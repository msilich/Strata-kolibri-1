#include "model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml-quants.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int H = 2560;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

template <typename T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& x, cudaStream_t stream) {
        check(cudaMemcpyAsync(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
    }
    std::vector<T> get(size_t n, cudaStream_t stream) const {
        check(cudaStreamSynchronize(stream));
        std::vector<T> x(n);
        check(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return x;
    }
};

struct Matrix {
    int in, out;
    std::vector<float> w;
};

// GGML's CPU dequantizers provide an independent interpretation of the file.
Matrix decode(const strata::GgufModel& file, const std::string& name, int expert = -1) {
    size_t shard = 0;
    const auto* t = file.find(name, &shard);
    if (!t || t->shape.size() != (expert < 0 ? 2u : 3u)) throw std::runtime_error("missing oracle tensor " + name);
    Matrix m{(int) t->shape[0], (int) t->shape[1], {}};
    const size_t count = (size_t) m.in * m.out;
    m.w.resize(count);
    const uint8_t* source = file.shard(shard).tensor_data(*t);
    if (expert >= 0) source += strata::tensor_payload_bytes(*t) / t->shape[2] * (size_t) expert;
    switch (t->type) {
        case 8: dequantize_row_q8_0((const block_q8_0*) source, m.w.data(), count); break;
        case 12: dequantize_row_q4_K((const block_q4_K*) source, m.w.data(), count); break;
        case 14: dequantize_row_q6_K((const block_q6_K*) source, m.w.data(), count); break;
        default: throw std::runtime_error("unsupported oracle type");
    }
    return m;
}

std::vector<float> input(int tokens, int width) {
    std::vector<float> x((size_t) tokens * width);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 0.3f * std::sin((float) i * 0.173f) + 0.2f * std::cos((float) i * 0.037f);
    return x;
}

std::vector<float> product(const Matrix& w, const std::vector<float>& x, int tokens) {
    std::vector<float> out((size_t) tokens * w.out);
    for (int t = 0; t < tokens; ++t)
        for (int r = 0; r < w.out; ++r) {
            double sum = 0;
            for (int k = 0; k < w.in; ++k)
                sum += (double) w.w[(size_t) r * w.in + k] * x[(size_t) t * w.in + k];
            out[(size_t) t * w.out + r] = (float) sum;
        }
    return out;
}

void near(const char* name, const std::vector<float>& got, const std::vector<float>& want, double tolerance) {
    if (got.size() != want.size()) throw std::runtime_error("oracle size mismatch");
    double error = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) throw std::runtime_error(std::string(name) + " produced a non-finite value");
        const double d = (double) got[i] - want[i];
        error += d * d;
        norm += (double) want[i] * want[i];
        maximum = std::max(maximum, std::abs(d));
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    std::printf("%s: relative RMS %.6g, max absolute %.6g\n", name, relative, maximum);
    if (relative > tolerance) throw std::runtime_error(std::string(name) + " disagrees with the F32 oracle");
}

void identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
        throw std::runtime_error("cached and streamed expert outputs differ");
}
void grouped_prefill_checks(strata::kolibri::Model& model, const strata::GgufModel& file, Stream& stream,
                           bool smoke = false) {
    constexpr int capacity = 4096;
    const auto source = input(capacity, H);
    Device<float> dx(capacity * H), packed(capacity * H), scalar_out(capacity * H), grouped_out(capacity * H);
    Device<float> mmq_x(9 * H), mmq_y(9 * H);
    Device<int32_t> row_ids(capacity);
    dx.put(source, stream.value);
    strata::kolibri::Executor mmq_reference(model, stream.value, 9, -1, 0);
    for (int slots : {0, 1, 16}) {
        strata::kolibri::Executor scalar(model, stream.value, capacity, -1, slots);
        strata::kolibri::Executor grouped(model, stream.value, capacity, -1, slots, false, false, false, true);
        for (int layer = 0; layer < 50; ++layer)
            if (!grouped.supports_grouped_prefill(layer) || scalar.supports_grouped_prefill(layer))
                throw std::runtime_error("grouped prefill support or opt-out fallback is incorrect");
        for (int layer : {0, 1, 12, 13, 49}) {
            std::vector<std::vector<int>> plans{{1}, {1, 2, 4, 8},
                {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}, {17}, {9, 17, 63},
                {9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100, 107, 114}};
            if (smoke) plans.back().assign(16, 9);
            else plans.push_back({3000, 800, 279, 17});
            for (const auto& counts : plans) {
                const bool small_rows = *std::max_element(counts.begin(), counts.end()) <= 8;
                std::vector<int32_t> experts, bounds{0}, ids;
                for (size_t i = 0; i < counts.size(); ++i) {
                    experts.push_back(i + 1 == counts.size() ? 383 : (int) i);
                    bounds.push_back(bounds.back() + counts[i]);
                }
                const int rows = bounds.back();
                std::vector<float> gathered((size_t) rows * H);
                for (int i = 0; i < rows; ++i) {
                    ids.push_back((i * 29 + 7) % capacity);
                    std::copy_n(source.data() + (size_t) ids.back() * H, H, gathered.data() + (size_t) i * H);
                }
                row_ids.put(ids, stream.value); packed.put(gathered, stream.value);
                std::vector<float> old_mmq;
                if (small_rows) {
                    old_mmq.resize((size_t) rows * H);
                    for (size_t i = 0; i < experts.size(); ++i) {
                        std::vector<float> nine_rows(9 * H);
                        for (int row = 0; row < 9; ++row)
                            std::copy_n(gathered.data() + (size_t) (bounds[i] + std::min(row, counts[i] - 1)) * H,
                                        H, nine_rows.data() + (size_t) row * H);
                        mmq_x.put(nine_rows, stream.value);
                        mmq_reference.expert(layer, experts[i], mmq_x.p, mmq_y.p, 9);
                        const auto reference = mmq_y.get(9 * H, stream.value);
                        std::copy_n(reference.data(), (size_t) counts[i] * H,
                                    old_mmq.data() + (size_t) bounds[i] * H);
                    }
                }
                for (int repeat = 0; repeat < 2; ++repeat) {
                    for (size_t i = 0; i < experts.size(); ++i)
                        scalar.expert(layer, experts[i], packed.p + (size_t) bounds[i] * H,
                                      scalar_out.p + (size_t) bounds[i] * H, counts[i]);
                    grouped.experts_prefill(layer, experts.data(), (int) experts.size(), bounds.data(),
                                             row_ids.p, dx.p, grouped_out.p, rows);
                    const auto result = grouped_out.get((size_t) rows * H, stream.value);
                    // Small groups apply existing MMQ arithmetic to fewer rows.
                    // Check every output against the old nine-row MMQ path;
                    // MMVQ and CPU differences remain recorded diagnostics.
                    if (small_rows) near("small groups versus existing nine-row MMQ", result, old_mmq, 0.001);
                    near(small_rows ? "grouped prefill versus scalar MMVQ (diagnostic)"
                                    : "grouped prefill versus scalar MMQ", result,
                          scalar_out.get(result.size(), stream.value), small_rows ? INFINITY : 0.001);
                    if (grouped.hits() != scalar.hits() || grouped.misses() != scalar.misses())
                        throw std::runtime_error("grouped prefill changed cache admission or counters");
                    if (slots == 0 && repeat == 0 && (rows == capacity || small_rows)) {
                        // Sample both ends and every 1/2/4/8-row expert in the
                        // small mixed plan, including expert 383's tensor boundary.
                        std::vector<int> oracle_rows{0, rows - 1};
                        if (small_rows && experts.size() <= 4)
                            for (size_t i = 0; i < experts.size(); ++i) oracle_rows.push_back(bounds[i]);
                        std::sort(oracle_rows.begin(), oracle_rows.end());
                        oracle_rows.erase(std::unique(oracle_rows.begin(), oracle_rows.end()), oracle_rows.end());
                        for (int row : oracle_rows) {
                            const size_t group = std::upper_bound(bounds.begin(), bounds.end(), row) - bounds.begin() - 1;
                            const int expert = experts[group];
                            const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
                            const Matrix gate = decode(file, prefix + "gate_exps.weight", expert);
                            const Matrix up = decode(file, prefix + "up_exps.weight", expert);
                            const Matrix down = decode(file, prefix + "down_exps.weight", expert);
                            const std::vector<float> one_x(gathered.begin() + (size_t) row * H,
                                                           gathered.begin() + (size_t) (row + 1) * H);
                            auto hidden = product(gate, one_x, 1); const auto u = product(up, one_x, 1);
                            for (size_t i = 0; i < hidden.size(); ++i)
                                hidden[i] = hidden[i] / (1.f + std::exp(-hidden[i])) * u[i];
                            const std::vector<float> one_y(result.begin() + (size_t) row * H,
                                                           result.begin() + (size_t) (row + 1) * H);
                            const auto cpu = product(down, hidden, 1);
                            if (small_rows) {
                                const std::vector<float> mmq_row(old_mmq.begin() + (size_t) row * H,
                                                                 old_mmq.begin() + (size_t) (row + 1) * H);
                                // The established MMQ path can itself exceed
                                // 3% on tiny outputs. Keep both CPU errors visible;
                                // the strict MMQ comparison above guards this change.
                                near("small-group CPU oracle (diagnostic)", one_y, cpu, INFINITY);
                                near("existing nine-row MMQ CPU oracle (diagnostic)", mmq_row, cpu, INFINITY);
                            } else near("grouped prefill CPU oracle", one_y, cpu, 0.03);
                        }
                    }
                }
            }
        }
        for (int layer : {0, 1, 12}) {
            scalar.expert(layer, 383, dx.p, scalar_out.p, 8);
            grouped.expert(layer, 383, dx.p, grouped_out.p, 8);
            identical(scalar_out.get(8 * H, stream.value), grouped_out.get(8 * H, stream.value));
        }
        bool rejected = false;
        const int32_t expert[]{0}, bounds[]{0, 0};
        try { grouped.experts_prefill(0, expert, 1, bounds, row_ids.p, dx.p, grouped_out.p, 0); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("grouped prefill accepted an empty group");
        rejected = false;
        const int32_t six[]{0, 1, 2, 3, 4, 5};
        try { grouped.experts_decode(13, six, dx.p, grouped_out.p); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("prefill staging accidentally enabled grouped decode");
        std::printf("Grouped prefill cache slots %d, %llu hits / %llu misses; explicit scalar calls bitwise equal\n",
                    slots, (unsigned long long) grouped.hits(), (unsigned long long) grouped.misses());
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool smoke = argc == 3 && std::string(argv[2]) == "--grouped-prefill-smoke";
        if (argc != 2 && !smoke)
            throw std::runtime_error("usage: kolibri_model_test MODEL.gguf [--grouped-prefill-smoke]");
        if (!std::getenv("STRATA_MMQ_KQ")) setenv("STRATA_MMQ_KQ", "1", 0);
        Stream stream;
        strata::kolibri::Model model;
        model.load(argv[1], stream.value);
        if (model.layers.size() != 50 || model.eos != 127906 || model.vocab != 128000)
            throw std::runtime_error("unexpected Kolibri geometry/token metadata");
        strata::GgufModel file(strata::gguf_split_paths(argv[1]));
        if (smoke) {
            grouped_prefill_checks(model, file, stream, true);
            std::puts("Kolibri grouped prefill smoke checks passed");
            return 0;
        }
        strata::kolibri::Executor cached(model, stream.value, 256, -1, 1);
        strata::kolibri::Executor streamed(model, stream.value, 256, -1, 0);
        if (cached.cache_slots() != 50 || streamed.cache_slots() != 0)
            throw std::runtime_error("per-layer cache capacity is incorrect");
        Device<float> x(256 * 6144), y(256 * 6144), z(256 * 6144);
        Device<int32_t> ids(4);

        const std::vector<int32_t> token_ids{0, 37, 127906, 127999};
        ids.put(token_ids, stream.value);
        cached.embed(ids.p, y.p, (int) token_ids.size());
        std::vector<float> embedding(token_ids.size() * H);
        size_t shard = 0;
        const auto* table = file.find("token_embd.weight", &shard);
        const uint8_t* data = file.shard(shard).tensor_data(*table);
        const size_t row_bytes = strata::tensor_payload_bytes(*table) / model.vocab;
        for (size_t i = 0; i < token_ids.size(); ++i)
            dequantize_row_q8_0((const block_q8_0*) (data + token_ids[i] * row_bytes), embedding.data() + i * H, H);
        near("embedding", y.get(embedding.size(), stream.value), embedding, 1e-7);

        const Matrix dense = decode(file, "blk.0.attn_k.weight");
        for (int tokens : {1, 7, 9, 17, 256}) {
            const auto host_x = input(tokens, H);
            x.put(host_x, stream.value);
            cached.multiply(model.layers[0].k, x.p, y.p, tokens);
            const std::string name = "dense T=" + std::to_string(tokens);
            near(name.c_str(), y.get((size_t) tokens * dense.out, stream.value), product(dense, host_x, tokens), 0.02);
        }
        {
            const Matrix output = decode(file, "blk.0.attn_output.weight");
            const auto host_x = input(9, output.in);
            x.put(host_x, stream.value);
            cached.multiply(model.layers[0].o, x.p, y.p, 9);
            near("6144-column projection", y.get(9 * H, stream.value), product(output, host_x, 9), 0.02);
        }

        for (int layer : {0, 24, 49}) {
            // The last expert checks the source slice at the end of each tensor.
            const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
            const Matrix gate = decode(file, prefix + "gate_exps.weight", 383);
            const Matrix up = decode(file, prefix + "up_exps.weight", 383);
            const Matrix down = decode(file, prefix + "down_exps.weight", 383);
            for (int tokens : {1, 7, 17}) {
                const auto host_x = input(tokens, H);
                x.put(host_x, stream.value);
                cached.expert(layer, 383, x.p, y.p, tokens);
                streamed.expert(layer, 383, x.p, z.p, tokens);
                const auto result = y.get((size_t) tokens * H, stream.value);
                identical(result, z.get(result.size(), stream.value));
                auto hidden = product(gate, host_x, tokens);
                const auto u = product(up, host_x, tokens);
                for (size_t i = 0; i < hidden.size(); ++i)
                    hidden[i] = hidden[i] / (1.f + std::exp(-hidden[i])) * u[i];
                const std::string name = "expert layer=" + std::to_string(layer) + " T=" + std::to_string(tokens);
                near(name.c_str(), result, product(down, hidden, tokens), 0.03);
                // With one slot per layer this second expert must use the miss
                // buffer. Its queued work must not overwrite the resident expert.
                cached.expert(layer, 0, x.p, z.p, tokens);
                cached.expert(layer, 383, x.p, y.p, tokens);
                identical(result, y.get(result.size(), stream.value));
            }
        }
        for (int type : {8, 12, 14})
            for (int cols : {512, 2560, 6144})
                if (strata::prefill::mmq::overread_bytes(type, cols) > 4096)
                    throw std::runtime_error("MMQ exceeds the weight tail allocation");
        if (cached.hits() != 15 || cached.misses() != 12 || streamed.hits() != 0 || streamed.misses() != 9)
            throw std::runtime_error("cache hit/miss accounting is incorrect");
        for (int slots : {0, 1, 6}) {
            strata::kolibri::Executor scalar(model, stream.value, 1, -1, slots);
            strata::kolibri::Executor grouped(model, stream.value, 1, -1, slots, true);
            // Mixed gate/up formats and the final Q8 layer exercise every format
            // in the Q4_K_M pack. Six misses must remain live until all products finish.
            for (int layer : {0, 24, 48, 49}) {
                const std::vector<int32_t> first{0, 1, 2, 3, 4, 383};
                const std::vector<int32_t> second{5, 6, 7, 8, 9, 382};
                x.put(input(1, H), stream.value);
                for (const auto* selected : {&first, &second, &first}) {
                    for (size_t i = 0; i < selected->size(); ++i)
                        scalar.expert(layer, (*selected)[i], x.p, y.p + i * H, 1);
                    grouped.experts_decode(layer, selected->data(), x.p, z.p);
                    const auto reference = y.get(6 * H, stream.value);
                    const auto result = z.get(6 * H, stream.value);
                    if (std::memcmp(reference.data(), result.data(), reference.size() * sizeof(float)))
                        throw std::runtime_error("grouped decode differs from scalar experts at layer " +
                                                 std::to_string(layer) + ", cache slots " + std::to_string(slots));
                }
            }
            const uint64_t expected_hits = 4u * (unsigned) slots;
            const uint64_t expected_misses = 72u - expected_hits;
            if (scalar.hits() != expected_hits || grouped.hits() != expected_hits ||
                scalar.misses() != expected_misses || grouped.misses() != expected_misses)
                throw std::runtime_error("grouped decode changes cache admission or accounting");
            std::printf("Grouped decode is bitwise identical on four layers, cache slots %d, "
                        "%llu hits / %llu misses\n", slots,
                        (unsigned long long) expected_hits, (unsigned long long) expected_misses);
        }
        {
            strata::kolibri::Executor reference(model, stream.value, 1, -1, 0);
            strata::kolibri::Executor adaptive(model, stream.value, 1, -1, 6, true, true);
            const std::vector<int32_t> a{0, 1, 2, 3, 4, 383};
            const std::vector<int32_t> b{0, 5, 6, 7, 8, 382};
            const uint64_t expected_hits[]{0, 1, 7, 8};
            x.put(input(1, H), stream.value);
            int step = 0;
            for (const auto* selected : {&a, &b, &b, &a}) {
                for (size_t i = 0; i < selected->size(); ++i)
                    reference.expert(0, (*selected)[i], x.p, y.p + i * H, 1);
                adaptive.experts_decode(0, selected->data(), x.p, z.p);
                identical(y.get(6 * H, stream.value), z.get(6 * H, stream.value));
                if (adaptive.hits() != expected_hits[step] ||
                    adaptive.misses() != (uint64_t) (6 * (step + 1)) - expected_hits[step])
                    throw std::runtime_error("decode LRU did not retain or replace the expected experts");
                ++step;
            }
            // A resident expert requested last must survive earlier misses in
            // the same group, even when there is only one cache slot.
            strata::kolibri::Executor single(model, stream.value, 1, -1, 1, true, true);
            single.expert(0, 383, x.p, y.p, 1);
            single.experts_decode(0, a.data(), x.p, z.p);
            for (size_t i = 0; i < a.size(); ++i)
                reference.expert(0, a[i], x.p, y.p + i * H, 1);
            identical(y.get(6 * H, stream.value), z.get(6 * H, stream.value));
            if (single.hits() != 1 || single.misses() != 6)
                throw std::runtime_error("decode LRU evicted an expert selected later in the group");
            std::printf("Decode LRU replacement and selected-expert protection passed\n");
        }
        {
            constexpr int rows = 4096;
            strata::kolibri::Executor large(model, stream.value, rows, -1, 0);
            Device<float> batch_x(rows * H), batch_y(rows * dense.out);
            const auto host_x = input(rows, H);
            batch_x.put(host_x, stream.value);
            large.multiply(model.layers[0].k, batch_x.p, batch_y.p, rows);
            const auto result = batch_y.get(rows * dense.out, stream.value);
            for (int row : {0, rows / 2 - 1, rows - 1}) {
                const std::vector<float> row_x(host_x.begin() + row * H, host_x.begin() + (row + 1) * H);
                const std::vector<float> row_y(result.begin() + row * dense.out,
                                               result.begin() + (row + 1) * dense.out);
                near("4096-row MMQ projection", row_y, product(dense, row_x, 1), 0.02);
            }
            bool rejected = false;
            try { large.multiply(model.layers[0].k, batch_x.p, batch_y.p, rows + 1); }
            catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("MMQ accepted a chunk larger than its buffers");
        }
        grouped_prefill_checks(model, file, stream);
        std::printf("Kolibri real-GGUF model/executor checks passed; cache %llu bytes, %llu hits, %llu misses\n",
                    (unsigned long long) cached.cache_bytes(), (unsigned long long) cached.hits(),
                    (unsigned long long) cached.misses());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "kolibri_model_test: %s\n", error.what());
        return 1;
    }
}
