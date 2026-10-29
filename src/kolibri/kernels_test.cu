// Synthetic mathematical checks; no model files or downloads are required.
// Attention uses a CPU F64 reference over the exact F16 cache sent to CUDA.
#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace k = strata::kolibri::kernels;
constexpr int Q = k::kQueryHeads * k::kHeadDim;
constexpr int KV = k::kKvHeads * k::kHeadDim;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <typename T> struct Device {
    T* p = nullptr;
    size_t count;
    explicit Device(size_t n) : count(n) { check(cudaMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& data) : Device(data.size()) { upload(data); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& data) {
        if (data.size() != count) throw std::runtime_error("test buffer size mismatch");
        check(cudaMemcpy(p, data.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> data(count);
        check(cudaMemcpy(data.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
        return data;
    }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(const std::string& label, const std::vector<float>& actual,
           const std::vector<double>& expected, double atol = 2e-6, double rtol = 2e-6) {
    require(actual.size() == expected.size(), "comparison shape mismatch");
    double worst = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || error > atol + rtol * std::abs(expected[i])) {
            std::fprintf(stderr, "%s[%zu]: actual %.10g, expected %.10g, error %.6g\n",
                         label.c_str(), i, actual[i], expected[i], error);
            throw std::runtime_error(label + " mismatch");
        }
        worst = std::max(worst, error);
    }
    std::printf("PASS %-45s max abs error %.3g\n", label.c_str(), worst);
}

std::vector<float> random_values(size_t count, unsigned seed, float scale = 1.f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-scale, scale);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(rng);
    return values;
}

uint16_t half_bits(float value) {
    const __half half = __float2half_rn(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float half_value(uint16_t bits) {
    __half half;
    std::memcpy(&half, &bits, sizeof(bits));
    return __half2float(half);
}

void norm_and_helpers(cudaStream_t stream) {
    constexpr int rows = 3, width = 2560;
    constexpr float eps = 1e-5f;
    auto input = random_values(rows * width, 12, 4.f);
    auto weights = random_values(width, 13, 2.f);
    std::fill_n(input.begin(), width, 0.f);
    std::vector<double> expected(input.size());
    for (int row = 0; row < rows; ++row) {
        double sum = 0;
        for (int d = 0; d < width; ++d) sum += (double) input[row * width + d] * input[row * width + d];
        const double scale = 1.0 / std::sqrt(sum / width + eps);
        for (int d = 0; d < width; ++d) expected[row * width + d] = input[row * width + d] * scale * weights[d];
    }
    Device<float> x(input), w(weights), out(input.size());
    k::rms_norm(x.p, w.p, out.p, rows, width, eps, stream);
    close("RMS norm, zero and nonzero rows", out.download(), expected);
    k::rms_norm(x.p, w.p, x.p, rows, width, eps, stream);
    close("RMS norm, in place", x.download(), expected);

    auto second = random_values(input.size(), 14);
    Device<float> b(second);
    x.upload(input);
    for (size_t i = 0; i < input.size(); ++i) expected[i] = (double) input[i] + second[i];
    k::add(x.p, b.p, x.p, input.size(), stream);
    close("Residual add, alias first input", x.download(), expected, 5e-7, 0);
    x.upload(input);
    k::add(x.p, b.p, b.p, input.size(), stream);
    close("Residual add, alias second input", b.download(), expected, 5e-7, 0);

    constexpr int ffn_width = 1024;
    auto gate = random_values(4 * ffn_width, 15, 8.f);
    auto up = random_values(gate.size(), 16, 2.f);
    gate[0] = -100.f; gate[1] = 100.f; gate[2] = 0.f;
    expected.resize(gate.size());
    for (size_t i = 0; i < gate.size(); ++i)
        expected[i] = ((double) gate[i] / (1.0 + std::exp(-(double) gate[i]))) * up[i];
    Device<float> dg(gate), du(up), dy(gate.size());
    k::swiglu(dg.p, du.p, dy.p, 4, ffn_width, stream);
    close("SwiGLU, including saturated gates", dy.download(), expected);

    constexpr int gather_width = 259;
    auto source = random_values(5 * gather_width, 17);
    std::vector<int32_t> ids{4, 0, 4, 2, 1};
    std::vector<float> multipliers{0.25f, -0.5f, 1.5f, 0.f, 2.f};
    Device<float> ds(source), gathered(ids.size() * gather_width), dm(multipliers);
    Device<int32_t> di(ids);
    expected.resize(ids.size() * gather_width);
    for (size_t row = 0; row < ids.size(); ++row)
        for (int d = 0; d < gather_width; ++d) expected[row * gather_width + d] = source[ids[row] * gather_width + d];
    k::gather_rows(ds.p, di.p, gathered.p, (int) ids.size(), gather_width, stream);
    close("Gather, reordered and duplicated rows", gathered.download(), expected, 0, 0);

    auto initial = random_values(source.size(), 18);
    Device<float> scattered(initial);
    expected.assign(initial.begin(), initial.end());
    for (size_t row = 0; row < ids.size(); ++row)
        for (int d = 0; d < gather_width; ++d)
            expected[ids[row] * gather_width + d] += (double) source[row * gather_width + d] * multipliers[row];
    k::scatter_add_weighted(ds.p, di.p, dm.p, scattered.p, (int) ids.size(), gather_width, stream);
    close("Weighted scatter, repeated destinations", scattered.download(), expected);
}

std::vector<double> qk_reference(const std::vector<float>& input, const std::vector<float>& weights,
                                 int heads, int64_t first, bool local, float theta, float eps) {
    std::vector<double> expected(input.size());
    const int rows = (int) input.size() / (heads * k::kHeadDim);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            const size_t offset = ((size_t) row * heads + head) * k::kHeadDim;
            double sum = 0;
            for (int d = 0; d < k::kHeadDim; ++d) sum += (double) input[offset + d] * input[offset + d];
            double normalized[k::kHeadDim];
            for (int d = 0; d < k::kHeadDim; ++d)
                normalized[d] = input[offset + d] * weights[d] / std::sqrt(sum / k::kHeadDim + eps);
            for (int d = 0; d < k::kHeadDim / 2; ++d) {
                const double angle = local ? (first + row) / std::pow((double) theta, (double) d / 64) : 0;
                expected[offset + d] = normalized[d] * std::cos(angle) - normalized[d + 64] * std::sin(angle);
                expected[offset + d + 64] = normalized[d + 64] * std::cos(angle) + normalized[d] * std::sin(angle);
            }
        }
    }
    return expected;
}

void qk_norm_rope(cudaStream_t stream) {
    constexpr int rows = 4;
    constexpr float theta = 1000000.f, eps = 1e-5f;
    auto q = random_values(rows * Q, 20);
    auto keys = random_values(rows * KV, 21);
    auto qw = random_values(k::kHeadDim, 22, 2.f);
    auto kw = random_values(k::kHeadDim, 23, 2.f);
    std::fill_n(q.begin(), k::kHeadDim, 0.f);
    Device<float> dq(q), dk(keys), dqw(qw), dkw(kw);
    for (const int64_t first : {int64_t(0), int64_t(511), int64_t(32764), int64_t(16777217)}) {
        dq.upload(q); dk.upload(keys);
        k::qk_norm_rope(dq.p, dk.p, dqw.p, dkw.p, rows, first, true, theta, eps, stream);
        close("Local Q norm/NeoX RoPE at " + std::to_string(first), dq.download(),
              qk_reference(q, qw, k::kQueryHeads, first, true, theta, eps));
        close("Local K norm/NeoX RoPE at " + std::to_string(first), dk.download(),
              qk_reference(keys, kw, k::kKvHeads, first, true, theta, eps));
    }
    dq.upload(q); dk.upload(keys);
    k::qk_norm_rope(dq.p, dk.p, dqw.p, dkw.p, rows, 32764, false, theta, eps, stream);
    close("Global Q norm without RoPE", dq.download(), qk_reference(q, qw, k::kQueryHeads, 32764, false, theta, eps));
    close("Global K norm without RoPE", dk.download(), qk_reference(keys, kw, k::kKvHeads, 32764, false, theta, eps));
}

void append(cudaStream_t stream) {
    constexpr int rows = k::kChunk, offset = 7;
    auto keys = random_values(rows * KV, 31, 5.f);
    auto values = random_values(rows * KV, 32, 5.f);
    keys[0] = 0.f; keys[1] = -0.f; keys[2] = 0.00001f;
    std::vector<uint16_t> initial((offset + rows + 1) * KV, half_bits(-17.f));
    Device<float> dk(keys), dv(values);
    Device<uint16_t> kc(initial), vc(initial);
    k::append_kv(dk.p, dv.p, kc.p, vc.p, rows, offset, stream);
    const auto actual_k = kc.download(), actual_v = vc.download();
    for (size_t i = 0; i < initial.size(); ++i) {
        const bool written = i >= offset * KV && i < (offset + rows) * KV;
        require(actual_k[i] == (written ? half_bits(keys[i - offset * KV]) : initial[i]), "F16 key append or guard mismatch");
        require(actual_v[i] == (written ? half_bits(values[i - offset * KV]) : initial[i]), "F16 value append or guard mismatch");
    }
    std::puts("PASS F16 KV append, 256 rows and offset guards");
}

void routing(cudaStream_t stream) {
    constexpr int rows = k::kChunk;
    auto logits = random_values(rows * k::kExperts, 40, 3.f);
    auto bias = random_values(k::kExperts, 41, 5.f);
    // Cancellation gives exact ties across all three 128-expert partitions.
    for (int expert = 0; expert < k::kExperts; ++expert) {
        logits[expert] = -bias[expert];
        logits[k::kExperts + expert] = -bias[expert] - 8.f;
        logits[3 * k::kExperts + expert] = -80.f;
        logits[4 * k::kExperts + expert] = 80.f;
    }
    const std::vector<int32_t> tied{0, 127, 128, 255, 256, 383};
    for (int expert : tied) logits[k::kExperts + expert] = -bias[expert];
    Device<float> dl(logits), db(bias), dw(rows * k::kTopK);
    Device<int32_t> di(rows * k::kTopK);
    k::route(dl.p, db.p, di.p, dw.p, rows, stream);
    const auto ids = di.download();
    const auto weights = dw.download();
    std::vector<double> expected(weights.size());
    for (int row = 0; row < rows; ++row) {
        std::vector<int> order(k::kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const float av = logits[row * k::kExperts + a] + bias[a];
            const float bv = logits[row * k::kExperts + b] + bias[b];
            return av > bv || (av == bv && a < b);
        });
        for (int rank = 0; rank < k::kTopK; ++rank) {
            const int index = row * k::kTopK + rank;
            require(ids[index] == order[rank], "routing expert ID mismatch");
            expected[index] = 1.0 / (1.0 + std::exp(-(double) logits[row * k::kExperts + order[rank]]));
        }
    }
    for (int rank = 0; rank < k::kTopK; ++rank) {
        require(ids[rank] == rank, "all-expert ties must choose lowest IDs");
        require(ids[k::kTopK + rank] == tied[rank], "cross-partition ties must choose lowest IDs");
    }
    require(std::accumulate(weights.begin() + 4 * k::kTopK, weights.begin() + 5 * k::kTopK, 0.f) == 6.f,
            "routing weights must not be renormalized");
    close("384 experts/top 6, bias, ties, sigmoid", weights, expected, 2e-7, 2e-6);
}

std::vector<double> attention_reference(const std::vector<float>& queries,
                                        const std::vector<uint16_t>& key_bits,
                                        const std::vector<uint16_t>& value_bits,
                                        int64_t key_first, int64_t query_first, bool local) {
    std::vector<double> keys(key_bits.size()), values(value_bits.size()), output(queries.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = half_value(key_bits[i]);
        values[i] = half_value(value_bits[i]);
    }
    const int rows = (int) queries.size() / Q;
    for (int row = 0; row < rows; ++row) {
        const int64_t position = query_first + row;
        const int64_t oldest = local ? std::max(int64_t(0), position - 512) : 0;
        const int begin = (int) (oldest - key_first), end = (int) (position - key_first + 1);
        for (int head = 0; head < k::kQueryHeads; ++head) {
            const int kv_head = head / 12;
            const size_t query_offset = ((size_t) row * k::kQueryHeads + head) * k::kHeadDim;
            std::vector<double> probabilities(end - begin);
            for (int token = begin; token < end; ++token) {
                const size_t key_offset = (size_t) token * KV + kv_head * k::kHeadDim;
                double dot = 0;
                for (int d = 0; d < k::kHeadDim; ++d) dot += queries[query_offset + d] * keys[key_offset + d];
                probabilities[token - begin] = dot / std::sqrt(128.0);
            }
            const double maximum = *std::max_element(probabilities.begin(), probabilities.end());
            double sum = 0;
            for (double& probability : probabilities) {
                probability = std::exp(probability - maximum);
                sum += probability;
            }
            double* result = output.data() + query_offset;
            for (int token = begin; token < end; ++token) {
                const double weight = probabilities[token - begin] / sum;
                const double* value = values.data() + (size_t) token * KV + kv_head * k::kHeadDim;
                for (int d = 0; d < k::kHeadDim; ++d) result[d] += weight * value[d];
            }
        }
    }
    return output;
}

void attention_case(const char* label, int rows, int64_t query_first, int64_t key_first,
                    bool local, cudaStream_t stream, bool large_scores = false) {
    const int key_count = (int) (query_first + rows - key_first);
    auto queries = random_values((size_t) rows * Q, 51);
    auto keys = random_values((size_t) key_count * KV, 52);
    auto values = random_values(keys.size(), 53, 3.f);
    if (large_scores) {
        // Exact large equal maxima and strongly negative scores exercise stable softmax.
        std::fill(queries.begin(), queries.end(), 0.f);
        for (size_t i = 0; i < queries.size(); i += k::kHeadDim) queries[i] = 1024.f;
        for (size_t i = 0; i < keys.size(); i += k::kHeadDim) keys[i] = (i / KV % 3 == 0) ? 32.f : -32.f;
    }
    std::vector<uint16_t> key_bits(keys.size()), value_bits(values.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        key_bits[i] = half_bits(keys[i]);
        value_bits[i] = half_bits(values[i]);
    }
    Device<float> dq(queries), out(queries.size());
    Device<uint16_t> dk(key_bits), dv(value_bits);
    k::attention(dq.p, dk.p, dv.p, out.p, rows, key_count, key_first, query_first, local, stream);
    const auto actual = out.download();
    close(label, actual, attention_reference(queries, key_bits, value_bits, key_first, query_first, local), 3e-6, 3e-6);
}

void attention(cudaStream_t stream) {
    attention_case("Local attention, first token", 1, 0, 0, true, stream);
    attention_case("Local attention, positions 511 through 514", 4, 511, 0, true, stream);
    attention_case("Local attention, 256-token boundary chunk", 256, 512, 0, true, stream);
    attention_case("Local attention, retained cache at 513", 4, 1025, 513, true, stream);
    attention_case("Local attention, retained cache near 32K", 4, 32764, 32252, true, stream);
    attention_case("Global attention, 256-token causal chunk", 256, 0, 0, false, stream);
    attention_case("Global attention, full 32768-token cache", 1, 32767, 0, false, stream);
    attention_case("Global attention, large positive/negative scores", 4, 40, 0, false, stream, true);

    bool rejected = false;
    try { k::attention(nullptr, nullptr, nullptr, nullptr, 1, 512, 2, 513, true, stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "local attention accepted missing required history");
    rejected = false;
    try { k::attention(nullptr, nullptr, nullptr, nullptr, 2, 1, 0, 0, false, stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "attention accepted a missing current key");
    std::puts("PASS Attention rejects incomplete causal cache");
}
}  // namespace

int main() {
    int devices = 0;
    const auto error = cudaGetDeviceCount(&devices);
    if (error != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "SKIP Kolibri CUDA math: no usable CUDA device\n");
        return 77;
    }
    cudaStream_t stream = nullptr;
    try {
        check(cudaStreamCreate(&stream));
        norm_and_helpers(stream);
        qk_norm_rope(stream);
        append(stream);
        routing(stream);
        attention(stream);
        check(cudaStreamSynchronize(stream));
        check(cudaStreamDestroy(stream));
        std::puts("PASS all Kolibri CUDA mathematical checks");
        return 0;
    } catch (const std::exception& e) {
        cudaStreamDestroy(stream);
        std::fprintf(stderr, "FAIL Kolibri CUDA math: %s\n", e.what());
        return 1;
    }
}
