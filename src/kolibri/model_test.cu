#include "model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml-quants.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kolibri_model_test MODEL.gguf");
        Stream stream;
        strata::kolibri::Model model;
        model.load(argv[1], stream.value);
        if (model.layers.size() != 50 || model.eos != 127906 || model.vocab != 128000)
            throw std::runtime_error("unexpected Kolibri geometry/token metadata");
        strata::GgufModel file(strata::gguf_split_paths(argv[1]));
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
        std::printf("Kolibri real-GGUF model/executor checks passed; cache %llu bytes, %llu hits, %llu misses\n",
                    (unsigned long long) cached.cache_bytes(), (unsigned long long) cached.hits(),
                    (unsigned long long) cached.misses());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "kolibri_model_test: %s\n", error.what());
        return 1;
    }
}
