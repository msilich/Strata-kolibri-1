#include "dense_decode.cuh"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml-quants.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
namespace K = strata::kolibri::kernels;
namespace N = strata::kernels;
constexpr int H = 2560;

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<typename T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) { check(cudaMalloc(&p, count * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& v) { check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

void run(std::vector<int> rows, cudaStream_t stream) {
    K::DenseDecodeProjection projection{};
    projection.count = int(rows.size());
    size_t weight_bytes = 4096, output_floats = 32;
    size_t weight_offset[3], output_offset[3];
    for (int i = 0; i < projection.count; ++i) {
        weight_offset[i] = weight_bytes;
        output_offset[i] = output_floats;
        weight_bytes += N::native_mmvq_weight_bytes(8, H, rows[i]) + 4096;
        output_floats += rows[i] + 32;
    }
    std::vector<unsigned char> weights(weight_bytes, 0xa5);
    for (int i = 0; i < projection.count; ++i) {
        std::vector<float> w(size_t(H) * rows[i]);
        uint32_t seed = 1957 + i;
        for (float& v : w) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            v = (int(seed & 65535) - 32768) * (0.05f / 32768);
        }
        quantize_row_q8_0_ref(w.data(), reinterpret_cast<block_q8_0*>(weights.data() + weight_offset[i]), w.size());
    }
    Device<unsigned char> weight(weights.size()), q8(N::native_q8_1_bytes(H));
    weight.put(weights);
    Device<float> input(H), got(output_floats), want(output_floats);
    for (int i = 0; i < projection.count; ++i) {
        projection.weights[i] = weight.p + weight_offset[i];
        projection.outputs[i] = got.p + output_offset[i];
        projection.rows[i] = rows[i];
    }
    for (int mode = 0; mode < 3; ++mode) {
        std::vector<float> x(H), initial(output_floats, 12345.f);
        for (int i = 0; i < H; ++i)
            x[i] = mode == 0 ? 0.f : std::sin(i * 0.173f) * (mode == 1 ? 0.3f : 64.f) + std::cos(i * 0.037f) * 0.2f;
        input.put(x); got.put(initial); want.put(initial);
        N::native_quantize_q8_1(input.p, q8.p, H, 1, stream);
        for (int i = 0; i < projection.count; ++i)
            N::native_mmvq(8, projection.weights[i], q8.p, want.p + output_offset[i], H, rows[i], 1, stream);
        K::dense_decode_q8(projection, q8.p, stream);
        check(cudaStreamSynchronize(stream));
        std::vector<float> a(output_floats), b(output_floats);
        check(cudaMemcpy(a.data(), got.p, a.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(), want.p, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
            throw std::runtime_error("batched dense Q8 differs from native scalar output or overwrote a guard");
    }
    std::printf("dense rows");
    for (int n : rows) std::printf(" %d", n);
    std::puts(": bitwise scalar parity for zero/normal/large inputs; output guards intact");
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        Stream stream;
        run({6144, 512, 512}, stream.value);
        run({512, 6144, 512}, stream.value);
        run({512, 512}, stream.value);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Kolibri dense decode test failed: %s\n", error.what());
        return 1;
    }
}
