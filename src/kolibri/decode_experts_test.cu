#include "decode_experts.cuh"
#include "kernels.cuh"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml-quants.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace K = strata::kolibri::kernels;
namespace N = strata::kernels;
constexpr int H = K::kDecodeExpertInput, F = K::kDecodeExpertHidden, E = K::kDecodeExpertCount;

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
template<typename T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) { check(cudaMalloc(&p, count * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<T>& v) { check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
};

void equal(const char* name, const void* a, const void* b, size_t bytes, cudaStream_t stream) {
    check(cudaStreamSynchronize(stream));
    std::vector<unsigned char> x(bytes), y(bytes);
    check(cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost));
    const auto mismatch = std::mismatch(x.begin(), x.end(), y.begin());
    if (mismatch.first != x.end())
        throw std::runtime_error(std::string(name) + " differs at byte " + std::to_string(mismatch.first - x.begin()));
}

void swiglu_q8_case(cudaStream_t stream) {
    constexpr int n = 32768;
    std::vector<float> gate(n, 32.f), up(n);
    const uint32_t special[] = {0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
        0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
        0x7fc12345, 0xffc12345, 0x7f800001, 0xff800001, 0x3f800000, 0xbf800000};
    for (int i = 0; i < n; ++i) {
        const int block = i / 32, lane = i % 32;
        if (i < 1024) {
            std::memcpy(&gate[i], &special[i % std::size(special)], 4);
            std::memcpy(&up[i], &special[(i / std::size(special)) % std::size(special)], 4);
        } else if (i < 2048) {
            // Uniform blocks expose FTZ and signed-zero behavior of native Q8.
            std::memcpy(&up[i], &special[block % 8], 4);
        } else {
            // SiLU(32) rounds to 32. Division by 32 is exact, so these probe
            // Q8 half-step neighbors over a wide scale range.
            const float d = std::ldexp(1.f, (block % 250) - 126);
            float value = lane == 31 ? 127.f * d : ((block * 31 + lane) % 253 - 126 + 0.5f) * d;
            if (lane != 31 && block % 3)
                value = std::nextafter(value, block % 3 == 1 ? -INFINITY : INFINITY);
            up[i] = value / 32.f;
        }
    }
    Device<float> g(n), u(n), hidden(n + 64), reference(n + 64);
    const size_t bytes = N::native_q8_1_bytes(n);
    Device<unsigned char> q8(bytes + 128), ref_q8(bytes + 128);
    g.put(gate); u.put(up);
    for (int count : {96, 512, 3072, n}) {
        check(cudaMemsetAsync(hidden.p, 0xa5, (n + 64) * 4, stream));
        check(cudaMemsetAsync(reference.p, 0xa5, (n + 64) * 4, stream));
        check(cudaMemsetAsync(q8.p, 0xa5, bytes + 128, stream));
        check(cudaMemsetAsync(ref_q8.p, 0xa5, bytes + 128, stream));
        K::swiglu(g.p, u.p, reference.p + 32, 1, count, stream);
        N::native_quantize_q8_1(reference.p + 32, ref_q8.p + 64, count, 1, stream);
        K::swiglu_quantize_q8_1(g.p, u.p, hidden.p + 32, q8.p + 64, count, stream);
        equal("fused SwiGLU hidden and guards", hidden.p, reference.p, (n + 64) * 4, stream);
        equal("fused native Q8 bytes and guards", q8.p, ref_q8.p, bytes + 128, stream);
    }
    std::puts("SwiGLU/Q8: exact bytes at half steps, special values, FTZ boundaries and guarded tails");
}

// CPU GGML quantizers produce valid, varied weights independently of both CUDA
// paths. Padding and a nontrivial pointer order exercise the grouped addressing.
void quantize(int type, unsigned char* dst, uint32_t seed) {
    std::vector<float> weights(size_t(H) * F);
    for (float& x : weights) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        x = (int(seed & 65535) - 32768) * (0.05f / 32768);
    }
    switch (type) {
        case 12: quantize_row_q4_K_ref(weights.data(), reinterpret_cast<block_q4_K*>(dst), weights.size()); break;
        case 14: quantize_row_q6_K_ref(weights.data(), reinterpret_cast<block_q6_K*>(dst), weights.size()); break;
        case 8: quantize_row_q8_0_ref(weights.data(), reinterpret_cast<block_q8_0*>(dst), weights.size()); break;
    }
}

void run_case(std::array<int, 3> types, cudaStream_t stream) {
    const auto aligned = [](size_t n) { return (n + 4095) / 4096 * 4096; };
    K::DecodeExpertWeights table{};
    table.gate_type = types[0]; table.up_type = types[1]; table.down_type = types[2];
    table.gate_offset = 4096;
    table.up_offset = table.gate_offset + aligned(N::native_mmvq_weight_bytes(types[0], H, F));
    table.down_offset = table.up_offset + aligned(N::native_mmvq_weight_bytes(types[1], H, F));
    const size_t stride = table.down_offset + aligned(N::native_mmvq_weight_bytes(types[2], F, H));
    std::vector<unsigned char> blobs(E * stride, 0xa5);
    for (int expert = 0; expert < E; ++expert) {
        auto* base = blobs.data() + expert * stride;
        quantize(types[0], base + table.gate_offset, 1729 + expert * 3);
        quantize(types[1], base + table.up_offset, 1730 + expert * 3);
        quantize(types[2], base + table.down_offset, 1731 + expert * 3);
    }
    Device<unsigned char> weights(blobs.size()); weights.put(blobs);
    constexpr int permutation[E] = {5, 1, 4, 0, 3, 2};
    for (int i = 0; i < E; ++i) table.expert[i] = weights.p + permutation[i] * stride;
    Device<float> input(H), got(E * H), want(E * H);
    Device<float> scratch(K::kDecodeExpertScratchFloats), reference(K::kDecodeExpertScratchFloats);
    Device<unsigned char> q8(K::kDecodeExpertQ8Bytes), ref_q8(K::kDecodeExpertQ8Bytes);
    const size_t input_bytes = N::native_q8_1_bytes(H), hidden_bytes = N::native_q8_1_bytes(F);
    for (int mode = 0; mode < 3; ++mode) {
        std::vector<float> x(H);
        for (int i = 0; i < H; ++i)
            x[i] = mode == 1 ? 0.f : std::sin(i * 0.173f) * (mode == 2 ? 16.f : 0.3f) + std::cos(i * 0.037f) * 0.2f;
        input.put(x);
        K::decode_experts(table, input.p, got.p, q8.p, scratch.p, stream);
        for (int expert = 0; expert < E; ++expert) {
            const auto* base = static_cast<const unsigned char*>(table.expert[expert]);
            float* gate = reference.p + expert * F;
            float* up = reference.p + E * F + expert * F;
            float* hidden = reference.p + 2 * E * F + expert * F;
            N::native_quantize_q8_1(input.p, ref_q8.p, H, 1, stream);
            N::native_mmvq(types[0], base + table.gate_offset, ref_q8.p, gate, H, F, 1, stream);
            N::native_quantize_q8_1(input.p, ref_q8.p, H, 1, stream);
            N::native_mmvq(types[1], base + table.up_offset, ref_q8.p, up, H, F, 1, stream);
            K::swiglu(gate, up, hidden, 1, F, stream);
            auto* hq8 = ref_q8.p + input_bytes + expert * hidden_bytes;
            N::native_quantize_q8_1(hidden, hq8, F, 1, stream);
            N::native_mmvq(types[2], base + table.down_offset, hq8, want.p + expert * H, F, H, 1, stream);
        }
        equal("gate/up/accurate SwiGLU", scratch.p, reference.p, K::kDecodeExpertScratchFloats * sizeof(float), stream);
        equal("Q8 input/hidden", q8.p, ref_q8.p, K::kDecodeExpertQ8Bytes, stream);
        equal("down projection", got.p, want.p, E * H * sizeof(float), stream);


        // Shared dense projections can retain the normalized Q8 input while
        // other projections use their own scratch. Poison this call's unused
        // input workspace so accidentally consuming it is observable.
        check(cudaMemsetAsync(q8.p, 0xa5, input_bytes, stream));
        K::decode_experts(table, input.p, got.p, q8.p, scratch.p, stream, ref_q8.p);
        equal("reused Q8 gate/up/hidden", scratch.p, reference.p, K::kDecodeExpertScratchFloats * sizeof(float), stream);
        equal("reused Q8 hidden quantization", q8.p + input_bytes, ref_q8.p + input_bytes, E * hidden_bytes, stream);
        equal("reused Q8 down projection", got.p, want.p, E * H * sizeof(float), stream);
        std::vector<unsigned char> guard(input_bytes);
        check(cudaMemcpy(guard.data(), q8.p, guard.size(), cudaMemcpyDeviceToHost));
        if (!std::all_of(guard.begin(), guard.end(), [](unsigned char x) { return x == 0xa5; }))
            throw std::runtime_error("reused Q8 input was unnecessarily requantized");
    }
    std::printf("types %d/%d/%d: gate/up/hidden/Q8/down and reused input bitwise equal\n",
                types[0], types[1], types[2]);
}

void combine_case(cudaStream_t stream) {
    Device<float> outputs(E * H), weights(E), got(H), want(H);
    Device<int32_t> row(1); row.put({0});
    constexpr int32_t ranks[E] = {5, 1, 4, 0, 3, 2};
    const std::vector<float> route_weights = {0.61f, 0.29f, 0.97f, 0.03125f, 1.f, 0.75f};
    weights.put(route_weights);
    std::vector<float> x(E * H), shared(H);
    for (int i = 0; i < H; ++i) {
        shared[i] = std::cos(i * 0.131f) * 3.f;
        for (int e = 0; e < E; ++e) x[e * H + i] = std::sin((e * H + i) * 0.217f) * (e + 1);
    }
    // Cancellation makes addition order observable; subnormals exercise the
    // original global atomicAdd behavior after the rounded multiplication.
    shared[0] = 16777216.f;
    for (int e = 0; e < E; ++e) x[e * H] = e == 0 ? -16777216.f / route_weights[ranks[e]] : 0.7f;
    shared[1] = 0.f; shared[2] = 0x1p-126f; shared[3] = -0.f;
    for (int e = 0; e < E; ++e) {
        x[e * H + 1] = 0x1p-149f * (e + 1);
        x[e * H + 2] = -0x1p-126f;
        x[e * H + 3] = e & 1 ? 0.f : -0.f;
    }
    // Exceptional values retain the original atomic NaN behavior, including
    // payload/sign bits. Finite sum overflow exercises the register path.
    const uint32_t exceptional[] = {0x7f800000, 0xff800000, 0x7fc12345,
                                    0xffc12345, 0x7f800001, 0xff800001};
    for (int j = 0; j < 6; ++j) {
        std::memcpy(&shared[4 + j], &exceptional[j], sizeof(float));
        std::memcpy(&x[10 + j], &exceptional[j], sizeof(float));
    }
    shared[16] = 1.f;
    x[16] = 0x1p-24f / route_weights[ranks[0]];
    shared[17] = 0x1.fffffep127f;
    x[17] = 0x1.fffffep127f;
    for (int e = 1; e < E; ++e) x[e * H + 17] = 0x1.fffffep127f;
    outputs.put(x); got.put(shared); want.put(shared);
    K::combine_decode_experts(outputs.p, weights.p, ranks, got.p, stream);
    for (int e = 0; e < E; ++e)
        K::scatter_add_weighted(outputs.p + e * H, row.p, weights.p + ranks[e], want.p, 1, H, stream);
    equal("shared-first ascending weighted atomic additions", got.p, want.p, H * sizeof(float), stream);
    std::puts("combine: bitwise equal to six original scatters, including cancellation, subnormals and non-finite payloads");
}
}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        Stream stream;
        swiglu_q8_case(stream.value);
        for (const auto& types : {std::array<int, 3>{12, 14, 8}, {12, 8, 14}, {12, 14, 14}, {12, 12, 12}, {8, 8, 8}, {14, 14, 14}})
            run_case(types, stream.value);
        combine_case(stream.value);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Kolibri grouped decode test failed: %s\n", error.what());
        return 1;
    }
}
