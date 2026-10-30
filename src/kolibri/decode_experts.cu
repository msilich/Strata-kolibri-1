#include "decode_experts.cuh"
#include "decode_cache.cuh"
#include "kernels.cuh"
#include "strata/kernels/native_mmvq.hpp"

// Reuse the pinned 3cf03257 device arithmetic from which native_mmvq.cu was
// adapted. This translation unit has the same --use_fast_math contract; SwiGLU
// remains in kernels.cu, which deliberately does not use fast math.
#include "ggml-cuda/vecdotq.cuh"

#include <stdexcept>

namespace strata::kolibri::kernels {
namespace {
constexpr int kWarp = 32;
constexpr int kWarps = 4;

template<int Type> struct Format;
template<> struct Format<12> {
    using Block = block_q4_K;
    static constexpr int elements = 256, threads = 16, vdr = 2;
    __device__ __forceinline__ static float dot(const void* row, const block_q8_1* x, int block, int iqs) {
        return vec_dot_q4_K_q8_1(row, x, block, iqs);
    }
};
template<> struct Format<14> {
    using Block = block_q6_K;
    static constexpr int elements = 256, threads = 32, vdr = 1;
    __device__ __forceinline__ static float dot(const void* row, const block_q8_1* x, int block, int iqs) {
        return vec_dot_q6_K_q8_1(row, x, block, iqs);
    }
};
template<> struct Format<8> {
    using Block = block_q8_0;
    static constexpr int elements = 32, threads = 4, vdr = 2;
    __device__ __forceinline__ static float dot(const void* row, const block_q8_1* x, int block, int iqs) {
        return vec_dot_q8_0_q8_1(row, x, block, iqs);
    }
};
static_assert(sizeof(block_q4_K) == 144 && sizeof(block_q6_K) == 210 &&
              sizeof(block_q8_0) == 34 && sizeof(block_q8_1) == 36);

// One physical warp owns several rows and retains four separate logical-warp
// accumulation chains, followed by the original ascending additions and XOR
// tree. This also preserves the small-K down projection's zero partial sums.
template<int Type, bool SharedInput, int Rows, bool Pair, bool DevicePlan>
__launch_bounds__(kWarp * kWarps, 1)
__global__ void grouped_warp(DecodeExpertWeights weights, size_t role_offset,
                             const block_q8_1* __restrict__ input, float* __restrict__ output,
                             int input_size, int output_size, const decode_cache::Plan* plan) {
    if constexpr (DevicePlan) if (plan->receipt.status != decode_cache::Ok) return;
    using F = Format<Type>;
    const int n_in = SharedInput ? input_size : kDecodeExpertHidden;
    const int n_out = SharedInput ? output_size : kDecodeExpertInput;
    if constexpr (Pair) {
        role_offset = blockIdx.z == 0 ? weights.gate_offset : weights.up_offset;
        output += size_t(blockIdx.z) * kDecodeExpertCount * n_out;
    }
    const int lane = int(threadIdx.x) & 31;
    const int row0 = (int(blockIdx.x) * kWarps + int(threadIdx.x) / kWarp) * Rows;
    if (row0 >= n_out) return;
    const int expert = int(blockIdx.y), blocks_per_row = n_in / F::elements;
    const auto* w = reinterpret_cast<const typename F::Block*>(
        static_cast<const char*>(DevicePlan ? plan->weights[expert] : weights.expert[expert]) + role_offset);
    const block_q8_1* x = input + (SharedInput ? 0 : expert * (n_in / 32));
    float partial[kWarps][Rows] = {};
    constexpr int stride = kWarp * kWarps / F::threads;
    for (int base = 0; base < blocks_per_row; base += stride) {
#pragma unroll
        for (int warp = 0; warp < kWarps; ++warp) {
            const int block = base + (warp * kWarp + lane) / F::threads;
            if (block < blocks_per_row) {
#pragma unroll
                for (int i = 0; i < Rows; ++i) {
                    const int row = min(row0 + i, n_out - 1);
                    partial[warp][i] += F::dot(w + size_t(row) * blocks_per_row,
                        x + block * (F::elements / 32), block, F::vdr * (lane % F::threads));
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
        float sum = partial[0][i];
#pragma unroll
        for (int warp = 1; warp < kWarps; ++warp) sum += partial[warp][i];
#pragma unroll
        for (int offset = kWarp / 2; offset > 0; offset /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, offset, kWarp);
        if (lane == 0 && row0 + i < n_out) output[expert * n_out + row0 + i] = sum;
    }
}

template<int Type, bool SharedInput, bool Pair, bool DevicePlan>
void launch(const DecodeExpertWeights& weights, size_t offset, const block_q8_1* x,
            float* y, int n_in, int n_out, cudaStream_t stream, const decode_cache::Plan* plan) {
    constexpr int rows = SharedInput ? 2 : (Type == 12 ? 4 : 8);
    grouped_warp<Type, SharedInput, rows, Pair, DevicePlan><<<dim3((n_out + 4 * rows - 1) / (4 * rows),
        kDecodeExpertCount, Pair ? 2 : 1), 128, 0, stream>>>(
        weights, offset, x, y, n_in, n_out, plan);
}

template<bool SharedInput, bool Pair = false, bool DevicePlan = false>
void project(int type, const DecodeExpertWeights& weights, size_t role_offset,
             const block_q8_1* x, float* y, int n_in, int n_out, cudaStream_t stream, const decode_cache::Plan* plan = nullptr) {
    switch (type) {
        case 12: launch<12, SharedInput, Pair, DevicePlan>(weights, role_offset, x, y, n_in, n_out, stream, plan); break;
        case 14: launch<14, SharedInput, Pair, DevicePlan>(weights, role_offset, x, y, n_in, n_out, stream, plan); break;
        case 8: launch<8, SharedInput, Pair, DevicePlan>(weights, role_offset, x, y, n_in, n_out, stream, plan); break;
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace

void decode_experts(const DecodeExpertWeights& weights, const float* input,
                    float* output, void* q8_scratch, float* float_scratch, void* stream, const void* normalized_q8) {
    for (int type : {weights.gate_type, weights.up_type, weights.down_type})
        if (type != 12 && type != 14 && type != 8)
            throw std::invalid_argument("Kolibri grouped decode requires Q4_K, Q6_K or Q8_0 expert roles");
    auto* scratch_q8 = static_cast<block_q8_1*>(q8_scratch);
    const auto* input_q8 = normalized_q8 ? static_cast<const block_q8_1*>(normalized_q8) : scratch_q8;
    auto* hidden_q8 = scratch_q8 + kDecodeExpertInput / 32;
    float* gate = float_scratch;
    float* up = gate + kDecodeExpertCount * kDecodeExpertHidden;
    float* hidden = up + kDecodeExpertCount * kDecodeExpertHidden;
    const auto s = static_cast<cudaStream_t>(stream);
    if (!normalized_q8)
        strata::kernels::native_quantize_q8_1(input, scratch_q8, kDecodeExpertInput, 1, stream);
    if (weights.gate_type == weights.up_type) {
        // Gate and up occupy contiguous scratch planes; each CTA keeps its
        // original dot/reduction while grid.z selects the role.
        project<true, true>(weights.gate_type, weights, weights.gate_offset, input_q8, gate,
                   kDecodeExpertInput, kDecodeExpertHidden, s);
    } else {
        project<true>(weights.gate_type, weights, weights.gate_offset, input_q8, gate,
                   kDecodeExpertInput, kDecodeExpertHidden, s);
        project<true>(weights.up_type, weights, weights.up_offset, input_q8, up,
                   kDecodeExpertInput, kDecodeExpertHidden, s);
    }
    swiglu_quantize_q8_1(gate, up, hidden, hidden_q8, kDecodeExpertCount * kDecodeExpertHidden, stream);
    project<false>(weights.down_type, weights, weights.down_offset, hidden_q8, output,
                    kDecodeExpertHidden, kDecodeExpertInput, s);
}

void decode_experts_device(const DecodeExpertWeights& weights, const decode_cache::Plan* plan, const float* input,
                    float* output, void* q8_scratch, float* float_scratch, void* stream, const void* normalized_q8) {
    for (int type : {weights.gate_type, weights.up_type, weights.down_type})
        if (type != 12 && type != 14 && type != 8)
            throw std::invalid_argument("Kolibri grouped decode requires Q4_K, Q6_K or Q8_0 expert roles");
    auto* scratch_q8 = static_cast<block_q8_1*>(q8_scratch);
    const auto* input_q8 = normalized_q8 ? static_cast<const block_q8_1*>(normalized_q8) : scratch_q8;
    auto* hidden_q8 = scratch_q8 + kDecodeExpertInput / 32;
    float* gate = float_scratch;
    float* up = gate + kDecodeExpertCount * kDecodeExpertHidden;
    float* hidden = up + kDecodeExpertCount * kDecodeExpertHidden;
    const auto s = static_cast<cudaStream_t>(stream);
    if (!normalized_q8)
        strata::kernels::native_quantize_q8_1(input, scratch_q8, kDecodeExpertInput, 1, stream);
    if (weights.gate_type == weights.up_type) {
        // Gate and up occupy contiguous scratch planes; each CTA keeps its
        // original dot/reduction while grid.z selects the role.
        project<true, true, true>(weights.gate_type, weights, weights.gate_offset, input_q8, gate,
                   kDecodeExpertInput, kDecodeExpertHidden, s, plan);
    } else {
        project<true, false, true>(weights.gate_type, weights, weights.gate_offset, input_q8, gate,
                   kDecodeExpertInput, kDecodeExpertHidden, s, plan);
        project<true, false, true>(weights.up_type, weights, weights.up_offset, input_q8, up,
                   kDecodeExpertInput, kDecodeExpertHidden, s, plan);
    }
    swiglu_quantize_q8_1(gate, up, hidden, hidden_q8, kDecodeExpertCount * kDecodeExpertHidden, stream);
    project<false, false, true>(weights.down_type, weights, weights.down_offset, hidden_q8, output,
                    kDecodeExpertHidden, kDecodeExpertInput, s, plan);
}

}  // namespace strata::kolibri::kernels
