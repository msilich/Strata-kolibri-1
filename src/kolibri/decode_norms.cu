#include "decode_norms.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace strata::kolibri::kernels {
namespace {
constexpr int Width = 2560, Threads = 256, Values = Width / Threads;


// The old shared tree first joins lanes 128, 64 and 32 apart. Warp zero can
// perform those same additions privately, then finish its identical lane-zero
// tree with shuffles. The caller synchronizes before reading and after scale.
__device__ __forceinline__ float block_sum_lane_zero(const float* sums) {
    const int lane = threadIdx.x;
    const float a = __fadd_rn(sums[lane], sums[lane + 128]);
    const float b = __fadd_rn(sums[lane + 32], sums[lane + 160]);
    const float c = __fadd_rn(sums[lane + 64], sums[lane + 192]);
    const float d = __fadd_rn(sums[lane + 96], sums[lane + 224]);
    float sum = __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
    for (int offset = 16; offset; offset /= 2)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, offset));
    return sum;
}

__global__ void residual_norm_kernel(const float* __restrict__ update, const float* __restrict__ post_weight,
                                     float* __restrict__ residual, const float* __restrict__ next_weight,
                                     float* __restrict__ normalized, float eps) {
    __shared__ float sums[Threads], post_scale, next_scale;
    const int t = threadIdx.x;
    float value[Values];
    float sum = 0.f;
    for (int j = 0; j < Values; ++j) {
        value[j] = update[t + j * Threads];
        // norm_kernel's sum += x*x contracts to one FMA. Keep its order.
        sum = __fmaf_rn(value[j], value[j], sum);
    }
    sums[t] = sum;
    __syncthreads();
    if (t < 32) {
        const float total = block_sum_lane_zero(sums);
        if (t == 0) post_scale = rsqrtf(__fadd_rn(__fdiv_rn(total, (float) Width), eps));
    }
    __syncthreads();
    sum = 0.f;
    for (int j = 0; j < Values; ++j) {
        const int i = t + j * Threads;
        // These products and the addition were separate kernels originally.
        const float post = __fmul_rn(__fmul_rn(value[j], post_scale), post_weight[i]);
        value[j] = __fadd_rn(residual[i], post);
        residual[i] = value[j];
        sum = __fmaf_rn(value[j], value[j], sum);
    }
    sums[t] = sum;
    __syncthreads();
    if (t < 32) {
        const float total = block_sum_lane_zero(sums);
        if (t == 0) next_scale = rsqrtf(__fadd_rn(__fdiv_rn(total, (float) Width), eps));
    }
    __syncthreads();
    for (int j = 0; j < Values; ++j) {
        const int i = t + j * Threads;
        normalized[i] = __fmul_rn(__fmul_rn(value[j], next_scale), next_weight[i]);
    }
}
}  // namespace

void decode_residual_norm(const float* update, const float* post_weight,
                          float* residual, const float* next_weight,
                          float* normalized, float eps, void* stream) {
    residual_norm_kernel<<<1, Threads, 0, (cudaStream_t) stream>>>(
        update, post_weight, residual, next_weight, normalized, eps);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("Kolibri decode residual norm: ") + cudaGetErrorString(error));
}
}  // namespace strata::kolibri::kernels
