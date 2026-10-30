#include "decode_experts.cuh"
#include "decode_cache.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::kolibri::kernels {
namespace {
struct Ranks { int32_t value[kDecodeExpertCount]; };

// This file is compiled without fast math, like the original scatter kernel:
// multiplication retains subnormals before atomicAdd applies its own semantics.
template<bool DevicePlan>
__global__ void combine(const float* outputs, const float* weights, Ranks ranks, float* destination, const decode_cache::Plan* plan) {
    if constexpr (DevicePlan) if (plan->receipt.status != decode_cache::Ok) return;
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= kDecodeExpertInput) return;
    float value = destination[i], product[kDecodeExpertCount];
    bool finite = (__float_as_uint(value) & 0x7fffffffU) < 0x7f800000U;
#pragma unroll
    for (int expert = 0; expert < kDecodeExpertCount; ++expert) {
        product[expert] = __fmul_rn(outputs[expert * kDecodeExpertInput + i], weights[DevicePlan ? plan->ranks[expert] : ranks.value[expert]]);
        finite = finite && ((__float_as_uint(product[expert]) & 0x7fffffffU) < 0x7f800000U);
    }
    // PTX leaves NaN payloads unspecified. Keep the original atomic operation
    // for exceptional values, without writing the destination beforehand.
    if (!finite) {
#pragma unroll
        for (int expert = 0; expert < kDecodeExpertCount; ++expert)
            atomicAdd(destination + i, product[expert]);
        return;
    }
    // One thread owns this feature. Preserve global atomicAdd's RN/FTZ
    // arithmetic and expert order while retaining the partial sum in a register.
#pragma unroll
    for (int expert = 0; expert < kDecodeExpertCount; ++expert)
        asm("add.rn.ftz.f32 %0, %0, %1;" : "+f"(value) : "f"(product[expert]));
    destination[i] = value;
}
}  // namespace

void combine_decode_experts(const float* outputs, const float* route_weights,
                            const int32_t* sorted_ranks, float* destination, void* stream) {
    Ranks ranks;
    for (int i = 0; i < kDecodeExpertCount; ++i) ranks.value[i] = sorted_ranks[i];
    combine<false><<<(kDecodeExpertInput + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        outputs, route_weights, ranks, destination, nullptr);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void combine_decode_experts_device(const float* outputs, const float* route_weights,
                                  const decode_cache::Plan* plan, float* destination, void* stream) {
    combine<true><<<(kDecodeExpertInput + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        outputs, route_weights, Ranks{}, destination, plan);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kolibri::kernels
