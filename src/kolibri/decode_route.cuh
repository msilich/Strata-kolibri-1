// Internal one-warp router body shared by standalone and cache-plan decode.
#pragma once
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace strata::kolibri::kernels::detail {
// Accurate math; preserve candidate comparisons, masking, and sigmoid order.
__device__ __forceinline__ void route_decode_body(const float* logits, const float* bias,
                                                  int32_t* ids, float* weights) {
    const int lane = threadIdx.x;
    float candidates[kExperts / 32];
#pragma unroll
    for (int j = 0; j < kExperts / 32; ++j) candidates[j] = logits[lane + 32 * j] + bias[lane + 32 * j];
    int selected_for_lane = 0;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        float best = -INFINITY;
        int id = kExperts;
#pragma unroll
        for (int j = 0; j < kExperts / 32; ++j) {
            const int candidate_id = lane + 32 * j;
            if (candidates[j] > best || (candidates[j] == best && candidate_id < id)) {
                best = candidates[j]; id = candidate_id;
            }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, id, mask, 32);
            if (other > best || (other == best && other_id < id)) { best = other; id = other_id; }
        }
        if (lane == rank) selected_for_lane = id;
#pragma unroll
        for (int j = 0; j < kExperts / 32; ++j)
            if (lane + 32 * j == id) candidates[j] = -INFINITY;
    }
    if (lane < kTopK) {
        const float value = logits[selected_for_lane];
        const float e = expf(value >= 0.f ? -value : value);
        ids[lane] = selected_for_lane;
        weights[lane] = value >= 0.f ? 1.f / (1.f + e) : e / (1.f + e);
    }
}
}  // namespace strata::kolibri::kernels::detail
