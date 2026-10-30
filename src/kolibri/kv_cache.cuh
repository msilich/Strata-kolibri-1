#pragma once

#include "kernels.cuh"

#include <cstddef>
#include <cstdint>

namespace strata::kolibri::kv {

// Strata's symmetric INT8 storage contract: one FP16 scale per 64 values,
// quantization against that stored scale, and round-to-nearest-even codes.
// Kolibri keeps consecutive token/head records instead of QSA's paged pools.
// Aligned records let attention load eight codes with one 64-bit instruction.
struct alignas(16) Int8Head {
    int8_t values[kernels::kHeadDim];
    uint16_t scales[kernels::kHeadDim / 64];
};
static_assert(sizeof(Int8Head) == 144);
inline constexpr size_t Int8CellBytes = kernels::kKvHeads * sizeof(Int8Head);

void append_int8(const float* keys, const float* values, void* kcache, void* vcache,
                 int rows, int offset, void* stream);

}  // namespace strata::kolibri::kv

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace strata::kolibri::kv::detail {

// Exactly 128 threads, one complete KV head. K and V have independent scales.
__device__ __forceinline__ void store_pair(float key, float value, Int8Head* kc, Int8Head* vc) {
    __shared__ float maxima[2][4];
    __shared__ unsigned invalid[2][4];
    const int t = threadIdx.x, warp = t / 32, group = t / 64;
    float km = fabsf(key), vm = fabsf(value);
    const unsigned kb = __ballot_sync(0xffffffffu, !isfinite(key));
    const unsigned vb = __ballot_sync(0xffffffffu, !isfinite(value));
    for (int o = 16; o; o /= 2) {
        km = fmaxf(km, __shfl_xor_sync(0xffffffffu, km, o));
        vm = fmaxf(vm, __shfl_xor_sync(0xffffffffu, vm, o));
    }
    if ((t & 31) == 0) {
        maxima[0][warp] = km; maxima[1][warp] = vm;
        invalid[0][warp] = kb; invalid[1][warp] = vb;
    }
    __syncthreads();
    const float ka = fmaxf(maxima[0][2 * group], maxima[0][2 * group + 1]);
    const float va = fmaxf(maxima[1][2 * group], maxima[1][2 * group + 1]);
    const __half ks = __float2half_rn((invalid[0][2 * group] | invalid[0][2 * group + 1]) ? NAN : ka / 127.f);
    const __half vs = __float2half_rn((invalid[1][2 * group] | invalid[1][2 * group + 1]) ? NAN : va / 127.f);
    const float ksf = __half2float(ks), vsf = __half2float(vs);
    const int kq = ksf > 0.f ? max(-127, min(127, __float2int_rn(key / ksf))) : 0;
    const int vq = vsf > 0.f ? max(-127, min(127, __float2int_rn(value / vsf))) : 0;
    kc->values[t] = (int8_t) kq; vc->values[t] = (int8_t) vq;
    if ((t & 63) == 0) {
        kc->scales[group] = __half_as_ushort(ks);
        vc->scales[group] = __half_as_ushort(vs);
    }
}

__device__ __forceinline__ uint2 half4(uint32_t codes, __half2 scale) {
    // 0x64xx represents 1024 + xx exactly in FP16. Bias signed bytes by 128,
    // then subtract 1152 before scaling. Separate rounded operations preserve
    // FP32(code * stored_scale) -> FP16, including subnormal results.
    const uint32_t biased = codes ^ 0x80808080u;
    const uint32_t lo = __byte_perm(biased, 0x64646464u, 0x5140);
    const uint32_t hi = __byte_perm(biased, 0x64646464u, 0x7362);
    const __half2 bias = __float2half2_rn(1152.f);
    const __half2 a = __hmul2_rn(__hsub2_rn(*reinterpret_cast<const __half2*>(&lo), bias), scale);
    const __half2 b = __hmul2_rn(__hsub2_rn(*reinterpret_cast<const __half2*>(&hi), bias), scale);
    return make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}

__device__ __forceinline__ uint4 load8(const void* cache, size_t head, int piece) {
    const auto* row = static_cast<const Int8Head*>(cache) + head;
    const uint2 codes = __ldg(reinterpret_cast<const uint2*>(row->values + piece * 8));
    const __half2 scale = __half2half2(__ushort_as_half(__ldg(row->scales + piece / 8)));
    const uint2 a = half4(codes.x, scale), b = half4(codes.y, scale);
    return make_uint4(a.x, a.y, b.x, b.y);
}

}  // namespace strata::kolibri::kv::detail
#endif
