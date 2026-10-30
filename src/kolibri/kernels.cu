#include "kernels.cuh"
#include "decode_route.cuh"
#include "kv_cache.cuh"
#define STRATA_Q81_FASTMATH_DIV 1
#include "strata/kernels/q8_1_finite.hpp"
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kolibri::kernels {
namespace {

void checked(const char* operation) {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("Kolibri ") + operation + ": " + cudaGetErrorString(error));
}

__device__ float warp_sum(float x) {
    for (int d = 16; d; d /= 2) x += __shfl_xor_sync(0xffffffffu, x, d);
    return x;
}

__device__ float warp_max(float x) {
    for (int d = 16; d; d /= 2) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
    return x;
}

__global__ void norm_kernel(const float* x, const float* w, float* out, int width, float eps) {
    __shared__ float sums[256];
    const int i = threadIdx.x;
    const size_t row = (size_t) blockIdx.x * width;
    float sum = 0.f;
    for (int j = i; j < width; j += 256) sum += x[row + j] * x[row + j];
    sums[i] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (i < d) sums[i] += sums[i + d];
        __syncthreads();
    }
    const float scale = rsqrtf(sums[0] / width + eps);
    for (int j = i; j < width; j += 256) out[row + j] = x[row + j] * scale * w[j];
}

__global__ void add_kernel(const float* a, const float* b, float* out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void qk_kernel(float* q, float* k, const float* qw, const float* kw,
                          int64_t first_pos, bool local, float theta, float eps) {
    __shared__ float sums[kHeadDim], normalized[kHeadDim];
    const int i = threadIdx.x, h = blockIdx.y, row = blockIdx.x;
    const bool query = h < kQueryHeads;
    float* x = query ? q + ((size_t) row * kQueryHeads + h) * kHeadDim
                     : k + ((size_t) row * kKvHeads + h - kQueryHeads) * kHeadDim;
    const float* w = query ? qw : kw;
    const float value = x[i];
    sums[i] = value * value;
    __syncthreads();
    for (int d = kHeadDim / 2; d; d /= 2) {
        if (i < d) sums[i] += sums[i + d];
        __syncthreads();
    }
    normalized[i] = value * rsqrtf(sums[0] / kHeadDim + eps) * w[i];
    __syncthreads();
    float result = normalized[i];
    if (local) {
        // Compute the angle in double so large absolute positions retain their
        // precision even when the surrounding engine enables fast math.
        const int p = i % (kHeadDim / 2);
        const double angle = (double) (first_pos + row) * pow((double) theta, -2.0 * p / kHeadDim);
        double sine, cosine;
        sincos(angle, &sine, &cosine);
        const float a = normalized[p], b = normalized[p + kHeadDim / 2];
        result = i < kHeadDim / 2 ? a * (float) cosine - b * (float) sine
                                 : b * (float) cosine + a * (float) sine;
    }
    x[i] = result;
}

// Generate the same double-precision angles as qk_kernel, once per forward
// position/pair rather than once for each head and each local layer.
__global__ void rope_table_kernel(float* sin_cos, int rows, int64_t first_pos, float theta) {
    const int pair = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= rows * (kHeadDim / 2)) return;
    const int row = pair / (kHeadDim / 2), p = pair % (kHeadDim / 2);
    const double angle = (double) (first_pos + row) * pow((double) theta, -2.0 * p / kHeadDim);
    double sine, cosine;
    sincos(angle, &sine, &cosine);
    sin_cos[2 * pair] = (float) sine;
    sin_cos[2 * pair + 1] = (float) cosine;
}

__global__ void rope_table_decode_graph_kernel(float* sin_cos, const int32_t* position, float theta) {
    const int pair = threadIdx.x;
    if (pair >= kHeadDim / 2) return;
    const double angle = (double) *position * pow((double) theta, -2.0 * pair / kHeadDim);
    double sine, cosine;
    sincos(angle, &sine, &cosine);
    sin_cos[2 * pair] = (float) sine;
    sin_cos[2 * pair + 1] = (float) cosine;
}

__global__ void qk_cached_kernel(float* q, float* k, const float* qw, const float* kw,
                                 const float* sin_cos, float eps) {
    // Keep qk_kernel's reduction and normalization order unchanged.
    __shared__ float sums[kHeadDim], normalized[kHeadDim];
    const int i = threadIdx.x, h = blockIdx.y, row = blockIdx.x;
    const bool query = h < kQueryHeads;
    float* x = query ? q + ((size_t) row * kQueryHeads + h) * kHeadDim
                     : k + ((size_t) row * kKvHeads + h - kQueryHeads) * kHeadDim;
    const float* w = query ? qw : kw;
    const float value = x[i];
    sums[i] = value * value;
    __syncthreads();
    for (int d = kHeadDim / 2; d; d /= 2) {
        if (i < d) sums[i] += sums[i + d];
        __syncthreads();
    }
    normalized[i] = value * rsqrtf(sums[0] / kHeadDim + eps) * w[i];
    __syncthreads();
    const int p = i % (kHeadDim / 2);
    const float sine = sin_cos[row * kHeadDim + 2 * p];
    const float cosine = sin_cos[row * kHeadDim + 2 * p + 1];
    const float a = normalized[p], b = normalized[p + kHeadDim / 2];
    x[i] = i < kHeadDim / 2 ? a * cosine - b * sine : b * cosine + a * sine;
}

// T=1 only. Preserve the original norm/RoPE arithmetic and write the same
// F32 Q/K values while the four K heads also append their F16 K/V values.
template <bool Local, bool DevicePosition = false, bool Int8 = false>
__global__ void qk_append_decode_kernel(float* q, float* k, const float* v,
                                        const float* qw, const float* kw, const float* sin_cos,
                                        __half* kc, __half* vc, size_t offset, float eps,
                                        const int32_t* position = nullptr) {
    if constexpr (DevicePosition) offset = (size_t) *position * kKvHeads * kHeadDim;
    __shared__ float sums[kHeadDim], normalized[kHeadDim];
    const int i = threadIdx.x, h = blockIdx.y;
    const bool query = h < kQueryHeads;
    float* x = query ? q + h * kHeadDim : k + (h - kQueryHeads) * kHeadDim;
    const float* w = query ? qw : kw;
    const float value = x[i];
    sums[i] = value * value;
    __syncthreads();
    if (i < 32) {
        // The same d=64,32 pairs, followed by the original d=16..1 tree.
        float partial = __fadd_rn(__fadd_rn(sums[i], sums[i + 64]),
                                  __fadd_rn(sums[i + 32], sums[i + 96]));
        for (int d = 16; d; d /= 2)
            partial = __fadd_rn(partial, __shfl_down_sync(0xffffffffu, partial, d));
        if (i == 0) sums[0] = partial;
    }
    __syncthreads();
    normalized[i] = value * rsqrtf(sums[0] / kHeadDim + eps) * w[i];
    __syncthreads();
    float result = normalized[i];
    if constexpr (Local) {
        const int p = i % (kHeadDim / 2);
        const float sine = sin_cos[2 * p], cosine = sin_cos[2 * p + 1];
        const float a = normalized[p], b = normalized[p + kHeadDim / 2];
        result = i < kHeadDim / 2 ? a * cosine - b * sine : b * cosine + a * sine;
    }
    x[i] = result;
    if (!query) {
        const size_t j = (size_t) (h - kQueryHeads) * kHeadDim + i;
        if constexpr (Int8) {
            const size_t head = offset / kHeadDim + h - kQueryHeads;
            kv::detail::store_pair(result, v[j], reinterpret_cast<kv::Int8Head*>(kc) + head,
                                    reinterpret_cast<kv::Int8Head*>(vc) + head);
        } else {
            kc[offset + j] = __float2half_rn(result);
            vc[offset + j] = __float2half_rn(v[j]);
        }
    }
}


__global__ void append_kernel(const float* k, const float* v, __half* kc, __half* vc, size_t n, size_t offset) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        kc[offset + i] = __float2half_rn(k[i]);
        vc[offset + i] = __float2half_rn(v[i]);
    }
}

// One block per query/head; 32-key tiles and online softmax keep storage bounded
// at long context. Every softmax reduction and weighted sum stays in F32.
__global__ void attention_kernel(const float* q, const __half* kc, const __half* vc, float* out,
                                 int key_count, int64_t key_first, int64_t query_first, bool local) {
    __shared__ float scores[32], tile_max, tile_sum;
    const int i = threadIdx.x, lane = i % 32, warp = i / 32;
    const int head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const int64_t position = query_first + blockIdx.x;
    const int64_t first = local && position >= kWindow ? position - (kWindow - 1) : 0;
    const int begin = first > key_first ? (int) (first - key_first) : 0;
    const int end = (int) ((position - key_first + 1) < key_count ? (position - key_first + 1) : key_count);
    const size_t qbase = ((size_t) blockIdx.x * kQueryHeads + head) * kHeadDim;
    float qr[4];
    for (int d = 0; d < 4; ++d) qr[d] = q[qbase + lane + 32 * d];
    float maximum = -INFINITY, denominator = 0.f, accumulator = 0.f;
    for (int start = begin; start < end; start += 32) {
        for (int key = warp; key < 32; key += 4) {
            float dot = 0.f;
            if (start + key < end) {
                const size_t kbase = ((size_t) (start + key) * kKvHeads + kv_head) * kHeadDim;
                for (int d = 0; d < 4; ++d) dot += qr[d] * __half2float(kc[kbase + lane + 32 * d]);
            }
            dot = warp_sum(dot);
            if (lane == 0) scores[key] = start + key < end ? dot * 0.08838834764831845f : -INFINITY;
        }
        __syncthreads();
        if (warp == 0) {
            const float m = warp_max(scores[lane]);
            const float weight = expf(scores[lane] - m);
            const float sum = warp_sum(weight);
            scores[lane] = weight;
            if (lane == 0) { tile_max = m; tile_sum = sum; }
        }
        __syncthreads();
        const float next_max = fmaxf(maximum, tile_max);
        const float old_scale = expf(maximum - next_max), new_scale = expf(tile_max - next_max);
        float weighted = 0.f;
        const int count = end - start < 32 ? end - start : 32;
        for (int key = 0; key < count; ++key) {
            const size_t vbase = ((size_t) (start + key) * kKvHeads + kv_head) * kHeadDim;
            weighted += scores[key] * __half2float(vc[vbase + i]);
        }
        accumulator = accumulator * old_scale + weighted * new_scale;
        denominator = denominator * old_scale + tile_sum * new_scale;
        maximum = next_max;
        __syncthreads();
    }
    out[qbase + i] = accumulator / denominator;
}

constexpr int kAttentionSplitKeys = 512;
constexpr int kLocalAttentionSplitKeys = 128;
constexpr int kAttentionPartial = kHeadDim + 2;

// The baseline's F32 32-key tile equations, with a bounded independent key range.
// Keep the baseline kernel intact so opting out retains its launch and arithmetic.
template <int SplitKeys>
__global__ void attention_split_kernel(const float* q, const __half* kc, const __half* vc,
                                       float* scratch, int key_count, int parts) {
    __shared__ float scores[32], tile_max, tile_sum;
    const int i = threadIdx.x, lane = i % 32, warp = i / 32;
    const int head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const int begin = blockIdx.x * SplitKeys;
    const int end = min(begin + SplitKeys, key_count);
    const size_t qbase = (size_t) head * kHeadDim;
    float qr[4];
    for (int d = 0; d < 4; ++d) qr[d] = q[qbase + lane + 32 * d];
    float maximum = -INFINITY, denominator = 0.f, accumulator = 0.f;
    for (int start = begin; start < end; start += 32) {
        for (int key = warp; key < 32; key += 4) {
            float dot = 0.f;
            if (start + key < end) {
                const size_t kbase = ((size_t) (start + key) * kKvHeads + kv_head) * kHeadDim;
                for (int d = 0; d < 4; ++d) dot += qr[d] * __half2float(kc[kbase + lane + 32 * d]);
            }
            dot = warp_sum(dot);
            if (lane == 0) scores[key] = start + key < end ? dot * 0.08838834764831845f : -INFINITY;
        }
        __syncthreads();
        if (warp == 0) {
            const float m = warp_max(scores[lane]);
            const float weight = expf(scores[lane] - m);
            const float sum = warp_sum(weight);
            scores[lane] = weight;
            if (lane == 0) { tile_max = m; tile_sum = sum; }
        }
        __syncthreads();
        const float next_max = fmaxf(maximum, tile_max);
        const float old_scale = expf(maximum - next_max), new_scale = expf(tile_max - next_max);
        float weighted = 0.f;
        const int count = end - start < 32 ? end - start : 32;
        for (int key = 0; key < count; ++key) {
            const size_t vbase = ((size_t) (start + key) * kKvHeads + kv_head) * kHeadDim;
            weighted += scores[key] * __half2float(vc[vbase + i]);
        }
        accumulator = accumulator * old_scale + weighted * new_scale;
        denominator = denominator * old_scale + tile_sum * new_scale;
        maximum = next_max;
        __syncthreads();
    }
    float* partial = scratch + ((size_t) head * parts + blockIdx.x) * kAttentionPartial;
    partial[i] = accumulator;
    if (i == 0) { partial[kHeadDim] = maximum; partial[kHeadDim + 1] = denominator; }
}

__device__ __forceinline__ float attention_merge_value(const float* scratch, int parts) {
    extern __shared__ float scales[];
    __shared__ float reduced[4], maximum, denominator;
    const int i = threadIdx.x, lane = i % 32, warp = i / 32;
    const float* partials = scratch + (size_t) blockIdx.x * parts * kAttentionPartial;
    float m = -INFINITY;
    for (int part = i; part < parts; part += kHeadDim)
        m = fmaxf(m, partials[(size_t) part * kAttentionPartial + kHeadDim]);
    m = warp_max(m);
    if (lane == 0) reduced[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = warp_max(lane < 4 ? reduced[lane] : -INFINITY);
        if (lane == 0) maximum = m;
    }
    __syncthreads();
    float sum = 0.f;
    for (int part = i; part < parts; part += kHeadDim) {
        const float* partial = partials + (size_t) part * kAttentionPartial;
        const float scale = expf(partial[kHeadDim] - maximum);
        scales[part] = scale;
        sum += partial[kHeadDim + 1] * scale;
    }
    sum = warp_sum(sum);
    if (lane == 0) reduced[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = warp_sum(lane < 4 ? reduced[lane] : 0.f);
        if (lane == 0) denominator = sum;
    }
    __syncthreads();
    float result = 0.f;
    for (int part = 0; part < parts; ++part)
        result += partials[(size_t) part * kAttentionPartial + i] * scales[part];
    return result / denominator;
}

__global__ void attention_split_merge(const float* scratch, float* out, int parts) {
    const float value = attention_merge_value(scratch, parts);
    out[(size_t) blockIdx.x * kHeadDim + threadIdx.x] = value;
}

__global__ void route_kernel(const float* logits, const float* bias, int32_t* ids, float* weights) {
    __shared__ float values[128];
    __shared__ int indices[128];
    const int lane = threadIdx.x, row = blockIdx.x;
    const float* input = logits + (size_t) row * kExperts;
    float candidates[3];
    for (int j = 0; j < 3; ++j) candidates[j] = input[lane + 128 * j] + bias[lane + 128 * j];
    for (int rank = 0; rank < kTopK; ++rank) {
        float best = -INFINITY;
        int id = kExperts;
        for (int j = 0; j < 3; ++j) {
            const int candidate_id = lane + 128 * j;
            if (candidates[j] > best || (candidates[j] == best && candidate_id < id)) {
                best = candidates[j]; id = candidate_id;
            }
        }
        values[lane] = best; indices[lane] = id;
        __syncthreads();
        for (int d = 64; d; d /= 2) {
            if (lane < d && (values[lane + d] > values[lane] ||
                            (values[lane + d] == values[lane] && indices[lane + d] < indices[lane]))) {
                values[lane] = values[lane + d]; indices[lane] = indices[lane + d];
            }
            __syncthreads();
        }
        const int selected = indices[0];
        if (lane == 0) {
            const float value = input[selected];
            const float e = expf(value >= 0.f ? -value : value);
            ids[row * kTopK + rank] = selected;
            weights[row * kTopK + rank] = value >= 0.f ? 1.f / (1.f + e) : e / (1.f + e);
        }
        for (int j = 0; j < 3; ++j) if (lane + 128 * j == selected) candidates[j] = -INFINITY;
        __syncthreads();
    }
}

// Single decode row; the value/ID comparator matches route_kernel. NaN
// candidates never replace the sentinel; masking remains -INFINITY, including
// the original repeated-ID behavior when all remaining values are -INFINITY.
__global__ void route_decode_kernel(const float* logits, const float* bias, int32_t* ids, float* weights) {
    detail::route_decode_body(logits, bias, ids, weights);
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (gate[i] / (1.f + expf(-gate[i]))) * up[i];
}

__device__ __forceinline__ float quantizer_ftz(float x) {
    float result;
    asm("mul.rn.ftz.f32 %0, %1, 0f3F800000;" : "=f"(result) : "f"(x));
    return result;
}
__global__ void swiglu_q8_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                      float* __restrict__ hidden, block_q8_1* __restrict__ q8, int count) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i >= count) return; // Only whole warps return.
    const float xi = (gate[i] / (1.f + expf(-gate[i]))) * up[i];
    hidden[i] = xi;
    float amax, sum = xi;
    asm("abs.ftz.f32 %0, %1;" : "=f"(amax) : "f"(xi));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffff, amax, offset, 32);
        asm("max.ftz.f32 %0, %1, %2;" : "=f"(amax) : "f"(amax), "f"(other));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffff, sum, offset, 32);
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(sum) : "f"(sum), "f"(other));
    }
    // The producer must retain subnormals, but the separate native quantizer
    // is compiled with --use_fast_math. Flush only at that quantizer boundary.
    const float d = strata::kernels::q8_1_finite(quantizer_ftz(strata::kernels::q81_div(amax, 127.f)));
    q8[i / 32].qs[i % 32] = strata::kernels::q8_1_quant(quantizer_ftz(xi), d, amax);
    if (i % 32 == 0) q8[i / 32].ds = strata::kernels::q8_1_ds(d, sum);
}

template<bool DevicePosition = false>
__global__ void attention_split_merge_q8(const float* scratch, float* out, block_q8_1* q8, int parts,
                                         const int32_t* position = nullptr, int split = 0) {
    if constexpr (DevicePosition) parts = (*position + 1 + split - 1) / split;
    const int i = threadIdx.x;
    const float xi = attention_merge_value(scratch, parts);
    out[(size_t) blockIdx.x * kHeadDim + i] = xi;
    q8 += (size_t) blockIdx.x * (kHeadDim / 32);
    float amax, sum = xi;
    asm("abs.ftz.f32 %0, %1;" : "=f"(amax) : "f"(xi));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffff, amax, offset, 32);
        asm("max.ftz.f32 %0, %1, %2;" : "=f"(amax) : "f"(amax), "f"(other));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffff, sum, offset, 32);
        asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(sum) : "f"(sum), "f"(other));
    }
    // The producer must retain subnormals, but the separate native quantizer
    // is compiled with --use_fast_math. Flush only at that quantizer boundary.
    const float d = strata::kernels::q8_1_finite(quantizer_ftz(strata::kernels::q81_div(amax, 127.f)));
    q8[i / 32].qs[i % 32] = strata::kernels::q8_1_quant(quantizer_ftz(xi), d, amax);
    if (i % 32 == 0) q8[i / 32].ds = strata::kernels::q8_1_ds(d, sum);
}

__global__ void gather_kernel(const float* x, const int32_t* ids, float* out, int width, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[(size_t) ids[i / width] * width + i % width];
}

__global__ void scatter_kernel(const float* x, const int32_t* ids, const float* weights, float* out, int width, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out + (size_t) ids[i / width] * width + i % width, x[i] * weights[i / width]);
}

}  // namespace

void rms_norm(const float* x, const float* w, float* out, int rows, int width, float eps, void* stream) {
    if (rows < 1 || width < 1 || eps <= 0.f) throw std::invalid_argument("Kolibri RMS norm: invalid shape or epsilon");
    norm_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, w, out, width, eps);
    checked("RMS norm");
}

void add(const float* a, const float* b, float* out, size_t n, void* stream) {
    if (!n) return;
    add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(a, b, out, n);
    checked("residual add");
}

void qk_norm_rope(float* q, float* k, const float* qw, const float* kw, int rows, int64_t first_pos,
                  bool local, float theta, float eps, void* stream) {
    if (rows < 1 || rows > kChunk || first_pos < 0 || theta <= 0.f || eps <= 0.f)
        throw std::invalid_argument("Kolibri QK norm: invalid shape, position or norm parameters");
    qk_kernel<<<dim3(rows, kQueryHeads + kKvHeads), kHeadDim, 0, (cudaStream_t) stream>>>(
        q, k, qw, kw, first_pos, local, theta, eps);
    checked("QK norm/RoPE");
}

void rope_table(float* sin_cos, int rows, int64_t first_pos, float theta, void* stream) {
    if (rows < 1 || rows > kChunk || first_pos < 0 || theta <= 0.f)
        throw std::invalid_argument("Kolibri RoPE table: invalid shape, position or theta");
    rope_table_kernel<<<(rows * (kHeadDim / 2) + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        sin_cos, rows, first_pos, theta);
    checked("RoPE table");
}

void rope_table_decode_graph(float* sin_cos, const int32_t* position, float theta, void* stream) {
    rope_table_decode_graph_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(sin_cos, position, theta);
    checked("graph RoPE table");
}

void qk_norm_rope_cached(float* q, float* k, const float* qw, const float* kw, int rows,
                         const float* sin_cos, float eps, void* stream) {
    if (rows < 1 || rows > kChunk || eps <= 0.f)
        throw std::invalid_argument("Kolibri cached QK norm: invalid shape or epsilon");
    qk_cached_kernel<<<dim3(rows, kQueryHeads + kKvHeads), kHeadDim, 0, (cudaStream_t) stream>>>(
        q, k, qw, kw, sin_cos, eps);
    checked("cached QK norm/RoPE");
}

void qk_norm_rope_append_decode(float* q, float* k, const float* v, const float* qw, const float* kw,
                                const float* sin_cos, bool local, uint16_t* kc, uint16_t* vc,
                                int dst_offset, float eps, void* stream) {
    if (dst_offset < 0 || eps <= 0.f || (local && !sin_cos))
        throw std::invalid_argument("Kolibri decode QK/KV: invalid offset, epsilon or RoPE table");
    const dim3 grid(1, kQueryHeads + kKvHeads);
    const size_t offset = (size_t) dst_offset * kKvHeads * kHeadDim;
    if (local)
        qk_append_decode_kernel<true><<<grid, kHeadDim, 0, (cudaStream_t) stream>>>(
            q, k, v, qw, kw, sin_cos, (__half*) kc, (__half*) vc, offset, eps);
    else
        qk_append_decode_kernel<false><<<grid, kHeadDim, 0, (cudaStream_t) stream>>>(
            q, k, v, qw, kw, sin_cos, (__half*) kc, (__half*) vc, offset, eps);
    checked("decode QK norm/RoPE and KV append");
}


void qk_norm_append_decode_graph(float* q, float* k, const float* v, const float* qw, const float* kw,
                                uint16_t* kc, uint16_t* vc, const int32_t* position, float eps, void* stream) {
    qk_append_decode_kernel<false, true><<<dim3(1, kQueryHeads + kKvHeads), kHeadDim, 0, (cudaStream_t) stream>>>(
        q, k, v, qw, kw, nullptr, (__half*) kc, (__half*) vc, 0, eps, position);
    checked("graph QK norm and KV append");
}

void qk_norm_append_decode_int8(float* q, float* k, const float* v, const float* qw, const float* kw,
                               void* kc, void* vc, int offset, const int32_t* position, float eps, void* stream) {
    const dim3 grid(1, kQueryHeads + kKvHeads);
    if (position)
        qk_append_decode_kernel<false, true, true><<<grid, kHeadDim, 0, (cudaStream_t) stream>>>(
            q, k, v, qw, kw, nullptr, (__half*) kc, (__half*) vc, 0, eps, position);
    else
        qk_append_decode_kernel<false, false, true><<<grid, kHeadDim, 0, (cudaStream_t) stream>>>(
            q, k, v, qw, kw, nullptr, (__half*) kc, (__half*) vc,
            (size_t) offset * kKvHeads * kHeadDim, eps);
    checked("INT8 QK norm and KV append");
}

void append_kv(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int rows, int dst_offset, void* stream) {
    if (rows < 1 || rows > kChunk || dst_offset < 0) throw std::invalid_argument("Kolibri KV append: invalid shape or offset");
    const size_t n = (size_t) rows * kKvHeads * kHeadDim;
    append_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        k, v, (__half*) kc, (__half*) vc, n, (size_t) dst_offset * kKvHeads * kHeadDim);
    checked("KV append");
}

void attention(const float* q, const uint16_t* kc, const uint16_t* vc, float* out, int rows, int key_count,
               int64_t key_first, int64_t query_first, bool local, void* stream) {
    const int64_t needed_first = local && query_first >= kWindow ? query_first - (kWindow - 1) : 0;
    if (rows < 1 || rows > kChunk || key_count < 1 || key_first < 0 || query_first < key_first ||
        query_first + rows > key_first + key_count || key_first > needed_first || (!local && key_first != 0))
        throw std::invalid_argument("Kolibri attention: incomplete causal KV range or invalid shape");
    attention_kernel<<<dim3(rows, kQueryHeads), kHeadDim, 0, (cudaStream_t) stream>>>(
        q, (const __half*) kc, (const __half*) vc, out, key_count, key_first, query_first, local);
    checked("attention");
}

size_t attention_split_scratch_floats(int key_count) {
    if (key_count < 1 || key_count > 262144)
        throw std::invalid_argument("Kolibri split attention: key count must be in 1..262144");
    return (size_t) ((key_count - 1) / kAttentionSplitKeys + 1) * kQueryHeads * kAttentionPartial;
}

void attention_split_decode(const float* q, const uint16_t* kc, const uint16_t* vc, float* out,
                            int key_count, int64_t query_position, float* scratch, void* stream) {
    attention_split_scratch_floats(key_count);
    if (query_position < 0 || query_position >= key_count || !scratch)
        throw std::invalid_argument("Kolibri split attention: invalid query position or missing scratch");
    const int causal_count = (int) query_position + 1;
    const int parts = (causal_count - 1) / kAttentionSplitKeys + 1;
    attention_split_kernel<kAttentionSplitKeys><<<dim3(parts, kQueryHeads), kHeadDim, 0, (cudaStream_t) stream>>>(
        q, (const __half*) kc, (const __half*) vc, scratch, causal_count, parts);
    checked("split attention");
    attention_split_merge<<<kQueryHeads, kHeadDim, (size_t) parts * sizeof(float), (cudaStream_t) stream>>>(
        scratch, out, parts);
    checked("split attention merge");
}

size_t attention_local_split_scratch_floats() {
    return (size_t) ((kWindow - 1) / kLocalAttentionSplitKeys + 1) * kQueryHeads * kAttentionPartial;
}

void attention_local_split_decode(const float* q, const uint16_t* kc, const uint16_t* vc,
                                  float* out, float* scratch, void* stream) {
    if (!scratch) throw std::invalid_argument("Kolibri local split attention: missing scratch");
    constexpr int parts = (kWindow - 1) / kLocalAttentionSplitKeys + 1;
    attention_split_kernel<kLocalAttentionSplitKeys><<<dim3(parts, kQueryHeads), kHeadDim, 0, (cudaStream_t) stream>>>(
        q, (const __half*) kc, (const __half*) vc, scratch, kWindow, parts);
    checked("local split attention");
    attention_split_merge<<<kQueryHeads, kHeadDim, (size_t) parts * sizeof(float), (cudaStream_t) stream>>>(
        scratch, out, parts);
    checked("local split attention merge");
}

void attention_merge_partials(const float* scratch, float* out, int parts, void* stream) {
    attention_split_merge<<<kQueryHeads, kHeadDim, (size_t) parts * sizeof(float), (cudaStream_t) stream>>>(
        scratch, out, parts);
    checked("tensor attention merge");
}

void attention_merge_partials_q8(const float* scratch, float* out, void* q8, int parts, void* stream) {
    if (!q8) throw std::invalid_argument("Kolibri merge Q8 requires output scratch");
    attention_split_merge_q8<><<<kQueryHeads, kHeadDim, (size_t) parts * sizeof(float), (cudaStream_t) stream>>>(
        scratch, out, static_cast<block_q8_1*>(q8), parts);
    checked("attention merge Q8");
}

void attention_merge_partials_q8_graph(const float* scratch, float* out, void* q8,
                                       const int32_t* position, int split, int max_parts, void* stream) {
    attention_split_merge_q8<true><<<kQueryHeads, kHeadDim, (size_t) max_parts * sizeof(float), (cudaStream_t) stream>>>(
        scratch, out, static_cast<block_q8_1*>(q8), 0, position, split);
    checked("graph attention merge Q8");
}

void route(const float* logits, const float* bias, int32_t* ids, float* weights, int rows, void* stream) {
    if (rows < 1) throw std::invalid_argument("Kolibri routing: empty input");
    route_kernel<<<rows, 128, 0, (cudaStream_t) stream>>>(logits, bias, ids, weights);
    checked("routing");
}

void route_decode(const float* logits, const float* bias, int32_t* ids, float* weights, void* stream) {
    route_decode_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(logits, bias, ids, weights);
    checked("decode routing");
}

void swiglu(const float* gate, const float* up, float* out, int rows, int width, void* stream) {
    if (rows < 1 || width < 1) throw std::invalid_argument("Kolibri SwiGLU: invalid shape");
    const size_t n = (size_t) rows * width;
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gate, up, out, n);
    checked("SwiGLU");
}

void swiglu_quantize_q8_1(const float* gate, const float* up, float* hidden,
                         void* q8, int count, void* stream) {
    if (count <= 0 || count % 32) throw std::invalid_argument("Kolibri SwiGLU Q8: invalid block width");
    swiglu_q8_kernel<<<(count + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        gate, up, hidden, static_cast<block_q8_1*>(q8), count);
    checked("SwiGLU Q8");
}

void gather_rows(const float* x, const int32_t* ids, float* out, int rows, int width, void* stream) {
    if (rows < 1 || width < 1) throw std::invalid_argument("Kolibri gather: invalid shape");
    const size_t n = (size_t) rows * width;
    gather_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, ids, out, width, n);
    checked("gather");
}

void scatter_add_weighted(const float* x, const int32_t* ids, const float* weights, float* out,
                          int rows, int width, void* stream) {
    if (rows < 1 || width < 1) throw std::invalid_argument("Kolibri scatter: invalid shape");
    const size_t n = (size_t) rows * width;
    scatter_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, ids, weights, out, width, n);
    checked("weighted scatter");
}

}  // namespace strata::kolibri::kernels
