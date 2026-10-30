// Kolibri global prefill: four queries' 48 grouped heads fill three MMA row
// tiles. Scores, online softmax and output accumulators stay in registers.
#include "kernels.cuh"
#include "kv_cache.cuh"
#include "prefill_attention.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::kolibri::kernels {
namespace {
constexpr int HD = kHeadDim, G = kQueryHeads / kKvHeads;
constexpr int QB = 4, CH = 32, STRIDE = HD, THREADS = QB * G / 16 * 32;
static_assert(HD == 128 && G == 12 && kKvHeads == 4, "Kolibri packed attention geometry");
__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}
__device__ __forceinline__ uint32_t pack(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
// Eight-element vectors remain aligned while consecutive ldmatrix rows use
// different shared-memory banks, without a padded row stride.
__device__ __forceinline__ int swizzle(int row, int dim) {
    return dim ^ ((row & 7) * 8);
}
__device__ __forceinline__ void load_matrix(uint32_t* r, const __half* p, bool transpose) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t address = (uint32_t)__cvta_generic_to_shared(p);
    if (transpose)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                     : "=r"(r[0]), "=r"(r[1])
                     : "r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(r[0]), "=r"(r[1])
                     : "r"(address));
#else
    __trap();
#endif
}
__global__ void __launch_bounds__(THREADS, 3)
    packed_prefill_kernel(const float* __restrict__ q, const __half* __restrict__ keys,
                          const __half* __restrict__ values, float* __restrict__ output, int rows,
                          int first_query) {
    constexpr int NW = QB * G / 16, NT = NW * 32, NB = CH / 8;
    __shared__ __align__(16) __half sk[CH][STRIDE], sv[CH][STRIDE];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int g = lane >> 2, c = lane & 3, kvh = blockIdx.y;
    const int qr = blockIdx.x * QB;
    const int rr[2] = {warp * 16 + g, warp * 16 + g + 8};
    const int qi[2] = {qr + rr[0] / G, qr + rr[1] / G};
    const int head[2] = {rr[0] % G, rr[1] % G};
    float qm[2] = {};
    for (int d = c; d < HD; d += 4) {
#pragma unroll
        for (int r = 0; r < 2; r++)
            if (qi[r] < rows)
                qm[r] = fmaxf(qm[r], fabsf(q[((size_t)qi[r] * kQueryHeads + kvh * G + head[r]) * HD + d]));
    }
    float qup[2], qdown[2];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        qm[r] = fmaxf(qm[r], __shfl_xor_sync(0xffffffff, qm[r], 1));
        qm[r] = fmaxf(qm[r], __shfl_xor_sync(0xffffffff, qm[r], 2));
        int exponent = 0;
        if (qm[r] > 0)
            frexpf(qm[r], &exponent);
        qup[r] = ldexpf(1.f, 14 - exponent);
        qdown[r] = ldexpf(0.08838834764831845f, exponent - 14);
    }
    uint32_t qh[8][4], ql[8][4];
#pragma unroll
    for (int d = 0; d < 8; d++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int r = j & 1, dim = d * 16 + 2 * c + (j >= 2 ? 8 : 0);
            float a = 0, b = 0;
            if (qi[r] < rows) {
                const float* p = q + ((size_t)qi[r] * kQueryHeads + kvh * G + head[r]) * HD + dim;
                a = p[0] * qup[r];
                b = p[1] * qup[r];
            }
            qh[d][j] = pack(a, b);
            const float2 h = __half22float2(*reinterpret_cast<const __half2*>(&qh[d][j]));
            ql[d][j] = pack(a - h.x, b - h.y);
        }
    }
    float acc[16][4] = {}, maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {};
    const int end = first_query + min(rows, qr + QB);
    for (int first = 0; first < end; first += CH) {
        __syncthreads();
        for (int i = t; i < CH * 16; i += NT) {
            const int row = i / 16, piece = i % 16;
            const size_t off = ((size_t)(first + row) * kKvHeads + kvh) * HD;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            const int size = first + row < end ? 16 : 0;
            const void* kp = first + row < end ? keys + off + piece * 8 : keys;
            const void* vp = first + row < end ? values + off + piece * 8 : values;
            const uint32_t ks = (uint32_t)__cvta_generic_to_shared(&sk[row][swizzle(row, piece * 8)]);
            const uint32_t vs = (uint32_t)__cvta_generic_to_shared(&sv[row][swizzle(row, piece * 8)]);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(ks), "l"(kp), "r"(size)
                         : "memory");
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(vs), "l"(vp), "r"(size)
                         : "memory");
#else
            __trap();
#endif
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
#endif
        __syncthreads();
        float score[NB][4] = {}, low[NB][4] = {};
#pragma unroll
        for (int d = 0; d < 8; d++) {
#pragma unroll
            for (int n = 0; n < NB; n++) {
                uint32_t b[2];
                const int kr = n * 8 + (lane & 7), kd = d * 16 + 8 * ((lane >> 3) & 1);
                load_matrix(b, &sk[kr][swizzle(kr, kd)], false);
                mma(score[n], qh[d], b);
                mma(low[n], ql[d], b);
            }
        }
        float next[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < NB; n++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int r = j / 2, key = first + n * 8 + 2 * c + (j & 1);
                score[n][j] = qi[r] < rows && key <= first_query + qi[r]
                                  ? (score[n][j] + low[n][j]) * qdown[r]
                                  : -INFINITY;
                next[r] = fmaxf(next[r], score[n][j]);
            }
        }
        float alpha[2], sum[2] = {};
#pragma unroll
        for (int r = 0; r < 2; r++) {
            next[r] = fmaxf(next[r], __shfl_xor_sync(0xffffffff, next[r], 1));
            next[r] = fmaxf(next[r], __shfl_xor_sync(0xffffffff, next[r], 2));
            next[r] = fmaxf(next[r], maximum[r]);
            alpha[r] = maximum[r] == -INFINITY ? 0.f : expf(maximum[r] - next[r]);
        }
#pragma unroll
        for (int n = 0; n < NB; n++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int r = j / 2;
                score[n][j] = score[n][j] == -INFINITY ? 0.f : expf(score[n][j] - next[r]);
                sum[r] += score[n][j];
            }
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            sum[r] += __shfl_xor_sync(0xffffffff, sum[r], 1);
            sum[r] += __shfl_xor_sync(0xffffffff, sum[r], 2);
            denominator[r] = fmaf(denominator[r], alpha[r], sum[r]);
            maximum[r] = next[r];
        }
        if (first + CH <= first_query + qr + 1) {
            uint32_t ph[CH / 16][4], pl[CH / 16][4];
#pragma unroll
            for (int p = 0; p < CH / 16; p++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int n = p * 2 + j / 2, r = j & 1;
                    const float a = score[n][r * 2] * 16384.f, b = score[n][r * 2 + 1] * 16384.f;
                    ph[p][j] = pack(a, b);
                    const float2 h = __half22float2(*reinterpret_cast<const __half2*>(&ph[p][j]));
                    pl[p][j] = pack(a - h.x, b - h.y);
                }
            }
#pragma unroll
            for (int d = 0; d < 16; d++) {
                float high[4] = {}, lowv[4] = {};
#pragma unroll
                for (int p = 0; p < CH / 16; p++) {
                    const int vr = p * 16 + (lane & 15), vd = d * 8;
                    uint32_t b[2];
                    load_matrix(b, &sv[vr][swizzle(vr, vd)], true);
                    mma(high, ph[p], b);
                    mma(lowv, pl[p], b);
                }
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[d][j] = fmaf(acc[d][j], alpha[j / 2], (high[j] + lowv[j]) / 16384.f);
            }
        } else {
// The final causal tile has per-row masks. Avoid zero times a
// future nonfinite V, which would otherwise contaminate early rows.
#pragma unroll
            for (int d = 0; d < 16; d++) {
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[d][j] *= alpha[j / 2];
            }
#pragma unroll
            for (int n = 0; n < NB; n++) {
#pragma unroll
                for (int b = 0; b < 8; b++) {
                    const int key = first + n * 8 + b;
                    const float p0 = __shfl_sync(0xffffffff, score[n][b & 1], g * 4 + b / 2);
                    const float p1 = __shfl_sync(0xffffffff, score[n][2 + (b & 1)], g * 4 + b / 2);
#pragma unroll
                    for (int d = 0; d < 16; d++) {
                        const int dim = d * 8 + 2 * c;
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const int r = j / 2;
                            if (qi[r] < rows && key <= first_query + qi[r])
                                acc[d][j] =
                                    fmaf(r ? p1 : p0,
                                         __half2float(sv[n * 8 + b][swizzle(n * 8 + b, dim + (j & 1))]),
                                         acc[d][j]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int d = 0; d < 16; d++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int r = j / 2;
            if (qi[r] < rows)
                output[((size_t)qi[r] * kQueryHeads + kvh * G + head[r]) * HD + d * 8 + 2 * c + (j & 1)] =
                    acc[d][j] / denominator[r];
        }
    }
}
__global__ void unpack_kv(const void* keys, const void* values, __half* out_k, __half* out_v, int count) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)count * kKvHeads * (HD / 8);
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t head = i / (HD / 8);
        const int piece = i % (HD / 8);
        reinterpret_cast<uint4*>(out_k)[i] = ::strata::kolibri::kv::detail::load8(keys, head, piece);
        reinterpret_cast<uint4*>(out_v)[i] = ::strata::kolibri::kv::detail::load8(values, head, piece);
    }
}

} // namespace
void prefill_attention_packed(const float* q, const void* k, const void* v, float* out, int rows,
                              int key_count, int64_t first, bool int8_kv, void* stream, void* workspace) {
    if (rows < 1 || rows > kChunk || first < 0 || first + rows > key_count || (int8_kv && !workspace))
        throw std::invalid_argument(
            "Kolibri packed prefill: incomplete causal KV range, missing workspace or invalid shape");
    const __half* keys = static_cast<const __half*>(k);
    const __half* values = static_cast<const __half*>(v);
    if (int8_kv) {
        auto* temporary_k = static_cast<__half*>(workspace);
        auto* temporary_v = temporary_k + (size_t)key_count * kKvHeads * HD;
        const int blocks = (key_count * kKvHeads * (HD / 8) + 255) / 256;
        // Convert once per layer/chunk, rather than once for every query tile.
        unpack_kv<<<blocks, 256, 0, (cudaStream_t)stream>>>(k, v, temporary_k, temporary_v, key_count);
        const auto error = cudaGetLastError();
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        keys = temporary_k;
        values = temporary_v;
    }
    const dim3 grid((rows + QB - 1) / QB, kKvHeads);
    packed_prefill_kernel<<<grid, THREADS, 0, (cudaStream_t)stream>>>(q, keys, values, out, rows, (int)first);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kolibri::kernels
