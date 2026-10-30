// Kolibri decode uses the validated prefill hi/lo MMA equations on independent
// key partitions. Twelve query heads share each FP16 KV tile; F32 partials are
// merged by the same stable reduction as the original split decode path.
#include "tensor_decode_attention.cuh"
#include "kernels.cuh"
#include "kv_cache.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kolibri::kernels {
namespace {
constexpr int HD = kHeadDim, G = kQueryHeads / kKvHeads;
constexpr int CH = 32, THREADS = 128, STRIDE = HD + 8;
static_assert(HD == 128 && G == 12 && kKvHeads == 4, "Kolibri MMA geometry");

__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();  // Setup rejects this opt-in on devices below sm_80.
#endif
}

__device__ __forceinline__ uint32_t pack(float a, float b) {
    const __half2 halves = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&halves);
}

struct Shared {
    __half qh[16][STRIDE], ql[16][STRIDE];
    __half k[CH][STRIDE], v[CH][STRIDE];
    float scores[16][CH + 1];
    float qmax[4], alpha[16], denominator[16], maximum[16];
};
static_assert(sizeof(Shared) < 48 * 1024, "tensor decode uses ordinary shared memory");

template<int SplitKeys, bool DevicePosition = false, bool Int8 = false>
__global__ void __launch_bounds__(THREADS) tensor_decode_kernel(
        const float* __restrict__ q, const __half* __restrict__ keys,
        const __half* __restrict__ values, float* __restrict__ scratch,
        int key_count, int parts, const int32_t* position = nullptr) {
    if constexpr (DevicePosition) {
        key_count = *position + 1;
        parts = (key_count + SplitKeys - 1) / SplitKeys;
        if ((int) blockIdx.x >= parts) return;
    }
    __shared__ __align__(16) Shared s;
    const int part = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int group = lane >> 2, column = lane & 3;
    q += (size_t) kvh * G * HD;
    const int begin = part * SplitKeys, end = min(begin + SplitKeys, key_count);

    float qm = 0.f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
    for (int offset = 16; offset; offset >>= 1)
        qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, offset));
    if (lane == 0) s.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(s.qmax[0], s.qmax[1]), fmaxf(s.qmax[2], s.qmax[3]));
    int exponent = 0;
    if (qm > 0.f) frexpf(qm, &exponent);
    // Exact power-of-two scaling preserves the low half's normal FP16 range.
    const float qup = ldexpf(1.f, 14 - exponent);
    const float qdown = ldexpf(0.08838834764831845f, exponent - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int head = i / HD, d = i % HD;
        const float value = head < G ? q[(size_t) head * HD + d] * qup : 0.f;
        const __half hi = __float2half_rn(value);
        s.qh[head][d] = hi;
        s.ql[head][d] = __float2half_rn(value - __half2float(hi));
    }
    if (t < 16) { s.maximum[t] = -INFINITY; s.denominator[t] = 0.f; }
    float accumulator[4][4] = {};

    for (int first = begin; first < end; first += CH) {
        const int count = min(CH, end - first);
        __syncthreads();  // Previous P.V has finished reading K/V and probabilities.
        constexpr int PIECES = HD / 8;
        for (int i = t; i < CH * PIECES; i += THREADS) {
            const int row = i / PIECES, piece = i % PIECES;
            uint4 key = make_uint4(0, 0, 0, 0), value = key;
            if (row < count) {
                const size_t offset = ((size_t) (first + row) * kKvHeads + kvh) * HD;
                if constexpr (Int8) {
                    key = kv::detail::load8(keys, offset / HD, piece);
                    value = kv::detail::load8(values, offset / HD, piece);
                } else {
                    key = __ldg(reinterpret_cast<const uint4*>(keys + offset) + piece);
                    value = __ldg(reinterpret_cast<const uint4*>(values + offset) + piece);
                }
            }
            *reinterpret_cast<uint4*>(&s.k[row][piece * 8]) = key;
            *reinterpret_cast<uint4*>(&s.v[row][piece * 8]) = value;
        }
        __syncthreads();

        // Four warps each compute eight keys for all twelve query heads. Keep
        // hi and lo in independent F32 chains before combining their products.
        const int key_base = warp * 8;
        float high[4] = {}, low[4] = {};
#pragma unroll
        for (int d = 0; d < HD; d += 16) {
            uint32_t ah[4], al[4], b[2];
            ah[0] = *reinterpret_cast<const uint32_t*>(&s.qh[group][d + 2 * column]);
            ah[1] = *reinterpret_cast<const uint32_t*>(&s.qh[group + 8][d + 2 * column]);
            ah[2] = *reinterpret_cast<const uint32_t*>(&s.qh[group][d + 2 * column + 8]);
            ah[3] = *reinterpret_cast<const uint32_t*>(&s.qh[group + 8][d + 2 * column + 8]);
            al[0] = *reinterpret_cast<const uint32_t*>(&s.ql[group][d + 2 * column]);
            al[1] = *reinterpret_cast<const uint32_t*>(&s.ql[group + 8][d + 2 * column]);
            al[2] = *reinterpret_cast<const uint32_t*>(&s.ql[group][d + 2 * column + 8]);
            al[3] = *reinterpret_cast<const uint32_t*>(&s.ql[group + 8][d + 2 * column + 8]);
            b[0] = *reinterpret_cast<const uint32_t*>(&s.k[key_base + group][d + 2 * column]);
            b[1] = *reinterpret_cast<const uint32_t*>(&s.k[key_base + group][d + 2 * column + 8]);
            mma(high, ah, b);
            mma(low, al, b);
        }
        const int cell = key_base + 2 * column;
        s.scores[group][cell] = cell < count ? (high[0] + low[0]) * qdown : -INFINITY;
        s.scores[group][cell + 1] = cell + 1 < count ? (high[1] + low[1]) * qdown : -INFINITY;
        s.scores[group + 8][cell] = cell < count ? (high[2] + low[2]) * qdown : -INFINITY;
        s.scores[group + 8][cell + 1] = cell + 1 < count ? (high[3] + low[3]) * qdown : -INFINITY;
        __syncthreads();

        // Eight lanes reduce one head. Use the baseline's natural-exp F32
        // softmax, retaining an F32 running maximum and denominator.
        const int row = t >> 3, sub = t & 7;
        float score[4], maximum = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            score[j] = s.scores[row][sub * 4 + j];
            maximum = fmaxf(maximum, score[j]);
        }
        for (int offset = 1; offset < 8; offset <<= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
        const float previous = s.maximum[row], next = fmaxf(previous, maximum);
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float probability = score[j] == -INFINITY ? 0.f : expf(score[j] - next);
            s.scores[row][sub * 4 + j] = probability;
            sum += probability;
        }
        for (int offset = 1; offset < 8; offset <<= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if (sub == 0) {
            const float alpha = previous == -INFINITY ? 0.f : expf(previous - next);
            s.alpha[row] = alpha;
            s.denominator[row] = fmaf(s.denominator[row], alpha, sum);
            s.maximum[row] = next;
        }
        __syncthreads();

        // Each warp owns 32 output dimensions. Scaling P by 2^14 keeps its
        // low FP16 part useful; V is already exactly the cached FP16 value.
        float phigh[4][4] = {}, plow[4][4] = {};
#pragma unroll
        for (int piece = 0; piece < CH / 16; ++piece) {
            const int a = piece * 16 + 2 * column, b = a + 8;
            const float p[8] = {
                s.scores[group][a] * 16384.f, s.scores[group][a + 1] * 16384.f,
                s.scores[group + 8][a] * 16384.f, s.scores[group + 8][a + 1] * 16384.f,
                s.scores[group][b] * 16384.f, s.scores[group][b + 1] * 16384.f,
                s.scores[group + 8][b] * 16384.f, s.scores[group + 8][b + 1] * 16384.f};
            uint32_t ah[4], al[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                ah[j] = pack(p[2 * j], p[2 * j + 1]);
                const float2 high = __half22float2(*reinterpret_cast<const __half2*>(&ah[j]));
                al[j] = pack(p[2 * j] - high.x, p[2 * j + 1] - high.y);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = warp * 32 + j * 8 + group;
                const __half2 v0 = __halves2half2(s.v[a][d], s.v[a + 1][d]);
                const __half2 v1 = __halves2half2(s.v[b][d], s.v[b + 1][d]);
                const uint32_t value[2] = {*reinterpret_cast<const uint32_t*>(&v0),
                                           *reinterpret_cast<const uint32_t*>(&v1)};
                mma(phigh[j], ah, value);
                mma(plow[j], al, value);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            accumulator[j][0] = fmaf(accumulator[j][0], s.alpha[group], (phigh[j][0] + plow[j][0]) / 16384.f);
            accumulator[j][1] = fmaf(accumulator[j][1], s.alpha[group], (phigh[j][1] + plow[j][1]) / 16384.f);
            accumulator[j][2] = fmaf(accumulator[j][2], s.alpha[group + 8], (phigh[j][2] + plow[j][2]) / 16384.f);
            accumulator[j][3] = fmaf(accumulator[j][3], s.alpha[group + 8], (phigh[j][3] + plow[j][3]) / 16384.f);
        }
    }
    __syncthreads();
    constexpr int PARTIAL = HD + 2;
    float* low = scratch + ((size_t) (kvh * G + group) * parts + part) * PARTIAL;
    float* high = group + 8 < G ?
        scratch + ((size_t) (kvh * G + group + 8) * parts + part) * PARTIAL : nullptr;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = warp * 32 + j * 8 + 2 * column;
        low[d] = accumulator[j][0];
        low[d + 1] = accumulator[j][1];
        if (high) {
            high[d] = accumulator[j][2];
            high[d + 1] = accumulator[j][3];
        }
    }
    if (t < G) {
        float* partial = scratch + ((size_t) (kvh * G + t) * parts + part) * PARTIAL;
        partial[HD] = s.maximum[t];
        partial[HD + 1] = s.denominator[t];
    }
}

int partition_keys(int key_count, bool local) {
    return local || key_count < 8192 ? 32 : 128;
}

}  // namespace

size_t tensor_decode_attention_scratch_floats(int max_key_count) {
    if (max_key_count < 1 || max_key_count > 262144)
        throw std::invalid_argument("Kolibri tensor decode: key count must be in 1..262144");
    // The global partition grows at 8K; an allocation for a longer context
    // must also hold the larger partition count immediately before that step.
    const int short_parts = (std::min(max_key_count, 8191) + 31) / 32;
    const int long_parts = (max_key_count + 127) / 128;
    const int local_parts = (kWindow + 31) / 32;
    return (size_t) std::max({short_parts, long_parts, local_parts}) * kQueryHeads * (HD + 2);
}

static void tensor_decode_attention_impl(const float* q, const uint16_t* keys, const uint16_t* values,
                                         float* output, float* scratch, int key_count, bool local, void* stream, void* q8) {
    tensor_decode_attention_scratch_floats(key_count);
    if (!scratch || (local && key_count != kWindow))
        throw std::invalid_argument("Kolibri tensor decode: full local window and scratch required");
    const int split = partition_keys(key_count, local);
    const int parts = (key_count + split - 1) / split;
    if (split == 32)
        tensor_decode_kernel<32><<<dim3(parts, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
            q, (const __half*) keys, (const __half*) values, scratch, key_count, parts);
    else
        tensor_decode_kernel<128><<<dim3(parts, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
            q, (const __half*) keys, (const __half*) values, scratch, key_count, parts);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error("Kolibri tensor decode: " + std::string(cudaGetErrorString(error)));
    if (q8) attention_merge_partials_q8(scratch, output, q8, parts, stream);
    else attention_merge_partials(scratch, output, parts, stream);
}

void tensor_decode_attention(const float* q, const uint16_t* keys, const uint16_t* values,
                             float* output, float* scratch, int key_count, bool local, void* stream) {
    tensor_decode_attention_impl(q, keys, values, output, scratch, key_count, local, stream, nullptr);
}

void tensor_decode_attention_q8(const float* q, const uint16_t* keys, const uint16_t* values,
                                float* output, float* scratch, int key_count, bool local, void* stream, void* q8) {
    if (!q8) throw std::invalid_argument("Kolibri tensor merge Q8 requires output scratch");
    tensor_decode_attention_impl(q, keys, values, output, scratch, key_count, local, stream, q8);
}

void tensor_decode_attention_q8_graph(const float* q, const uint16_t* keys, const uint16_t* values,
                                      float* output, float* scratch, const int32_t* position,
                                      int max_key_count, void* stream, void* q8) {
    tensor_decode_attention_scratch_floats(max_key_count);
    if (!scratch || !position || !q8)
        throw std::invalid_argument("Kolibri graph attention requires position and output scratch");
    const int split = partition_keys(max_key_count, false);
    const int parts = (max_key_count + split - 1) / split;
    if (split == 32)
        tensor_decode_kernel<32, true><<<dim3(parts, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
            q, (const __half*) keys, (const __half*) values, scratch, 0, 0, position);
    else
        tensor_decode_kernel<128, true><<<dim3(parts, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
            q, (const __half*) keys, (const __half*) values, scratch, 0, 0, position);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error("Kolibri graph tensor decode: " + std::string(cudaGetErrorString(error)));
    attention_merge_partials_q8_graph(scratch, output, q8, position, split, parts, stream);
}
void tensor_decode_attention_int8(const float* q, const void* keys, const void* values,
                                  float* output, float* scratch, int key_count, void* stream,
                                  void* q8, const int32_t* position) {
    tensor_decode_attention_scratch_floats(key_count);
    if (!scratch || (position && !q8))
        throw std::invalid_argument("Kolibri INT8 decode requires scratch and graph Q8 output");
    const int split = partition_keys(key_count, false), parts = (key_count + split - 1) / split;
    if (position) {
        if (split == 32)
            tensor_decode_kernel<32, true, true><<<dim3(parts, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
                q, (const __half*) keys, (const __half*) values, scratch, 0, 0, position);
        else
            tensor_decode_kernel<128, true, true><<<dim3(parts, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
                q, (const __half*) keys, (const __half*) values, scratch, 0, 0, position);
    } else {
        if (split == 32)
            tensor_decode_kernel<32, false, true><<<dim3(parts, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
                q, (const __half*) keys, (const __half*) values, scratch, key_count, parts);
        else
            tensor_decode_kernel<128, false, true><<<dim3(parts, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
                q, (const __half*) keys, (const __half*) values, scratch, key_count, parts);
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error("Kolibri INT8 decode: " + std::string(cudaGetErrorString(error)));
    if (position) attention_merge_partials_q8_graph(scratch, output, q8, position, split, parts, stream);
    else if (q8) attention_merge_partials_q8(scratch, output, q8, parts, stream);
    else attention_merge_partials(scratch, output, parts, stream);
}
}  // namespace strata::kolibri::kernels
