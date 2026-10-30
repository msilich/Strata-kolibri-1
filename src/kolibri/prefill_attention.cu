// Kolibri-only adaptation of qsa_prompt_attn.cu's FP16-KV MMA layout and hi/lo
// operands. Contiguous causal KV replaces the sparse page-table/selection path.
#include "prefill_attention.cuh"
#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kolibri::kernels {
namespace {
constexpr int HD = kHeadDim, G = kQueryHeads / kKvHeads;
constexpr int CH = 32, THREADS = 128, STRIDE = HD + 8;
static_assert(HD == 128 && G == 12 && kKvHeads == 4, "Kolibri MMA geometry");

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

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
static_assert(sizeof(Shared) < 48 * 1024, "prefill attention uses ordinary shared memory");

__global__ void __launch_bounds__(THREADS) prefill_kernel(
        const float* __restrict__ q, const __half* __restrict__ keys,
        const __half* __restrict__ values, float* __restrict__ output,
        int64_t key_first, int64_t query_first, bool local) {
    __shared__ __align__(16) Shared s;
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int group = lane >> 2, column = lane & 3;
    q += ((size_t) qi * kQueryHeads + kvh * G) * HD;
    output += ((size_t) qi * kQueryHeads + kvh * G) * HD;
    const int64_t position = query_first + qi;
    const int64_t oldest = local && position >= kWindow ? position - (kWindow - 1) : 0;
    const int begin = oldest > key_first ? (int) (oldest - key_first) : 0;
    const int end = (int) (position - key_first + 1);

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
                key = __ldg(reinterpret_cast<const uint4*>(keys + offset) + piece);
                value = __ldg(reinterpret_cast<const uint4*>(values + offset) + piece);
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
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = warp * 32 + j * 8 + 2 * column;
        output[group * HD + d] = accumulator[j][0] / s.denominator[group];
        output[group * HD + d + 1] = accumulator[j][1] / s.denominator[group];
        if (group + 8 < G) {
            output[(group + 8) * HD + d] = accumulator[j][2] / s.denominator[group + 8];
            output[(group + 8) * HD + d + 1] = accumulator[j][3] / s.denominator[group + 8];
        }
    }
}
struct TwoQueryState {
    __half qh[16][STRIDE], ql[16][STRIDE];
    float scores[16][CH + 1];
    float qmax[4], alpha[16], denominator[16], maximum[16];
};
struct TwoQueryShared {
    TwoQueryState query[2];
    __half k[CH][STRIDE], v[CH][STRIDE];
};
static_assert(sizeof(TwoQueryState) == 11024 && sizeof(TwoQueryShared) == 39456);
static_assert(sizeof(TwoQueryShared) < 48 * 1024, "two-query attention uses ordinary shared memory");

__global__ void __launch_bounds__(2 * THREADS) prefill_two_query_kernel(
        const float* __restrict__ q, const __half* __restrict__ keys,
        const __half* __restrict__ values, float* __restrict__ output,
        int rows, int64_t query_first) {
    __shared__ __align__(16) TwoQueryShared shared;
    const int cohort = threadIdx.x / THREADS;
    const int qi = 2 * blockIdx.x + cohort, kvh = blockIdx.y;
    const int t = threadIdx.x % THREADS, lane = t & 31, warp = t >> 5;
    const int group = lane >> 2, column = lane & 3;
    TwoQueryState& s = shared.query[cohort];
    const bool active = qi < rows;
    if (active) {
        q += ((size_t) qi * kQueryHeads + kvh * G) * HD;
        output += ((size_t) qi * kQueryHeads + kvh * G) * HD;
    }
    const int end = active ? (int) (query_first + qi + 1) : 0;
    const int pair_end = (int) (query_first + min(rows, 2 * (int) blockIdx.x + 2));

    float qm = 0.f;
    if (active) for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
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
        const float value = active && head < G ? q[(size_t) head * HD + d] * qup : 0.f;
        const __half hi = __float2half_rn(value);
        s.qh[head][d] = hi;
        s.ql[head][d] = __float2half_rn(value - __half2float(hi));
    }
    if (t < 16) { s.maximum[t] = -INFINITY; s.denominator[t] = 0.f; }
    float accumulator[4][4] = {};

    for (int first = 0; first < pair_end; first += CH) {
        const bool working = active && first < end;
        const int count = working ? min(CH, end - first) : 0;
        const int pair_count = min(CH, pair_end - first);
        __syncthreads();  // Previous P.V has finished reading K/V and probabilities.
        constexpr int PIECES = HD / 8;
        for (int i = threadIdx.x; i < CH * PIECES; i += 2 * THREADS) {
            const int row = i / PIECES, piece = i % PIECES;
            uint4 key = make_uint4(0, 0, 0, 0), value = key;
            if (row < pair_count) {
                const size_t offset = ((size_t) (first + row) * kKvHeads + kvh) * HD;
                key = __ldg(reinterpret_cast<const uint4*>(keys + offset) + piece);
                value = __ldg(reinterpret_cast<const uint4*>(values + offset) + piece);
            }
            *reinterpret_cast<uint4*>(&shared.k[row][piece * 8]) = key;
            *reinterpret_cast<uint4*>(&shared.v[row][piece * 8]) = value;
        }
        __syncthreads();

        if (working) {
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
            b[0] = *reinterpret_cast<const uint32_t*>(&shared.k[key_base + group][d + 2 * column]);
            b[1] = *reinterpret_cast<const uint32_t*>(&shared.k[key_base + group][d + 2 * column + 8]);
            mma(high, ah, b);
            mma(low, al, b);
        }
        const int cell = key_base + 2 * column;
        s.scores[group][cell] = cell < count ? (high[0] + low[0]) * qdown : -INFINITY;
        s.scores[group][cell + 1] = cell + 1 < count ? (high[1] + low[1]) * qdown : -INFINITY;
        s.scores[group + 8][cell] = cell < count ? (high[2] + low[2]) * qdown : -INFINITY;
        s.scores[group + 8][cell + 1] = cell + 1 < count ? (high[3] + low[3]) * qdown : -INFINITY;
        }
        __syncthreads();

        if (working) {
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
        }
        __syncthreads();

        if (working) {
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
                // Each query keeps the old zero-filled V tail even when its
                // partner has a valid extra key in this shared tile.
                const __half zero = __ushort_as_half(0);
                const __half2 v0 = __halves2half2(a < count ? shared.v[a][d] : zero,
                                                a + 1 < count ? shared.v[a + 1][d] : zero);
                const __half2 v1 = __halves2half2(b < count ? shared.v[b][d] : zero,
                                                b + 1 < count ? shared.v[b + 1][d] : zero);
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
    }
    __syncthreads();
    if (active) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = warp * 32 + j * 8 + 2 * column;
        output[group * HD + d] = accumulator[j][0] / s.denominator[group];
        output[group * HD + d + 1] = accumulator[j][1] / s.denominator[group];
        if (group + 8 < G) {
            output[(group + 8) * HD + d] = accumulator[j][2] / s.denominator[group + 8];
            output[(group + 8) * HD + d + 1] = accumulator[j][3] / s.denominator[group + 8];
        }
    }
    }
}
}  // namespace

bool prefill_attention_supported() {
    int device = 0, major = 0;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    return major >= 8;
}

void prefill_attention(const float* q, const uint16_t* keys, const uint16_t* values,
                       float* output, int rows, int key_count, int64_t key_first,
                       int64_t query_first, bool local, void* stream) {
    const int64_t needed_first = local && query_first >= kWindow ? query_first - (kWindow - 1) : 0;
    if (rows < 1 || rows > kChunk || key_count < 1 || key_first < 0 || query_first < key_first ||
        query_first + rows > key_first + key_count || key_first > needed_first || (!local && key_first != 0))
        throw std::invalid_argument("Kolibri prefill attention: incomplete causal KV range or invalid shape");
    prefill_kernel<<<dim3(rows, kKvHeads), THREADS, 0, (cudaStream_t) stream>>>(
        q, (const __half*) keys, (const __half*) values, output, key_first, query_first, local);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error("Kolibri prefill attention: " + std::string(cudaGetErrorString(error)));
}
void prefill_attention_two_query(const float* q, const uint16_t* keys, const uint16_t* values,
             float* output, int rows, int key_count, int64_t query_first, void* stream) {
    if (rows < 1 || rows > kChunk || key_count < 1 || query_first < 0 ||
        query_first + rows > key_count)
        throw std::invalid_argument("Kolibri two-query prefill attention: incomplete causal KV range or invalid shape");
    prefill_two_query_kernel<<<dim3((rows + 1) / 2, kKvHeads), 2 * THREADS, 0, (cudaStream_t) stream>>>(
        q, (const __half*) keys, (const __half*) values, output, rows, query_first);
    check(cudaGetLastError());
}
}  // namespace strata::kolibri::kernels
