#include "kernels.cuh"

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

__global__ void swiglu_kernel(const float* gate, const float* up, float* out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (gate[i] / (1.f + expf(-gate[i]))) * up[i];
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

void route(const float* logits, const float* bias, int32_t* ids, float* weights, int rows, void* stream) {
    if (rows < 1) throw std::invalid_argument("Kolibri routing: empty input");
    route_kernel<<<rows, 128, 0, (cudaStream_t) stream>>>(logits, bias, ids, weights);
    checked("routing");
}

void swiglu(const float* gate, const float* up, float* out, int rows, int width, void* stream) {
    if (rows < 1 || width < 1) throw std::invalid_argument("Kolibri SwiGLU: invalid shape");
    const size_t n = (size_t) rows * width;
    swiglu_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gate, up, out, n);
    checked("SwiGLU");
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
