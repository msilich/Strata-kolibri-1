#include "dense_decode.cuh"

// Same pinned Q8_0 arithmetic and --use_fast_math contract as native_mmvq.cu.
#include "ggml-cuda/vecdotq.cuh"

#include <stdexcept>

namespace strata::kolibri::kernels {
namespace {
constexpr int kInput = 2560, kWarp = 32, kWarps = 4;

__launch_bounds__(kWarp * kWarps, 1)
__global__ void project(DenseDecodeProjection projection, const block_q8_1* __restrict__ x) {
    int matrix = 0, row = int(blockIdx.x);
    if (row >= projection.rows[0]) {
        row -= projection.rows[0];
        matrix = 1;
        if (row >= projection.rows[1]) { row -= projection.rows[1]; matrix = 2; }
    }
    constexpr int blocks_per_row = kInput / 32;
    const auto* weights = static_cast<const block_q8_0*>(projection.weights[matrix]) + row * blocks_per_row;
    const int tid = kWarp * int(threadIdx.y) + int(threadIdx.x);
    const int iqs = 2 * (tid % 4);
    float sum = 0.f;
    for (int block = tid / 4; block < blocks_per_row; block += kWarp)
        sum += vec_dot_q8_0_q8_1(weights, x + block, block, iqs);

    __shared__ float partial[kWarps - 1][kWarp];
    if (threadIdx.y > 0) partial[threadIdx.y - 1][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int warp = 0; warp < kWarps - 1; ++warp) sum += partial[warp][threadIdx.x];
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, kWarp);
    if (threadIdx.x == 0) projection.outputs[matrix][row] = sum;
}

// Fixed pointers avoid a thread-local projection table while retaining the
// shared experts' original dot products and reduction order.
__launch_bounds__(kWarp * kWarps, 1)
__global__ void project_pair(const block_q8_0* w0, const block_q8_0* w1,
                             float* y0, float* y1,
                             const block_q8_1* __restrict__ x, int rows0) {
    const bool second = int(blockIdx.x) >= rows0;
    const int row = int(blockIdx.x) - (second ? rows0 : 0);
    constexpr int blocks_per_row = kInput / 32;
    const auto* weights = (second ? w1 : w0) + row * blocks_per_row;
    float* output = second ? y1 : y0;
    const int tid = kWarp * int(threadIdx.y) + int(threadIdx.x);
    const int iqs = 2 * (tid % 4);
    float sum = 0.f;
    for (int block = tid / 4; block < blocks_per_row; block += kWarp)
        sum += vec_dot_q8_0_q8_1(weights, x + block, block, iqs);

    __shared__ float partial[kWarps - 1][kWarp];
    if (threadIdx.y > 0) partial[threadIdx.y - 1][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int warp = 0; warp < kWarps - 1; ++warp) sum += partial[warp][threadIdx.x];
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, kWarp);
    if (threadIdx.x == 0) output[row] = sum;
}

template<int Rows>
__launch_bounds__(kWarp * kWarps, 1)
__global__ void project_warp(DenseDecodeProjection projection, const block_q8_1* __restrict__ x, int total_rows) {
    const int lane = int(threadIdx.x) & 31;
    const int row0 = (int(blockIdx.x) * kWarps + int(threadIdx.x) / kWarp) * Rows;
    if (row0 >= total_rows) return;
    const block_q8_0* weights[Rows];
    float* output[Rows];
#pragma unroll
    for (int i = 0; i < Rows; ++i) {
        int row = min(row0 + i, total_rows - 1), matrix = 0;
        if (row >= projection.rows[0]) {
            row -= projection.rows[0]; matrix = 1;
            if (row >= projection.rows[1]) { row -= projection.rows[1]; matrix = 2; }
        }
        weights[i] = static_cast<const block_q8_0*>(projection.weights[matrix]) + row * (kInput / 32);
        output[i] = projection.outputs[matrix] + row;
    }
    float partial[kWarps][Rows] = {};
    // The four logical warps retain exactly the original block ownership and
    // accumulation chains. Only their execution is moved into one physical warp.
    for (int base = 0; base < kInput / 32; base += 32) {
#pragma unroll
        for (int warp = 0; warp < kWarps; ++warp) {
            const int block = base + warp * 8 + lane / 4;
            if (block < kInput / 32) {
#pragma unroll
                for (int i = 0; i < Rows; ++i)
                    partial[warp][i] += vec_dot_q8_0_q8_1(weights[i], x + block, block, 2 * (lane % 4));
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
        if (lane == 0 && row0 + i < total_rows) *output[i] = sum;
    }
}
}  // namespace

void dense_decode_q8(const DenseDecodeProjection& projection, const void* input_q8, void* stream) {
    if (projection.count < 2 || projection.count > 3)
        throw std::invalid_argument("Kolibri dense decode requires two or three projections");
    int rows = 0;
    for (int i = 0; i < projection.count; ++i) {
        if (!projection.weights[i] || !projection.outputs[i] || projection.rows[i] <= 0)
            throw std::invalid_argument("Kolibri dense decode requires valid Q8 matrices and outputs");
        rows += projection.rows[i];
    }
    const auto* x = static_cast<const block_q8_1*>(input_q8);
    const auto s = static_cast<cudaStream_t>(stream);
    // The QKV geometry benefits from virtual warps; the smaller shared pair
    // retains the original block layout, which measured faster on the RTX 5090.
    if (projection.count == 3 && rows == 7168)
        project_warp<2><<<(rows + 7) / 8, kWarp * kWarps, 0, s>>>(projection, x, rows);
    else if (projection.count == 2 && projection.rows[0] == 512 && projection.rows[1] == 512)
        project_pair<<<rows, dim3(kWarp, kWarps), 0, s>>>(
            static_cast<const block_q8_0*>(projection.weights[0]),
            static_cast<const block_q8_0*>(projection.weights[1]),
            projection.outputs[0], projection.outputs[1], x, projection.rows[0]);
    else project<<<rows, dim3(kWarp, kWarps), 0, s>>>(projection, x);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kolibri::kernels
