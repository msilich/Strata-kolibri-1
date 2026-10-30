#include "kv_cache.cuh"

#include <stdexcept>
#include <string>

namespace strata::kolibri::kv {
namespace {
__global__ void append_kernel(const float* keys, const float* values, Int8Head* kc, Int8Head* vc, int offset) {
    const int head = blockIdx.x * kernels::kKvHeads + blockIdx.y;
    const size_t out = (size_t) (offset + blockIdx.x) * kernels::kKvHeads + blockIdx.y;
    const size_t i = (size_t) head * kernels::kHeadDim + threadIdx.x;
    detail::store_pair(keys[i], values[i], kc + out, vc + out);
}
}

void append_int8(const float* keys, const float* values, void* kc, void* vc,
                 int rows, int offset, void* stream) {
    if (rows < 1 || rows > kernels::kChunk || offset < 0)
        throw std::invalid_argument("Kolibri INT8 KV: invalid append shape or offset");
    append_kernel<<<dim3(rows, kernels::kKvHeads), kernels::kHeadDim, 0, (cudaStream_t) stream>>>(
        keys, values, static_cast<Int8Head*>(kc), static_cast<Int8Head*>(vc), offset);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error("Kolibri INT8 KV append: " + std::string(cudaGetErrorString(error)));
}
}  // namespace strata::kolibri::kv
