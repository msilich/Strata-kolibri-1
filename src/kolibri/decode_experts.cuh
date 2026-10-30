#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kolibri::decode_cache { struct Plan; }

namespace strata::kolibri::kernels {

inline constexpr int kDecodeExpertCount = 6;
inline constexpr int kDecodeExpertInput = 2560;
inline constexpr int kDecodeExpertHidden = 512;
inline constexpr size_t kDecodeExpertQ8Bytes = (2560 + 6 * 512) / 32 * 36;
inline constexpr size_t kDecodeExpertScratchFloats = 3 * 6 * 512;

// Host-side table of stable device pointers, passed to kernels by value. Roles
// are row-major GGUF Q4_K (12), Q6_K (14), or Q8_0 (8), independently typed.
// All six blobs have these same role offsets, including any allocator padding.
struct DecodeExpertWeights {
    const void* expert[kDecodeExpertCount];
    size_t gate_offset, up_offset, down_offset;
    int gate_type, up_type, down_type;
};

// Opt-in single-token routed FFN. Input is [2560], output is [6,2560] in table
// order, without routing weights or shared-expert addition. All buffers are
// disjoint device allocations (at least four-byte aligned); stream is non-null.
// Scratch is caller-owned: kDecodeExpertQ8Bytes and kDecodeExpertScratchFloats.
// Enqueues without allocation or synchronization. Uses native Q8_1 quantization,
// the generic native MMVQ reduction order, and accurate F32 SwiGLU.
// normalized_q8 optionally supplies input's native Q8_1 blocks (2880 bytes),
// skipping its quantization. It must stay live and disjoint from scratch/output.
void decode_experts(const DecodeExpertWeights& weights, const float* input,
                    float* output, void* q8_scratch, float* float_scratch,
                    void* stream, const void* normalized_q8 = nullptr);

// Output rows are in ascending expert-ID order. sorted_ranks is a host array
// mapping each row back to its device route_weights entry. The destination
// already contains the shared expert; each row is accumulated in order with
// the same rounded multiplication and global atomic-add arithmetic as scatter_add_weighted.
// The destination must not overlap inputs or be accessed concurrently.
void combine_decode_experts(const float* outputs, const float* route_weights,
                            const int32_t* sorted_ranks, float* destination,
                            void* stream);

// Device cache plans supply pointers and sorted router ranks.
// Invalid plans skip all weight-pointer reads and combination.
void decode_experts_device(const DecodeExpertWeights& roles, const decode_cache::Plan* plan,
                           const float* input, float* output, void* q8_scratch,
                           float* float_scratch, void* stream, const void* normalized_q8);
void combine_decode_experts_device(const float* outputs, const float* route_weights,
                                  const decode_cache::Plan* plan, float* destination, void* stream);

}  // namespace strata::kolibri::kernels
