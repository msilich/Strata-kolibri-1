#pragma once

#include <cstdint>

namespace strata::kolibri::kernels {

// Query the current CUDA device once at setup. The opt-in MMA kernel requires
// sm_80+; callers keep attention() on devices that do not support it.
bool prefill_attention_supported();

// Kolibri-only FP16-KV tensor-core attention. Q/output are F32 [rows,48,128];
// K/V are F16 bits [key_count,4,128], contiguous in token order. The cache spans
// key_first_pos..key_first_pos+key_count-1 and contains each query's causal range.
// Local queries use at most the current key plus 512 predecessors. Absolute
// RoPE has already been applied to Q/K. Rows are 1..4096; no global workspace,
// allocation, synchronization or per-call device capability query is needed.
// Call only after prefill_attention_supported() succeeds. Q and probabilities
// use scaled FP16 hi+lo parts; accumulation, softmax and output remain F32.
// This changes summation order and is not bitwise identical to attention().
void prefill_attention(const float* q, const uint16_t* kcache, const uint16_t* vcache,
                       float* out, int rows, int key_count, int64_t key_first_pos,
                       int64_t query_first_pos, bool local, void* stream);

// Global-only variant: two query cohorts share each K/V tile while retaining
// the original per-query arithmetic and causal masking. Same layouts and
// capability requirement as prefill_attention(); the cache starts at position 0.
void prefill_attention_two_query(const float* q, const uint16_t* kcache, const uint16_t* vcache,
                                 float* out, int rows, int key_count,
                                 int64_t query_first_pos, void* stream);
// Global-only packed query/head tiles. Same precision and causal range as the
// tensor prefill path, with a different summation order. INT8 requires a
// stream-ordered temporary buffer of 2*key_count*4*128 FP16 elements; FP16 needs
// no workspace. The stored KV format is unchanged.
void prefill_attention_packed(const float* q, const void* kcache, const void* vcache,
                              float* out, int rows, int key_count, int64_t query_first_pos,
                              bool int8_kv, void* stream, void* workspace = nullptr);
// Global contiguous INT8 KV, dequantized directly into the shared FP16 tiles.
void prefill_attention_int8(const float* q, const void* kcache, const void* vcache,
                            float* out, int rows, int key_count, int64_t query_first_pos,
                            bool two_query, void* stream);

}  // namespace strata::kolibri::kernels
