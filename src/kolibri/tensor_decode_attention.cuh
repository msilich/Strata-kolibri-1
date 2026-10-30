#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kolibri::kernels {

// Holds every global partition count up to max_key_count and the full local
// window. Allocate once; storage may be shared with other attention scratch.
size_t tensor_decode_attention_scratch_floats(int max_key_count);

// Single-query tensor-core attention using scaled FP16 hi/lo Q/P, FP16 KV and
// F32 softmax/partials/output. Requires prefill_attention_supported() at setup.
// Q/output are F32 [48,128], KV is F16 bits [key_count,4,128], already rotated.
// Global keys are the complete causal prefix; local keys are exactly the
// contiguous 513-key window ending at the query. No allocation or sync occurs.
// Summation order differs from the baseline and F32 split kernels.
void tensor_decode_attention(const float* q, const uint16_t* kcache, const uint16_t* vcache,
                             float* out, float* scratch, int key_count, bool local, void* stream);
// Additional native Q8_1 output from the unchanged F32 merge; q8 has 6912 bytes
// and does not alias any attention input, partial or F32 output buffer.
void tensor_decode_attention_q8(const float* q, const uint16_t* kcache, const uint16_t* vcache,
                                float* out, float* scratch, int key_count, bool local, void* stream, void* q8);

}  // namespace strata::kolibri::kernels
