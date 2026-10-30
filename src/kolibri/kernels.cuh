#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kolibri::kernels {

// All buffers are device pointers. Calls enqueue work on stream without allocating
// or synchronizing. Float matrices are token-major; cache values are F16 bits.
inline constexpr int kQueryHeads = 48;
inline constexpr int kKvHeads = 4;
inline constexpr int kHeadDim = 128;
inline constexpr int kExperts = 384;
inline constexpr int kTopK = 6;
inline constexpr int kWindow = 513;
inline constexpr int kChunk = 4096;

// out may alias x; weights has width entries.
void rms_norm(const float* x, const float* weights, float* out, int rows, int width, float eps, void* stream);
// out may alias either input.
void add(const float* a, const float* b, float* out, size_t n, void* stream);

// In-place per-head RMS norm, then full-head NeoX RoPE on local layers only.
// q [rows,48,128], k [rows,4,128], norm weights [128]; positions are absolute.
void qk_norm_rope(float* q, float* k, const float* qw, const float* kw, int rows,
                  int64_t first_pos, bool local, float theta, float eps, void* stream);
// Opt-in local-layer reuse: table holds rows*128 floats, interleaved sine/cosine
// for 64 NeoX pairs per row. Prepare once for the forward's absolute positions;
// reuse across local layers on the same stream. Table must not alias Q/K/weights.
void rope_table(float* sin_cos, int rows, int64_t first_pos, float theta, void* stream);
// Graph variants read the validated absolute position from device memory.
void rope_table_decode_graph(float* sin_cos, const int32_t* position, float theta, void* stream);
void qk_norm_rope_cached(float* q, float* k, const float* qw, const float* kw, int rows,
                         const float* sin_cos, float eps, void* stream);
// Opt-in T=1 equivalent of QK norm/RoPE followed by KV append. Local layers
// supply the existing 128-float RoPE table; global layers skip RoPE. Q/K remain
// F32 in place, V is unchanged, and caches must not alias Q/K/V/weights/table.
void qk_norm_rope_append_decode(float* q, float* k, const float* v, const float* qw, const float* kw,
                                const float* sin_cos, bool local, uint16_t* kcache, uint16_t* vcache,
                                int dst_offset, float eps, void* stream);
void qk_norm_append_decode_graph(float* q, float* k, const float* v, const float* qw, const float* kw,
                                uint16_t* kcache, uint16_t* vcache, const int32_t* position,
                                float eps, void* stream);
// k/v [rows,4,128] into token-major caches beginning at dst_offset.
void append_kv(const float* k, const float* v, uint16_t* kcache, uint16_t* vcache,
               int rows, int dst_offset, void* stream);

// q/out [rows,48,128]; caches [key_count,4,128]. Keys are consecutive absolute
// positions beginning at key_first_pos and include every query's current key.
// Local callers supply at least min(query_first_pos,512) preceding keys. Global
// callers supply every key from position zero. No attention output gate.
void attention(const float* q, const uint16_t* kcache, const uint16_t* vcache, float* out,
               int rows, int key_count, int64_t key_first_pos, int64_t query_first_pos,
               bool local, void* stream);

// Opt-in global single-query attention. Q, scores, softmax and sums stay F32;
// independent key ranges are merged with a stable softmax rescale. Caches begin
// at absolute position zero; key_count is 1..262144. Scratch is caller-owned,
// sized for key_count with the helper below, and does not alias inputs or output.
size_t attention_split_scratch_floats(int key_count);
void attention_split_decode(const float* q, const uint16_t* kcache, const uint16_t* vcache,
                            float* out, int key_count, int64_t query_position,
                            float* scratch, void* stream);

// Opt-in single-query local attention for the full 513-key window only.
// Cache indices 0..512 cover query_position-512..query_position; absolute RoPE
// has already been applied to Q/K. Use attention() for shorter windows/prefill.
// Scratch is caller-owned, does not alias inputs/output, and may reuse the
// global split-attention buffer when it is at least this many floats.
size_t attention_local_split_scratch_floats();
void attention_local_split_decode(const float* q, const uint16_t* kcache, const uint16_t* vcache,
                                  float* out, float* scratch, void* stream);

// Internal split-attention reduction, shared by F32 and tensor-core producers.
// Partials are [48,parts,130]: numerator[128], maximum, denominator.
void attention_merge_partials(const float* scratch, float* out, int parts, void* stream);
// Same F32 merge plus its native Q8_1 image (192 blocks, 6912 bytes).
// q8 is disjoint from the input partials and F32 output.
void attention_merge_partials_q8(const float* scratch, float* out, void* q8, int parts, void* stream);
void attention_merge_partials_q8_graph(const float* scratch, float* out, void* q8,
                                       const int32_t* position, int split, int max_parts, void* stream);

// Finite logits [rows,384], selection bias [384], outputs [rows,6]. Selection is
// descending logits+bias, ties to lower expert ID. Weights are unbiased sigmoid;
// they are not normalized over the selected experts.
void route(const float* logits, const float* bias, int32_t* ids, float* weights, int rows, void* stream);
// Opt-in single-row equivalent, with warp selection and the same sigmoid weights.
void route_decode(const float* logits, const float* bias, int32_t* ids, float* weights, void* stream);
void swiglu(const float* gate, const float* up, float* out, int rows, int width, void* stream);
// Accurate SwiGLU and native Q8_1 bytes in one launch. count is a positive
// multiple of 32; q8 holds count/32 * 36 bytes. All four buffers are disjoint.
void swiglu_quantize_q8_1(const float* gate, const float* up, float* hidden,
                         void* q8, int count, void* stream);

// IDs address valid source/destination rows. Gather output does not alias x.
void gather_rows(const float* x, const int32_t* row_ids, float* out, int rows, int width, void* stream);
// Add weighted rows to an initialized output. Duplicate destination IDs are
// supported using atomic adds; expert-group callers normally have distinct IDs.
void scatter_add_weighted(const float* x, const int32_t* row_ids, const float* weights,
                          float* out, int rows, int width, void* stream);

}  // namespace strata::kolibri::kernels
