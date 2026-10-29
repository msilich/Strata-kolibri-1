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
inline constexpr int kChunk = 256;

// out may alias x; weights has width entries.
void rms_norm(const float* x, const float* weights, float* out, int rows, int width, float eps, void* stream);
// out may alias either input.
void add(const float* a, const float* b, float* out, size_t n, void* stream);

// In-place per-head RMS norm, then full-head NeoX RoPE on local layers only.
// q [rows,48,128], k [rows,4,128], norm weights [128]; positions are absolute.
void qk_norm_rope(float* q, float* k, const float* qw, const float* kw, int rows,
                  int64_t first_pos, bool local, float theta, float eps, void* stream);
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

// Finite logits [rows,384], selection bias [384], outputs [rows,6]. Selection is
// descending logits+bias, ties to lower expert ID. Weights are unbiased sigmoid;
// they are not normalized over the selected experts.
void route(const float* logits, const float* bias, int32_t* ids, float* weights, int rows, void* stream);
void swiglu(const float* gate, const float* up, float* out, int rows, int width, void* stream);

// IDs address valid source/destination rows. Gather output does not alias x.
void gather_rows(const float* x, const int32_t* row_ids, float* out, int rows, int width, void* stream);
// Add weighted rows to an initialized output. Duplicate destination IDs are
// supported using atomic adds; expert-group callers normally have distinct IDs.
void scatter_add_weighted(const float* x, const int32_t* row_ids, const float* weights,
                          float* out, int rows, int width, void* stream);

}  // namespace strata::kolibri::kernels
