#pragma once

namespace strata::kolibri::kernels {

// Host-side table, passed to the kernel by value. Every matrix is row-major
// GGUF Q8_0 with 2560 input features; output row counts may differ. Buffers are
// disjoint device pointers and output order follows the table.
struct DenseDecodeProjection {
    const void* weights[3];
    float* outputs[3];
    int rows[3];
    int count;
};

// Two or three projections of the same already-quantized Q8_1 input (2880
// bytes). Preserves native_mmvq's per-row dot and four-warp reduction order.
// Enqueues on the explicit non-null stream without allocation or synchronization.
void dense_decode_q8(const DenseDecodeProjection& projection, const void* input_q8, void* stream);

}  // namespace strata::kolibri::kernels
