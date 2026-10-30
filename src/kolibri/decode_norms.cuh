#pragma once

namespace strata::kolibri::kernels {

// One Kolibri decode row (2560 values): normalize update, add it to residual,
// then normalize the new residual. Each operation retains the original F32
// rounding and 256-thread RMS reduction order. All five arrays must be disjoint.
void decode_residual_norm(const float* update, const float* post_weight,
                          float* residual, const float* next_weight,
                          float* normalized, float eps, void* stream);

}  // namespace strata::kolibri::kernels
