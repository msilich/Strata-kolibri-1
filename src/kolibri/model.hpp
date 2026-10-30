#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::kolibri {

struct Matrix {
    int type = -1, in = 0, out = 0;
    const uint8_t* device = nullptr;
};

struct Layer {
    bool local = false;
    const float *attn_norm = nullptr, *attn_post_norm = nullptr;
    const float *ffn_norm = nullptr, *ffn_post_norm = nullptr;
    const float *q_norm = nullptr, *k_norm = nullptr;
    const float *router = nullptr, *router_bias = nullptr;
    Matrix q, k, v, o, shared_gate, shared_up, shared_down;
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    void load(const std::string& path, cudaStream_t stream);

    std::vector<Layer> layers;
    Matrix embedding, head;
    const float* out_norm = nullptr;
    int vocab = 128000, eos = -1;
    float eps = 1e-6f, rope_base = 10000.f;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Executor;
};

// Compute uses the caller stream; optional pinned prefill uploads fence at gather.
// Token-major contiguous inputs and outputs. Construct after the
// caller's KV cache and workspaces so the remaining VRAM determines the cache.
// expert_slots is per layer; cache_slots() reports the total across all layers.
class Executor {
public:
    Executor(Model& model, cudaStream_t stream, int max_tokens = 256,
             double expert_vram_gib = -1, int expert_slots = -1, bool grouped_decode = false,
             bool decode_cache_lru = false, bool fused_decode = false, bool grouped_prefill = false,
             bool pin_experts = false, bool overlap_prefill_uploads = false, bool gpu_expert_cache = false,
             bool overlap_decode_cache = false);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    void multiply(const Matrix& matrix, const float* x, float* out, int tokens);
    // Opt-in T=1 Q8_0 projections sharing a 2560-wide input, count is 2 or 3.
    // Returns its Q8_1 input, valid until the next multiply_decode on this stream;
    // ordinary multiply uses separate scratch and does not overwrite it.
    const void* multiply_decode(const Matrix* matrices, float* const* outputs, int count, const float* input);
    // T=1 shared expert only; keeps multiply_decode's input Q8 live for routing.
    void shared_down_decode(const Matrix& matrix, const float* gate, const float* up,
                            float* hidden, float* out);
    // T1 attention merge may write 6912 native Q8_1 bytes here. The O consumer
    // must be queued before any ordinary multiply/shared-down uses this scratch.
    void* attention_decode_q8();
    void attention_output_decode_q8(const Matrix& matrix, float* out);
    void expert(int layer, int expert, const float* x, float* out, int tokens);
    bool supports_grouped_prefill(int layer) const;
    // Up to 16 ascending experts, each with at least one row. Host bounds partition
    // <=max_tokens device row IDs selecting x; output uses that grouped order.
    // Q4_K/Q6_K/Q8_0 roles, including mixed formats; admission remains non-evicting.
    void experts_prefill(int layer, const int32_t* experts, int count, const int32_t* bounds,
                         const int32_t* row_ids, const float* x, float* out, int rows);
    // Six distinct expert IDs in ascending order, one input token, six output rows.
    // Requires grouped_decode at construction; cache misses remain live together.
    // input_q8 may reuse multiply_decode's quantization of x; otherwise x is
    // quantized here. Its device buffer must remain live through this call.
    void experts_decode(int layer, const int32_t* experts, const float* x, float* out,
                        const void* input_q8 = nullptr);
    // Optional device cache. Completion reconciles every valid receipt
    // before reporting invalid router IDs; CPU cache remains usable afterward.
    void begin_gpu_decode();
    // With overlap_decode_cache, called after the router GEMM, before the shared
    // expert chain. Routes and Plan are produced together; consumers wait later.
    void prepare_experts_decode_gpu(int layer, const float* logits, const float* bias,
                                    int32_t* route_ids, float* route_weights);
    void experts_decode_gpu(int layer, const int32_t* route_ids, const float* input,
                            float* outputs, const float* route_weights, float* destination,
                            const void* input_q8);
    void complete_gpu_decode();
    // Opt-in KV growth boundary: drain both streams, validate the full remaining
    // KV commitment, and empty/reopen a smaller cache only when VRAM is needed.
    // Counters/tick and pinned source arenas survive; no work may be queued here.
    void prepare_kv_growth(size_t additional_bytes, size_t remaining_bytes);
    void embed(const int32_t* ids, float* out, int tokens);
    uint64_t hits() const;
    uint64_t misses() const;
    uint64_t cache_bytes() const;
    int cache_slots() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::kolibri
