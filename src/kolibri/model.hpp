#pragma once

#include <cuda_runtime_api.h>

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

// One stream, token-major contiguous inputs and outputs. Construct after the
// caller's KV cache and workspaces so the remaining VRAM determines the cache.
// expert_slots is per layer; cache_slots() reports the total across all layers.
class Executor {
public:
    Executor(Model& model, cudaStream_t stream, int max_tokens = 256,
             double expert_vram_gib = -1, int expert_slots = -1);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    void multiply(const Matrix& matrix, const float* x, float* out, int tokens);
    void expert(int layer, int expert, const float* x, float* out, int tokens);
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
