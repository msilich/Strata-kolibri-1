#include "model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned_ring.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace strata::kolibri {
namespace {
namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;
constexpr int kLayers = 50, kHidden = 2560, kExperts = 384, kFf = 512;
constexpr size_t kTail = 4096;

void check(cudaError_t e, const char* operation) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Kolibri: ") + operation + ": " + cudaGetErrorString(e));
}

size_t aligned(size_t bytes, size_t alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}

struct Allocations {
    std::vector<void*> pointers;
    ~Allocations() { for (void* p : pointers) cudaFree(p); }
    uint8_t* allocate(size_t bytes) {
        void* p = nullptr;
        check(cudaMalloc(&p, bytes), "allocate device memory");
        pointers.push_back(p);
        return static_cast<uint8_t*>(p);
    }
};

struct ExpertRole {
    int type = -1, in = 0, out = 0;
    const uint8_t* source = nullptr;
    size_t bytes = 0, offset = 0;
};

struct ExpertLayer {
    std::array<ExpertRole, 3> roles;
    size_t stride = 0;
};

__global__ void swiglu(const float* gate, const float* up, float* h, int n) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) h[i] = gate[i] / (1.f + expf(-gate[i])) * up[i];
}
}  // namespace

struct Model::Impl {
    Allocations memory;
    std::unique_ptr<strata::GgufModel> file;
    std::vector<ExpertLayer> experts;
};

Model::Model() = default;
Model::~Model() = default;

void Model::load(const std::string& path, cudaStream_t stream) {
    if (impl_) throw std::runtime_error("Kolibri: this model is already loaded");
    auto next = std::make_unique<Impl>();
    next->file = std::make_unique<strata::GgufModel>(strata::gguf_split_paths(path));
    const auto& file = *next->file;
    const auto& meta = file.meta();
    const auto* arch = meta.get("general.architecture");
    if (!arch || arch->type != strata::MetaType::STRING || arch->s != "kolibri1")
        throw std::runtime_error("Kolibri: expected GGUF architecture kolibri1");
    auto expect = [&](const char* suffix, double wanted) {
        const std::string key = std::string("kolibri1.") + suffix;
        const auto* v = meta.get(key);
        if (!v || !v->is_num() || v->num() != wanted)
            throw std::runtime_error("Kolibri: unsupported or missing " + key);
    };
    expect("block_count", kLayers);
    expect("embedding_length", kHidden);
    expect("attention.head_count", 48);
    expect("attention.head_count_kv", 4);
    expect("attention.key_length", 128);
    expect("attention.value_length", 128);
    expect("rope.dimension_count", 128);
    expect("rope.freq_base", 10000);
    expect("expert_count", kExperts);
    expect("expert_used_count", 6);
    expect("expert_feed_forward_length", kFf);
    expect("expert_shared_feed_forward_length", kFf);
    expect("expert_shared_count", 1);
    expect("expert_gating_func", 2);
    expect("expert_weights_norm", 0);
    expect("attention.sliding_window", 513);
    expect("vocab_size", 128000);
    const auto* eos_token = meta.get("tokenizer.ggml.eos_token_id");
    if (!eos_token || !eos_token->is_num() || !std::isfinite(eos_token->num()) ||
        eos_token->num() < 0 || eos_token->num() >= 128000 ||
        eos_token->num() != std::floor(eos_token->num()))
        throw std::runtime_error("Kolibri: invalid or missing EOS token ID");
    const auto* epsilon = meta.get("kolibri1.attention.layer_norm_rms_epsilon");
    if (!epsilon || !epsilon->is_num() || !std::isfinite(epsilon->num()) || epsilon->num() <= 0)
        throw std::runtime_error("Kolibri: invalid RMS normalization epsilon");
    const auto* pattern = meta.get("kolibri1.attention.sliding_window_pattern");
    if (!pattern || pattern->type != strata::MetaType::ARRAY || pattern->items.size() != kLayers)
        throw std::runtime_error("Kolibri: missing 50-layer sliding window pattern");
    for (int l = 0; l < kLayers; ++l)
        if (pattern->items[l].type != strata::MetaType::BOOL ||
            (pattern->items[l].u != 0) != (l % 5 != 4))
            throw std::runtime_error("Kolibri: expected four local layers followed by one global layer");

    auto tensor = [&](const std::string& name, std::vector<uint64_t> shape) {
        size_t shard = 0;
        const auto* t = file.find(name, &shard);
        if (!t || t->shape != shape || !file.in_bounds(*t, shard))
            throw std::runtime_error("Kolibri: missing, truncated or incorrectly shaped tensor " + name);
        return std::make_pair(t, file.shard(shard).tensor_data(*t));
    };
    auto floating = [&](const std::string& name, std::vector<uint64_t> shape) {
        const auto [t, source] = tensor(name, std::move(shape));
        if (t->type != 0)
            throw std::runtime_error("Kolibri: expected F32 normalization/router tensor " + name);
        const size_t bytes = (size_t) t->elements() * sizeof(float);
        for (size_t i = 0; i < bytes; i += sizeof(float)) {
            float value;
            std::memcpy(&value, source + i, sizeof value);
            if (!std::isfinite(value)) throw std::runtime_error("Kolibri: non-finite tensor " + name);
        }
        uint8_t* d = next->memory.allocate(bytes);
        check(cudaMemcpyAsync(d, source, bytes, cudaMemcpyHostToDevice, stream), "upload F32 tensor");
        return reinterpret_cast<const float*>(d);
    };
    auto matrix = [&](const std::string& name, int in, int out) {
        const auto [t, source] = tensor(name, {(uint64_t) in, (uint64_t) out});
        if (t->type != 8)
            throw std::runtime_error("Kolibri: this initial path requires Q8_0 dense weights: " + name);
        const size_t bytes = K::native_mmvq_weight_bytes((int) t->type, in, out);
        uint8_t* d = next->memory.allocate(bytes + kTail);
        check(cudaMemcpyAsync(d, source, bytes, cudaMemcpyHostToDevice, stream), "upload dense matrix");
        check(cudaMemsetAsync(d + bytes, 0, kTail, stream), "pad dense matrix");
        return Matrix{(int) t->type, in, out, d};
    };
    std::vector<Layer> loaded(kLayers);
    next->experts.resize(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Layer& y = loaded[l];
        y.local = l % 5 != 4;
        y.attn_norm = floating(p + "attn_norm.weight", {kHidden});
        y.attn_post_norm = floating(p + "post_attention_norm.weight", {kHidden});
        y.ffn_norm = floating(p + "ffn_norm.weight", {kHidden});
        y.ffn_post_norm = floating(p + "post_ffw_norm.weight", {kHidden});
        y.q_norm = floating(p + "attn_q_norm.weight", {128});
        y.k_norm = floating(p + "attn_k_norm.weight", {128});
        y.router = floating(p + "ffn_gate_inp.weight", {kHidden, kExperts});
        y.router_bias = floating(p + "exp_probs_b.bias", {kExperts});
        y.q = matrix(p + "attn_q.weight", kHidden, 48 * 128);
        y.k = matrix(p + "attn_k.weight", kHidden, 4 * 128);
        y.v = matrix(p + "attn_v.weight", kHidden, 4 * 128);
        y.o = matrix(p + "attn_output.weight", 48 * 128, kHidden);
        y.shared_gate = matrix(p + "ffn_gate_shexp.weight", kHidden, kFf);
        y.shared_up = matrix(p + "ffn_up_shexp.weight", kHidden, kFf);
        y.shared_down = matrix(p + "ffn_down_shexp.weight", kFf, kHidden);
        ExpertLayer& e = next->experts[l];
        const char* roles[] = {"gate", "up", "down"};
        size_t at = 0;
        for (int r = 0; r < 3; ++r) {
            auto& role = e.roles[r];
            role.in = r == 2 ? kFf : kHidden;
            role.out = r == 2 ? kHidden : kFf;
            const auto [t, source] = tensor(p + "ffn_" + roles[r] + "_exps.weight",
                                          {(uint64_t) role.in, (uint64_t) role.out, kExperts});
            if (t->type != 8 && t->type != 12 && t->type != 14)
                throw std::runtime_error("Kolibri: expert weights must use Q4_K, Q6_K or Q8_0");
            role.type = (int) t->type;
            role.source = source;
            role.bytes = K::native_mmvq_weight_bytes(role.type, role.in, role.out);
            role.offset = at;
            at = aligned(at + role.bytes + kTail, 16);
        }
        // Each MMQ product has one matrix, so its pointer need not be an integer
        // number of another role's quantization blocks. Match ExpertCache's slots.
        e.stride = aligned(at, 256);
    }
    Matrix emb = matrix("token_embd.weight", kHidden, 128000);
    Matrix output = matrix("output.weight", kHidden, 128000);
    const float* norm = floating("output_norm.weight", {kHidden});
    check(cudaStreamSynchronize(stream), "finish model upload");
    layers = std::move(loaded);
    embedding = emb;
    head = output;
    out_norm = norm;
    eos = (int) eos_token->num();
    eps = (float) epsilon->num();
    impl_ = std::move(next);
}

struct Executor::Impl {
    Model& model;
    cudaStream_t stream;
    int max_tokens;
    Allocations memory;
    strata::core::ExpertCache cache;
    strata::core::PinnedRing uploads;
    MMQ::Context mmq;
    std::map<std::pair<int, int>, bool> mmq_support;
    void *q8 = nullptr, *mq8 = nullptr;
    float *gate = nullptr, *up = nullptr, *hidden = nullptr;
    int32_t *bounds = nullptr, *ids = nullptr;
    uint8_t* miss = nullptr;
    std::vector<uint8_t> pageable;
    uint64_t hits = 0, misses = 0;

    Impl(Model& m, cudaStream_t s, int t) : model(m), stream(s), max_tokens(t) {}
    ~Impl() { cudaStreamSynchronize(stream); }

    void check_tokens(int tokens) const {
        if (tokens <= 0 || tokens > max_tokens)
            throw std::runtime_error("Kolibri: token batch exceeds the executor workspace");
    }
    void multiply(const Matrix& w, const float* x, float* out, int tokens) {
        check_tokens(tokens);
        if (!w.device || w.in <= 0 || w.in > 6144 || w.out <= 0 || !K::native_mmvq_supported(w.type))
            throw std::runtime_error("Kolibri: invalid matrix passed to executor");
        const auto key = std::make_pair(w.type, w.out);
        auto cap = mmq_support.find(key);
        if (cap == mmq_support.end())
            cap = mmq_support.emplace(key, MMQ::built() && MMQ::fits(w.type, w.out)).first;
        if (tokens > 8 && cap->second) {
            MMQ::quantize(x, nullptr, mq8, w.type, w.in, w.in, tokens, stream);
            MMQ::Product p;
            p.w = w.device;
            p.type = w.type;
            p.w_rows = w.out;
            p.w_cols = w.in;
            p.expert_bytes = K::native_mmvq_weight_bytes(w.type, w.in, w.out);
            p.n = 1;
            p.xq = mq8;
            p.bounds = bounds + 2 * tokens;
            p.ids = ids;
            p.total_rows = p.max_rows = tokens;
            p.dst = out;
            p.ld_dst = w.out;
            mmq.run(p, stream);
        } else {
            // The quantizer accepts at most eight columns, but all 32-value
            // blocks stay within a token when the contiguous batch is flattened.
            K::native_quantize_q8_1(x, q8, w.in * tokens, 1, stream);
            const size_t row = K::native_q8_1_bytes(w.in);
            for (int first = 0; first < tokens; first += 8)
                K::native_mmvq(w.type, w.device, (const uint8_t*) q8 + first * row,
                               out + (size_t) first * w.out, w.in, w.out, std::min(8, tokens - first), stream);
        }
        check(cudaPeekAtLastError(), "matrix multiplication");
    }
};

Executor::Executor(Model& model, cudaStream_t stream, int max_tokens, double expert_vram_gib, int expert_slots) {
    if (!model.impl_) throw std::runtime_error("Kolibri: load a model before creating the executor");
    if (!stream) throw std::runtime_error("Kolibri: the executor requires an explicit CUDA stream");
    if (max_tokens < 1 || max_tokens > 256 || expert_slots < -1 || expert_slots > kExperts ||
        !std::isfinite(expert_vram_gib) || (expert_vram_gib < 0 && expert_vram_gib != -1))
        throw std::runtime_error("Kolibri: invalid executor/cache settings");
    auto p = std::make_unique<Impl>(model, stream, max_tokens);
    p->q8 = p->memory.allocate(K::native_q8_1_bytes(6144) * (size_t) max_tokens);
    p->mq8 = p->memory.allocate(MMQ::q8_bytes(max_tokens, 6144));
    p->gate = (float*) p->memory.allocate((size_t) max_tokens * kFf * sizeof(float));
    p->up = (float*) p->memory.allocate((size_t) max_tokens * kFf * sizeof(float));
    p->hidden = (float*) p->memory.allocate((size_t) max_tokens * kFf * sizeof(float));
    // Bounds are always {0, tokens}; keep every allowed batch size on device.
    std::vector<int32_t> host_bounds(2 * (max_tokens + 1), 0);
    for (int t = 1; t <= max_tokens; ++t) host_bounds[2 * t + 1] = t;
    p->bounds = (int32_t*) p->memory.allocate(host_bounds.size() * sizeof(int32_t));
    check(strata::core::PinnedRing::direct(p->bounds, host_bounds.data(), host_bounds.size() * sizeof(int32_t), stream),
          "upload MMQ bounds");
    p->ids = (int32_t*) p->memory.allocate((size_t) max_tokens * sizeof(int32_t));
    MMQ::iota(p->ids, max_tokens, stream);
    size_t max_blob = 0, one_per_layer = 0;
    for (const auto& e : model.impl_->experts) {
        max_blob = std::max(max_blob, e.stride);
        one_per_layer += e.stride;
    }
    p->miss = p->memory.allocate(max_blob);
    p->pageable.resize(max_blob);
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "query available VRAM");
    constexpr size_t reserve = (size_t) 768 << 20;
    size_t budget = free > reserve ? free - reserve : 0;
    if (expert_vram_gib >= 0) {
        const double wanted = expert_vram_gib * 1073741824.0;
        if (wanted > (double) budget)
            throw std::runtime_error("Kolibri: requested expert VRAM exceeds free memory after the 768 MiB reserve");
        budget = (size_t) wanted;
    }
    const int slots = expert_slots >= 0 ? expert_slots :
        (int) std::min<size_t>(kExperts, budget / one_per_layer);
    if ((size_t) slots * one_per_layer > budget)
        throw std::runtime_error("Kolibri: requested expert slots exceed the cache budget");
    if (slots > 0) {
        std::vector<int64_t> sizes;
        sizes.reserve((size_t) kLayers * slots);
        for (const auto& e : model.impl_->experts)
            sizes.insert(sizes.end(), slots, (int64_t) e.stride);
        std::string error;
        if (!p->cache.open_sized(sizes, kLayers, kExperts, error))
            throw std::runtime_error("Kolibri: " + error);
        p->cache.set_per_layer_admission(true);
    }
    std::fprintf(stderr, "Kolibri: %d expert slots per layer, %.2f GiB VRAM; other experts streamed from GGUF\n",
                 slots, (double) p->cache.bytes() / 1073741824.0);
    impl_ = std::move(p);
}

Executor::~Executor() = default;

void Executor::multiply(const Matrix& matrix, const float* x, float* out, int tokens) {
    impl_->multiply(matrix, x, out, tokens);
}

void Executor::expert(int layer, int expert, const float* x, float* out, int tokens) {
    auto& p = *impl_;
    p.check_tokens(tokens);
    if (layer < 0 || layer >= kLayers || expert < 0 || expert >= kExperts)
        throw std::runtime_error("Kolibri: expert index outside the model");
    const auto& e = p.model.impl_->experts[layer];
    int slot = p.cache.valid() ? p.cache.slot_of(layer, expert) : -1;
    uint8_t* weights = nullptr;
    if (slot >= 0) {
        ++p.hits;
        weights = p.cache.device_slot(slot);
    } else {
        ++p.misses;
        // Uploads and every consumer use this one stream. A newly admitted slot
        // is therefore read only after its copy, even before the host waits.
        slot = p.cache.valid() ? p.cache.admit(layer, expert) : -1;
        weights = slot >= 0 ? p.cache.device_slot(slot) : p.miss;
        uint8_t* stage = static_cast<uint8_t*>(p.uploads.acquire(e.stride));
        uint8_t* host = stage ? stage : p.pageable.data();
        std::memset(host, 0, e.stride);
        for (const auto& r : e.roles)
            std::memcpy(host + r.offset, r.source + (size_t) expert * r.bytes, r.bytes);
        if (stage) check(p.uploads.submit(weights, stage, e.stride, p.stream), "upload expert");
        else check(strata::core::PinnedRing::direct(weights, host, e.stride, p.stream), "upload expert");
    }
    auto matrix = [&](int role) {
        const auto& r = e.roles[role];
        return Matrix{r.type, r.in, r.out, weights + r.offset};
    };
    p.multiply(matrix(0), x, p.gate, tokens);
    p.multiply(matrix(1), x, p.up, tokens);
    const int n = tokens * kFf;
    swiglu<<<(n + 255) / 256, 256, 0, p.stream>>>(p.gate, p.up, p.hidden, n);
    check(cudaPeekAtLastError(), "expert SwiGLU");
    p.multiply(matrix(2), p.hidden, out, tokens);
}

void Executor::embed(const int32_t* ids, float* out, int tokens) {
    impl_->check_tokens(tokens);
    const auto& m = impl_->model.embedding;
    K::iq_embed_rows(m.type, m.device, K::iq_row_bytes(m.type, m.in), ids, tokens, m.in, out, impl_->stream);
    check(cudaPeekAtLastError(), "token embedding");
}

uint64_t Executor::hits() const { return impl_->hits; }
uint64_t Executor::misses() const { return impl_->misses; }
uint64_t Executor::cache_bytes() const { return (uint64_t) impl_->cache.bytes(); }
int Executor::cache_slots() const { return (int) impl_->cache.slots(); }

}  // namespace strata::kolibri
