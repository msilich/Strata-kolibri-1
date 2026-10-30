// Optional Kolibri driver. The GGUF weights and existing model paths are unchanged.
#include "model.hpp"
#include "kernels.cuh"
#include "decode_experts.cuh"
#include "decode_norms.cuh"
#include "prefill_attention.cuh"
#include "tensor_decode_attention.cuh"
#include "kv_cache.cuh"
#include "strata/core/pinned_ring.hpp"
#include "strata/core/vmm.hpp"
#include "strata/core/logit_bias.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/qwen35/sampling.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <poll.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace K = strata::kolibri::kernels;
using strata::kolibri::Model;
using strata::kolibri::Executor;
using strata::qwen35::Sampling;
using Clock = std::chrono::steady_clock;

namespace {
constexpr int H = 2560, Q = 6144, KV = 512, FF = 512, NE = 384, TOP = 6;
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void bc(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string((int) e));
}
double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { ck(cudaStreamCreate(&value)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Blas {
    cublasHandle_t value = nullptr;
    explicit Blas(cudaStream_t s) {
        bc(cublasCreate(&value));
        bc(cublasSetStream(value, s));
        bc(cublasSetMathMode(value, CUBLAS_PEDANTIC_MATH));
    }
    ~Blas() { if (value) cublasDestroy(value); }
};
template<class T> struct Buffer {
    T* p = nullptr;
    Buffer() = default;
    explicit Buffer(size_t n) { alloc(n); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void alloc(size_t n) { ck(cudaMalloc(&p, n * sizeof(T))); }
};

struct DecodeRouteDownload {
    cudaStream_t stream = nullptr;
    int32_t* ids = nullptr;
    cudaEvent_t ready = nullptr;
    void init(cudaStream_t s) {
        stream = s;
        ck(cudaHostAlloc(&ids, TOP * sizeof(int32_t), cudaHostAllocDefault));
        ck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    }
    ~DecodeRouteDownload() {
        // Also covers an exception after a copy but before its event was recorded.
        if (ids) cudaStreamSynchronize(stream);
        if (ready) cudaEventDestroy(ready);
        if (ids) cudaFreeHost(ids);
    }
};

// Stream spans include host submission gaps. CPU spans include existing waits;
// neither is a sum of kernel execution times. No extra synchronization is used.
class ForwardProfile {
public:
    enum Stage { Embed, Qkv, Kv, LocalAttention, GlobalAttention, AttentionOutput,
                 RouterShared, RouteDownload, RouteGroup, Experts, FfnPost, Head, Count };
    explicit ForwardProfile(cudaStream_t s) : stream(s) {
        for (auto& event : events) ck(cudaEventCreate(&event));
    }
    ~ForwardProfile() { for (auto event : events) if (event) cudaEventDestroy(event); }
    void begin(int64_t p, int t, bool d, uint64_t h, uint64_t m) {
        position = p; tokens = t; decode = d; hits = h; misses = m;
        used = 0; cpu.fill(0); started = last = Clock::now();
        mark(Embed);
    }
    void mark(Stage next) {
        const auto now = Clock::now();
        if (used) cpu[(size_t) stages[used - 1]] +=
            std::chrono::duration<double, std::milli>(now - last).count();
        if (used == events.size()) throw std::runtime_error("profile event capacity exceeded");
        ck(cudaEventRecord(events[used], stream));
        stages[used++] = next; last = now;
    }
    void end() {
        wall = ms(started);
        mark(Head);
    }
    void report(uint64_t h, uint64_t m) {
        std::array<double, Count> gpu{};
        for (size_t i = 1; i < used; ++i) {
            float elapsed = 0;
            ck(cudaEventElapsedTime(&elapsed, events[i - 1], events[i]));
            gpu[(size_t) stages[i - 1]] += elapsed;
        }
        static const char* names[] = {"embed", "qkv", "kv", "local_attention", "global_attention",
            "attention_output", "router_shared", "route_download", "route_group", "experts", "ffn_post", "head"};
        std::fprintf(stderr, "{\"kolibri_profile\":1,\"phase\":\"%s\",\"position\":%lld,\"tokens\":%d,"
                     "\"submit_wall_ms\":%.6f,\"forward_wall_ms\":%.6f,\"cache_hits\":%llu,\"cache_misses\":%llu,\"stages\":{",
                     decode ? "decode" : "prefill", (long long) position, tokens, wall, ms(started),
                     (unsigned long long) (h - hits), (unsigned long long) (m - misses));
        for (size_t i = 0; i < Count; ++i)
            std::fprintf(stderr, "%s\"%s\":{\"stream_ms\":%.6f,\"cpu_ms\":%.6f}",
                         i ? "," : "", names[i], gpu[i], cpu[i]);
        std::fprintf(stderr, "}}\n");
    }
private:
    cudaStream_t stream;
    std::array<cudaEvent_t, 602> events{};
    std::array<Stage, 602> stages{};
    std::array<double, Count> cpu{};
    size_t used = 0;
    int64_t position = 0;
    int tokens = 0;
    bool decode = false;
    uint64_t hits = 0, misses = 0;
    double wall = 0;
    Clock::time_point started, last;
};

class Engine {
public:
    Stream stream;
    Blas blas{stream.value};
    Model model;
    const int context, chunk;
    const bool split_attention, split_local_attention, grouped_experts, fast_prefill, fused_decode, tensor_decode, gpu_greedy, gpu_expert_cache, kv_grow, overlap_decode_cache, two_query_prefill, decode_graphs, kv_int8;
    int64_t pos = 0;
    std::unique_ptr<Executor> exec;
    std::ofstream layer_dump, router_dump, route_ids_dump;

    Engine(const std::string& path, int ctx, int ch, double cache_gib, int cache_slots, bool profiling, bool split,
           bool split_local, bool grouped, bool decode_cache_lru, bool prefill, bool fused, bool tensor,
           bool grouped_prefill, bool pin_experts, bool greedy, bool overlap_prefill_uploads, bool device_cache, bool grow_kv,
           bool overlap_decode = false, bool two_query = false, bool graphs = false, bool int8_kv = false)
        : context(ctx), chunk(ch), split_attention(split), split_local_attention(split_local), grouped_experts(grouped),
          fast_prefill(prefill), fused_decode(fused), tensor_decode(tensor), gpu_greedy(greedy), gpu_expert_cache(device_cache), kv_grow(grow_kv),
          overlap_decode_cache(overlap_decode), two_query_prefill(two_query), decode_graphs(graphs), kv_int8(int8_kv) {
        if (kv_int8 && (!fast_prefill || !tensor_decode))
            throw std::runtime_error("--kv int8 requires --prefill-attention --tensor-decode-attention");
        if (two_query_prefill && !fast_prefill)
            throw std::runtime_error("--two-query-prefill requires --prefill-attention");
        if ((fast_prefill || tensor_decode) && !K::prefill_attention_supported())
            throw std::runtime_error("tensor attention requires CUDA compute capability 8.0 or newer");
        if (kv_grow && !strata::core::vmm_available())
            throw std::runtime_error("--kv-grow requires CUDA virtual memory support");
        model.load(path, stream.value);
        const size_t t = (size_t) chunk;
        x.alloc(t * H); norm.alloc(t * H); mixed.alloc(t * H); moe.alloc(t * H);
        q.alloc(t * Q); k.alloc(t * KV); v.alloc(t * KV); attn.alloc(t * Q);
        if (fast_prefill || fused_decode) rope_sin_cos.alloc(t * K::kHeadDim);
        if (split_attention || split_local_attention || tensor_decode)
            attention_scratch.alloc(std::max({split_attention ? K::attention_split_scratch_floats(context) : 0,
                                              split_local_attention ? K::attention_local_split_scratch_floats() : 0,
                                              tensor_decode ? K::tensor_decode_attention_scratch_floats(context) : 0}));
        gate.alloc(t * FF); up.alloc(t * FF); hidden.alloc(t * FF);
        router.alloc(t * NE); route_ids.alloc(t * TOP); route_weights.alloc(t * TOP);
        group_ids.alloc(t); group_weights.alloc(t); gathered.alloc(t * H); expert_out.alloc(t * H);
        if (grouped_experts) grouped_out.alloc(TOP * H);
        ids.alloc(t); logits.alloc(model.vocab);
        if (decode_graphs) graph_position.alloc(1);
        if (gpu_greedy) {
            greedy_scratch.alloc(strata::kernels::argmax_rows_scratch_bytes(1));
            greedy_status.alloc(strata::kernels::kNonfiniteBlocks + 1);
            ck(cudaMemsetAsync(greedy_scratch.p, 0, strata::kernels::argmax_rows_scratch_bytes(1), stream.value));
        }
        const size_t local_cells = fused_decode ? (size_t) (512 + chunk) : 512;
        local_k.alloc(local_cells * KV); local_v.alloc(local_cells * KV);
        if (kv_grow) {
            kv_granularity = strata::core::vmm_granularity();
            kv_cells = std::min(context, 32768);
        }
        for (size_t l = 0; l < model.layers.size(); ++l) {
            const size_t cells = model.layers[l].local ? (size_t) (512 + chunk) : (size_t) context;
            if (kv_grow && !model.layers[l].local) {
                for (auto* range : {&global_k[l], &global_v[l]}) {
                    if (!range->reserve(cells * global_cell_bytes()) ||
                        !range->map_range(0, kv_chunks(kv_cells), [] { return strata::core::VmmChunk(0); }))
                        throw std::runtime_error("Kolibri: initial global KV virtual-memory allocation failed");
                    kv_full_bytes += (size_t) range->chunks() * kv_granularity;
                    kv_mapped_bytes += (size_t) range->mapped_count() * kv_granularity;
                }
            } else {
                const size_t words = model.layers[l].local ? KV : global_cell_bytes() / sizeof(uint16_t);
                caches[l].k.alloc(cells * words); caches[l].v.alloc(cells * words);
            }
        }
        exec = std::make_unique<Executor>(model, stream.value, chunk, cache_gib, cache_slots, grouped_experts,
                                          decode_cache_lru, fused_decode, grouped_prefill, pin_experts, overlap_prefill_uploads, gpu_expert_cache, overlap_decode_cache);
        if (fused_decode && grouped_experts) route_download.init(stream.value);
        if (profiling) profile = std::make_unique<ForwardProfile>(stream.value);
        if (kv_grow) {
            exec->prepare_kv_growth(0, kv_full_bytes - kv_mapped_bytes);
            std::fprintf(stderr, "Kolibri KV growth: %d / %d cells mapped; high-water mark retained until exit\n",
                         kv_cells, context);
        }
        if (two_query_prefill)
            std::fprintf(stderr, "Kolibri two-query global prefill enabled (at least two prompt tokens per chunk)\n");
        if (kv_int8)
            std::fprintf(stderr, "Kolibri global KV: INT8, %zu bytes per layer/token; local KV remains FP16\n",
                         2 * global_cell_bytes());
        std::fprintf(stderr, "Kolibri-1: context %d, chunk %d, expert cache %d slots / %.2f GiB\n",
                     context, chunk, exec->cache_slots(), (double) exec->cache_bytes() / (1ull << 30));
    }

    ~Engine() { if (kv_grow) cudaStreamSynchronize(stream.value); }

    // Reset keeps mapped capacity and the current expert cache. Only new token
    // positions grow global KV; local history keeps its existing pointer swaps.
    void reset() {
        if (gpu_expert_cache) exec->complete_gpu_decode();
        decode_graph.clear();
        pos = 0;
    }

    void ensure_kv(int cells) {
        if (!kv_grow || cells <= kv_cells) return;
        if (cells > context) throw std::runtime_error("KV growth exceeds configured context");
        // A resize can replace the expert cache addresses captured in the graph.
        decode_graph.clear();
        const int target = std::min(context, (cells + 32767) / 32768 * 32768);
        const int64_t first = kv_chunks(kv_cells), end = kv_chunks(target);
        size_t additional = 0;
        for (size_t l = 0; l < model.layers.size(); ++l)
            if (!model.layers[l].local) additional += 2 * (size_t) (end - first) * kv_granularity;
        exec->prepare_kv_growth(additional, kv_full_bytes - kv_mapped_bytes);
        for (size_t l = 0; l < model.layers.size(); ++l) {
            if (model.layers[l].local) continue;
            for (auto* range : {&global_k[l], &global_v[l]})
                if (!range->map_range(first, end, [] { return strata::core::VmmChunk(0); }))
                    throw std::runtime_error("Kolibri: global KV growth allocation failed; configured context unchanged");
        }
        // append_kv writes every newly visible token before any attention reads
        // it. Partial allocation failure is fatal; VmmRange owns its cleanup.
        kv_mapped_bytes += additional;
        kv_cells = target;
        std::fprintf(stderr, "Kolibri KV growth: %d / %d cells mapped, %.2f GiB global KV\n",
                     kv_cells, context, double(kv_mapped_bytes) / (1ull << 30));
    }

    // A chunk and decode use the same layer equations. Local history is compacted
    // only after every query has consumed the old history and the whole new chunk.
    void forward(const int32_t* tokens, int T, bool decode = false) {
        if (T < 1 || T > chunk || pos + T > context) throw std::runtime_error("context or chunk limit exceeded");
        for (int i = 0; i < T; ++i)
            if (tokens[i] < 0 || tokens[i] >= model.vocab) throw std::runtime_error("token outside vocabulary");
        if (kv_grow) ensure_kv((int) pos + T);
        auto s = stream.value;
        const bool fuse_decode = fused_decode && decode && T == 1;
        const bool group_decode = grouped_experts && decode && T == 1;
        const bool device_decode = gpu_expert_cache && fuse_decode && group_decode &&
            !layer_dump.is_open() && !router_dump.is_open() && !route_ids_dump.is_open();
        const bool overlap_route = fuse_decode && group_decode && !device_decode;
        const bool cache_rope = (fast_prefill && !decode) || fuse_decode;
        if (device_decode) exec->begin_gpu_decode();
        try {
        if (profile) profile->begin(pos, T, decode, exec->hits(), exec->misses());
        const int graph_limit = std::min(context, std::min(pos + 1 < 8192 ? 8191 : context,
                                                         ((int(pos) + 1 + 1023) / 1024) * 1024));
        auto enqueue = [&](bool graph) {
        if (!graph) ck(transfer.h2d(ids.p, tokens, (size_t) T * sizeof(int32_t), s));
        exec->embed(ids.p, x.p, T);
        if (graph) K::rope_table_decode_graph(rope_sin_cos.p, graph_position.p, model.rope_base, s);
        else if (cache_rope) K::rope_table(rope_sin_cos.p, T, pos, model.rope_base, s);
        std::vector<float> dumps;
        std::vector<float> router_dumps;
        std::vector<int32_t> route_ids_dumps;
        if (layer_dump.is_open()) dumps.resize((size_t) T * 50 * H);
        if (router_dump.is_open()) router_dumps.resize((size_t) T * 50 * NE);
        if (route_ids_dump.is_open()) route_ids_dumps.resize((size_t) T * 50 * TOP);
        for (int l = 0; l < 50; ++l) {
            const auto& L = model.layers[(size_t) l];
            profile_mark(ForwardProfile::Qkv);
            if (!fuse_decode || l == 0)
                K::rms_norm(x.p, L.attn_norm, norm.p, T, H, model.eps, s);
            if (fuse_decode) {
                const strata::kolibri::Matrix matrices[] = {L.q, L.k, L.v};
                float* outputs[] = {q.p, k.p, v.p};
                exec->multiply_decode(matrices, outputs, 3, norm.p);
            } else {
                exec->multiply(L.q, norm.p, q.p, T);
                exec->multiply(L.k, norm.p, k.p, T);
                exec->multiply(L.v, norm.p, v.p, T);
            }
            profile_mark(ForwardProfile::Kv);
            const int prior = L.local ? (int) std::min<int64_t>(pos, 512) : (int) pos;
            auto& cache = caches[(size_t) l];
            auto* cache_k = kv_grow && !L.local ? reinterpret_cast<uint16_t*>(global_k[l].base()) : cache.k.p;
            auto* cache_v = kv_grow && !L.local ? reinterpret_cast<uint16_t*>(global_v[l].base()) : cache.v.p;
            if (kv_int8 && !L.local && fuse_decode) {
                K::qk_norm_append_decode_int8(q.p, k.p, v.p, L.q_norm, L.k_norm, cache_k, cache_v,
                                              prior, graph ? graph_position.p : nullptr, model.eps, s);
            } else if (graph && !L.local) {
                K::qk_norm_append_decode_graph(q.p, k.p, v.p, L.q_norm, L.k_norm,
                                               cache_k, cache_v, graph_position.p, model.eps, s);
            } else if (fuse_decode) {
                K::qk_norm_rope_append_decode(q.p, k.p, v.p, L.q_norm, L.k_norm, rope_sin_cos.p,
                                               L.local, cache_k, cache_v, prior, model.eps, s);
            } else {
                if (cache_rope && L.local)
                    K::qk_norm_rope_cached(q.p, k.p, L.q_norm, L.k_norm, T, rope_sin_cos.p, model.eps, s);
                else
                    K::qk_norm_rope(q.p, k.p, L.q_norm, L.k_norm, T, pos, L.local, model.rope_base, model.eps, s);
                if (kv_int8 && !L.local) strata::kolibri::kv::append_int8(k.p, v.p, cache_k, cache_v, T, prior, s);
                else K::append_kv(k.p, v.p, cache_k, cache_v, T, prior, s);
            }
            profile_mark(L.local ? ForwardProfile::LocalAttention : ForwardProfile::GlobalAttention);
            bool attention_q8_ready = false;
            if (kv_int8 && !L.local) {
                if (decode && T == 1) {
                    K::tensor_decode_attention_int8(q.p, cache_k, cache_v, attn.p, attention_scratch.p,
                                                     graph ? graph_limit : prior + T, s,
                                                     fuse_decode ? exec->attention_decode_q8() : nullptr,
                                                     graph ? graph_position.p : nullptr);
                    attention_q8_ready = fuse_decode;
                } else {
                    K::prefill_attention_int8(q.p, cache_k, cache_v, attn.p, T, prior + T,
                                              pos, two_query_prefill, s);
                }
            } else if (graph && !L.local) {
                K::tensor_decode_attention_q8_graph(q.p, cache_k, cache_v, attn.p, attention_scratch.p,
                                                     graph_position.p, graph_limit, s, exec->attention_decode_q8());
                attention_q8_ready = true;
            } else if (two_query_prefill && !decode && !L.local && T >= 2)
                K::prefill_attention_two_query(q.p, cache_k, cache_v, attn.p, T, prior + T, pos, s);
            else if (fast_prefill && !decode)
                K::prefill_attention(q.p, cache_k, cache_v, attn.p, T, prior + T,
                                     L.local ? pos - prior : 0, pos, L.local, s);
            else if (tensor_decode && decode && T == 1 && (!L.local || prior == 512)) {
                if (fuse_decode) {
                    K::tensor_decode_attention_q8(q.p, cache_k, cache_v, attn.p, attention_scratch.p,
                                                  prior + T, L.local, s, exec->attention_decode_q8());
                    attention_q8_ready = true;
                } else
                    K::tensor_decode_attention(q.p, cache_k, cache_v, attn.p, attention_scratch.p,
                                              prior + T, L.local, s);
            }
            else if (split_attention && decode && T == 1 && !L.local && prior + T >= 1024)
                K::attention_split_decode(q.p, cache_k, cache_v, attn.p, prior + T,
                                          pos, attention_scratch.p, s);
            else if (split_local_attention && decode && T == 1 && L.local && prior == 512)
                K::attention_local_split_decode(q.p, cache_k, cache_v, attn.p, attention_scratch.p, s);
            else
                K::attention(q.p, cache_k, cache_v, attn.p, T, prior + T,
                             L.local ? pos - prior : 0, pos, L.local, s);
            profile_mark(ForwardProfile::AttentionOutput);
            if (L.local && prior + T > 512) {
                const size_t bytes = 512 * KV * sizeof(uint16_t);
                const size_t offset = (size_t) (prior + T - 512) * KV;
                ck(cudaMemcpyAsync(local_k.p, cache.k.p + offset, bytes, cudaMemcpyDeviceToDevice, s));
                ck(cudaMemcpyAsync(local_v.p, cache.v.p + offset, bytes, cudaMemcpyDeviceToDevice, s));
                if (fuse_decode && !graph) {
                    // Equal-capacity buffers retain unique ownership across reset/prefill.
                    std::swap(cache.k.p, local_k.p); std::swap(cache.v.p, local_v.p);
                } else {
                    ck(cudaMemcpyAsync(cache.k.p, local_k.p, bytes, cudaMemcpyDeviceToDevice, s));
                    ck(cudaMemcpyAsync(cache.v.p, local_v.p, bytes, cudaMemcpyDeviceToDevice, s));
                }
            }
            if (attention_q8_ready) exec->attention_output_decode_q8(L.o, mixed.p);
            else exec->multiply(L.o, attn.p, mixed.p, T);
            if (fuse_decode)
                K::decode_residual_norm(mixed.p, L.attn_post_norm, x.p, L.ffn_norm, norm.p, model.eps, s);
            else {
                K::rms_norm(mixed.p, L.attn_post_norm, norm.p, T, H, model.eps, s);
                K::add(x.p, norm.p, x.p, (size_t) T * H, s);
            }
            profile_mark(ForwardProfile::RouterShared);
            if (!fuse_decode) K::rms_norm(x.p, L.ffn_norm, norm.p, T, H, model.eps, s);
            const float one = 1.f, zero = 0.f;
            bc(cublasSgemm(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, NE, T, H, &one,
                           L.router, H, norm.p, H, &zero, router.p, NE));
            if (device_decode && overlap_decode_cache)
                exec->prepare_experts_decode_gpu(l, router.p, L.router_bias, route_ids.p, route_weights.p);
            else if (fuse_decode) K::route_decode(router.p, L.router_bias, route_ids.p, route_weights.p, s);
            else K::route(router.p, L.router_bias, route_ids.p, route_weights.p, T, s);
            if (overlap_route) {
                ck(cudaMemcpyAsync(route_download.ids, route_ids.p, TOP * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
                ck(cudaEventRecord(route_download.ready, s));
            }
            const void* shared_input_q8 = nullptr;
            if (fuse_decode) {
                const strata::kolibri::Matrix matrices[] = {L.shared_gate, L.shared_up};
                float* outputs[] = {gate.p, up.p};
                shared_input_q8 = exec->multiply_decode(matrices, outputs, 2, norm.p);
            } else {
                exec->multiply(L.shared_gate, norm.p, gate.p, T);
                exec->multiply(L.shared_up, norm.p, up.p, T);
            }
            if (fuse_decode) {
                exec->shared_down_decode(L.shared_down, gate.p, up.p, hidden.p, moe.p);
            } else {
                K::swiglu(gate.p, up.p, hidden.p, T, FF, s);
                exec->multiply(L.shared_down, hidden.p, moe.p, T);
            }
            if (device_decode) {
                profile_mark(ForwardProfile::RouteDownload);
                profile_mark(ForwardProfile::RouteGroup);
                profile_mark(ForwardProfile::Experts);
                exec->experts_decode_gpu(l, route_ids.p, norm.p, grouped_out.p, route_weights.p, moe.p, shared_input_q8);
            } else {
            profile_mark(ForwardProfile::RouteDownload);
            std::vector<int32_t> selected((size_t) T * TOP);
            std::vector<float> weights(group_decode ? 0 : (size_t) T * TOP);
            if (overlap_route) {
                // Only the IDs must be ready for host cache lookup/staging. The
                // routed kernels follow shared work (and dense_q8) on this stream.
                ck(cudaEventSynchronize(route_download.ready));
                std::copy_n(route_download.ids, TOP, selected.data());
            } else {
                ck(transfer.d2h_sync(selected.data(), route_ids.p, selected.size() * sizeof(int32_t), s));
            }
            if (!group_decode)
                ck(transfer.d2h_sync(weights.data(), route_weights.p, weights.size() * sizeof(float), s));
            profile_mark(ForwardProfile::RouteGroup);
            if (router_dump.is_open()) {
                std::vector<float> row((size_t) T * NE);
                ck(transfer.d2h_sync(row.data(), router.p, row.size() * sizeof(float), s));
                for (int t = 0; t < T; ++t)
                    std::copy_n(row.data() + (size_t) t * NE, NE, router_dumps.data() + ((size_t) t * 50 + l) * NE);
            }
            if (route_ids_dump.is_open())
                for (int t = 0; t < T; ++t)
                    std::copy_n(selected.data() + (size_t) t * TOP, TOP,
                                route_ids_dumps.data() + ((size_t) t * 50 + l) * TOP);
            if (group_decode) {
                std::array<int32_t, TOP> order{};
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return selected[a] < selected[b]; });
                std::array<int32_t, TOP> sorted{};
                for (int j = 0; j < TOP; ++j) sorted[j] = selected[order[j]];
                profile_mark(ForwardProfile::Experts);
                exec->experts_decode(l, sorted.data(), norm.p, grouped_out.p, shared_input_q8);
                K::combine_decode_experts(grouped_out.p, route_weights.p, order.data(), moe.p, s);
            } else {
                std::array<std::vector<int32_t>, NE> rows;
                std::array<std::vector<float>, NE> scales;
                for (int t = 0; t < T; ++t) for (int j = 0; j < TOP; ++j) {
                    const int i = t * TOP + j, e = selected[(size_t) i];
                    if (e < 0 || e >= NE) throw std::runtime_error("non-finite or invalid router output");
                    rows[(size_t) e].push_back(t); scales[(size_t) e].push_back(weights[(size_t) i]);
                }
                profile_mark(ForwardProfile::Experts);
                if (!decode && exec->supports_grouped_prefill(l)) {
                    std::vector<int32_t> experts, bounds{0}, ids;
                    std::vector<float> factors;
                    const auto flush = [&] {
                        if (experts.empty()) return;
                        ck(transfer.h2d(group_ids.p, ids.data(), ids.size() * sizeof(int32_t), s));
                        ck(transfer.h2d(group_weights.p, factors.data(), factors.size() * sizeof(float), s));
                        exec->experts_prefill(l, experts.data(), (int) experts.size(), bounds.data(), group_ids.p,
                                              norm.p, expert_out.p, (int) ids.size());
                        // Separate launches preserve the original ascending-expert
                        // FP32 accumulation order, including tokens in several experts.
                        for (size_t i = 0; i < experts.size(); ++i) {
                            const int first = bounds[i], n = bounds[i + 1] - first;
                            K::scatter_add_weighted(expert_out.p + (size_t) first * H, group_ids.p + first,
                                                    group_weights.p + first, moe.p, n, H, s);
                        }
                        experts.clear(); bounds.resize(1); ids.clear(); factors.clear();
                    };
                    for (int e = 0; e < NE; ++e) {
                        const int n = (int) rows[e].size();
                        if (!n) continue;
                        if (experts.size() == 16 || ids.size() + n > (size_t) chunk) flush();
                        experts.push_back(e);
                        ids.insert(ids.end(), rows[e].begin(), rows[e].end());
                        factors.insert(factors.end(), scales[e].begin(), scales[e].end());
                        bounds.push_back((int) ids.size());
                    }
                    flush();
                } else {
                    for (int e = 0; e < NE; ++e) {
                        const int n = (int) rows[(size_t) e].size();
                        if (!n) continue;
                        ck(transfer.h2d(group_ids.p, rows[(size_t) e].data(), (size_t) n * sizeof(int32_t), s));
                        ck(transfer.h2d(group_weights.p, scales[(size_t) e].data(), (size_t) n * sizeof(float), s));
                        K::gather_rows(norm.p, group_ids.p, gathered.p, n, H, s);
                        exec->expert(l, e, gathered.p, expert_out.p, n);
                        K::scatter_add_weighted(expert_out.p, group_ids.p, group_weights.p, moe.p, n, H, s);
                    }
                }
            }
            }
            profile_mark(ForwardProfile::FfnPost);
            if (fuse_decode) {
                const float* next_norm = l + 1 < 50 ? model.layers[(size_t) l + 1].attn_norm : model.out_norm;
                K::decode_residual_norm(moe.p, L.ffn_post_norm, x.p, next_norm, norm.p, model.eps, s);
            } else {
                K::rms_norm(moe.p, L.ffn_post_norm, norm.p, T, H, model.eps, s);
                K::add(x.p, norm.p, x.p, (size_t) T * H, s);
            }
            if (layer_dump.is_open()) {
                std::vector<float> row((size_t) T * H);
                ck(transfer.d2h_sync(row.data(), x.p, row.size() * sizeof(float), s));
                for (int t = 0; t < T; ++t)
                    std::copy_n(row.data() + (size_t) t * H, H, dumps.data() + ((size_t) t * 50 + l) * H);
            }
        }
        if (!dumps.empty()) {
            layer_dump.write(reinterpret_cast<const char*>(dumps.data()), (std::streamsize) (dumps.size() * sizeof(float)));
            if (!layer_dump) throw std::runtime_error("cannot write layer dump");
        }
        if (!router_dumps.empty()) {
            router_dump.write(reinterpret_cast<const char*>(router_dumps.data()),
                              (std::streamsize) (router_dumps.size() * sizeof(float)));
            if (!router_dump) throw std::runtime_error("cannot write router dump");
        }
        if (!route_ids_dumps.empty()) {
            route_ids_dump.write(reinterpret_cast<const char*>(route_ids_dumps.data()),
                                 (std::streamsize) (route_ids_dumps.size() * sizeof(int32_t)));
            if (!route_ids_dump) throw std::runtime_error("cannot write route ID dump");
        }
        profile_mark(ForwardProfile::Head);
        if (!fuse_decode)
            K::rms_norm(x.p + (size_t) (T - 1) * H, model.out_norm, norm.p, 1, H, model.eps, s);
        exec->multiply(model.head, norm.p, logits.p, 1);
        };
        const bool use_graph = decode_graphs && device_decode && graph_warmed && pos >= 512;
        if (use_graph) {
            // Input staging is outside capture: PinnedRing may rotate its host
            // buffers. The graph reads only these fixed device allocations.
            const int32_t position = (int32_t) pos;
            ck(transfer.h2d(ids.p, tokens, sizeof(int32_t), s));
            ck(transfer.h2d(graph_position.p, &position, sizeof(position), s));
            try {
                if (!decode_graph.executable || decode_graph.limit != graph_limit) {
                    decode_graph.clear();
                    decode_graph.capture(s, [&] { enqueue(true); });
                    decode_graph.limit = graph_limit;
                    std::fprintf(stderr, "Kolibri decode graph: %zu nodes, global keys <= %d\n",
                                 decode_graph.nodes, graph_limit);
                }
                ck(cudaGraphLaunch(decode_graph.executable, s));
                exec->gpu_decode_graph_submitted();
            } catch (...) {
                // Capture does not execute its cache plans. Never reconcile old
                // receipts as if a failed capture had processed a real token.
                exec->cancel_gpu_decode_capture();
                throw;
            }
        } else {
            // Warm the exact cuBLAS/native decode paths before capturing them.
            // This is an actual token, so the cache is never advanced speculatively.
            decode_graph.clear();
            enqueue(false);
            if (device_decode) graph_warmed = true;
        }
        if (profile) profile->end();
        if (device_decode) exec->complete_gpu_decode();
        else ck(cudaStreamSynchronize(s));
        if (profile) profile->report(exec->hits(), exec->misses());
        pos += T;
        } catch (...) {
            // Partial forwards may already have queued cache mutations. Replay
            // every completed valid receipt before propagating the exception.
            if (device_decode) exec->complete_gpu_decode();
            throw;
        }
    }
    void host_logits(std::vector<float>& dst) {
        if (gpu_expert_cache) exec->complete_gpu_decode();
        dst.resize(model.vocab);
        ck(transfer.d2h_sync(dst.data(), logits.p, dst.size() * sizeof(float), stream.value));
        if (!std::all_of(dst.begin(), dst.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("model produced non-finite logits");
    }
    int greedy_token() {
        if (gpu_expert_cache) exec->complete_gpu_decode();
        constexpr int flags = strata::kernels::kNonfiniteBlocks;
        strata::kernels::argmax_rows(logits.p, 1, model.vocab, greedy_scratch.p,
                                    reinterpret_cast<int32_t*>(greedy_status.p + flags), stream.value);
        strata::kernels::logits_nonfinite_rows(logits.p, 1, model.vocab, greedy_status.p, stream.value);
        std::array<uint32_t, flags + 1> result{};
        ck(transfer.d2h_sync(result.data(), greedy_status.p, sizeof(result), stream.value));
        if (std::any_of(result.begin(), result.begin() + flags, [](uint32_t v) { return v != 0; }))
            throw std::runtime_error("model produced non-finite logits");
        return (int) result[flags];
    }
private:
    std::unique_ptr<ForwardProfile> profile;
    void profile_mark(ForwardProfile::Stage stage) { if (profile) profile->mark(stage); }
    struct Cache { Buffer<uint16_t> k, v; };
    std::array<Cache, 50> caches;
    std::array<strata::core::VmmRange, 50> global_k, global_v;
    size_t kv_granularity = 0, kv_full_bytes = 0, kv_mapped_bytes = 0;
    int kv_cells = 0;
    size_t global_cell_bytes() const {
        return kv_int8 ? strata::kolibri::kv::Int8CellBytes : KV * sizeof(uint16_t);
    }
    int64_t kv_chunks(int cells) const {
        return (int64_t) (((size_t) cells * global_cell_bytes() + kv_granularity - 1) / kv_granularity);
    }
    Buffer<float> x, norm, mixed, moe, q, k, v, attn, gate, up, hidden, router;
    Buffer<float> route_weights, group_weights, gathered, expert_out, logits;
    Buffer<float> grouped_out;
    Buffer<int32_t> route_ids, group_ids, ids;
    Buffer<uint16_t> local_k, local_v;
    Buffer<float> attention_scratch, rope_sin_cos;
    Buffer<uint8_t> greedy_scratch;
    Buffer<uint32_t> greedy_status;
    strata::core::PinnedRing transfer;
    DecodeRouteDownload route_download;
    Buffer<int32_t> graph_position;
    bool graph_warmed = false;
    struct DecodeGraph {
        cudaGraphExec_t executable = nullptr;
        int limit = 0;
        size_t nodes = 0;
        ~DecodeGraph() { clear(); }
        void clear() {
            if (executable) cudaGraphExecDestroy(executable);
            executable = nullptr;
            limit = 0;
        }
        template<class Submit> void capture(cudaStream_t stream, Submit submit) {
            cudaGraph_t graph = nullptr;
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try { submit(); }
            catch (...) {
                cudaStreamEndCapture(stream, &graph);
                if (graph) cudaGraphDestroy(graph);
                throw;
            }
            const auto ended = cudaStreamEndCapture(stream, &graph);
            if (ended != cudaSuccess) {
                if (graph) cudaGraphDestroy(graph);
                ck(ended);
            }
            const auto counted = cudaGraphGetNodes(graph, nullptr, &nodes);
            if (counted != cudaSuccess) { cudaGraphDestroy(graph); ck(counted); }
            const auto instantiated = cudaGraphInstantiate(&executable, graph, 0);
            cudaGraphDestroy(graph);
            ck(instantiated);
        }
    } decode_graph;
};

std::vector<int32_t> parse_ids(const std::string& text) {
    std::vector<int32_t> result;
    std::istringstream input(text);
    std::string token;
    while (std::getline(input, token, ',')) {
        size_t end = 0;
        long value = std::stol(token, &end);
        if (end != token.size() || value < 0 || value > INT32_MAX) throw std::runtime_error("invalid token IDs");
        result.push_back((int32_t) value);
    }
    if (result.empty()) throw std::runtime_error("empty token IDs");
    return result;
}

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist,
           std::mt19937_64& rng, const std::vector<float>& bias) {
    strata::qwen35::apply_penalties(lg, s, hist);
    if (!bias.empty()) for (size_t i = 0; i < lg.size(); ++i) lg[i] += bias[i];
    if (s.temperature <= 0) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const size_t count = s.top_k > 0 ? std::min((size_t) s.top_k, lg.size()) : lg.size();
    std::vector<int> order(lg.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int a, int b) {
        return lg[(size_t) a] != lg[(size_t) b] ? lg[(size_t) a] > lg[(size_t) b] : a < b;
    });
    order.resize(count);
    std::vector<double> probabilities(count);
    double sum = 0, highest = lg[(size_t) order[0]];
    for (size_t i = 0; i < count; ++i) sum += probabilities[i] = std::exp((lg[(size_t) order[i]] - highest) / s.temperature);
    double cumulative = 0;
    size_t keep = 0;
    for (; keep < count; ++keep) {
        if (keep && probabilities[keep] < s.min_p * probabilities[0]) break;
        cumulative += probabilities[keep];
        if (cumulative / sum >= s.top_p) { ++keep; break; }
    }
    keep = std::max<size_t>(1, keep);
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.begin() + keep);
    return order[distribution(rng)];
}

void print_logprobs(const std::vector<float>& lg, int token, int count, const std::vector<int32_t>& allowed) {
    double mx = *std::max_element(lg.begin(), lg.end()), sum = 0;
    for (float v : lg) sum += std::exp((double) v - mx);
    const double lse = mx + std::log(sum);
    std::printf("LP %.6f", lg[(size_t) token] - lse);
    std::vector<int> order;
    if (!allowed.empty()) {
        for (int id : allowed) if (id != token) std::printf(" %d:%.6f", id, lg[(size_t) id] - lse);
        std::printf("\n");
        return;
    }
    order.resize(lg.size()); std::iota(order.begin(), order.end(), 0);
    const size_t n = std::min<size_t>((size_t) count, order.size());
    std::partial_sort(order.begin(), order.begin() + n, order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    for (size_t i = 0; i < n; ++i) std::printf(" %d:%.6f", order[i], lg[(size_t) order[i]] - lse);
    std::printf("\n");
}

// Polling makes shutdown joinable even when stdin is still open. STOP epochs
// also cover cancellation arriving after GEN was queued but before it was read.
class Input {
public:
    struct Command { std::string line; uint64_t epoch; };
    std::atomic<uint64_t> cancelled{0};
    Input() : reader([this] { read(); }) {}
    ~Input() { shutting_down = true; reader.join(); }
    bool next(Command& command) {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return eof || !queue.empty(); });
        if (queue.empty()) return false;
        command = std::move(queue.front()); queue.pop_front(); return true;
    }
private:
    std::atomic<bool> shutting_down{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Command> queue;
    bool eof = false;
    std::thread reader;
    void line(std::string text) {
        if (!text.empty() && text.back() == '\r') text.pop_back();
        if (text == "STOP") { ++cancelled; return; }
        std::lock_guard<std::mutex> lock(mu);
        queue.push_back({std::move(text), cancelled.load()}); cv.notify_one();
    }
    void read() {
        std::string pending;
        while (!shutting_down) {
            pollfd fd{STDIN_FILENO, POLLIN, 0};
            const int ready = ::poll(&fd, 1, 100);
            if (ready < 0) { if (errno == EINTR) continue; break; }
            if (!ready) continue;
            char buffer[8192];
            const ssize_t n = ::read(STDIN_FILENO, buffer, sizeof(buffer));
            if (n <= 0) break;
            pending.append(buffer, (size_t) n);
            size_t end;
            while ((end = pending.find('\n')) != std::string::npos) {
                line(pending.substr(0, end)); pending.erase(0, end + 1);
            }
        }
        if (!pending.empty()) line(pending);
        std::lock_guard<std::mutex> lock(mu); eof = true; cv.notify_one();
    }
};

struct Request {
    int max_new = 0, logprobs = -1;
    Sampling sampling;
    std::vector<int32_t> ids, allowed;
    std::vector<float> bias;
};

Request request(const std::string& line, int vocab, int ctx) {
    Request r; r.sampling.top_k = 128;
    std::istringstream input(line);
    std::string verb, item;
    if (!(input >> verb >> r.max_new) || verb != "GEN" || r.max_new < 1)
        throw std::runtime_error("expected GEN <max_new> [key=value ...] <id,id,...>");
    while (input >> item) {
        const size_t eq = item.find('=');
        if (eq == std::string::npos) {
            r.ids = parse_ids(item);
            if (input >> item) throw std::runtime_error("extra request fields");
            break;
        }
        const std::string key = item.substr(0, eq), v = item.substr(eq + 1);
        if (key == "temperature") r.sampling.temperature = std::stof(v);
        else if (key == "top_p") r.sampling.top_p = std::stof(v);
        else if (key == "top_k") r.sampling.top_k = std::stoi(v);
        else if (key == "min_p") r.sampling.min_p = std::stof(v);
        else if (key == "penalty_last_n") r.sampling.last_n = std::stoi(v);
        else if (key == "penalty_repeat") r.sampling.repeat = std::stof(v);
        else if (key == "penalty_freq") r.sampling.freq = std::stof(v);
        else if (key == "penalty_present") r.sampling.present = std::stof(v);
        else if (key == "seed") r.sampling.seed = std::stoull(v);
        else if (key == "logprobs") r.logprobs = std::stoi(v);
        else if (key == "allowed") r.allowed = parse_ids(v);
        else if (key == "logit_bias") {
            std::string error;
            if (!strata::core::parse_logit_bias(v, vocab, r.bias, error)) throw std::runtime_error(error);
        } else if (key != "ckpt" && key != "pin" && key != "cvec")
            throw std::runtime_error("unsupported request option: " + key);
    }
    const auto& p = r.sampling;
    for (float f : {p.temperature, p.top_p, p.min_p, p.repeat, p.freq, p.present})
        if (!std::isfinite(f)) throw std::runtime_error("non-finite sampling parameter");
    if (p.temperature < 0 || p.top_p <= 0 || p.top_p > 1 || p.min_p < 0 || p.min_p > 1 || p.repeat <= 0 || p.top_k < 0 ||
        r.logprobs < -1 || r.logprobs > 20) throw std::runtime_error("invalid sampling parameters");
    if (r.ids.empty() || r.ids.size() + (size_t) r.max_new > (size_t) ctx)
        throw std::runtime_error("empty prompt or prompt + response exceeds context");
    for (const auto& list : {r.ids, r.allowed}) for (int id : list)
        if (id < 0 || id >= vocab) throw std::runtime_error("token ID outside vocabulary");
    return r;
}

int serve(Engine& engine) {
    std::printf("INFO family=kolibri1 sampling_top_k=128 batch_slots=0\nREADY %d stop\n", engine.context);
    std::fflush(stdout);
    Input input;
    Input::Command command;
    std::vector<float> lg;
    while (input.next(command)) {
        if (command.line.empty()) continue;
        Request r;
        try { r = request(command.line, engine.model.vocab, engine.context); }
        catch (const std::exception& error) { std::printf("ERR %s\n", error.what()); std::fflush(stdout); continue; }
        const auto cancelled = [&] { return input.cancelled.load() != command.epoch; };
        const auto& sampling = r.sampling;
        const bool gpu_pick = engine.gpu_greedy && sampling.temperature == 0.f && r.bias.empty() && r.logprobs < 0 &&
            (sampling.last_n == 0 || (sampling.repeat == 1.f && sampling.freq == 0.f && sampling.present == 0.f));
        engine.reset();
        auto start = Clock::now();
        size_t read = 0;
        while (read < r.ids.size() && !cancelled()) {
            const int n = (int) std::min<size_t>(engine.chunk, r.ids.size() - read);
            engine.forward(r.ids.data() + read, n); read += n;
            std::printf("PP %zu %zu %.0f %.1f\n", read, r.ids.size(), ms(start), 1000.0 * read / std::max(ms(start), 0.001));
            std::fflush(stdout);
        }
        const double prefill_ms = ms(start);
        auto decode = Clock::now();
        int produced = 0;
        std::string finish = "length";
        auto history = r.ids;
        std::mt19937_64 rng(r.sampling.seed);
        while (produced < r.max_new && !cancelled()) {
            int token;
            if (gpu_pick) {
                token = engine.greedy_token();
                std::printf("T %d\n", token);
            } else {
                engine.host_logits(lg);
                const auto raw_logits = r.logprobs >= 0 ? lg : std::vector<float>{};
                token = sample(lg, r.sampling, history, rng, r.bias);
                std::printf("T %d\n", token);
                if (r.logprobs >= 0) print_logprobs(raw_logits, token, r.logprobs, r.allowed);
            }
            std::fflush(stdout);
            ++produced; history.push_back(token);
            if (token == engine.model.eos) { finish = "stop"; break; }
            if (produced < r.max_new && !cancelled()) engine.forward(&history.back(), 1, true);
        }
        if (cancelled()) finish = "cancel";
        std::printf("DONE %d %zu %.1f %.1f %s 0 0 0 0 0 0 0 0.0 %zu 0\n", produced, r.ids.size(), prefill_ms,
                     ms(decode), finish.c_str(), read);
        std::fflush(stdout);
        std::fprintf(stderr, "Kolibri: %zu prompt tokens / %.0f ms; %d output tokens / %.0f ms; cache hits=%llu misses=%llu\n",
                     read, prefill_ms, produced, ms(decode), (unsigned long long) engine.exec->hits(),
                     (unsigned long long) engine.exec->misses());
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string model, tokens, forced, layer_path, logits_path, router_path, route_ids_path;
        std::string kv_type = "fp16";
        int ctx = 32768, chunk = 256, n = 64, slots = -1;
        double vram = -1;
        bool server = false, profiling = false, split_attention = false, split_local_attention = false,
             grouped_experts = false, decode_cache_lru = false, prefill_attention = false,
             fused_decode = false, tensor_decode_attention = false, grouped_prefill = false, pin_experts = false,
             gpu_greedy = false, overlap_prefill_uploads = false, gpu_expert_cache = false, kv_grow = false,
             overlap_decode_cache = false, two_query_prefill = false, decode_graphs = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--version") { std::cout << "strata-kolibri " << STRATA_VERSION << '\n'; return 0; }
            else if (arg == "--families") { std::cout << "kolibri1\n"; return 0; }
            else if (arg == "--help") {
                std::cout << "strata-kolibri --native MODEL [--max-context 1..262144 (default 32768)] [--chunk 1..4096 (default 256)]\n"
                             "  --serve | --tokens ID,ID,... [--n 64] [--force ID,ID,...]\n"
                             "  --expert-vram GiB | --expert-slots N (per layer, 0 disables cache)\n"
                             "  --kv fp16|int8 (default fp16; int8 changes global KV only)\n"
                             "  --kv-grow (experimental: map global KV as needed, shrink expert cache; retain high-water mark until exit)\n"
                             "  --dump-layers FILE --dump-full-logits FILE (diagnostic binary F32)\n"
                             "  --dump-router FILE (F32 [position,50,384]) --dump-route-ids FILE (I32 [position,50,6])\n"
                             "  --profile (diagnostic per-forward CPU/stream stage times as JSON on stderr)\n"
                             "  --split-k-attention (experimental global decode attention)\n"
                             "  --split-k-local-attention (experimental full-window local decode attention)\n"
                             "  --grouped-experts (experimental grouped single-token decode experts)\n"
                             "  --grouped-prefill (experimental batched Q4/Q6/Q8 prompt experts)\n"
                             "  --pin-experts (experimental pinned expert RAM; Linux only)\n"
                             "  --overlap-prefill-uploads (experimental; requires --pin-experts --grouped-prefill)\n"
                             "  --gpu-expert-cache (experimental; requires pin/grouped/LRU/fused decode)\n"
                             "  --overlap-decode-cache (experimental; requires --gpu-expert-cache --overlap-prefill-uploads)\n"
                             "  --gpu-greedy (GPU token choice for plain temperature-zero serving requests)\n"
                             "  --decode-cache-lru (experimental decode replacement; requires --grouped-experts)\n"
                             "  --prefill-attention (experimental tensor-core prompt attention; CUDA sm_80+)\n"
                             "  --two-query-prefill (share global prompt attention work across token pairs; requires --prefill-attention)\n"
                             "  --fused-decode (experimental combined decode projections and residual norms)\n"
                             "  --tensor-decode-attention (experimental tensor-core decode attention; CUDA sm_80+)\n";
                std::cout << "  --decode-graphs (optional token graph; requires device cache and tensor decode attention)\n";
                return 0;
            } else if (arg == "--native" || arg == "--model") model = next();
            else if (arg == "--max-context") ctx = std::stoi(next());
            else if (arg == "--chunk") chunk = std::stoi(next());
            else if (arg == "--kv") kv_type = next();
            else if (arg == "--expert-vram") vram = std::stod(next());
            else if (arg == "--expert-slots") slots = std::stoi(next());
            else if (arg == "--tokens") tokens = next();
            else if (arg == "--force") forced = next();
            else if (arg == "--n" || arg == "--max-new") n = std::stoi(next());
            else if (arg == "--dump-layers") layer_path = next();
            else if (arg == "--dump-full-logits") logits_path = next();
            else if (arg == "--dump-router") router_path = next();
            else if (arg == "--dump-route-ids") route_ids_path = next();
            else if (arg == "--serve") server = true;
            else if (arg == "--profile") profiling = true;
            else if (arg == "--split-k-attention") split_attention = true;
            else if (arg == "--split-k-local-attention") split_local_attention = true;
            else if (arg == "--grouped-experts") grouped_experts = true;
            else if (arg == "--grouped-prefill") grouped_prefill = true;
            else if (arg == "--pin-experts") pin_experts = true;
            else if (arg == "--overlap-prefill-uploads") overlap_prefill_uploads = true;
            else if (arg == "--gpu-expert-cache") gpu_expert_cache = true;
            else if (arg == "--overlap-decode-cache") overlap_decode_cache = true;
            else if (arg == "--kv-grow") kv_grow = true;
            else if (arg == "--gpu-greedy") gpu_greedy = true;
            else if (arg == "--decode-cache-lru") decode_cache_lru = true;
            else if (arg == "--prefill-attention") prefill_attention = true;
            else if (arg == "--two-query-prefill") two_query_prefill = true;
            else if (arg == "--fused-decode") fused_decode = true;
            else if (arg == "--tensor-decode-attention") tensor_decode_attention = true;
            else if (arg == "--decode-graphs") decode_graphs = true;
            else throw std::runtime_error("unknown option: " + arg);
        }
        if (model.empty() || (kv_type != "fp16" && kv_type != "int8") || ctx < 1 || ctx > 262144 || chunk < 1 || chunk > K::kChunk || n < 1 ||
            slots < -1 || slots > 384 || !std::isfinite(vram) || (vram < 0 && vram != -1) || (!server && tokens.empty()))
            throw std::runtime_error("invalid options; use --help");
        if (decode_cache_lru && !grouped_experts)
            throw std::runtime_error("--decode-cache-lru requires --grouped-experts");
        if (overlap_prefill_uploads && (!pin_experts || !grouped_prefill))
            throw std::runtime_error("--overlap-prefill-uploads requires --pin-experts and --grouped-prefill");
#if !defined(__linux__)
        if (pin_experts) throw std::runtime_error("--pin-experts currently requires Linux");
#endif
        if (gpu_expert_cache && !(pin_experts && grouped_experts && decode_cache_lru && fused_decode))
            throw std::runtime_error("--gpu-expert-cache requires --pin-experts --grouped-experts --decode-cache-lru --fused-decode");
        if (overlap_decode_cache && !(gpu_expert_cache && overlap_prefill_uploads))
            throw std::runtime_error("--overlap-decode-cache requires --gpu-expert-cache --overlap-prefill-uploads");
        if (decode_graphs && (!gpu_expert_cache || !tensor_decode_attention || profiling ||
                              !layer_path.empty() || !router_path.empty() || !route_ids_path.empty()))
            throw std::runtime_error("--decode-graphs requires --gpu-expert-cache --tensor-decode-attention and no --profile or layer/router dumps");
        auto ids = server ? std::vector<int32_t>{} : parse_ids(tokens);
        auto force = forced.empty() ? std::vector<int32_t>{} : parse_ids(forced);
        if (!force.empty()) n = (int) force.size() + 1;
        if (!server && ids.size() + (size_t) n > (size_t) ctx) throw std::runtime_error("prompt + response exceeds context");
        if (!std::getenv("STRATA_MMQ_KQ")) setenv("STRATA_MMQ_KQ", "1", 0);
        if (!std::getenv("CUDA_MODULE_LOADING")) setenv("CUDA_MODULE_LOADING", "EAGER", 0);
        Engine engine(model, ctx, chunk, vram, slots, profiling, split_attention, split_local_attention,
                      grouped_experts, decode_cache_lru, prefill_attention, fused_decode, tensor_decode_attention,
                      grouped_prefill, pin_experts, gpu_greedy, overlap_prefill_uploads, gpu_expert_cache, kv_grow,
                      overlap_decode_cache, two_query_prefill, decode_graphs, kv_type == "int8");
        if (server) return serve(engine);
        if (!layer_path.empty()) {
            engine.layer_dump.open(layer_path, std::ios::binary);
            if (!engine.layer_dump) throw std::runtime_error("cannot open layer dump");
        }
        if (!router_path.empty()) {
            engine.router_dump.open(router_path, std::ios::binary);
            if (!engine.router_dump) throw std::runtime_error("cannot open router dump");
        }
        if (!route_ids_path.empty()) {
            engine.route_ids_dump.open(route_ids_path, std::ios::binary);
            if (!engine.route_ids_dump) throw std::runtime_error("cannot open route ID dump");
        }
        std::ofstream output;
        if (!logits_path.empty()) {
            output.open(logits_path, std::ios::binary);
            if (!output) throw std::runtime_error("cannot open logit dump");
        }
        const auto start = Clock::now();
        for (size_t p = 0; p < ids.size(); p += (size_t) chunk)
            engine.forward(ids.data() + p, (int) std::min<size_t>(chunk, ids.size() - p));
        std::vector<float> lg;
        std::cout << "gen:";
        for (int i = 0; i < n; ++i) {
            engine.host_logits(lg);
            if (output.is_open()) {
                output.write(reinterpret_cast<const char*>(lg.data()), (std::streamsize) (lg.size() * sizeof(float)));
                if (!output) throw std::runtime_error("cannot write logit dump");
            }
            const int32_t token = (size_t) i < force.size() ? force[(size_t) i] :
                (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
            std::cout << ' ' << token << std::flush;
            if (i + 1 < n && (token != engine.model.eos || !force.empty())) engine.forward(&token, 1, true);
            else break;
        }
        std::cout << '\n';
        std::fprintf(stderr, "Kolibri: %.0f ms total; cache hits=%llu misses=%llu\n", ms(start),
                     (unsigned long long) engine.exec->hits(), (unsigned long long) engine.exec->misses());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-kolibri: %s\n", e.what());
        return 1;
    }
}
