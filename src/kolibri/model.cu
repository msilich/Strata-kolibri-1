#include "model.hpp"
#include "decode_experts.cuh"
#include "decode_cache.cuh"
#include "dense_decode.cuh"
#include "kernels.cuh"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/pinned_ring.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
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

#if defined(__linux__)
size_t proc_kib(const char* path, const char* wanted) {
    FILE* file = std::fopen(path, "r");
    if (!file) throw std::runtime_error("Kolibri: cannot read pinning memory status");
    char line[256], key[64];
    unsigned long long kib = 0;
    bool found = false;
    while (std::fgets(line, sizeof line, file)) {
        if (std::sscanf(line, "%63[^:]: %llu kB", key, &kib) == 2 && std::strcmp(key, wanted) == 0) {
            found = true;
            break;
        }
    }
    std::fclose(file);
    if (!found) throw std::runtime_error("Kolibri: missing pinning memory status field");
    return (size_t) kib * 1024;
}
struct HostMemory {
    size_t available = proc_kib("/proc/meminfo", "MemAvailable");
    size_t swap_free = proc_kib("/proc/meminfo", "SwapFree");
    size_t process_swap = proc_kib("/proc/self/status", "VmSwap");
};
struct ReadOnlyFile {
    int fd;
    explicit ReadOnlyFile(const std::string& path) : fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
        if (fd < 0) throw std::runtime_error("Kolibri: cannot open GGUF shard for source release");
    }
    ~ReadOnlyFile() { ::close(fd); }
};
#endif

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

void stage_expert(const ExpertLayer& layer, int expert, uint8_t* host) {
    size_t end = 0;
    for (const auto& role : layer.roles) {
        // Only padding can retain stale bytes; every payload is overwritten.
        std::memset(host + end, 0, role.offset - end);
        std::memcpy(host + role.offset, role.source + (size_t) expert * role.bytes, role.bytes);
        end = role.offset + role.bytes;
    }
    std::memset(host + end, 0, layer.stride - end);
}

// Mixed formats need independent packed matrices: their quantization blocks
// have different byte strides. All Kolibri role payloads/offsets are 16-byte aligned.
__global__ void gather_mixed_roles(MMQ::GatherGroup group, size_t gate_bytes, size_t up_bytes, size_t down_bytes,
                                   size_t up_offset, size_t down_offset, uint8_t* gate, uint8_t* up, uint8_t* down) {
    const int role = blockIdx.z, expert = blockIdx.y;
    const size_t bytes = role == 0 ? gate_bytes : (role == 1 ? up_bytes : down_bytes);
    const size_t offset = role == 0 ? 0 : (role == 1 ? up_offset : down_offset);
    uint4* dst = reinterpret_cast<uint4*>(role == 0 ? gate : (role == 1 ? up : down));
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bytes / 16) {
        const uint4* src = reinterpret_cast<const uint4*>(group.blob[expert] + offset);
        dst[(size_t) expert * (bytes / 16) + i] = src[i];
    } else if (expert == group.n - 1 && i < (bytes + kTail) / 16) {
        dst[(size_t) expert * (bytes / 16) + i] = make_uint4(0, 0, 0, 0);
    }
}

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
    cudaStream_t prefill_upload = nullptr;
    cudaEvent_t raw_reusable = nullptr, copy_ready = nullptr;
    bool prefill_sequence = false;
    int max_tokens;
    Allocations memory;
    strata::core::ExpertCache cache;
    strata::core::PinnedRing uploads;
    MMQ::Context mmq;
    std::map<std::pair<int, int>, bool> mmq_support;
    void *q8 = nullptr, *mq8 = nullptr;
    void* dense_q8 = nullptr;
    float *gate = nullptr, *up = nullptr, *hidden = nullptr;
    int32_t *bounds = nullptr, *ids = nullptr;
    uint8_t* miss = nullptr;
    uint8_t* grouped_miss = nullptr;
    size_t grouped_stride = 0;
    void* grouped_q8 = nullptr;
    float* grouped_float = nullptr;
    std::array<bool, kLayers> prefill_layers{};
    uint8_t *prefill_miss = nullptr, *prefill_gu = nullptr, *prefill_down = nullptr;
    int32_t* prefill_bounds = nullptr;
    size_t prefill_stride = 0, prefill_up_offset = 0;
    std::vector<uint64_t> decode_recency;
    uint64_t decode_tick = 0;
    std::vector<uint8_t> pageable;
    uint64_t hits = 0, misses = 0;

    std::array<std::array<uint8_t*, 3>, kLayers> pinned_sources{};
    std::array<std::array<std::unique_ptr<strata::core::PinnedArena>, 3>, kLayers> pinned_arenas{};
    std::array<std::array<size_t, 3>, 2> miss_layout{};
    size_t miss_bytes = 0, group_miss_bytes = 0;
    bool pinned = false;
    bool gpu_cache_enabled = false, gpu_cache_current = false, gpu_active = false;
    int gpu_pending_layers = 0;
    bool overlap_decode_cache = false, gpu_aux_unfenced = false;
    int gpu_prepared_layer = -1;
    cudaEvent_t decode_route_ready = nullptr, decode_cache_ready = nullptr;
    struct Completion {
        decode_cache::Receipt receipt[kLayers];
        decode_cache::Counters counters;
    };
    decode_cache::LayerState* gpu_state = nullptr;
    decode_cache::Plan* gpu_plans = nullptr;
    Completion *gpu_completion = nullptr, *host_completion = nullptr;
    std::vector<decode_cache::LayerState> host_state;
    std::array<decode_cache::LayerView, kLayers> gpu_view{};
    std::array<decode_cache::Mirror, kLayers> gpu_mirror{};

    Impl(Model& m, cudaStream_t s, int t) : model(m), stream(s), max_tokens(t) {}
    ~Impl() {
        cudaStreamSynchronize(stream);
        if (prefill_upload) cudaStreamSynchronize(prefill_upload);
        if (host_completion) cudaFreeHost(host_completion);
        if (decode_route_ready) cudaEventDestroy(decode_route_ready);
        if (decode_cache_ready) cudaEventDestroy(decode_cache_ready);
        if (copy_ready) cudaEventDestroy(copy_ready);
        if (raw_reusable) cudaEventDestroy(raw_reusable);
        if (prefill_upload) cudaStreamDestroy(prefill_upload);
        // Owned arenas unregister/unmap only after both streams have drained.
    }

    void start_prefill_uploads() {
        check(cudaStreamCreateWithFlags(&prefill_upload, cudaStreamNonBlocking), "create prefill upload stream");
        check(cudaEventCreateWithFlags(&raw_reusable, cudaEventDisableTiming), "create raw reuse event");
        check(cudaEventCreateWithFlags(&copy_ready, cudaEventDisableTiming), "create prefill copy event");
        // pin_sources queued the cache-padding memset on the compute stream.
        check(cudaEventRecord(raw_reusable, stream), "record initial cache padding readiness");
    }

    void refresh_gpu_cache_views() {
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto& e = model.impl_->experts[layer];
            int64_t lo = 0, hi = 0;
            cache.layer_slot_range(layer, lo, hi);
            gpu_view[layer] = {gpu_state + layer, hi > lo ? cache.device_slot((int)lo) : nullptr,
                               grouped_miss, e.stride, grouped_stride, (int32_t)(hi - lo)};
        }
    }

    void initialize_gpu_cache() {
        host_state.resize(kLayers);
        gpu_state = reinterpret_cast<decode_cache::LayerState*>(memory.allocate(kLayers * sizeof(decode_cache::LayerState)));
        gpu_plans = reinterpret_cast<decode_cache::Plan*>(memory.allocate(kLayers * sizeof(decode_cache::Plan)));
        gpu_completion = reinterpret_cast<Completion*>(memory.allocate(sizeof(Completion)));
        check(cudaHostAlloc(&host_completion, sizeof(Completion), cudaHostAllocPortable), "allocate cache receipts");
        check(cudaMemsetAsync(gpu_completion, 0, sizeof(Completion), stream), "initialize cache receipts");
        check(cudaMemsetAsync(grouped_float, 0, kernels::kDecodeExpertScratchFloats * sizeof(float), stream),
              "initialize invalid-plan scratch");
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto& e = model.impl_->experts[layer];
            auto& mirror = gpu_mirror[layer];
            mirror.stride = e.stride;
            for (int role = 0; role < 3; ++role) {
                void* alias = nullptr;
                check(cudaHostGetDevicePointer(&alias, pinned_sources[layer][role], 0), "map cache source arena");
                mirror.role[role] = static_cast<const uint8_t*>(alias);
                mirror.bytes[role] = e.roles[role].bytes;
                mirror.offset[role] = e.roles[role].offset;
            }
        }
        refresh_gpu_cache_views();
        gpu_cache_enabled = true;
    }

    void begin_gpu() {
        if (!gpu_cache_enabled || gpu_active) throw std::runtime_error("Kolibri: invalid device cache begin");
        if (!gpu_cache_current) {
            for (int layer = 0; layer < kLayers; ++layer) {
                auto& state = host_state[layer];
                int64_t lo = 0, hi = 0;
                cache.layer_slot_range(layer, lo, hi);
                state.next_free = 0;
                for (int expert = 0; expert < kExperts; ++expert) {
                    const int slot = cache.valid() ? cache.slot_of(layer, expert) : -1;
                    state.slot[expert] = slot < 0 ? -1 : slot - (int)lo;
                    state.next_free += slot >= 0;
                    state.age[expert] = decode_recency[(size_t)layer * kExperts + expert];
                }
            }
            host_completion->counters = {decode_tick, hits, misses};
            check(cudaMemcpyAsync(gpu_state, host_state.data(), kLayers * sizeof(decode_cache::LayerState), cudaMemcpyHostToDevice, stream),
                  "upload current host cache state");
            check(cudaMemcpyAsync(&gpu_completion->counters, &host_completion->counters,
                                  sizeof(decode_cache::Counters), cudaMemcpyHostToDevice, stream), "upload cache counters");
            gpu_cache_current = true;
        }
        gpu_pending_layers = 0;
        gpu_active = true;
    }

    void complete_gpu() {
        if (!gpu_active) return;
        // GU/Down may throw after prepare but before the compute-stream wait.
        // Such plans still mutated the cache and must finish before receipt D2H.
        if (gpu_aux_unfenced) {
            check(cudaStreamSynchronize(prefill_upload), "drain pending decode cache preparation");
            gpu_aux_unfenced = false;
        }
        gpu_prepared_layer = -1;
        check(cudaMemcpyAsync(host_completion, gpu_completion, sizeof(Completion), cudaMemcpyDeviceToHost, stream),
              "download cache receipts");
        check(cudaStreamSynchronize(stream), "complete device cache token");
        const int layers = gpu_pending_layers;
        gpu_active = false;
        gpu_pending_layers = 0;
        int invalid_layer = -1;
        for (int layer = 0; layer < layers; ++layer) {
            const auto& receipt = host_completion->receipt[layer];
            if (receipt.status != decode_cache::Ok) {
                if (invalid_layer < 0) invalid_layer = layer;
                continue;
            }
            for (int id : receipt.ids)
                decode_recency[(size_t)layer * kExperts + id] = receipt.counters.tick;
            int64_t lo = 0, hi = 0;
            cache.layer_slot_range(layer, lo, hi);
            for (int i = 0; i < receipt.mutation_count; ++i) {
                const auto m = receipt.mutation[i];
                if (m.kind == decode_cache::Admit) {
                    if (cache.admit(layer, m.new_expert) != lo + m.slot)
                        throw std::runtime_error("Kolibri: device admission disagrees with host cache");
                } else {
                    if (m.kind != decode_cache::Replace || cache.slot_of(layer, m.old_expert) != lo + m.slot ||
                        cache.slot_of(layer, m.new_expert) >= 0)
                        throw std::runtime_error("Kolibri: device replacement disagrees with host cache");
                    cache.replace(layer, m.old_expert, m.new_expert);
                    decode_recency[(size_t)layer * kExperts + m.old_expert] = receipt.counters.tick;
                }
            }
        }
        decode_tick = host_completion->counters.tick;
        hits = host_completion->counters.hits;
        misses = host_completion->counters.misses;
        // GPU staging can change the shared miss arena's layout without the
        // CPU upload helper seeing it. Its next use must clear layout padding.
        miss_layout[1] = {};
        if (invalid_layer >= 0)
            throw std::runtime_error("Kolibri: invalid device router IDs at layer " + std::to_string(invalid_layer));
    }

    void cpu_cache_use() {
        if (gpu_cache_enabled) {
            complete_gpu();
            gpu_cache_current = false;
        }
    }

    void pin_sources() {
#if !defined(__linux__)
        throw std::runtime_error("Kolibri: --pin-experts currently requires Linux");
#else
        size_t total = 0;
        for (const auto& layer : model.impl_->experts)
            for (const auto& role : layer.roles) total += kExperts * role.bytes;
        const HostMemory before;
        // Experimental startup limits; no system settings or automatic retries.
        constexpr size_t reserve = size_t(8) << 30;
        constexpr size_t system_swap_limit = size_t(256) << 20, process_swap_limit = size_t(64) << 20;
        if (before.available < total + reserve)
            throw std::runtime_error("Kolibri: pinning needs the expert payload plus 8 GiB available RAM");
        const long system_page = ::sysconf(_SC_PAGESIZE);
        if (system_page <= 0) throw std::runtime_error("Kolibri: cannot query page size");
        const size_t page = (size_t) system_page;
        size_t minimum_available = before.available;
        auto guard = [&] {
            const HostMemory now;
            minimum_available = std::min(minimum_available, now.available);
            const size_t growth = before.swap_free > now.swap_free ? before.swap_free - now.swap_free : 0;
            if (now.available < reserve || growth > system_swap_limit || now.process_swap > process_swap_limit)
                throw std::runtime_error("Kolibri: pinning stopped at the 8 GiB available RAM, 256 MiB new system swap, "
                                         "or 64 MiB process swap startup limit");
            return now.available;
        };
        const auto started = std::chrono::steady_clock::now();
        const char* names[] = {"gate", "up", "down"};
        for (int l = 0; l < kLayers; ++l) for (int r = 0; r < 3; ++r) {
            const auto& role = model.impl_->experts[l].roles[r];
            const size_t bytes = kExperts * role.bytes;
            if (guard() < bytes + reserve)
                throw std::runtime_error("Kolibri: pinned role allocation would consume the 8 GiB RAM reserve");
            const std::string name = "blk." + std::to_string(l) + ".ffn_" + names[r] + "_exps.weight";
            size_t shard_index = 0;
            const auto* tensor = model.impl_->file->find(name, &shard_index);
            if (!tensor) throw std::runtime_error("Kolibri: missing expert source tensor");
            const auto& shard = model.impl_->file->shard(shard_index);
            const uint64_t file_offset = shard.data_start() + tensor->offset;
            ReadOnlyFile fd(shard.path());
            auto release = [&](size_t offset, size_t amount) {
                const uintptr_t pointer = reinterpret_cast<uintptr_t>(role.source + offset);
                const uintptr_t begin = aligned(pointer, page), end = (pointer + amount) / page * page;
                if (end <= begin) return;
                const size_t count = end - begin;
                if (::madvise(reinterpret_cast<void*>(begin), count, MADV_DONTNEED) != 0)
                    throw std::runtime_error(std::string("Kolibri: expert source madvise: ") + std::strerror(errno));
                const int error = ::posix_fadvise(fd.fd, (off_t) (file_offset + offset + begin - pointer),
                                                (off_t) count, POSIX_FADV_DONTNEED);
                if (error) throw std::runtime_error(std::string("Kolibri: expert source fadvise: ") + std::strerror(error));
            };
            auto& arena = pinned_arenas[l][r];
            arena = std::make_unique<strata::core::PinnedArena>(bytes, std::vector<uint64_t>{0, bytes},
                                                              strata::core::PinnedArena::Deferred{});
            if (!arena->valid() || arena->capacity != bytes ||
                arena->backing != strata::core::PageBacking::NormalPages ||
                arena->mapping_base != arena->base || arena->mapping_bytes != bytes)
                throw std::runtime_error("Kolibri: dedicated anonymous expert arena required: " + arena->note);
            // Avoid THP allocation failures triggering background reclaim during registration.
            if (::madvise(arena->base, arena->capacity, MADV_NOHUGEPAGE) != 0)
                throw std::runtime_error(std::string("Kolibri: expert arena MADV_NOHUGEPAGE: ") + std::strerror(errno));
            std::atomic<int> ready{0};
            arena->register_slices(ready);
            if (arena->registered_bytes != arena->capacity || arena->registered_slices != 1)
                throw std::runtime_error("Kolibri: one fully registered expert arena slice required: " + arena->note);
            pinned_sources[l][r] = arena->data();
            guard();
            constexpr size_t tile = size_t(64) << 20;
            for (size_t offset = 0; offset < bytes; offset += tile) {
                const size_t amount = std::min(tile, bytes - offset);
                guard();
                std::memcpy(pinned_sources[l][r] + offset, role.source + offset, amount);
                release(offset, amount);
                guard();
            }
            // A final pass releases full pages shared by neighbouring copy tiles.
            release(0, bytes);
            guard();
            void* device_alias = nullptr;
            check(cudaHostGetDevicePointer(&device_alias, arena->base, 0), "query pinned expert device alias");
        }
        if (cache.valid())
            check(cudaMemsetAsync(cache.device_slot(0), 0, cache.bytes(), stream), "clear pinned cache padding");
        pinned = true;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::fprintf(stderr, "Kolibri: %.2f GiB pinned expert RAM prepared in %.2f s; minimum available RAM %.2f GiB\n",
                     (double) total / (1ull << 30), seconds, (double) minimum_available / (1ull << 30));
#endif
    }

    void upload_pinned(int layer, int expert, uint8_t* weights, int miss_kind, cudaStream_t upload_stream = nullptr) {
        const auto& e = model.impl_->experts[layer];
        const cudaStream_t copy_stream = upload_stream ? upload_stream : stream;
        if (miss_kind >= 0) {
            const std::array<size_t, 3> layout{e.roles[0].bytes, e.roles[1].bytes, e.roles[2].bytes};
            if (miss_layout[miss_kind] != layout) {
                // A larger preceding role can overlap a new layout's padding.
                // Prefill and decode share this group arena and its tracker.
                uint8_t* arena = miss_kind == 0 ? miss : (prefill_miss ? prefill_miss : grouped_miss);
                check(cudaMemsetAsync(arena, 0, miss_kind == 0 ? miss_bytes : group_miss_bytes, copy_stream),
                      "clear reused pinned expert padding");
                miss_layout[miss_kind] = layout;
            }
        }
        for (int r = 0; r < 3; ++r) {
            const auto& role = e.roles[r];
            check(cudaMemcpyAsync(weights + role.offset, pinned_sources[layer][r] + (size_t) expert * role.bytes,
                                  role.bytes, cudaMemcpyHostToDevice, copy_stream), "upload pinned expert role");
        }
    }

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

Executor::Executor(Model& model, cudaStream_t stream, int max_tokens, double expert_vram_gib, int expert_slots,
                   bool grouped_decode, bool decode_cache_lru, bool fused_decode, bool grouped_prefill, bool pin_experts,
                   bool overlap_prefill_uploads, bool gpu_expert_cache, bool overlap_decode_cache) {
    if (!model.impl_) throw std::runtime_error("Kolibri: load a model before creating the executor");
    if (!stream) throw std::runtime_error("Kolibri: the executor requires an explicit CUDA stream");
    if (max_tokens < 1 || max_tokens > kernels::kChunk || expert_slots < -1 || expert_slots > kExperts ||
        !std::isfinite(expert_vram_gib) || (expert_vram_gib < 0 && expert_vram_gib != -1))
        throw std::runtime_error("Kolibri: invalid executor/cache settings");
    if (decode_cache_lru && !grouped_decode)
        throw std::runtime_error("Kolibri: decode cache LRU requires grouped decode");
    if (overlap_prefill_uploads && (!pin_experts || !grouped_prefill))
        throw std::runtime_error("Kolibri: prefill upload overlap requires pinned experts and grouped prefill");
    if (gpu_expert_cache && !(pin_experts && grouped_decode && decode_cache_lru && fused_decode))
        throw std::runtime_error("Kolibri: device cache requires pin/grouped/LRU/fused decode");
    if (overlap_decode_cache && !(gpu_expert_cache && overlap_prefill_uploads))
        throw std::runtime_error("Kolibri: decode cache overlap requires device cache and prefill upload stream");
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
    p->miss_bytes = max_blob;
    p->pageable.resize(max_blob);
    if (grouped_prefill) {
        size_t max_gate = 0, max_up = 0, max_down = 0;
        int layers = 0;
        for (int l = 0; l < kLayers; ++l) {
            const auto& e = model.impl_->experts[l];
            const auto& g = e.roles[0]; const auto& u = e.roles[1]; const auto& d = e.roles[2];
            if (!MMQ::built() || !MMQ::fits(g.type, g.out) ||
                !MMQ::fits(u.type, u.out) || !MMQ::fits(d.type, d.out)) continue;
            p->prefill_layers[l] = true; ++layers;
            max_gate = std::max(max_gate, g.bytes); max_up = std::max(max_up, u.bytes);
            max_down = std::max(max_down, d.bytes);
        }
        if (layers) {
            p->prefill_stride = max_blob;
            p->prefill_miss = p->memory.allocate(MMQ::kGatherGroupMax * max_blob);
            p->group_miss_bytes = MMQ::kGatherGroupMax * max_blob;
            p->prefill_up_offset = MMQ::kGatherGroupMax * max_gate + kTail;
            p->prefill_gu = p->memory.allocate(MMQ::kGatherGroupMax * (max_gate + max_up) + 2 * kTail);
            p->prefill_down = p->memory.allocate(MMQ::kGatherGroupMax * max_down + kTail);
            p->prefill_bounds = (int32_t*) p->memory.allocate((MMQ::kGatherGroupMax + 1) * sizeof(int32_t));
            const size_t bytes = MMQ::kGatherGroupMax * (max_blob + max_gate + max_up + max_down) +
                                 3 * kTail + (MMQ::kGatherGroupMax + 1) * sizeof(int32_t);
            std::fprintf(stderr, "Kolibri grouped prefill: %d layers, %.6f MiB workspace; staging shared with grouped decode\n",
                         layers, (double) bytes / (1 << 20));
        }
    }
    if (grouped_decode) {
        p->grouped_stride = max_blob;
        p->grouped_miss = p->prefill_miss ? p->prefill_miss :
            p->memory.allocate(kernels::kDecodeExpertCount * max_blob);
        if (!p->prefill_miss) p->group_miss_bytes = kernels::kDecodeExpertCount * max_blob;
        p->grouped_q8 = p->memory.allocate(kernels::kDecodeExpertQ8Bytes);
        p->grouped_float = (float*) p->memory.allocate(kernels::kDecodeExpertScratchFloats * sizeof(float));
    }
    if (decode_cache_lru) p->decode_recency.resize(kLayers * kExperts);
    if (fused_decode) p->dense_q8 = p->memory.allocate(K::native_q8_1_bytes(kHidden));
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
    if (pin_experts) p->pin_sources();
    if (overlap_prefill_uploads) p->start_prefill_uploads();
    if (gpu_expert_cache) p->initialize_gpu_cache();
    if (overlap_decode_cache) {
        check(cudaEventCreateWithFlags(&p->decode_route_ready, cudaEventDisableTiming), "create decode route event");
        check(cudaEventCreateWithFlags(&p->decode_cache_ready, cudaEventDisableTiming), "create decode cache event");
        p->overlap_decode_cache = true;
    }
    std::fprintf(stderr, "Kolibri: %d expert slots per layer, %.2f GiB VRAM; other experts streamed from %s\n",
                 slots, (double) p->cache.bytes() / 1073741824.0, pin_experts ? "progressive pinned RAM" : "GGUF");
    impl_ = std::move(p);
}

Executor::~Executor() = default;

void Executor::multiply(const Matrix& matrix, const float* x, float* out, int tokens) {
    impl_->multiply(matrix, x, out, tokens);
}

const void* Executor::multiply_decode(const Matrix* matrices, float* const* outputs, int count, const float* input) {
    auto& p = *impl_;
    if (!p.dense_q8 || count < 2 || count > 3)
        throw std::runtime_error("Kolibri: fused decode requires two or three projections and opt-in workspace");
    kernels::DenseDecodeProjection projection{};
    projection.count = count;
    for (int i = 0; i < count; ++i) {
        const auto& matrix = matrices[i];
        if (matrix.type != 8 || matrix.in != kHidden || matrix.out <= 0 || !matrix.device || !outputs[i])
            throw std::runtime_error("Kolibri: fused decode requires Q8_0 matrices with 2560 input features");
        projection.weights[i] = matrix.device;
        projection.outputs[i] = outputs[i];
        projection.rows[i] = matrix.out;
    }
    K::native_quantize_q8_1(input, p.dense_q8, kHidden, 1, p.stream);
    kernels::dense_decode_q8(projection, p.dense_q8, p.stream);
    return p.dense_q8;
}

void Executor::shared_down_decode(const Matrix& matrix, const float* gate, const float* up,
                                  float* hidden, float* out) {
    auto& p = *impl_;
    if (!p.dense_q8 || matrix.type != 8 || matrix.in != kFf || matrix.out != kHidden || !matrix.device)
        throw std::runtime_error("Kolibri: fused shared down requires opt-in Q8_0 512-to-2560 projection");
    // Ordinary projection scratch is separate from the live shared/router input.
    kernels::swiglu_quantize_q8_1(gate, up, hidden, p.q8, kFf, p.stream);
    K::native_mmvq(matrix.type, matrix.device, p.q8, out, kFf, kHidden, 1, p.stream);
    check(cudaPeekAtLastError(), "shared down multiplication");
}

void* Executor::attention_decode_q8() {
    if (!impl_->dense_q8)
        throw std::runtime_error("Kolibri: attention Q8 scratch requires fused decode");
    return impl_->q8;
}

void Executor::attention_output_decode_q8(const Matrix& matrix, float* out) {
    auto& p = *impl_;
    if (!p.dense_q8 || matrix.type != 8 || matrix.in != 6144 || matrix.out != kHidden || !matrix.device)
        throw std::runtime_error("Kolibri: prequantized attention output requires fused Q8_0 6144-to-2560 projection");
    K::native_mmvq(matrix.type, matrix.device, p.q8, out, matrix.in, matrix.out, 1, p.stream);
    check(cudaPeekAtLastError(), "prequantized attention output multiplication");
}

void Executor::expert(int layer, int expert, const float* x, float* out, int tokens) {
    auto& p = *impl_;
    p.cpu_cache_use();
    if (p.prefill_upload) p.prefill_sequence = false;
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
        if (p.pinned) {
            p.upload_pinned(layer, expert, weights, slot >= 0 ? -1 : 0);
        } else {
            uint8_t* stage = static_cast<uint8_t*>(p.uploads.acquire(e.stride));
            uint8_t* host = stage ? stage : p.pageable.data();
            stage_expert(e, expert, host);
            if (stage) check(p.uploads.submit(weights, stage, e.stride, p.stream), "upload expert");
            else check(strata::core::PinnedRing::direct(weights, host, e.stride, p.stream), "upload expert");
        }
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

bool Executor::supports_grouped_prefill(int layer) const {
    return layer >= 0 && layer < kLayers && impl_->prefill_layers[layer];
}

void Executor::experts_prefill(int layer, const int32_t* experts, int count, const int32_t* bounds,
                              const int32_t* row_ids, const float* x, float* out, int rows) {
    auto& p = *impl_;
    p.cpu_cache_use();
    p.check_tokens(rows);
    if (!supports_grouped_prefill(layer) || count < 1 || count > MMQ::kGatherGroupMax ||
        !experts || !bounds || !row_ids || bounds[0] != 0 || bounds[count] != rows)
        throw std::runtime_error("Kolibri: invalid grouped prefill layer or shape");
    int max_rows = 0;
    for (int i = 0; i < count; ++i) {
        const int n = bounds[i + 1] - bounds[i];
        if (experts[i] < 0 || experts[i] >= kExperts || (i && experts[i] <= experts[i - 1]) || n <= 0)
            throw std::runtime_error("Kolibri: grouped prefill needs ascending experts with positive rows");
        max_rows = std::max(max_rows, n);
    }
    const auto& e = p.model.impl_->experts[layer];
    MMQ::GatherGroup group;
    group.n = count;
    if (p.prefill_upload) {
        // A scalar/decode call may have reused the shared raw arena since the
        // preceding gather. Fence its queued consumers before resuming overlap.
        if (!p.prefill_sequence)
            check(cudaEventRecord(p.raw_reusable, p.stream), "fence compute use of raw expert storage");
        check(cudaStreamWaitEvent(p.prefill_upload, p.raw_reusable, 0), "wait to reuse raw expert storage");
    }
    // Admission order is unchanged; gathered packed weights outlive raw reuse.
    for (int i = 0; i < count; ++i) {
        int slot = p.cache.valid() ? p.cache.slot_of(layer, experts[i]) : -1;
        uint8_t* weights = nullptr;
        if (slot >= 0) { ++p.hits; weights = p.cache.device_slot(slot); }
        else {
            ++p.misses;
            slot = p.cache.valid() ? p.cache.admit(layer, experts[i]) : -1;
            weights = slot >= 0 ? p.cache.device_slot(slot) : p.prefill_miss + i * p.prefill_stride;
            if (p.pinned) {
                p.upload_pinned(layer, experts[i], weights, slot >= 0 ? -1 : 1, p.prefill_upload);
            } else {
                uint8_t* stage = static_cast<uint8_t*>(p.uploads.acquire(e.stride));
                uint8_t* host = stage ? stage : p.pageable.data();
                stage_expert(e, experts[i], host);
                if (stage) check(p.uploads.submit(weights, stage, e.stride, p.stream), "upload prefill expert");
                else check(strata::core::PinnedRing::direct(weights, host, e.stride, p.stream), "upload prefill expert");
            }
        }
        group.blob[i] = weights;
    }
    if (p.prefill_upload) {
        check(cudaEventRecord(p.copy_ready, p.prefill_upload), "record prefill upload readiness");
        check(cudaStreamWaitEvent(p.stream, p.copy_ready, 0), "wait for prefill expert uploads");
    }
    const auto& g = e.roles[0]; const auto& u = e.roles[1]; const auto& d = e.roles[2];
    const bool homogeneous = g.type == u.type && g.bytes == u.bytes;
    if (homogeneous) {
        if (!MMQ::gather_native_group(group, u.offset, g.bytes, d.offset, d.bytes,
                                 p.prefill_gu, 2 * g.bytes, p.prefill_down, d.bytes, p.stream,
                                 0, nullptr, kTail))
            throw std::runtime_error("Kolibri: grouped prefill weight alignment is invalid");
    } else {
        const size_t span = std::max({g.bytes, u.bytes, d.bytes}) + kTail;
        const unsigned blocks = (unsigned) ((span / sizeof(uint4) + 255) / 256);
        gather_mixed_roles<<<dim3(blocks, count, 3), 256, 0, p.stream>>>(
            group, g.bytes, u.bytes, d.bytes, u.offset, d.offset,
            p.prefill_gu, p.prefill_gu + p.prefill_up_offset, p.prefill_down);
        check(cudaPeekAtLastError(), "gather mixed prefill roles");
    }
    if (p.prefill_upload) {
        check(cudaEventRecord(p.raw_reusable, p.stream), "release raw experts after gather");
        p.prefill_sequence = true;
    }
    check(p.uploads.h2d(p.prefill_bounds, bounds, (count + 1) * sizeof(int32_t), p.stream), "upload prefill bounds");
    MMQ::quantize(x, row_ids, p.mq8, g.type, kHidden, kHidden, rows, p.stream);
    MMQ::Product product;
    product.w = p.prefill_gu; product.type = g.type; product.w_rows = kFf; product.w_cols = kHidden;
    product.expert_bytes = 2 * g.bytes; product.n = count; product.xq = p.mq8;
    product.bounds = p.prefill_bounds; product.ids = p.ids; product.total_rows = rows; product.max_rows = max_rows;
    product.dst = p.gate; product.ld_dst = kFf;
    if (!homogeneous) product.expert_bytes = g.bytes;
    p.mmq.run(product, p.stream);
    product.w = p.prefill_gu + g.bytes; product.dst = p.up;
    if (!homogeneous) {
        // For the three supported types, Q4_K needs DS4; Q6_K and Q8_0 use D4.
        if ((g.type == 12) != (u.type == 12))
            MMQ::quantize(x, row_ids, p.mq8, u.type, kHidden, kHidden, rows, p.stream);
        product.w = p.prefill_gu + p.prefill_up_offset;
        product.type = u.type; product.expert_bytes = u.bytes;
    }
    p.mmq.run(product, p.stream);
    const int n = rows * kFf;
    swiglu<<<(n + 255) / 256, 256, 0, p.stream>>>(p.gate, p.up, p.hidden, n);
    check(cudaPeekAtLastError(), "grouped prefill SwiGLU");
    MMQ::quantize(p.hidden, nullptr, p.mq8, d.type, kFf, kFf, rows, p.stream);
    product.w = p.prefill_down; product.type = d.type; product.w_rows = kHidden; product.w_cols = kFf;
    product.expert_bytes = d.bytes; product.dst = out; product.ld_dst = kHidden;
    p.mmq.run(product, p.stream);
}

void Executor::experts_decode(int layer, const int32_t* experts, const float* x, float* out, const void* input_q8) {
    auto& p = *impl_;
    p.cpu_cache_use();
    if (p.prefill_upload) p.prefill_sequence = false;
    if (!p.grouped_miss || layer < 0 || layer >= kLayers)
        throw std::runtime_error("Kolibri: invalid grouped decode executor/layer");
    for (int i = 0; i < kernels::kDecodeExpertCount; ++i)
        if (experts[i] < 0 || experts[i] >= kExperts || (i && experts[i] <= experts[i - 1]))
            throw std::runtime_error("Kolibri: grouped expert IDs must be distinct and ascending");
    const auto& e = p.model.impl_->experts[layer];
    kernels::DecodeExpertWeights group{};
    group.gate_offset = e.roles[0].offset;
    group.up_offset = e.roles[1].offset;
    group.down_offset = e.roles[2].offset;
    group.gate_type = e.roles[0].type;
    group.up_type = e.roles[1].type;
    group.down_type = e.roles[2].type;
    const bool lru = !p.decode_recency.empty();
    if (lru) {
        ++p.decode_tick;
        // Protect every selected expert before considering a replacement.
        for (int i = 0; i < kernels::kDecodeExpertCount; ++i)
            p.decode_recency[(size_t) layer * kExperts + experts[i]] = p.decode_tick;
    }
    std::array<std::pair<int, int>, kernels::kDecodeExpertCount> replacements{};
    int replacement_count = 0;
    for (int i = 0; i < kernels::kDecodeExpertCount; ++i) {
        const int expert = experts[i];
        int slot = p.cache.valid() ? p.cache.slot_of(layer, expert) : -1;
        uint8_t* weights = nullptr;
        if (slot >= 0) {
            ++p.hits;
            weights = p.cache.device_slot(slot);
        } else {
            ++p.misses;
            slot = p.cache.valid() ? p.cache.admit(layer, expert) : -1;
            if (slot < 0 && lru && p.cache.valid()) {
                int victim = -1;
                uint64_t oldest = p.decode_tick;
                for (int candidate = 0; candidate < kExperts; ++candidate) {
                    const uint64_t age = p.decode_recency[(size_t) layer * kExperts + candidate];
                    if (age < oldest && p.cache.slot_of(layer, candidate) >= 0) {
                        oldest = age;
                        victim = candidate;
                    }
                }
                if (victim >= 0) {
                    slot = p.cache.slot_of(layer, victim);
                    // Reserve this slot until the queued copy is published.
                    p.decode_recency[(size_t) layer * kExperts + victim] = p.decode_tick;
                    replacements[replacement_count++] = {victim, expert};
                }
            }
            // Selected slots and each uncached miss remain live until all six
            // consumers finish on this stream.
            weights = slot >= 0 ? p.cache.device_slot(slot) : p.grouped_miss + (size_t) i * p.grouped_stride;
            if (p.pinned) {
                p.upload_pinned(layer, expert, weights, slot >= 0 ? -1 : 1);
            } else {
                uint8_t* stage = static_cast<uint8_t*>(p.uploads.acquire(e.stride));
                uint8_t* host = stage ? stage : p.pageable.data();
                stage_expert(e, expert, host);
                if (stage) check(p.uploads.submit(weights, stage, e.stride, p.stream), "upload grouped expert");
                else check(strata::core::PinnedRing::direct(weights, host, e.stride, p.stream), "upload grouped expert");
            }
        }
        group.expert[i] = weights;
    }
    if (replacement_count) {
        // ExpertCache::replace publishes completed copies. Wait once for the
        // batch, not once per expert; no selected slot is a replacement victim.
        check(cudaStreamSynchronize(p.stream), "complete decode cache replacements");
        for (int i = 0; i < replacement_count; ++i)
            p.cache.replace(layer, replacements[i].first, replacements[i].second);
    }
    kernels::decode_experts(group, x, out, p.grouped_q8, p.grouped_float, p.stream, input_q8);
}

void Executor::begin_gpu_decode() { impl_->begin_gpu(); }
void Executor::complete_gpu_decode() { impl_->complete_gpu(); }

void Executor::gpu_decode_graph_submitted() {
    auto& p = *impl_;
    if (!p.gpu_active || p.gpu_aux_unfenced || p.gpu_prepared_layer >= 0 ||
        (p.gpu_pending_layers != 0 && p.gpu_pending_layers != kLayers))
        throw std::runtime_error("Kolibri: invalid graph receipt bookkeeping");
    p.gpu_pending_layers = kLayers;
    p.prefill_sequence = false;
}

void Executor::cancel_gpu_decode_capture() {
    auto& p = *impl_;
    p.gpu_active = false;
    p.gpu_pending_layers = 0;
    p.gpu_aux_unfenced = false;
    p.gpu_prepared_layer = -1;
    p.gpu_cache_current = false;
    p.prefill_sequence = false;
}

void Executor::prepare_experts_decode_gpu(int layer, const float* logits, const float* bias,
                                          int32_t* route_ids, float* route_weights) {
    auto& p = *impl_;
    if (!p.overlap_decode_cache || !p.gpu_active || p.gpu_aux_unfenced || p.gpu_prepared_layer >= 0 ||
        layer != p.gpu_pending_layers || layer < 0 || layer >= kLayers || !route_ids || !route_weights || !logits || !bias)
        throw std::runtime_error("Kolibri: invalid overlapped cache preparation order");
    // This event follows the router GEMM and includes all prior expert
    // consumers, prefill gathers, and the initial cache-state H2D upload.
    check(cudaEventRecord(p.decode_route_ready, p.stream), "record decode routes ready");
    p.gpu_aux_unfenced = true;
    p.prefill_sequence = false;
    check(cudaStreamWaitEvent(p.prefill_upload, p.decode_route_ready, 0), "wait for decode routes/raw storage");
    auto* plan = p.gpu_plans + layer;
    decode_cache::route_plan(logits, bias, route_ids, route_weights, p.gpu_view[layer],
                             &p.gpu_completion->counters, plan, p.prefill_upload, &p.gpu_completion->receipt[layer]);
    check(cudaPeekAtLastError(), "route and plan overlapped device expert cache");
    ++p.gpu_pending_layers;
    decode_cache::copy(p.gpu_mirror[layer], plan, p.prefill_upload);
    check(cudaPeekAtLastError(), "copy overlapped mapped expert cache misses");
    check(cudaEventRecord(p.decode_cache_ready, p.prefill_upload), "record decode cache ready");
    p.gpu_prepared_layer = layer;
}

void Executor::prepare_kv_growth(size_t additional_bytes, size_t remaining_bytes) {
    auto& p = *impl_;
    p.complete_gpu();
    check(cudaStreamSynchronize(p.stream), "drain compute before KV growth");
    if (p.prefill_upload)
        check(cudaStreamSynchronize(p.prefill_upload), "drain uploads before KV growth");
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total), "query VRAM before KV growth");
    constexpr size_t reserve = size_t(768) << 20;
    const size_t old_bytes = (size_t) p.cache.bytes();
    const size_t reclaimable = free + old_bytes;
    if (additional_bytes > remaining_bytes || reclaimable < reserve || remaining_bytes > reclaimable - reserve)
        throw std::runtime_error("Kolibri: configured KV capacity does not fit with an empty expert cache and 768 MiB reserve");
    const size_t budget = reclaimable - reserve - additional_bytes;
    if (old_bytes <= budget) return;
    size_t one_per_layer = 0;
    for (const auto& e : p.model.impl_->experts) one_per_layer += e.stride;
    const size_t slots = std::min<size_t>(kExperts, budget / one_per_layer);
    p.cache.close();
    if (slots) {
        std::vector<int64_t> sizes;
        sizes.reserve(kLayers * slots);
        for (const auto& e : p.model.impl_->experts)
            sizes.insert(sizes.end(), slots, (int64_t) e.stride);
        std::string error;
        if (!p.cache.open_sized(sizes, kLayers, kExperts, error))
            throw std::runtime_error("Kolibri: expert cache reallocation for KV growth failed: " + error);
        p.cache.set_per_layer_admission(true);
    }
    check(cudaMemGetInfo(&free, &total), "check VRAM after expert cache resize");
    if (free < reserve || additional_bytes > free - reserve)
        throw std::runtime_error("Kolibri: KV growth cannot preserve the 768 MiB reserve after cache resize");
    std::fill(p.decode_recency.begin(), p.decode_recency.end(), 0);
    p.miss_layout = {};
    p.gpu_cache_current = false;
    p.prefill_sequence = false;
    if (p.gpu_cache_enabled) p.refresh_gpu_cache_views();
    // open_sized zeroed the new cache; the next prefill renews raw_reusable.
    // Old device plans are replaced before use, and begin_gpu uploads empty state.
    std::fprintf(stderr, "Kolibri KV growth: expert cache emptied, %.2f -> %.2f GiB (%zu slots per layer)\n",
                 double(old_bytes) / (1ull << 30), double(p.cache.bytes()) / (1ull << 30), slots);
}

void Executor::experts_decode_gpu(int layer, const int32_t* route_ids, const float* input,
                                  float* outputs, const float* route_weights, float* destination,
                                  const void* input_q8) {
    auto& p = *impl_;
    if (p.overlap_decode_cache) {
        if (!p.gpu_active || layer != p.gpu_prepared_layer || layer != p.gpu_pending_layers - 1 ||
            layer < 0 || layer >= kLayers || !input_q8)
            throw std::runtime_error("Kolibri: overlapped cache consumer without matching preparation");
        check(cudaStreamWaitEvent(p.stream, p.decode_cache_ready, 0), "wait for prepared decode cache");
        // From here the compute stream orders receipt reads even if a later
        // launch throws; the exceptional upload-only drain is no longer needed.
        p.gpu_aux_unfenced = false;
        p.gpu_prepared_layer = -1;
        const auto& e = p.model.impl_->experts[layer];
        kernels::DecodeExpertWeights roles{};
        roles.gate_offset = e.roles[0].offset; roles.up_offset = e.roles[1].offset; roles.down_offset = e.roles[2].offset;
        roles.gate_type = e.roles[0].type; roles.up_type = e.roles[1].type; roles.down_type = e.roles[2].type;
        auto* plan = p.gpu_plans + layer;
        kernels::decode_experts_device(roles, plan, input, outputs, p.grouped_q8, p.grouped_float, p.stream, input_q8);
        kernels::combine_decode_experts_device(outputs, route_weights, plan, destination, p.stream);
        return;
    }
    if (!p.gpu_active || layer != p.gpu_pending_layers || layer >= kLayers || !input_q8)
        throw std::runtime_error("Kolibri: invalid device-cache layer/input ordering");
    // Device copies reuse the raw prefill arena on the compute stream. The
    // next CPU prefill sequence must fence those consumers before uploading.
    if (p.prefill_upload) p.prefill_sequence = false;
    const auto& e = p.model.impl_->experts[layer];
    kernels::DecodeExpertWeights roles{};
    roles.gate_offset = e.roles[0].offset; roles.up_offset = e.roles[1].offset; roles.down_offset = e.roles[2].offset;
    roles.gate_type = e.roles[0].type; roles.up_type = e.roles[1].type; roles.down_type = e.roles[2].type;
    auto* plan = p.gpu_plans + layer;
    decode_cache::plan(route_ids, p.gpu_view[layer], &p.gpu_completion->counters, plan, p.stream,
                      &p.gpu_completion->receipt[layer]);
    check(cudaPeekAtLastError(), "plan device expert cache");
    ++p.gpu_pending_layers;
    decode_cache::copy(p.gpu_mirror[layer], plan, p.stream);
    check(cudaPeekAtLastError(), "copy mapped expert cache misses");
    kernels::decode_experts_device(roles, plan, input, outputs, p.grouped_q8, p.grouped_float, p.stream, input_q8);
    kernels::combine_decode_experts_device(outputs, route_weights, plan, destination, p.stream);
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
