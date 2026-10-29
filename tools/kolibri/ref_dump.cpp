// Kolibri parity against llama.cpp 9df0b73c707900b7277a46b5ae52ff7e7e567d13.
// Reuse the existing generic layer/logit dumper with F16 KV and GPU execution
// of host-resident experts through llama.cpp's native MoE cache. Use --ngl 99.
// Build like tools/qwen35/q36_llama_dump.cpp, additionally linking -lggml-cpu.
// STRATA_DUMP_TENSORS=DIR writes compact per-layer router/FFN-input traces.
// STRATA_DUMP_UBATCH=1 retains every traced prompt position except skipped final-layer rows.
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cerrno>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {
ggml_backend_sched_eval_callback original_callback = nullptr;
std::string trace_directory;
std::map<std::string, std::ofstream> traces;

bool kolibri_trace(ggml_tensor* tensor, bool ask, void* user) {
    const std::string name = ggml_get_name(tensor);
    bool wanted = false;
    for (const char* prefix : {"ffn_norm-", "ffn_moe_logits-", "ffn_moe_logits_biased-",
                               "ffn_moe_topk-", "ffn_moe_weights-"})
        wanted |= name.compare(0, std::strlen(prefix), prefix) == 0;
    if (ask) return wanted || (original_callback && original_callback(tensor, ask, user));
    if (wanted) {
        const char* suffix = tensor->type == GGML_TYPE_F32 ? ".f32" : ".i32";
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32) {
            std::fprintf(stderr, "unexpected diagnostic tensor type: %s\n", name.c_str());
            std::exit(2);
        }
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        // Top-k is a strided view; write its logical elements without padding.
        std::vector<uint8_t> packed((size_t) ggml_nelements(tensor) * 4);
        for (size_t element = 0; element < packed.size() / 4; ++element) {
            size_t index = element, offset = 0;
            for (int d = 0; d < 4; ++d) {
                offset += (index % tensor->ne[d]) * tensor->nb[d];
                index /= tensor->ne[d];
            }
            std::memcpy(packed.data() + element * 4, bytes.data() + offset, 4);
        }
        auto& out = traces.try_emplace(name, trace_directory + "/" + name + suffix, std::ios::binary).first->second;
        out.write(reinterpret_cast<const char*>(packed.data()), (std::streamsize) packed.size());
        if (!out) {
            std::fprintf(stderr, "cannot write diagnostic tensor: %s\n", name.c_str());
            std::exit(2);
        }
    }
    return !original_callback || original_callback(tensor, ask, user);
}

llama_model_params kolibri_model_defaults() {
    auto params = llama_model_default_params();
    static const llama_model_tensor_buft_override overrides[] = {
        {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    params.tensor_buft_overrides = overrides;
    return params;
}

llama_context_params kolibri_context_defaults() {
    auto params = llama_context_default_params();
    size_t mib = 24576;
    if (const char* value = std::getenv("STRATA_DUMP_MOE_CACHE_MIB")) {
        char* end = nullptr;
        errno = 0;
        const auto parsed = std::strtoull(value, &end, 10);
        if (errno || value == end || *end || !parsed || parsed > std::numeric_limits<size_t>::max() / (1024 * 1024)) {
            std::fprintf(stderr, "invalid STRATA_DUMP_MOE_CACHE_MIB\n");
            std::exit(2);
        }
        mib = (size_t) parsed;
    }
    params.moe_cache_size = mib * 1024 * 1024;
    params.type_k = params.type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return params;
}

llama_context* kolibri_init_context(llama_model* model, llama_context_params params) {
    // The pinned reference's GPU expert cache handles at most 32 tokens per microbatch.
    int ubatch = 32;
    if (const char* value = std::getenv("STRATA_DUMP_UBATCH")) ubatch = std::atoi(value);
    if (ubatch < 1 || ubatch > 32) {
        std::fprintf(stderr, "STRATA_DUMP_UBATCH must be 1..32 for the GPU expert cache\n");
        std::exit(2);
    }
    params.n_ubatch = std::min(params.n_ubatch, (uint32_t) ubatch);
    if (const char* directory = std::getenv("STRATA_DUMP_TENSORS")) {
        trace_directory = directory;
        std::filesystem::create_directories(trace_directory);
        original_callback = params.cb_eval;
        params.cb_eval = kolibri_trace;
    }
    return llama_init_from_model(model, params);
}
}  // namespace

// Apply the two defaults only to the reused standalone diagnostic program.
#define llama_model_default_params kolibri_model_defaults
#define llama_context_default_params kolibri_context_defaults
#define llama_init_from_model kolibri_init_context
#include "../qwen35/q36_llama_dump.cpp"
#undef llama_init_from_model
#undef llama_context_default_params
#undef llama_model_default_params
