// CPU-only protocol fixture: run the real request loop without model weights.
#define main kolibri_model_main
#include "engine.cu"
#undef main

namespace {
struct ProtocolEngine {
    struct { int vocab = 32, eos = 31; } model;
    struct Counters {
        uint64_t hits() const { return 0; }
        uint64_t misses() const { return 0; }
    } counters;
    Counters* exec = &counters;
    int context = 256, chunk = 8;
    bool gpu_greedy = true;
    int64_t pos = 0;
    std::vector<int32_t> processed;

    void reset(bool reuse_prefix = false) {
        if (!reuse_prefix) { processed.clear(); pos = 0; }
    }
    void forward(const int32_t* tokens, int n, bool = false) {
        if (n < 1 || n > chunk || pos + n > context) throw std::runtime_error("fixture context exceeded");
        // Give STOP a processing boundary; tests accept either side of an
        // in-flight forward, rather than relying on a particular scheduler.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        processed.insert(processed.end(), tokens, tokens + n);
        pos += n;
    }
    int greedy_token() const {
        uint32_t value = 7;
        for (int token : processed) value = value * 33 + (uint32_t) token + 1;
        return 1 + value % 29;
    }
    void host_logits(std::vector<float>& logits) const {
        logits.assign(model.vocab, -10.f);
        logits[greedy_token()] = 10.f;
    }
};
}

int main(int argc, char** argv) {
    bool cache = false;
    ProtocolEngine engine;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--prefix-cache") cache = true;
        else if (option == "--cpu-sampling") engine.gpu_greedy = false;
        else return 1;
    }
    return serve(engine, cache);
}
