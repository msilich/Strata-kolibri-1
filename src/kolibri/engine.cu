// Optional Kolibri driver. The GGUF weights and existing model paths are unchanged.
#include "model.hpp"
#include "kernels.cuh"
#include "strata/core/pinned_ring.hpp"
#include "strata/core/logit_bias.hpp"
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

class Engine {
public:
    Stream stream;
    Blas blas{stream.value};
    Model model;
    const int context, chunk;
    int64_t pos = 0;
    std::unique_ptr<Executor> exec;
    std::ofstream layer_dump, router_dump, route_ids_dump;

    Engine(const std::string& path, int ctx, int ch, double cache_gib, int cache_slots)
        : context(ctx), chunk(ch) {
        model.load(path, stream.value);
        const size_t t = (size_t) chunk;
        x.alloc(t * H); norm.alloc(t * H); mixed.alloc(t * H); moe.alloc(t * H);
        q.alloc(t * Q); k.alloc(t * KV); v.alloc(t * KV); attn.alloc(t * Q);
        gate.alloc(t * FF); up.alloc(t * FF); hidden.alloc(t * FF);
        router.alloc(t * NE); route_ids.alloc(t * TOP); route_weights.alloc(t * TOP);
        group_ids.alloc(t); group_weights.alloc(t); gathered.alloc(t * H); expert_out.alloc(t * H);
        ids.alloc(t); logits.alloc(model.vocab);
        local_k.alloc(512 * KV); local_v.alloc(512 * KV);
        for (size_t l = 0; l < model.layers.size(); ++l) {
            const size_t cells = model.layers[l].local ? (size_t) (512 + chunk) : (size_t) context;
            caches[l].k.alloc(cells * KV); caches[l].v.alloc(cells * KV);
        }
        exec = std::make_unique<Executor>(model, stream.value, chunk, cache_gib, cache_slots);
        std::fprintf(stderr, "Kolibri-1: context %d, chunk %d, expert cache %d slots / %.2f GiB\n",
                     context, chunk, exec->cache_slots(), (double) exec->cache_bytes() / (1ull << 30));
    }

    void reset() { pos = 0; }

    // A chunk and decode use the same layer equations. Local history is compacted
    // only after every query has consumed the old history and the whole new chunk.
    void forward(const int32_t* tokens, int T) {
        if (T < 1 || T > chunk || pos + T > context) throw std::runtime_error("context or chunk limit exceeded");
        for (int i = 0; i < T; ++i)
            if (tokens[i] < 0 || tokens[i] >= model.vocab) throw std::runtime_error("token outside vocabulary");
        auto s = stream.value;
        ck(transfer.h2d(ids.p, tokens, (size_t) T * sizeof(int32_t), s));
        exec->embed(ids.p, x.p, T);
        std::vector<float> dumps;
        std::vector<float> router_dumps;
        std::vector<int32_t> route_ids_dumps;
        if (layer_dump.is_open()) dumps.resize((size_t) T * 50 * H);
        if (router_dump.is_open()) router_dumps.resize((size_t) T * 50 * NE);
        if (route_ids_dump.is_open()) route_ids_dumps.resize((size_t) T * 50 * TOP);
        for (int l = 0; l < 50; ++l) {
            const auto& L = model.layers[(size_t) l];
            K::rms_norm(x.p, L.attn_norm, norm.p, T, H, model.eps, s);
            exec->multiply(L.q, norm.p, q.p, T);
            exec->multiply(L.k, norm.p, k.p, T);
            exec->multiply(L.v, norm.p, v.p, T);
            K::qk_norm_rope(q.p, k.p, L.q_norm, L.k_norm, T, pos, L.local, model.rope_base, model.eps, s);
            const int prior = L.local ? (int) std::min<int64_t>(pos, 512) : (int) pos;
            auto& cache = caches[(size_t) l];
            K::append_kv(k.p, v.p, cache.k.p, cache.v.p, T, prior, s);
            K::attention(q.p, cache.k.p, cache.v.p, attn.p, T, prior + T,
                         L.local ? pos - prior : 0, pos, L.local, s);
            if (L.local && prior + T > 512) {
                const size_t bytes = 512 * KV * sizeof(uint16_t);
                const size_t offset = (size_t) (prior + T - 512) * KV;
                ck(cudaMemcpyAsync(local_k.p, cache.k.p + offset, bytes, cudaMemcpyDeviceToDevice, s));
                ck(cudaMemcpyAsync(local_v.p, cache.v.p + offset, bytes, cudaMemcpyDeviceToDevice, s));
                ck(cudaMemcpyAsync(cache.k.p, local_k.p, bytes, cudaMemcpyDeviceToDevice, s));
                ck(cudaMemcpyAsync(cache.v.p, local_v.p, bytes, cudaMemcpyDeviceToDevice, s));
            }
            exec->multiply(L.o, attn.p, mixed.p, T);
            K::rms_norm(mixed.p, L.attn_post_norm, norm.p, T, H, model.eps, s);
            K::add(x.p, norm.p, x.p, (size_t) T * H, s);
            K::rms_norm(x.p, L.ffn_norm, norm.p, T, H, model.eps, s);
            const float one = 1.f, zero = 0.f;
            bc(cublasSgemm(blas.value, CUBLAS_OP_T, CUBLAS_OP_N, NE, T, H, &one,
                           L.router, H, norm.p, H, &zero, router.p, NE));
            K::route(router.p, L.router_bias, route_ids.p, route_weights.p, T, s);
            exec->multiply(L.shared_gate, norm.p, gate.p, T);
            exec->multiply(L.shared_up, norm.p, up.p, T);
            K::swiglu(gate.p, up.p, hidden.p, T, FF, s);
            exec->multiply(L.shared_down, hidden.p, moe.p, T);
            std::vector<int32_t> selected((size_t) T * TOP);
            std::vector<float> weights((size_t) T * TOP);
            ck(transfer.d2h_sync(selected.data(), route_ids.p, selected.size() * sizeof(int32_t), s));
            ck(transfer.d2h_sync(weights.data(), route_weights.p, weights.size() * sizeof(float), s));
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
            std::array<std::vector<int32_t>, NE> rows;
            std::array<std::vector<float>, NE> scales;
            for (int t = 0; t < T; ++t) for (int j = 0; j < TOP; ++j) {
                const int i = t * TOP + j, e = selected[(size_t) i];
                if (e < 0 || e >= NE) throw std::runtime_error("non-finite or invalid router output");
                rows[(size_t) e].push_back(t); scales[(size_t) e].push_back(weights[(size_t) i]);
            }
            for (int e = 0; e < NE; ++e) {
                const int n = (int) rows[(size_t) e].size();
                if (!n) continue;
                ck(transfer.h2d(group_ids.p, rows[(size_t) e].data(), (size_t) n * sizeof(int32_t), s));
                ck(transfer.h2d(group_weights.p, scales[(size_t) e].data(), (size_t) n * sizeof(float), s));
                K::gather_rows(norm.p, group_ids.p, gathered.p, n, H, s);
                exec->expert(l, e, gathered.p, expert_out.p, n);
                K::scatter_add_weighted(expert_out.p, group_ids.p, group_weights.p, moe.p, n, H, s);
            }
            K::rms_norm(moe.p, L.ffn_post_norm, norm.p, T, H, model.eps, s);
            K::add(x.p, norm.p, x.p, (size_t) T * H, s);
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
        K::rms_norm(x.p + (size_t) (T - 1) * H, model.out_norm, norm.p, 1, H, model.eps, s);
        exec->multiply(model.head, norm.p, logits.p, 1);
        ck(cudaStreamSynchronize(s));
        pos += T;
    }
    void host_logits(std::vector<float>& dst) {
        dst.resize(model.vocab);
        ck(transfer.d2h_sync(dst.data(), logits.p, dst.size() * sizeof(float), stream.value));
        if (!std::all_of(dst.begin(), dst.end(), [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("model produced non-finite logits");
    }
private:
    struct Cache { Buffer<uint16_t> k, v; };
    std::array<Cache, 50> caches;
    Buffer<float> x, norm, mixed, moe, q, k, v, attn, gate, up, hidden, router;
    Buffer<float> route_weights, group_weights, gathered, expert_out, logits;
    Buffer<int32_t> route_ids, group_ids, ids;
    Buffer<uint16_t> local_k, local_v;
    strata::core::PinnedRing transfer;
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
            engine.host_logits(lg);
            const auto raw_logits = r.logprobs >= 0 ? lg : std::vector<float>{};
            const int token = sample(lg, r.sampling, history, rng, r.bias);
            std::printf("T %d\n", token);
            if (r.logprobs >= 0) print_logprobs(raw_logits, token, r.logprobs, r.allowed);
            std::fflush(stdout);
            ++produced; history.push_back(token);
            if (token == engine.model.eos) { finish = "stop"; break; }
            if (produced < r.max_new && !cancelled()) engine.forward(&history.back(), 1);
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
        int ctx = 32768, chunk = 256, n = 64, slots = -1;
        double vram = -1;
        bool server = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
                return argv[++i];
            };
            if (arg == "--version") { std::cout << "strata-kolibri " << STRATA_VERSION << '\n'; return 0; }
            else if (arg == "--families") { std::cout << "kolibri1\n"; return 0; }
            else if (arg == "--help") {
                std::cout << "strata-kolibri --native MODEL [--max-context 32768] [--chunk 256]\n"
                             "  --serve | --tokens ID,ID,... [--n 64] [--force ID,ID,...]\n"
                             "  --expert-vram GiB | --expert-slots N (per layer, 0 disables cache)\n"
                             "  --dump-layers FILE --dump-full-logits FILE (diagnostic binary F32)\n"
                             "  --dump-router FILE (F32 [position,50,384]) --dump-route-ids FILE (I32 [position,50,6])\n";
                return 0;
            } else if (arg == "--native" || arg == "--model") model = next();
            else if (arg == "--max-context") ctx = std::stoi(next());
            else if (arg == "--chunk") chunk = std::stoi(next());
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
            else throw std::runtime_error("unknown option: " + arg);
        }
        if (model.empty() || ctx < 1 || ctx > 32768 || chunk < 1 || chunk > 256 || n < 1 ||
            slots < -1 || slots > 384 || !std::isfinite(vram) || (vram < 0 && vram != -1) || (!server && tokens.empty()))
            throw std::runtime_error("invalid options; use --help");
        auto ids = server ? std::vector<int32_t>{} : parse_ids(tokens);
        auto force = forced.empty() ? std::vector<int32_t>{} : parse_ids(forced);
        if (!force.empty()) n = (int) force.size() + 1;
        if (!server && ids.size() + (size_t) n > (size_t) ctx) throw std::runtime_error("prompt + response exceeds context");
        if (!std::getenv("STRATA_MMQ_KQ")) setenv("STRATA_MMQ_KQ", "1", 0);
        if (!std::getenv("CUDA_MODULE_LOADING")) setenv("CUDA_MODULE_LOADING", "EAGER", 0);
        Engine engine(model, ctx, chunk, vram, slots);
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
            if (i + 1 < n && (token != engine.model.eos || !force.empty())) engine.forward(&token, 1);
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
