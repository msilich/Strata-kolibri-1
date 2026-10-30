// Independent numerical checks and resident timing for tensor decode attention.
#include "prefill_attention.cuh"
#include "tensor_decode_attention.cuh"
#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace k = strata::kolibri::kernels;
constexpr int Q = k::kQueryHeads * k::kHeadDim, KV = k::kKvHeads * k::kHeadDim;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

template<class T> struct Device {
    T* p = nullptr;
    size_t count;
    explicit Device(size_t n) : count(n) { check(cudaMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& v) : Device(v.size()) {
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    std::vector<T> download() {
        std::vector<T> out(count);
        check(cudaMemcpy(out.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
};

uint16_t half_bits(float x) {
    const __half h = __float2half_rn(x);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}
float half_value(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}
std::vector<float> random_values(size_t count, unsigned seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    std::vector<float> values(count);
    for (auto& value : values) value = distribution(random);
    return values;
}

struct Case {
    const char* name;
    int rows, key_count;
    int64_t key_first, query_first;
    bool local;
    int variant = 0;  // 1: extreme scores; 2: Q low bits; 3: P low bits; 4: zero Q.
};

// Independent scalar F64 attention over the exact cached FP16 values.
std::vector<double> reference(const Case& c, const std::vector<float>& q,
                              const std::vector<uint16_t>& kb, const std::vector<uint16_t>& vb,
                              int row) {
    const int64_t position = c.query_first + row;
    const int begin = c.local ? (int) (std::max(int64_t(0), position - 512) - c.key_first) : 0;
    const int end = (int) (position - c.key_first + 1);
    std::vector<double> output(Q, 0.);
    for (int head = 0; head < k::kQueryHeads; ++head) {
        std::vector<double> weights(end - begin);
        const int kv_head = head / (k::kQueryHeads / k::kKvHeads);
        const float* query = q.data() + (size_t) row * Q + head * k::kHeadDim;
        for (int token = begin; token < end; ++token) {
            double score = 0.;
            const size_t offset = (size_t) token * KV + kv_head * k::kHeadDim;
            for (int d = 0; d < k::kHeadDim; ++d) score += (double) query[d] * half_value(kb[offset + d]);
            weights[token - begin] = score / std::sqrt(128.);
        }
        const double maximum = *std::max_element(weights.begin(), weights.end());
        double sum = 0.;
        for (double& value : weights) { value = std::exp(value - maximum); sum += value; }
        for (int token = begin; token < end; ++token) {
            const double weight = weights[token - begin] / sum;
            const size_t offset = (size_t) token * KV + kv_head * k::kHeadDim;
            for (int d = 0; d < k::kHeadDim; ++d)
                output[head * k::kHeadDim + d] += weight * half_value(vb[offset + d]);
        }
    }
    return output;
}

void compare(const char* label, const float* actual, const double* expected, size_t count, double atol) {
    double worst = 0.;
    for (size_t i = 0; i < count; ++i) {
        const double error = std::abs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || error > atol + 3e-6 * std::abs(expected[i])) {
            std::fprintf(stderr, "%s[%zu] actual=%.10g expected=%.10g error=%.6g\n", label, i, actual[i], expected[i], error);
            throw std::runtime_error(std::string(label) + " mismatch");
        }
        worst = std::max(worst, error);
    }
    std::printf("PASS %-38s max_abs=%.6g\n", label, worst);
}

void run_case(const Case& c, cudaStream_t stream, bool benchmark) {
    auto queries = random_values((size_t) c.rows * Q, 101);
    auto key_values = random_values((size_t) c.key_count * KV, 102);
    auto value_values = random_values(key_values.size(), 103);
    if (c.variant) {
        std::fill(queries.begin(), queries.end(), 0.f);
        if (c.variant != 4) {
            std::fill(key_values.begin(), key_values.end(), 0.f);
            for (size_t i = 0; i < queries.size(); i += k::kHeadDim) {
                queries[i] = c.variant == 1 ? 1024.f : c.variant == 2 ? 1.0001f : 1.f;
                if (c.variant == 2) queries[i + 1] = -1.f;
            }
            for (size_t i = 0; i < key_values.size(); i += k::kHeadDim) {
                const int token = (int) (i / KV);
                const float sign = token % 2 ? -1.f : 1.f;
                key_values[i] = sign * (c.variant == 3 ? 0.002f : 32.f);
                if (c.variant == 2) key_values[i + 1] = sign * 32.f;
                if (c.variant >= 2)
                    for (int d = 0; d < k::kHeadDim; ++d) value_values[i + d] = sign * (c.variant == 3 ? 8192.f : 1.f);
            }
        }
    }
    std::vector<uint16_t> keys(key_values.size()), values(value_values.size());
    for (size_t i = 0; i < keys.size(); ++i) { keys[i] = half_bits(key_values[i]); values[i] = half_bits(value_values[i]); }
    constexpr size_t GUARD = 32;
    constexpr float SENTINEL = -12345.f;
    const size_t scratch_count = std::max(k::tensor_decode_attention_scratch_floats(c.key_count),
        c.local ? k::attention_local_split_scratch_floats() : k::attention_split_scratch_floats(c.key_count));
    Device<float> q(queries), output(std::vector<float>(queries.size() + GUARD, SENTINEL));
    Device<float> scratch(std::vector<float>(scratch_count + GUARD, SENTINEL));
    Device<uint16_t> dk(keys), dv(values);
    const auto run = [&](int mode) {
        if (mode == 2) k::tensor_decode_attention(q.p, dk.p, dv.p, output.p, scratch.p,
                                                   c.key_count, c.local, stream);
        else if (mode == 1 && c.local)
            k::attention_local_split_decode(q.p, dk.p, dv.p, output.p, scratch.p, stream);
        else if (mode == 1)
            k::attention_split_decode(q.p, dk.p, dv.p, output.p, c.key_count,
                                       c.key_count - 1, scratch.p, stream);
        else k::attention(q.p, dk.p, dv.p, output.p, 1, c.key_count, 0,
                           c.key_count - 1, c.local, stream);
    };
    if (benchmark) {
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        run(1); run(2);
        constexpr int repeats = 50;
        for (int sample = 0; sample < 5; ++sample) {
            float elapsed[2];
            for (int phase = 0; phase < 2; ++phase) {
                const int tensor = (sample + phase) % 2;
                check(cudaEventRecord(begin, stream));
                for (int i = 0; i < repeats; ++i) run(tensor + 1);
                check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
                check(cudaEventElapsedTime(&elapsed[tensor], begin, end));
            }
            std::printf("TENSOR_DECODE_BENCH keys=%d local=%d sample=%d repeats=%d split_us=%.6f tensor_us=%.6f speedup=%.3f\n",
                         c.key_count, (int) c.local, sample, repeats,
                         1000 * elapsed[0] / repeats, 1000 * elapsed[1] / repeats, elapsed[0] / elapsed[1]);
        }
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
        return;
    }
    const auto expected = reference(c, queries, keys, values, 0);
    const double atol = c.variant == 3 ? 0.003 : 3e-6;
    for (int mode = 0; mode < 3; ++mode) {
        run(mode); check(cudaStreamSynchronize(stream));
        const auto actual = output.download();
        const auto partials = scratch.download();
        const std::string label = std::string(c.name) + (mode == 0 ? " baseline/F64" : mode == 1 ? " split/F64" : " tensor/F64");
        compare(label.c_str(), actual.data(), expected.data(), Q, atol);
        for (size_t i = queries.size(); i < actual.size(); ++i)
            require(actual[i] == SENTINEL, "decode attention output guard overwritten");
        for (size_t i = scratch_count; i < partials.size(); ++i)
            require(partials[i] == SENTINEL, "decode attention scratch guard overwritten");
        if (c.variant == 2) require(std::abs(actual[0]) > 0.0002f, "Q low-half contribution was lost");
        if (c.variant == 3) require(std::abs(actual[0]) > 1.f, "P low-half contribution was lost");
    }
}
void graph_replay(cudaStream_t stream) {
    constexpr int Guard = 32, Q8Bytes = Q / 32 * 36;
    constexpr float Sentinel = -12345.f;
    for (int limit : {8191, 9216, 32768}) {
        auto key_values = random_values((size_t) limit * KV, 221);
        auto value_values = random_values(key_values.size(), 222);
        std::vector<uint16_t> keys(key_values.size()), values(key_values.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            keys[i] = half_bits(key_values[i]); values[i] = half_bits(value_values[i]);
        }
        Device<float> q(random_values(Q, 220));
        Device<uint16_t> dk(keys), dv(values);
        Device<int32_t> position(std::vector<int32_t>{limit - 1});
        Device<float> eager(std::vector<float>(Q + Guard, Sentinel)), output(std::vector<float>(Q + Guard, Sentinel));
        const size_t size = k::tensor_decode_attention_scratch_floats(limit);
        Device<float> scratch(std::vector<float>(size + Guard, Sentinel));
        Device<uint8_t> eager_q8(std::vector<uint8_t>(Q8Bytes + Guard, 0xa5));
        Device<uint8_t> graph_q8(std::vector<uint8_t>(Q8Bytes + Guard, 0xa5));
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        k::tensor_decode_attention_q8_graph(q.p, dk.p, dv.p, output.p, scratch.p,
                                            position.p, limit, stream, graph_q8.p);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        const std::vector<int> positions = limit == 8191 ? std::vector<int>{512, 1023, 1024, 7936, 8190}
                                                        : std::vector<int>{limit - 129, limit - 128, limit - 1};
        for (int pos : positions) {
            check(cudaMemcpy(position.p, &pos, sizeof(pos), cudaMemcpyHostToDevice));
            k::tensor_decode_attention_q8(q.p, dk.p, dv.p, eager.p, scratch.p, pos + 1, false, stream, eager_q8.p);
            check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
            const auto expected = eager.download(), actual = output.download();
            require(!std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)),
                    "graph attention changed output or guards at dynamic key count");
            require(eager_q8.download() == graph_q8.download(), "graph attention changed Q8 bytes or guards");
            const auto partials = scratch.download();
            for (size_t i = size; i < partials.size(); ++i)
                require(partials[i] == Sentinel, "graph attention exceeded scratch capacity");
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        std::printf("PASS graph attention replay, limit=%d, dynamic partition count and Q8 bytes\n", limit);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--bench";
    if (argc > 1 && !benchmark) { std::fprintf(stderr, "usage: kolibri-tensor-attention-test [--bench]\n"); return 1; }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { std::puts("SKIP no CUDA device"); return 77; }
    cudaStream_t stream = nullptr;
    try {
        if (!k::prefill_attention_supported()) { std::puts("SKIP tensor MMA requires sm_80+"); return 77; }
        check(cudaStreamCreate(&stream));
        if (benchmark) {
            for (const Case c : {Case{"global 2K", 1, 2048, 0, 2047, false},
                                Case{"global 8K", 1, 8192, 0, 8191, false},
                                Case{"global 32K", 1, 32768, 0, 32767, false},
                                Case{"local full window", 1, 513, 0, 512, true}}) run_case(c, stream, true);
        } else {
            graph_replay(stream);
            for (int keys : {1, 31, 32, 33, 2048, 8191, 8192, 8193, 32768, 32769, 262144})
                run_case(Case{"global causal prefix", 1, keys, 0, keys - 1, false}, stream, false);
            for (const Case c : {Case{"local singleton last partition", 1, 513, 0, 512, true},
                                Case{"local extreme scores", 1, 513, 0, 512, true, 1},
                                Case{"global extreme scores", 1, 8193, 0, 8192, false, 1},
                                Case{"Q hi plus low precision", 1, 2, 0, 1, false, 2},
                                Case{"P hi plus low precision", 1, 2, 0, 1, false, 3},
                                Case{"zero local query", 1, 513, 0, 512, true, 4}}) run_case(c, stream, false);
            size_t previous = 0;
            for (int keys = 1; keys <= 262144; ++keys) {
                const size_t size = k::tensor_decode_attention_scratch_floats(keys);
                require(size >= previous, "tensor scratch capacity decreases across partition transition");
                previous = size;
            }
            for (int keys : {0, 262145}) {
                bool rejected = false;
                try { k::tensor_decode_attention_scratch_floats(keys); }
                catch (const std::invalid_argument&) { rejected = true; }
                require(rejected, "tensor decode accepted invalid context size");
            }
            bool rejected = false;
            try { k::tensor_decode_attention(nullptr, nullptr, nullptr, nullptr, nullptr, 513, true, stream); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "tensor decode accepted missing scratch");
            Device<float> scratch(k::tensor_decode_attention_scratch_floats(513));
            rejected = false;
            try { k::tensor_decode_attention(nullptr, nullptr, nullptr, nullptr, scratch.p, 512, true, stream); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "tensor decode accepted incomplete local window");
        }
        check(cudaStreamSynchronize(stream)); check(cudaStreamDestroy(stream));
        std::puts(benchmark ? "DONE Kolibri tensor decode timing" : "PASS all Kolibri tensor decode attention checks");
        return 0;
    } catch (const std::exception& error) {
        if (stream) cudaStreamDestroy(stream);
        std::fprintf(stderr, "FAIL Kolibri tensor decode: %s\n", error.what());
        return 1;
    }
}
