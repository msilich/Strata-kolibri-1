// Focused numerical checks and optional resident-input timing for Kolibri prefill.
#include "prefill_attention.cuh"
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

// Independent scalar F64 attention over the exact cached FP16 values. Sampling
// query rows bounds CPU cost for the 4096-row shape; every GPU output is also
// compared with the original F32 kernel below.
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
    Device<float> q(queries), output(queries.size());
    Device<uint16_t> dk(keys), dv(values);
    const auto run = [&](bool tensor) {
        if (tensor) k::prefill_attention(q.p, dk.p, dv.p, output.p, c.rows, c.key_count,
                                         c.key_first, c.query_first, c.local, stream);
        else k::attention(q.p, dk.p, dv.p, output.p, c.rows, c.key_count,
                          c.key_first, c.query_first, c.local, stream);
    };
    if (benchmark) {
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        run(false); run(true);
        constexpr int repeats = 3;
        for (int sample = 0; sample < 3; ++sample) {
            float elapsed[2];
            for (int phase = 0; phase < 2; ++phase) {
                const bool tensor = (sample + phase) % 2;
                check(cudaEventRecord(begin, stream));
                for (int i = 0; i < repeats; ++i) run(tensor);
                check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
                check(cudaEventElapsedTime(&elapsed[tensor], begin, end));
            }
            std::printf("PREFILL_ATTENTION_BENCH rows=%d keys=%d local=%d sample=%d repeats=%d baseline_ms=%.6f tensor_ms=%.6f speedup=%.3f\n",
                         c.rows, c.key_count, (int) c.local, sample, repeats,
                         elapsed[0] / repeats, elapsed[1] / repeats, elapsed[0] / elapsed[1]);
        }
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
        return;
    }
    run(false); check(cudaStreamSynchronize(stream)); const auto baseline = output.download();
    run(true); check(cudaStreamSynchronize(stream)); const auto actual = output.download();
    const double atol = c.variant == 3 ? 0.003 : 3e-6;
    const std::vector<double> base64(baseline.begin(), baseline.end());
    compare(c.name, actual.data(), base64.data(), actual.size(), atol);
    std::vector<int> rows{0, c.rows / 2, c.rows - 1};
    std::sort(rows.begin(), rows.end()); rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (int row : rows) {
        const auto expected = reference(c, queries, keys, values, row);
        const std::string label = std::string(c.name) + " F64 row " + std::to_string(row);
        compare(label.c_str(), actual.data() + (size_t) row * Q, expected.data(), Q, atol);
    }
    if (c.variant == 2) require(std::abs(actual[0]) > 0.0002f, "Q low-half contribution was lost");
    if (c.variant == 3) require(std::abs(actual[0]) > 1.f, "P low-half contribution was lost");
}

void run_two_query_case(const char* name, int rows, int key_count, int64_t query_first,
                        cudaStream_t stream, uint16_t future_value = 0) {
    auto queries = random_values((size_t) rows * Q, 201);
    std::vector<uint16_t> keys((size_t) key_count * KV), values(keys.size());
    std::mt19937 random(202);
    std::uniform_real_distribution<float> distribution(-1.f, 1.f);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = half_bits(distribution(random));
        values[i] = half_bits(distribution(random));
    }
    // Query 0 must retain its original zero V padding even though query 1
    // shares this tile and can already see the nonfinite final value.
    if (future_value)
        std::fill_n(values.begin() + (size_t) (query_first + 1) * KV, KV, future_value);
    constexpr size_t guard = 32;
    constexpr float sentinel = 12345.25f;
    const std::vector<float> initial(queries.size() + 2 * guard, sentinel);
    Device<float> q(queries), old_output(initial), new_output(initial);
    Device<uint16_t> dk(keys), dv(values);
    check(cudaStreamSynchronize(nullptr));  // Complete pageable setup copies before kernel work.
    k::prefill_attention(q.p, dk.p, dv.p, old_output.p + guard, rows, key_count,
                         0, query_first, false, stream);
    k::prefill_attention_two_query(q.p, dk.p, dv.p, new_output.p + guard,
                                   rows, key_count, query_first, stream);
    check(cudaStreamSynchronize(stream));
    const auto baseline = old_output.download(), actual = new_output.download();
    for (size_t i = 0; i < queries.size(); ++i) {
        const float a = baseline[guard + i], b = actual[guard + i];
        // The visible nonfinite row has no portable NaN payload contract.
        // All earlier rows must remain finite and bitwise identical.
        const bool exceptional_row = future_value && i >= Q;
        if (exceptional_row && std::isnan(a) && std::isnan(b)) continue;
        if ((!exceptional_row && (!std::isfinite(a) || !std::isfinite(b))) ||
            std::memcmp(&a, &b, sizeof(float)) != 0) {
            std::fprintf(stderr, "%s[%zu] baseline=%.9g two_query=%.9g\n", name, i, a, b);
            throw std::runtime_error("two-query prefill changed output bits or causality");
        }
    }
    for (const auto* output : {&baseline, &actual}) {
        for (size_t i = 0; i < guard; ++i) {
            require(std::memcmp(&(*output)[i], &sentinel, sizeof(float)) == 0,
                    "two-query prefill changed leading output guard");
            require(std::memcmp(&(*output)[guard + queries.size() + i], &sentinel, sizeof(float)) == 0,
                    "two-query prefill changed trailing output guard");
        }
    }
    std::printf("PASS %-38s exact tensor outputs and guards%s\n", name,
                future_value ? " (visible NaNs compared by class)" : "");
}

void two_query_checks(cudaStream_t stream, bool bounded) {
    run_two_query_case("two-query CH32 boundary", 2, 33, 31, stream);
    run_two_query_case("two-query odd causal tails", 3, 33, 30, stream);
    run_two_query_case("two-query odd final tile", 3, 34, 31, stream);
    run_two_query_case("two-query single first row", 1, 1, 0, stream);
    run_two_query_case("two-query future NaN V", 2, 32, 30, stream, 0x7e00);
    run_two_query_case("two-query future Inf V", 2, 32, 30, stream, 0x7c00);
    if (!bounded) {
        run_two_query_case("two-query maximum chunk", k::kChunk, k::kChunk, 0, stream);
        run_two_query_case("two-query 262144-key cache", 3, 262144, 262141, stream);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::string(argv[1]) == "--bench";
    const bool two_query_memcheck = argc == 2 && std::string(argv[1]) == "--two-query-memcheck";
    if (argc > 1 && !benchmark && !two_query_memcheck) {
        std::fprintf(stderr, "usage: kolibri-prefill-test [--bench|--two-query-memcheck]\n"); return 1;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { std::puts("SKIP no CUDA device"); return 77; }
    cudaStream_t stream = nullptr;
    try {
        if (!k::prefill_attention_supported()) { std::puts("SKIP prefill MMA requires sm_80+"); return 77; }
        check(cudaStreamCreate(&stream));
        if (benchmark) {
            for (const Case c : {Case{"global 256/2K", 256, 2048, 0, 1792, false},
                                Case{"global 256/32K", 256, 32768, 0, 32512, false},
                                Case{"global 1024/32K", 1024, 32768, 0, 31744, false},
                                Case{"local 256", 256, 768, 32000, 32512, true},
                                Case{"local 1024", 1024, 1536, 31232, 31744, true}}) run_case(c, stream, true);
        } else if (two_query_memcheck) {
            two_query_checks(stream, true);
        } else {
            for (const Case c : {Case{"global future keys masked", 4, 8, 0, 0, false},
                                Case{"global 32-key boundary", 4, 35, 0, 30, false},
                                Case{"global full 32K", 4, 32768, 0, 32764, false},
                                Case{"global 4096-row chunk", 4096, 4096, 0, 0, false},
                                Case{"local first queries", 4, 8, 0, 0, true},
                                Case{"local window boundary", 4, 515, 0, 511, true},
                                Case{"local shifted cache", 4, 516, 32252, 32764, true},
                                Case{"local 4096-row chunk", 4096, 4608, 15872, 16384, true},
                                Case{"global extreme scores", 4, 1027, 0, 1023, false, 1},
                                Case{"local extreme scores", 4, 516, 32252, 32764, true, 1},
                                Case{"Q hi plus low precision", 1, 2, 0, 1, false, 2},
                                Case{"P hi plus low precision", 1, 2, 0, 1, false, 3},
                                Case{"zero query", 4, 515, 0, 511, true, 4}}) run_case(c, stream, false);
            bool rejected = false;
            try { k::prefill_attention(nullptr, nullptr, nullptr, nullptr, 1, 512, 2, 513, true, stream); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "prefill accepted missing local history");
            rejected = false;
            try { k::prefill_attention(nullptr, nullptr, nullptr, nullptr, k::kChunk + 1, k::kChunk + 1, 0, 0, false, stream); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "prefill accepted an oversized query chunk");
            two_query_checks(stream, false);
        }
        check(cudaStreamSynchronize(stream)); check(cudaStreamDestroy(stream));
        std::puts(benchmark ? "DONE Kolibri prefill attention timing" :
                  two_query_memcheck ? "PASS bounded two-query prefill checks" :
                                      "PASS all Kolibri prefill attention checks");
        return 0;
    } catch (const std::exception& error) {
        if (stream) cudaStreamDestroy(stream);
        std::fprintf(stderr, "FAIL Kolibri prefill attention: %s\n", error.what());
        return 1;
    }
}
