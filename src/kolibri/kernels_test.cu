// Synthetic mathematical checks; no model files or downloads are required.
// Attention uses a CPU F64 reference over the exact F16 cache sent to CUDA.
#include "kernels.cuh"
#include "decode_cache.cuh"
#include "kv_cache.cuh"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace k = strata::kolibri::kernels;
constexpr int Q = k::kQueryHeads * k::kHeadDim;
constexpr int KV = k::kKvHeads * k::kHeadDim;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <typename T> struct Device {
    T* p = nullptr;
    size_t count;
    explicit Device(size_t n) : count(n) { check(cudaMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& data) : Device(data.size()) { upload(data); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void upload(const std::vector<T>& data) {
        if (data.size() != count) throw std::runtime_error("test buffer size mismatch");
        check(cudaMemcpy(p, data.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> data(count);
        check(cudaMemcpy(data.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
        return data;
    }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void close(const std::string& label, const std::vector<float>& actual,
           const std::vector<double>& expected, double atol = 2e-6, double rtol = 2e-6) {
    require(actual.size() == expected.size(), "comparison shape mismatch");
    double worst = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i]);
        if (!std::isfinite(actual[i]) || error > atol + rtol * std::abs(expected[i])) {
            std::fprintf(stderr, "%s[%zu]: actual %.10g, expected %.10g, error %.6g\n",
                         label.c_str(), i, actual[i], expected[i], error);
            throw std::runtime_error(label + " mismatch");
        }
        worst = std::max(worst, error);
    }
    std::printf("PASS %-45s max abs error %.3g\n", label.c_str(), worst);
}

std::vector<float> random_values(size_t count, unsigned seed, float scale = 1.f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-scale, scale);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(rng);
    return values;
}

uint16_t half_bits(float value) {
    const __half half = __float2half_rn(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float half_value(uint16_t bits) {
    __half half;
    std::memcpy(&half, &bits, sizeof(bits));
    return __half2float(half);
}

void norm_and_helpers(cudaStream_t stream) {
    constexpr int rows = 3, width = 2560;
    constexpr float eps = 1e-5f;
    auto input = random_values(rows * width, 12, 4.f);
    auto weights = random_values(width, 13, 2.f);
    std::fill_n(input.begin(), width, 0.f);
    std::vector<double> expected(input.size());
    for (int row = 0; row < rows; ++row) {
        double sum = 0;
        for (int d = 0; d < width; ++d) sum += (double) input[row * width + d] * input[row * width + d];
        const double scale = 1.0 / std::sqrt(sum / width + eps);
        for (int d = 0; d < width; ++d) expected[row * width + d] = input[row * width + d] * scale * weights[d];
    }
    Device<float> x(input), w(weights), out(input.size());
    k::rms_norm(x.p, w.p, out.p, rows, width, eps, stream);
    close("RMS norm, zero and nonzero rows", out.download(), expected);
    k::rms_norm(x.p, w.p, x.p, rows, width, eps, stream);
    close("RMS norm, in place", x.download(), expected);

    auto second = random_values(input.size(), 14);
    Device<float> b(second);
    x.upload(input);
    for (size_t i = 0; i < input.size(); ++i) expected[i] = (double) input[i] + second[i];
    k::add(x.p, b.p, x.p, input.size(), stream);
    close("Residual add, alias first input", x.download(), expected, 5e-7, 0);
    x.upload(input);
    k::add(x.p, b.p, b.p, input.size(), stream);
    close("Residual add, alias second input", b.download(), expected, 5e-7, 0);

    constexpr int ffn_width = 1024;
    auto gate = random_values(4 * ffn_width, 15, 8.f);
    auto up = random_values(gate.size(), 16, 2.f);
    gate[0] = -100.f; gate[1] = 100.f; gate[2] = 0.f;
    expected.resize(gate.size());
    for (size_t i = 0; i < gate.size(); ++i)
        expected[i] = ((double) gate[i] / (1.0 + std::exp(-(double) gate[i]))) * up[i];
    Device<float> dg(gate), du(up), dy(gate.size());
    k::swiglu(dg.p, du.p, dy.p, 4, ffn_width, stream);
    close("SwiGLU, including saturated gates", dy.download(), expected);

    constexpr int gather_width = 259;
    auto source = random_values(5 * gather_width, 17);
    std::vector<int32_t> ids{4, 0, 4, 2, 1};
    std::vector<float> multipliers{0.25f, -0.5f, 1.5f, 0.f, 2.f};
    Device<float> ds(source), gathered(ids.size() * gather_width), dm(multipliers);
    Device<int32_t> di(ids);
    expected.resize(ids.size() * gather_width);
    for (size_t row = 0; row < ids.size(); ++row)
        for (int d = 0; d < gather_width; ++d) expected[row * gather_width + d] = source[ids[row] * gather_width + d];
    k::gather_rows(ds.p, di.p, gathered.p, (int) ids.size(), gather_width, stream);
    close("Gather, reordered and duplicated rows", gathered.download(), expected, 0, 0);

    auto initial = random_values(source.size(), 18);
    Device<float> scattered(initial);
    expected.assign(initial.begin(), initial.end());
    for (size_t row = 0; row < ids.size(); ++row)
        for (int d = 0; d < gather_width; ++d)
            expected[ids[row] * gather_width + d] += (double) source[row * gather_width + d] * multipliers[row];
    k::scatter_add_weighted(ds.p, di.p, dm.p, scattered.p, (int) ids.size(), gather_width, stream);
    close("Weighted scatter, repeated destinations", scattered.download(), expected);
}

std::vector<double> qk_reference(const std::vector<float>& input, const std::vector<float>& weights,
                                 int heads, int64_t first, bool local, float theta, float eps) {
    std::vector<double> expected(input.size());
    const int rows = (int) input.size() / (heads * k::kHeadDim);
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            const size_t offset = ((size_t) row * heads + head) * k::kHeadDim;
            double sum = 0;
            for (int d = 0; d < k::kHeadDim; ++d) sum += (double) input[offset + d] * input[offset + d];
            double normalized[k::kHeadDim];
            for (int d = 0; d < k::kHeadDim; ++d)
                normalized[d] = input[offset + d] * weights[d] / std::sqrt(sum / k::kHeadDim + eps);
            for (int d = 0; d < k::kHeadDim / 2; ++d) {
                const double angle = local ? (first + row) / std::pow((double) theta, (double) d / 64) : 0;
                expected[offset + d] = normalized[d] * std::cos(angle) - normalized[d + 64] * std::sin(angle);
                expected[offset + d + 64] = normalized[d + 64] * std::cos(angle) + normalized[d] * std::sin(angle);
            }
        }
    }
    return expected;
}

void qk_norm_rope(cudaStream_t stream) {
    constexpr int rows = 4;
    constexpr float theta = 1000000.f, eps = 1e-5f;
    auto q = random_values(rows * Q, 20);
    auto keys = random_values(rows * KV, 21);
    auto qw = random_values(k::kHeadDim, 22, 2.f);
    auto kw = random_values(k::kHeadDim, 23, 2.f);
    std::fill_n(q.begin(), k::kHeadDim, 0.f);
    Device<float> dq(q), dk(keys), dqw(qw), dkw(kw);
    for (const int64_t first : {int64_t(0), int64_t(511), int64_t(32764), int64_t(16777217)}) {
        dq.upload(q); dk.upload(keys);
        k::qk_norm_rope(dq.p, dk.p, dqw.p, dkw.p, rows, first, true, theta, eps, stream);
        close("Local Q norm/NeoX RoPE at " + std::to_string(first), dq.download(),
              qk_reference(q, qw, k::kQueryHeads, first, true, theta, eps));
        close("Local K norm/NeoX RoPE at " + std::to_string(first), dk.download(),
              qk_reference(keys, kw, k::kKvHeads, first, true, theta, eps));
    }
    dq.upload(q); dk.upload(keys);
    k::qk_norm_rope(dq.p, dk.p, dqw.p, dkw.p, rows, 32764, false, theta, eps, stream);
    close("Global Q norm without RoPE", dq.download(), qk_reference(q, qw, k::kQueryHeads, 32764, false, theta, eps));
    close("Global K norm without RoPE", dk.download(), qk_reference(keys, kw, k::kKvHeads, 32764, false, theta, eps));
}

void qk_cached(cudaStream_t stream) {
    constexpr int guard = 128;
    constexpr float theta = 10000.f, eps = 1e-6f;
    const auto qw = random_values(k::kHeadDim, 24, 2.f);
    const auto kw = random_values(k::kHeadDim, 25, 2.f);
    Device<float> dqw(qw), dkw(kw);
    struct Case { int rows; int64_t first; };
    for (const auto test : {Case{1, 0}, Case{1, 511}, Case{1, 512}, Case{1, 513}, Case{1, 32767},
                            Case{512, 0}, Case{512, 511}, Case{512, 32256},
                            Case{4096, 0}, Case{4096, 28672}}) {
        auto q = random_values((size_t) test.rows * Q + 2 * guard, 26, 4.f);
        auto keys = random_values((size_t) test.rows * KV + 2 * guard, 27, 4.f);
        for (int d = 0; d < k::kHeadDim; ++d) {
            q[guard + d] = d & 1 ? -0.f : 0.f;
            keys[guard + d] = std::ldexp((float) (d % 17 - 8), -147);
        }
        std::vector<float> table((size_t) test.rows * k::kHeadDim + 2 * guard, 12345.f);
        Device<float> reference_q(q), reference_k(keys), cached_q(q), cached_k(keys), angles(table);
        k::rope_table(angles.p + guard, test.rows, test.first, theta, stream);
        k::qk_norm_rope(reference_q.p + guard, reference_k.p + guard, dqw.p, dkw.p,
                        test.rows, test.first, true, theta, eps, stream);
        k::qk_norm_rope_cached(cached_q.p + guard, cached_k.p + guard, dqw.p, dkw.p,
                               test.rows, angles.p + guard, eps, stream);
        const auto expected_q = reference_q.download(), expected_k = reference_k.download();
        const auto got_q = cached_q.download(), got_k = cached_k.download(), got_angles = angles.download();
        require(!std::memcmp(expected_q.data(), got_q.data(), got_q.size() * sizeof(float)),
                "cached RoPE Q differs from original or overwrote guard");
        require(!std::memcmp(expected_k.data(), got_k.data(), got_k.size() * sizeof(float)),
                "cached RoPE K differs from original or overwrote guard");
        for (int i = 0; i < guard; ++i) {
            require(got_q[i] == q[i] && got_q[got_q.size() - 1 - i] == q[q.size() - 1 - i], "Q guard overwritten");
            require(got_k[i] == keys[i] && got_k[got_k.size() - 1 - i] == keys[keys.size() - 1 - i], "K guard overwritten");
            require(got_angles[i] == 12345.f && got_angles[got_angles.size() - 1 - i] == 12345.f, "RoPE table guard overwritten");
        }
        // Rebuilding a row as a decode call must preserve both table and Q/K bits.
        if (test.rows > 1) for (const int row : {0, test.rows / 2, test.rows - 1}) {
            Device<float> one_q(std::vector<float>(q.begin() + guard + (size_t) row * Q,
                                                  q.begin() + guard + (size_t) (row + 1) * Q));
            Device<float> one_k(std::vector<float>(keys.begin() + guard + (size_t) row * KV,
                                                  keys.begin() + guard + (size_t) (row + 1) * KV));
            Device<float> one_table(k::kHeadDim);
            k::rope_table(one_table.p, 1, test.first + row, theta, stream);
            k::qk_norm_rope_cached(one_q.p, one_k.p, dqw.p, dkw.p, 1, one_table.p, eps, stream);
            const auto single_q = one_q.download(), single_k = one_k.download(), single_table = one_table.download();
            require(!std::memcmp(single_q.data(), got_q.data() + guard + (size_t) row * Q, Q * sizeof(float)),
                    "cached RoPE Q chunk/decode mismatch");
            require(!std::memcmp(single_k.data(), got_k.data() + guard + (size_t) row * KV, KV * sizeof(float)),
                    "cached RoPE K chunk/decode mismatch");
            require(!std::memcmp(single_table.data(), got_angles.data() + guard + row * k::kHeadDim, k::kHeadDim * sizeof(float)),
                    "RoPE table chunk/decode mismatch");
        }
        std::printf("PASS cached QK/RoPE bitwise, rows=%d first=%lld, guards and chunk/decode\n",
                    test.rows, (long long) test.first);
    }
}

void qk_append_decode(cudaStream_t stream) {
    constexpr int guard = 128;
    constexpr float theta = 10000.f, eps = 1e-6f;
    auto q = random_values(Q + 2 * guard, 28, 4.f);
    auto keys = random_values(KV + 2 * guard, 29, 4.f);
    auto values = random_values(KV + 2 * guard, 30, 5.f);
    for (int i = 0; i < k::kHeadDim; ++i) {
        q[guard + i] = i & 1 ? -0.f : 0.f;
        keys[guard + i] = std::ldexp((float) (i % 17 - 8), -147);
    }
    const float edge[] = {0.f, -0.f, std::ldexp(1.f, -24), std::ldexp(1.f, -14),
                          1.f + std::ldexp(1.f, -11), 1.f + 3 * std::ldexp(1.f, -11),
                          65504.f, 65520.f, -65520.f, INFINITY, -INFINITY};
    std::copy(std::begin(edge), std::end(edge), values.begin() + guard);
    const uint32_t nan_bits = 0x7fc01234u;
    std::memcpy(values.data() + guard + 11, &nan_bits, sizeof nan_bits);
    Device<float> dqw(random_values(k::kHeadDim, 24, 2.f)), dkw(random_values(k::kHeadDim, 25, 2.f));
    Device<float> dv(values), table(k::kHeadDim);
    struct Case { bool local; int64_t position; int offset; };
    // The large local position tests the cached angles; the global 32k crossing
    // tests cache addressing without adding full-262k allocations to this suite.
    for (const auto test : {Case{true, 0, 0}, Case{true, 511, 511}, Case{true, 512, 512},
                            Case{true, 262143, 512}, Case{false, 0, 0}, Case{false, 32768, 32768}}) {
        Device<float> reference_q(q), reference_k(keys), fused_q(q), fused_k(keys);
        std::vector<uint16_t> initial((size_t) (test.offset + 2) * KV, half_bits(-17.f));
        Device<uint16_t> kc(initial), vc(initial);
        k::rope_table(table.p, 1, test.position, theta, stream);
        if (test.local)
            k::qk_norm_rope_cached(reference_q.p + guard, reference_k.p + guard, dqw.p, dkw.p,
                                   1, table.p, eps, stream);
        else
            k::qk_norm_rope(reference_q.p + guard, reference_k.p + guard, dqw.p, dkw.p,
                            1, test.position, false, theta, eps, stream);
        k::append_kv(reference_k.p + guard, dv.p + guard, kc.p, vc.p, 1, test.offset, stream);
        check(cudaStreamSynchronize(stream));
        const auto expected_k = kc.download(), expected_v = vc.download();
        kc.upload(initial); vc.upload(initial);
        k::qk_norm_rope_append_decode(fused_q.p + guard, fused_k.p + guard, dv.p + guard,
                                      dqw.p, dkw.p, test.local ? table.p : nullptr, test.local,
                                      kc.p, vc.p, test.offset, eps, stream);
        check(cudaStreamSynchronize(stream));
        const auto expected_q = reference_q.download(), expected_f32_k = reference_k.download();
        const auto actual_q = fused_q.download(), actual_f32_k = fused_k.download();
        require(!std::memcmp(expected_q.data(), actual_q.data(), q.size() * sizeof(float)),
                "decode QK append changes F32 Q or guards");
        require(!std::memcmp(expected_f32_k.data(), actual_f32_k.data(), keys.size() * sizeof(float)),
                "decode QK append changes F32 K or guards");
        require(kc.download() == expected_k && vc.download() == expected_v,
                "decode QK append changes cache bytes or guards");
        const auto actual_v = dv.download();
        require(!std::memcmp(values.data(), actual_v.data(), values.size() * sizeof(float)),
                "decode QK append modifies V source");
        std::printf("PASS decode QK/KV bitwise, local=%d position=%lld offset=%d, guards and V source\n",
                    test.local, (long long) test.position, test.offset);

        Device<int32_t> position(std::vector<int32_t>{(int32_t) test.position});
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        k::rope_table_decode_graph(table.p, position.p, theta, stream);
        if (test.local)
            k::qk_norm_rope_append_decode(fused_q.p + guard, fused_k.p + guard, dv.p + guard,
                                          dqw.p, dkw.p, table.p, true, kc.p, vc.p, test.offset, eps, stream);
        else
            k::qk_norm_append_decode_graph(fused_q.p + guard, fused_k.p + guard, dv.p + guard,
                                            dqw.p, dkw.p, kc.p, vc.p, position.p, eps, stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        for (int step = 0; step < 2; ++step) {
            const int32_t next = (int32_t) test.position + (step ? (test.position == 262143 ? -1 : 1) : 0);
            position.upload({next});
            reference_q.upload(q); reference_k.upload(keys);
            fused_q.upload(q); fused_k.upload(keys); kc.upload(initial); vc.upload(initial);
            k::rope_table(table.p, 1, next, theta, stream);
            check(cudaStreamSynchronize(stream));
            const auto expected_angles = table.download();
            if (test.local)
                k::qk_norm_rope_cached(reference_q.p + guard, reference_k.p + guard, dqw.p, dkw.p,
                                       1, table.p, eps, stream);
            else
                k::qk_norm_rope(reference_q.p + guard, reference_k.p + guard, dqw.p, dkw.p,
                                1, next, false, theta, eps, stream);
            k::append_kv(reference_k.p + guard, dv.p + guard, kc.p, vc.p, 1,
                          test.local ? test.offset : next, stream);
            check(cudaStreamSynchronize(stream));
            const auto graph_expected_k = kc.download(), graph_expected_v = vc.download();
            kc.upload(initial); vc.upload(initial);
            check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
            const auto graph_q = fused_q.download(), graph_k = fused_k.download();
            const auto ref_q = reference_q.download(), ref_k = reference_k.download();
            require(table.download() == expected_angles, "graph RoPE position did not advance");
            require(!std::memcmp(ref_q.data(), graph_q.data(), ref_q.size() * sizeof(float)) &&
                    !std::memcmp(ref_k.data(), graph_k.data(), ref_k.size() * sizeof(float)),
                    "graph QK differs from eager normalization");
            require(kc.download() == graph_expected_k && vc.download() == graph_expected_v,
                    "graph KV append used stale position or changed guards");
        }
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
        std::puts("PASS graph replay advances RoPE and global KV position bitwise");
    }
}


__global__ void dequant_int8_test(const strata::kolibri::kv::Int8Head* cache, uint16_t* output) {
    const int piece = threadIdx.x;
    *reinterpret_cast<uint4*>(output + blockIdx.x * 128 + piece * 8) =
        strata::kolibri::kv::detail::load8(cache, blockIdx.x, piece);
}

void int8_cache(cudaStream_t stream) {
    namespace kv = strata::kolibri::kv;
    constexpr int rows = 3, offset = 2;
    auto keys = random_values(rows * KV, 471, 4.f), values = random_values(rows * KV, 472, 5.f);
    std::fill_n(keys.begin(), 64, 0.f);
    std::fill_n(values.begin(), 64, 1e-8f);
    keys[64] = 127.f; keys[65] = 1.5f; keys[66] = 2.5f; keys[67] = -1.5f;
    std::vector<kv::Int8Head> initial((offset + rows + 1) * k::kKvHeads);
    std::memset(initial.data(), 0x5a, initial.size() * sizeof(kv::Int8Head));
    Device<float> dk(keys), dv(values);
    Device<kv::Int8Head> kc(initial), vc(initial);
    kv::append_int8(dk.p, dv.p, kc.p, vc.p, rows, offset, stream);
    check(cudaStreamSynchronize(stream));
    auto reference = [&](const std::vector<float>& input) {
        auto expected = initial;
        for (int head = 0; head < rows * k::kKvHeads; ++head) {
            auto& out = expected[offset * k::kKvHeads + head];
            for (int group = 0; group < 2; ++group) {
                float amax = 0.f;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, std::abs(input[head * 128 + group * 64 + i]));
                out.scales[group] = half_bits(amax / 127.f);
                const float scale = half_value(out.scales[group]);
                for (int i = 0; i < 64; ++i) {
                    const int j = group * 64 + i;
                    out.values[j] = scale > 0.f ? (int8_t) std::clamp((int) std::nearbyintf(input[head * 128 + j] / scale), -127, 127) : 0;
                }
            }
        }
        return expected;
    };
    auto expected_k = reference(keys), expected_v = reference(values);
    auto got_k = kc.download(), got_v = vc.download();
    require(!std::memcmp(got_k.data(), expected_k.data(), got_k.size() * sizeof(kv::Int8Head)) &&
            !std::memcmp(got_v.data(), expected_v.data(), got_v.size() * sizeof(kv::Int8Head)),
            "INT8 KV differs from CPU scale/code oracle or overwrites guards");

    // The fused decode must normalize Q/K exactly as before, then store the
    // same INT8 bytes as the independent append kernel, including graph replay.
    auto queries = random_values(Q, 473, 3.f), one_key = random_values(KV, 474, 4.f);
    auto one_value = random_values(KV, 475, 5.f);
    Device<float> qw(random_values(k::kHeadDim, 476)), kw(random_values(k::kHeadDim, 477));
    Device<float> rq(queries), rk(one_key), fq(queries), fk(one_key), v(one_value);
    Device<int32_t> position(std::vector<int32_t>{offset});
    cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    k::qk_norm_append_decode_int8(fq.p, fk.p, v.p, qw.p, kw.p, kc.p, vc.p, 0, position.p, 1e-6f, stream);
    check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&executable, graph, 0));
    for (int pos : {offset, offset + 1}) {
        rq.upload(queries); rk.upload(one_key); fq.upload(queries); fk.upload(one_key);
        kc.upload(initial); vc.upload(initial); position.upload({pos});
        k::qk_norm_rope(rq.p, rk.p, qw.p, kw.p, 1, pos, false, 10000.f, 1e-6f, stream);
        kv::append_int8(rk.p, v.p, kc.p, vc.p, 1, pos, stream);
        check(cudaStreamSynchronize(stream)); expected_k = kc.download(); expected_v = vc.download();
        kc.upload(initial); vc.upload(initial);
        check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
        got_k = kc.download(); got_v = vc.download();
        require(!std::memcmp(got_k.data(), expected_k.data(), got_k.size() * sizeof(kv::Int8Head)) &&
                !std::memcmp(got_v.data(), expected_v.data(), got_v.size() * sizeof(kv::Int8Head)),
                "fused INT8 KV graph changes bytes or uses a stale position");
        const auto eq = rq.download(), ek = rk.download(), aq = fq.download(), ak = fk.download();
        require(!std::memcmp(eq.data(), aq.data(), eq.size() * sizeof(float)) &&
                !std::memcmp(ek.data(), ak.data(), ek.size() * sizeof(float)), "INT8 KV changed Q/K normalization");
    }
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    keys[0] = NAN; dk.upload(keys);
    kv::append_int8(dk.p, dv.p, kc.p, vc.p, rows, offset, stream); check(cudaStreamSynchronize(stream));
    require(std::isnan(half_value(kc.download()[offset * k::kKvHeads].scales[0])),
            "INT8 KV silently discarded a non-finite source");
    std::vector<kv::Int8Head> records;
    for (uint16_t scale : {0x0000, 0x0001, 0x03ff, 0x0400, 0x1400, 0x3c00, 0x5400, 0x7bff, 0x7c00, 0x7e00}) {
        for (int h = 0; h < 2; ++h) {
            kv::Int8Head record{};
            record.scales[0] = record.scales[1] = scale;
            for (int d = 0; d < 128; ++d) record.values[d] = (int8_t) (h * 128 + d - 128);
            records.push_back(record);
        }
    }
    Device<kv::Int8Head> encoded(records);
    Device<uint16_t> unpacked(records.size() * 128);
    dequant_int8_test<<<records.size(), 16, 0, stream>>>(encoded.p, unpacked.p);
    check(cudaGetLastError()); check(cudaStreamSynchronize(stream));
    const auto decoded = unpacked.download();
    for (size_t h = 0; h < records.size(); ++h) for (int d = 0; d < 128; ++d) {
        const float expected = (float) records[h].values[d] * half_value(records[h].scales[d / 64]);
        require(std::isnan(expected) ? std::isnan(half_value(decoded[h * 128 + d])) :
                                      decoded[h * 128 + d] == half_bits(expected),
                "packed INT8 dequantization changes FP16 rounding or non-finite classification");
    }
    std::puts("PASS INT8 KV CPU oracle, rounding, underflow, guards, fused graph positions and non-finite scale");
}

void append(cudaStream_t stream) {
    constexpr int rows = k::kChunk, offset = 7;
    auto keys = random_values(rows * KV, 31, 5.f);
    auto values = random_values(rows * KV, 32, 5.f);
    keys[0] = 0.f; keys[1] = -0.f; keys[2] = 0.00001f;
    std::vector<uint16_t> initial((offset + rows + 1) * KV, half_bits(-17.f));
    Device<float> dk(keys), dv(values);
    Device<uint16_t> kc(initial), vc(initial);
    k::append_kv(dk.p, dv.p, kc.p, vc.p, rows, offset, stream);
    const auto actual_k = kc.download(), actual_v = vc.download();
    for (size_t i = 0; i < initial.size(); ++i) {
        const bool written = i >= offset * KV && i < (offset + rows) * KV;
        require(actual_k[i] == (written ? half_bits(keys[i - offset * KV]) : initial[i]), "F16 key append or guard mismatch");
        require(actual_v[i] == (written ? half_bits(values[i - offset * KV]) : initial[i]), "F16 value append or guard mismatch");
    }
    std::printf("PASS F16 KV append, %d rows and offset guards\n", rows);
}

void routing(cudaStream_t stream) {
    constexpr int rows = k::kChunk;
    auto logits = random_values(rows * k::kExperts, 40, 3.f);
    auto bias = random_values(k::kExperts, 41, 5.f);
    // Cancellation gives exact ties across all three 128-expert partitions.
    for (int expert = 0; expert < k::kExperts; ++expert) {
        logits[expert] = -bias[expert];
        logits[k::kExperts + expert] = -bias[expert] - 8.f;
        logits[3 * k::kExperts + expert] = -80.f;
        logits[4 * k::kExperts + expert] = 80.f;
    }
    const std::vector<int32_t> tied{0, 127, 128, 255, 256, 383};
    for (int expert : tied) logits[k::kExperts + expert] = -bias[expert];
    Device<float> dl(logits), db(bias), dw(rows * k::kTopK);
    Device<int32_t> di(rows * k::kTopK);
    k::route(dl.p, db.p, di.p, dw.p, rows, stream);
    const auto ids = di.download();
    const auto weights = dw.download();
    std::vector<double> expected(weights.size());
    for (int row = 0; row < rows; ++row) {
        std::vector<int> order(k::kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const float av = logits[row * k::kExperts + a] + bias[a];
            const float bv = logits[row * k::kExperts + b] + bias[b];
            return av > bv || (av == bv && a < b);
        });
        for (int rank = 0; rank < k::kTopK; ++rank) {
            const int index = row * k::kTopK + rank;
            require(ids[index] == order[rank], "routing expert ID mismatch");
            expected[index] = 1.0 / (1.0 + std::exp(-(double) logits[row * k::kExperts + order[rank]]));
        }
    }
    for (int rank = 0; rank < k::kTopK; ++rank) {
        require(ids[rank] == rank, "all-expert ties must choose lowest IDs");
        require(ids[k::kTopK + rank] == tied[rank], "cross-partition ties must choose lowest IDs");
    }
    require(std::accumulate(weights.begin() + 4 * k::kTopK, weights.begin() + 5 * k::kTopK, 0.f) == 6.f,
            "routing weights must not be renormalized");
    close("384 experts/top 6, bias, ties, sigmoid", weights, expected, 2e-7, 2e-6);
    Device<int32_t> decode_ids(k::kTopK);
    Device<float> decode_weights(k::kTopK);
    for (int row = 0; row < 16; ++row) {
        k::route_decode(dl.p + (size_t) row * k::kExperts, db.p, decode_ids.p, decode_weights.p, stream);
        const auto actual_ids = decode_ids.download();
        const auto actual_weights = decode_weights.download();
        require(!std::memcmp(actual_ids.data(), ids.data() + row * k::kTopK, k::kTopK * sizeof(int32_t)),
                "decode router IDs differ from original");
        require(!std::memcmp(actual_weights.data(), weights.data() + row * k::kTopK, k::kTopK * sizeof(float)),
                "decode router sigmoid bits differ from original");
    }
    // Check existing non-finite behavior only where the original selects valid
    // IDs. All-NaN rows are excluded: the original reads beyond the logits.
    for (int trial = 0; trial < 3; ++trial) {
        std::vector<float> special(k::kExperts, -INFINITY), zero_bias(k::kExperts, 0.f);
        if (trial == 1) for (int i = 0; i < k::kExperts; ++i) special[i] = i % 13 == 0 ? INFINITY : float(i % 7);
        if (trial == 2) {
            const uint32_t nan_bits = 0x7fc12345;
            for (float& value : special) std::memcpy(&value, &nan_bits, sizeof(float));
            special[383] = 0.f;
        }
        Device<float> input(special), bias_zero(zero_bias), old_weights(k::kTopK);
        Device<int32_t> old_ids(k::kTopK);
        k::route(input.p, bias_zero.p, old_ids.p, old_weights.p, 1, stream);
        k::route_decode(input.p, bias_zero.p, decode_ids.p, decode_weights.p, stream);
        const auto expected_ids = old_ids.download(), actual_ids = decode_ids.download();
        const auto expected_weights = old_weights.download(), actual_weights = decode_weights.download();
        require(!std::memcmp(expected_ids.data(), actual_ids.data(), k::kTopK * sizeof(int32_t)),
                "decode router non-finite IDs differ from original");
        require(!std::memcmp(expected_weights.data(), actual_weights.data(), k::kTopK * sizeof(float)),
                "decode router non-finite weights differ from original");
    }
    std::puts("PASS decode warp router, bitwise ties/sigmoid and valid non-finite behavior");
}

void route_cache(cudaStream_t stream) {
    namespace d = strata::kolibri::decode_cache;
    constexpr int stride = 64;
    Device<uint8_t> storage(12 * stride);
    Device<d::LayerState> state(1);
    Device<d::Counters> counters(1);
    Device<d::Plan> plan(1);
    Device<d::Receipt> receipt(1);
    Device<int32_t> ids(8);
    Device<float> weights(8);
    const auto equal_counts = [](const d::Counters& a, const d::Counters& b) {
        require(a.tick == b.tick && a.hits == b.hits && a.misses == b.misses, "route/cache counters differ");
    };
    const auto equal_receipts = [&](const d::Receipt& a, const d::Receipt& b) {
        require(a.status == b.status && a.copy_count == b.copy_count && a.mutation_count == b.mutation_count,
                "route/cache receipt status/counts differ");
        equal_counts(a.counters, b.counters);
        for (int i = 0; i < d::Selected; ++i) {
            const auto x = a.mutation[i], y = b.mutation[i];
            require(a.ids[i] == b.ids[i] && x.kind == y.kind && x.old_expert == y.old_expert &&
                    x.new_expert == y.new_expert && x.slot == y.slot, "route/cache ordered receipt differs");
        }
    };
    // Admission, all hits, tied-age replacement, protected last expert with one
    // slot, routing ties, and safe non-finite duplicate-ID rejection.
    for (int trial = 0; trial < 6; ++trial) {
        const int slots = trial == 3 ? 1 : 6;
        std::vector<int32_t> ranked{383, 7, 4, 0, 3, 2};
        if (trial == 4) ranked = {0, 1, 2, 3, 4, 5};
        if (trial == 5) ranked.assign(6, 0);
        auto sorted = ranked;
        std::sort(sorted.begin(), sorted.end());
        std::vector<float> logits(d::Experts, -8.f), bias(d::Experts, 0.f);
        for (int i = 0; i < 6; ++i) { logits[ranked[i]] = 0.f; bias[ranked[i]] = 20.f - i; }
        // Bias chooses the rank, but the exactly representable sigmoid is 0.5.
        if (trial >= 4) {
            std::fill(logits.begin(), logits.end(), trial == 4 ? 0.f : -INFINITY);
            std::fill(bias.begin(), bias.end(), 0.f);
        }
        Device<float> dl(logits), db(bias);
        d::LayerState initial{};
        std::fill_n(initial.slot, d::Experts, -1);
        if (trial == 1 || trial == 2 || trial == 5) {
            const auto resident = trial == 1 ? sorted : std::vector<int32_t>{0, 1, 2, 3, 4, 383};
            for (int i = 0; i < 6; ++i) { initial.slot[resident[i]] = i; initial.age[resident[i]] = 3; }
            initial.next_free = 6;
        }
        if (trial == 3) { initial.slot[383] = 0; initial.age[383] = 3; initial.next_free = 1; }
        const d::Counters before{9, 10, 20};
        const d::LayerView view{state.p, storage.p, storage.p + 6 * stride, stride, stride, slots};
        struct Result { std::vector<int32_t> ids; std::vector<float> weights; d::LayerState state;
                        d::Counters counts; d::Plan plan; d::Receipt receipt; };
        const auto run = [&](bool fused) {
            state.upload({initial}); counters.upload({before}); plan.upload({d::Plan{}}); receipt.upload({d::Receipt{}});
            ids.upload(std::vector<int32_t>(8, -123)); weights.upload(std::vector<float>(8, -123.f));
            // Existing blocking test stream orders Device's default-stream setup copies.
            if (fused) d::route_plan(dl.p, db.p, ids.p + 1, weights.p + 1, view, counters.p, plan.p, stream, receipt.p);
            else {
                k::route_decode(dl.p, db.p, ids.p + 1, weights.p + 1, stream);
                d::plan(ids.p + 1, view, counters.p, plan.p, stream, receipt.p);
            }
            check(cudaStreamSynchronize(stream));
            return Result{ids.download(), weights.download(), state.download()[0], counters.download()[0],
                          plan.download()[0], receipt.download()[0]};
        };
        const auto separate = run(false), fused = run(true);
        require(separate.ids == fused.ids && !std::memcmp(separate.weights.data(), fused.weights.data(), 8 * sizeof(float)),
                "route/cache ID or weight bits differ");
        require(fused.ids.front() == -123 && fused.ids.back() == -123 &&
                fused.weights.front() == -123.f && fused.weights.back() == -123.f, "route/cache output guard changed");
        equal_counts(separate.counts, fused.counts);
        equal_receipts(separate.plan.receipt, fused.plan.receipt);
        equal_receipts(separate.receipt, fused.receipt);
        equal_receipts(fused.plan.receipt, fused.receipt);
        for (int i = 0; i < 6; ++i) {
            require(fused.ids[i + 1] == ranked[i] && fused.weights[i + 1] == (trial == 5 ? 0.f : .5f),
                    "route/cache expected rank or unbiased sigmoid changed");
            require(separate.plan.ranks[i] == fused.plan.ranks[i] && separate.plan.slots[i] == fused.plan.slots[i] &&
                    separate.plan.weights[i] == fused.plan.weights[i] && separate.plan.copies[i].expert == fused.plan.copies[i].expert &&
                    separate.plan.copies[i].destination == fused.plan.copies[i].destination, "route/cache plan fields differ");
        }
        const int hits[] = {0, 6, 5, 1, 0, 0}, misses[] = {6, 0, 1, 5, 6, 0};
        auto expected = initial;
        if (trial != 5) {
            for (int id : sorted) expected.age[id] = 10;
            if (trial == 0 || trial == 4) {
                for (int i = 0; i < 6; ++i) expected.slot[sorted[i]] = i;
                expected.next_free = 6;
            }
            if (trial == 2) { expected.slot[1] = -1; expected.slot[7] = 1; expected.age[1] = 10; }
        }
        equal_counts(fused.counts, {trial == 5 ? 9u : 10u, uint64_t(10 + hits[trial]), uint64_t(20 + misses[trial])});
        require(fused.state.next_free == expected.next_free && separate.state.next_free == expected.next_free,
                "route/cache next-free changed");
        for (int i = 0; i < d::Experts; ++i)
            require(fused.state.slot[i] == expected.slot[i] && fused.state.age[i] == expected.age[i] &&
                    separate.state.slot[i] == expected.slot[i] && separate.state.age[i] == expected.age[i],
                    "route/cache expected slot/recency/protection changed");
        require(fused.receipt.status == (trial == 5 ? d::DuplicateExpert : d::Ok) &&
                fused.receipt.copy_count == misses[trial] &&
                fused.receipt.mutation_count == (trial == 0 || trial == 4 ? 6 : trial == 2 ? 1 : 0),
                "route/cache expected receipt changed");
        if (trial == 5) continue;
        for (int i = 0; i < 6; ++i) {
            const int slot = expected.slot[sorted[i]];
            require(fused.plan.receipt.ids[i] == sorted[i] && fused.plan.ranks[i] >= 0 && fused.plan.ranks[i] < 6 &&
                    ranked[fused.plan.ranks[i]] == sorted[i] &&
                    fused.plan.slots[i] == slot && fused.plan.weights[i] ==
                    (slot >= 0 ? view.cache_base + slot * stride : view.miss_base + i * stride),
                    "route/cache expected sorted pointers/ranks changed");
        }
        int copy = 0;
        for (int i = 0; i < 6; ++i) {
            if (initial.slot[sorted[i]] < 0) {
                require(fused.plan.copies[copy].expert == sorted[i] &&
                        fused.plan.copies[copy].destination == fused.plan.weights[i], "route/cache copy order changed");
                ++copy;
            }
            if (trial == 0 || trial == 4) {
                const auto m = fused.receipt.mutation[i];
                require(m.kind == d::Admit && m.old_expert == -1 && m.new_expert == sorted[i] && m.slot == i,
                        "route/cache admission order changed");
            }
        }
        if (trial == 2) {
            const auto m = fused.receipt.mutation[0];
            require(m.kind == d::Replace && m.old_expert == 1 && m.new_expert == 7 && m.slot == 1,
                    "route/cache tied-age victim changed");
        }
    }
    std::puts("PASS fused route/cache: exact fields, biased ranks, hits/misses, protected slots and invalid routes");
}

std::vector<double> attention_reference(const std::vector<float>& queries,
                                        const std::vector<uint16_t>& key_bits,
                                        const std::vector<uint16_t>& value_bits,
                                        int64_t key_first, int64_t query_first, bool local) {
    std::vector<double> keys(key_bits.size()), values(value_bits.size()), output(queries.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = half_value(key_bits[i]);
        values[i] = half_value(value_bits[i]);
    }
    const int rows = (int) queries.size() / Q;
    for (int row = 0; row < rows; ++row) {
        const int64_t position = query_first + row;
        const int64_t oldest = local ? std::max(int64_t(0), position - 512) : 0;
        const int begin = (int) (oldest - key_first), end = (int) (position - key_first + 1);
        for (int head = 0; head < k::kQueryHeads; ++head) {
            const int kv_head = head / 12;
            const size_t query_offset = ((size_t) row * k::kQueryHeads + head) * k::kHeadDim;
            std::vector<double> probabilities(end - begin);
            for (int token = begin; token < end; ++token) {
                const size_t key_offset = (size_t) token * KV + kv_head * k::kHeadDim;
                double dot = 0;
                for (int d = 0; d < k::kHeadDim; ++d) dot += queries[query_offset + d] * keys[key_offset + d];
                probabilities[token - begin] = dot / std::sqrt(128.0);
            }
            const double maximum = *std::max_element(probabilities.begin(), probabilities.end());
            double sum = 0;
            for (double& probability : probabilities) {
                probability = std::exp(probability - maximum);
                sum += probability;
            }
            double* result = output.data() + query_offset;
            for (int token = begin; token < end; ++token) {
                const double weight = probabilities[token - begin] / sum;
                const double* value = values.data() + (size_t) token * KV + kv_head * k::kHeadDim;
                for (int d = 0; d < k::kHeadDim; ++d) result[d] += weight * value[d];
            }
        }
    }
    return output;
}

void attention_case(const char* label, int rows, int64_t query_first, int64_t key_first,
                    bool local, cudaStream_t stream, bool large_scores = false) {
    const int key_count = (int) (query_first + rows - key_first);
    auto queries = random_values((size_t) rows * Q, 51);
    auto keys = random_values((size_t) key_count * KV, 52);
    auto values = random_values(keys.size(), 53, 3.f);
    if (large_scores) {
        // Exact large equal maxima and strongly negative scores exercise stable softmax.
        std::fill(queries.begin(), queries.end(), 0.f);
        for (size_t i = 0; i < queries.size(); i += k::kHeadDim) queries[i] = 1024.f;
        for (size_t i = 0; i < keys.size(); i += k::kHeadDim) keys[i] = (i / KV % 3 == 0) ? 32.f : -32.f;
    }
    std::vector<uint16_t> key_bits(keys.size()), value_bits(values.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        key_bits[i] = half_bits(keys[i]);
        value_bits[i] = half_bits(values[i]);
    }
    Device<float> dq(queries), out(queries.size());
    Device<uint16_t> dk(key_bits), dv(value_bits);
    k::attention(dq.p, dk.p, dv.p, out.p, rows, key_count, key_first, query_first, local, stream);
    const auto actual = out.download();
    const auto expected = attention_reference(queries, key_bits, value_bits, key_first, query_first, local);
    close(label, actual, expected, 3e-6, 3e-6);
    if (!local) {
        Device<float> scratch(k::attention_split_scratch_floats(key_count));
        // Each decode sees the same cache, including future rows of the chunk.
        // Its causal result must agree with both the chunk and the F64 oracle.
        for (int row = 0; row < rows; ++row)
            k::attention_split_decode(dq.p + (size_t) row * Q, dk.p, dv.p, out.p + (size_t) row * Q,
                                      key_count, query_first + row, scratch.p, stream);
        const auto split = out.download();
        close(std::string(label) + ", split/F64", split, expected, 3e-6, 3e-6);
        close(std::string(label) + ", split/baseline", split,
              std::vector<double>(actual.begin(), actual.end()), 3e-6, 3e-6);
    } else if (rows == 1 && key_count == k::kWindow) {
        Device<float> scratch(k::attention_local_split_scratch_floats());
        k::attention_local_split_decode(dq.p, dk.p, dv.p, out.p, scratch.p, stream);
        const auto split = out.download();
        close(std::string(label) + ", local split/F64", split, expected, 3e-6, 3e-6);
        close(std::string(label) + ", local split/baseline", split,
              std::vector<double>(actual.begin(), actual.end()), 3e-6, 3e-6);
    }
}

void attention_head_mapping(cudaStream_t stream, bool local = false) {
    const int keys = local ? k::kWindow : 1025;
    const auto queries = random_values(Q, 61);
    const auto keys_f32 = random_values(keys * KV, 62);
    std::vector<uint16_t> key_bits(keys * KV), values(keys * KV);
    std::vector<double> expected(Q);
    for (int token = 0; token < keys; ++token) {
        for (int head = 0; head < k::kKvHeads; ++head) {
            for (int d = 0; d < k::kHeadDim; ++d) {
                const size_t index = (size_t) token * KV + head * k::kHeadDim + d;
                key_bits[index] = half_bits(keys_f32[index]);
                values[index] = half_bits((head + 1) * (d % 2 ? 1.f : -1.f));
            }
        }
    }
    for (int head = 0; head < k::kQueryHeads; ++head)
        for (int d = 0; d < k::kHeadDim; ++d)
            expected[head * k::kHeadDim + d] = (head / 12 + 1) * (d % 2 ? 1. : -1.);
    Device<float> q(queries), output(Q), scratch(local ? k::attention_local_split_scratch_floats()
                                                     : k::attention_split_scratch_floats(keys));
    Device<uint16_t> dk(key_bits), dv(values);
    if (local) k::attention_local_split_decode(q.p, dk.p, dv.p, output.p, scratch.p, stream);
    else k::attention_split_decode(q.p, dk.p, dv.p, output.p, keys, keys - 1, scratch.p, stream);
    close(local ? "Local split, all four GQA head groups" : "Split attention, all four GQA head groups",
          output.download(), expected, 3e-6, 3e-6);
}

void attention_local_tail(cudaStream_t stream) {
    // The fifth partition contains just the current key. Check both uniform
    // weights and an extreme score that makes this tail dominate the result.
    for (bool dominant : {false, true}) {
        std::vector<float> queries(Q, 0.f);
        std::vector<uint16_t> keys(k::kWindow * KV, half_bits(0.f)), values(keys);
        for (int head = 0; head < k::kQueryHeads; ++head)
            queries[head * k::kHeadDim] = dominant ? 1024.f : 0.f;
        for (int head = 0; head < k::kKvHeads; ++head) {
            const size_t offset = (size_t) (k::kWindow - 1) * KV + head * k::kHeadDim;
            keys[offset] = half_bits(32.f);
            for (int d = 0; d < k::kHeadDim; ++d)
                values[offset + d] = half_bits((head + 1) * (d % 2 ? 1.f : -1.f));
        }
        Device<float> q(queries), output(Q), scratch(k::attention_local_split_scratch_floats());
        Device<uint16_t> dk(keys), dv(values);
        const auto expected = attention_reference(queries, keys, values, 0, k::kWindow - 1, true);
        k::attention(q.p, dk.p, dv.p, output.p, 1, k::kWindow, 0, k::kWindow - 1, true, stream);
        close(dominant ? "Local baseline, dominant current key" : "Local baseline, uniform tail weight",
              output.download(), expected, 3e-6, 3e-6);
        k::attention_local_split_decode(q.p, dk.p, dv.p, output.p, scratch.p, stream);
        close(dominant ? "Local split, dominant current key" : "Local split, uniform tail weight",
              output.download(), expected, 3e-6, 3e-6);
    }
}

void attention_merge_q8(cudaStream_t stream) {
    constexpr int partial_width = k::kHeadDim + 2, guard = 32;
    const size_t q8_bytes = strata::kernels::native_q8_1_bytes(Q);
    const std::vector<float> initial(Q + 2 * guard, -12345.f);
    const std::vector<uint8_t> initial_q8(q8_bytes + 2 * guard, 0xa5);
    // These are the first/global, local513, 8K, pre-8K and full262K partition counts.
    for (const int parts : {1, 17, 64, 256, 2048}) {
        auto partials = random_values((size_t) k::kQueryHeads * parts * partial_width, 72 + parts);
        for (int head = 0; head < k::kQueryHeads; ++head) {
            for (int part = 0; part < parts; ++part) {
                float* row = partials.data() + ((size_t) head * parts + part) * partial_width;
                row[k::kHeadDim] = parts == 1 ? 0.f : (float) (part % 19 - 9);
                row[k::kHeadDim + 1] = 1.f + part % 7;
                if (parts != 1) continue;
                // One partial isolates quantizer boundaries after the original merge.
                for (int d = 0; d < k::kHeadDim; ++d) {
                    const int block = (head * k::kHeadDim + d) / 32, lane = d % 32;
                    const int mode = block % 9, variant = block / 9;
                    const float sign = variant % 2 ? -1.f : 1.f;
                    if (mode == 0) row[d] = lane % 2 ? -0.f : 0.f;
                    else if (mode == 1) row[d] = std::ldexp((float) (lane - 16), -147);
                    else if (mode == 2) {
                        const float edge[] = {0x1p-149f, -0x1p-149f, 0x1p-126f, -0x1p-126f,
                                              std::nextafter(0x1p-126f, 0.f), -std::nextafter(0x1p-126f, 0.f)};
                        row[d] = edge[lane % 6];
                    } else if (mode >= 3 && mode <= 5) {
                        row[d] = lane == 31 ? 127.f : (float) (lane - 15) + .5f;
                        if (lane != 31 && mode != 3)
                            row[d] = std::nextafter(row[d], mode == 4 ? -INFINITY : INFINITY);
                    } else if (mode == 6) {
                        const float sums[] = {65504.f / 32.f, 65520.f / 32.f, 65504.f};
                        row[d] = sign * sums[variant % 3];
                    } else if (mode == 7) {
                        const float scales[] = {65504.f * 127.f,
                                                std::nextafter(65504.f * 127.f, INFINITY), 1.e8f};
                        row[d] = sign * scales[variant % 3];
                    }
                }
            }
        }
        Device<float> input(partials), reference(initial), fused(initial);
        Device<uint8_t> reference_q8(initial_q8), fused_q8(initial_q8);
        // The suite's blocking stream orders Device's setup copies on the default stream.
        k::attention_merge_partials(input.p, reference.p + guard, parts, stream);
        strata::kernels::native_quantize_q8_1(reference.p + guard, reference_q8.p + guard, Q, 1, stream);
        k::attention_merge_partials_q8(input.p, fused.p + guard, fused_q8.p + guard, parts, stream);
        check(cudaStreamSynchronize(stream));
        const auto expected = reference.download(), actual = fused.download();
        const auto expected_q8 = reference_q8.download(), actual_q8 = fused_q8.download();
        require(!std::memcmp(expected.data(), actual.data(), initial.size() * sizeof(float)),
                "attention merge Q8 changes F32 output or guards");
        require(expected_q8 == actual_q8, "attention merge Q8 differs from native quantizer bytes or guards");
        for (int i = 0; i < guard; ++i) {
            require(actual[i] == initial[i] && actual[actual.size() - 1 - i] == initial.back(),
                    "attention merge Q8 overwrote F32 guard");
            require(actual_q8[i] == 0xa5 && actual_q8[actual_q8.size() - 1 - i] == 0xa5,
                    "attention merge Q8 overwrote byte guard");
        }
        require(std::all_of(actual.begin() + guard, actual.end() - guard,
                            [](float x) { return std::isfinite(x); }), "finite merge input produced nonfinite output");
        for (size_t block = 0; block < Q / 32; ++block) {
            uint16_t ds[2];
            std::memcpy(ds, actual_q8.data() + guard + block * 36, sizeof(ds));
            require(std::isfinite(half_value(ds[0])) && std::isfinite(half_value(ds[1])),
                    "attention Q8 scale or sum exceeds finite half range");
        }
        std::printf("PASS Attention merge Q8, parts=%d, F32 and all %zu bytes exact, guards\n", parts, q8_bytes);
    }
}

void attention(cudaStream_t stream) {
    attention_case("Local attention, first token", 1, 0, 0, true, stream);
    attention_case("Local attention, positions 511 through 514", 4, 511, 0, true, stream);
    attention_case("Local attention, 256-token boundary chunk", 256, 512, 0, true, stream);
    attention_case("Local attention, retained cache at 513", 4, 1025, 513, true, stream);
    attention_case("Local attention, retained cache near 32K", 4, 32764, 32252, true, stream);
    attention_case("Local decode, first full window", 1, 512, 0, true, stream);
    attention_case("Local decode, shifted window near 32K", 1, 32767, 32255, true, stream);
    attention_case("Local decode, extreme scores", 1, 32767, 32255, true, stream, true);
    attention_case("Global attention, first token", 1, 0, 0, false, stream);
    attention_case("Global attention, 256-token causal chunk", 256, 0, 0, false, stream);
    attention_case("Global attention, positions 511 through 514", 4, 511, 0, false, stream);
    attention_case("Global attention, positions 1023 through 1026", 4, 1023, 0, false, stream);
    attention_case("Global attention, full 32768-token cache", 1, 32767, 0, false, stream);
    attention_case("Global attention, large positive/negative scores", 4, 40, 0, false, stream, true);
    attention_case("Global attention, split extreme scores", 4, 1023, 0, false, stream, true);
    attention_head_mapping(stream);
    attention_head_mapping(stream, true);
    attention_local_tail(stream);

    bool rejected = false;
    try { k::attention(nullptr, nullptr, nullptr, nullptr, 1, 512, 2, 513, true, stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "local attention accepted missing required history");
    rejected = false;
    try { k::attention(nullptr, nullptr, nullptr, nullptr, 2, 1, 0, 0, false, stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "attention accepted a missing current key");
    rejected = false;
    Device<float> scratch(k::attention_split_scratch_floats(512));
    try { k::attention_split_decode(nullptr, nullptr, nullptr, nullptr, 512, 512, scratch.p, stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "split attention accepted a missing current key");
    rejected = false;
    try { k::attention_local_split_decode(nullptr, nullptr, nullptr, nullptr, nullptr, stream); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "local split attention accepted missing scratch");
    std::puts("PASS Attention rejects incomplete causal cache");
}

// Optional synthetic timing, separate from the mathematical checks. Time both
// paths on the same resident inputs; each sample averages repeated CUDA events.
void attention_benchmark(cudaStream_t stream, bool local = false) {
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    const std::vector<int> counts = local ? std::vector<int>{k::kWindow}
                                         : std::vector<int>{1, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    for (int count : counts) {
        const auto queries = random_values(Q, 71);
        const auto values = random_values((size_t) count * KV, 72);
        std::vector<uint16_t> bits(values.size());
        for (size_t i = 0; i < bits.size(); ++i) bits[i] = half_bits(values[i]);
        Device<float> q(queries), output(Q), scratch(local ? k::attention_local_split_scratch_floats()
                                                         : k::attention_split_scratch_floats(count));
        Device<uint16_t> kv(bits);
        auto run = [&](bool split) {
            if (split && local) k::attention_local_split_decode(q.p, kv.p, kv.p, output.p, scratch.p, stream);
            else if (split) k::attention_split_decode(q.p, kv.p, kv.p, output.p, count, count - 1, scratch.p, stream);
            else k::attention(q.p, kv.p, kv.p, output.p, 1, count, 0, count - 1, local, stream);
        };
        for (int i = 0; i < 5; ++i) { run(false); run(true); }
        check(cudaStreamSynchronize(stream));
        constexpr int repeats = 20;
        for (int sample = 0; sample < 5; ++sample) {
            float milliseconds[2];
            for (int phase = 0; phase < 2; ++phase) {
                const bool split = (phase + sample) % 2;
                check(cudaEventRecord(begin, stream));
                for (int i = 0; i < repeats; ++i) run(split);
                check(cudaEventRecord(end, stream));
                check(cudaEventSynchronize(end));
                check(cudaEventElapsedTime(&milliseconds[split], begin, end));
            }
            std::printf("%s keys=%d sample=%d repeats=%d baseline_us=%.3f split_us=%.3f speedup=%.3f\n",
                         local ? "LOCAL_ATTENTION_BENCH" : "ATTENTION_BENCH", count, sample, repeats, milliseconds[0] * 1000 / repeats,
                         milliseconds[1] * 1000 / repeats, milliseconds[0] / milliseconds[1]);
        }
    }
    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));
}
}  // namespace

int main(int argc, char** argv) {
    const bool local_benchmark = argc == 2 && std::string(argv[1]) == "--bench-local-attention";
    const bool benchmark = local_benchmark || (argc == 2 && std::string(argv[1]) == "--bench-attention");
    if (argc > 1 && !benchmark) {
        std::fprintf(stderr, "usage: kolibri-math-test [--bench-attention|--bench-local-attention]\n");
        return 1;
    }
    int devices = 0;
    const auto error = cudaGetDeviceCount(&devices);
    if (error != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "SKIP Kolibri CUDA math: no usable CUDA device\n");
        return 77;
    }
    cudaStream_t stream = nullptr;
    try {
        check(cudaStreamCreate(&stream));
        if (benchmark) attention_benchmark(stream, local_benchmark);
        else {
            norm_and_helpers(stream);
            qk_norm_rope(stream);
            qk_cached(stream);
            qk_append_decode(stream);
            int8_cache(stream);
            append(stream);
            routing(stream);
            route_cache(stream);
            attention(stream);
            attention_merge_q8(stream);
        }
        check(cudaStreamSynchronize(stream));
        check(cudaStreamDestroy(stream));
        std::puts(benchmark ? "DONE Kolibri attention benchmark" : "PASS all Kolibri CUDA mathematical checks");
        return 0;
    } catch (const std::exception& e) {
        cudaStreamDestroy(stream);
        std::fprintf(stderr, "FAIL Kolibri CUDA math: %s\n", e.what());
        return 1;
    }
}
