#include "decode_norms.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kolibri::kernels;
namespace {
constexpr int H = 2560;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Device {
    float* p = nullptr;
    Device() { check(cudaMalloc(&p, H * sizeof(float))); }
    ~Device() { cudaFree(p); }
    void upload(const std::vector<float>& values) {
        check(cudaMemcpy(p, values.data(), H * sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<float> download() const {
        std::vector<float> result(H);
        check(cudaMemcpy(result.data(), p, H * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    }
};
void same(const std::vector<float>& a, const std::vector<float>& b, const char* label, int trial) {
    for (int i = 0; i < H; ++i) {
        uint32_t aa, bb;
        std::memcpy(&aa, &a[i], sizeof(aa));
        std::memcpy(&bb, &b[i], sizeof(bb));
        if (aa != bb)
            throw std::runtime_error(std::string(label) + " trial " + std::to_string(trial) +
                " index " + std::to_string(i) + " expected bits " + std::to_string(aa) +
                " got " + std::to_string(bb));
    }
}
}  // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        Device update, post, next, reference, candidate, temporary, reference_norm, candidate_norm;
        std::mt19937 random(20261029);
        std::uniform_real_distribution<float> uniform(-1.f, 1.f);
        for (int trial = 0; trial < 36; ++trial) {
            std::vector<float> u(H), x(H), w(H), n(H);
            for (int i = 0; i < H; ++i) {
                u[i] = uniform(random) * std::ldexp(1.f, (trial % 9 - 4) * 8);
                x[i] = uniform(random) * std::ldexp(1.f, (trial % 7 - 3) * 5);
                w[i] = uniform(random) * 2.f;
                n[i] = uniform(random) * 2.f;
                if (trial == 0) u[i] = x[i] = 0.f;
                if (trial == 1) { u[i] = (i & 1) ? 1.f : -1.f; x[i] = -u[i]; w[i] = 1.f; }
                if (trial == 2) { u[i] = std::ldexp((float) (i % 17 - 8), -147); x[i] = -u[i]; }
                if (trial == 3) { u[i] = (i & 1) ? -0.f : 0.f; x[i] = u[i]; }
                if (trial == 4 && i % 13 == 0) u[i] = std::nextafter(u[i], 0.f);
            }
            update.upload(u); post.upload(w); next.upload(n);
            reference.upload(x); candidate.upload(x);
            const float eps = (trial & 1) ? 1.e-5f : 1.e-6f;
            K::rms_norm(update.p, post.p, temporary.p, 1, H, eps, nullptr);
            K::add(reference.p, temporary.p, reference.p, H, nullptr);
            K::rms_norm(reference.p, next.p, reference_norm.p, 1, H, eps, nullptr);
            K::decode_residual_norm(update.p, post.p, candidate.p, next.p, candidate_norm.p, eps, nullptr);
            check(cudaDeviceSynchronize());
            same(reference.download(), candidate.download(), "residual", trial);
            same(reference_norm.download(), candidate_norm.download(), "normalized", trial);
        }
        std::puts("PASS: fused decode residual and next RMS are bitwise equal in 36 cases");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
