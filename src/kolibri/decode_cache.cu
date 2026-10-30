#include "decode_cache.cuh"
#include "decode_route.cuh"
#include <cuda_runtime.h>

namespace strata::kolibri::decode_cache {
__device__ __forceinline__ void plan_body(const int32_t* routes, LayerView view, Counters* counters, Plan* output, Receipt* receipt) {
    constexpr unsigned All = 0xffffffffu;
    const int lane = threadIdx.x;
    __shared__ Plan result;
    __shared__ int slot, miss, needs_victim;
    if (!lane) result = Plan{};
    __syncwarp();
    const int id = lane < Selected ? routes[lane] : 0;
    const unsigned invalid = __ballot_sync(All, lane < Selected && (id < 0 || id >= Experts));
    if (invalid) {
        if (!lane) { result.receipt.status = InvalidExpert; *output = result; if (receipt) *receipt = result.receipt; }
        return;
    }
    if (lane < Selected) { result.receipt.ids[lane] = id; result.ranks[lane] = lane; }
    __syncwarp();
    if (!lane) {
        for (int i = 1; i < Selected; ++i) {
            const int expert = result.receipt.ids[i], rank = result.ranks[i];
            int j = i;
            while (j && result.receipt.ids[j - 1] > expert) {
                result.receipt.ids[j] = result.receipt.ids[j - 1]; result.ranks[j] = result.ranks[j - 1]; --j;
            }
            result.receipt.ids[j] = expert; result.ranks[j] = rank;
        }
        for (int i = 1; i < Selected; ++i) if (result.receipt.ids[i] == result.receipt.ids[i - 1]) {
            result = Plan{}; result.receipt.status = DuplicateExpert; break;
        }
    }
    __syncwarp();
    if (result.receipt.status != Ok) { if (!lane) { *output = result; if (receipt) *receipt = result.receipt; } return; }
    auto& state = *view.state;
    if (!lane) ++counters->tick;
    __syncwarp();
    const uint64_t tick = counters->tick;
    if (lane < Selected) state.age[result.receipt.ids[lane]] = tick;
    __syncwarp();
    const int resident = lane < Selected ? state.slot[result.receipt.ids[lane]] : -1;
    const unsigned hits = __ballot_sync(All, lane < Selected && resident >= 0);
    if (hits == (1u << Selected) - 1) {
        if (lane < Selected) {
            result.slots[lane] = resident;
            result.weights[lane] = view.cache_base + size_t(resident) * view.cache_stride;
        }
        if (!lane) { counters->hits += Selected; result.receipt.counters = *counters; }
        __syncwarp();
        if (!lane) { *output = result; if (receipt) *receipt = result.receipt; }
        return;
    }
    for (int i = 0; i < Selected; ++i) {
        const int expert = result.receipt.ids[i];
        if (!lane) {
            slot = state.slot[expert]; miss = slot < 0; needs_victim = 0;
            if (!miss) ++counters->hits;
            else {
                ++counters->misses;
                if (state.next_free < view.slots) {
                    slot = state.next_free++;
                    state.slot[expert] = slot;
                    result.receipt.mutation[result.receipt.mutation_count++] = {Admit, -1, expert, slot};
                } else needs_victim = view.slots > 0;
            }
        }
        __syncwarp();
        if (needs_victim) {
            uint64_t oldest = tick;
            int victim = Experts;
            for (int candidate = lane; candidate < Experts; candidate += 32) {
                const uint64_t age = state.age[candidate];
                if (state.slot[candidate] >= 0 && age < tick &&
                    (age < oldest || (age == oldest && candidate < victim))) { oldest = age; victim = candidate; }
            }
            for (int offset = 16; offset; offset >>= 1) {
                const uint64_t other_age = __shfl_down_sync(All, (unsigned long long)oldest, offset);
                const int other_id = __shfl_down_sync(All, victim, offset);
                if (other_age < oldest || (other_age == oldest && other_id < victim)) {
                    oldest = other_age; victim = other_id;
                }
            }
            if (!lane && victim < Experts) {
                slot = state.slot[victim];
                state.age[victim] = tick;
                result.receipt.mutation[result.receipt.mutation_count++] = {Replace, victim, expert, slot};
            }
        }
        __syncwarp();
        if (!lane) {
            uint8_t* destination = slot >= 0 ? view.cache_base + size_t(slot) * view.cache_stride
                                            : view.miss_base + size_t(i) * view.miss_stride;
            if (miss) result.copies[result.receipt.copy_count++] = {expert, destination};
            result.slots[i] = slot; result.weights[i] = destination;
        }
        __syncwarp();
    }
    if (!lane) {
        // Same delayed publication as Executor::experts_decode. The following
        // copy and consumers share this stream; host replay waits for completion.
        for (int i = 0; i < result.receipt.mutation_count; ++i) {
            const auto mutation = result.receipt.mutation[i];
            if (mutation.kind == Replace) {
                state.slot[mutation.new_expert] = state.slot[mutation.old_expert];
                state.slot[mutation.old_expert] = -1;
            }
        }
        result.receipt.counters = *counters;
        *output = result; if (receipt) *receipt = result.receipt;
    }
}

__global__ void plan_kernel(const int32_t* routes, LayerView view, Counters* counters, Plan* output, Receipt* receipt) {
    plan_body(routes, view, counters, output, receipt);
}

__global__ void route_plan_kernel(const float* logits, const float* bias, int32_t* routes, float* weights,
                                  LayerView view, Counters* counters, Plan* output, Receipt* receipt) {
    kernels::detail::route_decode_body(logits, bias, routes, weights);
    __syncwarp();
    plan_body(routes, view, counters, output, receipt);
}

__global__ void copy_kernel(Mirror mirror, const Plan* plan) {
    const unsigned count = plan->receipt.copy_count;
    if (!count || plan->receipt.status != Ok) return;
    unsigned tiles[3];
    size_t span[3];
#pragma unroll
    for (int role = 0; role < 3; ++role) {
        span[role] = (role == 2 ? mirror.stride : mirror.offset[role + 1]) - mirror.offset[role];
        tiles[role] = unsigned((span[role] + 4095) / 4096);
    }
    const unsigned per_expert = tiles[0] + tiles[1] + tiles[2];
    for (unsigned work = blockIdx.x; work < count * per_expert; work += gridDim.x) {
        const unsigned miss = work / per_expert;
        unsigned tile = work - miss * per_expert, role = 0;
        if (tile >= tiles[0]) { tile -= tiles[0]; role = 1;
            if (tile >= tiles[1]) { tile -= tiles[1]; role = 2; }
        }
        const size_t i = size_t(tile) * blockDim.x + threadIdx.x;
        if (i < span[role] / sizeof(uint4)) {
            const auto descriptor = plan->copies[miss];
            const auto* source = reinterpret_cast<const uint4*>(mirror.role[role] +
                size_t(descriptor.expert) * mirror.bytes[role]);
            auto* destination = reinterpret_cast<uint4*>(descriptor.destination + mirror.offset[role]);
            destination[i] = i < mirror.bytes[role] / sizeof(uint4) ? source[i] : make_uint4(0, 0, 0, 0);
        }
    }
}
void plan(const int32_t* routes, LayerView view, Counters* counters, Plan* output, void* stream, Receipt* receipt) {
    plan_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(routes, view, counters, output, receipt);
}
void route_plan(const float* logits, const float* bias, int32_t* routes, float* weights,
                LayerView view, Counters* counters, Plan* output, void* stream, Receipt* receipt) {
    route_plan_kernel<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, bias, routes, weights, view, counters, output, receipt);
}
void copy(Mirror mirror, const Plan* plan, void* stream) {
    copy_kernel<<<64, 256, 0, static_cast<cudaStream_t>(stream)>>>(mirror, plan);
}
}  // namespace strata::kolibri::decode_cache
