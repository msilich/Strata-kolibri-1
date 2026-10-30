// Device-owned expert selection and cache receipts for optional Kolibri decode.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kolibri::decode_cache {
constexpr int Experts = 384, Selected = 6;
enum Status : int32_t { Ok = 0, InvalidExpert = 1, DuplicateExpert = 2 };
enum MutationKind : int32_t { Admit = 1, Replace = 2 };
struct Counters { uint64_t tick, hits, misses; };
struct LayerState {
    int32_t slot[Experts];
    uint64_t age[Experts];
    int32_t next_free;
};
// Fixed per-layer cache; no shrink, sealed cache, cross-layer admission or loan.
// Initial slots form [0,next_free); replacements preserve that invariant.
struct LayerView {
    LayerState* state;
    uint8_t* cache_base;
    uint8_t* miss_base;
    size_t cache_stride, miss_stride;
    int32_t slots;
};
struct Mutation { int32_t kind, old_expert, new_expert, slot; };
// Drain the 50 receipts in forward order at token completion. Protect ids at
// counters.tick, replay admit/replace in order, and mark replaced victims at
// that tick. This also restores CPU recency/counters without copying age[384].
struct Receipt {
    int32_t status, copy_count, mutation_count;
    int32_t ids[Selected];
    Mutation mutation[Selected];
    Counters counters;
};
struct Copy { int32_t expert; uint8_t* destination; };
struct Plan {
    Receipt receipt;
    int32_t ranks[Selected], slots[Selected];
    uint8_t* weights[Selected];
    Copy copies[Selected];
};
struct Mirror {
    const uint8_t* role[3];
    size_t bytes[3], offset[3], stride;
};
// One stream owns all state. Invalid route IDs leave state/counters untouched.
// Caller supplies six IDs in router rank order. Pointers remain device-owned.
void plan(const int32_t* routes, LayerView view, Counters* counters, Plan* output, void* stream, Receipt* receipt = nullptr);
// One decode row: route_decode arithmetic followed by the same cache plan.
// The caller fences the router GEMM before this stream and consumers afterward.
void route_plan(const float* logits, const float* bias, int32_t* routes, float* weights,
                LayerView view, Counters* counters, Plan* output, void* stream, Receipt* receipt = nullptr);
// Coalesced uint4 copies consume device-owned plans.
// Role sizes/offsets/stride are multiples of16. Each copy also zeros role tails:
// the shared miss arena can contain a preceding layer with a different layout.
void copy(Mirror mirror, const Plan* plan, void* stream);
}  // namespace strata::kolibri::decode_cache
