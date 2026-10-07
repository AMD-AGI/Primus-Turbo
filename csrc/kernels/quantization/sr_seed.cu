/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Seeds of the stochastic-rounding quantizers. Every SR launch takes the next seed of its
// quantizer's stream: splitmix64 of the base seed mixed with the stream id and the stream's launch
// counter. Setting the base (set_sr_seed, called by the trainer at the start of every step with a
// hash of the run seed, the global rank and the iteration) resets every counter, so the SR bits of
// a step are a function of (base, launch index within the step) alone: different per rank and per
// run seed, and the same again when a run resumes at that step. The counters live on the host, so
// the seed is a kernel argument: a captured CUDA graph would freeze it (and replay the same SR
// bits).

#include <atomic>
#include <cstdint>

#include "primus_turbo/quantization.h"

namespace primus_turbo {

namespace {
std::atomic<uint64_t> g_sr_base{0};
std::atomic<uint64_t> g_sr_counter[static_cast<int>(SRStream::kCount)];
// One-shot seed for the next launch of a stream (sr_override_next_seed): a caller that needs the SR bits of one pack
// to be a function of its own key (e.g. a weight packed on whichever rank owns it) takes it without moving the
// stream's counter, so the other launches keep their seeds.
std::atomic<int64_t> g_sr_override[static_cast<int>(SRStream::kCount)];

struct InitOverride {
    InitOverride() {
        for (auto &o : g_sr_override)
            o.store(-1, std::memory_order_relaxed);
    }
} g_init_override;

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
} // namespace

void sr_set_base_seed(const uint64_t base) {
    g_sr_base.store(base, std::memory_order_relaxed);
    for (auto &c : g_sr_counter)
        c.store(0, std::memory_order_relaxed);
}

void sr_override_next_seed(const SRStream stream, const uint32_t seed) {
    g_sr_override[static_cast<int>(stream)].store(static_cast<int64_t>(seed), std::memory_order_relaxed);
}

uint32_t sr_next_seed(const SRStream stream) {
    const int64_t ov = g_sr_override[static_cast<int>(stream)].exchange(-1, std::memory_order_relaxed);
    if (ov >= 0)
        return static_cast<uint32_t>(ov);
    const uint64_t k =
        g_sr_counter[static_cast<int>(stream)].fetch_add(1, std::memory_order_relaxed);
    const uint64_t s = splitmix64(k ^ (static_cast<uint64_t>(stream) << 56));
    return static_cast<uint32_t>(splitmix64(g_sr_base.load(std::memory_order_relaxed) ^ s) >> 32);
}

} // namespace primus_turbo
