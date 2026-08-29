#pragma once

// [agent modifed]: new file -- Turbo-only runtime switches. Nothing upstream
// lives here, so the legacy sources stay comparable with a plain diff.

#include <cstdlib>
#include <string>

namespace primus_turbo::deep_ep {

// The cheap fence replaces the release store's system-scope write-back with a per-wave
// s_waitcnt (st_release_sys_global in legacy/utils.cuh). That waitcnt covers only the wave
// publishing the channel tail, while the payload rows are written by the other waves of the
// group, and the relaxed store drops the write-back the receiver needs -- so a receiver can
// see the new tail while those rows still sit in the sender's L2 and read the previous
// round's contents. Measured on gfx950 (EP=8, DSv3, intranode dispatch): 7 of 8 short runs
// carried 1-6 corrupted rows out of 8192, and 50-iter bf16 loss landed at 6.2066 against a
// 4.5206 DeepEP-off reference. The corruption is silent -- it lands in gradients -- so it
// must never be the default. The plain release store costs 2.7% more per step; set
// PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE=0 to take the cheap fence back with its hazard.
inline bool is_enable_cheap_fence() {
    static const bool enabled = []() {
        const char* v = std::getenv("PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE");
        if (v == nullptr or v[0] == '\0')
            return false;
        return std::atoi(v) == 0;
    }();
    return enabled;
}

}  // namespace primus_turbo::deep_ep
