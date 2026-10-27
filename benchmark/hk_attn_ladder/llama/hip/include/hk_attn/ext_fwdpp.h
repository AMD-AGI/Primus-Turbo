/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/ext_fwdpp.h -- the few primitives k_attn_fwd_pp needs beyond prims.h (owner: fwd-pp).
// Each lowers to the instruction named in its comment; hip/tests/t_ext_fwdpp.hip +
// hk/lab/test_ext_fwdpp.py check them.
//
//   simd_id()          s_getreg_b32 hwreg(HW_REG_WAVE_HW_ID1) bits 9:8 (mb_xcd: the wave-level SIMD
//   field) bar_window_open()  sched_barrier + s_wait_dscnt 0 + s_barrier_signal -1 + IR fence +
//   sched_barrier: the
//                      signal half of a barrier whose window (until bar_window_close) carries a DS
//                      burst
//   bar_window_close() sched_barrier + s_barrier_wait + IR fence + sched_barrier
//   any_lane(p)        v_cmp + s_ballot -> wave-uniform bool (one SGPR test, uniform s_cbranch)
//   log2_hw(x)         v_log_f32 (LSE)
#pragma once
#include "sync.h"
#include "types.h"

namespace hk_attn {

// SIMD (0..3) of the calling wave: HW_REG_WAVE_HW_ID1 (id 23) bits 9:8 (s_getreg simm16 = (size-1)
// << 11 | offset << 6 | id); mb_xcd measured these as the bits that differ between the waves of a
// 4-wave workgroup
HKA_DEV int simd_id() {
    return (int) ((__builtin_amdgcn_s_getreg((1 << 11) | (8 << 6) | 23)) & 3u);
}
// the wave's slot field (bits 4:0) of HW_ID1, for the SIMD probe only
HKA_DEV int wave_slot_id() {
    return (int) (__builtin_amdgcn_s_getreg((4 << 11) | (0 << 6) | 23) & 31u);
}

// barrier window: signal (after this wave's LDS accesses complete), then DS loads that read data
// published at an EARLIER barrier (they are covered by this wave's NEXT signal), then the wait. The
// IR fence keeps the window's loads from being hoisted above the signal (where the release's dscnt
// 0 would wait for them).
HKA_DEV void bar_window_open() {
    sched_barrier<0>();
    lds_release();
    barrier_signal();
    compiler_fence();
    sched_barrier<0>();
}
template <int TW> HKA_DEV void bar_window_open_tdm() {
    sched_barrier<0>();
    wait_tensorcnt<TW>();
    lds_release();
    barrier_signal();
    compiler_fence();
    sched_barrier<0>();
}
HKA_DEV void bar_window_close() {
    sched_barrier<0>();
    barrier_wait();
    compiler_fence();
    sched_barrier<0>();
}

// wave-uniform OR of a per-lane predicate (v_cmp writes an SGPR mask; != 0 is an SALU test)
HKA_DEV bool any_lane(bool p) {
    return __builtin_amdgcn_ballot_w32(p) != 0u;
}
// v_log_f32 (log2, no denormal handling; LSE of a positive normal row sum)
HKA_DEV float log2_hw(float x) {
    return __builtin_amdgcn_logf(x);
}

} // namespace hk_attn
