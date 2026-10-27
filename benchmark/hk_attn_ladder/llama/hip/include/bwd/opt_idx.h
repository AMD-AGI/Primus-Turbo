/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// bwd/opt_idx.h -- index formulas and protocol constants of the HK_BWD_OPT_* variants of
// k_hk_bwd_g4 (hip/src/k_attn_bwd_v0.hip; hk/notes/bwd_v0_opt.md), on top of bwd/idx.h (b0::).
// Host-compilable like bwd/idx.h: the bounds checker hk/lab/bounds/bwd_v0_check.cpp replays these.
// Only the v0 kernel source (variant builds) and that checker include this file; the default v0
// build does not depend on it.
#pragma once
#include "idx.h"

namespace b0opt {

// ---- HK_BWD_OPT_TDM3: wave 3 issues one REAL 16-B TDM op per step instead of skipping the ring op
// behind a role branch (fwd v0 style: constant per-wave tensorcnt, no branch). The op moves the 8
// bf16 of K row kvw0 of head hkv (the WG's own first K row: always inside K, read-only) into wave
// 3's own 16-B LDS cell DUMMY_OFF, which nothing reads. DUMMY_OFF is the first byte after the LD
// ring's RING_MAX reserved slots: segment 0, outside every window of the v0 LDS map (dO ring, LD
// ring, Q ring, dS^T, K / V images; checked below and replayed by the bounds checker). Tensor dims
// == tile dims (8 x 1), row stride 0 (one row).
constexpr int DUMMY_OFF  = b0::OFF_LD + b0::RING_MAX * b0::LD_B; // 44800
constexpr int DUMMY_COLS = 8;                                    // bf16
constexpr int DUMMY_B    = 2 * DUMMY_COLS;                       // 16
static_assert(DUMMY_OFF % 16 == 0 && DUMMY_OFF + DUMMY_B <= b0::OFF_Q,
              "dummy cell: segment 0, below the Q ring");
static_assert(DUMMY_OFF >= b0::OFF_LD + b0::RING_MAX * b0::LD_B, "dummy cell above the LD ring");
B0_HD int64_t dummy_src_byte(int b, int Skv, int Hkv, int hkv, int kvw0) {
    return b0::kv_tile_byte(b, Skv, Hkv, hkv, kvw0);
}

// ---- HK_BWD_OPT_ONEPHASE: one memory phase per step. The step's signal leaves at most
// ONEPHASE_UNCOVERED of the 32 E-operand tr16 in flight, so that the readback issued right after
// the wait (40 b128 + 16 tr16) never has more than 63 DS ops outstanding (6-bit DS counter).
constexpr int READBACK_DS =
    2 * b0::NDT * 2 * 2 + 2 * 4 + 2 * b0::NKC * 2;   // qA / dA 32, LD 8, dSa 16 = 56
constexpr int ONEPHASE_UNCOVERED = 63 - READBACK_DS; // 7
static_assert(READBACK_DS == 56 && ONEPHASE_UNCOVERED >= 0 &&
                  ONEPHASE_UNCOVERED <= b0::FULL_UNCOVERED,
              "one-phase DS budget");

} // namespace b0opt
