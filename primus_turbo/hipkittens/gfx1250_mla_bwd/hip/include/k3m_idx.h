/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k3m_idx.h -- the masked-ring index math of k_dkdv96ds (X3: the diagonal (masked) iterations go
// through the same 3-stage Q/dO TDM ring as the full loop, prefetch distance 2). constexpr
// __host__ __device__ functions of scalars; the kernel calls exactly these and the CPU replay
// (scratch x3 tools/ring_replay.py) compiles this file for the host.
//
// One flat tile sequence per workgroup: j = 0 .. N-1, N = n_mask + n_full = G * nqp_eff, tile T_j =
// (query pair qp_start + j / G, q head j % G); j < n_mask are the masked tiles (they straddle the
// diagonal), the rest are the full loop's. Tile j lives in stage sigma(j) = (j - n_mask) mod 3, so
// the full loop's first tile (j = n_mask) lands in stage 0 as the full loop's code assumes.
#pragma once
#include "idx.h"

#define K3M_FN static inline constexpr __host__ __device__

// byte offset of stage sigma(0) = ((-n_mask) mod 3) * K1_QDO_B (n_mask >= 0)
K3M_FN int k3m_st0(int n_mask) {
    const int r = n_mask % K1_DEPTH;
    return r == 0 ? 0 : (K1_DEPTH - r) * K1_QDO_B;
}
