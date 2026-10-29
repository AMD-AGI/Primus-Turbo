/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_dqds_c -- bwd_hk_dqds2: the recompute-free dQ kernel of the S1 line (dQ = dS K from the S1 scratch, dslayout v2)
// with 384-query workgroups: 4 waves (1 per SIMD, one workgroup per CU, LDS 274432 B) share ONE K ring (K bytes per
// call / 2 vs bwd_hk_dqds's k_dqds_b); role-split TDM issue (waves 0 / 3 the dS of kv32 rows 2u / 2u + 1 for all 12 q
// pairs, waves 1 / 2 the two 32-row K halves: one op per wave per 64-kv unit, the two TDM engines balanced), K and dS
// rings 3 stages each (every op 2 units ahead), <= 2 TDM ops in flight per wave, barrier every 64 kv.
// grid (HG, ntile + nhead, B*Hq/HG), block 128 = 4 waves: grid.y < ntile a 384-query tile of [q_off, Sq), grid.y ==
// ntile (q_off == 256) the head [0, 256) (uniform branch, bwd_hk_q2 (b)). Body: include/dqds/dqds_impl.h; index math
// idx.h.
#include "dqds/dqds_impl.h"

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128)))
void k_dqds_c(const __bf16* DS, const __bf16* K, __bf16* DQ, int Sq, int Skv, int Hq, int Hkv, int G, int nkvt,
              int cshift, int causal, int q_off, int ntile, int HG) {
  dq::dqds_kernel(DS, K, DQ, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, HG);
}
