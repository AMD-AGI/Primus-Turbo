/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_dqds_c -- dQ of the DeepSeek-V3 MLA attention backward on gfx1250: dQ = dS K, with dS read back from the dS
// scratch k_dkdv96ds wrote (dslayout.h, scale included), so S and dP are not recomputed. K [B, Skv, Hkv, 192],
// DQ [B, Sq, Hq, 192] bf16.
// grid (HG, ntile + nhead, B*Hq/HG), block 128 = 4 waves (1 per SIMD, one workgroup per CU, LDS 274432 B):
// grid.y < ntile a 384-query tile of [q_off, Sq), longest first; grid.y == ntile (q_off == 256 only) the head
// [0, 256) as one 4-wave x 64-query tile (a uniform branch). The 4 waves share one K ring; TDM issue is split by
// role, one op per wave per 64-kv unit (waves 0 / 3: the dS of kv32 rows 2u / 2u + 1 for all q pairs of the tile,
// waves 1 / 2: the two 32-row K halves); K and dS rings of 3 stages (every op 2 units ahead), <= 2 TDM ops in
// flight per wave, one barrier per 64 kv. Body: include/dq_kernel.h; index math: include/dq_idx.h.
#include "dq_kernel.h"

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128)))
void k_dqds_c(const __bf16* DS, const __bf16* K, __bf16* DQ, int Sq, int Skv, int Hq, int Hkv, int G, int nkvt,
              int cshift, int causal, int q_off, int ntile, int HG) {
  dq::dqds_kernel(DS, K, DQ, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, q_off, ntile, HG);
}
