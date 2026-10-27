/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_attn_fwd_pp -- gfx1250 d128 GQA flash-attention forward, 8 waves (2 per SIMD) with a staggered ping-pong of the
// lo (waves 0-3) and hi (waves 4-7) groups (hk/notes/fwd_pp_design.md). Body: include/fwdpp/body.h; index math:
// include/fwdpp/idx.h (replayed on the host by hk/lab/bounds/fwd_pp_check.cpp).
// grid (B * Hq/4 * Sq/64, 1, 1) longest-first, block 256, static LDS 327680 B (one WG per CU). Kernarg layout
// (104 B, no hidden arguments): Q K V O LSE (ptr), o_bytes lse_bytes (i64, TRUE byte extents), B Sq Skv Hq Hkv G
// nqb nhq4 nkv_st cst causal (i32; host-derived: nqb = Sq/64, nhq4 = Hq/4, nkv_st = Skv/64, cst = (Skv-Sq)/64),
// c2 = softmax_scale * log2(e) (f32). Host contract (fwd_pp.py): G % 4 == 0, Sq % 64 == 0, Skv % 64 == 0, causal ->
// Sq <= Skv, 2 B Sq Hq 128 < 2^31.
#include "fwdpp/body.h"

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void k_attn_fwd_pp(const __bf16 *Q, const __bf16 *K, const __bf16 *V, __bf16 *O, float *LSE, long long o_bytes,
                   long long lse_bytes, int B, int Sq, int Skv, int Hq, int Hkv, int G, int nqb, int nhq4, int nkv_st,
                   int cst, int causal, float c2) {
  const fwdpp::Args a{Q, K, V, O, LSE, o_bytes, lse_bytes, B, Sq, Skv, Hq, Hkv, G, nqb, nhq4, nkv_st, cst, causal, c2};
  fwdpp::kernel(a);
}
