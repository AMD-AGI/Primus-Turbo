/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk_attn/wmma.h -- v_wmma_f32_16x16x32_bf16 (llvm.amdgcn.wmma.f32.16x16x32.bf16, neg_a/neg_b 0,
// c_mod 0), the only matrix instruction of the attention kernels.
//
//   D[m][n] = sum_{k < 32} A[m][k] * B[k][n] + C[m][n]      (fp32 accumulate of bf16 products)
//   A: lane (r, h) element e = A[r][op_k(l, e)]        B: lane (r, h) element e = B[op_k(l, e)][r]
//   C, D: lane (r, h) element i = C[8h + i][r]         op_k(l, e) = (e / 8) * 16 + 8h + e % 8
// (layout.h; checked bit-exactly by t_prims t_wmma / t_ld_operands).
//
// Attention orientations used by the kernels (A x B -> C, lanes index the C column):
//   fwd  S^T = K Q^T    A = K rows (kv x d)      B = Q rows as B^T (q x d)   C lane = q, rows = kv
//   fwd  O^T += V^T P^T A = V^T (tr16 of V)      B = pack_k32(S^T tiles)     C lane = q, rows = d
//   bwd  S = Q K^T      A = Q rows (q x d)       B = K rows as B^T (kv x d)  C lane = kv, rows = q
//   bwd  dV^T += dO^T P A = dO^T (tr16 of dO)    B = pack_k32(S tiles)       C lane = kv, rows = d
//   bwd  dQ = dS K      A = dS (tr16 of dS^T)    B = K (tr16 of K)           C lane = d, rows = q
//
// Reuse hints (matrix_a_reuse / matrix_b_reuse, instruction OPSEL bits). Convention of the proven
// REF k_dkdv96ds (dV/dK runs, reuse_b for d > 0) and FlyDSL's fused bwd (reuseA = dtile > 0): set
// REUSE_A (REUSE_B) from the second WMMA of a run with a fixed A (B) operand on. UNVERIFIED: the
// bit's direction (same operand as the PREVIOUS WMMA, or kept for the NEXT one) and whether the
// hardware trusts it or checks the registers. LLVM does not keep hinted runs adjacent: without
// fences the scheduler interleaved both t_wmma runs (every reuse_b and the last reuse_a WMMA followed
// one with different operand registers); dsv3 p2_x12 saw ~200 such reordered hints with bitwise
// identical results. t_wmma now fences the order: c4 adjacent (each hinted WMMA follows one with the
// same B) and c5 interleaved with c0..c3 (each follows one with a different A); test_prims's ISA
// check isa_reuse_hints requires both patterns. No measured speed benefit on a pure WMMA stream
// (p1_research: 8.00 vs 8.00 cycles), so kernels keep REUSE_A = REUSE_B = false at least until the
// GPU t_wmma run passes (and then list their hinted WMMAs with test_prims.reuse_hint_pairs).
#pragma once
#include "types.h"

namespace hk_attn {

template <bool REUSE_A = false, bool REUSE_B = false>
HKA_DEV acc_t wmma(opnd_t a, opnd_t b, acc_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, (short) 0, c, REUSE_A,
                                                   REUSE_B);
}

// D = A B with C = 0 (the zero accumulator is an inline constant: `v_wmma ..., 0`)
template <bool REUSE_A = false, bool REUSE_B = false>
HKA_DEV acc_t wmma0(opnd_t a, opnd_t b) {
    return wmma<REUSE_A, REUSE_B>(a, b, zero_acc());
}

} // namespace hk_attn
