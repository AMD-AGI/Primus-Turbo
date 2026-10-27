/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// dslayout.h -- the S1 dS^T scratch contract, version 1.
//
// Owner: bwd_hk_k1ds (writer, k_dkdv64ds). Consumer: bwd_hk_dqds (reader, k_dqds), which carries a
// BYTE-IDENTICAL copy of this file and checks its sha256 against the writer's. Any change is a new
// version (DSL_VERSION), agreed by both sides before either builds against it.
//
// WHAT is stored: exactly the bf16 values that feed k_dkdv64's dK WMMA (dK^T += Q^T dS, the a_dsk B
// operand):
//     dS[q][kv] = bf16( P[q][kv] * fma(dP[q][kv], scale, -scale * delta[q]) ) (DSL_SCALED == 1)
//     P[q][kv]  = exp2( fma(S[q][kv], scale * LOG2E, -LOG2E * lse[q]) ),  S = Q K^T, dP = dO V^T
//     (fp32 WMMA)
// and exactly +-0 where the causal mask removes the entry (kv > q + cshift, cshift = Skv - Sq:
// bottom-right). DSL_SCALED == 0 is reserved for a writer that drops the scale (dS' = P * (dP -
// delta)); its reader then multiplies by scale in fp32. Both sides static_assert the value they
// were written for.
//
// WHERE (every offset in BYTES and computed in int64; requires Sq % 32 == 0 and Skv % 32 == 0, so
// cshift is a multiple of 32):
//   * one region per (batch, q head), bh = bat * Hq + qh, regions back to back, dsl_nblk_bh()
//   blocks each
//     (the e2e fold [1, s, b*h, d] and the b2h128 launch give the same bh order);
//   * inside a region, kv16 rows kt = 0 .. Skv/16 - 1 in order; row kt holds the q pairs
//     qp = dsl_qp_first(kt) .. Sq/32 - 1 in order ("packed causal": the leading q pairs of a row
//     are fully masked for all 16 kv rows and are NOT stored; non-causal: dsl_qp_first == 0, a
//     dense rectangle);
//   * a block (bh, kt, qp) is the [16 kv][32 q] bf16 tile, row-major: element (kv, q) with kv = kv
//   row - 16 kt
//     in [0, 16) and q = query - 32 qp in [0, 32) sits at block byte kv * 64 + q * 2; 1024 B per
//     block, so a 128-B line of a block holds kv rows 2l and 2l + 1.
//   dsl_qp_first(kt) = causal ? max(0, floor(kt / 2) - cshift / 32) : 0 is the first q pair with an
//   attended entry (16 kt <= 32 qp + 31 + cshift); it is <= Sq/32 - 1 for every kt < Skv/16, so
//   every row is non-empty. Size per region (prod s4096 causal): 16512 blocks = 16.9 MB; b2
//   h128: 4.33 GB (the dense [kt][qp] rectangle of bwd_r6_b..r7 was 8.59 GB).
// Granularity kv16 (not kv32) so a writer with 48 kv rows per wave (k3) exports whole blocks too.
//
// Reader hint: for a fixed q pair the blocks of kv rows 2j and 2j + 1 are dsl_row_len(2j) blocks
// apart (the two rows share qp_first), so a [2 rows][n q pairs] slab is one 2-D tile of row stride
// dsl_row_len(2j) * 1024 B.
#pragma once
#include <stdint.h>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#define DSL_FN static inline constexpr __host__ __device__

#define DSL_VERSION 1
#define DSL_SCALED 1   // dS carries the softmax scale (the dK operand as k_dkdv64 consumes it)
#define DSL_KV 16      // kv rows per block
#define DSL_Q 32       // queries per block (one q pair of 16-row tiles)
#define DSL_ROW_B 64   // bytes of one block row (32 bf16)
#define DSL_BLK_B 1024 // bytes of one block

DSL_FN int dsl_nqp(int Sq) {
    return Sq / DSL_Q;
} // q pairs
DSL_FN int dsl_nkt(int Skv) {
    return Skv / DSL_KV;
} // kv16 rows
// first stored q pair of kv16 row kt (kt >= 0; cshift = Skv - Sq, a multiple of 32)
DSL_FN int dsl_qp_first(int kt, int cshift, int causal) {
    const int f = kt / 2 - cshift / 32;
    return (causal != 0 && f > 0) ? f : 0;
}
DSL_FN int dsl_row_len(int kt, int Sq, int cshift, int causal) {
    return dsl_nqp(Sq) - dsl_qp_first(kt, cshift, causal);
}
// sum_{k < K} floor(k / 2), K >= 0
DSL_FN int64_t dsl_tri(int64_t K) {
    const int64_t p = K / 2;
    return p * (p - 1) + (K % 2) * p;
}
// sum_{k < K} dsl_qp_first(k): the blocks the packing removes from rows 0 .. K - 1
DSL_FN int64_t dsl_skip(int K, int cshift, int causal) {
    const int64_t m  = cshift / 32;
    const int64_t Kl = K;
    if (causal == 0 || K <= 0)
        return 0;
    if (m <= 0)
        return dsl_tri(Kl) - m * Kl;
    return Kl > 2 * m ? dsl_tri(Kl - 2 * m) : 0;
}
// blocks before kv16 row kt inside one region
DSL_FN int64_t dsl_row_off(int kt, int Sq, int cshift, int causal) {
    return (int64_t) kt * dsl_nqp(Sq) - dsl_skip(kt, cshift, causal);
}
// blocks of one (batch, q head) region
DSL_FN int64_t dsl_nblk_bh(int Sq, int Skv, int causal) {
    return dsl_row_off(dsl_nkt(Skv), Sq, Skv - Sq, causal);
}
// global block index of (bh, kt, qp); only qp >= dsl_qp_first(kt) is stored
DSL_FN int64_t dsl_blk(int64_t bh, int kt, int qp, int Sq, int Skv, int causal) {
    return bh * dsl_nblk_bh(Sq, Skv, causal) + dsl_row_off(kt, Sq, Skv - Sq, causal) +
           (int64_t) (qp - dsl_qp_first(kt, Skv - Sq, causal));
}
// byte of element (kv row kvr, query qr) of head bh (kvr in row kt = kvr / 16, qr in pair qp = qr /
// 32)
DSL_FN int64_t dsl_byte(int64_t bh, int kvr, int qr, int Sq, int Skv, int causal) {
    return dsl_blk(bh, kvr / DSL_KV, qr / DSL_Q, Sq, Skv, causal) * DSL_BLK_B +
           (int64_t) (kvr % DSL_KV) * DSL_ROW_B + (int64_t) (qr % DSL_Q) * 2;
}
// is (kv16 row kt, q pair qp) stored?
DSL_FN bool dsl_stored(int kt, int qp, int Sq, int Skv, int causal) {
    return qp >= dsl_qp_first(kt, Skv - Sq, causal) && qp < dsl_nqp(Sq) && kt >= 0 &&
           kt < dsl_nkt(Skv);
}
// total scratch bytes for B batches x Hq q heads
DSL_FN int64_t dsl_total_bytes(int B, int Hq, int Sq, int Skv, int causal) {
    return (int64_t) B * Hq * dsl_nblk_bh(Sq, Skv, causal) * DSL_BLK_B;
}
