/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// dslayout.h -- the dS scratch: k_dkdv96ds stores into it every bf16 dS tile it feeds its dK WMMA,
// k_dqds_c reads them back (dQ = dS K). Both kernels static_assert DSL_VERSION and DSL_SCALED; a
// layout change bumps DSL_VERSION.
//
// What: dS[q][kv] = bf16(scale * P[q][kv] * (dP[q][kv] - delta[q])) with P = exp(scale * S - lse),
// S = Q K^T, dP = dO V^T (DSL_SCALED 1: the softmax scale is included; 0, a dS without it that the
// reader scales in fp32, is reserved), and exactly +-0 where the causal mask removes the entry
// (kv > q + cshift, cshift = Skv - Sq: bottom-right).
//
// Where (byte offsets, int64; Sq % 32 == 0 and Skv % 32 == 0, so cshift % 32 == 0):
//   * one region per (batch, q head), bh = bat * Hq + qh, regions back to back, dsl_nblk_bh()
//     blocks each;
//   * inside a region the kv16 rows kt = 0 .. Skv / 16 - 1 in order; row kt holds the q pairs
//     qp = dsl_qp_first(kt) .. Sq / 32 - 1 in order (packed causal: the leading q pairs of a row
//     are fully masked for all 16 kv rows and are not stored; non-causal: dsl_qp_first == 0, a
//     dense rectangle);
//   * a block (bh, kt, qp) is the [16 kv][32 q] bf16 tile in the order of the writer's dK operand:
//     64 chunks of 16 B, chunk (kv, c) = the queries [8 c, 8 c + 8) of kv row kv (kv in [0, 16),
//     c in [0, 4)) at block byte dsl_chunk(kv, c) = (c / 2) * 512 + ((c % 2) * 16 + kv) * 16, so
//     element (kv, q), q in [0, 32), sits at dsl_chunk(kv, q / 8) + (q % 8) * 2. A writer lane (kv
//     row r = lane % 16, half h = lane / 16) holds chunks (r, h) and (r, h + 2) in its dK operand
//     (dwords 0..3, 4..7) and stores them at lane * 16 and 512 + lane * 16: two whole 512-B stores
//     per block, no cross-lane exchange. 1024 B per block.
// dsl_qp_first(kt) = causal ? max(0, floor(kt / 2) - cshift / 32) : 0 is the first q pair with an
// attended entry (16 kt <= 32 qp + 31 + cshift); it is <= Sq / 32 - 1 for every kt < Skv / 16, so
// no row is empty. s4096 causal: 16512 blocks (16.9 MB) per region, 4.33 GB at b2 h128. Blocks are
// 16 kv rows tall so that a k_dkdv96ds wave (48 kv rows) stores whole blocks.
//
// Reader hint: for a fixed q pair the blocks of kv rows 2j and 2j + 1 are dsl_row_len(2j) blocks
// apart (the two rows share qp_first), so a [2 rows][n q pairs] slab is one 2-D tile of row stride
// dsl_row_len(2j) * 1024 B.
#pragma once
#include <stdint.h>

#define DSL_FN static inline constexpr __host__ __device__

#define DSL_VERSION 2
#define DSL_SCALED 1   // dS includes the softmax scale (the dK operand as k_dkdv96ds consumes it)
#define DSL_KV 16      // kv rows per block
#define DSL_Q 32       // queries per block (one q pair of 16-row tiles)
#define DSL_BLK_B 1024 // bytes of one block

// q pairs
DSL_FN int dsl_nqp(int Sq) {
    return Sq / DSL_Q;
}
// kv16 rows
DSL_FN int dsl_nkt(int Skv) {
    return Skv / DSL_KV;
}
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
// block byte of chunk (kv row kv in [0, 16), query chunk c in [0, 4): queries [8 c, 8 c + 8))
DSL_FN int dsl_chunk(int kv, int c) {
    return (c >> 1) * 512 + ((c & 1) * 16 + kv) * 16;
}
// byte of element (kv row kvr, query qr) of region bh (kvr in row kt = kvr / 16, qr in q pair
// qp = qr / 32)
DSL_FN int64_t dsl_byte(int64_t bh, int kvr, int qr, int Sq, int Skv, int causal) {
    return dsl_blk(bh, kvr / DSL_KV, qr / DSL_Q, Sq, Skv, causal) * DSL_BLK_B +
           (int64_t) dsl_chunk(kvr % DSL_KV, (qr % DSL_Q) / 8) + (int64_t) (qr % 8) * 2;
}
// is (kv16 row kt, q pair qp) stored?
DSL_FN bool dsl_stored(int kt, int qp, int Sq, int Skv, int causal) {
    return qp >= dsl_qp_first(kt, Skv - Sq, causal) && qp < dsl_nqp(Sq) && kt >= 0 &&
           kt < dsl_nkt(Skv);
}
// scratch bytes for B batches x Hq q heads (limits.ds_layout_bytes)
DSL_FN int64_t dsl_total_bytes(int B, int Hq, int Sq, int Skv, int causal) {
    return (int64_t) B * Hq * dsl_nblk_bh(Sq, Skv, causal) * DSL_BLK_B;
}
