/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k1ds_idx.h -- every index / address formula of k_dkdv64ds's dS^T export (bwd_hk_k1ds), as
// constexpr
// __host__ __device__ functions of scalars. The kernel calls exactly these; bounds_proof.py
// compiles this file for the host and replays them over every launched shape. The layout is
// dslayout.h v2.
//
// Per wave (kv block bid, wave wv): kv16 rows kt0 = (64 bid + 32 wv) / 16 (kv sub-tile kh = 0) and
// kt0 + 1 (kh = 1). One wave-iteration (query pair qt, q head gh of the group) exports two blocks
// (bh0 + gh, kt0 + kh, qt), each with two buffer_store_b128 (t = 0: block rows 0..7, t = 1:
// rows 8..15; 512 contiguous bytes = 4 whole 128-B lines). Store address = descriptor base (64-bit,
// per wave) + soffset (SGPR, per iteration and kh) + voffset (VGPR, per lane, loop-invariant) + 512
// t (instruction offset). A block is stored only if qt >= qp_first(kt0 + kh).
#pragma once
#include "dslayout.h"
#include <stdint.h>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#define K1DS_FN static inline constexpr __host__ __device__
#define K1DS_ST_B 512 // bytes of one buffer_store_b128 of the export (32 lanes x 16 B)

// ---- branch-free forms of dslayout.h's row arithmetic (dslayout.h's own functions branch; the
// kernel's prologue would carry ~40 scalar branches). bounds_proof.py DL2 proves each == its
// dslayout.h function for every launched shape and an exhaustive grid of (Sq, Skv, causal, kt). m =
// cshift / 32 (cshift % 32 == 0).
K1DS_FN int k1ds_m(int cshift) {
    return cshift >> 5;
}
K1DS_FN int64_t k1ds_tri(int64_t K) { // sum_{k < K} floor(k / 2), K >= 0
    const int64_t p = K >> 1;
    return p * (p - 1) + (K & 1) * p;
}
K1DS_FN int64_t k1ds_skip(int K, int m, int causal) { // == dsl_skip(K, 32 m, causal), K >= 0
    const int64_t Kl = K, ml = m;
    const int64_t r = Kl - 2 * ml;
    const int64_t a = k1ds_tri(Kl) - ml * Kl;  // m <= 0: every row skips floor(k/2) - m
    const int64_t b = k1ds_tri(r > 0 ? r : 0); // m > 0: rows k >= 2m skip floor(k/2) - m
    const int64_t v = ml <= 0 ? a : b;
    return causal != 0 ? v : 0;
}
K1DS_FN int k1ds_qpf_kt(int kt, int m, int causal) { // == dsl_qp_first(kt, 32 m, causal)
    const int f = (kt >> 1) - m;
    return (causal != 0 && f > 0) ? f : 0;
}
K1DS_FN int64_t k1ds_row_off(int kt, int nqp, int m,
                             int causal) { // == dsl_row_off(kt, 32 nqp, 32 m, causal)
    return (int64_t) kt * nqp - k1ds_skip(kt, m, causal);
}
K1DS_FN int64_t k1ds_nblk_bh(int Sq, int Skv, int causal) { // == dsl_nblk_bh
    return k1ds_row_off(Skv / DSL_KV, Sq / DSL_Q, k1ds_m(Skv - Sq), causal);
}

// ---- the kernel's export index math
// ------------------------------------------------------------------------- the wave's first kv16
// row
K1DS_FN int k1ds_kt0(int bid, int wv) {
    return (bid * 64 + wv * 32) / DSL_KV;
}
// first q-head region of the workgroup: bh0 = bat*Hq + hkv*G; the group's q heads gh = 0..G-1 are
// bh0 + gh
K1DS_FN int64_t k1ds_bh0(int bat, int Hq, int hkv, int G) {
    return (int64_t) bat * Hq + (int64_t) hkv * G;
}
// the wave's descriptor base, bytes from the scratch start (int64): region bh0, first block of row
// kt0
K1DS_FN int64_t k1ds_base(int bat, int Hq, int hkv, int G, int bid, int wv, int Sq, int Skv,
                          int causal) {
    return (k1ds_bh0(bat, Hq, hkv, G) * k1ds_nblk_bh(Sq, Skv, causal) +
            k1ds_row_off(k1ds_kt0(bid, wv), Sq / DSL_Q, k1ds_m(Skv - Sq), causal)) *
           DSL_BLK_B;
}
// the wave's TRUE extent (int64 bytes): rows kt0, kt0 + 1 of region bh0 + G - 1 end it
K1DS_FN int64_t k1ds_ext(int G, int bid, int wv, int Sq, int Skv, int causal) {
    const int kt0 = k1ds_kt0(bid, wv), nqp = Sq / DSL_Q, m = k1ds_m(Skv - Sq);
    return ((int64_t) (G - 1) * k1ds_nblk_bh(Sq, Skv, causal) +
            k1ds_row_off(kt0 + 2, nqp, m, causal) - k1ds_row_off(kt0, nqp, m, causal)) *
           DSL_BLK_B;
}
// num_records passed to the descriptor: the true extent (ds_mode != 0) or 0 (ds_mode 0, route dsw0:
// the range check drops every store at the address stage -- same binary, no write traffic)
K1DS_FN int64_t k1ds_nrec(int ds_mode, int G, int bid, int wv, int Sq, int Skv, int causal) {
    return ds_mode != 0 ? k1ds_ext(G, bid, wv, Sq, Skv, causal) : 0;
}
// q-head stride in bytes (one region); int32 (bounds_proof: the group's span < 2^31)
K1DS_FN int k1ds_hstride(int Sq, int Skv, int causal) {
    return (int) (k1ds_nblk_bh(Sq, Skv, causal) * DSL_BLK_B);
}
// first stored q pair of kv sub-tile kh
K1DS_FN int k1ds_qpf(int kh, int bid, int wv, int Sq, int Skv, int causal) {
    return k1ds_qpf_kt(k1ds_kt0(bid, wv) + kh, k1ds_m(Skv - Sq), causal);
}
// is (kh, qt) stored? (uniform per wave and iteration)
K1DS_FN bool k1ds_put(int kh, int qt, int bid, int wv, int Sq, int Skv, int causal) {
    return qt >= k1ds_qpf(kh, bid, wv, Sq, Skv, causal);
}
// row constant of kv sub-tile kh: bytes from the wave base to row kt0 + kh, minus its qp_first
// blocks (may be negative; rowc + 1024 qt >= 0 whenever the block is stored)
K1DS_FN int k1ds_rowc(int kh, int bid, int wv, int Sq, int Skv, int causal) {
    const int kt0 = k1ds_kt0(bid, wv), nqp = Sq / DSL_Q, m = k1ds_m(Skv - Sq);
    return (int) ((k1ds_row_off(kt0 + kh, nqp, m, causal) - k1ds_row_off(kt0, nqp, m, causal) -
                   (int64_t) k1ds_qpf(kh, bid, wv, Sq, Skv, causal)) *
                  DSL_BLK_B);
}
// soffset of block (bh0 + gh, kt0 + kh, qt) relative to the wave base: gh * hstride + 1024 qt +
// rowc(kh)
K1DS_FN int k1ds_so(int kh, int qt, int gh, int bid, int wv, int Sq, int Skv, int causal) {
    return gh * k1ds_hstride(Sq, Skv, causal) + qt * DSL_BLK_B +
           k1ds_rowc(kh, bid, wv, Sq, Skv, causal);
}
// voffset of lane `lane` (loop-invariant, dslayout v2): lane (r = lane % 16, h = lane / 16) stores its dK
// operand's dwords 0..3 = chunk (r, h) at dsl_chunk(r, h) = lane * 16 and dwords 4..7 = chunk (r, h + 2) at
// dsl_chunk(r, h + 2) = 512 + lane * 16 (store t adds 512 t)
K1DS_FN int k1ds_voff(int lane) {
    return dsl_chunk(lane % 16, lane / 16);
}
// counters (query pair, q head) after the peeled full iteration 0: k1_wrap_q / k1_wrap_g (0, 0, G)
// in integer form (G >= 1): gh = min(G - 1, 1), qi = 1 - gh. The select form made clang carry (G >
// 1) as a lane mask and rebuild the value through v_cndmask + v_readfirstlane, which isa_static
// S5's partial-def rule cannot prove untainted.
K1DS_FN int k1ds_g1(int G) {
    return (G - 1) < 1 ? (G - 1) : 1;
}
K1DS_FN int k1ds_q1(int G) {
    return 1 - k1ds_g1(G);
}
