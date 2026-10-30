/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// dkdv_idx.h -- every index / address formula of k_dkdv96ds, as constexpr functions of scalars. i32
// arithmetic, except the TDM element offsets and the dS scratch offsets (i64).
//
// Names: lane in [0, 32), row = lane % 16, half = lane / 16. A workgroup (2 waves) owns a kv block
// of 32 nkv rows at kv0g (nkv 3: the 96-row blocks, nkv 2: the 64-row tail blocks); wave w owns the
// 16 nkv kv rows from kv0 = kv0g + 16 nkv w, as nkv kv sub-tiles kh of 16 rows. A query tile is a q
// pair of 32 queries in two 16-row halves hh. stage_off = s * K1_QDO_B (s in 0..2) is a byte offset
// in the workgroup's LDS. Prefixes: k1_ the ring, TDM, operand, NL / ND and global-store math
// (independent of nkv); k3_ the block decode, causal split and epilogue images per nkv; k1ds_ /
// k3ds_ the dS^T export (dslayout.h); k3m_ the tile sequence's ring stages.
#pragma once
#include "dslayout.h"
#include <stdint.h>

#define K1_FN static inline constexpr __host__ __device__

// ---- layout
#define K1_D_QK 192                     // q / k / dq / dk head dim
#define K1_D_V 128                      // v / o / do / dv head dim
#define K1_XK (K1_D_QK * 2 + 16)        // 400 B: padded LDS row of a Q tile / dK image
#define K1_XV (K1_D_V * 2 + 16)         // 272 B: padded LDS row of a dO tile / dV image
#define K1_QOFF (32 * K1_XV)            // 8704: Q tile offset inside a ring stage
#define K1_QDO_B (32 * (K1_XV + K1_XK)) // 21504: one ring stage ([32][D_V] dO + [32][D_QK] Q)
#define K1_DEPTH 3                      // ring stages
#define K1_RING_B (K1_DEPTH * K1_QDO_B) // 64512
#define K1_NW 2                         // waves per workgroup
#define K1_LDS K1_RING_B                // the ring; the epilogue images reuse it (k3_epi_w)
#define K1_TDM_ROWS 16                  // rows per wave per TDM op (32-row tile, 2 waves)
#define K1_TW 3                         // tensorcnt that retires all but the newest tile's 3 ops
#define K1_NEG (-3.0e38f)               // masked logit (exp2 -> +0)
#define K1_LOG2E 1.4426950408889634f    // f32 0x3FB8AA3B

// ---- workgroup decode (head-group launch): grid (HG, k3_nblk(Skv), B * Hkv / HG), head
// hb = bat * Hkv + hkv = bz * HG + bx
K1_FN int k1_hb(int bz, int bx, int HG) {
    return bz * HG + bx;
}
K1_FN int k1_bat(int bz, int bx, int HG, int Hkv) {
    return k1_hb(bz, bx, HG) / Hkv;
}
K1_FN int k1_hkv(int bz, int bx, int HG, int Hkv) {
    return k1_hb(bz, bx, HG) - k1_bat(bz, bx, HG, Hkv) * Hkv;
}

// ---- causal split at workgroup level (bottom-right: query q sees kv <= q + cshift): the q pairs
// before qp_start see no kv row of the block and are skipped; the workgroup runs nqp_eff q pairs.
K1_FN int k1_qp_start(int kv0g, int cshift, int causal) {
    const int c  = kv0g - cshift;
    const int qs = (c < 0 ? 0 : c) / 32;
    return causal != 0 ? qs : 0;
}
K1_FN int k1_nqp_eff(int kv0g, int cshift, int causal, int nqt) {
    return nqt / 2 - k1_qp_start(kv0g, cshift, causal);
}

// ---- (q pair, q head) counters of the flat tile index ii = q * G + g, division-free
K1_FN int k1_wrap_q(int qc, int gc, int G) {
    return gc + 1 < G ? qc : qc + 1;
}
K1_FN int k1_wrap_g(int qc, int gc, int G) {
    return gc + 1 < G ? gc + 1 : 0;
}
// q pair t clamped to [0, nqt2)
K1_FN int k1_clampqt(int t, int nqt2) {
    t = t < nqt2 ? t : nqt2 - 1;
    return t < 0 ? 0 : t;
}
// the next ring stage (byte offset)
K1_FN int k1_ncur(int cur) {
    return cur == 2 * K1_QDO_B ? 0 : cur + K1_QDO_B;
}
// full-loop iteration ii of n (counters (qi, gh) = the decomposition of ii): the NL / ND prefetch
// tile min(ii + 1, n - 1) and the TDM prefetch tile min(ii + 3, n - 1), as (q, g) relative to the
// loop's first pair (every ternary selects between computed values: selects, not branches)
K1_FN int k1_nx_q(int ii, int n, int qi, int gh, int G) {
    const int qn = k1_wrap_q(qi, gh, G);
    return ii + 1 < n ? qn : qi;
}
K1_FN int k1_nx_g(int ii, int n, int qi, int gh, int G) {
    const int gn = k1_wrap_g(qi, gh, G);
    return ii + 1 < n ? gn : gh;
}
K1_FN int k1_pf_q(int ii, int n, int qi, int gh, int G) {
    const int qn = k1_wrap_q(qi, gh, G), gn = k1_wrap_g(qi, gh, G);
    const int qj = ii + 1 < n ? qn : qi;
    const int q2 = k1_wrap_q(qn, gn, G), g2 = k1_wrap_g(qn, gn, G);
    const int q2c = ii + 2 < n ? q2 : qj;
    const int q3  = k1_wrap_q(q2, g2, G);
    return ii + 3 < n ? q3 : q2c;
}
K1_FN int k1_pf_g(int ii, int n, int qi, int gh, int G) {
    const int qn = k1_wrap_q(qi, gh, G), gn = k1_wrap_g(qi, gh, G);
    const int gj = ii + 1 < n ? gn : gh;
    const int q2 = k1_wrap_q(qn, gn, G), g2 = k1_wrap_g(qn, gn, G);
    const int g2c = ii + 2 < n ? g2 : gj;
    const int g3  = k1_wrap_g(q2, g2, G);
    return ii + 3 < n ? g3 : g2c;
}
// prologue tiles 1 and 2 of a sequence of n tiles: the decomposition of min(j, n - 1), j = 1, 2
K1_FN int k1_pro_q(int j, int n, int G) {
    const int q1w = k1_wrap_q(0, 0, G), g1w = k1_wrap_g(0, 0, G);
    const int q1  = 1 < n ? q1w : 0;
    const int q2w = k1_wrap_q(q1w, g1w, G);
    const int q2  = 2 < n ? q2w : q1;
    return j == 1 ? q1 : q2;
}
K1_FN int k1_pro_g(int j, int n, int G) {
    const int q1w = k1_wrap_q(0, 0, G), g1w = k1_wrap_g(0, 0, G);
    const int g1  = 1 < n ? g1w : 0;
    const int g2w = k1_wrap_g(q1w, g1w, G);
    const int g2  = 2 < n ? g2w : g1;
    return j == 1 ? g1 : g2;
}

// ---- K / V fragments (the S / dP B operands): vec8 (16-B) index of lane `lane`, kv sub-tile kh,
// k-step dt, piece u. Lane (row, half) holds kv row kv0 + kh * 16 + row, d = dt * 32 + half * 8 +
// [0, 8) (u = 0) and dt * 32 + 16 + half * 8 + [0, 8) (u = 1). Byte offset = 16 * index (i32),
// buffer extent nk_b / nv_b.
K1_FN int k1_kbase(int bat, int Skv, int Hkv, int hkv) {
    return bat * Skv * (Hkv * (K1_D_QK / 8)) + hkv * (K1_D_QK / 8);
}
K1_FN int k1_vbase(int bat, int Skv, int Hkv, int hkv) {
    return bat * Skv * (Hkv * (K1_D_V / 8)) + hkv * (K1_D_V / 8);
}
K1_FN int k1_kfrag(int kbase, int Hkv, int kv0, int kh, int dt, int u, int lane) {
    return kbase + (kv0 + kh * 16 + lane % 16) * (Hkv * (K1_D_QK / 8)) + lane / 16 + dt * 4 + u * 2;
}
K1_FN int k1_vfrag(int vbase, int Hkv, int kv0, int kh, int dt, int u, int lane) {
    return vbase + (kv0 + kh * 16 + lane % 16) * (Hkv * (K1_D_V / 8)) + lane / 16 + dt * 4 + u * 2;
}
K1_FN int k1_nk_b(int B, int Skv, int Hkv) {
    return B * Skv * Hkv * (K1_D_QK * 2);
}
K1_FN int k1_nv_b(int B, int Skv, int Hkv) {
    return B * Skv * Hkv * (K1_D_V * 2);
}

// ---- NL / ND (k_delta's row constants, [B, Hq, Sq] fp32): 8 vec4 loads per (q pair, q head),
// entry e = hh * 4 + which * 2 + u (which 0 = NL, 1 = ND) = queries hh * 16 + half * 8 + u * 4 +
// [0, 4) at byte soff (SGPR) + 32 * half + hh * 64 + u * 16 (VGPR: k1_nl_voff + k1_nl_sconst)
K1_FN int k1_qh(int hkv, int G, int gh) {
    return hkv * G + gh;
}
K1_FN int k1_nl_soff(int bat, int Hq, int qh, int Sq, int q0) {
    return ((bat * Hq + qh) * Sq + q0) * 4;
}
K1_FN int k1_nl_sconst(int e) {
    return (e / 4) * 64 + (e % 2) * 16;
}
K1_FN int k1_nl_voff(int lane) {
    return (lane / 16) * 32;
}
K1_FN int k1_nl_b(int B, int Hq, int Sq) {
    return B * Hq * Sq * 4;
}

// ---- TDM: one ring stage = op 0 dO [32][128] (272-B LDS rows), op 1 Q cols [0, 128) and op 2 Q
// cols [128, 192) (400-B rows); wave w moves rows [16 w, 16 w + 16) of every op. Element offsets
// are i64.
// tile columns
K1_FN int k1_tdm_w(int op) {
    return op == 2 ? 64 : 128;
}
// tensor row length (elements)
K1_FN int k1_tdm_d(int op) {
    return op == 0 ? K1_D_V : K1_D_QK;
}
// first column
K1_FN int k1_tdm_c0(int op) {
    return op == 2 ? 128 : 0;
}
// LDS row stride (bytes)
K1_FN int k1_tdm_xrow(int op) {
    return op == 0 ? K1_XV : K1_XK;
}
// pad elements per row
K1_FN int k1_tdm_pad(int op) {
    return (k1_tdm_xrow(op) - 2 * k1_tdm_w(op)) / 2;
}
K1_FN int64_t k1_tdm_row0(int bat, int Sq, int q0, int Hq, int qh) {
    return (int64_t) (bat * Sq + q0) * (int64_t) Hq + (int64_t) qh;
}
K1_FN int64_t k1_tdm_gelem(int bat, int Sq, int q0, int Hq, int qh, int op, int wv) {
    return k1_tdm_row0(bat, Sq, q0, Hq, qh) * k1_tdm_d(op) + k1_tdm_c0(op) +
           (int64_t) (wv % 2) * K1_TDM_ROWS * (int64_t) Hq * k1_tdm_d(op);
}
// outer stride (elements)
K1_FN int k1_tdm_stride(int Hq, int op) {
    return Hq * k1_tdm_d(op);
}
K1_FN int k1_tdm_lds(int stage_off, int op, int wv) {
    return stage_off + (op == 0 ? 0 : K1_QOFF) + k1_tdm_c0(op) * 2 +
           (wv % 2) * K1_TDM_ROWS * k1_tdm_xrow(op);
}
// outer extent of the descriptor: the slice height. The descriptor is built once; every tile is a q
// pair clamped to [0, Sq / 32), so its 16-row slices lie inside Sq and no row is clamped.
K1_FN int k1_tdm_ext(void) {
    return K1_TDM_ROWS;
}
// group 1 word 0: data size 2 B (1 << 16), pad enable (1 << 20), pad interval (<< 22) =
// log2(w / 2 dwords) - 1, pad amount (<< 25) = pad / 2 dwords - 1; ops 0..2: 0x07510000,
// 0x47510000, 0x87110000
K1_FN int k1_ilog2(int x) {
    return x <= 1 ? 0 : 1 + k1_ilog2(x / 2);
}
K1_FN uint32_t k1_tdm_w0(int op) {
    return (1u << 16) | (1u << 20) | ((uint32_t) (k1_ilog2(k1_tdm_w(op) / 2) - 1) << 22) |
           ((uint32_t) (k1_tdm_pad(op) / 2 - 1) << 25);
}

// ---- LDS reads of a stage: address = stage_off + lane part + compile-time constant (the kernel
// adds the constant to a base pointer, so it lands in the DS offset field).
// S / dP A operands (ds_load_b128), entry hh * 20 + (Q: dt * 2 + u | dO: 12 + dt * 2 + u): lane
// (row, half) reads row hh * 16 + row, bytes half * 16 + dt * 64 + u * 32 of the Q (isq 1, dt < 6)
// or dO (isq 0, dt < 4) tile
K1_FN int k1_rd_lane(int isq, int lane) {
    return (lane % 16) * (isq ? K1_XK : K1_XV) + (lane / 16) * 16;
}
K1_FN int k1_rd_const(int hh, int isq, int dt, int u) {
    return (isq ? K1_QOFF : 0) + hh * 16 * (isq ? K1_XK : K1_XV) + dt * 64 + u * 32;
}
// dV / dK A operands dO^T / Q^T (ds_load_tr16_b128): d tile d (dO d < 8, Q d < 12), piece p (rows
// + 16 p)
K1_FN int k1_lane_r(int lane) {
    return (lane / 16) * 8 + lane % 8;
}
K1_FN int k1_lane_c(int lane) {
    return ((lane / 8) % 2) * 8;
}
K1_FN int k1_tr_lane(int isq, int lane) {
    return k1_lane_r(lane) * (isq ? K1_XK : K1_XV) + k1_lane_c(lane) * 2;
}
K1_FN int k1_tr_const(int isq, int d, int p) {
    return (isq ? K1_QOFF : 0) + d * 32 + p * 16 * (isq ? K1_XK : K1_XV);
}

// ---- epilogue global store: vec8 index of (kv row kv0 + 4 rb + l / 8, d = li * 64 + (l % 8) * 8)
// in dv / dk (the image readback, k3_epi_rd)
K1_FN int k1_epi_g(int bat, int Skv, int Hkv, int hkv, int kv0, int isk, int rb, int li, int lane) {
    const int base = isk ? k1_kbase(bat, Skv, Hkv, hkv) : k1_vbase(bat, Skv, Hkv, hkv);
    const int rs   = Hkv * ((isk ? K1_D_QK : K1_D_V) / 8);
    return base + (kv0 + rb * 4 + lane / 8) * rs + lane % 8 + li * 8;
}

// ---- kv blocks: grid y = k3_nblk(Skv) blocks in ascending kv (longest first under bottom-right
// causal). Blocks [0, n96) are 96-row blocks (2 waves x 48 rows, nkv 3) at kv0g = 96 bid; the n64
// 64-row tail blocks (nkv 2) follow at 96 n96 + 64 j. Skv % 96 == 0: no tail; 64: one; 32: two
// (96 (n - 1) + 128). The host rejects any other Skv (limits.k3_ok).
#define K3_NKV 3              // 16-row kv sub-tiles per wave in a 96-row block
#define K3_ROWS (32 * K3_NKV) // 96 kv rows per block
K1_FN int k3_n96(int Skv) {
    const int n = Skv / K3_ROWS, r = Skv % K3_ROWS;
    return r == 32 ? n - 1 : n;
}
K1_FN int k3_n64(int Skv) {
    const int r = Skv % K3_ROWS;
    return r == 0 ? 0 : r == 64 ? 1 : r == 32 ? 2 : -1;
}
K1_FN int k3_nblk(int Skv) {
    return k3_n96(Skv) + k3_n64(Skv);
}
K1_FN int k3_kv0g(int bid, int n96) {
    return bid < n96 ? bid * K3_ROWS : n96 * K3_ROWS + (bid - n96) * 64;
}
K1_FN int k3_kv0(int kv0g, int wv, int nkv) {
    return kv0g + wv * 16 * nkv;
}
// the q pairs [qp_start, qp_start + k3_nmaskp) straddle the diagonal of the block of `rows` kv rows
// at kv0g (some of its entries masked): they run with the causal mask; 0 when not causal
K1_FN int k3_nmaskp(int kv0g, int rows, int cshift, int causal, int nqt) {
    const int nqt2    = nqt / 2;
    const int qs      = k1_qp_start(kv0g, cshift, causal);
    const int u       = kv0g + rows - 1 - cshift;
    const int qsu     = (u + 31) / 32;
    int       qsf     = u < 0 ? 0 : qsu;
    qsf               = qsf < nqt2 ? qsf : nqt2;
    int nm            = qsf - qs;
    nm                = nm < 0 ? 0 : nm;
    const int nqp_eff = nqt2 - qs;
    nm                = nm < nqp_eff ? nm : nqp_eff;
    return causal != 0 ? nm : 0;
}
// epilogue images of a wave with nkv sub-tiles: row-major dV [16 nkv][D_V] (272-B rows) at
// w * k3_epi_w(nkv), dK [16 nkv][D_QK] (400-B rows) after it. C fragment (kh, d) of lane
// (row, half) = kv row kh * 16 + row, cols d * 16 + half * 8 + [0, 8).
K1_FN int k3_epi_dk(int nkv) {
    return 16 * nkv * K1_XV;
}
K1_FN int k3_epi_w(int nkv) {
    return 16 * nkv * (K1_XV + K1_XK);
}
static_assert(K1_NW * k3_epi_w(K3_NKV) <= K1_LDS, "the epilogue images must fit in the ring");
K1_FN int k3_epi_wr(int nkv, int wv, int isk, int kh, int d, int lane) {
    const int x = isk ? K1_XK : K1_XV;
    return wv * k3_epi_w(nkv) + (isk ? k3_epi_dk(nkv) : 0) + (lane % 16) * x + (lane / 16) * 16 +
           kh * 16 * x + d * 32;
}
// whole-line readback: lane l reads kv row 4 rb + l / 8 (rb in [0, 4 nkv)), bytes li * 128 +
// (l % 8) * 16 of the image; its global store is k1_epi_g
K1_FN int k3_epi_rd(int nkv, int wv, int isk, int rb, int li, int lane) {
    const int x = isk ? K1_XK : K1_XV;
    return wv * k3_epi_w(nkv) + (isk ? k3_epi_dk(nkv) : 0) + (lane / 8) * x + (lane % 8) * 16 +
           rb * 4 * x + li * 128;
}

// ---- tile sequence of a workgroup: j = 0 .. N - 1, N = n_mask + n_full = G * nqp_eff, tile
// T_j = (q pair qp_start + j / G, q head j % G); the n_mask tiles of the diagonal come first. Tile
// j lives in ring stage sigma(j) = (j - n_mask) mod 3, so the full loop's first tile (j = n_mask)
// is in stage 0. k3m_st0: the byte offset of stage sigma(0) = ((-n_mask) mod 3) * K1_QDO_B.
K1_FN int k3m_st0(int n_mask) {
    const int r = n_mask % K1_DEPTH;
    return r == 0 ? 0 : (K1_DEPTH - r) * K1_QDO_B;
}
// counters (q pair, q head) after full iteration 0 (the 64-row tail body peels it): k1_wrap_q /
// k1_wrap_g (0, 0, G) in integer form (G >= 1): gh = min(G - 1, 1), qi = 1 - gh. The select form
// made clang carry (G > 1) as a lane mask and rebuild the value with v_cndmask + v_readfirstlane.
K1_FN int k1ds_g1(int G) {
    return (G - 1) < 1 ? (G - 1) : 1;
}
K1_FN int k1ds_q1(int G) {
    return 1 - k1ds_g1(G);
}

// ---- dS^T export (dslayout.h). A wave owns the nkv kv16 rows from kt0 = kv0 / 16; one iteration
// (q pair qt, q head gh) stores the blocks (bh0 + gh, kt0 + kh, qt), kh < nkv, each with two
// buffer_store_b128 of 512 contiguous bytes (4 whole 128-B lines). Store address = descriptor base
// (64-bit, per wave) + soffset (SGPR: gh * hstride + 1024 qt + rowc(kh)) + voffset (VGPR, per
// lane, loop-invariant: k1ds_voff) + 512 t. A block is stored only if qt >= qp_first(kt0 + kh).
#define K1DS_ST_B 512 // bytes of one buffer_store_b128 of the export (32 lanes x 16 B)

// branch-free forms of dslayout.h's row arithmetic (its own functions branch: the prologue would
// carry ~40 scalar branches); m = cshift / 32 (cshift % 32 == 0)
K1_FN int k1ds_m(int cshift) {
    return cshift >> 5;
}
K1_FN int64_t k1ds_tri(int64_t K) { // sum_{k < K} floor(k / 2), K >= 0
    const int64_t p = K >> 1;
    return p * (p - 1) + (K & 1) * p;
}
K1_FN int64_t k1ds_skip(int K, int m, int causal) { // == dsl_skip(K, 32 m, causal), K >= 0
    const int64_t Kl = K, ml = m;
    const int64_t r = Kl - 2 * ml;
    const int64_t a = k1ds_tri(Kl) - ml * Kl;  // m <= 0: every row skips floor(k / 2) - m
    const int64_t b = k1ds_tri(r > 0 ? r : 0); // m > 0: rows k >= 2 m skip floor(k / 2) - m
    const int64_t v = ml <= 0 ? a : b;
    return causal != 0 ? v : 0;
}
K1_FN int k1ds_qpf_kt(int kt, int m, int causal) { // == dsl_qp_first(kt, 32 m, causal)
    const int f = (kt >> 1) - m;
    return (causal != 0 && f > 0) ? f : 0;
}
// == dsl_row_off(kt, 32 nqp, 32 m, causal)
K1_FN int64_t k1ds_row_off(int kt, int nqp, int m, int causal) {
    return (int64_t) kt * nqp - k1ds_skip(kt, m, causal);
}
K1_FN int64_t k1ds_nblk_bh(int Sq, int Skv, int causal) { // == dsl_nblk_bh
    return k1ds_row_off(Skv / DSL_KV, Sq / DSL_Q, k1ds_m(Skv - Sq), causal);
}
// first region of the workgroup: bh0 = bat * Hq + hkv * G; its q heads gh = 0 .. G - 1 are bh0 + gh
K1_FN int64_t k1ds_bh0(int bat, int Hq, int hkv, int G) {
    return (int64_t) bat * Hq + (int64_t) hkv * G;
}
// q-head stride in bytes (one region), i32: it scales gh, which is 0 under MHA (limits.py)
K1_FN int k1ds_hstride(int Sq, int Skv, int causal) {
    return (int) (k1ds_nblk_bh(Sq, Skv, causal) * DSL_BLK_B);
}
// voffset of lane `lane` (loop-invariant): lane (r = lane % 16, h = lane / 16) stores its dK
// operand's dwords 0..3 = chunk (r, h) at dsl_chunk(r, h) = lane * 16 and dwords 4..7 = chunk
// (r, h + 2) at dsl_chunk(r, h + 2) = 512 + lane * 16 (store t adds 512 t)
K1_FN int k1ds_voff(int lane) {
    return dsl_chunk(lane % 16, lane / 16);
}
// the wave's first kv16 row (kv0 is a multiple of 16)
K1_FN int k3ds_kt0(int kv0) {
    return kv0 / DSL_KV;
}
// the wave's descriptor base, in bytes from the scratch start (int64): region bh0, row kt0
K1_FN int64_t k3ds_base(int bat, int Hq, int hkv, int G, int kt0, int Sq, int Skv, int causal) {
    return (k1ds_bh0(bat, Hq, hkv, G) * k1ds_nblk_bh(Sq, Skv, causal) +
            k1ds_row_off(kt0, Sq / DSL_Q, k1ds_m(Skv - Sq), causal)) *
           DSL_BLK_B;
}
// the wave's true extent (int64 bytes): up to the end of row kt0 + nkv - 1 of region bh0 + G - 1
K1_FN int64_t k3ds_ext(int G, int kt0, int nkv, int Sq, int Skv, int causal) {
    const int nqp = Sq / DSL_Q, m = k1ds_m(Skv - Sq);
    return ((int64_t) (G - 1) * k1ds_nblk_bh(Sq, Skv, causal) +
            k1ds_row_off(kt0 + nkv, nqp, m, causal) - k1ds_row_off(kt0, nqp, m, causal)) *
           DSL_BLK_B;
}
// num_records of the descriptor: the true extent, or 0 for ds_mode 0 (the range check then drops
// every store: the export off, same code)
K1_FN int64_t k3ds_nrec(int ds_mode, int G, int kt0, int nkv, int Sq, int Skv, int causal) {
    return ds_mode != 0 ? k3ds_ext(G, kt0, nkv, Sq, Skv, causal) : 0;
}
// first stored q pair of kv sub-tile kh (row kt0 + kh)
K1_FN int k3ds_qpf(int kh, int kt0, int Sq, int Skv, int causal) {
    return k1ds_qpf_kt(kt0 + kh, k1ds_m(Skv - Sq), causal);
}
// is (kh, qt) stored? (uniform per wave and iteration)
K1_FN bool k3ds_put(int kh, int qt, int kt0, int Sq, int Skv, int causal) {
    return qt >= k3ds_qpf(kh, kt0, Sq, Skv, causal);
}
// row constant of kv sub-tile kh: bytes from the wave base to row kt0 + kh, minus its qp_first
// blocks (may be negative; rowc + 1024 qt >= 0 whenever the block is stored)
K1_FN int k3ds_rowc(int kh, int kt0, int Sq, int Skv, int causal) {
    const int nqp = Sq / DSL_Q, m = k1ds_m(Skv - Sq);
    return (int) ((k1ds_row_off(kt0 + kh, nqp, m, causal) - k1ds_row_off(kt0, nqp, m, causal) -
                   (int64_t) k3ds_qpf(kh, kt0, Sq, Skv, causal)) *
                  DSL_BLK_B);
}
// q-pair / q-head part of a soffset (the kernel adds rowc(kh) at the store): gh * hstride + 1024 qt
K1_FN int k3ds_sq(int qt, int gh, int Sq, int Skv, int causal) {
    return gh * k1ds_hstride(Sq, Skv, causal) + qt * DSL_BLK_B;
}
