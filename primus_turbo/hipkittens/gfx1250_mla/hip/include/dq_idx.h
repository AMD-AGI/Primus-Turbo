/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// dq_idx.h -- every index formula of k_dqds_c, dQ[q, :] = sum_kv dS[q, kv] K[kv, :] with dS read
// from the dS scratch (dslayout.h, scale included): global offsets, LDS addresses, TDM descriptor
// fields, ring stages, loop bounds and the zeroing predicate, as constexpr functions of scalars
// (i32, except the 64-bit global element offsets).
//
// Tiles: the main tiles of [q_off, Sq) are 384 queries (4 waves x 96: NQW 6 sub-tiles of 16 per
// wave), the merged head tile [0, 256) (q_off == 256 only) is 4 waves x 64 queries (NQW 4);
// Sq % 384 is 0 or 256. Block 128 = 4 waves, 1 per SIMD, one workgroup per CU (LDS 274432 B); the
// 4 waves share one K ring.
//
// TDM issue is split by role, one op per wave per 64-kv unit u: waves 0 / 3 move the dS of kv32
// row 2u + (w >> 1) for all NB_G q pairs of the workgroup (one 2-D op, <= 24 KB), waves 1 / 2 move
// the K rows [64u + 32 (w >> 1), +32) (12 KB). Each CU TDM engine then carries one dS op and one K
// op per unit, whether the engines are shared by warp parity ({0, 2} / {1, 3}) or by SIMD pair
// ({0, 1} / {2, 3}). The K and dS rings have 3 stages each, every op is issued 2 units ahead, and
// a wave waits for all but 1 op before every barrier: <= 2 TDM ops in flight per wave.
//
// WMMA orientation (dQ^T): acc[qh][dt] (16 d x 16 q) += A (16 d x 32 kv) * B (32 kv x 16 q) with
//   A = K^T  : tr16 of the K image [kv][d] (lane = d row, K-dim = kv)
//   B = dS^T : tr16 of the dS block images [kv16][q32] (lane = q column, K-dim = kv: kv 0..15 from
//              block row 2r, kv 16..31 from block row 2r + 1)
// Per output: the kv32 rows ascending, one WMMA per (sub-tile, d tile, kv32 row) whose dS block has
// an attended entry. Skipping a fully masked block's WMMAs is exact: they would add +-0 to an
// accumulator that is never -0.
#pragma once

#define DQ_HD __host__ __device__ __attribute__((always_inline))

#include "dslayout.h"

namespace dq {

// ---- layout constants
constexpr int D_QK      = 192;       // q/k head dim (dq row)
constexpr int DV8_QK    = D_QK / 8;  // vec8 tiles per dq row (24)
constexpr int NDT       = D_QK / 16; // 16-wide d tiles of dQ^T (12)
constexpr int WAVE      = 32;
constexpr int NWAVE     = 4;             // waves per workgroup (both tile kinds), 1 per SIMD
constexpr int KV32      = 32;            // kv rows per WMMA step (kv32 row r = block rows 2r, 2r+1)
constexpr int MAIN_NQW  = 6;             // main tile: 4 waves x 96 queries (6 sub-tiles of 16)
constexpr int HEAD_NQW  = 4;             // merged head tile: 4 waves x 64 queries = 256
constexpr int DS_BLK_EL = DSL_BLK_B / 2; // 512 bf16 per dslayout block [16 kv][32 q]
// dS block images in LDS: the TDM pads 16 B after every 256 data bytes (16 dslayout chunks), so the
// 32 tr16 lanes of one operand (16 chunks of one 256-B group + 16 of the next) hit every bank
// exactly twice (64-B padding: 3 times)
constexpr int       DSI_GRP_B    = 256 + 16;      // one padded 256-B group (272 B)
constexpr int       DSI_BLK_B    = 4 * DSI_GRP_B; // one block image (1088 B)
constexpr unsigned  TDIM0        = 0x7FFFFFFFu;   // inner extent: never clamps
constexpr int       DQ_IMG_ROW_B = 400;           // dQ epilogue image row (384 B data + 16 B pad)
constexpr int       DQ_LINE_EL   = 64;            // bf16 per 128-B dq line
constexpr long long DQ_EXTENT_B  = 1LL << 30;     // dq buffer extent (> dq bytes: limits.MAX_BH_S)
constexpr int       RSRC_FLAGS   = 0x27000;       // buffer descriptor word 3

// ---- the rings: unit u = kv rows [64u, 64u + 64) = kv32 rows 2u, 2u + 1; one workgroup barrier
// P_u per unit.
// K ring: C_KDEPTH stages of 64 rows x 480 B (192 columns, 32 B of pad after every 128 B, so the
//   16 rows x 2 column halves of a half-wave K^T tr16 hit 16 distinct 4-bank groups); half h = rows
//   [32h, 32h + 32) of the unit, written by the K wave of half h (wave 1 + h; one 12-KB op).
// dS ring: C_DSDEPTH stages of 2 slots; slot h = kv32 row 2u + h for all NB_G q pairs of the
//   workgroup (2 block rows x NB_G block images), written by the dS wave of half h (wave 3h; one
//   2-D op).
// Op order per wave: op(0) op(1) | P_0 | op(2) | P_1 | op(3) ..., op(v) = the wave's role op of
//   unit v. P_{u+1} sits in unit u after its last ring read; before it a wave needs op(u + 1)
//   landed and only op(u + 2) may stay in flight (tensorcnt C_TW = 1); after it the wave issues
//   op(u + 3) into unit u's (now free) stages. Ops of units >= nu are 16-B dummies (the op count
//   stays uniform, no duplicate reads).
constexpr int C_KROW_B  = 480;
constexpr int C_KHALF_B = KV32 * C_KROW_B; // 15360 (one K wave's 32 rows)
constexpr int C_KSTG_B  = 2 * C_KHALF_B;   // 30720 (64 rows)
constexpr int C_KDEPTH  = 3;
constexpr int C_DSDEPTH = 3;
constexpr int C_DS_OFF  = C_KDEPTH * C_KSTG_B; // 92160
constexpr int C_TW      = 1;

// ---- integer helpers
DQ_HD constexpr int imin(int a, int b) {
    return a < b ? a : b;
}
DQ_HD constexpr int imax(int a, int b) {
    return a > b ? a : b;
}
DQ_HD constexpr int floordiv(int a, int b) { // b > 0
    int q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// ---- per-tile geometry (nqw = 16-row query sub-tiles per wave)
// queries per workgroup
DQ_HD constexpr int bqwg(int nqw) {
    return 16 * nqw * NWAVE;
}
// dS q pairs (blocks) per workgroup
DQ_HD constexpr int nbg(int nqw) {
    return NWAVE * nqw / 2;
}
// one kv32 row x NB_G block images
DQ_HD constexpr int c_slot_b(int nqw) {
    return 2 * nbg(nqw) * DSI_BLK_B;
}
// 2 slots (kv32 rows 2u, 2u + 1)
DQ_HD constexpr int c_dsstg_b(int nqw) {
    return 2 * c_slot_b(nqw);
}
// LDS: the rings (248832 B), then one dQ epilogue image (16 rows) per wave: 274432 B, one
// workgroup per CU
constexpr int LDS_C     = C_DS_OFF + C_DSDEPTH * c_dsstg_b(MAIN_NQW);
constexpr int EPI_IMG_B = 16 * DQ_IMG_ROW_B;

DQ_HD constexpr int lds_bytes() {
    return LDS_C + NWAVE * EPI_IMG_B;
}

// ---- roles (wave-uniform): waves 0 / 3 issue dS (kv32 row 2u + half), waves 1 / 2 issue K (rows
// 32 half of the unit); half = w >> 1, dS iff (w & 1) == (w >> 1)
DQ_HD constexpr bool c_role_ds(int wv) {
    return (wv & 1) == (wv >> 1);
}
DQ_HD constexpr int c_half(int wv) {
    return wv >> 1;
}

// ---- lane decomposition
DQ_HD constexpr int lane_row(int lane) {
    return lane % 16;
}
DQ_HD constexpr int lane_half(int lane) {
    return lane / 16;
}
// tr16 row
DQ_HD constexpr int lane_r(int lane) {
    return (lane / 16) * 8 + lane % 8;
}
// tr16 column (elements)
DQ_HD constexpr int lane_c(int lane) {
    return ((lane / 8) % 2) * 8;
}

// ---- workgroup decode: grid (HG, ntile + nhead, B * Hq / HG), x = the head in the group. With
// HG % 8 == 0 every head's workgroups run on XCD x % 8 (the dispatcher deals workgroups to the 8
// XCDs round robin), and the remap gives each XCD a contiguous range of HG / 8 heads.
DQ_HD constexpr int xcd_remap(int gx, int ngrp) {
    const unsigned x = (unsigned) gx, g = (unsigned) ngrp;
    const unsigned r = (x % 8u) * (g / 8u) + x / 8u;
    return (g % 8u == 0u) ? (int) r : gx;
}
// bh = bat*Hq + qh
DQ_HD constexpr int head_hb(int gx, int gz, int HG) {
    return gz * HG + xcd_remap(gx, HG);
}
DQ_HD constexpr int head_bat(int hb, int Hq) {
    return hb / Hq;
}
DQ_HD constexpr int head_qh(int hb, int bat, int Hq) {
    return hb - bat * Hq;
}
DQ_HD constexpr int head_kv(int qh, int G) {
    return qh / G;
}
// y < ntile: the 384-query tiles of [q_off, Sq), longest first; y == ntile (q_off == 256 only):
// the head [0, 256) as one 4-wave x 64-query workgroup
DQ_HD constexpr bool tile_is_head(int gy, int ntile) {
    return gy >= ntile;
}
DQ_HD constexpr int tile_q0g(int q_off, int ntile, int gy) {
    return q_off + (ntile - 1 - gy) * bqwg(MAIN_NQW);
}
DQ_HD constexpr int head_q0g(int q_off) {
    return q_off - bqwg(HEAD_NQW);
}
// q pair (0-based in the workgroup's nbg) of the k-th pair of wave wv: snake order over the waves
// (main tile w, 7 - w, 8 + w; head tile w, 7 - w), so the diagonal units' fully masked pairs spread
// over all waves
DQ_HD constexpr int wave_pair(int wv, int k, int nqw) {
    return 4 * k + ((k & 1) ? 3 - wv : wv);
}
// first query of the wave's 16-query sub-tile h (q pair h / 2, half h % 2)
DQ_HD constexpr int subtile_q0(int q0g, int wv, int h, int nqw) {
    return q0g + 32 * wave_pair(wv, h / 2, nqw) + 16 * (h % 2);
}
// first q pair of the workgroup (q0g % 32 == 0)
DQ_HD constexpr int tile_qp0(int q0g) {
    return q0g / 32;
}

// ---- kv range (kv32 rows) of the workgroup tile: the rows with an entry attended by one of its
// queries (bottom-right causal)
DQ_HD constexpr int kv_n32(int q0g, int nqw, int cshift, int nkvt, int causal) {
    int lim = floordiv(q0g + bqwg(nqw) + cshift + (KV32 - 1), KV32);
    lim     = imax(lim, 1);
    lim     = imin(lim, nkvt);
    return causal != 0 ? lim : nkvt;
}
// 64-kv units (n32 is even on every supported shape)
DQ_HD constexpr int n_units(int n32) {
    return (n32 + 1) / 2;
}
// units [0, n_full_units) have no fully masked block in the workgroup's span [qp0g, qp0g + nbg) in
// either kv32 row (unit u full iff dsl_qp_first(2 (2u + 1)) <= qp0g, i.e. 2u + 1 - cshift / 32 <=
// qp0g); causal only
DQ_HD constexpr int n_full_units(int qp0g, int cshift, int nu, int causal) {
    const int t = floordiv(qp0g + cshift / 32 + 1, 2);
    return causal != 0 ? imax(imin(t, nu), 0) : nu;
}
// units run by the steady-state body (no selects, no dummy op: every op it issues -- op(u + 3) --
// must be real: u + 3 < nu); the rest run the generic masked body, the last unit its own tail
DQ_HD constexpr int steady_units(int nfu, int nu) {
    return imin(imin(nfu, nu - 1), imax(nu - 3, 0));
}

// ---- dS TDM op: kv32 row r (block rows 2r, 2r + 1) x the stored q pairs [v0, blo + nbs) -> an
// LDS slot. 2-D: tile0 = nv * 512 elements (contiguous inside a block row), tile1 = 2 rows, row
// stride = the block-row length of row 2r (dslayout's reader hint: rows 2r and 2r + 1 share
// qp_first), 16 B of pad after every 256 B (block images of DSI_BLK_B). The leading q pairs (< v0)
// are not stored and not read: the op starts at v0 and lands right-aligned, so row 2r's block lb
// sits at slot + lb * DSI_BLK_B and row 2r + 1's at slot + (nbs + lb - sh) * DSI_BLK_B,
// sh = v0 - blo.
DQ_HD constexpr int dsop_v0(int blo, int r, int cshift, int causal) {
    return imax(blo, dsl_qp_first(2 * r, cshift, causal));
}
DQ_HD constexpr int dsop_nv(int blo, int nbs, int r, int cshift, int causal) {
    return blo + nbs - dsop_v0(blo, r, cshift, causal);
}
// The op's source (elements: dsl_blk(bh, 2r, v0) * 512) and row stride (dsl_row_len(2r) * 512) in
// branch-free SALU form for the hot path (dsl_skip's branches would split the WMMA runs that the
// descriptor is prepared in), valid for cshift >= 0 (the shape rules: seqlen_q <= seqlen_kv): with
// m = cshift / 32, dsl_qp_first(2r) = f2r = causal ? max(0, r - m) : 0, and the packing removes
// sum_{k < 2r} dsl_qp_first(k) = t (t + 1), t = causal ? max(r - m - 1, 0) : 0, blocks before kv16
// row 2r.
DQ_HD constexpr int f2r(int r, int m, int causal) {
    return causal != 0 ? imax(r - m, 0) : 0;
}
DQ_HD constexpr long long row_off2(int r, int nqp, int m, int causal) {
    const long long t = causal != 0 ? (long long) imax(r - m - 1, 0) : 0LL;
    return 2LL * (long long) r * (long long) nqp - t * (t + 1);
}
DQ_HD constexpr long long dsop_src_el_hot(long long bh, long long nblk, int r, int blo, int nqp,
                                          int m, int causal) {
    return (bh * nblk + row_off2(r, nqp, m, causal) +
            (long long) imax(blo - f2r(r, m, causal), 0)) *
           DS_BLK_EL;
}
DQ_HD constexpr long long dsop_stride_el_hot(int r, int nqp, int m, int causal) {
    return (long long) (nqp - f2r(r, m, causal)) * DS_BLK_EL;
}
DQ_HD constexpr int dsop_nv_hot(int blo, int nbs, int r, int m, int causal) {
    return blo + nbs - imax(blo, f2r(r, m, causal));
}
DQ_HD constexpr int dsop_lds(int slot, int nbs, int nv) {
    return slot + (nbs - nv) * DSI_BLK_B;
}
// <= nbs: reads stay in the slot
DQ_HD constexpr int dsop_sh(int nbs, int nv) {
    return nbs - imax(nv, 0);
}
// D# group-1 word 0 of a dS op: bf16, pad on, interval 64 dwords (enc 5), amount 4 dwords (enc 3)
constexpr unsigned DS_W0 = (1u << 16) | (1u << 20) | (5u << 22) | (3u << 25);

// ---- K TDM ops
// element offset of K[bat, kv, hkv, col]
DQ_HD constexpr long long k_el(int bat, int Skv, int kv, int Hkv, int hkv, int col) {
    return ((long long) (bat * Skv + kv) * (long long) Hkv + (long long) hkv) * (long long) D_QK +
           (long long) col;
}
DQ_HD constexpr long long k_stride_el(int Hkv) {
    return (long long) Hkv * (long long) D_QK;
}
// the K wave of half h moves rows [64u + 32h, +32) of unit u, all 192 columns (480-B LDS rows)
DQ_HD constexpr int c_kop_kv0(int u, int h) {
    return 64 * u + 32 * h;
}
DQ_HD constexpr int c_kstage(int u) {
    return (u % C_KDEPTH) * C_KSTG_B;
}
DQ_HD constexpr int c_kop_lds(int u, int h) {
    return c_kstage(u) + h * C_KHALF_B;
}
DQ_HD constexpr int c_kop_dim1(int Skv, int u, int h) {
    return imax(Skv - c_kop_kv0(u, h), 0);
}
// D# group-1 word 0 of a K op: bf16, pad on, interval 32 dwords (enc 4), amount 8 dwords (enc 7)
constexpr unsigned BK_W0 = (1u << 16) | (1u << 20) | (4u << 22) | (7u << 25);

// ---- dS ring
DQ_HD constexpr int c_dsstage(int u, int nqw) {
    return C_DS_OFF + (u % C_DSDEPTH) * c_dsstg_b(nqw);
}
DQ_HD constexpr int c_dsslot(int u, int nqw, int h) {
    return c_dsstage(u, nqw) + h * c_slot_b(nqw);
}

// ---- LDS read (burst) addresses: base VGPR = stage / slot offset + lane part; per-operand DS
// immediates
DQ_HD constexpr int c_k_lane(int lane) {
    return lane_r(lane) * C_KROW_B + lane_c(lane) * 2;
}
DQ_HD constexpr int c_k_imm(int dt, int h) {
    return dt * 32 + (dt / 4) * 32 + h * 16 * C_KROW_B;
}
// the tr16 lane reads chunk (kv row lane_r, query chunk 2 qhalf + lane_c / 8) of a block image, at
// dsl_chunk = qhalf * 512 + g * 16 with g = (lane_c / 8) * 16 + lane_r, i.e. image byte
// qhalf * 544 + (g / 16) * 272 + (g % 16) * 16 (16 B of pad after every 256 B): ds_lane(lane) +
// ds_half(qhalf)
DQ_HD constexpr int ds_lane(int lane) {
    const int g = (lane_c(lane) / 8) * 16 + lane_r(lane);
    return (g / 16) * DSI_GRP_B + (g % 16) * 16;
}
// image offset of the 16-query half qhalf of a block (its second 512 data bytes = 2 padded 256-B
// groups)
DQ_HD constexpr int ds_half(int qhalf) {
    return qhalf * 2 * DSI_GRP_B;
}
// row-1 image base of a slot whose op landed with shift sh (nbs blocks per row); the workgroup's
// NB_G pairs land in q order, the wave's sub-tile h reads pair wave_pair(wv, h / 2)
DQ_HD constexpr int ds_row1(int slot, int nbs, int sh) {
    return slot + (nbs - sh) * DSI_BLK_B;
}

// ---- dQ epilogue, one 16-row sub-tile at a time: per-wave row-major image [16][400 B] after the
// rings, whole-line readback
DQ_HD constexpr int epi_img(int wv) {
    return LDS_C + wv * EPI_IMG_B;
}
DQ_HD constexpr int epi_w_lane(int lane) {
    return lane_row(lane) * DQ_IMG_ROW_B + lane_half(lane) * 16;
}
DQ_HD constexpr int epi_w_imm(int dtile) {
    return dtile * 32;
}
DQ_HD constexpr int epi_r_lane(int lane) {
    return (lane / 8) * DQ_IMG_ROW_B + (lane % 8) * 16;
}
DQ_HD constexpr int epi_r_imm(int rb, int li) {
    return rb * 4 * DQ_IMG_ROW_B + li * 128;
}
// vec8 index of dq[bat, q0 + 4rb + l/8, qh, li*64 + (l%8)*8] (q0: the sub-tile's first query,
// rb < 4)
DQ_HD constexpr int dq_vec8(int bat, int Sq, int Hq, int qh, int q0, int lane, int rb, int li) {
    return ((bat * Sq + q0 + rb * 4 + lane / 8) * Hq + qh) * DV8_QK + lane % 8 +
           li * (DQ_LINE_EL / 8);
}

} // namespace dq
