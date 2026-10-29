/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// idx.h -- every index formula of bwd_hk_dqds2's recompute-free dQ kernel k_dqds_c (main 384-query
// tiles + the merged 256-query head tile), in ONE place.
//
//   dQ[q, :] = sum_kv dS[q, kv] K[kv, :]   (dS read from the S1 scratch in the dslayout v2
//   contract, scale included)
//
// constexpr __host__ __device__ functions: the HIP kernel (src/k_dqds_c.hip via
// include/dqds/dqds_impl.h) computes every global offset, LDS address, TDM descriptor field, ring
// stage, loop bound and zeroing predicate through these functions, and bounds_proof.py compiles
// THIS file for x86 and replays them over every proof shape via ctypes. dS offsets come only from
// include/dqds/dslayout.h (the writer's file, byte-identical copy, sha256 checked by
// bounds_proof.py and the gates).
//
// What changed against bwd_hk_dqds variant B (k_dqds_b, the base of this file):
//   * 4 waves per workgroup (block 128, 1 wave per SIMD, ONE workgroup per CU: LDS 274432 B): the
//   main tile is 384
//     queries (4 x 96, NQW 6) and shares ONE K ring, so every K tile moved by the TDM feeds 384
//     queries instead of 192 (K bytes per call / 2); the head tile [0, 256) is 4 x 64 queries (NQW
//     4); Sq % 384 must be 0 or 256.
//   * role-split TDM issue, ONE op per wave per 64-kv unit: waves 0 / 3 move dS kv32 row 2u + (w >>
//   1) for all NB_G q
//     pairs of the workgroup (2-D, <= 24 KB), waves 1 / 2 move K rows [64u + 32 (w >> 1), +32) (12
//     KB). Each CU TDM engine moves one dS op + one K op per unit (36 KB, balanced) whether warp
//     parity (HK cdna5 global_to_shared.cuh: warps {0, 2} share one engine, {1, 3} the other) or
//     the SIMD pair picks it.
//   * K ring and dS ring both 3 stages: every op is issued 2 units ahead (B: K 1 unit, dS 2), wait
//   1 before every
//     barrier, <= 2 TDM ops in flight per wave.
// The per-output accumulation (kv32 rows ascending, one WMMA per (sub-tile, d tile, kv32 row), the
// same A / B operand bytes, fully masked q pairs' operands selected to +0) is k_dqds_b's: dq is
// bitwise k_dqds_b's (the extra zero-operand WMMAs of the taller tile add +-0 to a non -0
// accumulator).
//
// WMMA orientation (dQ^T, bwd_hk_q1's): acc[qh][dt] (16 d x 16 q) += A (16 d x 32 kv) * B (32 kv x
// 16 q) with
//   A = K^T  : tr16 of the K image [kv][d]  (lane = d row,   K-dim = kv), bwd_hk_q1's bks()
//   B = dS^T : tr16 of the dS block images [kv16][q32] (lane = q column, K-dim = kv: kv 0..15 from
//   block row 2r,
//              kv 16..31 from block row 2r + 1)
//
// Integer semantics as bwd_hk_q1's idx.h (floordiv for possibly negative dividends; i32 where the
// kernel uses i32, the proof re-checks every value in Python integers). Mutation hooks: DQ_MUT
// selects one deliberately wrong formula (bounds_proof.py negative controls); never defined in a
// kernel build.
#pragma once

#if defined(__HIP__) || defined(__HIPCC__) || defined(__CUDACC__)
#define DQ_HD __host__ __device__ __attribute__((always_inline))
#else
#define DQ_HD
#endif

#ifndef DQ_MUT
#define DQ_MUT 0
#endif

#include "dqds/dslayout.h"

namespace dq {

// ---- layout constants
// -----------------------------------------------------------------------------------------
constexpr int D_QK      = 192;       // q/k head dim (dq row)
constexpr int DV8_QK    = D_QK / 8;  // vec8 tiles per dq row (24)
constexpr int NDT       = D_QK / 16; // 16-wide d tiles of dQ^T (12)
constexpr int WAVE      = 32;
constexpr int NWAVE     = 4;            // waves per workgroup (both tile kinds), 1 per SIMD
constexpr int BLOCK     = NWAVE * WAVE; // 128 threads
constexpr int KV32      = 32; // kv rows per WMMA step (one "kv32 row" r = kv16 block rows 2r, 2r+1)
constexpr int MAIN_NQW  = 6;  // main tile: 4 waves x 96 queries (6 sub-tiles of 16) = 384
constexpr int HEAD_NQW  = 4;  // merged head tile: 4 waves x 64 queries = 256 (q_off == 256 only)
constexpr int MAIN_ROWS = 16 * MAIN_NQW * NWAVE; // 384
constexpr int HEAD_ROWS = 16 * HEAD_NQW * NWAVE; // 256
constexpr int DS_BLK_EL = DSL_BLK_B / 2;         // 512 bf16 per dslayout block [16 kv][32 q]
// dS block images in LDS: the TDM pads 16 B after every 256 data bytes (16 dslayout v2 chunks), so the 32 tr16 lanes
// of one operand (16 chunks of one 256-B group + 16 of the next) hit every bank exactly twice (64-B padding: 3 times)
constexpr int DSI_GRP_B = 256 + 16;                // one padded 256-B group (272 B)
constexpr int DSI_ROW_B = DSI_GRP_B;               // (proof-table name)
constexpr int       DSI_BLK_B    = 4 * DSI_GRP_B;  // one block image (1088 B)
constexpr unsigned  TDIM0        = 0x7FFFFFFFu; // inner extent: never clamps (bwd_hk_q1's TDM_DIM0)
constexpr int       DQ_IMG_ROW_B = 400; // dQ epilogue image row (bwd_hk_q1: 384 B data + 16 B pad)
constexpr int       DQ_LINE_EL   = 64;  // bf16 per 128-B dq line
constexpr long long DQ_EXTENT_B  = 1LL
                                  << 30; // dq buffer resource extent (impl asserts q/dq <= 1 GiB)
constexpr int RSRC_FLAGS = 0x27000;      // bwd_hk_q1's V# word 3

// ---- the rings (unit u = kv rows [64u, 64u + 64) = kv32 rows 2u, 2u + 1; ONE workgroup barrier
// P_u per unit) ------ K ring: C_KDEPTH stages of 64 rows x 480 B (k_dqds_b's K image: 192 columns,
// 32 B pad every 128 B, so the 16 rows x
//   2 column halves of a half-wave K^T tr16 hit 16 distinct 4-bank groups); half h = rows [32h, 32h
//   + 32) of the unit, written by the K wave of half h (wave 1 + h; one 12-KB op).
// dS ring: C_DSDEPTH stages of 2 slots; slot h = kv32 row 2u + h for ALL NB_G q pairs of the
// workgroup (2 block rows x
//   NB_G block images of 1088 B), written by the dS wave of half h (wave 3h; one 2-D op, <= 24 KB
//   at the main tile).
// Op order per wave: op(0) op(1) | P_0 | op(2) | P_1 | op(3) ...: op(v) = the wave's role op of
// unit v. P_{u+1} sits in
//   unit u after its last ring read; before it a wave needs op(u + 1) retired and only op(u + 2)
//   may stay in flight: wait C_TW = 1; after it the wave issues op(u + 3) into unit u's (now free)
//   stages. In flight: <= 2 per wave. Ops of units >= nu are 16-B dummies (count stays uniform, no
//   duplicate reads).
constexpr int C_KROW_B       = 480;
constexpr int C_KHALF_B      = KV32 * C_KROW_B; // 15360 (one K wave's 32 rows)
constexpr int C_KSTG_B       = 2 * C_KHALF_B;   // 30720 (64 rows)
constexpr int C_KDEPTH       = 3;
constexpr int C_DSDEPTH      = 3;
constexpr int C_DS_OFF       = C_KDEPTH * C_KSTG_B; // 92160
constexpr int C_TW           = 1;
constexpr int C_AHEAD        = 2; // units between an op's issue barrier and its use barrier
constexpr int C_MAX_INFLIGHT = 2;

// ---- integer helpers
// --------------------------------------------------------------------------------------------
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
// ----------------------------------------------
DQ_HD constexpr int bqw(int nqw) {
    return 16 * nqw;
} // queries per wave
DQ_HD constexpr int bqwg(int nqw) {
    return 16 * nqw * NWAVE;
} // per workgroup
DQ_HD constexpr int nbw(int nqw) {
    return nqw / 2;
} // dS q pairs (blocks) per wave
DQ_HD constexpr int nbg(int nqw) {
    return NWAVE * nqw / 2;
} // per workgroup
DQ_HD constexpr int c_slot_b(int nqw) {
    return 2 * nbg(nqw) * DSI_BLK_B;
} // one kv32 row x NB_G block images
DQ_HD constexpr int c_dsstg_b(int nqw) {
    return 2 * c_slot_b(nqw);
} // 2 slots (kv32 rows 2u, 2u + 1)
constexpr int LDS_C = C_DS_OFF + C_DSDEPTH * c_dsstg_b(MAIN_NQW); // 248832: the rings
constexpr int EPI_IMG_B = 16 * DQ_IMG_ROW_B; // one wave's dQ epilogue image: 16 rows x 400 B, after the rings
DQ_HD constexpr int lds_bytes() {
    return LDS_C + NWAVE * EPI_IMG_B; // 274432: one workgroup per CU
}

// ---- roles (wave-uniform): waves 0 / 3 issue dS (kv32 row 2u + half), waves 1 / 2 issue K (rows
// 32 half) ------------ half = w >> 1; dS iff (w & 1) == (w >> 1). Each CU TDM engine then carries
// one dS op and one K op per unit under BOTH candidate engine maps: warp parity ({0, 2} / {1, 3},
// HipKittens cdna5 global_to_shared.cuh -- a comment, unmeasured) and SIMD pairs ({0, 1} / {2, 3}).
DQ_HD constexpr bool c_role_ds(int wv) {
#if DQ_MUT == 17
    return wv < 2; // MUTATION: both dS waves on one SIMD pair's engine
#else
    return (wv & 1) == (wv >> 1);
#endif
}
DQ_HD constexpr int c_half(int wv) {
#if DQ_MUT == 18
    return 0; // MUTATION: every wave moves half 0
#else
    return wv >> 1;
#endif
}

// ---- lane decomposition (bwd_hk_q1)
// ---------------------------------------------------------------------------
DQ_HD constexpr int lane_row(int lane) {
    return lane % 16;
}
DQ_HD constexpr int lane_half(int lane) {
    return lane / 16;
}
DQ_HD constexpr int lane_r(int lane) {
    return (lane / 16) * 8 + lane % 8;
} // tr16 row
DQ_HD constexpr int lane_c(int lane) {
    return ((lane / 8) % 2) * 8;
} // tr16 column (elements)

// ---- workgroup decode (bwd_hk_q2: head-group launch, XCD-major remap, longest-first, merged head)
// ------------
DQ_HD constexpr int xcd_remap(int gx, int ngrp) {
    const unsigned x = (unsigned) gx, g = (unsigned) ngrp;
#if DQ_MUT == 1
    const unsigned r = (x % 8u) * (g / 8u); // MUTATION: drops + x/8
#else
    const unsigned r = (x % 8u) * (g / 8u) + x / 8u;
#endif
    return (g % 8u == 0u) ? (int) r : gx;
}
DQ_HD constexpr int head_hb(int gx, int gz, int HG) {
    return gz * HG + xcd_remap(gx, HG);
} // bh = bat*Hq + qh
DQ_HD constexpr int head_bat(int hb, int Hq) {
    return hb / Hq;
}
DQ_HD constexpr int head_qh(int hb, int bat, int Hq) {
    return hb - bat * Hq;
}
DQ_HD constexpr int head_kv(int qh, int G) {
    return qh / G;
}
// grid (HG, ntile + nhead, B*Hq/HG): y < ntile the 384-query tiles of [q_off, Sq), longest first; y
// == ntile (only when q_off == 256) the head [0, 256) as one 4-wave x 64-query workgroup (bwd_hk_q2
// (b))
DQ_HD constexpr bool tile_is_head(int gy, int ntile) {
#if DQ_MUT == 2
    return gy >= ntile - 1; // MUTATION: the last main tile runs as a head
#else
    return gy >= ntile;
#endif
}
DQ_HD constexpr int tile_q0g(int q_off, int ntile, int gy) {
    return q_off + (ntile - 1 - gy) * bqwg(MAIN_NQW);
}
DQ_HD constexpr int head_q0g(int q_off) {
    return q_off - bqwg(HEAD_NQW);
}
// q pair (0-based in the workgroup's nbg) of the k-th pair of wave wv: snake order over the waves (main tile
// w, 7 - w, 8 + w; head tile w, 7 - w), so the diagonal units' fully masked pairs spread over all waves
DQ_HD constexpr int wave_pair(int wv, int k, int nqw) {
#if DQ_MUT == 3
    return 4 * k + wv; // MUTATION: two waves share pairs
#else
    return 4 * k + ((k & 1) ? 3 - wv : wv);
#endif
}
// first query of the wave's 16-query sub-tile h (q pair h / 2, half h % 2)
DQ_HD constexpr int subtile_q0(int q0g, int wv, int h, int nqw) {
    return q0g + 32 * wave_pair(wv, h / 2, nqw) + 16 * (h % 2);
}
DQ_HD constexpr int tile_qp0(int q0g) {
    return q0g / 32;
} // first q pair of the workgroup (q0g % 32 == 0)

// ---- kv range (kv32 rows) of the WORKGROUP tile: bwd_hk_q1's kv_nkvt_eff with BQWG = 64 nqw
// -------------------
DQ_HD constexpr int kv_n32(int q0g, int nqw, int cshift, int nkvt, int causal) {
    int lim = floordiv(q0g + bqwg(nqw) + cshift + (KV32 - 1), KV32);
#if DQ_MUT == 4
    lim = lim - 2; // MUTATION: drops the last unit
#endif
    lim = imax(lim, 1);
    lim = imin(lim, nkvt);
    return causal != 0 ? lim : nkvt;
}
DQ_HD constexpr int n_units(int n32) {
    return (n32 + 1) / 2;
} // 64-kv units (n32 is even on every plan shape)
// units [0, n_full_units) have no fully masked block in the workgroup's span [qp0g, qp0g + nbg) in
// either kv32 row (unit u full iff dsl_qp_first(2 (2u + 1)) <= qp0g, i.e. 2u + 1 - cshift/32 <=
// qp0g); causal only
DQ_HD constexpr int n_full_units(int qp0g, int cshift, int nu, int causal) {
#if DQ_MUT == 5
    const int t = floordiv(qp0g + cshift / 32 + 3, 2); // MUTATION: a masked unit runs unmasked
#else
    const int t = floordiv(qp0g + cshift / 32 + 1, 2);
#endif
    return causal != 0 ? imax(imin(t, nu), 0) : nu;
}
// units run by the steady-state body (no selects, no dummy op: every op it issues -- op(u + 3) --
// must be real: u + 3 < nu); the rest run the generic masked body, the last unit its own tail
DQ_HD constexpr int steady_units(int nfu, int nu) {
#if DQ_MUT == 16
    return imin(imin(nfu, nu - 1), imax(nu - 2, 0)); // MUTATION: issues a unit past nu
#else
    return imin(imin(nfu, nu - 1), imax(nu - 3, 0));
#endif
}

// ---- dS zeroing predicate: the B operand of q pair qp at kv32 row r is selected to zero iff the
// block holds no attended entry (both of its kv16 rows are unstored in dslayout, dsl_qp_first(2r)
// == dsl_qp_first(2r + 1) because cshift % 32 == 0) or r is a padding row (r >= n32; never on a
// plan shape, kept as a guard)
DQ_HD constexpr bool ds_zero(int qp, int r, int n32, int cshift, int causal) {
#if DQ_MUT == 6
    return r >= n32 ||
           qp <= dsl_qp_first(2 * r, cshift, causal); // MUTATION: zeroes the diagonal block
#else
    return r >= n32 || qp < dsl_qp_first(2 * r, cshift, causal);
#endif
}

// ---- dS TDM op: kv32 row r (block rows 2r, 2r + 1) x stored q pairs [v0, blo + nbs) -> LDS slot
// ----------------- 2-D: tile0 = nv * 512 elements (contiguous inside a block row), tile1 = 2 rows,
// row stride = the block-row length of row 2r (dslayout reader hint: rows 2r and 2r + 1 share
// qp_first), pad 8 elements every 32 (80-B LDS rows). The leading q pairs (< v0) are not stored and
// are NOT read: the op starts at v0 and lands right-aligned, so row 2r's block lb sits at slot +
// lb*1280 and row 2r+1's at slot + (nbs + lb - sh)*1280, sh = v0 - blo.
DQ_HD constexpr int dsop_v0(int blo, int r, int cshift, int causal) {
    return imax(blo, dsl_qp_first(2 * r, cshift, causal));
}
DQ_HD constexpr int dsop_nv(int blo, int nbs, int r, int cshift, int causal) {
    return blo + nbs - dsop_v0(blo, r, cshift, causal);
}
DQ_HD constexpr long long dsop_src_el(long long bh, int r, int v0, int Sq, int Skv, int causal) {
#if DQ_MUT == 7
    return dsl_blk(bh, 2 * r, v0, Sq, Skv, causal) * DS_BLK_EL +
           DS_BLK_EL; // MUTATION: one block late
#else
    return dsl_blk(bh, 2 * r, v0, Sq, Skv, causal) * DS_BLK_EL;
#endif
}
DQ_HD constexpr long long dsop_stride_el(int r, int Sq, int cshift, int causal) {
#if DQ_MUT == 8
    return (long long) dsl_nqp(Sq) * DS_BLK_EL; // MUTATION: the dense row length
#else
    return (long long) dsl_row_len(2 * r, Sq, cshift, causal) * DS_BLK_EL;
#endif
}
// The same source / stride in branch-free SALU form for the kernel's hot path (dsl_skip's branches
// split the WMMA runs into blocks when the descriptor is prepared inside them), valid for cshift >=
// 0 (the plan refuses cshift < 0): with m = cshift / 32, f(2r) = causal ? max(0, r - m) : 0 and the
// packing removes sum_{k < 2r} f(k) = t (t + 1), t = causal ? max(r - m - 1, 0) : 0, blocks before
// kv16 row 2r. bounds_proof.py D1 checks every op of every proof shape against dsl_blk (the
// writer's formula) block by block.
DQ_HD constexpr int f2r(int r, int m, int causal) {
    return causal != 0 ? imax(r - m, 0) : 0;
}
DQ_HD constexpr long long row_off2(int r, int nqp, int m, int causal) {
    const long long t = causal != 0 ? (long long) imax(r - m - 1, 0) : 0LL;
    return 2LL * (long long) r * (long long) nqp - t * (t + 1);
}
DQ_HD constexpr long long dsop_src_el_hot(long long bh, long long nblk, int r, int blo, int nqp,
                                          int m, int causal) {
#if DQ_MUT == 15
    return (bh * nblk + row_off2(r, nqp, m, causal) +
            (long long) imax(blo - f2r(r, m, causal), 0)) *
               DS_BLK_EL +
           DS_BLK_EL; // MUTATION: one block late (hot path)
#else
    return (bh * nblk + row_off2(r, nqp, m, causal) +
            (long long) imax(blo - f2r(r, m, causal), 0)) *
           DS_BLK_EL;
#endif
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
DQ_HD constexpr int dsop_sh(int nbs, int nv) {
    return nbs - imax(nv, 0);
} // <= nbs: reads stay in the slot
// D# group-1 word 0 of a dS op: bf16, pad on, interval 64 dw (enc 5), amount 4 dw (enc 3)
constexpr unsigned DS_W0 = (1u << 16) | (1u << 20) | (5u << 22) | (3u << 25);
// LDS byte of element (block row rr in {0,1}, stored block j, kv k, q c) of an op landing at L
// (stream pad model: destination byte b -> b + (b / 256) * 16)
DQ_HD constexpr int dsop_elem_lds(int L, int nv, int rr, int j, int k, int c) {
    return L + ((rr * nv + j) * DSL_BLK_B + dsl_chunk(k, c / 8) + 2 * (c % 8)) / 256 * DSI_GRP_B +
           (dsl_chunk(k, c / 8) + 2 * (c % 8)) % 256;
}

// ---- K TDM ops
// ------------------------------------------------------------------------------------------------
// element offset of K[bat, kv, hkv, col]
DQ_HD constexpr long long k_el(int bat, int Skv, int kv, int Hkv, int hkv, int col) {
    return ((long long) (bat * Skv + kv) * (long long) Hkv + (long long) hkv) * (long long) D_QK +
           (long long) col;
}
DQ_HD constexpr long long k_stride_el(int Hkv) {
    return (long long) Hkv * (long long) D_QK;
}
// K wave of half h moves rows [64u + 32h, +32) of unit u, all 192 columns, pad 8 dw every 32 dw
// (480-B rows)
DQ_HD constexpr int c_kop_kv0(int u, int h) {
#if DQ_MUT == 9
    return 64 * u; // MUTATION: both K waves move rows 0..31
#else
    return 64 * u + 32 * h;
#endif
}
DQ_HD constexpr int c_kstage(int u) {
#if DQ_MUT == 10
    return (u % (C_KDEPTH - 1)) * C_KSTG_B; // MUTATION: a 2-stage K ring
#else
    return (u % C_KDEPTH) * C_KSTG_B;
#endif
}
DQ_HD constexpr int c_kop_lds(int u, int h) {
    return c_kstage(u) + h * C_KHALF_B;
}
DQ_HD constexpr int c_kop_dim1(int Skv, int u, int h) {
    return imax(Skv - c_kop_kv0(u, h), 0);
}
constexpr unsigned BK_W0 = (1u << 16) | (1u << 20) | (4u << 22) | (7u << 25);
// LDS byte (from the op's start) of K element (row k, col c) of a K op: 480-B rows, 32 B after
// every 128 B
DQ_HD constexpr int c_k_elem_lds(int k, int c) {
    return k * C_KROW_B + 2 * c + (2 * c / 128) * 32;
}

// ---- dS ring
// ---------------------------------------------------------------------------------------------------
DQ_HD constexpr int c_dsstage(int u, int nqw) {
#if DQ_MUT == 11
    return C_DS_OFF + (u % (C_DSDEPTH - 1)) * c_dsstg_b(nqw); // MUTATION: a 2-stage dS ring
#else
    return C_DS_OFF + (u % C_DSDEPTH) * c_dsstg_b(nqw);
#endif
}
DQ_HD constexpr int c_dsslot(int u, int nqw, int h) {
    return c_dsstage(u, nqw) + h * c_slot_b(nqw);
}

// ---- LDS read (burst) addresses: base VGPR = stage/slot offset + lane part; per-operand DS
// immediates ---------
DQ_HD constexpr int c_k_lane(int lane) {
    return lane_r(lane) * C_KROW_B + lane_c(lane) * 2;
}
DQ_HD constexpr int c_k_imm(int dt, int h) {
#if DQ_MUT == 12
    return dt * 32 + h * 16 * C_KROW_B; // MUTATION: ignores the in-row pads
#else
    return dt * 32 + (dt / 4) * 32 + h * 16 * C_KROW_B;
#endif
}
// dslayout v2: the tr16 lane reads chunk (kv row lane_r, query chunk 2 qhalf + lane_c / 8) of a block image,
// at dsl_chunk = qhalf * 512 + g * 16 with g = (lane_c / 8) * 16 + lane_r, i.e. image byte qhalf * 544 + (g / 16) * 272
// + (g % 16) * 16 (the TDM pads 16 B after every 256 B): ds_lane(lane) + ds_half(qhalf)
DQ_HD constexpr int ds_lane(int lane) {
    const int g = (lane_c(lane) / 8) * 16 + lane_r(lane);
    return (g / 16) * DSI_GRP_B + (g % 16) * 16;
}
// image offset of the 16-query half qhalf of a block (its second 512 data bytes = 2 padded 256-B groups)
DQ_HD constexpr int ds_half(int qhalf) {
    return qhalf * 2 * DSI_GRP_B;
}
// operand of query sub-tile qh (q pair lb = qh / 2 relative to the reader's first block, 16-query
// half qh % 2)
DQ_HD constexpr int ds_imm(int qh) {
    return (qh / 2) * DSI_BLK_B + ds_half(qh % 2);
}
// row-1 image base of a slot whose op landed with shift sh (nbs blocks per row)
DQ_HD constexpr int ds_row1(int slot, int nbs, int sh) {
#if DQ_MUT == 13
    return slot + nbs * DSI_BLK_B; // MUTATION: ignores the right-aligned shift
#else
    return slot + (nbs - sh) * DSI_BLK_B;
#endif
}
// the workgroup's NB_G pairs land in q order in a slot; the wave's sub-tile h reads pair wave_pair(wv, h / 2)

// ---- dQ epilogue (bwd_hk_q1's, one 16-row sub-tile at a time): per-wave row-major image [16][400 B] after the
// rings, whole-line readback ----
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
#if DQ_MUT == 14
    return rb * 4 * DQ_IMG_ROW_B + li * 64; // MUTATION: half lines
#else
    return rb * 4 * DQ_IMG_ROW_B + li * 128;
#endif
}
// vec8 index of dq[bat, q0 + 4rb + l/8, qh, li*64 + (l%8)*8] (q0: the sub-tile's first query, rb < 4)
DQ_HD constexpr int dq_vec8(int bat, int Sq, int Hq, int qh, int q0, int lane, int rb, int li) {
    return ((bat * Sq + q0 + rb * 4 + lane / 8) * Hq + qh) * DV8_QK + lane % 8 +
           li * (DQ_LINE_EL / 8);
}

// ---- constants for the host proof (bounds_proof.py reads them through ctypes)
// -------------------------------
DQ_HD constexpr long long kconst(int i) {
    return i == 0    ? D_QK
           : i == 1  ? NDT
           : i == 2  ? MAIN_NQW
           : i == 3  ? HEAD_NQW
           : i == 4  ? DS_BLK_EL
           : i == 5  ? DSI_ROW_B
           : i == 6  ? DSI_BLK_B
           : i == 7  ? NWAVE
           : i == 8  ? BLOCK
           : i == 9  ? MAIN_ROWS
           : i == 10 ? HEAD_ROWS
           : i == 11 ? C_KROW_B
           : i == 12 ? C_KHALF_B
           : i == 13 ? C_KSTG_B
           : i == 14 ? C_KDEPTH
           : i == 15 ? C_DSDEPTH
           : i == 16 ? C_DS_OFF
           : i == 17 ? C_TW
           : i == 18 ? C_AHEAD
           : i == 19 ? LDS_C
           : i == 20 ? C_MAX_INFLIGHT
           : i == 21 ? (long long) DS_W0
           : i == 22 ? (long long) BK_W0
           : i == 23 ? DQ_IMG_ROW_B
           : i == 24 ? DQ_EXTENT_B
           : i == 25 ? (long long) TDIM0
           : i == 26 ? DSL_VERSION
           : i == 27 ? DSL_SCALED
                     : -1;
}

} // namespace dq
