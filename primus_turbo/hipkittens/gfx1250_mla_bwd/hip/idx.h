/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// idx.h -- every index / address formula of src/k_dkdv96.hip (bwd_hk_k3), as constexpr __host__
// __device__ functions of scalars. The kernel calls exactly these; bounds_proof.py compiles this
// file for the host (extern "C" wrappers, ctypes) and replays them over every launched shape.
// Arithmetic widths follow FlyDSL's k_dkdv64 (bwd_r10_c kernels.py _dkdv_impl): i32 everywhere
// except the TDM global element offsets (i64).
//
// Inherited from bwd_hk_k1 unchanged (the k1_* functions; the ring, TDM, readback, tr16, K/V
// fragment and NL/ND math does not depend on the kv rows per wave). New for k3 (the k3_* functions
// at the end): the 96-row block decode with 64-row tail blocks, the causal split of a block of
// `rows` kv rows, and the per-wave epilogue images of nkv 16-row sub-tiles. k1's 64-row versions
// (k1_kv0g, k1_kv0, k1_nmaskp, k1_epi_wr, k1_epi_rd) are kept: bounds_proof.py ties the k3
// functions at rows 64 / nkv 2 to them.
//
// Names: lane in [0, 32), row = lane % 16, half = lane / 16; wave w in {0, 1} of a block of 32*nkv
// kv rows owns kv rows kv0 = kv0g + 16*nkv*w; stage_off = s * K1_QDO_B (s in 0..2) is a byte offset
// in the workgroup's LDS.
#pragma once
#include <stdint.h>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#define K1_FN static inline constexpr __host__ __device__

// ---- layout (single source; bounds_proof.py reads them through k1_const)
// ---------------------------------
#define K1_D_QK 192                        // q / k / dq / dk head dim
#define K1_D_V 128                         // v / o / do / dv head dim
#define K1_XK (K1_D_QK * 2 + 16)           // 400 B: padded LDS row of a Q tile / dK image
#define K1_XV (K1_D_V * 2 + 16)            // 272 B: padded LDS row of a dO tile / dV image
#define K1_QOFF (32 * K1_XV)               // 8704: Q tile offset inside a ring stage
#define K1_QDO_B (32 * (K1_XV + K1_XK))    // 21504: one ring stage ([32][D_V] dO + [32][D_QK] Q)
#define K1_DEPTH 3                         // ring stages
#define K1_RING_B (K1_DEPTH * K1_QDO_B)    // 64512
#define K1_EPI_DK (32 * K1_XV)             // 8704: dK image offset in a wave's epilogue area
#define K1_EPI_W (32 * K1_XV + 32 * K1_XK) // 21504: one wave's dV + dK images
#define K1_NW 2                            // waves per workgroup
#define K1_LDS (K1_RING_B > K1_NW * K1_EPI_W ? K1_RING_B : K1_NW * K1_EPI_W) // 64512 B static LDS
#define K1_TDM_ROWS 16               // rows per wave per TDM op (32-row tile, 2 waves)
#define K1_TW 3                      // tensorcnt that retires all but the newest stage
#define K1_NEG (-3.0e38f)            // masked logit (exp2 -> 0)
#define K1_LOG2E 1.4426950408889634f // f32 0x3FB8AA3B

K1_FN int k1_const(int which) {
    return which == 0    ? K1_D_QK
           : which == 1  ? K1_D_V
           : which == 2  ? K1_XK
           : which == 3  ? K1_XV
           : which == 4  ? K1_QOFF
           : which == 5  ? K1_QDO_B
           : which == 6  ? K1_DEPTH
           : which == 7  ? K1_RING_B
           : which == 8  ? K1_EPI_DK
           : which == 9  ? K1_EPI_W
           : which == 10 ? K1_NW
           : which == 11 ? K1_LDS
           : which == 12 ? K1_TDM_ROWS
           : which == 13 ? K1_TW
                         : -1;
}

// ---- workgroup decode (head-group launch, bwd_r4_a): grid (HG, Skv/64, B*Hkv/HG)
// ------------------------
K1_FN int k1_hb(int bz, int bx, int HG) {
    return bz * HG + bx;
}
K1_FN int k1_bat(int bz, int bx, int HG, int Hkv) {
    return k1_hb(bz, bx, HG) / Hkv;
}
K1_FN int k1_hkv(int bz, int bx, int HG, int Hkv) {
    return k1_hb(bz, bx, HG) - k1_bat(bz, bx, HG, Hkv) * Hkv;
}
K1_FN int k1_kv0g(int bid) {
    return bid * 64;
}
K1_FN int k1_kv0(int bid, int wv) {
    return bid * 64 + wv * 32;
}

// ---- causal split at workgroup level (bottom-right: query q sees kv <= q + cshift)
// -----------------------
K1_FN int k1_qp_start(int kv0g, int cshift, int causal) {
    const int c  = kv0g - cshift;
    const int qs = (c < 0 ? 0 : c) / 32;
    return causal != 0 ? qs : 0;
}
K1_FN int k1_nmaskp(int kv0g, int cshift, int causal, int nqt) {
    const int nqt2    = nqt / 2;
    const int qs      = k1_qp_start(kv0g, cshift, causal);
    const int u       = kv0g + 63 - cshift;
    const int qsu     = (u + 31) / 32;
    int       qsf     = u < 0 ? 0 : qsu;
    qsf               = qsf < nqt2 ? qsf : nqt2;
    int nm            = qsf - qs;
    nm                = nm < 0 ? 0 : nm;
    const int nqp_eff = nqt2 - qs;
    nm                = nm < nqp_eff ? nm : nqp_eff;
    return causal != 0 ? nm : 0;
}
K1_FN int k1_nqp_eff(int kv0g, int cshift, int causal, int nqt) {
    return nqt / 2 - k1_qp_start(kv0g, cshift, causal);
}

// ---- division-free (query pair, q head) counters: flat ii = q*G + g
// ---------------------------------------
K1_FN int k1_wrap_q(int qc, int gc, int G) {
    return gc + 1 < G ? qc : qc + 1;
}
K1_FN int k1_wrap_g(int qc, int gc, int G) {
    return gc + 1 < G ? gc + 1 : 0;
}
K1_FN int k1_clampqt(int t, int nqt2) {
    t = t < nqt2 ? t : nqt2 - 1;
    return t < 0 ? 0 : t;
}
K1_FN int k1_ncur(int cur) {
    return cur == 2 * K1_QDO_B ? 0 : cur + K1_QDO_B;
}
// full loop iteration ii (counters (qi, gh) == decomposition of ii) of n: the NL/ND prefetch tile
// min(ii+1, n-1) and the TDM prefetch tile min(ii+3, n-1), as (q, g) relative to the loop's first
// pair (every ternary below selects between already-computed locals, so clang emits selects, not
// branches)
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
// prologue tiles 1 and 2 of the full loop (n iterations): decomposition of min(j, n-1), j = 1, 2
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

// ---- K / V fragments: vec8 (16-B) tile index of lane `lane`, kv sub-tile kh, k-step dt, piece u
// --------- gfrag: lane (row, half) holds kv row kv0 + kh*16 + row, d = dt*32 + half*8 + [0, 8) (u
// = 0) and dt*32 + 16 + half*8 + [0, 8) (u = 1). Byte offset = 16 * tile (i32), buffer extent nk_b
// / nv_b.
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

// ---- NL / ND (k_delta's -LOG2E*lse, -scale*delta, [B, Hq, Sq] fp32): 8 vec4 loads per (pair, q
// head) ---- entry e = hh*4 + which*2 + u (which 0 = NL, 1 = ND): byte soff + hh*64 + u*16 (SGPR) +
// 32*half (VGPR)
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

// ---- TDM: one ring stage = op 0 dO [32][128] (stride 272 B), op 1 Q cols [0,128), op 2 Q cols
// [128,192) (stride 400 B); wave w moves rows [16w, 16w+16) of every op (num_warps 2). Element
// offsets are i64.
K1_FN int k1_tdm_w(int op) {
    return op == 2 ? 64 : 128;
} // tile columns
K1_FN int k1_tdm_d(int op) {
    return op == 0 ? K1_D_V : K1_D_QK;
} // tensor row length (elements)
K1_FN int k1_tdm_c0(int op) {
    return op == 2 ? 128 : 0;
} // first column
K1_FN int k1_tdm_xrow(int op) {
    return op == 0 ? K1_XV : K1_XK;
} // LDS row stride (bytes)
K1_FN int k1_tdm_pad(int op) {
    return (k1_tdm_xrow(op) - 2 * k1_tdm_w(op)) / 2;
} // pad elements per row
K1_FN int64_t k1_tdm_row0(int bat, int Sq, int q0, int Hq, int qh) {
    return (int64_t) (bat * Sq + q0) * (int64_t) Hq + (int64_t) qh;
}
K1_FN int64_t k1_tdm_gelem(int bat, int Sq, int q0, int Hq, int qh, int op, int wv) {
    return k1_tdm_row0(bat, Sq, q0, Hq, qh) * k1_tdm_d(op) + k1_tdm_c0(op) +
           (int64_t) (wv % 2) * K1_TDM_ROWS * (int64_t) Hq * k1_tdm_d(op);
}
K1_FN int k1_tdm_stride(int Hq, int op) {
    return Hq * k1_tdm_d(op);
} // outer stride (elements)
K1_FN int k1_tdm_lds(int stage_off, int op, int wv) {
    return stage_off + (op == 0 ? 0 : K1_QOFF) + k1_tdm_c0(op) * 2 +
           (wv % 2) * K1_TDM_ROWS * k1_tdm_xrow(op);
}
// outer-extent field: FlyDSL wrote max(Sq - q0 - 16w, 0); k1 writes the slice height K1_TDM_ROWS
// (prebuilt descriptor; bounds_proof K1 proves FlyDSL's value >= 16 for every issued tile, so no
// live row is clamped)
K1_FN int k1_tdm_ext_fly(int Sq, int q0, int wv) {
    const int e = Sq - q0 - (wv % 2) * K1_TDM_ROWS;
    return e < 0 ? 0 : e;
}
K1_FN int k1_tdm_ext(void) {
    return K1_TDM_ROWS;
}
// descriptor group 1 word 0: data size 2 B (1 << 16), pad enable (1 << 20), pad interval log2(w/2
// dwords)-1
// (<< 22), pad amount pad/2 dwords - 1 (<< 25) (HK tdm.cuh build_tdm_descriptor_2d == FlyDSL's
// constants 0x07510000 / 0x47510000 / 0x87110000)
K1_FN int k1_ilog2(int x) {
    return x <= 1 ? 0 : 1 + k1_ilog2(x / 2);
}
K1_FN uint32_t k1_tdm_w0(int op) {
    return (1u << 16) | (1u << 20) | ((uint32_t) (k1_ilog2(k1_tdm_w(op) / 2) - 1) << 22) |
           ((uint32_t) (k1_tdm_pad(op) / 2 - 1) << 25);
}

// ---- LDS reads of a stage (address = stage_off + lane part + compile-time constant; the kernel
// adds the constant to a per-iteration base pointer so it lands in the DS offset field)
// ---------------------------- readback (ds_load_b128): entry hh*20 + isq?(dt*2+u):(12+dt*2+u);
// lane (row, half) reads row hh*16 + row, bytes half*16 + dt*64 + u*32 of the Q (isq 1, dt < 6) or
// dO (isq 0, dt < 4) tile
K1_FN int k1_rd_lane(int isq, int lane) {
    return (lane % 16) * (isq ? K1_XK : K1_XV) + (lane / 16) * 16;
}
K1_FN int k1_rd_const(int hh, int isq, int dt, int u) {
    return (isq ? K1_QOFF : 0) + hh * 16 * (isq ? K1_XK : K1_XV) + dt * 64 + u * 32;
}
K1_FN int k1_rd(int stage_off, int hh, int isq, int dt, int u, int lane) {
    return stage_off + k1_rd_lane(isq, lane) + k1_rd_const(hh, isq, dt, u);
}
// B-operand transposed reads (ds_load_tr16_b128): dtile d (dO d < 8, Q d < 12), piece p (rows +16p)
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
K1_FN int k1_tr(int stage_off, int isq, int d, int p, int lane) {
    return stage_off + k1_tr_lane(isq, lane) + k1_tr_const(isq, d, p);
}

// ---- epilogue (bwd_r9_a): per-wave row-major dV [32][D_V] (stride 272) at w*EPI_W, dK [32][D_QK]
// (400) at w*EPI_W + EPI_DK; C fragment (kh, d) of lane (row, half) = kv row kh*16 + row, cols d*16
// + half*8 + [0, 8)
K1_FN int k1_epi_wr(int wv, int isk, int kh, int d, int lane) {
    const int x = isk ? K1_XK : K1_XV;
    return wv * K1_EPI_W + (isk ? K1_EPI_DK : 0) + (lane % 16) * x + (lane / 16) * 16 +
           kh * 16 * x + d * 32;
}
// whole-line readback: lane l reads kv row 4rb + l/8, bytes li*128 + (l%8)*16 of the image
K1_FN int k1_epi_rd(int wv, int isk, int rb, int li, int lane) {
    const int x = isk ? K1_XK : K1_XV;
    return wv * K1_EPI_W + (isk ? K1_EPI_DK : 0) + (lane / 8) * x + (lane % 8) * 16 + rb * 4 * x +
           li * 128;
}
// its global store: vec8 index of (kv row kv0 + 4rb + l/8, d = li*64 + (l%8)*8) in dv / dk
K1_FN int k1_epi_g(int bat, int Skv, int Hkv, int hkv, int kv0, int isk, int rb, int li, int lane) {
    const int base = isk ? k1_kbase(bat, Skv, Hkv, hkv) : k1_vbase(bat, Skv, Hkv, hkv);
    const int rs   = Hkv * ((isk ? K1_D_QK : K1_D_V) / 8);
    return base + (kv0 + rb * 4 + lane / 8) * rs + lane % 8 + li * 8;
}

// ---- k3: 96-row kv blocks (2 waves x 48 rows, nkv 3) and 64-row tail blocks (the k1 body, nkv 2)
// ---------- grid y = k3_nblk(Skv) blocks, ascending kv (longest first under bottom-right causal):
// blocks [0, n96) are 96-row blocks at kv0g = 96*bid; the n64 tail blocks follow at 96*n96 + 64*j.
// Skv % 96 == 0 -> no tail;
// == 64 -> one 64-row tail (prod: 42 + 1); == 32 -> two 64-row tails (96*(n-1) + 128). Any other
// Skv is refused (k3_ok == 0; impl._plan routes it to bwd_hk_k1's k_dkdv64).
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
K1_FN int k3_ok(int Skv) {
    return (Skv > 0 && k3_n64(Skv) >= 0 && k3_n96(Skv) >= 0) ? 1 : 0;
}
K1_FN int k3_nblk(int Skv) {
    return k3_n96(Skv) + k3_n64(Skv);
}
K1_FN int k3_kv0g(int bid, int n96) {
    return bid < n96 ? bid * K3_ROWS : n96 * K3_ROWS + (bid - n96) * 64;
}
K1_FN int k3_rows(int bid, int n96) {
    return bid < n96 ? K3_ROWS : 64;
}
K1_FN int k3_kv0(int kv0g, int wv, int nkv) {
    return kv0g + wv * 16 * nkv;
}
// causal split of a block of `rows` kv rows at kv0g (k1_nmaskp with 63 -> rows - 1; qp_start /
// nqp_eff do not depend on the block height: k1_qp_start / k1_nqp_eff)
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
// epilogue images of a wave with nkv sub-tiles: row-major dV [16 nkv][D_V] (stride 272) at
// w*EPI_W(nkv), dK [16 nkv][D_QK] (400) at w*EPI_W + EPI_DK; nkv 3: 2 x 32256 = 64512 B = the ring
// allocation (K1_LDS)
K1_FN int k3_epi_dk(int nkv) {
    return 16 * nkv * K1_XV;
}
K1_FN int k3_epi_w(int nkv) {
    return 16 * nkv * (K1_XV + K1_XK);
}
K1_FN int k3_epi_wr(int nkv, int wv, int isk, int kh, int d, int lane) {
    const int x = isk ? K1_XK : K1_XV;
    return wv * k3_epi_w(nkv) + (isk ? k3_epi_dk(nkv) : 0) + (lane % 16) * x + (lane / 16) * 16 +
           kh * 16 * x + d * 32;
}
// whole-line readback: lane l reads kv row 4rb + l/8 (rb in [0, 4 nkv)), bytes li*128 + (l%8)*16 of
// the image; its global store is k1_epi_g (kv row kv0 + 4rb + l/8)
K1_FN int k3_epi_rd(int nkv, int wv, int isk, int rb, int li, int lane) {
    const int x = isk ? K1_XK : K1_XV;
    return wv * k3_epi_w(nkv) + (isk ? k3_epi_dk(nkv) : 0) + (lane / 8) * x + (lane % 8) * 16 +
           rb * 4 * x + li * 128;
}
