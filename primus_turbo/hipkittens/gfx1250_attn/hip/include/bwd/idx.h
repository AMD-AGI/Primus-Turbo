/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// bwd/idx.h -- every index formula of the gfx1250 d128 GQA backward v0 (k_hk_bwd_prep[_z],
// k_hk_bwd_g4, k_hk_bwd_dqcvt in hip/src/k_attn_bwd_v0.hip), in ONE place. Verbatim copy of the
// reviewed design sketch hk/notes/bwd_v0_idx_sketch.h (hk/notes/bwd_v0_design.md) up to the section
// "kernel-side additions", which holds what the kernels and the bounds checker
// (hk/lab/bounds/bwd_v0_check.cpp) add.
//
//   dV = P^T dO, dS = P o (dP - delta), dK = scale dS^T Q, dQ = scale dS K,
//   S = Q K^T (unscaled), P = exp2(S * scale * log2e - LSE * log2e), dP = dO V^T, delta = rowsum(dO
//   o O), causal bottom-right (kv <= q + Skv - Sq), BSHD contiguous, LSE natural log fp32 [B, Hq,
//   Sq].
//
// constexpr __host__ __device__ functions: the kernels compute every global offset, LDS address,
// TDM operand, ring slot, loop bound and wave kind through these, and the bounds checker compiles
// THIS file for x86 and replays them. B0_IDX_SELFTEST adds b0::selftest() (LDS map, global bounds +
// exactly-once coverage, causal kinds, dQ_acc C-fragment bijection, ring / barrier protocol of v0
// AND v1 with negative controls); never part of a kernel build.
//
// Geometry: a workgroup (WG) = 4 waves (one per SIMD) = 128 kv rows [kvw0, kvw0 + 128) of one (b,
// hkv); wave w owns kv rows kvw0 + 32 w + [0, 32) = two 16-row kv tiles kh. A step = 32 queries
// (one q pair qt, two 16-row q tiles hh / qh16) of one q head gh of the GQA group; steps run heads
// inner, pairs descending from the last pair down to the diagonal. WMMA orientation (ASM): S = Q
// K^T, dP = dO V^T (C lane = kv, rows = q), dV^T += dO^T P, dK^T += Q^T dS (C lane = kv, rows = d;
// P / dS packed from the S / dP C fragments as B operands), dQ = dS K (C lane = d, rows = q; A = dS
// via tr16 of the dS^T exchange tile). dQ of step s is computed one step late (in step s + 1, under
// its softmax) and atomically added right after; its A operands are read from the dS^T tile at the
// end of step s. C layout: lane l, element i <-> row 8 (l / 16) + i, column l % 16
// (hk_attn/layout.h).
#pragma once

#if defined(__HIP__) || defined(__HIPCC__)
#define B0_HD __host__ __device__ __attribute__((always_inline)) constexpr
#else
#define B0_HD constexpr
#endif

#include <stdint.h>

namespace b0 {

// ---- shape-independent constants
// ---------------------------------------------------------------------------
constexpr int D     = 128; // q / k / v head dim
constexpr int WAVE  = 32;
constexpr int NWAVE = 4; // 1 wave per SIMD, 1 WG per CU
constexpr int BLOCK = NWAVE * WAVE;
constexpr int KVW   = 32;          // kv rows per wave (2 kv tiles kh)
constexpr int BKV   = NWAVE * KVW; // 128 kv rows per WG
constexpr int QS    = 32;          // queries per step (one q pair = q tiles qh16 0, 1)
constexpr int NDT   = D / 32;      // 4: 32-deep k-steps of S = Q K^T and dP = dO V^T
constexpr int NDO   = D / 16;      // 8: 16-wide d tiles of dV^T / dK^T
constexpr int NKC   = BKV / 32;    // 4: 32-kv chunks of the dQ contraction
constexpr int ROW_B = D * 2;       // 256: one bf16 head row

// LDS pitches (bytes). XP: Q / dO ring tiles and the K / V images (rows loader AND tr loader on the
// same tile, 32 + 32 per wave-step; 272 = conflict-free b128 rows, 288 = conflict-free tr16 under
// the half-wave model; mb tr16pitch decides, 272 is the FlyDSL / REF-proven value). SP: dS^T
// exchange tile [128 kv][32 q] read by 16 tr16 and written by 4 C^T stores per wave-step: 96 =
// conflict-free tr16, 2-way stores (default); 80 = FlyDSL's S_ROW_B: conflict-free stores, five
// 2-way 16-B collisions per tr16 half-wave.
#ifndef B0_XP
#define B0_XP 272
#endif
#ifndef B0_SP
#define B0_SP 96
#endif
constexpr int XP = B0_XP;
constexpr int SP = B0_SP;

// ring / exchange depths: v0 uses RING 4 (lead 3) and NDS 2; v1 uses RING 5 (lead 3) and NDS 3. The
// LDS map reserves the v1 sizes so v1 is a code change only.
constexpr int RING_V0 = 4, RING_V1 = 5, RING_MAX = 5;
constexpr int NDS_V0 = 2, NDS_V1 = 3, NDS_MAX = 3;
constexpr int LEAD = 3; // TDM op of tile k is issued in step k - LEAD

constexpr int QT_B = QS * XP;  // 8704 @272: one Q or dO tile [32 q][XP]
constexpr int LD_B = QS * 8;   // 256: the NL / delta records of one step (32 q x {NL, delta} fp32)
constexpr int DS_B = BKV * SP; // 12288 @96: one dS^T exchange tile [128 kv][SP]
constexpr int KV_B = BKV * XP; // 34816 @272: one K or V image [128 kv][XP]

// LDS map (byte offsets in the single static allocation; 64-KB segment = off >> 16). dO ring + LD
// ring in segment 0, Q ring in segment 1, dS^T tiles in segment 2, K / V images in segments 3-4
// (prologue only in v0; v1 prefetches the paired block's K / V there).
constexpr int OFF_DO    = 0;
constexpr int OFF_LD    = OFF_DO + RING_MAX * QT_B; // 43520
constexpr int OFF_Q     = 65536;
constexpr int OFF_DS    = 131072;
constexpr int OFF_K     = 196608;
constexpr int OFF_V     = OFF_K + KV_B; // 231424
constexpr int LDS_BYTES = OFF_V + KV_B; // 266240 @272 (<= 327680)
static_assert(OFF_LD + RING_MAX * LD_B <= OFF_Q, "dO + LD rings fit segment 0");
static_assert(OFF_Q + RING_MAX * QT_B <= OFF_DS, "Q ring fits segment 1");
static_assert(OFF_DS + NDS_MAX * DS_B <= OFF_K, "dS^T tiles fit segment 2");
static_assert(LDS_BYTES <= 327680, "320 KB LDS");

// TDM roles (wave-uniform): wave 0 issues dO, wave 1 Q, wave 2 the LD records (one ring op per
// issuer per step); wave 3 issues the K and V images in the prologue only (v1: also the paired
// block's K / V prefetch).
constexpr int ISS_DO = 0, ISS_Q = 1, ISS_LD = 2, ISS_KV = 3;
constexpr int TCNT_STEADY =
    1; // s_wait_tensorcnt before every step signal: retires op(k), op(k+1) flies

// TDM group-1 word 0 (hk_attn::lay::tdm_w0): data_size, pad_enable, pad_interval log2(row dw) - 1,
// pad dw - 1
B0_HD int ilog2(int x) {
    return x <= 1 ? 0 : 1 + ilog2(x / 2);
}
B0_HD uint32_t tdm_w0(int cols, int pitch_b, int elem_b) {
    const int      row_b = cols * elem_b, pad_b = pitch_b - row_b;
    const uint32_t ds = (uint32_t) ilog2(elem_b) << 16;
    return pad_b == 0 ? ds
                      : ds | (1u << 20) | ((uint32_t) (ilog2(row_b / 4) - 1) << 22) |
                            ((uint32_t) (pad_b / 4 - 1) << 25);
}
constexpr uint32_t W0_QDO = tdm_w0(D, XP, 2);   // 0x07510000 @272, 0x0F510000 @288
constexpr uint32_t W0_LD  = tdm_w0(64, 256, 4); // 0x00020000: one row of 64 fp32, no pad
constexpr uint32_t W0_KV  = W0_QDO;

constexpr float LOG2E = 1.4426950408889634f;

// ---- grid decode
// --------------------------------------------------------------------------------------------- v0:
// grid (Hkv, Skv / 128, B), blockIdx = (hkv, blk, b): x fastest = the 8 kv heads of one block (one
// per XCD under round-robin dispatch), y ascending = longest first (bottom-right causal), z = batch
// (FlyDSL).
struct Work {
    int b, hkv, blk;
};
B0_HD Work work_v0(int bx, int by, int bz) {
    return Work{bz, bx, by};
}
// v1 (persistent pairing): grid (Hkv, nblk / 2, B); sub 0 = block by (long), sub 1 = block nblk - 1
// - by (short). Needs v1_ok(nblk) (an odd nblk leaves the middle block without a WG: run v0). Every
// WG runs exactly G (2 nqp - 4 (nblk - 1)) steps when Sq == Skv (prod 1040) and 2 G nqp when
// non-causal; with Sq < Skv the blocks with 128 blk <= cshift all have nqp pairs, so the pairs are
// correct but unbalanced.
B0_HD Work work_v1(int bx, int by, int bz, int sub, int nblk) {
    return Work{bz, bx, sub == 0 ? by : nblk - 1 - by};
}
B0_HD bool v1_ok(int nblk) {
    return nblk >= 2 && nblk % 2 == 0;
}
B0_HD int kvw0_of(int blk) {
    return blk * BKV;
}

// ---- host gate (impl.py, after FlyDSL's unsupported_reason): the only shapes k_hk_bwd_g4 is
// launched with. 0 = launch, else the first failed rule. It buys the kernel invariant n_all >= G
// for every WG (selftest): Sq = 0 would give n = 0, step_clamp -> -1, tile_qt -1 and a prologue TDM
// read before DO / Q / LD.
enum { GATE_OK = 0, GATE_EMPTY, GATE_G, GATE_SQ, GATE_SKV, GATE_CAUSAL, GATE_BYTES };
B0_HD int host_gate(int64_t B, int64_t Sq, int64_t Skv, int64_t Hq, int64_t Hkv, int causal,
                    int G) {
    if (B < 1 || Sq < 1 || Skv < 1 || Hq < 1 || Hkv < 1)
        return GATE_EMPTY;
    if (Hq != (int64_t) G * Hkv)
        return GATE_G; // the compiled G (4); other G -> FlyDSL
    if (Sq % 64 != 0)
        return GATE_SQ; // with Sq >= 1: Sq >= 64
    if (Skv % BKV != 0)
        return GATE_SKV; // with Skv >= 1: Skv >= 128
    if (causal && Sq > Skv)
        return GATE_CAUSAL;
    const int64_t lim = (int64_t) 1 << 31; // int32 voffsets / kernel int args
    if (B * Sq * Hq * ROW_B >= lim || B * Hq * Sq * 512 >= lim || (int64_t) G * Sq * 512 >= lim ||
        B * Skv * Hkv * ROW_B >= lim || (int64_t) (BKV - 1) * Hkv * ROW_B + ROW_B >= lim)
        return GATE_BYTES;
    return GATE_OK;
}

// ---- causal split (WG-uniform: kvw0 only, never the wave)
// ------------------------------------------------------
B0_HD int nqp_of(int Sq) {
    return Sq / QS;
}
// first q pair with any visible kv of the block: 32 qt + 31 + cshift >= kvw0
B0_HD int qp_start(int kvw0, int cshift, int causal) {
    const int c = kvw0 - cshift;
    return causal ? (c < 0 ? 0 : c) / QS : 0;
}
// first q pair that sees all 128 kv rows of the block: kvw0 + 127 <= 32 qt + cshift
B0_HD int qp_full(int kvw0, int cshift, int causal, int nqp) {
    if (!causal)
        return 0;
    const int u = kvw0 + BKV - 1 - cshift;
    const int q = u <= 0 ? 0 : (u + QS - 1) / QS;
    return q < nqp ? q : nqp;
}
B0_HD int n_pairs(int kvw0, int cshift, int causal, int nqp) {
    return nqp - qp_start(kvw0, cshift, causal);
}
B0_HD int n_maskp(int kvw0, int cshift, int causal, int nqp) {
    int       m  = qp_full(kvw0, cshift, causal, nqp) - qp_start(kvw0, cshift, causal);
    const int np = n_pairs(kvw0, cshift, causal, nqp);
    m            = m < 0 ? 0 : m;
    return m < np ? m : np;
}
B0_HD int n_all(int kvw0, int cshift, int causal, int nqp, int G) {
    return G * n_pairs(kvw0, cshift, causal, nqp);
}
B0_HD int n_mask(int kvw0, int cshift, int causal, int nqp, int G) {
    return G * n_maskp(kvw0, cshift, causal, nqp);
}
B0_HD int n_full(int kvw0, int cshift, int causal, int nqp, int G) {
    return n_all(kvw0, cshift, causal, nqp, G) - n_mask(kvw0, cshift, causal, nqp, G);
}

// ---- step -> tile (heads inner, pairs descending from nqp - 1; s clamped into [0, n - 1])
// ----------------------- Steps [0, n_full) are unmasked for the whole block, [n_full, n) are the
// diagonal pairs. Clamping makes the prefetch of steps >= n reload tile n - 1 (in bounds, never
// read for compute), so every issuer issues exactly one TDM op per step and the tensorcnt
// immediates never change.
B0_HD int step_clamp(int s, int n) {
    return s < 0 ? 0 : (s >= n ? n - 1 : s);
}
B0_HD int tile_qt(int s, int n, int G, int nqp) {
    return nqp - 1 - step_clamp(s, n) / G;
} // G pow2: shift
B0_HD int tile_gh(int s, int n, int G) {
    return step_clamp(s, n) % G;
} // G pow2: mask
B0_HD int qh_of(int hkv, int G, int gh) {
    return hkv * G + gh;
}

// wave kind of a step (wave-uniform): 0 FULL (no mask), 1 DIAG (mask), 2 DEAD (every pair masked:
// skip the S/dP/softmax/dV/dK work, store a zero dS^T, keep every barrier, atomic and dQ WMMA)
enum { K_FULL = 0, K_DIAG = 1, K_DEAD = 2 };
B0_HD int wave_kind(int kvw0, int w, int qt, int cshift, int causal) {
    if (!causal)
        return K_FULL;
    const int kv_lo = kvw0 + KVW * w, kv_hi = kv_lo + KVW - 1, q_lo = QS * qt, q_hi = q_lo + QS - 1;
    if (kv_lo > q_hi + cshift)
        return K_DEAD;
    if (kv_hi <= q_lo + cshift)
        return K_FULL;
    return K_DIAG;
}
// mask offset of S tile (hh, kh) of wave w (hk_attn mask_col_gt_row: masked iff n > m + off)
B0_HD int mask_off(int qt, int hh, int kvw0, int w, int kh, int cshift) {
    return (QS * qt + 16 * hh) + cshift - (kvw0 + KVW * w + 16 * kh);
}

// ---- ring slots / exchange buffers
// --------------------------------------------------------------------------------
B0_HD int slot_of(int k, int ring) {
    return k % ring;
}
B0_HD int dsb_of(int t, int nds) {
    return t % nds;
}
B0_HD int lds_do(int slot) {
    return OFF_DO + slot * QT_B;
}
B0_HD int lds_q(int slot) {
    return OFF_Q + slot * QT_B;
}
B0_HD int lds_ld(int slot) {
    return OFF_LD + slot * LD_B;
}
B0_HD int lds_ds(int buf) {
    return OFF_DS + buf * DS_B;
}

// ---- TDM sources (int64 byte offsets from the tensor base; tensor_dim0 = tile columns,
// tensor_dim1 = the tile rows, both exact: every issued tile lies inside its tensor, proved by the
// selftest) --------------------------- dO / Q tile: 32 rows q0..q0+31 of head qh, BSHD viewed as
// [B*Sq, Hq*D] (row stride Hq*D elements)
B0_HD int64_t qdo_tile_byte(int b, int Sq, int Hq, int qh, int q0) {
    return (((int64_t) b * Sq + q0) * Hq + qh) * ROW_B;
}
B0_HD int64_t qdo_tile_end(int b, int Sq, int Hq, int qh, int q0) { // exclusive
    return qdo_tile_byte(b, Sq, Hq, qh, q0) + (int64_t) (QS - 1) * Hq * ROW_B + ROW_B;
}
B0_HD int64_t qdo_bytes(int B, int Sq, int Hq) {
    return (int64_t) B * Sq * Hq * ROW_B;
}
// LD = prep output [B, Hq, Sq / 4] records of 32 B {NL[4 q], delta[4 q]}: one step = 256 contiguous
// bytes
B0_HD int64_t ld_rec_byte(int b, int Hq, int qh, int Sq, int q) {
    return (((int64_t) b * Hq + qh) * Sq + q) * 8;
} // q % 4 == 0
B0_HD int64_t ld_bytes(int B, int Hq, int Sq) {
    return (int64_t) B * Hq * Sq * 8;
}
// K / V image: 128 rows kvw0.. of kv head hkv, [B*Skv, Hkv*D]
B0_HD int64_t kv_tile_byte(int b, int Skv, int Hkv, int hkv, int kvw0) {
    return (((int64_t) b * Skv + kvw0) * Hkv + hkv) * ROW_B;
}
B0_HD int64_t kv_tile_end(int b, int Skv, int Hkv, int hkv, int kvw0) {
    return kv_tile_byte(b, Skv, Hkv, hkv, kvw0) + (int64_t) (BKV - 1) * Hkv * ROW_B + ROW_B;
}
B0_HD int64_t kv_bytes(int B, int Skv, int Hkv) {
    return (int64_t) B * Skv * Hkv * ROW_B;
}

// ---- LDS operand lane maps (== hk_attn::lay; lane (r, h) = (l % 16, l / 16))
// -----------------------------------
B0_HD int rows_lane(int lane, int pitch) {
    return (lane % 16) * pitch + (lane / 16) * 16;
}
B0_HD int rows_imm(int pitch, int r0, int k0, int half) {
    return r0 * pitch + 2 * k0 + 32 * half;
}
B0_HD int tr_lane(int lane, int pitch) {
    return ((lane / 16) * 8 + lane % 8) * pitch + ((lane / 8) % 2) * 16;
}
B0_HD int tr_imm(int pitch, int k0, int r0, int half) {
    return (k0 + 16 * half) * pitch + 2 * r0;
}
B0_HD int ct_imm(int pitch, int n0, int m0) {
    return n0 * pitch + 2 * m0;
} // + rows_lane: row n0 + r, cols m0 + 8h
// element maps of the loaders (for the emulation in the selftest): (row, col) of element e of lane
// l
B0_HD int op_k(int lane, int e) {
    return (e / 8) * 16 + (lane / 16) * 8 + e % 8;
}
// LD readback: lane (r, h) of q tile hh reads 4 b128 at lds_ld(slot) + ld_lane + ld_imm(hh, k):
// k 0 = NL[16hh + 8h + 0..3], 1 = delta[0..3], 2 = NL[4..7], 3 = delta[4..7] (accumulator rows 8h +
// i)
B0_HD int ld_lane(int lane) {
    return (lane / 16) * 64;
}
B0_HD int ld_imm(int hh, int k) {
    return hh * 128 + k * 16;
}

// per-step operand fragments (all relative to a tile base; r0 / k0 compile-time):
//   S  A  qA[hh][dt] = rows<XP>(lds_q(slot),  r0 16hh,          k0 32dt)   Q[q0 + 16hh + r][32dt +
//   op_k] dP A  dA[hh][dt] = rows<XP>(lds_do(slot), r0 16hh,          k0 32dt) S  B  kf[kh][dt] =
//   rows<XP>(OFF_K,        r0 32w + 16kh,    k0 32dt)   K[kvw0 + 32w + 16kh + r][32dt + op_k] dP B
//   vf[kh][dt] = rows<XP>(OFF_V,        r0 32w + 16kh,    k0 32dt) dV A  dOT[dt]    =
//   tr<XP>(lds_do(slot),   k0 0,             r0 16dt)   dO[q0 + op_k][16dt + r] dK A  QT[dt]     =
//   tr<XP>(lds_q(slot),    k0 0,             r0 16dt) dQ A  dSa[qh][kc]= tr<SP>(lds_ds(buf),    k0
//   32kc,          r0 16qh)   dS^T[32kc + op_k][16qh + r] dQ B  kt[j][kc]  = tr<XP>(OFF_K, k0 32kc,
//   r0 32w + 16j)  K[kvw0 + 32kc + op_k][32w + 16j + r] dS^T store (hh, kh): ds_store_b128 at
//   lds_ds(buf) + rows_lane(l, SP) + ct_imm(SP, 32w + 16kh, 16hh)

// ---- dQ_acc: fp32, WMMA C-fragment order, one 16 KB block per (b, qh, q pair)
// ---------------------------------- block ((b Hq + qh) nqp + qt); inside: slot k = (qh16 2 + j) 8
// + i (512 B), + w 128 B + lane 4 B. Element (q, d): q = 32 qt + 16 qh16 + 8 (l / 16) + i, d = 32 w
// + 16 j + l % 16. One atomic = one aligned 128-B line. Per-WG buffer resource over its GQA group:
// base dqa_grp_byte, TRUE extent dqa_grp_bytes; the per-step part lives in voffset (range-checked;
// provably non-negative so the slot constant folds into offset:IMM).
B0_HD int64_t dqa_grp_byte(int b, int Hq, int hkv, int G, int Sq) {
    return (((int64_t) b * Hq + (int64_t) hkv * G) * nqp_of(Sq)) * 16384;
}
B0_HD int64_t dqa_grp_bytes(int G, int Sq) {
    return (int64_t) G * nqp_of(Sq) * 16384;
}
B0_HD int64_t dqa_bytes(int B, int Hq, int Sq) {
    return (int64_t) B * Hq * nqp_of(Sq) * 16384;
}
B0_HD int dqa_voff(int lane, int w, int gh, int qt, int nqp) {
    return ((gh * nqp + qt) << 14) | (w << 7) | (lane << 2);
}
B0_HD int dqa_imm(int qh16, int j, int i) {
    return ((qh16 * 2 + j) * 8 + i) * 512;
}
B0_HD int frag_q(int qh16, int lane, int i) {
    return 16 * qh16 + 8 * (lane / 16) + i;
} // within the step's 32 q
B0_HD int frag_d(int w, int j, int lane) {
    return 32 * w + 16 * j + lane % 16;
}

// ---- epilogue: dV^T / dK^T accumulator (dt, kh) of lane (r, h) -> bf16 row kv = kvw0 + 32w + 16kh
// + r, d = 16dt + 8h + [0, 8) straight from registers (C^T store, 16 B per lane). Per-WG resource:
// base = the WG's first row of head hkv, TRUE extent = its 128-row slab. dK is multiplied by scale
// (fp32) first. ----------
B0_HD int64_t epi_base_byte(int b, int Skv, int Hkv, int hkv, int kvw0) {
    return kv_tile_byte(b, Skv, Hkv, hkv, kvw0);
}
B0_HD int64_t epi_bytes(int Hkv) {
    return (int64_t) (BKV - 1) * Hkv * ROW_B + ROW_B;
}
B0_HD int epi_voff(int lane, int w, int kh, int dt, int Hkv) {
    return (KVW * w + 16 * kh + lane % 16) * Hkv * ROW_B + (16 * dt + 8 * (lane / 16)) * 2;
}

// ---- dQ convert (grid (Sq / 32, Hq, B), 256 threads): fp32 C-fragment block -> scale -> bf16 dQ
// rows ----------
constexpr int CVT_PITCH = 272; // LDS [32 q][128 d] bf16 image
B0_HD int64_t cvt_src_byte(int b, int Hq, int qh, int nqp, int qt) {
    return (((int64_t) b * Hq + qh) * nqp + qt) * 16384;
}
// float p in [0, 4096) of a block -> (q, d) inside the step (inverse of dqa_voff / dqa_imm)
B0_HD int cvt_q(int p) {
    const int k = p / 128, ln = p % 32;
    return 16 * (k / 16) + 8 * (ln / 16) + k % 8;
}
B0_HD int cvt_d(int p) {
    const int k = p / 128, e = p % 128, ln = e % 32;
    return 32 * (e / 32) + 16 * ((k / 8) % 2) + ln % 16;
}
B0_HD int cvt_lds(int q, int d) {
    return q * CVT_PITCH + 2 * d;
}
B0_HD int64_t cvt_dst_byte(int b, int Sq, int Hq, int qh, int qt, int row, int chunk) {
    return (((int64_t) b * Sq + QS * qt + row) * Hq + qh) * ROW_B + 16 * chunk;
}

// ---- prep (grid (Sq / 64, Hq, B), 256 threads): 64 rows (b, q0 + [0, 64), h)
// -------------------------------------
B0_HD int64_t prep_row_byte(int b, int Sq, int Hq, int h, int q) {
    return (((int64_t) b * Sq + q) * Hq + h) * ROW_B;
}
B0_HD int64_t lse_byte(int b, int Hq, int h, int Sq, int q) {
    return (((int64_t) b * Hq + h) * Sq + q) * 4;
}
// thread t, pass p (0..3): row q0 + 16p + t / 16, bytes 16 (t % 16) of it; group g = t (< 16)
// writes record q0 + 4g: NL at ld_rec_byte, delta at + 16
B0_HD int prep_row(int t, int p) {
    return 16 * p + t / 16;
}

// ======================= kernel-side additions (not in the design sketch)
// =======================================
constexpr int G4   = 4; // the compiled GQA ratio (symbol k_hk_bwd_g4)
constexpr int RING = RING_V0, NDS = NDS_V0;

// TDM roles (wave = role): global byte offset of the tile of step (qt, gh) from the role tensor's
// base. Affine in (qt, gh), so the kernel builds it as role_base + qt * role_qstep + gh *
// role_hstep (int64 SALU, steps derived from this function) and the checker replays exactly that
// sum.
B0_HD int64_t role_tile_byte(int role, int b, int Sq, int Hq, int hkv, int G, int qt, int gh) {
    return role == ISS_LD ? ld_rec_byte(b, Hq, qh_of(hkv, G, gh), Sq, QS * qt)
                          : qdo_tile_byte(b, Sq, Hq, qh_of(hkv, G, gh), QS * qt);
}
B0_HD int64_t role_qstep(int role, int Sq, int Hq) {
    return role_tile_byte(role, 0, Sq, Hq, 0, 1, 1, 0) -
           role_tile_byte(role, 0, Sq, Hq, 0, 1, 0, 0);
}
B0_HD int64_t role_hstep(int role, int Sq, int Hq) {
    return role_tile_byte(role, 0, Sq, Hq, 0, 1, 0, 1) -
           role_tile_byte(role, 0, Sq, Hq, 0, 1, 0, 0);
}
B0_HD int64_t role_step_byte(int64_t base, int64_t qstep, int64_t hstep, int qt, int gh) {
    return base + (int64_t) qt * qstep + (int64_t) gh * hstep;
}
// LDS slot of the role's ring and the group-1 parameters of its tile (dO / Q: [32][128] bf16 at
// pitch XP, row stride Hq*D elements; LD: one row of 64 fp32 = the 32 queries' records, unpadded)
B0_HD int role_lds(int role, int slot) {
    return role == ISS_DO ? lds_do(slot) : role == ISS_Q ? lds_q(slot) : lds_ld(slot);
}
// role_lds is affine in the slot: the kernel selects base and stride once (no per-step role branch)
B0_HD int role_lds0(int role) {
    return role_lds(role, 0);
}
B0_HD int role_lds_step(int role) {
    return role_lds(role, 1) - role_lds(role, 0);
}
B0_HD int role_lds_at(int lds0, int lstep, int slot) {
    return lds0 + slot * lstep;
}
B0_HD int64_t role_stride_el(int role, int Hq) {
    return role == ISS_LD ? 64 : (int64_t) Hq * D;
}
B0_HD int role_rows(int role) {
    return role == ISS_LD ? 1 : QS;
}
B0_HD int role_cols(int role) {
    return role == ISS_LD ? 64 : D;
}
B0_HD int role_elem(int role) {
    return role == ISS_LD ? 4 : 2;
}
B0_HD int role_pitch(int role) {
    return role == ISS_LD ? LD_B : XP;
}
B0_HD int64_t kv_stride_el(int Hkv) {
    return (int64_t) Hkv * D;
} // K / V image: 128 rows of [B*Skv, Hkv*D]

// per-wave parts of the LDS operand bases (runtime w folded into the lane base, so every fragment
// coordinate stays a compile-time immediate): rows of the wave's kv rows in the K / V images, tr of
// the K image at the wave's dQ columns d = 32 w + .., and the wave's 32 rows of a dS^T tile
B0_HD int kimg_rows_wave(int w) {
    return KVW * w * XP;
}
B0_HD int kimg_tr_wave(int w) {
    return 2 * KVW * w;
}
B0_HD int ds_rows_wave(int w) {
    return KVW * w * SP;
}

// dQ(s - 1) is added in step s (s = 0: dQ(-1) = 0 into the clamped tile 0); the tile of the atomics
// of step s
B0_HD int atom_qt(int s, int n, int G, int nqp) {
    return tile_qt(s - 1, n, G, nqp);
}
B0_HD int atom_gh(int s, int n, int G) {
    return tile_gh(s - 1, n, G);
}

// epilogue: per-kh lane voffset (dt lands in the instruction offset, + 32 dt)
B0_HD int epi_voff_kh(int lane, int w, int kh, int Hkv) {
    return epi_voff(lane, w, kh, 0, Hkv);
}
B0_HD int epi_imm(int dt) {
    return 32 * dt;
}

// prep (grid (Sq / 64, Hq, B), 256 threads) and convert (grid (Sq / 32, Hq, B), 256 threads)
constexpr int PREP_THREADS = 256, PREP_ROWS = 64, PREP_LDS_B = PREP_ROWS * 4;
constexpr int CVT_THREADS = 256, CVT_LDS_B = QS * CVT_PITCH; // 8704
B0_HD int     prep_chunk(int t) {
    return t % 16;
}
B0_HD int prep_zero_off(int t, int i) {
    return 16 * (t + PREP_THREADS * i);
} // i = 0..7: 32 KB per WG
B0_HD int cvt_src_off(int t, int i) {
    return 16 * (t + CVT_THREADS * i);
} // i = 0..3: the 16 KB block
B0_HD int cvt_row(int t, int i) {
    return (t + CVT_THREADS * i) / 16;
} // i = 0..1: 32 rows x 16 chunks
B0_HD int cvt_chunk(int t, int i) {
    return (t + CVT_THREADS * i) % 16;
}

// ISA expectation of k_hk_bwd_g4 (hk/lab/test_ext_bwd.py lint): DS loads in flight at each
// full-step signal
constexpr int FULL_UNCOVERED = 2 * NDO * 2; // the 32 E-operand tr16 (dOT[0..7], QT[0..7], 2 each)

} // namespace b0

#ifdef B0_IDX_SELFTEST
// ======================================= reference replay (host only)
// =============================================
#include <algorithm>
#include <cstdio>
#include <deque>
#include <vector>

namespace b0 {

#define B0_CHECK(cond, ...)                                                                        \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                       \
            std::printf(__VA_ARGS__);                                                              \
            std::printf("\n");                                                                     \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

struct Shape {
    int  B, Sq, Skv, Hq, Hkv, causal;
    bool exhaustive;
};

// --- 1. dQ_acc C-fragment map: (w, lane, qh16, j, i) -> float in the 16 KB block is a bijection,
// cvt inverts it
inline int check_cfrag() {
    std::vector<int> hit(4096, 0);
    for (int w = 0; w < NWAVE; ++w)
        for (int l = 0; l < WAVE; ++l)
            for (int qh = 0; qh < 2; ++qh)
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 8; ++i) {
                        const int byte = dqa_voff(l, w, 0, 0, 1) + dqa_imm(qh, j, i);
                        B0_CHECK(byte % 4 == 0 && byte >= 0 && byte < 16384, "cfrag byte %d", byte);
                        const int p = byte / 4;
                        ++hit[p];
                        B0_CHECK(cvt_q(p) == frag_q(qh, l, i) && cvt_d(p) == frag_d(w, j, l),
                                 "cvt inverse at p %d", p);
                        // one atomic instruction (fixed w, qh, j, i; lanes 0..31) = one aligned
                        // 128-B line
                        B0_CHECK((byte - 4 * l) % 128 == 0, "line alignment");
                    }
    for (int p = 0; p < 4096; ++p)
        B0_CHECK(hit[p] == 1, "cfrag float %d hit %d times", p, hit[p]);
    return 0;
}

// --- 2. LDS: every lane access of every loader / store lies inside its region; regions disjoint
// ---------------
inline int check_lds() {
    struct R {
        int         lo, hi;
        const char *n;
    } regs[] = {{OFF_DO, OFF_DO + RING_MAX * QT_B, "dO ring"},
                {OFF_LD, OFF_LD + RING_MAX * LD_B, "LD ring"},
                {OFF_Q, OFF_Q + RING_MAX * QT_B, "Q ring"},
                {OFF_DS, OFF_DS + NDS_MAX * DS_B, "dS^T"},
                {OFF_K, OFF_K + KV_B, "K image"},
                {OFF_V, OFF_V + KV_B, "V image"}};
    for (auto &a : regs) {
        B0_CHECK(a.lo % 1024 == 0 || a.lo % 16 == 0, "align %s", a.n);
        B0_CHECK(a.hi <= LDS_BYTES, "%s past LDS", a.n);
        for (auto &b : regs)
            B0_CHECK(&a == &b || a.hi <= b.lo || b.hi <= a.lo, "%s overlaps %s", a.n, b.n);
    }
    auto in = [](int a, int lo, int hi) { return a >= lo && a + 16 <= hi && a % 16 == 0; };
    for (int l = 0; l < WAVE; ++l)
        for (int w = 0; w < NWAVE; ++w) {
            for (int sl = 0; sl < RING_MAX; ++sl) {
                const int qlo = lds_q(sl), dlo = lds_do(sl);
                for (int hh = 0; hh < 2; ++hh)
                    for (int dt = 0; dt < NDT; ++dt)
                        for (int u = 0; u < 2; ++u) {
                            B0_CHECK(in(qlo + rows_lane(l, XP) + rows_imm(XP, 16 * hh, 32 * dt, u),
                                        qlo, qlo + 32 * XP - (XP - ROW_B)),
                                     "Q rows");
                            B0_CHECK(in(dlo + rows_lane(l, XP) + rows_imm(XP, 16 * hh, 32 * dt, u),
                                        dlo, dlo + 32 * XP - (XP - ROW_B)),
                                     "dO rows");
                        }
                for (int dt = 0; dt < NDO; ++dt)
                    for (int u = 0; u < 2; ++u) {
                        B0_CHECK(in(qlo + tr_lane(l, XP) + tr_imm(XP, 0, 16 * dt, u), qlo,
                                    qlo + 32 * XP - (XP - ROW_B)),
                                 "Q tr");
                        B0_CHECK(in(dlo + tr_lane(l, XP) + tr_imm(XP, 0, 16 * dt, u), dlo,
                                    dlo + 32 * XP - (XP - ROW_B)),
                                 "dO tr");
                    }
                for (int hh = 0; hh < 2; ++hh)
                    for (int k = 0; k < 4; ++k)
                        B0_CHECK(in(lds_ld(sl) + ld_lane(l) + ld_imm(hh, k), lds_ld(sl),
                                    lds_ld(sl) + LD_B),
                                 "LD");
            }
            for (int bf = 0; bf < NDS_MAX; ++bf) {
                const int lo = lds_ds(bf), hi = lo + BKV * SP - (SP - 64);
                for (int hh = 0; hh < 2; ++hh)
                    for (int kh = 0; kh < 2; ++kh)
                        B0_CHECK(in(lo + rows_lane(l, SP) + ct_imm(SP, KVW * w + 16 * kh, 16 * hh),
                                    lo, hi),
                                 "dS^T st");
                for (int qh = 0; qh < 2; ++qh)
                    for (int kc = 0; kc < NKC; ++kc)
                        for (int u = 0; u < 2; ++u)
                            B0_CHECK(
                                in(lo + tr_lane(l, SP) + tr_imm(SP, 32 * kc, 16 * qh, u), lo, hi),
                                "dS^T tr");
            }
            for (int kh = 0; kh < 2; ++kh)
                for (int dt = 0; dt < NDT; ++dt)
                    for (int u = 0; u < 2; ++u) {
                        B0_CHECK(in(OFF_K + rows_lane(l, XP) +
                                        rows_imm(XP, KVW * w + 16 * kh, 32 * dt, u),
                                    OFF_K, OFF_K + KV_B),
                                 "kf");
                        B0_CHECK(in(OFF_V + rows_lane(l, XP) +
                                        rows_imm(XP, KVW * w + 16 * kh, 32 * dt, u),
                                    OFF_V, OFF_V + KV_B),
                                 "vf");
                    }
            for (int j = 0; j < 2; ++j)
                for (int kc = 0; kc < NKC; ++kc)
                    for (int u = 0; u < 2; ++u)
                        B0_CHECK(
                            in(OFF_K + tr_lane(l, XP) + tr_imm(XP, 32 * kc, KVW * w + 16 * j, u),
                               OFF_K, OFF_K + KV_B),
                            "kt");
        }
    // dS^T exchange: the 4 waves' C^T stores tile [128 kv][32 q] exactly once; the tr loads read
    // exactly it
    std::vector<int> cell(BKV * 32, 0);
    for (int w = 0; w < NWAVE; ++w)
        for (int l = 0; l < WAVE; ++l)
            for (int hh = 0; hh < 2; ++hh)
                for (int kh = 0; kh < 2; ++kh)
                    for (int e = 0; e < 8; ++e) {
                        const int a =
                            rows_lane(l, SP) + ct_imm(SP, KVW * w + 16 * kh, 16 * hh) + 2 * e;
                        const int kv = a / SP, q = (a % SP) / 2;
                        B0_CHECK(kv == KVW * w + 16 * kh + l % 16 &&
                                     q == 16 * hh + 8 * (l / 16) + e && q < 32,
                                 "dS^T st map");
                        ++cell[kv * 32 + q];
                    }
    for (int c = 0; c < BKV * 32; ++c)
        B0_CHECK(cell[c] == 1, "dS^T cell %d", c);
    return 0;
}

// --- 3. global bounds, coverage and causal kinds for one shape
// ---------------------------------------------------
inline int check_shape(const Shape &s, int G, bool verbose = true) {
    const int nqp = nqp_of(s.Sq), nblk = s.Skv / BKV, cshift = s.Skv - s.Sq;
    B0_CHECK(host_gate(s.B, s.Sq, s.Skv, s.Hq, s.Hkv, s.causal, G) == GATE_OK,
             "host gate rejects B%d Sq%d Skv%d Hq%d Hkv%d", s.B, s.Sq, s.Skv, s.Hq, s.Hkv);
    const int64_t qb = qdo_bytes(s.B, s.Sq, s.Hq), lb = ld_bytes(s.B, s.Hq, s.Sq),
                  kb = kv_bytes(s.B, s.Skv, s.Hkv);
    // exactly-once coverage of (b, qh, q pair) per (b, hkv, blk) and of the visible (q, kv) pairs
    std::vector<uint8_t> pair_seen;
    int64_t              steps = 0, wave_steps_live = 0, vis_cells = 0;
    for (int b = 0; b < s.B; ++b)
        for (int hkv = 0; hkv < s.Hkv; ++hkv)
            for (int blk = 0; blk < nblk; ++blk) {
                const Work wk   = work_v0(hkv, blk, b);
                const int  kvw0 = kvw0_of(wk.blk);
                const int  n    = n_all(kvw0, cshift, s.causal, nqp, G),
                          nf    = n_full(kvw0, cshift, s.causal, nqp, G);
                B0_CHECK(n >= G && n % G == 0 && nf % G == 0 && nf <= n, "split n %d nf %d", n, nf);
                B0_CHECK(n % 2 == 0 || G % 2 == 1, "x2 unroll parity");
                pair_seen.assign((size_t) G * nqp, 0);
                // K / V images
                B0_CHECK(kv_tile_byte(b, s.Skv, s.Hkv, hkv, kvw0) >= 0 &&
                             kv_tile_end(b, s.Skv, s.Hkv, hkv, kvw0) <= kb,
                         "kv tile");
                // epilogue resource
                B0_CHECK(epi_base_byte(b, s.Skv, s.Hkv, hkv, kvw0) + epi_bytes(s.Hkv) <= kb,
                         "epi extent");
                for (int st = 0; st < n + LEAD + 1; ++st) { // every prefetched (clamped) tile too
                    const int qt = tile_qt(st, n, G, nqp), gh = tile_gh(st, n, G),
                              qh = qh_of(hkv, G, gh), q0 = QS * qt;
                    B0_CHECK(qt >= qp_start(kvw0, cshift, s.causal) && qt < nqp && gh >= 0 &&
                                 gh < G,
                             "tile range");
                    B0_CHECK(qdo_tile_byte(b, s.Sq, s.Hq, qh, q0) >= 0 &&
                                 qdo_tile_end(b, s.Sq, s.Hq, qh, q0) <= qb,
                             "qdo tile");
                    B0_CHECK(ld_rec_byte(b, s.Hq, qh, s.Sq, q0) >= 0 &&
                                 ld_rec_byte(b, s.Hq, qh, s.Sq, q0) + LD_B <= lb,
                             "ld tile");
                    if (st >= n)
                        continue;
                    ++steps;
                    B0_CHECK(!pair_seen[(size_t) gh * nqp + qt], "pair twice");
                    pair_seen[(size_t) gh * nqp + qt] = 1;
                    // atomics of this step: inside the WG's group region
                    const int64_t gb = dqa_grp_byte(b, s.Hq, hkv, G, s.Sq),
                                  ge = gb + dqa_grp_bytes(G, s.Sq);
                    B0_CHECK(ge <= dqa_bytes(s.B, s.Hq, s.Sq), "grp extent");
                    const int vmax = dqa_voff(31, 3, gh, qt, nqp) + dqa_imm(1, 1, 7) + 4;
                    B0_CHECK(dqa_voff(0, 0, gh, qt, nqp) >= 0 && gb + vmax <= ge, "atomic extent");
                    B0_CHECK((gb + dqa_voff(0, 0, gh, qt, nqp)) / 16384 ==
                                 ((int64_t) b * s.Hq + qh) * nqp + qt,
                             "atomic block");
                    // causal kinds vs the true mask
                    const bool masked_step = st >= nf;
                    for (int w = 0; w < NWAVE; ++w) {
                        const int k = wave_kind(kvw0, w, qt, cshift, s.causal);
                        if (!masked_step)
                            B0_CHECK(k == K_FULL, "full step with masked wave");
                        if (k != K_DEAD)
                            ++wave_steps_live;
                        if (s.exhaustive) {
                            int vis = 0;
                            for (int qq = 0; qq < QS; ++qq)
                                for (int kk = 0; kk < KVW; ++kk) {
                                    const int  q = q0 + qq, kv = kvw0 + KVW * w + kk;
                                    const bool v = !s.causal || kv <= q + cshift;
                                    vis += v;
                                    // the in-tile mask (hh = qq / 16, kh = kk / 16, m = qq % 16, n
                                    // = kk % 16) agrees
                                    const int off = mask_off(qt, qq / 16, kvw0, w, kk / 16, cshift);
                                    const bool mk = s.causal && (kk % 16) > (qq % 16) + off;
                                    B0_CHECK(mk == !v, "mask_off");
                                }
                            vis_cells += vis;
                            if (k == K_DEAD)
                                B0_CHECK(vis == 0, "dead wave with visible pair");
                            if (k == K_FULL)
                                B0_CHECK(vis == QS * KVW, "full wave with masked pair");
                        }
                    }
                }
                // every (head, pair) with a visible kv of this block is visited exactly once
                for (int gh = 0; gh < G; ++gh)
                    for (int qt = 0; qt < nqp; ++qt) {
                        const bool any = !s.causal || (QS * qt + QS - 1 + cshift >= kvw0);
                        B0_CHECK((bool) pair_seen[(size_t) gh * nqp + qt] == any,
                                 "coverage gh %d qt %d", gh, qt);
                    }
            }
    if (s.exhaustive) {
        int64_t want = 0; // visible (q, kv) pairs x heads
        for (int q = 0; q < s.Sq; ++q)
            for (int kv = 0; kv < s.Skv; ++kv)
                want += !s.causal || kv <= q + cshift;
        want *= (int64_t) s.B * s.Hq;
        B0_CHECK(vis_cells == want, "visible cells %lld want %lld", (long long) vis_cells,
                 (long long) want);
        // epilogue: every (b, kv, hkv, d) written exactly once
        std::vector<uint8_t> e((size_t) kb / 2, 0);
        for (int b = 0; b < s.B; ++b)
            for (int hkv = 0; hkv < s.Hkv; ++hkv)
                for (int blk = 0; blk < nblk; ++blk)
                    for (int w = 0; w < NWAVE; ++w)
                        for (int l = 0; l < WAVE; ++l)
                            for (int kh = 0; kh < 2; ++kh)
                                for (int dt = 0; dt < NDO; ++dt) {
                                    const int64_t a =
                                        epi_base_byte(b, s.Skv, s.Hkv, hkv, kvw0_of(blk)) +
                                        epi_voff(l, w, kh, dt, s.Hkv);
                                    B0_CHECK(epi_voff(l, w, kh, dt, s.Hkv) + 16 <= epi_bytes(s.Hkv),
                                             "epi voff");
                                    for (int x = 0; x < 8; ++x) {
                                        const int64_t el = a / 2 + x;
                                        B0_CHECK(el >= 0 && el < (int64_t) e.size() && !e[el],
                                                 "epi twice");
                                        e[el]             = 1;
                                        const int64_t row = el / (s.Hkv * D),
                                                      col = el % (s.Hkv * D);
                                        B0_CHECK(col / D == hkv &&
                                                     row == (int64_t) b * s.Skv + kvw0_of(blk) +
                                                                KVW * w + 16 * kh + l % 16 &&
                                                     col % D == 16 * dt + 8 * (l / 16) + x,
                                                 "epi map");
                                    }
                                }
        for (size_t i = 0; i < e.size(); ++i)
            B0_CHECK(e[i], "epi hole %zu", i);
    }
    // v1 pairing: every block exactly once when v1_ok; the stated balance when non-causal or Sq ==
    // Skv; an odd nblk leaves the middle block unvisited by the (Hkv, nblk / 2, B) grid (why v1_ok
    // gates it)
    {
        std::vector<int> seen(nblk, 0);
        int64_t          smin = INT64_MAX, smax = 0;
        for (int by = 0; by < nblk / 2; ++by) {
            int64_t st = 0;
            for (int sub = 0; sub < 2; ++sub) {
                const Work wk = work_v1(0, by, 0, sub, nblk);
                B0_CHECK(wk.blk >= 0 && wk.blk < nblk, "v1 block range");
                ++seen[wk.blk];
                st += n_all(kvw0_of(wk.blk), cshift, s.causal, nqp, G);
            }
            smin = std::min(smin, st);
            smax = std::max(smax, st);
        }
        if (v1_ok(nblk)) {
            for (int blk = 0; blk < nblk; ++blk)
                B0_CHECK(seen[blk] == 1, "v1 block %d seen %d times", blk, seen[blk]);
            if (!s.causal)
                B0_CHECK(smin == smax && smax == 2LL * G * nqp, "v1 balance (non-causal)");
            else if (cshift == 0)
                B0_CHECK(smin == smax && smax == (int64_t) G * (2 * nqp - 4 * (nblk - 1)),
                         "v1 balance");
        } else {
            B0_CHECK(nblk % 2 == 0 || seen[nblk / 2] == 0,
                     "odd nblk: middle block expected unvisited");
        }
    }
    if (verbose)
        std::printf("  shape B%d Sq%d Skv%d Hq%d Hkv%d c%d: %lld WG-steps, %lld live wave-steps\n",
                    s.B, s.Sq, s.Skv, s.Hq, s.Hkv, s.causal, (long long) steps,
                    (long long) wave_steps_live);
    return 0;
}

// --- 3b. host gate: rejects empty / misaligned / oversized shapes; every accepted shape keeps
// n_all >= G -------
inline int check_gate() {
    struct C {
        int64_t B, Sq, Skv, Hq, Hkv;
        int     causal, want;
    } cs[] = {{4, 8192, 8192, 32, 8, 1, GATE_OK},
              {1, 256, 128, 4, 1, 0, GATE_OK}, // non-causal Sq > Skv is fine
              {1, 0, 128, 4, 1, 1, GATE_EMPTY},
              {1, 0, 128, 4, 1, 0, GATE_EMPTY}, // Sq = 0: the review case
              {0, 64, 128, 4, 1, 1, GATE_EMPTY},
              {1, 64, 0, 4, 1, 0, GATE_EMPTY},
              {1, 64, 128, 0, 0, 1, GATE_EMPTY},
              {1, 64, 128, 8, 1, 1, GATE_G},
              {1, 96, 128, 4, 1, 1, GATE_SQ},
              {1, 64, 192, 4, 1, 1, GATE_SKV},
              {1, 256, 128, 4, 1, 1, GATE_CAUSAL},
              {16, 8192, 8192, 32, 8, 1, GATE_BYTES}, // dQ_acc 2 GiB
              {1, 64, 1LL << 26, 4, 1, 0, GATE_BYTES}};
    for (const C &c : cs) {
        const int got = host_gate(c.B, c.Sq, c.Skv, c.Hq, c.Hkv, c.causal, 4);
        B0_CHECK(got == c.want, "gate B%lld Sq%lld Skv%lld Hq%lld Hkv%lld c%d: %d want %d",
                 (long long) c.B, (long long) c.Sq, (long long) c.Skv, (long long) c.Hq,
                 (long long) c.Hkv, c.causal, got, c.want);
    }
    // Sq = 0 would reach the kernel with n = 0: step_clamp(0, 0) = -1 -> tile_qt = -1 (q0 = -32,
    // TDM before DO / Q)
    B0_CHECK(tile_qt(0, 0, 4, 0) == -1 && n_all(0, 128, 1, 0, 4) == 0, "Sq = 0 failure mode");
    // sweep: every accepted (Sq, Skv, causal) of a grid passes the full shape check, quietly (n >=
    // G inside)
    int nacc = 0;
    for (int Sq = 0; Sq <= 640; Sq += 32)
        for (int Skv = 0; Skv <= 768; Skv += 64)
            for (int causal = 0; causal < 2; ++causal) {
                if (host_gate(1, Sq, Skv, 4, 1, causal, 4) != GATE_OK)
                    continue;
                ++nacc;
                if (check_shape(Shape{1, Sq, Skv, 4, 1, causal, Sq * Skv <= 128 * 1024}, 4, false))
                    return 1;
            }
    std::printf("  host gate: %d shape cases, %d accepted sweep shapes checked\n",
                (int) (sizeof(cs) / sizeof(cs[0])), nacc);
    return 0;
}

// --- 4. ring / barrier protocol replay (all waves run one program; issuers = one op per step)
// --------------------- Event model: SIG(x) / WAIT(x) barrier phases, DSCNT(n) (all but the n
// youngest DS ops retired), TCNT(n) (all but the n youngest TDM ops of the issuer retired), DS
// reads / stores tagged with what they touch, TDM issue. Cross-wave happens-before: op retired
// before SIG(x) (in program order) -> anything after WAIT(x).
enum Ev { SIG, WAIT, DSCNT, TCNT, TDM, RD_SLOT, ST_DS, RD_DS };
struct E {
    Ev  ev;
    int a;
}; // a = barrier id / count / tile index
struct Prog {
    std::vector<E> v;
    void           add(Ev e, int a) { v.push_back(E{e, a}); }
};
// v0 step s (6.6 / 6.7 of the spec). FULL: P4 stores dS(s), then the 32 E tr16 of slot s; P5 TCNT
// 1, DSCNT 32 (modelled as DSCNT 1: the one younger "op" stands for those 32 tr16), SIG s; P6 E; P7
// WAIT s; P8 TDM(s + 3); P9 readback of slot s+1, then dSa = RD dS(s) (consumed by dQ(s) in step s
// + 1). MASKED: DSCNT 0 and the E tr16 after SIG. Prologue: readback of tile 0; dSa(-1) = 0 in
// registers (no LDS read). `uncov` = the modelled younger-op count at the FULL signal (1; 2 is a
// negative control: the dS^T stores would still be in flight at the signal).
inline Prog prog_v0(int n, int nf, int uncov = 1) {
    Prog p;
    p.add(TDM, 0);
    p.add(TDM, 1);
    p.add(TDM, 2);
    p.add(TCNT, 2);
    p.add(DSCNT, 0);
    p.add(SIG, -1);
    p.add(WAIT, -1);
    p.add(RD_SLOT, 0);
    for (int s = 0; s < n; ++s) {
        const bool m = s >= nf;
        p.add(ST_DS, s);
        if (!m)
            p.add(RD_SLOT, s);
        p.add(TCNT, TCNT_STEADY);
        p.add(DSCNT, m ? 0 : uncov);
        p.add(SIG, s);
        if (m)
            p.add(RD_SLOT, s);
        p.add(WAIT, s);
        p.add(TDM, s + LEAD);
        p.add(RD_SLOT, s + 1);
        p.add(RD_DS, s);
    }
    p.add(TCNT, 0);
    p.add(DSCNT, 0);
    p.add(SIG, 1 << 20);
    p.add(WAIT, 1 << 20);
    return p;
}
// v1 step t (11.x of the spec): WAIT t-1; TDM(t+3); RD slot t-1 (dV/dK(t-1) tr16, part 2); RD slot
// t+1 (readback for S/dP(t+1)); ST dS(t); RD dS(t-1) (dQ(t-1) A operands, the younger loads at the
// signal); TCNT 1; DSCNT 1; SIG t; RD slot t (tr16 prefetch for dV/dK(t), part 1). Drain step t =
// n: no softmax.
inline Prog prog_v1(int n) {
    Prog p;
    p.add(TDM, 0);
    p.add(TDM, 1);
    p.add(TDM, 2);
    p.add(TCNT, 1);
    p.add(DSCNT, 0);
    p.add(SIG, -1);
    p.add(WAIT, -1);
    p.add(RD_SLOT, 0); // readback for S/dP(0)
    for (int t = 0; t <= n; ++t) {
        if (t > 0)
            p.add(WAIT, t - 1);
        p.add(TDM, t + LEAD);
        if (t > 0)
            p.add(RD_SLOT, t - 1);
        if (t + 1 < n)
            p.add(RD_SLOT, t + 1);
        if (t < n)
            p.add(ST_DS, t);
        if (t > 0)
            p.add(RD_DS, t - 1);
        p.add(TCNT, TCNT_STEADY);
        p.add(DSCNT, t > 0 ? 1 : 0);
        p.add(SIG, t);
        if (t < n)
            p.add(RD_SLOT, t);
    }
    p.add(WAIT, n);
    p.add(TCNT, 0);
    p.add(DSCNT, 0);
    p.add(SIG, 1 << 20);
    p.add(WAIT, 1 << 20);
    return p;
}
inline int check_protocol(const Prog &p, int ring, int nds, int n, const char *name,
                          bool expect_ok) {
    // positions of SIG / WAIT per barrier id, retire position of every DS op / TDM op
    struct Op {
        Ev  ev;
        int tile;
        int pos;
        int retire;
    };
    std::vector<Op>                  ds, tdm;
    std::deque<int>                  dsq, tq;
    std::vector<std::pair<int, int>> sig, wt; // (id, pos)
    int                              inflight_max = 0;
    for (int i = 0; i < (int) p.v.size(); ++i) {
        const E &e = p.v[i];
        switch (e.ev) {
        case SIG:
            sig.push_back({e.a, i});
            break;
        case WAIT:
            wt.push_back({e.a, i});
            break;
        case DSCNT:
            while ((int) dsq.size() > e.a) {
                ds[dsq.front()].retire = i;
                dsq.pop_front();
            }
            break;
        case TCNT:
            while ((int) tq.size() > e.a) {
                tdm[tq.front()].retire = i;
                tq.pop_front();
            }
            break;
        case TDM:
            tdm.push_back({TDM, e.a, i, -1});
            tq.push_back((int) tdm.size() - 1);
            inflight_max = std::max(inflight_max, (int) tq.size());
            break;
        default:
            ds.push_back({e.ev, e.a, i, -1});
            dsq.push_back((int) ds.size() - 1);
            break;
        }
    }
    int bad = 0;
    B0_CHECK(dsq.empty() && tq.empty(), "%s: ops not retired at exit", name);
    B0_CHECK(sig.size() == wt.size(), "%s: signal / wait count", name);
    for (size_t k = 0; k < sig.size(); ++k)
        B0_CHECK(sig[k].first == wt[k].first && sig[k].second < wt[k].second &&
                     (k == 0 || wt[k - 1].second < sig[k].second),
                 "%s: barrier phases out of order", name);
    // hb(x_retire, y): some barrier with retire < SIG < WAIT < y
    auto hb = [&](int retire, int y) {
        for (size_t k = 0; k < sig.size(); ++k)
            if (retire >= 0 && retire < sig[k].second && wt[k].second < y)
                return true;
        return false;
    };
    auto tdm_of = [&](int k) -> const Op * {
        for (auto &o : tdm)
            if (o.tile == k)
                return &o;
        return nullptr;
    };
    for (auto &r : ds) {
        if (r.ev == RD_SLOT) { // RAW: the TDM that filled this slot retired, then a barrier
            const Op *t = tdm_of(r.tile);
            if (!t || !hb(t->retire, r.pos)) {
                if (expect_ok)
                    std::printf("  %s: RAW slot tile %d at %d\n", name, r.tile, r.pos);
                bad = 1;
            }
        }
        if (r.ev == RD_DS) { // RAW: every wave's dS(t) store retired, then a barrier
            for (auto &st : ds)
                if (st.ev == ST_DS && st.tile == r.tile && !hb(st.retire, r.pos)) {
                    if (expect_ok)
                        std::printf("  %s: RAW dS %d\n", name, r.tile);
                    bad = 1;
                }
        }
    }
    for (auto &t : tdm) { // WAR: every read of the slot's previous occupant retired, then a barrier
        for (auto &r : ds)
            if (r.ev == RD_SLOT && r.tile < t.tile &&
                slot_of(r.tile, ring) == slot_of(t.tile, ring) && r.pos < t.pos &&
                !hb(r.retire, t.pos)) {
                if (expect_ok)
                    std::printf("  %s: WAR slot tile %d by TDM %d\n", name, r.tile, t.tile);
                bad = 1;
            }
        for (auto &r : ds) // and no read of an older occupant happens after this overwrite
            if (r.ev == RD_SLOT && r.tile < t.tile &&
                slot_of(r.tile, ring) == slot_of(t.tile, ring) && r.pos > t.pos) {
                if (expect_ok)
                    std::printf("  %s: stale read tile %d after TDM %d\n", name, r.tile, t.tile);
                bad = 1;
            }
    }
    for (auto &st : ds) // WAR on the dS^T tiles
        if (st.ev == ST_DS)
            for (auto &r : ds)
                if (r.ev == RD_DS && r.tile < st.tile &&
                    dsb_of(r.tile, nds) == dsb_of(st.tile, nds) && !hb(r.retire, st.pos)) {
                    if (expect_ok)
                        std::printf("  %s: WAR dS %d by store %d\n", name, r.tile, st.tile);
                    bad = 1;
                }
    if (inflight_max > 3) {
        if (expect_ok)
            std::printf("  %s: %d TDM ops in flight\n", name, inflight_max);
        bad = 1;
    }
    (void) n;
    B0_CHECK(bad == (expect_ok ? 0 : 1), "%s: protocol %s", name,
             expect_ok ? "violated" : "negative control passed");
    return 0;
}

inline int selftest() {
    std::printf("b0 selftest: XP %d SP %d LDS %d B, W0_QDO 0x%08x W0_LD 0x%08x\n", XP, SP,
                LDS_BYTES, W0_QDO, W0_LD);
    if (check_cfrag() || check_lds() || check_gate())
        return 1;
    const Shape shapes[] = {
        {4, 8192, 8192, 32, 8, 1, false},                               // prod
        {4, 8192, 8192, 32, 8, 0, false},                               // prod, not causal
        {1, 256, 256, 4, 1, 1, true},     {2, 192, 384, 8, 2, 1, true}, // Sq < Skv
        {1, 128, 512, 4, 1, 1, true},     {1, 64, 128, 4, 1, 1, true},
        {2, 128, 256, 8, 2, 0, true},     {1, 320, 640, 4, 1, 1, true}};
    for (const Shape &s : shapes)
        if (check_shape(s, s.Hq / s.Hkv))
            return 1;
    for (int n = 4; n <= 48; n += 4) // G 4: n = 4 * pairs; nf = n - 16 (prod: 4 masked pairs) or 0
        for (int nf : {0, n >= 16 ? n - 16 : 0, n}) {
            if (check_protocol(prog_v0(n, nf), RING_V0, NDS_V0, n, "v0", true))
                return 1;
            if (check_protocol(prog_v0(n, nf), 3, NDS_V0, n, "v0 ring 3 (neg)", n < 2))
                return 1;
            if (check_protocol(prog_v0(n, nf), RING_V0, 1, n, "v0 nds 1 (neg)", n < 2))
                return 1;
            if (nf > 0 && check_protocol(prog_v0(n, nf, 2), RING_V0, NDS_V0, n,
                                         "v0 stores uncovered (neg)", false))
                return 1;
        }
    for (int n = 1; n <= 48; ++n) {
        if (check_protocol(prog_v1(n), RING_V1, NDS_V1, n, "v1", true))
            return 1;
        if (n >= 4 && check_protocol(prog_v1(n), 4, NDS_V1, n, "v1 ring 4 (neg)", false))
            return 1;
        if (n >= 4 && check_protocol(prog_v1(n), RING_V1, 2, n, "v1 nds 2 (neg)", false))
            return 1;
    }
    std::printf("b0 selftest: PASS\n");
    return 0;
}

} // namespace b0

int main() {
    return b0::selftest();
}
#endif // B0_IDX_SELFTEST
