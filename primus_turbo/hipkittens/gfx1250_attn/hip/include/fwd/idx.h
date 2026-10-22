/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// fwd/idx.h -- every index formula of the gfx1250 d128 GQA forward v0 (k_attn_fwd_v0), in ONE
// place. Verbatim copy of hk/notes/fwd_v0_idx_sketch.h (rev 2, after review; spec
// hk/notes/fwd_v0_design.md): the kernel (hip/src/k_attn_fwd_v0.hip) and its bounds checker
// (hk/lab/bounds/fwd_v0_check.cpp) both include THIS file.
//
//   O[b, p, hq, :] = softmax_kv(scale * Q[b, p, hq, :] K[b, kv, hkv, :]^T) V[b, kv, hkv, :],
//   causal bottom-right (kv <= p + Skv - Sq), LSE[b, hq, p] = natural log of the row sum (scaled
//   scores).
//
// constexpr __host__ __device__ functions: the kernel computes every global offset, LDS address,
// TDM descriptor word, ring slot, loop bound, mask offset and barrier / TDM event through these,
// and the bounds checker compiles THIS file for x86 and replays them. F0_IDX_SELFTEST adds
// f0::selftest(), the reference replay (LDS bounds + bank groups + TDM landing model, global bounds
// + exactly-once coverage + TRUE TDM extents, causal masks, an fp32 emulation of the scheduled
// online softmax, ring / barrier protocol, work maps); never part of a kernel build.
//
// Geometry: a workgroup (WG) = 4 waves (one per SIMD) = 64 query positions x 4 q heads (one head
// quad hq4 = heads [4 hq4, 4 hq4 + 4), all of kv head hkv = 4 hq4 / G). Wave w owns positions [64
// qb + 16 w, +16) of all 4 heads: q-tile h (0..3) = head 4 hq4 + h, lane row r = lane % 16 =
// position 64 qb + 16 w + r. kv streams in 64-row stages t (LDS ring, KD slots) of two 32-row
// chunks c = 2t + hc; every wave issues one 32-row TDM op per stage (waves 0 / 1: K rows [0, 32) /
// [32, 64), waves 2 / 3: the same of V). WMMA orientation: S^T = K Q^T (A = K, B = Q^T), O^T += V^T
// P^T (A = V^T via ds_load_tr16_b128, B = P^T packed from the S^T C fragments). C layout: lane l,
// element i <-> row 8 (l / 16) + i, column l % 16.
#pragma once

#if defined(__HIP__) || defined(__HIPCC__)
#define F0_HD __host__ __device__ __attribute__((always_inline)) constexpr
#else
#define F0_HD constexpr
#endif

#include <stdint.h>

// LDS bank model the pitches are chosen for (mb_tr16pitch settles it): 0 = "lin16" (a b128 / tr16
// is served per half-wave, lanes 0-15 / 16-31), 1 = "colhalf" (per tr16 column half, lanes with (l
// / 8) % 2 == 0 / 1).
#ifndef F0_BANK_GROUPING
#define F0_BANK_GROUPING 0
#endif

namespace f0 {

// ---- shape-independent constants
// -------------------------------------------------------------------------------
constexpr int D     = 128; // q/k/v head dim
constexpr int WAVE  = 32;
constexpr int NWAVE = 4; // 1 wave per SIMD, 1 WG per CU
constexpr int BLOCK = NWAVE * WAVE;
constexpr int QB    = 64;     // query positions per WG
constexpr int QW    = 16;     // query positions per wave (one 16-row q-tile per head)
constexpr int NH    = 4;      // q heads per WG = q-tiles per wave
constexpr int ST    = 64;     // kv rows per ring stage
constexpr int CH    = 32;     // kv rows per chunk (WMMA K depth of PV)
constexpr int OPR   = 32;     // kv rows per TDM op (half a stage; one op per wave per stage)
constexpr int NDK   = D / 32; // 4: WMMA k-steps of S^T = K Q^T
constexpr int NDT   = D / 16; // 8: 16-wide d tiles of O^T

// Pitches, PROVISIONAL until mb_tr16pitch: a conflict-free b128 needs row r at bank 4 r under lin16
// (8 r under colhalf), a conflict-free tr16 the other one, so K (b128) and V (tr16) swap 272 / 288
// between the models; the Q / O-image row (4 heads x 256 B) gets a 16 / 32-B pad per 1024 B. One
// switch, nothing else changes.
constexpr int BANK_GROUPING = F0_BANK_GROUPING;
static_assert(BANK_GROUPING == 0 || BANK_GROUPING == 1, "bank grouping: 0 lin16, 1 colhalf");
constexpr int K_PITCH = BANK_GROUPING == 0 ? 272 : 288; // K stage row: 256 B + pad (b128 reads)
constexpr int V_PITCH = BANK_GROUPING == 0 ? 288 : 272; // V stage row: 256 B + pad (tr16 reads)
constexpr int Q_PITCH =
    BANK_GROUPING == 0 ? 1040 : 1056;  // Q staging / O image row: 4 heads x 256 B + pad
constexpr int KD       = 4;            // ring depth (v1: 5)
constexpr int K_SLOT_B = ST * K_PITCH; // 17408 / 18432
constexpr int V_SLOT_B = ST * V_PITCH; // 18432 / 17408
constexpr int QO_W_B   = QW * Q_PITCH; // 16640 / 16896: one wave's Q staging / O image region

// LDS map (byte offsets from the single static allocation; 64-KB segments s = off >> 16)
constexpr int K_OFF     = 0;      // K ring: segment 0 (+ 4 / 8 KB of segment 1)
constexpr int DUMMY_OFF = 130048; // 4 x 16 B dummy-op cells (one per wave), segment 1
constexpr int V_OFF     = 131072; // V ring: segment 2 (+ 8 / 4 KB of segment 3)
constexpr int LDS_BYTES = 327680; // the whole 320 KB: one WG per CU
constexpr int QO_OFF    = LDS_BYTES - NWAVE * QO_W_B; // 4 per-wave Q / O regions, up to the end

// TDM roles: every wave issues exactly one 32-row op per stage (same bytes per wave: balanced under
// any wave -> SIMD -> TDM engine map); the role is a wave-uniform SALU select, never a branch.
F0_HD int op_role_v(int w) {
    return w >> 1;
} // waves 0, 1: K; waves 2, 3: V
F0_HD int op_half(int w) {
    return w & 1;
} // rows [32 half, +32) of the stage
F0_HD int issuer_of(int role_v, int half) {
    return 2 * role_v + half;
}
constexpr int TCNT_STEADY = 1; // s_wait_tensorcnt before every SIGNAL (op(t + 1) may fly)

// TDM group-1 word 0 (prims layout.h tdm_w0): data_size 2 B (1 << 16), pad_enable (1 << 20),
// pad_interval 2^(enc + 1) dw = the tile row (enc << 22), pad_amount enc + 1 dw = pitch - row (enc
// << 25)
F0_HD uint32_t tdm_w0(int row_b, int pitch_b) {
    int lg = 0;
    for (int x = row_b / 4; x > 1; x /= 2)
        ++lg;
    return (1u << 16) | (1u << 20) | ((uint32_t) (lg - 1) << 22) |
           ((uint32_t) ((pitch_b - row_b) / 4 - 1) << 25);
}
constexpr uint32_t K_W0 = tdm_w0(2 * D, K_PITCH);      // lin16: 0x07510000
constexpr uint32_t V_W0 = tdm_w0(2 * D, V_PITCH);      // lin16: 0x0f510000
constexpr uint32_t Q_W0 = tdm_w0(2 * NH * D, Q_PITCH); // lin16: 0x07d10000 (1024-B row, 16-B pad)
constexpr uint32_t G0_TYPE = 0x80000000u;              // group-0 word 3 type field [31:30] = 2

constexpr float LOG2E   = 1.4426950408889634f;
constexpr float LN2     = 0.6931471805599453f;
constexpr float TH2     = 8.0f * LOG2E; // lazy-rescale threshold: 8 nat in the log2 domain
constexpr float BIG_NEG = -1.0e30f;     // running-max seed (finite: no -inf - -inf)

// ---- integer helpers
// ---------------------------------------------------------------------------------------------
F0_HD int imin(int a, int b) {
    return a < b ? a : b;
}
F0_HD int imax(int a, int b) {
    return a > b ? a : b;
}

// ---- work decode
// -------------------------------------------------------------------------------------------------
struct Work {
    int b, hq4, qb;
};
// v0: 1-D grid of B * nhq4 * nqb WGs, FlyDSL's longest-first order: the first B * nhq4 dispatched
// WGs take the last (heaviest) q block of every (b, hq4), the next ones the one before, ... (XCD =
// i % 8 then sees one hq4).
F0_HD Work work_v0(int i, int B, int nhq4, int nqb) {
    const int nh = B * nhq4, rank = i / nh, rem = i - rank * nh;
    return Work{rem / nhq4, rem % nhq4, nqb - 1 - rank};
}
// v1 (persistent, reference only): W WGs, item j of WG g. Pairs (x, nqb - 1 - x) of one head hb = b
// * nhq4 + hq4 (heavy block first); WG g = xcd + nxcd * loc processes, at pair step j / 2, pair p =
// xcd * (W / nxcd) * ppw + (j / 2) * (W / nxcd) + loc, so the W / nxcd WGs of one XCD work on the
// same head at the same time. Requires nqb even, (B * nhq4 * nqb / 2) % W == 0, W % nxcd == 0 (else
// the host falls back to v0 order).
F0_HD Work work_v1(int g, int j, int W, int nxcd, int B, int nhq4, int nqb) {
    const int np = B * nhq4 * (nqb / 2), ppw = np / W, wx = W / nxcd;
    const int xcd = g % nxcd, loc = g / nxcd;
    const int p  = xcd * wx * ppw + (j / 2) * wx + loc;
    const int hb = p / (nqb / 2), pr = p % (nqb / 2);
    return Work{hb / nhq4, hb % nhq4, (j % 2) == 0 ? nqb - 1 - pr : pr};
}
F0_HD int items_per_wg_v1(int W, int B, int nhq4, int nqb) {
    return 2 * (B * nhq4 * (nqb / 2) / W);
}

F0_HD int hkv_of(int hq4, int G) {
    return (NH * hq4) / G;
} // G % 4 == 0 (v0)
F0_HD int wave_pos0(int qb, int w) {
    return QB * qb + QW * w;
}
// stages of the WG: causal -> through the diagonal stage qb + cst (cst = (Skv - Sq) / 64 >= 0);
// else all
F0_HD int n_stages(int qb, int cst, int nkv_st, int causal) {
    return causal != 0 ? qb + 1 + cst : nkv_st;
}

// ---- causal mask (branch-free; prims frag.h mask_row_gt_col)
// ------------------------------------------------------- Only the last stage n - 1 (chunks 2n - 2,
// 2n - 1) can hold masked elements: kv_rel = kv - 64 (n - 1), query row r of wave w sees kv_rel <=
// 16 w + r. kv tile u = 2 (c - (2n - 2)) + kt of that stage: element i of lane l is kv_rel = 16 u +
// 8 hf + i, visible <=> 8 hf + i <= r + 16 (w - u) <=> NOT (i > thr_rgc(l, off = 16 (w - u))). off
// >= 16 masks nothing, off <= -16 everything; every wave runs every chunk's WMMAs and softmax.
constexpr int MASK_NONE = 64; // offset that masks nothing (thr >= 56 > 7)
F0_HD bool    last_stage_chunk(int c, int n) {
    return c >= 2 * n - 2;
}
F0_HD int mask_off(int w, int c, int n, int kt, int causal) {
    return causal != 0 && last_stage_chunk(c, n) ? QW * (w - (2 * (c - (2 * n - 2)) + kt))
                                                 : MASK_NONE;
}
F0_HD int thr_rgc(int lane, int off) {
    return lane % 16 - 8 * (lane / 16) + off;
} // prims layout.h
F0_HD bool masked(int lane, int i, int off) {
    return i > thr_rgc(lane, off);
} // mask_row_gt_col test

// ---- ring
// --------------------------------------------------------------------------------------------------------
F0_HD int slot(int k) {
    return k % KD;
}
F0_HD int k_slot_off(int k) {
    return K_OFF + slot(k) * K_SLOT_B;
}
F0_HD int v_slot_off(int k) {
    return V_OFF + slot(k) * V_SLOT_B;
}
F0_HD int dummy_off(int w) {
    return DUMMY_OFF + 16 * w;
}
F0_HD bool op_real(int k, int n) {
    return k < n;
}
F0_HD int n_ops(int n) {
    return n + 2;
} // per wave: ops 0..n+1, ops >= n are 16-B dummies

// ---- global element offsets (int64)
// -------------------------------------------------------------------------------
F0_HD int64_t q_el(int b, int Sq, int p, int Hq, int hq) {
    return ((int64_t) b * Sq + p) * Hq * D + (int64_t) hq * D;
}
F0_HD int64_t kv_el(int b, int Skv, int kv, int Hkv, int hkv) {
    return ((int64_t) b * Skv + kv) * Hkv * D + (int64_t) hkv * D;
}
F0_HD int64_t lse_el(int b, int Hq, int hq, int Sq, int p) {
    return ((int64_t) b * Hq + hq) * Sq + p;
}

// ---- TDM descriptors (2-D only; TRUE extents: tensor_dim1 = rows below the origin, clamped to the
// tile) -----------
struct G1 {
    uint32_t w[8];
};
// group 1 of a 2-D op: dim0 = contiguous (elements), dim1 = rows, stride0 = row stride (elements,
// 48 bit)
F0_HD G1 g1_2d(uint32_t w0, uint32_t tdim0, uint32_t tdim1, uint32_t tile0, uint32_t tile1,
               uint64_t stride0) {
    return G1{{w0, (tdim0 & 0xFFFFu) << 16, (tdim0 >> 16) | ((tdim1 & 0xFFFFu) << 16),
               (tdim1 >> 16) | (tile0 << 16), tile1, (uint32_t) stride0,
               (uint32_t) (stride0 >> 32) & 0xFFFFu, 0u}};
}
// Q op of wave w: 2-D [16 positions][512 = 4 heads x 128] (the 4 heads of a position are 1 KB
// contiguous in BSHD), row stride Hq * 128, LDS row Q_PITCH: element (h, p, d) lands at p * Q_PITCH
// + 2 (128 h + d)
F0_HD int q_op_dim1(int Sq, int qb, int w) {
    return imin(imax(Sq - wave_pos0(qb, w), 0), QW);
}
F0_HD G1 q_g1(int Hq, int dim1) {
    return g1_2d(Q_W0, NH * D, (uint32_t) dim1, NH * D, QW, (uint64_t) Hq * D);
}
F0_HD int64_t q_op_el(int b, int Sq, int Hq, int hq4, int qb, int w) {
    return q_el(b, Sq, wave_pos0(qb, w), Hq, NH * hq4);
}
F0_HD int q_op_lds(int w) {
    return QO_OFF + w * QO_W_B;
}
// K / V op k of wave w: rows [64 k + 32 half, +32) of (b, hkv) of the role tensor, 2-D [32][128];
// dummy (k >= n): 8 x 1 at row 0 (always inside) into the wave's own cell
F0_HD int kv_op_row0(int w, int k) {
    return ST * k + OPR * op_half(w);
}
F0_HD int kv_op_dim1(int Skv, int w, int k) {
    return imin(imax(Skv - kv_op_row0(w, k), 0), OPR);
}
F0_HD G1 kv_g1(int role_v, int Hkv, int dim1) {
    return g1_2d(role_v ? V_W0 : K_W0, D, (uint32_t) dim1, D, OPR, (uint64_t) Hkv * D);
}
F0_HD G1 dummy_g1(int role_v) {
    return g1_2d(role_v ? V_W0 : K_W0, 8, 1, 8, 1, 0);
}
F0_HD int64_t kv_op_el(int b, int Skv, int Hkv, int hkv, int w, int k, int n) {
    return kv_el(b, Skv, op_real(k, n) ? kv_op_row0(w, k) : 0, Hkv, hkv);
}
F0_HD int kv_op_lds(int w, int k, int n) {
    return !op_real(k, n) ? dummy_off(w)
           : op_role_v(w) ? v_slot_off(k) + OPR * op_half(w) * V_PITCH
                          : k_slot_off(k) + OPR * op_half(w) * K_PITCH;
}
// TDM landing model (row t of an op at lds + t * pitch): LDS byte of stage row `row` (0..63),
// column col
F0_HD int k_land(int k, int row, int col) {
    return k_slot_off(k) + row * K_PITCH + 2 * col;
}
F0_HD int v_land(int k, int row, int col) {
    return v_slot_off(k) + row * V_PITCH + 2 * col;
}
F0_HD int q_land(int w, int h, int p, int col) {
    return q_op_lds(w) + p * Q_PITCH + 2 * (D * h + col);
}

// ---- LDS read addresses: per-lane base (VGPR, once per stage) + compile-time DS immediate
// ------------------------ K A operand of (chunk half hc, kv tile kt, k-step k): v16 = [b128(half
// 0), b128(half 1)]; lane: row l % 16, d = 32 k + 16 half + 8 (l / 16) .. +8
F0_HD int k_lane(int l) {
    return (l % 16) * K_PITCH + (l / 16) * 16;
}
F0_HD int k_imm(int hc, int kt, int k, int half) {
    return (CH * hc + 16 * kt) * K_PITCH + 64 * k + 32 * half;
}
// V^T A operand of (chunk half hc, d tile dt): v16 = [tr16(rows 0..15), tr16(rows 16..31)] of the
// chunk; lane address row 8 (l / 16) + l % 8, col 8 ((l / 8) % 2) (FlyDSL VManager16bV2)
F0_HD int v_lane(int l) {
    return (8 * (l / 16) + l % 8) * V_PITCH + ((l / 8) % 2) * 16;
}
F0_HD int v_imm(int hc, int half, int dt) {
    return (CH * hc + 16 * half) * V_PITCH + 32 * dt;
}
// Q^T B operand of (q-tile h, k-step k): lane: position l % 16 of head h, same d split as K
F0_HD int q_lane(int w, int l) {
    return q_op_lds(w) + (l % 16) * Q_PITCH + (l / 16) * 16;
}
F0_HD int q_imm(int h, int k, int half) {
    return 2 * D * h + 64 * k + 32 * half;
}
// O image (the wave's dead Q region, same layout): write = the O^T C fragment of (h, dt)
// (lane: position l % 16, d = 16 dt + 8 (l / 16) .. +8)
F0_HD int ow_lane(int w, int l) {
    return q_op_lds(w) + (l % 16) * Q_PITCH + (l / 16) * 16;
}
F0_HD int ow_imm(int h, int dt) {
    return 2 * D * h + 32 * dt;
}
// read-back of store s (0..31): position p = s / 2, bytes [512 (s % 2), +512) of the position's 1
// KB (heads 2 (s % 2), +1); lane l moves 16-B chunk or_chunk(l): identity under lin16, under
// colhalf each lane group takes 16 consecutive chunks (conflict-free both ways; the global store
// uses the same chunk)
F0_HD int or_chunk(int l) {
    return BANK_GROUPING == 0 ? l : l % 8 + 8 * (l / 16) + 16 * ((l / 8) % 2);
}
F0_HD int or_lane(int w, int l) {
    return q_op_lds(w) + 16 * or_chunk(l);
}
F0_HD int or_imm(int s) {
    return (s / 2) * Q_PITCH + 512 * (s % 2);
}

// ---- global stores (byte offsets; host asserts every extent < 2^31)
// ------------------------------------------------
F0_HD int64_t o_store_b(int b, int Sq, int Hq, int hq4, int qb, int w, int s, int l) {
    return 2 * q_el(b, Sq, wave_pos0(qb, w) + s / 2, Hq, NH * hq4) + 512 * (s % 2) +
           16 * or_chunk(l);
}
// LSE store k (0, 1): lane l writes q-tile 2 (l / 16) + k, row l % 16
F0_HD int64_t lse_store_b(int b, int Hq, int Sq, int hq4, int qb, int w, int k, int l) {
    return 4 * lse_el(b, Hq, NH * hq4 + 2 * (l / 16) + k, Sq, wave_pos0(qb, w) + l % 16);
}

// ---- iteration schedule (v0: LEAD 2, KD 4)
// -------------------------------------------------------------------------- iteration c (chunk c =
// 2t + hc, c in [0, 2n)):
//   A V(c) burst | B QK(c+1) || tails(c) h0..h2 | D barrier point | E K(c+2) burst | F PV(c) ||
//   SM1(c+1), tail(c) h3 | C rescale(c+1) (the only branch: uniform, at the iteration end).
// Prologue: Q op, ops 0 / 1, tensorcnt 1, Q burst, P_0 (full barrier), op 2, tensorcnt 1,
// SIGNAL(P_1), K(0) + K(1) bursts, QK(0), SM1(0) (m2 = mx2). Code: stages [0, main_stages)
// unmasked, stage n - 2 if n >= 2 (its SM1(2n-2) masked), the last stage n - 1 (SM1(2n - 1) masked,
// no QK(2n) / K(2n) / K(2n+1) / SM1(2n)), epilogue, exit. Barrier phase P_t (t in 1..n) publishes
// stage t and frees stage t - 2:
//   SIGNAL(P_t) at D(2t - 3) (prologue for t = 1), WAIT(P_t) at D(2t - 2) for t <= n - 1, then op(t
//   + 2) by every wave; WAIT(P_n) is the exit. The last stage has no barrier op.
enum BarOp : int { BAR_NONE = 0, BAR_SIGNAL = 1, BAR_WAIT = 2 };
F0_HD int bar_op(int c, int n) {
    if (last_stage_chunk(c, n))
        return BAR_NONE;
    return (c & 1) == 0 ? BAR_WAIT : BAR_SIGNAL;
}
F0_HD int bar_phase(int c) {
    return (c & 1) == 0 ? c / 2 + 1 : (c + 3) / 2;
}
F0_HD int op_at(int c, int n) {
    return bar_op(c, n) == BAR_WAIT ? c / 2 + 3 : -1;
} // issued right after WAIT
F0_HD int exit_phase(int n) {
    return n;
}
// chunks read by the bursts (stage c / 2, half c % 2: exactly the rows of one wave's op)
F0_HD int chunk_stage(int c) {
    return c / 2;
}
F0_HD int chunk_half(int c) {
    return c & 1;
}
F0_HD int v_read_chunk(int c) {
    return c;
} // A(c)
F0_HD int k_read_chunk(int c, int n) {
    return c + 2 < 2 * n ? c + 2 : -1;
} // E(c), -1 = no burst
// stage bodies: [0, main_stages) unmasked; stage n - 2 masked at its odd F iff has_pen_stage; then
// the last stage
F0_HD int main_stages(int n) {
    return imax(n - 2, 0);
}
F0_HD bool has_pen_stage(int n) {
    return n >= 2;
}

// ---- kernel-side (hot) forms
// ------------------------------------------------------------------------------------------
// Branch-free versions of kv_op_lds / kv_op_el and the real-or-dummy group 1, used by the kernel's
// op descriptor inside the stage loop: both candidates are evaluated, then plain scalar selects
// (s_cselect). The reference forms' nested conditionals with arithmetic in their arms compile to
// uniform branches in the loop (first build). hk/lab/bounds/fwd_v0_check.cpp proves hot ==
// reference for every (w, k, n) it replays (and n up to 300), and the selftest below for n
// = 1..140. k >= 0 (an op index).
F0_HD int kv_op_lds_hot(int w, int k, int n) {
    const int s = (int) ((uint32_t) k % (uint32_t) KD), half = op_half(w);
    const int kb = K_OFF + s * K_SLOT_B + OPR * half * K_PITCH;
    const int vb = V_OFF + s * V_SLOT_B + OPR * half * V_PITCH;
    const int rb = op_role_v(w) != 0 ? vb : kb;
    const int db = dummy_off(w);
    return op_real(k, n) ? rb : db;
}
F0_HD int64_t kv_op_el_hot(int b, int Skv, int Hkv, int hkv, int w, int k, int n) {
    const int r0  = kv_op_row0(w, k);
    const int row = op_real(k, n) ? r0 : 0;
    return kv_el(b, Skv, row, Hkv, hkv);
}
// kv_op_dim1 without a VALU v_med3_i32 (an i32 clamp of a uniform value is selected as v_med3 +
// readfirstlane; the i64 form compiles to v_cmp_*_i64 -> SGPR mask + s_cselect: no VGPR is written)
F0_HD int kv_op_dim1_hot(int Skv, int w, int k) {
    const int64_t r = (int64_t) Skv - (int64_t) kv_op_row0(w, k);
    return (int) (r < 0 ? 0 : r > OPR ? OPR : r);
}
// group 1 of op k: kv_g1(role_v, Hkv, dim1) if real, else dummy_g1(role_v) (dim1 = kv_op_dim1 of
// the op)
F0_HD G1 kv_op_g1_hot(int role_v, int Hkv, int dim1, bool real) {
    const G1 a = kv_g1(role_v, Hkv, dim1), d = dummy_g1(role_v);
    G1       r{};
    for (int i = 0; i < 8; ++i)
        r.w[i] = real ? a.w[i] : d.w[i];
    return r;
}

// ---- constants for the host proof (ctypes)
// ---------------------------------------------------------------------------
F0_HD long long kconst(int i) {
    return i == 0    ? D
           : i == 1  ? NWAVE
           : i == 2  ? QB
           : i == 3  ? QW
           : i == 4  ? NH
           : i == 5  ? ST
           : i == 6  ? CH
           : i == 7  ? K_PITCH
           : i == 8  ? V_PITCH
           : i == 9  ? KD
           : i == 10 ? K_SLOT_B
           : i == 11 ? V_SLOT_B
           : i == 12 ? QO_W_B
           : i == 13 ? K_OFF
           : i == 14 ? DUMMY_OFF
           : i == 15 ? V_OFF
           : i == 16 ? QO_OFF
           : i == 17 ? LDS_BYTES
           : i == 18 ? OPR
           : i == 19 ? Q_PITCH
           : i == 20 ? TCNT_STEADY
           : i == 21 ? (long long) K_W0
           : i == 22 ? (long long) V_W0
           : i == 23 ? (long long) Q_W0
           : i == 24 ? MASK_NONE
           : i == 25 ? BANK_GROUPING
                     : -1;
}

} // namespace f0

#ifdef F0_IDX_SELFTEST
// ======================================= reference replay (host only)
// =============================================
#include <cmath>
#include <cstdio>
#include <vector>

namespace f0 {

#define F0_CHECK(cond, ...)                                                                        \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                       \
            std::printf(__VA_ARGS__);                                                              \
            std::printf("\n");                                                                     \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

// bank model: each lane group of `grouping` is served in one pass; its 16 accesses of 16 B must
// cover 64 distinct banks (4 B each)
inline bool in_group(int grouping, int g, int l) {
    return grouping == 0 ? l / 16 == g : (l / 8) % 2 == g;
}
inline int bank_ok(const int *addr32, int grouping) {
    for (int g = 0; g < 2; ++g) {
        bool used[64] = {};
        for (int l = 0; l < 32; ++l)
            if (in_group(grouping, g, l))
                for (int j = 0; j < 4; ++j) {
                    const int bank = (addr32[l] / 4 + j) % 64;
                    if (used[bank])
                        return 0;
                    used[bank] = true;
                }
    }
    return 1;
}
// one access pattern = its instructions' lane addresses; must be conflict-free under BANK_GROUPING
struct Pat {
    const char                   *name;
    std::vector<std::vector<int>> ins;
};
inline int pat_ok(const Pat &p, int grouping) {
    for (auto &a : p.ins)
        if (!bank_ok(a.data(), grouping))
            return 0;
    return 1;
}

inline int check_lds(bool verbose) {
    // regions disjoint and inside the allocation; v1 ring depth 5 still fits below the next region
    struct R {
        int         lo, hi;
        const char *n;
    } rs[] = {{K_OFF, K_OFF + KD * K_SLOT_B, "K"},
              {V_OFF, V_OFF + KD * V_SLOT_B, "V"},
              {QO_OFF, QO_OFF + NWAVE * QO_W_B, "QO"},
              {DUMMY_OFF, DUMMY_OFF + 16 * NWAVE, "dummy"}};
    for (auto &a : rs) {
        F0_CHECK(a.lo >= 0 && a.hi <= LDS_BYTES && a.lo % 16 == 0, "region %s", a.n);
        for (auto &b : rs)
            F0_CHECK(&a == &b || a.hi <= b.lo || b.hi <= a.lo, "overlap %s %s", a.n, b.n);
    }
    F0_CHECK(K_OFF + 5 * K_SLOT_B <= DUMMY_OFF && V_OFF + 5 * V_SLOT_B <= QO_OFF, "v1 depth 5");
    for (int w = 0; w < NWAVE; ++w) // dummy cells: one per wave, inside the dummy region
        F0_CHECK(dummy_off(w) == DUMMY_OFF + 16 * w && kv_op_lds(w, 5, 5) == dummy_off(w),
                 "dummy cell %d", w);
    // TDM row / pad encodings == the pitches (pad_interval = row dwords, pad_amount = pad dwords)
    const uint32_t w0s[3]  = {K_W0, V_W0, Q_W0};
    const int      rowb[3] = {2 * D, 2 * D, 2 * NH * D}, pitch[3] = {K_PITCH, V_PITCH, Q_PITCH};
    for (int j = 0; j < 3; ++j) {
        const int ivl_b = 4 << (int) (((w0s[j] >> 22) & 7u) + 1),
                  pad_b = 4 * (int) (((w0s[j] >> 25) & 127u) + 1);
        F0_CHECK(ivl_b == rowb[j] && pad_b == pitch[j] - rowb[j], "tdm w0 %d", j);
        F0_CHECK(pitch[j] - rowb[j] > 0 && pitch[j] - rowb[j] <= 512 && rowb[j] <= 1024,
                 "tdm tile %d", j);
    }
    std::vector<Pat> pats;
    // K A operands: reads == landing of the op that wrote the half, inside that half, DS immediates
    // < 64 KB
    for (int hc = 0; hc < 2; ++hc) {
        Pat p{hc ? "K b128 (half 1)" : "K b128 (half 0)", {}};
        for (int kt = 0; kt < 2; ++kt)
            for (int k = 0; k < NDK; ++k)
                for (int half = 0; half < 2; ++half) {
                    std::vector<int> ad(32);
                    for (int l = 0; l < 32; ++l) {
                        const int row = CH * hc + 16 * kt + l % 16,
                                  col = 32 * k + 16 * half + 8 * (l / 16);
                        ad[l]         = k_slot_off(1) + k_lane(l) + k_imm(hc, kt, k, half);
                        F0_CHECK(ad[l] == k_land(1, row, col) && row / OPR == hc, "K read model");
                        const int w       = issuer_of(0, hc),
                                  r_in_op = row - OPR * op_half(w); // the op of the half's issuer
                        F0_CHECK(ad[l] == kv_op_lds(w, 1, 2) + r_in_op * K_PITCH + 2 * col,
                                 "K op landing");
                        F0_CHECK(k_imm(hc, kt, k, half) < 65536 && ad[l] % 16 == 0, "K imm");
                    }
                    p.ins.push_back(ad);
                }
        pats.push_back(p);
    }
    for (int hc = 0; hc < 2; ++hc) {
        Pat p{hc ? "V tr16 (half 1)" : "V tr16 (half 0)", {}};
        for (int half = 0; half < 2; ++half)
            for (int dt = 0; dt < NDT; ++dt) {
                std::vector<int> ad(32);
                for (int l = 0; l < 32; ++l) {
                    const int row = CH * hc + 16 * half + 8 * (l / 16) + l % 8,
                              col = 16 * dt + 8 * ((l / 8) % 2);
                    ad[l]         = v_slot_off(2) + v_lane(l) + v_imm(hc, half, dt);
                    F0_CHECK(ad[l] == v_land(2, row, col) && row / OPR == hc, "V read model");
                    const int w = issuer_of(1, hc), r_in_op = row - OPR * op_half(w);
                    F0_CHECK(ad[l] == kv_op_lds(w, 2, 3) + r_in_op * V_PITCH + 2 * col,
                             "V op landing");
                    F0_CHECK(v_imm(hc, half, dt) < 65536 && ad[l] % 16 == 0, "V imm");
                }
                p.ins.push_back(ad);
            }
        pats.push_back(p);
    }
    Pat pq{"Q^T b128", {}}, pw{"O image write", {}}, pr{"O read-back", {}};
    for (int w = 0; w < NWAVE; ++w) {
        for (int h = 0; h < NH; ++h)
            for (int k = 0; k < NDK; ++k)
                for (int half = 0; half < 2; ++half) {
                    std::vector<int> ad(32);
                    for (int l = 0; l < 32; ++l) {
                        ad[l] = q_lane(w, l) + q_imm(h, k, half);
                        F0_CHECK(ad[l] == q_land(w, h, l % 16, 32 * k + 16 * half + 8 * (l / 16)),
                                 "Q read model");
                        F0_CHECK(q_imm(h, k, half) < 65536 && ad[l] % 16 == 0, "Q imm");
                    }
                    pq.ins.push_back(ad);
                }
        // O image: every (h, p, d chunk) written once and read once, inside the wave's region
        std::vector<int> wr(QO_W_B / 16, 0), rd(QO_W_B / 16, 0);
        for (int h = 0; h < NH; ++h)
            for (int dt = 0; dt < NDT; ++dt) {
                std::vector<int> ad(32);
                for (int l = 0; l < 32; ++l) {
                    ad[l] = ow_lane(w, l) + ow_imm(h, dt);
                    F0_CHECK(ad[l] == q_land(w, h, l % 16, 16 * dt + 8 * (l / 16)),
                             "O write model");
                    F0_CHECK(ad[l] >= q_op_lds(w) && ad[l] + 16 <= q_op_lds(w) + QO_W_B,
                             "O write region");
                    wr[(ad[l] - q_op_lds(w)) / 16]++;
                }
                pw.ins.push_back(ad);
            }
        for (int s = 0; s < 2 * QW; ++s) {
            std::vector<int> ad(32);
            for (int l = 0; l < 32; ++l) {
                ad[l] = or_lane(w, l) + or_imm(s);
                const int ch =
                    32 * (s % 2) + or_chunk(l); // 16-B chunk of the position's 1 KB: head ch / 16
                F0_CHECK(ad[l] == q_land(w, ch / 16, s / 2, 8 * (ch % 16)), "O read model");
                F0_CHECK(or_imm(s) < 65536, "O read imm");
                rd[(ad[l] - q_op_lds(w)) / 16]++;
            }
            pr.ins.push_back(ad);
        }
        for (size_t i = 0; i < wr.size(); ++i)
            F0_CHECK(wr[i] == rd[i] && wr[i] <= 1, "O image coverage");
    }
    pats.push_back(pq);
    pats.push_back(pw);
    pats.push_back(pr);
    // Q TDM landing (one 2-D op per wave: tile row p = position, column 128 h + d)
    for (int w = 0; w < NWAVE; ++w)
        for (int p = 0; p < QW; ++p)
            for (int h = 0; h < NH; ++h)
                F0_CHECK(q_op_lds(w) + p * Q_PITCH + 2 * (D * h + 8) == q_land(w, h, p, 8),
                         "Q op landing");
    // bank groups: every pattern conflict-free under the selected model; the other model is
    // reported only
    for (auto &p : pats) {
        if (verbose)
            std::printf("  banks %-16s lin16 %-8s colhalf %s\n", p.name,
                        pat_ok(p, 0) ? "ok" : "conflict", pat_ok(p, 1) ? "ok" : "conflict");
        F0_CHECK(pat_ok(p, BANK_GROUPING), "%s banks (grouping %d)", p.name, BANK_GROUPING);
    }
    return 0;
}

// causal: the code's mask (applied at SM1 of the last stage's chunks with mask_off) == kv <= p +
// Skv - Sq (relative: kv <= 64 (n - 1) + 16 w + r) for every wave / chunk / element; chunk 0 has a
// visible key per row
inline int check_mask() {
    for (int n = 1; n < 9; ++n)
        for (int causal = 0; causal < 2; ++causal)
            for (int w = 0; w < NWAVE; ++w)
                for (int c = 0; c < 2 * n; ++c)
                    for (int kt = 0; kt < 2; ++kt)
                        for (int l = 0; l < 32; ++l)
                            for (int i = 0; i < 8; ++i) {
                                const int  kv = CH * c + 16 * kt + 8 * (l / 16) + i;
                                const bool def =
                                    causal == 0 || kv <= ST * (n - 1) + QW * w + l % 16;
                                const bool code = !(last_stage_chunk(c, n) &&
                                                    masked(l, i, mask_off(w, c, n, kt, causal)));
                                F0_CHECK(def == code,
                                         "mask n %d causal %d w %d c %d kt %d l %d i %d", n, causal,
                                         w, c, kt, l, i);
                                if (c == 0 && kt == 0 && l < 16 && i == 0)
                                    F0_CHECK(def, "chunk 0 kv 0 visible");
                            }
    return 0;
}

// fp32 emulation of one q-tile (16 query rows of wave w) through the scheduled softmax: SM1(c)
// (lane max, cross- half max, ballot over the wave) and C(c) before tails(c); m2 = mx2 at chunk 0
// (prologue), lazy rescale after; p = exp2(fma(s, c2, -m2)); per-lane partial row sums; masks of
// the last stage as -inf. vs a double reference.
inline int check_softmax(int n, int causal, int w, uint32_t seed) {
    const float        c2  = 0.08838834764831845f * LOG2E;
    const int          nkv = ST * n;
    std::vector<float> S((size_t) nkv * QW), V((size_t) nkv);
    auto               rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float) (seed >> 8) / 16777216.0f;
    };
    for (int kv = 0; kv < nkv; ++kv) {
        V[kv] = 2.0f * rnd() - 1.0f;
        for (int r = 0; r < QW; ++r)
            S[(size_t) kv * QW + r] = 80.0f * rnd() - 40.0f + 500.0f * kv / nkv; // rising max
    }
    auto  vis = [&](int kv, int r) { return causal == 0 || kv <= ST * (n - 1) + QW * w + r; };
    float m2[32], ll[32], O[32];
    for (int l = 0; l < 32; ++l) {
        m2[l] = BIG_NEG;
        ll[l] = 0.0f;
        O[l]  = 0.0f;
    }
    int rescales = 0;
    for (int c = 0; c < 2 * n; ++c) {
        float s[32][16], mx[32];
        for (int l = 0; l < 32; ++l) {
            mx[l] = -INFINITY;
            for (int kt = 0; kt < 2; ++kt)
                for (int i = 0; i < 8; ++i) {
                    const int  kv = CH * c + 16 * kt + 8 * (l / 16) + i;
                    const bool m =
                        last_stage_chunk(c, n) && masked(l, i, mask_off(w, c, n, kt, causal));
                    s[l][8 * kt + i] = m ? -INFINITY : S[(size_t) kv * QW + l % 16];
                    mx[l]            = std::fmax(mx[l], s[l][8 * kt + i]);
                }
        }
        float mx2[32];
        bool  bal = false;
        for (int l = 0; l < 32; ++l) {
            mx2[l] = std::fmax(mx[l], mx[l ^ 16]) * c2;
            bal    = bal || (mx2[l] - m2[l] > TH2);
        }
        if (c == 0) {
            for (int l = 0; l < 32; ++l) {
                F0_CHECK(std::isfinite(mx2[l]), "chunk 0 max");
                m2[l] = mx2[l];
            }
        } else if (bal) {
            ++rescales;
            for (int l = 0; l < 32; ++l) {
                const float m2n = std::fmax(m2[l], mx2[l]), corr = std::exp2(m2[l] - m2n);
                O[l] *= corr;
                ll[l] *= corr;
                m2[l] = m2n;
            }
        }
        for (int l = 0; l < 32; ++l) {
            float rs = 0.0f, pv = 0.0f;
            for (int e = 0; e < 16; ++e) {
                const float p  = std::exp2(std::fma(s[l][e], c2, -m2[l]));
                const int   kv = CH * c + 16 * (e / 8) + 8 * (l / 16) + e % 8;
                rs += p;
                pv += p * V[kv];
            }
            ll[l] += rs;
            O[l] += pv; // O: this lane's half of the column (the WMMA sums both halves)
        }
    }
    F0_CHECK(n < 2 || rescales > 0, "emulation never rescaled");
    for (int r = 0; r < QW; ++r) {
        const float lt = ll[r] + ll[r + 16], o = (O[r] + O[r + 16]) / lt,
                    lse = (m2[r] + std::log2(lt)) * LN2;
        double mref = -1e300, sum = 0.0, acc = 0.0;
        for (int kv = 0; kv < nkv; ++kv)
            if (vis(kv, r))
                mref = std::fmax(mref, (double) S[(size_t) kv * QW + r]);
        for (int kv = 0; kv < nkv; ++kv)
            if (vis(kv, r)) {
                const double e =
                    std::exp(0.08838834764831845 * ((double) S[(size_t) kv * QW + r] - mref));
                sum += e;
                acc += e * V[kv];
            }
        const double oref = acc / sum, lref = 0.08838834764831845 * mref + std::log(sum);
        F0_CHECK(std::isfinite(o) && std::isfinite(lse), "softmax finite n %d w %d r %d", n, w, r);
        F0_CHECK(std::fabs(o - oref) <= 1e-4 * (std::fabs(oref) + 1.0) &&
                     std::fabs(lse - lref) <= 1e-4 * (std::fabs(lref) + 1.0),
                 "softmax n %d causal %d w %d r %d: o %g / %g lse %g / %g", n, causal, w, r, o,
                 oref, lse, lref);
    }
    return 0;
}

// ---- ring / barrier protocol replay
// -------------------------------------------------------------------------------
enum Ev : int { E_TDM, E_TCNT, E_RD_Q, E_RD_K, E_RD_V, E_SIG, E_WAIT, E_BAR };
struct Event {
    int type, arg;
}; // E_TDM: op (-1 = Q); E_RD_K / E_RD_V: chunk; E_SIG / E_WAIT / E_BAR: phase

// the program order of wave w's memory / barrier events (all waves run the same program; only the
// op content differs by role)
inline std::vector<Event> wave_events(int w, int n) {
    (void) w;
    std::vector<Event> e;
    e.push_back({E_TDM, -1}); // own Q op
    e.push_back({E_TDM, 0});
    e.push_back({E_TDM, 1});
    e.push_back({E_TCNT, TCNT_STEADY}); // Q and op 0 landed
    e.push_back({E_RD_Q, 0});
    e.push_back({E_BAR, 0}); // P_0
    e.push_back({E_TDM, 2});
    e.push_back({E_TCNT, TCNT_STEADY});
    e.push_back({E_SIG, 1});
    e.push_back({E_RD_K, 0});
    e.push_back({E_RD_K, 1});
    for (int c = 0; c < 2 * n; ++c) {
        e.push_back({E_RD_V, v_read_chunk(c)}); // A(c): before D (load-bearing for WAR)
        const int bo = bar_op(c, n);
        if (bo == BAR_WAIT) {
            e.push_back({E_WAIT, bar_phase(c)});
            e.push_back({E_TDM, op_at(c, n)});
        }
        if (bo == BAR_SIGNAL) {
            e.push_back({E_TCNT, TCNT_STEADY});
            e.push_back({E_SIG, bar_phase(c)});
        }
        if (k_read_chunk(c, n) >= 0)
            e.push_back({E_RD_K, k_read_chunk(c, n)}); // E(c)
    }
    e.push_back({E_TCNT, 0});
    e.push_back({E_WAIT, exit_phase(n)}); // exit
    return e;
}

inline int check_protocol(int n) {
    std::vector<Event> ev[NWAVE];
    for (int w = 0; w < NWAVE; ++w)
        ev[w] = wave_events(w, n);
    for (int role = 0; role < 2; ++role)
        for (int half = 0; half < 2; ++half) {
            const int a = issuer_of(role, half);
            F0_CHECK(a >= 0 && a < NWAVE && op_role_v(a) == role && op_half(a) == half,
                     "issuer map");
        }
    // barrier sequence: identical across waves, SIGNAL / WAIT alternate in phase order, n + 1
    // phases
    std::vector<int> sig[NWAVE], wt[NWAVE];
    for (int w = 0; w < NWAVE; ++w) {
        sig[w].assign(n + 1, -1);
        wt[w].assign(n + 1, -1);
        int  expect = 0;
        bool open   = false;
        for (int i = 0; i < (int) ev[w].size(); ++i) {
            const Event &x = ev[w][i];
            if (x.type == E_BAR) {
                F0_CHECK(!open && x.arg == expect, "bar order");
                sig[w][x.arg] = wt[w][x.arg] = i;
                ++expect;
            }
            if (x.type == E_SIG) {
                F0_CHECK(!open && x.arg == expect, "sig order n %d", n);
                sig[w][x.arg] = i;
                open          = true;
            }
            if (x.type == E_WAIT) {
                F0_CHECK(open && x.arg == expect, "wait order n %d", n);
                wt[w][x.arg] = i;
                open         = false;
                ++expect;
            }
        }
        F0_CHECK(expect == n + 1 && !open, "n %d: %d phases", n, expect);
    }
    // happens-before (i in wave a) -> (j in wave b): the first phase a signals after i is waited by
    // b before j
    std::vector<int> nsig[NWAVE];
    for (int w = 0; w < NWAVE; ++w) {
        nsig[w].assign(ev[w].size(), n + 1);
        for (int i = (int) ev[w].size() - 1, x = n + 1; i >= 0; --i) {
            nsig[w][i] = x;
            for (int y = 0; y <= n; ++y)
                if (sig[w][y] == i)
                    x = y;
        }
    }
    auto hb = [&](int a, int i, int b, int j) {
        const int x = nsig[a][i];
        return x <= n && wt[b][x] >= 0 && wt[b][x] < j;
    };
    // TDM retire points per wave (in-order completion; s_wait_tensorcnt N retires all but the
    // newest N)
    std::vector<int> ops[NWAVE], ret_at[NWAVE], iss_at[NWAVE];
    for (int a = 0; a < NWAVE; ++a) {
        int retired = 0, nops = 0;
        for (int i = 0; i < (int) ev[a].size(); ++i) {
            const Event &x = ev[a][i];
            if (x.type == E_TDM) {
                ops[a].push_back(x.arg);
                ret_at[a].push_back(-1);
                iss_at[a].push_back(i);
                ++nops;
                F0_CHECK(nops - retired <= 3, "n %d: > 3 TDM in flight", n);
            }
            if (x.type == E_TCNT)
                for (; retired < nops - x.arg; ++retired)
                    ret_at[a][retired] = i;
            if (x.type == E_RD_Q)
                F0_CHECK(nops >= 1 && ret_at[a][0] >= 0, "Q read before its op retired");
        }
        F0_CHECK(nops == 1 + n_ops(n) && retired == nops, "n %d: op count", n);
    }
    // RAW: a read of chunk ch (stage s, half h) follows the retire of op(s) of the half's issuer
    // through a barrier; WAR: a real op k >= KD of wave a is issued after every read of (its role,
    // stage k - KD, its half)
    for (int b = 0; b < NWAVE; ++b)
        for (int j = 0; j < (int) ev[b].size(); ++j) {
            const Event &x = ev[b][j];
            if (x.type != E_RD_K && x.type != E_RD_V)
                continue;
            const int role = x.type == E_RD_V, s = chunk_stage(x.arg), h = chunk_half(x.arg),
                      a = issuer_of(role, h);
            F0_CHECK(op_real(s, n), "n %d: read of a dummy stage %d", n, s);
            int r = -1;
            for (size_t q = 0; q < ops[a].size(); ++q)
                if (ops[a][q] == s)
                    r = ret_at[a][q];
            F0_CHECK(r >= 0 && hb(a, r, b, j), "n %d: RAW stage %d role %d half %d wave %d", n, s,
                     role, h, b);
            for (int a2 = 0; a2 < NWAVE; ++a2)
                if (op_role_v(a2) == role && op_half(a2) == h)
                    for (size_t q = 0; q < ops[a2].size(); ++q) {
                        const int k = ops[a2][q];
                        if (k >= KD && op_real(k, n) && k - KD == s)
                            F0_CHECK(hb(b, j, a2, iss_at[a2][q]),
                                     "n %d: WAR op %d wave %d vs read wave %d", n, k, a2, b);
                    }
        }
    return 0;
}

// ---- global bounds + exactly-once coverage + TRUE TDM extents for one shape
// ----------------------------------------
inline int check_shape(int B, int Sq, int Skv, int Hq, int Hkv, int causal) {
    const int G = Hq / Hkv, nqb = Sq / QB, nhq4 = Hq / NH, nkv_st = Skv / ST, cst = (Skv - Sq) / ST;
    const int64_t q_n = (int64_t) B * Sq * Hq * D, kv_n = (int64_t) B * Skv * Hkv * D,
                  lse_n = (int64_t) B * Hq * Sq;
    F0_CHECK(2 * q_n < (1LL << 31) && 4 * lse_n < (1LL << 31), "extents");
    std::vector<uint8_t> ocov((size_t) (q_n / 8), 0), lcov((size_t) lse_n, 0),
        seen((size_t) B * nhq4 * nqb, 0);
    for (int i = 0; i < B * nhq4 * nqb; ++i) {
        const Work wk = work_v0(i, B, nhq4, nqb);
        F0_CHECK(wk.b >= 0 && wk.b < B && wk.hq4 >= 0 && wk.hq4 < nhq4 && wk.qb >= 0 && wk.qb < nqb,
                 "work");
        seen[((size_t) wk.b * nhq4 + wk.hq4) * nqb + wk.qb]++;
        if (i > 0)
            F0_CHECK(work_v0(i - 1, B, nhq4, nqb).qb >= wk.qb, "LPT order");
        const int hkv = hkv_of(wk.hq4, G), n = n_stages(wk.qb, cst, nkv_st, causal);
        F0_CHECK(n >= 1 && n <= nkv_st, "n_stages");
        for (int w = 0; w < NWAVE; ++w) {
            for (int k = 0; k < n_ops(n);
                 ++k) { // K / V ops: footprint inside the tensor, clamp == tile here
                const int64_t e0 = kv_op_el(wk.b, Skv, Hkv, hkv, w, k, n);
                int64_t       e1 = e0 + 8;
                if (op_real(k, n)) {
                    const int d1 = kv_op_dim1(Skv, w, k);
                    F0_CHECK(d1 == OPR, "kv tensor_dim1 %d", d1);
                    e1 = e0 + (int64_t) (d1 - 1) * Hkv * D + D;
                }
                F0_CHECK(e0 >= 0 && e1 <= kv_n, "kv op footprint");
            }
            const int64_t e0 = q_op_el(wk.b, Sq, Hq, wk.hq4, wk.qb, w);
            const int     d1 = q_op_dim1(Sq, wk.qb, w);
            F0_CHECK(d1 == QW, "q tensor_dim1 %d", d1);
            F0_CHECK(e0 >= 0 && e0 + (int64_t) (d1 - 1) * Hq * D + NH * D <= q_n, "q op footprint");
            for (int s = 0; s < 2 * QW; ++s)
                for (int l = 0; l < 32; ++l) {
                    const int64_t ob = o_store_b(wk.b, Sq, Hq, wk.hq4, wk.qb, w, s, l);
                    F0_CHECK(ob >= 0 && ob + 16 <= 2 * q_n && ob % 16 == 0, "o store");
                    ocov[(size_t) (ob / 16)]++;
                }
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 32; ++l) {
                    const int64_t lb = lse_store_b(wk.b, Hq, Sq, wk.hq4, wk.qb, w, k, l);
                    F0_CHECK(lb >= 0 && lb + 4 <= 4 * lse_n, "lse store");
                    lcov[(size_t) (lb / 4)]++;
                }
        }
    }
    for (auto x : seen)
        F0_CHECK(x == 1, "work bijection");
    for (auto x : ocov)
        F0_CHECK(x == 1, "O coverage");
    for (auto x : lcov)
        F0_CHECK(x == 1, "LSE coverage");
    return 0;
}

// v1 work map: bijection + equal stages per WG (prod-like shapes)
inline int check_v1(int W, int nxcd, int B, int nhq4, int nqb, int cst) {
    std::vector<uint8_t> seen((size_t) B * nhq4 * nqb, 0);
    long long            ref = -1;
    for (int g = 0; g < W; ++g) {
        long long st = 0;
        for (int j = 0; j < items_per_wg_v1(W, B, nhq4, nqb); ++j) {
            const Work wk = work_v1(g, j, W, nxcd, B, nhq4, nqb);
            seen[((size_t) wk.b * nhq4 + wk.hq4) * nqb + wk.qb]++;
            st += wk.qb + 1 + cst;
        }
        F0_CHECK(ref < 0 || st == ref, "v1 balance");
        ref = st;
    }
    for (auto x : seen)
        F0_CHECK(x == 1, "v1 bijection");
    return 0;
}

inline int selftest(bool verbose = false) {
    if (verbose) {
        const G1 q = q_g1(32, QW), kv = kv_g1(0, 8, OPR), vv = kv_g1(1, 8, OPR), dm = dummy_g1(0);
        std::printf("  grouping %d: K_PITCH %d V_PITCH %d Q_PITCH %d QO_OFF %d\n", BANK_GROUPING,
                    K_PITCH, V_PITCH, Q_PITCH, QO_OFF);
        const G1   *gs[4] = {&q, &kv, &vv, &dm};
        const char *nm[4] = {"Q op", "K op", "V op", "dummy"};
        for (int j = 0; j < 4; ++j) {
            std::printf("  g1 %-6s", nm[j]);
            for (int x = 0; x < 8; ++x)
                std::printf(" %08x", gs[j]->w[x]);
            std::printf("\n");
        }
    }
    F0_CHECK(kv_op_dim1(100, 1, 1) == 4 && kv_op_dim1(100, 0, 2) == 0 &&
                 q_op_dim1(72, 0, 3) == 16 && q_op_dim1(72, 1, 0) == 8,
             "clamp units");
    if (check_lds(verbose) || check_mask())
        return 1;
    for (int n = 1; n <= 140; ++n)
        if (check_protocol(n))
            return 1;
    for (int n = 1; n <= 140; ++n) // kernel-side hot forms == reference forms
        for (int w = 0; w < NWAVE; ++w)
            for (int k = 0; k < n_ops(n); ++k) {
                F0_CHECK(kv_op_lds_hot(w, k, n) == kv_op_lds(w, k, n),
                         "kv_op_lds_hot n %d w %d k %d", n, w, k);
                for (int skv = 0; skv <= 64 * n + 128; skv += 16)
                    F0_CHECK(kv_op_dim1_hot(skv, w, k) == kv_op_dim1(skv, w, k), "kv_op_dim1_hot");
                F0_CHECK(kv_op_el_hot(1, 64 * n + 64, 8, 3, w, k, n) ==
                             kv_op_el(1, 64 * n + 64, 8, 3, w, k, n),
                         "kv_op_el_hot");
                const G1 h = kv_op_g1_hot(op_role_v(w), 8, OPR, op_real(k, n));
                const G1 r = op_real(k, n) ? kv_g1(op_role_v(w), 8, OPR) : dummy_g1(op_role_v(w));
                for (int i = 0; i < 8; ++i)
                    F0_CHECK(h.w[i] == r.w[i], "kv_op_g1_hot n %d w %d k %d word %d", n, w, k, i);
            }
    for (int n = 1; n <= 6; ++n)
        for (int causal = 0; causal < 2; ++causal)
            for (int w = 0; w < NWAVE; ++w)
                if (check_softmax(n, causal, w, 12345u + 7u * n + 3u * w + causal))
                    return 1;
    const int shapes[][6] = {
        {1, 64, 4, 1, 64, 1},         {1, 128, 8, 2, 128, 1},  {2, 256, 16, 2, 256, 1},
        {1, 192, 16, 1, 192, 1},      {1, 128, 4, 1, 256, 1},  {2, 192, 8, 1, 320, 1},
        {1, 256, 8, 2, 128 + 256, 0}, {2, 512, 32, 8, 512, 1}, {4, 8192, 32, 8, 8192, 1}};
    for (auto &s : shapes)
        if (check_shape(s[0], s[1], s[4], s[2], s[3], s[5])) {
            std::printf("shape B%d Sq%d\n", s[0], s[1]);
            return 1;
        }
    if (check_v1(256, 8, 4, 8, 128, 0) || check_v1(64, 8, 2, 4, 64, 0))
        return 1;
    std::printf("f0 selftest OK (bank grouping %s)\n", BANK_GROUPING == 0 ? "lin16" : "colhalf");
    return 0;
}

} // namespace f0
#endif // F0_IDX_SELFTEST
