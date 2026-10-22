/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// fwdpp/idx.h -- every index formula of the gfx1250 d128 GQA forward k_attn_fwd_pp (8 waves, 2
// waves / SIMD, staggered ping-pong; hk/notes/fwd_pp_design.md), in ONE place, as constexpr
// __host__ __device__ functions of scalars (host-compilable: no HIP header). The kernel computes
// every global offset, LDS address, TDM descriptor word, ring slot and mask offset through these,
// and its barrier / burst / TDM / compute program is the stage plan cl_plan (body.h expands it at
// compile time); the bounds checker (hk/lab/bounds/fwd_pp_check.cpp) compiles this file for x86 and
// replays them (kernel_events = the same expansion of the plan).
//
//   O[b, p, hq, :] = softmax_kv(scale * Q[b, p, hq, :] K[b, kv, hkv, :]^T) V[b, kv, hkv, :],
//   causal bottom-right (kv <= p + Skv - Sq), LSE[b, hq, p] = natural log of the row sum (scaled
//   scores).
//
// Geometry: a WG (256 threads, 8 waves) = 64 query positions x 4 q heads (head quad hq4, kv head
// 4 hq4 / G). Wave w: SIMD pair s = w & 3 (w and w + 4 share a SIMD), group g = w >> 2 (0 lo, 1
// hi); it owns positions [64 qb + 16 s, +16) of heads 4 hq4 + 2 g + h, h = 0, 1 (its 2 q-tiles;
// lane row r = position). kv streams in 64-row stages (depth-4 ring) issued by the 4 lo waves only
// (one 32-row op each per stage: waves 0 / 1 K rows [0, 32) / [32, 64), waves 2 / 3 the same of V).
// Per wave: 32-kv chunks c, two clusters per chunk, X(c) = QK(c+1) || exp(c) and Y(c) = PV(c) ||
// cvt, sums(c), SM1(c+1), rescale; one WG barrier per cluster; the hi group runs one cluster behind
// (PP_STAGGER = 1). WMMA orientation as fwd_v0: S^T = K Q^T (A = K rows, B = Q^T rows), O^T += V^T
// P^T (A = V^T via tr16).
//
// PP_IDX_SELFTEST adds pp::selftest() (host only): LDS map / landing / bank model / segment plan,
// causal masks, an fp32 emulation of the scheduled softmax, the stage plan (cl_plan, the kernel's
// program) == the cluster-indexed spec and kernel_events == wave_events, the barrier / ring
// protocol replay of kernel_events for n = 1..140 and both lags, global bounds + exactly-once
// coverage + TRUE TDM extents on the proof shapes, work map.
#pragma once

#if defined(__HIP__) || defined(__HIPCC__)
#define PP_HD __host__ __device__ __attribute__((always_inline)) constexpr
#else
#define PP_HD constexpr
#endif

#include <stdint.h>

#ifndef PP_STAGGER
#define PP_STAGGER 1 // hi-group lag in clusters: 1 = staggered ping-pong, 0 = lockstep (A/B arm)
#endif

namespace pp {

// ---- shape-independent constants
// ------------------------------------------------------------------------------
constexpr int D     = 128; // q/k/v head dim
constexpr int WAVE  = 32;
constexpr int NWAVE = 8; // 2 waves per SIMD, 1 WG per CU
constexpr int NPAIR = 4; // SIMD pairs (= SIMDs)
constexpr int BLOCK = NWAVE * WAVE;
constexpr int QB    = 64;     // query positions per WG
constexpr int QW    = 16;     // query positions per wave (= per SIMD pair)
constexpr int NH    = 4;      // q heads per WG
constexpr int NHW   = 2;      // q heads (q-tiles) per wave
constexpr int ST    = 64;     // kv rows per ring stage
constexpr int CH    = 32;     // kv rows per chunk
constexpr int OPR   = 32;     // kv rows per TDM op (one stage half)
constexpr int NDK   = D / 32; // 4 k-steps of S^T = K Q^T
constexpr int NDT   = D / 16; // 8 d tiles of O^T
constexpr int LAG   = PP_STAGGER;
static_assert(LAG == 0 || LAG == 1,
              "PP_STAGGER: 0 or 1 (a lag of 2 clusters aligns X with X and breaks the WAR bound)");

// pitches (mb_tr16pitch: lin16 bank grouping confirmed): K b128 272, V tr16 288, Q / O image b128
// 528 = 33 x 16 B
constexpr int K_PITCH  = 272;
constexpr int V_PITCH  = 288;
constexpr int Q_PITCH  = 528;
constexpr int KD       = 4;             // ring depth (stages)
constexpr int SEG      = 65536;         // LDS segment
constexpr int K_HALF_B = OPR * K_PITCH; // 8704: one stage half of K (32 rows)
constexpr int V_HALF_B = OPR * V_PITCH; // 9216
constexpr int QO_W_B   = QW * Q_PITCH;  // 8448: one wave's Q staging / O image
constexpr int KH_OFF0 = 0 * SEG, VH_OFF0 = 1 * SEG, KH_OFF1 = 2 * SEG,
              VH_OFF1    = 3 * SEG; // half rings
constexpr int QO_SEG_OFF = 36864;   // Q / O regions: segment (w & 3), after the half rings
constexpr int DUMMY_SEG_OFF =
    QO_SEG_OFF + 2 * QO_W_B;      // 53760: lo wave s's two 16-B dummy cells, segment s
constexpr int LDS_BYTES = 327680; // whole 320 KB: one WG per CU
static_assert(KD * K_HALF_B <= QO_SEG_OFF && KD * V_HALF_B <= QO_SEG_OFF,
              "half rings below the Q / O regions");
static_assert(DUMMY_SEG_OFF + 32 <= SEG, "dummy cells inside their segment");

constexpr float LOG2E     = 1.4426950408889634f;
constexpr float LN2       = 0.6931471805599453f;
constexpr float TH2       = 8.0f * LOG2E; // lazy-rescale threshold: 8 nat in the log2 domain
constexpr int   MASK_NONE = 64;           // mask offset that masks nothing

// ---- integer helpers
// --------------------------------------------------------------------------------------------
PP_HD int imin(int a, int b) {
    return a < b ? a : b;
}
PP_HD int imax(int a, int b) {
    return a > b ? a : b;
}

// ---- waves
// --------------------------------------------------------------------------------------------------------
// measured wave -> SIMD of 8-wave WGs: [0, 3, 2, 1, 0, 3, 2, 1] (p1_fwd 3.1; t_ext_fwdpp t_pp_simd
// re-checks): waves w and w + 4 share a SIMD -> pair s = w & 3 holds lo wave s and hi wave s + 4
PP_HD int pair_of(int w) {
    return w & 3;
}
PP_HD int grp_of(int w) {
    return w >> 2;
}
PP_HD int lag_of(int g) {
    return LAG * g;
} // barriers the group runs behind

// ---- work decode (fwd_v0 work_v0 = FlyDSL _lpt_block_id folded into one dimension)
// -------------------------------
struct Work {
    int b, hq4, qb;
};
PP_HD Work work_v0(int i, int B, int nhq4, int nqb) {
    const int nh = B * nhq4, rank = i / nh, rem = i - rank * nh;
    return Work{rem / nhq4, rem % nhq4, nqb - 1 - rank};
}
PP_HD int hkv_of(int hq4, int G) {
    return (NH * hq4) / G;
} // G % 4 == 0
PP_HD int wave_pos0(int qb, int w) {
    return QB * qb + QW * pair_of(w);
}
PP_HD int wave_head0(int hq4, int w) {
    return NH * hq4 + NHW * grp_of(w);
}
PP_HD int n_stages(int qb, int cst, int nkv_st, int causal) {
    return causal != 0 ? qb + 1 + cst : nkv_st;
}

// ---- causal mask (branch-free, prims mask_row_gt_col on an S^T accumulator)
// ------------------------------------------ last stage n - 1 only: kv tile u = 2 (c - (2n - 2)) +
// kt; element i of lane l visible iff 8 hf + i <= r + 16 (s - u)
PP_HD bool last_stage_chunk(int c, int n) {
    return c >= 2 * n - 2;
}
PP_HD int mask_off(int s, int c, int n, int kt, int causal) {
    return causal != 0 && last_stage_chunk(c, n) ? QW * (s - (2 * (c - (2 * n - 2)) + kt))
                                                 : MASK_NONE;
}
PP_HD int thr_rgc(int lane, int off) {
    return lane % 16 - 8 * (lane / 16) + off;
} // prims layout.h
PP_HD bool masked(int lane, int i, int off) {
    return i > thr_rgc(lane, off);
}

// ---- LDS map
// ----------------------------------------------------------------------------------------------------
PP_HD int slot(int k) {
    return k % KD;
}
PP_HD int k_half_lds(int k, int hh) {
    return (hh ? KH_OFF1 : KH_OFF0) + slot(k) * K_HALF_B;
} // stage k half hh
PP_HD int v_half_lds(int k, int hh) {
    return (hh ? VH_OFF1 : VH_OFF0) + slot(k) * V_HALF_B;
}
PP_HD int k_chunk_lds(int c) {
    return k_half_lds(c >> 1, c & 1);
} // chunk c = stage c / 2, half c % 2
PP_HD int v_chunk_lds(int c) {
    return v_half_lds(c >> 1, c & 1);
}
PP_HD int qo_lds(int w) {
    return SEG * pair_of(w) + QO_SEG_OFF + QO_W_B * grp_of(w);
}
// dummy op k of lo wave s lands in cell k & 1: consecutive dummies are in flight together (two
// async writes of one cell would be an unordered WAW), ops k and k + 2 never are (<= 2 ops in
// flight after the prologue)
PP_HD int dummy_lds(int s, int k) {
    return SEG * s + DUMMY_SEG_OFF + 16 * (k & 1);
}
// wave w's 8-B scratch in the free segment 4: {c2, c2} written and read back once (keeps the pk_fma
// constant pair a VGPR value the allocator cannot rematerialize from SGPRs; x5: an SGPR-sourced
// v_mov_b64 per tile)
constexpr int C2_OFF = 4 * SEG;
PP_HD int     c2_lds(int w) {
    return C2_OFF + 16 * w;
} // 16 B per wave (one hazard granule)
PP_HD int seg_of(int off) {
    return off >> 16;
}

// LDS read addresses: per-lane base (VGPR) + compile-time DS immediate (< 64 KB)
// K A operand (kv tile kt, k-step k, register half e): row 16 kt + r of the chunk, d = 32 k + 16 e
// + 8 hf
PP_HD int k_lane(int l) {
    return (l % 16) * K_PITCH + (l / 16) * 16;
}
PP_HD int k_imm(int kt, int k, int e) {
    return 16 * kt * K_PITCH + 64 * k + 32 * e;
}
// V^T A operand (register half e, d tile dt): tr16 lane address row 16 e + 8 hf + l % 8, col 16 dt
// + 8 ((l / 8) % 2)
PP_HD int v_lane(int l) {
    return (8 * (l / 16) + l % 8) * V_PITCH + ((l / 8) % 2) * 16;
}
PP_HD int v_imm(int e, int dt) {
    return 16 * e * V_PITCH + 32 * dt;
}
// Q^T B operand (q-tile h, k-step k, half e): row r = position, bytes 256 h + 64 k + 32 e + 16 hf
PP_HD int q_lane(int w, int l) {
    return qo_lds(w) + (l % 16) * Q_PITCH + (l / 16) * 16;
}
PP_HD int q_imm(int h, int k, int e) {
    return 2 * D * h + 64 * k + 32 * e;
}
// O image write (C^T of O^T (h, dt)): row r, bytes 256 h + 32 dt + 16 hf  (same lane base as Q)
PP_HD int ow_imm(int h, int dt) {
    return 2 * D * h + 32 * dt;
}
// O read-back of position p: lane l moves bytes [16 l, +16) of the position's 512 B (heads 2 g, 2 g
// + 1)
PP_HD int or_lane(int w, int l) {
    return qo_lds(w) + 16 * l;
}
PP_HD int or_imm(int p) {
    return p * Q_PITCH;
}

// ---- global element offsets (int64)
// ---------------------------------------------------------------------------------
PP_HD int64_t q_el(int b, int Sq, int p, int Hq, int hq) {
    return ((int64_t) b * Sq + p) * Hq * D + (int64_t) hq * D;
}
PP_HD int64_t kv_el(int b, int Skv, int kv, int Hkv, int hkv) {
    return ((int64_t) b * Skv + kv) * Hkv * D + (int64_t) hkv * D;
}
PP_HD int64_t lse_el(int b, int Hq, int hq, int Sq, int p) {
    return ((int64_t) b * Hq + hq) * Sq + p;
}

// ---- TDM (2-D descriptors; TRUE extents: tensor_dim1 = rows below the origin clamped to the tile)
// ------------------ group-1 word 0 (prims layout.h tdm_w0): data_size 2 B, pad_enable,
// pad_interval = the row, pad_amount = pitch - row
PP_HD uint32_t tdm_w0(int row_b, int pitch_b) {
    int lg = 0;
    for (int x = row_b / 4; x > 1; x /= 2)
        ++lg;
    return (1u << 16) | (1u << 20) | ((uint32_t) (lg - 1) << 22) |
           ((uint32_t) ((pitch_b - row_b) / 4 - 1) << 25);
}
constexpr uint32_t K_W0    = tdm_w0(2 * D, K_PITCH);       // 0x07510000
constexpr uint32_t V_W0    = tdm_w0(2 * D, V_PITCH);       // 0x0f510000
constexpr uint32_t Q_W0    = tdm_w0(2 * NHW * D, Q_PITCH); // 512-B row, 16-B pad
constexpr uint32_t G0_TYPE = 0x80000000u;                  // group-0 word 3 type field [31:30] = 2
struct G1 {
    uint32_t w[8];
};
PP_HD G1 g1_2d(uint32_t w0, uint32_t tdim0, uint32_t tdim1, uint32_t tile0, uint32_t tile1,
               uint64_t stride0) {
    return G1{{w0, (tdim0 & 0xFFFFu) << 16, (tdim0 >> 16) | ((tdim1 & 0xFFFFu) << 16),
               (tdim1 >> 16) | (tile0 << 16), tile1, (uint32_t) stride0,
               (uint32_t) (stride0 >> 32) & 0xFFFFu, 0u}};
}
// Q op of wave w: [16 positions][256 = 2 heads x 128], row stride Hq * 128, LDS pitch Q_PITCH
PP_HD int q_op_dim1(int Sq, int qb, int w) {
    return imin(imax(Sq - wave_pos0(qb, w), 0), QW);
}
PP_HD G1 q_g1(int Hq, int dim1) {
    return g1_2d(Q_W0, NHW * D, (uint32_t) dim1, NHW * D, QW, (uint64_t) Hq * D);
}
PP_HD int64_t q_op_el(int b, int Sq, int Hq, int hq4, int qb, int w) {
    return q_el(b, Sq, wave_pos0(qb, w), Hq, wave_head0(hq4, w));
}
// K / V op k of lo wave s (role: s < 2 K, else V; half s & 1): rows [64 k + 32 half, +32); dummy (k
// >= n): 8 x 1 at row 0 (always inside) into the wave's own cell
PP_HD int op_role_v(int s) {
    return s >> 1;
}
PP_HD int op_half(int s) {
    return s & 1;
}
PP_HD int issuer_of(int role_v, int half) {
    return 2 * role_v + half;
} // lo wave index
PP_HD bool op_real(int k, int n) {
    return k < n;
}
PP_HD int n_ops(int n) {
    return n + 2;
} // per lo wave: ops 0 .. n + 1 (>= n: dummies)
PP_HD int kv_op_row0(int s, int k) {
    return ST * k + OPR * op_half(s);
}
PP_HD int kv_op_dim1(int Skv, int s, int k) {
    return imin(imax(Skv - kv_op_row0(s, k), 0), OPR);
}
PP_HD G1 kv_g1(int role_v, int Hkv, int dim1) {
    return g1_2d(role_v ? V_W0 : K_W0, D, (uint32_t) dim1, D, OPR, (uint64_t) Hkv * D);
}
PP_HD G1 dummy_g1(int role_v) {
    return g1_2d(role_v ? V_W0 : K_W0, 8, 1, 8, 1, 0);
}
PP_HD int64_t kv_op_el(int b, int Skv, int Hkv, int hkv, int s, int k, int n) {
    return kv_el(b, Skv, op_real(k, n) ? kv_op_row0(s, k) : 0, Hkv, hkv);
}
PP_HD int kv_op_lds(int s, int k, int n) {
    return !op_real(k, n) ? dummy_lds(s, k)
           : op_role_v(s) ? v_half_lds(k, op_half(s))
                          : k_half_lds(k, op_half(s));
}

// branch-free form of one lo-wave op (what the kernel issues): integer mask selects only, so every
// word stays SALU (a bool select of the descriptor became VALU + v_readfirstlane in the first
// build). lds = LDS byte offset, el = element offset from the role tensor base, g1 = group 1. ==
// kv_op_lds / kv_op_el / kv_g1 | dummy_g1.
PP_HD uint32_t umask(bool c) {
    return 0u - (uint32_t) c;
}
PP_HD uint32_t usel(uint32_t m, uint32_t a, uint32_t b) {
    return (a & m) | (b & ~m);
}
struct KvDesc {
    uint32_t lds;
    int64_t  el;
    uint32_t g1[8];
};
// dim1 = kv_op_dim1(Skv, s, k), passed in (the kernel makes it an SGPR first: LLVM turns the clamp
// into v_med3_i32)
PP_HD KvDesc kv_op_desc(int b, int Skv, int Hkv, int hkv, int s, int k, int n, int dim1) {
    const uint32_t m  = umask(op_real(k, n));
    const int      rv = op_role_v(s), hh = op_half(s);
    const uint32_t lr = (uint32_t) (rv ? v_half_lds(k, hh) : k_half_lds(k, hh));
    KvDesc         d{};
    d.lds = usel(m, lr, (uint32_t) dummy_lds(s, k));
    d.el  = kv_el(b, Skv, 0, Hkv, hkv) +
           (int64_t) (int) usel(m, (uint32_t) kv_op_row0(s, k), 0u) * Hkv * D;
    const G1 gr = kv_g1(rv, Hkv, dim1), gd = dummy_g1(rv);
    for (int i = 0; i < 8; ++i)
        d.g1[i] = usel(m, gr.w[i], gd.w[i]);
    return d;
}

// ---- global stores (byte offsets; host asserts 2 B Sq Hq D < 2^31 and 4 B Hq Sq < 2^31)
// ----------------------------- O: read-back p of wave w -> uniform part (soffset) + lane part 16 l
// (voffset)
PP_HD int64_t o_store_soff(int b, int Sq, int Hq, int hq4, int qb, int w, int p) {
    return 2 * q_el(b, Sq, wave_pos0(qb, w) + p, Hq, wave_head0(hq4, w));
}
PP_HD int o_store_voff(int l) {
    return 16 * l;
}
// LSE: lane l stores q-tile hf = l / 16, position r: uniform part + lane part 4 (hf Sq + r)
PP_HD int64_t lse_store_soff(int b, int Hq, int Sq, int hq4, int qb, int w) {
    return 4 * lse_el(b, Hq, wave_head0(hq4, w), Sq, wave_pos0(qb, w));
}
PP_HD int64_t lse_store_voff(int Sq, int l) {
    return 4 * ((int64_t) (l / 16) * Sq + l % 16);
}

// ---- cluster schedule (per wave; every function below is what the kernel executes at cluster j)
// ------------------- clusters j = 0 .. 4n + 1: X for even j, Y for odd j, chunk c = j / 2 - 1:
//   X(c): QK(c+1) -> S[(c+1) & 1] (c+1 < 2n)  ||  exp(c) in place on S[c & 1] (c >= 0)
//   Y(c): PV(c) (c >= 0) || cvt P(c), row sums(c), SM1(c+1) (c+1 < 2n), rescale C(c+1) at the end
// burst of cluster j, issued in the barrier window (after the signal, before the wait) that
// precedes j:
//   X(c): V(c)    Y(c): K(c+2) (c+2 < 2n)    X(-1): K(0), after the wait of the last prologue
//   barrier
PP_HD int n_clusters(int n) {
    return 4 * n + 2;
}
PP_HD bool cl_is_x(int j) {
    return (j & 1) == 0;
}
PP_HD int cl_chunk(int j) {
    return j / 2 - 1;
}
PP_HD int qk_chunk(int j, int n) {
    return cl_is_x(j) && cl_chunk(j) + 1 < 2 * n ? cl_chunk(j) + 1 : -1;
}
PP_HD int exp_chunk(int j) {
    return cl_is_x(j) && j >= 2 ? cl_chunk(j) : -1;
}
PP_HD int pv_chunk(int j) {
    return !cl_is_x(j) && j >= 3 ? cl_chunk(j) : -1;
}
PP_HD int sm1_chunk(int j, int n) {
    return !cl_is_x(j) && cl_chunk(j) + 1 < 2 * n ? cl_chunk(j) + 1 : -1;
}
PP_HD int burst_k_chunk(int j, int n) {
    return j == 0 ? 0 : (!cl_is_x(j) && cl_chunk(j) + 2 < 2 * n ? cl_chunk(j) + 2 : -1);
}
PP_HD int burst_v_chunk(int j) {
    return cl_is_x(j) && j >= 2 ? cl_chunk(j) : -1;
}
// barrier B_b preceding cluster j of group g (j >= 1); the prologue runs B_0 .. B_{lag} (full
// barriers)
PP_HD int bar_before(int g, int j) {
    return j + lag_of(g);
}
PP_HD int n_barriers(int n) {
    return n_clusters(n) + 1;
} // B_0 .. B_J per wave
PP_HD int n_exit_barriers(int g) {
    return 1 - lag_of(g);
} // lo: 1 (B_J); hi: 1 - LAG
// lo TDM: op issued right after the wait that opens cluster j (-1: none); X(2u), u <= n - 2 -> op u
// + 3. The prologue issues ops 0, 1 (before B_0) and 2 (after B_0).
PP_HD int op_issue(int j, int n) {
    return cl_is_x(j) && j >= 2 && (j - 2) % 4 == 0 && (j - 2) / 4 <= n - 2 ? (j - 2) / 4 + 3 : -1;
}
// lo TDM: s_wait_tensorcnt at the end of cluster j (before its next signal; -1: none): Y(2u+1),
// 2u+1 <= 2n - 3, and Y(-1) -> 1 (retires op u + 2, i.e. publishes stage u + 2 at B_{4u+6})
PP_HD int tcnt_end(int j, int n) {
    return !cl_is_x(j) && (j - 1) % 4 == 0 && cl_chunk(j) <= 2 * n - 3 ? 1 : -1;
}
constexpr int TCNT_PRO = 1; // lo prologue: Q, op 0 retired, op 1 in flight
// stage bodies: stage t = clusters 4t + 2 .. 4t + 5; [0, main_stages) unmasked, n - 2 (if n >= 2)
// masked at its last Y, the last stage n - 1 (no op issue, no tensorcnt, no K bursts, no QK(2n) /
// SM1(2n))
PP_HD int main_stages(int n) {
    return imax(n - 2, 0);
}
PP_HD bool has_pen_stage(int n) {
    return n >= 2;
}

// ---- stage plan: THE kernel's program (body.h expands it at compile time; kernel_events below on
// the host) ------- The cluster-indexed functions above are the specification; the plan is the
// stage-relative form the kernel is generated from. selftest: kernel_events == wave_events (n
// = 1..140, every wave) and every plan field == the spec function of its cluster; fwd_pp_check
// replays kernel_events; test_ext_fwdpp checks the ISA's event order == kernel_events(w, 3)
// (prologue, one loop body, penultimate, last, exit) for both groups. Stage t = clusters j = 4t + 2
// + i, i = 0..3 (X, Y, X, Y), chunk base c0 = 2t; the prologue is stage -1, clusters i = 2, 3
// (X(-1), Y(-1)). Chunk fields are offsets from c0, -1 = none.
enum StageMode : int { SM_PRO = -1, SM_MAIN = 0, SM_PEN = 1, SM_LAST = 2 };
struct ClPlan {
    int win; // the burst sits in the barrier window before the cluster (0: after the last prologue
             // wait)
    int kc, vc;  // K / V burst chunk
    int op;      // lo: TDM op t + op issued right after the opening wait
    int tcnt;    // lo: s_wait_tensorcnt at the end of the cluster
    int qk, ex;  // X: QK chunk (into S[chunk & 1]), exp chunk (in place)
    int pv, sm1; // Y: PV / cvt / row-sum chunk, SM1 chunk
    int masked;  // Y: SM1 applies mask_off (the sites where the SM1 chunk can be in the last stage)
    int seed;    // Y: SM1 seeds m2 (chunk 0: no vote, no rescale)
};
PP_HD int cl_first(int M) {
    return M == SM_PRO ? 2 : 0;
}
PP_HD ClPlan cl_plan(int M, int i) {
    const bool last = M == SM_LAST, pro = M == SM_PRO;
    ClPlan     p{1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0};
    if (i == 0) { // X(c0): V(c0) | op t + 3 | exp(c0) || QK(c0 + 1)
        p.vc = 0;
        p.op = last ? -1 : 3;
        p.ex = 0;
        p.qk = 1;
    } else if (i == 1) { // Y(c0): K(c0 + 2) | PV(c0) || SM1(c0 + 1)
        p.kc     = last ? -1 : 2;
        p.pv     = 0;
        p.sm1    = 1;
        p.masked = last;
    } else if (i == 2) { // X(c0 + 1): V(c0 + 1) | exp(c0 + 1) || QK(c0 + 2);  prologue X(-1): K(0)
                         // after B_lag | QK(0)
        p.win = !pro;
        p.kc  = pro ? 2 : -1;
        p.vc  = pro ? -1 : 1;
        p.ex  = pro ? -1 : 1;
        p.qk  = last ? -1 : 2;
    } else { // Y(c0 + 1): K(c0 + 3) | PV(c0 + 1) || SM1(c0 + 2) | tensorcnt 1;  prologue Y(-1):
             // K(1) | SM1(0)
        p.kc     = last ? -1 : 3;
        p.pv     = pro ? -1 : 1;
        p.sm1    = last ? -1 : 2;
        p.masked = M == SM_PEN || pro;
        p.seed   = pro;
        p.tcnt   = last ? -1 : 1;
    }
    return p;
}
// the stage mode of stage t (-1 = prologue) of a WG with n stages: body.h runs [0, main_stages)
// MAIN, n - 2 PEN
PP_HD int stage_mode(int t, int n) {
    return t < 0 ? SM_PRO : t < main_stages(n) ? SM_MAIN : t == n - 1 ? SM_LAST : SM_PEN;
}
// prologue / exit TDM program of a lo wave (around the plan's clusters): ops [0, PRO_OPS) before
// B_0, then s_wait_tensorcnt TCNT_PRO; op PRO_OP_B0 after B_0; s_wait_tensorcnt EXIT_TCNT before
// the exit barrier. hi: Q op, s_wait_tensorcnt 0.
constexpr int PRO_OPS = 2, PRO_OP_B0 = 2, EXIT_TCNT = 0;
// what body.h precomputes one cluster ahead must not depend on the next stage's mode (MAIN, PEN or
// LAST)
static_assert(cl_plan(SM_MAIN, 0).vc == 0 && cl_plan(SM_PEN, 0).vc == 0 &&
                  cl_plan(SM_LAST, 0).vc == 0 && cl_plan(SM_MAIN, 0).kc < 0 &&
                  cl_plan(SM_PEN, 0).kc < 0 && cl_plan(SM_LAST, 0).kc < 0,
              "next stage's first burst: V(c0) for every mode");
static_assert(cl_plan(SM_MAIN, 0).op == cl_plan(SM_PEN, 0).op &&
                  cl_plan(SM_MAIN, 0).op == PRO_OP_B0 + 1,
              "the carried descriptor is op t + 3 for MAIN and PEN");

// ---- constants for the host proof (ctypes)
// -----------------------------------------------------------------------------
PP_HD long long kconst(int i) {
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
           : i == 10 ? Q_PITCH
           : i == 11 ? LDS_BYTES
           : i == 12 ? (long long) K_W0
           : i == 13 ? (long long) V_W0
           : i == 14 ? (long long) Q_W0
           : i == 15 ? LAG
           : i == 16 ? BLOCK
                     : -1;
}

} // namespace pp

// ================================== host-only: the wave program as an event list
// ==================================
#if !defined(__HIP_DEVICE_COMPILE__) && (defined(PP_IDX_SELFTEST) || defined(PP_IDX_EVENTS))
#include <string>
#include <vector>
namespace pp {
// Event list of wave w for a WG with n stages, in program order. E_TDM a: op a (-1 = own Q op);
// E_TCNT a: s_wait_tensorcnt a; E_SIG / E_WAIT a: barrier B_a; E_RD_Q: the Q burst (+ c2 cell);
// E_RD_K / E_RD_V a: burst of chunk a; E_EPI: O image write + read-back + global stores; E_CL a:
// start of cluster a (bookkeeping only).
//   kernel_events  the kernel's program: the stage plan expanded exactly as body.h expands it
//   (replayed by
//                  fwd_pp_check and check_protocol; its token form is matched against the ISA by
//                  test_ext_fwdpp)
//   wave_events    the specification from the cluster-indexed functions, with mutations for
//   negative controls;
//                  selftest: kernel_events == wave_events(MU_NONE) for n = 1..140 and every wave
enum Ev : int { E_TDM, E_TCNT, E_RD_Q, E_RD_K, E_RD_V, E_SIG, E_WAIT, E_EPI, E_CL };
struct Event {
    int type, a;
};
inline bool operator==(const Event &x, const Event &y) {
    return x.type == y.type && x.a == y.a;
}
inline std::vector<Event> kernel_events(int w, int n) {
    const int          g  = grp_of(w);
    const bool         lo = g == 0;
    std::vector<Event> e;
    e.push_back({E_TDM, -1}); // own Q op
    if (lo) {
        for (int k = 0; k < PRO_OPS; ++k)
            e.push_back({E_TDM, k});
        e.push_back({E_TCNT, TCNT_PRO});
    } else {
        e.push_back({E_TCNT, 0});
    }
    e.push_back({E_RD_Q, 0});
    int b = 0;
    e.push_back({E_SIG, b});
    e.push_back({E_WAIT, b++}); // B_0
    if (lo)
        e.push_back({E_TDM, PRO_OP_B0});
    for (int k = 0; k < lag_of(g); ++k) {
        e.push_back({E_SIG, b});
        e.push_back({E_WAIT, b++});
    }                              // hi: the stagger
    for (int t = -1; t < n; ++t) { // body.h: stage(-1, PRO); MAIN loop; PEN (n >= 2); LAST
        const int M = stage_mode(t, n), c0 = 2 * t;
        for (int i = cl_first(M); i < 4; ++i) {
            const ClPlan p = cl_plan(M, i);
            if (p.win)
                e.push_back({E_SIG, b});
            if (p.kc >= 0)
                e.push_back({E_RD_K, c0 + p.kc});
            if (p.vc >= 0)
                e.push_back({E_RD_V, c0 + p.vc});
            if (p.win)
                e.push_back({E_WAIT, b++});
            e.push_back({E_CL, 4 * t + 2 + i});
            if (lo && p.op >= 0)
                e.push_back({E_TDM, t + p.op});
            if (lo && p.tcnt >= 0)
                e.push_back({E_TCNT, p.tcnt});
        }
    }
    if (lo)
        e.push_back({E_TCNT, EXIT_TCNT});
    for (int k = 0; k < n_exit_barriers(g); ++k) {
        e.push_back({E_SIG, b});
        if (k == n_exit_barriers(g) - 1)
            e.push_back({E_EPI, 0});
        e.push_back({E_WAIT, b++});
    }
    if (n_exit_barriers(g) == 0)
        e.push_back({E_EPI, 0});
    return e;
}
// the ISA token form of an event list (test_ext_fwdpp matches it against the disassembly): T TDM
// issue, C<a> tensorcnt, S / W barrier signal / wait, q Q burst, k / v K / V burst, e epilogue;
// E_CL has none
inline std::string event_tokens(const std::vector<Event> &ev) {
    std::string s;
    for (const Event &x : ev) {
        const char *t = x.type == E_TDM    ? "T"
                        : x.type == E_TCNT ? (x.a == 0   ? "C0"
                                              : x.a == 1 ? "C1"
                                              : x.a == 2 ? "C2"
                                                         : "C3")
                        : x.type == E_SIG  ? "S"
                        : x.type == E_WAIT ? "W"
                        : x.type == E_RD_Q ? "q"
                        : x.type == E_RD_K ? "k"
                        : x.type == E_RD_V ? "v"
                        : x.type == E_EPI  ? "e"
                                           : nullptr;
        if (t) {
            if (!s.empty())
                s += ' ';
            s += t;
        }
    }
    return s;
}
// mutations for negative controls (0 = the kernel)
enum Mut : int {
    MU_NONE = 0,
    MU_TCNT2,
    MU_ISSUE_EARLY,
    MU_NO_LAG_BAR,
    MU_PUBLISH_LATE,
    MU_NO_EXIT_TCNT,
    MU_BURST_AFTER
};
inline std::vector<Event> wave_events(int w, int n, int mut = MU_NONE) {
    const int          g = grp_of(w), x = lag_of(g), J = n_clusters(n);
    const bool         lo = g == 0;
    std::vector<Event> e;
    if (lo) {
        e.push_back({E_TDM, -1});
        e.push_back({E_TDM, 0});
        e.push_back({E_TDM, 1});
        e.push_back({E_TCNT, TCNT_PRO});
    } else {
        e.push_back({E_TDM, -1});
        e.push_back({E_TCNT, 0});
    }
    e.push_back({E_RD_Q, 0});
    e.push_back({E_SIG, 0});
    e.push_back({E_WAIT, 0});
    if (lo)
        e.push_back({E_TDM, 2});
    for (int b = 1; b <= x; ++b)
        if (!(mut == MU_NO_LAG_BAR && w == 7)) {
            e.push_back({E_SIG, b});
            e.push_back({E_WAIT, b});
        }
    e.push_back({E_RD_K, 0});
    e.push_back({E_CL, 0});
    for (int j = 1; j < J; ++j) {
        if (lo) {
            int tc = tcnt_end(j - 1, n);
            if (mut == MU_PUBLISH_LATE)
                tc = tcnt_end(j - 3, n); // retire one Y later than the publishing barrier
            if (tc >= 0)
                e.push_back({E_TCNT, mut == MU_TCNT2 ? 2 : tc});
        }
        const int b  = bar_before(g, j);
        const int kc = burst_k_chunk(j, n), vc = burst_v_chunk(j);
        e.push_back({E_SIG, b});
        if (mut != MU_BURST_AFTER) {
            if (kc >= 0)
                e.push_back({E_RD_K, kc});
            if (vc >= 0)
                e.push_back({E_RD_V, vc});
        }
        e.push_back({E_WAIT, b});
        if (mut == MU_BURST_AFTER) { // bursts after the wait: legal (a stricter order), used as a
                                     // positive variant
            if (kc >= 0)
                e.push_back({E_RD_K, kc});
            if (vc >= 0)
                e.push_back({E_RD_V, vc});
        }
        e.push_back({E_CL, j});
        if (lo) {
            int op = op_issue(j, n);
            if (mut == MU_ISSUE_EARLY)
                op = op_issue(j + 1, n); // one cluster earlier (end of Y(2u-1) window)
            if (op >= 0)
                e.push_back({E_TDM, op});
        }
    }
    if (lo && mut != MU_NO_EXIT_TCNT)
        e.push_back({E_TCNT, 0});
    for (int k = 0; k < n_exit_barriers(g); ++k) {
        const int b = J + x + k;
        e.push_back({E_SIG, b});
        if (k == n_exit_barriers(g) - 1)
            e.push_back({E_EPI, 0});
        e.push_back({E_WAIT, b});
    }
    if (n_exit_barriers(g) == 0)
        e.push_back({E_EPI, 0});
    return e;
}
} // namespace pp
#endif

#ifdef PP_IDX_SELFTEST
// ======================================= reference replay (host only)
// =============================================
#include <cmath>
#include <cstdio>

namespace pp {

#define PP_CHECK(cond, ...)                                                                        \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                       \
            std::printf(__VA_ARGS__);                                                              \
            std::printf("\n");                                                                     \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

// lin16 bank model (mb_tr16pitch): lanes 0-15 / 16-31 are served in separate passes; the 16
// accesses of 16 B of a pass must cover 64 distinct 4-B banks
inline int bank_ok(const int *addr32) {
    for (int g = 0; g < 2; ++g) {
        bool used[64] = {};
        for (int l = 16 * g; l < 16 * g + 16; ++l)
            for (int j = 0; j < 4; ++j) {
                const int bank = (addr32[l] / 4 + j) % 64;
                if (used[bank])
                    return 0;
                used[bank] = true;
            }
    }
    return 1;
}

inline int check_lds() {
    struct R {
        int         lo, hi;
        const char *n;
    };
    std::vector<R> rs;
    for (int h = 0; h < 2; ++h) {
        rs.push_back({k_half_lds(0, h), k_half_lds(0, h) + KD * K_HALF_B, h ? "K h1" : "K h0"});
        rs.push_back({v_half_lds(0, h), v_half_lds(0, h) + KD * V_HALF_B, h ? "V h1" : "V h0"});
    }
    for (int w = 0; w < NWAVE; ++w)
        rs.push_back({qo_lds(w), qo_lds(w) + QO_W_B, "QO"});
    for (int s = 0; s < NPAIR; ++s)
        rs.push_back({dummy_lds(s, 0), dummy_lds(s, 0) + 32, "dummy"});
    for (int w = 0; w < NWAVE; ++w)
        rs.push_back({c2_lds(w), c2_lds(w) + 16, "c2"});
    for (auto &a : rs) {
        PP_CHECK(a.lo >= 0 && a.hi <= LDS_BYTES && a.lo % 8 == 0, "region %s", a.n);
        PP_CHECK(seg_of(a.lo) == seg_of(a.hi - 1), "region %s crosses a segment", a.n);
        for (auto &b : rs)
            PP_CHECK(&a == &b || a.hi <= b.lo || b.hi <= a.lo, "overlap %s %s", a.n, b.n);
    }
    // segment plan: K h0 / V h0 / K h1 / V h1 in segments 0 / 1 / 2 / 3; the concurrent bursts of
    // the two groups (window before lo X(c): lo V(c) || hi K(c+1); before lo Y(c): lo K(c+2) || hi
    // V(c)) in different segments of different parity
    for (int c = 0; c < 64; ++c) {
        const int a = seg_of(v_chunk_lds(c)), b = seg_of(k_chunk_lds(c + 1));
        const int x = seg_of(k_chunk_lds(c + 2)), y = seg_of(v_chunk_lds(c));
        PP_CHECK(a != b && (a & 1) != (b & 1) && x != y && (x & 1) != (y & 1), "segment plan c %d",
                 c);
    }
    // TDM w0 encodings == the pitches
    const uint32_t w0s[3]  = {K_W0, V_W0, Q_W0};
    const int      rowb[3] = {2 * D, 2 * D, 2 * NHW * D}, pitch[3] = {K_PITCH, V_PITCH, Q_PITCH};
    for (int j = 0; j < 3; ++j) {
        const int ivl_b = 4 << (int) (((w0s[j] >> 22) & 7u) + 1),
                  pad_b = 4 * (int) (((w0s[j] >> 25) & 127u) + 1);
        PP_CHECK(ivl_b == rowb[j] && pad_b == pitch[j] - rowb[j], "tdm w0 %d", j);
    }
    PP_CHECK(K_W0 == 0x07510000u && V_W0 == 0x0f510000u, "K / V w0 (prims constants)");
    int ad[32];
    // K A operands: lane reads == landing of the issuing lo wave's op, all lanes conflict-free
    for (int c = 0; c < 8; ++c)
        for (int kt = 0; kt < 2; ++kt)
            for (int k = 0; k < NDK; ++k)
                for (int e = 0; e < 2; ++e) {
                    for (int l = 0; l < 32; ++l) {
                        const int row = 16 * kt + l % 16,
                                  col = 32 * k + 16 * e + 8 * (l / 16); // row within the chunk
                        ad[l]         = k_chunk_lds(c) + k_lane(l) + k_imm(kt, k, e);
                        const int s   = issuer_of(0, c & 1);
                        PP_CHECK(ad[l] == kv_op_lds(s, c >> 1, 1 << 20) + row * K_PITCH + 2 * col,
                                 "K landing");
                        PP_CHECK(k_imm(kt, k, e) < 65536 && ad[l] % 16 == 0, "K imm");
                    }
                    PP_CHECK(bank_ok(ad), "K b128 banks");
                }
    for (int c = 0; c < 8; ++c)
        for (int e = 0; e < 2; ++e)
            for (int dt = 0; dt < NDT; ++dt) {
                for (int l = 0; l < 32; ++l) {
                    const int row = 16 * e + 8 * (l / 16) + l % 8,
                              col = 16 * dt + 8 * ((l / 8) % 2);
                    ad[l]         = v_chunk_lds(c) + v_lane(l) + v_imm(e, dt);
                    const int s   = issuer_of(1, c & 1);
                    PP_CHECK(ad[l] == kv_op_lds(s, c >> 1, 1 << 20) + row * V_PITCH + 2 * col,
                             "V landing");
                    PP_CHECK(v_imm(e, dt) < 65536 && ad[l] % 16 == 0, "V imm");
                }
                PP_CHECK(bank_ok(ad), "V tr16 banks");
            }
    for (int w = 0; w < NWAVE; ++w) {
        std::vector<int> wr(QO_W_B / 16, 0), rd(QO_W_B / 16, 0);
        for (int h = 0; h < NHW; ++h) {
            for (int k = 0; k < NDK; ++k)
                for (int e = 0; e < 2; ++e) {
                    for (int l = 0; l < 32;
                         ++l) { // Q op lands element (p, col) at qo_lds + p * Q_PITCH + 2 col
                        ad[l] = q_lane(w, l) + q_imm(h, k, e);
                        PP_CHECK(ad[l] == qo_lds(w) + (l % 16) * Q_PITCH +
                                              2 * (D * h + 32 * k + 16 * e + 8 * (l / 16)),
                                 "Q read");
                    }
                    PP_CHECK(bank_ok(ad), "Q^T b128 banks");
                }
            for (int dt = 0; dt < NDT; ++dt) {
                for (int l = 0; l < 32; ++l) {
                    ad[l] = q_lane(w, l) + ow_imm(h, dt);
                    PP_CHECK(ad[l] == qo_lds(w) + (l % 16) * Q_PITCH +
                                          2 * (D * h + 16 * dt + 8 * (l / 16)),
                             "O write");
                    PP_CHECK(ad[l] >= qo_lds(w) && ad[l] + 16 <= qo_lds(w) + QO_W_B,
                             "O write region");
                    wr[(ad[l] - qo_lds(w)) / 16]++;
                }
                PP_CHECK(bank_ok(ad), "O image write banks");
            }
        }
        for (int p = 0; p < QW; ++p) {
            for (int l = 0; l < 32; ++l) {
                ad[l] = or_lane(w, l) + or_imm(p);
                PP_CHECK(ad[l] == qo_lds(w) + p * Q_PITCH + 16 * l && or_imm(p) < 65536, "O read");
                rd[(ad[l] - qo_lds(w)) / 16]++;
            }
            PP_CHECK(bank_ok(ad), "O read-back banks");
        }
        for (size_t i = 0; i < wr.size(); ++i)
            PP_CHECK(wr[i] == rd[i] && wr[i] <= 1, "O image coverage");
    }
    return 0;
}

// causal mask == kv <= p + Skv - Sq (relative: kv <= 64 (n - 1) + 16 s + r) for every pair / chunk
// / element
inline int check_mask() {
    for (int n = 1; n < 9; ++n)
        for (int causal = 0; causal < 2; ++causal)
            for (int s = 0; s < NPAIR; ++s)
                for (int c = 0; c < 2 * n; ++c)
                    for (int kt = 0; kt < 2; ++kt)
                        for (int l = 0; l < 32; ++l)
                            for (int i = 0; i < 8; ++i) {
                                const int  kv = CH * c + 16 * kt + 8 * (l / 16) + i;
                                const bool def =
                                    causal == 0 || kv <= ST * (n - 1) + QW * s + l % 16;
                                const bool code = !(last_stage_chunk(c, n) &&
                                                    masked(l, i, mask_off(s, c, n, kt, causal)));
                                PP_CHECK(def == code,
                                         "mask n %d causal %d s %d c %d kt %d l %d i %d", n, causal,
                                         s, c, kt, l, i);
                                if (c == 0 && kt == 0 && l < 16 && i == 0)
                                    PP_CHECK(def, "chunk 0 kv 0 visible");
                            }
    // the kernel's mask sites: SM1(c) is masked iff last_stage_chunk(c, n); SM1(c) runs in Y(c - 1)
    // (Y(-1) for c 0)
    for (int n = 1; n < 9; ++n)
        for (int j = 0; j < n_clusters(n); ++j) {
            const int c = sm1_chunk(j, n);
            if (c < 0)
                continue;
            const bool pen = j == 4 * (n - 2) + 5, last = j == 4 * (n - 1) + 3,
                       pro = j == 1 && n == 1;
            PP_CHECK(last_stage_chunk(c, n) == (pen || last || pro), "mask site n %d j %d", n, j);
        }
    return 0;
}

// fp32 emulation of one wave (2 q-tiles x 16 queries of pair s) through the scheduled softmax:
// SM1(c) (lane max, cross-half max, one ballot over both q-tiles) and the rescale C(c) before
// exp(c); m2 = mx2 at chunk 0; per-lane partial row sums; masks of the last stage as -inf. vs a
// double reference.
inline int check_softmax(int n, int causal, int s, uint32_t seed) {
    const float        c2  = 0.08838834764831845f * LOG2E;
    const int          nkv = ST * n;
    std::vector<float> S((size_t) nkv * QW * NHW), V((size_t) nkv);
    auto               rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float) (seed >> 8) / 16777216.0f;
    };
    for (int kv = 0; kv < nkv; ++kv) {
        V[kv] = 2.0f * rnd() - 1.0f;
        for (int q = 0; q < QW * NHW; ++q) // rising max (forces rescales); q-tile 1 rises faster
            S[(size_t) kv * QW * NHW + q] =
                80.0f * rnd() - 40.0f + (q >= QW ? 900.0f : 300.0f) * kv / nkv;
    }
    auto  sv  = [&](int kv, int h, int r) { return S[(size_t) kv * QW * NHW + QW * h + r]; };
    auto  vis = [&](int kv, int r) { return causal == 0 || kv <= ST * (n - 1) + QW * s + r; };
    float m2[NHW][32], ll[NHW][32], O[NHW][32];
    for (int h = 0; h < NHW; ++h)
        for (int l = 0; l < 32; ++l) {
            m2[h][l] = -1e30f;
            ll[h][l] = 0.0f;
            O[h][l]  = 0.0f;
        }
    int rescales = 0;
    for (int c = 0; c < 2 * n; ++c) {
        float x[NHW][32][16], mx2[NHW][32];
        bool  any = false;
        for (int h = 0; h < NHW; ++h) {
            float mx[32];
            for (int l = 0; l < 32; ++l) {
                mx[l] = -INFINITY;
                for (int kt = 0; kt < 2; ++kt)
                    for (int i = 0; i < 8; ++i) {
                        const int  kv = CH * c + 16 * kt + 8 * (l / 16) + i;
                        const bool m =
                            last_stage_chunk(c, n) && masked(l, i, mask_off(s, c, n, kt, causal));
                        x[h][l][8 * kt + i] = m ? -INFINITY : sv(kv, h, l % 16);
                        mx[l]               = std::fmax(mx[l], x[h][l][8 * kt + i]);
                    }
            }
            for (int l = 0; l < 32; ++l) {
                mx2[h][l] = std::fmax(mx[l], mx[l ^ 16]) * c2;
                any       = any || (mx2[h][l] - m2[h][l] > TH2);
            }
        }
        if (c == 0) {
            for (int h = 0; h < NHW; ++h)
                for (int l = 0; l < 32; ++l) {
                    PP_CHECK(std::isfinite(mx2[h][l]), "chunk 0 max");
                    m2[h][l] = mx2[h][l];
                }
        } else if (any) {
            ++rescales;
            for (int h = 0; h < NHW; ++h)
                for (int l = 0; l < 32; ++l) {
                    const float m2n  = std::fmax(m2[h][l], mx2[h][l]),
                                corr = std::exp2(m2[h][l] - m2n);
                    O[h][l] *= corr;
                    ll[h][l] *= corr;
                    m2[h][l] = m2n;
                }
        }
        for (int h = 0; h < NHW; ++h)
            for (int l = 0; l < 32; ++l) {
                float rs = 0.0f, pv = 0.0f;
                for (int e = 0; e < 16; ++e) {
                    const float p  = std::exp2(std::fma(x[h][l][e], c2, -m2[h][l]));
                    const int   kv = CH * c + 16 * (e / 8) + 8 * (l / 16) + e % 8;
                    rs += p;
                    pv += p * V[kv];
                }
                ll[h][l] += rs;
                O[h][l] += pv;
            }
    }
    PP_CHECK(n < 2 || rescales > 0, "emulation never rescaled");
    for (int h = 0; h < NHW; ++h)
        for (int r = 0; r < QW; ++r) {
            PP_CHECK(m2[h][r] == m2[h][r + 16], "m2 identical in l and l ^ 16");
            const float lt = ll[h][r] + ll[h][r + 16], o = (O[h][r] + O[h][r + 16]) / lt;
            const float lse  = (m2[h][r] + std::log2(lt)) * LN2;
            double      mref = -1e300, sum = 0.0, acc = 0.0;
            for (int kv = 0; kv < nkv; ++kv)
                if (vis(kv, r))
                    mref = std::fmax(mref, (double) sv(kv, h, r));
            for (int kv = 0; kv < nkv; ++kv)
                if (vis(kv, r)) {
                    const double ex =
                        std::exp(0.08838834764831845 * ((double) sv(kv, h, r) - mref));
                    sum += ex;
                    acc += ex * V[kv];
                }
            const double oref = acc / sum, lref = 0.08838834764831845 * mref + std::log(sum);
            PP_CHECK(std::isfinite(o) && std::isfinite(lse) && lt > 0.0f,
                     "softmax finite n %d s %d r %d", n, s, r);
            PP_CHECK(std::fabs(o - oref) <= 1e-4 * (std::fabs(oref) + 1.0) &&
                         std::fabs(lse - lref) <= 1e-4 * (std::fabs(lref) + 1.0),
                     "softmax n %d causal %d s %d h %d r %d: o %g / %g lse %g / %g", n, causal, s,
                     h, r, o, oref, lse, lref);
        }
    return 0;
}

// ---- barrier / ring protocol replay (all 8 waves, happens-before over the workgroup barrier
// phases) --------------- returns 0 iff: identical barrier counts with strict SIGNAL / WAIT
// alternation; every wave reaches its exit (the deadlock-freedom induction); <= 3 TDM ops in flight
// per wave; exactly 1 + n_ops ops per lo wave, all retired at the exit; Q read after its own
// retire; RAW: every K / V chunk read after the issuer's retire of its op via a barrier phase; WAR:
// every real op k >= KD issued after every read (both groups) of stage k - KD via a phase.
inline int check_protocol(int n, int mut = MU_NONE) {
    std::vector<Event> ev[NWAVE];
    for (int w = 0; w < NWAVE; ++w)
        ev[w] = mut == MU_NONE ? kernel_events(w, n) : wave_events(w, n, mut);
    const int        NB = n_barriers(n);
    std::vector<int> sig[NWAVE], wt[NWAVE];
    for (int w = 0; w < NWAVE; ++w) {
        sig[w].assign(NB, -1);
        wt[w].assign(NB, -1);
        int  expect = 0;
        bool open   = false;
        for (int i = 0; i < (int) ev[w].size(); ++i) {
            const Event &x = ev[w][i];
            if (x.type == E_SIG) {
                PP_CHECK(!open && x.a == expect, "n %d w %d: signal %d out of order", n, w, x.a);
                sig[w][x.a] = i;
                open        = true;
            }
            if (x.type == E_WAIT) {
                PP_CHECK(open && x.a == expect, "n %d w %d: wait %d out of order", n, w, x.a);
                wt[w][x.a] = i;
                open       = false;
                ++expect;
            }
        }
        PP_CHECK(expect == NB && !open, "n %d w %d: %d barrier phases, %d expected", n, w, expect,
                 NB);
    }
    // execution: simulate the barrier semantics (a wait returns when every wave signalled that
    // phase); every wave must run to its end (deadlock freedom on this program)
    {
        std::vector<int> pc(NWAVE, 0);
        std::vector<int> nsig(NB, 0);
        bool             progress = true;
        while (progress) {
            progress = false;
            for (int w = 0; w < NWAVE; ++w)
                while (pc[w] < (int) ev[w].size()) {
                    const Event &x = ev[w][pc[w]];
                    if (x.type == E_WAIT && nsig[x.a] < NWAVE)
                        break;
                    if (x.type == E_SIG)
                        ++nsig[x.a];
                    ++pc[w];
                    progress = true;
                }
        }
        for (int w = 0; w < NWAVE; ++w)
            PP_CHECK(pc[w] == (int) ev[w].size(), "n %d: wave %d deadlocks at event %d", n, w,
                     pc[w]);
    }
    // happens-before (event i of wave a) -> (event j of wave b): the first phase a signals after i
    // is waited by b before j
    std::vector<int> nxt[NWAVE];
    for (int w = 0; w < NWAVE; ++w) {
        nxt[w].assign(ev[w].size(), NB);
        for (int i = (int) ev[w].size() - 1, x = NB; i >= 0; --i) {
            nxt[w][i] = x;
            if (ev[w][i].type == E_SIG)
                x = ev[w][i].a;
        }
    }
    auto hb = [&](int a, int i, int b, int j) {
        if (a == b)
            return i < j;
        const int x = nxt[a][i];
        return x < NB && wt[b][x] >= 0 && wt[b][x] < j;
    };
    // TDM: issue / retire points per wave (in-order completion; s_wait_tensorcnt N retires all but
    // the newest N)
    std::vector<int> ops[NWAVE], ret_at[NWAVE], iss_at[NWAVE];
    for (int a = 0; a < NWAVE; ++a) {
        int retired = 0, nops = 0;
        for (int i = 0; i < (int) ev[a].size(); ++i) {
            const Event &x = ev[a][i];
            if (x.type == E_TDM) {
                ops[a].push_back(x.a);
                ret_at[a].push_back(-1);
                iss_at[a].push_back(i);
                ++nops;
                PP_CHECK(nops - retired <= 3, "n %d w %d: > 3 TDM ops in flight", n, a);
            }
            if (x.type == E_TCNT)
                for (; retired < nops - x.a; ++retired)
                    ret_at[a][retired] = i;
            if (x.type == E_RD_Q)
                PP_CHECK(nops >= 1 && ret_at[a][0] >= 0, "n %d w %d: Q read before its op retired",
                         n, a);
            if (x.type == E_SIG || x.type == E_EPI) {
            }
        }
        PP_CHECK(retired == nops, "n %d w %d: %d TDM ops in flight at the exit", n, a,
                 nops - retired);
        PP_CHECK(nops == (grp_of(a) == 0 ? 1 + n_ops(n) : 1), "n %d w %d: %d ops", n, a, nops);
        for (int q = 1; q < (int) ops[a].size(); ++q)
            PP_CHECK(ops[a][q] == q - 1, "n %d w %d: op order", n, a);
    }
    // RAW and WAR per (role, half, stage)
    int nread = 0;
    for (int b = 0; b < NWAVE; ++b)
        for (int j = 0; j < (int) ev[b].size(); ++j) {
            const Event &x = ev[b][j];
            if (x.type != E_RD_K && x.type != E_RD_V)
                continue;
            ++nread;
            const int role = x.type == E_RD_V, stg = x.a >> 1, half = x.a & 1,
                      a = issuer_of(role, half);
            PP_CHECK(op_real(stg, n), "n %d: wave %d reads chunk %d of a dummy stage", n, b, x.a);
            int r = -1;
            for (size_t q = 0; q < ops[a].size(); ++q)
                if (ops[a][q] == stg)
                    r = ret_at[a][q];
            PP_CHECK(r >= 0 && hb(a, r, b, j),
                     "n %d: RAW chunk %d role %d read by wave %d not after op %d of wave %d", n,
                     x.a, role, b, stg, a);
            for (size_t q = 0; q < ops[a].size(); ++q) {
                const int k = ops[a][q];
                if (k >= KD && op_real(k, n) && k - KD == stg)
                    PP_CHECK(hb(b, j, a, iss_at[a][q]),
                             "n %d: WAR op %d of wave %d vs read of chunk %d by wave %d", n, k, a,
                             x.a, b);
            }
        }
    PP_CHECK(nread == NWAVE * 4 * n, "n %d: %d chunk reads, %d expected", n, nread, NWAVE * 4 * n);
    return 0;
}

// the cluster schedule: per wave every chunk is QK'd, exp'd, PV'd and SM1'd exactly once, in
// dependency order
inline int check_schedule(int n) {
    const int        J = n_clusters(n);
    std::vector<int> qk(2 * n, -1), ex(2 * n, -1), pv(2 * n, -1), sm(2 * n, -1), bk(2 * n, -1),
        bv(2 * n, -1);
    for (int j = 0; j < J; ++j) {
        if (qk_chunk(j, n) >= 0) {
            PP_CHECK(qk[qk_chunk(j, n)] < 0, "qk twice");
            qk[qk_chunk(j, n)] = j;
        }
        if (exp_chunk(j) >= 0) {
            PP_CHECK(ex[exp_chunk(j)] < 0, "exp twice");
            ex[exp_chunk(j)] = j;
        }
        if (pv_chunk(j) >= 0) {
            PP_CHECK(pv[pv_chunk(j)] < 0, "pv twice");
            pv[pv_chunk(j)] = j;
        }
        if (sm1_chunk(j, n) >= 0) {
            PP_CHECK(sm[sm1_chunk(j, n)] < 0, "sm1 twice");
            sm[sm1_chunk(j, n)] = j;
        }
        if (burst_k_chunk(j, n) >= 0)
            bk[burst_k_chunk(j, n)] = j;
        if (burst_v_chunk(j) >= 0)
            bv[burst_v_chunk(j)] = j;
    }
    for (int c = 0; c < 2 * n; ++c) {
        PP_CHECK(qk[c] >= 0 && ex[c] >= 0 && pv[c] >= 0 && sm[c] >= 0 && bk[c] >= 0 && bv[c] >= 0,
                 "n %d chunk %d missing", n, c);
        PP_CHECK(bk[c] <= qk[c] && qk[c] < sm[c] && sm[c] < ex[c] && ex[c] < pv[c] &&
                     bv[c] <= pv[c],
                 "n %d chunk %d order", n, c);
        PP_CHECK(bv[c] == ex[c] && pv[c] == ex[c] + 1,
                 "n %d chunk %d: V burst opens X(c), PV in Y(c)", n, c);
        // registers: S[c & 1] is written by QK(c) and read until Y(c); QK(c + 2) (the same buffer)
        // only after it
        if (c + 2 < 2 * n)
            PP_CHECK(qk[c + 2] > pv[c], "n %d chunk %d: S buffer reuse", n, c);
        // ka: K(c) burst before cluster bk[c], consumed by QK(c); the next K burst (c + 1) after
        // QK(c)
        if (c + 1 < 2 * n)
            PP_CHECK(bk[c + 1] > qk[c], "n %d chunk %d: ka reuse", n, c);
        // va: V(c + 1) burst after PV(c)
        if (c + 1 < 2 * n)
            PP_CHECK(bv[c + 1] > pv[c], "n %d chunk %d: va reuse", n, c);
        // rescale C(c) right after SM1(c) (end of the same Y), before exp(c)
    }
    // stage bodies: stage t = clusters 4t + 2 .. 4t + 5; op issue at its first, tensorcnt at its
    // last
    for (int t = 0; t < n; ++t) {
        PP_CHECK(op_issue(4 * t + 2, n) == (t <= n - 2 ? t + 3 : -1), "op issue stage %d", t);
        PP_CHECK(tcnt_end(4 * t + 5, n) == (t <= n - 2 ? 1 : -1), "tcnt stage %d", t);
        for (int j = 4 * t + 3; j <= 4 * t + 5; ++j)
            PP_CHECK(op_issue(j, n) < 0, "op issue only at X(2t)");
        for (int j = 4 * t + 2; j <= 4 * t + 4; ++j)
            PP_CHECK(tcnt_end(j, n) < 0, "tcnt only at Y(2t+1)");
        PP_CHECK(burst_k_chunk(4 * t + 3, n) == (t <= n - 2 ? 2 * t + 2 : -1) &&
                     burst_k_chunk(4 * t + 5, n) == (t <= n - 2 ? 2 * t + 3 : -1),
                 "K bursts stage %d", t);
        PP_CHECK(qk_chunk(4 * t + 4, n) == (t <= n - 2 ? 2 * t + 2 : -1) &&
                     qk_chunk(4 * t + 2, n) == 2 * t + 1,
                 "QK stage %d", t);
        PP_CHECK(sm1_chunk(4 * t + 5, n) == (t <= n - 2 ? 2 * t + 2 : -1) &&
                     sm1_chunk(4 * t + 3, n) == 2 * t + 1,
                 "SM1 stage %d", t);
    }
    PP_CHECK(tcnt_end(1, n) == 1 && op_issue(0, n) < 0 && op_issue(1, n) < 0, "prologue clusters");
    return 0;
}

// the stage plan (what body.h is generated from) == the cluster-indexed specification, cluster by
// cluster, and its expansion kernel_events == wave_events for every wave
inline int check_plan(int n) {
    int j = 0;
    for (int t = -1; t < n; ++t) {
        const int M = stage_mode(t, n), c0 = 2 * t;
        PP_CHECK(t < 0 || M == (t < n - 2    ? SM_MAIN
                                : t == n - 2 ? SM_PEN
                                             : SM_LAST),
                 "n %d: stage %d mode %d", n, t, M);
        for (int i = cl_first(M); i < 4; ++i, ++j) {
            const ClPlan p  = cl_plan(M, i);
            auto         ch = [&](int off) { return off >= 0 ? c0 + off : -1; };
            PP_CHECK(j == 4 * t + 2 + i && cl_is_x(j) == (i % 2 == 0) && p.win == (j >= 1),
                     "n %d j %d: cluster index", n, j);
            PP_CHECK(ch(p.kc) == burst_k_chunk(j, n) && ch(p.vc) == burst_v_chunk(j),
                     "n %d j %d: bursts", n, j);
            PP_CHECK((p.op >= 0 ? t + p.op : -1) == op_issue(j, n) && p.tcnt == tcnt_end(j, n),
                     "n %d j %d: TDM", n, j);
            PP_CHECK(ch(p.qk) == qk_chunk(j, n) && ch(p.ex) == exp_chunk(j) &&
                         ch(p.pv) == pv_chunk(j) && ch(p.sm1) == sm1_chunk(j, n),
                     "n %d j %d: compute", n, j);
            PP_CHECK(cl_is_x(j) ? p.pv < 0 && p.sm1 < 0 && !p.masked && !p.seed
                                : p.qk < 0 && p.ex < 0,
                     "n %d j %d: kind", n, j);
            PP_CHECK(p.sm1 < 0 || !last_stage_chunk(c0 + p.sm1, n) || p.masked,
                     "n %d j %d: unmasked last-stage SM1", n, j);
            PP_CHECK(p.seed == (p.sm1 >= 0 && c0 + p.sm1 == 0), "n %d j %d: seed", n, j);
        }
    }
    PP_CHECK(j == n_clusters(n), "n %d: %d clusters", n, j);
    for (int w = 0; w < NWAVE; ++w)
        PP_CHECK(kernel_events(w, n) == wave_events(w, n),
                 "n %d w %d: kernel_events != wave_events", n, w);
    return 0;
}

// ---- global bounds + exactly-once coverage + TRUE TDM extents for one shape
// ----------------------------------------
inline int check_shape(int B, int Sq, int Skv, int Hq, int Hkv, int causal) {
    const int G = Hq / Hkv, nqb = Sq / QB, nhq4 = Hq / NH, nkv_st = Skv / ST, cst = (Skv - Sq) / ST;
    const int64_t q_n = (int64_t) B * Sq * Hq * D, kv_n = (int64_t) B * Skv * Hkv * D,
                  lse_n = (int64_t) B * Hq * Sq;
    PP_CHECK(2 * q_n < (1LL << 31) && 4 * lse_n < (1LL << 31) && G % 4 == 0, "extents / G");
    std::vector<uint8_t> ocov((size_t) (q_n / 8), 0), lcov((size_t) lse_n, 0),
        seen((size_t) B * nhq4 * nqb, 0);
    for (int i = 0; i < B * nhq4 * nqb; ++i) {
        const Work wk = work_v0(i, B, nhq4, nqb);
        PP_CHECK(wk.b >= 0 && wk.b < B && wk.hq4 >= 0 && wk.hq4 < nhq4 && wk.qb >= 0 && wk.qb < nqb,
                 "work");
        seen[((size_t) wk.b * nhq4 + wk.hq4) * nqb + wk.qb]++;
        if (i > 0)
            PP_CHECK(work_v0(i - 1, B, nhq4, nqb).qb >= wk.qb, "LPT order");
        const int hkv = hkv_of(wk.hq4, G), n = n_stages(wk.qb, cst, nkv_st, causal);
        PP_CHECK(n >= 1 && n <= nkv_st && hkv < Hkv, "n_stages / hkv");
        for (int w = 0; w < NWAVE; ++w) {
            if (grp_of(w) == 0)
                for (int k = 0; k < n_ops(n); ++k) {
                    const int     s  = pair_of(w);
                    const int64_t e0 = kv_op_el(wk.b, Skv, Hkv, hkv, s, k, n);
                    int64_t       e1 = e0 + 8;
                    if (op_real(k, n)) {
                        const int d1 = kv_op_dim1(Skv, s, k);
                        PP_CHECK(d1 == OPR, "kv tensor_dim1 %d", d1);
                        e1 = e0 + (int64_t) (d1 - 1) * Hkv * D + D;
                    }
                    PP_CHECK(e0 >= 0 && e1 <= kv_n, "kv op footprint");
                }
            const int64_t e0 = q_op_el(wk.b, Sq, Hq, wk.hq4, wk.qb, w);
            const int     d1 = q_op_dim1(Sq, wk.qb, w);
            PP_CHECK(d1 == QW, "q tensor_dim1 %d", d1);
            PP_CHECK(e0 >= 0 && e0 + (int64_t) (d1 - 1) * Hq * D + NHW * D <= q_n,
                     "q op footprint");
            for (int p = 0; p < QW; ++p)
                for (int l = 0; l < 32; ++l) {
                    const int64_t ob =
                        o_store_soff(wk.b, Sq, Hq, wk.hq4, wk.qb, w, p) + o_store_voff(l);
                    PP_CHECK(ob >= 0 && ob + 16 <= 2 * q_n && ob % 16 == 0, "o store");
                    ocov[(size_t) (ob / 16)]++;
                }
            for (int l = 0; l < 32; ++l) {
                const int64_t lb =
                    lse_store_soff(wk.b, Hq, Sq, wk.hq4, wk.qb, w) + lse_store_voff(Sq, l);
                PP_CHECK(lb >= 0 && lb + 4 <= 4 * lse_n, "lse store");
                lcov[(size_t) (lb / 4)]++;
            }
        }
    }
    for (auto x : seen)
        PP_CHECK(x == 1, "work bijection");
    for (auto x : ocov)
        PP_CHECK(x == 1, "O coverage");
    for (auto x : lcov)
        PP_CHECK(x == 1, "LSE coverage");
    return 0;
}

inline int selftest(bool verbose = false) {
    if (verbose) {
        const G1 q = q_g1(32, QW), kv = kv_g1(0, 8, OPR), vv = kv_g1(1, 8, OPR), dm = dummy_g1(0);
        std::printf("  LAG %d K_W0 %08x V_W0 %08x Q_W0 %08x\n", LAG, K_W0, V_W0, Q_W0);
        const G1   *gs[4] = {&q, &kv, &vv, &dm};
        const char *nm[4] = {"Q op", "K op", "V op", "dummy"};
        for (int j = 0; j < 4; ++j) {
            std::printf("  g1 %-6s", nm[j]);
            for (int x = 0; x < 8; ++x)
                std::printf(" %08x", gs[j]->w[x]);
            std::printf("\n");
        }
    }
    PP_CHECK(kv_op_dim1(100, 1, 1) == 4 && kv_op_dim1(100, 0, 2) == 0 &&
                 q_op_dim1(72, 0, 3) == 16 && q_op_dim1(72, 1, 0) == 8,
             "clamp units");
    for (int n = 1; n <= 9; ++n)
        for (int s = 0; s < NPAIR; ++s)
            for (int k = 0; k < n_ops(n) + 1; ++k)
                for (int Skv : {64 * n, 64 * n + 64, 100}) {
                    const KvDesc d = kv_op_desc(1, Skv, 3, 2, s, k, n, kv_op_dim1(Skv, s, k));
                    const G1     g = op_real(k, n) ? kv_g1(op_role_v(s), 3, kv_op_dim1(Skv, s, k))
                                                   : dummy_g1(op_role_v(s));
                    PP_CHECK(d.lds == (uint32_t) kv_op_lds(s, k, n) &&
                                 d.el == kv_op_el(1, Skv, 3, 2, s, k, n),
                             "kv_op_desc lds / el");
                    for (int i = 0; i < 8; ++i)
                        PP_CHECK(d.g1[i] == g.w[i], "kv_op_desc g1 word %d", i);
                }
    if (check_lds() || check_mask())
        return 1;
    for (int n = 1; n <= 140; ++n)
        if (check_schedule(n) || check_plan(n) || check_protocol(n) ||
            check_protocol(n, MU_BURST_AFTER))
            return 1;
    // negative controls of the protocol replay: each must fail for some n
    const int muts[] = {MU_TCNT2, MU_ISSUE_EARLY, MU_NO_LAG_BAR, MU_PUBLISH_LATE, MU_NO_EXIT_TCNT};
    const char *mnames[] = {"tensorcnt 2", "op issue one cluster early",
                            "hi lag barrier missing (w7)", "stage published one barrier late",
                            "no exit tensorcnt"};
    for (int m = 0; m < 5; ++m) {
        if (LAG == 0 && muts[m] == MU_NO_LAG_BAR)
            continue;
        if (LAG == 0 && muts[m] == MU_ISSUE_EARLY)
            continue; // lockstep: one cluster earlier is still after the reads
        bool caught = false;
        for (int n = 1; n <= 12 && !caught; ++n) {
            std::printf("  [negctl %s n %d] ", mnames[m], n);
            caught = check_protocol(n, muts[m]) != 0;
            if (!caught)
                std::printf("passed\n");
        }
        PP_CHECK(caught, "negative control '%s' not caught", mnames[m]);
    }
    for (int n = 1; n <= 6; ++n)
        for (int causal = 0; causal < 2; ++causal)
            for (int s = 0; s < NPAIR; ++s)
                if (check_softmax(n, causal, s, 12345u + 7u * n + 3u * s + causal))
                    return 1;
    const int shapes[][6] = {{1, 256, 8, 2, 256, 1},    {1, 64, 4, 1, 64, 1},
                             {1, 128, 8, 1, 128, 1},    {2, 256, 16, 2, 256, 1},
                             {1, 192, 16, 1, 192, 1},   {1, 128, 4, 1, 256, 1},
                             {2, 192, 8, 2, 320, 1},    {1, 256, 8, 2, 128 + 256, 0},
                             {1, 4096, 32, 8, 4096, 1}, {4, 8192, 32, 8, 8192, 1}};
    for (auto &s : shapes)
        if (check_shape(s[0], s[1], s[4], s[2], s[3], s[5])) {
            std::printf("shape B%d Sq%d\n", s[0], s[1]);
            return 1;
        }
    std::printf("pp selftest OK (LAG %d)\n", LAG);
    return 0;
}

} // namespace pp
#endif // PP_IDX_SELFTEST
