/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// fwd1/idx.h -- every index formula of the gfx1250 d128 GQA forward v1 (k_attn_fwd_v1), in ONE
// place: the kernel (hip/src/k_attn_fwd_v1.hip) and its bounds checker
// (hk/lab/bounds/fwd_v1_check.cpp) both include THIS file. Notes: hk/notes/fwd_v1_impl.md.
//
//   O[b, p, hq, :] = softmax_kv(scale * Q[b, p, hq, :] K[b, kv, hkv, :]^T) V[b, kv, hkv, :],
//   causal bottom-right (kv <= p + Skv - Sq), LSE[b, hq, p] = natural log of the row sum.
//
// v1 = v0's item body (fwd/idx.h, namespace f0: 64 positions x one head quad per item, 4 waves, one
// 32-row TDM op per wave per 64-kv stage, depth-4 ring, one split-barrier phase per stage) run by a
// PERSISTENT grid of W workgroups. Self-contained (namespace f1): no dependency on fwd/idx.h.
//
// Persistent schedule. Workgroup g runs the item list wg_item(g, 0 .. J-1) (J = wg_items(g); J = 0 only for a
// grid larger than #units, which fwd_v1.plan never launches: the kernel then returns before any memory op or
// barrier, and the checker replays such a grid):
//   MODE_ITEMS  items in v0's longest-first order, item v = j W + g (W = #items: exactly v0's grid)
//   MODE_PAIRS  units = zig-zag pairs (qb_hi = nqb-1-pr, then qb_lo = pr) of one head (b, hq4), plus
//               (odd nqb) one single middle block per head after all pairs. Unit v = s W + rank(g),
//               rank XCD-major (XCC = wg % 8): at step s the W/8 workgroups of one XCD take W/8
//               consecutive units, i.e. the same head. Every pair has the same stage count.
// The K/V op stream, ring slots and barrier phases are WG-LIFETIME counters: global stage T of item j
// = T0_j + t (T0_j = sum of the stage counts of items 0..j-1), slot T % KD, phase P_T. Every stage
// of every item runs WAIT(P_{T+1}) + op(T+3) at its even barrier point and tensorcnt + SIGNAL(P_{T+2})
// at its odd one, so the next item's first stages (and its Q tile) are loaded during the current
// item's last stages. N = sum of all stage counts; ops >= N are 16-B dummies; exit = WAIT(P_{N+1}).
// Q tiles: Q(j+1) is issued at the even barrier point of item j's LAST stage (before its K/V op),
// hence tensorcnt 2 at that stage's odd point and tensorcnt 1 at item j+1's prologue; item J's "Q" is a
// dummy. Q and the O image have SEPARATE per-wave LDS regions (Q(j+1) lands while item j's epilogue
// writes its O image).
// Strided layouts (k_attn_fwd_v1 built with -DHK_STRIDES=1): Q, K, V, O each BSHD-contiguous or SBHD storage,
// addressed by (batch, seq) element strides through the *_s functions (section "strided layouts"); the BSHD
// functions above are the default build's and the strided ones equal them at the BSHD strides (check_strides).
#pragma once

#if defined(__HIP__) || defined(__HIPCC__)
#define F1_HD __host__ __device__ __attribute__((always_inline)) constexpr
#else
#define F1_HD constexpr
#endif

#include <stdint.h>

// K / V burst placement (a compile-time variant; kernel and checker must agree):
//   0  v0: A(c) = V(c) tr16 burst, E(c) = K(c+2) b128 burst after the barrier point (two DS bursts)
//   1  one burst per chunk at A(c): K(c+1) (consumed by QK(c+1) in B(c): its DS latency is exposed) then V(c)
//   2  one burst per chunk at E(c): K(c+2) then V(c+1) into the other V set (V double-buffered, +64 VGPR);
//      every burst has a whole compute region of slack before its first use (default)
//   3  one burst per STAGE at the even E(2t): K(2t+2), V(2t+1), K(2t+3), V(2t+2) (the last stage: V(2t+1) only);
//      the odd chunk has no DS burst (its barrier point is tensorcnt + SIGNAL only); 2 K sets, 3 V sets
//      (hk/notes/fwd_v11.md)
#ifndef FWD1_MDS
#define FWD1_MDS 2
#endif
// epilogue placement: 0 epilogue(j) before item j+1's prologue; 1 item j+1's prologue loads first,
// then epilogue(j)'s normalize / convert runs under QK(0) of item j+1
#ifndef FWD1_EPI
#define FWD1_EPI 0
#endif

namespace f1 {

// ---- shape-independent constants ----------------------------------------------------------------
constexpr int D     = 128;
constexpr int WAVE  = 32;
constexpr int NWAVE = 4; // 1 wave per SIMD, 1 WG per CU
constexpr int BLOCK = NWAVE * WAVE;
constexpr int QB    = 64; // query positions per item
constexpr int QW    = 16; // query positions per wave
constexpr int NH    = 4;  // q heads per item = q-tiles per wave
constexpr int ST    = 64; // kv rows per ring stage
constexpr int CH    = 32; // kv rows per chunk
constexpr int OPR   = 32; // kv rows per TDM op (one op per wave per stage)
constexpr int NDK   = D / 32;
constexpr int NDT   = D / 16;
constexpr int NXCD  = 8; // XCC = workgroup id % 8 (measured, mb_xcd)

// pitches: the lin16 bank model, GPU-validated by v0 (272 / 288 / 1040)
constexpr int K_PITCH  = 272;
constexpr int V_PITCH  = 288;
constexpr int Q_PITCH  = 1040;
constexpr int KD       = 4;
constexpr int K_SLOT_B = ST * K_PITCH; // 17408
constexpr int V_SLOT_B = ST * V_PITCH; // 18432
constexpr int QO_W_B   = QW * Q_PITCH; // 16640: one wave's Q staging / O image

// LDS map (byte offsets; 64-KB segment = off >> 16)
constexpr int K_OFF     = 0;                        // K ring [0, 69632)
constexpr int O_OFF_LO  = K_OFF + KD * K_SLOT_B;    // O images of waves 0..2: [69632, 119552)
constexpr int DUMMY_OFF = 130048;                   // 4 K/V dummy cells + 4 Q dummy cells (128 B)
constexpr int V_OFF     = 131072;                   // V ring [131072, 204800)
constexpr int O_OFF_HI  = V_OFF + KD * V_SLOT_B;    // O image of wave 3: [204800, 221440)
constexpr int LDS_BYTES = 327680;                   // the whole 320 KB: one WG per CU
constexpr int Q_OFF     = LDS_BYTES - NWAVE * QO_W_B; // 4 per-wave Q regions [261120, 327680)

F1_HD int op_role_v(int w) { return w >> 1; } // waves 0, 1: K; waves 2, 3: V
F1_HD int op_half(int w) { return w & 1; }
F1_HD int issuer_of(int role_v, int half) { return 2 * role_v + half; }
constexpr int TCNT_STEADY = 1; // odd barrier point of every stage but an item's last
constexpr int TCNT_LAST   = 2; // odd barrier point of an item's last stage (Q(j+1) in flight too)
constexpr int TCNT_ITEM   = 1; // item prologue: own Q(j) landed (op(T0+2) may fly)

F1_HD uint32_t tdm_w0(int row_b, int pitch_b) {
    int lg = 0;
    for (int x = row_b / 4; x > 1; x /= 2) ++lg;
    return (1u << 16) | (1u << 20) | ((uint32_t) (lg - 1) << 22) |
           ((uint32_t) ((pitch_b - row_b) / 4 - 1) << 25);
}
constexpr uint32_t K_W0    = tdm_w0(2 * D, K_PITCH);      // 0x07510000
constexpr uint32_t V_W0    = tdm_w0(2 * D, V_PITCH);      // 0x0f510000
constexpr uint32_t Q_W0    = tdm_w0(2 * NH * D, Q_PITCH); // 0x07d10000
constexpr uint32_t G0_TYPE = 0x80000000u;

constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.6931471805599453f;
constexpr float TH2   = 8.0f * LOG2E; // lazy-rescale threshold: 8 nat in the log2 domain

F1_HD int imin(int a, int b) { return a < b ? a : b; }
F1_HD int imax(int a, int b) { return a > b ? a : b; }

// ---- work items -----------------------------------------------------------------------------------
struct Work {
    int b, hq4, qb;
};
// v0's longest-first order (== f0::work_v0): item i -> (b, hq4, qb), the first B * nhq4 items take the
// heaviest q block of every head
F1_HD Work work_lpt(int i, int B, int nhq4, int nqb) {
    const int nh = B * nhq4, rank = i / nh, rem = i - rank * nh;
    return Work{rem / nhq4, rem % nhq4, nqb - 1 - rank};
}
// kv head of head quad hq4 = (NH hq4) / G. G is a power of two (host gate: G in {4, 8, 16}), so the quotient is
// a shift by log2 G (s_ff1 + s_lshr; no integer-division sequence in the item loop). The checker computes the head
// map independently ((4 hq4 + h) Hkv / Hq per q head) and the selftest compares hkv_of with the plain quotient.
F1_HD int hkv_of(int hq4, int G) { return (NH * hq4) >> __builtin_ctz((unsigned) G); }
F1_HD int wave_pos0(int qb, int w) { return QB * qb + QW * w; }
F1_HD int n_stages(int qb, int cst, int nkv_st, int causal) {
    return causal != 0 ? qb + 1 + cst : nkv_st;
}

// ---- division by a runtime constant (host-computed magic numbers; no division sequence in the kernel) ---
// Granlund-Montgomery round-up method for numerators n < 2^31: l = ceil(log2 d), m = ceil(2^(31+l) / d)
// (< 2^32), n / d == (n m) >> (31 + l) for every 0 <= n < 2^31 (proof: m d - 2^(31+l) < d <= 2^l). In SALU:
// s_mul_u64 + s_lshr_b64. The host passes (m, 31 + l) per divisor as kernel arguments (fwd_v1.py fdiv_of
// mirrors fdiv_of; hk/lab/test_ext_fwd1.py compares them).
struct Fdiv {
    uint32_t m;
    int      sh;
};
F1_HD Fdiv fdiv_of(uint32_t d) {
    int l = 0;
    while ((uint64_t(1) << l) < (uint64_t) d) ++l;
    return Fdiv{(uint32_t) (((uint64_t(1) << (31 + l)) + d - 1) / d), 31 + l};
}
F1_HD uint32_t fdiv(uint32_t n, const Fdiv &f) { return (uint32_t) (((uint64_t) n * f.m) >> f.sh); }

// ---- persistent schedule --------------------------------------------------------------------------
enum : int { MODE_ITEMS = 0, MODE_PAIRS = 1 };
// dW, dnp, dh4, dnh: fdiv_of(W), fdiv_of(max(nqb / 2, 1)), fdiv_of(nhq4), fdiv_of(B nhq4)
struct Sched {
    int  mode, W, B, nhq4, nqb;
    Fdiv dW, dnp, dh4, dnh;
};
F1_HD Sched make_sched(int mode, int W, int B, int nhq4, int nqb) {
    return Sched{mode, W, B, nhq4, nqb, fdiv_of((uint32_t) W), fdiv_of((uint32_t) imax(nqb / 2, 1)),
                 fdiv_of((uint32_t) nhq4), fdiv_of((uint32_t) (B * nhq4))};
}
F1_HD int n_items(int B, int nhq4, int nqb) { return B * nhq4 * nqb; }
F1_HD int n_pair_units(int B, int nhq4, int nqb) { return B * nhq4 * (nqb / 2); }
F1_HD int n_units(int mode, int B, int nhq4, int nqb) {
    return mode == MODE_PAIRS ? B * nhq4 * ((nqb + 1) / 2) : n_items(B, nhq4, nqb);
}
F1_HD int n_units(const Sched &s) { return n_units(s.mode, s.B, s.nhq4, s.nqb); }
// unit order position of workgroup g: XCD-major in pairs mode (a permutation of [0, W) when W % 8 == 0)
F1_HD int wg_rank(const Sched &s, int g) {
    return s.mode == MODE_PAIRS && s.W % NXCD == 0 ? (g % NXCD) * (s.W / NXCD) + g / NXCD : g;
}
// number of s >= 0 with s W + r < lim (unsigned: every operand is non-negative and < 2^31)
F1_HD int count_below(int lim, int r, const Fdiv &dW) {
    return r < lim ? (int) fdiv((uint32_t) (lim - 1 - r), dW) + 1 : 0;
}
F1_HD int wg_items(const Sched &s, int g) {
    const int r = wg_rank(s, g);
    if (s.mode != MODE_PAIRS) return count_below(n_items(s.B, s.nhq4, s.nqb), r, s.dW);
    const int np = count_below(n_pair_units(s.B, s.nhq4, s.nqb), r, s.dW);
    const int nu = count_below(n_units(s), r, s.dW);
    return 2 * np + (nu - np);
}
// item j (0 <= j < wg_items) of workgroup g. Every quotient by a runtime value is an fdiv (selects, no
// branch in the kernel: both arms are cheap SALU)
F1_HD Work wg_item(const Sched &s, int g, int j) {
    const int r = wg_rank(s, g);
    if (s.mode != MODE_PAIRS) { // v0's longest-first order (== work_lpt(j W + r)), by fdiv
        const uint32_t i = (uint32_t) (j * s.W + r), nh = (uint32_t) (s.B * s.nhq4);
        const uint32_t rank = fdiv(i, s.dnh), rem = i - rank * nh, b = fdiv(rem, s.dh4);
        return Work{(int) b, (int) (rem - b * (uint32_t) s.nhq4), s.nqb - 1 - (int) rank};
    }
    const uint32_t npair = (uint32_t) (s.nqb / 2), NP = (uint32_t) n_pair_units(s.B, s.nhq4, s.nqb);
    const int      np    = count_below((int) NP, r, s.dW);
    const bool     pair  = j < 2 * np;
    // pair unit j / 2 (heavy block first) or single middle block (odd nqb) number j - np
    const uint32_t v  = (uint32_t) ((pair ? j >> 1 : j - np) * s.W + r);
    const uint32_t hp = fdiv(v, s.dnp), pr = v - hp * npair;
    const uint32_t hb = pair ? hp : v - NP;
    const int      qb = pair ? ((j & 1) == 0 ? s.nqb - 1 - (int) pr : (int) pr) : s.nqb / 2;
    const uint32_t b  = fdiv(hb, s.dh4);
    return Work{(int) b, (int) (hb - b * (uint32_t) s.nhq4), qb};
}

// ---- causal mask (branch-free; prims frag.h mask_row_gt_col), per item as v0 ----------------------
constexpr int MASK_NONE = 64;
F1_HD bool last_stage_chunk(int c, int n) { return c >= 2 * n - 2; }
F1_HD int mask_off(int w, int c, int n, int kt, int causal) {
    return causal != 0 && last_stage_chunk(c, n) ? QW * (w - (2 * (c - (2 * n - 2)) + kt)) : MASK_NONE;
}
F1_HD int thr_rgc(int lane, int off) { return lane % 16 - 8 * (lane / 16) + off; }
F1_HD bool masked(int lane, int i, int off) { return i > thr_rgc(lane, off); }

// ---- ring (global stage T) ----------------------------------------------------------------------------
F1_HD int slot(int T) { return T & (KD - 1); }
static_assert((KD & (KD - 1)) == 0, "slot = T & (KD - 1)");
F1_HD int k_slot_off(int T) { return K_OFF + slot(T) * K_SLOT_B; }
F1_HD int v_slot_off(int T) { return V_OFF + slot(T) * V_SLOT_B; }
F1_HD int dummy_off(int w) { return DUMMY_OFF + 16 * w; }      // K/V dummy cell of wave w
F1_HD int q_dummy_off(int w) { return DUMMY_OFF + 64 + 16 * w; } // Q dummy cell of wave w
F1_HD int o_lds(int w) { return w < 3 ? O_OFF_LO + w * QO_W_B : O_OFF_HI; }
F1_HD int q_op_lds(int w) { return Q_OFF + w * QO_W_B; }

// ---- global element offsets (int64) ------------------------------------------------------------------
F1_HD int64_t q_el(int b, int Sq, int p, int Hq, int hq) {
    return ((int64_t) b * Sq + p) * Hq * D + (int64_t) hq * D;
}
F1_HD int64_t kv_el(int b, int Skv, int kv, int Hkv, int hkv) {
    return ((int64_t) b * Skv + kv) * Hkv * D + (int64_t) hkv * D;
}
F1_HD int64_t lse_el(int b, int Hq, int hq, int Sq, int p) { return ((int64_t) b * Hq + hq) * Sq + p; }

// ---- TDM descriptors (2-D; TRUE extents: tensor_dim1 = rows below the origin clamped to the tile) ---
struct G1 {
    uint32_t w[8];
};
F1_HD G1 g1_2d(uint32_t w0, uint32_t tdim0, uint32_t tdim1, uint32_t tile0, uint32_t tile1, uint64_t stride0) {
    return G1{{w0, (tdim0 & 0xFFFFu) << 16, (tdim0 >> 16) | ((tdim1 & 0xFFFFu) << 16),
               (tdim1 >> 16) | (tile0 << 16), tile1, (uint32_t) stride0, (uint32_t) (stride0 >> 32) & 0xFFFFu, 0u}};
}
// Q op of wave w for item (b, hq4, qb): [16 positions][512 = 4 heads x 128] at row stride Hq * 128
F1_HD int q_op_dim1(int Sq, int qb, int w) { return imin(imax(Sq - wave_pos0(qb, w), 0), QW); }
F1_HD G1 q_g1(int Hq, int dim1) { return g1_2d(Q_W0, NH * D, (uint32_t) dim1, NH * D, QW, (uint64_t) Hq * D); }
F1_HD int64_t q_op_el(int b, int Sq, int Hq, int hq4, int qb, int w) {
    return q_el(b, Sq, wave_pos0(qb, w), Hq, NH * hq4);
}
// K / V op of wave w for local stage s of an item: rows [64 s + 32 half, +32) of (b, hkv)
F1_HD int kv_op_row0(int w, int s) { return ST * s + OPR * op_half(w); }
// clamp(Skv - row0, 0, 32) in 32-bit unsigned arithmetic (s_sub / s_min_u32: no VALU, no i64 compare)
F1_HD int kv_op_dim1(int Skv, int w, int s) {
    const uint32_t a = (uint32_t) Skv, r0 = (uint32_t) kv_op_row0(w, s);
    const uint32_t d = a > r0 ? a - r0 : 0u;
    return (int) (d < (uint32_t) OPR ? d : (uint32_t) OPR);
}
F1_HD G1 kv_g1(int role_v, int Hkv, int dim1) {
    return g1_2d(role_v ? V_W0 : K_W0, D, (uint32_t) dim1, D, OPR, (uint64_t) Hkv * D);
}
// 16-B dummy: an 8 x 1 element tile at row 0 of a tensor that exists (== hk_attn::tdm_g1_dummy)
F1_HD G1 dummy_g1(int role_v) { return g1_2d(role_v ? V_W0 : K_W0, 8, 1, 8, 1, 0); }
// LDS address of a REAL K / V op of wave w at global stage T
F1_HD int kv_op_lds(int w, int T) {
    return op_role_v(w) ? v_slot_off(T) + OPR * op_half(w) * V_PITCH : k_slot_off(T) + OPR * op_half(w) * K_PITCH;
}
F1_HD int64_t kv_op_el(int b, int Skv, int Hkv, int hkv, int w, int s) {
    return kv_el(b, Skv, kv_op_row0(w, s), Hkv, hkv);
}

// ---- the K / V op stream: global stage -> owning item ---------------------------------------------------
// The k-th stage (k = 0, 1, 2) of the stream after the first j1 items of the workgroup (j1 = j + 1 when
// asked at item j; j1 = 0 in the workgroup prologue): owning item and its local stage, or none (a dummy)
struct StRef {
    int real, b, hkv, s;
};
struct Shape {
    int B, Sq, Skv, Hq, Hkv, G, nqb, nhq4, nkv_st, cst, causal;
};
F1_HD StRef stream_stage(const Shape &sh, const Sched &sc, int g, int J, int j1, int k) {
    int rem = k;
    for (int i = j1; i < J && i < j1 + 3; ++i) {
        const Work wk = wg_item(sc, g, i);
        const int  ni = n_stages(wk.qb, sh.cst, sh.nkv_st, sh.causal);
        if (rem < ni) return StRef{1, wk.b, hkv_of(wk.hq4, sh.G), rem};
        rem -= ni;
    }
    return StRef{0, 0, 0, 0};
}

// kernel form of stream_stage for k = 0, 1, 2: items j1, j1 + 1, j1 + 2 decoded once each (an index past
// the list is clamped to the last item and gets 0 stages), then plain selects
F1_HD void stream3(const Shape &sh, const Sched &sc, int g, int J, int j1, StRef (&out)[3]) {
    const Work w0 = wg_item(sc, g, imin(j1, J - 1)), w1 = wg_item(sc, g, imin(j1 + 1, J - 1)),
               w2 = wg_item(sc, g, imin(j1 + 2, J - 1));
    const int  n0 = j1 < J ? n_stages(w0.qb, sh.cst, sh.nkv_st, sh.causal) : 0;
    const int  n1 = j1 + 1 < J ? n_stages(w1.qb, sh.cst, sh.nkv_st, sh.causal) : 0;
    const int  n2 = j1 + 2 < J ? n_stages(w2.qb, sh.cst, sh.nkv_st, sh.causal) : 0;
    const int  h0 = hkv_of(w0.hq4, sh.G), h1 = hkv_of(w1.hq4, sh.G), h2 = hkv_of(w2.hq4, sh.G);
    for (int k = 0; k < 3; ++k) { // selects only (no runtime-indexed local array: that would be scratch)
        const int  r1 = k - n0, r2 = k - n0 - n1;
        const bool i0 = k < n0, i1 = !i0 && r1 < n1, re = i0 || i1 || r2 < n2;
        const int  s  = i0 ? k : i1 ? r1 : r2;
        out[k]        = StRef{re ? 1 : 0, i0 ? w0.b : i1 ? w1.b : w2.b, i0 ? h0 : i1 ? h1 : h2, re ? s : 0};
    }
}

// compact K / V op (2 SGPRs): d1 > 0 = a real op of d1 rows at element el of the role tensor, d1 == 0 = a
// 16-B dummy (el 0). Element offsets fit 31 bits: the host gate keeps every K / V / Q byte extent < 2^31.
struct OpC {
    uint32_t el;
    int      d1;
};
F1_HD OpC op_c(const Shape &sh, const StRef &r, int w) {
    const uint32_t el = (uint32_t) kv_op_el(r.b, sh.Skv, sh.Hkv, r.hkv, w, r.s);
    const int      d1 = kv_op_dim1(sh.Skv, w, r.s);
    return OpC{r.real ? el : 0u, r.real ? d1 : 0};
}
// the item's own op at local stage s: el0 + s * sst (el0 = its stage-0 element, sst = elements per stage)
F1_HD uint32_t kv_el0(int b, int Skv, int Hkv, int hkv, int w) { return (uint32_t) kv_op_el(b, Skv, Hkv, hkv, w, 0); }
F1_HD uint32_t kv_sst(int Hkv) { return (uint32_t) (ST * Hkv * D); }
F1_HD OpC own_c(int Skv, uint32_t el0, uint32_t sst, int w, int s) {
    return OpC{el0 + (uint32_t) s * sst, kv_op_dim1(Skv, w, s)};
}
// compact Q op: d1 > 0 = item's Q tile (d1 rows) at element el of Q, 0 = a 16-B dummy of Q row 0
F1_HD OpC q_c(int real, const Work &x, int Sq, int Hq, int w) {
    const uint32_t el = (uint32_t) q_op_el(x.b, Sq, Hq, x.hq4, x.qb, w);
    const int      d1 = q_op_dim1(Sq, x.qb, w);
    return OpC{real ? el : 0u, real ? d1 : 0};
}

// ---- LDS read addresses (per-lane base + compile-time DS immediate), as v0 ----------------------------
F1_HD int k_lane(int l) { return (l % 16) * K_PITCH + (l / 16) * 16; }
F1_HD int k_imm(int hc, int kt, int k, int half) { return (CH * hc + 16 * kt) * K_PITCH + 64 * k + 32 * half; }
F1_HD int v_lane(int l) { return (8 * (l / 16) + l % 8) * V_PITCH + ((l / 8) % 2) * 16; }
F1_HD int v_imm(int hc, int half, int dt) { return (CH * hc + 16 * half) * V_PITCH + 32 * dt; }
F1_HD int q_lane(int w, int l) { return q_op_lds(w) + (l % 16) * Q_PITCH + (l / 16) * 16; }
F1_HD int q_imm(int h, int k, int half) { return 2 * D * h + 64 * k + 32 * half; }
// O image (the wave's O region, Q's layout): write = the O^T C fragment of (h, dt)
F1_HD int ow_lane(int w, int l) { return o_lds(w) + (l % 16) * Q_PITCH + (l / 16) * 16; }
F1_HD int ow_imm(int h, int dt) { return 2 * D * h + 32 * dt; }
// read-back of store s (0..31): position s / 2, bytes [512 (s % 2), +512) of the position's 1 KB
F1_HD int or_lane(int w, int l) { return o_lds(w) + 16 * l; }
F1_HD int or_imm(int s) { return (s / 2) * Q_PITCH + 512 * (s % 2); }
// landing models (row t of an op at lds + t * pitch)
F1_HD int k_land(int T, int row, int col) { return k_slot_off(T) + row * K_PITCH + 2 * col; }
F1_HD int v_land(int T, int row, int col) { return v_slot_off(T) + row * V_PITCH + 2 * col; }
F1_HD int q_land(int w, int h, int p, int col) { return q_op_lds(w) + p * Q_PITCH + 2 * (D * h + col); }
F1_HD int o_land(int w, int h, int p, int col) { return o_lds(w) + p * Q_PITCH + 2 * (D * h + col); }

// ---- global stores (byte offsets; host asserts every extent < 2^31) ---------------------------------------
F1_HD int64_t o_store_b(int b, int Sq, int Hq, int hq4, int qb, int w, int s, int l) {
    return 2 * q_el(b, Sq, wave_pos0(qb, w) + s / 2, Hq, NH * hq4) + 512 * (s % 2) + 16 * l;
}
F1_HD int64_t lse_store_b(int b, int Hq, int Sq, int hq4, int qb, int w, int k, int l) {
    return 4 * lse_el(b, Hq, NH * hq4 + 2 * (l / 16) + k, Sq, wave_pos0(qb, w) + l % 16);
}

// ---- strided layouts (HK_STRIDES builds only; hk/notes/fwd_v1_strides.md) ---------------------------------------
// Under -DHK_STRIDES=1 the kernel addresses Q, K, V and O, each independently, through two int64 ELEMENT strides,
// Lay{sb (batch), ss (seq)}; the head stride stays D and the element stride 1:
//   element (b, s, h, d) at b sb + s ss + h D + d.
// The host (fwd_v1.py layout_of) admits exactly two layouts per tensor of shape [B, S, H, 128]:
//   BSHD contiguous            sb = S H D, ss = H D       (the functions above are this layout)
//   SBHD storage, BSHD view    sb = H D,   ss = B H D     ([S, B, H, D] contiguous viewed through permute(1, 0, 2, 3))
// Both are bijections onto [0, B S H D) (the tensor's true extent; a batch's rows interleave with the other batches'
// in SBHD, so every extent / resource is the whole tensor). The TDM row stride (stride0 of group 1, 48 bits) is ss;
// the host keeps every stride in [D, 2^31) (the 31-bit compact offsets need el < 2^30 anyway). The default build
// (no HK_STRIDES) never calls these functions: its code objects are the validated ones byte for byte.
struct Lay {
    int64_t sb, ss;
};
F1_HD Lay lay_bshd(int S, int H) { return Lay{(int64_t) S * H * D, (int64_t) H * D}; }
F1_HD Lay lay_sbhd(int B, int H) { return Lay{(int64_t) H * D, (int64_t) B * H * D}; }
F1_HD int64_t el_s(const Lay &L, int b, int s, int h) { return (int64_t) b * L.sb + (int64_t) s * L.ss + (int64_t) h * D; }
// Q op: [16 positions][512 = 4 heads x 128] at row stride ss (the 4 heads of a quad are contiguous in both layouts)
F1_HD G1 q_g1_s(int64_t ss, int dim1) { return g1_2d(Q_W0, NH * D, (uint32_t) dim1, NH * D, QW, (uint64_t) ss); }
F1_HD int64_t q_op_el_s(const Lay &L, int b, int hq4, int qb, int w) { return el_s(L, b, wave_pos0(qb, w), NH * hq4); }
// K / V op: rows [64 s + 32 half, +32) of (b, hkv) at row stride ss of the role's tensor. The kernel holds this
// stride across its stage loop in ONE SGPR (kv_ss32: the host keeps ss < 2^31, so the narrowing is exact and the
// descriptor's high stride word is 0), as the default build holds Hkv; the int64 kernel argument is the contract
F1_HD uint32_t kv_ss32(int64_t ss) { return (uint32_t) ss; }
F1_HD G1 kv_g1_s(int role_v, uint32_t ss32, int dim1) {
    return g1_2d(role_v ? V_W0 : K_W0, D, (uint32_t) dim1, D, OPR, (uint64_t) ss32);
}
F1_HD int64_t kv_op_el_s(const Lay &L, int b, int hkv, int w, int s) { return el_s(L, b, kv_op_row0(w, s), hkv); }
F1_HD OpC op_c_s(const Shape &sh, const Lay &L, const StRef &r, int w) {
    const uint32_t el = (uint32_t) kv_op_el_s(L, r.b, r.hkv, w, r.s);
    const int      d1 = kv_op_dim1(sh.Skv, w, r.s);
    return OpC{r.real ? el : 0u, r.real ? d1 : 0};
}
F1_HD uint32_t kv_el0_s(const Lay &L, int b, int hkv, int w) { return (uint32_t) kv_op_el_s(L, b, hkv, w, 0); }
F1_HD uint32_t kv_sst_s(uint32_t ss32) { return (uint32_t) ST * ss32; } // elements per 64-row stage (own_c's step)
F1_HD OpC q_c_s(int real, const Work &x, int Sq, const Lay &L, int w) {
    const uint32_t el = (uint32_t) q_op_el_s(L, x.b, x.hq4, x.qb, w);
    const int      d1 = q_op_dim1(Sq, x.qb, w);
    return OpC{real ? el : 0u, real ? d1 : 0};
}
// O store s (0..31) of lane l: position s / 2, bytes [512 (s % 2) + 16 l, +16) of the position's 4-head row
F1_HD int64_t o_store_b_s(const Lay &L, int b, int hq4, int qb, int w, int s, int l) {
    return 2 * el_s(L, b, wave_pos0(qb, w) + s / 2, NH * hq4) + 512 * (s % 2) + 16 * l;
}

// ---- stage bodies of an item (n stages) ------------------------------------------------------------------
// main loop [0, n - 3): op(T + 3) is the item's own stage t + 3; stage n - 3 (n >= 3): op = stream stage
// 0 after the item; stage n - 2 (n >= 2, masked SM1 at its odd F): stream stage 1; stage n - 1 (last,
// masked SM1 at its even F, Q(j+1) before its op): stream stage 2.
F1_HD int main_stages(int n) { return imax(n - 3, 0); }
F1_HD bool has_x_stage(int n) { return n >= 3; }
F1_HD bool has_pen_stage(int n) { return n >= 2; }

// ---- constants for the host proof (ctypes / Python mirror) ------------------------------------------------
F1_HD long long kconst(int i) {
    return i == 0 ? D : i == 1 ? NWAVE : i == 2 ? QB : i == 3 ? QW : i == 4 ? NH : i == 5 ? ST
         : i == 6 ? CH : i == 7 ? K_PITCH : i == 8 ? V_PITCH : i == 9 ? KD : i == 10 ? K_SLOT_B
         : i == 11 ? V_SLOT_B : i == 12 ? QO_W_B : i == 13 ? K_OFF : i == 14 ? DUMMY_OFF : i == 15 ? V_OFF
         : i == 16 ? Q_OFF : i == 17 ? LDS_BYTES : i == 18 ? OPR : i == 19 ? Q_PITCH : i == 20 ? O_OFF_LO
         : i == 21 ? (long long) K_W0 : i == 22 ? (long long) V_W0 : i == 23 ? (long long) Q_W0
         : i == 24 ? MASK_NONE : i == 25 ? O_OFF_HI : i == 26 ? NXCD : i == 27 ? FWD1_MDS
         : i == 28 ? FWD1_EPI : -1;
}

} // namespace f1

#ifdef F1_IDX_SELFTEST
// ======================================= reference replay (host only) ===============================
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace f1 {

#define F1_CHECK(cond, ...)                                                                          \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                        \
            std::printf(__VA_ARGS__);                                                                \
            std::printf("\n");                                                                       \
            return 1;                                                                                \
        }                                                                                            \
    } while (0)

// lin16 bank model: each half-wave's 16 accesses of 16 B must cover 64 distinct 4-B banks
inline int bank_ok(const int *a32) {
    for (int g = 0; g < 2; ++g) {
        bool used[64] = {};
        for (int l = 16 * g; l < 16 * g + 16; ++l)
            for (int j = 0; j < 4; ++j) {
                const int bank = (a32[l] / 4 + j) % 64;
                if (used[bank]) return 0;
                used[bank] = true;
            }
    }
    return 1;
}

inline int check_lds() {
    struct R {
        int lo, hi;
        const char *n;
    } rs[] = {{K_OFF, K_OFF + KD * K_SLOT_B, "K"},   {V_OFF, V_OFF + KD * V_SLOT_B, "V"},
              {Q_OFF, Q_OFF + NWAVE * QO_W_B, "Q"},  {O_OFF_LO, O_OFF_LO + 3 * QO_W_B, "O0..2"},
              {O_OFF_HI, O_OFF_HI + QO_W_B, "O3"},   {DUMMY_OFF, DUMMY_OFF + 32 * NWAVE, "dummies"}};
    for (auto &a : rs) {
        F1_CHECK(a.lo >= 0 && a.hi <= LDS_BYTES && a.lo % 16 == 0, "region %s", a.n);
        for (auto &b : rs) F1_CHECK(&a == &b || a.hi <= b.lo || b.hi <= a.lo, "overlap %s %s", a.n, b.n);
    }
    for (int w = 0; w < NWAVE; ++w) {
        F1_CHECK((o_lds(w) - q_op_lds(0)) % 256 == 0, "O region bank alignment == Q region's (w %d)", w);
    }
    const uint32_t w0s[3] = {K_W0, V_W0, Q_W0};
    const int rowb[3] = {2 * D, 2 * D, 2 * NH * D}, pitch[3] = {K_PITCH, V_PITCH, Q_PITCH};
    for (int j = 0; j < 3; ++j) {
        const int ivl_b = 4 << (int) (((w0s[j] >> 22) & 7u) + 1), pad_b = 4 * (int) (((w0s[j] >> 25) & 127u) + 1);
        F1_CHECK(ivl_b == rowb[j] && pad_b == pitch[j] - rowb[j], "tdm w0 %d", j);
    }
    // read / write models, landing, immediates, banks (every access pattern of the kernel)
    int a[32];
    for (int T = 0; T < KD; ++T)
        for (int hc = 0; hc < 2; ++hc) {
            for (int kt = 0; kt < 2; ++kt)
                for (int k = 0; k < NDK; ++k)
                    for (int half = 0; half < 2; ++half) {
                        for (int l = 0; l < 32; ++l) {
                            const int row = CH * hc + 16 * kt + l % 16, col = 32 * k + 16 * half + 8 * (l / 16);
                            a[l] = k_slot_off(T) + k_lane(l) + k_imm(hc, kt, k, half);
                            F1_CHECK(a[l] == k_land(T, row, col), "K read model");
                            const int iw = issuer_of(0, hc);
                            F1_CHECK(a[l] == kv_op_lds(iw, T) + (row - OPR * hc) * K_PITCH + 2 * col, "K landing");
                            F1_CHECK(k_imm(hc, kt, k, half) < 65536, "K imm");
                        }
                        F1_CHECK(bank_ok(a), "K banks");
                    }
            for (int half = 0; half < 2; ++half)
                for (int dt = 0; dt < NDT; ++dt) {
                    for (int l = 0; l < 32; ++l) {
                        const int row = CH * hc + 16 * half + 8 * (l / 16) + l % 8, col = 16 * dt + 8 * ((l / 8) % 2);
                        a[l] = v_slot_off(T) + v_lane(l) + v_imm(hc, half, dt);
                        F1_CHECK(a[l] == v_land(T, row, col), "V read model");
                        const int iw = issuer_of(1, hc);
                        F1_CHECK(a[l] == kv_op_lds(iw, T) + (row - OPR * hc) * V_PITCH + 2 * col, "V landing");
                    }
                    F1_CHECK(bank_ok(a), "V banks (tr16 lane groups as b128 under lin16)");
                }
        }
    for (int w = 0; w < NWAVE; ++w) {
        for (int h = 0; h < NH; ++h)
            for (int k = 0; k < NDK; ++k)
                for (int half = 0; half < 2; ++half) {
                    for (int l = 0; l < 32; ++l) {
                        a[l] = q_lane(w, l) + q_imm(h, k, half);
                        F1_CHECK(a[l] == q_land(w, h, l % 16, 32 * k + 16 * half + 8 * (l / 16)), "Q read model");
                    }
                    F1_CHECK(bank_ok(a), "Q banks");
                }
        std::vector<int> wr(QO_W_B / 16, 0), rd(QO_W_B / 16, 0);
        for (int h = 0; h < NH; ++h)
            for (int dt = 0; dt < NDT; ++dt) {
                for (int l = 0; l < 32; ++l) {
                    a[l] = ow_lane(w, l) + ow_imm(h, dt);
                    F1_CHECK(a[l] == o_land(w, h, l % 16, 16 * dt + 8 * (l / 16)), "O write model");
                    F1_CHECK(a[l] >= o_lds(w) && a[l] + 16 <= o_lds(w) + QO_W_B, "O write region");
                    wr[(a[l] - o_lds(w)) / 16]++;
                }
                F1_CHECK(bank_ok(a), "O write banks");
            }
        for (int s = 0; s < 2 * QW; ++s) {
            for (int l = 0; l < 32; ++l) {
                a[l]         = or_lane(w, l) + or_imm(s);
                const int ch = 32 * (s % 2) + l;
                F1_CHECK(a[l] == o_land(w, ch / 16, s / 2, 8 * (ch % 16)), "O read model");
                rd[(a[l] - o_lds(w)) / 16]++;
            }
            F1_CHECK(bank_ok(a), "O read banks");
        }
        for (size_t i = 0; i < wr.size(); ++i) F1_CHECK(wr[i] == rd[i] && wr[i] <= 1, "O image coverage");
    }
    return 0;
}

// fdiv == integer division: every divisor 1..4096 and the powers of two / their neighbours up to 2^31 - 1,
// against numerators 0..4096, multiples of d and their neighbours, and the top of the range
inline int check_fdiv() {
    std::vector<uint32_t> ds;
    for (uint32_t d = 1; d <= 4096; ++d) ds.push_back(d);
    for (int k = 12; k <= 31; ++k)
        for (int e = -1; e <= 1; ++e) {
            const int64_t d = (int64_t(1) << k) + e;
            if (d >= 1 && d < (int64_t(1) << 31)) ds.push_back((uint32_t) d);
        }
    ds.push_back(0x7FFFFFFFu);
    ds.push_back(1000003u);
    for (uint32_t d : ds) {
        const Fdiv f = fdiv_of(d);
        F1_CHECK(f.sh >= 31 && f.sh <= 62 && (uint64_t) f.m * d >= (uint64_t(1) << f.sh) &&
                     (uint64_t) f.m * d - (uint64_t(1) << f.sh) < d && (d == 1 || (uint64_t(1) << (f.sh - 32)) < d),
                 "fdiv_of(%u) = (%u, %d)", d, f.m, f.sh);
        auto chk = [&](int64_t n) {
            if (n < 0 || n >= (int64_t(1) << 31)) return true;
            return fdiv((uint32_t) n, f) == (uint32_t) n / d;
        };
        for (int64_t n = 0; n <= 4096; ++n) F1_CHECK(chk(n), "fdiv %lld / %u", (long long) n, d);
        for (int64_t q = 1; q <= 4096; ++q)
            for (int e = -1; e <= 1; ++e) F1_CHECK(chk(q * d + e), "fdiv %lld / %u", (long long) (q * d + e), d);
        for (int64_t n = (int64_t(1) << 31) - 4096; n < (int64_t(1) << 31); ++n) F1_CHECK(chk(n), "fdiv top / %u", d);
        for (int64_t q = ((int64_t(1) << 31) - 1) / d, k = 0; k < 4 && q >= 0; --q, ++k)
            for (int e = -1; e <= 1; ++e) F1_CHECK(chk(q * d + e), "fdiv near top / %u", d);
    }
    return 0;
}

inline int check_mask() {
    for (int n = 1; n < 9; ++n)
        for (int causal = 0; causal < 2; ++causal)
            for (int w = 0; w < NWAVE; ++w)
                for (int c = 0; c < 2 * n; ++c)
                    for (int kt = 0; kt < 2; ++kt)
                        for (int l = 0; l < 32; ++l)
                            for (int i = 0; i < 8; ++i) {
                                const int  kv   = CH * c + 16 * kt + 8 * (l / 16) + i;
                                const bool def  = causal == 0 || kv <= ST * (n - 1) + QW * w + l % 16;
                                const bool code = !(last_stage_chunk(c, n) && masked(l, i, mask_off(w, c, n, kt, causal)));
                                F1_CHECK(def == code, "mask n %d c %d", n, c);
                            }
    return 0;
}

// bf16 RNE of an fp32 value (the P^T pack's rounding)
inline float bf16r(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

// fp32 emulation of one q-tile through v1's scheduled softmax, both decision forms:
//   sm1l = 0: v0 (x2 = c2 max_full; rescale iff some lane has x2 - m2 > TH2; m2n = max(m2, x2))
//   sm1l = 1: lean (rescale iff some lane's OWN-HALF max mx > (m2 + TH2) c2inv; inside the rescale the
//             cross-half max and m2n = max(m2, c2 max_full))
//   lsum = 0: fp32 per-lane row sums of p; 1: row sums of bf16(p) by WMMA (rows of all-ones A)
// O always accumulates bf16(p) v (the WMMA's operand). vs a double reference.
inline int check_softmax(int n, int causal, int w, uint32_t seed, int sm1l, int lsum) {
    const float c2 = 0.08838834764831845f * LOG2E, c2inv = (float) (1.0 / (0.08838834764831845 * 1.4426950408889634));
    const int   nkv = ST * n;
    std::vector<float> S((size_t) nkv * QW), V((size_t) nkv);
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float) (seed >> 8) / 16777216.0f;
    };
    for (int kv = 0; kv < nkv; ++kv) {
        V[kv] = 2.0f * rnd() - 1.0f;
        for (int r = 0; r < QW; ++r) S[(size_t) kv * QW + r] = 80.0f * rnd() - 40.0f + 500.0f * kv / nkv;
    }
    auto  vis = [&](int kv, int r) { return causal == 0 || kv <= ST * (n - 1) + QW * w + r; };
    float m2[32], ll[32], O[32];
    for (int l = 0; l < 32; ++l) ll[l] = O[l] = 0.0f, m2[l] = 0.0f;
    int rescales = 0;
    for (int c = 0; c < 2 * n; ++c) {
        float s[32][16], mx[32];
        for (int l = 0; l < 32; ++l) {
            mx[l] = -INFINITY;
            for (int kt = 0; kt < 2; ++kt)
                for (int i = 0; i < 8; ++i) {
                    const int  kv = CH * c + 16 * kt + 8 * (l / 16) + i;
                    const bool m  = last_stage_chunk(c, n) && masked(l, i, mask_off(w, c, n, kt, causal));
                    s[l][8 * kt + i] = m ? -INFINITY : S[(size_t) kv * QW + l % 16];
                    mx[l]            = std::fmax(mx[l], s[l][8 * kt + i]);
                }
        }
        if (c == 0) {
            for (int l = 0; l < 32; ++l) {
                m2[l] = std::fmax(mx[l], mx[l ^ 16]) * c2;
                F1_CHECK(std::isfinite(m2[l]), "chunk 0 max");
            }
        } else {
            bool bal = false;
            for (int l = 0; l < 32; ++l)
                bal = bal || (sm1l ? mx[l] > (m2[l] + TH2) * c2inv : std::fmax(mx[l], mx[l ^ 16]) * c2 - m2[l] > TH2);
            if (bal) {
                ++rescales;
                float m2n[32];
                for (int l = 0; l < 32; ++l) m2n[l] = std::fmax(m2[l], std::fmax(mx[l], mx[l ^ 16]) * c2);
                for (int l = 0; l < 32; ++l) {
                    const float corr = std::exp2(m2[l] - m2n[l]);
                    O[l] *= corr;
                    ll[l] *= corr;
                    m2[l] = m2n[l];
                }
            }
        }
        for (int l = 0; l < 32; ++l) {
            float rs = 0.0f, pv = 0.0f;
            for (int e = 0; e < 16; ++e) {
                const float p  = std::exp2(std::fma(s[l][e], c2, -m2[l]));
                F1_CHECK(p <= 4096.0f, "p %g beyond the lazy bound", p);
                const int   kv = CH * c + 16 * (e / 8) + 8 * (l / 16) + e % 8;
                rs += lsum ? bf16r(p) : p;
                pv += bf16r(p) * V[kv];
            }
            ll[l] += rs;
            O[l] += pv;
        }
    }
    F1_CHECK(n < 2 || rescales > 0, "emulation never rescaled");
    for (int r = 0; r < QW; ++r) {
        const float lt = ll[r] + ll[r + 16], o = (O[r] + O[r + 16]) / lt, lse = (m2[r] + std::log2(lt)) * LN2;
        double mref = -1e300, sum = 0.0, acc = 0.0;
        for (int kv = 0; kv < nkv; ++kv)
            if (vis(kv, r)) mref = std::fmax(mref, (double) S[(size_t) kv * QW + r]);
        for (int kv = 0; kv < nkv; ++kv)
            if (vis(kv, r)) {
                const double e = std::exp(0.08838834764831845 * ((double) S[(size_t) kv * QW + r] - mref));
                sum += e;
                acc += e * V[kv];
            }
        const double oref = acc / sum, lref = 0.08838834764831845 * mref + std::log(sum);
        F1_CHECK(std::isfinite(o) && std::isfinite(lse), "softmax finite");
        // bf16(p) in O: 2^-8 relative per weight. The bf16 row sum (lsum 1) moves LSE by at most -ln(1 - 2^-9) =
        // 1.955e-3 nat (bf16 RNE: relative error <= 2^-9 per term, whatever the row length), on top of the fp32
        // tolerance of lsum 0 (exp2 / log2 / accumulation at |S| up to 540). hk/lab/fwd1_lsum_sim.py measures
        // the typical size on lab's inputs: max 0.8-1.6e-3, mean 0.5-2.2e-4 nat
        const double lse_tol = 1e-4 * (std::fabs(lref) + 1.0) + (lsum ? 1.956e-3 : 0.0);
        F1_CHECK(std::fabs(o - oref) <= 8e-3 * (std::fabs(oref) + 1.0) && std::fabs(lse - lref) <= lse_tol,
                 "softmax n %d causal %d w %d r %d sm1l %d lsum %d: o %g / %g lse %g / %g", n, causal, w, r, sm1l,
                 lsum, o, oref, lse, lref);
    }
    return 0;
}

// ---- the reference memory / barrier program of one wave (all 4 waves run the same program) ------------
enum Ev : int { E_TDM, E_QTDM, E_TCNT, E_RD_Q, E_RD_K, E_RD_V, E_SIG, E_WAIT, E_BAR, E_EPI };
// E_TDM: K/V op index k; E_QTDM: item index (== J: the dummy after the last item); E_RD_K / E_RD_V: global
// chunk; E_RD_Q / E_EPI: item; E_SIG / E_WAIT / E_BAR: phase; E_TCNT: count
struct Event {
    int type, arg;
};
inline std::vector<Event> wave_events(const std::vector<int> &ns, int mds, int epi) {
    std::vector<Event> e;
    const int          J = (int) ns.size();
    std::vector<int>   T0(J + 1, 0);
    for (int j = 0; j < J; ++j) T0[j + 1] = T0[j] + ns[j];
    const int N = T0[J];
    e.push_back({E_QTDM, 0});
    e.push_back({E_TDM, 0});
    e.push_back({E_TDM, 1});
    e.push_back({E_TCNT, TCNT_STEADY});
    e.push_back({E_BAR, 0});
    e.push_back({E_TDM, 2});
    e.push_back({E_TCNT, TCNT_STEADY});
    e.push_back({E_SIG, 1});
    auto item_pro = [&](int j) {
        e.push_back({E_TCNT, TCNT_ITEM});
        e.push_back({E_RD_Q, j});
        e.push_back({E_RD_K, 2 * T0[j]});
        if (mds != 1) e.push_back({E_RD_K, 2 * T0[j] + 1});
        if (mds == 2 || mds == 3) e.push_back({E_RD_V, 2 * T0[j]});
    };
    item_pro(0);
    for (int j = 0; j < J; ++j) {
        const int n = ns[j];
        for (int t = 0; t < n; ++t) {
            const int  T = T0[j] + t, c = 2 * T;
            const bool last = t == n - 1;
            // even chunk
            if (mds == 1) e.push_back({E_RD_K, c + 1});
            if (mds == 0 || mds == 1) e.push_back({E_RD_V, c});
            e.push_back({E_WAIT, T + 1});
            if (last) e.push_back({E_QTDM, j + 1});
            e.push_back({E_TDM, T + 3});
            if (mds == 0 && !last) e.push_back({E_RD_K, c + 2});
            if (mds == 2) {
                if (!last) e.push_back({E_RD_K, c + 2});
                e.push_back({E_RD_V, c + 1});
            }
            if (mds == 3) { // the stage's one burst (in the kernel's order)
                if (!last) e.push_back({E_RD_K, c + 2});
                e.push_back({E_RD_V, c + 1});
                if (!last) e.push_back({E_RD_K, c + 3});
                if (!last) e.push_back({E_RD_V, c + 2});
            }
            // odd chunk
            if (mds == 1 && !last) e.push_back({E_RD_K, c + 2});
            if (mds == 0 || mds == 1) e.push_back({E_RD_V, c + 1});
            e.push_back({E_TCNT, last ? TCNT_LAST : TCNT_STEADY});
            e.push_back({E_SIG, T + 2});
            if (mds == 0 && !last) e.push_back({E_RD_K, c + 3});
            if (mds == 2 && !last) {
                e.push_back({E_RD_K, c + 3});
                e.push_back({E_RD_V, c + 2});
            }
        }
        if (epi == 1 && j + 1 < J) {
            item_pro(j + 1);
            e.push_back({E_EPI, j});
        } else {
            e.push_back({E_EPI, j});
            if (j + 1 < J) item_pro(j + 1);
        }
    }
    e.push_back({E_TCNT, 0});
    e.push_back({E_WAIT, N + 1});
    return e;
}

// replay of the reference program for one item list: identical barrier sequences, signal / wait
// alternation, <= 3 TDM ops in flight, RAW (every K / V read after the issuer's retire of that stage via a
// barrier phase; every Q read after its own Q op's retire) and WAR (every real op k >= KD after all reads
// of stage k - KD of its half; every Q op after the previous item's Q read), every op retired at exit,
// every chunk of every item read exactly once per role
inline int check_protocol(const std::vector<int> &ns, int mds, int epi) {
    const int          J = (int) ns.size();
    int                N = 0;
    for (int x : ns) N += x;
    const std::vector<Event> ev = wave_events(ns, mds, epi);
    const int          nph = N + 2;
    std::vector<int>   sig(nph, -1), wt(nph, -1);
    {
        int  expect = 0;
        bool open   = false;
        for (int i = 0; i < (int) ev.size(); ++i) {
            const Event &x = ev[i];
            if (x.type == E_BAR) {
                F1_CHECK(!open && x.arg == expect, "bar order");
                sig[x.arg] = wt[x.arg] = i;
                ++expect;
            }
            if (x.type == E_SIG) {
                F1_CHECK(!open && x.arg == expect, "sig order");
                sig[x.arg] = i;
                open       = true;
            }
            if (x.type == E_WAIT) {
                F1_CHECK(open && x.arg == expect, "wait order");
                wt[x.arg] = i;
                open      = false;
                ++expect;
            }
        }
        F1_CHECK(expect == nph && !open, "%d phases, want %d", expect, nph);
    }
    // all waves run this program: happens-before (i in wave a) -> (j in wave b) iff the first phase
    // signaled at or after i is waited before j
    std::vector<int> nsig(ev.size(), nph);
    for (int i = (int) ev.size() - 1, x = nph; i >= 0; --i) {
        nsig[i] = x;
        for (int y = 0; y < nph; ++y)
            if (sig[y] == i) x = y;
    }
    auto hb = [&](int i, int j) {
        const int x = nsig[i];
        return x < nph && wt[x] >= 0 && wt[x] < j;
    };
    // TDM retire points (in-order completion; tensorcnt n retires all but the n newest)
    struct Op {
        int kind, arg, iss, ret;
    };
    std::vector<Op> ops;
    int             retired = 0;
    for (int i = 0; i < (int) ev.size(); ++i) {
        const Event &x = ev[i];
        if (x.type == E_TDM || x.type == E_QTDM) {
            ops.push_back({x.type, x.arg, i, -1});
            F1_CHECK((int) ops.size() - retired <= 3, "> 3 TDM ops in flight");
        }
        if (x.type == E_TCNT)
            for (; retired < (int) ops.size() - x.arg; ++retired) ops[retired].ret = i;
    }
    F1_CHECK(retired == (int) ops.size(), "ops in flight at exit");
    int nkv = 0, nq = 0;
    for (auto &o : ops) (o.kind == E_TDM ? nkv : nq)++;
    F1_CHECK(nkv == N + 3 && nq == J + 1, "op counts kv %d (want %d) q %d (want %d)", nkv, N + 3, nq, J + 1);
    std::vector<int> rdk(2 * N, 0), rdv(2 * N, 0);
    for (int i = 0; i < (int) ev.size(); ++i) {
        const Event &x = ev[i];
        if (x.type == E_RD_Q) {
            int r = -1;
            for (auto &o : ops)
                if (o.kind == E_QTDM && o.arg == x.arg) r = o.ret;
            F1_CHECK(r >= 0 && r < i, "Q(%d) read before its op retired", x.arg);
        }
        if (x.type == E_QTDM && x.arg >= 1) { // WAR: after the read of the previous item's Q
            int rdp = -1;
            for (int y = 0; y < i; ++y)
                if (ev[y].type == E_RD_Q && ev[y].arg == x.arg - 1) rdp = y;
            F1_CHECK(rdp >= 0, "Q op %d before the read of Q(%d)", x.arg, x.arg - 1);
        }
        if (x.type != E_RD_K && x.type != E_RD_V) continue;
        F1_CHECK(x.arg >= 0 && x.arg < 2 * N, "read of chunk %d outside [0, %d)", x.arg, 2 * N);
        (x.type == E_RD_K ? rdk : rdv)[x.arg]++;
        const int st = x.arg / 2;
        // RAW: the op of stage st (any issuer wave: same program) retired, then a barrier to here
        int r = -1;
        for (auto &o : ops)
            if (o.kind == E_TDM && o.arg == st) r = o.ret;
        F1_CHECK(r >= 0 && hb(r, i), "RAW chunk %d (stage %d)", x.arg, st);
        // WAR: the op that next writes this slot (op st + KD) is issued after this read in every wave
        for (auto &o : ops)
            if (o.kind == E_TDM && o.arg == st + KD && o.arg < N) F1_CHECK(hb(i, o.iss), "WAR op %d vs read of chunk %d", o.arg, x.arg);
    }
    for (int c = 0; c < 2 * N; ++c) F1_CHECK(rdk[c] == 1 && rdv[c] == 1, "chunk %d read K %d V %d times", c, rdk[c], rdv[c]);
    return 0;
}

// stream_stage == the flattened stage stream of the workgroup; schedule bijection / balance / XCD heads
inline int check_sched(const Shape &sh, const Sched &sc, bool expect_balance) {
    const int        U = n_units(sc), nit = n_items(sh.B, sh.nhq4, sh.nqb);
    std::vector<int> seen((size_t) nit, 0);
    long long        ref = -1;
    F1_CHECK(sc.W >= 1 && sc.W <= U, "W %d outside [1, %d]", sc.W, U);
    for (int g = 0; g < sc.W; ++g) {
        const int J = wg_items(sc, g);
        F1_CHECK(J >= 1, "wg %d has no item", g);
        std::vector<int> flat_item, flat_s;
        long long        st = 0;
        for (int j = 0; j < J; ++j) {
            const Work wk = wg_item(sc, g, j);
            F1_CHECK(wk.b >= 0 && wk.b < sh.B && wk.hq4 >= 0 && wk.hq4 < sh.nhq4 && wk.qb >= 0 && wk.qb < sh.nqb, "decode");
            seen[((size_t) wk.b * sh.nhq4 + wk.hq4) * sh.nqb + wk.qb]++;
            if (sc.mode == MODE_ITEMS) {
                const Work ref = work_lpt(j * sc.W + wg_rank(sc, g), sh.B, sh.nhq4, sh.nqb);
                F1_CHECK(ref.b == wk.b && ref.hq4 == wk.hq4 && ref.qb == wk.qb, "items mode != work_lpt (g %d j %d)", g, j);
            }
            const int n = n_stages(wk.qb, sh.cst, sh.nkv_st, sh.causal);
            st += n;
            for (int s = 0; s < n; ++s) flat_item.push_back(j), flat_s.push_back(s);
            if (sc.mode == MODE_PAIRS && (j & 1) == 1 && j < 2 * count_below(n_pair_units(sh.B, sh.nhq4, sh.nqb), wg_rank(sc, g), sc.dW)) {
                const Work hi = wg_item(sc, g, j - 1);
                F1_CHECK(hi.b == wk.b && hi.hq4 == wk.hq4 && hi.qb == sh.nqb - 1 - wk.qb && hi.qb > wk.qb, "pair");
            }
        }
        // stream_stage / stream3 after every j1 (incl. 0) vs the flattened list
        int T0 = 0;
        for (int j1 = 0; j1 <= J; ++j1) {
            StRef s3[3];
            stream3(sh, sc, g, J, j1, s3);
            for (int k = 0; k < 3; ++k) {
                const StRef r = stream_stage(sh, sc, g, J, j1, k);
                F1_CHECK(s3[k].real == r.real && (!r.real || (s3[k].b == r.b && s3[k].hkv == r.hkv && s3[k].s == r.s)),
                         "stream3 != stream_stage g %d j1 %d k %d", g, j1, k);
                const int   T = T0 + k;
                if (T < (int) flat_item.size()) {
                    const Work wk = wg_item(sc, g, flat_item[T]);
                    F1_CHECK(r.real && r.b == wk.b && r.hkv == hkv_of(wk.hq4, sh.G) && r.s == flat_s[T], "stream_stage g %d j1 %d k %d", g, j1, k);
                } else
                    F1_CHECK(!r.real, "stream_stage past the end");
            }
            if (j1 < J) T0 += n_stages(wg_item(sc, g, j1).qb, sh.cst, sh.nkv_st, sh.causal);
        }
        if (expect_balance) {
            F1_CHECK(ref < 0 || st == ref, "pairs balance: wg %d has %lld stages, wg 0 %lld", g, st, ref);
            ref = st;
        }
    }
    for (int x : seen) F1_CHECK(x == 1, "item bijection");
    // XCD-major pairs: at step s the workgroups of one XCD process units of one head (prod-like shapes)
    if (sc.mode == MODE_PAIRS && sc.W % NXCD == 0 && (sh.nqb / 2) % (sc.W / NXCD) == 0 && sh.nqb % 2 == 0)
        for (int x = 0; x < NXCD; ++x)
            for (int j = 0; j < wg_items(sc, x); j += 2) {
                const Work a = wg_item(sc, x, j);
                for (int g = x; g < sc.W; g += NXCD) {
                    if (j >= wg_items(sc, g)) continue;
                    const Work b = wg_item(sc, g, j);
                    F1_CHECK(a.b == b.b && a.hq4 == b.hq4, "XCD %d step %d: heads differ", x, j / 2);
                }
            }
    return 0;
}

// strided layouts (the HK_STRIDES functions): with the BSHD strides they equal the default build's functions (same
// elements, same descriptor words); with the SBHD strides every element map equals the independent formula
// ((s B + b) H + h) D and is a bijection onto [0, B S H D); every descriptor holds the row stride exactly (48-bit field)
inline int check_strides() {
    const int shapes[][5] = {{1, 256, 256, 8, 2},     {1, 64, 64, 4, 1},       {2, 192, 320, 16, 2}, {3, 320, 320, 8, 2},
                             {2, 512, 512, 32, 8},    {1, 192, 192, 32, 2},    {4, 8192, 8192, 32, 8},
                             {64, 64, 64, 2040, 510}};
    for (auto &x : shapes) {
        const int  B = x[0], Sq = x[1], Skv = x[2], Hq = x[3], Hkv = x[4];
        const bool big = (int64_t) B * Sq * Hq > 1 << 20;
        const Shape sh{B, Sq, Skv, Hq, Hkv, Hq / Hkv, Sq / QB, Hq / NH, Skv / ST, (Skv - Sq) / ST, 1};
        const Lay   qB = lay_bshd(Sq, Hq), kB = lay_bshd(Skv, Hkv), qS = lay_sbhd(B, Hq), kS = lay_sbhd(B, Hkv);
        auto same_g1 = [](const G1 &a, const G1 &b) {
            for (int i = 0; i < 8; ++i)
                if (a.w[i] != b.w[i]) return false;
            return true;
        };
        auto g1_stride = [](const G1 &g) { return (int64_t) ((uint64_t) g.w[5] | ((uint64_t) (g.w[6] & 0xFFFFu) << 32)); };
        F1_CHECK(kv_sst_s(kv_ss32(kB.ss)) == kv_sst(Hkv), "kv_sst_s(BSHD) != kv_sst (B %d)", B);
        for (int d1 = 0; d1 <= OPR; ++d1) {
            F1_CHECK(same_g1(q_g1_s(qB.ss, imin(d1, QW)), q_g1(Hq, imin(d1, QW))), "q_g1_s(BSHD) != q_g1");
            for (int rv = 0; rv < 2; ++rv) F1_CHECK(same_g1(kv_g1_s(rv, kv_ss32(kB.ss), d1), kv_g1(rv, Hkv, d1)), "kv_g1_s(BSHD) != kv_g1");
        }
        for (const Lay &L : {qS, kS, qB, kB}) {
            F1_CHECK(L.ss >= D && L.ss < (int64_t(1) << 31) && L.sb >= D && L.sb < (int64_t(1) << 31), "stride range");
            F1_CHECK((int64_t) kv_ss32(L.ss) == L.ss && (int64_t) kv_sst_s(kv_ss32(L.ss)) == ST * L.ss, "32-bit K / V stride");
            F1_CHECK(g1_stride(q_g1_s(L.ss, QW)) == L.ss && g1_stride(kv_g1_s(1, kv_ss32(L.ss), OPR)) == L.ss, "stride field");
        }
        // Q / O: every (b, hq4, qb, w), stores of sampled lanes
        for (int b = 0; b < B; ++b)
            for (int hq4 = 0; hq4 < Hq / NH; ++hq4)
                for (int qb = 0; qb < Sq / QB; ++qb)
                    for (int w = 0; w < NWAVE; ++w) {
                        const Work wk{b, hq4, qb};
                        const OpC  a = q_c_s(1, wk, Sq, qB, w), c = q_c(1, wk, Sq, Hq, w), s = q_c_s(1, wk, Sq, qS, w);
                        F1_CHECK(a.el == c.el && a.d1 == c.d1 && s.d1 == c.d1, "q_c_s(BSHD) != q_c");
                        const int p0 = wave_pos0(qb, w);
                        F1_CHECK(s.el == (uint32_t) (((int64_t) p0 * B + b) * Hq + NH * hq4) * D, "q_c_s(SBHD) element");
                        for (int st = 0; st < 2 * QW; ++st)
                            for (int l : {0, 1, 15, 16, 31}) {
                                F1_CHECK(o_store_b_s(qB, b, hq4, qb, w, st, l) == o_store_b(b, Sq, Hq, hq4, qb, w, st, l), "o_store_b_s(BSHD)");
                                const int64_t ch = 32 * (st % 2) + l, want = 2 * ((((int64_t) p0 + st / 2) * B + b) * Hq + NH * hq4 + ch / 16) * D + 16 * (ch % 16);
                                F1_CHECK(o_store_b_s(qS, b, hq4, qb, w, st, l) == want, "o_store_b_s(SBHD)");
                                if (big && l == 0) break;
                            }
                    }
        // K / V: every (b, hkv, stage, w)
        for (int b = 0; b < B; ++b)
            for (int hkv = 0; hkv < Hkv; ++hkv)
                for (int s = 0; s < Skv / ST; ++s)
                    for (int w = 0; w < NWAVE; ++w) {
                        const StRef r{1, b, hkv, s};
                        const OpC   a = op_c_s(sh, kB, r, w), c = op_c(sh, r, w), t = op_c_s(sh, kS, r, w);
                        F1_CHECK(a.el == c.el && a.d1 == c.d1 && t.d1 == c.d1, "op_c_s(BSHD) != op_c");
                        F1_CHECK(t.el == (uint32_t) ((((int64_t) kv_op_row0(w, s)) * B + b) * Hkv + hkv) * D, "op_c_s(SBHD) element");
                        if (s == 0) {
                            F1_CHECK(kv_el0_s(kB, b, hkv, w) == kv_el0(b, Skv, Hkv, hkv, w), "kv_el0_s(BSHD)");
                            for (const Lay &L : {kB, kS})
                                for (int s2 = 0; s2 < Skv / ST; ++s2)
                                    F1_CHECK(own_c(Skv, kv_el0_s(L, b, hkv, w), kv_sst_s(kv_ss32(L.ss)), w, s2).el == op_c_s(sh, L, StRef{1, b, hkv, s2}, w).el,
                                             "own_c over the strided el0 / sst != op_c_s");
                        }
                    }
        // bijection onto [0, B S H D) (rows of D elements), both layouts, small shapes
        if (!big)
            for (int t = 0; t < 2; ++t) {
                const int        S = t ? Skv : Sq, H = t ? Hkv : Hq;
                const Lay        Ls[2] = {lay_bshd(S, H), lay_sbhd(B, H)};
                for (const Lay &L : Ls) {
                    std::vector<int> hit((size_t) B * S * H, 0);
                    for (int b = 0; b < B; ++b)
                        for (int s = 0; s < S; ++s)
                            for (int h = 0; h < H; ++h) {
                                const int64_t e = el_s(L, b, s, h);
                                F1_CHECK(e >= 0 && e % D == 0 && e / D < (int64_t) hit.size(), "element outside the tensor");
                                hit[(size_t) (e / D)]++;
                            }
                    for (int v : hit) F1_CHECK(v == 1, "layout not a bijection");
                }
            }
    }
    return 0;
}

inline int selftest(bool verbose = false) {
    if (verbose) {
        const G1 q = q_g1(32, QW), kv = kv_g1(0, 8, OPR), vv = kv_g1(1, 8, OPR), dm = dummy_g1(0);
        const G1 *gs[4] = {&q, &kv, &vv, &dm};
        const char *nm[4] = {"Q op", "K op", "V op", "dummy"};
        for (int j = 0; j < 4; ++j) {
            std::printf("  g1 %-6s", nm[j]);
            for (int x = 0; x < 8; ++x) std::printf(" %08x", gs[j]->w[x]);
            std::printf("\n");
        }
    }
    F1_CHECK(kv_op_dim1(100, 1, 1) == 4 && kv_op_dim1(100, 0, 2) == 0 && kv_op_dim1(1 << 20, 1, 3) == 32 &&
                 q_op_dim1(72, 0, 3) == 16 && q_op_dim1(72, 1, 0) == 8,
             "clamp units");
    for (int G : {4, 8, 16})
        for (int hq4 = 0; hq4 < 4096; ++hq4) F1_CHECK(hkv_of(hq4, G) == (NH * hq4) / G, "hkv_of(%d, %d) shift != quotient", hq4, G);
    if (check_fdiv() || check_lds() || check_mask() || check_strides()) return 1;
    // protocol: every item list of length 1..4 with n in 1..5, plus long pair-like lists, every variant
    std::vector<std::vector<int>> lists;
    for (int len = 1; len <= 4; ++len) {
        int tot = 1;
        for (int i = 0; i < len; ++i) tot *= 5;
        for (int code = 0; code < tot; ++code) {
            std::vector<int> ns;
            for (int i = 0, x = code; i < len; ++i, x /= 5) ns.push_back(1 + x % 5);
            lists.push_back(ns);
        }
    }
    for (int npq = 1; npq <= 9; ++npq) {
        std::vector<int> ns;
        for (int pr = 0; pr < npq; ++pr) ns.push_back(2 * npq - pr), ns.push_back(pr + 1);
        lists.push_back(ns);
    }
    lists.push_back(std::vector<int>(1, 140));
    for (int mds = 0; mds <= 3; ++mds)
        for (int epi = 0; epi <= 1; ++epi)
            for (auto &ns : lists)
                if (check_protocol(ns, mds, epi)) {
                    std::printf("  list");
                    for (int x : ns) std::printf(" %d", x);
                    std::printf(" mds %d epi %d\n", mds, epi);
                    return 1;
                }
    for (int n = 1; n <= 6; ++n)
        for (int causal = 0; causal < 2; ++causal)
            for (int w = 0; w < NWAVE; ++w)
                for (int v = 0; v < 4; ++v)
                    if (check_softmax(n, causal, w, 12345u + 7u * n + 3u * w + causal, v & 1, v >> 1)) return 1;
    // schedules: proof shapes x modes x W
    const int shapes[][6] = {{1, 64, 4, 1, 64, 1},   {1, 128, 8, 2, 128, 1}, {2, 256, 16, 2, 256, 1},
                             {1, 192, 16, 1, 192, 1}, {1, 128, 4, 1, 256, 1}, {2, 192, 8, 1, 320, 1},
                             {1, 256, 8, 2, 384, 0},  {1, 256, 8, 2, 128, 0}, {2, 512, 32, 8, 512, 1},
                             {3, 320, 8, 2, 320, 1},  {4, 8192, 32, 8, 8192, 1}, {2, 192, 16, 2, 320, 1},
                             {1, 192, 32, 2, 192, 1}};
    for (auto &x : shapes) {
        Shape sh{x[0], x[1], x[4], x[2], x[3], x[2] / x[3], x[1] / QB, x[2] / NH, x[4] / ST,
                 x[5] ? (x[4] - x[1]) / ST : 0, x[5]};
        const int nit = n_items(sh.B, sh.nhq4, sh.nqb);
        // items mode: W = #items (v0's grid) and smaller W (several items per workgroup, longest first)
        for (int W : {nit, (nit + 1) / 2, 3, 1})
            if (W >= 1 && W <= nit && check_sched(sh, make_sched(MODE_ITEMS, W, sh.B, sh.nhq4, sh.nqb), false)) return 1;
        const int U = n_units(MODE_PAIRS, sh.B, sh.nhq4, sh.nqb);
        const int Ws[] = {1, 2, 3, 5, 7, 8, 16, 24, 32, 64, 128, 256, U};
        for (int W : Ws)
            if (W <= U) {
                const Sched sc = make_sched(MODE_PAIRS, W, sh.B, sh.nhq4, sh.nqb);
                const bool  bal = sh.nqb % 2 == 0 && U % W == 0;
                if (check_sched(sh, sc, bal)) {
                    std::printf("  shape B%d Sq%d W %d\n", sh.B, sh.Sq, W);
                    return 1;
                }
            }
    }
    std::printf("f1 selftest OK (mds %d epi %d)\n", FWD1_MDS, FWD1_EPI);
    return 0;
}

} // namespace f1
#endif // F1_IDX_SELFTEST
