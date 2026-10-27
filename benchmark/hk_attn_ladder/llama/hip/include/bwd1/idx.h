/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// bwd1/idx.h -- index / schedule constants of the gfx1250 d128 GQA backward v1 (hip/src/k_attn_bwd_v1.hip;
// design hk/notes/bwd_v1_design.md, implementation note hk/notes/bwd_v1_impl.md). v1 keeps every address
// formula, LDS map, TDM role, dQ_acc layout, prep / convert map and the host gate of v0 (bwd/idx.h, namespace
// b0) and adds only what its re-timed step changes (namespace b1):
//   * head rotation: kv block blk starts its head cycle at gh = blk % G (BWD1_HROT), so the WGs of one XCC
//     (same b, hkv) add into different dQ_acc blocks at any time;
//   * the step program: TDM op(s + 3) at the top of step s, dSa(s - 1) and the E-operand tr16 before the dS^T
//     stores (covered), readback(s + 1) split around the signal (NUNC uncovered loads before it), dS buffer 1
//     zeroed in the prologue (dQ(-1) = 0 read from LDS);
//   * the compile-time schedule knobs and the counts derived from them (the bounds checker
//     hk/lab/bounds/bwd_v1_check.cpp is compiled with the same -D flags and replays exactly these).
// Host-compilable (g++ -std=c++20); no HIP include.
#pragma once
#include "bwd/idx.h"
#if defined(BWD1_OPT_TDM3) && BWD1_OPT_TDM3
#include "bwd/opt_idx.h" // b0opt::DUMMY_OFF / dummy_src_byte (the v0 TDM3 dummy cell, unused by the v1 LDS map)
#endif

// ---- compile-time knobs (design 5.) ------------------------------------------------------------------------------
#ifndef BWD1_SCHED   // 0: memory as bursts at the protocol positions; 1: interleaved under the WMMA phases
#define BWD1_SCHED 1
#endif
#ifndef BWD1_ATOM    // dQ atomics: 0 burst after the dS^T stores; 1 interleaved under E1; 2 under E1 + E2
#define BWD1_ATOM 1
#endif
#ifndef BWD1_NWIN    // dV / dK WMMAs (of 32) inside the barrier window (after the signal)
#define BWD1_NWIN 16
#endif
#ifndef BWD1_HROT    // head rotation by kv block
#define BWD1_HROT 1
#endif
#ifndef BWD1_SKIP_DEAD  // 1: DEAD waves of a diagonal step skip S/dP / softmax / dV / dK (0: debug, bitwise-equal dk / dv)
#define BWD1_SKIP_DEAD 1
#endif
// ---- v1.1 optimisation knobs (hk/notes/bwd_v11.md; every one 0 = the validated v1 code) ---------------------------
#ifndef BWD1_OPT_MASKED  // diagonal steps on the FULL step's interleaved schedule: 1 every wave kind computes it;
#define BWD1_OPT_MASKED 0 //   2 FULL / DIAG waves do, DEAD waves keep v1's burst body (wave-uniform kind branch)
#endif
#ifndef BWD1_OPT_TDM3    // 1: wave 3 issues a real 16-B dummy TDM op per step (b0opt::DUMMY_OFF): no role branch
#define BWD1_OPT_TDM3 0
#endif
#ifndef BWD1_OPT_REGS    // bit mask of llvm.fake.use register holds (kernel-scope steps), see v1_step.inc:
#define BWD1_OPT_REGS 0  //   1 X.qA / X.dA to the end of G1, 2 dSa to the end of C, 4 E operands to the end of E2,
#endif                   //   8 dQ(s - 1) atomic data + voffset through G0 of step s + 1, 16 E2 operands likewise

namespace b1 {

using namespace b0;

constexpr int SCHED = BWD1_SCHED, ATOM = BWD1_ATOM, NWIN = BWD1_NWIN, HROT = BWD1_HROT;
static_assert(SCHED == 0 || SCHED == 1, "BWD1_SCHED");
static_assert(ATOM >= 0 && ATOM <= 2, "BWD1_ATOM");
static_assert(NWIN >= 8 && NWIN <= 16 && NWIN % 2 == 0, "BWD1_NWIN: 8..16, even (QT loads in the first 8 E1 gaps)");
static_assert(HROT == 0 || HROT == 1, "BWD1_HROT");
constexpr int OPT_MASKED = BWD1_OPT_MASKED, OPT_TDM3 = BWD1_OPT_TDM3, OPT_REGS = BWD1_OPT_REGS;
static_assert(OPT_MASKED >= 0 && OPT_MASKED <= 2, "BWD1_OPT_MASKED");
static_assert(OPT_MASKED != 2 || BWD1_SKIP_DEAD, "BWD1_OPT_MASKED=2 is the DEAD-skipping split");
static_assert(OPT_TDM3 == 0 || OPT_TDM3 == 1, "BWD1_OPT_TDM3");
static_assert(OPT_REGS >= 0 && OPT_REGS <= 31, "BWD1_OPT_REGS: bit mask 0..31");
static_assert(!OPT_MASKED || SCHED == 1, "BWD1_OPT_MASKED needs the interleaved schedule (BWD1_SCHED=1)");
// BWD1_OPT_MASKED: VALU-class ops the causal mask adds to one softmax phase (2 blocks x (8 v_cmp + 8 v_cndmask +
// threshold)); only sizes the sched_group_barrier VALU groups of a diagonal step's G1 / C
constexpr int MASK_VALU = 34;
// checker replay of a wave's step: live = computes S/dP / softmax / dV / dK and issues the E tr16 (BWD1_OPT_MASKED=1:
// every kind); il = runs the interleaved body (readback items before the signal = RB1, else 0)
B0_HD bool wave_live(bool masked, int kind) {
  return OPT_MASKED == 1 || !masked || !BWD1_SKIP_DEAD || kind != b0::K_DEAD;
}
B0_HD bool wave_il(bool masked, int kind) {
  return SCHED == 1 && (!masked || (OPT_MASKED == 1) || (OPT_MASKED == 2 && kind != b0::K_DEAD));
}

// ring / exchange depths and the TDM lead of v1 (design 2.): op(k) issued at the top of step k - 3, retired
// at SIGNAL B_{k-2} (tensorcnt 1), read from step k - 1 on; tile s - 1's slot is free at the top of step s
constexpr int RING = 4, NDS = 2, LEAD = 3, TCNT = 1;
static_assert(RING <= RING_MAX && NDS <= NDS_MAX, "LDS map reserves the depths");

// E = dV^T (16 WMMA) then dK^T (16 WMMA); E1 = the first NE1 before the signal, E2 = the last NWIN
constexpr int NE = 2 * NDO * 2, NE1 = NE - NWIN;

// readback(s + 1) as 24 items in consumption order: items 0..15 = the S / dP A operands (dt-major: qA hh 0,
// qA hh 1, dA hh 0, dA hh 1), 2 ds_load_b128 each; items 16..23 = the LD records ld[hh][k], 1 ds_load_b128 each
constexpr int RB_ITEMS = 24, RB_LOADS = 40;
B0_HD int rb_loads(int items) { return 2 * (items < 16 ? items : 16) + (items > 16 ? items - 16 : 0); }
// SCHED 1: items before the signal (in E1) = what E2 cannot take at <= 3 loads per WMMA gap (none at NWIN 16);
// SCHED 0: none (one burst after the signal)
constexpr int RB_E2_LOADS = SCHED ? (3 * NWIN < RB_LOADS ? 3 * NWIN : RB_LOADS) : RB_LOADS;
B0_HD int rb_split_items() {
  for (int it = 0; it <= RB_ITEMS; ++it)
    if (RB_LOADS - rb_loads(it) <= RB_E2_LOADS) return it;
  return RB_ITEMS;
}
constexpr int RB1 = rb_split_items();          // items 0..RB1-1 before the signal (uncovered)
constexpr int NUNC = rb_loads(RB1);            // DS loads in flight past a FULL step's signal (s_wait_dscnt)
static_assert(NUNC >= 0 && NUNC < 64, "dscnt is a 6-bit field");
// item decode: kind 0 qA, 1 dA (hh, dt), 2 ld (hh, k)
B0_HD int rb_kind(int it) { return it < 16 ? (it % 4) / 2 : 2; }
B0_HD int rb_hh(int it) { return it < 16 ? it % 2 : (it - 16) / 4; }
B0_HD int rb_dt(int it) { return it < 16 ? it / 4 : -1; }
B0_HD int rb_k(int it) { return it < 16 ? -1 : (it - 16) % 4; }

// dQ atomics per E phase (32 per wave-step)
constexpr int NATOM = 32;
constexpr int ATOM_E1 = ATOM == 0 ? 0 : ATOM == 1 ? NATOM : (NATOM * NE1) / NE;
constexpr int ATOM_E2 = ATOM == 0 ? 0 : NATOM - ATOM_E1;

// even split of X items over N groups: items of group g
B0_HD int share(int x, int n, int g) { return (x * (g + 1)) / n - (x * g) / n; }

// ---- head rotation: step s of kv block blk -> head gh (steps heads-inner, pairs descending as v0) -----------
B0_HD int rot_of(int blk, int G) { return HROT ? blk % G : 0; }
B0_HD int tile_gh(int s, int n, int G, int rot) { return (b0::tile_gh(s, n, G) + rot) % G; }
// the dQ atomics issued in step s belong to tile s - 1 (s = n: the drain); s = 0 adds dQ(-1) = 0 into tile 0
B0_HD int atom_qt(int s, int n, int G, int nqp) { return b0::tile_qt(s - 1, n, G, nqp); }
B0_HD int atom_gh(int s, int n, int G, int rot) { return tile_gh(s - 1, n, G, rot); }

// ---- HK_STRIDES: per-tensor (batch, seq) element strides (hk/notes/bwd_v1_strides.md) --------------------------------
// -DHK_STRIDES=1 builds address every BSHD-family global tensor of the three kernels (prep: dO, O; main: Q, K, V, dO,
// dK, dV; convert: dQ) as element (b, s, h, d) at b sb + s ss + h D + d, with (sb, ss) = two int64 kernel arguments per
// tensor appended after the v1 arguments (order: k_attn_bwd_v1.hip, bwd_v1.py STRIDE_TENSORS). The head stride stays D
// and the d stride 1. The host admits exactly two layouts per tensor, each tensor independently (strides_ok):
//   BSHD contiguous      (sb, ss) = (S H D, H D)
//   SBHD-storage view    (sb, ss) = (H D, B H D)   (a [b, s, h, d] permute of a contiguous [s, b, h, d] tensor)
// Both are dense: the addressed elements are exactly [0, B S H D) of the allocation, so the whole-tensor resources
// (prep dO / O, convert dQ) keep their extent B S H D * 2 and every per-WG window (TDM tiles, the dK / dV slabs) lies
// inside the tensor (bounds proof: hk/lab/bounds/bwd_v1_check.cpp, every layout combination). LD, dQ_acc and LSE are
// workspaces / [B, Hq, Sq] and keep their layouts. Default (0): the validated v1 kernels, byte-identical (every
// use below sits under #if HK_STRIDES in the kernel source; these functions only exist for it and the checker).
#ifndef HK_STRIDES
#define HK_STRIDES 0
#endif
constexpr int STRIDES = HK_STRIDES;
static_assert(STRIDES == 0 || STRIDES == 1, "HK_STRIDES: 0 or 1");

enum { LAYOUT_BSHD = 0, LAYOUT_SBHD = 1, LAYOUT_NONE = -1 };
B0_HD int64_t bshd_sb(int64_t S, int64_t H) { return S * H * D; }
B0_HD int64_t bshd_ss(int64_t H) { return H * D; }
B0_HD int64_t sbhd_sb(int64_t H) { return H * D; }
B0_HD int64_t sbhd_ss(int64_t B, int64_t H) { return B * H * D; }
// layout of the strides (sb, ss) of a [B, S, H, D] tensor (BSHD first: with B == 1 both describe the same elements)
B0_HD int layout_of(int64_t B, int64_t S, int64_t H, int64_t sb, int64_t ss) {
  if (sb == bshd_sb(S, H) && ss == bshd_ss(H)) return LAYOUT_BSHD;
  if (sb == sbhd_sb(H) && ss == sbhd_ss(B, H)) return LAYOUT_SBHD;
  return LAYOUT_NONE;
}
// TDM group 1 holds the row stride in 48 bits of elements (hk_attn/tdm.h tdm_g1); the per-lane offsets of prep /
// convert / the epilogue use the byte row stride as int32
constexpr int64_t TDM_STRIDE_MAX = ((int64_t) 1 << 48) - 1;
constexpr int64_t I32_LIM = (int64_t) 1 << 31;
B0_HD bool strides_ok(int64_t B, int64_t S, int64_t H, int64_t sb, int64_t ss) {
  return B >= 1 && S >= 1 && H >= 1 && layout_of(B, S, H, sb, ss) != LAYOUT_NONE && ss <= TDM_STRIDE_MAX &&
         2 * ss < I32_LIM && B * S * H * ROW_B < I32_LIM;
}
// dK / dV (the per-WG slab resources and their int32 voffsets): 127 seq strides + one row inside int32
B0_HD bool slab_ok(int64_t ss) { return (int64_t) (BKV - 1) * ss * 2 + ROW_B < I32_LIM; }
// gate code of a strided launch (after b0::host_gate passed): every tensor's strides admitted
constexpr int GATE_LAYOUT = b0::GATE_BYTES + 1;

// byte offset of row (b, s, h) (element d = 0)
B0_HD int64_t row_byte_s(int64_t sb, int64_t ss, int b, int s, int h) {
  return ((int64_t) b * sb + (int64_t) s * ss + (int64_t) h * D) * 2;
}
// dO / Q tile: 32 rows q0.. of head qh, row stride ss (TDM tensor_dim1 = 32 rows exactly as b0)
B0_HD int64_t qdo_tile_byte_s(int64_t sb, int64_t ss, int b, int qh, int q0) { return row_byte_s(sb, ss, b, q0, qh); }
// K / V image (128 rows kvw0.. of kv head hkv) and the dK / dV slab base
B0_HD int64_t kv_tile_byte_s(int64_t sb, int64_t ss, int b, int hkv, int kvw0) { return row_byte_s(sb, ss, b, kvw0, hkv); }
// TDM roles (b0::role_tile_byte with the role tensor's strides; LD records unchanged); qstep / hstep derived from it
// exactly as b0 does, so the kernel's affine sum base + qt qstep + gh hstep is replayed unchanged
B0_HD int64_t role_tile_byte_s(int role, int64_t sb, int64_t ss, int b, int Sq, int Hq, int hkv, int G, int qt, int gh) {
  return role == ISS_LD ? ld_rec_byte(b, Hq, qh_of(hkv, G, gh), Sq, QS * qt)
                        : qdo_tile_byte_s(sb, ss, b, qh_of(hkv, G, gh), QS * qt);
}
B0_HD int64_t role_qstep_s(int role, int64_t ss, int Sq, int Hq) {
  return role_tile_byte_s(role, 0, ss, 0, Sq, Hq, 0, 1, 1, 0) - role_tile_byte_s(role, 0, ss, 0, Sq, Hq, 0, 1, 0, 0);
}
B0_HD int64_t role_hstep_s(int role, int64_t ss, int Sq, int Hq) {
  return role_tile_byte_s(role, 0, ss, 0, Sq, Hq, 0, 1, 0, 1) - role_tile_byte_s(role, 0, ss, 0, Sq, Hq, 0, 1, 0, 0);
}
B0_HD int64_t role_stride_el_s(int role, int64_t ss) { return role == ISS_LD ? 64 : ss; }
// BWD1_OPT_TDM3's dummy source: the first 16 B of K row kvw0 (b0opt::dummy_src_byte with K's strides)
B0_HD int64_t dummy_src_byte_s(int64_t k_sb, int64_t k_ss, int b, int hkv, int kvw0) {
  return kv_tile_byte_s(k_sb, k_ss, b, hkv, kvw0);
}
// epilogue: the WG's slab of rows kvw0 .. kvw0 + 127 of head hkv (TRUE extent of its writes: 127 seq strides + one row)
// and the lane voffsets with the int32 byte row stride ss_b = 2 ss (slab_ok)
B0_HD int64_t epi_bytes_s(int64_t ss) { return (int64_t) (BKV - 1) * ss * 2 + ROW_B; }
B0_HD int epi_voff_s(int lane, int w, int kh, int dt, int ss_b) {
  return (KVW * w + 16 * kh + lane % 16) * ss_b + (16 * dt + 8 * (lane / 16)) * 2;
}
B0_HD int epi_voff_kh_s(int lane, int w, int kh, int ss_b) { return epi_voff_s(lane, w, kh, 0, ss_b); }
// prep rows (dO, O: each its own strides) and convert rows (dQ)
B0_HD int64_t prep_row_byte_s(int64_t sb, int64_t ss, int b, int h, int q) { return row_byte_s(sb, ss, b, q, h); }
B0_HD int64_t cvt_dst_byte_s(int64_t sb, int64_t ss, int b, int qh, int qt, int row, int chunk) {
  return row_byte_s(sb, ss, b, QS * qt + row, qh) + 16 * chunk;
}

}  // namespace b1
