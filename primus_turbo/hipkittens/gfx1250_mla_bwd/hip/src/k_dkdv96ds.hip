/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_dkdv96ds (bwd_hk_k3ds): bwd_hk_k3's k_dkdv96
// PLUS the S1 write side of bwd_hk_k1ds: every wave-iteration exports the bf16 dS^T it feeds the dK WMMA (a_dsk)
// into the scratch DS in dslayout.h v1. The dK/dV computation is k3's, instruction for instruction in operands and
// per-accumulator WMMA order, so dk / dv stay bitwise equal to bwd_hk_k3's (and bwd_r10_c's); the exported bytes are
// k1ds's (same S/dP chains in the same lanes, same layout).
//
// Export:
//   * regroup (k1ds's DPP recipe): a_dsk[kh] lane (row r, half h) holds 16-B chunks h and h + 2 of kv row 16 kh + r;
//     per dword k < 4, X[k] = DPP row_ror:8 (a[4+k] -> lanes r >= 8, bank mask 0xC; old a[k]), Y[k] = DPP row_ror:8
//     (a[k] -> lanes r < 8, bank mask 0x3; old a[4+k]); each buffer_store_b128 is 512 contiguous bytes = 4 whole
//     128-B lines; 2 stores per kv16 block.
//   * descriptor per wave: 64-bit base = scratch + region bh0 = bat*Hq + hkv*G, row kt0 = kv0 / 16 (int64 math,
//     k3ds_idx.h), num_records = the wave's TRUE extent (its NKV kv16 rows over the G q heads; route dsw0: ds_mode 0
//     -> num_records 0, every store dropped at the address stage); soffset = gh * hstride + 1024 qt + rowc(kh)
//     (SGPR), voffset per lane, loop-invariant (k1ds_voff; +512 for Y).
//   * placement, 96-row body (NKV 3): full loop -> this iteration's 3 blocks right after the dO^T tr16 burst, at the
//     head of the DS-free dV WMMA run (24 WMMAs before the Q^T burst); no payload crosses the back edge (k3 holds 998
//     VGPRs; k1ds's deferred payload would add 24 at the S/dP peak). Masked loop -> at the end of the body, a block
//     only if qt >= qp_first(its kv16 row).
//   * placement, 64-row tail body (NKV 2, k1's body): k1ds's -- the payload of iteration ii stored in iteration ii+1's
//     S/dP window (no DS op there), the first full iteration peeled, the last payload after the loop; masked loop at
//     the end of the body.
//
// k_dkdv96 (bwd_hk_k3): bwd_hk_k1's k_dkdv64 with 48 kv rows per wave (NKV 3): 96 kv rows per 2-wave
// workgroup on the SAME 3-stage Q/dO TDM ring, so every staged 32-row Q/dO tile feeds 1.5x the WMMAs (TDM bytes,
// LDS writes, barriers, NL/ND loads and the per-workgroup fixed cost per FLOP all drop to 2/3).
//
//   grid (HG, k3_nblk(Skv), B*Hkv/HG), block 64 = 2 waves (one per SIMD, 1 wave/SIMD). Blocks [0, n96) are 96-row
//   blocks (wave w owns kv rows 96*bid + 48*w, NKV 3, the k3 body below); the n64 = 0..2 tail blocks after them
//   are 64-row blocks running bwd_hk_k1's body (NKV 2) behind a uniform branch on blockIdx.y (built with
//   -mllvm -structurizecfg-skip-uniform-regions so the two bodies stay separate regions). Ascending blocks =
//   longest first under bottom-right causal; the tail runs last.
//   per iteration and wave (NKV 3): S = Q K^T 2 hh x 3 kh x 6 = 36, dP = dO V^T 2 x 3 x 4 = 24, dV^T += dO^T P
//   3 x 8 = 24, dK^T += Q^T dS 3 x 12 = 36: 120 WMMA for the same 40 b128 + 40 tr16 LDS reads as k1's 80.
//
// Numerics: the k1 operands in the k1 lanes; every S/dP accumulator chains over dt ascending, every dV/dK
// accumulator gets exactly one WMMA per (query pair, q head) in the same ascending order as bwd_r10_c's (a kv
// row's extra fully-masked pairs below the diagonal add exact zeros: P = exp2(-3e38) = +0, dS = +0 * x), so dk /
// dv are meant to be bitwise equal to bwd_r10_c's.
//
// Register plan of the k3 body (k1 864 + 256 for the third sub-tile: accumulators 480, K/V fragments 240):
//   (1) the Q/dO readback is carried per hh: only hh0's 20 b128 (80 VGPR) cross the dV/dK phase (read back after
//       BARRIER(ii) as in k1); hh1's 20 are read from the current stage at the top of the next iteration, as one
//       burst under hh0's S/dP WMMAs (the stage is not overwritten before the tile's own BARRIER).
//   (2) the dV/dK B operands come in two tr16 bursts: dO^T (16 tr16, 64 VGPR) -> the 24 dV WMMAs -> Q^T (24 tr16,
//       96 VGPR) into the dead b_do registers -> the 36 dK WMMAs (k1 held both, 160 VGPR).
//   The ring protocol is k1's: the ring barrier sits after the Q^T burst and dK kh0, so every DS read of stage
//   cur precedes BARRIER(ii) and the prefetch TDM into cur follows it.
//
// Ring protocol (k1's = FlyDSL's full loop; X3: the masked iterations ride the same ring; the CPU replay
// scratch/hk_opt/x3_dkdv_mask/tools/ring_replay.py checks both bodies): both waves run one program, the wave index
// only offsets addresses, so per wave #barriers = 1 + G*nmaskp + n_full + 1.
//   one tile sequence T_0 .. T_{N-1} (masked tiles first), tile j in stage sigma(j) = (j - G*nmaskp) mod 3.
//   prologue: TDM T_0, T_1 -> sigma(0), sigma(1) -> tensorcnt 3 -> BARRIER -> readback (k3: hh0) of sigma(0)
//     -> k1: TDM T_2 -> sigma(2) | k3: T_2 -> sigma(2) becomes the pending prefetch.
//   masked iteration j: the full iteration below with the causal mask on, stage sigma(j), tiles min(j+k, N-1),
//     dS^T export (put blocks only) at its end.
//   k1 full iteration ii (stage cur = ii%3): tr16 of cur; WMMAs; tensorcnt 3 (own half of (ii+1)%3); BARRIER(ii);
//     readback of (ii+1)%3; TDM of tile min(ii+3, n-1) into cur; WMMAs.
//   k3 full iteration ii: readback hh1 of cur; the pending prefetch TDM (tile min(ii+2, n-1) -> stage (ii+2)%3, whose
//     last reads preceded BARRIER(ii-1)); tr16 dO^T, tr16 Q^T of cur; prep of tile min(ii+3, n-1) -> cur (pending,
//     issued at the top of ii+1; never after the last iteration); tensorcnt 3; BARRIER(ii); readback hh0 of (ii+1)%3.
//     Moving the issue behind the hh1 readback keeps every barrier region in the order readback -> TDM -> tr16
//     (isa_static S3 R4); tile ii+3 is still retired by iteration ii+2's tensorcnt 3 before BARRIER(ii+2).
//   exit: tensorcnt 0 -> BARRIER -> epilogue images overwrite the ring.
#include "hk1.h"
#include "hk1ds.h"
#include "idx.h"
#include "dslayout.h"
#include "k1ds_idx.h"
#include "k3ds_idx.h"
#include "k3m_idx.h"

using namespace hk1;

static_assert(DSL_VERSION == 1 && DSL_SCALED == 1, "k_dkdv96ds writes dslayout v1 with the scale inside dS");

namespace {
constexpr int NDT_QK = K1_D_QK / 32;   // 6: WMMA k-steps of S = Q K^T
constexpr int NDT_V = K1_D_V / 32;     // 4: WMMA k-steps of dP = dO V^T
constexpr int NDO_V = K1_D_V / 16;     // 8: dV^T output tiles across d
constexpr int NDO_QK = K1_D_QK / 16;   // 12: dK^T output tiles across d
constexpr int NQE = 2 * NDT_QK;        // 12 Q readback entries per hh
constexpr int NHB = NQE + 2 * NDT_V;   // 20 entries per hh
constexpr int NRB = 2 * NHB;           // 40 readback entries of a stage
constexpr int NLD = 8;                 // carried NL/ND vec4 entries
constexpr int KH0_SALU = 2;            // SALU per WMMA of the TDM-prep run (FlyDSL KH0_SALU)
// sched_barrier masks (llvm.amdgcn.sched.barrier): what may cross the fence
constexpr int SB_ALU = 0x002 | 0x004 | 0x400;   // VALU, SALU, transcendental
}  // namespace

extern "C" __global__ __launch_bounds__(64) void k_dkdv96ds(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ DO, const float* __restrict__ NL, const float* __restrict__ ND,
    bf16* __restrict__ DV, bf16* __restrict__ DK, bf16* __restrict__ DS,
    float scale, int Sq, int Skv, int Hq, int Hkv, int G, int nqt, int cshift, int causal, int B, int HG,
    int ds_mode) {
  __shared__ __attribute__((aligned(1024))) unsigned char smem[K1_LDS];
  lock_simd();                           // the champion's first statement (rocdl.disable_xdl_arb_stall)
  const uint32_t lds0 = lds_addr(smem);
  const int tid = (int)threadIdx.x;
  const int lane = tid % 32;
  const int wv = wave_id(tid);
  const int half = lane / 16;
  const int row = lane % 16;
  const int bat = k1_bat((int)blockIdx.z, (int)blockIdx.x, HG, Hkv);
  const int hkv = k1_hkv((int)blockIdx.z, (int)blockIdx.x, HG, Hkv);
  const int bid = (int)blockIdx.y;
  const int n96 = k3_n96(Skv);

  // TRUE byte extents on every buffer descriptor (FlyDSL _bv): an over-read returns 0, never the next page
  const int nk_b = k1_nk_b(B, Skv, Hkv), nv_b = k1_nv_b(B, Skv, Hkv), nl_b = k1_nl_b(B, Hq, Sq);
  const auto r_k = rsrc(K, nk_b), r_v = rsrc(V, nv_b), r_nl = rsrc(NL, nl_b), r_nd = rsrc(ND, nl_b);
  const int kbase = k1_kbase(bat, Skv, Hkv, hkv), vbase = k1_vbase(bat, Skv, Hkv, hkv);
  const float c1 = scale * K1_LOG2E;     // softmax exp2 multiplier
  const int nqt2 = nqt / 2;              // query tiles come in pairs of 16 rows

  // ---- prebuilt TDM descriptors (k1): group 1 per op (loop-invariant), per-op LDS / global bases ----------
  i32x8 g1[3];
  uint32_t ldsb[3];
  uint64_t gb[3];
  int64_t gstep_q[3], gstep_g[3];
  static_for<3>([&](auto op) {
    const int w = k1_tdm_w(op);
    const int64_t st = (int64_t)k1_tdm_stride(Hq, op);
    const int ext = k1_tdm_ext();
    g1[op] = i32x8{(int)k1_tdm_w0(op), (int)0xFFFF0000u, (int)(((uint32_t)ext & 0xFFFFu) << 16 | 0x7FFFu),
                   (int)(((uint32_t)ext >> 16) & 0xFFFFu | ((uint32_t)w << 16)), K1_TDM_ROWS,
                   (int)(uint32_t)st, (int)((uint32_t)((uint64_t)st >> 32) & 0xFFFFu), 0};
    ldsb[op] = lds0 + (uint32_t)k1_tdm_lds(0, op, wv);
    const uint64_t base = (uint64_t)(uintptr_t)(op == 0 ? DO : Q);
    gb[op] = base + 2u * (uint64_t)k1_tdm_gelem(bat, Sq, 0, Hq, k1_qh(hkv, G, 0), op, wv);
    gstep_q[op] = 2 * (k1_tdm_gelem(bat, Sq, 32, Hq, k1_qh(hkv, G, 0), op, wv)
                       - k1_tdm_gelem(bat, Sq, 0, Hq, k1_qh(hkv, G, 0), op, wv));
    gstep_g[op] = 2 * (k1_tdm_gelem(bat, Sq, 0, Hq, k1_qh(hkv, G, 1), op, wv)
                       - k1_tdm_gelem(bat, Sq, 0, Hq, k1_qh(hkv, G, 0), op, wv));
  });
  // the TDM of tile (query pair qt, q head gh) into the stage at byte offset stage_off: this wave's 16 rows.
  // tdm_prep = the scalar math (group 0: LDS address, 64-bit global address), tdm_issue = the 3 ops
  struct TdmG0 { i32x4 g0[3]; };
  auto tdm_prep = [&](int qt, int gh, int stage_off) {
    TdmG0 t;
    static_for<3>([&](auto op) {
      const uint64_t ga = gb[op] + (uint64_t)((int64_t)qt * gstep_q[op] + (int64_t)gh * gstep_g[op]);
      t.g0[op] = i32x4{1, (int)(ldsb[op] + (uint32_t)stage_off), (int)(uint32_t)ga,
                       (int)((uint32_t)(ga >> 32) | 0x80000000u)};
    });
    return t;
  };
  auto tdm_issue = [&](const TdmG0& t) { static_for<3>([&](auto op) { tdm_load(t.g0[op], g1[op]); }); };
  auto tdm_tile = [&](int qt, int gh, int stage_off) { tdm_issue(tdm_prep(qt, gh, stage_off)); };

  // ---- NL / ND: the 8 vec4 loads of (query pair qt, q head gh); voffset 32*half, soffset uniform -------
  const int voff_l = k1_nl_voff(lane);
  auto ldl = [&](int qt, int gh, f32x4 (&out)[NLD]) {
    const int sb = uniform(k1_nl_soff(bat, Hq, k1_qh(hkv, G, gh), Sq, qt * 32));
    static_for<NLD>([&](auto e) {
      // X1a: the constant goes on the VGPR offset (3 loop-invariant VGPRs voff + 16 / 64 / 80), so all 8 loads share
      // one SGPR soffset and no address register is rewritten while earlier loads still read it (s_wait_xcnt WAR)
      out[e] = buf_load_f32x4(((e % 4) / 2) == 0 ? r_nl : r_nd, voff_l + k1_nl_sconst(e), sb);
    });
  };

  // ---- readback of a stage's S/dP B operands: entry hh*20 + (Q: dt*2+u | dO: 12 + dt*2+u) --------------
  struct RdBase { const lds_u8* q; const lds_u8* v; };
  auto rd_base = [&](int stage_off) {
    return RdBase{lds_ptr(lds0 + (uint32_t)(stage_off + k1_rd_lane(1, lane))),
                  lds_ptr(lds0 + (uint32_t)(stage_off + k1_rd_lane(0, lane)))};
  };
  // one hh half (20 entries: Q dt*2+u, then dO 12 + dt*2+u) into out[0, 20)
  auto rdqd_hh = [&](const RdBase& b, auto hh, bf16x8 (&out)[NHB]) {
    static_for<NDT_QK>([&](auto dt) {
      static_for<2>([&](auto u) { out[dt * 2 + u] = lds_load_b128(b.q, k1_rd_const(hh, 1, dt, u)); });
    });
    static_for<NDT_V>([&](auto dt) {
      static_for<2>([&](auto u) { out[NQE + dt * 2 + u] = lds_load_b128(b.v, k1_rd_const(hh, 0, dt, u)); });
    });
  };
  // both halves (k1 body): entry hh*20 + ...
  auto rdqd = [&](const RdBase& b, bf16x8 (&out)[NRB]) {
    static_for<2>([&](auto hh) {
      static_for<NDT_QK>([&](auto dt) {
        static_for<2>([&](auto u) { out[hh * NHB + dt * 2 + u] = lds_load_b128(b.q, k1_rd_const(hh, 1, dt, u)); });
      });
      static_for<NDT_V>([&](auto dt) {
        static_for<2>([&](auto u) { out[hh * NHB + NQE + dt * 2 + u] = lds_load_b128(b.v, k1_rd_const(hh, 0, dt, u)); });
      });
    });
  };
  // B operands of the dV/dK WMMAs: rows [lane_r] and [lane_r + 16] concatenate into a v16 operand
  auto tr = [&](const lds_u8* base, int isq, int d) -> bf16x16 {
    return cat(lds_tr16(base, k1_tr_const(isq, d, 0)), lds_tr16(base, k1_tr_const(isq, d, 1)));
  };
  auto tr_base = [&](int stage_off, int isq) { return lds_ptr(lds0 + (uint32_t)(stage_off + k1_tr_lane(isq, lane))); };

  // ======================= one body per block height: NKV 3 (96 rows, k3) or NKV 2 (64-row tail, k1) =========
  auto body = [&](auto nkv_c) __attribute__((always_inline)) {
    constexpr int NKV = decltype(nkv_c)::value;
    constexpr bool K3 = NKV == K3_NKV;
    constexpr int ROWS = 32 * NKV;
    constexpr int NST = NKV * (NDO_V + NDO_QK);   // accumulators: dV (kh, d) then dK (kh, d)
    constexpr int DK0 = NKV * NDO_V;
    const int kv0g = k3_kv0g(bid, n96);
    const int kv0 = k3_kv0(kv0g, wv, NKV);

    // ---- K / V fragments of this wave's 16*NKV kv rows: invariant over every query and q head -------------
    bf16x16 kf[NKV][NDT_QK], vf[NKV][NDT_V];
    static_for<NKV>([&](auto kh) {
      static_for<NDT_QK>([&](auto dt) {
        kf[kh][dt] = cat(buf_load_b128(r_k, 16 * k1_kfrag(kbase, Hkv, kv0, kh, dt, 0, lane), 0),
                         buf_load_b128(r_k, 16 * k1_kfrag(kbase, Hkv, kv0, kh, dt, 1, lane), 0));
      });
      static_for<NDT_V>([&](auto dt) {
        vf[kh][dt] = cat(buf_load_b128(r_v, 16 * k1_vfrag(vbase, Hkv, kv0, kh, dt, 0, lane), 0),
                         buf_load_b128(r_v, 16 * k1_vfrag(vbase, Hkv, kv0, kh, dt, 1, lane), 0));
      });
    });

    // ---- dS^T export (k3ds_idx.h over k1ds_idx.h, dslayout.h v1): per-wave descriptor, row constants, voffset --
    const int kt0 = k3ds_kt0(kv0);
    const int64_t ds_base = k3ds_base(bat, Hq, hkv, G, kt0, Sq, Skv, causal);
    const auto r_ds = rsrc64((const unsigned char*)DS + ds_base, k3ds_nrec(ds_mode, G, kt0, NKV, Sq, Skv, causal));
    int ds_rc[NKV];
    static_for<NKV>([&](auto kh) { ds_rc[kh] = k3ds_rowc(kh, kt0, Sq, Skv, causal); });
    const int vds = k1ds_voff(lane);
    // the two stores of one kv16 block: DPP row_ror:8 regroup of a_dsk[kh] (X -> block rows 0..7, Y -> 8..15)
    auto ds_store = [&](const bf16x16& a, int so) {
      const u32x8 w = __builtin_bit_cast(u32x8, a);
      const u32x4 x = {dpp_ror8<0xC>(w[0], w[4]), dpp_ror8<0xC>(w[1], w[5]), dpp_ror8<0xC>(w[2], w[6]),
                       dpp_ror8<0xC>(w[3], w[7])};
      const u32x4 y = {dpp_ror8<0x3>(w[4], w[0]), dpp_ror8<0x3>(w[5], w[1]), dpp_ror8<0x3>(w[6], w[2]),
                       dpp_ror8<0x3>(w[7], w[3])};
      buf_store_u32x4(x, r_ds, vds, so);
      buf_store_u32x4(y, r_ds, vds + K1DS_ST_B, so);
    };

    // ---- S/dP + softmax/dS of one (hh, kh): S = Q K^T and dP = dO V^T (A = the Q / dO rows of hh, B = K / V of
    // kh): C lane = kv row, element si = q row hh*16 + half*8 + si; results into ph/dh[kh][hh*8 + si]
    auto sdp_one = [&](auto do_mask, auto hh, auto kh, const bf16x16 (&qfr)[NDT_QK], const bf16x16 (&dfr)[NDT_V],
                       const f32x4 (&pre)[NLD], int q0, bf16 (&ph)[NKV][16], bf16 (&dh)[NKV][16]) {
      const int qb = q0 + hh * 16 + half * 8;
      f32x8 s = {0, 0, 0, 0, 0, 0, 0, 0}, p = {0, 0, 0, 0, 0, 0, 0, 0};
      static_for<NDT_QK>([&](auto dt) {
        s = wmma<false, false>(qfr[dt], kf[kh][dt], s);
        if constexpr (dt < NDT_V) p = wmma<false, false>(dfr[dt], vf[kh][dt], p);
      });
      const int kvl = kv0 + kh * 16 + row;
      static_for<8>([&](auto si) {
        const float nl = pre[hh * 4 + si / 4][si % 4];
        const float nd = pre[hh * 4 + 2 + si / 4][si % 4];
        float tt = __builtin_fmaf(s[si.value], c1, nl);
        if constexpr (decltype(do_mask)::value) {
          // causal, bottom-right: query q attends kv <= q + cshift; kv = this lane's row
          tt = ((kvl > qb + si + cshift) & (causal != 0)) ? K1_NEG : tt;
        }
        const float pf = exp2_raw(tt);
        ph[kh][hh * 8 + si] = to_bf16(pf);
        dh[kh][hh * 8 + si] = to_bf16(pf * __builtin_fmaf(p[si.value], scale, nd));
      });
    };
    auto frags = [&](const bf16x8* ops, bf16x16 (&qfr)[NDT_QK], bf16x16 (&dfr)[NDT_V]) {
      static_for<NDT_QK>([&](auto dt) { qfr[dt] = cat(ops[dt * 2], ops[dt * 2 + 1]); });
      static_for<NDT_V>([&](auto dt) { dfr[dt] = cat(ops[NQE + dt * 2], ops[NQE + dt * 2 + 1]); });
    };
    auto pack = [&](const bf16 (&ph)[NKV][16], const bf16 (&dh)[NKV][16], bf16x16 (&a_pk)[NKV],
                    bf16x16 (&a_dsk)[NKV]) {
      static_for<NKV>([&](auto kh) {
        a_pk[kh] = bf16x16{ph[kh][0], ph[kh][1], ph[kh][2], ph[kh][3], ph[kh][4], ph[kh][5], ph[kh][6], ph[kh][7],
                           ph[kh][8], ph[kh][9], ph[kh][10], ph[kh][11], ph[kh][12], ph[kh][13], ph[kh][14],
                           ph[kh][15]};
        a_dsk[kh] = bf16x16{dh[kh][0], dh[kh][1], dh[kh][2], dh[kh][3], dh[kh][4], dh[kh][5], dh[kh][6], dh[kh][7],
                            dh[kh][8], dh[kh][9], dh[kh][10], dh[kh][11], dh[kh][12], dh[kh][13], dh[kh][14],
                            dh[kh][15]};
      });
    };
    // k1 body: both hh halves from the carried 40-entry readback (k1's sdp, unchanged order)
    auto sdp = [&](auto do_mask, const bf16x8 (&ops)[NRB], const f32x4 (&pre)[NLD], int q0,
                   bf16x16 (&a_pk)[NKV], bf16x16 (&a_dsk)[NKV]) {
      bf16 ph[NKV][16], dh[NKV][16];
      static_for<2>([&](auto hh) {
        bf16x16 qfr[NDT_QK], dfr[NDT_V];
        frags(ops + hh * NHB, qfr, dfr);
        static_for<NKV>([&](auto kh) { sdp_one(do_mask, hh, kh, qfr, dfr, pre, q0, ph, dh); });
      });
      pack(ph, dh, a_pk, a_dsk);
    };
    // k3 body: hh0 from the carried 20-entry readback op0; hh1's 20 entries are read from rb1 (the same stage) as
    // one burst at the top, so its latency hides under hh0's 30 S/dP WMMAs (fences let only VALU/SALU/trans cross)
    auto sdp3 = [&](auto do_mask, const bf16x8 (&op0)[NHB], const RdBase& rb1, const f32x4 (&pre)[NLD], int q0,
                    bf16x16 (&a_pk)[NKV], bf16x16 (&a_dsk)[NKV], auto&& after_burst) {
      bf16 ph[NKV][16], dh[NKV][16];
      bf16x8 op1[NHB];
      {
        bf16x16 qfr[NDT_QK], dfr[NDT_V];
        frags(op0, qfr, dfr);
        sched_barrier<SB_ALU>();
        rdqd_hh(rb1, std::integral_constant<int, 1>{}, op1);
        sched_barrier<SB_ALU>();
        // full loop: the pending prefetch TDM right behind the burst (ring region order rb, TDM, tr: isa_static S3
        // R4); no fence after it, so the hh0 S/dP chains may issue around the 3 ops
        after_burst();
        static_for<NKV>([&](auto k) {
          sdp_one(do_mask, std::integral_constant<int, 0>{}, k, qfr, dfr, pre, q0, ph, dh);
        });
      }
      {
        bf16x16 qfr[NDT_QK], dfr[NDT_V];
        frags(op1, qfr, dfr);
        static_for<NKV>([&](auto kh) {
          sdp_one(do_mask, std::integral_constant<int, 1>{}, kh, qfr, dfr, pre, q0, ph, dh);
        });
      }
      pack(ph, dh, a_pk, a_dsk);
    };
    // dV^T += dO^T P (B = P^T of kv sub-tile kh held across the run: reuse hint from the 2nd WMMA on),
    // dK^T += Q^T dS; C fragment = kv row kh*16 + row, d = dtile*16 + half*8 + [0, 8)
    auto dv_kh = [&](auto kh, f32x8 (&acc)[NST], const bf16x16 (&b_do)[NDO_V], const bf16x16 (&a_pk)[NKV]) {
      static_for<NDO_V>([&](auto d) {
        acc[kh * NDO_V + d] = wmma<false, (d > 0)>(b_do[d], a_pk[kh], acc[kh * NDO_V + d]);
      });
    };
    auto dk_kh = [&](auto kh, f32x8 (&acc)[NST], const bf16x16 (&b_q)[NDO_QK], const bf16x16 (&a_dsk)[NKV]) {
      static_for<NDO_QK>([&](auto d) {
        acc[DK0 + kh * NDO_QK + d] = wmma<false, (d > 0)>(b_q[d], a_dsk[kh], acc[DK0 + kh * NDO_QK + d]);
      });
    };
    // dK^T of kv sub-tile kh, d tiles [D0, D1) (the split-barrier window takes a prefix of dK kh1/kh2)
    auto dk_rng = [&](auto kh, auto d0, auto d1, f32x8 (&acc)[NST], const bf16x16 (&b_q)[NDO_QK],
                      const bf16x16 (&a_dsk)[NKV]) {
      static_for<decltype(d1)::value - decltype(d0)::value>([&](auto i) {
        constexpr int d = decltype(d0)::value + decltype(i)::value;
        acc[DK0 + kh * NDO_QK + d] = wmma<false, (d > 0)>(b_q[d], a_dsk[kh], acc[DK0 + kh * NDO_QK + d]);
      });
    };
    auto dvdk = [&](auto kh, f32x8 (&acc)[NST], const bf16x16 (&b_do)[NDO_V], const bf16x16 (&b_q)[NDO_QK],
                    const bf16x16 (&a_pk)[NKV], const bf16x16 (&a_dsk)[NKV]) {
      dv_kh(kh, acc, b_do, a_pk);
      dk_kh(kh, acc, b_q, a_dsk);
    };

    // ---- loop bounds: causal tile skip at workgroup level (block id only, both waves equal) -------------
    const int qp_start = k1_qp_start(kv0g, cshift, causal);
    const int nmaskp = k3_nmaskp(kv0g, ROWS, cshift, causal, nqt);
    const int nqp_eff = k1_nqp_eff(kv0g, cshift, causal, nqt);
    const int n_mask = G * nmaskp;
    const int n_full = G * (nqp_eff - nmaskp);

    f32x8 acc[NST];
    static_for<NST>([&](auto i) { acc[i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; });

    // ---- X3: the masked (diagonal) iterations ride the full loop's 3-stage Q/dO ring (k3m_idx.h) ----------
    // Flat tile sequence j < N = n_mask + n_full, T_j = (qp_start + j / G, j % G) in stage sigma(j) = (j - n_mask)
    // mod 3 (the full loop's first tile lands in stage 0); every prefetch is clamped to T_{N-1}.
    //   prologue (the ring is empty): TDM T_0 -> sigma(0), T_1 -> sigma(1); NL/ND of T_0; tensorcnt 3; BARRIER;
    //     readback of sigma(0) (k3: hh0 | k1: both halves); k3: T_2 -> sigma(2) pending | k1: TDM T_2 -> sigma(2).
    //   masked iteration j = the full iteration's ring protocol with the causal mask on:
    //     k3: NL/ND of T_{j+1}; hh1 readback of sigma(j), pending TDM T_{j+2} -> sigma(j+2); S/dP; tr16 dO^T; dV
    //         (+ prep of T_{j+3} -> sigma(j)); tr16 Q^T; dK kh0; tensorcnt 3; BARRIER(j); readback hh0 of
    //         sigma(j+1); dK kh1, kh2; export.
    //     k1: NL/ND of T_{j+1}; S/dP; tr16 dO^T, Q^T; dvdk kh0; tensorcnt 3; BARRIER(j); readback of sigma(j+1);
    //         TDM T_{j+3} -> sigma(j); dvdk kh1; export.
    //   (A fully masked wave-iteration, e.g. wave 1 at qp_start, still runs its WMMAs: skipping them behind a uniform
    //   branch measured no gain -- wave 0 bounds that iteration -- and raised SGPR pressure: X3b, p2_x3_dkdv_mask.md.)
    // After the last masked iteration the state is the full loop's entry state: qd = readback of stage 0 =
    // T_{n_mask}, T_{n_mask+1} in flight to stage 1, T_{n_mask+2} -> stage 2 (k3 pending | k1 issued), preA = NL/ND
    // of T_{n_mask}. Barriers per wave: 1 + n_mask + n_full + 1 (was 2 n_mask + 2 + n_full + 1).
    const int qt0 = qp_start + nmaskp;
    const int n_seq = n_mask + n_full;
    // NL/ND live in two register sets (k1): iteration ii reads one and prefetches ii+1 into the other, the loop
    // body is unrolled by 2 to alternate them (the masked loop copies preB -> preA instead).
    f32x4 preA[NLD], preB[NLD];
    // carried readback: k1 both hh halves (40 entries); k3 hh0 only (20)
    constexpr int NCAR = K3 ? NHB : NRB;
    bf16x8 qd[NCAR];
    TdmG0 tdp_pend;                          // k3: the prefetch TDM prepped in iteration ii-1 (or the prologue)
    {
      const int s0 = k3m_st0(n_mask);
      const int s1 = k1_ncur(s0), s2 = k1_ncur(s1);
      sched_barrier<0>();
      tdm_tile(k1_clampqt(qp_start, nqt2), 0, s0);
      tdm_tile(k1_clampqt(qp_start + k1_pro_q(1, n_seq, G), nqt2), k1_pro_g(1, n_seq, G), s1);
      ldl(k1_clampqt(qp_start, nqt2), 0, preA);
      sched_barrier<0>();
      tensor_wait<K1_TW>();
      sched_barrier<0>();
      lds_barrier();                         // RAW: every wave's half of sigma(0) landed
      sched_barrier<0>();
      if constexpr (K3) {
        rdqd_hh(rd_base(s0), std::integral_constant<int, 0>{}, qd);
        sched_barrier<0>();
        // T_2 -> sigma(2) is issued at the top of iteration 0, after its hh1 readback
        tdp_pend = tdm_prep(k1_clampqt(qp_start + k1_pro_q(2, n_seq, G), nqt2), k1_pro_g(2, n_seq, G), s2);
      } else {
        rdqd(rd_base(s0), qd);
        sched_barrier<0>();                  // readback before the TDM (one barrier region, R4 order)
        tdm_tile(k1_clampqt(qp_start + k1_pro_q(2, n_seq, G), nqt2), k1_pro_g(2, n_seq, G), s2);
      }
      sched_barrier<0>();
    }
    {
      int qi = 0, gh = 0, mst = k3m_st0(n_mask);
      for (int it = 0; it < n_mask; ++it) {
        const int qt = qp_start + qi;
        const int mnx = k1_ncur(mst);        // sigma(j+1)
        sched_barrier<0>();
        ldl(qp_start + k1_nx_q(it, n_seq, qi, gh, G), k1_nx_g(it, n_seq, qi, gh, G), preB);
        const int pfq = qp_start + k1_pf_q(it, n_seq, qi, gh, G), pfg = k1_pf_g(it, n_seq, qi, gh, G);
        bf16x16 a_pk[NKV], a_dsk[NKV];
        if constexpr (K3) {
          sdp3(std::true_type{}, qd, rd_base(mst), preA, qt * 32, a_pk, a_dsk, [&] { tdm_issue(tdp_pend); });
          const lds_u8* tv = tr_base(mst, 0);
          const lds_u8* tk = tr_base(mst, 1);
          const RdBase rbb = rd_base(mnx);
          sched_barrier<SB_ALU>();
          bf16x16 b_do[NDO_V];
          static_for<NDO_V>([&](auto d) { b_do[d] = tr(tv, 0, d); });
          sched_barrier<0>();
          tdp_pend = tdm_prep(pfq, pfg, mst);
          static_for<NKV>([&](auto kh) { dv_kh(kh, acc, b_do, a_pk); });
          static_for<NKV * NDO_V>([&](auto) {
            sched_group_barrier<0x008, 1, 0>();
            sched_group_barrier<0x004, KH0_SALU, 0>();
          });
          sched_barrier<0>();
          bf16x16 b_q[NDO_QK];
          static_for<NDO_QK>([&](auto d) { b_q[d] = tr(tk, 1, d); });
          sched_barrier<0>();
          dk_kh(std::integral_constant<int, 0>{}, acc, b_q, a_dsk);
          sched_barrier<0>();
          tensor_wait<K1_TW>();
          sched_barrier<0>();
          lds_barrier();
          sched_barrier<0>();
          rdqd_hh(rbb, std::integral_constant<int, 0>{}, qd);
          sched_barrier<0>();
          static_for<NKV - 1>([&](auto k) { dk_kh(std::integral_constant<int, k + 1>{}, acc, b_q, a_dsk); });
        } else {
          sdp(std::true_type{}, qd, preA, qt * 32, a_pk, a_dsk);
          const lds_u8* tv = tr_base(mst, 0);
          const lds_u8* tk = tr_base(mst, 1);
          const RdBase rbb = rd_base(mnx);
          sched_barrier<0>();
          bf16x16 b_do[NDO_V], b_q[NDO_QK];
          static_for<NDO_V>([&](auto d) { b_do[d] = tr(tv, 0, d); });
          sched_barrier<0>();
          static_for<NDO_QK>([&](auto d) { b_q[d] = tr(tk, 1, d); });
          sched_barrier<0>();
          const TdmG0 tdp = tdm_prep(pfq, pfg, mst);
          dvdk(std::integral_constant<int, 0>{}, acc, b_do, b_q, a_pk, a_dsk);
          static_for<NDO_V + NDO_QK>([&](auto) {
            sched_group_barrier<0x008, 1, 0>();
            sched_group_barrier<0x004, KH0_SALU, 0>();
          });
          sched_barrier<0>();
          tensor_wait<K1_TW>();
          sched_barrier<0>();
          lds_barrier();
          sched_barrier<0>();
          rdqd(rbb, qd);
          sched_barrier<0x004>();            // readback before the TDM; SALU may cross
          tdm_issue(tdp);
          sched_barrier<0>();
          dvdk(std::integral_constant<int, 1>{}, acc, b_do, b_q, a_pk, a_dsk);
        }
        // export of this masked iteration's dS^T: a block only if its kv16 row has an attended entry in pair qt
        sched_barrier<0>();
        const int so_m = k3ds_sq(qt, gh, Sq, Skv, causal);
        static_for<NKV>([&](auto kh) {
          if (k3ds_put(kh, qt, kt0, Sq, Skv, causal)) ds_store(a_dsk[kh], so_m + ds_rc[kh]);
        });
        sched_barrier<0>();
        static_for<NLD>([&](auto e) { preA[e] = preB[e]; });
        const int qn = k1_wrap_q(qi, gh, G), gn = k1_wrap_g(qi, gh, G);
        qi = qn;
        gh = gn;
        mst = mnx;
      }
    }
    int cur = 0, qi_c = 0, gh_c = 0;
    // k1ds (64-row tail body): the dS^T payload of the previous full iteration (its a_dsk) and the q-pair / q-head
    // part of its soffset, stored in the S/dP window of the next iteration (no DS op there) or after the loop
    bf16x16 pend[NKV];
    int pso = 0;
    // one full iteration ii: S/dP of the carried B operands (stage ii%3's tile, read back in ii-1) and NL/ND
    // `pre`; prefetch NL/ND of ii+1 into `nxt`; EXP (tail body): store iteration ii-1's dS^T in this S/dP window
    // X1d: g1 = true_type is the MHA (G == 1) specialization of the 96-row body's loop: the (query pair, q head)
    // counters are (ii, 0) there, so the clamped prefetch tiles min(ii+1, n-1) / min(ii+3, n-1) replace the
    // k1_nx_* / k1_pf_* select chains (same values: qi == ii, gh == 0 for G == 1) and no counter is carried.
    auto full_iter = [&](auto exp, auto g1, int ii, const f32x4 (&pre)[NLD], f32x4 (&nxt)[NLD]) {
      constexpr bool G1 = decltype(g1)::value;
      const int qi = G1 ? ii : qi_c;
      const int gh = G1 ? 0 : gh_c;
      const int qn = k1_wrap_q(qi, gh, G), gn = k1_wrap_g(qi, gh, G);
      const int ncur = k1_ncur(cur);         // (ii+1)%3: next iteration's stage == this one's readback stage
      // NL/ND prefetch for ii+1 (clamped) at the top
      sched_barrier<0>();
      if constexpr (G1) {
        ldl(qt0 + (ii + 1 < n_full ? ii + 1 : n_full - 1), 0, nxt);
      } else {
        ldl(qt0 + k1_nx_q(ii, n_full, qi, gh, G), k1_nx_g(ii, n_full, qi, gh, G), nxt);
      }
      bf16x16 a_pk[NKV], a_dsk[NKV];
      if constexpr (K3) {
        // (1) hh1 of stage cur is read back here, under hh0's S/dP WMMAs; right behind it the pending prefetch
        // TDM: tile min(ii+2, n-1) -> stage (ii+2)%3, whose last reads were before BARRIER(ii-1)
        sdp3(std::false_type{}, qd, rd_base(cur), pre, (qt0 + qi) * 32, a_pk, a_dsk, [&] { tdm_issue(tdp_pend); });
        const lds_u8* tv = tr_base(cur, 0);
        const lds_u8* tk = tr_base(cur, 1);
        const RdBase rbb = rd_base(ncur);
        sched_barrier<SB_ALU>();             // the softmax / pack VALU tail may overlap the burst
        // (2) tr16 burst 1: dO^T of stage cur
        bf16x16 b_do[NDO_V];
        static_for<NDO_V>([&](auto d) { b_do[d] = tr(tv, 0, d); });
        sched_barrier<0>();
        // k3ds: this iteration's dS^T (all NKV blocks) at the head of the DS-free dV WMMA run, 24 WMMAs before the
        // Q^T burst; the soffset from this iteration's counters (they advance only at the end of full_iter)
        const int so_c = k3ds_sq(qt0 + qi, gh, Sq, Skv, causal);
        // X1b1s: regroup all NKV blocks first (distinct staging registers), then the 2*NKV stores
        u32x4 dsx[NKV], dsy[NKV];
        static_for<NKV>([&](auto kh) {
          const u32x8 w = __builtin_bit_cast(u32x8, a_dsk[kh]);
          dsx[kh] = u32x4{dpp_ror8<0xC>(w[0], w[4]), dpp_ror8<0xC>(w[1], w[5]), dpp_ror8<0xC>(w[2], w[6]),
                          dpp_ror8<0xC>(w[3], w[7])};
          dsy[kh] = u32x4{dpp_ror8<0x3>(w[4], w[0]), dpp_ror8<0x3>(w[5], w[1]), dpp_ror8<0x3>(w[6], w[2]),
                          dpp_ror8<0x3>(w[7], w[3])};
        });
        sched_barrier<0>();
        static_for<NKV>([&](auto kh) {
          buf_store_u32x4(dsx[kh], r_ds, vds, so_c + ds_rc[kh]);
          buf_store_u32x4(dsy[kh], r_ds, vds + K1DS_ST_B, so_c + ds_rc[kh]);
        });
        // X1b1: the export is pinned here, under the dO^T tr16 latency, ahead of the first dV WMMA (in the
        // non-expert mode a store issued inside the WMMA run waits for every earlier WMMA)
        sched_barrier<0>();
        // the prefetch tile min(ii+3, n-1) -> stage cur: its scalar math rides the dV WMMA run ({1 WMMA,
        // KH0_SALU SALU} groups); the 3 ops issue at the top of iteration ii+1 (never, after the last iteration)
        if constexpr (G1) {
          tdp_pend = tdm_prep(qt0 + (ii + 3 < n_full ? ii + 3 : n_full - 1), 0, cur);
        } else {
          tdp_pend = tdm_prep(qt0 + k1_pf_q(ii, n_full, qi, gh, G), k1_pf_g(ii, n_full, qi, gh, G), cur);
        }
        // (2) r3 c1: dV d-outer / kh-inner (A = b_do[d] for 3 consecutive WMMAs), the 24 Q^T tr16 of stage cur
        // interleaved {1 WMMA : 2 tr16} behind the first 12 dV WMMAs (each dV accumulator still gets exactly one WMMA
        // per iteration: bits unchanged). The compiler puts the Q^T targets into the b_do registers that just died,
        // which costs an expert-mode va_vdst(0) drain per pair; manifest "asm_patch" (hip/asm/k_dkdv96ds.s.patch,
        // applied by hsaco.py to the compiler's own .s) re-targets those tr16 pairs to registers that are free over
        // the value's live range and recomputes the va_vdst waits with LLVM's rules (r3 c1, p4_c1.md).
        bf16x16 b_q[NDO_QK];
        static_for<NDO_V>([&](auto d) {
          static_for<NKV>([&](auto kh) {
            acc[kh * NDO_V + d] = wmma<(kh > 0), false>(b_do[d], a_pk[kh], acc[kh * NDO_V + d]);
          });
        });
        static_for<NDO_QK>([&](auto d) { b_q[d] = tr(tk, 1, d); });
        static_for<NKV * NDO_V>([&](auto i) {
          sched_group_barrier<0x008, 1, 0>();
          if constexpr (i < NDO_QK) sched_group_barrier<0x100, 2, 0>();
          sched_group_barrier<0x004, KH0_SALU, 0>();
        });
        sched_barrier<0>();
        dk_kh(std::integral_constant<int, 0>{}, acc, b_q, a_dsk);
        // the ring barrier after the last DS read of stage cur: tensorcnt 3 retires this wave's half of
        // (ii+1)%3, the release fence drains every wave's DS reads of cur (WAR for the prefetch below)
        sched_barrier<0>();
        tensor_wait<K1_TW>();
        sched_barrier<0>();
        // X1c: split ring barrier. arrive (s_wait_dscnt 0 + release + signal) -> the first XS of the 24 pure-register
        // dK kh1/kh2 WMMAs -> wait (+ acquire) -> hh0 readback of ncur -> the rest of dK kh1/kh2. WAR/RAW as before:
        // every wave signals only after its own tensorcnt 3 and its last DS read of stage cur.
        constexpr int XS = 24;
        lds_arrive();
        sched_barrier<0>();
        // window: global WMMA index j = (kh - 1) * NDO_QK + d over kh 1..NKV-1
        static_for<NKV - 1>([&](auto k) {
          constexpr int lo = k * NDO_QK, hi = lo + NDO_QK;
          constexpr int e = XS < lo ? 0 : (XS > hi ? NDO_QK : XS - lo);
          dk_rng(std::integral_constant<int, k + 1>{}, std::integral_constant<int, 0>{}, std::integral_constant<int, e>{},
                 acc, b_q, a_dsk);
        });
        sched_barrier<0>();
        lds_wait();
        sched_barrier<0>();
        rdqd_hh(rbb, std::integral_constant<int, 0>{}, qd);   // (1) hh0 of ncur; qd died in hh0's S/dP
        sched_barrier<0>();
        static_for<NKV - 1>([&](auto k) {
          constexpr int lo = k * NDO_QK, hi = lo + NDO_QK;
          constexpr int e = XS < lo ? 0 : (XS > hi ? NDO_QK : XS - lo);
          dk_rng(std::integral_constant<int, k + 1>{}, std::integral_constant<int, e>{}, std::integral_constant<int, NDO_QK>{},
                 acc, b_q, a_dsk);
        });
        sched_barrier<0>();
      } else {
        // the k1 body + k1ds's export: the previous iteration's payload in this S/dP window (no DS op there), the
        // NL/ND loads in front of the stores (a load issued behind stores waits for them: bwd_r7 diagnosis); ALU /
        // WMMA / TRANS may cross this fence, no VMEM or DS op
        if constexpr (decltype(exp)::value) {
          sched_barrier<0x040F>();
          static_for<NKV>([&](auto kh) { ds_store(pend[kh], pso + ds_rc[kh]); });
        }
        sdp(std::false_type{}, qd, pre, (qt0 + qi) * 32, a_pk, a_dsk);
        const lds_u8* tv = tr_base(cur, 0);
        const lds_u8* tk = tr_base(cur, 1);
        const RdBase rbb = rd_base(ncur);
        sched_barrier<0>();
        bf16x16 b_do[NDO_V], b_q[NDO_QK];
        static_for<NDO_V>([&](auto d) { b_do[d] = tr(tv, 0, d); });
        sched_barrier<0>();
        static_for<NDO_QK>([&](auto d) { b_q[d] = tr(tk, 1, d); });
        sched_barrier<0>();
        sched_barrier<0>();
        const TdmG0 tdp = tdm_prep(qt0 + k1_pf_q(ii, n_full, qi, gh, G), k1_pf_g(ii, n_full, qi, gh, G), cur);
        dvdk(std::integral_constant<int, 0>{}, acc, b_do, b_q, a_pk, a_dsk);
        static_for<NDO_V + NDO_QK>([&](auto) {
          sched_group_barrier<0x008, 1, 0>();
          sched_group_barrier<0x004, KH0_SALU, 0>();
        });
        sched_barrier<0>();
        tensor_wait<K1_TW>();
        sched_barrier<0>();
        lds_barrier();
        sched_barrier<0>();
        rdqd(rbb, qd);                         // qd's previous contents died in the S/dP phase
        sched_barrier<0x004>();                // readback before the TDM; SALU may cross
        tdm_issue(tdp);
        sched_barrier<0>();
        dvdk(std::integral_constant<int, 1>{}, acc, b_do, b_q, a_pk, a_dsk);
        sched_barrier<0>();
      }
      if constexpr (!K3) {
        // k1ds: this iteration's payload and the q-pair / q-head part of its soffset (counters not yet advanced)
        static_for<NKV>([&](auto kh) { pend[kh] = a_dsk[kh]; });
        pso = k3ds_sq(qt0 + qi, gh, Sq, Skv, causal);
      }
      cur = ncur;
      if constexpr (!G1) {
        qi_c = qn;
        gh_c = gn;
      }
    };
    if constexpr (K3) {
      auto loop = [&](auto g1) {
        int ii = 0;
        for (; ii + 1 < n_full; ii += 2) {
          full_iter(std::false_type{}, g1, ii, preA, preB);
          full_iter(std::false_type{}, g1, ii + 1, preB, preA);
        }
        if (ii < n_full) full_iter(std::false_type{}, g1, ii, preA, preB);
      };
      if (G == 1) {
        loop(std::true_type{});
      } else {
        loop(std::false_type{});
      }
    } else if (n_full > 0) {
      // k1ds: iteration 0 is peeled (nothing pending), then k1's x2 unrolled body and odd tail from iteration 1
      full_iter(std::false_type{}, std::false_type{}, 0, preA, preB);
      qi_c = k1ds_q1(G);                     // == k1_wrap_q(0, 0, G) (bounds_proof DL3), integer form
      gh_c = k1ds_g1(G);                     // == k1_wrap_g(0, 0, G)
      int ii = 1;
      for (; ii + 1 < n_full; ii += 2) {
        full_iter(std::true_type{}, std::false_type{}, ii, preB, preA);
        full_iter(std::true_type{}, std::false_type{}, ii + 1, preA, preB);
      }
      if (ii < n_full) full_iter(std::true_type{}, std::false_type{}, ii, preB, preA);
      // the last full iteration's payload
      sched_barrier<0>();
      static_for<NKV>([&](auto kh) { ds_store(pend[kh], pso + ds_rc[kh]); });
      sched_barrier<0>();
    }
    // retire the last (clamped) prefetch -- or the prologue's -- before the epilogue reuses the ring
    tensor_wait<0>();
    sched_barrier<0>();
    lds_barrier();
    sched_barrier<0>();

    // ---- epilogue (bwd_r9_a): C fragments -> this wave's row-major dV / dK images -> whole-line stores ----
    static_for<NKV>([&](auto kh) {
      static_for<NDO_QK>([&](auto d) {
        if constexpr (d < NDO_V)
          lds_store_b128(lds0 + (uint32_t)k3_epi_wr(NKV, wv, 0, kh, d, lane), to_bf16x8(acc[kh * NDO_V + d]));
        lds_store_b128(lds0 + (uint32_t)k3_epi_wr(NKV, wv, 1, kh, d, lane), to_bf16x8(acc[DK0 + kh * NDO_QK + d]));
      });
    });
    sched_barrier<0>();
    __builtin_amdgcn_wave_barrier();
    const auto r_dv = rsrc(DV, nv_b), r_dk = rsrc(DK, nk_b);
    static_for<4 * NKV>([&](auto rb) {
      static_for<K1_D_V * 2 / 128>([&](auto li) {
        const bf16x8 vv = lds_load_b128(lds0 + (uint32_t)k3_epi_rd(NKV, wv, 0, rb, li, lane));
        buf_store_b128(vv, r_dv, 16 * k1_epi_g(bat, Skv, Hkv, hkv, kv0, 0, rb, li, lane), 0);
      });
      static_for<K1_D_QK * 2 / 128>([&](auto li) {
        const bf16x8 kk = lds_load_b128(lds0 + (uint32_t)k3_epi_rd(NKV, wv, 1, rb, li, lane));
        buf_store_b128(kk, r_dk, 16 * k1_epi_g(bat, Skv, Hkv, hkv, kv0, 1, rb, li, lane), 0);
      });
    });
  };

  // uniform (SGPR) branch on the block id: 96-row blocks first, then the 64-row tail blocks
  if (bid < n96) {
    body(std::integral_constant<int, K3_NKV>{});
  } else {
    body(std::integral_constant<int, 2>{});
  }
}
