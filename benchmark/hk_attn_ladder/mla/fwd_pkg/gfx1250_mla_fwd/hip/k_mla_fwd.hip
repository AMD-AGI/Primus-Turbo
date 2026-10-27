/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd -- HK forward of the DeepSeek-V3 MLA attention on gfx1250 (round 4, W-S1a: the marker shell).
//   qk 192 / v 128, bf16, MHA, bottom-right causal (query i sees kv j <= i + Skv - Sq).
//   Q [B, Sq, H, 192], K [B, Skv, H, 192], V [B, Skv, H, 128] -> O [B, Sq, H, 128] bf16, LSE [B, H, Sq] fp32 (natural
//   log). The math of the shipped FlyDSL forward, bit for bit: fp32 online softmax, p = exp2(fma(s, c, -m)) with
//   c = scale * log2(e), m in log2 units, the lazy-rescale decision of FlyDSL (m moves only when the row max grew by
//   more than 1 nat in some row of the 16-row q tile), d = fma(corr, d, ls + peer(ls)) with FlyDSL's sum tree,
//   out = bf16(O * (d > 0 ? 1 / d : 0)), lse = (m + log2 d) * ln 2. O is multiplied by corr on every tile (corr is
//   exactly 1 when m did not move), and the fully masked q tiles of the diagonal tiles run as exact no-ops.
//
// grid (ceil(Sq / 256), H, B) through FlyDSL's longest-first / head-grouped block remap; block 128 = 4 waves, one per
// SIMD, one workgroup per CU (LDS 274432 B). Wave w owns the 16-row q tiles {w, w + 4, w + 8, w + 12} of the 256-row
// block (slot i = tile w + 4 i). K / V go through a 4-stage LDS ring of 64-kv tiles (K 64 x 400 B, V 64 x 272 B; wave
// 0 issues the 3 TDM ops of tile t + 3 at the start of tile t, branch-free: waves 1..3 issue NULL descriptors, and
// past the last tile a 1-row copy of tile n - 1 into the free stage). Q stays in its LDS staging area for the block.
//
// This file is the SHELL (LLVM compiles the prologue, the TDM descriptors and issue, the ring barriers' waits, the
// loop control and the O / LSE stores). The per-tile work is spliced in after compilation at the HKGEN markers, inline
// asm statements whose operands carry the shell's values into the generated code (gen/fwdgen.py, gen/build.py):
//   INIT          once per block: m = -1e30, d = 0, O = 0, constants; takes the Q fragment base addresses
//   TILE_C/TILE_M per 64-kv tile (C: no mask, M: the causal mask on all four q tiles): Q / K / V fragment reads from
//                 LDS, s_wait_dscnt 0 + s_wait_tensorcnt 6 + s_barrier_signal (B_{t+1}) as soon as this wave's
//                 reads of the stage are done, QK -> softmax -> rescale -> PV
//   FINAL_i       per q tile i: 1 / d, lse, O * (1 / d) -> bf16 fragments in the marker's output registers
// Contract of the generated code: it writes only VGPRs above every VGPR the shell's code references (its own state
// lives there across the markers), the marker outputs, and the clobbered SGPRs / vcc / scc; no global memory access;
// LDS reads at the marker addresses plus constant offsets; MSB mode 0 at its end; one s_barrier_signal per TILE.
// -DHKGEN_REF=1 builds the same per-tile math in HIP instead of the markers (the reference the generated code is
// checked against, and a class-1 kernel of its own).
#include "hkf.h"

#pragma clang fp contract(off)

using namespace hkf;

#ifndef HKGEN_REF
#define HKGEN_REF 0
#endif
// HKGEN_DIAG=1: the diagonal tiles get specialized bodies TILE_D0..3 (tile ncc + k: q tiles < k skipped, as FlyDSL
// skips them; the causal mask on q tile k only). Requires Sq == Skv (the host guarantees it).
#ifndef HKGEN_DIAG
#define HKGEN_DIAG 0
#endif

// SGPRs the generated code may use as temporaries (and vcc / scc): declared clobbered at every marker
#define HKGEN_CLOBBERS "memory", "vcc", "scc", "s90", "s91", "s92", "s93", "s94", "s95", "s96", "s97", "s98", \
                       "s99", "s100", "s101", "s102", "s103"

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int BM = 256;          // q rows per workgroup
constexpr int QT = 4;            // 16-row q tiles per wave
constexpr int NB = 64;           // kv rows per ring stage (tile)
constexpr int NKS = D_QK / 32;   // 6 k-steps of QK
constexpr int NDT = D_V / 16;    // 8 d tiles of O
constexpr int NKV = NB / 16;     // 4 kv tiles of 16 per tile
constexpr int XK = 400, XV = 272;  // LDS row strides (bytes)
constexpr int K_B = NB * XK;       // 25600
constexpr int V_B = NB * XV;       // 17408
constexpr int STG_B = K_B + V_B;   // 43008
constexpr int NSTG = 4;
constexpr int Q_OFF = NSTG * STG_B;     // 172032
constexpr int LDS_B = Q_OFF + BM * XK;  // 274432
static_assert(LDS_B <= 327680 && 2 * LDS_B > 327680, "one workgroup per CU");
constexpr unsigned W0_A = 0x47510000u;  // 128 columns, pad 72 elements -> 400-B rows
constexpr unsigned W0_B = 0x87110000u;  // 64 columns, pad 136 elements -> 400-B rows
constexpr unsigned W0_V = 0x07510000u;  // 128 columns, pad 8 elements -> 272-B rows
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.69314718055994531f;  // fp32 0x3f317218 = FlyDSL's fp32(1 / LOG2E)

HKF_INLINE i32x8 tdm_g1(unsigned w0, unsigned tdim1, unsigned tile0, unsigned tile1, unsigned long long stride_el) {
  return i32x8{(int)w0, (int)0xFFFF0000u, (int)(((tdim1 & 0xFFFFu) << 16) | 0x7FFFu),
               (int)(((tdim1 >> 16) & 0xFFFFu) | (tile0 << 16)), (int)tile1, (int)(unsigned)stride_el,
               (int)((unsigned)(stride_el >> 32) & 0xFFFFu), 0};
}
HKF_INLINE i32x4 tdm_g0(unsigned lds, unsigned long long ga, int count = 1) {
  // count 0 = the NULL descriptor: moves nothing (the branch-free way to keep a TDM op off a wave)
  return i32x4{count, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | 0x80000000u)};
}

// FlyDSL's _remap: (q block, head, batch) of the hardware dispatch rank, longest-first and head-grouped
HKF_INLINE void block_ids(int gx, int gy, int gz, int& bx, int& by, int& bz) {
  const int lin = (int)__builtin_amdgcn_workgroup_id_x() +
                  gx * ((int)__builtin_amdgcn_workgroup_id_y() + gy * (int)__builtin_amdgcn_workgroup_id_z());
  const int gyz = gy * gz;
  int rank = lin / gyz;
  int rem = lin - rank * gyz;
  const int r = (rank & 15) * gyz + rem;
  const int c = r & 255;
  const bool grouped = ((gx & 15) | (gyz & 15)) == 0;
  const int lin2 = grouped ? ((rank & -16) + (c >> 4)) * gyz + ((r >> 8) << 4) + (c & 15) : lin;
  rank = lin2 / gyz;
  rem = lin2 - rank * gyz;
  bx = gx - 1 - rank;
  by = rem % gy;
  bz = rem / gy;
}

#if HKGEN_REF
HKF_INLINE f32x8 zero8() { return f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; }
HKF_INLINE float fmax2(float a, float b) { return __builtin_fmaxf(a, b); }
// FlyDSL _tree_reduce on one list: balanced 3-ary, (a + b) + c, a trailing pair a + b, a trailing single passed on
template <int N>
HKF_INLINE float tree_sum(const float (&x)[N]) {
  if constexpr (N == 1) {
    return x[0];
  } else {
    constexpr int M = N / 3 + ((N % 3) ? 1 : 0);
    float y[M];
    static_for<M>([&](auto j) {
      constexpr int i = 3 * j;
      if constexpr (N - i >= 3) {
        y[j] = (x[i] + x[i + 1]) + x[i + 2];
      } else if constexpr (N - i == 2) {
        y[j] = x[i] + x[i + 1];
      } else {
        y[j] = x[i];
      }
    });
    return tree_sum<M>(y);
  }
}
// max over the values: max3 of consecutive triples (a trailing pair max2, a trailing single passed on), per level
template <int N>
HKF_INLINE float tree_max(const float (&x)[N]) {
  if constexpr (N == 1) {
    return x[0];
  } else {
    constexpr int M = N / 3 + ((N % 3) ? 1 : 0);
    float y[M];
    static_for<M>([&](auto j) {
      constexpr int i = 3 * j;
      if constexpr (N - i >= 3) {
        y[j] = __builtin_fmaxf(__builtin_fmaxf(x[i], x[i + 1]), x[i + 2]);
      } else if constexpr (N - i == 2) {
        y[j] = __builtin_fmaxf(x[i], x[i + 1]);
      } else {
        y[j] = x[i];
      }
    });
    return tree_max<M>(y);
  }
}
#endif
}  // namespace

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_mla_fwd(const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
               bf16* __restrict__ O, float* __restrict__ LSE, float scale, int Sq, int Skv, int H, int B, int nqb) {
  __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
  lock_simd();
  const uint32_t lds0 = lds_addr(smem);
  const int tid = (int)__builtin_amdgcn_workitem_id_x();
  const int lane = tid & 31;
  const int wv = wave_id(tid);
  int qb, hh, bb;
  block_ids(nqb, H, B, qb, hh, bb);
  const int q0 = qb * BM;
  const int cshift = Skv - Sq;
  const int qv = min(q0 + BM, Sq);
  const int n = (qv + cshift + NB - 1) / NB;          // 64-kv tiles (ring stages) of the block
  const int ncc = min((q0 + cshift + 1) / NB, n);     // tiles [0, ncc) need no mask for any row of the block
  const float c2 = scale * LOG2E;

  const unsigned long long krow = (unsigned long long)H * D_QK;
  const unsigned long long vrow = (unsigned long long)H * D_V;
  const unsigned long long gk0 = (unsigned long long)(uintptr_t)K + 2ull * (((unsigned long long)bb * Skv * H + hh) * D_QK);
  const unsigned long long gv0 = (unsigned long long)(uintptr_t)V + 2ull * (((unsigned long long)bb * Skv * H + hh) * D_V);
  const unsigned long long gq0 =
      (unsigned long long)(uintptr_t)Q + 2ull * ((((unsigned long long)bb * Sq + q0) * H + hh) * D_QK);

  auto issue_tile = [&](int t) {
    const int kv0 = t * NB;
    const unsigned st = lds0 + (unsigned)((t & (NSTG - 1)) * STG_B);
    const unsigned rows = (unsigned)(Skv - kv0);
    const unsigned long long gk = gk0 + 2ull * (unsigned long long)kv0 * krow;
    const unsigned long long gv = gv0 + 2ull * (unsigned long long)kv0 * vrow;
    tdm_load(tdm_g0(st, gk), tdm_g1(W0_A, rows, 128, NB, krow));
    tdm_load(tdm_g0(st + 256, gk + 256), tdm_g1(W0_B, rows, 64, NB, krow));
    tdm_load(tdm_g0(st + K_B, gv), tdm_g1(W0_V, rows, 128, NB, vrow));
  };
  // branch-free in-loop issue (every wave executes it, waves 1..3 with NULL descriptors): tile t into stage t % 4, or
  // -- past the last tile -- a 1-row copy of tile n - 1 into that (free) stage, so wave 0 keeps exactly 3 ops per tile
  // in flight and its tensor waits stay static.
  const int cnt0 = wv == 0 ? 1 : 0;
  auto issue_tile_bf = [&](int t) {
    const bool real = t < n;
    const int kv0 = (real ? t : n - 1) * NB;
    const unsigned st = lds0 + (unsigned)((t & (NSTG - 1)) * STG_B);
    const unsigned rows = (unsigned)(Skv - kv0);
    const unsigned th = real ? NB : 1;
    const unsigned long long gk = gk0 + 2ull * (unsigned long long)kv0 * krow;
    const unsigned long long gv = gv0 + 2ull * (unsigned long long)kv0 * vrow;
    tdm_load(tdm_g0(st, gk, cnt0), tdm_g1(W0_A, rows, 128, th, krow));
    tdm_load(tdm_g0(st + 256, gk + 256, cnt0), tdm_g1(W0_B, rows, 64, th, krow));
    tdm_load(tdm_g0(st + K_B, gv, cnt0), tdm_g1(W0_V, rows, 128, th, vrow));
  };

  // ---- prologue: tiles 0..2 in flight, Q staged; barrier B_0 = Q and tile 0 visible ---------------------------------
  if (wv == 0) {
    issue_tile(0);
    issue_tile_bf(1);
    issue_tile_bf(2);
    tensor_wait<6>();
  } else if (wv == 1) {
    tdm_load(tdm_g0(lds0 + Q_OFF, gq0), tdm_g1(W0_A, (unsigned)(Sq - q0), 128, BM, krow));
    tensor_wait<0>();
  } else if (wv == 2) {
    tdm_load(tdm_g0(lds0 + Q_OFF + 256, gq0 + 256), tdm_g1(W0_B, (unsigned)(Sq - q0), 64, BM, krow));
    tensor_wait<0>();
  }
  bar_signal();
  bar_wait();

  // per-lane LDS bases of the fragment reads (FlyDSL's lane patterns): Q B operands (q tiles 0, 1 from qa, 2, 3 from
  // qb), K A operands (+ kv16 * 16 * XK + ks * 64 + {0, 32}), V^T A operands via ds_load_tr16_b128 (+ (32 kt + 16 h)
  // * XV + dt * 32)
  const unsigned qa = lds0 + Q_OFF + (unsigned)((16 * wv + (lane & 15)) * XK + (lane >> 4) * 16);
  const unsigned qb2 = qa + (unsigned)(2 * 64 * XK);
  const unsigned k_lane = (unsigned)((lane & 15) * XK + (lane >> 4) * 16);
  const unsigned v_lane = (unsigned)((((lane >> 4) * 8) + (lane & 7)) * XV + ((lane >> 3) & 1) * 16);
  // causal: element e of kv16 tile j of tile t (kv = 64 t + 16 j + 8 (lane / 16) + e) is masked for slot i iff
  // 16 j + e > dl0 - 64 t + 64 i
  const int dl0 = q0 + 16 * wv + (lane & 15) + cshift - (lane >> 4) * 8;

#if HKGEN_REF
  f32x8 o[QT][NDT];
  float m[QT], d[QT];
  static_for<QT>([&](auto i) {
    m[i] = -1.0e30f;
    d[i] = 0.f;
    static_for<NDT>([&](auto dt) { o[i][dt] = zero8(); });
  });
  // first: q tiles below it are skipped; mask_only < 0: the mask on every q tile, else on that q tile only
  auto tile = [&](unsigned kb, unsigned vb, bool masked, int dl, int first, int mask_only) __attribute__((always_inline)) {
    const lds_u8* kp = lds_ptr(kb);
    const lds_u8* vp = lds_ptr(vb);
    bf16x16 kf[NKV][NKS];
    static_for<NKS>([&](auto ks) {
      static_for<NKV>([&](auto j) {
        kf[j][ks] = cat(lds_load_b128(kp, j * 16 * XK + ks * 64), lds_load_b128(kp, j * 16 * XK + ks * 64 + 32));
      });
    });
    bf16x16 vf[NDT][2];
    static_for<NDT>([&](auto dt) {
      static_for<2>([&](auto kt) {
        vf[dt][kt] = cat(lds_tr16(vp, kt * 32 * XV + dt * 32), lds_tr16(vp, (kt * 32 + 16) * XV + dt * 32));
      });
    });
    __builtin_amdgcn_s_wait_tensorcnt((short)6);
    bar_signal();  // B_{t+1}: this wave's reads of the stage are done, wave 0's tile t + 1 landed
    static_for<QT>([&](auto i) {
      if (i < first) return;
      const lds_u8* qp = lds_ptr((i < 2 ? qa : qb2) + (unsigned)((i & 1) * 64 * XK));
      f32x8 s[NKV];
      static_for<NKS>([&](auto ks) {
        const bf16x16 qf = cat(lds_load_b128(qp, ks * 64), lds_load_b128(qp, ks * 64 + 32));
        static_for<NKV>([&](auto j) {
          if constexpr (ks == 0) {
            s[j] = wmma(kf[j][0], qf, zero8());
          } else {
            s[j] = wmma(kf[j][ks], qf, s[j]);
          }
        });
      });
      if (masked && (mask_only < 0 || mask_only == i)) {
        const int dli = dl + 64 * i;
        static_for<NKV>([&](auto j) {
          static_for<8>([&](auto e) { s[j][(int)e] = (j * 16 + e > dli) ? -__builtin_inff() : s[j][(int)e]; });
        });
      }
      float x32[32];
      static_for<NKV>([&](auto j) { static_for<8>([&](auto e) { x32[j * 8 + e] = s[j][(int)e]; }); });
      const float mx = tree_max<32>(x32);  // the generated code's grouping (max is exact: any grouping is equal)
      const float lm = fmax2(mx, peer(mx));
      const float need = __builtin_fmaf(lm, c2, -m[i]);
      const bool resc = __builtin_amdgcn_ballot_w32(need > LOG2E) != 0u;
      const float rmax = lm * c2;
      const float mfull = fmax2(m[i], rmax);
      const float mn = resc ? mfull : m[i];
      const float corr = exp2_raw(m[i] - mn);
      float p[32];
      static_for<NKV>([&](auto j) {
        static_for<8>([&](auto e) { p[j * 8 + e] = exp2_raw(__builtin_fmaf(s[j][(int)e], c2, -mn)); });
      });
      const float ls = tree_sum<32>(p);
      const float tot = ls + peer(ls);
      d[i] = __builtin_fmaf(corr, d[i], tot);
      m[i] = mn;
      static_for<NDT>([&](auto dt) { o[i][dt] = o[i][dt] * corr; });
      static_for<2>([&](auto kt) {
        f32x8 pa, pb;
        static_for<8>([&](auto e) {
          pa[(int)e] = p[(2 * kt) * 8 + e];
          pb[(int)e] = p[(2 * kt + 1) * 8 + e];
        });
        const bf16x16 pf = cat(to_bf16x8(pa), to_bf16x8(pb));
        static_for<NDT>([&](auto dt) { o[i][dt] = wmma(vf[dt][kt], pf, o[i][dt]); });
      });
    });
  };
#else
  asm volatile("; HKGEN INIT qa=%0 qb=%1 c2=%2" ::"v"(qa), "v"(qb2), "s"(c2) : HKGEN_CLOBBERS);
#endif

  // ---- the tiles: [0, ncc) need no mask, [ncc, n) carry the causal mask (diagonal tiles, kv tail) -------------------
  int t = 0;
  for (; t < ncc; ++t) {
    issue_tile_bf(t + 3);
    const unsigned st = lds0 + (unsigned)((t & (NSTG - 1)) * STG_B);
#if HKGEN_REF
    tile(st + k_lane, st + K_B + v_lane, false, 0, 0, -1);
#else
    asm volatile("; HKGEN TILE_C kb=%0 vb=%1" ::"v"(st + k_lane), "v"(st + K_B + v_lane) : HKGEN_CLOBBERS);
#endif
    bar_wait();  // B_{t+1}
  }
#if HKGEN_DIAG
  static_for<4>([&](auto k) {  // the diagonal tiles ncc + k (Sq == Skv: at most 4)
    if (ncc + k < n) {
      const int t2 = ncc + k;
      issue_tile_bf(t2 + 3);
      const unsigned st = lds0 + (unsigned)((t2 & (NSTG - 1)) * STG_B);
#if HKGEN_REF
      tile(st + k_lane, st + K_B + v_lane, true, dl0 - NB * t2, k, k);
#else
      if constexpr (k == 0) {
        asm volatile("; HKGEN TILE_D0 kb=%0 vb=%1 dl=%2" ::"v"(st + k_lane), "v"(st + K_B + v_lane),
                     "v"(dl0 - NB * t2) : HKGEN_CLOBBERS);
      } else if constexpr (k == 1) {
        asm volatile("; HKGEN TILE_D1 kb=%0 vb=%1 dl=%2" ::"v"(st + k_lane), "v"(st + K_B + v_lane),
                     "v"(dl0 - NB * t2) : HKGEN_CLOBBERS);
      } else if constexpr (k == 2) {
        asm volatile("; HKGEN TILE_D2 kb=%0 vb=%1 dl=%2" ::"v"(st + k_lane), "v"(st + K_B + v_lane),
                     "v"(dl0 - NB * t2) : HKGEN_CLOBBERS);
      } else {
        asm volatile("; HKGEN TILE_D3 kb=%0 vb=%1 dl=%2" ::"v"(st + k_lane), "v"(st + K_B + v_lane),
                     "v"(dl0 - NB * t2) : HKGEN_CLOBBERS);
      }
#endif
      bar_wait();  // B_{t2+1}
    }
  });
#else
  for (; t < n; ++t) {
    issue_tile_bf(t + 3);
    const unsigned st = lds0 + (unsigned)((t & (NSTG - 1)) * STG_B);
#if HKGEN_REF
    tile(st + k_lane, st + K_B + v_lane, true, dl0 - NB * t, 0, -1);
#else
    asm volatile("; HKGEN TILE_M kb=%0 vb=%1 dl=%2" ::"v"(st + k_lane), "v"(st + K_B + v_lane), "v"(dl0 - NB * t)
                 : HKGEN_CLOBBERS);
#endif
    bar_wait();  // B_{t+1}
  }
#endif
  tensor_wait<0>();  // no TDM write may land after the workgroup retires

  // ---- epilogue: per q tile, out = bf16(O^T / d) and lse = (m + log2 d) ln 2 ------------------------------------------
  const auto r_o = rsrc(O, (int)min((long long)B * Sq * H * D_V * 2, 0x7FFFFFFFll));
  const auto r_l = rsrc(LSE, (int)min((long long)B * H * Sq * 4, 0x7FFFFFFFll));
  static_for<QT>([&](auto i) {
    i32x32 ob;
    float lse;
#if HKGEN_REF
    const float inv = d[i] > 0.f ? 1.0f / d[i] : 0.f;
    static_for<NDT>([&](auto dt) {
      const u32x4 w = __builtin_bit_cast(u32x4, to_bf16x8(o[i][dt] * inv));
      static_for<4>([&](auto r) { ob[4 * dt + (int)r] = (int)w[(int)r]; });
    });
    lse = (m[i] + __builtin_log2f(d[i])) * LN2;
#else
    if constexpr (i == 0) {
      asm volatile("; HKGEN FINAL_0 o=%0 lse=%1" : "=v"(ob), "=v"(lse) : : HKGEN_CLOBBERS);
    } else if constexpr (i == 1) {
      asm volatile("; HKGEN FINAL_1 o=%0 lse=%1" : "=v"(ob), "=v"(lse) : : HKGEN_CLOBBERS);
    } else if constexpr (i == 2) {
      asm volatile("; HKGEN FINAL_2 o=%0 lse=%1" : "=v"(ob), "=v"(lse) : : HKGEN_CLOBBERS);
    } else {
      asm volatile("; HKGEN FINAL_3 o=%0 lse=%1" : "=v"(ob), "=v"(lse) : : HKGEN_CLOBBERS);
    }
#endif
    const int q = q0 + 16 * (wv + 4 * i) + (lane & 15);
    if (q < Sq) {
      const int obase = (int)((((long long)bb * Sq + q) * H + hh) * D_V + (lane >> 4) * 8) * 2;
      static_for<NDT>([&](auto dt) {
        buf_store_b128(u32x4{(unsigned)ob[4 * dt], (unsigned)ob[4 * dt + 1], (unsigned)ob[4 * dt + 2],
                             (unsigned)ob[4 * dt + 3]},
                       r_o, obase + dt * 32, 0);
      });
      if ((lane >> 4) == 0) buf_store_b32(lse, r_l, (int)((((long long)bb * H + hh) * Sq + q) * 4), 0);
    }
  });
}
