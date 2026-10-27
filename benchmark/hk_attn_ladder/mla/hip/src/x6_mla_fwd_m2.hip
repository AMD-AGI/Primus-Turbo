/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd (x6 M2) -- HIP forward of the DeepSeek-V3 MLA attention on gfx1250:
//   qk 192 / v 128, bf16, MHA, bottom-right causal (query i sees kv j <= i + Skv - Sq).
//   Q [B, Sq, H, 192], K [B, Skv, H, 192], V [B, Skv, H, 128] -> O [B, Sq, H, 128] bf16,
//   LSE [B, H, Sq] fp32 natural log. The FlyDSL forward's math: fp32 online softmax with
//   p = exp2(s * scale * log2e - m), m in log2 units, lazy rescale (threshold 1 nat, wave-uniform
//   per 16-row q tile), lse = (m + log2 l) * ln 2.
//
// grid (ceil(Sq / 256), H, B) through FlyDSL's longest-first / head-grouped block remap; block 128 =
// 4 waves, 1 wave per SIMD, one workgroup per CU (LDS 274432 B). Wave w owns the 16-row q tiles
// {w, w + 4, w + 8, w + 12} of the 256-row block (slot i = tile w + 4 i): every wave runs the same
// number of kv chunks.
//
// M2: the online softmax advances per 32-kv chunk c (S^T chunk = 4 slots x 2 kv tiles = 64 VGPR, two
// buffers sE / sO by chunk parity) and one wave software-pipelines two chunks:
//   phase 1 of iteration c:  QK(c + 1) (48 WMMA)  ||  p = exp2(s c - m) of chunk c, V^T(c) reads (16 tr16)
//   phase 2 of iteration c:  PV(c) (32 WMMA)       ||  bf16 P(c), row sums of c, K(c + 2) reads (24 b128),
//                                                     masks + row max of chunk c + 1, rescale ballots
//   then (rare, uniform branch) the lazy rescale of O / l for chunk c + 1.
// Ring: 4 stages of 64 kv rows (K 64 x 400 B, V 64 x 272 B), wave 0 issues the 3 TDM ops of tile T + 3
// right after the wait of barrier B_{T+1} (iteration 2T); every wave signals B_{T+2} in iteration
// 2T + 1 after its last read of stage T (V(2T + 1)), wave 0 after tile T + 2 landed. Stage T + 1 is
// first read by the K(2T + 2) reads of iteration 2T's phase 2, after the wait of B_{T+1}.
// Masked chunks (any row of the block masked) carry the causal mask on all 4 slots (fully masked slots
// still run their WMMAs: p = 0).
#include "hk1.h"

using namespace hk1;

extern "C" __device__ void x6_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
#define X6_WAIT_DSCNT(n) x6_llvm_s_wait_dscnt((short) (n))

#ifndef X6_SGB
#define X6_SGB 1  // sched_group_barrier interleave patterns in the two phases
#endif
#ifndef X6_P1_VALU
#define X6_P1_VALU 2  // VALU per QK WMMA in phase 1
#endif
#ifndef X6_P2_VALU
#define X6_P2_VALU 3  // VALU per PV WMMA in phase 2
#endif
#ifndef X6_BURST
#define X6_BURST 1  // 1: all LDS reads of an iteration (V(c), K(c + 2)) as one burst at the head of phase 1
#endif

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int NW = 4;            // waves per workgroup
constexpr int BM = 256;          // q rows per workgroup
constexpr int QT = 4;            // 16-row q tiles per wave
constexpr int NB = 64;           // kv rows per ring stage (tile)
constexpr int NKC = 2;           // kv 16-row tiles per 32-kv chunk
constexpr int NKS = D_QK / 32;   // 6 k-steps of QK
constexpr int NDT = D_V / 16;    // 8 d tiles of O
constexpr int XK = 400, XV = 272;  // LDS row strides (bytes)
constexpr int K_B = NB * XK;       // 25600
constexpr int V_B = NB * XV;       // 17408
constexpr int STG_B = K_B + V_B;   // 43008
constexpr int NSTG = 4;
constexpr int Q_OFF = NSTG * STG_B;  // 172032
constexpr int LDS_B = Q_OFF + BM * XK;  // 274432
static_assert(LDS_B <= 327680 && 2 * LDS_B > 327680, "one workgroup per CU");
constexpr unsigned W0_A = 0x47510000u;  // 128 columns, pad 72 elements -> 400-B rows
constexpr unsigned W0_B = 0x87110000u;  // 64 columns, pad 136 elements -> 400-B rows
constexpr unsigned W0_V = 0x07510000u;  // 128 columns, pad 8 elements -> 272-B rows
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.69314718055994531f;
constexpr float BIG_NEG = -1.0e30f;
constexpr float THR = 1.0f * LOG2E;
constexpr float NEG_INF = -__builtin_inff();
// sched_group_barrier masks (LLVM IGroupLP)
constexpr int SG_VALU = 0x002, SG_SALU = 0x004, SG_WMMA = 0x008, SG_DSR = 0x100, SG_TRANS = 0x400;

typedef float f32x2 __attribute__((ext_vector_type(2)));
template <int N> using ic = std::integral_constant<int, N>;

HK1_INLINE f32x8 zero8() { return f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; }
HK1_INLINE float peer(float x) {
  const unsigned u = __builtin_bit_cast(unsigned, x);
  return __builtin_bit_cast(float, (unsigned)__builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xFEDCBA98u, false, false));
}
HK1_INLINE float fmax2(float a, float b) { return __builtin_fmaxf(a, b); }
HK1_INLINE float fmax3(float a, float b, float c) { return __builtin_fmaxf(__builtin_fmaxf(a, b), c); }

HK1_INLINE i32x8 tdm_g1(unsigned w0, unsigned tdim1, unsigned tile0, unsigned tile1, unsigned long long stride_el) {
  return i32x8{(int)w0, (int)0xFFFF0000u, (int)(((tdim1 & 0xFFFFu) << 16) | 0x7FFFu),
               (int)(((tdim1 >> 16) & 0xFFFFu) | (tile0 << 16)), (int)tile1, (int)(unsigned)stride_el,
               (int)((unsigned)(stride_el >> 32) & 0xFFFFu), 0};
}
HK1_INLINE i32x4 tdm_g0(unsigned lds, unsigned long long ga, int count = 1) {
  // count 0 = the NULL descriptor (HK tdm.cuh): moves nothing; the branch-free way to keep a TDM op off a wave
  return i32x4{count, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | 0x80000000u)};
}
HK1_INLINE void wg_sync() {
  sched_barrier<0>();
  X6_WAIT_DSCNT(0);
  __syncthreads();
  sched_barrier<0>();
}
HK1_INLINE void bar_signal() {  // this wave's LDS reads retired, release, arrive
  sched_barrier<0>();
  X6_WAIT_DSCNT(0);
  __builtin_amdgcn_fence(MO_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier_signal(-1);
  sched_barrier<0>();
}
HK1_INLINE void bar_wait() {
  sched_barrier<0>();
  __builtin_amdgcn_s_barrier_wait(-1);
  __builtin_amdgcn_fence(MO_ACQUIRE, "workgroup", "local");
  sched_barrier<0>();
}

HK1_INLINE void block_ids(int gx, int gy, int gz, int& bx, int& by, int& bz) {
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

// phase-1 interleave: 48 QK WMMA; the 16 V tr16 + 24 K b128 reads one per WMMA (V first); X6_P1_VALU VALU per WMMA
HK1_INLINE void sgb_phase1() {
#if X6_SGB && X6_BURST
  static_for<48>([&](auto) {
    sched_group_barrier<SG_WMMA, 1, 0>();
    sched_group_barrier<SG_VALU | SG_TRANS, X6_P1_VALU, 0>();
  });
#elif X6_SGB
  static_for<40>([&](auto) {
    sched_group_barrier<SG_WMMA, 1, 0>();
    sched_group_barrier<SG_DSR, 1, 0>();
    sched_group_barrier<SG_VALU | SG_TRANS, X6_P1_VALU, 0>();
  });
  static_for<8>([&](auto) {
    sched_group_barrier<SG_WMMA, 1, 0>();
    sched_group_barrier<SG_VALU | SG_TRANS, X6_P1_VALU, 0>();
  });
#endif
}
// phase-2 interleave: 32 PV WMMA, X6_P2_VALU VALU per WMMA
HK1_INLINE void sgb_phase2() {
#if X6_SGB
  static_for<32>([&](auto) {
    sched_group_barrier<SG_WMMA, 1, 0>();
    sched_group_barrier<SG_VALU | SG_TRANS, X6_P2_VALU, 0>();
  });
#endif
}
}  // namespace

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(128, 128), amdgpu_waves_per_eu(1, 1)))
void k_mla_fwd(const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
               bf16* __restrict__ O, float* __restrict__ LSE, float scale, int Sq, int Skv, int H, int B,
               int nqb) {
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
  const int n = (qv + cshift + NB - 1) / NB;                 // 64-kv tiles (ring stages) of the block
  const int ncc = 2 * min((q0 + cshift + 1) / NB, n);        // chunks [0, ncc) need no mask
  const int NC = 2 * n;                                      // 32-kv chunks
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
  // branch-free in-loop issue (every wave executes it, waves 1..3 with NULL descriptors): tile t into stage
  // t % 4, or -- past the last tile -- a 1-row copy of tile n - 1 into that (free) stage, so wave 0 keeps exactly
  // 3 ops per tile in flight and its tensor waits stay static.
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

  // ---- prologue: tiles 0..2 in flight, Q staged; B_0 = Q and tile 0 visible ------------------------------
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
  wg_sync();

  bf16x16 qf[QT][NKS];
  {
    const unsigned qlane = lds0 + Q_OFF + (unsigned)((16 * wv + (lane & 15)) * XK + (lane >> 4) * 16);
    static_for<QT>([&](auto i) {
      const lds_u8* qp = lds_ptr(qlane + (unsigned)(i * 64 * XK));
      static_for<NKS>([&](auto ks) {
        qf[i][ks] = cat(lds_load_b128(qp, ks * 64), lds_load_b128(qp, ks * 64 + 32));
      });
    });
  }
  f32x8 o[QT][NDT];
  float m[QT], l[QT];
  static_for<QT>([&](auto i) {
    m[i] = BIG_NEG;
    l[i] = 0.f;
    static_for<NDT>([&](auto d) { o[i][d] = zero8(); });
  });

  const unsigned k_lane = (unsigned)((lane & 15) * XK + (lane >> 4) * 16);
  const unsigned v_lane = (unsigned)((((lane >> 4) * 8) + (lane & 7)) * XV + ((lane >> 3) & 1) * 16);
  // causal: element e of kv tile kt of chunk c (kv = 32 c + 16 kt + 8 (lane / 16) + e) is masked for slot i
  // iff 16 kt + e > dl0 + 64 i - 32 c
  const int dl0 = q0 + 16 * wv + (lane & 15) + cshift - (lane >> 4) * 8;

  auto stage_of = [&](int c) { return lds0 + (unsigned)(((c >> 1) & (NSTG - 1)) * STG_B); };
  bf16x16 kA[NKC][NKS], kB[NKC][NKS];  // K fragments: odd chunks in kA, even chunks in kB
  auto load_k = [&](int c, bf16x16 (&kf)[NKC][NKS]) __attribute__((always_inline)) {
    const lds_u8* kb = lds_ptr(stage_of(c) + k_lane + (unsigned)((c & 1) * 32 * XK));
    static_for<NKS>([&](auto ks) {
      static_for<NKC>([&](auto kt) {
        kf[kt][ks] = cat(lds_load_b128(kb, kt * 16 * XK + ks * 64), lds_load_b128(kb, kt * 16 * XK + ks * 64 + 32));
      });
    });
  };
  auto load_v = [&](int c, bf16x16 (&vf)[NDT]) __attribute__((always_inline)) {
    const lds_u8* vb = lds_ptr(stage_of(c) + K_B + v_lane + (unsigned)((c & 1) * 32 * XV));
    static_for<NDT>([&](auto dt) { vf[dt] = cat(lds_tr16(vb, dt * 32), lds_tr16(vb, 16 * XV + dt * 32)); });
  };
  auto qk = [&](const bf16x16 (&kf)[NKC][NKS], f32x8 (&s)[QT][NKC]) __attribute__((always_inline)) {
    static_for<NKS>([&](auto ks) {
      static_for<NKC>([&](auto kt) {
        static_for<QT>([&](auto i) {
          if constexpr (ks == 0) {
            s[i][kt] = wmma<false, false>(kf[kt][0], qf[i][0], zero8());
          } else {
            s[i][kt] = wmma<false, false>(kf[kt][ks], qf[i][ks], s[i][kt]);
          }
        });
      });
    });
  };
  float rmax[QT];
  bool resc[QT];
  // masks (MASK) + row max of chunk c, rescale ballots
  auto rowmax = [&](int c, f32x8 (&s)[QT][NKC], auto mask_c) __attribute__((always_inline)) {
    static_for<QT>([&](auto i) {
      if constexpr (decltype(mask_c)::value) {
        const int dl = dl0 + 64 * i - 32 * c;
        static_for<NKC>([&](auto kt) {
          static_for<8>([&](auto e) { s[i][kt][(int)e] = (kt * 16 + e > dl) ? NEG_INF : s[i][kt][(int)e]; });
        });
      }
      const float a = fmax3(s[i][0][0], s[i][0][1], s[i][0][2]);
      const float b = fmax3(s[i][0][3], s[i][0][4], s[i][0][5]);
      const float cc = fmax3(s[i][0][6], s[i][0][7], s[i][1][0]);
      const float d = fmax3(s[i][1][1], s[i][1][2], s[i][1][3]);
      const float e = fmax3(s[i][1][4], s[i][1][5], s[i][1][6]);
      const float x = fmax3(a, b, cc);
      const float y = fmax3(d, e, s[i][1][7]);
      float mx = fmax2(x, y);
      mx = fmax2(mx, peer(mx));
      rmax[i] = mx * c2;
      resc[i] = __builtin_amdgcn_ballot_w32((rmax[i] - m[i]) > THR) != 0u;
    });
  };
  auto rescale = [&]() __attribute__((always_inline)) {
    const bool any = resc[0] | resc[1] | resc[2] | resc[3];
    if (any) {
      static_for<QT>([&](auto i) {  // corr = 1 exactly on the slots that keep m
        const float mn = resc[i] ? fmax2(m[i], rmax[i]) : m[i];
        const float corr = exp2_raw(m[i] - mn);
        m[i] = mn;
        l[i] = l[i] * corr;
        static_for<NDT>([&](auto d) { o[i][d] = o[i][d] * corr; });
      });
    }
  };
  // p = exp2(s c - m) in place (fp32)
  auto expo = [&](f32x8 (&s)[QT][NKC]) __attribute__((always_inline)) {
    const f32x2 cc = {c2, c2};
    static_for<QT>([&](auto i) {
      const f32x2 nm = {-m[i], -m[i]};
      static_for<NKC>([&](auto kt) {
        static_for<4>([&](auto j) {
          const f32x2 sv = {s[i][kt][2 * j], s[i][kt][2 * j + 1]};
          const f32x2 av = __builtin_elementwise_fma(sv, cc, nm);
          s[i][kt][2 * j] = exp2_raw(av[0]);
          s[i][kt][2 * j + 1] = exp2_raw(av[1]);
        });
      });
    });
  };
  // bf16 P^T, O^T += V^T P^T, row sums -> l
  auto pv_sum = [&](f32x8 (&p)[QT][NKC], const bf16x16 (&vf)[NDT]) __attribute__((always_inline)) {
    static_for<QT>([&](auto i) {
      const bf16x16 pb = cat(to_bf16x8(p[i][0]), to_bf16x8(p[i][1]));
      static_for<NDT>([&](auto dt) { o[i][dt] = wmma<false, false>(vf[dt], pb, o[i][dt]); });
    });
    static_for<2>([&](auto pr) {  // slots (2 pr, 2 pr + 1) as packed pairs
      constexpr int i0 = 2 * pr, i1 = 2 * pr + 1;
      f32x2 t8[8];
      static_for<8>([&](auto j) {
        const f32x2 a = {p[i0][j / 4][2 * (j % 4)], p[i1][j / 4][2 * (j % 4)]};
        const f32x2 b = {p[i0][j / 4][2 * (j % 4) + 1], p[i1][j / 4][2 * (j % 4) + 1]};
        t8[j] = a + b;
      });
      const f32x2 t40 = t8[0] + t8[1], t41 = t8[2] + t8[3], t42 = t8[4] + t8[5], t43 = t8[6] + t8[7];
      const f32x2 sm = (t40 + t41) + (t42 + t43);
      l[i0] = l[i0] + (sm[0] + peer(sm[0]));
      l[i1] = l[i1] + (sm[1] + peer(sm[1]));
    });
  };

  // ---- one pipelined iteration on chunk c (sc = S(c), max done; sn <- S(c + 1)) ------------------------------
  // kcur = K(c + 1) (consumed by QK(c + 1)), knext <- K(c + 2) (read in phase 1, from stage T + 1)
  auto iter = [&](int c, f32x8 (&sc)[QT][NKC], f32x8 (&sn)[QT][NKC], bf16x16 (&kcur)[NKC][NKS],
                  bf16x16 (&knext)[NKC][NKS], auto odd_c, auto next_c, auto mask_c) __attribute__((always_inline)) {
    constexpr bool ODD = decltype(odd_c)::value != 0;
    constexpr bool NEXT = decltype(next_c)::value != 0;
    const int T = c >> 1;
    if constexpr (!ODD) {
      bar_wait();            // B_{T+1}: tile T + 1 visible, stage T - 1 free
      issue_tile_bf(T + 3);  // branch-free: every wave runs the same straight-line code
    }
    sched_barrier<0>();
    // phase 1
    bf16x16 vf[NDT];
#if X6_BURST
    load_v(c, vf);
    if constexpr (NEXT) load_k(min(c + 2, NC - 1), knext);
    sched_barrier<0>();
    expo(sc);
    if constexpr (NEXT) {
      qk(kcur, sn);
      sgb_phase1();
    }
#else
    load_v(c, vf);
    expo(sc);
    if constexpr (NEXT) {
      qk(kcur, sn);
      load_k(min(c + 2, NC - 1), knext);
      sgb_phase1();
    }
#endif
    sched_barrier<0>();
    if constexpr (ODD && NEXT) {
      tensor_wait<3>();      // wave 0: tile T + 2 landed (T + 3 may stay in flight)
      bar_signal();          // B_{T+2}: this wave is done with stage T
    }
    sched_barrier<0>();
    // phase 2
    pv_sum(sc, vf);
    if constexpr (NEXT) {
      rowmax(c + 1, sn, mask_c);
      sgb_phase2();
    }
    sched_barrier<0>();
    if constexpr (NEXT) rescale();
  };

  f32x8 sE[QT][NKC], sO[QT][NKC];
  // chunk 0: QK, max, rescale; K(1)
  load_k(0, kB);
  qk(kB, sE);
  sched_barrier<0>();
  rowmax(0, sE, ic<1>{});  // the mask is a no-op on a clean chunk
  rescale();
  load_k(min(1, NC - 1), kA);
  tensor_wait<3>();  // wave 0: tile 1 landed
  bar_signal();      // B_1

  // pairs (2T, 2T + 1) whose next chunks 2T + 1, 2T + 2 are both clean; then the masked pairs (the first may
  // mask a clean chunk: a no-op); one variant per loop, so no register shuffles at the back edges
  const int tA = max(ncc / 2 - 1, 0);
  int T = 0;
  for (; T < tA; ++T) {
    iter(2 * T, sE, sO, kA, kB, ic<0>{}, ic<1>{}, ic<0>{});
    iter(2 * T + 1, sO, sE, kB, kA, ic<1>{}, ic<1>{}, ic<0>{});
  }
  for (; T < n - 1; ++T) {
    iter(2 * T, sE, sO, kA, kB, ic<0>{}, ic<1>{}, ic<1>{});
    iter(2 * T + 1, sO, sE, kB, kA, ic<1>{}, ic<1>{}, ic<1>{});
  }
  iter(NC - 2, sE, sO, kA, kB, ic<0>{}, ic<1>{}, ic<1>{});  // chunk NC - 1 is always masked (ncc <= NC - 2)
  iter(NC - 1, sO, sE, kB, kA, ic<1>{}, ic<0>{}, ic<0>{});
  tensor_wait<0>();  // no TDM write may land after the workgroup retires

  // ---- epilogue: O = O^T / l, lse = (m + log2 l) ln 2 ------------------------------------------------------
  const auto r_o = rsrc(O, (int)min((long long)B * Sq * H * D_V * 2, 0x7FFFFFFFll));
  const auto r_l = rsrc(LSE, (int)min((long long)B * H * Sq * 4, 0x7FFFFFFFll));
  static_for<QT>([&](auto i) {
    const int q = q0 + 16 * (wv + 4 * i) + (lane & 15);
    if (q < Sq) {
      const float inv = l[i] > 0.f ? 1.0f / l[i] : 0.f;
      const int obase = (int)((((long long)bb * Sq + q) * H + hh) * D_V + (lane >> 4) * 8) * 2;
      static_for<NDT>([&](auto d) { buf_store_b128(to_bf16x8(o[i][d] * inv), r_o, obase + d * 32, 0); });
      if ((lane >> 4) == 0) {
        const float lse = (m[i] + __builtin_log2f(l[i])) * LN2;
        __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, lse), r_l,
                                              (int)((((long long)bb * H + hh) * Sq + q) * 4), 0, 0);
      }
    }
  });
}
