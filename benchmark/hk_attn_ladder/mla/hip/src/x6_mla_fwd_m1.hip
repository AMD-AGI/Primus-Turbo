/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_mla_fwd (x6 M1) -- HIP forward of the DeepSeek-V3 MLA attention on gfx1250:
//   qk 192 / v 128, bf16, MHA, bottom-right causal (query i sees kv j <= i + Skv - Sq).
//   Q [B, Sq, H, 192], K [B, Skv, H, 192], V [B, Skv, H, 128] -> O [B, Sq, H, 128] bf16,
//   LSE [B, H, Sq] fp32 natural log. The FlyDSL forward's math: fp32 online softmax with
//   p = exp2(s * scale * log2e - m), m in log2 units, lazy rescale (threshold 1 nat, wave-uniform
//   per 16-row q tile), lse = (m + log2 l) * ln 2.
//
// grid (ceil(Sq / 256), H, B) through FlyDSL's longest-first / head-grouped block remap; block 128 =
// 4 waves, 1 wave per SIMD, one workgroup per CU (LDS 274432 B). Wave w owns the 16-row q tiles
// {w, w + 4, w + 8, w + 12} of the 256-row block (slot i = tile w + 4 i), so every wave runs the same
// number of 64-kv tiles; on the 4 diagonal tiles of an aligned block slot i < u is fully masked (its
// WMMAs are skipped), i == u carries the causal mask and i > u is clean.
//
// GEMM orientation (FlyDSL's): S^T = K Q^T (A = K rows from LDS, ds_load_b128; B = Q^T fragments
// resident in VGPRs), the S^T accumulator converted to bf16 is directly the B operand P^T of
// O^T += V^T P^T (A = V^T via ds_load_tr16_b128).
//
// Ring: 4 stages of (K 64 x 400 B, V 64 x 272 B), wave 0 issues the 3 TDM ops of tile t + 3 after the
// barrier of tile t; one workgroup barrier per tile.
#include "hk1.h"

using namespace hk1;

extern "C" __device__ void x6_llvm_s_wait_dscnt(short) __asm("llvm.amdgcn.s.wait.dscnt");
#define X6_WAIT_DSCNT(n) x6_llvm_s_wait_dscnt((short) (n))

#ifndef X6_EXPERT
#define X6_EXPERT 0
#endif

namespace {
constexpr int D_QK = 192, D_V = 128;
constexpr int NW = 4;            // waves per workgroup
constexpr int BM = 256;          // q rows per workgroup
constexpr int QT = 4;            // 16-row q tiles per wave
constexpr int NB = 64;           // kv rows per tile
constexpr int NKT = NB / 16;     // 4 kv 16-row tiles
constexpr int NKS = D_QK / 32;   // 6 k-steps of QK
constexpr int NDT = D_V / 16;    // 8 d tiles of O
constexpr int NPS = NB / 32;     // 2 k-steps of PV
constexpr int XK = 400, XV = 272;  // LDS row strides (bytes)
constexpr int K_B = NB * XK;       // 25600
constexpr int V_B = NB * XV;       // 17408
constexpr int STG_B = K_B + V_B;   // 43008
constexpr int NSTG = 4;
constexpr int Q_OFF = NSTG * STG_B;  // 172032
constexpr int LDS_B = Q_OFF + BM * XK;  // 274432
static_assert(LDS_B <= 327680 && 2 * LDS_B > 327680, "one workgroup per CU");
// D# group-1 word 0: bf16, pad on, interval / amount (HK tdm.cuh build_tdm_descriptor_2d encoding)
constexpr unsigned W0_A = 0x47510000u;  // 128 columns, pad 72 elements -> 400-B rows
constexpr unsigned W0_B = 0x87110000u;  // 64 columns, pad 136 elements -> 400-B rows
constexpr unsigned W0_V = 0x07510000u;  // 128 columns, pad 8 elements -> 272-B rows
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.69314718055994531f;
constexpr float BIG_NEG = -1.0e30f;
constexpr float THR = 1.0f * LOG2E;  // rescale threshold (1 nat) in log2 units
constexpr float NEG_INF = -__builtin_inff();

typedef float f32x2 __attribute__((ext_vector_type(2)));

HK1_INLINE f32x8 zero8() { return f32x8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; }

HK1_INLINE float peer(float x) {  // the value of lane l ^ 16
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
HK1_INLINE i32x4 tdm_g0(unsigned lds, unsigned long long ga) {
  return i32x4{1, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | 0x80000000u)};
}

HK1_INLINE void wg_sync() {
  sched_barrier<0>();
  X6_WAIT_DSCNT(0);
  __syncthreads();
  sched_barrier<0>();
}

// FlyDSL fwd_kernel._block_id: longest-first, KV_GROUP 16 q blocks x KV_HEAD_CHUNK 16 heads together
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
  const int qv = min(q0 + BM, Sq);                       // end of the block's valid rows
  const int n_tiles = (qv + cshift + NB - 1) / NB;         // tiles holding a key some row attends
  const int n_clean = min((q0 + cshift + 1) / NB, n_tiles);  // tiles no row of the block masks
  const bool aligned = (cshift % NB) == 0;
  const float c2 = scale * LOG2E;

  const unsigned long long krow = (unsigned long long)H * D_QK;  // K/Q row stride (elements)
  const unsigned long long vrow = (unsigned long long)H * D_V;
  const unsigned long long gk0 = (unsigned long long)(uintptr_t)K + 2ull * (((unsigned long long)bb * Skv * H + hh) * D_QK);
  const unsigned long long gv0 = (unsigned long long)(uintptr_t)V + 2ull * (((unsigned long long)bb * Skv * H + hh) * D_V);
  const unsigned long long gq0 =
      (unsigned long long)(uintptr_t)Q + 2ull * ((((unsigned long long)bb * Sq + q0) * H + hh) * D_QK);

  // ---- TDM: the 3 ops of kv tile t into stage t % 4 (wave 0 only) ----------------------------------------
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

  if (wv == 0) {
    issue_tile(0);
    if (n_tiles > 1) issue_tile(1);
    if (n_tiles > 2) issue_tile(2);
  } else if (wv == 1) {
    tdm_load(tdm_g0(lds0 + Q_OFF, gq0), tdm_g1(W0_A, (unsigned)(Sq - q0), 128, BM, krow));
    tensor_wait<0>();
  } else if (wv == 2) {
    tdm_load(tdm_g0(lds0 + Q_OFF + 256, gq0 + 256), tdm_g1(W0_B, (unsigned)(Sq - q0), 64, BM, krow));
    tensor_wait<0>();
  }
  wg_sync();

  // ---- resident Q^T fragments (B operand): slot i = q tile wv + 4 i -----------------------------------
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
  // causal: element e of kv tile kt (kv = 64 t + 16 kt + 8 (lane / 16) + e) is masked for slot i iff
  // 16 kt + e > dl0 + 64 (i - t)
  const int dl0 = q0 + 16 * wv + (lane & 15) + cshift - (lane >> 4) * 8;

  // ---- one 64-kv tile: ULO = first live slot, MM = 0 no mask / 1 mask slot ULO / 2 mask every live slot ---
  auto tile = [&](int t, auto ulo_c, auto mm_c) __attribute__((always_inline)) {
    constexpr int ULO = decltype(ulo_c)::value;
    constexpr int MM = decltype(mm_c)::value;
    const unsigned st = lds0 + (unsigned)((t & (NSTG - 1)) * STG_B);
    const lds_u8* kb = lds_ptr(st + k_lane);
    const lds_u8* vb = lds_ptr(st + K_B + v_lane);

    // S^T = K Q^T: s[i][kt] (lane: q column lane % 16, kv rows 16 kt + 8 (lane / 16) + [0, 8))
    f32x8 s[QT][NKT];
    static_for<NKS>([&](auto ks) {
      bf16x16 kf[NKT];
      static_for<NKT>([&](auto kt) {
        kf[kt] = cat(lds_load_b128(kb, kt * 16 * XK + ks * 64), lds_load_b128(kb, kt * 16 * XK + ks * 64 + 32));
      });
      static_for<NKT>([&](auto kt) {
        static_for<QT>([&](auto i) {
          if constexpr (i >= ULO) {
            if constexpr (ks == 0) {
              s[i][kt] = wmma<false, false>(kf[kt], qf[i][0], zero8());
            } else {
              s[i][kt] = wmma<false, false>(kf[kt], qf[i][ks], s[i][kt]);
            }
          }
        });
      });
    });

    // V^T fragments (A operand of PV): d tile dt, kv step ps
    bf16x16 vf[NDT][NPS];
    static_for<NDT>([&](auto dt) {
      static_for<NPS>([&](auto ps) {
        vf[dt][ps] = cat(lds_tr16(vb, (ps * 32) * XV + dt * 32), lds_tr16(vb, (ps * 32 + 16) * XV + dt * 32));
      });
    });

    // softmax part 1: mask, row max, wave-uniform rescale flags
    float rmax[QT];
    bool resc[QT];
    static_for<QT>([&](auto i) {
      if constexpr (i >= ULO) {
        constexpr bool DO_MASK = (MM == 2) || (MM == 1 && i == ULO);
        if constexpr (DO_MASK) {
          const int dl = dl0 + 64 * (i - t);
          static_for<NKT>([&](auto kt) {
            static_for<8>([&](auto e) {
              s[i][kt][(int)e] = (kt * 16 + e > dl) ? NEG_INF : s[i][kt][(int)e];
            });
          });
        }
        float a = fmax3(s[i][0][0], s[i][0][1], s[i][0][2]);
        float b = fmax3(s[i][0][3], s[i][0][4], s[i][0][5]);
        float c = fmax3(s[i][0][6], s[i][0][7], s[i][1][0]);
        float d = fmax3(s[i][1][1], s[i][1][2], s[i][1][3]);
        float e = fmax3(s[i][1][4], s[i][1][5], s[i][1][6]);
        float f = fmax3(s[i][1][7], s[i][2][0], s[i][2][1]);
        float g = fmax3(s[i][2][2], s[i][2][3], s[i][2][4]);
        float h = fmax3(s[i][2][5], s[i][2][6], s[i][2][7]);
        float x = fmax3(s[i][3][0], s[i][3][1], s[i][3][2]);
        float y = fmax3(s[i][3][3], s[i][3][4], s[i][3][5]);
        float z = fmax2(s[i][3][6], s[i][3][7]);
        a = fmax3(a, b, c);
        d = fmax3(d, e, f);
        g = fmax3(g, h, x);
        y = fmax2(y, z);
        a = fmax3(a, d, g);
        a = fmax2(a, y);
        a = fmax2(a, peer(a));
        rmax[i] = a * c2;
        resc[i] = __builtin_amdgcn_ballot_w32((rmax[i] - m[i]) > THR) != 0u;
      }
    });
    bool any = false;
    static_for<QT>([&](auto i) {
      if constexpr (i >= ULO) any = any | resc[i];
    });
    if (any) {
      static_for<QT>([&](auto i) {
        if constexpr (i >= ULO) {
          const float mn = resc[i] ? fmax2(m[i], rmax[i]) : m[i];
          const float corr = exp2_raw(m[i] - mn);
          m[i] = mn;
          l[i] = l[i] * corr;
          if (resc[i]) {
            static_for<NDT>([&](auto d) { o[i][d] = o[i][d] * corr; });
          }
        }
      });
    }

    // softmax part 2: p = exp2(s c - m), row sums, bf16 P^T
    bf16x16 pf[QT][NPS];
    float ps_[QT];
    static_for<QT>([&](auto i) {
      if constexpr (i >= ULO) {
        const f32x2 cc = {c2, c2};
        const f32x2 nm = {-m[i], -m[i]};
        float p[32];
        static_for<16>([&](auto j) {
          const f32x2 sv = {s[i][(2 * j) / 8][(2 * j) % 8], s[i][(2 * j + 1) / 8][(2 * j + 1) % 8]};
          const f32x2 av = __builtin_elementwise_fma(sv, cc, nm);
          p[2 * j] = exp2_raw(av[0]);
          p[2 * j + 1] = exp2_raw(av[1]);
        });
        static_for<NPS>([&](auto q) {
          f32x8 lo, hi;
          static_for<8>([&](auto e) {
            lo[(int)e] = p[q * 16 + e];
            hi[(int)e] = p[q * 16 + 8 + e];
          });
          pf[i][q] = cat(to_bf16x8(lo), to_bf16x8(hi));
        });
        // sum tree (pairs, then 3-ary)
        float t16[16];
        static_for<16>([&](auto j) { t16[j] = p[2 * j] + p[2 * j + 1]; });
        float t8[8];
        static_for<8>([&](auto j) { t8[j] = t16[2 * j] + t16[2 * j + 1]; });
        float t4[4];
        static_for<4>([&](auto j) { t4[j] = t8[2 * j] + t8[2 * j + 1]; });
        const float sm = (t4[0] + t4[1]) + (t4[2] + t4[3]);
        ps_[i] = sm + peer(sm);
        l[i] = l[i] + ps_[i];
      }
    });

    // O^T += V^T P^T
    static_for<NDT>([&](auto dt) {
      static_for<NPS>([&](auto q) {
        static_for<QT>([&](auto i) {
          if constexpr (i >= ULO) o[i][dt] = wmma<false, false>(vf[dt][q], pf[i][q], o[i][dt]);
        });
      });
    });
  };

  for (int t = 0; t < n_tiles; ++t) {
    if (wv == 0) {
      if (t + 2 < n_tiles) {
        tensor_wait<6>();
      } else if (t + 1 < n_tiles) {
        tensor_wait<3>();
      } else {
        tensor_wait<0>();
      }
    }
    wg_sync();
    if (wv == 0 && t + 3 < n_tiles) issue_tile(t + 3);
    if (t < n_clean) {
      tile(t, std::integral_constant<int, 0>{}, std::integral_constant<int, 0>{});
    } else if (aligned) {
      const int u = t - n_clean;
      if (u == 0) {
        tile(t, std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{});
      } else if (u == 1) {
        tile(t, std::integral_constant<int, 1>{}, std::integral_constant<int, 1>{});
      } else if (u == 2) {
        tile(t, std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
      } else {
        tile(t, std::integral_constant<int, 3>{}, std::integral_constant<int, 1>{});
      }
    } else {
      tile(t, std::integral_constant<int, 0>{}, std::integral_constant<int, 2>{});
    }
  }

  // ---- epilogue: O = O^T / l (bf16, 16 B per lane per (slot, d tile)), lse = (m + log2 l) ln 2 ----------
  const auto r_o = rsrc(O, (int)min((long long)B * Sq * H * D_V * 2, 0x7FFFFFFFll));
  const auto r_l = rsrc(LSE, (int)min((long long)B * H * Sq * 4, 0x7FFFFFFFll));
  static_for<QT>([&](auto i) {
    const int q = q0 + 16 * (wv + 4 * i) + (lane & 15);
    if (q < Sq) {
      const float inv = l[i] > 0.f ? 1.0f / l[i] : 0.f;
      const int obase = (int)((((long long)bb * Sq + q) * H + hh) * D_V + (lane >> 4) * 8) * 2;
      static_for<NDT>([&](auto d) {
        buf_store_b128(to_bf16x8(o[i][d] * inv), r_o, obase + d * 32, 0);
      });
      if ((lane >> 4) == 0) {
        const float lse = (m[i] + __builtin_log2f(l[i])) * LN2;
        __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, lse), r_l,
                                              (int)((((long long)bb * H + hh) * Sq + q) * 4), 0, 0);
      }
    }
  });
}
