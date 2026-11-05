/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_dkdv96ds -- dK and dV of the DeepSeek-V3 MLA attention backward on gfx1250 (qk 192 / v 128, bf16, bottom-right
// causal: query q sees kv <= q + cshift, cshift = Skv - Sq), plus the dS scratch: every bf16 dS^T tile the kernel
// feeds its dK WMMA is also stored into DS (dslayout.h), where k_dqds_c reads it back for dQ.
//   Q [B, Sq, Hq, 192], dO [B, Sq, Hq, 128], K / dK [B, Skv, Hkv, 192], V / dV [B, Skv, Hkv, 128] bf16; NL, ND
//   [B, Hq, Sq] fp32 (k_delta's row constants); G = Hq / Hkv q heads per kv head.
//
// grid (HG, k3_nblk(Skv), B*Hkv/HG), block 64 = 2 waves (one per SIMD). A work item is one kv block of one kv head:
// blocks [0, n96) are 96-row blocks (wave w owns the 48 kv rows from 96 bid + 48 w: NKV 3 sub-tiles of 16), the 0..2
// blocks after them 64-row tail blocks (NKV 2). The two heights are two bodies behind a uniform branch on the block
// index (built with -structurizecfg-skip-uniform-regions so they stay separate regions). Ascending blocks = longest
// first under bottom-right causal.
//
// A workgroup runs one sequence of query tiles (q pairs of 32 queries x the G q heads) over its block: first the
// n_mask tiles that straddle the diagonal (causal mask on), then the n_full tiles below it. The Q / dO tiles go
// through a 3-stage LDS ring filled by the TDM (a stage = [32][D_V] dO + [32][D_QK] Q, each wave moving its 16
// rows); K / V of the wave's rows stay in registers (the S / dP B operands), dV / dK accumulate in registers and
// leave through per-wave row-major images in LDS (whole-line global stores). Per tile and wave (NKV 3): S = Q K^T
// 2 hh x 3 kh x 6 = 36 WMMAs, dP = dO V^T 2 x 3 x 4 = 24, dV^T += dO^T P 3 x 8 = 24, dK^T += Q^T dS 3 x 12 = 36:
// 120 WMMAs on 40 ds_load_b128 (the S / dP A operands Q, dO) + 40 ds_load_tr16_b128 (the dV / dK A operands
// dO^T, Q^T). Every dV / dK accumulator takes one WMMA per tile, in sequence order; a masked entry contributes exact
// zeros (P = exp2(-3e38) = +0, dS = +0 * x).
//
// HKGEN selects the build:
//   1 (manifest: the shipped code object): the marker shell. LLVM compiles the work-item fetch, the descriptors,
//     the first two ring tiles, the barriers around the tile loop and the epilogue's addressing; inline-asm markers
//     (; HKGEN INIT / LOOP / FINAL nkv=...) stand where the per-tile work goes, which tools/bwdgen generates
//     (numerics: tools/bwdgen/bwdgen.py) and hip/asm/k_dkdv96ds.s.patch splices in.
//   0: the reference, the whole kernel in HIP. tools/bwdgen compiles it to learn the VGPR-MSB operand-field map;
//     it is not shipped. Its softmax, P = exp2(c1 S + NL) and dS = P (scale dP + ND), expects the unfolded row
//     constants NL = -log2(e) lse and ND = -scale delta; the shipped k_delta folds them for the generated code
//     (NL += log2(scale), ND = -delta).
//
// Ring protocol (both waves run one program: the wave index only offsets addresses; one barrier per tile, so per
// wave 1 + N + 1 barriers). Tile j of the sequence lives in stage sigma(j) = (j - n_mask) mod 3; every prefetch is
// clamped to the last tile T_{N-1}.
//   prologue: TDM T_0, T_1 -> tensorcnt 3 -> BARRIER -> readback of sigma(0) (NKV 3: hh0 only) -> NKV 2: TDM T_2
//     | NKV 3: T_2 becomes the pending prefetch.
//   NKV 2 tile j (stage cur): S / dP; tr16 of cur; dV / dK kh0; tensorcnt 3 (this wave's half of T_{j+1}); BARRIER;
//     readback of the next stage; TDM of T_{j+3} into cur; dV / dK kh1.
//   NKV 3 tile j: readback hh1 of cur, then the pending TDM (T_{j+2}, into the stage whose last reads preceded the
//     previous BARRIER); S / dP; tr16 dO^T, Q^T of cur; prep of T_{j+3} -> cur (pending: issued in the next tile,
//     never after the last); tensorcnt 3; BARRIER (split: arrive, 24 dK WMMAs, wait); readback hh0 of the next
//     stage. The marker shell's generated LOOP follows the same protocol (the TDM of T_{j+2} at the top of tile j).
//   So every DS read of stage cur precedes the tile's BARRIER and the TDM into cur follows it (WAR); each wave's
//   half of the next tile has landed before the wave signals the BARRIER, so both halves are readable after it (RAW).
//   exit: tensorcnt 0 -> BARRIER -> the epilogue images overwrite the ring.
//
// Register plan of the NKV 3 reference body (accumulators 480 VGPRs, K / V fragments 240):
//   (1) the S / dP operands are carried per hh: only hh0's 20 b128 (80 VGPRs) cross the dV / dK phase (read back
//       after the BARRIER); hh1's 20 are read from the current stage at the top of the next tile, as one burst
//       under hh0's S / dP WMMAs (the stage is not overwritten before the tile's own BARRIER).
//   (2) the dV / dK A operands come in two tr16 bursts: dO^T (16 tr16, 64 VGPRs) -> the 24 dV WMMAs -> Q^T (24
//       tr16, 96 VGPRs, in the dead dO^T registers) -> the 36 dK WMMAs.
#include "dkdv_prims.h"
#include "dkdv_idx.h"

using namespace hk1;

static_assert(DSL_VERSION == 2 && DSL_SCALED == 1, "k_dkdv96ds writes dslayout v2 with the scale inside dS");

namespace {
constexpr int NDT_QK = K1_D_QK / 32;   // 6: WMMA k-steps of S = Q K^T
constexpr int NDT_V = K1_D_V / 32;     // 4: WMMA k-steps of dP = dO V^T
constexpr int NDO_V = K1_D_V / 16;     // 8: dV^T output tiles across d
constexpr int NDO_QK = K1_D_QK / 16;   // 12: dK^T output tiles across d
constexpr int NQE = 2 * NDT_QK;        // 12 Q readback entries per hh
constexpr int NHB = NQE + 2 * NDT_V;   // 20 entries per hh
constexpr int NRB = 2 * NHB;           // 40 readback entries of a stage
constexpr int NLD = 8;                 // carried NL/ND vec4 entries
constexpr int KH0_SALU = 2;            // SALU instructions per WMMA of a TDM-prep run
#ifndef HKGEN
#define HKGEN 0        // 1: the marker shell (manifest), 0: the reference (see above)
#endif
// one wave's K / V staging area of the marker shell, after the ring: 16 * K3_NKV rows of V (K1_XV), then of K (K1_XK)
#define K3_KV_W (16 * K3_NKV * (K1_XV + K1_XK))
// SGPRs the generated code may use as temporaries (and vcc / scc): declared clobbered at every marker
#define HKGEN_CLOBBERS "memory", "vcc", "scc", "s90", "s91", "s92", "s93", "s94", "s95", "s96", "s97", "s98", \
                       "s99", "s100", "s101", "s102", "s103"
// the LOOP region keeps its loop state, TDM descriptors and per-tile scalars in s48..s103
#define HKGEN_LOOP_CLOBBERS HKGEN_CLOBBERS, "s48", "s49", "s50", "s51", "s52", "s53", "s54", "s55", "s56", "s57", \
    "s58", "s59", "s60", "s61", "s62", "s63", "s64", "s65", "s66", "s67", "s68", "s69", "s70", "s71", "s72", "s73",  \
    "s74", "s75", "s76", "s77", "s78", "s79", "s80", "s81", "s82", "s83", "s84", "s85", "s86", "s87", "s88", "s89"
// the two 16-B halves of a v16 operand: a dK operand's dwords 0..3 / 4..7 = its two dslayout chunks
HK1_INLINE bf16x8 lo8(bf16x16 v) { return __builtin_shufflevector(v, v, 0, 1, 2, 3, 4, 5, 6, 7); }
HK1_INLINE bf16x8 hi8(bf16x16 v) { return __builtin_shufflevector(v, v, 8, 9, 10, 11, 12, 13, 14, 15); }
// sched_barrier masks (llvm.amdgcn.sched.barrier): what may cross the fence
constexpr int SB_ALU = 0x002 | 0x004 | 0x400;   // VALU, SALU, transcendental
}  // namespace

extern "C" __global__ __launch_bounds__(64) void k_dkdv96ds(
    const bf16* __restrict__ Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
    const bf16* __restrict__ DO, const float* __restrict__ NL, const float* __restrict__ ND,
    bf16* __restrict__ DV, bf16* __restrict__ DK, bf16* __restrict__ DS,
    float scale, int Sq, int Skv, int Hq, int Hkv, int G, int nqt, int cshift, int causal, int B, int HG,
    int ds_mode) {
  __shared__ __attribute__((aligned(1024))) unsigned char smem[K1_LDS + (HKGEN ? K1_NW * K3_KV_W : 0)];
  lock_simd();                           // SCHED_MODE.DISABLE_XDL_ARB_STALL
  const uint32_t lds0 = lds_addr(smem);
  const int tid = (int)threadIdx.x;
  const int lane = tid % 32;
  const int wv = wave_id(tid);
  const int half = lane / 16;
  const int row = lane % 16;
#if HKGEN
  // ---- marker shell: work item li = (bz * nblk + by) * HG + bx (the grid's order: per bz the longest kv blocks
  // first) from a ticket counter (an int 64 B before DS, in the scratch's front guard, zeroed by the host before each
  // launch): the workgroups take the items in the order they start. The dispatcher deals the workgroups to the 8
  // XCDs round robin, a fixed number each; on a slower XCD they start later and take the later, shorter items instead
  // of that XCD finishing last. Which workgroup runs an item does not change what it computes.
  __shared__ int s_item[1];
  if (tid == 0)
    s_item[0] = __hip_atomic_fetch_add(reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(DS) - 64), 1,
                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  lds_barrier();
  const int k3_item = __builtin_amdgcn_readfirstlane(s_item[0]);
#define K3_BX (k3_item % HG)
#define K3_BY (k3_item / HG % k3_nblk(Skv))
#define K3_BZ (k3_item / (HG * k3_nblk(Skv)))
#else
#define K3_BX ((int)blockIdx.x)
#define K3_BY ((int)blockIdx.y)
#define K3_BZ ((int)blockIdx.z)
#endif
  const int bat = k1_bat(K3_BZ, K3_BX, HG, Hkv);
  const int hkv = k1_hkv(K3_BZ, K3_BX, HG, Hkv);
  const int bid = K3_BY;
  const int n96 = k3_n96(Skv);

  // true byte extents on every buffer descriptor: an access past them is dropped (a load returns 0)
  const int nk_b = k1_nk_b(B, Skv, Hkv), nv_b = k1_nv_b(B, Skv, Hkv), nl_b = k1_nl_b(B, Hq, Sq);
  const auto r_k = rsrc(K, nk_b), r_v = rsrc(V, nv_b), r_nl = rsrc(NL, nl_b), r_nd = rsrc(ND, nl_b);
  const int kbase = k1_kbase(bat, Skv, Hkv, hkv), vbase = k1_vbase(bat, Skv, Hkv, hkv);
  const float c1 = scale * K1_LOG2E;     // softmax exp2 multiplier
  const int nqt2 = nqt / 2;              // q pairs (nqt: 16-row query tiles)

  // ---- Q / dO ring TDM descriptors: group 1 per op (loop-invariant), per-op LDS / global bases
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

  // ---- NL / ND: the 8 vec4 loads of (query pair qt, q head gh); voffset 32*half + constant, soffset uniform
  const int voff_l = k1_nl_voff(lane);
  auto ldl = [&](int qt, int gh, f32x4 (&out)[NLD]) {
    const int sb = uniform(k1_nl_soff(bat, Hq, k1_qh(hkv, G, gh), Sq, qt * 32));
    static_for<NLD>([&](auto e) {
      // the constant goes on the VGPR offset (3 loop-invariant VGPRs voff + 16 / 64 / 80), so all 8 loads share one
      // SGPR soffset and no address register is rewritten while earlier loads still read it (an s_wait_xcnt WAR)
      out[e] = buf_load_f32x4(((e % 4) / 2) == 0 ? r_nl : r_nd, voff_l + k1_nl_sconst(e), sb);
    });
  };

  // ---- readback of a stage's S/dP A operands (Q, dO rows): entry hh*20 + (Q: dt*2+u | dO: 12 + dt*2+u)
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
  // both halves (64-row tail body): entry hh*20 + ...
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
  // A operands of the dV/dK WMMAs (dO^T, Q^T of a stage): the tr16 pieces p = 0, 1 (rows + 16 p) form a v16 operand
  auto tr = [&](const lds_u8* base, int isq, int d) -> bf16x16 {
    return cat(lds_tr16(base, k1_tr_const(isq, d, 0)), lds_tr16(base, k1_tr_const(isq, d, 1)));
  };
  auto tr_base = [&](int stage_off, int isq) { return lds_ptr(lds0 + (uint32_t)(stage_off + k1_tr_lane(isq, lane))); };

  // ======================= one body per block height: NKV 3 (96-row blocks) or NKV 2 (64-row tail blocks) =======
  auto body = [&](auto nkv_c) __attribute__((always_inline)) {
    constexpr int NKV = decltype(nkv_c)::value;
    constexpr bool K3 = NKV == K3_NKV;
    constexpr int ROWS = 32 * NKV;
    constexpr int NST = NKV * (NDO_V + NDO_QK);   // accumulators: dV (kh, d) then dK (kh, d)
    constexpr int DK0 = NKV * NDO_V;
    const int kv0g = k3_kv0g(bid, n96);
    const int kv0 = k3_kv0(kv0g, wv, NKV);

    // ---- K / V fragments of this wave's 16*NKV kv rows: invariant over every query and q head (the reference
    // loads them with load_kv(); the marker shell's INIT region loads its own)
    bf16x16 kf[NKV][NDT_QK], vf[NKV][NDT_V];
    auto load_kv = [&]() __attribute__((always_inline)) {
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
    };

    // XCNT drain before the export's scalar math, 64-row tail body only (once per tail workgroup): in the reference
    // the allocator gives that math r_k's SGPRs while the last K fragment loads may still be in address translation,
    // and LLVM emits no s_wait_xcnt there (SIInsertWaitcnts of the manifest's toolchain lets a younger VMEM whose
    // vdst overlaps an older one's vaddr stand in for the XCNT wait, which also clears the older loads' SGPR
    // sources). No builtin exists for s_wait_xcnt; the fence keeps the math below the wait.
    if constexpr (!K3) {
      asm volatile("s_wait_xcnt 0x0" ::: "memory");
      sched_barrier<0>();
    }

    // ---- dS^T export (dslayout.h): per-wave descriptor (base: region bh0 = bat*Hq + hkv*G, row kt0 = kv0 / 16;
    // num_records: the wave's true extent, 0 for ds_mode 0), row constants, voffset
    const int kt0 = k3ds_kt0(kv0);
    const int64_t ds_base = k3ds_base(bat, Hq, hkv, G, kt0, Sq, Skv, causal);
    const auto r_ds = rsrc64((const unsigned char*)DS + ds_base, k3ds_nrec(ds_mode, G, kt0, NKV, Sq, Skv, causal));
    int ds_rc[NKV];
    static_for<NKV>([&](auto kh) { ds_rc[kh] = k3ds_rowc(kh, kt0, Sq, Skv, causal); });
    const int vds = k1ds_voff(lane);
    // the two stores of one kv16 block: a_dsk[kh]'s dwords 0..3 (queries 0..15) at vds = lane * 16, dwords 4..7
    // (queries 16..31) at 512 + lane * 16 -- the operand's own bytes, no cross-lane exchange
    auto ds_store = [&](const bf16x16& a, int so) {
      buf_store_b128(lo8(a), r_ds, vds, so);
      buf_store_b128(hi8(a), r_ds, vds + K1DS_ST_B, so);
    };

    // ---- epilogue: C fragments -> this wave's row-major dV / dK images (epi_images; the marker shell's FINAL region
    // writes the same images), then whole-line loads + global stores (epi_stores)
    auto epi_images = [&](const auto& acc) __attribute__((always_inline)) {
      static_for<NKV>([&](auto kh) {
        static_for<NDO_QK>([&](auto d) {
          if constexpr (d < NDO_V)
            lds_store_b128(lds0 + (uint32_t)k3_epi_wr(NKV, wv, 0, kh, d, lane), to_bf16x8(acc[kh * NDO_V + d]));
          lds_store_b128(lds0 + (uint32_t)k3_epi_wr(NKV, wv, 1, kh, d, lane),
                         to_bf16x8(acc[NKV * NDO_V + kh * NDO_QK + d]));
        });
      });
    };
    auto epi_stores = [&]() __attribute__((always_inline)) {
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
          tt = (kvl > qb + si + cshift) ? K1_NEG : tt;  // no causal test: n_mask = G * k3_nmaskp() is 0 unless causal
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
    // 64-row tail body: both hh halves from the carried 40-entry readback
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
    // 96-row body: hh0 from the carried 20-entry readback op0; hh1's 20 entries are read from rb1 (the same stage)
    // as one burst at the top, so its latency hides under hh0's 30 S/dP WMMAs (fences let only VALU/SALU/trans cross)
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
        // full tiles: the pending prefetch TDM right behind the burst; no fence after it, so the hh0 S/dP chains may
        // issue around the 3 ops
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
    // dV^T += dO^T P, dK^T += Q^T dS (A = the dO^T / Q^T d tile, B = the P / dS operand of kv sub-tile kh, the same
    // over the run: reuse_b from the 2nd WMMA on); C fragment = kv row kh*16 + row, d = dtile*16 + half*8 + [0, 8)
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

    // ---- loop bounds: causal tile skip at workgroup level (block id only, both waves equal)
    const int qp_start = k1_qp_start(kv0g, cshift, causal);
    const int nmaskp = k3_nmaskp(kv0g, ROWS, cshift, causal, nqt);
    const int nqp_eff = k1_nqp_eff(kv0g, cshift, causal, nqt);
    const int n_mask = G * nmaskp;
    const int n_full = G * (nqp_eff - nmaskp);

    f32x8 acc[NST];
    static_for<NST>([&](auto i) { acc[i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0}; });
#if HKGEN
    // ---- marker shell: the per-tile work as generated regions at inline-asm markers (tools/bwdgen; the asm patch
    // splices them into this shell's .s). The generated loop assumes G == 1 (limits.py admits MHA only): tile k is
    // q pair qp_start + k, in stage k mod 3. Marker contract (checked by tools/bwdgen/build.py): the generated code
    // writes only VGPRs above every VGPR this shell references between INIT and FINAL, the clobbered SGPRs, vcc and
    // scc; one workgroup barrier per tile.
    //   INIT  (c1, c1), (1 / scale, 1 / scale); the wave's K / V rows by TDM into its staging area, accumulators = 0,
    //         then the K / V fragments from the staging area
    //   LOOP  tile 0's hh0 operands and NL/ND, then every tile k = 0 .. N-1 (the first n_mask diagonal: causal mask
    //         on, the dS^T stores of unstored blocks dropped): TDM of tile k+2; S / dP, softmax, dV / dK; the dS^T
    //         stores of tile k; NL/ND of tile k+1; tensorcnt 3 + barrier after the last read of stage k; hh0
    //         operands of tile k+1
    //   FINAL dV * (1 / scale), dK -> bf16 images (epi_images' layout), whole-line readback, global stores
    {  // both block heights (NKV 3: 96 rows, NKV 2: the 64-row tail); the markers carry nkv
      const int N = n_mask + n_full;
      auto stg = [&](int t) { return (t % K1_DEPTH) * K1_QDO_B; };
      auto tq = [&](int t) { return k1_clampqt(qp_start + (t < N ? t : N - 1), nqt2); };
      auto nls = [&](int qt) { return uniform(k1_nl_soff(bat, Hq, k1_qh(hkv, G, 0), Sq, qt * 32)); };
      const uint32_t rdq_l = lds0 + (uint32_t)k1_rd_lane(1, lane), rdv_l = lds0 + (uint32_t)k1_rd_lane(0, lane);
      const uint32_t trv_l = lds0 + (uint32_t)k1_tr_lane(0, lane), trk_l = lds0 + (uint32_t)k1_tr_lane(1, lane);
      // vk / vv / sk / sv (the K / V fragment byte offsets: lane part, kv sub-tile part) are unused since INIT loads
      // K / V by TDM. They stay until the next asm-patch regeneration: dropping them renumbers an IR block name in
      // the compiler's .s, which asm_input_sha256 pins.
      const int vk = 16 * k1_kfrag(0, Hkv, 0, 0, 0, 0, lane), vv = 16 * k1_vfrag(0, Hkv, 0, 0, 0, 0, lane);
      int sk[NKV], sv[NKV];
      static_for<NKV>([&](auto kh) {
        sk[kh] = 16 * k1_kfrag(kbase, Hkv, kv0, kh, 0, 0, 0);
        sv[kh] = 16 * k1_vfrag(vbase, Hkv, kv0, kh, 0, 0, 0);
      });
      // this wave's K / V staging area (16 * NKV rows of V, then of K); INIT moves the rows there by TDM (ops as a
      // ring stage's: V cols [0, 128), K cols [0, 128), K cols [128, 192)), waits for every TDM op, then reads the
      // fragments (ds_load_b128 at lane part + kh * 16 rows + dt * 64 + u * 32)
      const uint32_t kv_l = lds0 + K1_LDS + (uint32_t)(wv * K3_KV_W);
      const uint64_t kv_kg = (uint64_t)(uintptr_t)K + 2u * (uint64_t)(((int64_t)(bat * Skv + kv0) * Hkv + hkv) * K1_D_QK);
      const uint64_t kv_vg = (uint64_t)(uintptr_t)V + 2u * (uint64_t)(((int64_t)(bat * Skv + kv0) * Hkv + hkv) * K1_D_V);
      // the first two Q / dO tiles, ahead of INIT: their latency overlaps INIT's K / V loads
      if (N > 0) {
        sched_barrier<0>();
        tdm_tile(tq(0), 0, stg(0));
        tdm_tile(tq(1), 0, stg(1));
        sched_barrier<0>();
      }
      (void)vk, (void)vv, (void)sk, (void)sv;
      asm volatile("; HKGEN INIT nkv=%8 lk=%0 lv=%1 kg=%2 vg=%3 hkv=%4 kvl=%5 c1=%6 sc=%7"
                   ::"v"(kv_l + (uint32_t)(16 * NKV * K1_XV + (lane % 16) * K1_XK + (lane / 16) * 16)),
                   "v"(kv_l + (uint32_t)((lane % 16) * K1_XV + (lane / 16) * 16)), "s"(kv_kg), "s"(kv_vg), "s"(Hkv),
                   "s"(kv_l), "s"(c1), "s"(scale), "i"(NKV)
                   : HKGEN_CLOBBERS);  // its TDM descriptors in s92..s103, one op at a time
      if (N > 0) {
        sched_barrier<0>();
        tensor_wait<K1_TW>();                    // own half of tile 0
        sched_barrier<0>();
        lds_barrier();                           // RAW: both halves of tile 0
        sched_barrier<0>();
        // LOOP: the hh0 operands of tile 0 + its NL/ND, then every tile k = 0 .. N-1 (the first n_mask diagonal) in a
        // generated loop (unrolled by the ring depth: the stage is static per body). Every per-tile scalar is derived
        // inside from these loop-invariant inputs: no shell code runs between the tiles.
        //   tdm: per op o (dO, Q 0..127, Q 128..191): global byte base of tile 0's slice (lo, hi), bytes per tile,
        //        LDS byte of stage 0; then the outer strides (elements) of op 0 / ops 1, 2; N; n_mask
        //   sc:  qp_start, the last query pair, NL/ND soffset of pair 0, thr0 = cshift - kv0, qp_first of kv16 rows
        //        kt0 + kh (dS^T blocks stored iff qt >= it), 0
        //   rc:  the dS^T row constants k3ds_rowc(kh) (soffset of pair qt = 1024 qt + rc[kh]), 0
        const int dl = row - 8 * half;           // the mask's lane part: kv row - q row offset of this lane
        {
          i32x16 tdmv;
          static_for<3>([&](auto op) {
            const uint64_t ga0 = gb[op] + (uint64_t)((int64_t)qp_start * gstep_q[op]);
            tdmv[4 * op + 0] = (int)(uint32_t)ga0;
            tdmv[4 * op + 1] = (int)(uint32_t)(ga0 >> 32);
            tdmv[4 * op + 2] = (int)gstep_q[op];
            tdmv[4 * op + 3] = (int)ldsb[op];
          });
          tdmv[12] = (int)k1_tdm_stride(Hq, 0);
          tdmv[13] = (int)k1_tdm_stride(Hq, 1);
          tdmv[14] = N;
          tdmv[15] = n_mask;
          const i32x8 scv = {qp_start, nqt2 - 1, nls(0), cshift - kv0, k3ds_qpf(0, kt0, Sq, Skv, causal),
                             k3ds_qpf(1, kt0, Sq, Skv, causal), k3ds_qpf(NKV - 1, kt0, Sq, Skv, causal), 0};
          const i32x4 rcv = {ds_rc[0], ds_rc[1], ds_rc[NKV - 1], 0};
          asm volatile("; HKGEN LOOP nkv=%13 lq=%0 lv=%1 ltv=%2 ltk=%3 nlv=%4 vds=%5 dl=%6 tdm=%7 sc=%8 rc=%9 rnl=%10 "
                       "rnd=%11 rds=%12"
                       ::"v"(rdq_l), "v"(rdv_l), "v"(trv_l), "v"(trk_l), "v"(voff_l), "v"(vds), "v"(dl), "s"(tdmv),
                       "s"(scv), "s"(rcv), "s"(r_nl), "s"(r_nd), "s"(r_ds), "i"(NKV)
                       : HKGEN_LOOP_CLOBBERS);
        }
      }
      tensor_wait<0>();                          // no TDM may land after the ring is reused
      sched_barrier<0>();
      lds_barrier();                             // every wave's last ring read before the images overwrite it
      sched_barrier<0>();
      // FINAL: images (lane bases ev / ek), whole-line readback (rv / rk), global stores: byte 16 * k1_epi_g = lane
      // part gv / gk + row base sov / sok + rb * rsv / rsk (4 kv rows) + li * 128
      {
        const int rsv = Hkv * (K1_D_V / 8), rsk = Hkv * (K1_D_QK / 8);
        const auto r_dv = rsrc(DV, nv_b), r_dk = rsrc(DK, nk_b);
        asm volatile("; HKGEN FINAL nkv=%12 ev=%0 ek=%1 rv=%2 rk=%3 gv=%4 gk=%5 sov=%6 sok=%7 rsv=%8 rsk=%9 rdv=%10 "
                     "rdk=%11"
                     ::"v"(lds0 + (uint32_t)k3_epi_wr(NKV, wv, 0, 0, 0, lane)),
                     "v"(lds0 + (uint32_t)k3_epi_wr(NKV, wv, 1, 0, 0, lane)),
                     "v"(lds0 + (uint32_t)k3_epi_rd(NKV, wv, 0, 0, 0, lane)),
                     "v"(lds0 + (uint32_t)k3_epi_rd(NKV, wv, 1, 0, 0, lane)), "v"(16 * ((lane / 8) * rsv + lane % 8)),
                     "v"(16 * ((lane / 8) * rsk + lane % 8)), "s"(16 * (vbase + kv0 * rsv)), "s"(16 * (kbase + kv0 * rsk)),
                     "s"(64 * rsv), "s"(64 * rsk), "s"(r_dv), "s"(r_dk), "i"(NKV)
                     : HKGEN_CLOBBERS);
      }
      return;
    }
#endif
    load_kv();

    // ---- the masked (diagonal) tiles ride the full loop's 3-stage Q/dO ring (dkdv_idx.h: tile sequence)
    // Flat tile sequence j < N = n_mask + n_full, T_j = (qp_start + j / G, j % G) in stage sigma(j) = (j - n_mask)
    // mod 3 (the full loop's first tile lands in stage 0); every prefetch is clamped to T_{N-1}.
    //   prologue (the ring is empty): TDM T_0 -> sigma(0), T_1 -> sigma(1); NL/ND of T_0; tensorcnt 3; BARRIER;
    //     readback of sigma(0) (NKV 3: hh0 | NKV 2: both halves); NKV 3: T_2 -> sigma(2) pending | NKV 2: TDM T_2 ->
    //     sigma(2).
    //   masked iteration j = the full iteration's ring protocol with the causal mask on:
    //     NKV 3: NL/ND of T_{j+1}; hh1 readback of sigma(j), pending TDM T_{j+2} -> sigma(j+2); S/dP; tr16 dO^T; dV
    //         (+ prep of T_{j+3} -> sigma(j)); tr16 Q^T; dK kh0; tensorcnt 3; BARRIER(j); readback hh0 of
    //         sigma(j+1); dK kh1, kh2; export.
    //     NKV 2: NL/ND of T_{j+1}; S/dP; tr16 dO^T, Q^T; dvdk kh0; tensorcnt 3; BARRIER(j); readback of sigma(j+1);
    //         TDM T_{j+3} -> sigma(j); dvdk kh1; export.
    //   A fully masked wave-iteration (e.g. wave 1 at qp_start) still runs its WMMAs: wave 0 bounds that iteration,
    //   so skipping them behind a uniform branch gains nothing, and it costs SGPRs.
    // After the last masked iteration the state is the full loop's entry state: qd = readback of stage 0 =
    // T_{n_mask}, T_{n_mask+1} in flight to stage 1, T_{n_mask+2} -> stage 2 (NKV 3 pending | NKV 2 issued), preA =
    // NL/ND of T_{n_mask}.
    const int qt0 = qp_start + nmaskp;
    const int n_seq = n_mask + n_full;
    // NL/ND live in two register sets: iteration ii reads one and prefetches ii+1 into the other, the loop
    // body is unrolled by 2 to alternate them (the masked loop copies preB -> preA instead).
    f32x4 preA[NLD], preB[NLD];
    // carried readback: NKV 2 both hh halves (40 entries); NKV 3 hh0 only (20)
    constexpr int NCAR = K3 ? NHB : NRB;
    bf16x8 qd[NCAR];
    TdmG0 tdp_pend;                          // NKV 3: the prefetch TDM prepped in iteration ii-1 (or the prologue)
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
        sched_barrier<0>();                  // readback before the TDM
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
    // 64-row tail body: the dS^T payload of the previous full iteration (its a_dsk) and the q-pair / q-head
    // part of its soffset, stored in the S/dP window of the next iteration (no DS op there) or after the loop
    bf16x16 pend[NKV];
    int pso = 0;
    // one full iteration ii: S/dP of the carried A operands (stage ii%3's tile, read back in ii-1) and NL/ND
    // `pre`; prefetch NL/ND of ii+1 into `nxt`; EXP (tail body): store iteration ii-1's dS^T in this S/dP window.
    // g1 = true_type is the MHA (G == 1) specialization of the 96-row body's loop: the (query pair, q head)
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
        // dS^T export: this iteration's blocks (all NKV) at the head of the DS-free dV WMMA run, 24 WMMAs before the
        // Q^T burst (no payload crosses the back edge: it would add 24 VGPRs at the S/dP register peak); the soffset
        // from this iteration's counters (they advance only at the end of full_iter)
        const int so_c = k3ds_sq(qt0 + qi, gh, Sq, Skv, causal);
        // the 2*NKV stores straight from the a_dsk operands
        static_for<NKV>([&](auto kh) {
          buf_store_b128(lo8(a_dsk[kh]), r_ds, vds, so_c + ds_rc[kh]);
          buf_store_b128(hi8(a_dsk[kh]), r_ds, vds + K1DS_ST_B, so_c + ds_rc[kh]);
        });
        // the export is pinned here, under the dO^T tr16 latency, ahead of the first dV WMMA (without expert
        // scheduling a store issued inside the WMMA run waits for every earlier WMMA)
        sched_barrier<0>();
        // the prefetch tile min(ii+3, n-1) -> stage cur: its scalar math rides the dV WMMA run ({1 WMMA,
        // KH0_SALU SALU} groups); the 3 ops issue at the top of iteration ii+1 (not after the last iteration)
        if constexpr (G1) {
          tdp_pend = tdm_prep(qt0 + (ii + 3 < n_full ? ii + 3 : n_full - 1), 0, cur);
        } else {
          tdp_pend = tdm_prep(qt0 + k1_pf_q(ii, n_full, qi, gh, G), k1_pf_g(ii, n_full, qi, gh, G), cur);
        }
        // (2) dV d-outer / kh-inner (A = b_do[d] for NKV consecutive WMMAs: reuse_a), the 24 Q^T tr16 of stage cur
        // interleaved {1 WMMA : 2 tr16} behind the first 12 dV WMMAs (each dV accumulator still gets exactly one WMMA
        // per iteration)
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
        // (ii+1)%3, the release fence drains every wave's DS reads of cur (WAR for the TDM into cur, issued at the
        // top of the next iteration)
        sched_barrier<0>();
        tensor_wait<K1_TW>();
        sched_barrier<0>();
        // split ring barrier: arrive (s_wait_dscnt 0 + release + signal) -> the first XS of the 24 pure-register
        // dK kh1/kh2 WMMAs -> wait (+ acquire) -> hh0 readback of ncur -> the rest of dK kh1/kh2. WAR / RAW as with
        // lds_barrier: every wave signals only after its own tensorcnt 3 and its last DS read of stage cur.
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
        // 64-row tail body: the previous iteration's dS^T payload in this S/dP window (no DS op there), behind the
        // NL/ND loads (a load issued behind stores waits for them); ALU / WMMA / TRANS may cross this fence, no VMEM
        // or DS op
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
        // this iteration's dS^T payload and the q-pair / q-head part of its soffset (counters not yet advanced)
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
      // 64-row tail body: iteration 0 is peeled (nothing pending), then the x2-unrolled body and the odd tail
      full_iter(std::false_type{}, std::false_type{}, 0, preA, preB);
      qi_c = k1ds_q1(G);                     // == k1_wrap_q(0, 0, G), integer form
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

    // ---- epilogue: C fragments -> this wave's row-major dV / dK images -> whole-line stores
    epi_images(acc);
    epi_stores();
  };

  // uniform (SGPR) branch on the block id: 96-row blocks first, then the 64-row tail blocks
  if (bid < n96) {
    body(std::integral_constant<int, K3_NKV>{});
  } else {
    body(std::integral_constant<int, 2>{});
  }
#undef K3_BX
#undef K3_BY
#undef K3_BZ
}
