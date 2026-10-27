/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k_attn_fwd_v0 -- HK gfx1250 flash-attention forward for Llama-3.1 attention: head dim 128, bf16 BSHD,
// GQA with G % 4 == 0, bottom-right causal (query p sees kv <= p + Skv - Sq) or full.
//
//   O[b, p, hq, :] = softmax_kv(scale Q K^T) V (bf16),  LSE[b, hq, p] = ln sum_kv exp(scale s) (fp32)
//
// Spec hk/notes/fwd_v0_design.md rev 2; every index formula is fwd/idx.h (f0::, replayed on the host by
// hk/lab/bounds/fwd_v0_check.cpp); implementation notes and ISA census hk/notes/fwd_v0_impl.md.
//
// grid (B * Hq/4 * Sq/64), longest first (f0::work_v0); block 128 = 4 waves, 1 wave per SIMD, 1 workgroup
// per CU (static LDS 320 KB). A workgroup = 64 positions x one head quad; wave w = positions
// [64 qb + 16 w, +16) of the 4 heads = 4 q-tiles h, Q^T resident (128 VGPR).
// kv streams in 64-row stages through a depth-4 TDM ring: every wave issues ONE 32-row op per stage (waves
// 0 / 1: K rows [0, 32) / [32, 64), waves 2 / 3: the same of V; role = SALU select, never a branch); ops
// past the last stage are real 16-B dummies into the wave's own cell, so every s_wait_tensorcnt count is a
// constant. One split-barrier phase per stage: SIGNAL(P_t) at D(2t - 3) after s_wait_tensorcnt 1,
// WAIT(P_t) at D(2t - 2) followed by every wave's op(t + 2).
// Iteration c (32-kv chunk; S^T double-buffered by chunk parity, compile-time in the 2-chunk stage body):
//   A  V(c) burst (16 ds_load_tr16_b128)
//   B  QK(c+1) -> S[c+1 & 1] (32 WMMA)  ||  tails(c) h0..h2 (fma, exp2, bf16 pack, row sums)
//   D  even c: WAIT + op(c/2 + 3);  odd c: tensorcnt 1, LDS release, SIGNAL
//   E  K(c+2) burst (16 ds_load_b128)
//   F  PV(c) h0,h1 || SM1(c+1) (max, ballot);  PV h2 || tail(c) h3;  PV h3 || row sums h3
//   C  lazy rescale of O / l for chunk c+1 (the only branch: wave-uniform, at the iteration end)
// Stages [0, n-2) loop; stage n-2 (if n >= 2) peeled (its odd F holds the first masked SM1); the last stage
// n-1 has no barrier op, no QK(2n) and no K burst. Exit: epilogue (wave-private region + VMEM), tensorcnt 0,
// WAIT(P_n). No wave-dependent branch anywhere; every barrier is executed by every wave equally often.
#include "fwd/idx.h"
#include "hk_attn/ext_fwd.h"
#include "hk_attn/prims.h"

#pragma clang fp contract(off)

// sched_group_barrier interleave patterns (fwd_v0_design 8.1), a bit mask: 1 = region B, 2 = region F; 0 =
// plain program order. The default build (FWD_SGB 0 + -amdgpu-sched-strategy=coexec) won the static census
// (hk/notes/fwd_v0_impl.md section 3). Correctness never depends on it.
#ifndef FWD_SGB
#define FWD_SGB 0
#endif

// HK_FWD_OPT_* (hk/notes/fwd_v0_opt.md): compile-time variants from the ATT analysis (hk/notes/att_v0.md section
// 4, fwd items 1-4), each independent, default 0. With every one 0 (the manifest build) the code object is the pinned
// v0 one, byte for byte. Kernel ABI, grid, block and LDS map are the same in every variant.
//   RESC   the rescale decision: the common path keeps only max(x2 - m2) over the 4 q-tiles -> ONE compare ->
//          uniform branch (no per-head s_cmp / s_cselect / v_cndmask / m2n); the per-head decision of v0 (same
//          m2n, same corr, bit-identical results) runs inside the rarely taken block, which is laid out of line
//          (__builtin_expect), so the common path falls through and the loop gets a bottom test.
//   DESC   op(t+3)'s descriptor, the loop control and the LDS bases of the next stage move into the even chunk's
//          QK run (an SGPR value fence keeps IR passes from hoisting them into region A); the tensor_dim1 clamp is
//          32-bit SALU (f0::kv_op_dim1_s32: s_max_u32 / s_sub / s_min_u32, no v_cmp_*_i64 -> SGPR round trips);
//          the V / K burst VGPR bases are computed one stage ahead (no VALU -> DS address stall).
//   TAIL   the row sums of q-tiles 0..2 and the P^T pack of q-tile 2 move from the QK run (region B) into the PV
//          run (region F), which has issue slack; B ends with the exps / packs PV needs first.
//   MERGE  one DS burst per chunk: V(c+1) joins K(c+2) in region E (va double-buffered by chunk parity, +64
//          VGPR); region A is gone. Same barriers, TDM ops and waits; only where V is read moves (every V read
//          still precedes the SIGNAL that releases its stage: hk/lab/bounds/fwd_v0_check.cpp -DHK_FWD_OPT_MERGE).
//   ALL    sets the default of every one above to 1.
#ifndef HK_FWD_OPT_ALL
#define HK_FWD_OPT_ALL 0
#endif
#ifndef HK_FWD_OPT_RESC
#define HK_FWD_OPT_RESC HK_FWD_OPT_ALL
#endif
#ifndef HK_FWD_OPT_DESC
#define HK_FWD_OPT_DESC HK_FWD_OPT_ALL
#endif
#ifndef HK_FWD_OPT_TAIL
#define HK_FWD_OPT_TAIL HK_FWD_OPT_ALL
#endif
#ifndef HK_FWD_OPT_MERGE
#define HK_FWD_OPT_MERGE HK_FWD_OPT_ALL
#endif

#define FWD_INL __attribute__((always_inline))

using namespace hk_attn;

namespace {

constexpr int NH = f0::NH, NDK = f0::NDK, NDT = f0::NDT;
template <int N> using ic = std::integral_constant<int, N>;

static_assert(f0::BLOCK == 128 && f0::LDS_BYTES == 327680, "4 waves, one workgroup per CU");

struct Tdm { // one TDM op: groups 0 / 1 (wave-uniform SGPRs)
    u32x4 g0;
    u32x8 g1;
};
HKA_DEV Tdm make_tdm(uint32_t lds, uint64_t ga, const f0::G1 &g) {
    return Tdm{tdm_g0(lds, ga), tdm_g1_words(g.w)};
}
HKA_DEV void issue(const Tdm &d) {
    tdm_load(d.g0, d.g1);
}

// max over the 16 values of one query in this lane (kv tiles 0 and 1): 7 v_max3 + 1 v_max
HKA_DEV float max16(const acc_t &a, const acc_t &b) {
    const float x0 = max3(a[0], a[1], a[2]), x1 = max3(a[3], a[4], a[5]), x2 = max3(a[6], a[7], b[0]);
    const float x3 = max3(b[1], b[2], b[3]), x4 = max3(b[4], b[5], b[6]);
    return __builtin_fmaxf(max3(x0, x1, x2), max3(x3, x4, b[7]));
}

// sched_group_barrier run: NG groups of {NWM WMMA, NV VALU, NT TRANS, NS SALU} (in this order)
template <int REGION, int NG, int NWM, int NV, int NT, int NS = 0> HKA_DEV void sgb_run() {
    if constexpr ((FWD_SGB & REGION) != 0) static_for<NG>([&](auto) {
        sched_group_barrier<sg::WMMA, NWM, 0>();
        if constexpr (NV > 0) sched_group_barrier<sg::VALU, NV, 0>();
        if constexpr (NT > 0) sched_group_barrier<sg::TRANS, NT, 0>();
        if constexpr (NS > 0) sched_group_barrier<sg::SALU, NS, 0>();
    });
}

#if HK_FWD_OPT_RESC
// identity v_mov_b32_dpp (exact): the rare block's new m2 goes through it, so InstCombine cannot push the tails'
// fneg(m2) through the loop phi (that keeps a negated copy of m2 live and costs phi moves on the common path)
HKA_DEV float opaque_f32(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), 0xE4, 0xF, 0xF, false));
}
#endif
#if HK_FWD_OPT_DESC
// SGPR value fence (no instruction): x is "redefined" here, so nothing computed from it can be placed before this
// point by IR passes or the DAG; it pins op(t+3)'s descriptor and the next stage's LDS bases into region B
// (sched_barrier only constrains the machine scheduler). The only inline asm of the kernel; empty, one SGPR.
HKA_DEV void sgpr_fence(int &x) {
    asm volatile("" : "+s"(x));
}
// VGPR value fence on a loop-carried LDS base (one v_mov_b32_dpp quad_perm:[0,1,2,3], the identity): without an
// opaque op InstCombine folds the two incoming "lane base + slot offset" adds of the phi into one add after it, i.e.
// back to the loop head right before the burst (VALU -> DS address stall). (An empty asm "+v" also blocks the fold
// but pins the value to v0..v255, which costs a copy at the loop head.)
HKA_DEV const lds_u8 *vgpr_fence(const lds_u8 *p) {
    const int x = (int) (uint32_t) (uintptr_t) p;
    return lds_ptr((uint32_t) __builtin_amdgcn_mov_dpp(x, 0xE4, 0xF, 0xF, false));
}
#endif

} // namespace

extern "C" __global__ __launch_bounds__(f0::BLOCK) void k_attn_fwd_v0(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, __bf16 *O, float *LSE, long long o_bytes,
    long long lse_bytes, int B, int Sq, int Skv, int Hq, int Hkv, int G, int nqb, int nhq4, int nkv_st,
    int cst, int causal, float c2) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[f0::LDS_BYTES];
    lock_simd();

    // ---- identity and work (wave-uniform except the lane) -------------------------------------------------
    const int      w    = wave_id();
    const int      l    = lane_id();
    const f0::Work wk   = f0::work_v0((int) __builtin_amdgcn_workgroup_id_x(), B, nhq4, nqb);
    const int      hkv  = f0::hkv_of(wk.hq4, G);
    const int      n    = f0::n_stages(wk.qb, cst, nkv_st, causal);
    const int      rv   = f0::op_role_v(w); // 0: this wave moves K halves, 1: V halves (SALU selects only)
    const uint32_t lds0 = lds_addr(smem);

    // ---- TDM descriptors (pure SALU) ----------------------------------------------------------------------
    const uint64_t kv_base = (uint64_t) (uintptr_t) (rv != 0 ? V : K);
    // op k of this wave: rows [64 k + 32 (w & 1), +32) of its role tensor into slot k % 4, or (k >= n) a 16-B
    // dummy of row 0 into its own cell
    // (hot forms: scalar selects, no branch, no VGPR)
    auto c_desc = [&](int k) FWD_INL -> Tdm {
#if HK_FWD_OPT_DESC
        const int dim1 = f0::kv_op_dim1_s32(Skv, w, k); // 32-bit SALU clamp (no v_cmp_*_i64 -> SGPR)
#else
        const int dim1 = f0::kv_op_dim1_hot(Skv, w, k);
#endif
        return make_tdm(lds0 + (uint32_t) f0::kv_op_lds_hot(w, k, n),
                        kv_base + 2ull * (uint64_t) f0::kv_op_el_hot(wk.b, Skv, Hkv, hkv, w, k, n),
                        f0::kv_op_g1_hot(rv, Hkv, dim1, f0::op_real(k, n)));
    };

    // ---- LDS operand bursts (per-lane base + compile-time DS immediates) ----------------------------------
    // K A operands of chunk 2 stage + HC (consumption order of qk: k outer, kt inner)
    auto ld_k = [&](opnd_t(&ka)[2][NDK], int stage, auto HC) FWD_INL {
        const lds_u8 *kp = lds_ptr(lds0 + (uint32_t) (f0::k_slot_off(stage) + f0::k_lane(l)));
        static_for<NDK>([&](auto k) {
            static_for<2>([&](auto kt) {
                ka[kt][k] = cat(ds_load_b128(kp, f0::k_imm(HC, kt, k, 0)), ds_load_b128(kp, f0::k_imm(HC, kt, k, 1)));
            });
        });
    };
    // V^T A operands of chunk 2 stage + HC (tr16 at pitch 288)
    auto ld_v = [&](opnd_t(&va)[NDT], int stage, auto HC) FWD_INL {
        const lds_u8 *vp = lds_ptr(lds0 + (uint32_t) (f0::v_slot_off(stage) + f0::v_lane(l)));
        static_for<NDT>([&](auto dt) {
            va[dt] = cat(ds_load_tr16_b128(vp, f0::v_imm(HC, 0, dt)), ds_load_tr16_b128(vp, f0::v_imm(HC, 1, dt)));
        });
    };

#if HK_FWD_OPT_DESC || HK_FWD_OPT_MERGE
    // the same bursts at a base computed ahead (lane part + slot offset: one v_add, off the DS path)
    auto k_base = [&](int stage) FWD_INL {
        return lds_ptr(lds0 + (uint32_t) (f0::k_slot_off(stage) + f0::k_lane(l)));
    };
    auto v_base = [&](int stage) FWD_INL {
        return lds_ptr(lds0 + (uint32_t) (f0::v_slot_off(stage) + f0::v_lane(l)));
    };
    auto ld_k_at = [&](opnd_t(&ka)[2][NDK], const lds_u8 *kp, auto HC) FWD_INL {
        static_for<NDK>([&](auto k) {
            static_for<2>([&](auto kt) {
                ka[kt][k] = cat(ds_load_b128(kp, f0::k_imm(HC, kt, k, 0)), ds_load_b128(kp, f0::k_imm(HC, kt, k, 1)));
            });
        });
    };
    auto ld_v_at = [&](opnd_t(&va)[NDT], const lds_u8 *vp, auto HC) FWD_INL {
        static_for<NDT>([&](auto dt) {
            va[dt] = cat(ds_load_tr16_b128(vp, f0::v_imm(HC, 0, dt)), ds_load_tr16_b128(vp, f0::v_imm(HC, 1, dt)));
        });
    };
#endif

    // ---- state ------------------------------------------------------------------------------------------------
    opnd_t   qb[NH][NDK];     // Q^T B operands (resident)
    opnd_t   ka[2][NDK];      // K A operands of the next QK
#if HK_FWD_OPT_MERGE
    opnd_t   vab[2][NDT];     // V^T A operands, by chunk parity: PV(c) reads vab[c & 1], E(c) loads V(c+1)
#else
    opnd_t   va[NDT];         // V^T A operands of the current PV
#endif
    opnd_t   P[NH];           // P^T B operands
    acc_t    S[2][NH][2];     // S^T of chunks c (tails in place) and c + 1
    acc_t    oacc[NH][NDT];   // O^T
    f32x2    lsum[NH];        // per-lane partial row sums (halves added in the epilogue)
    float    m2[NH], m2n[NH]; // running max (log2 domain), its candidate for the next chunk
#if HK_FWD_OPT_RESC
    f32x2 m2p[NH / 2]; // RESC: the running max as q-tile pairs (64-bit loop phis: the tails' pk_fma broadcast one
                       // half with op_sel instead of moves building {-m2, -m2} on the common path)
#define M2(h) m2p[(h) >> 1][(h) & 1]
#else
#define M2(h) m2[h]
#endif
    uint32_t bal[NH];         // rescale ballots (wave-uniform)
#if HK_FWD_OPT_RESC
    float    x2c[NH];         // column max (log2 domain) of chunk c + 1, for the rarely taken rescale block
    uint32_t need = 0u;       // ballot of "some q-tile's max moved by > 8 nat" (wave-uniform)
#endif
#if HK_FWD_OPT_DESC
    // V base of stage t (A bursts; MERGE: even E), K base of stage t + 1 (E bursts); each is advanced in place right
    // after its last use of the stage (odd chunk: V in B after A / even E, K in F after E), so old and new value are
    // never live together (no phi copy at the loop head)
    const lds_u8 *vbs = v_base(0), *kbs = k_base(1);
#endif

    // ---- phases ----------------------------------------------------------------------------------------------
    auto qk = [&](acc_t(&s)[NH][2]) FWD_INL { // S^T(c') = K(c') Q^T, C = 0 inline on k = 0
        static_for<NDK>([&](auto k) {
            static_for<2>([&](auto kt) {
                static_for<NH>([&](auto h) {
                    if constexpr (decltype(k)::value == 0) s[h][kt] = wmma0(ka[kt][0], qb[h][0]);
                    else s[h][kt] = wmma(ka[kt][k], qb[h][k], s[h][kt]);
                });
            });
        });
    };
#if HK_FWD_OPT_MERGE
    auto pvm = [&](const opnd_t(&v)[NDT], auto H) FWD_INL { // O^T += V^T P^T of q-tile H
        static_for<NDT>([&](auto dt) { oacc[H][dt] = wmma(v[dt], P[H], oacc[H][dt]); });
    };
#else
    auto pv = [&](auto H) FWD_INL { // O^T += V^T P^T of q-tile H
        static_for<NDT>([&](auto dt) { oacc[H][dt] = wmma(va[dt], P[H], oacc[H][dt]); });
    };
#endif
    // SM1 of chunk c1 (in S buffer s): [causal mask] -> column max -> ballot of "moved by > 8 nat" -> m2n;
    // FIRST (chunk 0, prologue): m2 = max directly (O and l are 0)
    auto sm1 = [&](acc_t(&s)[NH][2], int c1, auto MASKED, auto FIRST) FWD_INL {
        if constexpr (decltype(MASKED)::value != 0) {
            const int off0 = f0::mask_off(w, c1, n, 0, causal), off1 = f0::mask_off(w, c1, n, 1, causal);
            static_for<NH>([&](auto h) {
                s[h][0] = mask_row_gt_col(s[h][0], l, off0);
                s[h][1] = mask_row_gt_col(s[h][1], l, off1);
            });
        }
        static_for<NH>([&](auto h) {
            float mx = max16(s[h][0], s[h][1]);
            mx             = __builtin_fmaxf(mx, xhalf(mx));
            const float x2 = mx * c2;
            if constexpr (decltype(FIRST)::value != 0) {
                M2(h) = x2;
            } else {
#if HK_FWD_OPT_RESC
                x2c[h] = x2;
#else
                bal[h] = ballot(x2 - M2(h) > f0::TH2);
                m2n[h] = any_set(bal[h]) ? __builtin_fmaxf(M2(h), x2) : M2(h);
#endif
            }
        });
#if HK_FWD_OPT_RESC
        // one compare for the branch: max_h (x2 - m2) > TH2 <=> some q-tile's ballot of v0 is non-zero (exact: the
        // differences are the same fp32 values, max is exact, no NaN reaches it)
        if constexpr (decltype(FIRST)::value == 0) {
            const float d = __builtin_fmaxf(__builtin_fmaxf(x2c[0] - M2(0), x2c[1] - M2(1)),
                                            __builtin_fmaxf(x2c[2] - M2(2), x2c[3] - M2(3)));
            need = ballot(d > f0::TH2);
        }
#endif
    };
    // tail of q-tile H of chunk c (S buffer s): p = exp2(s c2 - m2) in place, P^T pack
    auto tail = [&](acc_t(&s)[NH][2], auto H) FWD_INL {
        constexpr int h  = decltype(H)::value;
#if HK_FWD_OPT_RESC
        // -m2 of the q-tile pair, then the half (op_sel broadcast + neg modifiers, no {-m2, -m2} moves)
        const f32x2 nmp = -m2p[h >> 1], cc = {c2, c2};
        const f32x2 nm  = __builtin_shufflevector(nmp, nmp, h & 1, h & 1);
#else
        const f32x2   cc = {c2, c2}, nm = {-M2(h), -M2(h)};
#endif
        static_for<2>([&](auto kt) {
            static_for<4>([&](auto j) {
                constexpr int jj = decltype(j)::value;
                const f32x2   a  = pk_fma(acc_pair<jj>(s[h][kt]), cc, nm);
                s[h][kt][2 * jj]     = exp2_hw(a[0]);
                s[h][kt][2 * jj + 1] = exp2_hw(a[1]);
            });
        });
        P[h] = pack_pt_fwd(s[h][0], s[h][1]);
    };
#if HK_FWD_OPT_TAIL
    // the same in two parts: p = exp2(s c2 - m2) in place, then the P^T pack
    auto tail_exp = [&](acc_t(&s)[NH][2], auto H) FWD_INL {
        constexpr int h  = decltype(H)::value;
#if HK_FWD_OPT_RESC
        // -m2 of the q-tile pair, then the half (op_sel broadcast + neg modifiers, no {-m2, -m2} moves)
        const f32x2 nmp = -m2p[h >> 1], cc = {c2, c2};
        const f32x2 nm  = __builtin_shufflevector(nmp, nmp, h & 1, h & 1);
#else
        const f32x2   cc = {c2, c2}, nm = {-M2(h), -M2(h)};
#endif
        static_for<2>([&](auto kt) {
            static_for<4>([&](auto j) {
                constexpr int jj = decltype(j)::value;
                const f32x2   a  = pk_fma(acc_pair<jj>(s[h][kt]), cc, nm);
                s[h][kt][2 * jj]     = exp2_hw(a[0]);
                s[h][kt][2 * jj + 1] = exp2_hw(a[1]);
            });
        });
    };
    auto tail_pack = [&](acc_t(&s)[NH][2], auto H) FWD_INL {
        constexpr int h = decltype(H)::value;
        P[h]            = pack_pt_fwd(s[h][0], s[h][1]);
    };
#endif
    // per-lane row sum of q-tile H (7 v_pk_add tree + 1 accumulate; the two halves of the pair are added
    // in the epilogue)
    auto rsum = [&](acc_t(&s)[NH][2], auto H) FWD_INL {
        constexpr int h = decltype(H)::value;
        const f32x2   a = pk_add(pk_add(acc_pair<0>(s[h][0]), acc_pair<1>(s[h][0])),
                                 pk_add(acc_pair<2>(s[h][0]), acc_pair<3>(s[h][0])));
        const f32x2   b = pk_add(pk_add(acc_pair<0>(s[h][1]), acc_pair<1>(s[h][1])),
                                 pk_add(acc_pair<2>(s[h][1]), acc_pair<3>(s[h][1])));
        lsum[h] = pk_add(lsum[h], pk_add(a, b));
    };
    // C: lazy rescale (uniform branch, rarely taken); corr = 1 exactly for q-tiles whose m2 stays
    auto rescale = [&]() FWD_INL {
#if HK_FWD_OPT_RESC
        // rarely taken, laid out of line; inside: v0's per-q-tile decision (same m2n and corr bits)
        if (__builtin_expect(any_set(need), 0)) {
            static_for<NH>([&](auto h) {
                const uint32_t b    = ballot(x2c[h] - M2(h) > f0::TH2);
                const float    mn   = any_set(b) ? __builtin_fmaxf(M2(h), x2c[h]) : M2(h);
                const float    corr = exp2_hw(M2(h) - mn);
                const f32x2    cv   = {corr, corr};
                static_for<NDT>([&](auto dt) { oacc[h][dt] = oacc[h][dt] * corr; });
                lsum[h] = pk_mul(lsum[h], cv);
                M2(h)   = opaque_f32(mn);
            });
        }
#else
        if (any_set(bal[0] | bal[1] | bal[2] | bal[3])) {
            static_for<NH>([&](auto h) {
                const float corr = exp2_hw(M2(h) - m2n[h]);
                const f32x2 cv   = {corr, corr};
                static_for<NDT>([&](auto dt) { oacc[h][dt] = oacc[h][dt] * corr; });
                lsum[h] = pk_mul(lsum[h], cv);
            });
        }
        static_for<NH>([&](auto h) { M2(h) = m2n[h]; });
#endif
    };

    // ---- one iteration on chunk c = 2 t + PAR -----------------------------------------------------------------
    // QKN: QK(c+1) present; BAR: 0 none, 1 WAIT(P_{t+1}) + op(t + 3), 2 tensorcnt 1 + SIGNAL(P_{t+2});
    // KB: K(c+2) burst present; SM: 0 none, 1 SM1(c+1) + C(c+1), 2 the same with the causal mask
    auto iter = [&](int t, auto PAR, auto QKN, auto BAR, auto KB, auto SM) FWD_INL {
        constexpr int  par = decltype(PAR)::value, nx = 1 - par;
        constexpr bool qkn = decltype(QKN)::value != 0, kb = decltype(KB)::value != 0;
        constexpr int  bar = decltype(BAR)::value, sm = decltype(SM)::value;
        const int      c   = 2 * t + par;
#if !HK_FWD_OPT_MERGE
        // A: V(c)
#if HK_FWD_OPT_DESC
        ld_v_at(va, vbs, PAR);
#else
        ld_v(va, t, PAR);
#endif
        sched_barrier<0>();
#endif
        // B: QK(c+1) || tails(c) h0..h2 (+ the descriptor of op(t + 3) on even c)
#if HK_FWD_OPT_DESC
        int tk = t; // fenced: op(t+3)'s descriptor (even c) and the next stage's V base (odd c) are computed in B
        if constexpr (bar != 0) sgpr_fence(tk);
        __builtin_assume(tk >= 0);
        const Tdm dn = bar == 1 ? c_desc(tk + 3) : Tdm{};
        if constexpr (bar == 2) vbs = vgpr_fence(v_base(tk + 1)); // stage t+1: A(2t+2) (MERGE: E(2t+1))
#else
        const Tdm dn = bar == 1 ? c_desc(t + 3) : Tdm{};
#endif
        if constexpr (qkn) qk(S[nx]);
#if HK_FWD_OPT_TAIL
        tail_exp(S[par], ic<0>{});
        tail_pack(S[par], ic<0>{});
        tail_exp(S[par], ic<1>{});
        tail_pack(S[par], ic<1>{});
        tail_exp(S[par], ic<2>{}); // h2's P^T pack and the row sums of h0..h2: region F
#else
        tail(S[par], ic<0>{});
        rsum(S[par], ic<0>{});
        tail(S[par], ic<1>{});
        rsum(S[par], ic<1>{});
        tail(S[par], ic<2>{});
        rsum(S[par], ic<2>{});
#endif
        if constexpr (qkn && bar == 1) {
            sgb_run<1, 15, 1, 2, 1, 1>();
            sgb_run<1, 1, 1, 2, 1>();
            sgb_run<1, 16, 1, 3, 2>();
        } else if constexpr (qkn) {
            sgb_run<1, 16, 1, 2, 1>();
            sgb_run<1, 16, 1, 3, 2>();
        }
        sched_barrier<0>();
        // D: barrier point
        if constexpr (bar == 1) {
            split_wait_lds(); // WAIT(P_{t+1}): stage t+1 published, stage t-1 free
            issue(dn);        // op(t + 3) into slot (t + 3) % 4
            sched_barrier<0>();
        } else if constexpr (bar == 2) {
            split_signal_tdm_lds<f0::TCNT_STEADY>(); // SIGNAL(P_{t+2}): own op(t+2) landed, V(c) retired
        }
#if HK_FWD_OPT_MERGE
        // E: K(c+2) (stage t+1, half PAR; ka of K(c+1) is dead after B), then V(c+1) (stage (c+1)/2, half 1-PAR)
        // into the other V set: one DS burst per chunk; both are consumed in the next iteration (QK(c+2) in B,
        // PV(c+1) in F), V(c) of this F was loaded at E(c-1)
        if constexpr (kb || qkn) {
            // K before V (QK(c+2) in the next B waits on K only; the scheduler may otherwise put V first, which
            // makes that QK run wait for the whole 32-load burst)
#if HK_FWD_OPT_DESC
            if constexpr (kb) ld_k_at(ka, kbs, PAR);
            if constexpr (kb && qkn) sched_barrier<0>();
            if constexpr (qkn) ld_v_at(vab[nx], vbs, ic<nx>{}); // odd c: vbs already advanced to stage t+1 in B
#else
            const lds_u8 *vp = v_base(t + par); // V(c+1)'s base before the K loads (no VALU -> DS stall between)
            if constexpr (kb) ld_k(ka, t + 1, PAR);
            if constexpr (kb && qkn) sched_barrier<0>();
            if constexpr (qkn) ld_v_at(vab[nx], vp, ic<nx>{});
#endif
            sched_barrier<0>();
        }
        auto pv = [&](auto H) FWD_INL { pvm(vab[par], H); };
#else
        // E: K(c+2) (stage t+1, half PAR; ka of K(c+1) is dead after B)
        if constexpr (kb) {
#if HK_FWD_OPT_DESC
            ld_k_at(ka, kbs, PAR);
#else
            ld_k(ka, t + 1, PAR);
#endif
            sched_barrier<0>();
        }
#endif
        // F: PV(c) || SM1(c+1), tail h3, row sums h3 (TAIL: + h2's pack, row sums h0..h2)
#if HK_FWD_OPT_DESC
        if constexpr (bar == 2) { // K base of stage t+2 (E(2t+2), E(2t+3)), after E(2t+1)'s burst
            int tf = t;
            sgpr_fence(tf);
            __builtin_assume(tf >= 0);
            kbs = vgpr_fence(k_base(tf + 2));
        }
#endif
        pv(ic<0>{});
        pv(ic<1>{});
        if constexpr (sm != 0) sm1(S[nx], c + 1, ic<sm == 2>{}, ic<0>{});
        if constexpr (sm != 0) sgb_run<2, 16, 1, 4, 0>();
#if HK_FWD_OPT_TAIL
        tail_pack(S[par], ic<2>{});
        rsum(S[par], ic<0>{});
        rsum(S[par], ic<1>{});
#endif
        pv(ic<2>{});
        tail(S[par], ic<3>{});
#if HK_FWD_OPT_TAIL
        rsum(S[par], ic<2>{});
#endif
        sgb_run<2, 8, 1, 2, 2>();
        pv(ic<3>{});
        rsum(S[par], ic<3>{});
        sgb_run<2, 8, 1, 1, 0>();
        sched_barrier<0>();
        // C: rescale for chunk c+1
        if constexpr (sm != 0) rescale();
    };

    // ---- prologue -------------------------------------------------------------------------------------------
    issue(make_tdm(lds0 + (uint32_t) f0::q_op_lds(w),
                   (uint64_t) (uintptr_t) Q + 2ull * (uint64_t) f0::q_op_el(wk.b, Sq, Hq, wk.hq4, wk.qb, w),
                   f0::q_g1(Hq, f0::q_op_dim1(Sq, wk.qb, w))));
    issue(c_desc(0));
    issue(c_desc(1));
    sched_barrier<0>();
    wait_tensorcnt<f0::TCNT_STEADY>(); // own Q and own op 0 landed
    {
        const lds_u8 *qp = lds_ptr(lds0 + (uint32_t) f0::q_lane(w, l));
        static_for<NH>([&](auto h) {
            static_for<NDK>([&](auto k) {
                qb[h][k] = cat(ds_load_b128(qp, f0::q_imm(h, k, 0)), ds_load_b128(qp, f0::q_imm(h, k, 1)));
            });
        });
    }
    lds_barrier(); // P_0: every wave's op 0 (stage 0) published
    issue(c_desc(2));
    split_signal_tdm_lds<f0::TCNT_STEADY>(); // SIGNAL(P_1): own op 1 landed
    {
        opnd_t k0[2][NDK];
        ld_k(k0, 0, ic<0>{}); // K(0)
        ld_k(ka, 0, ic<1>{}); // K(1), consumed by QK(1) in iteration 0
#if HK_FWD_OPT_MERGE
        ld_v(vab[0], 0, ic<0>{}); // V(0), consumed by PV(0) in iteration 0
#endif
        sched_barrier<0>();
        static_for<NH>([&](auto h) {
            static_for<NDT>([&](auto dt) { oacc[h][dt] = zero_acc(); });
            lsum[h] = f32x2{0.f, 0.f};
        });
        static_for<NDK>([&](auto k) {
            static_for<2>([&](auto kt) {
                static_for<NH>([&](auto h) {
                    if constexpr (decltype(k)::value == 0) S[0][h][kt] = wmma0(k0[kt][0], qb[h][0]);
                    else S[0][h][kt] = wmma(k0[kt][k], qb[h][k], S[0][h][kt]);
                });
            });
        });
    }
    sm1(S[0], 0, ic<1>{}, ic<1>{}); // SM1(0): masked (effective iff n == 1), m2 = max
    sched_barrier<0>();

    // ---- stages ---------------------------------------------------------------------------------------------
    const int nmain = f0::main_stages(n);
    for (int t = 0; t < nmain; ++t) {
        iter(t, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<1>{});
        iter(t, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<1>{});
    }
    if (f0::has_pen_stage(n)) { // WG-uniform, outside any loop: stage n-2, its SM1(2n-2) masked
        const int t = n - 2;
        iter(t, ic<0>{}, ic<1>{}, ic<1>{}, ic<1>{}, ic<1>{});
        iter(t, ic<1>{}, ic<1>{}, ic<2>{}, ic<1>{}, ic<2>{});
    }
    {
        const int t = n - 1; // last stage: no barrier op, no K burst; SM1(2n-1) masked; no QK(2n)
        iter(t, ic<0>{}, ic<1>{}, ic<0>{}, ic<0>{}, ic<2>{});
        iter(t, ic<1>{}, ic<0>{}, ic<0>{}, ic<0>{}, ic<0>{});
    }

    // ---- epilogue (wave-private Q region + VMEM; no barrier needed before it) --------------------------------
    float lt[NH];
    static_for<NH>([&](auto h) {
        const float ll = lsum[h][0] + lsum[h][1];
        lt[h]          = ll + xhalf(ll); // identical bits in lanes l and l ^ 16
    });
    {
        lds_u8 *ow = lds_ptr(lds0 + (uint32_t) f0::ow_lane(w, l));
        static_for<NH>([&](auto h) {
            const float inv = lt[h] > 0.f ? rcp_ieee(lt[h]) : 0.f;
            static_for<NDT>([&](auto dt) {
                ds_store_b128(ow, f0::ow_imm(h, dt), cvt_pk_bf16(oacc[h][dt] * inv));
            });
        });
    }
    compiler_fence(); // read-back after the image writes (same-wave DS ops complete in order)
    {
        const rsrc_t  ro = make_rsrc(O, o_bytes);
        const lds_u8 *rp = lds_ptr(lds0 + (uint32_t) f0::or_lane(w, l));
        static_for<2 * f0::QW>([&](auto s) {
            const u32x4 v = ds_load_b128_u32(rp, f0::or_imm(s));
            buf_store_b128(ro, (int) f0::o_store_b(wk.b, Sq, Hq, wk.hq4, wk.qb, w, s, l), 0, v);
        });
    }
    {
        const rsrc_t rl = make_rsrc(LSE, lse_bytes);
        float        lse[NH];
        static_for<NH>([&](auto h) { lse[h] = (M2(h) + log2_hw(lt[h])) * f0::LN2; });
        const bool hi = l >= 16;
        static_for<2>([&](auto k) {
            buf_store_f32(rl, (int) f0::lse_store_b(wk.b, Hq, Sq, wk.hq4, wk.qb, w, k, l), 0,
                          hi ? lse[2 + k] : lse[k]);
        });
    }
    // ---- exit: every TDM op of this wave landed (incl. dummies), then WAIT(P_n) -------------------------------
    sched_barrier<0>();
    wait_tensorcnt<0>();
    split_wait_lds();
}
