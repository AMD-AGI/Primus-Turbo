/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// fwdpp/body.h -- k_attn_fwd_pp: gfx1250 d128 GQA flash-attention forward, 8 waves (2 per SIMD),
// staggered ping-pong (hk/notes/fwd_pp_design.md; every index formula in fwdpp/idx.h, namespace
// pp).
//
// WG = 64 positions x 4 q heads of one kv head; wave w (pair s = w & 3, group g = w >> 2) owns
// positions [64 qb + 16 s, +16) of heads 4 hq4 + 2 g + {0, 1}. Per wave, per 32-kv chunk c, two
// clusters:
//   X(c) = QK(c+1) (16 WMMA, S^T = K Q^T into S[(c+1) & 1])  ||  exp(c) in place on S[c & 1]
//   [s_setprio 1] Y(c) = PV(c) (16 WMMA, O^T += V^T P^T)  ||  cvt P(c), row sums(c), SM1(c+1) (max
//   + ballot + new m2), rescale
// one WG barrier per cluster; the K / V burst of a cluster is issued inside the barrier window
// before it (after the signal, before the wait). The hi group executes one extra barrier first
// (PP_STAGGER), so on every SIMD the X of one wave runs against the Y of the other. K / V stages
// (64 rows, depth-4 ring) are TDM'd by the 4 lo waves only (one 32-row op each per stage),
// published by `s_wait_tensorcnt 1` before the signal at the end of every odd Y. Default scheduling
// mode; no inline asm; no atomics (bitwise deterministic). The program (which burst / TDM op /
// tensorcnt / compute each cluster of each stage mode carries) is the constexpr stage plan
// pp::cl_plan, expanded here at compile time by `cluster` and on the host by pp::kernel_events
// (selftest: == the cluster-indexed spec wave_events; the bounds checker replays it; the CPU suite
// matches its token form against the ISA).
//
// Hazard discipline (x6: VALU writing a register an in-flight WMMA of the same wave reads costs 4
// v_nop; the register allocator reuses just-dead operand registers for temporaries): every cluster
// is two scheduling regions,
//   R1 = all of the cluster's VALU + the WMMAs whose operands stay live past R1,   R2 = the
//   remaining WMMAs only: X: R1 = exp(c) || QK(c+1) of q-tile 0 (ka and qb stay live: q-tile 1
//   reads them)   R2 = QK(c+1) of q-tile 1 Y: R1 = cvt P, sums, SM1, new m2 || PV h0 dt 0..6 (va,
//   P[0], P[1] live)           R2 = PV h0 dt 7, PV h1
// so no VALU ever follows the last read of a WMMA operand register inside a cluster; the next VALU
// of the wave is behind a barrier window (signal, a 16-op DS burst, wait). The rescale branch only
// scales O and l (m2 and corr are computed branch-free in R1), so its join carries no copies.
#pragma once
#pragma clang fp contract(off)

#include "fwdpp/idx.h"
#include "hk_attn/ext_fwdpp.h"
#include "hk_attn/prims.h"

#ifndef PP_PRIO
#define PP_PRIO 1 // 1: s_setprio 1 around every X cluster; 2: around every Y; 0: none
#endif
#ifndef PP_SGB
#define PP_SGB 1 // sched_group_barrier patterns inside the R1 regions (0: program order)
#endif
#ifndef PP_LOCK_SIMD
#define PP_LOCK_SIMD                                                                               \
    0 // SCHED_MODE.DISABLE_XDL_ARB_STALL (a 1-wave/SIMD knob; off for 2 waves/SIMD)
#endif
#ifndef PP_C2_VGPR
#define PP_C2_VGPR                                                                                 \
    1 // {c2, c2} through an LDS round trip once: a VGPR pair, never rematerialized from SGPRs
#endif

namespace fwdpp {
using namespace hk_attn;

// the kernel's only LDS: K / V half rings, per-wave Q staging / O image regions, dummy cells, c2
// cells
__shared__ __attribute__((aligned(1024))) unsigned char pp_smem[pp::LDS_BYTES];

struct Args {
    const __bf16 *Q, *K, *V;
    __bf16       *O;
    float        *LSE;
    long long     o_bytes, lse_bytes;
    int           B, Sq, Skv, Hq, Hkv, G, nqb, nhq4, nkv_st, cst, causal;
    float         c2;
};

#define PP_INL __attribute__((always_inline))
template <int V> using ic = std::integral_constant<int, V>;
typedef __attribute__((address_space(3))) f32x2 lds_f32x2;

struct Tdm {
    u32x4 g0;
    u32x8 g1;
};
HKA_DEV u32x8 g1v(const pp::G1 &g) {
    return u32x8{g.w[0], g.w[1], g.w[2], g.w[3], g.w[4], g.w[5], g.w[6], g.w[7]};
}

template <int ON> HKA_DEV void prio_on() {
    if constexpr (ON) {
        sched_barrier<0>();
        set_prio<1>();
        sched_barrier<0>();
    }
}
template <int ON> HKA_DEV void prio_off() {
    if constexpr (ON) {
        sched_barrier<0>();
        set_prio<0>();
        sched_barrier<0>();
    }
}
// region fence: nothing crosses (the R1 / R2 split and the cluster ends)
HKA_DEV void region_end() {
    sched_barrier<0>();
}

// X R1: 8 x {1 WMMA, 2 VALU (pk_fma), 4 TRANS (exp)}
HKA_DEV void sgb_x1() {
    if constexpr (PP_SGB) {
        static_for<8>([&](auto) PP_INL {
            sched_group_barrier<sg::WMMA, 1, 0>();
            sched_group_barrier<sg::VALU, 2, 0>();
            sched_group_barrier<sg::TRANS, 4, 0>();
        });
    }
    region_end();
}
// Y R1: the 8 cvt of P[0], then 7 x {1 WMMA, NV VALU} (cvt P[1], sums, SM1, m2 update)
template <int NV> HKA_DEV void sgb_y1() {
    if constexpr (PP_SGB) {
        sched_group_barrier<sg::VALU, 8, 0>();
        static_for<7>([&](auto) PP_INL {
            sched_group_barrier<sg::WMMA, 1, 0>();
            sched_group_barrier<sg::VALU, NV, 0>();
        });
    }
    region_end();
}

template <int GRP> HKA_DEV void run(const Args &a) {
    constexpr bool LO  = GRP == 0;
    constexpr int  LAG = pp::lag_of(GRP);
    constexpr int  PX = PP_PRIO == 1, PY = PP_PRIO == 2;

    // ---- identity (all uniform except lane)
    // ----------------------------------------------------------------------
    const int      w    = wave_id();
    const int      lane = lane_id();
    const int      s    = pp::pair_of(w);
    const pp::Work wk   = pp::work_v0((int) __builtin_amdgcn_workgroup_id_x(), a.B, a.nhq4, a.nqb);
    const int      hkv  = pp::hkv_of(wk.hq4, a.G);
    const int      n    = pp::n_stages(wk.qb, a.cst, a.nkv_st, a.causal);
    const uint32_t lds0 = lds_addr(pp_smem);

    // ---- TDM descriptors
    // -------------------------------------------------------------------------------------------
    // own Q tile [16 positions][2 heads x 128] -> Q staging region (pitch 528)
    const uint64_t ga_q = (uint64_t) (uintptr_t) a.Q +
                          2ull * (uint64_t) pp::q_op_el(wk.b, a.Sq, a.Hq, wk.hq4, wk.qb, w);
    const Tdm dq = Tdm{tdm_g0(lds0 + (uint32_t) pp::qo_lds(w), ga_q),
                       g1v(pp::q_g1(a.Hq, pp::q_op_dim1(a.Sq, wk.qb, w)))};
    // lo wave s: its role op k (K if s < 2 else V, rows [64k + 32 (s & 1), +32)); k >= n: 16-B
    // dummy into its cell
    const uint64_t ga_role = (uint64_t) (uintptr_t) (pp::op_role_v(s) ? a.V : a.K);
    auto kv_desc = [&](int k) PP_INL -> Tdm { // pp::kv_op_desc: mask selects only (all SALU)
        const int        d1 = uniform(pp::kv_op_dim1(a.Skv, s, k));
        const pp::KvDesc d  = pp::kv_op_desc(wk.b, a.Skv, a.Hkv, hkv, s, k, n, d1);
        const uint64_t   ga = ga_role + 2ull * (uint64_t) d.el;
        return Tdm{tdm_g0(lds0 + d.lds, ga),
                   u32x8{d.g1[0], d.g1[1], d.g1[2], d.g1[3], d.g1[4], d.g1[5], d.g1[6], d.g1[7]}};
    };
    auto issue = [&](const Tdm &d) PP_INL { tdm_load(d.g0, d.g1); };

    // ---- registers
    // ---------------------------------------------------------------------------------------------------
    opnd_t qb[2][4];   // Q^T B operands [q-tile][k-step]
    acc_t  O[2][8];    // O^T accumulators [q-tile][d tile]
    acc_t  S[2][2][2]; // S^T [chunk parity][q-tile][kv tile]
    opnd_t P[2];       // P^T B operands [q-tile]
    opnd_t ka[2][4];   // K A operands of one chunk [kv tile][k-step]
    opnd_t va[8];      // V^T A operands of one chunk [d tile]
    float  m2[2];      // running max (log2 domain, scaled), identical in l and l ^ 16
    f32x2  l2;         // per-lane partial row sums {q-tile 0, q-tile 1}
    float  m2o[2];     // m2 before the last SM1 (the rescale branch's corr = exp2(m2o - m2))
    f32x2  c2v;        // {c2, c2}
    static_for<2>([&](auto H) PP_INL {
        static_for<8>([&](auto DT) PP_INL { O[H][DT] = zero_acc(); });
        static_for<2>([&](auto KT) PP_INL {
            S[0][H][KT] = zero_acc();
            S[1][H][KT] = zero_acc();
        });
    });

    // ---- LDS bursts (lane base = chunk base (SGPR) + lane offset (VGPR), compile-time DS
    // immediates) --------------- lane bases of a chunk's burst, computed in the R1 of the cluster
    // before the burst's window (a fresh VGPR written right after a WMMA run lands in a just-freed
    // operand register: 4 v_nop)
    const uint32_t k_lane = lds0 + (uint32_t) pp::k_lane(lane),
                   v_lane = lds0 + (uint32_t) pp::v_lane(lane);
    auto kbase   = [&](int c) PP_INL { return lds_ptr(k_lane + (uint32_t) pp::k_chunk_lds(c)); };
    auto vbase   = [&](int c) PP_INL { return lds_ptr(v_lane + (uint32_t) pp::v_chunk_lds(c)); };
    auto burst_k = [&](const lds_u8 *kb) PP_INL {
        static_for<2>([&](auto KT) PP_INL {
            static_for<4>([&](auto K) PP_INL {
                ka[KT][K] = cat(ds_load_b128(kb, pp::k_imm(KT, K, 0)),
                                ds_load_b128(kb, pp::k_imm(KT, K, 1)));
            });
        });
    };
    auto burst_v = [&](const lds_u8 *vb) PP_INL {
        static_for<8>([&](auto DT) PP_INL {
            va[DT] = cat(ds_load_tr16_b128(vb, pp::v_imm(0, DT)),
                         ds_load_tr16_b128(vb, pp::v_imm(1, DT)));
        });
    };

    // ---- cluster pieces
    // ------------------------------------------------------------------------------------------ QK
    // of q-tile H into S[PAR][H]: k-outer, 2 accumulator chains (the co-resident wave's WMMAs fill
    // the pipe)
    auto qk = [&](auto PAR, auto H) PP_INL {
        constexpr int pr = decltype(PAR)::value, h = decltype(H)::value;
        static_for<4>([&](auto K) PP_INL {
            static_for<2>([&](auto KT) PP_INL {
                if constexpr (K == 0)
                    S[pr][h][KT] = wmma0(ka[KT][0], qb[h][0]);
                else
                    S[pr][h][KT] = wmma(ka[KT][K], qb[h][K], S[pr][h][KT]);
            });
        });
    };
    // exp in place: S = exp2(fma(S, c2, -m2)) (4 v_pk_fma + 8 v_exp per accumulator)
    auto expc = [&](auto PAR) PP_INL {
        constexpr int pr = decltype(PAR)::value;
        static_for<2>([&](auto H) PP_INL {
            const f32x2 nm = {-m2[H], -m2[H]};
            static_for<2>([&](auto KT) PP_INL {
                acc_t &x = S[pr][H][KT];
                static_for<4>([&](auto J) PP_INL {
                    const f32x2 t =
                        __builtin_elementwise_fma(f32x2{x[2 * J], x[2 * J + 1]}, c2v, nm);
                    x[2 * J]     = exp2_hw(t[0]);
                    x[2 * J + 1] = exp2_hw(t[1]);
                });
            });
        });
    };
    // PV of (h, dt)
    auto pv1 = [&](auto H, auto DT) PP_INL { O[H][DT] = wmma(va[DT], P[H], O[H][DT]); };
    // row sums of S[PAR] (pk_add tree over the two q-tiles) and l += rs
    auto sums = [&](auto PAR) PP_INL {
        constexpr int pr = decltype(PAR)::value;
        f32x2         x[16];
        static_for<16>(
            [&](auto J) PP_INL { x[J] = f32x2{S[pr][0][J / 8][J % 8], S[pr][1][J / 8][J % 8]}; });
        static_for<8>([&](auto J) PP_INL { x[J] = x[2 * J] + x[2 * J + 1]; });
        static_for<4>([&](auto J) PP_INL { x[J] = x[2 * J] + x[2 * J + 1]; });
        static_for<2>([&](auto J) PP_INL { x[J] = x[2 * J] + x[2 * J + 1]; });
        l2 = l2 + (x[0] + x[1]);
    };
    // SM1 on S[PAR] of chunk c: [mask], column max (lane tree + permlanex16), scaled; the rescale
    // vote; new m2 and corr branch-free (corr == 1 exactly when m2 does not move). SEED: chunk 0,
    // m2 = mx2 (no vote).
    auto sm1 = [&](auto PAR, auto MASKED, auto SEED, int c) PP_INL -> bool {
        constexpr int pr = decltype(PAR)::value;
        if constexpr (decltype(MASKED)::value) {
            static_for<2>([&](auto KT) PP_INL {
                const int off = pp::mask_off(s, c, n, KT, a.causal);
                static_for<2>([&](auto H) PP_INL {
                    S[pr][H][KT] = mask_row_gt_col(S[pr][H][KT], lane, off);
                });
            });
        }
        f32x2 mx2;
        static_for<2>([&](auto H) PP_INL { mx2[(int) H] = col_max<2>(S[pr][H]); });
        mx2 = mx2 * c2v;
        if constexpr (decltype(SEED)::value) {
            m2[0] = mx2[0];
            m2[1] = mx2[1];
            return false;
        } else {
            const bool go = any_lane((mx2[0] - m2[0] > pp::TH2) | (mx2[1] - m2[1] > pp::TH2));
            static_for<2>([&](auto H) PP_INL {
                m2o[H] = m2[H];
                m2[H]  = go ? __builtin_fmaxf(m2[H], mx2[(int) H]) : m2[H];
            });
            return go;
        }
    };
    // rescale C: the only branch (wave-uniform, rarely taken), after the cluster's last WMMA; O and
    // l only
    auto rescale = [&](bool go) PP_INL {
        if (__builtin_expect(go, 0)) {
            const f32x2 corr = {exp2_hw(m2o[0] - m2[0]), exp2_hw(m2o[1] - m2[1])};
            static_for<2>([&](auto H) PP_INL {
                static_for<8>([&](auto DT) PP_INL { O[H][DT] = O[H][DT] * corr[(int) H]; });
            });
            l2 = l2 * corr;
        }
    };
    // cluster X: R1 = exp(EP) || QK(QP) q-tile 0 || r1() (next burst base); R2 = QK(QP) q-tile 1
    auto cluster_x = [&](auto EXP, auto QK, auto EP, auto QP, auto r1) PP_INL {
        prio_on<PX>();
        if constexpr (decltype(EXP)::value)
            expc(EP);
        if constexpr (decltype(QK)::value)
            qk(QP, ic<0>{});
        r1();
        sgb_x1();
        if constexpr (decltype(QK)::value)
            qk(QP, ic<1>{});
        region_end();
        prio_off<PX>();
    };
    // cluster Y: R1 = cvt P(PP), sums(PP), SM1(SP of chunk c) || PV h0 dt 0..6; R2 = PV h0 dt 7, PV
    // h1; rescale. PV 0: the prologue's Y(-1) (SM1 only)
    auto cluster_y = [&](auto PV, auto PP, auto SM, auto SP, auto MASKED, auto SEED, int c, auto r1)
                         PP_INL {
                             constexpr int  pr = decltype(PP)::value;
                             constexpr bool pv = decltype(PV)::value;
                             prio_on<PY>();
                             if constexpr (pv) {
                                 P[0] = pack_pt_fwd(S[pr][0][0], S[pr][0][1]);
                                 P[1] = pack_pt_fwd(S[pr][1][0], S[pr][1][1]);
                                 static_for<7>([&](auto DT) PP_INL { pv1(ic<0>{}, DT); });
                                 sums(PP);
                             }
                             bool go = false;
                             if constexpr (decltype(SM)::value)
                                 go = sm1(SP, MASKED, SEED, c);
                             r1();
                             if constexpr (pv) {
                                 sgb_y1<(decltype(MASKED)::value ? 14 : 8)>();
                                 pv1(ic<0>{}, ic<7>{});
                                 static_for<8>([&](auto DT) PP_INL { pv1(ic<1>{}, DT); });
                             }
                             region_end();
                             prio_off<PY>();
                             if constexpr (decltype(SM)::value && !decltype(SEED)::value)
                                 rescale(go);
                         };

    // ---- prologue
    // ------------------------------------------------------------------------------------------------------
    issue(dq);
    if constexpr (LO) {
        static_assert(pp::PRO_OPS == 2, "prologue ops 0, 1");
        issue(kv_desc(0));
        issue(kv_desc(1));
        wait_tensorcnt<pp::TCNT_PRO>(); // own Q + op 0 landed
    } else {
        wait_tensorcnt<0>(); // own Q landed
    }
    sched_barrier<0>();
    {
        const lds_u8 *qbase = lds_ptr(lds0 + (uint32_t) pp::q_lane(w, lane));
        static_for<2>([&](auto H) PP_INL {
            static_for<4>([&](auto K) PP_INL {
                qb[H][K] = cat(ds_load_b128(qbase, pp::q_imm(H, K, 0)),
                               ds_load_b128(qbase, pp::q_imm(H, K, 1)));
            });
        });
    }
    if constexpr (PP_C2_VGPR) {
        volatile lds_f32x2 *cp = (volatile lds_f32x2 *) lds_ptr(lds0 + (uint32_t) pp::c2_lds(w));
        *cp                    = f32x2{a.c2, a.c2};
        c2v                    = *cp;
    } else {
        c2v = f32x2{a.c2, a.c2};
    }
    lds_barrier(); // B_0: stage 0 published (op 0 of every lo wave)
    if constexpr (LO) {
        issue(kv_desc(pp::PRO_OP_B0));
    } else if constexpr (LAG == 1) {
        lds_barrier(); // B_1: the stagger (hi runs one cluster behind from here on)
    }
    l2     = f32x2{0.f, 0.f};
    m2o[0] = m2o[1] = 0.f;
    // loop-carried: the lane base of the next K / V burst (computed in the preceding cluster's R1)
    // and the lo wave's next TDM op (computed in the R1 of the Y that precedes its issue)
    constexpr pp::ClPlan PRO0  = pp::cl_plan(pp::SM_PRO, pp::cl_first(pp::SM_PRO)),
                         MAIN0 = pp::cl_plan(pp::SM_MAIN, 0);
    static_assert(PRO0.kc >= 0 && PRO0.vc < 0 && !PRO0.win,
                  "the first burst: K, after the last prologue wait");
    const lds_u8 *kb = kbase(-2 + PRO0.kc), *vb = kbase(-2 + PRO0.kc);
    Tdm           dn  = kv_desc(0 + MAIN0.op);
    auto          nop = []() PP_INL {};

    // ---- cluster i of stage t in mode M: the expansion of pp::cl_plan(M, i) (pp::kernel_events is
    // the same expansion on the host): [window: signal, burst, wait] [lo: TDM op] cluster [lo:
    // s_wait_tensorcnt]. The R1 of a cluster computes the lane base of the next cluster's burst
    // (and, at i = 3, the next stage's TDM descriptor).
    auto cluster = [&](int t, auto MODE, auto I) PP_INL {
        constexpr int        M = decltype(MODE)::value, i = decltype(I)::value;
        constexpr pp::ClPlan p  = pp::cl_plan(M, i);
        const int            c0 = 2 * t;
        auto                 r1 = [&]() PP_INL {
            if constexpr (i < 3) {
                constexpr pp::ClPlan q = pp::cl_plan(M, i + 1);
                if constexpr (q.kc >= 0)
                    kb = kbase(c0 + q.kc);
                if constexpr (q.vc >= 0)
                    vb = vbase(c0 + q.vc);
            } else if constexpr (M != pp::SM_LAST) { // the next stage's cluster 0 (static_assert:
                                                     // mode-independent)
                vb = vbase(c0 + 2 + MAIN0.vc);
                if constexpr (LO && M != pp::SM_PRO)
                    dn = kv_desc(t + 1 + MAIN0.op);
            }
        };
        constexpr bool has_r1 =
            i < 3 ? (pp::cl_plan(M, i + 1).kc >= 0 || pp::cl_plan(M, i + 1).vc >= 0)
                  : M != pp::SM_LAST;
        if constexpr (p.win)
            bar_window_open();
        else
            sched_barrier<0>();
        if constexpr (p.kc >= 0)
            burst_k(kb);
        if constexpr (p.vc >= 0)
            burst_v(vb);
        if constexpr (p.win)
            bar_window_close();
        if constexpr (LO && p.op >= 0) {
            issue(dn); // op t + 3: after B_{4t+2}, the WAR bound of slot (t + 3) % 4
            sched_barrier<0>();
        }
        if constexpr (i % 2 == 0) {
            if constexpr (has_r1)
                cluster_x(ic<(p.ex >= 0)>{}, ic<(p.qk >= 0)>{}, ic<(p.ex >= 0 ? p.ex & 1 : 0)>{},
                          ic<(p.qk >= 0 ? p.qk & 1 : 0)>{}, r1);
            else
                cluster_x(ic<(p.ex >= 0)>{}, ic<(p.qk >= 0)>{}, ic<(p.ex >= 0 ? p.ex & 1 : 0)>{},
                          ic<(p.qk >= 0 ? p.qk & 1 : 0)>{}, nop);
        } else {
            constexpr int pp_ = p.pv >= 0 ? p.pv & 1 : 0, sp = p.sm1 >= 0 ? p.sm1 & 1 : 0;
            if constexpr (has_r1)
                cluster_y(ic<(p.pv >= 0)>{}, ic<pp_>{}, ic<(p.sm1 >= 0)>{}, ic<sp>{},
                          ic<p.masked>{}, ic<p.seed>{}, c0 + p.sm1, r1);
            else
                cluster_y(ic<(p.pv >= 0)>{}, ic<pp_>{}, ic<(p.sm1 >= 0)>{}, ic<sp>{},
                          ic<p.masked>{}, ic<p.seed>{}, c0 + p.sm1, nop);
        }
        if constexpr (LO && p.tcnt >= 0) {
            sched_barrier<0>();
            wait_tensorcnt<p.tcnt>(); // retires op t + 2 (prologue: op 1): published at the next
                                      // signal
        }
    };
    auto stage = [&](int t, auto MODE) PP_INL {
        static_for<4>([&](auto I) PP_INL {
            if constexpr (decltype(I)::value >= pp::cl_first(decltype(MODE)::value))
                cluster(t, MODE, I);
        });
    };

    // prologue clusters X(-1) (K(0) after the last prologue wait, QK(0)) and Y(-1) (K(1) window,
    // SM1(0) seed); stages [0, n - 2) MAIN, n - 2 PEN (masked SM1(2n - 2)), n - 1 LAST
    // (pp::stage_mode)
    stage(-1, ic<pp::SM_PRO>{});
#pragma clang loop unroll(disable)
    for (int t = 0; t < pp::main_stages(n); ++t)
        stage(t, ic<pp::SM_MAIN>{});
    if (pp::has_pen_stage(n))
        stage(n - 2, ic<pp::SM_PEN>{});
    stage(n - 1, ic<pp::SM_LAST>{});

    // ---- epilogue (own Q / O region + global stores) and exit
    // -----------------------------------------------------------
    auto epilogue = [&]() PP_INL {
        const rsrc_t ro = make_rsrc(a.O, a.o_bytes);
        const rsrc_t rl = make_rsrc(a.LSE, a.lse_bytes);
        float        lt[2], inv[2];
        static_for<2>([&](auto H) PP_INL {
            lt[H]  = l2[(int) H] + xhalf(l2[(int) H]);
            inv[H] = 1.0f / lt[H];
        });
        lds_u8 *ob = lds_ptr(lds0 + (uint32_t) pp::q_lane(w, lane));
        static_for<2>([&](auto H) PP_INL {
            static_for<8>([&](auto DT) PP_INL {
                ds_store_b128(ob, pp::ow_imm(H, DT), cvt_pk_bf16(O[H][DT] * inv[H]));
            });
        });
        const lds_u8 *rb = lds_ptr(lds0 + (uint32_t) pp::or_lane(w, lane));
        static_for<16>([&](auto PP) PP_INL {
            const u32x4 v = ds_load_b128_u32(rb, pp::or_imm(PP));
            buf_store_b128(ro, voff_nonneg(pp::o_store_voff(lane)),
                           (int) pp::o_store_soff(wk.b, a.Sq, a.Hq, wk.hq4, wk.qb, w, PP), v);
        });
        const float lse0 = (m2[0] + log2_hw(lt[0])) * pp::LN2;
        const float lse1 = (m2[1] + log2_hw(lt[1])) * pp::LN2;
        buf_store_f32(rl, voff_nonneg((int) pp::lse_store_voff(a.Sq, lane)),
                      (int) pp::lse_store_soff(wk.b, a.Hq, a.Sq, wk.hq4, wk.qb, w),
                      lane < 16 ? lse0 : lse1);
    };
    if constexpr (pp::n_exit_barriers(GRP) == 1) {
        if constexpr (LO)
            bar_window_open_tdm<pp::EXIT_TCNT>(); // every TDM op (incl. dummies) retired before B_J
        else
            bar_window_open();
        epilogue();
        bar_window_close();
    } else {
        epilogue(); // hi, PP_STAGGER 1: its last barrier B_J preceded its last cluster Y(2n-1)
    }
}

HKA_DEV void kernel(const Args &a) {
    if constexpr (PP_LOCK_SIMD)
        lock_simd();
    if (wave_id() < pp::NPAIR)
        run<0>(a);
    else
        run<1>(a);
}

} // namespace fwdpp
