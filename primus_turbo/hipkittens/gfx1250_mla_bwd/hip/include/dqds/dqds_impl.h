/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// dqds/dqds_impl.h (bwd_hk_dqds2) -- the recompute-free dQ kernel of the S1 line in HIP, 384-query
// tiles:
//
//   dQ[q, :] = sum_{kv attended by q} dS[q, kv] K[kv, :]      (dS from the S1 scratch, dslayout v1,
//   scale included)
//
// bwd_hk_dqds's k_dqds_b with the workgroup doubled to 4 waves (block 128, 1 wave per SIMD, one
// workgroup per CU), so one K ring feeds 384 queries, and role-split TDM issue (idx.h has the
// layouts and the arithmetic):
//   NQW 6 -> the 384-query tiles of [q_off, Sq): 4 waves x 96 queries, longest first
//   NQW 4 -> the head [0, 256) (q_off == 256): 4 waves x 64 queries, the same ring protocol
// Per kv32 row r (block rows 2r, 2r + 1) every wave runs NQW x 12 WMMA (dQ^T += K^T dS^T), operands
// burst-loaded one row ahead into the other register set (X / Y: 12 K^T + NQW dS^T v16 operands
// each):
//   half-step j:  [sync + TDM issue] -> burst of row j + 1 (24 + 2 NQW tr16) -> NQW x 12 WMMA on
//   row j
// Ring protocol: K ring 3 x 64 rows, dS ring 3 x 2 kv32-row slots; ONE op per wave per 64-kv unit
// (waves 0 / 3: the dS of kv32 row 2u + (w >> 1) for all NB_G q pairs of the workgroup, waves 1 /
// 2: K rows [64u + 32 (w >> 1), +32)); wait 1 -> barrier P_{u+1} -> op(u + 3) at the odd row of
// every unit only; <= 2 TDM ops in flight per wave; ops past the last unit are 16-B dummies. Fully
// masked q pairs are never loaded (the dS op starts at the first stored pair, dslayout) and their B
// operand is selected to zero (uniform condition, v_cndmask; never a multiply) in the masked units;
// full units carry no select. Epilogue: bwd_hk_q1's (dQ^T fragments -> per-wave row-major image in
// the dead ring -> whole-line b128 stores). Every index formula: ../../idx.h (replayed on the host
// by bounds_proof.py). No inline asm, no atomics, no named barriers, no clusters (prims.h).
#pragma once
#pragma clang fp contract(off)

#include "../q1/prims.h"
#include "idx.h" // arm root (build.py: -Iinclude -I.)

static_assert(DSL_VERSION == 1 && DSL_SCALED == 1,
              "bwd_hk_dqds2 reads dslayout v1 with the scale included");
static_assert(dq::LDS_C <= 327680 && 2 * dq::LDS_C > 327680,
              "one workgroup per CU, inside the 320-KiB LDS");

namespace q1 {
// The kernel's only LDS: the rings, later the per-wave dQ images (the name is the one the emulator
// expects).
__shared__ __attribute__((aligned(16))) char q1_smem[dq::lds_bytes()];
} // namespace q1

namespace dq {
using namespace q1;

#define DQ_INL __attribute__((always_inline))

// ---- X4 (needs -mllvm -amdgpu-expert-scheduling-mode, manifest extra_flags):
// * every unit but the last: split barrier P_{u+1}: arrive after the first X4_NS (x NACC / 72 on the head tile)
//   WMMAs of run(X), wait after run(X). The ring protocol is unchanged: before the arrive the wave's op(u + 1)
//   landed (tensorcnt 1) and its ring reads (burst(Y)) retired (dscnt 0); between arrive and wait only WMMAs;
//   op(u + 3) and burst(X) after the wait.
// * main-tile steady units: the operand bursts interleaved into the WMMA runs (expert scheduling: a DS read
//   no longer waits for earlier WMMAs):
//     region A: burst(Y, 2u + 1) under run(X) WMMAs [0, X4_NS): {X4_LA WMMA} then {1 tr16, 1 WMMA} x 36
//     region C: burst(X, 2u + 2) under run(Y):                  {X4_LC WMMA} then {1 tr16, 1 WMMA} x 36
// * masked units: the zero-select predicate in branch-free form (x4_ds_zero, same value as idx.h ds_zero), so
//   the selects stay in-place v_cndmask (no scalar branches / operand copies inside the run).
constexpr int X4_NS = 66, X4_LA = 8, X4_LC = 4;
Q1_DEV void x4_bar_signal() {
    __builtin_amdgcn_fence(3, "workgroup", "local"); // release
    __builtin_amdgcn_s_barrier_signal(-1);
}
Q1_DEV void x4_bar_wait() {
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(2, "workgroup", "local"); // acquire
}
// sched_group_barrier pattern of one interleaved region: L WMMA, then {1 DS read, R WMMA} x NT, then the rest
template <int L, int R, int NT, int NW> Q1_DEV void x4_sgb_il() {
    static_assert(L + NT * R <= NW, "interleave pattern longer than the WMMA run");
    if constexpr (L > 0) Q1_SGB(0x008, L);
    sfor<0, NT>([&](auto) DQ_INL {
        Q1_SGB(0x100, 1);
        Q1_SGB(0x008, R);
    });
    if constexpr (NW - L - NT * R > 0) Q1_SGB(0x008, NW - L - NT * R);
}

// idx.h ds_zero without the short circuit (cshift >= 0, as the plan requires): r >= n32 || qp <
// dsl_qp_first(2r) with dsl_qp_first(2r) = causal ? max(0, r - cshift / 32) : 0
Q1_DEV bool x4_ds_zero(int qp, int r, int n32, int cshift, int causal) {
    const int f   = r - cshift / 32;
    const int qpf = (causal != 0) & (f > 0) ? f : 0;
    return (bool) ((int) (r >= n32) | (int) (qp < qpf));
}

struct TdmDesc { // one TDM op's descriptor groups 0 / 1 (prepared before the barrier, issued after
                 // it)
    u32x4 g0;
    u32x8 g1;
};

template <int NQW> struct OpSet {
    bf16x16 kt[NDT]; // A operands: K^T of the 12 d tiles
    bf16x16 ds[NQW]; // B operands: dS^T of the wave's 16-query sub-tiles
};

template <int NQW>
Q1_DEV void dqds_body(const __bf16 *DS, const __bf16 *K, __bf16 *DQ, int Sq, int Skv, int Hq,
                      int Hkv, int G, int nkvt, int cshift, int causal, int q0g, int HG) {
    constexpr int NB_G = nbg(NQW);
    constexpr int NACC = NQW * NDT;

    // ---- identity
    // ----------------------------------------------------------------------------------------------
    const int  tid  = thread_id_x();
    const int  lane = tid % WAVE;
    const int  wv   = readfirstlane(tid / WAVE); // uniform (SGPR) wave index 0..3
    const bool dsw  = c_role_ds(wv); // uniform: this wave issues the dS ops (else the K ops)
    const int  hh   = c_half(wv);    // its kv32 row / K half inside a unit
    const int  hb   = head_hb(block_id_x(), block_id_z(), HG); // = bat*Hq + qh = dslayout's bh
    const int  bat  = head_bat(hb, Hq);
    const int  qh   = head_qh(hb, bat, Hq);
    const int  hkv  = head_kv(qh, G);
    const int  q0   = wave_q0(q0g, wv, NQW);
    const int  qp0g = tile_qp0(q0g);
    const int  qp0w = qp0g + wv * nbw(NQW);
    const int  n32  = kv_n32(q0g, NQW, cshift, nkvt, causal);
    const int  nu   = n_units(n32);
    const int  nfu  = n_full_units(qp0g, cshift, nu, causal);

    lds_char *const          lds   = (lds_char *) q1_smem;
    const unsigned           lds0  = lds_addr(lds);
    const unsigned long long ga_ds = (unsigned long long) DS;
    const unsigned long long ga_k  = (unsigned long long) K;
    const long long          kst   = k_stride_el(Hkv);
    // dummy op source: K[bat, 0, hkv, 0..7] (always inside K)
    const unsigned long long ga_dummy =
        ga_k + (unsigned long long) (k_el(bat, Skv, 0, Hkv, hkv, 0) * 2);

    // ---- one TDM op (bwd_hk_q1's word layout); every field is wave-uniform
    // ----------------------------------
    auto desc = [&](int lds_off, unsigned long long ga, unsigned w0, unsigned tdim1, unsigned tile0,
                    unsigned tile1, unsigned long long stride_el) DQ_INL -> TdmDesc {
        TdmDesc d;
        d.g0 = u32x4{1u, lds0 + (unsigned) lds_off, (unsigned) ga,
                     (unsigned) (ga >> 32) | 0x80000000u};
        d.g1 = u32x8{w0,
                     (TDIM0 & 0xFFFFu) << 16,
                     ((tdim1 & 0xFFFFu) << 16) | (TDIM0 >> 16),
                     ((tdim1 >> 16) & 0xFFFFu) | (tile0 << 16),
                     tile1,
                     (unsigned) stride_el,
                     (unsigned) (stride_el >> 32) & 0xFFFFu,
                     0u};
        return d;
    };

    // ---- operand bursts
    // ----------------------------------------------------------------------------------------- K^T
    // of a kv32 row whose K image (480-B rows) starts at LDS offset kimg
    auto burst_k = [&](OpSet<NQW> &S, int kimg) DQ_INL {
        const int kb = kimg + c_k_lane(lane);
        sfor<0, NDT>([&](auto D) DQ_INL {
            constexpr int dt = decltype(D)::value;
            S.kt[dt] = cat16(tr16(lds, kb + c_k_imm(dt, 0)), tr16(lds, kb + c_k_imm(dt, 1)));
        });
    };
    // dS^T of the wave's sub-tiles: row-0 image block 0 of the reader at r0, row-1 image block 0 at
    // r1
    auto burst_ds = [&](OpSet<NQW> &S, int r0, int r1) DQ_INL {
        const int d0 = r0 + ds_lane(lane);
        const int d1 = r1 + ds_lane(lane);
        sfor<0, NQW>([&](auto Q) DQ_INL {
            constexpr int h = decltype(Q)::value;
            S.ds[h]         = cat16(tr16(lds, d0 + ds_imm(h)), tr16(lds, d1 + ds_imm(h)));
        });
    };
    // the whole burst of kv32 row r (its stages hold landed, published data)
    auto burst = [&](OpSet<NQW> &S, int r) DQ_INL {
        const int u = r / 2, h = r % 2;
        const int slot = c_dsslot(u, NQW, h); // the op of dS wave h holds kv32 row 2u + h
        const int sh   = dsop_sh(NB_G, dsop_nv(qp0g, NB_G, r, cshift, causal));
        const int rb   = ds_wave_off(wv, NQW); // this wave's first q pair in the slot
        burst_k(S, c_kstage(u) + h * C_KHALF_B);
        burst_ds(S, slot + rb, ds_row1(slot, NB_G, sh) + rb);
    };

    // ---- the NQW x 12 WMMA run of one kv32 row (MASK: masked units select fully masked q pairs'
    // operands to zero) -
    f32x8 acc[NACC];
    sfor<0, NACC>([&](auto I) DQ_INL { acc[decltype(I)::value] = zero8(); });
    const bf16x16 zero16 = {};
    auto          run    = [&](const OpSet<NQW> &S, auto MASK, int r) DQ_INL {
        bf16x16 b[NQW];
        sfor<0, NQW>([&](auto Q) DQ_INL {
            constexpr int h = decltype(Q)::value;
            if constexpr (decltype(MASK)::value != 0) {
                b[h] = x4_ds_zero(qp0w + h / 2, r, n32, cshift, causal) ? zero16 : S.ds[h];
            } else {
                b[h] = S.ds[h];
            }
        });
        sfor<0, NQW>([&](auto Q) DQ_INL {
            constexpr int h = decltype(Q)::value;
            sfor<0, NDT>([&](auto D) DQ_INL {
                constexpr int dt  = decltype(D)::value;
                acc[h * NDT + dt] = wmma_bf16(S.kt[dt], b[h], acc[h * NDT + dt]);
            });
        });
        if constexpr (decltype(MASK)::value != 0) {
            // the selects ride the run: sub-tile 0's 8 first, then {1 WMMA, 1 VALU} (bwd_r6_a's
            // k_dqds192 pattern)
            Q1_SGB(0x002, 8);
            sfor<0, 8 * (NQW - 1)>([&](auto) DQ_INL {
                Q1_SGB(0x008, 1);
                Q1_SGB(0x002, 1);
            });
            Q1_SGB(0x008, NACC - 8 * (NQW - 1));
        }
        Q1_SCHED_BARRIER();
    };

    // X4: the WMMAs [LO, HI) of a run (the masked form selects its operands and carries the selects' pattern
    // in the first part only)
    auto run_rng = [&](const OpSet<NQW> &S, auto MASK, int r, auto LO, auto HI) DQ_INL {
        constexpr int lo = decltype(LO)::value, hi = decltype(HI)::value;
        bf16x16 b[NQW];
        sfor<0, NQW>([&](auto Q) DQ_INL {
            constexpr int h = decltype(Q)::value;
            if constexpr (decltype(MASK)::value != 0) {
                b[h] = x4_ds_zero(qp0w + h / 2, r, n32, cshift, causal) ? zero16 : S.ds[h];
            } else {
                b[h] = S.ds[h];
            }
        });
        sfor<lo, hi>([&](auto I) DQ_INL {
            constexpr int i = decltype(I)::value, h = i / NDT, dt = i % NDT;
            acc[i] = wmma_bf16(S.kt[dt], b[h], acc[i]);
        });
        if constexpr (decltype(MASK)::value != 0 && lo == 0) {
            Q1_SGB(0x002, 8);
            sfor<0, hi>([&](auto I) DQ_INL {
                Q1_SGB(0x008, 1);
                if constexpr (decltype(I)::value < 8 * (NQW - 1)) Q1_SGB(0x002, 1);
            });
        }
        Q1_SCHED_BARRIER();
    };
    // X4: the WMMAs [LO, HI) of a steady run (no selects, no trailing sched_barrier: interleaved regions)
    auto run_raw = [&](const OpSet<NQW> &S, auto LO, auto HI) DQ_INL {
        constexpr int lo = decltype(LO)::value, hi = decltype(HI)::value;
        sfor<lo, hi>([&](auto I) DQ_INL {
            constexpr int i = decltype(I)::value, h = i / NDT, dt = i % NDT;
            acc[i] = wmma_bf16(S.kt[dt], S.ds[h], acc[i]);
        });
    };

    // ---- this wave's role op of unit u (16-B dummy past the last unit)
    // ------------------------------------------ REAL = ic<1>: the caller guarantees u < nu (the
    // steady-state loop): no dummy selects. Branch-free SALU (the *_hot forms of idx.h, the role a
    // wave-uniform select) so the descriptor of the next barrier can be prepared inside the current
    // WMMA run.
    const int       nqp    = dsl_nqp(Sq);
    const int       mshift = cshift / 32;
    const long long nblk   = dsl_nblk_bh(Sq, Skv, causal);
    auto            c_desc = [&](int u, auto REAL) DQ_INL -> TdmDesc {
        const bool live = decltype(REAL)::value != 0 || u < nu;
        // K role: rows [64u + 32 hh, +32), all 192 columns
        const int       kv0 = c_kop_kv0(u, hh);
        const long long gel = k_el(bat, Skv, kv0, Hkv, hkv, 0);
        const TdmDesc   dk =
            desc(c_kop_lds(u, hh), live ? ga_k + (unsigned long long) (gel * 2) : ga_dummy, BK_W0,
                 live ? (unsigned) c_kop_dim1(Skv, u, hh) : 1u, live ? (unsigned) D_QK : 8u,
                 live ? (unsigned) KV32 : 1u, live ? (unsigned long long) kst : 0ull);
        // dS role: kv32 row 2u + hh x the workgroup's stored q pairs
        const int       r    = 2 * u + hh;
        const int       slot = c_dsslot(u, NQW, hh);
        const int       nv   = dsop_nv_hot(qp0g, NB_G, r, mshift, causal);
        const bool      go   = live && nv >= 1;
        const long long src = dsop_src_el_hot((long long) hb, nblk, r, qp0g, nqp, mshift, causal);
        const long long str = dsop_stride_el_hot(r, nqp, mshift, causal);
        const TdmDesc   dd  = desc(go ? dsop_lds(slot, NB_G, nv) : slot,
                                go ? ga_ds + (unsigned long long) (src * 2) : ga_dummy, DS_W0,
                                go ? 2u : 1u, go ? (unsigned) (nv * DS_BLK_EL) : 8u, go ? 2u : 1u,
                                go ? (unsigned long long) str : 0ull);
        TdmDesc d;
        d.g0 = dsw ? dd.g0 : dk.g0;
        d.g1 = dsw ? dd.g1 : dk.g1;
        return d;
    };
    auto issue   = [&](const TdmDesc &d) DQ_INL { tdm_load(d.g0, d.g1); };
    auto c_issue = [&](int u) DQ_INL { issue(c_desc(u, ic<0>{})); };

    // ---- one 64-kv unit: rows 2u (set X, burst into Y) and 2u + 1 (set Y, burst into X unless
    // LAST) --------------
    OpSet<NQW> X, Y;
    auto       unit = [&](int u, auto MASK, auto LAST, auto REAL) DQ_INL {
        constexpr bool last = decltype(LAST)::value != 0;
        const int      r0 = 2 * u, r1 = 2 * u + 1;
        if constexpr (!last && decltype(MASK)::value == 0 && NQW == MAIN_NQW) {
            constexpr int NS = X4_NS < NACC ? X4_NS : NACC;
            // region A: burst Y (row 2u + 1, unit u's stages) under run X part 1, the descriptor SALU too
            burst(Y, r1);
            const TdmDesc dn = c_desc(u + 3, REAL);
            run_raw(X, ic<0>{}, ic<NS>{});
            x4_sgb_il<X4_LA, 1, 24 + 2 * NQW, NS>();
            Q1_SCHED_BARRIER();
            Q1_WAIT_TENSORCNT(C_TW);
            Q1_WAIT_DSCNT(0);
            x4_bar_signal(); // P_{u+1} arrive: op(u+1) landed, unit u's ring reads retired
            Q1_SCHED_BARRIER();
            run_raw(X, ic<NS>{}, ic<NACC>{});
            Q1_SCHED_BARRIER();
            x4_bar_wait(); // P_{u+1}: unit u+1 published, unit u's stages free
            Q1_SCHED_BARRIER();
            issue(dn); // op(u + 3) into unit u's stages
            Q1_SCHED_BARRIER();
            // region C: burst X (row 2u + 2, unit u + 1's stages) under run Y
            burst(X, r1 + 1);
            run_raw(Y, ic<0>{}, ic<NACC>{});
            x4_sgb_il<X4_LC, 1, 24 + 2 * NQW, NACC>();
            Q1_SCHED_BARRIER();
            return;
        }
        if constexpr (!last) {
            // head-tile steady units and all masked units: split barrier only (bursts as in the shipped
            // kernel), arrive after X4_NS * NACC / 72 WMMAs
            constexpr int NS = X4_NS * NACC / (MAIN_NQW * NDT);
            burst(Y, r1);
            Q1_SCHED_BARRIER();
            const TdmDesc dn = c_desc(u + 3, REAL);
            run_rng(X, MASK, r0, ic<0>{}, ic<NS>{});
            Q1_WAIT_TENSORCNT(C_TW);
            Q1_WAIT_DSCNT(0);
            x4_bar_signal();
            Q1_SCHED_BARRIER();
            run_rng(X, MASK, r0, ic<NS>{}, ic<NACC>{});
            x4_bar_wait();
            Q1_SCHED_BARRIER();
            issue(dn);
            Q1_SCHED_BARRIER();
            burst(X, r1 + 1);
            Q1_SCHED_BARRIER();
            run(Y, MASK, r1);
            return;
        }
        // half-step r0
        burst(Y, r1);
        Q1_SCHED_BARRIER();
        TdmDesc dn;
        if constexpr (!last) {
            // the descriptor of op(u + 3), issued at P_{u+1} (pure SALU, scheduled into the WMMA
            // run below)
            dn = c_desc(u + 3, REAL);
        }
        run(X, MASK, r0);
        // half-step r1
        if constexpr (!last) {
            Q1_SCHED_BARRIER();
            Q1_WAIT_TENSORCNT(C_TW);
            wg_sync(); // P_{u+1}: unit u+1 published, unit u's stages free
            issue(dn); // op(u + 3) into unit u's stages
            Q1_SCHED_BARRIER();
            burst(X, r1 + 1);
            Q1_SCHED_BARRIER();
        }
        run(Y, MASK, r1);
    };

    // ---- prologue
    // ---------------------------------------------------------------------------------------------
    c_issue(0);
    c_issue(1);
    Q1_SCHED_BARRIER();
    Q1_WAIT_TENSORCNT(C_TW);
    wg_sync(); // P_0: unit 0 published
    c_issue(2);
    Q1_SCHED_BARRIER();
    burst(X, 0);
    Q1_SCHED_BARRIER();

    // ---- units: full ones (no selects), masked ones, then the last unit (masked form, no burst
    // past the end) ------ The steady-state loop also requires every op it issues to be real (u + 3
    // < nu): causal main tiles have exactly 6 masked units, the head 4, so this is the full-unit
    // loop itself; the generic body (selects + dummies) runs the rest.
    const int nloop = nu - 1;
    const int nf    = steady_units(nfu, nu);
    for (int u = 0; u < nf; ++u)
        unit(u, ic<0>{}, ic<0>{}, ic<1>{});
    for (int u = nf; u < nloop; ++u)
        unit(u, ic<1>{}, ic<0>{}, ic<0>{});
    unit(nloop, ic<1>{}, ic<1>{}, ic<0>{});

    // ---- exit: every wave's TDM landed (incl. dummy prefetches), every ring read retired
    // --------------------------
    Q1_WAIT_TENSORCNT(0);
    wg_sync();
    // dQ epilogue (bwd_hk_q1): dQ^T fragments (lane = q row, 8 consecutive d) -> row-major bf16
    // image, then whole 128-B lines back and one buffer_store_b128 per 4 lines' worth of lanes
    const rsrc_t g_dq = make_rsrc(DQ, DQ_EXTENT_B, RSRC_FLAGS);
    const int    img  = epi_img(wv, NQW);
    sfor<0, NQW>([&](auto Q) DQ_INL {
        constexpr int h = decltype(Q)::value;
        sfor<0, NDT>([&](auto D) DQ_INL {
            constexpr int dt = decltype(D)::value;
            lds_st(lds, img + epi_w_lane(lane) + epi_w_imm(h, dt), cvt_bf16x8(acc[h * NDT + dt]));
        });
    });
    sfor<0, bqw(NQW) / 4>([&](auto RB) DQ_INL {
        constexpr int rb = decltype(RB)::value;
        sfor<0, D_QK / DQ_LINE_EL>([&](auto LI) DQ_INL {
            constexpr int li = decltype(LI)::value;
            const bf16x8  vv = lds_ld(lds, img + epi_r_lane(lane) + epi_r_imm(rb, li));
            buf_st128(g_dq, 16 * dq_vec8(bat, Sq, Hq, qh, q0, lane, rb, li), vv);
        });
    });
}

// The kernel entry: grid (HG, ntile + nhead, B*Hq/HG), block 128.
Q1_DEV void dqds_kernel(const __bf16 *DS, const __bf16 *K, __bf16 *DQ, int Sq, int Skv, int Hq,
                        int Hkv, int G, int nkvt, int cshift, int causal, int q_off, int ntile,
                        int HG) {
    lock_simd(); // SCHED_MODE.DISABLE_XDL_ARB_STALL, as bwd_r6_a's k_dqds192 / k_dqg96 / k_dqds_b
    const int gy = block_id_y();
    if (!tile_is_head(gy, ntile)) {
        dqds_body<MAIN_NQW>(DS, K, DQ, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal,
                            tile_q0g(q_off, ntile, gy), HG);
    } else {
        dqds_body<HEAD_NQW>(DS, K, DQ, Sq, Skv, Hq, Hkv, G, nkvt, cshift, causal, head_q0g(q_off),
                            HG);
    }
}

} // namespace dq
