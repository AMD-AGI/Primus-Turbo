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
//   NQW 6 -> the 384-query tiles of [q_off, Sq): 4 waves x 96 queries (3 q pairs each), longest first
//   NQW 4 -> the head [0, 256) (q_off == 256): 4 waves x 64 queries (2 q pairs each), the same ring protocol
// q pairs are dealt to the waves in snake order (wave w: pairs w, 7 - w, 8 + w of the workgroup's 12; head
// tile w, 7 - w; idx.h wave_pair), so the fully masked pairs of the diagonal units spread over all waves.
// Per kv32 row r (block rows 2r, 2r + 1) every wave runs NQW x 12 WMMA (dQ^T += K^T dS^T), operands
// burst-loaded one row ahead into the other register set (X / Y: 12 K^T + NQW dS^T v16 operands
// each):
//   half-step j:  [sync + TDM issue] -> burst of row j + 1 (24 + 2 NQW tr16) -> NQW x 12 WMMA on
//   row j
// Ring protocol: K ring 3 x 64 rows, dS ring 3 x 2 kv32-row slots; ONE op per wave per 64-kv unit
// (waves 0 / 3: the dS of kv32 row 2u + (w >> 1) for all NB_G q pairs of the workgroup, waves 1 /
// 2: K rows [64u + 32 (w >> 1), +32)); wait 1 -> barrier P_{u+1} -> op(u + 3) at the odd row of
// every unit only; <= 2 TDM ops in flight per wave; ops past the last unit are 16-B dummies. Fully
// masked q pairs are never loaded (the dS op starts at the first stored pair, dslayout); in the masked
// units a wave-uniform branch skips their WMMAs (they would add +-0 to a non -0 accumulator); full units
// carry no branch. Epilogue: per 16-row sub-tile, dQ^T fragments -> this wave's row-major 16-row image
// after the rings -> whole-line b128 stores. Every index formula: ../../idx.h (replayed on the host
// by bounds_proof.py). No atomics, no named barriers, no clusters (prims.h); the only inline asm is an
// empty statement that keeps operands allocated (keep_ds).
//
// Scheduling (needs -mllvm -amdgpu-expert-scheduling-mode, manifest extra_flags):
// * every unit but the last: split barrier P_{u+1}: arrive after the first SPLIT_AT (x NACC / 72 on the head tile)
//   WMMAs of run(X), wait after run(X). The ring protocol is unchanged: before the arrive the wave's op(u + 1)
//   landed (tensorcnt 1) and its ring reads (burst(Y)) retired (dscnt 0); between arrive and wait only WMMAs;
//   op(u + 3) and burst(X) after the wait.
// * main-tile steady units: the operand bursts interleaved into the WMMA runs (expert scheduling: a DS read
//   no longer waits for earlier WMMAs):
//     region A: burst(Y, 2u + 1) under run(X) WMMAs [0, SPLIT_AT): {IL_LEAD_A WMMA} then {1 tr16, 1 WMMA} x 36
//     region C: burst(X, 2u + 2) under run(Y):                     {IL_LEAD_C WMMA} then {1 tr16, 1 WMMA} x 36
//   The steady loop is unrolled by the ring depth (3 units: the ring stage is a compile-time constant), so
//   every burst address is a precomputed base VGPR + immediate and the loop carries no VALU: with only WMMAs
//   in flight the compiler's expert-mode waits before the bursts are counted (va_vdst(k)) instead of full
//   drains (a VALU mixed with in-flight WMMAs forces va_vdst(0)). keep_ds() holds each run's dS operands to
//   the end of its region so the burst loads cannot be allocated to a register the newest WMMA just read.
// * masked units: a uniform branch per q pair skips the fully masked pairs (no operand selects).
#pragma once
#pragma clang fp contract(off)

#include "../q1/prims.h"
#include "idx.h" // arm root (build.py: -Iinclude -I.)

static_assert(DSL_VERSION == 1 && DSL_SCALED == 1,
              "bwd_hk_dqds2 reads dslayout v1 with the scale included");

static_assert(dq::lds_bytes() <= 327680 && 2 * dq::lds_bytes() > 327680,
              "one workgroup per CU, inside the 320-KiB LDS");

namespace q1 {
// The kernel's only LDS: the rings, then the per-wave dQ sub-tile images (the name is the one the emulator
// expects).
__shared__ __attribute__((aligned(16))) char q1_smem[dq::lds_bytes()];
} // namespace q1

namespace dq {
using namespace q1;

#define DQ_INL __attribute__((always_inline))

constexpr int SPLIT_AT = 66, IL_LEAD_A = 8, IL_LEAD_C = 4;
Q1_DEV void split_arrive() {
    __builtin_amdgcn_fence(3, "workgroup", "local"); // release
    __builtin_amdgcn_s_barrier_signal(-1);
}
Q1_DEV void split_wait() {
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(2, "workgroup", "local"); // acquire
}
template <int L, int R, int NT, int NW> Q1_DEV void sgb_interleave() {
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
Q1_DEV bool ds_zero_branchless(int qp, int r, int n32, int cshift, int causal) {
    const int f   = r - cshift / 32;
    const int qpf = (causal != 0) & (f > 0) ? f : 0;
    return (bool) ((int) (r >= n32) | (int) (qp < qpf));
}

struct TdmDesc { // one TDM op's descriptor groups 0 / 1 (prepared before the barrier, issued after it)
    u32x4 g0;
    u32x8 g1;
};

template <int NQW> struct OpSet {
    bf16x16 kt[NDT]; // A operands: K^T of the 12 d tiles
    bf16x16 ds[NQW]; // B operands: dS^T of the wave's 16-query sub-tiles
};
// emits no instruction: the dS operands of S stay allocated (live) up to this point
template <int NQW> Q1_DEV void keep_ds(const OpSet<NQW> &S) {
    static_assert(NQW == 6, "main tile only");
    asm volatile("" ::"v"(S.ds[0]), "v"(S.ds[1]), "v"(S.ds[2]), "v"(S.ds[3]), "v"(S.ds[4]), "v"(S.ds[5]));
}

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
    const int  qp0g = tile_qp0(q0g);
    const int  n32  = kv_n32(q0g, NQW, cshift, nkvt, causal);
    const int  nu   = n_units(n32);
    const int  nfu  = n_full_units(qp0g, cshift, nu, causal);

    lds_char *const lds = (lds_char *) q1_smem;

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
    // dS^T of the wave's sub-tiles: row-0 image of the workgroup's pair 0 at r0, row-1 image at r1
    auto burst_ds = [&](OpSet<NQW> &S, int r0, int r1) DQ_INL {
        const int d0 = r0 + ds_lane(lane);
        const int d1 = r1 + ds_lane(lane);
        sfor<0, NQW>([&](auto Q) DQ_INL {
            constexpr int h  = decltype(Q)::value;
            const int     po = wave_pair(wv, h / 2, NQW) * DSI_BLK_B + (h % 2) * 32;
            S.ds[h]          = cat16(tr16(lds, d0 + po), tr16(lds, d1 + po));
        });
    };
    // the whole burst of kv32 row r (its stages hold landed, published data)
    auto burst = [&](OpSet<NQW> &S, int r) DQ_INL {
        const int u = r / 2, h = r % 2;
        const int slot = c_dsslot(u, NQW, h); // the op of dS wave h holds kv32 row 2u + h
        const int sh   = dsop_sh(NB_G, dsop_nv(qp0g, NB_G, r, cshift, causal));
        burst_k(S, c_kstage(u) + h * C_KHALF_B);
        burst_ds(S, slot, ds_row1(slot, NB_G, sh));
    };

    // ---- the NQW x 12 WMMA run of one kv32 row
    f32x8 acc[NACC];
    sfor<0, NACC>([&](auto I) DQ_INL { acc[decltype(I)::value] = zero8(); });
    // the wave's k-th q pair holds no attended entry at kv32 row r (wave-uniform)
    auto pair_zero = [&](int k, int r) DQ_INL {
        return ds_zero_branchless(qp0g + wave_pair(wv, k, NQW), r, n32, cshift, causal);
    };
    // the WMMAs [LO, HI) of a run; MASK (masked units): a uniform branch per q pair skips the pair's WMMAs when
    // it is fully masked (its operands were never loaded)
    auto run_part = [&](const OpSet<NQW> &S, auto MASK, int r, auto LO, auto HI) DQ_INL {
        constexpr int lo = decltype(LO)::value, hi = decltype(HI)::value;
        if constexpr (decltype(MASK)::value != 0) {
            sfor<0, NQW / 2>([&](auto Kp) DQ_INL {
                constexpr int k  = decltype(Kp)::value;
                constexpr int a  = 2 * k * NDT > lo ? 2 * k * NDT : lo;
                constexpr int bb = 2 * (k + 1) * NDT < hi ? 2 * (k + 1) * NDT : hi;
                if constexpr (a < bb) {
                    if (!pair_zero(k, r)) {
                        sfor<a, bb>([&](auto I) DQ_INL {
                            constexpr int i = decltype(I)::value, h = i / NDT, dt = i % NDT;
                            acc[i] = wmma_bf16(S.kt[dt], S.ds[h], acc[i]);
                        });
                    }
                }
            });
            Q1_SCHED_BARRIER();
            return;
        }
        sfor<lo, hi>([&](auto I) DQ_INL {
            constexpr int i = decltype(I)::value, h = i / NDT, dt = i % NDT;
            acc[i] = wmma_bf16(S.kt[dt], S.ds[h], acc[i]);
        });
        Q1_SCHED_BARRIER();
    };
    auto run = [&](const OpSet<NQW> &S, auto MASK, int r) DQ_INL {
        run_part(S, MASK, r, ic<0>{}, ic<NACC>{});
    };
    // the WMMAs [LO, HI) of a steady run (no trailing sched_barrier: interleaved regions)
    auto run_steady = [&](const OpSet<NQW> &S, auto LO, auto HI) DQ_INL {
        constexpr int lo = decltype(LO)::value, hi = decltype(HI)::value;
        sfor<lo, hi>([&](auto I) DQ_INL {
            constexpr int i = decltype(I)::value, h = i / NDT, dt = i % NDT;
            acc[i] = wmma_bf16(S.kt[dt], S.ds[h], acc[i]);
        });
    };

    const unsigned           lds0  = lds_addr(lds);
    const unsigned long long ga_ds = (unsigned long long) DS;
    const unsigned long long ga_k  = (unsigned long long) K;
    const long long          kst   = k_stride_el(Hkv);
    // dummy op source: K[bat, 0, hkv, 0..7] (always inside K)
    const unsigned long long ga_dummy =
        ga_k + (unsigned long long) (k_el(bat, Skv, 0, Hkv, hkv, 0) * 2);

    // ---- one TDM op (bwd_hk_q1's word layout); every field is wave-uniform
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
    // ---- this wave's role op of unit u (16-B dummy past the last unit). REAL = ic<1>: the caller guarantees
    // u + 3 < nu (the steady-state loop): no dummy selects and nv >= 1 (rows < n32), so no runtime check keeps
    // the steady body branch-free. Branch-free SALU (the *_hot forms of idx.h, the role a wave-uniform select)
    // so the descriptor of the next barrier can be prepared inside the current WMMA run.
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
        const bool      go   = decltype(REAL)::value != 0 || (live && nv >= 1);
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
    // main-tile steady unit u (interleaved regions A / C); BY / BX issue the bursts of rows 2u + 1 / 2u + 2
    auto steady_unit = [&](int u, auto BY, auto BX) DQ_INL {
        constexpr int NS = SPLIT_AT < NACC ? SPLIT_AT : NACC;
        // region A: burst Y (row 2u + 1, unit u's stages) under run X part 1, the descriptor SALU too
        BY();
        const TdmDesc dn = c_desc(u + 3, ic<1>{});
        run_steady(X, ic<0>{}, ic<NS>{});
        sgb_interleave<IL_LEAD_A, 1, 24 + 2 * NQW, NS>();
        Q1_SCHED_BARRIER();
        keep_ds(X);
        Q1_WAIT_TENSORCNT(C_TW);
        Q1_WAIT_DSCNT(0);
        split_arrive(); // P_{u+1} arrive: op(u+1) landed, unit u's ring reads retired
        Q1_SCHED_BARRIER();
        run_steady(X, ic<NS>{}, ic<NACC>{});
        Q1_SCHED_BARRIER();
        split_wait(); // P_{u+1}: unit u+1 published, unit u's stages free
        Q1_SCHED_BARRIER();
        issue(dn); // op(u + 3) into unit u's stages
        Q1_SCHED_BARRIER();
        // region C: burst X (row 2u + 2, unit u + 1's stages) under run Y
        BX();
        run_steady(Y, ic<0>{}, ic<NACC>{});
        sgb_interleave<IL_LEAD_C, 1, 24 + 2 * NQW, NACC>();
        Q1_SCHED_BARRIER();
        keep_ds(Y);
    };

    auto unit = [&](int u, auto MASK, auto LAST, auto REAL) DQ_INL {
        constexpr bool last = decltype(LAST)::value != 0;
        const int      r0 = 2 * u, r1 = 2 * u + 1;
        if constexpr (!last && decltype(MASK)::value == 0 && NQW == MAIN_NQW) {
            steady_unit(u, [&]() DQ_INL { burst(Y, r1); }, [&]() DQ_INL { burst(X, r1 + 1); });
            return;
        }
        if constexpr (!last) {
            // head-tile steady units and all masked units: split barrier only, arrive after SPLIT_AT * NACC / 72
            // WMMAs
            constexpr int NS = SPLIT_AT * NACC / (MAIN_NQW * NDT);
            burst(Y, r1);
            Q1_SCHED_BARRIER();
            const TdmDesc dn = c_desc(u + 3, REAL);
            run_part(X, MASK, r0, ic<0>{}, ic<NS>{});
            Q1_WAIT_TENSORCNT(C_TW);
            Q1_WAIT_DSCNT(0);
            split_arrive();
            Q1_SCHED_BARRIER();
            run_part(X, MASK, r0, ic<NS>{}, ic<NACC>{});
            split_wait();
            Q1_SCHED_BARRIER();
            issue(dn);
            Q1_SCHED_BARRIER();
            burst(X, r1 + 1);
            Q1_SCHED_BARRIER();
            run(Y, MASK, r1);
            return;
        }
        // the last unit: no barrier, no op, no burst past the end
        burst(Y, r1);
        Q1_SCHED_BARRIER();
        run(X, MASK, r0);
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

    // ---- units: full ones (no branches), masked ones, then the last unit (masked form, no burst
    // past the end) ------ The steady-state loop also requires every op it issues to be real (u + 3
    // < nu): causal main tiles have exactly 6 masked units, the head 4, so this is the full-unit
    // loop itself; the generic body (skips + dummies) runs the rest.
    const int nloop = nu - 1;
    const int nf    = steady_units(nfu, nu);
    int       u     = 0;
    if constexpr (NQW == MAIN_NQW) {
        // the unrolled steady units burst row 2u + 2 assuming all of its pairs are stored (sh = 0); for the last
        // steady unit that row belongs to the first masked unit, whose stored pairs start mid-row when
        // (Skv - Sq) % 64 == 32: then the last steady unit runs as a generic unit (its bursts handle sh) below
        const int nfs = ((cshift & 63) == 32 && nf > 0) ? nf - 1 : nf;
        // steady units, unrolled by the ring depth: precomputed burst bases, K stages 0 / 1 off kb01
        // (immediates <= 54176), stage 2 off kb2; dS stage s off db[s] (the wave's pair 0; pair 2 = pair 0 + 8
        // blocks) and db1[s] (pair 1)
        const int kb01 = c_k_lane(lane);
        const int kb2  = kb01 + 2 * C_KSTG_B;
        const int dl   = ds_lane(lane) + wave_pair(wv, 0, NQW) * DSI_BLK_B;
        const int d1o  = (wave_pair(wv, 1, NQW) - wave_pair(wv, 0, NQW)) * DSI_BLK_B;
        int       db[3], db1[3];
        sfor<0, 3>([&](auto Sg) DQ_INL {
            constexpr int s = decltype(Sg)::value;
            db[s]           = dl + C_DS_OFF + s * c_dsstg_b(NQW);
            db1[s]          = db[s] + d1o;
        });
        // the burst of kv32 row 2u + H of a steady unit u with u % 3 == s (all NB_G pairs stored: sh = 0)
        auto burst_stage = [&](OpSet<NQW> &Sx, auto Sg, auto Hg) DQ_INL {
            constexpr int s = decltype(Sg)::value, H = decltype(Hg)::value;
            constexpr int ko = (s == 2 ? 0 : s * C_KSTG_B) + H * C_KHALF_B;
            const int     kb = s == 2 ? kb2 : kb01;
            sfor<0, NDT>([&](auto D) DQ_INL {
                constexpr int dt = decltype(D)::value;
                Sx.kt[dt] = cat16(tr16(lds, kb + ko + c_k_imm(dt, 0)), tr16(lds, kb + ko + c_k_imm(dt, 1)));
            });
            constexpr int so = H * c_slot_b(NQW);
            constexpr int r1 = NB_G * DSI_BLK_B;
            sfor<0, NQW>([&](auto Q) DQ_INL {
                constexpr int h = decltype(Q)::value, k = h / 2;
                static_assert(wave_pair(0, 2, MAIN_NQW) - wave_pair(0, 0, MAIN_NQW) == 8, "pair 2 = pair 0 + 8");
                const int     base = k == 1 ? db1[s] : db[s];
                constexpr int po   = (k == 2 ? 8 * DSI_BLK_B : 0) + (h % 2) * 32;
                Sx.ds[h] = cat16(tr16(lds, base + so + po), tr16(lds, base + so + r1 + po));
            });
        };
        auto unit_stage = [&](int uu, auto Sg) DQ_INL {
            constexpr int s  = decltype(Sg)::value;
            constexpr int s1 = (s + 1) % 3;
            steady_unit(uu, [&]() DQ_INL { burst_stage(Y, ic<s>{}, ic<1>{}); },
                        [&]() DQ_INL { burst_stage(X, ic<s1>{}, ic<0>{}); });
        };
        for (; u + 3 <= nfs; u += 3) {
            unit_stage(u, ic<0>{});
            unit_stage(u + 1, ic<1>{});
            unit_stage(u + 2, ic<2>{});
        }
        if (u < nfs) { // nfs % 3 == 1 or 2 (u % 3 == 0 here)
            unit_stage(u, ic<0>{});
            ++u;
            if (u < nfs) {
                unit_stage(u, ic<1>{});
                ++u;
            }
        }
    }
    for (; u < nf; ++u) // head tile, and a main tile's last steady unit when (Skv - Sq) % 64 == 32
        unit(u, ic<0>{}, ic<0>{}, ic<1>{});
    for (u = nf; u < nloop; ++u)
        unit(u, ic<1>{}, ic<0>{}, ic<0>{});
    unit(nloop, ic<1>{}, ic<1>{}, ic<0>{});

    // ---- exit: every wave's TDM landed (incl. dummies), every ring read retired
    Q1_WAIT_TENSORCNT(0);
    wg_sync();
    const rsrc_t g_dq = make_rsrc(DQ, DQ_EXTENT_B, RSRC_FLAGS);
    Q1_SCHED_BARRIER();
    // dQ epilogue, per 16-row sub-tile: 12 dQ^T fragments (lane = q row, 8 consecutive d) -> this wave's
    // row-major image after the rings -> whole 128-B lines back, one buffer_store_b128 per 4 lines' worth of lanes
    const int img = epi_img(wv);
    sfor<0, NQW>([&](auto Q) DQ_INL {
        constexpr int h = decltype(Q)::value;
        sfor<0, NDT>([&](auto D) DQ_INL {
            constexpr int dt = decltype(D)::value;
            lds_st(lds, img + epi_w_lane(lane) + epi_w_imm(dt), cvt_bf16x8(acc[h * NDT + dt]));
        });
        const int qs = subtile_q0(q0g, wv, h, NQW);
        sfor<0, 4>([&](auto RB) DQ_INL {
            constexpr int rb = decltype(RB)::value;
            sfor<0, D_QK / DQ_LINE_EL>([&](auto LI) DQ_INL {
                constexpr int li = decltype(LI)::value;
                const bf16x8  vv = lds_ld(lds, img + epi_r_lane(lane) + epi_r_imm(rb, li));
                buf_st128(g_dq, 16 * dq_vec8(bat, Sq, Hq, qh, qs, lane, rb, li), vv);
            });
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
