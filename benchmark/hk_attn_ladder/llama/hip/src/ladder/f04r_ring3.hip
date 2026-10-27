/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// f04r_ring3 -- Llama forward ladder (main line): f03b with a 3-slot TDM ring, prefetch distance 2.
//
// Kernel Specification
//   work      workgroup = 64 query positions x ONE q head; wave w = positions [16 w, +16)
//   grid      B * Hq * Sq/64, q block fastest (natural order)
//   kv        64-row blocks, a RING of 3 LDS slots: 3 x (K tile (pitch 272) + V tile (pitch 288)) = 105 KB
//   fill      TDM: per block every wave issues ONE 32-row op (waves 0 / 1: K rows [0, 32) / [32, 64), waves 2 / 3: the
//             same of V); block j + 2's ops are issued at the top of block j, two blocks ahead
//   sync      per block: one barrier phase, signalled at the end of block j - 1 (own op of block j landed, j + 1 may
//             fly) and waited at the top of block j, before block j + 2's ops: those overwrite block j - 1's slot,
//             which every wave finished reading before its signal (the signal drains its DS loads)
//   Q         Q^T operands loaded once from global straight into registers (32 VGPR)
//   S^T       K Q^T: 16 WMMA per block; the 16 K operands of the block loaded first, as one burst (128 VGPR)
//   softmax   online, eager: every block rescales O and l; causal mask on every block
//   PV        O^T += V^T P^T: 16 WMMA per block; the 16 V^T operands (tr16) issued as one burst BEFORE the softmax,
//             so their LDS latency hides under its VALU
//
// What changes from f03b: the tiles are requested two blocks ahead instead of one, so a block's TDM latency (and an
// L2 / HBM miss) is covered by two blocks of compute. Mostly an enabler: the next rung computes block j + 1's S^T
// while block j's softmax runs, and needs block j + 1 in LDS at the top of block j.
#include "ladder/lf.h"

using namespace hk_attn;

namespace {

constexpr int BUF_B = lf::K_TILE_B + lf::V_TILE_B; // one buffer: K tile, then V tile
constexpr int NSLOT = 3;
constexpr int LDS_B = NSLOT * BUF_B;                // 107520

// one TDM op = half a tile: 32 rows x 128 bf16 into LDS rows of the K or the V pitch (tdm.h)
using k_half = tdm_tile<32, lf::D, lf::K_PITCH>;
using v_half = tdm_tile<32, lf::D, lf::V_PITCH>;

} // namespace

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_fwd_f04r(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, __bf16 *O, float *LSE, long long q_bytes,
    long long kv_bytes, long long o_bytes, long long lse_bytes, int B, int Sq, int Skv, int Hq, int Hkv,
    int causal, float c2) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const lf::Args a{Q, K, V, O, LSE, q_bytes, kv_bytes, o_bytes, lse_bytes, B, Sq, Skv, Hq, Hkv, causal, c2};

    // ---- work: q block fastest ---------------------------------------------------------------------
    const int w = wave_id(), l = lane_id();
    const int nqb = Sq / lf::QB, i = (int) __builtin_amdgcn_workgroup_id_x();
    const int qb = i % nqb, hq = (i / nqb) % Hq, b = i / (nqb * Hq);
    const int hkv = hq / (Hq / Hkv);
    const int p0  = lf::QB * qb + lf::QW * w;
    const int n   = lf::n_kv_blocks(qb, Sq, Skv, causal);
    const int cshift = Skv - Sq;

    const uint32_t lds0 = lds_addr(smem);
    // slot of block j: j % 3 (K at its start, V after K)
    auto k_lds = [&](int j) { return lds0 + (uint32_t) ((j % NSLOT) * BUF_B); };
    auto v_lds = [&](int j) { return lds0 + (uint32_t) ((j % NSLOT) * BUF_B + lf::K_TILE_B); };
    const rsrc_t   rq = make_rsrc(Q, q_bytes);

    opnd_t qt[lf::NDK];
    lf::load_qt(qt, rq, lf::row_b(b, Sq, p0, Hq, hq), Hq * lf::D * 2, l);
    acc_t oacc[lf::NDT];
    static_for<lf::NDT>([&](auto dt) { oacc[dt] = zero_acc(); });
    float m2 = lf::BIG_NEG, lsum = 0.f;

    // this wave's op of block j: role (K / V) and half by wave, everything wave-uniform (SGPRs)
    const int role_v = w >> 1, half = w & 1;
    const u32x8 g1 = role_v ? tdm_g1<v_half>((int64_t) Hkv * lf::D, 32) : tdm_g1<k_half>((int64_t) Hkv * lf::D, 32);
    auto issue = [&](int j) {
        const int      row0 = lf::KVB * j + 32 * half; // Skv % 64 == 0: all 32 rows exist
        const uint32_t dst  = role_v ? v_lds(j) + (uint32_t) (32 * half * lf::V_PITCH)
                                     : k_lds(j) + (uint32_t) (32 * half * lf::K_PITCH);
        tdm_load(tdm_g0(dst, tdm_gaddr(role_v ? (const void *) V : (const void *) K, (int64_t) b * Skv + row0,
                                       (int64_t) Hkv * lf::D, (int64_t) hkv * lf::D)),
                 g1);
    };
    issue(0);
    if (n > 1) {
        issue(1);
        split_signal_tdm_lds<1>(); // P_0: own op of block 0 landed (block 1's may fly)
    } else {
        split_signal_tdm_lds<0>();
    }
    for (int j = 0; j < n; ++j) {
        // ---- WAIT(P_j): block j published, block j - 1 released by every wave; then block j + 2's ops -------
        const bool next = j + 1 < n, next2 = j + 2 < n; // workgroup-uniform
        split_wait_lds();
        if (next2) issue(j + 2);
        const lds_u8 *kbase = rows_base<lf::K_PITCH>(k_lds(j), l);
        const lds_u8 *vbase = tr_base<lf::V_PITCH>(v_lds(j), l);

        // ---- S^T = K Q^T (rows kv, lane = query) -----------------------------------------------------
        opnd_t ka[lf::NKT][lf::NDK]; // the block's K operands: one burst, then the WMMA run
        static_for<lf::NKT>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) { ka[kt][k] = load_rows<lf::K_PITCH>(kbase, 16 * kt, 32 * k); });
        });
        sched_barrier<0>();
        acc_t s[lf::NKT];
        static_for<lf::NKT>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) {
                if constexpr (decltype(k)::value == 0) s[kt] = wmma0(ka[kt][0], qt[0]);
                else s[kt] = wmma(ka[kt][k], qt[k], s[kt]);
            });
        });
        if (causal) // kv tile origin 64 j + 16 kt, q tile origin p0 (frag.h mask_row_gt_col)
            static_for<lf::NKT>([&](auto kt) {
                s[kt] = mask_row_gt_col(s[kt], l, p0 + cshift - (lf::KVB * j + 16 * kt));
            });

        // ---- the V^T operands of PV: one burst, issued before the softmax (lands under its VALU) ------------
        opnd_t va[lf::NKC][lf::NDT];
        static_for<lf::NKC>([&](auto c) {
            static_for<lf::NDT>([&](auto dt) { va[c][dt] = load_tr<lf::V_PITCH>(vbase, 32 * c, 16 * dt); });
        });
        sched_barrier<0>();

        // ---- online softmax, eager rescale ---------------------------------------------------------------
        const float m2n  = __builtin_fmaxf(m2, lf::qmax(s) * c2);
        const float corr = exp2_hw(m2 - m2n);
        lsum             = lsum * corr + lf::exp_tiles(s, c2, m2n);
        static_for<lf::NDT>([&](auto dt) { oacc[dt] = oacc[dt] * corr; });
        m2 = m2n;

        // ---- O^T += V^T P^T ---------------------------------------------------------------------------
        static_for<lf::NKC>([&](auto c) {
            const opnd_t pt = pack_pt_fwd(s[2 * c], s[2 * c + 1]);
            static_for<lf::NDT>([&](auto dt) {
                oacc[dt] = wmma(va[c][dt], pt, oacc[dt]);
            });
        });

        // ---- SIGNAL(P_j+1): own op of block j + 1 landed (j + 2's may fly), every DS read of block j retired ----
        if (next) {
            if (next2) split_signal_tdm_lds<1>();
            else split_signal_tdm_lds<0>();
        }
    }
    lf::store_out(a, oacc, lsum, m2, b, hq, p0, l);
}
