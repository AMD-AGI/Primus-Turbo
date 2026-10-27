/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// f08_lazy_rescale -- rung 08 of the Llama forward ladder: f07 with a lazy online-softmax rescale.
//
// Kernel Specification
//   work      workgroup = 64 query positions x FOUR q heads (a head quad, G % 4 == 0, so one kv head);
//             wave w = positions [16 w, +16) of the 4 heads = 4 q tiles
//   grid      B * Hq/4 * Sq/64, longest first: dispatch rank r takes q block nqb - 1 - r of every
//             (b, head quad) (v0's f0::work_v0, FlyDSL's order)
//   kv        64-row blocks, a RING of 4 LDS slots: 4 x (K tile (pitch 272) + V tile (pitch 288)) = 140 KB
//   fill      TDM: per block every wave issues ONE 32-row op (waves 0 / 1: K rows [0, 32) / [32, 64),
//             waves 2 / 3: the same of V) straight into the padded LDS tile; block j + 2's ops are
//             issued at the start of block j, so two blocks are in flight under every block's compute
//   sync      per block: 1 SPLIT barrier phase. SIGNAL(P_j+1) right after block j's S^T (own op of block
//             j + 1 landed, j + 2 may fly), WAIT(P_j+1) at the top of block j + 1, then block j + 3's ops;
//             softmax(j) and PV(j) run between signal and wait. Slot (j + 3) % 4 = block j - 1's: every
//             wave signalled P_j+1 after its S^T(j), i.e. after PV(j - 1) (and the signal drains its DS
//             loads), so no wave still reads it
//   Q         Q^T operands loaded once from global straight into registers (4 heads: 128 VGPR)
//   S^T       K Q^T: 64 WMMA per block; every K operand read from LDS feeds 4 WMMAs (one per head)
//   softmax   online, lazy: O and l are rescaled only when some query's max grew by more than 8 nat
//             (one ballot, one wave-uniform branch per head); causal mask on every block
//   PV        O^T += V^T P^T: 64 WMMA per block; every V^T operand (ds_load_tr16_b128) feeds 4
//
// What changes from f07: the eager rescale multiplies all 64 O^T accumulator values of every head
// by exp2(m_old - m_new) on every block, although after the first few blocks the running max
// rarely moves. Keeping the stale max is exact as long as exp2(s c2 - m) cannot overflow: the
// threshold of 8 nat bounds p by e^8 (bf16 / fp32 headroom is ample), and the final normalisation
// divides the same factor out of O and l (FlyDSL / v0's lazy rescale, TH2 = 8 log2 e).
#include "ladder/lf.h"

using namespace hk_attn;

namespace {

constexpr int BUF_B = lf::K_TILE_B + lf::V_TILE_B; // one buffer: K tile, then V tile
constexpr int NSLOT = 4;
constexpr float TH2 = 8.0f * 1.4426950408889634f; // lazy-rescale threshold: 8 nat in the log2 domain
constexpr int LDS_B = NSLOT * BUF_B;                // 143360

// one TDM op = half a tile: 32 rows x 128 bf16 into LDS rows of the K or the V pitch (tdm.h)
using k_half = tdm_tile<32, lf::D, lf::K_PITCH>;
using v_half = tdm_tile<32, lf::D, lf::V_PITCH>;

} // namespace

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_fwd_f08(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, __bf16 *O, float *LSE, long long q_bytes,
    long long kv_bytes, long long o_bytes, long long lse_bytes, int B, int Sq, int Skv, int Hq, int Hkv,
    int causal, float c2) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    lock_simd();
    const lf::Args a{Q, K, V, O, LSE, q_bytes, kv_bytes, o_bytes, lse_bytes, B, Sq, Skv, Hq, Hkv, causal, c2};

    // ---- work: q block fastest ---------------------------------------------------------------------
    const int w = wave_id(), l = lane_id();
    constexpr int NH = 4; // q heads per workgroup = q tiles per wave
    const int nqb = Sq / lf::QB, nhq4 = Hq / NH, i = (int) __builtin_amdgcn_workgroup_id_x();
    const int nh = B * nhq4, rank = i / nh, rem = i - rank * nh;
    const int qb = nqb - 1 - rank, hq4 = rem % nhq4, b = rem / nhq4;
    const int hkv = NH * hq4 / (Hq / Hkv);
    const int p0  = lf::QB * qb + lf::QW * w;
    const int n   = lf::n_kv_blocks(qb, Sq, Skv, causal);
    const int cshift = Skv - Sq;

    const uint32_t lds0 = lds_addr(smem);
    // slot of block j: j % 4 (K at its start, V after K)
    auto k_lds = [&](int j) { return lds0 + (uint32_t) ((j % NSLOT) * BUF_B); };
    auto v_lds = [&](int j) { return lds0 + (uint32_t) ((j % NSLOT) * BUF_B + lf::K_TILE_B); };
    const rsrc_t   rq = make_rsrc(Q, q_bytes);

    opnd_t qt[NH][lf::NDK];
    acc_t  oacc[NH][lf::NDT];
    float  m2[NH], lsum[NH];
    static_for<NH>([&](auto h) {
        lf::load_qt(qt[h], rq, lf::row_b(b, Sq, p0, Hq, NH * hq4 + h), Hq * lf::D * 2, l);
        static_for<lf::NDT>([&](auto dt) { oacc[h][dt] = zero_acc(); });
        m2[h]   = lf::BIG_NEG;
        lsum[h] = 0.f;
    });

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
        split_signal_tdm_lds<1>(); // SIGNAL(P_0): own op of block 0 landed (block 1's may fly)
    } else {
        split_signal_tdm_lds<0>();
    }
    for (int j = 0; j < n; ++j) {
        // ---- WAIT(P_j): block j published, block j - 2 released by every wave; block j + 2's ops ---------
        const bool next = j + 1 < n, next2 = j + 2 < n; // workgroup-uniform
        split_wait_lds();
        if (next2) issue(j + 2);
        const lds_u8 *kbase = rows_base<lf::K_PITCH>(k_lds(j), l);
        const lds_u8 *vbase = tr_base<lf::V_PITCH>(v_lds(j), l);

        // ---- S^T = K Q^T (rows kv, lane = query) -----------------------------------------------------
        acc_t s[NH][lf::NKT];
        static_for<lf::NKT>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) {
                const opnd_t ka = load_rows<lf::K_PITCH>(kbase, 16 * kt, 32 * k);
                static_for<NH>([&](auto h) {
                    if constexpr (decltype(k)::value == 0) s[h][kt] = wmma0(ka, qt[h][0]);
                    else s[h][kt] = wmma(ka, qt[h][k], s[h][kt]);
                });
            });
        });
        if (causal) // kv tile origin 64 j + 16 kt, q tile origin p0 (frag.h mask_row_gt_col)
            static_for<NH>([&](auto h) {
                static_for<lf::NKT>([&](auto kt) {
                    s[h][kt] = mask_row_gt_col(s[h][kt], l, p0 + cshift - (lf::KVB * j + 16 * kt));
                });
            });

        // ---- SIGNAL(P_j+1): own op of block j + 1 landed, K(j) read (DS drained by the signal) ------------
        if (next) {
            if (next2) split_signal_tdm_lds<1>();
            else split_signal_tdm_lds<0>();
        }

        // ---- online softmax, lazy rescale ----------------------------------------------------------------
        static_for<NH>([&](auto h) {
            const float x2 = lf::qmax(s[h]) * c2;
            if (any_set(ballot(x2 - m2[h] > TH2))) { // wave-uniform, rarely taken after the first block
                const float m2n  = __builtin_fmaxf(m2[h], x2);
                const float corr = exp2_hw(m2[h] - m2n);
                lsum[h] *= corr;
                static_for<lf::NDT>([&](auto dt) { oacc[h][dt] = oacc[h][dt] * corr; });
                m2[h] = m2n;
            }
            lsum[h] += lf::exp_tiles(s[h], c2, m2[h]);
        });

        // ---- O^T += V^T P^T ---------------------------------------------------------------------------
        static_for<lf::NKC>([&](auto c) {
            opnd_t pt[NH];
            static_for<NH>([&](auto h) { pt[h] = pack_pt_fwd(s[h][2 * c], s[h][2 * c + 1]); });
            static_for<lf::NDT>([&](auto dt) {
                const opnd_t va = load_tr<lf::V_PITCH>(vbase, 32 * c, 16 * dt);
                static_for<NH>([&](auto h) { oacc[h][dt] = wmma(va, pt[h], oacc[h][dt]); });
            });
        });

    }
    static_for<NH>([&](auto h) { lf::store_out(a, oacc[h], lsum[h], m2[h], b, NH * hq4 + h, p0, l); });
}
