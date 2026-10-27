/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// f04_longest_first -- rung 04 of the Llama forward ladder: f03 with the heaviest workgroups dispatched first.
//
// Kernel Specification
//   work      workgroup = 64 query positions x FOUR q heads (a head quad, G % 4 == 0, so one kv head);
//             wave w = positions [16 w, +16) of the 4 heads = 4 q tiles
//   grid      B * Hq/4 * Sq/64, longest first: dispatch rank r takes q block nqb - 1 - r of every
//             (b, head quad) (v0's f0::work_v0, FlyDSL's order)
//   kv        64-row blocks, TWO LDS buffers: 2 x (K tile (pitch 272) + V tile (pitch 288)) = 70 KB
//   fill      TDM: per block every wave issues ONE 32-row op (waves 0 / 1: K rows [0, 32) / [32, 64),
//             waves 2 / 3: the same of V) straight into the padded LDS tile; block j + 1's ops are
//             issued before block j's compute and waited for (s_wait_tensorcnt 0) after it
//   sync      per block: 1 full barrier (publishes block j + 1; block j's buffer is refilled one
//             block later, after every wave has passed that barrier)
//   Q         Q^T operands loaded once from global straight into registers (4 heads: 128 VGPR)
//   S^T       K Q^T: 64 WMMA per block; every K operand read from LDS feeds 4 WMMAs (one per head)
//   softmax   online, eager: every block rescales O and l; causal mask on every block
//   PV        O^T += V^T P^T: 64 WMMA per block; every V^T operand (ds_load_tr16_b128) feeds 4
//
// What changes from f03: only the order. Causal work grows with the q block (block qb sees qb + 1 kv
// blocks), and the natural order puts every head's heavy blocks at the end of its run, so the grid
// ends in a tail of heavy workgroups on few CUs. Heaviest first (longest-processing-time order)
// lets the light blocks fill in behind them.
#include "ladder/lf.h"

using namespace hk_attn;

namespace {

constexpr int BUF_B = lf::K_TILE_B + lf::V_TILE_B; // one buffer: K tile, then V tile
constexpr int LDS_B = 2 * BUF_B;                    // 71680

// one TDM op = half a tile: 32 rows x 128 bf16 into LDS rows of the K or the V pitch (tdm.h)
using k_half = tdm_tile<32, lf::D, lf::K_PITCH>;
using v_half = tdm_tile<32, lf::D, lf::V_PITCH>;

} // namespace

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_fwd_f04(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, __bf16 *O, float *LSE, long long q_bytes,
    long long kv_bytes, long long o_bytes, long long lse_bytes, int B, int Sq, int Skv, int Hq, int Hkv,
    int causal, float c2) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
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
    // buffer of block j: j & 1 (K at its start, V after K)
    auto k_lds = [&](int j) { return lds0 + (uint32_t) ((j & 1) * BUF_B); };
    auto v_lds = [&](int j) { return lds0 + (uint32_t) ((j & 1) * BUF_B + lf::K_TILE_B); };
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
    split_signal_tdm_lds<0>(); // own op landed
    split_wait_lds();          // block 0 published
    for (int j = 0; j < n; ++j) {
        // ---- block j + 1: TDM ops now, waited for and published after this block's compute -----------------
        const bool next = j + 1 < n; // workgroup-uniform
        if (next) issue(j + 1);
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

        // ---- online softmax, eager rescale ---------------------------------------------------------------
        static_for<NH>([&](auto h) {
            const float m2n  = __builtin_fmaxf(m2[h], lf::qmax(s[h]) * c2);
            const float corr = exp2_hw(m2[h] - m2n);
            lsum[h]          = lsum[h] * corr + lf::exp_tiles(s[h], c2, m2n);
            static_for<lf::NDT>([&](auto dt) { oacc[h][dt] = oacc[h][dt] * corr; });
            m2[h] = m2n;
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

        // ---- publish block j + 1 (its buffer was last read in block j - 1, before the previous barrier) --
        if (next) {
            split_signal_tdm_lds<0>(); // own op of block j + 1 landed
            split_wait_lds();
        }
    }
    static_for<NH>([&](auto h) { lf::store_out(a, oacc[h], lsum[h], m2[h], b, NH * hq4 + h, p0, l); });
}
