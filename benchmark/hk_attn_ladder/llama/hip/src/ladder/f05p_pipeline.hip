/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// f05p_pipeline -- Llama forward ladder (main line): f04r software-pipelined on 32-row chunks: chunk c + 1's S^T is
// computed under chunk c's softmax.
//
// Kernel Specification
//   work      workgroup = 64 query positions x ONE q head; wave w = positions [16 w, +16)
//   grid      B * Hq * Sq/64, q block fastest (natural order)
//   kv        64-row blocks = 2 chunks of 32 rows, a RING of 3 LDS slots (105 KB); TDM as f04r (block j + 2's ops in
//             the middle of block j)
//   pipeline  the top of chunk c holds S^T(c) (masked); chunk c loads V(c) (8 tr16 operands) and K(c + 1) (8
//             operands) as two bursts, then computes S^T(c + 1) = K(c + 1) Q^T (8 WMMA) and the softmax of S^T(c)
//             in ONE branch-free region (interleaved by LLVM's co-execution scheduler, -amdgpu-sched-strategy=coexec
//             with -structurizecfg-skip-uniform-regions: v0's build flags), then PV(c) (8 WMMA) and the causal mask
//             of S^T(c + 1) (branch-free offset)
//   sync      one barrier phase P_j+1 per block: signalled at the end of block j - 1 (own op of block j + 1 landed,
//             every DS read of block j - 1 retired), waited in the middle of block j, right before chunk 2j + 1 loads
//             K(2j + 2) (block j + 1); then block j + 2's ops go into block j - 1's slot
//   softmax   online, eager rescale per chunk
//
// What changes from f04r: f04r's softmax waits for its block's S^T and the next S^T waits for the softmax; here the
// next chunk's S^T and this chunk's softmax are independent work in one region, so the WMMA pipe runs while the
// VALU / transcendental units do the softmax (v0's "QK(c+1) || tails(c)"). 32-row chunks keep the operand bursts at
// 64 + 64 VGPRs (a 64-row block pipeline needs 128 + 128 and loses a wave per SIMD: 0.90x, measured).
#include "ladder/lf.h"

using namespace hk_attn;

namespace {

constexpr int BUF_B = lf::K_TILE_B + lf::V_TILE_B; // one slot: K tile, then V tile
constexpr int NSLOT = 3;
constexpr int LDS_B = NSLOT * BUF_B;                // 107520
constexpr int NO_MASK = 64;                         // a mask_row_gt_col offset that masks nothing (thr >= 56 > 7)

using k_half = tdm_tile<32, lf::D, lf::K_PITCH>;
using v_half = tdm_tile<32, lf::D, lf::V_PITCH>;

} // namespace

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_fwd_f05p(
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
    auto k_lds = [&](int j) { return lds0 + (uint32_t) ((j % NSLOT) * BUF_B); };
    auto v_lds = [&](int j) { return lds0 + (uint32_t) ((j % NSLOT) * BUF_B + lf::K_TILE_B); };
    const rsrc_t rq = make_rsrc(Q, q_bytes);

    opnd_t qt[lf::NDK];
    lf::load_qt(qt, rq, lf::row_b(b, Sq, p0, Hq, hq), Hq * lf::D * 2, l);
    acc_t oacc[lf::NDT];
    static_for<lf::NDT>([&](auto dt) { oacc[dt] = zero_acc(); });
    float m2 = lf::BIG_NEG, lsum = 0.f;

    const int   role_v = w >> 1, half = w & 1;
    const u32x8 g1 = role_v ? tdm_g1<v_half>((int64_t) Hkv * lf::D, 32) : tdm_g1<k_half>((int64_t) Hkv * lf::D, 32);
    auto issue = [&](int j) {
        const int      row0 = lf::KVB * j + 32 * half;
        const uint32_t dst  = role_v ? v_lds(j) + (uint32_t) (32 * half * lf::V_PITCH)
                                     : k_lds(j) + (uint32_t) (32 * half * lf::K_PITCH);
        tdm_load(tdm_g0(dst, tdm_gaddr(role_v ? (const void *) V : (const void *) K, (int64_t) b * Skv + row0,
                                       (int64_t) Hkv * lf::D, (int64_t) hkv * lf::D)),
                 g1);
    };

    // chunk c = kv rows [32 c, +32): block c / 2, half c % 2 (an S^T chunk = 2 accumulators of 16 kv rows)
    auto kload = [&](opnd_t (&ka)[2][lf::NDK], int c) {
        const lds_u8 *kbase = rows_base<lf::K_PITCH>(k_lds(c / 2), l);
        static_for<2>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) { ka[kt][k] = load_rows<lf::K_PITCH>(kbase, 32 * (c % 2) + 16 * kt, 32 * k); });
        });
    };
    auto qk = [&](acc_t (&s)[2], const opnd_t (&ka)[2][lf::NDK]) {
        static_for<2>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) {
                if constexpr (decltype(k)::value == 0) s[kt] = wmma0(ka[kt][0], qt[0]);
                else s[kt] = wmma(ka[kt][k], qt[k], s[kt]);
            });
        });
    };
    auto mask = [&](acc_t (&s)[2], int c) {
        static_for<2>([&](auto kt) {
            s[kt] = mask_row_gt_col(s[kt], l, causal ? p0 + cshift - (32 * c + 16 * kt) : NO_MASK);
        });
    };
    auto vload = [&](opnd_t (&va)[lf::NDT], int c) {
        const lds_u8 *vbase = tr_base<lf::V_PITCH>(v_lds(c / 2), l);
        static_for<lf::NDT>([&](auto dt) { va[dt] = load_tr<lf::V_PITCH>(vbase, 32 * (c % 2), 16 * dt); });
    };
    auto softmax = [&](acc_t (&s)[2]) { // eager rescale; s becomes P
        const float m2n  = __builtin_fmaxf(m2, lf::qmax(s) * c2);
        const float corr = exp2_hw(m2 - m2n);
        lsum             = lsum * corr + lf::exp_tiles(s, c2, m2n);
        static_for<lf::NDT>([&](auto dt) { oacc[dt] = oacc[dt] * corr; });
        m2 = m2n;
    };
    auto pv = [&](acc_t (&p)[2], const opnd_t (&va)[lf::NDT]) {
        const opnd_t pt = pack_pt_fwd(p[0], p[1]);
        static_for<lf::NDT>([&](auto dt) { oacc[dt] = wmma(va[dt], pt, oacc[dt]); });
    };
    // one pipelined chunk c (S^T(c) in sc, S^T(c + 1) into sn; chunk c + 1 exists)
    auto chunk = [&](int c, acc_t (&sc)[2], acc_t (&sn)[2]) {
        opnd_t va[lf::NDT], ka[2][lf::NDK];
        vload(va, c);
        kload(ka, c + 1);
        sched_barrier<0>();
        qk(sn, ka); // independent of the softmax: one region, interleaved by the co-execution scheduler
        softmax(sc);
        pv(sc, va);
        mask(sn, c + 1);
    };

    // ---- prologue: blocks 0 (and 1) requested, block 0 published, S^T(chunk 0) --------------------------
    issue(0);
    if (n > 1) issue(1);
    if (n > 1) split_signal_tdm_lds<1>(); // P_0: own op of block 0 landed (block 1's may fly)
    else split_signal_tdm_lds<0>();
    split_wait_lds();
    if (n > 1) split_signal_tdm_lds<0>(); // P_1: own op of block 1 landed (block 0's reads come later: no WAR)
    acc_t s0[2], s1[2];
    {
        opnd_t ka[2][lf::NDK];
        kload(ka, 0);
        sched_barrier<0>();
        qk(s0, ka);
    }
    mask(s0, 0);

    // ---- blocks: chunk 2j (its successor 2j + 1 is in block j), then P_j+1, block j + 2's ops, chunk 2j + 1 ------
    for (int j = 0; j < n; ++j) {
        chunk(2 * j, s0, s1);
        if (j + 1 < n) {
            split_wait_lds();                     // P_j+1: block j + 1 published; block j - 1 released by every wave
            if (j + 2 < n) issue(j + 2);          // into block j - 1's slot
            chunk(2 * j + 1, s1, s0);             // S^T(2j + 2) from block j + 1 into s0
            if (j + 2 < n) split_signal_tdm_lds<0>(); // P_j+2: own op of block j + 2 landed, block j's reads retired
        } else { // the last chunk
            opnd_t va[lf::NDT];
            vload(va, 2 * j + 1);
            softmax(s1);
            pv(s1, va);
        }
    }
    lf::store_out(a, oacc, lsum, m2, b, hq, p0, l);
}
