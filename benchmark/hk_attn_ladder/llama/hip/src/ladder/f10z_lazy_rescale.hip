/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// f10z_lazy_rescale -- Llama forward ladder (main line): f09m with a lazy rescale behind ONE unlikely branch per chunk.
//
// Kernel Specification
//   work      workgroup = 64 query positions x FOUR q heads (a head quad, G % 4 == 0: one kv head); wave w =
//             positions [16 w, +16) of the 4 heads = 4 q tiles; Q^T of 4 heads resident (128 VGPR)
//   grid      B * Hq/4 * Sq/64, longest first: rank r takes q block nqb - 1 - r of every (b, head quad)
//   kv        64-row blocks = 2 chunks of 32 rows, 3-slot TDM ring (105 KB), as f05p
//   pipeline  as f05p, every phase for 4 heads: per chunk the 8 K operands (one burst) feed 32 S^T WMMAs and the 8 V^T
//             operands (one burst) 32 PV WMMAs; S^T(c + 1) of the 4 heads interleaves with the 4 heads' softmax of c
//   sync      as f05p
//   softmax   online, LAZY, as v0 with HK_FWD_OPT_RESC: right after S^T(c + 1) is computed (after PV(c), outside the
//             interleaved region) the column max of each head is taken; ONE ballot over max_h (x2 - m2) > 8 nat
//             decides the rare rescale, which runs out of line (__builtin_expect) with v0's per-head decision; the
//             next chunk's region holds only the branch-free exps / row sums with the (possibly stale) max
//
// What changes from f09m: the eager rescale multiplies the 4 heads' O^T accumulators (256 values) by exp2(m_old -
// m_new) on every chunk, although after the first chunks the running max rarely moves. Keeping the stale max is exact
// while p = exp2(s c2 - m) cannot overflow (8 nat: p <= e^8) and the final normalisation divides the same factor out of
// O and l. One branch per chunk, laid out of line: a per-head branch at every chunk end (side rung f09z) cut the
// interleaved schedule and lost 5%.
#include "ladder/lf.h"

using namespace hk_attn;

namespace {

constexpr int BUF_B = lf::K_TILE_B + lf::V_TILE_B; // one slot: K tile, then V tile
constexpr int NSLOT = 3;
constexpr int LDS_B = NSLOT * BUF_B;                // 107520
constexpr float TH2 = 8.0f * 1.4426950408889634f; // lazy-rescale threshold: 8 nat in the log2 domain
constexpr int NO_MASK = 64; // a mask_row_gt_col offset that masks nothing (thr >= 56 > 7)
template <int N> using ic = std::integral_constant<int, N>;
#define INL __attribute__((always_inline)) // every phase lambda inlined (a non-inlined one takes its captures through the stack)

using k_half = tdm_tile<32, lf::D, lf::K_PITCH>;
using v_half = tdm_tile<32, lf::D, lf::V_PITCH>;

} // namespace

// one wave per SIMD by design (4 heads per wave): let the register allocator use the whole file
extern "C" __global__ __launch_bounds__(lf::BLOCK) __attribute__((amdgpu_waves_per_eu(1, 1))) void k_fwd_f10z(
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
    auto k_lds = [&](int j) INL { return lds0 + (uint32_t) ((j % NSLOT) * BUF_B); };
    auto v_lds = [&](int j) INL { return lds0 + (uint32_t) ((j % NSLOT) * BUF_B + lf::K_TILE_B); };
    const rsrc_t rq = make_rsrc(Q, q_bytes);

    opnd_t qt[NH][lf::NDK];
    acc_t  oacc[NH][lf::NDT];
    float  m2[NH], lsum[NH];
    static_for<NH>([&](auto h) {
        lf::load_qt(qt[h], rq, lf::row_b(b, Sq, p0, Hq, NH * hq4 + h), Hq * lf::D * 2, l);
        static_for<lf::NDT>([&](auto dt) { oacc[h][dt] = zero_acc(); });
        m2[h]   = lf::BIG_NEG;
        lsum[h] = 0.f;
    });

    const int   role_v = w >> 1, half = w & 1;
    const u32x8 g1 = role_v ? tdm_g1<v_half>((int64_t) Hkv * lf::D, 32) : tdm_g1<k_half>((int64_t) Hkv * lf::D, 32);
    auto issue = [&](int j) INL {
        const int      row0 = lf::KVB * j + 32 * half;
        const uint32_t dst  = role_v ? v_lds(j) + (uint32_t) (32 * half * lf::V_PITCH)
                                     : k_lds(j) + (uint32_t) (32 * half * lf::K_PITCH);
        tdm_load(tdm_g0(dst, tdm_gaddr(role_v ? (const void *) V : (const void *) K, (int64_t) b * Skv + row0,
                                       (int64_t) Hkv * lf::D, (int64_t) hkv * lf::D)),
                 g1);
    };

    // chunk c = kv rows [32 c, +32): block c / 2, half c % 2 (an S^T chunk = 2 accumulators of 16 kv rows)
    auto kload = [&](opnd_t (&ka)[2][lf::NDK], int c) INL {
        const lds_u8 *kbase = rows_base<lf::K_PITCH>(k_lds(c / 2), l);
        static_for<2>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) { ka[kt][k] = load_rows<lf::K_PITCH>(kbase, 32 * (c % 2) + 16 * kt, 32 * k); });
        });
    };
    auto qk = [&](acc_t (&s)[NH][2], const opnd_t (&ka)[2][lf::NDK]) INL { // each K operand feeds the 4 heads
        static_for<2>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) {
                static_for<NH>([&](auto h) {
                    if constexpr (decltype(k)::value == 0) s[h][kt] = wmma0(ka[kt][0], qt[h][0]);
                    else s[h][kt] = wmma(ka[kt][k], qt[h][k], s[h][kt]);
                });
            });
        });
    };
    auto mask = [&](acc_t (&s)[NH][2], int c) INL {
        static_for<NH>([&](auto h) {
            static_for<2>([&](auto kt) {
                s[h][kt] = mask_row_gt_col(s[h][kt], l, causal ? p0 + cshift - (32 * c + 16 * kt) : NO_MASK);
            });
        });
    };
    auto vload = [&](opnd_t (&va)[lf::NDT], int c) INL {
        const lds_u8 *vbase = tr_base<lf::V_PITCH>(v_lds(c / 2), l);
        static_for<lf::NDT>([&](auto dt) { va[dt] = load_tr<lf::V_PITCH>(vbase, 32 * (c % 2), 16 * dt); });
    };
    // SM1 of an S^T chunk: column max of each head, ONE ballot; the rare rescale out of line (v0 HK_FWD_OPT_RESC)
    auto sm1 = [&](acc_t (&s)[NH][2]) INL {
        // max over the heads of (column max - running max): no local array (one kept on the stack cost 2.4 KB scratch)
        float d = -__builtin_inff();
        static_for<NH>([&](auto h) { d = __builtin_fmaxf(d, lf::qmax(s[h]) * c2 - m2[h]); });
        if (__builtin_expect(any_set(ballot(d > TH2)), 0)) {
            static_for<NH>([&](auto h) {
                const float x2 = lf::qmax(s[h]) * c2; // recomputed in the rare block
                if (any_set(ballot(x2 - m2[h] > TH2))) {
                    const float m2n  = __builtin_fmaxf(m2[h], x2);
                    const float corr = exp2_hw(m2[h] - m2n);
                    lsum[h] *= corr;
                    static_for<lf::NDT>([&](auto dt) { oacc[h][dt] = oacc[h][dt] * corr; });
                    m2[h] = m2n;
                }
            });
        }
    };
    auto softmax = [&](acc_t (&s)[NH][2]) INL { // tails: branch-free exps / row sums with the current max; s becomes P
        static_for<NH>([&](auto h) { lsum[h] += lf::exp_tiles(s[h], c2, m2[h]); });
    };
    auto pv = [&](acc_t (&p)[NH][2], const opnd_t (&va)[lf::NDT]) INL { // each V^T operand feeds the 4 heads
        opnd_t pt[NH];
        static_for<NH>([&](auto h) { pt[h] = pack_pt_fwd(p[h][0], p[h][1]); });
        static_for<lf::NDT>([&](auto dt) {
            static_for<NH>([&](auto h) { oacc[h][dt] = wmma(va[dt], pt[h], oacc[h][dt]); });
        });
    };
    // one pipelined chunk c (S^T(c) in sc, S^T(c + 1) into sn; chunk c + 1 exists)
    // MASK: S^T(c + 1) belongs to the last (diagonal) block and is masked
    auto chunk = [&](int c, acc_t (&sc)[NH][2], acc_t (&sn)[NH][2], auto MASK) INL {
        opnd_t va[lf::NDT], ka[2][lf::NDK];
        vload(va, c);
        kload(ka, c + 1);
        sched_barrier<0>();
        qk(sn, ka); // independent of the softmax: one region, interleaved by the co-execution scheduler
        softmax(sc);
        pv(sc, va);
        if constexpr (decltype(MASK)::value) mask(sn, c + 1);
        sm1(sn); // after PV(c): a rescale applies to everything up to chunk c
    };

    // ---- prologue: blocks 0 (and 1) requested, block 0 published, S^T(chunk 0) --------------------------
    issue(0);
    if (n > 1) issue(1);
    if (n > 1) split_signal_tdm_lds<1>(); // P_0: own op of block 0 landed (block 1's may fly)
    else split_signal_tdm_lds<0>();
    split_wait_lds();
    if (n > 1) split_signal_tdm_lds<0>(); // P_1: own op of block 1 landed (block 0's reads come later: no WAR)
    acc_t s0[NH][2], s1[NH][2];
    {
        opnd_t ka[2][lf::NDK];
        kload(ka, 0);
        sched_barrier<0>();
        qk(s0, ka);
    }
    if (n == 1) mask(s0, 0); // block 0 is the diagonal block
    sm1(s0);                 // m2 starts at BIG_NEG: the first chunk always rescales (corr = 0, O and l are 0)

    // ---- blocks: chunk 2j (its successor 2j + 1 is in block j), then P_j+1, block j + 2's ops, chunk 2j + 1 ------
    // blocks 0 .. n - 3: nothing to mask; block n - 2: its second chunk computes the first S^T of the last block
    auto body = [&](int j, auto MASK_A, auto MASK_B) INL {
        chunk(2 * j, s0, s1, MASK_A);
        split_wait_lds();                     // P_j+1: block j + 1 published; block j - 1 released by every wave
        if (j + 2 < n) issue(j + 2);          // into block j - 1's slot
        chunk(2 * j + 1, s1, s0, MASK_B);     // S^T(2j + 2) from block j + 1 into s0
        if (j + 2 < n) split_signal_tdm_lds<0>(); // P_j+2: own op of block j + 2 landed, block j's reads retired
    };
    int j = 0;
    for (; j + 2 < n; ++j) body(j, ic<0>{}, ic<0>{});
    if (j + 1 < n) { // block n - 2: S^T(2n - 2) (first chunk of the last block) is masked
        body(j, ic<0>{}, ic<1>{});
        ++j;
    }
    // the last block j = n - 1: chunk 2j computes S^T(2j + 1), masked; then the last chunk
    chunk(2 * j, s0, s1, ic<1>{});
    {
        opnd_t va[lf::NDT];
        vload(va, 2 * j + 1);
        softmax(s1);
        pv(s1, va);
    }
    static_for<NH>([&](auto h) { lf::store_out(a, oacc[h], lsum[h], m2[h], b, NH * hq4 + h, p0, l); });
}
