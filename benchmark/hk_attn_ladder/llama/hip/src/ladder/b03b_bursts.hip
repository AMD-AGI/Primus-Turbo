/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// b03b_bursts -- Llama backward ladder (main line): b02 with LDS operands loaded in bursts, NL / delta early.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O), NL = -LSE log2 e                          grid (Sq/64, Hq, B), block 256
//   dkdv  workgroup = 64 kv rows of one (b, kv head); wave w = kv rows [16 w, +16), K / V resident (B operands)
//         loop: every visible 64-query block x every q head of the kv head's group:
//           fill Q and dO tiles (TDM, pitch 272, TWO buffers: waves 0 / 1 move Q rows [0, 32) / [32, 64),
//           waves 2 / 3 the same of dO; the next (q block, head)'s ops in flight during this one's compute,
//           s_wait_tensorcnt 0 + 1 full barrier after it),
//           NL / delta rows of the 4 q tiles: global loads issued first (consumed after the S / dP run),
//           S = Q K^T, dP = dO V^T (lane = kv): per q tile one burst of its 4 Q + 4 dO operands, issued before
//           the previous q tile's 8 WMMAs (two operand sets live), P, dS,
//           dV^T += dO^T P, dK^T += Q^T dS: chunk 0's 16 tr16 operands issued before the softmax, chunk 1's
//           before chunk 0's WMMAs
//         dK = scale dK^T, dV stored once                                  grid (Skv/64, Hkv, B), block 128
//   dq    workgroup = 64 queries of one q head; wave w = queries [16 w, +16), Q^T / dO^T resident
//         loop: every visible 64-row kv block: fill K and V tiles (TDM, two buffers, waves 0 / 1: K halves,
//           2 / 3: V halves; block j + 1's ops in flight during block j, 1 barrier),
//           S^T = K Q^T, dP^T = V dO^T (lane = q): per kv tile one burst of its 4 K + 4 V operands, issued
//           before the previous kv tile's 8 WMMAs, P^T, dS^T, dQ^T += K^T dS^T: the 16 K^T tr16 operands
//           issued before the softmax
//         dQ = scale dQ^T                                                   grid (Sq/64, Hq, B), block 128
// S and dP are computed twice (once per kernel): 7 GEMMs of 128-wide operands per (q, kv) pair instead of 5.
// What changes from b02: b02's thread traces spend 32-33% of the time in s_wait_dscnt (every WMMA waits for the
// LDS operand loaded right before it) and up to 9% in s_wait_loadcnt (the NL / delta rows, loaded from global
// right before the softmax that needs them). As in the forward ladder's f03b: a burst pays the LDS latency once
// per group of WMMAs, the next group's burst flies during this group's WMMAs, the tr16 operands of the dV / dK
// (dQ) WMMAs fly during the softmax, and the row vectors are requested a whole S / dP run ahead. The price is
// registers (two operand sets live).
#include "ladder/lb.h"

using namespace hk_attn;

namespace {

constexpr int P_B   = 2 * lf::D + 16;  // 272: every LDS tile here, rows and tr16 reads
constexpr int TILE  = 64 * P_B;         // 17408
constexpr int BUF_B = 2 * TILE;         // one buffer: two tiles, (Q, dO) in dkdv, (K, V) in dq
constexpr int LDS_B = 2 * BUF_B;        // two buffers

template <int N> using ic = std::integral_constant<int, N>;
using T_HALF = tdm_tile<32, lf::D, P_B>; // one op: 32 rows x 128 bf16 into LDS rows of pitch 272

// this wave's op: half (w & 1) of tile (w >> 1) of a buffer, rows row0 + 32 half of a [*, H, 128] tensor
HKA_DEV void issue_half(uint32_t buf, const void *t0, const void *t1, int b, int S, int row0, int H, int h, int w) {
    const int   half = w & 1;
    const void *src  = (w >> 1) ? t1 : t0;
    const u32x8 g1   = tdm_g1<T_HALF>((int64_t) H * lf::D, 32); // S % 64 == 0: all 32 rows exist
    tdm_load(tdm_g0(buf + (uint32_t) ((w >> 1) * TILE + 32 * half * P_B),
                    tdm_gaddr(src, (int64_t) b * S + row0 + 32 * half, (int64_t) H * lf::D, (int64_t) h * lf::D)),
             g1);
}

} // namespace

extern "C" __global__ __launch_bounds__(256) void k_bwd_b03b_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int Hq) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, Hq);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b03b_dkdv(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DK, __bf16 *DV, long long q_bytes, long long kv_bytes, long long row_bytes, int B, int Sq, int Skv,
    int Hq, int Hkv, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int kb = (int) __builtin_amdgcn_workgroup_id_x(), hkv = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int G = Hq / Hkv, kv0 = 64 * kb + 16 * w, cshift = Skv - Sq;
    const int q_stride = Hq * lf::D * 2, kv_stride = Hkv * lf::D * 2;

    const uint32_t lds0 = lds_addr(smem);
    auto q_lds  = [&](int it) { return lds0 + (uint32_t) ((it & 1) * BUF_B); };  // Q tile of iteration it
    auto do_lds = [&](int it) { return lds0 + (uint32_t) ((it & 1) * BUF_B + TILE); };
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, kv_bytes), rv = make_rsrc(V, kv_bytes);
    const rsrc_t   rdo = make_rsrc(DO, q_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);

    // K / V rows as B^T operands of S = Q K^T and dP = dO V^T (B[k = d][n = kv], lane = kv): resident
    opnd_t kb_op[lf::NDK], vb_op[lf::NDK];
    lf::load_qt(kb_op, rk, lf::row_b(b, Skv, kv0, Hkv, hkv), kv_stride, l);
    lf::load_qt(vb_op, rv, lf::row_b(b, Skv, kv0, Hkv, hkv), kv_stride, l);
    acc_t dkt[lf::NDT], dvt[lf::NDT]; // dK^T, dV^T: rows d, lane = kv
    static_for<lf::NDT>([&](auto dt) { dkt[dt] = zero_acc(); dvt[dt] = zero_acc(); });

    // iterations: (q block, head) pairs, heads inner: it -> q block qb0 + it / G, head hkv G + it % G
    const int qb0 = lb::first_q_block(kb, Sq, Skv, causal), nit = (Sq / 64 - qb0) * G;
    auto      issue = [&](int it) { // Q (waves 0, 1) and dO (waves 2, 3) of iteration it
        issue_half(q_lds(it), Q, DO, b, Sq, 64 * (qb0 + it / G), Hq, hkv * G + it % G, w);
    };
    if (nit > 0) {
        issue(0);
        split_signal_tdm_lds<0>();
        split_wait_lds();
    }
    for (int it = 0; it < nit; ++it) {
        {
            const int qb = qb0 + it / G, hq = hkv * G + it % G;
            const bool next = it + 1 < nit; // workgroup-uniform
            if (next) issue(it + 1);
            const lds_u8 *q_rows = rows_base<P_B>(q_lds(it), l), *do_rows = rows_base<P_B>(do_lds(it), l);
            const lds_u8 *q_tr = tr_base<P_B>(q_lds(it), l), *do_tr = tr_base<P_B>(do_lds(it), l);

            // NL / delta rows of the 4 q tiles (global): requested now, consumed after the S / dP run
            acc_t nl[4], del[4];
            static_for<4>([&](auto qt) {
                nl[qt]  = lb::rows8(rnl, b, Hq, hq, Sq, 64 * qb + 16 * qt, l);
                del[qt] = lb::rows8(rdel, b, Hq, hq, Sq, 64 * qb + 16 * qt, l);
            });
            // S = Q K^T, dP = dO V^T: 4 q tiles of 16 rows, lane = kv; per q tile one burst, then its 8 WMMAs
            acc_t s[4], dp[4];
            static_for<4>([&](auto qt) {
                opnd_t xa[2][lf::NDK]; // [Q, dO][k]
                static_for<lf::NDK>([&](auto k) {
                    xa[0][k] = load_rows<P_B>(q_rows, 16 * qt, 32 * k);
                    xa[1][k] = load_rows<P_B>(do_rows, 16 * qt, 32 * k);
                });
                sched_barrier<0>();
                static_for<lf::NDK>([&](auto k) {
                    if constexpr (decltype(k)::value == 0) {
                        s[qt]  = wmma0(xa[0][0], kb_op[0]);
                        dp[qt] = wmma0(xa[1][0], vb_op[0]);
                    } else {
                        s[qt]  = wmma(xa[0][k], kb_op[k], s[qt]);
                        dp[qt] = wmma(xa[1][k], vb_op[k], dp[qt]);
                    }
                });
                sched_barrier<0>();
            });
            // the dO^T / Q^T operands of dV / dK chunk c (tr16): chunk 0's burst flies during the softmax
            opnd_t ta[2][2][lf::NDT]; // [chunk][dO^T, Q^T][dt]
            auto   tload = [&](auto c) {
                static_for<lf::NDT>([&](auto dt) {
                    ta[c][0][dt] = load_tr<P_B>(do_tr, 32 * c, 16 * dt);
                    ta[c][1][dt] = load_tr<P_B>(q_tr, 32 * c, 16 * dt);
                });
            };
            tload(ic<0>{});
            sched_barrier<0>();
            // P and dS (row vectors NL / delta in accumulator-row order)
            static_for<4>([&](auto qt) {
                s[qt]  = lb::probs_qk(s[qt], c2, nl[qt], l, causal != 0, 64 * qb + 16 * qt + cshift - kv0);
                dp[qt] = lb::dsoft(s[qt], dp[qt], del[qt]);
            });
            // dV^T += dO^T P, dK^T += Q^T dS (32-query chunks); chunk 1's burst after chunk 0's WMMAs
            static_for<2>([&](auto c) {
                if constexpr (c == 1) {
                    tload(ic<1>{});
                    sched_barrier<0>();
                }
                const opnd_t pb = pack_p_bwd(s[2 * c], s[2 * c + 1]);
                const opnd_t sb = pack_p_bwd(dp[2 * c], dp[2 * c + 1]);
                static_for<lf::NDT>([&](auto dt) {
                    dvt[dt] = wmma(ta[c][0][dt], pb, dvt[dt]);
                    dkt[dt] = wmma(ta[c][1][dt], sb, dkt[dt]);
                });
                sched_barrier<0>();
            });
            // publish the next iteration's tiles (their buffer was last read an iteration earlier)
            if (next) {
                split_signal_tdm_lds<0>(); // own op of iteration it + 1 landed
                split_wait_lds();
            }
        }
    }
    const int row = lf::row_b(b, Skv, kv0 + l % 16, Hkv, hkv);
    lb::store_rows(make_rsrc(DK, kv_bytes), dkt, scale, row, l);
    lb::store_rows(make_rsrc(DV, kv_bytes), dvt, 1.0f, row, l);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b03b_dq(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DQ, long long q_bytes, long long kv_bytes, long long row_bytes, int B, int Sq, int Skv, int Hq, int Hkv,
    int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int qb = (int) __builtin_amdgcn_workgroup_id_x(), hq = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int hkv = hq / (Hq / Hkv), p0 = 64 * qb + 16 * w, cshift = Skv - Sq;
    const int q_stride = Hq * lf::D * 2, kv_stride = Hkv * lf::D * 2;

    const uint32_t lds0 = lds_addr(smem);
    auto k_lds = [&](int j) { return lds0 + (uint32_t) ((j & 1) * BUF_B); };
    auto v_lds = [&](int j) { return lds0 + (uint32_t) ((j & 1) * BUF_B + TILE); };
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, kv_bytes), rv = make_rsrc(V, kv_bytes);
    const rsrc_t   rdo = make_rsrc(DO, q_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);

    // Q^T and dO^T as B operands (B[k = d][n = q], lane = q): resident; NL / delta of the lane's query
    opnd_t qt_op[lf::NDK], dot_op[lf::NDK];
    lf::load_qt(qt_op, rq, lf::row_b(b, Sq, p0, Hq, hq), q_stride, l);
    lf::load_qt(dot_op, rdo, lf::row_b(b, Sq, p0, Hq, hq), q_stride, l);
    const int   vrow = lf::lse_b(b, Hq, hq, Sq, p0 + l % 16);
    const float nl = buf_load_f32(rnl, vrow, 0), del = buf_load_f32(rdel, vrow, 0);
    acc_t dqt[lf::NDT]; // dQ^T: rows d, lane = q
    static_for<lf::NDT>([&](auto dt) { dqt[dt] = zero_acc(); });

    const int n = lf::n_kv_blocks(qb, Sq, Skv, causal);
    auto issue = [&](int j) { issue_half(k_lds(j), K, V, b, Skv, 64 * j, Hkv, hkv, w); }; // K: waves 0, 1; V: 2, 3
    issue(0);
    split_signal_tdm_lds<0>();
    split_wait_lds();
    for (int j = 0; j < n; ++j) {
        const bool next = j + 1 < n; // workgroup-uniform
        if (next) issue(j + 1);
        const lds_u8 *k_rows = rows_base<P_B>(k_lds(j), l), *v_rows = rows_base<P_B>(v_lds(j), l);
        const lds_u8 *k_tr = tr_base<P_B>(k_lds(j), l);

        // S^T = K Q^T, dP^T = V dO^T: 4 kv tiles of 16 rows, lane = q; kv tile kt + 1's burst flies during tile kt
        opnd_t xa[2][2][lf::NDK]; // [set][K, V][k]
        auto   xload = [&](auto kt) {
            static_for<lf::NDK>([&](auto k) {
                xa[kt & 1][0][k] = load_rows<P_B>(k_rows, 16 * kt, 32 * k);
                xa[kt & 1][1][k] = load_rows<P_B>(v_rows, 16 * kt, 32 * k);
            });
        };
        acc_t s[4], dp[4];
        xload(ic<0>{});
        static_for<4>([&](auto kt) {
            if constexpr (kt + 1 < 4) xload(ic<decltype(kt)::value + 1>{});
            sched_barrier<0>();
            static_for<lf::NDK>([&](auto k) {
                if constexpr (decltype(k)::value == 0) {
                    s[kt]  = wmma0(xa[kt & 1][0][0], qt_op[0]);
                    dp[kt] = wmma0(xa[kt & 1][1][0], dot_op[0]);
                } else {
                    s[kt]  = wmma(xa[kt & 1][0][k], qt_op[k], s[kt]);
                    dp[kt] = wmma(xa[kt & 1][1][k], dot_op[k], dp[kt]);
                }
            });
            sched_barrier<0>();
        });
        // the K^T operands of dQ (tr16 of the K tile): one burst, flies during the softmax
        opnd_t ta[2][lf::NDT];
        static_for<2>([&](auto c) {
            static_for<lf::NDT>([&](auto dt) { ta[c][dt] = load_tr<P_B>(k_tr, 32 * c, 16 * dt); });
        });
        sched_barrier<0>();
        static_for<4>([&](auto kt) {
            s[kt]  = lb::probs_kq(s[kt], c2, nl, l, causal != 0, p0 + cshift - (64 * j + 16 * kt));
            dp[kt] = lb::dsoft(s[kt], dp[kt], del);
        });
        // dQ^T += K^T dS^T (32-row kv chunks)
        static_for<2>([&](auto c) {
            const opnd_t sb = pack_k32(dp[2 * c], dp[2 * c + 1]);
            static_for<lf::NDT>([&](auto dt) { dqt[dt] = wmma(ta[c][dt], sb, dqt[dt]); });
        });
        if (next) { // publish block j + 1 (its buffer was last read in block j - 1)
            split_signal_tdm_lds<0>();
            split_wait_lds();
        }
    }
    lb::store_rows(make_rsrc(DQ, q_bytes), dqt, scale, lf::row_b(b, Sq, p0 + l % 16, Hq, hq), l);
}
