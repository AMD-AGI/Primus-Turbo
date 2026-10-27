/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// b03_gqa4 -- rung 03 of the Llama backward ladder: b02 with four q heads per dQ workgroup.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O), NL = -LSE log2 e                          grid (Sq/64, Hq, B), block 256
//   dkdv  workgroup = 64 kv rows of one (b, kv head); wave w = kv rows [16 w, +16), K / V resident (B operands)
//         loop: every visible 64-query block x every q head of the kv head's group:
//           fill Q and dO tiles (TDM, pitch 272, TWO buffers: waves 0 / 1 move Q rows [0, 32) / [32, 64),
//           waves 2 / 3 the same of dO; the next (q block, head)'s ops in flight during this one's compute,
//           s_wait_tensorcnt 0 + 1 full barrier after it),
//           S = Q K^T, dP = dO V^T (lane = kv), P, dS, dV^T += dO^T P, dK^T += Q^T dS (operands by tr16)
//         dK = scale dK^T, dV stored once                                  grid (Skv/64, Hkv, B), block 128
//   dq    workgroup = 64 queries of FOUR q heads (one kv head, G % 4 == 0); wave w = queries [16 w, +16) of
//         the 4 heads, Q^T / dO^T of 4 heads resident; per kv block the 4 heads run one after the other
//         loop: every visible 64-row kv block: fill K and V tiles (TDM, two buffers, waves 0 / 1: K halves,
//           2 / 3: V halves; block j + 1's ops in flight during block j, 1 barrier),
//           S^T = K Q^T, dP^T = V dO^T (lane = q), P^T, dS^T, dQ^T += K^T dS^T
//         dQ = scale dQ^T                                                   grid (Sq/64, Hq/4, B), block 128
// S and dP are computed twice (once per kernel): 7 GEMMs of 128-wide operands per (q, kv) pair instead of 5.
// What changes from b02: the dQ kernel's K / V tiles feed four q heads (the forward ladder's f03): a quarter
// of the K / V bytes per WMMA. (dkdv already streams every head of its kv head over resident K / V.)
#include "ladder/lb.h"

using namespace hk_attn;

namespace {

constexpr int P_B   = 2 * lf::D + 16;  // 272: every LDS tile here, rows and tr16 reads
constexpr int TILE  = 64 * P_B;         // 17408
constexpr int BUF_B = 2 * TILE;         // one buffer: two tiles, (Q, dO) in dkdv, (K, V) in dq
constexpr int LDS_B = 2 * BUF_B;        // two buffers

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

extern "C" __global__ __launch_bounds__(256) void k_bwd_b03_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int Hq) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, Hq);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b03_dkdv(
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

            // S = Q K^T, dP = dO V^T: 4 q tiles of 16 rows, lane = kv
            acc_t s[4], dp[4];
            static_for<4>([&](auto qt) {
                static_for<lf::NDK>([&](auto k) {
                    const opnd_t qa = load_rows<P_B>(q_rows, 16 * qt, 32 * k);
                    const opnd_t da = load_rows<P_B>(do_rows, 16 * qt, 32 * k);
                    if constexpr (decltype(k)::value == 0) {
                        s[qt]  = wmma0(qa, kb_op[0]);
                        dp[qt] = wmma0(da, vb_op[0]);
                    } else {
                        s[qt]  = wmma(qa, kb_op[k], s[qt]);
                        dp[qt] = wmma(da, vb_op[k], dp[qt]);
                    }
                });
            });
            // P and dS (row vectors NL / delta in accumulator-row order)
            static_for<4>([&](auto qt) {
                const int   q0  = 64 * qb + 16 * qt;
                const acc_t nl  = lb::rows8(rnl, b, Hq, hq, Sq, q0, l);
                const acc_t del = lb::rows8(rdel, b, Hq, hq, Sq, q0, l);
                s[qt]  = lb::probs_qk(s[qt], c2, nl, l, causal != 0, q0 + cshift - kv0);
                dp[qt] = lb::dsoft(s[qt], dp[qt], del);
            });
            // dV^T += dO^T P, dK^T += Q^T dS (32-query chunks)
            static_for<2>([&](auto c) {
                const opnd_t pb = pack_p_bwd(s[2 * c], s[2 * c + 1]);
                const opnd_t sb = pack_p_bwd(dp[2 * c], dp[2 * c + 1]);
                static_for<lf::NDT>([&](auto dt) {
                    dvt[dt] = wmma(load_tr<P_B>(do_tr, 32 * c, 16 * dt), pb, dvt[dt]);
                    dkt[dt] = wmma(load_tr<P_B>(q_tr, 32 * c, 16 * dt), sb, dkt[dt]);
                });
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

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b03_dq(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DQ, long long q_bytes, long long kv_bytes, long long row_bytes, int B, int Sq, int Skv, int Hq, int Hkv,
    int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    constexpr int NH = 4;
    const int qb = (int) __builtin_amdgcn_workgroup_id_x(), hq4 = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int hkv = NH * hq4 / (Hq / Hkv), p0 = 64 * qb + 16 * w, cshift = Skv - Sq;
    const int q_stride = Hq * lf::D * 2, kv_stride = Hkv * lf::D * 2;

    const uint32_t lds0 = lds_addr(smem);
    auto k_lds = [&](int j) { return lds0 + (uint32_t) ((j & 1) * BUF_B); };
    auto v_lds = [&](int j) { return lds0 + (uint32_t) ((j & 1) * BUF_B + TILE); };
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, kv_bytes), rv = make_rsrc(V, kv_bytes);
    const rsrc_t   rdo = make_rsrc(DO, q_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);

    // Q^T and dO^T as B operands (B[k = d][n = q], lane = q): resident; NL / delta of the lane's query
    opnd_t qt_op[NH][lf::NDK], dot_op[NH][lf::NDK];
    float  nl[NH], del[NH];
    acc_t  dqt[NH][lf::NDT]; // dQ^T: rows d, lane = q
    static_for<NH>([&](auto h) {
        const int hq = NH * hq4 + h;
        lf::load_qt(qt_op[h], rq, lf::row_b(b, Sq, p0, Hq, hq), q_stride, l);
        lf::load_qt(dot_op[h], rdo, lf::row_b(b, Sq, p0, Hq, hq), q_stride, l);
        const int vrow = lf::lse_b(b, Hq, hq, Sq, p0 + l % 16);
        nl[h]  = buf_load_f32(rnl, vrow, 0);
        del[h] = buf_load_f32(rdel, vrow, 0);
        static_for<lf::NDT>([&](auto dt) { dqt[h][dt] = zero_acc(); });
    });

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

        static_for<NH>([&](auto h) { // the 4 heads in turn, on the same K / V tiles
            // S^T = K Q^T, dP^T = V dO^T: 4 kv tiles of 16 rows, lane = q
            acc_t s[4], dp[4];
            static_for<4>([&](auto kt) {
                static_for<lf::NDK>([&](auto k) {
                    const opnd_t ka = load_rows<P_B>(k_rows, 16 * kt, 32 * k);
                    const opnd_t va = load_rows<P_B>(v_rows, 16 * kt, 32 * k);
                    if constexpr (decltype(k)::value == 0) {
                        s[kt]  = wmma0(ka, qt_op[h][0]);
                        dp[kt] = wmma0(va, dot_op[h][0]);
                    } else {
                        s[kt]  = wmma(ka, qt_op[h][k], s[kt]);
                        dp[kt] = wmma(va, dot_op[h][k], dp[kt]);
                    }
                });
            });
            static_for<4>([&](auto kt) {
                s[kt]  = lb::probs_kq(s[kt], c2, nl[h], l, causal != 0, p0 + cshift - (64 * j + 16 * kt));
                dp[kt] = lb::dsoft(s[kt], dp[kt], del[h]);
            });
            // dQ^T += K^T dS^T (32-row kv chunks; K^T operands by tr16 of the K tile)
            static_for<2>([&](auto c) {
                const opnd_t sb = pack_k32(dp[2 * c], dp[2 * c + 1]);
                static_for<lf::NDT>([&](auto dt) {
                    dqt[h][dt] = wmma(load_tr<P_B>(k_tr, 32 * c, 16 * dt), sb, dqt[h][dt]);
                });
            });
        });
        if (next) { // publish block j + 1 (its buffer was last read in block j - 1)
            split_signal_tdm_lds<0>();
            split_wait_lds();
        }
    }
    static_for<NH>([&](auto h) {
        lb::store_rows(make_rsrc(DQ, q_bytes), dqt[h], scale, lf::row_b(b, Sq, p0 + l % 16, Hq, NH * hq4 + h), l);
    });
}
