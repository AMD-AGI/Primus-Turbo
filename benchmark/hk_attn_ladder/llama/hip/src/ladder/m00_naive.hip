/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// m00_naive -- rung 00 of the DeepSeek-V3 MLA backward ladder: the naive baseline (no parent). The Llama
// ladder's b00 (FlashAttention-2's backward in three plain kernels) with the MLA head dims: qk 192, v 128, MHA.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O) (128 wide), NL = -LSE log2 e              grid (Sq/64, H, B), block 256
//   dkdv  workgroup = 64 kv rows of one (b, head); wave w = kv rows [16 w, +16); K (6 k-steps) / V (4) resident
//         loop: every visible 64-query block: fill Q (64 x 192, pitch 400) and dO (64 x 128, pitch 272) by the
//         register path into ONE buffer (2 full barriers); S = Q K^T, dP = dO V^T (lane = kv), P, dS,
//         dV^T += dO^T P (8 d tiles), dK^T += Q^T dS (12 d tiles)
//         dK = scale dK^T, dV stored once                                   grid (Skv/64, H, B), block 128
//   dq    workgroup = 64 queries of one head; Q^T (6 k-steps) / dO^T (4) resident
//         loop: every visible 64-row kv block: fill K (pitch 400) and V (pitch 272), one buffer, 2 barriers;
//           S^T = K Q^T, dP^T = V dO^T (lane = q), P^T, dS^T, dQ^T += K^T dS^T (12 d tiles)
//         dQ = scale dQ^T                                                    grid (Sq/64, H, B), block 128
// S and dP are computed in both kernels: FlyDSL's MLA backward does the same (k_dkdv + k_dqg).
#include "ladder/lm.h"

using namespace hk_attn;

namespace {

constexpr int QT_B  = 64 * lm::PQ;  // 25600: a 64 x 192 tile (Q in dkdv, K in dq)
constexpr int VT_B  = 64 * lm::PV;  // 17408: a 64 x 128 tile (dO in dkdv, V in dq)
constexpr int LDS_B = QT_B + VT_B;  // 43008

} // namespace

extern "C" __global__ __launch_bounds__(256) void k_mla_m00_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int H) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, H);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_mla_m00_dkdv(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DK, __bf16 *DV, long long q_bytes, long long k_bytes, long long v_bytes, long long o_bytes,
    long long row_bytes, int B, int Sq, int Skv, int H, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int kb = (int) __builtin_amdgcn_workgroup_id_x(), h = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int kv0 = 64 * kb + 16 * w, cshift = Skv - Sq;

    const uint32_t lds0 = lds_addr(smem);
    lds_u8 *const  q_lds = lds_ptr(lds0), *const do_lds = lds_ptr(lds0 + QT_B);
    const lds_u8  *q_rows = rows_base<lm::PQ>(lds0, l), *do_rows = rows_base<lm::PV>(lds0 + QT_B, l);
    const lds_u8  *q_tr = tr_base<lm::PQ>(lds0, l), *do_tr = tr_base<lm::PV>(lds0 + QT_B, l);
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, k_bytes), rv = make_rsrc(V, v_bytes);
    const rsrc_t   rdo = make_rsrc(DO, o_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);

    // K / V rows as B^T operands of S = Q K^T (6 k-steps) and dP = dO V^T (4): resident, lane = kv
    opnd_t kb_op[lm::NKQ], vb_op[lm::NKV];
    lm::load_bt(kb_op, rk, lm::row<lm::DQK>(b, Skv, kv0, H, h), H * lm::DQK * 2, l);
    lm::load_bt(vb_op, rv, lm::row<lm::DV>(b, Skv, kv0, H, h), H * lm::DV * 2, l);
    acc_t dkt[lm::NTQ], dvt[lm::NTV]; // dK^T, dV^T: rows d, lane = kv
    static_for<lm::NTQ>([&](auto dt) { dkt[dt] = zero_acc(); });
    static_for<lm::NTV>([&](auto dt) { dvt[dt] = zero_acc(); });

    for (int qb = lb::first_q_block(kb, Sq, Skv, causal); qb < Sq / 64; ++qb) {
        lds_barrier(); // every wave done with the previous (Q, dO)
        lm::fill_tile<lm::DQK, lm::PQ>(q_lds, rq, lm::row<lm::DQK>(b, Sq, 64 * qb, H, h), H * lm::DQK * 2, tid);
        lm::fill_tile<lm::DV, lm::PV>(do_lds, rdo, lm::row<lm::DV>(b, Sq, 64 * qb, H, h), H * lm::DV * 2, tid);
        lds_barrier();

        // S = Q K^T (6 k-steps), dP = dO V^T (4): 4 q tiles of 16 rows, lane = kv
        acc_t s[4], dp[4];
        static_for<4>([&](auto qt) {
            static_for<lm::NKQ>([&](auto k) {
                const opnd_t qa = load_rows<lm::PQ>(q_rows, 16 * qt, 32 * k);
                if constexpr (decltype(k)::value == 0) s[qt] = wmma0(qa, kb_op[0]);
                else s[qt] = wmma(qa, kb_op[k], s[qt]);
            });
            static_for<lm::NKV>([&](auto k) {
                const opnd_t da = load_rows<lm::PV>(do_rows, 16 * qt, 32 * k);
                if constexpr (decltype(k)::value == 0) dp[qt] = wmma0(da, vb_op[0]);
                else dp[qt] = wmma(da, vb_op[k], dp[qt]);
            });
        });
        static_for<4>([&](auto qt) {
            const int   q0  = 64 * qb + 16 * qt;
            const acc_t nl  = lb::rows8(rnl, b, H, h, Sq, q0, l);
            const acc_t del = lb::rows8(rdel, b, H, h, Sq, q0, l);
            s[qt]  = lb::probs_qk(s[qt], c2, nl, l, causal != 0, q0 + cshift - kv0);
            dp[qt] = lb::dsoft(s[qt], dp[qt], del);
        });
        // dV^T += dO^T P (8 d tiles), dK^T += Q^T dS (12)
        static_for<2>([&](auto c) {
            const opnd_t pb = pack_p_bwd(s[2 * c], s[2 * c + 1]);
            const opnd_t sb = pack_p_bwd(dp[2 * c], dp[2 * c + 1]);
            static_for<lm::NTV>([&](auto dt) { dvt[dt] = wmma(load_tr<lm::PV>(do_tr, 32 * c, 16 * dt), pb, dvt[dt]); });
            static_for<lm::NTQ>([&](auto dt) { dkt[dt] = wmma(load_tr<lm::PQ>(q_tr, 32 * c, 16 * dt), sb, dkt[dt]); });
        });
    }
    lm::store_rows(make_rsrc(DK, k_bytes), dkt, scale, lm::row<lm::DQK>(b, Skv, kv0 + l % 16, H, h), l);
    lm::store_rows(make_rsrc(DV, v_bytes), dvt, 1.0f, lm::row<lm::DV>(b, Skv, kv0 + l % 16, H, h), l);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_mla_m00_dq(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DQ, long long q_bytes, long long k_bytes, long long v_bytes, long long o_bytes, long long row_bytes,
    int B, int Sq, int Skv, int H, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int qb = (int) __builtin_amdgcn_workgroup_id_x(), h = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int p0 = 64 * qb + 16 * w, cshift = Skv - Sq;

    const uint32_t lds0 = lds_addr(smem);
    lds_u8 *const  k_lds = lds_ptr(lds0), *const v_lds = lds_ptr(lds0 + QT_B);
    const lds_u8  *k_rows = rows_base<lm::PQ>(lds0, l), *v_rows = rows_base<lm::PV>(lds0 + QT_B, l);
    const lds_u8  *k_tr = tr_base<lm::PQ>(lds0, l);
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, k_bytes), rv = make_rsrc(V, v_bytes);
    const rsrc_t   rdo = make_rsrc(DO, o_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);

    // Q^T (6 k-steps) and dO^T (4) as B operands (lane = q): resident; NL / delta of the lane's query
    opnd_t qt_op[lm::NKQ], dot_op[lm::NKV];
    lm::load_bt(qt_op, rq, lm::row<lm::DQK>(b, Sq, p0, H, h), H * lm::DQK * 2, l);
    lm::load_bt(dot_op, rdo, lm::row<lm::DV>(b, Sq, p0, H, h), H * lm::DV * 2, l);
    const int   vrow = lf::lse_b(b, H, h, Sq, p0 + l % 16);
    const float nl = buf_load_f32(rnl, vrow, 0), del = buf_load_f32(rdel, vrow, 0);
    acc_t dqt[lm::NTQ]; // dQ^T: rows d, lane = q
    static_for<lm::NTQ>([&](auto dt) { dqt[dt] = zero_acc(); });

    const int n = lf::n_kv_blocks(qb, Sq, Skv, causal);
    for (int j = 0; j < n; ++j) {
        lds_barrier();
        lm::fill_tile<lm::DQK, lm::PQ>(k_lds, rk, lm::row<lm::DQK>(b, Skv, 64 * j, H, h), H * lm::DQK * 2, tid);
        lm::fill_tile<lm::DV, lm::PV>(v_lds, rv, lm::row<lm::DV>(b, Skv, 64 * j, H, h), H * lm::DV * 2, tid);
        lds_barrier();

        // S^T = K Q^T (6 k-steps), dP^T = V dO^T (4): 4 kv tiles of 16 rows, lane = q
        acc_t s[4], dp[4];
        static_for<4>([&](auto kt) {
            static_for<lm::NKQ>([&](auto k) {
                const opnd_t ka = load_rows<lm::PQ>(k_rows, 16 * kt, 32 * k);
                if constexpr (decltype(k)::value == 0) s[kt] = wmma0(ka, qt_op[0]);
                else s[kt] = wmma(ka, qt_op[k], s[kt]);
            });
            static_for<lm::NKV>([&](auto k) {
                const opnd_t va = load_rows<lm::PV>(v_rows, 16 * kt, 32 * k);
                if constexpr (decltype(k)::value == 0) dp[kt] = wmma0(va, dot_op[0]);
                else dp[kt] = wmma(va, dot_op[k], dp[kt]);
            });
        });
        static_for<4>([&](auto kt) {
            s[kt]  = lb::probs_kq(s[kt], c2, nl, l, causal != 0, p0 + cshift - (64 * j + 16 * kt));
            dp[kt] = lb::dsoft(s[kt], dp[kt], del);
        });
        // dQ^T += K^T dS^T (12 d tiles; K^T operands by tr16 of the K tile)
        static_for<2>([&](auto c) {
            const opnd_t sb = pack_k32(dp[2 * c], dp[2 * c + 1]);
            static_for<lm::NTQ>([&](auto dt) { dqt[dt] = wmma(load_tr<lm::PQ>(k_tr, 32 * c, 16 * dt), sb, dqt[dt]); });
        });
    }
    lm::store_rows(make_rsrc(DQ, q_bytes), dqt, scale, lm::row<lm::DQK>(b, Sq, p0 + l % 16, H, h), l);
}
