/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// b00_naive -- rung 00 of the Llama backward ladder: the naive baseline (no parent). FlashAttention-2's
// backward in three kernels, each written the simplest way that is correct.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O), NL = -LSE log2 e                          grid (Sq/64, Hq, B), block 256
//   dkdv  workgroup = 64 kv rows of one (b, kv head); wave w = kv rows [16 w, +16), K / V resident (B operands)
//         loop: every visible 64-query block x every q head of the kv head's group:
//           fill Q and dO tiles (global -> VGPR -> LDS, pitch 272, ONE buffer, 2 full barriers),
//           S = Q K^T, dP = dO V^T (lane = kv), P, dS, dV^T += dO^T P, dK^T += Q^T dS (operands by tr16)
//         dK = scale dK^T, dV stored once                                  grid (Skv/64, Hkv, B), block 128
//   dq    workgroup = 64 queries of one q head; wave w = queries [16 w, +16), Q^T / dO^T resident
//         loop: every visible 64-row kv block: fill K and V tiles (register path, one buffer, 2 barriers),
//           S^T = K Q^T, dP^T = V dO^T (lane = q), P^T, dS^T, dQ^T += K^T dS^T
//         dQ = scale dQ^T                                                   grid (Sq/64, Hq, B), block 128
// S and dP are computed twice (once per kernel): 7 GEMMs of 128-wide operands per (q, kv) pair instead of 5.
#include "ladder/lb.h"

using namespace hk_attn;

namespace {

constexpr int P_B   = 2 * lf::D + 16;  // 272: every LDS tile here, rows and tr16 reads
constexpr int TILE  = 64 * P_B;         // 17408
constexpr int LDS_B = 2 * TILE;         // two tiles: (Q, dO) in dkdv, (K, V) in dq

// global -> VGPR -> LDS of one 64 x 128 bf16 tile (thread t: 16-B chunk t % 16 of rows t / 16 + 8 j)
HKA_DEV void fill_tile(lds_u8 *dst, rsrc_t r, int row0_b, int stride_b, int tid) {
    const int c = tid % 16, r0 = tid / 16;
    u32x4     v[8];
    static_for<8>([&](auto j) { v[j] = buf_load_b128(r, row0_b + (r0 + 8 * j) * stride_b + 16 * c, 0); });
    static_for<8>([&](auto j) { ds_store_b128(dst, (r0 + 8 * j) * P_B + 16 * c, v[j]); });
}

} // namespace

extern "C" __global__ __launch_bounds__(256) void k_bwd_b00_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int Hq) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, Hq);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b00_dkdv(
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
    lds_u8 *const  q_lds = lds_ptr(lds0), *const do_lds = lds_ptr(lds0 + TILE);
    const lds_u8  *q_rows = rows_base<P_B>(lds0, l), *do_rows = rows_base<P_B>(lds0 + TILE, l);
    const lds_u8  *q_tr = tr_base<P_B>(lds0, l), *do_tr = tr_base<P_B>(lds0 + TILE, l);
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, kv_bytes), rv = make_rsrc(V, kv_bytes);
    const rsrc_t   rdo = make_rsrc(DO, q_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);

    // K / V rows as B^T operands of S = Q K^T and dP = dO V^T (B[k = d][n = kv], lane = kv): resident
    opnd_t kb_op[lf::NDK], vb_op[lf::NDK];
    lf::load_qt(kb_op, rk, lf::row_b(b, Skv, kv0, Hkv, hkv), kv_stride, l);
    lf::load_qt(vb_op, rv, lf::row_b(b, Skv, kv0, Hkv, hkv), kv_stride, l);
    acc_t dkt[lf::NDT], dvt[lf::NDT]; // dK^T, dV^T: rows d, lane = kv
    static_for<lf::NDT>([&](auto dt) { dkt[dt] = zero_acc(); dvt[dt] = zero_acc(); });

    const int nqb = Sq / 64;
    for (int qb = lb::first_q_block(kb, Sq, Skv, causal); qb < nqb; ++qb) {
        for (int g = 0; g < G; ++g) {
            const int hq = hkv * G + g;
            lds_barrier(); // every wave done with the previous (Q, dO)
            fill_tile(q_lds, rq, lf::row_b(b, Sq, 64 * qb, Hq, hq), q_stride, tid);
            fill_tile(do_lds, rdo, lf::row_b(b, Sq, 64 * qb, Hq, hq), q_stride, tid);
            lds_barrier();

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
        }
    }
    const int row = lf::row_b(b, Skv, kv0 + l % 16, Hkv, hkv);
    lb::store_rows(make_rsrc(DK, kv_bytes), dkt, scale, row, l);
    lb::store_rows(make_rsrc(DV, kv_bytes), dvt, 1.0f, row, l);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b00_dq(
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
    lds_u8 *const  k_lds = lds_ptr(lds0), *const v_lds = lds_ptr(lds0 + TILE);
    const lds_u8  *k_rows = rows_base<P_B>(lds0, l), *v_rows = rows_base<P_B>(lds0 + TILE, l);
    const lds_u8  *k_tr = tr_base<P_B>(lds0, l);
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
    for (int j = 0; j < n; ++j) {
        lds_barrier();
        fill_tile(k_lds, rk, lf::row_b(b, Skv, 64 * j, Hkv, hkv), kv_stride, tid);
        fill_tile(v_lds, rv, lf::row_b(b, Skv, 64 * j, Hkv, hkv), kv_stride, tid);
        lds_barrier();

        // S^T = K Q^T, dP^T = V dO^T: 4 kv tiles of 16 rows, lane = q
        acc_t s[4], dp[4];
        static_for<4>([&](auto kt) {
            static_for<lf::NDK>([&](auto k) {
                const opnd_t ka = load_rows<P_B>(k_rows, 16 * kt, 32 * k);
                const opnd_t va = load_rows<P_B>(v_rows, 16 * kt, 32 * k);
                if constexpr (decltype(k)::value == 0) {
                    s[kt]  = wmma0(ka, qt_op[0]);
                    dp[kt] = wmma0(va, dot_op[0]);
                } else {
                    s[kt]  = wmma(ka, qt_op[k], s[kt]);
                    dp[kt] = wmma(va, dot_op[k], dp[kt]);
                }
            });
        });
        static_for<4>([&](auto kt) {
            s[kt]  = lb::probs_kq(s[kt], c2, nl, l, causal != 0, p0 + cshift - (64 * j + 16 * kt));
            dp[kt] = lb::dsoft(s[kt], dp[kt], del);
        });
        // dQ^T += K^T dS^T (32-row kv chunks; K^T operands by tr16 of the K tile)
        static_for<2>([&](auto c) {
            const opnd_t sb = pack_k32(dp[2 * c], dp[2 * c + 1]);
            static_for<lf::NDT>([&](auto dt) { dqt[dt] = wmma(load_tr<P_B>(k_tr, 32 * c, 16 * dt), sb, dqt[dt]); });
        });
    }
    lb::store_rows(make_rsrc(DQ, q_bytes), dqt, scale, lf::row_b(b, Sq, p0 + l % 16, Hq, hq), l);
}
