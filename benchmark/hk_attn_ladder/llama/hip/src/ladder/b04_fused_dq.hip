/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// b04_fused_dq -- Llama backward ladder, side chain (first draft): b03 with dQ computed inside the dK / dV kernel.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O), NL = -LSE log2 e                          grid (Sq/64, Hq, B), block 256
//   dkdv  workgroup = 64 kv rows of one (b, kv head); wave w = kv rows [16 w, +16), K / V resident (B operands)
//         and the WG's K tile also in LDS (pitch 288, for dQ); loop over every visible 64-query block x every
//         q head of the group:
//           Q / dO tiles by TDM (pitch 272, two buffers, next iteration's ops in flight),
//           S = Q K^T, dP = dO V^T (lane = kv), P, dS,
//           dS^T -> LDS (each wave its 16 kv rows x 64 queries), barrier,
//           dQ(16 queries of wave w) = dS K: A = dS by tr16 of the dS^T tile, B = K by tr16 of the K tile,
//           fp32 device-scope atomics into dQ_acc [B, Sq, Hq, 128] (zeroed by the host),
//           dV^T += dO^T P, dK^T += Q^T dS
//         dK = scale dK^T, dV stored once                                  grid (Skv/64, Hkv, B), block 128
//   cvt   dq = bf16(scale dQ_acc)                                          grid (B Sq Hq 128 / 2048), block 256
//   sync  per iteration: 2 full barriers (dS^T published; next tiles published)
// What changes from b03: the dQ kernel and its second computation of S and dP are gone: 5 GEMMs per (q, kv)
// pair instead of 7. The price: dS crosses waves through LDS (one more barrier per iteration) and dQ is
// summed in fp32 atomics over the Skv / 64 workgroups of a (b, kv head) (dQ is no longer bitwise
// reproducible; dK / dV are).
#include "ladder/lb.h"

using namespace hk_attn;

namespace {

constexpr int P_B    = 2 * lf::D + 16; // 272: Q / dO tiles (rows and tr16 reads)
constexpr int KT_P   = 2 * lf::D + 32; // 288: the K tile of dQ (tr16 reads)
constexpr int DS_P   = 2 * 64 + 32;    // 160: dS^T tile [64 kv][64 q] (C^T stores, tr16 reads)
constexpr int TILE   = 64 * P_B;       // 17408
constexpr int BUF_B  = 2 * TILE;       // one buffer: Q tile, then dO tile
constexpr int K_OFF  = 2 * BUF_B;      // 69632
constexpr int DS_OFF = K_OFF + 64 * KT_P; // 88064
constexpr int LDS_B  = DS_OFF + 64 * DS_P; // 98304

using T_HALF = tdm_tile<32, lf::D, P_B>;  // one Q / dO op: 32 rows x 128 bf16
using T_KHALF = tdm_tile<32, lf::D, KT_P>; // one K op

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

extern "C" __global__ __launch_bounds__(256) void k_bwd_b04_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int Hq) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, Hq);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b04_dkdv(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DK, __bf16 *DV, float *DQA, long long q_bytes, long long kv_bytes, long long row_bytes,
    long long dqa_bytes, int B, int Sq, int Skv, int Hq, int Hkv, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id();
    const int kb = (int) __builtin_amdgcn_workgroup_id_x(), hkv = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int G = Hq / Hkv, kv0 = 64 * kb + 16 * w, cshift = Skv - Sq;
    const int kv_stride = Hkv * lf::D * 2;

    const uint32_t lds0 = lds_addr(smem);
    auto q_lds  = [&](int it) { return lds0 + (uint32_t) ((it & 1) * BUF_B); }; // Q tile of iteration it
    auto do_lds = [&](int it) { return lds0 + (uint32_t) ((it & 1) * BUF_B + TILE); };
    lds_u8 *const ds_rows = rows_base<DS_P>(lds0 + DS_OFF, l);                     // C^T stores of dS
    const lds_u8 *ds_tr   = tr_base<DS_P>(lds0 + DS_OFF + 2 * 16 * (uint32_t) w, l); // dS of this wave's queries
    const lds_u8 *k_tr    = tr_base<KT_P>(lds0 + K_OFF, l);
    const rsrc_t  rk = make_rsrc(K, kv_bytes), rv = make_rsrc(V, kv_bytes);
    const rsrc_t  rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes), rqa = make_rsrc(DQA, dqa_bytes);

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
    if (w < 2) // the K tile of dQ (waves 0, 1: rows [0, 32) / [32, 64)), once
        tdm_load(tdm_g0(lds0 + K_OFF + (uint32_t) (32 * w * KT_P),
                        tdm_gaddr(K, (int64_t) b * Skv + 64 * kb + 32 * w, (int64_t) Hkv * lf::D, (int64_t) hkv * lf::D)),
                 tdm_g1<T_KHALF>((int64_t) Hkv * lf::D, 32));
    if (nit > 0)
        issue(0);
    split_signal_tdm_lds<0>(); // own ops landed: K tile, iteration 0's Q / dO
    split_wait_lds();
    const int qa_row = Hq * lf::D * 4; // dQ_acc bytes per query position
    for (int it = 0; it < nit; ++it) {
        const int  qb = qb0 + it / G, hq = hkv * G + it % G;
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

        // ---- dS^T -> LDS (rows kv 16 w .., columns q), then dQ of this wave's 16 queries = dS K -------------
        static_for<4>([&](auto qt) { store_ct<DS_P>(ds_rows, 16 * w, 16 * qt, dp[qt]); });
        lds_barrier(); // every wave's dS^T in LDS
        {
            acc_t dq[lf::NDT]; // dQ: rows q (8 h + i), lane = d (16 dt + r)
            static_for<2>([&](auto c) {
                const opnd_t a = load_tr<DS_P>(ds_tr, 32 * c, 0); // A = dS [q][kv]
                static_for<lf::NDT>([&](auto dt) {
                    const opnd_t bk = load_tr<KT_P>(k_tr, 32 * c, 16 * dt); // B = K [kv][d]
                    if constexpr (decltype(c)::value == 0) dq[dt] = wmma0(a, bk);
                    else dq[dt] = wmma(a, bk, dq[dt]);
                });
            });
            // element i of lane (r, h): query 64 qb + 16 w + 8 h + i, d = 16 dt + r
            const int vo = lf::row_b(b, Sq, 64 * qb + 16 * w + 8 * (l / 16), Hq, hq) * 2 + 4 * (l % 16);
            static_for<lf::NDT>([&](auto dt) {
                static_for<8>([&](auto i) {
                    buf_atomic_add_f32(rqa, vo + 64 * dt + (int) i * qa_row, 0, dq[dt][(int) i]);
                });
            });
        }

        // dV^T += dO^T P, dK^T += Q^T dS (32-query chunks)
        static_for<2>([&](auto c) {
            const opnd_t pb = pack_p_bwd(s[2 * c], s[2 * c + 1]);
            const opnd_t sb = pack_p_bwd(dp[2 * c], dp[2 * c + 1]);
            static_for<lf::NDT>([&](auto dt) {
                dvt[dt] = wmma(load_tr<P_B>(do_tr, 32 * c, 16 * dt), pb, dvt[dt]);
                dkt[dt] = wmma(load_tr<P_B>(q_tr, 32 * c, 16 * dt), sb, dkt[dt]);
            });
        });
        // publish the next iteration's tiles; every wave's dS^T reads are done before its signal (DS drained)
        if (next) {
            split_signal_tdm_lds<0>();
            split_wait_lds();
        }
    }
    const int row = lf::row_b(b, Skv, kv0 + l % 16, Hkv, hkv);
    lb::store_rows(make_rsrc(DK, kv_bytes), dkt, scale, row, l);
    lb::store_rows(make_rsrc(DV, kv_bytes), dvt, 1.0f, row, l);
}

// dq = bf16(scale dQ_acc): thread t converts the 8 fp32 at byte 32 t (out of range: loads 0, stores dropped)
extern "C" __global__ __launch_bounds__(256) void k_bwd_b04_cvt(const float *DQA, __bf16 *DQ, long long dqa_bytes,
                                                                long long q_bytes, float scale) {
    const int    t  = (int) (__builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x());
    const rsrc_t ra = make_rsrc(DQA, dqa_bytes), rq = make_rsrc(DQ, q_bytes);
    const f32x4  a = buf_load_f32x4(ra, 32 * t, 0), c = buf_load_f32x4(ra, 32 * t + 16, 0);
    const acc_t  v = __builtin_shufflevector(a, c, 0, 1, 2, 3, 4, 5, 6, 7);
    buf_store_b128(rq, 16 * t, 0, cvt_pk_bf16(v * scale));
}
