/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// b06_deferred_dq -- rung 06 of the Llama backward ladder: b05 with dQ computed one iteration late.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O), NL = -LSE log2 e                          grid (Sq/64, Hq, B), block 256
//   dkdv  workgroup = 128 kv rows of one (b, kv head); wave w = kv rows [32 w, +32) = 2 kv tiles, K / V resident
//         (B operands) and the WG's K tile also in LDS (pitch 288, for dQ); loop over every visible 64-query
//         block x every q head of the group:
//           Q / dO tiles by TDM (pitch 272, two buffers, next iteration's ops in flight),
//           dQ of the PREVIOUS iteration (16 queries of wave w) = dS K over 128 kv from its dS^T tile (A by tr16,
//           B = K by tr16), fp32 device-scope atomics into dQ_acc [B, Sq, Hq, 128] (zeroed by the host),
//           S = Q K^T, dP = dO V^T, P, dS of this iteration; dS^T -> LDS buffer it & 1 (no barrier here),
//           dV^T += dO^T P, dK^T += Q^T dS
//         after the loop: dQ of the last iteration
//         dK = scale dK^T, dV stored once                                  grid (Skv/128, Hkv, B), block 128
//   cvt   dq = bf16(scale dQ_acc)                                          grid (B Sq Hq 128 / 2048), block 256
//   sync  per iteration: 1 full barrier (publishes the next Q / dO tiles AND this iteration's dS^T)
// What changes from b05: the barrier between storing dS^T and reading it back for dQ goes. The dS^T tile is
// double-buffered by iteration parity and read one iteration later, after the barrier that publishes the
// next Q / dO tiles anyway (v0's "dQ(s-1) in step s"): one barrier per iteration instead of two, and the
// dQ WMMAs + atomics no longer wait for the slowest wave's dS. Buffer it & 1 was last read (dQ of
// iteration it - 2) before every wave's signal at the end of iteration it - 1.
#include "ladder/lb.h"

using namespace hk_attn;

namespace {

constexpr int P_B    = 2 * lf::D + 16; // 272: Q / dO tiles (rows and tr16 reads)
constexpr int KT_P   = 2 * lf::D + 32; // 288: the K tile of dQ (tr16 reads)
constexpr int DS_P   = 2 * 64 + 32;    // 160: dS^T tile [128 kv][64 q] (C^T stores, tr16 reads)
constexpr int TILE   = 64 * P_B;       // 17408
constexpr int BUF_B  = 2 * TILE;       // one buffer: Q tile, then dO tile
constexpr int KVB    = 128;            // kv rows per workgroup
constexpr int K_OFF  = 2 * BUF_B;      // 69632
constexpr int DS_OFF = K_OFF + KVB * KT_P; // 106496
constexpr int DS_B   = KVB * DS_P;     // one dS^T tile [128 kv][64 q]
constexpr int LDS_B  = DS_OFF + 2 * DS_B; // 147456: two dS^T tiles (iteration parity)

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

extern "C" __global__ __launch_bounds__(256) void k_bwd_b06_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int Hq) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, Hq);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b06_dkdv(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DK, __bf16 *DV, float *DQA, long long q_bytes, long long kv_bytes, long long row_bytes,
    long long dqa_bytes, int B, int Sq, int Skv, int Hq, int Hkv, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id();
    const int kb = (int) __builtin_amdgcn_workgroup_id_x(), hkv = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int G = Hq / Hkv, kv0 = KVB * kb + 32 * w, cshift = Skv - Sq; // kv tile t of the wave: kv0 + 16 t
    const int kv_stride = Hkv * lf::D * 2;

    const uint32_t lds0 = lds_addr(smem);
    auto q_lds  = [&](int it) { return lds0 + (uint32_t) ((it & 1) * BUF_B); }; // Q tile of iteration it
    auto do_lds = [&](int it) { return lds0 + (uint32_t) ((it & 1) * BUF_B + TILE); };
    // dS^T tile of iteration it: buffer it & 1 (C^T stores of dS; tr16 reads of this wave's 16 queries)
    auto ds_rows = [&](int it) { return rows_base<DS_P>(lds0 + DS_OFF + (uint32_t) ((it & 1) * DS_B), l); };
    auto ds_tr   = [&](int it) { return tr_base<DS_P>(lds0 + DS_OFF + (uint32_t) ((it & 1) * DS_B + 32 * w), l); };
    const lds_u8 *k_tr    = tr_base<KT_P>(lds0 + K_OFF, l);
    const rsrc_t  rk = make_rsrc(K, kv_bytes), rv = make_rsrc(V, kv_bytes);
    const rsrc_t  rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes), rqa = make_rsrc(DQA, dqa_bytes);

    // K / V rows as B^T operands of S = Q K^T and dP = dO V^T (B[k = d][n = kv], lane = kv): resident
    opnd_t kb_op[2][lf::NDK], vb_op[2][lf::NDK];
    acc_t  dkt[2][lf::NDT], dvt[2][lf::NDT]; // dK^T, dV^T of kv tile t: rows d, lane = kv
    static_for<2>([&](auto t) {
        lf::load_qt(kb_op[t], rk, lf::row_b(b, Skv, kv0 + 16 * t, Hkv, hkv), kv_stride, l);
        lf::load_qt(vb_op[t], rv, lf::row_b(b, Skv, kv0 + 16 * t, Hkv, hkv), kv_stride, l);
        static_for<lf::NDT>([&](auto dt) { dkt[t][dt] = zero_acc(); dvt[t][dt] = zero_acc(); });
    });

    // iterations: (q block, head) pairs, heads inner: it -> q block qb0 + it / G, head hkv G + it % G
    // first query block that sees kv row 128 kb: the one holding query 128 kb - cshift (b0 / lb first_q_block)
    const int qf = KVB * kb - cshift, qb0 = causal == 0 || qf <= 0 ? 0 : qf / 64, nit = (Sq / 64 - qb0) * G;
    auto      issue = [&](int it) { // Q (waves 0, 1) and dO (waves 2, 3) of iteration it
        issue_half(q_lds(it), Q, DO, b, Sq, 64 * (qb0 + it / G), Hq, hkv * G + it % G, w);
    };
    // the K tile of dQ, once: wave w moves rows [32 w, +32)
    tdm_load(tdm_g0(lds0 + K_OFF + (uint32_t) (32 * w * KT_P),
                    tdm_gaddr(K, (int64_t) b * Skv + KVB * kb + 32 * w, (int64_t) Hkv * lf::D, (int64_t) hkv * lf::D)),
             tdm_g1<T_KHALF>((int64_t) Hkv * lf::D, 32));
    if (nit > 0)
        issue(0);
    split_signal_tdm_lds<0>(); // own ops landed: K tile, iteration 0's Q / dO
    split_wait_lds();
    const int qa_row = Hq * lf::D * 4; // dQ_acc bytes per query position
    // dQ of iteration it (16 queries of wave w) = dS K over the 128 kv rows, from dS^T buffer it & 1
    auto dq_of = [&](int it) {
        const int     qb = qb0 + it / G, hq = hkv * G + it % G;
        const lds_u8 *dst = ds_tr(it);
        acc_t         dq[lf::NDT]; // dQ: rows q (8 h + i), lane = d (16 dt + r)
        static_for<KVB / 32>([&](auto c) {
            const opnd_t a = load_tr<DS_P>(dst, 32 * c, 0); // A = dS [q][kv]
            static_for<lf::NDT>([&](auto dt) {
                const opnd_t bk = load_tr<KT_P>(k_tr, 32 * c, 16 * dt); // B = K [kv][d]
                if constexpr (decltype(c)::value == 0) dq[dt] = wmma0(a, bk);
                else dq[dt] = wmma(a, bk, dq[dt]);
            });
        });
        // element i of lane (r, h): query 64 qb + 16 w + 8 h + i, d = 16 dt + r
        const int vo = lf::row_b(b, Sq, 64 * qb + 16 * w + 8 * (l / 16), Hq, hq) * 2 + 4 * (l % 16);
        static_for<lf::NDT>([&](auto dt) {
            static_for<8>([&](auto i) { buf_atomic_add_f32(rqa, vo + 64 * dt + (int) i * qa_row, 0, dq[dt][(int) i]); });
        });
    };
    for (int it = 0; it < nit; ++it) {
        const int  qb = qb0 + it / G, hq = hkv * G + it % G;
        const bool next = it + 1 < nit; // workgroup-uniform
        if (next) issue(it + 1);
        if (it > 0) dq_of(it - 1); // its dS^T was published by the previous iteration's barrier
        const lds_u8 *q_rows = rows_base<P_B>(q_lds(it), l), *do_rows = rows_base<P_B>(do_lds(it), l);
        const lds_u8 *q_tr = tr_base<P_B>(q_lds(it), l), *do_tr = tr_base<P_B>(do_lds(it), l);

        // S = Q K^T, dP = dO V^T: 4 q tiles of 16 rows x 2 kv tiles, lane = kv; each A operand feeds 2 WMMAs
        acc_t s[4][2], dp[4][2];
        static_for<4>([&](auto qt) {
            static_for<lf::NDK>([&](auto k) {
                const opnd_t qa = load_rows<P_B>(q_rows, 16 * qt, 32 * k);
                const opnd_t da = load_rows<P_B>(do_rows, 16 * qt, 32 * k);
                static_for<2>([&](auto t) {
                    if constexpr (decltype(k)::value == 0) {
                        s[qt][t]  = wmma0(qa, kb_op[t][0]);
                        dp[qt][t] = wmma0(da, vb_op[t][0]);
                    } else {
                        s[qt][t]  = wmma(qa, kb_op[t][k], s[qt][t]);
                        dp[qt][t] = wmma(da, vb_op[t][k], dp[qt][t]);
                    }
                });
            });
        });
        // P and dS (row vectors NL / delta in accumulator-row order)
        static_for<4>([&](auto qt) {
            const int   q0  = 64 * qb + 16 * qt;
            const acc_t nl  = lb::rows8(rnl, b, Hq, hq, Sq, q0, l);
            const acc_t del = lb::rows8(rdel, b, Hq, hq, Sq, q0, l);
            static_for<2>([&](auto t) {
                s[qt][t]  = lb::probs_qk(s[qt][t], c2, nl, l, causal != 0, q0 + cshift - (kv0 + 16 * t));
                dp[qt][t] = lb::dsoft(s[qt][t], dp[qt][t], del);
            });
        });

        // ---- dS^T -> LDS buffer it & 1 (rows kv 32 w + 16 t .., columns q); read back by dQ(it) next iteration --
        {
            lds_u8 *const dsr = ds_rows(it);
            static_for<4>([&](auto qt) {
                static_for<2>([&](auto t) { store_ct<DS_P>(dsr, 32 * w + 16 * t, 16 * qt, dp[qt][t]); });
            });
        }

        // dV^T += dO^T P, dK^T += Q^T dS (32-query chunks; each dO^T / Q^T operand feeds both kv tiles)
        static_for<2>([&](auto c) {
            opnd_t pb[2], sb[2];
            static_for<2>([&](auto t) {
                pb[t] = pack_p_bwd(s[2 * c][t], s[2 * c + 1][t]);
                sb[t] = pack_p_bwd(dp[2 * c][t], dp[2 * c + 1][t]);
            });
            static_for<lf::NDT>([&](auto dt) {
                const opnd_t doa = load_tr<P_B>(do_tr, 32 * c, 16 * dt), qa = load_tr<P_B>(q_tr, 32 * c, 16 * dt);
                static_for<2>([&](auto t) {
                    dvt[t][dt] = wmma(doa, pb[t], dvt[t][dt]);
                    dkt[t][dt] = wmma(qa, sb[t], dkt[t][dt]);
                });
            });
        });
        // publish this iteration's dS^T and the next iteration's tiles (the signal drains this wave's DS ops)
        if (next) split_signal_tdm_lds<0>();
        else split_signal_lds();
        split_wait_lds();
    }
    if (nit > 0) dq_of(nit - 1);
    static_for<2>([&](auto t) {
        const int row = lf::row_b(b, Skv, kv0 + 16 * t + l % 16, Hkv, hkv);
        lb::store_rows(make_rsrc(DK, kv_bytes), dkt[t], scale, row, l);
        lb::store_rows(make_rsrc(DV, kv_bytes), dvt[t], 1.0f, row, l);
    });
}

// dq = bf16(scale dQ_acc): thread t converts the 8 fp32 at byte 32 t (out of range: loads 0, stores dropped)
extern "C" __global__ __launch_bounds__(256) void k_bwd_b06_cvt(const float *DQA, __bf16 *DQ, long long dqa_bytes,
                                                                long long q_bytes, float scale) {
    const int    t  = (int) (__builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x());
    const rsrc_t ra = make_rsrc(DQA, dqa_bytes), rq = make_rsrc(DQ, q_bytes);
    const f32x4  a = buf_load_f32x4(ra, 32 * t, 0), c = buf_load_f32x4(ra, 32 * t + 16, 0);
    const acc_t  v = __builtin_shufflevector(a, c, 0, 1, 2, 3, 4, 5, 6, 7);
    buf_store_b128(rq, 16 * t, 0, cvt_pk_bf16(v * scale));
}
