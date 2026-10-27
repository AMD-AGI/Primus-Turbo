/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// b07n_prefetch -- Llama backward ladder (main line): b06k with the next step's operands requested a step ahead.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O), NL = -LSE log2 e                          grid (Sq/64, Hq, B), block 256
//   dkdv  workgroup = 128 kv rows of one (b, kv head); wave w = kv rows [32 w, +32) = 2 kv tiles, K / V resident
//         (B operands, 128 VGPR); the WG's K image [128 kv][128 d] in LDS once (TDM), from which wave w keeps the
//         B operands of dQ for d columns [32 w, +32) over all 128 kv (K slice, 64 VGPR)
//         loop: steps of 32 queries x one q head of the group, heads inner:
//           Q / dO tiles [32][128] by TDM (pitch 272, two buffers; wave w moves 16 rows of Q (w < 2) or dO;
//           step s + 1's ops in flight during step s),
//           S = Q K^T, dP = dO V^T (2 q tiles x 2 kv tiles, lane = kv) from the 16 Q / dO operands read back in
//           step s - 1, each feeding 4 WMMAs, the 16 dO^T / Q^T tr16 operands of dV / dK issued before the softmax,
//           P, dS; dS^T -> LDS (rows kv 32 w + .., two buffers by step parity),
//           dV^T += dO^T P, dK^T += Q^T dS (32 WMMA),
//           steps ordered as v0's: q pairs descending, so the pairs every wave sees whole come first and run
//           in a loop of their own (no mask, no branch); in the diagonal steps the work goes by wave kind
//           (wave-uniform branch): FULL (every (q, kv) of the wave's 32 x 32
//           visible: no mask), DIAG (the masked code), DEAD (nothing visible: S / dP / softmax / dV / dK skipped,
//           a zero dS^T stored),
//           the NL / delta rows of step s + 1 requested (global, into the registers this step's softmax released),
//           ONE barrier (next Q / dO landed, every wave's dS^T stored),
//           dQ[32 q][d 32 w ..] = dS K over the 128 kv (A = dS by tr16 of the dS^T tile, B = the K slice: 16
//           WMMA; behind the dS loads the readback of step s + 1's 16 Q / dO operands, which lands under the dQ
//           WMMAs and atomics), 32 fp32 atomics per lane into dQ_acc in C-fragment order (each instruction one 128-B line)
//         dK = scale dK^T, dV stored once                                  grid (Skv/128, Hkv, B), block 128
//         work order: linear WG id i -> kv head i % Hkv, kv block (i / Hkv) % (Skv/128), batch last: every kv
//         block 0 (causal: the most steps) of the grid is dispatched first, the shortest blocks last
//   cvt   dq = bf16(scale dQ_acc), de-swizzled from the C-fragment order  grid (B Sq Hq 128 / 2048), block 256
// dQ_acc layout (fp32, 1 KB per 16 x 16 tile): byte ((((b Hq + h) Sq/16 + q / 16) 8 + d / 16) 1024 + 128 i + 4 l
// for element i of lane l = 16 hh + r of the C fragment (row q = 8 hh + i, column d = r).
//
// What changes from b06k: at one wave per SIMD nothing hides a wait: the NL / delta rows (global) were requested
// at the top of the step that needs them, and each q tile's Q / dO burst at the top of its S / dP run. Both are
// known a step ahead: the row vectors of step s + 1 go out once step s's softmax has consumed its own, and step
// s + 1's tiles are published by step s's barrier, so their A operands are read back right after it, issued
// behind the dS loads of step s's dQ (DS loads complete in order: the dQ WMMAs wait for theirs only) and landing
// under the dQ WMMAs and atomics (v0's "readback" phase). The price: 128 more VGPRs for a whole step's operands.
// (The same on b04f, before the work order and the wave kinds, measured 0.95x.)
#include "ladder/lb.h"

using namespace hk_attn;

namespace {

template <int N> using ic = std::integral_constant<int, N>;
#define INL __attribute__((always_inline))

constexpr int KVB    = 128;            // kv rows per workgroup
constexpr int QS     = 32;             // queries per step
constexpr int P_B    = 2 * lf::D + 16; // 272: Q / dO tiles and the K image (rows and tr16 reads)
constexpr int DS_P   = 2 * QS + 32;    // 96: dS^T tile [128 kv][32 q] (C^T stores, tr16 reads)
constexpr int TILE   = QS * P_B;       // 8704
constexpr int BUF_B  = 2 * TILE;       // one buffer: Q tile, then dO tile
constexpr int K_OFF  = 2 * BUF_B;      // 34816
constexpr int DS_OFF = K_OFF + KVB * P_B;     // 69632
constexpr int DS_B   = KVB * DS_P;            // 12288
constexpr int LDS_B  = DS_OFF + 2 * DS_B;     // 94208

using T_QDO = tdm_tile<16, lf::D, P_B>; // one Q / dO op: 16 rows x 128 bf16
using T_K   = tdm_tile<32, lf::D, P_B>; // one K image op: 32 rows

} // namespace

extern "C" __global__ __launch_bounds__(256) void k_bwd_b07n_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int Hq) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, Hq);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_bwd_b07n_dkdv(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DK, __bf16 *DV, float *DQA, long long q_bytes, long long kv_bytes, long long row_bytes,
    long long dqa_bytes, int B, int Sq, int Skv, int Hq, int Hkv, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id();
    const int nkb = Skv / KVB;
    const int wg  = (int) (__builtin_amdgcn_workgroup_id_x() +
                          nkb * (__builtin_amdgcn_workgroup_id_y() + Hkv * __builtin_amdgcn_workgroup_id_z()));
    const int hkv = wg % Hkv, kb = (wg / Hkv) % nkb, b = wg / (Hkv * nkb); // longest first
    const int G = Hq / Hkv, kv0 = KVB * kb + 32 * w, cshift = Skv - Sq; // kv tile t of the wave: kv0 + 16 t
    const int kv_stride = Hkv * lf::D * 2;

    const uint32_t lds0 = lds_addr(smem);
    auto q_lds  = [&](int it) { return lds0 + (uint32_t) ((it & 1) * BUF_B); }; // Q tile of step it
    auto do_lds = [&](int it) { return lds0 + (uint32_t) ((it & 1) * BUF_B + TILE); };
    auto ds_lds = [&](int it) { return lds0 + (uint32_t) (DS_OFF + (it & 1) * DS_B); };
    const rsrc_t rk = make_rsrc(K, kv_bytes), rv = make_rsrc(V, kv_bytes);
    const rsrc_t rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes), rqa = make_rsrc(DQA, dqa_bytes);

    // K / V rows as B^T operands of S = Q K^T and dP = dO V^T (B[k = d][n = kv], lane = kv): resident
    opnd_t kb_op[2][lf::NDK], vb_op[2][lf::NDK];
    acc_t  dkt[2][lf::NDT], dvt[2][lf::NDT]; // dK^T, dV^T of kv tile t: rows d, lane = kv
    static_for<2>([&](auto t) {
        lf::load_qt(kb_op[t], rk, lf::row_b(b, Skv, kv0 + 16 * t, Hkv, hkv), kv_stride, l);
        lf::load_qt(vb_op[t], rv, lf::row_b(b, Skv, kv0 + 16 * t, Hkv, hkv), kv_stride, l);
        static_for<lf::NDT>([&](auto dt) { dkt[t][dt] = zero_acc(); dvt[t][dt] = zero_acc(); });
    });

    // steps: (q pair, head), heads inner, pairs DESCENDING: it -> queries 32 (Sq/32 - 1 - it / G) .., head
    // hkv G + it % G, down to the first q pair that sees kv row 128 kb (the one holding query 128 kb - cshift);
    // steps [0, nf) are pairs every wave sees whole (FULL for the WG), [nf, nit) the diagonal pairs
    // is the one holding query 128 kb - cshift (the first that sees kv row 128 kb)
    const int qf = KVB * kb - cshift, qp0 = causal == 0 || qf <= 0 ? 0 : qf / QS, nit = (Sq / QS - qp0) * G;
    const int qe = KVB * kb + KVB - 1 - cshift; // a pair is FULL for the WG iff its first query >= qe
    const int qpf = causal == 0 || qe <= 0 ? 0 : (qe + QS - 1) / QS, nf = (Sq / QS - max(qpf, qp0)) * G;
    auto pair_q0 = [&](int it) { return QS * (Sq / QS - 1 - it / G); };
    const u32x8 g1 = tdm_g1<T_QDO>((int64_t) Hq * lf::D, 16);
    auto issue = [&](int it) { // wave w: rows [16 (w & 1), +16) of Q (w < 2) or dO of step it
        const int row0 = pair_q0(it) + 16 * (w & 1), hq = hkv * G + it % G;
        tdm_load(tdm_g0(q_lds(it) + (uint32_t) ((w >> 1) * TILE + 16 * (w & 1) * P_B),
                        tdm_gaddr((w >> 1) ? (const void *) DO : (const void *) Q, (int64_t) b * Sq + row0,
                                  (int64_t) Hq * lf::D, (int64_t) hq * lf::D)),
                 g1);
    };
    // the K image, once: wave w moves rows [32 w, +32)
    tdm_load(tdm_g0(lds0 + K_OFF + (uint32_t) (32 * w * P_B),
                    tdm_gaddr(K, (int64_t) b * Skv + KVB * kb + 32 * w, (int64_t) Hkv * lf::D, (int64_t) hkv * lf::D)),
             tdm_g1<T_K>((int64_t) Hkv * lf::D, 32));
    if (nit > 0)
        issue(0);
    split_signal_tdm_lds<0>(); // own ops landed: K image rows, step 0's Q / dO rows
    split_wait_lds();
    // the S / dP A operands of a step (Q / dO rows of its two q tiles) and its NL / delta rows: step 0's now
    opnd_t xa[2][2][lf::NDK]; // [q tile][Q, dO][k]
    acc_t  nl[2], del[2];
    auto   readback = [&](int it) INL {
        const lds_u8 *q_rows = rows_base<P_B>(q_lds(it), l), *do_rows = rows_base<P_B>(do_lds(it), l);
        static_for<2>([&](auto qt) {
            static_for<lf::NDK>([&](auto k) {
                xa[qt][0][k] = load_rows<P_B>(q_rows, 16 * qt, 32 * k);
                xa[qt][1][k] = load_rows<P_B>(do_rows, 16 * qt, 32 * k);
            });
        });
    };
    auto rows = [&](int it) INL {
        const int q0 = pair_q0(it), hq = hkv * G + it % G;
        static_for<2>([&](auto qt) {
            nl[qt]  = lb::rows8(rnl, b, Hq, hq, Sq, q0 + 16 * qt, l);
            del[qt] = lb::rows8(rdel, b, Hq, hq, Sq, q0 + 16 * qt, l);
        });
    };
    if (nit > 0) {
        readback(0);
        rows(0);
    }
    // the K slice: B operands of dQ = dS K, B[k = kv 32 c ..][n = d 32 w + 16 j ..] (tr16 of the K image)
    opnd_t kt[2][KVB / 32];
    {
        const lds_u8 *kc = tr_base<P_B>(lds0 + K_OFF + (uint32_t) (2 * 32 * w), l);
        static_for<2>([&](auto j) { static_for<KVB / 32>([&](auto c) { kt[j][c] = load_tr<P_B>(kc, 32 * c, 16 * j); }); });
    }
    // dQ_acc: this lane's byte in the 1-KB block of (q tile, d tile 2 w + j) of head hq, b (C-fragment order)
    const int nqt = Sq / 16;
    auto dqa_blk = [&](int hq, int qt, int j) { return ((((b * Hq + hq) * nqt + qt) * 8 + 2 * w + j) * 1024) + 4 * l; };

    auto step = [&](int it, auto diag_c) INL {
        constexpr bool DIAG = decltype(diag_c)::value; // a diagonal step: per-wave kinds
        const int  q0 = pair_q0(it), hq = hkv * G + it % G;
        const bool next = it + 1 < nit; // workgroup-uniform
        if (next) issue(it + 1);         // into the buffer of step it - 1: every wave's reads retired at its barrier
        const lds_u8 *q_tr = tr_base<P_B>(q_lds(it), l), *do_tr = tr_base<P_B>(do_lds(it), l);

        // the step's work by wave kind: FULL (no mask), DIAG (masked), DEAD (zero dS^T)
        lds_u8 *const ds_rows = rows_base<DS_P>(ds_lds(it), l);
        auto work = [&](auto mask_c) INL {
            constexpr bool MASK = decltype(mask_c)::value;
            // S = Q K^T, dP = dO V^T: 2 q tiles x 2 kv tiles, lane = kv, from the operands read back a step ago
            acc_t s[2][2], dp[2][2];
            static_for<2>([&](auto qt) {
                static_for<lf::NDK>([&](auto k) {
                    static_for<2>([&](auto t) {
                        if constexpr (decltype(k)::value == 0) {
                            s[qt][t]  = wmma0(xa[qt][0][0], kb_op[t][0]);
                            dp[qt][t] = wmma0(xa[qt][1][0], vb_op[t][0]);
                        } else {
                            s[qt][t]  = wmma(xa[qt][0][k], kb_op[t][k], s[qt][t]);
                            dp[qt][t] = wmma(xa[qt][1][k], vb_op[t][k], dp[qt][t]);
                        }
                    });
                });
            });
            sched_barrier<0>();
            // the dO^T / Q^T operands of dV / dK (tr16): one burst, flies during the softmax
            opnd_t ta[2][lf::NDT]; // [dO^T, Q^T][dt]
            static_for<lf::NDT>([&](auto dt) {
                ta[0][dt] = load_tr<P_B>(do_tr, 0, 16 * dt);
                ta[1][dt] = load_tr<P_B>(q_tr, 0, 16 * dt);
            });
            sched_barrier<0>();
            // P and dS (row vectors NL / delta in accumulator-row order); dS^T -> LDS buffer it & 1
            static_for<2>([&](auto qt) {
                static_for<2>([&](auto t) {
                    s[qt][t]  = lb::probs_qk(s[qt][t], c2, nl[qt], l, MASK, q0 + 16 * qt + cshift - (kv0 + 16 * t));
                    dp[qt][t] = lb::dsoft(s[qt][t], dp[qt][t], del[qt]);
                    store_ct<DS_P>(ds_rows, 32 * w + 16 * t, 16 * qt, dp[qt][t]);
                });
            });
            // dV^T += dO^T P, dK^T += Q^T dS (the 32 queries are one contraction chunk)
            static_for<2>([&](auto t) {
                const opnd_t pb = pack_p_bwd(s[0][t], s[1][t]);
                const opnd_t sb = pack_p_bwd(dp[0][t], dp[1][t]);
                static_for<lf::NDT>([&](auto dt) {
                    dvt[t][dt] = wmma(ta[0][dt], pb, dvt[t][dt]);
                    dkt[t][dt] = wmma(ta[1][dt], sb, dkt[t][dt]);
                });
            });
        };
        const int kind = !DIAG || kv0 + 31 <= q0 + cshift ? 0 : (kv0 > q0 + QS - 1 + cshift ? 2 : 1);
        if (kind == 0) {
            work(std::false_type{});
        } else if (kind == 1) {
            work(std::true_type{});
        } else {
            static_for<2>([&](auto qt) {
                static_for<2>([&](auto t) { store_ct<DS_P>(ds_rows, 32 * w + 16 * t, 16 * qt, zero_acc()); });
            });
        }
        // NL / delta rows of step it + 1 (global) into the registers this step's softmax released
        if (next) rows(it + 1);
        // one barrier: step it + 1's tiles landed (own op), every wave's dS^T stored (DS drained)
        split_signal_tdm_lds<0>();
        split_wait_lds();

        // dQ[32 q][32 d] = dS K over the 128 kv of the WG: A = dS by tr16 of the dS^T tile, B = the K slice
        {
            const lds_u8 *ds_tr = tr_base<DS_P>(ds_lds(it), l);
            opnd_t        dsa[2][KVB / 32];
            static_for<2>([&](auto qh) {
                static_for<KVB / 32>([&](auto c) { dsa[qh][c] = load_tr<DS_P>(ds_tr, 32 * c, 16 * qh); });
            });
            // step it + 1's A operands, behind the dS operands: they land under this step's dQ WMMAs and atomics
            if (next) readback(it + 1);
            sched_barrier<0>();
            acc_t dq[2][2];
            static_for<KVB / 32>([&](auto c) {
                static_for<2>([&](auto qh) {
                    static_for<2>([&](auto j) {
                        if constexpr (decltype(c)::value == 0) dq[qh][j] = wmma0(dsa[qh][0], kt[j][0]);
                        else dq[qh][j] = wmma(dsa[qh][c], kt[j][c], dq[qh][j]);
                    });
                });
            });
            static_for<2>([&](auto qh) {
                static_for<2>([&](auto j) { atomic_add_c<128>(rqa, dqa_blk(hq, q0 / 16 + qh, j), 0, dq[qh][j]); });
            });
        }
    };
    int it = 0;
    for (; it < nf; ++it) step(it, std::false_type{}); // FULL steps: no branch, no mask
    for (; it < nit; ++it) step(it, std::true_type{}); // diagonal steps
    static_for<2>([&](auto t) {
        const int row = lf::row_b(b, Skv, kv0 + 16 * t + l % 16, Hkv, hkv);
        lb::store_rows(make_rsrc(DK, kv_bytes), dkt[t], scale, row, l);
        lb::store_rows(make_rsrc(DV, kv_bytes), dvt[t], 1.0f, row, l);
    });
}

// dq = bf16(scale dQ_acc): thread t converts the 8 fp32 at byte 32 t of the C-fragment order = lanes 8 (j % 4) ..
// + 7, element i = j / 4 (j = t % 32) of block t / 32: one row of 8 consecutive d -> one 16-B store
extern "C" __global__ __launch_bounds__(256) void k_bwd_b07n_cvt(const float *DQA, __bf16 *DQ, long long dqa_bytes,
                                                                 long long q_bytes, float scale, int Sq, int Hq) {
    const int    t  = (int) (__builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x());
    const rsrc_t ra = make_rsrc(DQA, dqa_bytes), rq = make_rsrc(DQ, q_bytes);
    const f32x4  a = buf_load_f32x4(ra, 32 * t, 0), c = buf_load_f32x4(ra, 32 * t + 16, 0);
    const acc_t  v = __builtin_shufflevector(a, c, 0, 1, 2, 3, 4, 5, 6, 7);
    const int    blk = t / 32, j = t % 32, nqt = Sq / 16;
    const int    row = 8 * ((j % 4) / 2) + j / 4, d = 16 * (blk % 8) + 8 * (j % 2);
    const int    qt = (blk / 8) % nqt, bh = blk / 8 / nqt; // bh = b Hq + h
    const int    q = 16 * qt + row, hq = bh % Hq, bb = bh / Hq;
    buf_store_b128(rq, ((bb * Sq + q) * Hq + hq) * 2 * lf::D + 2 * d, 0, cvt_pk_bf16(v * scale)); // dropped if t is out of range
}
