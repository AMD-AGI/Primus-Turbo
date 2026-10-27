/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// m01_tdm_dbuf -- rung 01 of the DeepSeek-V3 MLA backward ladder: m00 with double-buffered TDM fills (the Llama
// ladder's b01 + b02 in one step: the next tiles in flight during this iteration, moved by the TDM).
//
// Kernel Specification
//   prep  delta = rowsum(dO o O) (128 wide), NL = -LSE log2 e              grid (Sq/64, H, B), block 256
//   dkdv  workgroup = 64 kv rows of one (b, head); wave w = kv rows [16 w, +16); K (6 k-steps) / V (4) resident
//         loop: every visible 64-query block: Q (64 x 192, pitch 400) and dO (64 x 128, pitch 272) by TDM into
//         TWO buffers (waves 0 / 1: Q rows [0, 32) / [32, 64) as a 128- and a 64-column op, waves 2 / 3: dO
//         halves; the next block's ops issued before this block's compute, 1 barrier); S, dP, P, dS,
//         dV^T += dO^T P (8 d tiles), dK^T += Q^T dS (12 d tiles)
//         dK = scale dK^T, dV stored once                                   grid (Skv/64, H, B), block 128
//   dq    workgroup = 64 queries of one head; Q^T (6 k-steps) / dO^T (4) resident
//         loop: every visible 64-row kv block: K (pitch 400) and V (pitch 272) by TDM, two buffers, 1 barrier;
//           S^T = K Q^T, dP^T = V dO^T (lane = q), P^T, dS^T, dQ^T += K^T dS^T (12 d tiles)
//         dQ = scale dQ^T                                                    grid (Sq/64, H, B), block 128
// S and dP are computed in both kernels: FlyDSL's MLA backward does the same (k_dkdv + k_dqg).
// 192-wide rows are 384 B, which the TDM cannot pad (its pad interval is a power-of-two row), so a 192-wide
// half tile is two ops: columns [0, 128) (256-B rows + 144-B pad) and [128, 192) (128-B rows + 272-B pad).
#include "ladder/lm.h"

using namespace hk_attn;

namespace {

constexpr int QT_B  = 64 * lm::PQ;  // 25600: a 64 x 192 tile (Q in dkdv, K in dq)
constexpr int VT_B  = 64 * lm::PV;  // 17408: a 64 x 128 tile (dO in dkdv, V in dq)
constexpr int BUF_B = QT_B + VT_B;  // one buffer: the 192-wide tile, then the 128-wide tile
constexpr int LDS_B = 2 * BUF_B;     // 86016

using T_W128 = tdm_tile<32, 128, lm::PQ>; // 192-wide tile, columns [0, 128)
using T_W64  = tdm_tile<32, 64, lm::PQ>;  // 192-wide tile, columns [128, 192)
using T_V    = tdm_tile<32, 128, lm::PV>; // 128-wide tile

// this wave's ops for a buffer at `buf`: waves 0 / 1 move rows [32 w, +32) of the 192-wide tensor t192 (two ops),
// waves 2 / 3 rows [32 (w - 2), +32) of the 128-wide tensor t128 (one op); rows row0 .. of (b, h)
HKA_DEV void issue_tiles(uint32_t buf, const void *t192, const void *t128, int b, int S, int row0, int H, int h,
                         int w) {
    const int half = w & 1;
    const int64_t r = (int64_t) b * S + row0 + 32 * half;
    if (w < 2) {
        const uint32_t dst = buf + (uint32_t) (32 * half * lm::PQ);
        tdm_load(tdm_g0(dst, tdm_gaddr(t192, r, (int64_t) H * lm::DQK, (int64_t) h * lm::DQK)),
                 tdm_g1<T_W128>((int64_t) H * lm::DQK, 32));
        tdm_load(tdm_g0(dst + 256, tdm_gaddr(t192, r, (int64_t) H * lm::DQK, (int64_t) h * lm::DQK + 128)),
                 tdm_g1<T_W64>((int64_t) H * lm::DQK, 32));
    } else {
        tdm_load(tdm_g0(buf + (uint32_t) (QT_B + 32 * half * lm::PV),
                        tdm_gaddr(t128, r, (int64_t) H * lm::DV, (int64_t) h * lm::DV)),
                 tdm_g1<T_V>((int64_t) H * lm::DV, 32));
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(256) void k_mla_m01_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int H) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, H);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_mla_m01_dkdv(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DK, __bf16 *DV, long long q_bytes, long long k_bytes, long long v_bytes, long long o_bytes,
    long long row_bytes, int B, int Sq, int Skv, int H, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int kb = (int) __builtin_amdgcn_workgroup_id_x(), h = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int kv0 = 64 * kb + 16 * w, cshift = Skv - Sq;

    const uint32_t lds0 = lds_addr(smem);
    auto buf = [&](int i) { return lds0 + (uint32_t) ((i & 1) * BUF_B); }; // Q tile at +0, dO tile at +QT_B
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, k_bytes), rv = make_rsrc(V, v_bytes);
    const rsrc_t   rdo = make_rsrc(DO, o_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);

    // K / V rows as B^T operands of S = Q K^T (6 k-steps) and dP = dO V^T (4): resident, lane = kv
    opnd_t kb_op[lm::NKQ], vb_op[lm::NKV];
    lm::load_bt(kb_op, rk, lm::row<lm::DQK>(b, Skv, kv0, H, h), H * lm::DQK * 2, l);
    lm::load_bt(vb_op, rv, lm::row<lm::DV>(b, Skv, kv0, H, h), H * lm::DV * 2, l);
    acc_t dkt[lm::NTQ], dvt[lm::NTV]; // dK^T, dV^T: rows d, lane = kv
    static_for<lm::NTQ>([&](auto dt) { dkt[dt] = zero_acc(); });
    static_for<lm::NTV>([&](auto dt) { dvt[dt] = zero_acc(); });

    const int qb0 = lb::first_q_block(kb, Sq, Skv, causal), nqb = Sq / 64;
    issue_tiles(buf(0), Q, DO, b, Sq, 64 * qb0, H, h, w);
    split_signal_tdm_lds<0>();
    split_wait_lds();
    for (int qb = qb0; qb < nqb; ++qb) {
        const int  i    = qb - qb0;
        const bool next = qb + 1 < nqb; // workgroup-uniform
        if (next) issue_tiles(buf(i + 1), Q, DO, b, Sq, 64 * (qb + 1), H, h, w);
        const lds_u8 *q_rows = rows_base<lm::PQ>(buf(i), l), *do_rows = rows_base<lm::PV>(buf(i) + QT_B, l);
        const lds_u8 *q_tr = tr_base<lm::PQ>(buf(i), l), *do_tr = tr_base<lm::PV>(buf(i) + QT_B, l);

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
        if (next) { // publish block qb + 1 (its buffer was last read an iteration earlier)
            split_signal_tdm_lds<0>();
            split_wait_lds();
        }
    }
    lm::store_rows(make_rsrc(DK, k_bytes), dkt, scale, lm::row<lm::DQK>(b, Skv, kv0 + l % 16, H, h), l);
    lm::store_rows(make_rsrc(DV, v_bytes), dvt, 1.0f, lm::row<lm::DV>(b, Skv, kv0 + l % 16, H, h), l);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_mla_m01_dq(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DQ, long long q_bytes, long long k_bytes, long long v_bytes, long long o_bytes, long long row_bytes,
    int B, int Sq, int Skv, int H, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int qb = (int) __builtin_amdgcn_workgroup_id_x(), h = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int p0 = 64 * qb + 16 * w, cshift = Skv - Sq;

    const uint32_t lds0 = lds_addr(smem);
    auto buf = [&](int j) { return lds0 + (uint32_t) ((j & 1) * BUF_B); }; // K tile at +0, V tile at +QT_B
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
    issue_tiles(buf(0), K, V, b, Skv, 0, H, h, w);
    split_signal_tdm_lds<0>();
    split_wait_lds();
    for (int j = 0; j < n; ++j) {
        const bool next = j + 1 < n; // workgroup-uniform
        if (next) issue_tiles(buf(j + 1), K, V, b, Skv, 64 * (j + 1), H, h, w);
        const lds_u8 *k_rows = rows_base<lm::PQ>(buf(j), l), *v_rows = rows_base<lm::PV>(buf(j) + QT_B, l);
        const lds_u8 *k_tr = tr_base<lm::PQ>(buf(j), l);

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
        if (next) { // publish block j + 1
            split_signal_tdm_lds<0>();
            split_wait_lds();
        }
    }
    lm::store_rows(make_rsrc(DQ, q_bytes), dqt, scale, lm::row<lm::DQK>(b, Sq, p0 + l % 16, H, h), l);
}
