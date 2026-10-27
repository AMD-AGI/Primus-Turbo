/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// m04b_bursts -- DeepSeek-V3 MLA backward ladder (main line): m03q with the dK / dV kernel's LDS operands in bursts
// and its NL / delta rows requested early.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O) (128 wide), NL = -LSE log2 e              grid (Sq/64, H, B), block 256
//   dkdv  workgroup = 64 kv rows of one (b, head); wave w = kv rows [16 w, +16); K (6 k-steps) / V (4) resident
//         loop: every visible 64-query block: Q (64 x 192, pitch 400) and dO (64 x 128, pitch 272) by TDM into
//         TWO buffers (waves 0 / 1: Q rows [0, 32) / [32, 64) as a 128- and a 64-column op, waves 2 / 3: dO
//         halves; the next block's ops issued before this block's compute, 1 barrier);
//         NL / delta rows of the 4 q tiles requested first (global); S, dP per q tile from one burst of its 6 Q +
//         4 dO operands; the 20 dO^T / Q^T tr16 operands of chunk 0 issued before the softmax, chunk 1's after
//         chunk 0's WMMAs; P, dS, dV^T += dO^T P (8 d tiles), dK^T += Q^T dS (12 d tiles)
//         + every dS tile stored as dS^T (bf16, C^T order: one 16-B store per lane and q tile) to a scratch
//           [B, H, Skv, Sq] (only the causal-visible blocks are written, and only those are read)
//         dK = scale dK^T, dV stored once                                   grid (Skv/64, H, B), block 128
//   dq    workgroup = 128 queries of one head, wave w = queries [32 w, +32) (2 q tiles); loop: every 64-row kv
//           block visible to the workgroup: the K tile (pitch 400) and the dS^T block [64 kv][128 q] (pitch 288)
//           by TDM, two buffers, 1 barrier; per 32-row kv chunk one burst of the 12 K^T operands + the wave's 2
//           dS^T operands (tr16), then dQ^T += K^T dS^T: 24 WMMAs, each K^T operand feeding both q tiles; a wave
//           past its own last visible kv block skips the chunk (wave-uniform; those dS^T bytes were never written)
//         dQ = scale dQ^T                                                    grid (Sq/128, H, B), block 128
// What changes from m03q: the dK / dV kernel (two thirds of m03q's time) reads every LDS operand right before its
// WMMA and the NL / delta rows from global right before the softmax that needs them. As in the Llama ladder's
// b03b: one burst per q tile for S / dP, the tr16 operands of the first dV / dK chunk in flight during the
// softmax, the row vectors requested a whole S / dP run ahead.
// 192-wide rows are 384 B, which the TDM cannot pad (its pad interval is a power-of-two row), so a 192-wide
// half tile is two ops: columns [0, 128) (256-B rows + 144-B pad) and [128, 192) (128-B rows + 272-B pad).
#include "ladder/lm.h"

using namespace hk_attn;

namespace {

constexpr int QT_B  = 64 * lm::PQ;  // 25600: a 64 x 192 tile (Q in dkdv, K in dq)
constexpr int VT_B  = 64 * lm::PV;  // 17408: a 64 x 128 tile (dO in dkdv, V in dq)
constexpr int BUF_B = QT_B + VT_B;  // one buffer: the 192-wide tile, then the 128-wide tile
constexpr int LDS_B = 2 * BUF_B;     // 86016
constexpr int DQ_QB = 128;               // queries per dQ workgroup
constexpr int DS_P  = 2 * DQ_QB + 32;     // 288: the dS^T block [64 kv][128 q] in the dQ kernel (tr16 reads)
constexpr int DST_B = 64 * DS_P;          // 18432
constexpr int QBUF_B = QT_B + DST_B;      // dQ kernel buffer: K tile, then the dS^T block
constexpr int QLDS_B = 2 * QBUF_B;        // 88064
using T_DS = tdm_tile<32, DQ_QB, DS_P>;   // 32 kv rows x 128 q of dS^T

// the scratch rows of (b, head h): dS^T [Skv][Sq] bf16
HKA_DEV rsrc_t ds_head(const __bf16 *DS, int b, int H, int h, int Sq, int Skv) {
    return make_rsrc(DS + ((int64_t) b * H + h) * Skv * Sq, (int64_t) Skv * Sq * 2);
}

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

extern "C" __global__ __launch_bounds__(256) void k_mla_m04b_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int H) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, H);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_mla_m04b_dkdv(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DK, __bf16 *DV, __bf16 *DS, long long q_bytes, long long k_bytes, long long v_bytes, long long o_bytes,
    long long row_bytes, long long ds_bytes, int B, int Sq, int Skv, int H, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int kb = (int) __builtin_amdgcn_workgroup_id_x(), h = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int kv0 = 64 * kb + 16 * w, cshift = Skv - Sq;

    const uint32_t lds0 = lds_addr(smem);
    auto buf = [&](int i) { return lds0 + (uint32_t) ((i & 1) * BUF_B); }; // Q tile at +0, dO tile at +QT_B
    const rsrc_t   rq = make_rsrc(Q, q_bytes), rk = make_rsrc(K, k_bytes), rv = make_rsrc(V, v_bytes);
    const rsrc_t   rdo = make_rsrc(DO, o_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);
    const rsrc_t   rds = ds_head(DS, b, H, h, Sq, Skv);

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

        // NL / delta rows of the 4 q tiles (global): requested now, consumed after the S / dP run
        acc_t nl[4], del[4];
        static_for<4>([&](auto qt) {
            nl[qt]  = lb::rows8(rnl, b, H, h, Sq, 64 * qb + 16 * qt, l);
            del[qt] = lb::rows8(rdel, b, H, h, Sq, 64 * qb + 16 * qt, l);
        });
        // S = Q K^T (6 k-steps), dP = dO V^T (4): 4 q tiles of 16 rows, lane = kv; per q tile one burst
        acc_t s[4], dp[4];
        static_for<4>([&](auto qt) {
            opnd_t qa[lm::NKQ], da[lm::NKV];
            static_for<lm::NKQ>([&](auto k) { qa[k] = load_rows<lm::PQ>(q_rows, 16 * qt, 32 * k); });
            static_for<lm::NKV>([&](auto k) { da[k] = load_rows<lm::PV>(do_rows, 16 * qt, 32 * k); });
            sched_barrier<0>();
            static_for<lm::NKQ>([&](auto k) {
                if constexpr (decltype(k)::value == 0) s[qt] = wmma0(qa[0], kb_op[0]);
                else s[qt] = wmma(qa[k], kb_op[k], s[qt]);
            });
            static_for<lm::NKV>([&](auto k) {
                if constexpr (decltype(k)::value == 0) dp[qt] = wmma0(da[0], vb_op[0]);
                else dp[qt] = wmma(da[k], vb_op[k], dp[qt]);
            });
            sched_barrier<0>();
        });
        // the dO^T / Q^T operands of dV / dK chunk c (tr16): chunk 0's burst flies during the softmax
        opnd_t tv[lm::NTV], tq[lm::NTQ];
        auto   tload = [&](int c) {
            static_for<lm::NTV>([&](auto dt) { tv[dt] = load_tr<lm::PV>(do_tr, 32 * c, 16 * dt); });
            static_for<lm::NTQ>([&](auto dt) { tq[dt] = load_tr<lm::PQ>(q_tr, 32 * c, 16 * dt); });
        };
        tload(0);
        sched_barrier<0>();
        static_for<4>([&](auto qt) {
            const int q0 = 64 * qb + 16 * qt;
            s[qt]  = lb::probs_qk(s[qt], c2, nl[qt], l, causal != 0, q0 + cshift - kv0);
            dp[qt] = lb::dsoft(s[qt], dp[qt], del[qt]);
            // dS^T rows kv0 + r, columns q0 + 8 h .. + 8 (gmem.h buf_store_ct_bf16; RNE, the bits dK uses)
            buf_store_ct_bf16(rds, (kv0 + l % 16) * Sq * 2 + (q0 + 8 * (l / 16)) * 2, 0, dp[qt]);
        });
        // dV^T += dO^T P (8 d tiles), dK^T += Q^T dS (12); chunk 1's burst after chunk 0's WMMAs
        static_for<2>([&](auto c) {
            if constexpr (c == 1) {
                tload(1);
                sched_barrier<0>();
            }
            const opnd_t pb = pack_p_bwd(s[2 * c], s[2 * c + 1]);
            const opnd_t sb = pack_p_bwd(dp[2 * c], dp[2 * c + 1]);
            static_for<lm::NTV>([&](auto dt) { dvt[dt] = wmma(tv[dt], pb, dvt[dt]); });
            static_for<lm::NTQ>([&](auto dt) { dkt[dt] = wmma(tq[dt], sb, dkt[dt]); });
            sched_barrier<0>();
        });
        if (next) { // publish block qb + 1 (its buffer was last read an iteration earlier)
            split_signal_tdm_lds<0>();
            split_wait_lds();
        }
    }
    lm::store_rows(make_rsrc(DK, k_bytes), dkt, scale, lm::row<lm::DQK>(b, Skv, kv0 + l % 16, H, h), l);
    lm::store_rows(make_rsrc(DV, v_bytes), dvt, 1.0f, lm::row<lm::DV>(b, Skv, kv0 + l % 16, H, h), l);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_mla_m04b_dq(
    const __bf16 *DS, const __bf16 *K, __bf16 *DQ, long long k_bytes, long long q_bytes, long long ds_bytes, int B,
    int Sq, int Skv, int H, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[QLDS_B];
    const int w = wave_id(), l = lane_id();
    const int qb = (int) __builtin_amdgcn_workgroup_id_x(), h = (int) __builtin_amdgcn_workgroup_id_y();
    const int b = (int) __builtin_amdgcn_workgroup_id_z();
    const int p0 = DQ_QB * qb + 32 * w;

    const uint32_t lds0 = lds_addr(smem);
    auto buf = [&](int j) { return lds0 + (uint32_t) ((j & 1) * QBUF_B); }; // K tile at +0, dS^T block at +QT_B
    // block j: waves 0 / 1 move K rows [32 w, +32) (two ops: 128 + 64 columns), waves 2 / 3 dS^T rows
    // [32 (w - 2), +32) of kv block j, columns [128 qb, +128) (row stride Sq)
    auto issue = [&](int j) {
        const int half = w & 1;
        if (w < 2) {
            const int64_t  r   = (int64_t) b * Skv + 64 * j + 32 * half;
            const uint32_t dst = buf(j) + (uint32_t) (32 * half * lm::PQ);
            tdm_load(tdm_g0(dst, tdm_gaddr(K, r, (int64_t) H * lm::DQK, (int64_t) h * lm::DQK)),
                     tdm_g1<T_W128>((int64_t) H * lm::DQK, 32));
            tdm_load(tdm_g0(dst + 256, tdm_gaddr(K, r, (int64_t) H * lm::DQK, (int64_t) h * lm::DQK + 128)),
                     tdm_g1<T_W64>((int64_t) H * lm::DQK, 32));
        } else {
            const __bf16 *ds_bh = DS + ((int64_t) b * H + h) * Skv * Sq;
            tdm_load(tdm_g0(buf(j) + (uint32_t) (QT_B + 32 * half * DS_P),
                            tdm_gaddr(ds_bh, 64 * j + 32 * half, Sq, DQ_QB * qb)),
                     tdm_g1<T_DS>((int64_t) Sq, 32));
        }
    };
    acc_t dqt[2][lm::NTQ]; // dQ^T of q tile qt: rows d, lane = q
    static_for<2>([&](auto qt) { static_for<lm::NTQ>([&](auto dt) { dqt[qt][dt] = zero_acc(); }); });

    // kv blocks with an exported dS^T block: the workgroup's (its last 64-query block's) and this wave's
    const int n = lf::n_kv_blocks(2 * qb + 1, Sq, Skv, causal), nw = lf::n_kv_blocks(2 * qb + w / 2, Sq, Skv, causal);
    issue(0);
    split_signal_tdm_lds<0>();
    split_wait_lds();
    for (int j = 0; j < n; ++j) {
        const bool next = j + 1 < n; // workgroup-uniform
        if (next) issue(j + 1);
        if (j < nw) { // wave-uniform
            const lds_u8 *k_tr  = tr_base<lm::PQ>(buf(j), l);
            const lds_u8 *ds_tr = tr_base<DS_P>(buf(j) + (uint32_t) (QT_B + 2 * 32 * w), l); // this wave's 32 queries
            // dQ^T += K^T dS^T per 32-row kv chunk: A = K^T [d][kv] (tr16 of the K tile), B = dS^T [kv][q]
            static_for<2>([&](auto c) {
                opnd_t ka[lm::NTQ], sb[2];
                static_for<2>([&](auto qt) { sb[qt] = load_tr<DS_P>(ds_tr, 32 * c, 16 * qt); });
                static_for<lm::NTQ>([&](auto dt) { ka[dt] = load_tr<lm::PQ>(k_tr, 32 * c, 16 * dt); });
                sched_barrier<0>();
                static_for<lm::NTQ>([&](auto dt) {
                    static_for<2>([&](auto qt) { dqt[qt][dt] = wmma(ka[dt], sb[qt], dqt[qt][dt]); });
                });
                sched_barrier<0>();
            });
        }
        if (next) { // publish block j + 1 (its buffer was last read an iteration earlier)
            split_signal_tdm_lds<0>();
            split_wait_lds();
        }
    }
    static_for<2>([&](auto qt) {
        lm::store_rows(make_rsrc(DQ, q_bytes), dqt[qt], scale, lm::row<lm::DQK>(b, Sq, p0 + 16 * qt + l % 16, H, h), l);
    });
}
