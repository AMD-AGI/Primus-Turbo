/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// m10t_ring3 -- DeepSeek-V3 MLA backward ladder (main line): m09r with a 3-slot Q / dO ring, ops two steps ahead.
//
// Kernel Specification
//   prep  delta = rowsum(dO o O) (128 wide), NL = -LSE log2 e              grid (Sq/64, H, B), block 256
//   dkdv  workgroup = 128 kv rows of one (b, head); wave w = kv rows [32 w, +32) = 2 kv tiles; K (6 k-steps) / V (4)
//         of both tiles resident (160 VGPR), dK^T / dV^T of both (320 VGPR)
//         loop: 32-query steps from the first one that sees kv row 128 kb: Q (32 x 192, pitch 400) and dO (32 x 128,
//         pitch 272) by TDM into a RING of 3 slots (wave 0: Q columns [0, 128), wave 1: [128, 192), waves 2 / 3: dO
//         rows [16 (w - 2), +16); waves 0 / 1 also the step's 32 NL / delta values (one 128-B row each); step s + 2's
//         ops issued at the top of step s into step s - 1's slot, the barrier waits for step s + 1's only);
//         (q tile 0's 10 S / dP operands and the NL / delta rows of the step were read back at the end of the
//         previous step); S, dP per q tile from one burst of its 6 Q +
//         4 dO operands, each feeding the 2 kv tiles (20 WMMA); the 20 dO^T / Q^T tr16 operands of dV / dK
//         issued before the softmax; P, dS (+ the dS^T export as m02); dV^T += dO^T P (2 x 8); the step barrier
//         (every DS read of the step's buffer retired, the next step's tiles published); the readback of the next
//         step's q tile 0 operands + NL / delta rows (LDS); dK^T += Q^T dS (2 x 12, registers only) while they land
//         dK = scale dK^T, dV stored once                                  grid (Skv/128, H, B), block 128
//         work order: linear WG id i -> head i % H, kv block (i / H) % (Skv/128) ascending, batch last
//   dq    workgroup = 256 queries of one head (the last one of a head may hold fewer: Sq % 64 == 0), wave w =
//           queries [64 w, +64) (4 q tiles); loop: every 64-row kv block visible to the workgroup: the K tile
//           (pitch 400) and the dS^T block [64 kv][256 q] (pitch 544; columns past Sq outside the TDM tensor:
//           zeros) by TDM, two buffers, 1 barrier; per 32-row kv chunk one burst of the 12 K^T operands + the
//           wave's 4 dS^T operands (tr16), then dQ^T += K^T dS^T: 48 WMMAs, each K^T operand feeding the 4 q
//           tiles; a wave past its own last visible kv block (or past Sq) skips the chunk (wave-uniform)
//         dQ = scale dQ^T                                                    grid (ceil(Sq/256), H, B), block 128
//         work order: linear WG id i -> head i % H, q block (i / H) % (Sq/128) DESCENDING, batch last
// What changes from m09r: m09r's thread trace spends 7% of the time in the s_wait_tensorcnt 0 before the step barrier:
// with the barrier moved up to the end of the dV run, the next step's TDM ops (issued at the top of the step) have
// less than a step to land. A third slot lets the ops go two steps ahead (hk_v0's 3-stage ring, prefetch distance
// 2): the barrier retires all but the newest op (s_wait_tensorcnt 2 on waves 0 / 1, which issue two ops a step, 1 on
// waves 2 / 3).
// 192-wide rows are 384 B, which the TDM cannot pad (its pad interval is a power-of-two row), so a 192-wide
// half tile is two ops: columns [0, 128) (256-B rows + 144-B pad) and [128, 192) (128-B rows + 272-B pad).
#include "ladder/lm.h"

using namespace hk_attn;

namespace {

constexpr int KVB   = 128;           // kv rows per dK / dV workgroup
constexpr int QS    = 32;            // queries per dK / dV step
constexpr int QT_B  = QS * lm::PQ;   // 12800: the step's Q tile (32 x 192)
constexpr int VT_B  = QS * lm::PV;   // 8704: the step's dO tile (32 x 128)
constexpr int LD_OFF = QT_B + VT_B;  // in a buffer: Q tile, dO tile, then NL [32], delta [32] (fp32)
constexpr int BUF_B = LD_OFF + 256;
constexpr int NSLOT = 3;             // Q / dO ring slots
constexpr int LDS_B = NSLOT * BUF_B; // 65280
constexpr int KT_B  = 64 * lm::PQ;   // 25600: a 64 x 192 K tile in the dQ kernel
constexpr int DQ_QB = 256;               // queries per dQ workgroup
constexpr int DQ_QW = DQ_QB / 4;         // queries per dQ wave
constexpr int DS_P  = 2 * DQ_QB + 32;     // 544: the dS^T block [64 kv][256 q] in the dQ kernel (tr16 reads)
constexpr int DST_B = 64 * DS_P;          // 34816
constexpr int QBUF_B = KT_B + DST_B;      // dQ kernel buffer: K tile, then the dS^T block
constexpr int QLDS_B = 2 * QBUF_B;        // 120832
using T_DS = tdm_tile<32, DQ_QB, DS_P>;   // 32 kv rows x 256 q of dS^T

// the scratch rows of (b, head h): dS^T [Skv][Sq] bf16
HKA_DEV rsrc_t ds_head(const __bf16 *DS, int b, int H, int h, int Sq, int Skv) {
    return make_rsrc(DS + ((int64_t) b * H + h) * Skv * Sq, (int64_t) Skv * Sq * 2);
}

using T_W128 = tdm_tile<32, 128, lm::PQ>; // 192-wide tile, columns [0, 128)
using T_W64  = tdm_tile<32, 64, lm::PQ>;  // 192-wide tile, columns [128, 192)
using T_V    = tdm_tile<32, 128, lm::PV>; // 128-wide tile

using T_V16  = tdm_tile<16, 128, lm::PV>; // 16 rows of a 128-wide tile
using T_LD   = tdm_tile<1, QS, 4 * QS, 4>; // one NL / delta op: 32 fp32

// this wave's op of a step's buffer at `buf`: wave 0 Q rows [q0, +32) columns [0, 128), wave 1 columns [128, 192),
// waves 2 / 3 dO rows [q0 + 16 (w - 2), +16) of (b, head h)
// (+ waves 0 / 1: the NL / delta row of (b, h), queries [q0, +32), from [B, H, Sq] fp32)
HKA_DEV void issue_step(uint32_t buf, const void *Q, const void *DO, const float *NL, const float *DEL, int b, int Sq,
                        int q0, int H, int h, int w) {
    if (w < 2) {
        tdm_load(tdm_g0(buf + (uint32_t) (256 * w),
                        tdm_gaddr(Q, (int64_t) b * Sq + q0, (int64_t) H * lm::DQK, (int64_t) h * lm::DQK + 128 * w)),
                 w ? tdm_g1<T_W64>((int64_t) H * lm::DQK, 32) : tdm_g1<T_W128>((int64_t) H * lm::DQK, 32));
        tdm_load(tdm_g0(buf + (uint32_t) (LD_OFF + 128 * w),
                        tdm_gaddr(w ? (const void *) DEL : (const void *) NL, (int64_t) b * H + h, Sq, q0, 4)),
                 tdm_g1<T_LD>((int64_t) Sq, 1));
    } else {
        tdm_load(tdm_g0(buf + (uint32_t) (QT_B + 16 * (w - 2) * lm::PV),
                        tdm_gaddr(DO, (int64_t) b * Sq + q0 + 16 * (w - 2), (int64_t) H * lm::DV, (int64_t) h * lm::DV)),
                 tdm_g1<T_V16>((int64_t) H * lm::DV, 16));
    }
}

} // namespace

extern "C" __global__ __launch_bounds__(256) void k_mla_m10t_prep(const __bf16 *DO, const __bf16 *O, const float *LSE,
                                                                 float *DEL, float *NL, long long o_bytes,
                                                                 long long row_bytes, int B, int Sq, int H) {
    lb::prep(DO, O, LSE, DEL, NL, o_bytes, row_bytes, Sq, H);
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_mla_m10t_dkdv(
    const __bf16 *Q, const __bf16 *K, const __bf16 *V, const __bf16 *DO, const float *DEL, const float *NL,
    __bf16 *DK, __bf16 *DV, __bf16 *DS, long long q_bytes, long long k_bytes, long long v_bytes, long long o_bytes,
    long long row_bytes, long long ds_bytes, int B, int Sq, int Skv, int H, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[LDS_B];
    const int w = wave_id(), l = lane_id(), tid = (int) __builtin_amdgcn_workitem_id_x();
    const int nkb = Skv / KVB;
    const int wg  = (int) (__builtin_amdgcn_workgroup_id_x() +
                          nkb * (__builtin_amdgcn_workgroup_id_y() + H * __builtin_amdgcn_workgroup_id_z()));
    const int h = wg % H, kb = (wg / H) % nkb, b = wg / (H * nkb); // longest first
    const int kv0 = KVB * kb + 32 * w, cshift = Skv - Sq; // kv tile t of the wave: kv0 + 16 t

    const uint32_t lds0 = lds_addr(smem);
    auto buf = [&](int i) { return lds0 + (uint32_t) ((i % NSLOT) * BUF_B); }; // Q tile at +0, dO tile at +QT_B
    const rsrc_t   rk = make_rsrc(K, k_bytes), rv = make_rsrc(V, v_bytes);
    const rsrc_t   rds = ds_head(DS, b, H, h, Sq, Skv);

    // K / V rows of both kv tiles as B^T operands of S = Q K^T (6 k-steps) and dP = dO V^T (4): resident, lane = kv
    opnd_t kb_op[2][lm::NKQ], vb_op[2][lm::NKV];
    acc_t  dkt[2][lm::NTQ], dvt[2][lm::NTV]; // dK^T, dV^T of kv tile t: rows d, lane = kv
    static_for<2>([&](auto t) {
        lm::load_bt(kb_op[t], rk, lm::row<lm::DQK>(b, Skv, kv0 + 16 * t, H, h), H * lm::DQK * 2, l);
        lm::load_bt(vb_op[t], rv, lm::row<lm::DV>(b, Skv, kv0 + 16 * t, H, h), H * lm::DV * 2, l);
        static_for<lm::NTQ>([&](auto dt) { dkt[t][dt] = zero_acc(); });
        static_for<lm::NTV>([&](auto dt) { dvt[t][dt] = zero_acc(); });
    });

    // steps: 32-query blocks from the one holding query 128 kb - cshift (the first that sees kv row 128 kb)
    const int qf = KVB * kb - cshift, qp0 = causal == 0 || qf <= 0 ? 0 : qf / QS, nit = Sq / QS - qp0;
    if (nit > 0) issue_step(buf(0), Q, DO, NL, DEL, b, Sq, QS * qp0, H, h, w);
    if (nit > 1) issue_step(buf(1), Q, DO, NL, DEL, b, Sq, QS * (qp0 + 1), H, h, w);
    split_signal_tdm_lds<0>();
    split_wait_lds();
    // the readback set of a step: q tile 0's S / dP operands and the NL / delta rows of both q tiles (LDS)
    opnd_t qa0[lm::NKQ], da0[lm::NKV];
    acc_t  nl[2], del[2];
    auto   readback = [&](int i) {
        const lds_u8 *q_rows = rows_base<lm::PQ>(buf(i), l), *do_rows = rows_base<lm::PV>(buf(i) + QT_B, l);
        static_for<lm::NKQ>([&](auto k) { qa0[k] = load_rows<lm::PQ>(q_rows, 0, 32 * k); });
        static_for<lm::NKV>([&](auto k) { da0[k] = load_rows<lm::PV>(do_rows, 0, 32 * k); });
        // NL / delta, accumulator-row order: lane (r, h) gets element 16 qt + 8 h + i (one 32-B chunk)
        const lds_u8 *ld = lds_ptr(buf(i) + (uint32_t) (LD_OFF + 32 * (l / 16)));
        auto          row8 = [&](int off) {
            const f32x4 x = __builtin_bit_cast(f32x4, ds_load_b128_u32(ld, off));
            const f32x4 y = __builtin_bit_cast(f32x4, ds_load_b128_u32(ld, off + 16));
            return (acc_t) __builtin_shufflevector(x, y, 0, 1, 2, 3, 4, 5, 6, 7);
        };
        static_for<2>([&](auto qt) {
            nl[qt]  = row8(64 * qt);
            del[qt] = row8(128 + 64 * qt);
        });
    };
    if (nit > 0) readback(0);
    for (int it = 0; it < nit; ++it) {
        const int  q0   = QS * (qp0 + it);
        const bool next = it + 1 < nit; // workgroup-uniform
        const bool ahead = it + 2 < nit; // workgroup-uniform
        if (ahead) issue_step(buf(it + 2), Q, DO, NL, DEL, b, Sq, q0 + 2 * QS, H, h, w); // into step it - 1's slot
        const lds_u8 *q_rows = rows_base<lm::PQ>(buf(it), l), *do_rows = rows_base<lm::PV>(buf(it) + QT_B, l);
        const lds_u8 *q_tr = tr_base<lm::PQ>(buf(it), l), *do_tr = tr_base<lm::PV>(buf(it) + QT_B, l);

        // S = Q K^T (6 k-steps), dP = dO V^T (4): 2 q tiles x 2 kv tiles, lane = kv; q tile 0 from the readback,
        // q tile 1 from one burst issued before q tile 0's WMMAs
        acc_t  s[2][2], dp[2][2];
        opnd_t qa1[lm::NKQ], da1[lm::NKV];
        static_for<lm::NKQ>([&](auto k) { qa1[k] = load_rows<lm::PQ>(q_rows, 16, 32 * k); });
        static_for<lm::NKV>([&](auto k) { da1[k] = load_rows<lm::PV>(do_rows, 16, 32 * k); });
        sched_barrier<0>();
        auto sdp = [&](auto qt, const opnd_t (&qa)[lm::NKQ], const opnd_t (&da)[lm::NKV]) {
            static_for<2>([&](auto t) {
                static_for<lm::NKQ>([&](auto k) {
                    if constexpr (decltype(k)::value == 0) s[qt][t] = wmma0(qa[0], kb_op[t][0]);
                    else s[qt][t] = wmma(qa[k], kb_op[t][k], s[qt][t]);
                });
                static_for<lm::NKV>([&](auto k) {
                    if constexpr (decltype(k)::value == 0) dp[qt][t] = wmma0(da[0], vb_op[t][0]);
                    else dp[qt][t] = wmma(da[k], vb_op[t][k], dp[qt][t]);
                });
            });
        };
        sdp(std::integral_constant<int, 0>{}, qa0, da0);
        sched_barrier<0>();
        sdp(std::integral_constant<int, 1>{}, qa1, da1);
        sched_barrier<0>();
        // the dO^T / Q^T operands of dV / dK (tr16; the 32 queries are one chunk): one burst, flies during the softmax
        opnd_t tv[lm::NTV], tq[lm::NTQ];
        static_for<lm::NTV>([&](auto dt) { tv[dt] = load_tr<lm::PV>(do_tr, 0, 16 * dt); });
        static_for<lm::NTQ>([&](auto dt) { tq[dt] = load_tr<lm::PQ>(q_tr, 0, 16 * dt); });
        sched_barrier<0>();
        static_for<2>([&](auto qt) {
            static_for<2>([&](auto t) {
                const int q1 = q0 + 16 * qt, k1 = kv0 + 16 * t;
                s[qt][t]  = lb::probs_qk(s[qt][t], c2, nl[qt], l, causal != 0, q1 + cshift - k1);
                dp[qt][t] = lb::dsoft(s[qt][t], dp[qt][t], del[qt]);
                // dS^T rows k1 + r, columns q1 + 8 h .. + 8 (gmem.h buf_store_ct_bf16; RNE, the bits dK uses)
                buf_store_ct_bf16(rds, (k1 + l % 16) * Sq * 2 + (q1 + 8 * (l / 16)) * 2, 0, dp[qt][t]);
            });
        });
        // dV^T += dO^T P (8 d tiles), each operand feeding both kv tiles
        opnd_t sb[2];
        static_for<2>([&](auto t) {
            const opnd_t pb = pack_p_bwd(s[0][t], s[1][t]);
            sb[t]           = pack_p_bwd(dp[0][t], dp[1][t]);
            static_for<lm::NTV>([&](auto dt) { dvt[t][dt] = wmma(tv[dt], pb, dvt[t][dt]); });
        });
        if (next) { // publish step it + 1 (its buffer was last read a step earlier); every DS read of this step retired
            if (ahead) { // own op(s) of step it + 1 landed, step it + 2's may fly
                if (w < 2) split_signal_tdm_lds<2>();
                else split_signal_tdm_lds<1>();
            } else {
                split_signal_tdm_lds<0>();
            }
            split_wait_lds();
            readback(it + 1); // lands under the dK WMMAs
        }
        // dK^T += Q^T dS (12 d tiles), registers only
        static_for<2>([&](auto t) {
            static_for<lm::NTQ>([&](auto dt) { dkt[t][dt] = wmma(tq[dt], sb[t], dkt[t][dt]); });
        });
    }
    static_for<2>([&](auto t) {
        lm::store_rows(make_rsrc(DK, k_bytes), dkt[t], scale, lm::row<lm::DQK>(b, Skv, kv0 + 16 * t + l % 16, H, h), l);
        lm::store_rows(make_rsrc(DV, v_bytes), dvt[t], 1.0f, lm::row<lm::DV>(b, Skv, kv0 + 16 * t + l % 16, H, h), l);
    });
}

extern "C" __global__ __launch_bounds__(lf::BLOCK) void k_mla_m10t_dq(
    const __bf16 *DS, const __bf16 *K, __bf16 *DQ, long long k_bytes, long long q_bytes, long long ds_bytes, int B,
    int Sq, int Skv, int H, int causal, float c2, float scale) {
    __shared__ __attribute__((aligned(1024))) unsigned char smem[QLDS_B];
    const int w = wave_id(), l = lane_id();
    const int nqb = (Sq + DQ_QB - 1) / DQ_QB;
    const int wg  = (int) (__builtin_amdgcn_workgroup_id_x() +
                          nqb * (__builtin_amdgcn_workgroup_id_y() + H * __builtin_amdgcn_workgroup_id_z()));
    const int h = wg % H, qb = nqb - 1 - (wg / H) % nqb, b = wg / (H * nqb); // longest first
    const int p0 = DQ_QB * qb + DQ_QW * w;
    const int qcols = min(DQ_QB, Sq - DQ_QB * qb); // the workgroup's queries (a multiple of 64)

    const uint32_t lds0 = lds_addr(smem);
    auto buf = [&](int j) { return lds0 + (uint32_t) ((j & 1) * QBUF_B); }; // K tile at +0, dS^T block at +KT_B
    // block j: waves 0 / 1 move K rows [32 w, +32) (two ops: 128 + 64 columns), waves 2 / 3 dS^T rows
    // [32 (w - 2), +32) of kv block j, columns [256 qb, +qcols) (row stride Sq; the TDM zero-fills the rest)
    u32x8 ds_g1 = tdm_g1<T_DS>((int64_t) Sq, 32); // tensor_dim0 = the workgroup's columns, not T_DS::cols
    ds_g1[1] = ((uint32_t) qcols & 0xFFFFu) << 16;
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
            tdm_load(tdm_g0(buf(j) + (uint32_t) (KT_B + 32 * half * DS_P),
                            tdm_gaddr(ds_bh, 64 * j + 32 * half, Sq, DQ_QB * qb)),
                     ds_g1);
        }
    };
    acc_t dqt[4][lm::NTQ]; // dQ^T of q tile qt: rows d, lane = q
    static_for<4>([&](auto qt) { static_for<lm::NTQ>([&](auto dt) { dqt[qt][dt] = zero_acc(); }); });

    // kv blocks with an exported dS^T block: the workgroup's (its last 64-query block's) and this wave's (none past Sq)
    const bool live = p0 < Sq; // wave-uniform
    const int  n  = lf::n_kv_blocks(4 * qb + qcols / 64 - 1, Sq, Skv, causal);
    const int  nw = live ? lf::n_kv_blocks(4 * qb + w, Sq, Skv, causal) : 0;
    issue(0);
    split_signal_tdm_lds<0>();
    split_wait_lds();
    for (int j = 0; j < n; ++j) {
        const bool next = j + 1 < n; // workgroup-uniform
        if (next) issue(j + 1);
        if (j < nw) { // wave-uniform
            const lds_u8 *k_tr  = tr_base<lm::PQ>(buf(j), l);
            const lds_u8 *ds_tr = tr_base<DS_P>(buf(j) + (uint32_t) (KT_B + 2 * DQ_QW * w), l); // this wave's queries
            // dQ^T += K^T dS^T per 32-row kv chunk: A = K^T [d][kv] (tr16 of the K tile), B = dS^T [kv][q]
            static_for<2>([&](auto c) {
                opnd_t ka[lm::NTQ], sb[4];
                static_for<4>([&](auto qt) { sb[qt] = load_tr<DS_P>(ds_tr, 32 * c, 16 * qt); });
                static_for<lm::NTQ>([&](auto dt) { ka[dt] = load_tr<lm::PQ>(k_tr, 32 * c, 16 * dt); });
                sched_barrier<0>();
                static_for<lm::NTQ>([&](auto dt) {
                    static_for<4>([&](auto qt) { dqt[qt][dt] = wmma(ka[dt], sb[qt], dqt[qt][dt]); });
                });
                sched_barrier<0>();
            });
        }
        if (next) { // publish block j + 1 (its buffer was last read an iteration earlier)
            split_signal_tdm_lds<0>();
            split_wait_lds();
        }
    }
    if (live)
        static_for<4>([&](auto qt) {
            lm::store_rows(make_rsrc(DQ, q_bytes), dqt[qt], scale, lm::row<lm::DQK>(b, Sq, p0 + 16 * qt + l % 16, H, h), l);
        });
}
