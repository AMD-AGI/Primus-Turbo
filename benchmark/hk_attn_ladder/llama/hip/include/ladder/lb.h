/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// ladder/lb.h -- what every rung below v0 of the Llama backward ladder (src/ladder/b0*.hip) shares: the
// kernel ABIs, constants, the prep kernel (delta, NL) and the row-vector helpers. Everything a rung
// changes lives in its own source.
//
// Problem: given q, k, v, o, do (bf16 BSHD, head dim 128, GQA hkv = hq / G) and lse [B, Hq, Sq] (natural
// log), dq, dk, dv of O = softmax(scale Q K^T) V, bottom-right causal (query p sees kv <= p + Skv - Sq).
//   P = exp2(s c2 + NL),  NL = -LSE log2 e,  c2 = scale log2 e       (the forward's probabilities)
//   dP = dO V^T,  delta = rowsum(dO o O),  dS = P o (dP - delta)
//   dV = P^T dO,  dK = scale dS^T Q,  dQ = scale dS K                  (dk / dv summed over the G heads)
// Masking follows hk_attn/frag.h: the exp2 ARGUMENT is masked after the fma with the finite MASK_NEG.
//
// Workspace (bwd_ladder.py): DEL [B, Hq, Sq] fp32 (delta), NL [B, Hq, Sq] fp32.
// Kernel ABIs (every rung):
//   prep (DO, O, LSE, DEL, NL, o_bytes, row_bytes, B, Sq, Hq)                  grid (Sq/64, Hq, B), block 256
//   dkdv (Q, K, V, DO, DEL, NL, DK, DV, q_bytes, kv_bytes, row_bytes, B, Sq, Skv, Hq, Hkv, causal, c2, scale)
//   dq   (Q, K, V, DO, DEL, NL, DQ, q_bytes, kv_bytes, row_bytes, B, Sq, Skv, Hq, Hkv, causal, c2, scale)
// fused rungs (b04 on): dkdv (Q, K, V, DO, DEL, NL, DK, DV, DQA, q_bytes, kv_bytes, row_bytes, dqa_bytes, B, Sq, Skv,
//   Hq, Hkv, causal, c2, scale) adds dQ_acc (fp32 [B, Sq, Hq, 128], zeroed by the host); cvt (DQA, DQ, dqa_bytes,
//   q_bytes, scale) replaces dq.
// row_bytes = byte extent of DEL / NL. Host gate (lab/bwd_ladder.py): d = 128, Sq % 64 == 0, Skv % 64 (b05 on: 128)
// == 0, causal -> Sq <= Skv, Hq % Hkv == 0 (b03: G % 4 == 0), every byte extent incl. dQ_acc < 2^31.
#pragma once
#include "ladder/lf.h"

namespace lb {

using namespace hk_attn;
using lf::D;
using lf::NDK;
using lf::NDT;

constexpr float LOG2E = 1.4426950408889634f;

// rows (q) of a 64-row block a 64-row kv block kb is visible to: causal -> from the q block holding
// kv = 64 kb - (Skv - Sq) on (q >= kv - cshift); full -> all
HKA_DEV int first_q_block(int kb, int Sq, int Skv, int causal) {
    if (causal == 0)
        return 0;
    const int q0 = 64 * kb - (Skv - Sq); // first query that sees kv row 64 kb
    return q0 <= 0 ? 0 : q0 / 64;
}

// ---- prep: delta = rowsum(dO o O) (fp32), NL = -LSE log2 e ------------------------------------------------
// 256 threads = 64 query rows x 4 threads; a thread reduces 32 d (4 x 16 B of dO and O), the 4 threads of
// a row combine with two DPP xor steps (lanes 4r .. 4r + 3 of a 16-lane row)
HKA_DEV void prep(const __bf16 *DO, const __bf16 *O, const float *LSE, float *DEL, float *NL, long long o_bytes,
                  long long row_bytes, int Sq, int Hq) {
    const int    tid = (int) __builtin_amdgcn_workitem_id_x();
    const int    qb = (int) __builtin_amdgcn_workgroup_id_x(), hq = (int) __builtin_amdgcn_workgroup_id_y();
    const int    b = (int) __builtin_amdgcn_workgroup_id_z();
    const int    p = 64 * qb + tid / 4, part = tid % 4;
    const rsrc_t ro = make_rsrc(O, o_bytes), rd = make_rsrc(DO, o_bytes);
    const int    vo = lf::row_b(b, Sq, p, Hq, hq) + 64 * part;
    float        acc = 0.f;
    static_for<4>([&](auto j) {
        const bf16x8 a = buf_load_bf16x8(rd, vo + 16 * j, 0), c = buf_load_bf16x8(ro, vo + 16 * j, 0);
        static_for<8>([&](auto e) { acc += (float) a[(int) e] * (float) c[(int) e]; });
    });
    acc += xor16<1>(acc);
    acc += xor16<2>(acc);
    if (part == 0) {
        const rsrc_t rl = make_rsrc(LSE, row_bytes), rdel = make_rsrc(DEL, row_bytes), rnl = make_rsrc(NL, row_bytes);
        const int    vr = lf::lse_b(b, Hq, hq, Sq, p);
        buf_store_f32(rdel, vr, 0, acc);
        buf_store_f32(rnl, vr, 0, -buf_load_f32(rl, vr, 0) * LOG2E);
    }
}

// ---- row vectors in accumulator-row order (dkdv: S = Q K^T, rows = q) -------------------------------------
// lane (r, h) gets X[p0 + 8 h + i], i < 8 (gmem.h load_rows8_f32) for the 16-row q tile at p0
HKA_DEV acc_t rows8(rsrc_t r, int b, int Hq, int hq, int Sq, int p0, int lane) {
    return load_rows8_f32(r, lf::lse_b(b, Hq, hq, Sq, p0), 0, lane);
}

// p = exp2(fma(s, c2, nl)) with the causal mask on the ARGUMENT (finite MASK_NEG, frag.h): S = Q K^T tiles
// (rows q from m0, cols kv from n0): masked iff kv > q + cshift <=> col > row + off, off = m0 + cshift - n0
HKA_DEV acc_t probs_qk(acc_t s, float c2, acc_t nl, int lane, bool masked, int off) {
    acc_t t;
    static_for<8>([&](auto i) { t[(int) i] = __builtin_fmaf(s[(int) i], c2, nl[(int) i]); });
    if (masked)
        t = mask_col_gt_row(t, lane, off, MASK_NEG);
    static_for<8>([&](auto i) { t[(int) i] = exp2_hw(t[(int) i]); });
    return t;
}
// the same on S^T = K Q^T tiles (rows kv from m0, cols q from n0, one nl per lane): masked iff row > col + off,
// off = n0 + cshift - m0
HKA_DEV acc_t probs_kq(acc_t s, float c2, float nl, int lane, bool masked, int off) {
    acc_t t;
    static_for<8>([&](auto i) { t[(int) i] = __builtin_fmaf(s[(int) i], c2, nl); });
    if (masked)
        t = mask_row_gt_col(t, lane, off, MASK_NEG);
    static_for<8>([&](auto i) { t[(int) i] = exp2_hw(t[(int) i]); });
    return t;
}

// dS = P o (dP - delta), delta per accumulator element (rows) or per lane
HKA_DEV acc_t dsoft(acc_t p, acc_t dp, acc_t del) {
    acc_t r;
    static_for<8>([&](auto i) { r[(int) i] = p[(int) i] * (dp[(int) i] - del[(int) i]); });
    return r;
}
HKA_DEV acc_t dsoft(acc_t p, acc_t dp, float del) {
    acc_t r;
    static_for<8>([&](auto i) { r[(int) i] = p[(int) i] * (dp[(int) i] - del); });
    return r;
}

// accumulator (rows d = 16 dt + 8 h + i, lane = row of the BSHD tensor) * mul -> bf16 row chunk
HKA_DEV void store_rows(rsrc_t r, const acc_t (&acc)[NDT], float mul, int row0_b, int lane) {
    const int vo = row0_b + (lane / 16) * 16;
    static_for<NDT>([&](auto dt) { buf_store_ct_bf16(r, vo + 32 * dt, 0, acc[dt] * mul); });
}

} // namespace lb
