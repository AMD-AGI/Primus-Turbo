/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// ladder/lf.h -- what every rung below v0 of the Llama forward ladder (src/ladder/f0*.hip) shares:
// the kernel ABI, the problem constants, the global offsets, the Q^T operand load straight from
// global memory and the epilogue. Everything a rung changes lives in its own source.
//
// Problem: O[b, p, hq, :] = softmax_kv(scale Q K^T) V, LSE[b, hq, p] = ln sum_kv exp(scale s), bf16
// BSHD, head dim 128, GQA (hkv = hq / G), bottom-right causal (query p sees kv <= p + Skv - Sq) or
// full. The math is v0's: S^T = K Q^T (lane = query, rows = kv), column softmax in the log2 domain
// (c2 = scale * log2 e), P^T packed from the S^T accumulators straight into the B operand of
// O^T += V^T P^T (hk_attn/wmma.h "fwd" orientations).
//
// Kernel ABI of every rung (lab/fwd_ladder.py):
//   (Q, K, V, O, LSE, q_bytes, kv_bytes, o_bytes, lse_bytes, B, Sq, Skv, Hq, Hkv, causal, c2)
// Host gate (fwd_ladder.py unsupported_reason): d = 128, Sq % 64 == 0, Skv % 64 == 0, causal ->
// Sq <= Skv, Hq % Hkv == 0 (head-packed rungs: G % 4 == 0), every byte extent < 2^31 (int32 buffer
// offsets). All global accesses are buffer accesses with the tensor's true byte extent (an access
// at or past it reads 0 / writes nothing) or TDM ops with true tensor extents.
#pragma once
#include "hk_attn/ext_fwd.h"
#include "hk_attn/prims.h"

namespace lf {

using namespace hk_attn;

constexpr int D     = 128;    // head dim
constexpr int NWAVE = 4;      // waves per workgroup
constexpr int BLOCK = NWAVE * 32;
constexpr int QB    = 64;     // query positions per workgroup
constexpr int QW    = 16;     // query positions per wave (one 16-wide q tile per head)
constexpr int KVB   = 64;     // kv rows per block (one LDS stage)
constexpr int NDK   = D / 32; // 4: WMMA k-steps of S^T = K Q^T
constexpr int NDT   = D / 16; // 8: 16-wide d tiles of O^T
constexpr int NKT   = KVB / 16; // 4: 16-row kv tiles of S^T per block
constexpr int NKC   = KVB / 32; // 2: 32-row kv chunks (the k depth of PV) per block

// LDS row pitches (lds.h): K is read with the rows loader (odd multiple of 16 B), V with tr16 (odd
// multiple of 32 B); v0's K_PITCH / V_PITCH
constexpr int K_PITCH = 2 * D + 16;   // 272
constexpr int V_PITCH = 2 * D + 32;   // 288
constexpr int K_TILE_B = KVB * K_PITCH; // 17408
constexpr int V_TILE_B = KVB * V_PITCH; // 18432

constexpr float LN2     = 0.6931471805599453f;
constexpr float BIG_NEG = -1.0e30f; // running-max seed (finite: exp2 of (-inf) - BIG_NEG is 0, no NaN)

struct Args {
    const __bf16 *Q;
    const __bf16 *K;
    const __bf16 *V;
    __bf16       *O;
    float        *LSE;
    long long     q_bytes, kv_bytes, o_bytes, lse_bytes;
    int           B, Sq, Skv, Hq, Hkv, causal;
    float         c2;
};

// ---- global byte offsets (int32: the host gate keeps every extent < 2^31) ------------------------
// row (b, p) of a [B, S, H, D] tensor, head h: element ((b S + p) H + h) D
HKA_DEV int row_b(int b, int S, int p, int H, int h) {
    return (((b * S + p) * H) + h) * D * 2;
}
HKA_DEV int lse_b(int b, int Hq, int hq, int Sq, int p) {
    return ((b * Hq + hq) * Sq + p) * 4;
}

// kv blocks a query block sees: causal -> through the block holding kv = last query + Skv - Sq
HKA_DEV int n_kv_blocks(int qb, int Sq, int Skv, int causal) {
    return causal != 0 ? qb + 1 + (Skv - Sq) / KVB : Skv / KVB;
}

// ---- Q^T B operands straight from global (once per wave; no LDS) ---------------------------------
// B[k = d][n = q] of S^T = K Q^T: lane (r, h) element e = Q[p0 + r][32 k + op_k(l, e)], two 16-B
// loads per operand (d = 32 k + 8 h + [0, 8) and + 16). row0_b: byte offset of (b, p0, hq, d 0);
// pitch_b: bytes between consecutive positions (Hq * D * 2).
HKA_DEV void load_qt(opnd_t (&qt)[NDK], rsrc_t rq, int row0_b, int pitch_b, int lane) {
    const int vo = row0_b + (lane % 16) * pitch_b + (lane / 16) * 16;
    static_for<NDK>([&](auto k) {
        qt[k] = cat(buf_load_bf16x8(rq, vo + 64 * k, 0), buf_load_bf16x8(rq, vo + 64 * k + 32, 0));
    });
}

// ---- softmax pieces on S^T accumulators (lane = query, rows = kv) ---------------------------------
// column max over the 16 * NT rows of this query (in-lane tree + the other lane half)
template <int NT> HKA_DEV float qmax(const acc_t (&s)[NT]) {
    return col_max(s);
}
// p = exp2(s c2 - m2) in place (m2: running max in the log2 domain); returns this lane's partial sum
template <int NT> HKA_DEV float exp_tiles(acc_t (&s)[NT], float c2, float m2) {
    static_for<NT>([&](auto t) { s[t] = exp2_tile(s[t], c2, -m2); });
    return lane_sum(s);
}

// ---- epilogue: O = O^T / l as bf16 rows, LSE (natural log) ---------------------------------------
// oacc[dt]: rows d = 16 dt + 8 h + i, column q = lane % 16. lsum: this lane's partial row sum (the
// other lane half holds the rest of the same query). m2: running max (log2 domain).
HKA_DEV void store_out(const Args &a, const acc_t (&oacc)[NDT], float lsum, float m2, int b, int hq,
                       int p0, int lane) {
    const float  lt  = lsum + xhalf(lsum);
    const float  inv = lt > 0.f ? rcp_ieee(lt) : 0.f;
    const rsrc_t ro  = make_rsrc(a.O, a.o_bytes);
    const int    vo  = row_b(b, a.Sq, p0 + lane % 16, a.Hq, hq) + (lane / 16) * 16;
    static_for<NDT>([&](auto dt) { buf_store_ct_bf16(ro, vo + 32 * dt, 0, oacc[dt] * inv); });
    const rsrc_t rl = make_rsrc(a.LSE, a.lse_bytes);
    store_cols16_f32(rl, lse_b(b, a.Hq, hq, a.Sq, p0), 0, (m2 + log2_hw(lt)) * LN2, lane);
}

} // namespace lf
