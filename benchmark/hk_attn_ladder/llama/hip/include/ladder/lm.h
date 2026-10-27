/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// ladder/lm.h -- the DeepSeek-V3 MLA backward rungs below hk_v0 (src/ladder/m0*.hip, MLA ladder): lb.h's
// FlashAttention-2 backward with the MLA head dims, qk 192 / v 128, MHA (one kv head per q head). The
// kernels see BSHD [B, S, H, D] tensors; the launcher (mla/lab/bwd_mladder.py) hands Megatron's SBHD over
// as [1, S, B*H, D] (the production fold, no copy).
//
// Kernel ABIs (every m rung; byte extents of every tensor, DEL / NL [B, H, Sq] fp32 from the prep kernel):
//   prep (DO, O, LSE, DEL, NL, o_bytes, row_bytes, B, Sq, H)                            lb.h's prep (dv 128)
//   dkdv (Q, K, V, DO, DEL, NL, DK, DV, q_bytes, k_bytes, v_bytes, o_bytes, row_bytes, B, Sq, Skv, H, causal,
//         c2, scale)
//   dq   (Q, K, V, DO, DEL, NL, DQ, q_bytes, k_bytes, v_bytes, o_bytes, row_bytes, B, Sq, Skv, H, causal, c2,
//         scale)
#pragma once
#include "ladder/lb.h"

namespace lm {

using namespace hk_attn;

constexpr int DQK = 192, DV = 128;
constexpr int NKQ = DQK / 32, NKV = DV / 32; // WMMA k-steps of S (over 192) and dP (over 128)
constexpr int NTQ = DQK / 16, NTV = DV / 16; // 16-wide d tiles of dQ^T / dK^T (12) and dV^T (8)
constexpr int PQ = 2 * DQK + 16;             // 400: LDS pitch of 192-wide tiles (rows and tr16 reads)
constexpr int PV = 2 * DV + 16;              // 272: LDS pitch of 128-wide tiles

// byte offset of (b, p, h, d 0) in a [B, S, H, W] bf16 tensor (int32: extents < 2^31 by the host gate)
template <int W> HKA_DEV int row(int b, int S, int p, int H, int h) {
    return (((b * S + p) * H) + h) * W * 2;
}

// rows of a W-wide tensor as B^T operands (B[k = d][n = row], lane = row): NK k-steps of 32 (lf::load_qt for any W)
template <int NK> HKA_DEV void load_bt(opnd_t (&op)[NK], rsrc_t r, int row0_b, int pitch_b, int lane) {
    const int vo = row0_b + (lane % 16) * pitch_b + (lane / 16) * 16;
    static_for<NK>([&](auto k) {
        op[k] = cat(buf_load_bf16x8(r, vo + 64 * k, 0), buf_load_bf16x8(r, vo + 64 * k + 32, 0));
    });
}

// global -> VGPR -> LDS of a 64-row tile of W bf16 per row (W / 8 chunks of 16 B per row; 128 threads)
template <int W, int PITCH> HKA_DEV void fill_tile(lds_u8 *dst, rsrc_t r, int row0_b, int stride_b, int tid) {
    constexpr int CH = W / 8, N = 64 * CH / 128; // chunks per row, chunks per thread
    static_assert((64 * CH) % 128 == 0, "whole chunks per thread");
    u32x4 v[N];
    static_for<N>([&](auto j) {
        const int i = tid + 128 * j;
        v[j]        = buf_load_b128(r, row0_b + (i / CH) * stride_b + 16 * (i % CH), 0);
    });
    static_for<N>([&](auto j) {
        const int i = tid + 128 * j;
        ds_store_b128(dst, (i / CH) * PITCH + 16 * (i % CH), v[j]);
    });
}

// accumulator tiles (rows d = 16 dt + 8 h + i, lane = row of the tensor) * mul -> bf16 row chunks
template <int NT> HKA_DEV void store_rows(rsrc_t r, const acc_t (&acc)[NT], float mul, int row_b, int lane) {
    const int vo = row_b + (lane / 16) * 16;
    static_for<NT>([&](auto dt) { buf_store_ct_bf16(r, vo + 32 * dt, 0, acc[dt] * mul); });
}

} // namespace lm
