/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace primus_turbo::hipkittens {

// HipKittens flash attention, gfx950 (CDNA4) only.
//
// One entry point per head dim, because each is a separately tuned kernel compiled from its own
// translation unit -- see the note at the top of each .hip source. Everything else is a runtime
// argument, so one build serves every batch, head count and sequence length.
//
// The caller is responsible for the envelope these kernels admit: bf16, SBHD and contiguous,
// bottom-right causal with an optional left window, Sq <= Skv, and Sq/Skv already padded up to
// the block sizes reported below. `primus_turbo/hipkittens/attention/gfx950/` enforces all of it
// on the Python side; the kernels themselves read out of bounds rather than failing.

void hk_attn_fwd_d64_blocks(int64_t *q_block, int64_t *kv_block);
void hk_attn_fwd_d128_blocks(int64_t *q_block, int64_t *kv_block);

void hk_attn_fwd_d64(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                     const at::Tensor &o, const at::Tensor &lse, int Sq, int Skv, int B, int Hq,
                     int Hkv, int window_left, float softmax_scale);

void hk_attn_fwd_d128(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                      const at::Tensor &o, const at::Tensor &lse, int Sq, int Skv, int B, int Hq,
                      int Hkv, int window_left, float softmax_scale);

// The backward runs three kernels with different tile shapes (prep, dq, dkdv) plus the fused
// single-pass variant, so it reports seven block sizes rather than two; the Python layer pads
// Sq/Skv to the lcm of them.
struct HkBwdBlocks {
    int64_t dq_q;
    int64_t dq_kv;
    int64_t dkdv_q;
    int64_t dkdv_kv;
    int64_t prep_q;
    int64_t fused_q;
    int64_t fused_kv;
};

void hk_attn_bwd_d64_blocks(HkBwdBlocks *out);
void hk_attn_bwd_d128_blocks(HkBwdBlocks *out);

// How many ways to split the GQA head group for dK/dV. The Python layer allocates the partials
// tensor, so it has to ask before launching; memoise on the shape.
int hk_attn_bwd_d64_dkdv_head_split(int Sq, int Skv, int B, int Hq, int Hkv, int window_left);
int hk_attn_bwd_d128_dkdv_head_split(int Sq, int Skv, int B, int Hq, int Hkv, int window_left);

// `wsk`/`wsv` are the dK/dV split-K partials, read only when the split the launcher picks is
// deeper than 1; pass dk/dv themselves when no workspace was allocated.
void hk_attn_bwd_d64(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                     const at::Tensor &o, const at::Tensor &dO, const at::Tensor &dq,
                     const at::Tensor &dk, const at::Tensor &dv, const at::Tensor &lse,
                     const at::Tensor &delta, const at::Tensor &lneg, const at::Tensor &wsk,
                     const at::Tensor &wsv, int Sq, int Skv, int B, int Hq, int Hkv,
                     int window_left, float softmax_scale, int n_split_req);

void hk_attn_bwd_d128(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                      const at::Tensor &o, const at::Tensor &dO, const at::Tensor &dq,
                      const at::Tensor &dk, const at::Tensor &dv, const at::Tensor &lse,
                      const at::Tensor &delta, const at::Tensor &lneg, const at::Tensor &wsk,
                      const at::Tensor &wsv, int Sq, int Skv, int B, int Hq, int Hkv,
                      int window_left, float softmax_scale, int n_split_req);

// The fused single-pass backward: one KV-outer pass emitting dQ as well, with the dQ
// contribution of each kv band going to the bf16 split-K workspace `ws`.
void hk_attn_bwd_fused_d64(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                           const at::Tensor &o, const at::Tensor &dO, const at::Tensor &dq,
                           const at::Tensor &dk, const at::Tensor &dv, const at::Tensor &ws,
                           const at::Tensor &lse, const at::Tensor &delta, int Sq, int Skv, int B,
                           int Hq, int Hkv, int window_left, float softmax_scale);

void hk_attn_bwd_fused_d128(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                            const at::Tensor &o, const at::Tensor &dO, const at::Tensor &dq,
                            const at::Tensor &dk, const at::Tensor &dv, const at::Tensor &ws,
                            const at::Tensor &lse, const at::Tensor &delta, int Sq, int Skv, int B,
                            int Hq, int Hkv, int window_left, float softmax_scale);

}  // namespace primus_turbo::hipkittens
