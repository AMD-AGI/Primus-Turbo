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

}  // namespace primus_turbo::hipkittens
