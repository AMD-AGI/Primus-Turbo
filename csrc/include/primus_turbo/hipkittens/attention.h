/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "primus_turbo/common.h"

namespace primus_turbo::hipkittens {

// HipKittens flash attention, gfx950 (CDNA4).
//
// Compiled into every arch configuration: each kernel body is guarded on __gfx950__ and
// asserts on any other device pass, so one multi-arch build can carry these sources. Nothing
// in the build stops them being CALLED on another card -- that is the Python layer's job.
//
// This is the kernel layer, so it takes raw pointers rather than tensors, like the rest of
// csrc/kernels. The torch-facing wrappers are in csrc/pytorch/attention/hk_attn.cpp.
//
// Every buffer is a contiguous 4-D extent described right-aligned as (d0, d1, d2, d3): SBHD
// for q/k/v/o and their gradients, [B, H, 1, S] for the fp32 per-row vectors. Callers must
// have padded the sequence axes to the block sizes reported below; the kernels read past a
// ragged tail rather than failing.
struct HkTensorDesc {
    const void *data;
    int d0;
    int d1;
    int d2;
    int d3;
};

// The tile sizes the caller must pad the sequence axes up to. Reported by the kernels rather
// than restated by the caller: a table that drifted from the build would pad to the wrong
// multiple and read past the end of a tensor instead of failing.
void hk_attn_fwd_d64_blocks(int64_t *q_block, int64_t *kv_block);
void hk_attn_fwd_d128_blocks(int64_t *q_block, int64_t *kv_block);

void hk_attn_fwd_d64(const HkTensorDesc &q, const HkTensorDesc &k, const HkTensorDesc &v,
                     const HkTensorDesc &o, const HkTensorDesc &lse, int Sq, int Skv, int B,
                     int Hq, int Hkv, int window_left, float softmax_scale);

void hk_attn_fwd_d128(const HkTensorDesc &q, const HkTensorDesc &k, const HkTensorDesc &v,
                      const HkTensorDesc &o, const HkTensorDesc &lse, int Sq, int Skv, int B,
                      int Hq, int Hkv, int window_left, float softmax_scale);

// The backward runs three kernels with different tile shapes (prep, dq, dkdv) plus the fused
// single-pass variant, so it reports seven block sizes rather than two; the caller pads to
// the lcm of them.
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

// How many ways to split the GQA head group for dK/dV. The caller allocates the partials
// tensor, so it has to ask before launching; memoise on the shape, since this simulates the
// per-XCD dispatch and a sub-millisecond kernel cannot pay for that on every launch.
int hk_attn_bwd_d64_dkdv_head_split(int Sq, int Skv, int B, int Hq, int Hkv, int window_left);
int hk_attn_bwd_d128_dkdv_head_split(int Sq, int Skv, int B, int Hq, int Hkv, int window_left);

// `wsk`/`wsv` are the dK/dV split-K partials, read only when n_split_req > 1; pass the dk/dv
// descriptors themselves when no workspace was allocated.
void hk_attn_bwd_d64(const HkTensorDesc &q, const HkTensorDesc &k, const HkTensorDesc &v,
                     const HkTensorDesc &o, const HkTensorDesc &dO, const HkTensorDesc &dq,
                     const HkTensorDesc &dk, const HkTensorDesc &dv, const HkTensorDesc &lse,
                     const HkTensorDesc &delta, const HkTensorDesc &lneg, const HkTensorDesc &wsk,
                     const HkTensorDesc &wsv, int Sq, int Skv, int B, int Hq, int Hkv,
                     int window_left, float softmax_scale, int n_split_req);

void hk_attn_bwd_d128(const HkTensorDesc &q, const HkTensorDesc &k, const HkTensorDesc &v,
                      const HkTensorDesc &o, const HkTensorDesc &dO, const HkTensorDesc &dq,
                      const HkTensorDesc &dk, const HkTensorDesc &dv, const HkTensorDesc &lse,
                      const HkTensorDesc &delta, const HkTensorDesc &lneg, const HkTensorDesc &wsk,
                      const HkTensorDesc &wsv, int Sq, int Skv, int B, int Hq, int Hkv,
                      int window_left, float softmax_scale, int n_split_req);

// The fused single-pass backward: one KV-outer pass emitting dQ as well, with the dQ
// contribution of each kv band going to the bf16 split-K workspace `ws`.
void hk_attn_bwd_fused_d64(const HkTensorDesc &q, const HkTensorDesc &k, const HkTensorDesc &v,
                           const HkTensorDesc &o, const HkTensorDesc &dO, const HkTensorDesc &dq,
                           const HkTensorDesc &dk, const HkTensorDesc &dv, const HkTensorDesc &ws,
                           const HkTensorDesc &lse, const HkTensorDesc &delta, int Sq, int Skv,
                           int B, int Hq, int Hkv, int window_left, float softmax_scale);

void hk_attn_bwd_fused_d128(const HkTensorDesc &q, const HkTensorDesc &k, const HkTensorDesc &v,
                            const HkTensorDesc &o, const HkTensorDesc &dO, const HkTensorDesc &dq,
                            const HkTensorDesc &dk, const HkTensorDesc &dv, const HkTensorDesc &ws,
                            const HkTensorDesc &lse, const HkTensorDesc &delta, int Sq, int Skv,
                            int B, int Hq, int Hkv, int window_left, float softmax_scale);

}  // namespace primus_turbo::hipkittens
