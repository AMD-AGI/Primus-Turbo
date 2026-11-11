###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

import functools

import torch

from .flash_attn_fwd_mla import build_flash_attn_dualwave_swp_module


@functools.lru_cache(maxsize=64)
def _fwd_module(hq, hkv, d_qk, d_v, causal, emit_lse, block_m, waves_per_eu, stagger, k_head, rows_per_wave):
    return build_flash_attn_dualwave_swp_module(
        num_heads=hq,
        head_dim=d_qk,
        head_dim_v=d_v,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=hkv,
        varlen=False,
        cross_seqlen=False,
        emit_lse=emit_lse,
        window_left=-1,
        sbhd=True,
        has_sink=False,
        block_m=block_m,
        waves_per_eu=waves_per_eu,
        dualwave_swp_enable_stagger=stagger,
        k_head=k_head,
        rows_per_wave=rows_per_wave,
        # One wave per SIMD keeps the 8-wave barrier schedule: one rendezvous per KV tile.
        fold_barriers=True if rows_per_wave == 64 else None,
    )


def mla_fwd_sbhd(
    q,
    k,
    v,
    causal=True,
    return_lse=False,
    block_m=256,
    waves_per_eu=None,
    stagger=False,
    k_head=1,
    rows_per_wave=64,
):
    """q [Sq,B,Hq,Dqk], k [Skv,B,Hkv,Dqk], v [Skv,B,Hkv,Dv], contiguous bf16.

    Returns O [Sq,B,Hq,Dv] (and LSE [B*Sq,Hq] fp32, natural log). Softmax scale is 1/sqrt(Dqk).
    ``rows_per_wave`` 64 runs a 256-row q block on one wave per SIMD (512 registers each).
    """
    if waves_per_eu is None:
        waves_per_eu = 1 if rows_per_wave == 64 else 2
    Sq, B, Hq, d_qk = q.shape
    Skv, Bk, Hkv, d_k = k.shape
    d_v = v.shape[-1]
    assert Bk == B and d_k == d_qk and v.shape[:3] == k.shape[:3]
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert Sq == Skv, "prototype covers self-attention only"
    mod = _fwd_module(
        Hq,
        Hkv,
        d_qk,
        d_v,
        bool(causal),
        bool(return_lse),
        block_m,
        waves_per_eu,
        stagger,
        k_head,
        rows_per_wave,
    )
    out = q.new_empty((Sq, B, Hq, d_v))
    lse = torch.empty((B * Sq, Hq) if return_lse else (0,), device=q.device, dtype=torch.float32)
    kw = {
        "seq_len_kv": Skv,
        "stride_q_n": B * Hq * d_qk,
        "stride_kv_n": B * Hkv * d_qk,
        "stream": torch.cuda.current_stream(q.device),
    }
    if return_lse:
        kw["debug_counts"] = lse
    mod(q, k, v, out, B, Sq, **kw)
    return (out, lse) if return_lse else out
