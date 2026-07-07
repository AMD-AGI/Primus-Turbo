###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL-v1 sparse-MLA forward (native FlyDSL MFMA) launcher.

Public API mirrors :func:`sparse_mla_fwd_v4_gluon` / ``sparse_mla_fwd_v4_triton``:

    sparse_mla_fwd_v4_flydsl(q, kv, topk_indices, attn_sink=None,
                             kv_lora_rank=512, scale=None) -> (o, lse)

The heavy lifting is the native FlyDSL MFMA kernel in
``dsa_fwd_v4_flydsl_kernel.build_dsa_fwd_module``; this module just marshals the
tensors, caches the built launcher per (H, D, TOPK, has_sink, block_n), and
launches it. See the kernel module docstring for the design.
"""

from __future__ import annotations

import os
import threading

import torch

# flydsl_v1 uses only the installed `flydsl` pip package + its own local kernel
# modules — it does NOT need the /workspace/FlyDSL-amd source tree.
# The pipe kernel is a superset of dsa_fwd_kernel: it adds an async global->LDS DMA
# gather (rocdl.raw_ptr_buffer_load_lds, bypassing the VGPR staging buffer) gated by
# `use_dma`. DMA is a bit-identical +6% on the dual (H>=128) path and is default-ON
# there; env PRIMUS_DSA_FLYDSL_FWD_DMA=0 falls back to the VGPR-staged gather.
from .dsa_fwd_pipe_kernel import build_dsa_fwd_module  # noqa: E402
from .dsa_fwd_m16_kernel import build_dsa_fwd_m16_module  # noqa: E402
from .dsa_fwd_tr16_kernel import build_dsa_fwd_tr16_module  # noqa: E402

_KERNEL_CACHE = {}
_KERNEL_CACHE_LOCK = threading.Lock()
_M16_CACHE = {}
_M16_LOCK = threading.Lock()
_TR16_CACHE = {}
_TR16_LOCK = threading.Lock()

# M=16 16x16x32 QK + ds_read_tr PV (path B): combines occupancy-2 with the fast
# hardware-transpose PV pipeline. Aims to beat gluon_v2. Env-gated for A/B.
_USE_TR16 = os.environ.get("PRIMUS_DSA_FLYDSL_FWD_TR16", "0") == "1"


def _get_tr16_kernel(num_heads, kv_lora_rank, d_qk, topk, has_sink, scale):
    key = (num_heads, kv_lora_rank, d_qk, topk, has_sink, round(float(scale), 8))
    with _TR16_LOCK:
        launch = _TR16_CACHE.get(key)
        if launch is None:
            launch = build_dsa_fwd_tr16_module(
                num_heads=num_heads, kv_lora_rank=kv_lora_rank, d_qk=d_qk,
                topk=topk, sm_scale=float(scale), has_sink=has_sink,
            )
            _TR16_CACHE[key] = launch
        return launch

# M=16 (16x16x32) forward: occupancy-2, faster than M=32 on RANDOM topk (205 vs
# 190) but SLOWER on the realistic [window++pool] topk the adapter builds (162 vs
# 197 end-to-end) — the M=32 kernel's ds_read_tr V pipeline handles the
# structured-gather locality better. Default OFF; env-gated for experiments.
_USE_M16 = os.environ.get("PRIMUS_DSA_FLYDSL_FWD_M16", "0") == "1"


def _get_m16_kernel(num_heads, kv_lora_rank, d_qk, topk, has_sink, scale):
    key = (num_heads, kv_lora_rank, d_qk, topk, has_sink, round(float(scale), 8))
    with _M16_LOCK:
        launch = _M16_CACHE.get(key)
        if launch is None:
            launch = build_dsa_fwd_m16_module(
                num_heads=num_heads, kv_lora_rank=kv_lora_rank, d_qk=d_qk,
                topk=topk, sm_scale=float(scale), has_sink=has_sink,
            )
            _M16_CACHE[key] = launch
        return launch

_DEFAULT_BLOCK_N = int(os.environ.get("PRIMUS_DSA_FLYDSL_FWD_BLOCK_N", "0"))  # 0 = shape-conditional
_DEFAULT_BLOCK_H = int(os.environ.get("PRIMUS_DSA_FLYDSL_FWD_BLOCK_H", "256"))
_DEFAULT_WPE = int(os.environ.get("PRIMUS_DSA_FLYDSL_FWD_WPE", "2"))


# Async DMA gather (buffer_load_lds): +6% bit-identical on the dual (H>=128) path.
# Only valid when single_latent is off (needs the unpadded dual V layout). Env
# override: 0 forces the VGPR-staged gather, 1 forces DMA even where auto is off.
_DMA_ENV = os.environ.get("PRIMUS_DSA_FLYDSL_FWD_DMA", "")


def _get_kernel(
    num_heads, kv_lora_rank, d_qk, topk, has_sink, block_n, block_h, single_latent, waves_per_eu, scale
):
    block_h = min(block_h, num_heads)
    while num_heads % block_h != 0:
        block_h -= 32
    if _DMA_ENV == "1":
        use_dma = True
    elif _DMA_ENV == "0":
        use_dma = False
    else:
        use_dma = True  # auto: DMA on both dual (H>=128) and single_latent (H<=64)
    key = (
        num_heads,
        kv_lora_rank,
        d_qk,
        topk,
        has_sink,
        block_n,
        block_h,
        single_latent,
        waves_per_eu,
        use_dma,
        round(float(scale), 8),
    )
    with _KERNEL_CACHE_LOCK:
        launch = _KERNEL_CACHE.get(key)
        if launch is None:
            launch = build_dsa_fwd_module(
                num_heads=num_heads,
                kv_lora_rank=kv_lora_rank,
                d_qk=d_qk,
                topk=topk,
                dtype_str="bf16",
                sm_scale=float(scale),
                has_sink=has_sink,
                block_n=block_n,
                block_h=block_h,
                single_latent=single_latent,
                waves_per_eu=waves_per_eu,
                use_dma=use_dma,
            )
            _KERNEL_CACHE[key] = launch
        return launch


def sparse_mla_fwd_v4_flydsl(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA forward (native FlyDSL MFMA, gfx950)."""
    assert q.is_contiguous() and topk_indices.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    kv_lora_rank = int(kv_lora_rank)
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    assert kv.shape[0] >= total_tokens and kv.shape[-1] == d_qk
    assert q.dtype == torch.bfloat16, "bf16 only"

    # Shape-conditional tile: flash (H<=64) is LDS/occupancy-bound -> small tile
    # (BLOCK_N=32) doubles occupancy (occ 1->2). pro (H>=128) is VGPR-bound
    # (occupancy stuck at 1 regardless of LDS) -> a smaller tile only adds
    # barrier/softmax overhead, so keep the larger BLOCK_N=64 (fewer tiles).
    # flash (H<=64) is LDS/occupancy-bound → single-latent (one shared tile) halves
    # LDS (occ 1→2) at BLOCK_N=64 (single-latent already halves LDS, so no need for
    # a smaller tile — smaller tiles only add barriers). pro (H>=128) is VGPR-bound
    # → dual swizzled tile (conflict-free QK), extra LDS is free (occ stuck at 1).
    single_latent = num_heads <= 64
    block_n = _DEFAULT_BLOCK_N if _DEFAULT_BLOCK_N else 64
    topk = topk_indices.shape[1]
    # Pad TOPK up to a multiple of block_n with -1 (masked) if needed.
    if topk % block_n != 0:
        pad = ((topk + block_n - 1) // block_n) * block_n - topk
        topk_indices = torch.cat(
            [topk_indices, torch.full((total_tokens, pad), -1, dtype=torch.int32, device=q.device)],
            dim=1,
        ).contiguous()
        topk = topk_indices.shape[1]

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.is_contiguous() and attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_heads,)
        sink = attn_sink
    else:
        # kernel always folds the sink; -inf makes it a no-op (af=1, sink_e=0)
        sink = torch.full((num_heads,), float("-inf"), dtype=torch.float32, device=q.device)

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    if (os.environ.get("PRIMUS_DSA_FLYDSL_FWD_TR16", "0") == "1") and (num_heads % 16 == 0) and (topk % 32 == 0):
        launch = _get_tr16_kernel(int(num_heads), kv_lora_rank, int(d_qk), int(topk), has_sink, scale)
        launch(
            q.reshape(-1), kv.reshape(-1), topk_indices.reshape(-1), sink.reshape(-1),
            o.reshape(-1), lse.reshape(-1), int(total_tokens),
        )
        return o, lse

    if _USE_M16 and (num_heads % 16 == 0) and (topk % 32 == 0):
        launch = _get_m16_kernel(int(num_heads), kv_lora_rank, int(d_qk), int(topk), has_sink, scale)
        launch(
            q.reshape(-1), kv.reshape(-1), topk_indices.reshape(-1), sink.reshape(-1),
            o.reshape(-1), lse.reshape(-1), int(total_tokens),
        )
        return o, lse

    launch = _get_kernel(
        int(num_heads),
        kv_lora_rank,
        int(d_qk),
        int(topk),
        has_sink,
        block_n,
        _DEFAULT_BLOCK_H,
        single_latent,
        _DEFAULT_WPE,
        scale,
    )
    launch(
        q.reshape(-1),
        kv.reshape(-1),
        topk_indices.reshape(-1),
        sink.reshape(-1),
        o.reshape(-1),
        lse.reshape(-1),
        int(total_tokens),
    )
    return o, lse


__all__ = ["sparse_mla_fwd_v4_flydsl"]
