###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL-v1 sparse-MLA backward — native FlyDSL dQ + shared dKV gather.

``sparse_mla_bwd_v4_flydsl(q, kv, o, do, topk, lse, ...) -> (dq, dkv, d_sink)``.

The **dQ** kernel is native FlyDSL MFMA (``dsa_bwd_dq_flydsl_kernel``): it
recomputes S = Q·Kᵀ, P = exp(scale·S − lse), dP = dO·Kᵀ, dS = P·(dP − Δ)·scale,
accumulates dQ = dS·K, and writes the per-token dS/P buffers. The dKV path
(intermediate GEMM + CSR inverted-topk scatter-reduce) reuses the shared,
proven Triton kernels — the dKV gather is a variable-length scatter-reduction
with no MFMA content, so there is nothing to gain from a FlyDSL rewrite there.

This runs the whole top-k as a single chunk (R_CHUNK = TOPK), so dQ needs no
cross-chunk read-modify-write and the dKV intermediate is built in one pass.

Depends only on the installed ``flydsl`` pip package (no /workspace source).
"""

from __future__ import annotations

import os
import threading

import torch
import triton

# The dKV-intermediate + CSR gather kernels are shared with the ported triton_v2
# backend (primus_turbo.triton.attention.deepseek.sparse_mla_v2).
from primus_turbo.triton.attention.deepseek.sparse_mla_v2._csr_helper import (
    _build_inverted_topk_slice,
)
from primus_turbo.triton.attention.deepseek.sparse_mla_v2.dsa_bwd_kernels import (
    _bwd_compute_dkv_intermediate,
    _bwd_dkv_gather_acc,
)

from .dsa_bwd_dq_kernel import build_dsa_bwd_dq_module  # noqa: E402
from .dsa_bwd_dq_m16_kernel import build_dsa_bwd_dq_m16_module  # noqa: E402
from .dsa_bwd_dkv_interm_kernel import build_dsa_bwd_dkv_interm_module  # noqa: E402

_DQ_CACHE = {}
_DQ_LOCK = threading.Lock()
_DQ_M16_CACHE = {}
_DQ_M16_LOCK = threading.Lock()
_BLOCK_N = int(os.environ.get("PRIMUS_DSA_FLYDSL_BWD_BLOCK_N", "64"))
_BLOCK_H = int(os.environ.get("PRIMUS_DSA_FLYDSL_BWD_BLOCK_H", "64"))

# M=16 dQ kernel (tr16-style: shared gather + 16x16x32 + ds_read_tr dQ). Cuts the
# M=32 dQ kernel's 3-accumulator VGPR blowup (512 VGPR / 140-196 spill). Env-gated.
_USE_DQ_M16 = os.environ.get("PRIMUS_DSA_FLYDSL_BWD_DQ_M16", "0") == "1"


def _get_dq_m16_kernel(num_heads, kv_lora_rank, d_qk, topk, scale):
    key = (num_heads, kv_lora_rank, d_qk, topk, round(float(scale), 8))
    with _DQ_M16_LOCK:
        launch = _DQ_M16_CACHE.get(key)
        if launch is None:
            launch = build_dsa_bwd_dq_m16_module(
                num_heads=num_heads, kv_lora_rank=kv_lora_rank, d_qk=d_qk,
                topk=topk, dtype_str="bf16", sm_scale=float(scale),
            )
            _DQ_M16_CACHE[key] = launch
        return launch


_INTERM_CACHE = {}
_INTERM_LOCK = threading.Lock()


def _get_interm_kernel(num_heads, kv_lora_rank, d_qk, topk):
    key = (num_heads, kv_lora_rank, d_qk, topk)
    with _INTERM_LOCK:
        launch = _INTERM_CACHE.get(key)
        if launch is None:
            launch = build_dsa_bwd_dkv_interm_module(
                num_heads=num_heads, kv_lora_rank=kv_lora_rank, d_qk=d_qk,
                topk=topk, dtype_str="bf16",
            )
            _INTERM_CACHE[key] = launch
        return launch


def _get_dq_kernel(num_heads, kv_lora_rank, d_qk, topk, single_latent, scale):
    block_h = min(_BLOCK_H, num_heads)
    while num_heads % block_h != 0:
        block_h -= 32
    key = (num_heads, kv_lora_rank, d_qk, topk, block_h, single_latent, round(float(scale), 8))
    with _DQ_LOCK:
        launch = _DQ_CACHE.get(key)
        if launch is None:
            launch = build_dsa_bwd_dq_module(
                num_heads=num_heads,
                kv_lora_rank=kv_lora_rank,
                d_qk=d_qk,
                topk=topk,
                dtype_str="bf16",
                sm_scale=float(scale),
                block_n=_BLOCK_N,
                block_h=block_h,
                single_latent=single_latent,
            )
            _DQ_CACHE[key] = launch
        return launch


def sparse_mla_bwd_v4_flydsl(q, kv, o, do, topk_indices, lse, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA backward: native FlyDSL dQ + Triton dKV gather."""
    assert q.is_contiguous() and o.is_contiguous() and do.is_contiguous()
    assert topk_indices.is_contiguous() and lse.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    D = int(kv_lora_rank)
    rope_rank = d_qk - D
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    num_kv = kv.shape[0]
    assert q.dtype == torch.bfloat16

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # pad topk to a BLOCK_N multiple (-1 = invalid); usually already padded upstream
    topk = topk_indices.shape[1]
    if topk % _BLOCK_N != 0:
        pad = ((topk + _BLOCK_N - 1) // _BLOCK_N) * _BLOCK_N - topk
        topk_p = torch.cat(
            [topk_indices, torch.full((total_tokens, pad), -1, dtype=torch.int32, device=q.device)], dim=1
        ).contiguous()
    else:
        topk_p = topk_indices
    TOPK = topk_p.shape[1]

    # Delta = rowsum(O * dO)  (o is [T,H,D])
    delta = (o[:, :, :D].float() * do.float()).sum(-1).contiguous()
    lse_c = lse.contiguous()

    # ---- native FlyDSL dQ (whole top-k) + dS/P buffers ----
    dq = torch.zeros(total_tokens, num_heads, d_qk, dtype=q.dtype, device=q.device)  # rope cols stay 0
    chunk_dS = torch.empty(total_tokens, num_heads, TOPK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, TOPK, dtype=torch.bfloat16, device=q.device)
    if (os.environ.get("PRIMUS_DSA_FLYDSL_BWD_DQ_M16", "0") == "1") and (num_heads % 16 == 0) and (TOPK % 32 == 0):
        dq_launch = _get_dq_m16_kernel(int(num_heads), D, int(d_qk), int(TOPK), scale)
        dq_launch(
            q.reshape(-1),
            kv.reshape(-1),
            do.reshape(-1),
            topk_p.reshape(-1),
            lse_c.reshape(-1),
            delta.reshape(-1),
            dq.reshape(-1),
            chunk_dS.reshape(-1),
            chunk_P.reshape(-1),
            int(total_tokens),
        )
    else:
        single_latent = num_heads <= 64
        dq_launch = _get_dq_kernel(int(num_heads), D, int(d_qk), int(TOPK), single_latent, scale)
        dq_launch(
            q.reshape(-1),
            kv.reshape(-1),
            do.reshape(-1),
            topk_p.reshape(-1),
            lse_c.reshape(-1),
            delta.reshape(-1),
            dq.reshape(-1),
            chunk_dS.reshape(-1),
            chunk_P.reshape(-1),
            int(total_tokens),
        )

    # ---- dKV: intermediate GEMM + CSR inverted-topk scatter-reduce ----
    HAS_ROPE = False
    interm = torch.empty(total_tokens, TOPK, d_qk, dtype=torch.bfloat16, device=q.device)
    if (os.environ.get("PRIMUS_DSA_FLYDSL_BWD_INTERM_MFMA", "0") == "1") and (num_heads % 16 == 0):
        # Native FlyDSL M=16 head-contraction MFMA (replaces the Triton dkv-interm).
        # rope cols of interm are never read by the gather; leave them undefined.
        # flydsl packs a tensor's byte-size as i32, so any single arg must stay
        # < 2^31 bytes. interm = T*TOPK*d_qk*2 can exceed that (e.g. H128 K512 ->
        # 2.4 GB). Chunk the launch over T so each sub-tensor fits; the kernel is
        # grid=(T,) per-token independent, so a T-slice is self-contained.
        interm_launch = _get_interm_kernel(int(num_heads), D, int(d_qk), int(TOPK))
        _bytes_per_tok = TOPK * d_qk * 2
        _max_tok = max(1, (2**31 - 1) // max(_bytes_per_tok, num_heads * d_qk * 2, num_heads * TOPK * 2))
        _t0 = 0
        while _t0 < total_tokens:
            _t1 = min(total_tokens, _t0 + _max_tok)
            interm_launch(
                q[_t0:_t1].reshape(-1), do[_t0:_t1].reshape(-1),
                chunk_dS[_t0:_t1].reshape(-1), chunk_P[_t0:_t1].reshape(-1),
                interm[_t0:_t1].reshape(-1), int(_t1 - _t0),
            )
            _t0 = _t1
    else:
        # BH_DKV=32/TK_DKV=64 whole-top-k (R_CHUNK=TOPK) in one dKV-intermediate pass.
        BH_DKV, TK_DKV = 32, 64
        num_hg_dkv = triton.cdiv(num_heads, BH_DKV)
        _bwd_compute_dkv_intermediate[(total_tokens,)](
            q,
            do,
            chunk_dS,
            chunk_P,
            interm,
            q.stride(0),
            q.stride(1),
            do.stride(0),
            do.stride(1),
            chunk_dS.stride(0),
            chunk_dS.stride(1),
            interm.stride(0),
            interm.stride(1),
            num_heads,
            R_CHUNK=TOPK,
            TILE_K=TK_DKV,
            BLOCK_H=BH_DKV,
            NUM_HG=num_hg_dkv,
            D_V=D,
            D_ROPE=rope_rank,
            HAS_ROPE=HAS_ROPE,
            num_warps=4,
        )

    dkv_acc = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    inv_ptr, inv_data = _build_inverted_topk_slice(topk_p, 0, TOPK, num_kv=num_kv)
    _bwd_dkv_gather_acc[(num_kv,)](
        interm,
        inv_ptr,
        inv_data,
        dkv_acc,
        interm.stride(1),
        dkv_acc.stride(0),
        D_V=D,
        D_ROPE=rope_rank,
        HAS_ROPE=HAS_ROPE,
        num_warps=4,
    )

    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dkv = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv, d_sink


__all__ = ["sparse_mla_bwd_v4_flydsl"]
