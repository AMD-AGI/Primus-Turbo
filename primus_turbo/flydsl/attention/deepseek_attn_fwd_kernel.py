###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 attention FORWARD FlyDSL launchers (ported from Primus).

This module is the Primus-Turbo adapter over the ported FlyDSL forward kernel
builders in :mod:`primus_turbo.flydsl.attention.kernels`:

* :func:`hca_attention_fwd_flydsl_kernel` — dense / sliding-window-causal (SWA)
  forward. Wraps ``sla_fwd_kernel.build_swa_fwd_module``. Scope (matching
  the reference SWA kernel): bf16, ``D == 512``, ``swa_window > 0``, ``Sq == Sk``,
  ``scale == 1/sqrt(D)``, MQA (``K_H == 1``) or MHA (``K_H == H``), and *no* sink /
  additive bias / HCA split-mask (those fall back to Triton in the dispatcher).
* :func:`csa_pool_attention_fwd_flydsl_kernel` — CSA fused forward with in-kernel
  gather from the compressed pool (local SWA + sparse top-K + optional sink) in
  one online softmax. Wraps ``csa_pool_fwd_kernel.build_csa_pool_fwd_module``.

Both return ``(out, lse)`` where ``out`` is ``[B, H, Sq, D]`` in input dtype and
``lse`` is fp32 ``[B, H, Sq]`` in the raw-qk-scaled domain (``m + ln(l)``, ``m``
absorbing ``sm_scale``) — the same LSE convention the Triton backward consumes.

MQA avoids the ``.expand().clone()`` K/V broadcast: the ``[B, 1, Sk, D]`` view is
passed directly and the kernel reads K/V with ``stride_kh == 0`` via ``mqa_kv``.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Optional, Tuple

import torch

# Match the reference launcher's runtime defaults (16 B DMA-to-LDS forward path,
# 2 waves/EU occupancy target). setdefault so an explicit override still wins.
os.environ.setdefault("FLYDSL_SLA_FWD_ENABLE_DMA", "1")
os.environ.setdefault("FLYDSL_WAVES_PER_EU", "2")

from primus_turbo.flydsl.attention.kernels.csa_pool_fwd_kernel import build_csa_pool_fwd_module
from primus_turbo.flydsl.attention.kernels.csa_pool_sparse_fwd_kernel import (
    build_csa_pool_sparse_fwd_module,
)
from primus_turbo.flydsl.attention.kernels.sla_fwd_kernel import build_swa_fwd_module

# ───────────────────────────────────────────────────────────────────────────
# SWA (dense) forward
# ───────────────────────────────────────────────────────────────────────────

_SWA_KERNEL_CACHE = {}
_SWA_KERNEL_CACHE_LOCK = threading.Lock()


def _get_swa_kernel(
    num_heads_q, head_dim, swa_window, dtype_str, block_m, block_n, waves_per_eu, mqa_kv, flat_work_group_size
):
    key = (
        num_heads_q,
        head_dim,
        swa_window,
        dtype_str,
        block_m,
        block_n,
        waves_per_eu,
        mqa_kv,
        flat_work_group_size,
    )
    with _SWA_KERNEL_CACHE_LOCK:
        if key in _SWA_KERNEL_CACHE:
            return _SWA_KERNEL_CACHE[key]
        launch = build_swa_fwd_module(
            num_heads=num_heads_q,
            head_dim=head_dim,
            swa_window=int(swa_window),
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            block_m=block_m,
            block_n=block_n,
            flat_work_group_size=flat_work_group_size,
            layout_bhld=True,
            mqa_kv=mqa_kv,
        )
        _SWA_KERNEL_CACHE[key] = launch
        return launch


def hca_attention_fwd_flydsl_kernel(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, K_H, Sk, D]
    v: torch.Tensor,  # [B, K_H, Sk, D]
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    additive_mask: Optional[torch.Tensor],
    scale: float,
    hca_local_seqlen: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense / SWA forward via the ported FlyDSL SWA kernel.

    The dispatcher's ``can_handle`` gates the unsupported cases (sink, additive
    bias, HCA split-mask, non-1/sqrt(D) scale, ``Sq != Sk``) to Triton, so the
    raises below are defensive — they only fire on a contract violation.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("rank-4 q/k/v required")
    B, HQ, Sq, D = q.shape
    Bk, HK, Sk, Dk = k.shape
    if (Bk, Sk, Dk) != (B, Sk, D) or v.shape != k.shape:
        raise ValueError("k/v shape mismatch w.r.t. q")
    if HK != 1 and HK != HQ:
        raise ValueError(f"K_H must be 1 or {HQ}; got {HK}")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("bf16 only")
    if D != 512:
        raise NotImplementedError("head_dim=512 only")
    if sink is not None:
        raise NotImplementedError("sink not supported in the FlyDSL SWA forward")
    if additive_mask is not None or int(hca_local_seqlen) != 0:
        raise NotImplementedError("HCA split-mask not supported in the FlyDSL SWA forward")
    if int(swa_window) <= 0:
        raise NotImplementedError("swa_window > 0 required")
    if Sq != Sk:
        raise NotImplementedError("SWA requires Sq == Sk")
    expected_scale = 1.0 / math.sqrt(D)
    if not math.isclose(float(scale), expected_scale, rel_tol=1e-4):
        raise NotImplementedError("scale=1/sqrt(D) only")

    mqa = (HK == 1) and (HQ > 1)

    q_bhld = q.contiguous()
    k_bhld = k.contiguous()
    v_bhld = v.contiguous()
    o_bhld = torch.empty_like(q_bhld)
    lse = torch.zeros((B, HQ, Sq), device=q.device, dtype=torch.float32)

    block_m = int(os.environ.get("PRIMUS_V4_FLYDSL_BLOCK_M", "128"))
    block_n = int(os.environ.get("PRIMUS_V4_FLYDSL_BLOCK_N", "32"))
    waves_per_eu = int(os.environ.get("FLYDSL_WAVES_PER_EU", "2"))
    fwgs_env = os.environ.get("PRIMUS_V4_FLYDSL_FWGS", "")
    flat_work_group_size = int(fwgs_env) if fwgs_env else None

    launch = _get_swa_kernel(HQ, D, int(swa_window), "bf16", block_m, block_n, waves_per_eu, mqa, flat_work_group_size)
    launch(
        q_bhld.view(-1),
        k_bhld.view(-1),
        v_bhld.view(-1),
        o_bhld.view(-1),
        lse.view(-1),
        B,
        Sq,
    )
    return o_bhld, lse


# ───────────────────────────────────────────────────────────────────────────
# CSA-from-pool fused forward (in-kernel gather from the compressed pool)
# ───────────────────────────────────────────────────────────────────────────

_CSA_POOL_KERNEL_CACHE = {}
_CSA_POOL_KERNEL_CACHE_LOCK = threading.Lock()

_CSA_SPARSE_KERNEL_CACHE = {}
_CSA_SPARSE_KERNEL_CACHE_LOCK = threading.Lock()


def _get_csa_sparse_kernel(num_heads_q, head_dim, dtype_str, waves_per_eu):
    key = (num_heads_q, head_dim, dtype_str, waves_per_eu)
    with _CSA_SPARSE_KERNEL_CACHE_LOCK:
        if key in _CSA_SPARSE_KERNEL_CACHE:
            return _CSA_SPARSE_KERNEL_CACHE[key]
        launch = build_csa_pool_sparse_fwd_module(
            num_heads=num_heads_q,
            head_dim=head_dim,
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
        )
        _CSA_SPARSE_KERNEL_CACHE[key] = launch
        return launch


def _csa_lse_merge(o_local, lse_local, o_sparse, lse_sparse, sink):
    """Stable joint softmax merge of the local + sparse branches (+ optional sink).

    lse = ln(exp(lse_local) + exp(lse_sparse) [+ exp(sink)]); the sink enters the
    normaliser only. out = o_local * exp(lse_local - lse) + o_sparse *
    exp(lse_sparse - lse). Computed in fp32 for the weights, output in o dtype.
    """
    m = torch.maximum(lse_local, lse_sparse)
    if sink is not None:
        B, H, S = lse_local.shape
        sink_bhs = sink.float().view(1, H, 1).expand(B, H, S)
        m = torch.maximum(m, sink_bhs)
    wl = torch.exp(lse_local - m)
    ws = torch.exp(lse_sparse - m)
    denom = wl + ws
    if sink is not None:
        denom = denom + torch.exp(sink_bhs - m)
    inv = 1.0 / denom
    wl = (wl * inv).unsqueeze(-1)
    ws = (ws * inv).unsqueeze(-1)
    out = o_local.float() * wl + o_sparse.float() * ws
    lse = m + torch.log(denom)
    return out.to(o_local.dtype), lse


_csa_merge_compiled = None


def _get_merge_fn():
    """torch.compile the elementwise merge once (it is memory-bound; the
    compiled fused kernel is ~8x faster than eager on the V4-Flash widths)."""
    global _csa_merge_compiled
    if _csa_merge_compiled is None:
        if os.environ.get("PRIMUS_V4_CSA_MERGE_COMPILE", "1") == "1":
            try:
                _csa_merge_compiled = torch.compile(_csa_lse_merge)
            except Exception:  # noqa: BLE001
                _csa_merge_compiled = _csa_lse_merge
        else:
            _csa_merge_compiled = _csa_lse_merge
    return _csa_merge_compiled


def _csa_pool_split_forward(q, k_local, v_local, pool, topk_idxs, sink, swa_window, scale):
    """MQA split CSA forward: local SWA + sparse head-block MFMA + LSE merge."""
    B, HQ, Sq, D = q.shape
    P = pool.shape[1]
    K_topk = topk_idxs.shape[2]
    waves_per_eu = int(os.environ.get("FLYDSL_WAVES_PER_EU", "2"))

    q_c = q.contiguous()
    pool_c = pool.contiguous()
    topk_i32 = topk_idxs.to(torch.int32).contiguous()

    # Local SWA branch (no sink; MQA K/V passed as [B, 1, Sq, D]).
    block_m = int(os.environ.get("PRIMUS_V4_FLYDSL_BLOCK_M", "128"))
    block_n = int(os.environ.get("PRIMUS_V4_FLYDSL_BLOCK_N", "32"))
    fwgs_env = os.environ.get("PRIMUS_V4_FLYDSL_FWGS", "")
    flat_work_group_size = int(fwgs_env) if fwgs_env else None
    swa_launch = _get_swa_kernel(
        HQ, D, int(swa_window), "bf16", block_m, block_n, waves_per_eu, True, flat_work_group_size
    )
    o_local = torch.empty_like(q_c)
    lse_local = torch.zeros((B, HQ, Sq), device=q.device, dtype=torch.float32)
    swa_launch(
        q_c.view(-1),
        k_local.contiguous().view(-1),
        v_local.contiguous().view(-1),
        o_local.view(-1),
        lse_local.view(-1),
        B,
        Sq,
    )

    # Sparse pool branch (head-block MFMA, in-kernel gather).
    sparse_launch = _get_csa_sparse_kernel(HQ, D, "bf16", waves_per_eu)
    o_sparse = torch.empty_like(q_c)
    lse_sparse = torch.zeros((B, HQ, Sq), device=q.device, dtype=torch.float32)
    sparse_launch(
        q_c.view(-1),
        pool_c.view(-1),
        topk_i32.view(-1),
        o_sparse.view(-1),
        lse_sparse.view(-1),
        B,
        Sq,
        int(K_topk),
        int(P),
    )

    return _get_merge_fn()(o_local, lse_local, o_sparse, lse_sparse, sink)


def _get_csa_pool_kernel(num_heads_q, head_dim, swa_window, dtype_str, block_n, block_k, waves_per_eu, has_sink, mqa_kv):
    key = (num_heads_q, head_dim, swa_window, dtype_str, block_n, block_k, waves_per_eu, has_sink, mqa_kv)
    with _CSA_POOL_KERNEL_CACHE_LOCK:
        if key in _CSA_POOL_KERNEL_CACHE:
            return _CSA_POOL_KERNEL_CACHE[key]
        launch = build_csa_pool_fwd_module(
            num_heads=num_heads_q,
            head_dim=head_dim,
            swa_window=int(swa_window),
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            block_n=block_n,
            block_k=block_k,
            has_sink=has_sink,
            has_sparse=True,
            mqa_kv=mqa_kv,
        )
        _CSA_POOL_KERNEL_CACHE[key] = launch
        return launch


def csa_pool_attention_fwd_flydsl_kernel(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, K_H, Sq, D]
    v_local: torch.Tensor,  # [B, K_H, Sq, D]
    pool: torch.Tensor,  # [B, P, D]
    topk_idxs: torch.Tensor,  # [B, Sq, K_topk] int (-1 masks a slot)
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CSA-from-pool fused forward via the FlyDSL in-kernel gather kernel.

    Gathers the per-query top-K keys from ``pool`` using ``topk_idxs`` inside
    the kernel (no materialised ``[B, Sq, K_topk, D]`` tensor), then runs the
    same joint online softmax (local SWA + sparse + optional sink) as the
    pre-gathered CSA forward. Returns ``(out, lse)`` with ``out`` bf16
    ``[B, H, Sq, D]`` and ``lse`` fp32 ``[B, H, Sq]`` (raw-qk-scaled domain).
    """
    if q.dim() != 4 or k_local.dim() != 4 or v_local.dim() != 4:
        raise ValueError("rank-4 q/k_local/v_local required")
    if pool.dim() != 3:
        raise ValueError("rank-3 pool [B, P, D] required")
    if topk_idxs.dim() != 3:
        raise ValueError("rank-3 topk_idxs [B, Sq, K_topk] required")
    B, HQ, Sq, D = q.shape
    HK = k_local.shape[1]
    if k_local.shape != (B, HK, Sq, D) or v_local.shape != k_local.shape:
        raise ValueError("k_local/v_local shape mismatch w.r.t. q")
    if HK != 1 and HK != HQ:
        raise ValueError(f"K_H must be 1 or {HQ}; got {HK}")
    Bp, P, Dp = pool.shape
    if Bp != B or Dp != D:
        raise ValueError(f"pool shape mismatch {tuple(pool.shape)}")
    Bt, Sqt, K_topk = topk_idxs.shape
    if Bt != B or Sqt != Sq:
        raise ValueError(f"topk_idxs shape mismatch {tuple(topk_idxs.shape)}")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(f"bf16 only; got {q.dtype}")
    if D != 512:
        raise NotImplementedError(f"head_dim=512 only; got {D}")
    if int(swa_window) <= 0:
        raise NotImplementedError("swa_window > 0 required")
    if int(K_topk) <= 0:
        raise NotImplementedError("K_topk > 0 required")
    expected_scale = 1.0 / math.sqrt(D)
    if not math.isclose(float(scale), expected_scale, rel_tol=1e-4):
        raise NotImplementedError(f"only scale=1/sqrt(D) supported; got {scale}")

    has_sink = sink is not None
    mqa = (HK == 1) and (HQ > 1)

    # ── Split-forward path (MQA): local SWA (MFMA) + sparse head-block MFMA +
    # host LSE merge. Mirrors the Triton P32 split. The monolithic per-row
    # kernel is the fallback for MHA (the SWA local kernel is MQA-only) or when
    # explicitly disabled. This is the fast path that engages the matrix cores
    # for the sparse branch's AV step (design §4.7 / AV-MFMA).
    _use_split = mqa and os.environ.get("PRIMUS_V4_CSA_SPLIT_FWD", "1") == "1"
    if _use_split:
        return _csa_pool_split_forward(
            q, k_local, v_local, pool, topk_idxs, sink, int(swa_window), float(scale)
        )

    if has_sink:
        sink_fp32 = sink.float().contiguous()
        if sink_fp32.shape != (HQ,):
            raise ValueError(f"sink shape must be ({HQ},); got {tuple(sink_fp32.shape)}")
    else:
        sink_fp32 = torch.zeros((max(HQ, 1),), dtype=torch.float32, device=q.device)

    q_c = q.contiguous()
    k_c = k_local.contiguous()
    v_c = v_local.contiguous()
    pool_c = pool.contiguous()
    # The kernel reads TOPK as int32 scalars.
    topk_i32 = topk_idxs.to(torch.int32).contiguous()
    o_bhld = torch.empty_like(q_c)
    lse = torch.zeros((B, HQ, Sq), device=q.device, dtype=torch.float32)

    block_n = int(os.environ.get("PRIMUS_V4_CSA_BLOCK_N", "8"))
    block_k = int(os.environ.get("PRIMUS_V4_CSA_BLOCK_K", "16"))
    waves_per_eu = int(os.environ.get("FLYDSL_WAVES_PER_EU", "2"))

    launch = _get_csa_pool_kernel(HQ, D, int(swa_window), "bf16", block_n, block_k, waves_per_eu, has_sink, mqa)
    launch(
        q_c.view(-1),
        k_c.view(-1),
        v_c.view(-1),
        pool_c.view(-1),
        topk_i32.view(-1),
        sink_fp32.view(-1),
        o_bhld.view(-1),
        lse.view(-1),
        B,
        Sq,
        int(K_topk),
        int(P),
    )
    return o_bhld, lse


__all__ = [
    "hca_attention_fwd_flydsl_kernel",
    "csa_pool_attention_fwd_flydsl_kernel",
]
