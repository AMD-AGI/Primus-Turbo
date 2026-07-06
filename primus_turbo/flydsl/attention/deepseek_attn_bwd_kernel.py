###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 attention BACKWARD FlyDSL launcher (ported from Primus).

:func:`hca_attention_bwd_flydsl_kernel` is the Primus-Turbo adapter over the
ported FlyDSL dense / sliding-window-causal (SWA) backward kernels in
:mod:`primus_turbo.flydsl.attention.kernels`. It runs a fully-FlyDSL three-stage
backward (no Triton dependency):

* preprocess — ``D[b,h,m] = sum_d out * dout`` (``build_swa_bwd_preprocess_module``)
* dQ (+ dsink via atomic add) — ``build_swa_bwd_dq_module``
* dK / dV (MQA head-loop accumulator) — ``build_swa_bwd_dkv_module``

Scope (the dispatcher's ``can_handle`` gates everything else to Triton): bf16,
``D == 512``, ``swa_window > 0``, ``Sq == Sk``, MQA (``K_H == 1``), optional sink,
``additive_mask is None`` and ``hca_local_seqlen == 0``. The HCA split-mask
backward and the CSA backward stay on Triton — the reference does not wire any
backward, the Primus-Turbo CSA path is MHA (the ported CSA backward kernel
assumes MQA), and neither is covered by the FlyDSL test parametrisation; the HCA
/ CSA backward kernel builders are vendored under ``kernels`` for a future round.

Returns ``(dq, dk, dv, dsink)`` in the input dtype (``dsink`` is ``None`` when
``sink is None``).
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import torch

from primus_turbo.flydsl.attention.kernels.hca_bwd_dkv_pool_kernel import build_hca_bwd_dkv_pool_module
from primus_turbo.flydsl.attention.kernels.hca_bwd_dq_pool_kernel import build_hca_bwd_dq_pool_module
from primus_turbo.flydsl.attention.kernels.sla_bwd_dkv_kernel import build_swa_bwd_dkv_module
from primus_turbo.flydsl.attention.kernels.sla_bwd_kernel import (
    build_swa_bwd_dq_module,
    build_swa_bwd_preprocess_module,
)

_PRE_CACHE = {}
_PRE_LOCK = threading.Lock()
_DQ_CACHE = {}
_DQ_LOCK = threading.Lock()
_DKV_CACHE = {}
_DKV_LOCK = threading.Lock()
_DQ_POOL_CACHE = {}
_DQ_POOL_LOCK = threading.Lock()
_DKV_POOL_CACHE = {}
_DKV_POOL_LOCK = threading.Lock()


def _get_preprocess(head_dim, dtype_str, block_rows):
    key = (head_dim, dtype_str, block_rows)
    with _PRE_LOCK:
        if key in _PRE_CACHE:
            return _PRE_CACHE[key]
        launch = build_swa_bwd_preprocess_module(head_dim=head_dim, dtype_str=dtype_str, block_rows=block_rows)
        _PRE_CACHE[key] = launch
        return launch


def _get_dq(num_heads, head_dim, swa_window, dtype_str, mqa_kv, has_sink):
    key = (num_heads, head_dim, swa_window, dtype_str, mqa_kv, has_sink)
    with _DQ_LOCK:
        if key in _DQ_CACHE:
            return _DQ_CACHE[key]
        launch = build_swa_bwd_dq_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
            has_sink=has_sink,
        )
        _DQ_CACHE[key] = launch
        return launch


def _get_dkv(num_heads, head_dim, swa_window, dtype_str, mqa_kv):
    key = (num_heads, head_dim, swa_window, dtype_str, mqa_kv)
    with _DKV_LOCK:
        if key in _DKV_CACHE:
            return _DKV_CACHE[key]
        launch = build_swa_bwd_dkv_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
        )
        _DKV_CACHE[key] = launch
        return launch


def _get_dq_pool(num_heads, head_dim, pool_size, hca_local_seqlen, dtype_str, mqa_kv):
    key = (num_heads, head_dim, pool_size, hca_local_seqlen, dtype_str, mqa_kv)
    with _DQ_POOL_LOCK:
        if key in _DQ_POOL_CACHE:
            return _DQ_POOL_CACHE[key]
        launch = build_hca_bwd_dq_pool_module(
            num_heads=num_heads,
            head_dim=head_dim,
            pool_size=pool_size,
            hca_local_seqlen=hca_local_seqlen,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
        )
        _DQ_POOL_CACHE[key] = launch
        return launch


def _get_dkv_pool(num_heads, head_dim, pool_size, hca_local_seqlen, dtype_str, mqa_kv):
    key = (num_heads, head_dim, pool_size, hca_local_seqlen, dtype_str, mqa_kv)
    with _DKV_POOL_LOCK:
        if key in _DKV_POOL_CACHE:
            return _DKV_POOL_CACHE[key]
        launch = build_hca_bwd_dkv_pool_module(
            num_heads=num_heads,
            head_dim=head_dim,
            pool_size=pool_size,
            hca_local_seqlen=hca_local_seqlen,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
        )
        _DKV_POOL_CACHE[key] = launch
        return launch


def _run_preprocess(out: torch.Tensor, dout: torch.Tensor) -> torch.Tensor:
    """D[b,h,m] = sum_d out * dout, fp32, via the FlyDSL preprocess kernel."""
    B, HQ, Sq, D = out.shape
    d_buf = torch.empty((B, HQ, Sq), device=out.device, dtype=torch.float32)
    out_f = out.contiguous().view(-1, D)
    dout_f = dout.contiguous().view(-1, D)
    delta_f = d_buf.view(-1)
    n_rows = out_f.shape[0]
    dtype_str = "bf16" if out.dtype == torch.bfloat16 else "f16"
    max_block_threads = 256
    threads_per_row = D // 8
    block_rows = 1
    for br in (8, 4, 2, 1):
        if n_rows % br == 0 and br * threads_per_row <= max_block_threads:
            block_rows = br
            break
    launch = _get_preprocess(D, dtype_str, block_rows)
    launch(out_f, dout_f, delta_f, n_rows)
    return d_buf


def _hca_split_mask_bwd(
    q: torch.Tensor,  # [B, HQ, Sq, D]
    k: torch.Tensor,  # [B, 1, Sq + P, D] (k_cat = [local | pool], MQA)
    v: torch.Tensor,  # [B, 1, Sq + P, D]
    out: torch.Tensor,  # [B, HQ, Sq, D]
    dout: torch.Tensor,  # [B, HQ, Sq, D]
    lse: torch.Tensor,  # [B, HQ, Sq] fp32 (raw, JOINT domain)
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
    additive_mask: torch.Tensor,  # [Sq, P] pool-only additive mask
    hca_local_seqlen: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """HCA split-mask backward (MQA, bf16).

    The HCA forward runs one joint online softmax over the local sliding-window
    keys (``k[:, :, :Sq]``) and the compressed pool keys (``k[:, :, Sq:Sq+P]``,
    biased by ``additive_mask``) plus the optional sink. The gradient splits by
    region:

    * dQ = local-stream dQ (SWA dq kernel) + pool-stream dQ (pool dq kernel,
      accumulated into the same fp32 buffer). The sink gradient comes from the
      local dq kernel (the pool stream never touches the sink).
    * dK / dV: the local rows from the SWA dkv kernel, the pool rows from the
      pool dkv kernel (atomic-add into the pool slice), then concatenated.

    Both streams consume the JOINT fp32 LSE / delta, so each stream's
    ``exp(qk*scale [+bias] - lse)`` is already the correct joint probability.
    """
    B, HQ, Sq, D = q.shape
    HK, Sk = k.shape[1], k.shape[2]
    pool_size = int(Sk) - int(hca_local_seqlen)
    if pool_size <= 0:
        raise NotImplementedError("HCA pool backward requires Sk > hca_local_seqlen")
    if int(hca_local_seqlen) != Sq:
        raise NotImplementedError("HCA pool backward requires hca_local_seqlen == Sq")

    dtype_str = "bf16"
    swa_window_int = int(swa_window)
    has_sink = sink is not None

    q_c = q.contiguous()
    k_cat = k.contiguous()
    v_cat = v.contiguous()
    dout_c = dout.contiguous()
    lse_c = lse.contiguous()
    # Pool kernels read ADD_MASK in the input element type (bf16/f16).
    add_mask = additive_mask.to(q.dtype).contiguous()

    k_local = k_cat[:, :, :Sq, :].contiguous()
    v_local = v_cat[:, :, :Sq, :].contiguous()

    d_buf = _run_preprocess(out.contiguous(), dout_c)

    # ---- dQ: local stream (SWA dq + sink), then pool stream (accumulate) ----
    dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
    if has_sink:
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink.contiguous()
    else:
        sink_arg = torch.zeros((HQ,), device=q.device, dtype=torch.float32)

    dq_local_launch = _get_dq(HQ, D, swa_window_int, dtype_str, True, has_sink)
    dq_local_launch(
        q_c,
        k_local,
        v_local,
        dout_c,
        lse_c,
        d_buf,
        dq_fp32,
        dsink_fp32,
        sink_arg,
        int(B),
        int(Sq),
        int(Sq),
    )

    # The pool-stream dQ kernel writes its contribution into a dedicated
    # zero-init bf16 buffer via packed 2xbf16 stores (no f32 RMW),
    # then we add it onto the f32 local dq. This de-serialises the pool dq
    # epilogue (drops the load+add and halves the store op count) while keeping
    # the f32 local-stream contract intact (dq_fp32 stays f32-accurate for the
    # local term; only the pool term is truncated to bf16 before the merge).
    dq_pool_bf16 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=q.dtype)
    dq_pool_launch = _get_dq_pool(HQ, D, int(pool_size), int(hca_local_seqlen), dtype_str, True)
    dq_pool_launch(
        q_c,
        k_cat,
        v_cat,
        dout_c,
        lse_c,
        d_buf,
        dq_pool_bf16,
        add_mask,
        int(B),
        int(Sq),
        int(Sk),
    )
    dq_fp32.add_(dq_pool_bf16.to(torch.float32))

    # ---- dK / dV: local stream rows, then pool stream rows (atomic-add) ----
    dk_local_fp32 = torch.zeros((B, HK, Sq, D), device=q.device, dtype=torch.float32)
    dv_local_fp32 = torch.zeros((B, HK, Sq, D), device=q.device, dtype=torch.float32)
    dkv_local_launch = _get_dkv(HQ, D, swa_window_int, dtype_str, True)
    dkv_local_launch(
        q_c,
        k_local,
        v_local,
        dout_c,
        lse_c,
        d_buf,
        dk_local_fp32,
        dv_local_fp32,
        int(B),
        int(Sq),
        int(Sq),
    )

    # Pool kernel atomic-adds into rows [hca_local_seqlen, hca_local_seqlen+P)
    # of these full-length [B, HK, Sk, D] buffers (the local rows stay zero).
    dk_pool_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    dv_pool_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    dkv_pool_launch = _get_dkv_pool(HQ, D, int(pool_size), int(hca_local_seqlen), dtype_str, True)
    dkv_pool_launch(
        q_c,
        k_cat,
        v_cat,
        dout_c,
        lse_c,
        d_buf,
        dk_pool_fp32,
        dv_pool_fp32,
        add_mask,
        int(B),
        int(Sq),
        int(Sk),
    )

    dk_out = torch.cat([dk_local_fp32, dk_pool_fp32[:, :, hca_local_seqlen:, :]], dim=2).to(k.dtype)
    dv_out = torch.cat([dv_local_fp32, dv_pool_fp32[:, :, hca_local_seqlen:, :]], dim=2).to(v.dtype)
    dq_out = dq_fp32.to(q.dtype)
    dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
    return dq_out, dk_out, dv_out, dsink_out


def hca_attention_bwd_flydsl_kernel(
    q: torch.Tensor,  # [B, HQ, Sq, D]
    k: torch.Tensor,  # [B, 1, Sk, D] (MQA)
    v: torch.Tensor,  # [B, 1, Sk, D]
    out: torch.Tensor,  # [B, HQ, Sq, D]
    dout: torch.Tensor,  # [B, HQ, Sq, D]
    lse: torch.Tensor,  # [B, HQ, Sq] fp32 (raw domain)
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    additive_mask: Optional[torch.Tensor],
    scale: float,
    hca_local_seqlen: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Dense / SWA backward via the ported FlyDSL kernels (MQA, bf16)."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("rank-4 q/k/v required")
    if dout.shape != out.shape or out.shape != q.shape:
        raise ValueError("shape mismatch among q / out / dout")
    B, HQ, Sq, D = q.shape
    HK, Sk = k.shape[1], k.shape[2]
    if k.shape != (B, HK, Sk, D) or v.shape != k.shape:
        raise ValueError("k/v shape mismatch")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("bf16 only")
    if HK != 1:
        raise NotImplementedError("FlyDSL dense backward supports MQA (K_H == 1) only")
    if int(swa_window) <= 0:
        raise NotImplementedError("swa_window > 0 required")

    # HCA split-mask backward: joint softmax over [local SWA | pool]. Composed
    # from the local SWA dq/dkv kernels (fed the joint LSE) + the ported HCA
    # pool dq/dkv kernels.
    if additive_mask is not None and int(hca_local_seqlen) > 0:
        return _hca_split_mask_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            sink=sink,
            swa_window=int(swa_window),
            scale=scale,
            additive_mask=additive_mask,
            hca_local_seqlen=int(hca_local_seqlen),
        )
    if additive_mask is not None or int(hca_local_seqlen) != 0:
        raise NotImplementedError(
            "HCA split-mask backward needs both additive_mask and hca_local_seqlen"
        )
    if Sq != Sk:
        raise NotImplementedError("SWA requires Sq == Sk")

    has_sink = sink is not None
    swa_window_int = int(swa_window)
    dtype_str = "bf16"

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    dout_c = dout.contiguous()
    lse_c = lse.contiguous()

    dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dk_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    dv_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
    if has_sink:
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink.contiguous()
    else:
        sink_arg = torch.zeros((HQ,), device=q.device, dtype=torch.float32)

    d_buf = _run_preprocess(out.contiguous(), dout_c)

    dq_launch = _get_dq(HQ, D, swa_window_int, dtype_str, True, has_sink)
    dq_launch(
        q_c,
        k_c,
        v_c,
        dout_c,
        lse_c,
        d_buf,
        dq_fp32,
        dsink_fp32,
        sink_arg,
        int(B),
        int(Sq),
        int(Sk),
    )

    dkv_launch = _get_dkv(HQ, D, swa_window_int, dtype_str, True)
    dkv_launch(
        q_c,
        k_c,
        v_c,
        dout_c,
        lse_c,
        d_buf,
        dk_fp32,
        dv_fp32,
        int(B),
        int(Sq),
        int(Sk),
    )

    dq_out = dq_fp32.to(q.dtype)
    dk_out = dk_fp32.to(k.dtype)
    dv_out = dv_fp32.to(v.dtype)
    dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
    return dq_out, dk_out, dv_out, dsink_out


__all__ = [
    "hca_attention_bwd_flydsl_kernel",
]
