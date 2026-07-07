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

import os
import threading
from typing import Optional, Tuple

import torch

from primus_turbo.flydsl.attention.kernels.csa_pool_bwd_kernel import (
    build_csa_pool_bwd_dpool_mfma_module,
    build_csa_pool_bwd_dpool_module,
    build_csa_pool_bwd_dq_mfma_module,
    build_csa_pool_bwd_module,
)
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
_CSA_BWD_CACHE = {}
_CSA_BWD_LOCK = threading.Lock()
_CSA_DPOOL_CACHE = {}
_CSA_DPOOL_LOCK = threading.Lock()


_CSA_DQ_MFMA_CACHE = {}
_CSA_DQ_MFMA_LOCK = threading.Lock()


def _get_csa_dq_mfma(num_heads, head_dim, dtype_str):
    key = (num_heads, head_dim, dtype_str)
    with _CSA_DQ_MFMA_LOCK:
        if key in _CSA_DQ_MFMA_CACHE:
            return _CSA_DQ_MFMA_CACHE[key]
        launch = build_csa_pool_bwd_dq_mfma_module(
            num_heads=num_heads, head_dim=head_dim, dtype_str=dtype_str,
        )
        _CSA_DQ_MFMA_CACHE[key] = launch
        return launch


def _get_csa_dpool(num_heads, head_dim, dtype_str, mqa_kv, k_block, use_mfma=False):
    key = (num_heads, head_dim, dtype_str, mqa_kv, k_block, use_mfma)
    with _CSA_DPOOL_LOCK:
        if key in _CSA_DPOOL_CACHE:
            return _CSA_DPOOL_CACHE[key]
        if use_mfma:
            # All-MFMA dpool (head-as-contraction). Same partial-buffer contract
            # as the scalar kernel; ~56x faster on the V4-Flash width.
            launch = build_csa_pool_bwd_dpool_mfma_module(
                num_heads=num_heads, head_dim=head_dim, dtype_str=dtype_str,
                mqa_kv=mqa_kv, k_block=k_block,
            )
        else:
            launch = build_csa_pool_bwd_dpool_module(
                num_heads=num_heads, head_dim=head_dim, dtype_str=dtype_str,
                mqa_kv=mqa_kv, k_block=k_block,
            )
        _CSA_DPOOL_CACHE[key] = launch
        return launch


def _get_csa_bwd(num_heads, head_dim, swa_window, dtype_str, mqa_kv, has_sink,
                 has_local=True, has_sparse=True, store_dpool=True):
    key = (num_heads, head_dim, swa_window, dtype_str, mqa_kv, has_sink,
           has_local, has_sparse, store_dpool)
    with _CSA_BWD_LOCK:
        if key in _CSA_BWD_CACHE:
            return _CSA_BWD_CACHE[key]
        launch = build_csa_pool_bwd_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
            has_sink=has_sink,
            has_local=has_local,
            has_sparse=has_sparse,
            store_dpool=store_dpool,
        )
        _CSA_BWD_CACHE[key] = launch
        return launch


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


def csa_pool_attention_bwd_flydsl_kernel(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, HK, Sq, D]
    v_local: torch.Tensor,  # [B, HK, Sq, D]
    pool: torch.Tensor,  # [B, P, D]
    topk_idxs: torch.Tensor,  # [B, Sq, K_topk]
    out: torch.Tensor,  # [B, H, Sq, D]
    dout: torch.Tensor,  # [B, H, Sq, D]
    lse: torch.Tensor,  # [B, H, Sq] fp32 (raw domain)
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """CSA-from-pool backward via the FlyDSL per-row scatter-add kernel (bf16).

    Returns ``(dq, dk_local, dv_local, dpool, dsink)`` in input dtype (``dsink``
    is ``None`` when ``sink is None``). ``dpool`` is fp32-accumulated in-kernel
    via atomic scatter-add then cast, matching the Triton reference contract.
    """
    if q.dim() != 4 or k_local.dim() != 4 or v_local.dim() != 4:
        raise ValueError("rank-4 q/k_local/v_local required")
    if pool.dim() != 3:
        raise ValueError("rank-3 pool [B, P, D] required")
    if topk_idxs.dim() != 3:
        raise ValueError("rank-3 topk_idxs [B, Sq, K_topk] required")
    if dout.shape != out.shape or out.shape != q.shape:
        raise ValueError("shape mismatch among q / out / dout")
    B, HQ, Sq, D = q.shape
    HK = k_local.shape[1]
    if HK != 1 and HK != HQ:
        raise NotImplementedError(f"K_H must be 1 or {HQ}; got {HK}")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("bf16 only")
    if int(swa_window) <= 0:
        raise NotImplementedError("swa_window > 0 required")
    Bp, P, Dp = pool.shape
    K_topk = topk_idxs.shape[2]
    if Bp != B or Dp != D or int(K_topk) <= 0:
        raise ValueError("pool / topk_idxs shape mismatch")

    dtype_str = "bf16"
    mqa = HK == 1
    has_sink = sink is not None

    q_c = q.contiguous()
    kl_c = k_local.contiguous()
    vl_c = v_local.contiguous()
    pool_c = pool.contiguous()
    dout_c = dout.contiguous()
    lse_c = lse.contiguous()
    topk_i32 = topk_idxs.to(torch.int32).contiguous()

    d_buf = _run_preprocess(out.contiguous(), dout_c)  # [B, HQ, Sq] fp32

    dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dk_fp32 = torch.zeros((B, HK, Sq, D), device=q.device, dtype=torch.float32)
    dv_fp32 = torch.zeros((B, HK, Sq, D), device=q.device, dtype=torch.float32)
    dpool_fp32 = torch.zeros((B, P, D), device=q.device, dtype=torch.float32)
    dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
    if has_sink:
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink.contiguous()
    else:
        sink_arg = torch.zeros((HQ,), device=q.device, dtype=torch.float32)

    # Split decomposition: the CSA joint softmax factors into a local SWA stream
    # + a sparse pool stream sharing the joint LSE / delta. dpool ALWAYS uses the
    # dedicated head-summed atomic-free partial kernel (works for MQA + MHA) to
    # avoid the pool-row atomic contention (Sq*H*K atomics on P rows) that
    # dominates the naive kernel.
    #
    # The dq/dk/dv path differs by head layout:
    #  * MQA (K_H == 1): reuse the fast MFMA SWA dq / dkv kernels for the local
    #    stream (fed the joint LSE, matching ``_hca_split_mask_bwd``) + the
    #    dq-only per-row CSA sparse kernel. dsink from the SWA dq kernel.
    #  * MHA (K_H == H): the SWA dq/dkv kernels are MQA-only, so use the
    #    monolithic per-row CSA kernel with store_dpool=False for dq/dk/dv/dsink.
    if mqa:
        dq_local_launch = _get_dq(HQ, D, int(swa_window), dtype_str, True, has_sink)
        dq_local_launch(
            q_c, kl_c, vl_c, dout_c, lse_c, d_buf,
            dq_fp32, dsink_fp32, sink_arg,
            int(B), int(Sq), int(Sq),
        )
        dkv_local_launch = _get_dkv(HQ, D, int(swa_window), dtype_str, True)
        dkv_local_launch(
            q_c, kl_c, vl_c, dout_c, lse_c, d_buf,
            dk_fp32, dv_fp32,
            int(B), int(Sq), int(Sq),
        )
        # Sparse dq: MFMA head-block kernel (dq[h,d]=scale*sum_k ds[h,k]*gathered[k,d])
        # when HQ % 16 == 0; else the per-row scalar kernel (store_dpool=False).
        dq_sparse_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
        _dq_mfma = (HQ % 16 == 0) and os.environ.get("PRIMUS_V4_CSA_BWD_DQ_MFMA", "1") == "1"
        if _dq_mfma:
            dq_sparse_launch = _get_csa_dq_mfma(HQ, D, dtype_str)
            dq_sparse_launch(
                q_c.view(-1), pool_c.view(-1), topk_i32.view(-1), dout_c.view(-1),
                lse_c.view(-1), d_buf.view(-1), dq_sparse_fp32.view(-1),
                int(B), int(Sq), int(K_topk), int(P),
            )
        else:
            sparse_launch = _get_csa_bwd(
                HQ, D, int(swa_window), dtype_str, mqa, False,
                has_local=False, has_sparse=True, store_dpool=False,
            )
            sparse_launch(
                q_c.view(-1), kl_c.view(-1), vl_c.view(-1), pool_c.view(-1),
                topk_i32.view(-1), dout_c.view(-1), lse_c.view(-1), d_buf.view(-1),
                sink_arg.view(-1),
                dq_sparse_fp32.view(-1), dk_fp32.view(-1), dv_fp32.view(-1),
                dpool_fp32.view(-1), dsink_fp32.view(-1),
                int(B), int(Sq), int(K_topk), int(P),
            )
        dq_fp32.add_(dq_sparse_fp32)
    else:
        # MHA: monolithic per-row kernel for dq/dk/dv/dsink (no dpool here).
        mono_launch = _get_csa_bwd(
            HQ, D, int(swa_window), dtype_str, mqa, has_sink,
            has_local=True, has_sparse=True, store_dpool=False,
        )
        mono_launch(
            q_c.view(-1), kl_c.view(-1), vl_c.view(-1), pool_c.view(-1),
            topk_i32.view(-1), dout_c.view(-1), lse_c.view(-1), d_buf.view(-1),
            sink_arg.view(-1),
            dq_fp32.view(-1), dk_fp32.view(-1), dv_fp32.view(-1),
            dpool_fp32.view(-1), dsink_fp32.view(-1),
            int(B), int(Sq), int(K_topk), int(P),
        )

    # dpool: dedicated head-summed atomic-free partial kernel + index_add reduce.
    # The all-MFMA variant (head-as-contraction) needs HQ % 16 == 0; otherwise
    # fall back to the scalar per-head kernel. Env can force the scalar path.
    k_block = int(os.environ.get("PRIMUS_V4_CSA_BWD_DPOOL_KBLOCK", "1"))
    _dpool_mfma = (HQ % 16 == 0) and os.environ.get("PRIMUS_V4_CSA_BWD_DPOOL_MFMA", "1") == "1"
    dpool_part = torch.zeros((B, Sq, K_topk, D), device=q.device, dtype=torch.float32)
    dpool_launch = _get_csa_dpool(HQ, D, dtype_str, mqa, k_block, use_mfma=_dpool_mfma)
    dpool_launch(
        q_c.view(-1), pool_c.view(-1), topk_i32.view(-1), dout_c.view(-1),
        lse_c.view(-1), d_buf.view(-1), dpool_part.view(-1),
        int(B), int(Sq), int(K_topk), int(P),
    )
    # Invalid slots (topk < 0 or >= P) stored 0 in the partial, so clamp their
    # index into range for the scatter — they contribute nothing.
    topk_flat = topk_i32.view(B, Sq * K_topk).long()
    topk_flat = torch.where((topk_flat >= 0) & (topk_flat < P), topk_flat, torch.zeros_like(topk_flat))
    part_flat = dpool_part.view(B, Sq * K_topk, D)
    for b in range(B):
        dpool_fp32[b].index_add_(0, topk_flat[b], part_flat[b])

    dq_out = dq_fp32.to(q.dtype)
    dk_out = dk_fp32.to(k_local.dtype)
    dv_out = dv_fp32.to(v_local.dtype)
    dpool_out = dpool_fp32.to(pool.dtype)
    dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
    return dq_out, dk_out, dv_out, dpool_out, dsink_out


__all__ = [
    "hca_attention_bwd_flydsl_kernel",
    "csa_pool_attention_bwd_flydsl_kernel",
]
