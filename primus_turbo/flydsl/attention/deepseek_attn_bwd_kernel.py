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

from primus_turbo.flydsl.attention.kernels.csa_bwd_full_kernel import build_csa_bwd_full_module
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
    # R5: launch the swa_bwd_dkv kernel as a 4-wavefront (256-thread)
    # workgroup so all 4 SIMDs of the LDS-capped CU (1 WG/CU at ~97 KB LDS)
    # issue MFMA. num_waves=4 splits the dV/dK D-axis 4 ways and the
    # GEMM1/GEMM3 D-contraction 2 ways per m-tile (K-split partial reduction
    # through LDS) so no wave sits idle. Pure intra-WG parallelism change.
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
            num_waves=4,
            # R13: enable the gfx950 sched_group_barrier MFMA/MEM
            # cluster-pair interleave in the dominant swa_bwd_dkv inner
            # loop. Set False to restore the byte-identical pre-R13
            # schedule (rollback).
            sched_interleave=True,
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

    dq_pool_launch = _get_dq_pool(HQ, D, int(pool_size), int(hca_local_seqlen), dtype_str, True)
    dq_pool_launch(
        q_c,
        k_cat,
        v_cat,
        dout_c,
        lse_c,
        d_buf,
        dq_fp32,
        add_mask,
        int(B),
        int(Sq),
        int(Sk),
    )

    # ---- dK / dV: overlap the two data-independent kernels on two streams ----
    # R10: dkv_local (swa_bwd_dkv, compute/VALU-bound) and dkv_pool
    # (hca_bwd_dkv_pool, memory/L2-bound) are provably independent — they write
    # DISJOINT outputs (dk_local/dv_local vs dk_pool/dv_pool) and only READ the
    # shared preprocess outputs (q_c/k_cat/v_cat/dout_c/lse_c/d_buf). Issuing the
    # memory-bound pool kernel on a second HIP stream concurrently with the
    # compute-bound local kernel lets the pool's VMEM/L2 traffic hide behind the
    # local kernel's MFMA/VALU work, raising aggregate CU/L2 utilization with
    # zero change to either kernel's math. This is a deterministic per-launch
    # scheduling change (no tensor-identity cache), so the overlap recurs
    # identically on every real-training backward step.
    dk_local_fp32 = torch.zeros((B, HK, Sq, D), device=q.device, dtype=torch.float32)
    dv_local_fp32 = torch.zeros((B, HK, Sq, D), device=q.device, dtype=torch.float32)
    # Pool kernel atomic-adds into rows [hca_local_seqlen, hca_local_seqlen+P)
    # of these full-length [B, HK, Sk, D] buffers (the local rows stay zero).
    dk_pool_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    dv_pool_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)

    dkv_local_launch = _get_dkv(HQ, D, swa_window_int, dtype_str, True)
    dkv_pool_launch = _get_dkv_pool(HQ, D, int(pool_size), int(hca_local_seqlen), dtype_str, True)

    cur_stream = torch.cuda.current_stream(device=q.device)
    pool_stream = torch.cuda.Stream(device=q.device)
    # The pool stream must observe the d_buf preprocess + the dk_pool/dv_pool
    # zero-init that were enqueued on the current stream above.
    pool_stream.wait_stream(cur_stream)

    with torch.cuda.stream(pool_stream):
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
            stream=pool_stream,
        )

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
        stream=cur_stream,
    )

    # Join: the current stream (which runs the trailing torch.cat) must wait for
    # the pool kernel to finish writing dk_pool/dv_pool before they are read.
    cur_stream.wait_stream(pool_stream)
    # The pool buffers were allocated on the current stream but consumed on the
    # pool stream; mark that so the caching allocator does not recycle them early.
    dk_pool_fp32.record_stream(pool_stream)
    dv_pool_fp32.record_stream(pool_stream)

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


# ───────────────────────────────────────────────────────────────────────────
# CSA fused backward (pre-gathered top-K), one-launch full output set
# ───────────────────────────────────────────────────────────────────────────

_CSA_BWD_CACHE = {}
_CSA_BWD_LOCK = threading.Lock()


def _get_csa_bwd_full(num_heads, head_dim, swa_window, dtype_str, has_sink, has_sparse, mqa_kv):
    key = (num_heads, head_dim, swa_window, dtype_str, has_sink, has_sparse, mqa_kv)
    with _CSA_BWD_LOCK:
        if key in _CSA_BWD_CACHE:
            return _CSA_BWD_CACHE[key]
        launch = build_csa_bwd_full_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            has_sink=has_sink,
            has_sparse=has_sparse,
            block_n=32,
            block_k=32,
            mqa_kv=mqa_kv,
        )
        _CSA_BWD_CACHE[key] = launch
        return launch


def csa_attention_bwd_flydsl_kernel(
    q: torch.Tensor,  # [B, HQ, Sq, D]
    k_local: torch.Tensor,  # [B, K_H, Sq, D]   K_H in {1, HQ}
    v_local: torch.Tensor,  # [B, K_H, Sq, D]
    gathered: torch.Tensor,  # [B, Sq, K_topk, D]
    sparse_mask: torch.Tensor,  # [B, Sq, K_topk]
    out: torch.Tensor,  # [B, HQ, Sq, D]
    dout: torch.Tensor,  # [B, HQ, Sq, D]
    lse: torch.Tensor,  # [B, HQ, Sq] fp32 (raw domain)
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """CSA fused backward via the ported FlyDSL full kernel (gathered path).

    One launch emits ``(dq, dk_local, dv_local, dgathered, dsink)``. Supports
    both MQA (``K_H == 1``, shared local K/V) and MHA (``K_H == HQ``, per-head
    local K/V) via the kernel's ``mqa_kv`` flag. The sparse / gathered branch is
    head-shared, so ``dgathered`` sums over all query heads (atomic).
    """
    if q.dim() != 4 or k_local.dim() != 4 or v_local.dim() != 4 or gathered.dim() != 4:
        raise ValueError("rank-4 q/k_local/v_local/gathered required")
    if dout.shape != out.shape or out.shape != q.shape:
        raise ValueError("shape mismatch among q / out / dout")
    B, HQ, Sq, D = q.shape
    K_H = k_local.shape[1]
    K_topk = gathered.shape[2]
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("bf16 only")
    if D % 64 != 0:
        raise NotImplementedError("head_dim must be a multiple of 64")
    if K_H != 1 and K_H != HQ:
        raise NotImplementedError(f"K_H must be 1 or {HQ}; got {K_H}")
    if int(swa_window) <= 0:
        raise NotImplementedError("swa_window > 0 required")
    if K_topk <= 0:
        raise NotImplementedError("K_topk > 0 required (K_topk == 0 short-circuits to dense)")

    mqa_kv = K_H == 1
    has_sink = sink is not None
    has_sparse = K_topk > 0
    dtype_str = "bf16"

    q_c = q.contiguous()
    k_c = k_local.contiguous()
    v_c = v_local.contiguous()
    gathered_c = gathered.contiguous()
    dout_c = dout.contiguous()
    lse_c = lse.contiguous()
    # The kernel reads SPARSE_MASK in the input element type (bf16).
    sparse_mask_c = sparse_mask.to(q.dtype).contiguous()

    d_buf = _run_preprocess(out.contiguous(), dout_c)

    dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    # dk_local / dv_local are always per-head [B, HQ, Sq, D] inside the kernel;
    # the MQA caller reduces over the head axis afterwards.
    dkl_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dvl_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dgathered_fp32 = torch.zeros((B, Sq, K_topk, D), device=q.device, dtype=torch.float32)
    dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
    if has_sink:
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink.contiguous()
    else:
        sink_arg = torch.zeros((HQ,), device=q.device, dtype=torch.float32)

    launch = _get_csa_bwd_full(HQ, D, int(swa_window), dtype_str, has_sink, has_sparse, mqa_kv)
    launch(
        q_c,
        k_c,
        v_c,
        gathered_c,
        sparse_mask_c,
        dout_c,
        lse_c,
        d_buf,
        sink_arg,
        dq_fp32,
        dkl_fp32,
        dvl_fp32,
        dgathered_fp32,
        dsink_fp32,
        int(B),
        int(Sq),
        int(K_topk),
    )

    dq_out = dq_fp32.to(q.dtype)
    if mqa_kv:
        # Reduce the per-head local grads back to the shared [B, 1, Sq, D] head.
        dk_out = dkl_fp32.sum(dim=1, keepdim=True).to(k_local.dtype)
        dv_out = dvl_fp32.sum(dim=1, keepdim=True).to(v_local.dtype)
    else:
        dk_out = dkl_fp32.to(k_local.dtype)
        dv_out = dvl_fp32.to(v_local.dtype)
    dgathered_out = dgathered_fp32.to(gathered.dtype)
    dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
    return dq_out, dk_out, dv_out, dgathered_out, dsink_out


__all__ = [
    "hca_attention_bwd_flydsl_kernel",
    "csa_attention_bwd_flydsl_kernel",
]
