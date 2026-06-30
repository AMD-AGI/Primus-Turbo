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

from primus_turbo.flydsl.attention.kernels.csa_bwd_dkv_kernel import build_csa_bwd_dkv_module
from primus_turbo.flydsl.attention.kernels.csa_bwd_dq_finalize_kernel import (
    build_csa_bwd_dgathered_finalize_module,
)
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


# ───────────────────────────────────────────────────────────────────────────
# CSA fused backward (pre-gathered top-K), one-launch full output set
# ───────────────────────────────────────────────────────────────────────────

_CSA_BWD_CACHE = {}
_CSA_BWD_LOCK = threading.Lock()
# Dedicated MFMA matrix-core kernel for the LOCAL-branch dk_local / dv_local
# contractions (split out of csa_bwd_full).
_CSA_BWD_DKV_CACHE = {}
_CSA_BWD_DKV_LOCK = threading.Lock()
# dgathered split-K finalize reduce (sums the per-head-group bf16 stripes of
# the packed-bf16 split scratch back into the fp32 dgathered).
_CSA_BWD_DGFIN_CACHE = {}
_CSA_BWD_DGFIN_LOCK = threading.Lock()


# Mirror the csa_bwd_full head-grouping (head_block=4, gated on MQA and
# HQ % 4 == 0) so the launcher can size the disjoint split scratch's group axis.
def _csa_bwd_num_head_groups(num_heads, mqa_kv):
    hb_req = 4
    if mqa_kv and (int(num_heads) % hb_req == 0):
        head_block = hb_req
    else:
        head_block = 1
    return int(num_heads) // head_block


def _get_csa_bwd_dgfinalize(head_dim, num_groups, dtype_str):
    key = (head_dim, num_groups, dtype_str)
    with _CSA_BWD_DGFIN_LOCK:
        if key in _CSA_BWD_DGFIN_CACHE:
            return _CSA_BWD_DGFIN_CACHE[key]
        launch = build_csa_bwd_dgathered_finalize_module(
            head_dim=head_dim,
            num_groups=num_groups,
            dtype_str=dtype_str,
        )
        _CSA_BWD_DGFIN_CACHE[key] = launch
        return launch


def _get_csa_bwd_full(num_heads, head_dim, swa_window, dtype_str, has_sink, has_sparse, mqa_kv):
    key = (num_heads, head_dim, swa_window, dtype_str, has_sink, has_sparse, mqa_kv)
    with _CSA_BWD_LOCK:
        if key in _CSA_BWD_CACHE:
            return _CSA_BWD_CACHE[key]
        # Head-group ownership sizing along the axis that carries the dgathered
        # atomic contention. DGATHERED is dense [B, Sq, K_topk, D] with NO head
        # axis, so every one of the NUM_HEAD_GROUPS = HQ/HEAD_BLOCK head-group
        # programs atomic_fadd's into the SAME [b, q, k_pos, d] word; the
        # in-register pre-sum only collapses the HEAD_BLOCK heads WITHIN a
        # program, leaving HQ/HEAD_BLOCK groups still serializing on the RMW and
        # re-reading the head-shared gathered g_afrag/g_vec lines once per group.
        # HEAD_BLOCK=4 with QROW_BLOCK=1 keeps the live VGPR footprint neutral
        # (the dq_accs / qk-dp caches scale with the HB*QR product, 4*1 == 2*2)
        # and keeps the LOCAL per-tile K/V reuse count identical (4 consumers
        # either way: 4 heads x 1 row vs 2 heads x 2 rows), but (a) doubles the
        # head-shared GATHERED line reuse within one program (g_afrag/g_vec
        # fetched once, reused across 4 heads instead of 2 -> higher L2 hit rate
        # on the dense gathered tensor) and (b) halves the cross-head-group
        # dgathered atomic contention (HQ/4 contending groups instead of HQ/2 --
        # single-owner-style in-register accumulation over twice as many heads
        # before one atomic_fadd). Gated inside the build on mqa_kv and
        # NUM_HEADS % 4 == 0; MHA / indivisible head counts fall back to
        # HEAD_BLOCK=1.
        launch = build_csa_bwd_full_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            has_sink=has_sink,
            has_sparse=has_sparse,
            block_n=32,
            block_k=32,
            qrow_block=1,
            head_block=4,
            mqa_kv=mqa_kv,
        )
        _CSA_BWD_CACHE[key] = launch
        return launch


def _get_csa_bwd_dkv(num_heads, head_dim, swa_window, dtype_str, mqa_kv):
    key = (num_heads, head_dim, swa_window, dtype_str, mqa_kv)
    with _CSA_BWD_DKV_LOCK:
        if key in _CSA_BWD_DKV_CACHE:
            return _CSA_BWD_DKV_CACHE[key]
        launch = build_csa_bwd_dkv_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
        )
        _CSA_BWD_DKV_CACHE[key] = launch
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
    # Warp-disjoint packed-bf16 split-K scratch for the gathered gradient.
    # The full kernel writes its disjoint head-group stripe with a
    # plain packed 2xbf16 store (no atomic_fadd RMW); the finalize reduce sums
    # the NUM_HEAD_GROUPS stripes back into dgathered_fp32.
    num_head_groups = _csa_bwd_num_head_groups(HQ, mqa_kv)
    dgathered_split_bf16 = torch.zeros(
        (B, Sq, K_topk, num_head_groups, D), device=q.device, dtype=q.dtype
    )
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
        dgathered_split_bf16,
        dsink_fp32,
        int(B),
        int(Sq),
        int(K_topk),
    )

    # Finalize: sum the NUM_HEAD_GROUPS disjoint bf16 stripes of the split
    # scratch into the fp32 dgathered (conflict-free element-parallel pass).
    dgfin_launch = _get_csa_bwd_dgfinalize(D, int(num_head_groups), dtype_str)
    dgfin_launch(
        dgathered_split_bf16.view(-1, num_head_groups * D),
        dgathered_fp32.view(-1, D),
        int(B * Sq * K_topk * D),
    )

    # The LOCAL-branch dk_local / dv_local contractions
    # (dV_local = P^T @ dO, dK_local = dS^T @ Q) are produced here by the
    # dedicated MFMA matrix-core kernel instead of the VALU atomic path inside
    # csa_bwd_full. Direct-stores per-head [B, HQ, Sq, D] (no atomics); the
    # buffers were zero-initialised above and csa_bwd_full no longer writes
    # them. Sq == Sk for the local branch.
    dkv_launch = _get_csa_bwd_dkv(HQ, D, int(swa_window), dtype_str, mqa_kv)
    dkv_launch(
        q_c,
        k_c,
        v_c,
        dout_c,
        lse_c,
        d_buf,
        dkl_fp32,
        dvl_fp32,
        int(B),
        int(Sq),
        int(Sq),
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
