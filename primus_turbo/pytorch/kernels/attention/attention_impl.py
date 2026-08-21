###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unified multi-backend selection for dense flash-attention (bf16).

Mirrors the GEMM convention (``kernels/gemm/gemm_impl.py``): ``KernelBackend`` subclasses
registered in an ``AutoKernelDispatcher``, selected by ``GlobalBackendManager`` /
``PRIMUS_TURBO_ATTN_BACKEND`` / autotune.

Unlike GEMM, a forward+backward pair must run on the *same* backend (saved-tensor and LSE
conventions differ), so the op layer resolves once per call and carries the enum on ctx.
``execute`` here runs the forward only; it exists so autotune can time each backend.
FP8 (triton) stays on its own ``flash_attn_fp8_func`` path.
"""

import math
from typing import Optional

import torch

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendEntry,
    BackendType,
    KernelBackend,
    TuneCache,
)
from primus_turbo.pytorch.core.utils import get_device_compute_capability
from primus_turbo.pytorch.kernels.attention.attention_aiter_impl import (
    attention_aiter_forward_impl,
    attention_aiter_varlen_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_forward_impl,
    flash_attn_varlen_flydsl_forward_impl,
)
from primus_turbo.pytorch.kernels.attention.attention_hipkittens_impl import (
    flash_attn_sbhd_hipkittens_forward_impl,
    hipkittens_attn_supported_impl,
)

_GFX950 = (9, 5)
# Narrowest left window the FlyDSL backward is correct on. The kv band comes from the window
# rounded down to a power of two, and the dQ reduce takes the low edge of a whole q BLOCK's
# band range. Below BLOCK_Q=64 a band is narrower than the block, so odd bands start mid-block
# and the reduce sums workspace the body never wrote (NaN). Narrower windows go to aiter.
_MIN_FLYDSL_WINDOW = 64


def _scale_ok(softmax_scale: Optional[float], head_dim: int) -> bool:
    """FlyDSL bakes softmax_scale = 1/sqrt(D); accept only None or that value."""
    return softmax_scale is None or abs(softmax_scale - 1.0 / math.sqrt(head_dim)) < 1e-6


def _sink_ok(sink: Optional[torch.Tensor], num_heads_q: int) -> bool:
    """A sink must be fp32 [Hq] for FlyDSL to fold it into the softmax denominator; None
    (no sink) is fine, anything else falls back to aiter."""
    return sink is None or (sink.dtype == torch.float32 and sink.numel() == num_heads_q)


def _gqa_group_ok(num_heads_q: int, num_heads_kv: int) -> bool:
    """G = Hq // Hkv must be a power of two in [1, 256], so it divides BLOCK_SIZE for the
    dkdv backward's cooperative (delta, lse) stage. Anything else falls back to aiter."""
    if num_heads_kv <= 0 or num_heads_q % num_heads_kv != 0:
        return False
    g = num_heads_q // num_heads_kv
    return 1 <= g <= 256 and (g & (g - 1)) == 0


def _flydsl_common_ok(
    q: torch.Tensor,
    causal: bool,
    window_size,
    softmax_scale: Optional[float],
    dropout_p: float,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
) -> bool:
    """Shared FlyDSL eligibility gate (gfx950, causal, D in {64,128}, bf16, ...).

    ``sink`` and the GQA group are checked separately, where Hq is known.
    """
    head_dim = q.shape[-1]
    return (
        get_device_compute_capability() >= _GFX950
        and bool(causal)
        and q.dtype == torch.bfloat16
        and head_dim in (64, 128)
        and int(window_size[1]) in (0, -1)
        and (int(window_size[0]) < 0 or int(window_size[0]) >= _MIN_FLYDSL_WINDOW)
        and _scale_ok(softmax_scale, head_dim)
        and dropout_p == 0.0
        and bias is None
        and alibi_slopes is None
    )


# =============================================================================
# Dense (bshd / sbhd / bhsd) forward backends
# =============================================================================


class DenseAttnFwdAiterBackend(KernelBackend):
    @staticmethod
    def can_handle(q: torch.Tensor, **kwargs) -> bool:
        return q.dtype in (torch.float16, torch.bfloat16) and q.ndim == 4

    @staticmethod
    def execute(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        sink,
        qkv_format,
        **kwargs,
    ):
        return attention_aiter_forward_impl(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=False,
            max_seqlen_q=q.size(1),
            max_seqlen_k=k.size(1),
            sink=sink,
            qkv_format=qkv_format,
        )


def _sbhd_layout(q: torch.Tensor, qkv_format: str) -> bool:
    """Whether q's bytes are in sbhd order, the one the FlyDSL kernels address.

    A batch of one is sbhd and bshd at once -- the same bytes, and strides that cannot name
    one -- so _infer_qkv_format has to break the tie and breaks it toward bshd. Reading only
    its answer would send every b == 1 shape to aiter, which for a prefill batch of one is
    most of them. bhsd is a different order at any batch and is not covered by this.
    """
    return qkv_format == "sbhd" or (qkv_format == "bshd" and q.shape[0] == 1)


class DenseAttnFwdFlydslBackend(KernelBackend):
    @staticmethod
    def can_handle(
        q,
        k=None,
        v=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        sink=None,
        qkv_format="bshd",
        return_softmax=False,
        **kwargs,
    ) -> bool:
        # sbhd only: the kernel is compiled to address that order and takes the [s,b,h,d]
        # view of these [b,s,h,d]-shaped tensors with no copy. Everything else goes to aiter.
        if k is None or v is None or not _sbhd_layout(q, qkv_format):
            return False
        # These kernels never materialise the dropout softmax matrix, so a caller that asked
        # for it has to go to aiter, which does.
        if return_softmax:
            return False
        if not _gqa_group_ok(q.shape[2], k.shape[2]) or not _sink_ok(sink, q.shape[2]):
            return False
        return _flydsl_common_ok(q, causal, window_size, softmax_scale, dropout_p, bias, alibi_slopes)

    @staticmethod
    def execute(q, k, v, softmax_scale, causal, window_size, return_lse=True, **kwargs):
        # Under autotune, off the op layer's path, so the sbhd view is taken here too.
        q, k, v = (t.permute(1, 0, 2, 3) for t in (q, k, v))
        return flash_attn_sbhd_flydsl_forward_impl(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_lse=return_lse,
        )


class DenseAttnFwdHipkittensBackend(KernelBackend):
    """HipKittens attention, gfx950 only.

    A narrow backend by construction: bf16, causal, sbhd, head dim 64 or 128, Sq <= Skv, no
    sink and no varlen. Everything outside that is refused here rather than computed wrongly
    -- these kernels read out of bounds or leave output unwritten instead of failing -- so a
    shape that does not qualify falls back to whichever backend does.
    """

    @staticmethod
    def can_handle(
        q,
        k=None,
        v=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        sink=None,
        qkv_format="bshd",
        return_softmax=False,
        **kwargs,
    ) -> bool:
        # Same layout gate as FlyDSL: the kernels address sbhd and take the [s,b,h,d] view of
        # these [b,s,h,d]-shaped tensors with no copy, so the bytes have to already be in that
        # order. _sbhd_layout is what decides that, rather than a stride test, because at
        # b == 1 the two orders are indistinguishable.
        if k is None or v is None or not _sbhd_layout(q, qkv_format):
            return False
        # No dropout softmax matrix here either, for the same reason as FlyDSL above.
        if return_softmax:
            return False
        # Ask the backend rather than restating its rules here, so the two cannot drift. It
        # answers on the sbhd view, which is what it will be handed.
        qs, ks, vs = (t.permute(1, 0, 2, 3) for t in (q, k, v))
        ok, _ = hipkittens_attn_supported_impl(
            qs,
            ks,
            vs,
            causal=causal,
            window_size=window_size,
            sink=sink,
            dropout_p=dropout_p,
            bias=bias,
            alibi_slopes=alibi_slopes,
        )
        return ok

    @staticmethod
    def execute(q, k, v, softmax_scale, causal, window_size, return_lse=True, **kwargs):
        # Under autotune, off the op layer's path, so the sbhd view is taken here too.
        q, k, v = (t.permute(1, 0, 2, 3) for t in (q, k, v))
        res = flash_attn_sbhd_hipkittens_forward_impl(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_lse=return_lse,
        )
        # Back to the [b, s, h, d] view the caller handed us.
        if return_lse:
            out, lse = res
            return out.permute(1, 0, 2, 3), lse
        return res.permute(1, 0, 2, 3)


_DENSE_FWD_BACKENDS = {
    BackendType.FLYDSL: BackendEntry(DenseAttnFwdFlydslBackend),
    BackendType.AITER: BackendEntry(DenseAttnFwdAiterBackend),
    BackendType.HIPKITTENS: BackendEntry(DenseAttnFwdHipkittensBackend),
}


class FlashAttnDenseDispatcher(AutoKernelDispatcher):
    _backends = _DENSE_FWD_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls,
        q,
        k,
        causal=True,
        window_size=(-1, -1),
        qkv_format="bshd",
        sink=None,
        return_softmax=False,
        **kwargs,
    ):
        b, s, hq, d = q.shape
        hkv = k.shape[2]
        # return_softmax belongs in the key because it decides eligibility, not just output:
        # only aiter can produce the dropout matrix, so a tuned entry cached without it would
        # hand a return_softmax call to a backend that silently drops it.
        return (
            b,
            s,
            hq,
            hkv,
            d,
            q.dtype,
            bool(causal),
            tuple(window_size),
            qkv_format,
            sink is not None,
            bool(return_softmax),
        )


# =============================================================================
# Varlen (thd) forward backends
# =============================================================================


def _uniform_cu_seqlens(cu_seqlens_q, cu_seqlens_k) -> bool:
    """True when every per-batch segment length is equal (FlyDSL varlen needs it)."""
    for cu in (cu_seqlens_q, cu_seqlens_k):
        seg = cu[1:] - cu[:-1]
        if seg.numel() == 0 or not bool((seg == seg[0]).all().item()):
            return False
    return True


class VarlenAttnFwdAiterBackend(KernelBackend):
    @staticmethod
    def can_handle(q: torch.Tensor, sink=None, **kwargs) -> bool:
        # No sink: the aiter varlen kernels take none, and the dense ones do, so saying so
        # here is what keeps a varlen sink from being dropped on the floor.
        return q.dtype in (torch.float16, torch.bfloat16) and q.ndim == 3 and sink is None

    @staticmethod
    def execute(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        **kwargs,
    ):
        return attention_aiter_varlen_forward_impl(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=True,
            return_softmax=False,
        )


class VarlenAttnFwdFlydslBackend(KernelBackend):
    @staticmethod
    def can_handle(
        q,
        k=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
        window_size=(-1, -1),
        bias=None,
        alibi_slopes=None,
        sink=None,
        **kwargs,
    ) -> bool:
        # varlen THD: q [total, Hq, D], k [total, Hkv, D].
        if k is None or not _gqa_group_ok(q.shape[1], k.shape[1]) or not _sink_ok(sink, q.shape[1]):
            return False
        if not _flydsl_common_ok(q, causal, window_size, softmax_scale, dropout_p, bias, alibi_slopes):
            return False
        return _uniform_cu_seqlens(cu_seqlens_q, cu_seqlens_k)

    @staticmethod
    def execute(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        return_lse=True,
        **kwargs,
    ):
        return flash_attn_varlen_flydsl_forward_impl(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            return_lse=return_lse,
        )


_VARLEN_FWD_BACKENDS = {
    BackendType.FLYDSL: BackendEntry(VarlenAttnFwdFlydslBackend),
    BackendType.AITER: BackendEntry(VarlenAttnFwdAiterBackend),
}


class FlashAttnVarlenDispatcher(AutoKernelDispatcher):
    _backends = _VARLEN_FWD_BACKENDS
    _cache = TuneCache(1024)

    @classmethod
    def make_key(
        cls, q, k, causal=True, window_size=(-1, -1), max_seqlen_q=0, max_seqlen_k=0, sink=None, **kwargs
    ):
        total_q, hq, d = q.shape
        total_k, hkv, _ = k.shape
        return (
            total_q,
            total_k,
            hq,
            hkv,
            d,
            q.dtype,
            bool(causal),
            tuple(window_size),
            int(max_seqlen_q),
            int(max_seqlen_k),
            sink is not None,
        )


# =============================================================================
# Backend resolution (used by the op layer to pick fwd == bwd backend)
# =============================================================================


def resolve_flash_attn_backend(varlen: bool, user_backend: Optional[BackendType], **kwargs) -> BackendType:
    """Resolve the dense/varlen flash-attn backend enum (default FLYDSL, else AITER)."""
    dispatcher = FlashAttnVarlenDispatcher if varlen else FlashAttnDenseDispatcher
    return dispatcher.resolve(BackendType.FLYDSL, user_backend, **kwargs)
