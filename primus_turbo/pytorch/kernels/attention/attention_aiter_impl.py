###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unified Aiter kernel dispatch for attention forward/backward.

Dispatch policy:
- Forward: use csrc when sink is None; use triton when sink is not None
- Backward: use csrc when sink is None; use triton when sink is not None
"""

from typing import Any, Optional, Tuple

import torch

from primus_turbo.common.aiter_utils import (
    check_aiter_version_once,
    raise_aiter_missing,
)
from primus_turbo.pytorch.core.backend import KernelBackend

_AITER_ATTN_KERNELS = None


def _get_aiter_attn_kernels():
    """Lazily import and cache the aiter attention kernels (see aiter_utils)."""
    global _AITER_ATTN_KERNELS
    if _AITER_ATTN_KERNELS is None:
        try:
            from aiter.ops.mha import (
                _flash_attn_backward,
                _flash_attn_forward,
                _flash_attn_varlen_backward,
                _flash_attn_varlen_forward,
            )
            from aiter.ops.triton.attention.mha import (
                _flash_attn_forward as _triton_flash_attn_forward,
            )
            from aiter.ops.triton.attention.mha_onekernel_bwd import (
                flash_attn_onekernel_backward as _triton_flash_attn_onekernel_backward,
            )
        except ImportError as exc:
            raise_aiter_missing(exc)
        check_aiter_version_once()
        _AITER_ATTN_KERNELS = {
            "flash_attn_forward": _flash_attn_forward,
            "flash_attn_backward": _flash_attn_backward,
            "flash_attn_varlen_forward": _flash_attn_varlen_forward,
            "flash_attn_varlen_backward": _flash_attn_varlen_backward,
            "triton_flash_attn_forward": _triton_flash_attn_forward,
            "triton_flash_attn_onekernel_backward": _triton_flash_attn_onekernel_backward,
        }
    return _AITER_ATTN_KERNELS


_torch_custom_op_wrapper = torch.library.custom_op


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _normalize_sink_window(causal: bool, window_size_left: int, window_size_right: int) -> Tuple[int, int]:
    """Map GPT-OSS style causal window_size=(left, 0) to the aiter Triton sentinel."""
    if causal and window_size_right == 0:
        return window_size_left, -1
    return window_size_left, window_size_right


# =============================================================================
# Forward Backend
# =============================================================================


_SUPPORTED_QKV_FORMATS = ["sbhd", "bshd", "bhsd"]


class AttnFwdAiterBackend(KernelBackend):

    @staticmethod
    def can_handle(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        return_lse: bool,
        return_softmax: bool,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
        qkv_format: Optional[str] = "bshd",
    ) -> bool:

        if sink is not None:
            head_dim_qk = q.size(-1)
            head_dim_v = v.size(-1)
            if head_dim_qk != head_dim_v or not _is_power_of_2(head_dim_qk):
                return False

        supported = qkv_format in _SUPPORTED_QKV_FORMATS

        return supported

    @staticmethod
    def execute(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        return_lse: bool,
        return_softmax: bool,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
        qkv_format: Optional[str] = "sbhd",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
        batch_size, seq_len, num_heads_qk, _ = q.size()
        _, _, _, head_dim_v = v.size()

        if qkv_format == "sbhd":
            out = torch.empty(
                (seq_len, batch_size, num_heads_qk, head_dim_v), dtype=q.dtype, device=q.device
            ).permute(1, 0, 2, 3)
        elif qkv_format == "bhsd":
            out = torch.empty(
                (batch_size, num_heads_qk, seq_len, head_dim_v), dtype=q.dtype, device=q.device
            ).permute(0, 2, 1, 3)
        else:
            out = torch.empty((batch_size, seq_len, num_heads_qk, head_dim_v), dtype=q.dtype, device=q.device)

        if sink is None:
            _, softmax_lse, S_dmask, rng_state = _get_aiter_attn_kernels()["flash_attn_forward"](
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                0,  # sink_size
                bias,
                alibi_slopes,
                None,  # q_descale
                None,  # k_descale
                None,  # v_descale
                return_lse,
                return_softmax,
                out=out,
            )
        else:
            if max_seqlen_q is None:
                max_seqlen_q = q.size(1)
            if max_seqlen_k is None:
                max_seqlen_k = k.size(1)

            window_size_left, window_size_right = _normalize_sink_window(
                causal, window_size_left, window_size_right
            )

            out, softmax_lse, S_dmask, philox_seed, philox_offset = _get_aiter_attn_kernels()[
                "triton_flash_attn_forward"
            ](
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                bias,
                alibi_slopes,
                return_lse,
                return_softmax,
                max_seqlen_q,
                max_seqlen_k,
                sink=sink,
            )
            rng_state = torch.tensor([philox_seed, philox_offset], dtype=torch.int64, device="cpu")

        return out, softmax_lse, S_dmask, rng_state


class AttnBwdAiterBackend(KernelBackend):

    @staticmethod
    def can_handle(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        dbias: Optional[torch.Tensor],
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        rng_state: Optional[torch.Tensor],
        is_v3_atomic_fp32: bool,
        how_v3_bf16_cvt: int,
        sink: Optional[torch.Tensor] = None,
        dsink: Optional[torch.Tensor] = None,
        qkv_format: Optional[str] = "bshd",
    ) -> bool:
        if sink is not None:
            head_dim_qk = q.size(-1)
            head_dim_v = v.size(-1)
            if head_dim_qk != head_dim_v or not _is_power_of_2(head_dim_qk):
                return False

        supported = qkv_format in _SUPPORTED_QKV_FORMATS

        return supported

    @staticmethod
    def execute(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        dbias: Optional[torch.Tensor],
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        rng_state: Optional[torch.Tensor],
        is_v3_atomic_fp32: bool,
        how_v3_bf16_cvt: int,
        sink: Optional[torch.Tensor] = None,
        dsink: Optional[torch.Tensor] = None,
        qkv_format: Optional[str] = "bshd",
    ):
        if sink is None:
            result = _get_aiter_attn_kernels()["flash_attn_backward"](
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                bias,
                alibi_slopes,
                deterministic,
                rng_state,
                is_v3_atomic_fp32,
                how_v3_bf16_cvt,
            )
        else:
            assert (
                isinstance(rng_state, torch.Tensor)
                and rng_state.device.type == "cpu"
                and rng_state.dtype == torch.int64
                and rng_state.numel() == 2
            ), "Triton backward requires rng_state to be a CPU int64 tensor of shape [2]"
            philox_seed = int(rng_state[0].item())
            philox_offset = int(rng_state[1].item())

            result = _get_aiter_attn_kernels()["triton_flash_attn_onekernel_backward"](
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                softmax_scale,
                alibi_slopes,
                causal,
                None,  # cu_seqlens_q
                None,  # cu_seqlens_k
                q.size(1),  # max_seqlen_q
                k.size(1),  # max_seqlen_k
                dropout_p,
                philox_seed,
                philox_offset,
                USE_INT64_STRIDES=True,
                sink=sink,
                dsink=dsink,
                sliding_window=window_size_left if window_size_left >= 0 else 0,
            )

        return (
            result,
            dq,
            dk,
            dv,
            dbias,
            dsink,
        )


@_torch_custom_op_wrapper("primus_turbo::attention_aiter_forward_impl", mutates_args=(), device_types="cuda")
def attention_aiter_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    sink: Optional[torch.Tensor] = None,
    qkv_format: Optional[str] = "bshd",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
        "bias": bias,
        "alibi_slopes": alibi_slopes,
        "return_lse": return_lse,
        "return_softmax": return_softmax,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "sink": sink,
        "qkv_format": qkv_format,
    }
    # TODO(ruibin): Add unified attention kernel dispatcher
    if not AttnFwdAiterBackend.can_handle(**kwargs):
        raise ValueError(
            f"AttnFwdAiterBackend cannot handle the given inputs: {_format_kwargs(kwargs)}. "
            f"Please check input constraints or choose a different backend."
        )

    return AttnFwdAiterBackend.execute(**kwargs)


@attention_aiter_forward_impl.register_fake
def _attention_aiter_forward_impl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    sink: Optional[torch.Tensor] = None,
    qkv_format: Optional[str] = "bshd",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len_q, num_heads_q, _ = q.shape
    _, _, _, head_dim_v = v.shape
    if qkv_format == "sbhd":
        out = torch.empty(
            (seq_len_q, batch_size, num_heads_q, head_dim_v), dtype=q.dtype, device=q.device
        ).permute(1, 0, 2, 3)
    elif qkv_format == "bhsd":
        out = torch.empty(
            (batch_size, num_heads_q, seq_len_q, head_dim_v), dtype=q.dtype, device=q.device
        ).permute(0, 2, 1, 3)
    else:
        out = torch.empty((batch_size, seq_len_q, num_heads_q, head_dim_v), dtype=q.dtype, device=q.device)
    softmax_lse = torch.empty((batch_size, num_heads_q, seq_len_q), dtype=torch.float32, device=q.device)
    S_dmask = torch.empty((0,), dtype=q.dtype, device=q.device)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, S_dmask, rng_state


@_torch_custom_op_wrapper(
    "primus_turbo::attention_aiter_backward_impl",
    mutates_args=("dq", "dk", "dv"),
    device_types="cuda",
)
def attention_aiter_backward_impl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor],
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dbias: Optional[torch.Tensor] = None,
    dsink: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    qkv_format: Optional[str] = "bshd",
) -> None:
    kwargs = {
        "dout": dout,
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "softmax_lse": softmax_lse,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
        "bias": bias,
        "alibi_slopes": alibi_slopes,
        "deterministic": deterministic,
        "rng_state": rng_state,
        "is_v3_atomic_fp32": is_v3_atomic_fp32,
        "how_v3_bf16_cvt": how_v3_bf16_cvt,
        "dbias": dbias,
        "dsink": dsink,
        "sink": sink,
        "qkv_format": qkv_format,
    }

    # TODO(ruibin): Add unified attention kernel dispatcher
    if not AttnBwdAiterBackend.can_handle(**kwargs):
        raise ValueError(
            f"AttnBwdAiterBackend cannot handle the given inputs: {_format_kwargs(kwargs)}. "
            f"Please check input constraints or choose a different backend."
        )

    AttnBwdAiterBackend.execute(**kwargs)


@attention_aiter_backward_impl.register_fake
def _attention_aiter_backward_impl_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor],
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dbias: Optional[torch.Tensor] = None,
    dsink: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
    qkv_format: Optional[str] = "bshd",
) -> None:
    return None


# =============================================================================
# Varlen Forward Backend
# =============================================================================


class AttnFwdAiterVarlenBackend(KernelBackend):

    @staticmethod
    def can_handle(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        return_lse: bool,
        return_softmax: bool,
    ) -> bool:
        if q.dtype not in (torch.bfloat16, torch.float16):
            return False
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            return False
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            return False
        return True

    @staticmethod
    def execute(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        return_lse: bool,
        return_softmax: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out, softmax_lse, S_dmask, rng_state = _get_aiter_attn_kernels()["flash_attn_varlen_forward"](
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_q_padded
            None,  # cu_seqlens_k_padded
            max_seqlen_q,
            max_seqlen_k,
            0,  # min_seqlen_q
            dropout_p,
            softmax_scale,
            causal=causal,
            logits_soft_cap=0.0,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            sink_size=0,
            bias=bias,
            alibi_slopes=alibi_slopes,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            return_lse=return_lse,
            return_softmax=return_softmax,
            how_v3_bf16_cvt=1,
            block_table=None,
            out=None,
        )
        return out, softmax_lse, S_dmask, rng_state


# =============================================================================
# Varlen Backward Backend
# =============================================================================


class AttnBwdAiterVarlenBackend(KernelBackend):

    @staticmethod
    def can_handle(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        rng_state: Optional[torch.Tensor],
        is_v3_atomic_fp32: bool,
        how_v3_bf16_cvt: int,
    ) -> bool:
        if q.dtype not in (torch.bfloat16, torch.float16):
            return False
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
            return False
        return True

    @staticmethod
    def execute(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        rng_state: Optional[torch.Tensor],
        is_v3_atomic_fp32: bool,
        how_v3_bf16_cvt: int,
    ):
        _get_aiter_attn_kernels()["flash_attn_varlen_backward"](
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            alibi_slopes,
            deterministic,
            rng_state=rng_state,
            is_v3_atomic_fp32=is_v3_atomic_fp32,
            how_v3_bf16_cvt=how_v3_bf16_cvt,
            cu_seqlens_q_padded=None,
            cu_seqlens_k_padded=None,
        )


@_torch_custom_op_wrapper(
    "primus_turbo::attention_aiter_varlen_forward_impl", mutates_args=(), device_types="cuda"
)
def attention_aiter_varlen_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
        "bias": bias,
        "alibi_slopes": alibi_slopes,
        "return_lse": return_lse,
        "return_softmax": return_softmax,
    }
    if not AttnFwdAiterVarlenBackend.can_handle(**kwargs):
        raise ValueError(
            f"AttnFwdAiterVarlenBackend cannot handle the given inputs: {_format_kwargs(kwargs)}. "
            f"Varlen requires bf16/fp16, 3-D q/k/v in thd layout, and int32 cu_seqlens."
        )
    return AttnFwdAiterVarlenBackend.execute(**kwargs)


@attention_aiter_varlen_forward_impl.register_fake
def _attention_aiter_varlen_forward_impl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_q, nheads_q, _ = q.shape
    head_dim_v = v.shape[-1]
    out = torch.empty((total_q, nheads_q, head_dim_v), dtype=q.dtype, device=q.device)
    softmax_lse = torch.empty((nheads_q, total_q), dtype=torch.float32, device=q.device)
    S_dmask = torch.empty((0,), dtype=q.dtype, device=q.device)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, S_dmask, rng_state


@_torch_custom_op_wrapper(
    "primus_turbo::attention_aiter_varlen_backward_impl",
    mutates_args=("dq", "dk", "dv"),
    device_types="cuda",
)
def attention_aiter_varlen_backward_impl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor],
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
) -> None:
    kwargs = {
        "dout": dout,
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "softmax_lse": softmax_lse,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
        "alibi_slopes": alibi_slopes,
        "deterministic": deterministic,
        "rng_state": rng_state,
        "is_v3_atomic_fp32": is_v3_atomic_fp32,
        "how_v3_bf16_cvt": how_v3_bf16_cvt,
    }
    if not AttnBwdAiterVarlenBackend.can_handle(**kwargs):
        raise ValueError(
            f"AttnBwdAiterVarlenBackend cannot handle the given inputs: {_format_kwargs(kwargs)}."
        )
    AttnBwdAiterVarlenBackend.execute(**kwargs)


@attention_aiter_varlen_backward_impl.register_fake
def _attention_aiter_varlen_backward_impl_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor],
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
) -> None:
    return None
