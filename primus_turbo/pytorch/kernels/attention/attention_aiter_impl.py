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
from aiter.ops.mha import _flash_attn_backward, _flash_attn_forward
from aiter.ops.triton.attention.mha import (
    _flash_attn_forward as _triton_flash_attn_forward,
)
from aiter.ops.triton.attention.mha_onekernel_bwd import flash_attn_onekernel_backward

from primus_turbo.pytorch.core.backend import KernelBackend
from primus_turbo.pytorch.core.utils import get_device_compute_capability


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


_SUPPORTED_QKV_FORMATS = ["sbhd", "bshd"]


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

        if sink is not None and qkv_format == "sbhd":
            # sink attention is not supported for sbhd format
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
        batch_size, seq_len, num_heads_qk, head_dim_qk = q.size()
        _, _, _, head_dim_v = v.size()

        out = None
        if sink is None:
            if qkv_format == "sbhd":
                out = torch.empty(
                    (seq_len, batch_size, num_heads_qk, head_dim_v), dtype=q.dtype, device=q.device
                ).permute(1, 0, 2, 3)
            else:
                # BSHD
                assert qkv_format == "bshd"
                out = torch.empty(
                    (batch_size, seq_len, num_heads_qk, head_dim_v), dtype=q.dtype, device=q.device
                )

            _, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
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
            assert qkv_format == "bshd", "Sink attention is not supported for sbhd format"

            if head_dim_qk != head_dim_v or not _is_power_of_2(head_dim_qk):
                raise ValueError(
                    "Triton sink attention requires head_dim_qk == head_dim_v and head_dim power-of-2"
                )

            if max_seqlen_q is None:
                max_seqlen_q = q.size(1)
            if max_seqlen_k is None:
                max_seqlen_k = k.size(1)
            window_size_left, window_size_right = _normalize_sink_window(
                causal, window_size_left, window_size_right
            )
            out, softmax_lse, S_dmask, philox_seed, philox_offset = _triton_flash_attn_forward(
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


# =============================================================================
# Backward Backends
# =============================================================================


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
        if sink is not None and qkv_format == "sbhd":
            # sink attention is not supported for sbhd format
            return False

        supported = qkv_format in _SUPPORTED_QKV_FORMATS
        # NOTE: gfx942 has numerical issue in fp16 atomic when layout is sbhd.
        if get_device_compute_capability() == (9, 4):
            supported &= not (qkv_format == "sbhd" and not is_v3_atomic_fp32)

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
            result = _flash_attn_backward(
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
            assert qkv_format == "bshd", "Sink attention is not supported for sbhd format"

            assert (
                isinstance(rng_state, torch.Tensor)
                and rng_state.device.type == "cpu"
                and rng_state.dtype == torch.int64
                and rng_state.numel() == 2
            ), "Triton backward requires rng_state to be a CPU int64 tensor of shape [2]"
            philox_seed = int(rng_state[0].item())
            philox_offset = int(rng_state[1].item())

            result = flash_attn_onekernel_backward(
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
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
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
            f"AttnFwdAiterBackend cannot handle the given inputs. "
            f"Please check input constraints or choose a different backend."
        )

    return AttnFwdAiterBackend.execute(**kwargs)


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
):
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
            f"AttnBwdAiterBackend cannot handle the given inputs. "
            f"Please check input constraints or choose a different backend."
        )

    return AttnBwdAiterBackend.execute(**kwargs)
