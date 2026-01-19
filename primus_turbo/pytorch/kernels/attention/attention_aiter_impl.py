###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unified Aiter kernel dispatch for attention forward/backward.

Dispatch policy (following AutoKernelDispatcher):
1. User specified backend (env or code) - highest priority
2. Auto tune (if enabled)
3. Default backend (AITER/C++)
4. Fallback: try all backends

Benefits:
- Clearly separates responsibilities between different implementations
- Avoids branching and duplicated logic at the Python layer
- Enables simple and effective reuse of aiter-triton in CP scenarios
"""

from typing import Any, Optional, Tuple

import torch
from aiter.ops.mha import _flash_attn_backward, _flash_attn_forward
from aiter.ops.triton.attention.mha import (
    _flash_attn_forward as _triton_flash_attn_forward,
)
from aiter.ops.triton.attention.mha_onekernel_bwd import flash_attn_onekernel_backward

from primus_turbo.pytorch.core.backend import (
    AutoKernelDispatcher,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    TuneCache,
)


def _is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


# =============================================================================
# Forward Backends
# =============================================================================


class AttnFwdAiterCsrcBackend(KernelBackend):
    """C++ backend for attention forward using aiter csrc implementation."""

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
    ) -> bool:
        # C++ backend does not support sink attention
        if sink is not None:
            return False
        return True

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
        max_seqlen_q: Optional[int] = None,  # same as Triton backend
        max_seqlen_k: Optional[int] = None,
        sink: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:

        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
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
        )
        # Return format: (out, lse, S_dmask, state_info)
        # state_info is ("csrc", rng_state, philox_seed=0, philox_offset=0) for csrc backend
        return out_padded, softmax_lse, S_dmask, ("csrc", rng_state, 0, 0)


class AttnFwdAiterTritonBackend(KernelBackend):
    """Triton backend for attention forward using aiter triton implementation."""

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
    ) -> bool:
        # Triton requires:
        # 1. head_dim_qk == head_dim_v
        # 2. head_dim must be a power of 2
        head_dim_qk = q.shape[-1]
        head_dim_v = v.shape[-1]
        if head_dim_qk != head_dim_v:
            return False
        if not _is_power_of_2(head_dim_qk):
            return False
        return True

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
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
        if max_seqlen_q is None:
            max_seqlen_q = q.shape[1]
        if max_seqlen_k is None:
            max_seqlen_k = k.shape[1]

        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = _triton_flash_attn_forward(
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
        # Return format: (out, lse, S_dmask, state_info)
        # state_info is ("triton", sink, philox_seed, philox_offset) for triton backend
        return out_padded, softmax_lse, S_dmask, ("triton", sink, philox_seed, philox_offset)


_ATTN_FWD_AITER_BACKENDS = {
    BackendType.AITER: AttnFwdAiterCsrcBackend,
    BackendType.TRITON: AttnFwdAiterTritonBackend,
}


class AttnFwdAiterKernelDispatcher(AutoKernelDispatcher):
    """
    Dispatcher for attention forward with Aiter backends.

    Uses AutoKernelDispatcher dispatch logic:
    1. User specified backend (env or code) - highest priority
    2. Auto tune (if enabled)
    3. Default backend (AITER/C++)
    4. Fallback: try all backends
    """

    _backends = _ATTN_FWD_AITER_BACKENDS
    _cache = TuneCache(256)

    @classmethod
    def make_key(cls, q, k, v, causal, sink, **kwargs):
        # Key based on shapes and key config options
        return (
            q.shape,
            k.shape,
            v.shape,
            q.dtype,
            causal,
            sink is not None,
        )


# =============================================================================
# Backward Backends
# =============================================================================


class AttnBwdAiterCsrcBackend(KernelBackend):
    """C++ backend for attention backward using aiter csrc implementation."""

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
        philox_seed: int = 0,
        philox_offset: int = 0,
    ) -> bool:
        # C++ backend does not support sink attention
        if sink is not None:
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
        philox_seed: int = 0,
        philox_offset: int = 0,
    ):

        return _flash_attn_backward(
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


class AttnBwdAiterTritonBackend(KernelBackend):
    """Triton backend for attention backward using aiter triton implementation."""

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
        philox_seed: int = 0,
        philox_offset: int = 0,
    ) -> bool:
        # Triton requires:
        # 1. head_dim_qk == head_dim_v
        # 2. head_dim must be a power of 2
        head_dim_qk = q.shape[-1]
        head_dim_v = v.shape[-1]
        if head_dim_qk != head_dim_v:
            return False
        if not _is_power_of_2(head_dim_qk):
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
        philox_seed: int = 0,
        philox_offset: int = 0,
    ):

        return flash_attn_onekernel_backward(
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
            q.shape[1],  # max_seqlen_q
            k.shape[1],  # max_seqlen_k
            dropout_p,
            philox_seed,
            philox_offset,
            True,  # USE_INT64_STRIDES
            sink=sink,
            dsink=dsink,
        )


_ATTN_BWD_AITER_BACKENDS = {
    BackendType.AITER: AttnBwdAiterCsrcBackend,
    BackendType.TRITON: AttnBwdAiterTritonBackend,
}


class AttnBwdAiterKernelDispatcher(AutoKernelDispatcher):
    """
    Dispatcher for attention backward with Aiter backends.

    Uses AutoKernelDispatcher dispatch logic:
    1. User specified backend (env or code) - highest priority
    2. Auto tune (if enabled)
    3. Default backend (AITER/C++)
    4. Fallback: try all backends
    """

    _backends = _ATTN_BWD_AITER_BACKENDS
    _cache = TuneCache(256)

    @classmethod
    def make_key(cls, q, k, v, causal, sink, **kwargs):
        return (
            q.shape,
            k.shape,
            v.shape,
            q.dtype,
            causal,
            sink is not None,
        )


def attention_aiter_forward_dispatch(
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
    max_seqlen_q: int,
    max_seqlen_k: int,
    sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
    """
    Unified dispatch for attention forward.

    Returns:
        Tuple of (out, softmax_lse, S_dmask, state_info)
        state_info contains backend-specific state needed for backward:
        - For csrc: ("csrc", rng_state, 0, 0)
        - For triton: ("triton", sink, philox_seed, philox_offset)
    """
    # Default backend is AITER (C++), prefer C++ over Triton
    default_backend_enum = BackendType.AITER
    # Get user-specified backend from GlobalBackendManager
    user_backend_enum = GlobalBackendManager.get_attention_backend()

    kwargs = dict(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        bias=bias,
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_softmax,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sink=sink,
    )

    return AttnFwdAiterKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)


def attention_aiter_backward_dispatch(
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
    philox_seed: int = 0,
    philox_offset: int = 0,
):
    """
    Unified dispatch for attention backward.
    """
    # Default backend is AITER (C++), prefer C++ over Triton
    default_backend_enum = BackendType.AITER
    # Get user-specified backend from GlobalBackendManager
    user_backend_enum = GlobalBackendManager.get_attention_backend()

    kwargs = dict(
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=softmax_lse,
        dq=dq,
        dk=dk,
        dv=dv,
        dbias=dbias,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        bias=bias,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        rng_state=rng_state,
        is_v3_atomic_fp32=is_v3_atomic_fp32,
        how_v3_bf16_cvt=how_v3_bf16_cvt,
        sink=sink,
        dsink=dsink,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
    )

    return AttnBwdAiterKernelDispatcher.dispatch(default_backend_enum, user_backend_enum, **kwargs)
