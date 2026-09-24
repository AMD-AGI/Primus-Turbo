###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers shared across the GEMM / grouped-GEMM op implementations."""

import threading
from typing import Callable, Optional, TypeVar

import torch

from primus_turbo.pytorch.core.low_precision import (
    Format,
    float8_e4m3,
    float8_e5m2,
)

_FUSED_GRAD_WRITE_STATE = "_primus_turbo_fused_grad_write_state"
_T = TypeVar("_T")


def _is_cuda_graph_capturing() -> bool:
    fn = getattr(torch.cuda, "is_current_stream_capturing", None)
    if fn is None:
        graphs = getattr(torch.cuda, "graphs", None)
        fn = getattr(graphs, "is_current_stream_capturing", None) if graphs is not None else None
    try:
        return bool(fn()) if fn is not None else False
    except Exception:
        # CPU-only builds and uninitialized drivers may expose the API but
        # raise RuntimeError, AssertionError, or AttributeError when called.
        return False


class _FusedGradWriteState:
    """Shared state for every forward that may write one Parameter this step."""

    def __init__(self):
        self.lock = threading.Lock()
        self.claimed = False
        self.overwrite_eligible = True
        self.has_overwrite_producer = False
        self.has_beta1_only_producer = False
        self.last_write_stream = None


class _FusedGradOverwriteClaim:
    """Select the beta=0 writer when its backward actually executes."""

    def __init__(self, parameter: torch.nn.Parameter, state: _FusedGradWriteState):
        self.parameter = parameter
        self.state = state

    def execute(self, launch: Callable[[bool], _T]) -> _T:
        """Choose beta and enqueue the write as one serialized operation.

        The lock stays held through kernel launch, so a beta=1 writer cannot be
        enqueued before the beta=0 winner. If autograd invokes tied-Parameter
        backwards on different CUDA streams, ``wait_stream`` extends that
        ordering to device execution before the next write is launched.
        """
        parameter = self.parameter
        state = self.state
        with state.lock:
            current_state = getattr(parameter, _FUSED_GRAD_WRITE_STATE, None)

            # A retained graph may be run again after Megatron starts a new
            # step. Falling back to beta=1 is not safe when the framework
            # skipped this slice's clear based on the previous epoch, so reject
            # before writing.
            if not bool(parameter.grad_added_to_main_grad) or current_state is not state:
                raise RuntimeError(
                    "fused gradient overwrite claim belongs to a stale write epoch; "
                    "retained-graph backward across gradient-buffer resets is unsupported"
                )

            # Python bookkeeping is not replayed by CUDA graphs. Capture beta=1
            # into the graph and permanently disable overwrite for this epoch.
            if _is_cuda_graph_capturing():
                state.overwrite_eligible = False
                overwrite = False
            elif not state.overwrite_eligible or state.claimed:
                overwrite = False
            else:
                state.claimed = True
                overwrite = True

            main_grad = parameter.main_grad
            current_stream = None
            if main_grad.is_cuda:
                current_stream = torch.cuda.current_stream(main_grad.device)
                if state.last_write_stream is not None and state.last_write_stream != current_stream:
                    current_stream.wait_stream(state.last_write_stream)

            result = launch(overwrite)
            if current_stream is not None:
                state.last_write_stream = current_stream
            return result


def _get_fp8_dtype(format: Format, is_fwd_stage: bool):
    if format == Format.E4M3:
        return float8_e4m3
    elif format == Format.E5M2:
        return float8_e5m2
    elif format == Format.HYBRID:
        return float8_e4m3 if is_fwd_stage else float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: {format}")


def _ensure_contiguous_grad_out(grad_out: torch.Tensor) -> torch.Tensor:
    # Some upstream reductions can produce expanded zero-stride grad_out views.
    # Custom grouped GEMM kernels expect dense layouts.
    return grad_out if grad_out.is_contiguous() else grad_out.contiguous()


def _setup_fused_grad_accum(
    b,
    fuse_bgrad_accum_pattern: Optional[str],
    *,
    supports_overwrite: bool = False,
):
    """Resolve the weight's gradient-accumulation buffer for the fused wgrad path.

    Returns ``(enabled, main_grad, overwrite_claim)``. When enabled, the wgrad GEMM
    writes straight into ``main_grad`` and the Function must return no gradient
    for ``b``.

    An overwrite-capable caller receives a claim object and must call
    ``execute()`` around its backward write. The first backward that actually
    executes for the Parameter receives True and may overwrite with beta=0;
    later writes receive False and accumulate with beta=1. Selection and launch
    are serialized so the overwrite is also the first device write. Deciding at
    the write, rather than in forward, keeps checkpoint recomputation, staged
    forwards, tied weights, and reverse backward order correct.

    The claim is restricted to a real ``nn.Parameter``. On the multi-microbatch
    path the caller hands us a per-microbatch quantized alias carrying a *copy*
    of the flag, and the real parameter's flag is only set later; an alias read
    would hand out beta=0 twice and silently drop a microbatch.

    Calls from fused paths that cannot overwrite register the epoch as
    ineligible. Mixing one of those producers with an overwrite-capable
    producer, or switching to one immediately after an owned beta=0 epoch, is
    rejected because the framework may already have skipped the buffer clear.
    """
    if fuse_bgrad_accum_pattern is None:
        return False, None, None

    assert fuse_bgrad_accum_pattern in ["megatron"], (
        "Only megatron support gradient accumulation fusion currently"
    )

    assert hasattr(b, "grad_added_to_main_grad"), (
        "b.grad_added_to_main_grad must be set up before the backward pass."
    )
    assert hasattr(b, "main_grad"), "b.main_grad must be set up before the backward pass."
    assert isinstance(b.main_grad, torch.Tensor) and (b.main_grad.shape == b.shape), (
        "b.main_grad must be a tensor with the same shape as b"
    )

    overwrite_claim = None
    if isinstance(b, torch.nn.Parameter):
        state = getattr(b, _FUSED_GRAD_WRITE_STATE, None)
        previous_state = None
        if not bool(b.grad_added_to_main_grad) or not isinstance(state, _FusedGradWriteState):
            previous_state = state
            state = _FusedGradWriteState()
            setattr(b, _FUSED_GRAD_WRITE_STATE, state)
        if not supports_overwrite:
            # If the previous epoch actually used beta=0, Primus may already
            # have skipped clearing this slice before the current forward. A
            # beta=1-only producer cannot safely consume that stale value.
            if isinstance(previous_state, _FusedGradWriteState) and previous_state.claimed:
                raise RuntimeError(
                    "fused gradient producer changed from beta=0 overwrite to beta=1-only; "
                    "the gradient buffer may have skipped its clear"
                )
            if state.has_overwrite_producer:
                raise RuntimeError(
                    "mixing beta=0-capable and beta=1-only fused gradient producers "
                    "for one Parameter in the same write epoch is unsupported"
                )
            state.has_beta1_only_producer = True
            state.overwrite_eligible = False
        else:
            if state.has_beta1_only_producer:
                raise RuntimeError(
                    "mixing beta=0-capable and beta=1-only fused gradient producers "
                    "for one Parameter in the same write epoch is unsupported"
                )
            state.has_overwrite_producer = True
            overwrite_claim = _FusedGradOverwriteClaim(b, state)

    # Preserve Megatron's existing fused-accumulation signal during forward.
    b.grad_added_to_main_grad = True
    return True, b.main_grad, overwrite_claim


_dummy_wgrads = {}


def _get_dummy_wgrad(shape: list, dtype: torch.dtype, zero=False) -> torch.Tensor:
    """Returns a dummy tensor of given shape.

    Supports arbitrary rank (2D for plain Linear weights, 3D for stacked
    grouped-linear weights ``(num_gemms, out_features, in_features)``, etc.).
    Tensors are cached by ``(shape, dtype)`` so each distinct weight layout
    only allocates one persistent buffer that gets reused across steps.
    """
    global _dummy_wgrads
    key = (tuple(shape), dtype)
    if key not in _dummy_wgrads:
        _dummy_wgrads[key] = torch.empty(
            shape,
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
    if zero:
        _dummy_wgrads[key].fill_(0)
    return _dummy_wgrads[key].detach()
