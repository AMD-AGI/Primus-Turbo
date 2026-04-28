###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
import os
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import torch
import torch.distributed as dist

from primus_turbo.common.constants import ENV_MOE_DISPATCH_COMBINE_BACKEND
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
)

# =========================================================================
# Buffer configuration
# =========================================================================


@dataclass
class EPBufferConfig:
    """Configuration for EP communication buffer initialization.

    Attributes:
        num_sms: Number of SMs to use in high-throughput kernels.
        dispatch_config: Optional user-provided dispatch config (from offline
            benchmarking). When ``None``, the backend's default for the current
            ``ep_size`` is used (``Buffer.get_dispatch_config(ep_size)``).
        combine_config: Optional user-provided combine config.  Same fallback
            behaviour as *dispatch_config*.
    """

    num_sms: int = 32
    dispatch_config: Any = None
    combine_config: Any = None


_DEFAULT_BUFFER_CONFIG = EPBufferConfig(
    num_sms=32,
    dispatch_config=None,
    combine_config=None,
)

_buffer_config: EPBufferConfig = _DEFAULT_BUFFER_CONFIG


def set_buffer_global_config(
    num_use_cu: int = 32,
    autotune_config: Optional[tuple] = None,
) -> None:
    """Store the SM count and optional per-operation configs.

    This is typically called once by the token dispatcher during ``__init__``.

    Args:
        num_use_cu: Number of SMs (compute units) for high-throughput kernels.
        autotune_config: Legacy parameter — a ``(dispatch_config, combine_config)``
            tuple obtained from offline benchmarking.  ``None`` means use the
            backend's built-in defaults for the current EP group size.
    """
    global _buffer_config
    dispatch_cfg, combine_cfg = autotune_config if autotune_config is not None else (None, None)
    _buffer_config = EPBufferConfig(
        num_sms=num_use_cu,
        dispatch_config=dispatch_cfg,
        combine_config=combine_cfg,
    )


# =========================================================================
# EPBackend Protocol
# =========================================================================


@runtime_checkable
class EPBackend(Protocol):
    """Structural (``typing.Protocol``) interface for Expert-Parallel
    communication backends.

    Each backend encapsulates a specific EP library (e.g. in-tree Turbo DeepEP,
    external ``deep_ep``, UCCL-EP, ...) and owns its own buffer lifecycle.
    Adding a new backend is a single-class change plus one
    ``register_ep_backend()`` call — any class that structurally conforms to
    this Protocol is accepted; no explicit inheritance is required.
    """

    @staticmethod
    def is_available() -> bool:
        """Return True if this backend's dependencies are importable."""
        ...

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_bytes: int,
        config: EPBufferConfig,
    ) -> None:
        """(Re-)create the communication buffer if needed."""
        ...

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[tuple] = None,
        topk_idx: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        num_worst_tokens: int = 0,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Union[List[int], torch.Tensor]],
        Optional[tuple],
    ]:
        """Execute dispatch (layout + send) and return
        ``(recv_x, recv_topk_idx, recv_topk_weights, tokens_per_expert, handle)``.
        """
        ...

    def combine(
        self,
        x: torch.Tensor,
        handle: tuple,
        topk_weights: Optional[torch.Tensor] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Execute combine and return ``(combined_x, combined_topk_weights)``."""
        ...


# =========================================================================
# _DeepEPLikeBackend — shared implementation for DeepEP-compatible backends
# =========================================================================


class _DeepEPLikeBackend:
    """Shared logic for all backends that follow the DeepEP Buffer protocol
    (``get_dispatch_layout`` / ``dispatch`` / ``combine`` / ``set_num_sms`` /
    ``get_dispatch_config`` / ``get_combine_config``).

    This is a plain implementation base class — it does **not** inherit from
    ``EPBackend``. Conformance to the ``EPBackend`` Protocol is checked
    structurally by the type system.

    Subclasses only need to override ``is_available``, ``_get_module``, and
    optionally ``_make_buffer_kwargs`` to supply backend-specific constructor
    arguments.
    """

    def __init__(self) -> None:
        self._buffer = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if this backend's dependencies are importable."""
        raise NotImplementedError

    @staticmethod
    def _get_module():
        """Return the Python module that exposes ``Buffer``, ``Config``,
        ``EventHandle``, ``EventOverlap`` (or a compatible ``utils`` sub-module).
        """
        raise NotImplementedError

    def _make_buffer_kwargs(self, group: dist.ProcessGroup) -> dict:
        """Extra keyword arguments forwarded to ``BufferClass(group, nvl, rdma, **kwargs)``."""
        return {}

    # ------------------------------------------------------------------
    # EPBackend interface
    # ------------------------------------------------------------------

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_bytes: int,
        config: EPBufferConfig,
    ) -> None:
        mod = self._get_module()
        BufferClass = mod.Buffer

        BufferClass.set_num_sms(config.num_sms)

        dispatch_config = config.dispatch_config or BufferClass.get_dispatch_config(group.size())
        combine_config = config.combine_config or BufferClass.get_combine_config(group.size())

        num_nvl_bytes, num_rdma_bytes = 0, 0
        for cfg in (dispatch_config, combine_config):
            num_nvl_bytes = max(
                cfg.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                num_nvl_bytes,
            )
            try:
                num_rdma_bytes = max(
                    cfg.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
            except (RuntimeError, AttributeError):
                pass

        buf_kwargs = self._make_buffer_kwargs(group)

        if (
            self._buffer is None
            or not isinstance(self._buffer, BufferClass)
            or self._buffer.group != group
            or self._buffer.num_nvl_bytes < num_nvl_bytes
            or self._buffer.num_rdma_bytes < num_rdma_bytes
        ):
            self._buffer = BufferClass(group, num_nvl_bytes, num_rdma_bytes, **buf_kwargs)

    # ----- helpers ------------------------------------------------------

    def _get_event_classes(self):
        mod = self._get_module()
        EventOverlapClass = mod.utils.EventOverlap if hasattr(mod, "utils") else mod.EventOverlap
        EventHandleClass = mod.utils.EventHandle if hasattr(mod, "utils") else mod.EventHandle
        return EventOverlapClass, EventHandleClass

    # ----- dispatch / combine -------------------------------------------

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[tuple] = None,
        topk_idx: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        num_worst_tokens: int = 0,
    ):
        EventOverlapClass, EventHandleClass = self._get_event_classes()
        buffer = self._buffer
        assert buffer is not None, "init_buffer() must be called before dispatch()"

        previous_event = EventOverlapClass(EventHandleClass()) if async_finish else None

        if handle is None:
            assert topk_idx is not None
            assert token_weights is not None
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                event,
            ) = buffer.get_dispatch_layout(
                topk_idx,
                num_experts,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )

            (
                recv_x,
                recv_token_indices,
                recv_token_probs,
                tokens_per_expert,
                handle,
                after_event,
            ) = buffer.dispatch(
                x,
                topk_idx=topk_idx,
                topk_weights=token_weights,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                previous_event=event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
                num_worst_tokens=num_worst_tokens,
            )
        else:
            recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle, after_event = (
                buffer.dispatch(
                    x,
                    handle=handle,
                    previous_event=previous_event,
                    async_finish=async_finish,
                    allocate_on_comm_stream=allocate_on_comm_stream,
                )
            )

        if async_finish:
            after_event.current_stream_wait()

        return recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle

    def combine(
        self,
        x: torch.Tensor,
        handle: tuple,
        topk_weights: Optional[torch.Tensor] = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ):
        EventOverlapClass, EventHandleClass = self._get_event_classes()
        buffer = self._buffer
        assert buffer is not None, "init_buffer() must be called before combine()"

        previous_event = EventOverlapClass(EventHandleClass()) if async_finish else None

        combined_x, combined_topk_weights, after_event = buffer.combine(
            x,
            handle=handle,
            topk_weights=None if topk_weights is None else topk_weights.float(),
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            previous_event=previous_event,
        )

        if async_finish:
            after_event.current_stream_wait()

        return combined_x, combined_topk_weights


# =========================================================================
# Concrete backends
# =========================================================================


class TurboEPBackend(_DeepEPLikeBackend):
    """In-tree Primus-Turbo DeepEP backend (always available)."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def _get_module():
        import primus_turbo.pytorch.deep_ep as turbo_ep

        return turbo_ep


class DeepEPBackend(_DeepEPLikeBackend):
    """External ``deep_ep`` package backend (optional)."""

    @staticmethod
    def is_available() -> bool:
        try:
            import deep_ep  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _get_module():
        import deep_ep

        return deep_ep

    def _make_buffer_kwargs(self, group: dist.ProcessGroup) -> dict:
        BufferClass = self._get_module().Buffer
        try:
            param = inspect.signature(BufferClass).parameters.get("is_intranode")
        except (TypeError, ValueError):
            param = None
        if param is not None and param.default is False:
            # uccl-ep special handle
            return {"is_intranode": group.size() <= 8}
        return {}


# =========================================================================
# Backend registry
# =========================================================================

_BACKEND_REGISTRY: Dict[str, Type[EPBackend]] = {
    "TURBO": TurboEPBackend,
    "DEEP_EP": DeepEPBackend,
}

_backend_instances: Dict[str, EPBackend] = {}


def clear_backend_instances():
    global _backend_instances

    _backend_instances.clear()


def register_ep_backend(name: str, cls: Type[EPBackend]) -> None:
    """Register a new EP backend class (e.g. ``UCCL_EP``)."""
    _BACKEND_REGISTRY[name] = cls


def _get_backend_instance(name: str) -> EPBackend:
    """Lazily create and cache a backend singleton."""
    if name not in _backend_instances:
        if name not in _BACKEND_REGISTRY:
            raise ValueError(f"Unknown EP backend '{name}'. Available: {list(_BACKEND_REGISTRY.keys())}")
        cls = _BACKEND_REGISTRY[name]
        if not cls.is_available():
            raise RuntimeError(
                f"EP backend '{name}' is registered but its dependencies are not "
                f"installed. Please install the required package."
            )
        _backend_instances[name] = cls()
    return _backend_instances[name]


# =========================================================================
# Backend selection
# =========================================================================

_BACKEND_TYPE_TO_NAME: Dict[BackendType, str] = {
    BackendType.TURBO: "TURBO",
    BackendType.DEEP_EP: "DEEP_EP",
}


def _resolve_backend_name() -> str:
    """Determine which EP backend to use.

    Priority (high -> low):
      1. ``GlobalBackendManager`` code-level setting (via ``set_moe_dispatch_combine_backend``)
      2. ``PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND`` env var (supports names beyond ``BackendType``)
      3. Default: ``TURBO``
    """
    user_backend = GlobalBackendManager.get_moe_dispatch_combine_backend(PrecisionType.BF16_FP16_FP32)
    if user_backend is not None:
        return _BACKEND_TYPE_TO_NAME.get(user_backend, user_backend.name)

    env_val = os.environ.get(ENV_MOE_DISPATCH_COMBINE_BACKEND)
    if env_val is not None:
        return env_val.strip().upper()

    return "TURBO"


# =========================================================================
# Utilities
# =========================================================================


def get_hidden_bytes(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Uses at least 2 bytes (bf16 size) so buffers work for both fp8 and bf16
    without reallocation.
    """
    inp = x if isinstance(x, torch.Tensor) else x[0]
    return inp.size(1) * max(inp.element_size(), 2)


def _ensure_buffer(
    group: dist.ProcessGroup,
    hidden_bytes: int,
    backend: EPBackend,
) -> None:
    """Make sure the backend's buffer is initialized."""
    if _buffer_config is None:
        raise RuntimeError(
            "set_buffer_global_config() must be called before dispatch/combine. "
            "This is typically done by the token dispatcher during __init__."
        )
    backend.init_buffer(group, hidden_bytes, _buffer_config)


# =========================================================================
# Public API — used by ``moe_dispatch_combine.py``
# =========================================================================


def moe_dispatch_impl(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    handle: Optional[tuple] = None,
    topk_idx: Optional[torch.Tensor] = None,
    token_weights: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
    num_worst_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    name = _resolve_backend_name()
    backend = _get_backend_instance(name)
    _ensure_buffer(group, get_hidden_bytes(x), backend)
    return backend.dispatch(
        x,
        handle=handle,
        topk_idx=topk_idx,
        token_weights=token_weights,
        num_experts=num_experts,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        num_worst_tokens=num_worst_tokens,
    )


def moe_combine_impl(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    handle: tuple,
    topk_weights: Optional[torch.Tensor] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    name = _resolve_backend_name()
    backend = _get_backend_instance(name)
    _ensure_buffer(group, get_hidden_bytes(x), backend)
    return backend.combine(
        x,
        handle=handle,
        topk_weights=topk_weights,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
