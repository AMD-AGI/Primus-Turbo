###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import dataclasses
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.distributed as dist

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.core.backend import (
    BackendType,
    GlobalBackendManager,
    PrecisionType,
    TuneCache,
)
from primus_turbo.pytorch.kernels.moe._backend import (
    DeepEPBackend,
    EPBackend,
    EPBufferConfig,
    MoriEPBackend,
    TurboEPBackend,
    UCCLEPBackend,
    detect_group_topology,
)

# =========================================================================
# Buffer configuration
# =========================================================================


# Per-backend default buffer configuration. Keys must match the names used
# in ``_BACKEND_REGISTRY`` so lookups can be done by backend name.
_DEFAULT_BUFFER_CONFIG_PER_BACKEND: Dict[str, EPBufferConfig] = {
    "TURBO": EPBufferConfig(
        num_sms=64,
        dispatch_config=None,
        combine_config=None,
    ),
    "DEEP_EP": EPBufferConfig(
        num_sms=64,
        dispatch_config=None,
        combine_config=None,
    ),
    "UCCL": EPBufferConfig(
        num_sms=64,
        dispatch_config=None,
        combine_config=None,
    ),
    "MORI": EPBufferConfig(num_sms=64, dispatch_config=None, combine_config=None),
}


# Deep copy of the defaults so mutations via ``set_buffer_global_config``
# don't leak into ``_DEFAULT_BUFFER_CONFIG_PER_BACKEND``.
_buffer_config_per_backend: Dict[str, EPBufferConfig] = {
    name: dataclasses.replace(cfg) for name, cfg in _DEFAULT_BUFFER_CONFIG_PER_BACKEND.items()
}


def _get_buffer_config(backend_name: str) -> EPBufferConfig:
    """Return the runtime buffer config for ``backend_name``, or a default."""
    cfg = _buffer_config_per_backend.get(backend_name)
    if cfg is not None:
        return cfg
    default = _DEFAULT_BUFFER_CONFIG_PER_BACKEND.get(backend_name)
    return dataclasses.replace(default) if default is not None else EPBufferConfig()


def set_buffer_global_config(
    num_use_cu: int = 32,
    autotune_config: Optional[tuple] = None,
    backend: Optional[str] = None,
) -> None:
    """Set the EP buffer config.

    Args:
        num_use_cu: SM count for high-throughput kernels.
        autotune_config: Optional ``(dispatch_config, combine_config)`` from
            offline tuning.
        backend: Target backend name (matches ``_BACKEND_REGISTRY``). When
            ``None``, the same config is broadcast to every registered
            backend (backward-compat with the pre-multi-backend API).

    If a backend has already allocated a buffer against its previous config, we
    *warn* on any actual change: the live buffer is sized to the old config and
    may be referenced by a captured CUDA graph, so silently swapping in a new
    ``num_sms`` would either re-trigger reallocation on the next
    ``_ensure_buffer`` (invalidating the captured pointer) or be ignored
    entirely. Equal configs are a no-op.
    """
    dispatch_cfg, combine_cfg = autotune_config if autotune_config is not None else (None, None)
    new_cfg = EPBufferConfig(
        num_sms=num_use_cu,
        dispatch_config=dispatch_cfg,
        combine_config=combine_cfg,
    )

    def _warn_if_live_buffer(name: str) -> None:
        old_cfg = _buffer_config_per_backend.get(name)
        if old_cfg is None or new_cfg == old_cfg:
            return
        inst = _backend_instances.get(name)
        if inst is not None and getattr(inst, "_buffer", None) is not None:
            warnings.warn(
                f"set_buffer_global_config called with a different config after "
                f"backend '{name}' was already initialized "
                f"(old={old_cfg!r}, new={new_cfg!r}). Previously-captured CUDA "
                "graphs continue to reference the old buffer; the new config only "
                "takes effect when the buffer is next reallocated.",
                stacklevel=2,
            )

    if backend is None:
        targets = set(_buffer_config_per_backend.keys()) | set(_BACKEND_REGISTRY.keys())
        for name in targets:
            _warn_if_live_buffer(name)
            _buffer_config_per_backend[name] = dataclasses.replace(new_cfg)
    else:
        _warn_if_live_buffer(backend)
        _buffer_config_per_backend[backend] = new_cfg


# =========================================================================
# Backend registry
# =========================================================================

_BACKEND_REGISTRY: Dict[str, Type[EPBackend]] = {
    "TURBO": TurboEPBackend,
    "DEEP_EP": DeepEPBackend,
    "MORI": MoriEPBackend,
    "UCCL": UCCLEPBackend,
}

_backend_instances: Dict[str, EPBackend] = {}


def clear_backend_instances():
    """Wipe cached backend singletons.

    Refuses to clear if any backend already has an initialized buffer — that
    buffer may be referenced by a captured CUDA graph or by an in-flight
    dispatch/combine on another stream. Call this only at process start, before
    any dispatch has run (or in tests after explicitly tearing down all
    captured graphs).
    """
    global _backend_instances
    for name, backend in _backend_instances.items():
        if getattr(backend, "_buffer", None) is not None:
            raise RuntimeError(
                f"Refusing to clear EP backend cache: backend '{name}' already has an "
                "initialized buffer. Dropping it now would invalidate buffer pointers "
                "referenced by captured CUDA graphs or in-flight kernels."
            )
    _backend_instances.clear()


def register_ep_backend(name: str, cls: Type[EPBackend]) -> None:
    """Register a new EP backend class (e.g. ``UCCL``)."""
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


def _release_other_backends(active_name: str) -> None:
    """Release comm buffers of every EP backend except ``active_name`` (and UCCL,
    whose destroy() corrupts the HIP context). Only one live EP backend per
    process: a leftover buffer otherwise deadlocks the active backend's init."""
    for name, inst in _backend_instances.items():
        if name in (active_name, "UCCL"):
            continue
        is_init = getattr(inst, "is_initialized", None)
        if is_init is None or not is_init():
            continue
        try:
            inst.release_buffer()
            logger.info(
                f"EP backend switch: released '{name}' buffer before using "
                f"'{active_name}' (one live EP backend per process).",
                rank=0,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logger.warning(f"EP backend switch: release_buffer('{name}') failed: {exc!r}")


# =========================================================================
# Autotuner — finds best (backend, dispatch_config, combine_config) per case
# =========================================================================


@dataclass(frozen=True)
class _EPAutoTuneKey:
    """Shape-based signature used to cache autotune results."""

    num_tokens: int
    hidden: int
    num_topk: int
    num_experts: int
    ep_size: int
    dtype: torch.dtype
    use_fp8: bool


@dataclass
class EPAutoTuneResult:
    """Result of :class:`MoEDispatchCombineAutoTuner.tune`.

    Attributes:
        backend_name: Winning backend registry name.
        dispatch_config: Best dispatch ``Config``.
        combine_config: Best combine ``Config``.
        dispatch_time_us: Dispatch latency of the winner (us).
        combine_time_us: Combine latency of the winner (us).
        per_backend: ``backend_name -> (d_time_us, c_time_us)`` of every
            candidate that completed tuning.
        tune_duration_s: Wall-clock seconds spent tuning (rank 0 clock).
    """

    backend_name: str
    dispatch_config: Any
    combine_config: Any
    dispatch_time_us: float
    combine_time_us: float
    per_backend: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    tune_duration_s: float = 0.0


class MoEDispatchCombineAutoTuner:
    """Picks the best ``(backend, dispatch_config, combine_config)`` per shape.

    The shape-keyed result is bound to the dispatch ``handle`` so the paired
    combine / cached dispatch / backward reuse it. Enable via
    ``PRIMUS_TURBO_AUTO_TUNE=1`` or :meth:`get_or_tune`.
    """

    # Shape-based cache shared across layers with identical input shape.
    _cache: TuneCache = TuneCache(capacity=1024)

    # ``handle -> tune result`` LRU. Handles are tuples (not weakref-able), so
    # we keep a strong ref alongside the result and rely on ``stored_handle
    # is handle`` to defend against ``id()`` reuse. Cap is small because a
    # dispatch/combine pair turns over quickly; the LRU evicts stale entries
    # before they pile up. ``clear()`` empties this cache as well.
    _HANDLE_CACHE_MAX: int = 128
    _handle_cache: "OrderedDict[int, Tuple[Any, EPAutoTuneResult]]" = OrderedDict()

    # Fallback for calls made before any handle mapping is registered.
    _current_result: Optional[EPAutoTuneResult] = None

    @staticmethod
    def make_key(
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_idx: Optional[torch.Tensor],
        num_experts: int,
        ep_size: int,
        num_topk: Optional[int] = None,
    ) -> _EPAutoTuneKey:
        inp = x if isinstance(x, torch.Tensor) else x[0]
        num_tokens, hidden = int(inp.size(0)), int(inp.size(1))
        if num_topk is not None:
            resolved_num_topk = int(num_topk)
        elif topk_idx is not None:
            resolved_num_topk = int(topk_idx.size(1))
        else:
            resolved_num_topk = 0
        return _EPAutoTuneKey(
            num_tokens=num_tokens,
            hidden=hidden,
            num_topk=resolved_num_topk,
            num_experts=int(num_experts),
            ep_size=int(ep_size),
            dtype=inp.dtype,
            use_fp8=isinstance(x, tuple),
        )

    # ------------------------------------------------------------------
    # Handle <-> result binding (dispatch/combine pair)
    # ------------------------------------------------------------------

    @classmethod
    def register_handle(cls, handle: Any, result: EPAutoTuneResult) -> None:
        """Bind ``result`` to ``handle`` for paired-call reuse.

        Args:
            handle: Handle returned by the backend's ``dispatch`` call.
            result: Tune result to associate with the handle.
        """
        if handle is None:
            return
        hid = id(handle)
        cache = cls._handle_cache
        if hid in cache:
            cache.move_to_end(hid)
        cache[hid] = (handle, result)
        if len(cache) > cls._HANDLE_CACHE_MAX:
            cache.popitem(last=False)

    @classmethod
    def lookup_handle(cls, handle: Any) -> Optional[EPAutoTuneResult]:
        """Return the result bound to ``handle``, or ``None`` if unknown.

        ``stored_handle is handle`` guards against ``id()`` collisions after
        the original handle was GC'd and its id was reused by a new one.
        """
        if handle is None:
            return None
        entry = cls._handle_cache.get(id(handle))
        if entry is None:
            return None
        stored_handle, res = entry
        if stored_handle is not handle:
            cls._handle_cache.pop(id(handle), None)
            return None
        cls._handle_cache.move_to_end(id(handle))
        return res

    @classmethod
    def discard_handle(cls, handle: Any) -> None:
        """Drop the cache entry for ``handle`` once the caller is done with it.

        Call this after the paired combine (and any backward that reuses the
        handle) so long-running jobs don't pin tune-result tensors via the
        LRU.
        """
        if handle is None:
            return
        cls._handle_cache.pop(id(handle), None)

    @classmethod
    def handle_cache_size(cls) -> int:
        """Return current entry count in the handle cache (tests / metrics)."""
        return len(cls._handle_cache)

    @classmethod
    def _candidate_backend_names(cls) -> List[str]:
        names: List[str] = []
        for name, impl in _BACKEND_REGISTRY.items():
            try:
                if impl.is_available():
                    names.append(name)
            except Exception:
                continue
        return names

    @classmethod
    @torch.no_grad()
    def tune(
        cls,
        group: dist.ProcessGroup,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        num_experts: int,
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        num_sms: int = 32,
        candidate_backends: Optional[Sequence[str]] = None,
        num_tests: int = 20,
        num_topk: Optional[int] = None,
        uniform_dispatch: bool = True,
    ) -> EPAutoTuneResult:
        """Tune across candidate backends + configs and return the best.

        Args:
            group: EP process group.
            x: Dispatch input (Tensor or ``(fp8_tensor, scales)`` tuple).
            num_experts: Total number of experts.
            topk_idx: ``[num_tokens, num_topk]`` expert indices (optional in
                uniform mode).
            topk_weights: ``[num_tokens, num_topk]`` expert weights (optional
                in uniform mode).
            num_sms: SM count to use for tuning.
            candidate_backends: Optional explicit backend name list.
            num_tests: Timed iterations per candidate.
            num_topk: Number of experts per token (uniform-mode fallback).
            uniform_dispatch: If True, resample topk internally for robustness.

        Returns:
            Best :class:`EPAutoTuneResult`. Not cached; use :meth:`get_or_tune`.
        """
        names = list(candidate_backends) if candidate_backends is not None else cls._candidate_backend_names()
        if not names:
            raise RuntimeError("MoE autotune: no EP backends are available.")

        tune_start = time.perf_counter()

        # Detect multi-node topology once; reused to guard backend candidates
        # whose worst-case buffer sizing is unsafe in inter-node EP.
        _, num_nodes = detect_group_topology(group)

        best: Optional[EPAutoTuneResult] = None
        best_total = float("inf")
        per_backend: Dict[str, Tuple[float, float]] = {}

        for name in names:
            if name not in _BACKEND_REGISTRY:
                logger.info(
                    f"MoE autotune: skipping backend '{name}'; not in "
                    f"backend registry ({list(_BACKEND_REGISTRY.keys())}).",
                    once=True,
                    rank=0,
                )
                continue
            try:
                available = _BACKEND_REGISTRY[name].is_available()
            except Exception as exc:
                logger.info(
                    f"MoE autotune: skipping backend '{name}'; "
                    f"is_available() raised {type(exc).__name__}: {exc!r}.",
                    once=True,
                    rank=0,
                )
                continue
            if not available:
                logger.info(
                    f"MoE autotune: skipping backend '{name}'; "
                    f"is_available()=False (dependencies missing or backend "
                    f"disabled).",
                    once=True,
                    rank=0,
                )
                continue
            backend = _get_backend_instance(name)
            _release_other_backends(name)
            try:
                d_cfg, c_cfg, d_t, c_t = backend.tune_configs(
                    group=group,
                    x=x,
                    num_experts=num_experts,
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    num_sms=num_sms,
                    num_tests=num_tests,
                    num_topk=num_topk,
                    uniform_dispatch=uniform_dispatch,
                )
            except NotImplementedError:
                # Backend opts out of config tuning. Skip quietly.
                logger.info(
                    f"MoE autotune: backend '{name}' does not support " f"tune_configs; skipping.",
                    once=True,
                    rank=0,
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                # repr() so pybind-backed errors aren't silently blank.
                logger.warning(
                    f"MoE autotune: backend '{name}' failed: \n" f"{type(exc).__name__}: {exc!r}",
                    exc_info=True,
                )
                continue
            finally:
                # Free this candidate's tuning buffer before the next one. UCCL
                # is exempt: its destroy() corrupts the HIP context (TODO: fix).
                if name != "UCCL":
                    try:
                        backend.release_buffer()
                    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                        logger.debug(f"MoE autotune: release_buffer('{name}') failed: {exc!r}")

            per_backend[name] = (round(d_t * 1e6, 3), round(c_t * 1e6, 3))
            total = d_t + c_t
            if total < best_total:
                best_total = total
                best = EPAutoTuneResult(
                    backend_name=name,
                    dispatch_config=d_cfg,
                    combine_config=c_cfg,
                    dispatch_time_us=round(d_t * 1e6, 3),
                    combine_time_us=round(c_t * 1e6, 3),
                )

        if best is None:
            raise RuntimeError(
                f"MoE autotune: all candidate backends failed ({names}). "
                f"Check logs and EP buffer configuration."
            )
        best.per_backend = per_backend
        best.tune_duration_s = time.perf_counter() - tune_start
        return best

    @classmethod
    @torch.no_grad()
    def get_or_tune(
        cls,
        group: dist.ProcessGroup,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        num_experts: int,
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        num_sms: int = 32,
        num_topk: Optional[int] = None,
        uniform_dispatch: bool = True,
        **tune_kwargs: Any,
    ) -> EPAutoTuneResult:
        """Return a cached tune result for this shape or run :meth:`tune` now.

        Args:
            group: EP process group.
            x: Dispatch input (Tensor or ``(fp8_tensor, scales)`` tuple).
            num_experts: Total number of experts.
            topk_idx: Optional ``[num_tokens, num_topk]`` expert indices.
            topk_weights: Optional ``[num_tokens, num_topk]`` expert weights.
            num_sms: SM count to use for tuning.
            num_topk: Number of experts per token. Also used in the shape
                cache key, so the same signature reuses the cached winner.
            uniform_dispatch: If True, resample topk internally for robustness.
            **tune_kwargs: Forwarded to :meth:`tune`.
        """
        key = cls.make_key(x, topk_idx, num_experts, group.size(), num_topk=num_topk)
        cached = cls._cache.get(key)
        if cached is not None:
            cls._current_result = cached
            return cached

        result = cls.tune(
            group=group,
            x=x,
            num_experts=num_experts,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_sms=num_sms,
            num_topk=num_topk,
            uniform_dispatch=uniform_dispatch,
            **tune_kwargs,
        )
        cls._cache.put(key, result)
        cls._current_result = result
        logger.info(
            f"[MoE AutoTune] key={key} -> backend={result.backend_name} "
            f"dispatch={result.dispatch_time_us:.1f}us combine={result.combine_time_us:.1f}us "
            f"tune_time={result.tune_duration_s:.2f}s "
            f"per_backend={result.per_backend}",
            rank=0,
        )
        return result

    @classmethod
    def current(cls) -> Optional[EPAutoTuneResult]:
        """Return the most recent tune result, or ``None`` if never tuned."""
        return cls._current_result

    @classmethod
    def set_current(cls, result: Optional[EPAutoTuneResult]) -> None:
        """Override the current runtime result (mainly for testing)."""
        cls._current_result = result

    @classmethod
    def clear(cls) -> None:
        """Drop every cache (shape, handle, current)."""
        cls._cache.clear()
        cls._handle_cache.clear()
        cls._current_result = None


# =========================================================================
# Backend selection
# =========================================================================

_DEFAULT_BACKEND_NAME = "TURBO"

# Explicit ``BackendType -> registry name`` map so renaming the enum doesn't
# silently break EP backend selection. Keys must mirror ``_BACKEND_REGISTRY``.
_BACKEND_TYPE_TO_NAME: Dict[BackendType, str] = {
    BackendType.TURBO: "TURBO",
    BackendType.DEEP_EP: "DEEP_EP",
    BackendType.MORI: "MORI",
    BackendType.UCCL: "UCCL",
}


def _get_backend_name() -> str:
    """Return the user-selected backend name, or ``TURBO`` by default."""
    bt = GlobalBackendManager.get_ep_backend(PrecisionType.BF16_FP16_FP32)
    if bt is None:
        return _DEFAULT_BACKEND_NAME
    return _BACKEND_TYPE_TO_NAME.get(bt, _DEFAULT_BACKEND_NAME)


# =========================================================================
# Utilities
# =========================================================================


def _maybe_tune_config(
    x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    group: dist.ProcessGroup,
    handle: Optional[tuple],
    topk_idx: Optional[torch.Tensor],
    token_weights: Optional[torch.Tensor],
    num_experts: Optional[int],
) -> Tuple[str, EPBufferConfig, Optional[EPAutoTuneResult]]:
    """Tune the config for the given shape or get default config."""
    backend_name = _get_backend_name()
    user_cfg = _get_buffer_config(backend_name)

    # Autotune off: use the user's backend + per-backend config.
    if not GlobalBackendManager.auto_tune_enabled():
        return backend_name, user_cfg, None

    # Autotune on: if a backend is pinned (env/code), tune only that one;
    # otherwise sweep all available backends.
    pinned_bt = GlobalBackendManager.get_ep_backend(PrecisionType.BF16_FP16_FP32)
    pinned_candidates = (
        [_BACKEND_TYPE_TO_NAME[pinned_bt]]
        if pinned_bt is not None and pinned_bt in _BACKEND_TYPE_TO_NAME
        else None
    )

    result: Optional[EPAutoTuneResult] = None

    if handle is not None:
        result = MoEDispatchCombineAutoTuner.lookup_handle(handle)
    else:
        if topk_idx is not None and token_weights is not None and num_experts is not None:
            result = MoEDispatchCombineAutoTuner.get_or_tune(
                group=group,
                x=x,
                num_experts=int(num_experts),
                topk_idx=topk_idx,
                topk_weights=token_weights,
                num_sms=user_cfg.num_sms,
                candidate_backends=pinned_candidates,
            )

        if result is None:
            result = MoEDispatchCombineAutoTuner.current()

    if result is not None:
        winner_cfg = _get_buffer_config(result.backend_name)
        cfg = EPBufferConfig(
            num_sms=winner_cfg.num_sms,
            dispatch_config=result.dispatch_config,
            combine_config=result.combine_config,
        )
        return result.backend_name, cfg, result

    # Nothing cached yet - fall back to user/default.
    return backend_name, user_cfg, None


# =========================================================================
# Public API - used by ``moe_dispatch_combine.py``
# =========================================================================


def _warn_if_graph_capture_incompatible(backend_name: str, op: str) -> None:
    """Warn when a graph-incompatible backend runs under active capture.

    Triggered iff ``torch.cuda.is_current_stream_capturing()`` is True and the
    backend's ``supports_cuda_graph()`` returns False. Does not raise;
    kernels will likely deadlock so the warning makes the cause obvious.
    """
    if not torch.cuda.is_current_stream_capturing():
        return
    cls = _BACKEND_REGISTRY.get(backend_name)
    if cls is None:
        return
    supports = getattr(cls, "supports_cuda_graph", None)
    if supports is None or supports():
        return
    logger.warning(
        f"EP backend '{backend_name}' does not support CUDA graph capture "
        f"(detected during moe_{op}); dispatch/combine may hang. Switch to "
        f"a graph-capable backend (e.g. PRIMUS_TURBO_EP_BACKEND=TURBO) or "
        f"disable capture for this layer.",
        once=True,
    )


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
    """Dispatch tokens to experts via the selected EP backend.

    Args:
        x: Input tensor, or ``(fp8_tensor, scales)`` tuple for FP8 path.
        group: EP process group.
        handle: Cached dispatch handle; ``None`` for a primary call.
        topk_idx: ``[num_tokens, num_topk]`` expert indices (primary only).
        token_weights: ``[num_tokens, num_topk]`` expert weights.
        num_experts: Total number of experts (primary only).
        async_finish: If True, return before the kernel finishes.
        allocate_on_comm_stream: Allocate outputs on the comm stream.
        num_worst_tokens: Worst-case receive token count (for padding).

    Returns:
        ``(recv_x, recv_topk_idx, recv_topk_weights, tokens_per_expert, handle)``.
    """
    name, cfg, result = _maybe_tune_config(x, group, handle, topk_idx, token_weights, num_experts)
    _warn_if_graph_capture_incompatible(name, "dispatch")
    backend = _get_backend_instance(name)
    if num_experts is None or topk_idx is None:
        assert backend.is_initialized(), "Backend is not initialized"
    else:
        _release_other_backends(name)
        # Unpack FP8 ``(tensor, scales)`` tuple to get shape metadata.
        fp8_dispatch = isinstance(x, tuple)
        x_inp = x[0] if fp8_dispatch else x
        backend.init_buffer(
            group,
            hidden_size=x_inp.size(1),
            num_experts=num_experts,
            num_topk=topk_idx.size(1),
            seqlen=x_inp.size(0),
            fp8_dispatch=fp8_dispatch,
            config=cfg,
        )
    outputs = backend.dispatch(
        x,
        handle=handle,
        topk_idx=topk_idx,
        token_weights=token_weights,
        num_experts=num_experts,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        num_worst_tokens=num_worst_tokens,
        config=cfg.dispatch_config,
    )

    # Bind the autotune result to the new handle for paired-call reuse.
    if result is not None:
        new_handle = outputs[-1]
        MoEDispatchCombineAutoTuner.register_handle(new_handle, result)

    return outputs


def moe_combine_impl(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    handle: tuple,
    topk_weights: Optional[torch.Tensor] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Combine expert outputs back to original token order via the selected backend.

    Args:
        x: Expert-side tensor to be combined.
        group: EP process group.
        handle: Handle returned by the paired ``moe_dispatch_impl`` call.
        topk_weights: Optional per-token expert weights.
        async_finish: If True, return before the kernel finishes.
        allocate_on_comm_stream: Allocate outputs on the comm stream.

    Returns:
        ``(combined_x, combined_topk_weights)``.
    """
    name, cfg, _ = _maybe_tune_config(x, group, handle, None, None, None)
    _warn_if_graph_capture_incompatible(name, "combine")
    backend = _get_backend_instance(name)
    assert backend.is_initialized(), "Backend is not initialized"
    return backend.combine(
        x,
        handle=handle,
        topk_weights=topk_weights,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
        config=cfg.combine_config,
    )
