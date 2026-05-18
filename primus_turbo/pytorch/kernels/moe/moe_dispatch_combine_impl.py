###############################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import dataclasses
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.core.backend import (
    GlobalBackendManager,
    PrecisionType,
    TuneCache,
)

# =========================================================================
# Buffer configuration
# =========================================================================


def _compute_expert_token_info_configs() -> List[triton.Config]:
    """Autotune space for :func:`compute_expert_token_info_kernel`."""
    configs: List[triton.Config] = []
    for block_m in (64, 128, 256, 512, 1024):
        for num_warps in (1, 2, 4, 8):
            # Wave=64, cap threads/CTA to 1024 (HW limit on CDNA).
            if num_warps * 64 > 1024:
                continue
            # Keep at least one element per thread in the M dimension.
            if block_m < num_warps * 64:
                continue
            for num_stages in (1, 2):
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE_M": block_m},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=_compute_expert_token_info_configs(),
    key=["num_tokens", "num_topk"],
    reset_to_zero=["num_recv_tokens_per_expert_ptr"],
)
@triton.jit
def compute_expert_token_info_kernel(
    recv_topk_idx_ptr,
    deepep_topk_idx_ptr,
    num_recv_tokens_per_expert_ptr,
    total_recv_ptr,
    num_tokens: tl.int32,
    num_topk: tl.int32,
    num_local_experts: tl.int32,
    expert_base: tl.int32,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HAS_TOTAL_RECV: tl.constexpr,
):
    """Count per-expert received tokens and emit a DeepEP-format topk_idx. will be removed in the future."""
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offs = tl.arange(0, BLOCK_SIZE_K)
    bin_offs = tl.arange(0, BLOCK_SIZE_N)

    if HAS_TOTAL_RECV:
        total_recv = tl.load(total_recv_ptr)
        row_limit = tl.minimum(total_recv, num_tokens)
    else:
        row_limit = num_tokens

    in_footprint = row_offs < num_tokens
    in_valid_rows = row_offs < row_limit
    col_mask = col_offs < num_topk

    footprint_mask = in_footprint[:, None] & col_mask[None, :]
    load_mask = in_valid_rows[:, None] & col_mask[None, :]

    safe_row = tl.where(in_footprint, row_offs, 0)
    expert_offs = safe_row[:, None] * num_topk + col_offs[None, :]

    topk_experts = tl.load(
        recv_topk_idx_ptr + expert_offs,
        mask=load_mask,
        other=0,
    )

    local_experts = topk_experts - expert_base
    valid = load_mask & (local_experts >= 0) & (local_experts < num_local_experts)
    # Clamp the experts of invalid lanes so they land on a legal bin; the
    # ``mask`` passed to ``tl.histogram`` makes sure they are not counted.
    safe_experts = tl.where(valid, local_experts, 0)

    # ``tl.histogram`` requires a flat 1D input; reshape the 2D tile.
    flat_experts = tl.reshape(safe_experts, [BLOCK_SIZE_M * BLOCK_SIZE_K])
    flat_mask = tl.reshape(valid, [BLOCK_SIZE_M * BLOCK_SIZE_K])
    local_counts = tl.histogram(flat_experts, BLOCK_SIZE_N, mask=flat_mask)

    # Flush CTA-local accumulator to global memory with one atomic per bin.
    bin_mask = (bin_offs < num_local_experts) & (local_counts > 0)
    tl.atomic_add(
        num_recv_tokens_per_expert_ptr + bin_offs,
        local_counts,
        sem="relaxed",
        scope="gpu",
        mask=bin_mask,
    )

    deepep_experts = tl.where(valid, local_experts, -1).to(topk_experts.dtype)
    tl.store(deepep_topk_idx_ptr + expert_offs, deepep_experts, mask=footprint_mask)


def compute_expert_token_info(
    recv_topk_idx: torch.Tensor,
    num_local_experts: int,
    *,
    expert_base: int = 0,
    total_recv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Count per-expert received tokens and emit ``recv_topk_idx`` in DeepEP
    layout (as a new tensor).
    """
    assert recv_topk_idx.is_cuda, "recv_topk_idx must be a CUDA tensor"
    assert recv_topk_idx.dim() == 2, "recv_topk_idx must be 2D [num_tokens, num_topk]"

    device = recv_topk_idx.device
    num_tokens, num_topk = recv_topk_idx.shape

    num_recv_tokens_per_expert = torch.zeros(num_local_experts, dtype=recv_topk_idx.dtype, device=device)
    deepep_like_topk_idx = torch.empty_like(recv_topk_idx)

    if num_tokens == 0 or num_topk == 0 or num_local_experts == 0:
        # Nothing to count; fill the output with the DeepEP padding sentinel
        # so callers can rely on a uniform layout.
        if deepep_like_topk_idx.numel() > 0:
            deepep_like_topk_idx.fill_(-1)
        return num_recv_tokens_per_expert, deepep_like_topk_idx

    has_total_recv = total_recv is not None
    if has_total_recv:
        assert total_recv.is_cuda, "total_recv must be a CUDA tensor"
        assert total_recv.numel() >= 1, "total_recv must have at least 1 element"
        # Triton expects an int32 pointer; coerce silently if the caller
        # passed (e.g.) int64.
        if total_recv.dtype != torch.int32:
            total_recv = total_recv.to(torch.int32)
        total_recv_ptr = total_recv
    else:
        # Triton requires a real pointer; reuse ``recv_topk_idx`` as a dummy
        # (the kernel won't dereference it because ``HAS_TOTAL_RECV=False``).
        total_recv_ptr = recv_topk_idx

    # ``tl.histogram`` requires ``num_bins`` to be a power of 2 on AMD Triton.
    block_size_n = triton.next_power_of_2(num_local_experts)

    # ``BLOCK_SIZE_M`` is picked by the autotuner; the grid size depends on it.
    def grid(META):
        return (triton.cdiv(num_tokens, META["BLOCK_SIZE_M"]),)  # noqa: E731

    compute_expert_token_info_kernel[grid](
        recv_topk_idx,
        deepep_like_topk_idx,
        num_recv_tokens_per_expert,
        total_recv_ptr,
        num_tokens,
        num_topk,
        num_local_experts,
        expert_base,
        BLOCK_SIZE_K=triton.next_power_of_2(num_topk),
        BLOCK_SIZE_N=block_size_n,
        HAS_TOTAL_RECV=has_total_recv,
    )
    return num_recv_tokens_per_expert, deepep_like_topk_idx


@dataclass
class EPBufferConfig:
    """Configuration for EP communication buffer initialization.

    Attributes:
        num_sms: Number of SMs used by high-throughput kernels.
        dispatch_config: Dispatch config; ``None`` means use backend default.
        combine_config: Combine config; ``None`` means use backend default.
    """

    num_sms: int = 64
    dispatch_config: Any = None
    combine_config: Any = None


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
    "UCCL_EP": EPBufferConfig(
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
    """Set the EP buffer config for a backend, or for all backends when
    ``backend`` is ``None`` (backward-compat). Accepts SM count and an
    optional ``(dispatch_config, combine_config)`` tuple.
    """
    dispatch_cfg, combine_cfg = autotune_config if autotune_config is not None else (None, None)
    new_cfg = EPBufferConfig(
        num_sms=num_use_cu,
        dispatch_config=dispatch_cfg,
        combine_config=combine_cfg,
    )
    if backend is None:
        targets = set(_buffer_config_per_backend.keys()) | set(_BACKEND_REGISTRY.keys())
        for name in targets:
            _buffer_config_per_backend[name] = dataclasses.replace(new_cfg)
    else:
        _buffer_config_per_backend[backend] = new_cfg


# =========================================================================
# Backend imports
#
# Imported after ``EPBufferConfig`` and ``compute_expert_token_info`` are
# defined so that ``_backend`` submodules can resolve them via deferred
# (in-function) imports without a circular-import deadlock.
# =========================================================================

from primus_turbo.pytorch.kernels.moe._backend import (  # noqa: E402
    DeepEPBackend,
    EPBackend,
    MoriEPBackend,
    TurboEPBackend,
    UCCLEPBackend,
)

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
    """Autotuner that picks the best *(backend, dispatch_config, combine_config)*
    for a given MoE dispatch / combine case.

    Tuning is driven by ``moe_dispatch`` on a shape-based key; the resulting
    :class:`EPAutoTuneResult` is bound to the returned ``handle`` so that the
    paired ``moe_combine`` / cached-``moe_dispatch`` / autograd backward can
    reuse it without any further lookup.

    Enable via ``PRIMUS_TURBO_AUTO_TUNE=1`` or by calling
    :meth:`get_or_tune` explicitly.
    """

    # Shape-based cache shared across layers with identical input shape.
    _cache: TuneCache = TuneCache(capacity=1024)

    _HANDLE_CACHE_MAX: int = 1024
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
        """Return the result bound to ``handle``, or ``None`` if unknown."""
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

        best: Optional[EPAutoTuneResult] = None
        best_total = float("inf")
        per_backend: Dict[str, Tuple[float, float]] = {}

        for name in names:
            if name not in _BACKEND_REGISTRY or not _BACKEND_REGISTRY[name].is_available():
                continue
            backend = _get_backend_instance(name)
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
                # Backend opts out of config tuning (e.g. Mori has no
                # user-tunable Config sweep space). Skip quietly.
                logger.info(
                    f"MoE autotune: backend '{name}' does not support " f"tune_configs; skipping.",
                    once=True,
                    rank=0,
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive
                # Use ``repr`` so pybind-backed errors aren't silently blank.
                logger.warning(
                    f"MoE autotune: backend '{name}' failed: \n" f"{type(exc).__name__}: {exc!r}",
                    exc_info=True,
                )
                continue
            finally:
                # Drop the worst-case buffer; next init_buffer sizes correctly.
                backend.release_buffer()

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


def _get_backend_name() -> str:
    """Return the user-selected backend name, or ``TURBO`` by default."""
    bt = GlobalBackendManager.get_ep_backend(PrecisionType.BF16_FP16_FP32)
    return bt.name if bt is not None else _DEFAULT_BACKEND_NAME


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

    # Autotune on: user's backend choice is ignored when we have a tuned
    # result to fall back to.
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
    backend = _get_backend_instance(name)
    if num_experts is None or topk_idx is None:
        assert backend.is_initialized(), "Backend is not initialized"
    else:
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
