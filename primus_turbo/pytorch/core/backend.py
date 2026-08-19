###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Hashable, List, Optional, Type

import torch

from primus_turbo.common.constants import (
    ENV_ATTN_BACKEND,
    ENV_AUTO_TUNE,
    ENV_GEMM_BACKEND,
    ENV_GROUPED_GEMM_BACKEND,
    ENV_MOE_DISPATCH_COMBINE_BACKEND,
)
from primus_turbo.common.logger import logger
from primus_turbo.triton.utils.origami import origami_clear_caches

try:
    HAVE_DEEP_EP = True
    import deep_ep  # noqa: F401
except ImportError:
    HAVE_DEEP_EP = False


__all__ = [
    "BackendChoice",
    "BackendEntry",
    "BackendType",
    "GlobalBackendManager",
    "KernelBackend",
    "PrecisionType",
    "TuneCache",
    "AutoKernelDispatcher",
]


class PrecisionType(Enum):
    FP4 = auto()
    FP8 = auto()
    BF16_FP16_FP32 = auto()


_PRECISION_TYPE_MAPPING = {
    "FP4": PrecisionType.FP4,
    "FP8": PrecisionType.FP8,
    "BF16": PrecisionType.BF16_FP16_FP32,
    "FP16": PrecisionType.BF16_FP16_FP32,
    "FP32": PrecisionType.BF16_FP16_FP32,
}
_PRECISION_TYPE_SET = set(_PRECISION_TYPE_MAPPING.values())
_OTHER_PRECISION_HOLDER = "OTHER"
# Accepted wherever a backend name is, to auto-tune that op/precision alone.
_AUTO_TUNE_HOLDER = "AUTOTUNE"


class BackendType(Enum):
    CK = auto()
    HIPBLASLT = auto()
    AITER = auto()
    TRITON = auto()
    DEEP_EP = auto()
    TURBO = auto()
    FLYDSL = auto()
    # HipKittens attention. gfx950-only at runtime: the kernels are built into the shared extension
    # in all arch configurations but are guarded on __gfx950__ and must be capability-gated before
    # launch (see is_gfx950()/hipkittens_attn_supported).
    HIPKITTENS = auto()


@dataclass
class BackendChoice:
    backend: Optional[BackendType] = None
    auto_tune: bool = False


class GlobalBackendManager:
    """
    Global Backend manager.

    Priority (high to low):
    1. Code settings - set_gemm_backend(), etc.
    2. Environment variables - PRIMUS_TURBO_GEMM_BACKEND, etc.
    3. Auto-tune - PRIMUS_TURBO_AUTO_TUNE=1 for every op, or
       set_*_backend(auto_tune=True) to auto-tune a single op/precision.
    4. Code defaults
    5. Fallback: try all backends
    """

    _gemm_backend: Optional[Dict[PrecisionType, BackendChoice]] = None
    _grouped_gemm_backend: Optional[Dict[PrecisionType, BackendChoice]] = None
    _moe_dispatch_combine_backend: Optional[Dict[PrecisionType, BackendChoice]] = None
    _attn_backend: Optional[Dict[PrecisionType, Optional[BackendType]]] = None
    _auto_tune: Optional[bool] = None
    _env_cache: Dict[str, Dict["PrecisionType", "BackendChoice"]] = {}

    @staticmethod
    def _parse_backend_choice(name: str) -> BackendChoice:
        """Turn one backend name from an env var into a ``BackendChoice``.

        Raises ``KeyError`` for names that are neither ``AUTOTUNE`` nor a
        ``BackendType`` member; callers decide whether that is fatal.
        """
        name = name.strip().upper()
        if name == _AUTO_TUNE_HOLDER:
            return BackendChoice(auto_tune=True)
        return BackendChoice(backend=BackendType[name])

    @classmethod
    def _extract_backend_from_env(cls, env_value: str) -> Dict[PrecisionType, BackendChoice]:
        """
        Extract the backend from the environment variable.
        Support formats. Example:
        1. ENV_KEY=backend -> All precison use the same backend
        2. ENV_KEY=<precision1>:<backend1>,<precision2>:<backend2>,... -> Each precision uses a different backend
        3. ENV_KEY=<precision1>:<backend1>,other:<backend2>,... -> precision1 use backend1, other precisions use backend2

        Precision types are defined in the _PRECISION_TYPE_MAPPING. Any backend
        slot also accepts "autotune" to auto-tune that precision alone.
        """
        if env_value in cls._env_cache:
            return cls._env_cache[env_value]

        precision_backend_dict = {}

        # Parse format 2 & 3
        env_lower = env_value.lower()
        if any(key_word in env_lower for key_word in ("fp4", "fp8", "bf16", "fp16", "fp32", "other")):
            precision_backend_pairs = env_value.split(",")
            other_precision_backend = None
            for pair in precision_backend_pairs:
                if pair.strip() == "":
                    continue
                precision, backend = pair.split(":")
                precision, backend = precision.strip().upper(), backend.strip().upper()
                if precision == _OTHER_PRECISION_HOLDER:
                    other_precision_backend = cls._parse_backend_choice(backend)
                    continue
                assert precision in _PRECISION_TYPE_MAPPING, f"Precision {precision} not supported."
                precision_backend_dict[_PRECISION_TYPE_MAPPING[precision]] = cls._parse_backend_choice(
                    backend
                )

            # Set rest precisions to the other precision backend
            for precision in _PRECISION_TYPE_MAPPING.values():
                if precision not in precision_backend_dict:
                    precision_backend_dict[precision] = other_precision_backend
        else:
            # Parse format 1: ENV_KEY=backend -> All precison use the same backend
            choice = cls._parse_backend_choice(env_value)
            for value in _PRECISION_TYPE_MAPPING.values():
                precision_backend_dict[value] = choice

        cls._env_cache[env_value] = precision_backend_dict
        return precision_backend_dict

    @classmethod
    def _clear_env_cache(cls) -> None:
        """Clear the cached parses of backend env vars.

        Replaces the previous ``_extract_backend_from_env.cache_clear()``
        contract from when this method was wrapped with ``functools.lru_cache``.
        Tests and any external callers that need to invalidate the cache
        should call this instead.
        """
        cls._env_cache.clear()

    @staticmethod
    def _updated_backend_table(
        table: Optional[Dict[PrecisionType, BackendChoice]],
        backend: Optional[BackendType],
        precision: Optional[PrecisionType],
        auto_tune: bool,
    ) -> Dict[PrecisionType, BackendChoice]:
        """Build the per-precision table produced by one ``set_*_backend`` call.

        ``precision=None`` applies the choice to every precision; otherwise only
        that precision is updated and the rest of ``table`` is kept.
        """
        if auto_tune:
            assert backend is None, "Backend must be None when auto-tune is enabled"

        choice = BackendChoice(backend=backend, auto_tune=auto_tune)
        if precision is None:
            return {p: choice for p in _PRECISION_TYPE_SET}

        table = table if table is not None else {}
        table[precision] = choice
        return table

    @classmethod
    def _backend_from_env(
        cls, env_key: str, precision: PrecisionType, allow_unknown_backend: bool = False
    ) -> Optional[BackendChoice]:
        """Read one precision's backend out of ``env_key``. None if unset or unusable.

        With ``allow_unknown_backend``, a name that is not a ``BackendType`` yields
        None instead of raising, for ops whose backends are not all modelled by the
        enum (MoE dispatch/combine accepts custom EP names such as ``UCCL_EP``).
        """
        env_value = os.environ.get(env_key, None)
        # Treat an empty / whitespace-only env var as missing (else
        # _extract_backend_from_env raises KeyError on BackendType['']).
        if env_value is None or not env_value.strip():
            return None

        try:
            choice = cls._extract_backend_from_env(env_value).get(precision, None)
        except KeyError:
            if allow_unknown_backend:
                return None
            raise

        if choice is None:
            logger.warning(
                f"Precision {precision.name} not found in the environment variable {env_key}. "
                f"Using default backend.",
                once=True,
            )
            return None

        return choice

    @classmethod
    def set_gemm_backend(
        cls,
        backend: Optional[BackendType] = None,
        precision: Optional[PrecisionType] = None,
        auto_tune: bool = False,
    ) -> None:
        """Set the GEMM backend in code."""
        cls._gemm_backend = cls._updated_backend_table(cls._gemm_backend, backend, precision, auto_tune)

    @classmethod
    def set_grouped_gemm_backend(
        cls,
        backend: Optional[BackendType] = None,
        precision: Optional[PrecisionType] = None,
        auto_tune: bool = False,
    ) -> None:
        """Set the Grouped GEMM backend in code."""
        cls._grouped_gemm_backend = cls._updated_backend_table(
            cls._grouped_gemm_backend, backend, precision, auto_tune
        )

    @classmethod
    def set_moe_dispatch_combine_backend(
        cls,
        backend: Optional[BackendType] = None,
        precision: Optional[PrecisionType] = None,
        auto_tune: bool = False,
    ) -> None:
        """Set the MoE dispatch/combine backend in code."""
        assert auto_tune is False, "Auto-tune is not supported for MOE dispatch combine backend"

        cls._moe_dispatch_combine_backend = cls._updated_backend_table(
            cls._moe_dispatch_combine_backend, backend, precision, auto_tune
        )

    @classmethod
    def set_auto_tune(cls, enabled: Optional[bool]) -> None:
        """Set whether auto-tune is enabled in code."""
        cls._auto_tune = enabled

    @classmethod
    def get_gemm_backend(cls, precision: PrecisionType) -> Optional[BackendChoice]:
        """Get the GEMM backend configuration. Returns None if not set."""
        if cls._gemm_backend is not None:
            # .get(): setting one precision leaves the others unconfigured.
            return cls._gemm_backend.get(precision)
        return cls._backend_from_env(ENV_GEMM_BACKEND, precision)

    @classmethod
    def get_grouped_gemm_backend(cls, precision: PrecisionType) -> Optional[BackendChoice]:
        """Get the Grouped GEMM backend configuration. Returns None if not set."""
        if cls._grouped_gemm_backend is not None:
            return cls._grouped_gemm_backend.get(precision)
        return cls._backend_from_env(ENV_GROUPED_GEMM_BACKEND, precision)

    @classmethod
    def set_attn_backend(
        cls, backend: Optional[BackendType] = None, precision: Optional[PrecisionType] = None
    ) -> None:
        """Set the attention backend in code; flash-attention and sparse-MLA share it."""
        if backend is None:
            cls._attn_backend = None
            return

        if cls._attn_backend is None:
            cls._attn_backend = {}

        if precision is None:
            cls._attn_backend = {precision: backend for precision in _PRECISION_TYPE_SET}
        else:
            cls._attn_backend[precision] = backend

    @classmethod
    def get_attn_backend(cls, precision: PrecisionType) -> Optional[BackendType]:
        """Get the attention backend configuration. Returns None if not set.

        Flash-attention and sparse-MLA read the same setting; they do not have the same
        backends, so each dispatcher drops a name it does not carry (see
        resolve_sparse_mla_fwd_backend) rather than failing on it."""
        if cls._attn_backend is not None:
            return cls._attn_backend.get(precision)
        # env parsing yields BackendChoice; attn dispatch consumes the bare enum.
        choice = cls._backend_from_env(ENV_ATTN_BACKEND, precision)
        return choice.backend if choice is not None else None

    @classmethod
    def get_moe_dispatch_combine_backend(cls, precision: PrecisionType) -> Optional[BackendChoice]:
        """Get the MoE dispatch combine backend configuration. Returns None if not set.

        If the environment variable contains a value that is not a valid ``BackendType``
        (e.g. a custom EP backend name like ``UCCL_EP``), this method returns ``None`` so
        the EP-specific backend registry in ``moe_dispatch_combine_impl`` can handle it.
        """
        if cls._moe_dispatch_combine_backend is not None:
            return cls._moe_dispatch_combine_backend.get(precision)

        choice = cls._backend_from_env(
            ENV_MOE_DISPATCH_COMBINE_BACKEND, precision, allow_unknown_backend=True
        )
        if choice is None:
            return None

        # Dispatch/combine picks an EP backend that owns persistent communication
        # buffers, so there is nothing to profile and swap per call.
        assert not choice.auto_tune, (
            f"{ENV_MOE_DISPATCH_COMBINE_BACKEND}=AUTOTUNE is not supported: "
            "MoE dispatch/combine has no auto-tune path."
        )

        if choice.backend == BackendType.DEEP_EP:
            assert HAVE_DEEP_EP, (
                "DeepEP is required for this module. Install from https://github.com/uccl-project/uccl or https://github.com/ROCm/DeepEP"
            )
        return choice

    @classmethod
    def auto_tune_enabled(cls) -> bool:
        """Check whether the global auto-tune switch is on.

        This is the process-wide switch only; an op may still be auto-tuned
        through a per-op request while this returns False.
        """
        if cls._auto_tune is not None:
            return cls._auto_tune
        return os.environ.get(ENV_AUTO_TUNE, "0") == "1"

    @classmethod
    def reset(cls) -> None:
        """Reset all backend settings and clear all dispatcher caches."""
        cls._gemm_backend = None
        cls._grouped_gemm_backend = None
        cls._moe_dispatch_combine_backend = None
        cls._attn_backend = None
        cls._auto_tune = None
        cls._env_cache = {}
        AutoKernelDispatcher.clear_all_caches()
        origami_clear_caches()


class KernelBackend(ABC):
    @staticmethod
    @abstractmethod
    def can_handle(**kwargs) -> bool:
        raise NotImplementedError("can_handle is not implemented")

    @staticmethod
    @abstractmethod
    def execute(**kwargs):
        raise NotImplementedError("execute is not implemented")


@dataclass(frozen=True)
class BackendEntry:
    """Metadata wrapper for a registered kernel backend.

    Attributes:
        impl: The kernel backend class.
        autotune: Whether this backend participates in auto-tuning.
                  Backends with autotune=False can still be selected via
                  explicit user configuration or as a fallback.
    """

    impl: Type[KernelBackend]
    autotune: bool = True


class TuneCache:
    """LRU cache for storing tuned backend results."""

    def __init__(self, capacity: int = 1024):
        self._capacity = capacity
        self._cache: OrderedDict[Hashable, Type[KernelBackend]] = OrderedDict()

    def get(self, key: Hashable) -> Optional[Type[KernelBackend]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: Hashable, value: Type[KernelBackend]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self._capacity:
            warnings.warn(
                f"TuneCache capacity ({self._capacity}) exceeded. "
                f"Input shapes changing frequently - AutoTune may not be beneficial. "
                f"Consider disabling AutoTune or using fixed shapes.",
                stacklevel=2,
            )
            self._cache.popitem(last=False)
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._cache


def _format_kwargs(kwargs: Dict[str, Any]) -> str:

    def _format_value(v):
        if isinstance(v, torch.Tensor):
            return f"Tensor(shape={v.shape}, dtype={v.dtype})"
        if isinstance(v, Enum):
            return f"{type(v).__name__}.{v.name}"
        return repr(v)

    return ", ".join(f"{k}={_format_value(v)}" for k, v in kwargs.items())


def _warn_fallback(backend_enum: BackendType, kwargs: dict) -> None:
    """Say once that the default backend was not eligible and we fell back.

    Skipped under torch.compile: dynamo cannot trace a logging.Logger call, and backend
    selection is host-side dispatch, so tracing into it would fail the compile over a
    diagnostic. Formatting the kwargs is not cheap either, and it sits inside the guard.
    """
    if torch.compiler.is_compiling():
        return
    logger.warning(
        f"For inputs: {_format_kwargs(kwargs)}, the default backend is not compatible, "
        f"fallback backend {backend_enum.name} is selected. The fallback backend may hurt performance!",
        once=True,
    )


def drop_unregistered_backend(dispatcher, user_backend: Optional[BackendType]) -> Optional[BackendType]:
    """Treat a backend this dispatcher has no entry for as "no preference".

    One env var names a backend for several dispatchers that do not carry the same set
    (both attention ones share PRIMUS_TURBO_ATTN_BACKEND, and only flash-attn has aiter),
    so a name meant for its sibling must not take this one down. Say so once, since the
    name being ignored here is otherwise indistinguishable from it being honoured.
    """
    if user_backend is None or user_backend in dispatcher._backends:
        return user_backend
    if not torch.compiler.is_compiling():
        logger.warning(
            f"Backend {user_backend.name} is not one of {dispatcher.__name__}'s "
            f"({[b.name for b in dispatcher._backends]}); ignoring it for this op.",
            once=True,
        )
    return None


class AutoKernelDispatcher(ABC):  # noqa: B024
    """
    Base class for auto kernel dispatcher.
    """

    _backends: Dict[BackendType, BackendEntry] = {}
    _cache: Optional[TuneCache] = None
    _warmup_iters: int = 10
    _profile_iters: int = 20
    _subclasses: List[Type["AutoKernelDispatcher"]] = []

    @staticmethod
    def _is_graph_capturing() -> bool:
        fn = getattr(torch.cuda, "is_current_stream_capturing", None)
        if fn is None:
            graphs = getattr(torch.cuda, "graphs", None)
            fn = getattr(graphs, "is_current_stream_capturing", None) if graphs is not None else None
        try:
            return bool(fn()) if fn is not None else False
        except Exception:
            return False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_backends" not in cls.__dict__:
            cls._backends = {}
        if "_cache" not in cls.__dict__:
            cls._cache = TuneCache()
        AutoKernelDispatcher._subclasses.append(cls)

    @classmethod
    def clear_all_caches(cls) -> None:
        """Clear caches for all dispatcher subclasses."""
        for subclass in cls._subclasses:
            if subclass._cache is not None:
                subclass._cache.clear()

    @classmethod
    def make_key(cls, **kwargs) -> Hashable:
        raise NotImplementedError("Subclass should implement make_key")

    @classmethod
    @torch.no_grad()
    def profile(cls, backend: Type[KernelBackend], **kwargs) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # warm-up
        for _ in range(cls._warmup_iters):
            backend.execute(**kwargs)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        start.record()
        for _ in range(cls._profile_iters):
            backend.execute(**kwargs)
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / cls._profile_iters

    @classmethod
    def tune(cls, **kwargs) -> Optional[Type[KernelBackend]]:
        """Profile all compatible backends and cache the fastest one."""
        key = cls.make_key(**kwargs)

        cached_backend = cls._cache.get(key)
        if cached_backend is not None:
            return cached_backend

        best_backend = None
        best_time = float("inf")
        for entry in cls._backends.values():
            if not entry.autotune:
                continue
            if entry.impl.can_handle(**kwargs):
                torch.cuda.synchronize()
                try:
                    cur_time = cls.profile(entry.impl, **kwargs)
                except Exception:
                    cur_time = float("inf")
                finally:
                    torch.cuda.synchronize()
                if cur_time < best_time:
                    best_time = cur_time
                    best_backend = entry.impl

        if best_backend is not None:
            cls._cache.put(key, best_backend)
        return best_backend

    @classmethod
    def dispatch(
        cls,
        default_backend_choice: BackendChoice,
        user_backend_choice: Optional[BackendChoice] = None,
        **kwargs,
    ) -> Any:
        # 1. User specified backend (env or code) - highest priority
        if user_backend_choice is not None and user_backend_choice.backend is not None:
            user_backend_enum = user_backend_choice.backend
            if user_backend_enum not in cls._backends:
                raise ValueError(
                    f"User specified backend {user_backend_enum.name} is not registered for {cls.__name__}. "
                    f"Available backends: {[b.name for b in cls._backends.keys()]}"
                )
            entry = cls._backends[user_backend_enum]
            if not entry.impl.can_handle(**kwargs):
                raise ValueError(
                    f"User specified backend {user_backend_enum.name} cannot handle the given inputs: {_format_kwargs(kwargs)}. "
                    f"Please check input constraints or choose a different backend."
                )
            return entry.impl.execute(**kwargs)

        # 2. Auto tune
        # NOTE: Skip autotune during cuda graph capture.
        if (
            (user_backend_choice is not None and user_backend_choice.auto_tune)
            or default_backend_choice.auto_tune
            or GlobalBackendManager.auto_tune_enabled()
        ) and not cls._is_graph_capturing():
            backend_cls = cls.tune(**kwargs)
            if backend_cls is not None:
                return backend_cls.execute(**kwargs)

        # 3. Default backend
        default_entry = cls._backends.get(default_backend_choice.backend)
        if default_entry is not None and default_entry.impl.can_handle(**kwargs):
            return default_entry.impl.execute(**kwargs)

        # 4. Fallback: try all backends
        for fallback_backend_enum, fallback_backend_entry in cls._backends.items():
            if fallback_backend_entry.impl.can_handle(**kwargs):
                _warn_fallback(fallback_backend_enum, kwargs)
                return fallback_backend_entry.impl.execute(**kwargs)

        raise ValueError(
            f"No compatible backend found for {cls.__name__} with inputs: {_format_kwargs(kwargs)}"
        )

    @classmethod
    def _enum_for_impl(cls, impl: Type[KernelBackend]) -> Optional[BackendType]:
        """Reverse-lookup the BackendType enum for a registered backend impl class."""
        for backend_enum, entry in cls._backends.items():
            if entry.impl is impl:
                return backend_enum
        return None

    @classmethod
    def resolve(
        cls, default_backend_enum: BackendType, user_backend_enum: Optional[BackendType] = None, **kwargs
    ) -> BackendType:
        """Select (but do not execute) the backend enum for the given inputs.

        Follows the same priority order as ``dispatch``: user > autotune >
        default > fallback. Exposed so callers that run a kernel across multiple
        passes (e.g. an autograd forward/backward pair) can pin the *same*
        backend for every pass by resolving once and threading the enum through.
        """
        # 1. User specified backend (env or code) - highest priority
        if user_backend_enum is not None:
            if user_backend_enum not in cls._backends:
                raise ValueError(
                    f"User specified backend {user_backend_enum.name} is not registered for {cls.__name__}. "
                    f"Available backends: {[b.name for b in cls._backends.keys()]}"
                )
            if not cls._backends[user_backend_enum].impl.can_handle(**kwargs):
                raise ValueError(
                    f"User specified backend {user_backend_enum.name} cannot handle the given inputs: {_format_kwargs(kwargs)}. "
                    f"Please check input constraints or choose a different backend."
                )
            return user_backend_enum

        # 2. Auto tune
        # NOTE: Skip autotune during cuda graph capture.
        if GlobalBackendManager.auto_tune_enabled() and not cls._is_graph_capturing():
            backend_cls = cls.tune(**kwargs)
            tuned_enum = cls._enum_for_impl(backend_cls) if backend_cls is not None else None
            if tuned_enum is not None:
                return tuned_enum

        # 3. Default backend
        default_entry = cls._backends.get(default_backend_enum)
        if default_entry is not None and default_entry.impl.can_handle(**kwargs):
            return default_backend_enum

        # 4. Fallback: try all backends
        for fallback_backend_enum, fallback_backend_entry in cls._backends.items():
            if fallback_backend_entry.impl.can_handle(**kwargs):
                _warn_fallback(fallback_backend_enum, kwargs)
                return fallback_backend_enum

        raise ValueError(
            f"No compatible backend found for {cls.__name__} with inputs: {_format_kwargs(kwargs)}"
        )
