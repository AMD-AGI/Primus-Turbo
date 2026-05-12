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
from functools import lru_cache
from typing import Any, Dict, Hashable, List, Optional, Type

import torch

from primus_turbo.common.constants import (
    ENV_AUTO_TUNE,
    ENV_GEMM_BACKEND,
    ENV_GROUPED_GEMM_BACKEND,
    ENV_MOE_DISPATCH_COMBINE_BACKEND,
)
from primus_turbo.common.logger import logger
from primus_turbo.triton.gemm.gemm_kernel import clear_origami_caches

try:
    HAVE_DEEP_EP = True
    import deep_ep  # fmt: skip
except ImportError:
    HAVE_DEEP_EP = False


__all__ = [
    "BackendEntry",
    "BackendType",
    "GlobalBackendManager",
    "KernelBackend",
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


class BackendType(Enum):
    CK = auto()
    HIPBLASLT = auto()
    AITER = auto()
    TRITON = auto()
    DEEP_EP = auto()
    TURBO = auto()


class GlobalBackendManager:
    """
    Global Backend manager.

    Priority (high to low):
    1. Code settings - set_gemm_backend(), etc.
    2. Environment variables - PRIMUS_TURBO_GEMM_BACKEND, etc.
    3. Auto-tune - PRIMUS_TURBO_AUTO_TUNE=1
    4. Code defaults
    5. Fallback: try all backends
    """

    _gemm_backend: Dict[PrecisionType, Optional[BackendType]] = None
    _grouped_gemm_backend: Dict[PrecisionType, Optional[BackendType]] = None
    _moe_dispatch_combine_backend: Dict[PrecisionType, Optional[BackendType]] = None
    _auto_tune: Optional[bool] = None

    @classmethod
    @lru_cache(maxsize=32)
    def _extract_backend_from_env(cls, env_value: str) -> Dict[PrecisionType, BackendType]:
        """
        Extract the backend from the environment variable.
        Support formats. Example:
        1. ENV_KEY=backend -> All precison use the same backend
        2. ENV_KEY=<precision1>:<backend1>,<precision2>:<backend2>,... -> Each precision uses a different backend
        3. ENV_KEY=<precision1>:<backend1>,other:<backend2>,... -> precision1 use backend1, other precisions use backend2

        Precision types are defined in the _PRECISION_TYPE_MAPPING.
        """
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
                    other_precision_backend = BackendType[backend]
                    continue
                assert precision in _PRECISION_TYPE_MAPPING, f"Precision {precision} not supported."
                precision_backend_dict[_PRECISION_TYPE_MAPPING[precision]] = BackendType[backend]

            # Set rest precisions to the other precision backend
            for precision in _PRECISION_TYPE_MAPPING.values():
                if precision not in precision_backend_dict:
                    precision_backend_dict[precision] = other_precision_backend
        else:
            # Parse format 1: ENV_KEY=backend -> All precison use the same backend
            for value in _PRECISION_TYPE_MAPPING.values():
                precision_backend_dict[value] = BackendType[env_value.upper()]

        return precision_backend_dict

    @classmethod
    def set_gemm_backend(
        cls, backend: Optional[BackendType] = None, precision: Optional[PrecisionType] = None
    ) -> None:
        """Set the GEMM backend in code."""
        if backend is None:
            cls._gemm_backend = None
            return

        if cls._gemm_backend is None:
            cls._gemm_backend = {}

        # backend is not None
        if precision is None:
            # preicision is None -> set all precisions to the same backend
            cls._gemm_backend = {precision: backend for precision in _PRECISION_TYPE_SET}
        else:
            cls._gemm_backend[precision] = backend

    @classmethod
    def set_grouped_gemm_backend(
        cls, backend: Optional[BackendType] = None, precision: Optional[PrecisionType] = None
    ) -> None:
        """Set the Grouped GEMM backend in code."""
        if backend is None:
            cls._grouped_gemm_backend = None
            return

        if cls._grouped_gemm_backend is None:
            cls._grouped_gemm_backend = {}

        # backend is not None
        if precision is None:
            # preicision is None -> set all precisions to the same backend
            cls._grouped_gemm_backend = {precision: backend for precision in _PRECISION_TYPE_SET}
        else:
            cls._grouped_gemm_backend[precision] = backend

    @classmethod
    def set_auto_tune(cls, enabled: Optional[bool]) -> None:
        """Set whether auto-tune is enabled in code."""
        cls._auto_tune = enabled

    @classmethod
    def get_gemm_backend(cls, precision: PrecisionType) -> Optional[BackendType]:
        """Get the GEMM backend configuration. Returns None if not set."""
        if cls._gemm_backend is not None:
            return cls._gemm_backend[precision]
        env_value = os.environ.get(ENV_GEMM_BACKEND, None)
        if env_value is not None:
            backend = cls._extract_backend_from_env(env_value).get(precision, None)
            if backend is None:
                logger.warning(
                    f"Precision {precision.name} not found in the environment variable {ENV_GEMM_BACKEND}. "
                    f"Using default backend.",
                    once=True,
                )
            return backend

        return None

    @classmethod
    def get_grouped_gemm_backend(cls, precision: PrecisionType) -> Optional[BackendType]:
        """Get the Grouped GEMM backend configuration. Returns None if not set."""
        if cls._grouped_gemm_backend is not None:
            return cls._grouped_gemm_backend[precision]
        env_value = os.environ.get(ENV_GROUPED_GEMM_BACKEND, None)
        if env_value is not None:
            backend = cls._extract_backend_from_env(env_value).get(precision, None)
            if backend is None:
                logger.warning(
                    f"Precision {precision.name} not found in the environment variable "
                    f"{ENV_GROUPED_GEMM_BACKEND}. Using default backend.",
                    once=True,
                )
            return backend

        return None

    @classmethod
    def get_moe_dispatch_combine_backend(cls, precision: PrecisionType) -> Optional[BackendType]:
        """Get the MoE dispatch combine backend configuration. Returns None if not set.

        If the environment variable contains a value that is not a valid ``BackendType``
        (e.g. a custom EP backend name like ``UCCL_EP``), this method returns ``None`` so
        the EP-specific backend registry in ``moe_dispatch_combine_impl`` can handle it.
        """
        if cls._moe_dispatch_combine_backend is not None:
            return cls._moe_dispatch_combine_backend[precision]
        env_value = os.environ.get(ENV_MOE_DISPATCH_COMBINE_BACKEND, None)
        if env_value is not None:
            try:
                backend = cls._extract_backend_from_env(env_value).get(precision, None)
            except KeyError:
                return None

            if backend is None:
                logger.warning(
                    f"Precision {precision.name} not found in the environment variable "
                    f"{ENV_MOE_DISPATCH_COMBINE_BACKEND}. Using default backend.",
                    once=True,
                )

            if backend == BackendType.DEEP_EP:
                assert (
                    HAVE_DEEP_EP
                ), "DeepEP is required for this module. Install from https://github.com/uccl-project/uccl or https://github.com/ROCm/DeepEP"
            return backend

        return None

    @classmethod
    def auto_tune_enabled(cls) -> bool:
        """Check if auto-tune is enabled."""
        if cls._auto_tune is not None:
            return cls._auto_tune
        return os.environ.get(ENV_AUTO_TUNE, "0") == "1"

    @classmethod
    def reset(cls) -> None:
        """Reset all backend settings and clear all dispatcher caches."""
        cls._gemm_backend = None
        cls._grouped_gemm_backend = None
        cls._auto_tune = None
        AutoKernelDispatcher.clear_all_caches()
        clear_origami_caches()


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


class AutoKernelDispatcher(ABC):
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
        cls, default_backend_enum: BackendType, user_backend_enum: Optional[BackendType] = None, **kwargs
    ) -> Any:
        # 1. User specified backend (env or code) - highest priority

        if user_backend_enum is not None:
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
        if GlobalBackendManager.auto_tune_enabled() and not cls._is_graph_capturing():
            backend_cls = cls.tune(**kwargs)
            if backend_cls is not None:
                return backend_cls.execute(**kwargs)

        # 3. Default backend
        default_entry = cls._backends.get(default_backend_enum)
        if default_entry is not None and default_entry.impl.can_handle(**kwargs):
            return default_entry.impl.execute(**kwargs)

        # 4. Fallback: try all backends
        for fallback_backend_enum, fallback_backend_entry in cls._backends.items():
            if fallback_backend_entry.impl.can_handle(**kwargs):
                logger.warning(
                    f"For inputs: {_format_kwargs(kwargs)}, the default backend is not compatible, fallback backend {fallback_backend_enum.name} is selected. The fallback backend may hurt performance!",
                    once=True,
                )
                return fallback_backend_entry.impl.execute(**kwargs)

        raise ValueError(
            f"No compatible backend found for {cls.__name__} with inputs: {_format_kwargs(kwargs)}"
        )
