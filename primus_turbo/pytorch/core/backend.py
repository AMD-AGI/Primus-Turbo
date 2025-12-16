###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import os
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum, auto
from typing import Any, Dict, Hashable, List, Optional, Type

import torch

__all__ = [
    "BackendType",
    "GlobalBackendManager",
    "KernelBackend",
    "TuneCache",
    "AutoKernelDispatcher",
]


class BackendType(Enum):
    CK = auto()
    HIPBLASLT = auto()
    AITER = auto()
    TRITON = auto()


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

    _gemm_backend: Optional[BackendType] = None
    _grouped_gemm_backend: Optional[BackendType] = None
    _auto_tune: Optional[bool] = None

    @classmethod
    def set_gemm_backend(cls, backend: Optional[BackendType]) -> None:
        """Set the GEMM backend in code."""
        cls._gemm_backend = backend

    @classmethod
    def set_grouped_gemm_backend(cls, backend: Optional[BackendType]) -> None:
        """Set the Grouped GEMM backend in code."""
        cls._grouped_gemm_backend = backend

    @classmethod
    def set_auto_tune(cls, enabled: Optional[bool]) -> None:
        """Set whether auto-tune is enabled in code."""
        cls._auto_tune = enabled

    @classmethod
    def get_gemm_backend(cls) -> Optional[BackendType]:
        """Get the GEMM backend configuration. Returns None if not set."""
        if cls._gemm_backend is not None:
            return cls._gemm_backend
        backend = os.environ.get("PRIMUS_TURBO_GEMM_BACKEND")
        if backend:
            return BackendType[backend.upper()]
        return None

    @classmethod
    def get_grouped_gemm_backend(cls) -> Optional[BackendType]:
        """Get the Grouped GEMM backend configuration. Returns None if not set."""
        if cls._grouped_gemm_backend is not None:
            return cls._grouped_gemm_backend
        backend = os.environ.get("PRIMUS_TURBO_GROUPED_GEMM_BACKEND")
        if backend:
            return BackendType[backend.upper()]
        return None

    @classmethod
    def auto_tune_enabled(cls) -> bool:
        """Check if auto-tune is enabled."""
        if cls._auto_tune is not None:
            return cls._auto_tune
        return os.environ.get("PRIMUS_TURBO_AUTO_TUNE", "0") == "1"

    @classmethod
    def reset(cls) -> None:
        """Reset all backend settings and clear all dispatcher caches."""
        cls._gemm_backend = None
        cls._grouped_gemm_backend = None
        cls._auto_tune = None
        AutoKernelDispatcher.clear_all_caches()


class KernelBackend(ABC):
    @staticmethod
    @abstractmethod
    def can_handle(**kwargs) -> bool:
        raise NotImplementedError("can_handle is not implemented")

    @staticmethod
    @abstractmethod
    def execute(**kwargs):
        raise NotImplementedError("execute is not implemented")


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


class AutoKernelDispatcher(ABC):
    """
    Base class for auto kernel dispatcher.
    """

    _backends: Dict[BackendType, Type[KernelBackend]] = {}
    _cache: Optional[TuneCache] = None
    _warmup_iters: int = 10
    _profile_iters: int = 20
    _subclasses: List[Type["AutoKernelDispatcher"]] = []

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
        for _ in range(cls._warmup_iters):
            backend.execute(**kwargs)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

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
        for backend in cls._backends.values():
            if backend.can_handle(**kwargs):
                torch.cuda.synchronize()
                try:
                    cur_time = cls.profile(backend, **kwargs)
                except Exception:
                    cur_time = float("inf")
                finally:
                    torch.cuda.synchronize()
                if cur_time < best_time:
                    best_time = cur_time
                    best_backend = backend

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
            backend_cls = cls._backends[user_backend_enum]
            if not backend_cls.can_handle(**kwargs):
                raise ValueError(
                    f"User specified backend {user_backend_enum.name} cannot handle the given inputs. "
                    f"Please check input constraints or choose a different backend."
                )
            return backend_cls.execute(**kwargs)

        # 2. Auto tune
        if GlobalBackendManager.auto_tune_enabled():
            backend_cls = cls.tune(**kwargs)
            if backend_cls is not None:
                return backend_cls.execute(**kwargs)

        # 3. Default backend
        default_backend_cls = cls._backends.get(default_backend_enum)
        if default_backend_cls is not None and default_backend_cls.can_handle(**kwargs):
            return default_backend_cls.execute(**kwargs)

        # 4. Fallback: try all backends
        for fallback_backend_cls in cls._backends.values():
            if fallback_backend_cls.can_handle(**kwargs):
                return fallback_backend_cls.execute(**kwargs)

        raise ValueError(f"No compatible backend found for {cls.__name__} with kwargs: {kwargs}")
