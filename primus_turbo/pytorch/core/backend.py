###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import importlib
import json
import os
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Hashable, List, Optional, Type

import torch

from primus_turbo.common.constants import (
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

# Packaged offline tune-config assets: primus_turbo/tuning/configs/<framework>/<arch>/<op>.json
_TUNE_CONFIGS_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tuning", "configs"))


class BackendType(Enum):
    CK = auto()
    HIPBLASLT = auto()
    AITER = auto()
    TRITON = auto()
    DEEP_EP = auto()
    TURBO = auto()
    FLYDSL = auto()


class GlobalBackendManager:
    """
    Global Backend manager.

    Priority (high to low):
    1. Code settings - set_gemm_backend(), etc.
    2. Environment variables - PRIMUS_TURBO_GEMM_BACKEND, etc.
    3. Auto-tune - PRIMUS_TURBO_AUTO_TUNE=1
    4. Offline tune-config lookup - packaged configs/<arch>/<op>.json (auto-loaded)
    5. Code defaults
    6. Fallback: try all backends
    """

    _gemm_backend: Dict[PrecisionType, Optional[BackendType]] = None
    _grouped_gemm_backend: Dict[PrecisionType, Optional[BackendType]] = None
    _moe_dispatch_combine_backend: Dict[PrecisionType, Optional[BackendType]] = None
    _auto_tune: Optional[bool] = None
    _env_cache: Dict[str, Dict["PrecisionType", "BackendType"]] = {}

    @classmethod
    def _extract_backend_from_env(cls, env_value: str) -> Dict[PrecisionType, BackendType]:
        """
        Extract the backend from the environment variable.
        Support formats. Example:
        1. ENV_KEY=backend -> All precison use the same backend
        2. ENV_KEY=<precision1>:<backend1>,<precision2>:<backend2>,... -> Each precision uses a different backend
        3. ENV_KEY=<precision1>:<backend1>,other:<backend2>,... -> precision1 use backend1, other precisions use backend2

        Precision types are defined in the _PRECISION_TYPE_MAPPING.
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
        # Treat an empty / whitespace-only env var as missing (else
        # _extract_backend_from_env raises KeyError on BackendType['']).
        if env_value is not None and env_value.strip():
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
        if env_value is not None and env_value.strip():
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
        if env_value is not None and env_value.strip():
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
                assert HAVE_DEEP_EP, (
                    "DeepEP is required for this module. Install from https://github.com/uccl-project/uccl or https://github.com/ROCm/DeepEP"
                )
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
    def execute(backend_config: Optional[dict] = None, **kwargs):
        """Run the kernel.

        Every backend receives ``backend_config`` (a flat dict, or None) and decides
        whether to use it: backends with an internal autotune (Triton blockwise /
        FlyDSL) apply the pinned config (None -> their fixed default, never bench);
        all others ignore it.
        """
        raise NotImplementedError("execute is not implemented")

    @staticmethod
    def tune_config(**kwargs) -> Optional[dict]:
        """Return this backend's chosen internal config (a flat, JSON-able dict).

        Default ``None`` = no internal autotune to pin. Backends with a per-shape
        internal autotune (Triton blockwise / FlyDSL) override this to run that
        search once and return the winning config; ``execute`` then accepts it back
        via ``backend_config=`` to skip the runtime search (see dispatcher pin flow).
        """
        return None


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


# make_key tuples may hold ints/bools/str, torch.dtype and Enum members. Encode to a
# JSON-safe form that round-trips to the *identical* hashable (a loaded key must match
# what make_key produces at runtime). Both functions recurse into tuples/lists.
def _encode_key(x: Any) -> Any:
    if isinstance(x, torch.dtype):
        return {"__dtype__": str(x).removeprefix("torch.")}
    if isinstance(x, Enum):
        return {"__enum__": f"{type(x).__module__}:{type(x).__qualname__}", "name": x.name}
    if isinstance(x, (list, tuple)):
        return [_encode_key(e) for e in x]
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    raise TypeError(f"Unsupported tune-cache key element: {type(x)!r}")


def _decode_key(x: Any) -> Any:
    if isinstance(x, dict):
        if "__dtype__" in x:
            return getattr(torch, x["__dtype__"])
        if "__enum__" in x:
            module, qualname = x["__enum__"].split(":")
            return getattr(importlib.import_module(module), qualname)[x["name"]]
        raise ValueError(f"Bad tune-cache key element: {x!r}")
    if isinstance(x, list):
        return tuple(_decode_key(e) for e in x)
    return x


@dataclass
class TuneEntry:
    """One tuned result: the chosen backend, its optional internal config, and perf.

    ``backend_config`` is the backend's own opaque flat dict (None = nothing to pin);
    ``perf`` is profiling metadata (e.g. ``{"time_ms": ...}``), informational only.
    """

    backend: Type[KernelBackend]
    backend_config: Optional[dict] = None
    perf: Optional[dict] = None


class TuneCache:
    """LRU cache mapping a tune key to its :class:`TuneEntry`."""

    def __init__(self, capacity: int = 1024):
        self._capacity = capacity
        self._entries: OrderedDict[Hashable, TuneEntry] = OrderedDict()

    def get(self, key: Hashable) -> Optional[TuneEntry]:
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        return None

    def put(self, key: Hashable, entry: TuneEntry) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
        elif len(self._entries) >= self._capacity:
            warnings.warn(
                f"TuneCache capacity ({self._capacity}) exceeded. "
                f"Input shapes changing frequently - AutoTune may not be beneficial. "
                f"Consider disabling AutoTune or using fixed shapes.",
                stacklevel=2,
            )
            self._entries.popitem(last=False)
        self._entries[key] = entry

    def clear(self) -> None:
        self._entries.clear()

    def items(self) -> List[tuple]:
        """Return ``(key, TuneEntry)`` pairs (for serialization)."""
        return list(self._entries.items())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries


def _format_kwargs(kwargs: Dict[str, Any]) -> str:
    def _format_value(v):
        if isinstance(v, torch.Tensor):
            return f"Tensor(shape={v.shape}, dtype={v.dtype})"
        if isinstance(v, Enum):
            return f"{type(v).__name__}.{v.name}"
        return repr(v)

    return ", ".join(f"{k}={_format_value(v)}" for k, v in kwargs.items())


class AutoKernelDispatcher(ABC):  # noqa: B024
    """
    Base class for auto kernel dispatcher.
    """

    _backends: Dict[BackendType, BackendEntry] = {}
    _cache: Optional[TuneCache] = None
    _warmup_iters: int = 10
    _profile_iters: int = 20
    _subclasses: List[Type["AutoKernelDispatcher"]] = []
    # Basename of this dispatcher's tune-config asset, e.g. "gemm_fp8" ->
    # configs/pytorch/<arch>/gemm_fp8.json. None => no packaged asset to auto-load.
    _tune_config_name: Optional[str] = None
    _tune_config_loaded: bool = False  # latch: packaged asset auto-loaded at most once

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
            subclass._tune_config_loaded = False  # allow the packaged asset to re-load on next dispatch

    @classmethod
    def dump_cache(cls, path: str) -> int:
        """Serialize this dispatcher's tuned cache to a JSON file.

        Each entry is ``{"key", "backend", "backend_config", "perf"}``.
        ``backend_config`` is the selected backend's internal config (``null`` when
        the backend has none / not pinned); ``perf`` carries profiling metadata
        (e.g. ``{"time_ms": ...}``). Returns entries written.
        """
        if cls._cache is None:
            return 0
        impl_to_name = {entry.impl: bt.name for bt, entry in cls._backends.items()}
        entries = []
        for key, tuned in cls._cache.items():
            name = impl_to_name.get(tuned.backend)
            if name is None:
                continue  # backend not registered in _backends; skip
            entries.append(
                {
                    "key": _encode_key(key),
                    "backend": name,
                    "backend_config": tuned.backend_config,
                    "perf": tuned.perf,
                }
            )
        with open(path, "w") as f:
            json.dump({"dispatcher": cls.__name__, "entries": entries}, f, indent=2)
        return len(entries)

    @classmethod
    def load_cache(cls, path: str) -> int:
        """Load a JSON tune cache produced by ``dump_cache`` into this dispatcher.

        Loads ``key`` -> ``backend`` + ``backend_config`` (``perf`` is informational).
        Unknown / unregistered backends are skipped (logged once). Returns entries loaded.
        """
        if cls._cache is None:
            return 0
        with open(path) as f:
            data = json.load(f)
        loaded = 0
        for entry_json in data.get("entries", []):
            encoded_key, name = entry_json["key"], entry_json["backend"]
            entry = cls._backends.get(BackendType.__members__.get(name))
            if entry is None:
                logger.warning(f"Tune cache {path}: backend '{name}' unavailable, skipped.", once=True)
                continue
            cls._cache.put(
                _decode_key(encoded_key),
                TuneEntry(entry.impl, entry_json.get("backend_config"), entry_json.get("perf")),
            )
            loaded += 1
        return loaded

    @classmethod
    def _ensure_tune_config_loaded(cls) -> None:
        """Lazily load this dispatcher's packaged tune-config asset (once).

        Reads ``tuning/configs/pytorch/<arch>/<_tune_config_name>.json`` — no env var.
        Skipped while auto-tune is on (offline tuning / pure runtime tune ignore
        the asset, so the fresh profiling cache is never polluted).
        """
        if cls._tune_config_name is None or cls._tune_config_loaded:
            return
        if GlobalBackendManager.auto_tune_enabled():
            return  # retry once auto-tune is turned off
        cls._tune_config_loaded = True  # latch first: attempt at most once, even on failure
        try:
            arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        except Exception:
            return
        path = os.path.join(_TUNE_CONFIGS_ROOT, "pytorch", arch, f"{cls._tune_config_name}.json")
        if os.path.isfile(path):
            logger.info(f"{cls.__name__}: loaded {cls.load_cache(path)} tuned entries from {path}")

    @classmethod
    def make_key(cls, **kwargs) -> Hashable:
        raise NotImplementedError("Subclass should implement make_key")

    @classmethod
    @torch.no_grad()
    def profile(cls, backend: Type[KernelBackend], backend_config: Optional[dict] = None, **kwargs) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # warm-up
        for _ in range(cls._warmup_iters):
            backend.execute(**kwargs, backend_config=backend_config)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        start.record()
        for _ in range(cls._profile_iters):
            backend.execute(**kwargs, backend_config=backend_config)
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / cls._profile_iters

    @classmethod
    def tune_backend(cls, **kwargs) -> tuple[Optional[Type[KernelBackend]], Optional[dict]]:
        """Profile all compatible backends and cache the fastest one.

        Returns ``(backend, backend_config)``. ``backend_config`` is the winning
        backend's internal config (``None`` for backends without an internal
        autotune), captured so ``dump_cache`` can persist it for later pinning.
        Distinct from ``KernelBackend.tune_config`` (which picks a backend's *internal*
        config); this picks the *backend*.
        """
        key = cls.make_key(**kwargs)

        cached = cls._cache.get(key)
        if cached is not None:
            return cached.backend, cached.backend_config

        best_backend = None
        best_cfg = None
        best_time = float("inf")
        for entry in cls._backends.values():
            if not entry.autotune:
                continue
            if entry.impl.can_handle(**kwargs):
                cfg = entry.impl.tune_config(**kwargs)  # backend picks its internal config (or None)
                torch.cuda.synchronize()
                try:
                    cur_time = cls.profile(entry.impl, cfg, **kwargs)
                except Exception:
                    cur_time = float("inf")
                finally:
                    torch.cuda.synchronize()
                if cur_time < best_time:
                    best_time = cur_time
                    best_backend = entry.impl
                    best_cfg = cfg

        if best_backend is not None:
            cls._cache.put(key, TuneEntry(best_backend, best_cfg, {"time_ms": best_time}))
        return best_backend, best_cfg

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
            # auto-tune only this forced backend's internal config; else default cfg.
            cfg = None
            if GlobalBackendManager.auto_tune_enabled() and not cls._is_graph_capturing():
                cfg = entry.impl.tune_config(**kwargs)
                cls._cache.put(cls.make_key(**kwargs), TuneEntry(entry.impl, cfg))
            return entry.impl.execute(**kwargs, backend_config=cfg)

        # 2. Auto tune (skipped during cuda graph capture)
        if GlobalBackendManager.auto_tune_enabled() and not cls._is_graph_capturing():
            backend_cls, backend_cfg = cls.tune_backend(**kwargs)
            if backend_cls is not None:
                return backend_cls.execute(**kwargs, backend_config=backend_cfg)

        # 3. Tuned cache lookup (no profiling); auto-loads the packaged asset once.
        cls._ensure_tune_config_loaded()
        if cls._cache:
            cached = cls._cache.get(cls.make_key(**kwargs))
            if cached is not None and cached.backend.can_handle(**kwargs):
                return cached.backend.execute(**kwargs, backend_config=cached.backend_config)

        # 4. Default backend
        default_entry = cls._backends.get(default_backend_enum)
        if default_entry is not None and default_entry.impl.can_handle(**kwargs):
            return default_entry.impl.execute(**kwargs, backend_config=None)

        # 5. Fallback: try all backends
        for fallback_backend_enum, fallback_backend_entry in cls._backends.items():
            if fallback_backend_entry.impl.can_handle(**kwargs):
                logger.warning(
                    f"For inputs: {_format_kwargs(kwargs)}, the default backend is not compatible, fallback backend {fallback_backend_enum.name} is selected. The fallback backend may hurt performance!",
                    once=True,
                )
                return fallback_backend_entry.impl.execute(**kwargs, backend_config=None)

        raise ValueError(
            f"No compatible backend found for {cls.__name__} with inputs: {_format_kwargs(kwargs)}"
        )
