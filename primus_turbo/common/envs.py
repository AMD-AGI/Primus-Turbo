###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vLLM (https://github.com/vllm-project/vllm), file vllm/envs.py.
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""The environment variables Primus-Turbo reads, declared in one place.

Read a value through the module attribute of the same name::

    import primus_turbo.common.envs as envs

    if envs.PRIMUS_TURBO_AUTO_TUNE:
        ...

Each variable is parsed by its getter in ``environment_variables`` the first time it is read, and
that value is returned for the rest of the process: setting the variable afterwards has no effect.
Code that changes one of these variables at run time writes it with ``set_env()``, which also
drops the cached value. Tests that set variables call ``reset_cache()`` (the repo's test conftest
does so around every test), or ``disable_envs_cache()`` to read on every access.

Build-time variables (``setup.py``, ``tools/``) are not declared here, nor are variables owned by
other projects (FlyDSL, Triton, NVSHMEM, the launcher's ``RANK`` / ``WORLD_SIZE``): code reads or
writes those through ``os.environ`` directly. The C++ extension reads its variables with
``getenv``; they are declared below so Python can discover and inspect them, but C++ reads and
caches them itself, so ``set_env()`` reaches C++ only before C++ first reads the variable.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    PRIMUS_TURBO_LOG_LEVEL: str = "WARNING"
    PRIMUS_TURBO_GEMM_BACKEND: str | None = None
    PRIMUS_TURBO_GROUPED_GEMM_BACKEND: str | None = None
    PRIMUS_TURBO_ATTN_BACKEND: str | None = None
    PRIMUS_TURBO_SPARSE_ATTN_BACKEND: str | None = None
    PRIMUS_TURBO_MLA_BWD_IMPL: str = "flydsl"
    PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND: str | None = None
    PRIMUS_TURBO_AUTO_TUNE: bool = False
    PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32: bool = True
    PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT: bool = False
    PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM: bool = False
    PRIMUS_TURBO_JAX_DEEPEP_MODE: str | None = None
    PT_MEGA_FP8_COMBINE_AUTOTUNE: bool = True
    PT_MEGA_FP8_COMBINE_AUTOTUNE_VERBOSE: bool = True
    PT_MEGA_FP8_COMBINE_CU_CANDIDATES: tuple[int, ...] = (16, 32, 48, 64, 80, 96, 128)
    PT_MEGA_FP8_COMBINE_TUNE_REPS: int = 3
    PT_MEGA_FP8_L1_COMBINE_CU: int | None = None
    PT_MEGA_FP8_L2_COMBINE_CU: int | None = None
    PRIMUS_TURBO_DEEPEP_CPU_TIMEOUT: int = 300
    PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE: bool = False
    PRIMUS_TURBO_ODC_GDA_BLOCK: int = 256
    PRIMUS_TURBO_ODC_GDA_PIPE: int = 1
    PRIMUS_TURBO_ODC_GDA_NUM_QP: int = 1


def backend_spec(name: str) -> Callable[[], str | None]:
    """A getter for a backend selection string; unset, empty and whitespace-only all read as None.

    Parsing the value needs ``BackendType``, so ``primus_turbo.pytorch.core.backend`` does it.
    """

    def _get() -> str | None:
        value = os.getenv(name)
        return value if value is not None and value.strip() else None

    return _get


def _positive_int(name: str, raw: str, what: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise ValueError(f"{name}={raw!r}: {what}")
    return value


def env_positive_int(
    name: str, default: int | None, what: str = "must be a positive integer"
) -> Callable[[], int | None]:
    """A getter for a positive integer; unset or empty reads as ``default``, anything else that is
    not a positive integer raises ``ValueError`` naming the variable."""

    def _get() -> int | None:
        raw = os.getenv(name)
        return _positive_int(name, raw, what) if raw else default

    return _get


def env_positive_int_list(
    name: str, default: tuple[int, ...], what: str, empty_what: str
) -> Callable[[], tuple[int, ...]]:
    """A getter for a comma- or space-separated list of positive integers; unset or empty reads as
    ``default``, a value with no integers in it raises ``ValueError``."""

    def _get() -> tuple[int, ...]:
        raw = os.getenv(name)
        if not raw:
            return default
        values = tuple(_positive_int(name, v, what) for v in raw.replace(",", " ").split())
        if not values:
            raise ValueError(f"{name}={raw!r}: {empty_what}")
        return values

    return _get


def env_clamped_int(name: str, default: int, lo: int, hi: int) -> Callable[[], int]:
    """A getter mirroring the C++ ``atoi`` + clamp idiom: a non-integer reads as 0 before clamping."""

    def _get() -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            value = 0
        return min(max(value, lo), hi)

    return _get


def _mla_bwd_impl() -> str:
    """PRIMUS_TURBO_MLA_BWD_IMPL: "flydsl" (also unset or blank) or "hk", any case."""
    raw = os.getenv("PRIMUS_TURBO_MLA_BWD_IMPL", "")
    value = raw.strip().lower() or "flydsl"
    if value not in ("flydsl", "hk"):
        raise ValueError(f"PRIMUS_TURBO_MLA_BWD_IMPL={raw!r}: expected flydsl or hk")
    return value


def _deepep_cpu_timeout() -> int:
    raw = os.getenv("PRIMUS_TURBO_DEEPEP_CPU_TIMEOUT")
    if not raw:
        return 300
    try:
        return int(raw)
    except ValueError:
        return 300


environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Logging ==================
    # Level of the ``primus_turbo`` logger: DEBUG, INFO, WARNING, ERROR or CRITICAL (any case).
    # An unknown name falls back to WARNING. Read once, when the logger singleton is created.
    "PRIMUS_TURBO_LOG_LEVEL": lambda: os.getenv("PRIMUS_TURBO_LOG_LEVEL", "WARNING").upper(),
    # ================== Backend selection ==================
    # GEMM backend, e.g. HIPBLASLT, AITER. Either one backend for every precision or a
    # per-precision list: "FP4:HIPBLASLT,FP8:AITER", with "other:<backend>" for the rest. Any slot
    # also accepts "autotune" to auto-tune that precision only, e.g. "FP8:autotune,other:HIPBLASLT".
    # Unset (or blank) leaves the choice to the dispatcher. Read at the first dispatch that has no
    # code-level setting (``GlobalBackendManager.set_gemm_backend``); the five backend variables
    # and PRIMUS_TURBO_AUTO_TUNE are read again after ``GlobalBackendManager.reset()``.
    "PRIMUS_TURBO_GEMM_BACKEND": backend_spec("PRIMUS_TURBO_GEMM_BACKEND"),
    # Grouped GEMM backend. Same format as PRIMUS_TURBO_GEMM_BACKEND.
    "PRIMUS_TURBO_GROUPED_GEMM_BACKEND": backend_spec("PRIMUS_TURBO_GROUPED_GEMM_BACKEND"),
    # Flash-attention backend (dense and varlen share it). Same per-precision format as
    # PRIMUS_TURBO_GEMM_BACKEND. Unset auto-selects: FLYDSL when eligible, else the op's fallback.
    # GLUON is explicit and forward-only, never auto-selected; it is dense-only, so selecting it
    # makes varlen dispatch fail before launch. Flash and sparse attention keep separate keys
    # because their backend sets differ (only flash has AITER, only sparse has TRITON).
    "PRIMUS_TURBO_ATTN_BACKEND": backend_spec("PRIMUS_TURBO_ATTN_BACKEND"),
    # Sparse-attention (DeepSeek-V4 sparse-MLA) backend, e.g. FLYDSL, TRITON, or "autotune".
    "PRIMUS_TURBO_SPARSE_ATTN_BACKEND": backend_spec("PRIMUS_TURBO_SPARSE_ATTN_BACKEND"),
    # Kernels of the gfx1250 DeepSeek-V3 MLA attention backward (qk 192 / v 128) under the FLYDSL
    # backend: "flydsl" (unset or blank) or "hk", the HipKittens kernels, which hand a call they
    # cannot take to FlyDSL (e.g. seqlen_q % 384 not 0 or 256, see their README). Read at the first
    # MLA backward (attention_flydsl_gfx1250_mla_impl.mla_bwd_impl); another value raises ValueError.
    "PRIMUS_TURBO_MLA_BWD_IMPL": _mla_bwd_impl,
    # MoE dispatch/combine EP backend: TURBO (the default when unset), DEEP_EP, or a custom EP name
    # such as UCCL_EP. "autotune" is rejected with an AssertionError.
    "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND": backend_spec("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND"),
    # ================== Auto-tune ==================
    # "1" turns auto-tuning across registered kernel backends on for every op, and the FlyDSL
    # kernel-level tuners with it. An explicit per-op backend still wins over it, and so does
    # ``GlobalBackendManager.set_auto_tune``. Read at the first dispatch or FlyDSL launch.
    "PRIMUS_TURBO_AUTO_TUNE": lambda: os.getenv("PRIMUS_TURBO_AUTO_TUNE", "0") == "1",
    # ================== Kernel behavior ==================
    # Whether the attention V3 (CK) backward accumulates with FP32 atomics. Only "0" disables it;
    # unset, "1" and any other value keep it on. Read at the first forward that does not pass
    # ``is_v3_atomic_fp32`` explicitly.
    "PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32": lambda: os.getenv("PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32", "1") != "0",
    # "1" lets the FP4 variable-K grouped GEMM (wgrad) overwrite ``out`` with beta=0 instead of
    # accumulating into it, when the caller allows it and no CUDA graph is being captured.
    "PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT": lambda: (
        os.getenv("PRIMUS_TURBO_WGRAD_ACCUM_OVERWRITE_OUT", "0") == "1"
    ),
    # ================== Expert parallel ==================
    # A non-zero integer runs the EP dispatch/combine kernels on the caller's current stream
    # instead of the Buffer's comm stream. Parsed as C++ does (``is_ep_force_current_stream`` in
    # deep_ep/configs.h, which caches it once per process): unset is off, and an empty
    # or non-integer value is an error (ValueError here, std::stoi throws in C++). The FlyDSL
    # intranode backend (``FlyDSLIntranodeRuntime``) reads it when it is constructed.
    # ``DeepEPTokenDispatcher`` writes "1" with ``set_env()`` when a dispatcher asks for
    # ``deepep_use_comm_stream=False``.
    "PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM": lambda: (
        int(os.getenv("PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM", "0")) != 0
    ),
    # JAX DeepEP launch mode: "inproc" (the default when unset) or "per_process"; case and "-"/"_"
    # are normalized by ``LaunchMode.from_str``. ``auto_detect_mode`` and ``setup(mode=...)`` write
    # it with ``set_env()``, ``reset_runtime()`` clears it, and the mode locks at the first DeepEP
    # call.
    "PRIMUS_TURBO_JAX_DEEPEP_MODE": lambda: os.getenv("PRIMUS_TURBO_JAX_DEEPEP_MODE"),
    # ================== Mega MoE (FlyDSL FP8 combine) ==================
    # Online tuning of the FP8 combine's CU split. On unless set to "0", "false" or "False".
    # Read once, when ``primus_turbo.flydsl.mega.fp8.combine_autotune`` is imported.
    "PT_MEGA_FP8_COMBINE_AUTOTUNE": lambda: (
        os.getenv("PT_MEGA_FP8_COMBINE_AUTOTUNE", "1") not in ("0", "false", "False")
    ),
    # One log line per shape when the tuned CU split locks in. Same values and timing as above.
    "PT_MEGA_FP8_COMBINE_AUTOTUNE_VERBOSE": lambda: (
        os.getenv("PT_MEGA_FP8_COMBINE_AUTOTUNE_VERBOSE", "1") not in ("0", "false", "False")
    ),
    # Candidate CU splits the combine tuner tries, comma- or space-separated positive integers.
    # Read once, at import of combine_autotune. A bad value raises ValueError there. 0 is not
    # "off" for a CU split: the combine reads it as "no PUSH role", and with the reduce on, as in
    # every production call, the reduce would then wait forever for arrivals nobody sends.
    "PT_MEGA_FP8_COMBINE_CU_CANDIDATES": env_positive_int_list(
        "PT_MEGA_FP8_COMBINE_CU_CANDIDATES",
        (16, 32, 48, 64, 80, 96, 128),
        "the combine CU split must be a positive block count",
        "need at least one candidate CU split",
    ),
    # Timed samples per candidate CU split. Read once, at import of combine_autotune. Must be
    # positive: an empty schedule would lock in at the first call, on a candidate never measured.
    "PT_MEGA_FP8_COMBINE_TUNE_REPS": env_positive_int("PT_MEGA_FP8_COMBINE_TUNE_REPS", 3),
    # Pin the stage-1 (L1) / stage-2 (L2) combine CU split and skip tuning for it. A positive
    # block count; unset leaves it to the tuner. Read once, at import of the fused_mega_moe
    # stage-1 / stage-2 FP8 impl modules.
    "PT_MEGA_FP8_L1_COMBINE_CU": env_positive_int(
        "PT_MEGA_FP8_L1_COMBINE_CU", None, "the combine CU split must be a positive block count"
    ),
    "PT_MEGA_FP8_L2_COMBINE_CU": env_positive_int(
        "PT_MEGA_FP8_L2_COMBINE_CU", None, "the combine CU split must be a positive block count"
    ),
    # ================== Read by the C++ extension ==================
    # These getters mirror the C++ parsing so Python can report the effective value; C++ reads
    # the variable itself and caches it once per process.
    #
    # DeepEP host-side timeout in seconds (deep_ep/configs.h). Unset, empty or a non-integer
    # reads as 300.
    "PRIMUS_TURBO_DEEPEP_CPU_TIMEOUT": _deepep_cpu_timeout,
    # A non-zero integer makes the intranode dispatch/combine channels publish their tails with
    # a full system-scope release/acquire instead of the cheap per-warp fence (gfx1250 always does).
    # C++ caches it separately in the dispatch and combine launchers, each at its first call.
    "PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE": lambda: (
        int(os.getenv("PRIMUS_TURBO_DEEPEP_DISABLE_CHEAP_FENCE") or "0") != 0
    ),
    # ODC (on-demand-comm) rocSHMEM GDA device-kernel knobs, consumed by
    # csrc/kernels/odc_rocshmem/odc_rocshmem_gda.cu for the multi-node reduce-scatter /
    # all-gather device kernels.
    #
    # Threads per block of the ODC GDA device kernels, clamped to [32, 1024].
    "PRIMUS_TURBO_ODC_GDA_BLOCK": env_clamped_int("PRIMUS_TURBO_ODC_GDA_BLOCK", 256, 32, 1024),
    # Reduce-scatter peer-pipeline batch depth (1 = serial), clamped to [1, 64].
    "PRIMUS_TURBO_ODC_GDA_PIPE": env_clamped_int("PRIMUS_TURBO_ODC_GDA_PIPE", 1, 1, 64),
    # rocSHMEM QPs/contexts for multi-QP NIC concurrency, clamped to [1, 256]; above 1 selects
    # the multi-QP path.
    "PRIMUS_TURBO_ODC_GDA_NUM_QP": env_clamped_int("PRIMUS_TURBO_ODC_GDA_NUM_QP", 1, 1, 256),
}


_cache: dict[str, Any] = {}
_cache_enabled = True


def __getattr__(name: str):
    """The value of ``name``, parsed at its first read and cached from then on."""
    if name not in environment_variables:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return _cache[name]
    except KeyError:
        pass
    value = environment_variables[name]()
    if _cache_enabled:
        _cache[name] = value
    return value


def __dir__():
    return list(environment_variables.keys())


def _check_declared(name: str) -> None:
    if name not in environment_variables:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_set(name: str) -> bool:
    """Whether ``name`` is present in the environment right now (never cached)."""
    _check_declared(name)
    return name in os.environ


def set_env(name: str, value: str | None) -> None:
    """Write ``name`` into ``os.environ`` (``None`` removes it) and drop its cached value.

    The next read of ``envs.<name>`` parses the new value. C++ sees the write only if it has not
    read and cached the variable yet.
    """
    _check_declared(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    _cache.pop(name, None)


def reset_cache(*names: str) -> None:
    """Drop the cached values of ``names`` (all variables if none are given).

    The next read of each one parses the environment again.
    """
    if not names:
        _cache.clear()
        return
    for name in names:
        _check_declared(name)
        _cache.pop(name, None)


def disable_envs_cache() -> None:
    """Read every variable from the environment on each access, until ``enable_envs_cache()``."""
    global _cache_enabled
    _cache_enabled = False
    _cache.clear()


def enable_envs_cache() -> None:
    """Go back to the default: cache each variable at its first read."""
    global _cache_enabled
    _cache_enabled = True
