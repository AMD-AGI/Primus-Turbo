###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Grouped GEMM on hipBLASLt for JAX (BF16/FP16), with autodiff.

Environment variables:
    PRIMUS_TURBO_HB_GG_TUNE             "subprocess" (default), "inproc", or "0" (hipBLASLt heuristic, much slower).
    PRIMUS_TURBO_HB_GG_TUNE_CANDIDATES  number of algorithms timed per problem signature (default 256).
    PRIMUS_TURBO_HB_GG_TUNE_MAX_ROWS    largest row count used for tuning (default 131072).
    PRIMUS_TURBO_HB_GG_TUNE_CACHE       tuning cache file (default ~/.cache/primus_turbo/hipblaslt_grouped_gemm_tune.json).
    PRIMUS_TURBO_HB_GG_HOST_ROWS        "auto" (default) or "max": rows per group of the host-side problem that sizes
                                        the launch grid; "max" is also correct but launches G times more work-groups.
    PRIMUS_TURBO_HB_GG_DEBUG            print each tuning result.
    PRIMUS_TURBO_HB_GG_VERBOSE          print the kernel chosen for each plan.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import warnings
from functools import lru_cache, partial

import jax
import jax.numpy as jnp

import primus_turbo.common.envs as envs
from primus_turbo.jax._C import (
    register_hipblaslt_grouped_gemm_algo,
    tune_hipblaslt_grouped_gemm,
)
from primus_turbo.jax.primitive.grouped_gemm.hipblaslt_grouped_gemm import (
    hipblaslt_grouped_gemm_p,
)

__all__ = ["grouped_gemm_hipblaslt"]

_tuned = {}


def _cache_path():
    return envs.PRIMUS_TURBO_HB_GG_TUNE_CACHE


@lru_cache(maxsize=1)
def _fingerprint():
    """Tuned algorithm indices are only valid for the same GPU type and hipBLASLt library."""
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    libs = sorted(glob.glob(os.path.join(rocm, "lib", "libhipblaslt.so*")))
    lib = f"{os.path.getsize(libs[-1])}-{int(os.path.getmtime(libs[-1]))}" if libs else "unknown"
    try:
        kind = jax.devices()[0].device_kind
    except Exception:  # pylint: disable=broad-except
        kind = "unknown"
    return f"{kind}|hipblaslt:{lib}"


def _load_cache():
    try:
        with open(_cache_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(key_str, value):
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cache = _load_cache()
        cache[key_str] = value
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        warnings.warn(f"could not write the hipBLASLt grouped GEMM tuning cache {path}: {e}")


_TUNE_SCRIPT = (
    "import sys\n"
    "from primus_turbo.jax._C import tune_hipblaslt_grouped_gemm as tune\n"
    "a = [int(x) for x in sys.argv[1:]]\n"
    "r = tune(*a[:6], bool(a[6]), bool(a[7]), a[8], a[9])\n"
    "print('PRIMUS_TURBO_TUNE_RESULT', r[0], r[1], r[2], flush=True)\n"
)


def _tune_in_subprocess(key, max_candidates):
    """Tune in a child process on the same device (index 0 of the visible devices, like the parent's
    current device). Some candidate kernels make wild reads that only fault under the parent's memory
    layout (XLA preallocated most of the device), so a crash here must not take the parent down."""
    args = [str(int(x)) for x in key] + [str(max_candidates)]
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", _TUNE_SCRIPT, *args], capture_output=True, text=True, timeout=900
        )
        if envs.PRIMUS_TURBO_HB_GG_DEBUG:
            print("[primus_turbo] tune subprocess stderr:", proc.stderr[-1500:], flush=True)
        for line in proc.stdout.splitlines():
            if line.startswith("PRIMUS_TURBO_TUNE_RESULT"):
                _, idx, ms, name = line.split(" ", 3)
                return int(idx), float(ms), name
    raise RuntimeError(f"tuning subprocess failed (rc={proc.returncode}): {proc.stderr[-500:]}")


def _tune(lhs, rhs, group_lens, transA, transB):
    """Pick the hipBLASLt grouped GEMM algorithm for this problem by timing candidates.

    Runs once per problem signature at trace time (never inside a running multi-device program, since
    timing synchronizes the device). PRIMUS_TURBO_HB_GG_TUNE: "subprocess" (default), "inproc", or "0"
    to use hipBLASLt's heuristic (much slower kernels for grouped problems). Results are kept in an on-disk
    cache (PRIMUS_TURBO_HB_GG_TUNE_CACHE, default ~/.cache/primus_turbo/hipblaslt_grouped_gemm_tune.json).
    """
    mode = envs.PRIMUS_TURBO_HB_GG_TUNE
    if mode == "0":
        return
    m, a_cols = lhs.shape
    if transA:
        g, (b_d1, b_d2), c_rows = group_lens.shape[0], rhs.shape, rhs.shape[1]
    else:
        g, b_d1, b_d2 = rhs.shape
        c_rows = b_d1 if transB else b_d2
    dtype = 2 if lhs.dtype == jnp.float8_e4m3fn else (0 if lhs.dtype == jnp.bfloat16 else 1)
    key = (g, m, a_cols, b_d1, b_d2, c_rows, transA, transB, dtype)
    if key in _tuned:
        return
    max_candidates = envs.PRIMUS_TURBO_HB_GG_TUNE_CANDIDATES
    # The tuner times at most PRIMUS_TURBO_HB_GG_TUNE_MAX_ROWS rows, so larger M share one cache entry.
    tune_m = max(g, min(m, envs.PRIMUS_TURBO_HB_GG_TUNE_MAX_ROWS) // g * g)
    tune_key = (g, tune_m, a_cols, tune_m if transA else b_d1, b_d2, c_rows, transA, transB, dtype)
    key_str = f"{_fingerprint()}|{max_candidates}|{','.join(str(int(x)) for x in tune_key)}"
    cached = _load_cache().get(key_str)
    if cached is not None:
        _tuned[key] = tuple(cached)
        register_hipblaslt_grouped_gemm_algo(g, a_cols, b_d1, b_d2, c_rows, transA, transB, dtype, cached[0])
        return
    try:
        if mode == "inproc":
            _tuned[key] = tune_hipblaslt_grouped_gemm(*key, max_candidates)
        else:
            _tuned[key] = _tune_in_subprocess(key, max_candidates)
            if _tuned[key][0] >= 0:
                register_hipblaslt_grouped_gemm_algo(
                    g, a_cols, b_d1, b_d2, c_rows, transA, transB, dtype, _tuned[key][0]
                )
        if _tuned[key][0] >= 0:
            _save_cache(key_str, list(_tuned[key]))
        if envs.PRIMUS_TURBO_HB_GG_DEBUG:
            print(f"[primus_turbo] tuned {key}: {_tuned[key]}", flush=True)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        _tuned[key] = None
        warnings.warn(f"hipBLASLt grouped GEMM tuning failed for {key}, using the heuristic: {e}")


def _bind(lhs, rhs, group_lens, transA, transB):
    _tune(lhs, rhs, group_lens, transA, transB)
    out, _ = hipblaslt_grouped_gemm_p.bind(lhs, rhs, group_lens, transA=transA, transB=transB)
    return out


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def grouped_gemm_hipblaslt(a, b, group_lens, transB=False):
    """Grouped GEMM on hipBLASLt (one grouped-GEMM launch, device-side group sizes), with autodiff.

    Args:
        a: [M, K], rows sorted by group.
        b: [G, K, N] ([G, N, K] if transB).
        group_lens: [G] int32 or int64; sum may be less than M.
        transB: whether b is transposed.

    Returns:
        [M, N]. Rows past sum(group_lens) are zero.
    """
    return _bind(a, b, group_lens, False, transB)


def _fwd(a, b, group_lens, transB):
    return _bind(a, b, group_lens, False, transB), (a, b, group_lens)


def _bwd(transB, ctx, grad):
    a, b, group_lens = ctx
    grad_a = _bind(grad, b, group_lens, False, not transB)
    lhs, rhs = (grad, a) if transB else (a, grad)
    grad_b = _bind(lhs, rhs, group_lens, True, False)
    return grad_a, grad_b, None


grouped_gemm_hipblaslt.defvjp(_fwd, _bwd)
