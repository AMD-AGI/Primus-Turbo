###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Offline autotune driver for the dense GEMM family.

The GEMM family shares one shape axis ``(m, n, k)``. This driver sweeps every
precision over a shape list, each enumerating its own grid on top of it, and dumps
one JSON per dispatcher under ``tuning/configs/<framework>/<arch>/``.

    python -m primus_turbo.tuning.offline_tune_gemm [--shapes s.json] [--gpus N]

``shapes.json``: ``{"mnk": [[m, n, k], ...]}``. Omit for one smoke-test shape.
``--gpus N`` shards the shapes over N single-GPU workers and merges their output.

To add a precision: write its ``_jobs_*`` builder and add a row to ``_PRECISIONS``.
"""

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

import torch

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.core.backend import GlobalBackendManager, TuneCache
from primus_turbo.pytorch.core.low_precision import (
    Float4QuantConfig,
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    check_mxfp4_support,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import GEMMFP4KernelDispatcher
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8KernelDispatcher
from primus_turbo.pytorch.kernels.gemm.gemm_impl import GEMMKernelDispatcher
from primus_turbo.pytorch.ops import gemm, gemm_fp4, gemm_fp8

# --- What gets swept ---------------------------------------------------------

_DEFAULT_MNK = [(4096, 4096, 4096)]

_DTYPES = (torch.bfloat16, torch.float16)
_FP8_FORMATS = (Format.E4M3, Format.E5M2, Format.HYBRID)
_FP8_GRANULARITIES = (
    ScalingGranularity.TENSORWISE,
    ScalingGranularity.ROWWISE,
    ScalingGranularity.BLOCKWISE,
    ScalingGranularity.MX_BLOCKWISE,
)

# Sub-byte dtypes whose stored extent is smaller than the logical one.
_VALUES_PER_ELEMENT = {torch.float4_e2m1fn_x2: 2}


# --- One job list per precision ----------------------------------------------


def _nt_fwd_bwd(op, dtype, device, **op_kwargs):
    """Build a job running one NT gemm fwd + bwd, so the two grad gemms get tuned too."""

    def run_one(m, n, k):
        # NT: a is [m, k], b is [n, k]; both need grad.
        a = torch.randn(m, k, dtype=dtype, device=device, requires_grad=True)
        b = torch.randn(n, k, dtype=dtype, device=device, requires_grad=True)
        out = op(a, b, False, True, dtype, **op_kwargs)
        out.backward(torch.randn_like(out))

    return run_one


def _fp8_config(fmt, gran) -> Float8QuantConfig:
    if gran == ScalingGranularity.BLOCKWISE:
        return Float8QuantConfig(granularity=gran, format=fmt, block_size=128)
    if gran == ScalingGranularity.MX_BLOCKWISE:
        return Float8QuantConfig(granularity=gran, format=fmt, block_size=32, scale_dtype=ScaleDtype.E8M0)
    return Float8QuantConfig(granularity=gran, format=fmt)


def _jobs_gemm(device):
    """bf16/fp16 dense: one job per dtype."""
    return [(str(dtype), _nt_fwd_bwd(gemm, dtype, device)) for dtype in _DTYPES]


def _jobs_gemm_fp8(device):
    """fp8: dtype x format x granularity."""
    return [
        (
            f"{dtype} {fmt.name}/{gran.name}",
            _nt_fwd_bwd(gemm_fp8, dtype, device, config=_fp8_config(fmt, gran)),
        )
        for dtype, fmt, gran in itertools.product(_DTYPES, _FP8_FORMATS, _FP8_GRANULARITIES)
    ]


def _jobs_gemm_fp4(device):
    """fp4: dtype x preshuffle. Format and granularity are fixed by Float4QuantConfig."""
    supported, reason = check_mxfp4_support()
    if not supported:
        logger.info(f"[gemm_fp4] skipped: {reason}")
        return []
    return [
        (
            f"{dtype} preshuffle={preshuffle}",
            _nt_fwd_bwd(gemm_fp4, dtype, device, config=Float4QuantConfig(use_preshuffle=preshuffle)),
        )
        for dtype, preshuffle in itertools.product(_DTYPES, (False, True))
    ]


# Name (also the asset basename), dispatcher, job builder. `run` and `_merge_shards`
# both walk this table, so a precision is described in exactly one place.
_PRECISIONS = (
    ("gemm", GEMMKernelDispatcher, _jobs_gemm),
    ("gemm_fp8", GEMMFP8KernelDispatcher, _jobs_gemm_fp8),
    ("gemm_fp4", GEMMFP4KernelDispatcher, _jobs_gemm_fp4),
)


# --- Sweeping ----------------------------------------------------------------


def _dump(dispatcher, out_dir, name) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = dispatcher.dump_cache(os.path.join(out_dir, f"{name}.json"))
    logger.info(f"[offline_tune_gemm] {name}.json: {n} entries")
    return n


def _annotate_perf(dispatcher) -> None:
    """Add tflops / gbps to every tuned entry's ``perf``.

    The dispatcher only times the kernel; FLOP and byte counts are op-specific. Assumes
    the GEMM key prefix ``(m, n, k, a_dtype, b_dtype, out_dtype)``.
    """
    for key, entry in dispatcher._cache.items():
        time_ms = (entry.perf or {}).get("time_ms")
        if not time_ms:
            continue
        m, n, k, a_dtype, b_dtype, out_dtype = key[:6]
        secs = time_ms * 1e-3
        # k is the *stored* contraction extent, so it already gives the right byte
        # counts; fp4 packs two values per byte, so FLOPs need the logical k back.
        logical_k = k * _VALUES_PER_ELEMENT.get(a_dtype, 1)
        moved = m * k * a_dtype.itemsize + n * k * b_dtype.itemsize + m * n * out_dtype.itemsize
        entry.perf["tflops"] = round(2 * m * n * logical_k / secs / 1e12, 2)
        entry.perf["gbps"] = round(moved / secs / 1e9, 1)


def _sweep(name, dispatcher, jobs, mnk_list, out_dir) -> None:
    """Run every ``(label, run_one)`` job over every shape, then annotate and dump."""
    # Offline tuning must retain every tuned key until dump. Each fwd+bwd emits up to
    # 3 dispatch keys (1 forward gemm + 2 backward grad gemms), so size the LRU to the
    # full grid x 3 (upper bound on distinct keys) and never evict.
    dispatcher._cache = TuneCache(capacity=max(3 * len(jobs) * len(mnk_list), 1024))
    for i, (label, run_one) in enumerate(jobs, 1):
        logger.info(f"[{name}] ({i}/{len(jobs)}) {label} x {len(mnk_list)} shapes")
        for m, n, k in mnk_list:
            try:
                run_one(m, n, k)
                torch.cuda.synchronize()
            except Exception as e:  # unsupported combo on this arch -> skip, keep sweeping
                logger.warning(f"[offline_tune_gemm] skip {name} {label}: {e}", once=True)
    torch.cuda.synchronize()
    _annotate_perf(dispatcher)
    _dump(dispatcher, out_dir, name)


def run(mnk_list, out_dir, device="cuda:0") -> None:
    """Tune every precision over ``mnk_list`` on a single device."""
    GlobalBackendManager.reset()
    GlobalBackendManager.set_auto_tune(True)
    try:
        for name, dispatcher, build_jobs in _PRECISIONS:
            jobs = build_jobs(device)
            if jobs:  # empty => this precision is unsupported here
                _sweep(name, dispatcher, jobs, mnk_list, out_dir)
    finally:
        GlobalBackendManager.set_auto_tune(None)
        GlobalBackendManager.reset()


# --- Multi-GPU sharding ------------------------------------------------------


def _merge_shards(shard_dirs, out_dir) -> None:
    """Union each precision's asset across the shards and dump it to ``out_dir``."""
    for name, dispatcher, _ in _PRECISIONS:
        paths = [p for p in (os.path.join(d, f"{name}.json") for d in shard_dirs) if os.path.isfile(p)]
        if not paths:  # precision produced nothing on this arch
            continue
        total = sum(len(json.load(open(p))["entries"]) for p in paths)
        dispatcher._cache = TuneCache(capacity=max(total, 1024))
        for p in paths:
            dispatcher.load_cache(p)
        _dump(dispatcher, out_dir, name)


def run_sharded(mnk_list, out_dir, gpus: int) -> None:
    """Tune the shapes across ``gpus`` single-GPU workers, then merge their JSONs.

    Shapes are dealt round-robin, so fewer shapes than GPUs leaves some idle. Pinning
    each worker to one device keeps it on the ordinary single-GPU path.
    """
    if gpus > torch.cuda.device_count():
        raise ValueError(f"--gpus {gpus} exceeds the {torch.cuda.device_count()} visible devices")
    with tempfile.TemporaryDirectory() as tmp:
        workers = []
        for rank in range(gpus):
            shard = mnk_list[rank::gpus]
            if not shard:
                continue
            d = os.path.join(tmp, f"rank{rank}")
            os.makedirs(d)
            shapes_path = os.path.join(d, "shapes.json")
            with open(shapes_path, "w") as f:
                json.dump({"mnk": [list(s) for s in shard]}, f)
            cmd = [sys.executable, "-m", "primus_turbo.tuning.offline_tune_gemm"]
            cmd += ["--shapes", shapes_path, "--out-dir", d]
            # ROCm reads HIP_VISIBLE_DEVICES; torch also honours the CUDA spelling.
            env = {**os.environ, "HIP_VISIBLE_DEVICES": str(rank), "CUDA_VISIBLE_DEVICES": str(rank)}
            with open(os.path.join(d, "log"), "w") as log:  # closing is fine: the child dup'd it
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            logger.info(f"[offline_tune_gemm] GPU {rank}: {len(shard)} shapes, pid {proc.pid}")
            workers.append((rank, d, proc))

        for rank, d, proc in workers:
            rc = proc.wait()
            if rc != 0:
                tail = "".join(open(os.path.join(d, "log")).readlines()[-20:])
                raise RuntimeError(f"[offline_tune_gemm] GPU {rank} worker failed (exit {rc}):\n{tail}")
            logger.info(f"[offline_tune_gemm] GPU {rank}: done")
        _merge_shards([d for _, d, _ in workers], out_dir)


# --- CLI ---------------------------------------------------------------------


def _arch_tag() -> str:
    """e.g. 'gfx950' / 'gfx942'."""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def main():
    p = argparse.ArgumentParser(description="Offline autotune for the GEMM family.")
    p.add_argument("--shapes", default=None, help="JSON: {'mnk': [[m,n,k], ...]}")
    p.add_argument("--gpus", type=int, default=1, help="Shard the shapes over N GPUs (default: 1).")
    p.add_argument("--out-dir", default=None, help="Output dir; defaults to the packaged config path.")
    args = p.parse_args()

    logger.set_level(logging.INFO)  # progress logs are INFO; default level would hide them

    mnk = [tuple(x) for x in json.load(open(args.shapes))["mnk"]] if args.shapes else _DEFAULT_MNK
    # Default is the canonical packaged path the runtime auto-loads from; --out-dir is
    # for the per-worker shards of a --gpus run.
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "pytorch", _arch_tag()
    )
    t0 = time.perf_counter()
    if args.gpus > 1:
        run_sharded(mnk, out_dir, args.gpus)
    else:
        run(mnk, out_dir)
    logger.info(f"[offline_tune_gemm] done -> {out_dir} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
