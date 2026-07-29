###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared machinery behind the ``offline_tune_*`` drivers.

A driver describes its op family with a :class:`Family` and calls :func:`main`.
Everything else — sweeping, perf annotation, dumping, multi-GPU sharding and the
CLI — lives here, so every family behaves the same way.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.core.backend import GlobalBackendManager, TuneCache
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)

# Sub-byte dtypes whose stored extent is smaller than the logical one.
_VALUES_PER_ELEMENT = {torch.float4_e2m1fn_x2: 2}

# The sweep grid the GEMM families share.
DTYPES = (torch.bfloat16, torch.float16)
FP8_FORMATS = (Format.E4M3, Format.E5M2, Format.HYBRID)
FP8_GRANULARITIES = (
    ScalingGranularity.TENSORWISE,
    ScalingGranularity.ROWWISE,
    ScalingGranularity.BLOCKWISE,
    ScalingGranularity.MX_BLOCKWISE,
)


def fp8_config(fmt, gran) -> Float8QuantConfig:
    """The quant config each fp8 granularity expects."""
    if gran == ScalingGranularity.BLOCKWISE:
        return Float8QuantConfig(granularity=gran, format=fmt, block_size=128)
    if gran == ScalingGranularity.MX_BLOCKWISE:
        return Float8QuantConfig(granularity=gran, format=fmt, block_size=32, scale_dtype=ScaleDtype.E8M0)
    return Float8QuantConfig(granularity=gran, format=fmt)


@dataclass(frozen=True)
class Family:
    """One op family's offline-tune driver.

    Attributes:
        module: dotted path of the driver, re-executed by a sharded run.
        shapes_key: key holding the shape list inside the shapes JSON.
        default_shapes: used when ``--shapes`` is omitted.
        precisions: ``(build_jobs, assets)`` rows, one per precision.
            ``build_jobs(device)`` returns ``(label, run_one(*shape))`` pairs, and
            ``assets`` lists the ``(name, dispatcher, perf_counts)`` this sweep fills —
            more than one when a single fwd+bwd feeds several dispatchers (e.g. grouped
            GEMM's forward and variable-K wgrad). ``name`` is the asset basename and
            ``perf_counts(key)`` returns ``(flops, bytes_moved)`` for that dispatcher's
            key layout.
    """

    module: str
    shapes_key: str
    default_shapes: Sequence[Sequence[int]]
    precisions: Sequence[tuple]

    @property
    def assets(self):
        """Every ``(name, dispatcher, perf_counts)`` this family can produce."""
        return [asset for _, assets in self.precisions for asset in assets]

    @property
    def tag(self) -> str:
        """Short name for log lines, e.g. 'offline_tune_gemm'."""
        return self.module.rsplit(".", 1)[-1]


def _dump(family: Family, dispatcher, out_dir: str, name: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = dispatcher.dump_cache(os.path.join(out_dir, f"{name}.json"))
    logger.info(f"[{family.tag}] {name}.json: {n} entries")
    return n


def logical_k(k: int, a_dtype) -> int:
    """The contraction extent in values, not stored elements (fp4 packs two per byte)."""
    return k * _VALUES_PER_ELEMENT.get(a_dtype, 1)


def _annotate_perf(dispatcher, perf_counts: Callable[[tuple], tuple | None]) -> None:
    """Add tflops / gbps to every tuned entry's ``perf``.

    The dispatcher only times the kernel; FLOP and byte counts depend on the op's operand
    layout, so each asset supplies its own ``perf_counts``. A key is a fingerprint, not
    always a faithful shape, so ``perf_counts`` may return None — then only ``time_ms``
    is reported rather than a number derived from extents that do not mean what they say.
    """
    for key, entry in dispatcher._cache.items():
        time_ms = (entry.perf or {}).get("time_ms")
        if not time_ms:
            continue
        counts = perf_counts(key)
        if counts is None:
            continue
        flops, moved = counts
        secs = time_ms * 1e-3
        entry.perf["tflops"] = round(flops / secs / 1e12, 2)
        entry.perf["gbps"] = round(moved / secs / 1e9, 1)


def _sweep(family: Family, assets, jobs, shapes, out_dir: str) -> None:
    """Run every ``(label, run_one)`` job over every shape, then annotate and dump."""
    precision = assets[0][0]  # by convention the row's first asset names the precision
    # Offline tuning must retain every tuned key until dump. One fwd+bwd emits up to 3
    # dispatch keys (1 forward gemm + 2 backward grad gemms), so size the LRU to the full
    # grid x 3 (upper bound on distinct keys) and never evict.
    capacity = max(3 * len(jobs) * len(shapes), 1024)
    for _, dispatcher, _ in assets:
        dispatcher._cache = TuneCache(capacity=capacity)
    for i, (label, run_one) in enumerate(jobs, 1):
        logger.info(f"[{precision}] ({i}/{len(jobs)}) {label} x {len(shapes)} shapes")
        for shape in shapes:
            try:
                run_one(*shape)
                torch.cuda.synchronize()
            except Exception as e:  # unsupported combo on this arch -> skip, keep sweeping
                logger.warning(f"[{family.tag}] skip {precision} {label} {tuple(shape)}: {e}", once=True)
    torch.cuda.synchronize()
    for name, dispatcher, perf_counts in assets:
        _annotate_perf(dispatcher, perf_counts)
        _dump(family, dispatcher, out_dir, name)


def run(family: Family, shapes, out_dir: str, device: str = "cuda:0") -> None:
    """Tune every precision of ``family`` over ``shapes`` on a single device."""
    GlobalBackendManager.reset()
    GlobalBackendManager.set_auto_tune(True)
    try:
        for build_jobs, assets in family.precisions:
            jobs = build_jobs(device)
            if jobs:  # empty => this precision is unsupported here
                _sweep(family, assets, jobs, shapes, out_dir)
    finally:
        GlobalBackendManager.set_auto_tune(None)
        GlobalBackendManager.reset()


def _merge_shards(family: Family, shard_dirs, out_dir: str) -> None:
    """Union each asset across the shards and dump it to ``out_dir``."""
    for name, dispatcher, _ in family.assets:
        paths = [p for p in (os.path.join(d, f"{name}.json") for d in shard_dirs) if os.path.isfile(p)]
        if not paths:  # precision produced nothing on this arch
            continue
        total = sum(len(json.load(open(p))["entries"]) for p in paths)
        dispatcher._cache = TuneCache(capacity=max(total, 1024))
        for p in paths:
            dispatcher.load_cache(p)
        _dump(family, dispatcher, out_dir, name)


def run_sharded(family: Family, shapes, out_dir: str, gpus: int) -> None:
    """Tune the shapes across ``gpus`` single-GPU workers, then merge their JSONs.

    Shapes are dealt round-robin, so fewer shapes than GPUs leaves some idle. Pinning each
    worker to one device keeps it on the ordinary single-GPU path.
    """
    if gpus > torch.cuda.device_count():
        raise ValueError(f"--gpus {gpus} exceeds the {torch.cuda.device_count()} visible devices")
    with tempfile.TemporaryDirectory() as tmp:
        workers = []
        for rank in range(gpus):
            shard = shapes[rank::gpus]
            if not shard:
                continue
            d = os.path.join(tmp, f"rank{rank}")
            os.makedirs(d)
            shapes_path = os.path.join(d, "shapes.json")
            with open(shapes_path, "w") as f:
                json.dump({family.shapes_key: [list(s) for s in shard]}, f)
            cmd = [sys.executable, "-m", family.module, "--shapes", shapes_path, "--out-dir", d]
            # ROCm reads HIP_VISIBLE_DEVICES; torch also honours the CUDA spelling.
            env = {**os.environ, "HIP_VISIBLE_DEVICES": str(rank), "CUDA_VISIBLE_DEVICES": str(rank)}
            with open(os.path.join(d, "log"), "w") as log:  # closing is fine: the child dup'd it
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            logger.info(f"[{family.tag}] GPU {rank}: {len(shard)} shapes, pid {proc.pid}")
            workers.append((rank, d, proc))

        for rank, d, proc in workers:
            rc = proc.wait()
            if rc != 0:
                tail = "".join(open(os.path.join(d, "log")).readlines()[-20:])
                raise RuntimeError(f"[{family.tag}] GPU {rank} worker failed (exit {rc}):\n{tail}")
            logger.info(f"[{family.tag}] GPU {rank}: done")
        _merge_shards(family, [d for _, d, _ in workers], out_dir)


def _arch_tag() -> str:
    """e.g. 'gfx950' / 'gfx942'."""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def main(family: Family) -> None:
    """Standard CLI for an ``offline_tune_*`` driver."""
    p = argparse.ArgumentParser(description=f"Offline autotune for the {family.tag} family.")
    p.add_argument("--shapes", default=None, help=f"JSON: {{'{family.shapes_key}': [[...], ...]}}")
    p.add_argument("--gpus", type=int, default=1, help="Shard the shapes over N GPUs (default: 1).")
    p.add_argument("--out-dir", default=None, help="Output dir; defaults to the packaged config path.")
    args = p.parse_args()

    logger.set_level(logging.INFO)  # progress logs are INFO; default level would hide them

    shapes = list(family.default_shapes)
    if args.shapes:
        spec = json.load(open(args.shapes))
        if family.shapes_key not in spec:  # e.g. "mnk" handed to the grouped driver
            raise ValueError(f"{args.shapes}: expected key '{family.shapes_key}', got {sorted(spec)}")
        shapes = [tuple(s) for s in spec[family.shapes_key]]
    # Default is the canonical packaged path the runtime auto-loads from; --out-dir is for
    # the per-worker shards of a --gpus run.
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "pytorch", _arch_tag()
    )
    t0 = time.perf_counter()
    if args.gpus > 1:
        run_sharded(family, shapes, out_dir, args.gpus)
    else:
        run(family, shapes, out_dir)
    logger.info(f"[{family.tag}] done -> {out_dir} ({time.perf_counter() - t0:.1f}s)")
