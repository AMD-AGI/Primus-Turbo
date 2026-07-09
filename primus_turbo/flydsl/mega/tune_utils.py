###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kineto-based device-time benchmark (``bench_kineto``) plus an EP-safe autotuner.

``Autotuner`` here SUBCLASSES flydsl's built-in autotuner and overrides only the
disk-cache I/O -- everything else (key building, benchmarking, config selection,
launch) is reused verbatim.

Why the override: flydsl's autotuner persists best configs to ONE shared JSON per
kernel with no concurrency control (``_save_disk_cache`` does a non-atomic
whole-file ``write_text``; ``_load_disk_cache`` swallows parse errors). Under EP
the 8 rank processes share that file, so torn reads and lost updates -- combine's
key even includes ``rank``, so a whole-file overwrite drops other ranks' entries
-- leave ranks with inconsistent caches. They then pick DIFFERENT configs for the
*collective* dispatch/combine kernels, whose cross-rank spin barrier never
handshakes -> deadlock.

Fix (minimal): guard the shared file with an flock, and make the write a
read-merge-write + atomic ``os.replace`` so no rank loses another rank's key or
reads a torn file. Once a run has tuned, every rank loads the same complete file
next run -> same config everywhere -> no re-tune, no miss, no hang. Compiled
binaries are already cached safely by flydsl's own JIT disk cache
(``FLYDSL_RUNTIME_CACHE_DIR``, default ``~/.flydsl/cache``; per-key lock + atomic
write) -- we just reuse it (never wipe it).
"""

import json
import os
import sys
from contextlib import contextmanager

import torch

# subclass flydsl's autotuner; reuse its Config verbatim
from flydsl import Config
from flydsl.autotune import Autotuner as _BaseAutotuner

try:
    import fcntl  # POSIX advisory file lock for the shared config cache
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

__all__ = ["bench_kineto", "Config", "autotune", "Autotuner"]

# Device ops to drop from the aggregate timing path (housekeeping, not the kernel).
_BENCH_DENYLIST = ("memset", "Memset", "fill", "Fill", "Copy", "memcpy")


class _empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _suppress_stdout_stderr:
    """Redirect stdout/stderr to /dev/null (silences flydsl JIT prints)."""

    def __enter__(self):
        # Flush pending Python-buffered output before swapping the fd.
        sys.stdout.flush()
        sys.stderr.flush()
        self._outnull = open(os.devnull, "w")
        self._errnull = open(os.devnull, "w")
        self._out_fd = os.dup(sys.stdout.fileno())
        self._err_fd = os.dup(sys.stderr.fileno())
        os.dup2(self._outnull.fileno(), sys.stdout.fileno())
        os.dup2(self._errnull.fileno(), sys.stderr.fileno())
        return self

    def __exit__(self, *_):
        # Flush buffered prints into /dev/null BEFORE restoring the real fd,
        # else block-buffered output (stdout piped) leaks after restore.
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._out_fd, sys.stdout.fileno())
        os.dup2(self._err_fd, sys.stderr.fileno())
        os.close(self._out_fd)
        os.close(self._err_fd)
        self._outnull.close()
        self._errnull.close()


def bench_kineto(fn, num_tests=20, num_warmups=5, flush_l2=True, trace_path=None, suppress_output=True):
    """Measure the per-iteration device time (us) of ``fn`` via the kineto profiler.

    Sums every device kernel ``fn`` launches (housekeeping memset/copy/fill ops
    dropped via ``_BENCH_DENYLIST``) -- no kernel name needed.
    """
    # Warm up outside the profiler so the flydsl JIT compiles before timing.
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    # L2 flush before timing (the flush memset runs outside the profiler window
    # and is denylisted, so it never lands in the aggregate sum).
    if flush_l2:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
        cache.zero_()

    suppress = _suppress_stdout_stderr if suppress_output else _empty_suppress
    schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
    with suppress():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        ) as prof:
            for _ in range(2):
                for _ in range(num_tests):
                    fn()
                torch.cuda.synchronize()
                prof.step()

    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Parse the profiling table (deep_ep style): the per-call device time is the
    # second-to-last column ("CUDA time avg"); "# of Calls" is the last column.
    prof_lines = prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=100).split("\n")
    units = {"ms": 1e3, "us": 1.0}  # -> microseconds

    def _line_us(line):
        time_str = line.split()[-2]
        for unit, scale in units.items():
            if unit in time_str:
                return float(time_str.replace(unit, "")) * scale
        return None

    total = 0.0
    for line in prof_lines:
        toks = line.split()
        if len(toks) < 2 or not toks[-1].isdigit():  # kernel rows end in call count
            continue
        if any(bad in line for bad in _BENCH_DENYLIST):
            continue
        us = _line_us(line)
        if us is not None:
            total += us
    return total


@contextmanager
def _file_lock(lock_path):
    """Exclusive advisory lock around the shared config cache (no-op without fcntl)."""
    if fcntl is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


class Autotuner(_BaseAutotuner):
    """flydsl's autotuner with EP-safe disk I/O.

    Overrides ONLY the shared-file load/save so concurrent ranks can't lose each
    other's keys or read a torn file (which made collective kernels pick different
    configs per rank -> spin-barrier deadlock). All tuning/launch logic is inherited."""

    def _lock_file(self):
        return self._cache_file.with_suffix(".lock")

    def _load_disk_cache(self):
        # lock so we never read while another rank is mid-write
        with _file_lock(self._lock_file()):
            super()._load_disk_cache()

    def _save_disk_cache(self):
        # lock + read-merge-write + atomic replace: keep every rank's key, never torn
        with _file_lock(self._lock_file()):
            disk = {}
            if self._cache_file.exists():
                try:
                    disk = json.loads(self._cache_file.read_text())
                except Exception:  # noqa: BLE001  corrupt/legacy file -> rebuild from ours
                    disk = {}
            # merge our freshly tuned keys on top of whatever peers already wrote
            for key, config in self.cache.items():
                disk[json.dumps(list(key))] = config.to_dict()
            # keep in-memory cache in sync with the merged on-disk view
            self.cache = {tuple(json.loads(k)): Config.from_dict(v) for k, v in disk.items()}
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_file.with_name(f"{self._cache_file.name}.tmp.{os.getpid()}")
            tmp.write_text(json.dumps(disk, indent=2))
            os.replace(tmp, self._cache_file)  # atomic on the same filesystem


def autotune(
    configs,
    key=None,
    warmup=5,
    rep=25,
    prune_configs_by=None,
    reset_to_zero=None,
    pre_hook=None,
    post_hook=None,
    do_bench=None,
):
    """Decorator: wrap a @flyc.jit function in the EP-safe Autotuner (flydsl-compatible)."""

    def decorator(fn):
        return Autotuner(
            fn,
            configs,
            key,
            warmup,
            rep,
            prune_configs_by=prune_configs_by,
            reset_to_zero=reset_to_zero,
            pre_hook=pre_hook,
            post_hook=post_hook,
            do_bench_fn=do_bench,
        )

    return decorator
