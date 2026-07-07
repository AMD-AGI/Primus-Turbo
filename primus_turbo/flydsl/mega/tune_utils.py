###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kineto-based device-time benchmark (``bench_kineto``) used by the autotuner.
"""

import os
import sys

import torch

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
