###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP-safe autotuner: flydsl's Autotuner with concurrency-safe disk cache I/O.
"""

import json
import os
import sys
from contextlib import contextmanager

# subclass flydsl's autotuner; reuse its Config verbatim
from flydsl import Config
from flydsl.autotune import Autotuner as _BaseAutotuner

try:
    import fcntl  # POSIX advisory file lock for the shared config cache
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

__all__ = ["Config", "autotune", "Autotuner"]


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
        # Flush into /dev/null before restoring the real fd, else buffered output leaks.
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._out_fd, sys.stdout.fileno())
        os.dup2(self._err_fd, sys.stderr.fileno())
        os.close(self._out_fd)
        os.close(self._err_fd)
        self._outnull.close()
        self._errnull.close()


@contextmanager
def _file_lock(lock_path):
    """Exclusive advisory lock around the shared config cache (local FS only; no-op without fcntl)."""
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
    """flydsl's autotuner with EP-safe disk I/O so concurrent ranks can't lose keys or read torn files."""

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
