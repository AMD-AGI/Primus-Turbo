###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
import importlib
import multiprocessing as mp
import os
import queue
import socket
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.core.low_precision import is_fp8_dtype


def skip_if_lt_x_gpu(min_gpus):
    """Decorator to skip test if less than min_gpus are available."""
    reason = f"Test requires at least {min_gpus} GPUs, but only {jax.local_device_count()} available"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if jax.local_device_count() < min_gpus:
                pytest.skip(reason)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_tolerances(dtype):
    """Get relative and absolute tolerances for different dtypes."""
    if dtype == jnp.float32:
        return dict(rtol=1e-4, atol=1e-4)
    elif dtype == jnp.float16:
        return dict(rtol=1e-2, atol=1e-2)
    elif dtype == jnp.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    elif is_fp8_dtype(dtype):
        return dict(rtol=1e-1, atol=1e-1)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


###################################################################


# Relative Error
# Note: x is ref
def relative_error(x: jnp.ndarray, y: jnp.ndarray):
    """Compute relative error between two arrays."""
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    return float(jnp.linalg.norm(x - y) / jnp.linalg.norm(x))


# MSE Error
def mean_squared_error(x: jnp.ndarray, y: jnp.ndarray):
    """Compute mean squared error between two arrays."""
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    return float(jnp.mean((x - y) ** 2))


# Max Abs Error
def max_abs_error(x: jnp.ndarray, y: jnp.ndarray):
    """Compute maximum absolute error between two arrays."""
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    return float(jnp.max(jnp.abs(x - y)))


# Cosine Similarity
def cosine_similarity(x: jnp.ndarray, y: jnp.ndarray):
    """Compute cosine similarity between two arrays."""
    x, y = x.flatten().astype(jnp.float32), y.flatten().astype(jnp.float32)
    dot_product = jnp.dot(x, y)
    norm_x = jnp.linalg.norm(x)
    norm_y = jnp.linalg.norm(y)
    return float(dot_product / (norm_x * norm_y))


# Symmetric Similarity
def symmetric_similarity_diff(x: jnp.ndarray, y: jnp.ndarray):
    """Compute symmetric similarity difference between two arrays."""
    x, y = x.astype(jnp.float64), y.astype(jnp.float64)
    denominator = jnp.sum(x * x + y * y)
    sim = 2 * jnp.sum(x * y) / denominator
    return float(1 - sim)


# SNR
# Note: x is ref
def compute_snr(x: jnp.ndarray, y: jnp.ndarray):
    """Compute Signal-to-Noise Ratio in dB. x is reference."""
    x, y = x.astype(jnp.float32), y.astype(jnp.float32)
    signal_power = jnp.linalg.norm(x) ** 2
    noise_power = jnp.linalg.norm(x - y) ** 2
    snr = 10 * jnp.log10(signal_power / (noise_power + 1e-12))
    return float(snr)


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert two arrays are close. Auto-converts to float32 for numpy compatibility."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


###################################################################
# Multi-process test harness (1 GPU per process)
###################################################################


def _find_free_port():
    """
    Return an ephemeral port that is (momentarily) free on localhost.

    NOTE: there is a TOCTOU race between this call returning and any later
    bind against the returned port (see PR #344 Copilot review thread).
    For multi-process tests on a busy host this can be flaky; tracked as a
    follow-up to add a retry-on-EADDRINUSE wrapper around
    ``jax.distributed.initialize``. The race window is small in practice
    and acceptable for the current test workload.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker_entry(rank, world_size, coordinator_port, fn_module, fn_name, error_queue):
    """Child-process entry point.

    All GPUs remain visible (module-level ``import jax`` in this file
    initialises the ROCm/HIP runtime before this function runs under the
    ``spawn`` multiprocessing context).  ``local_device_ids=[rank]`` tells
    JAX to bind this process to a single, distinct GPU so that RCCL sees
    unique ``(bus_id, device_index)`` pairs across ranks.
    """
    try:
        import jax  # noqa: E402 – intentionally late import

        jax.distributed.initialize(
            coordinator_address=f"localhost:{coordinator_port}",
            num_processes=world_size,
            process_id=rank,
            local_device_ids=[rank],
        )

        mod = importlib.import_module(fn_module)
        fn = getattr(mod, fn_name)
        fn(rank, world_size)
    except Exception:
        error_queue.put((rank, traceback.format_exc()))


class JaxMultiProcessTestCase:
    """Pytest-friendly base class for 1-GPU-per-process JAX tests.

    Worker functions **must** be top-level (module-scope) so that they can be
    resolved by name in the child process.  They receive ``(rank, world_size)``
    as positional arguments::

        def _my_worker(rank, world_size):
            import jax
            ...

        class TestFoo(JaxMultiProcessTestCase):
            @pytest.mark.multigpu
            def test_bar(self):
                self.run_multiprocess(_my_worker)
    """

    TIMEOUT = 120  # per-process join timeout in seconds

    def run_multiprocess(self, test_fn, world_size=None):
        """Spawn *world_size* processes and execute *test_fn* in each.

        Parameters
        ----------
        test_fn : callable(rank: int, world_size: int) -> None
            Must be a **top-level** function (required for cross-process
            name resolution).
        world_size : int, optional
            Defaults to ``jax.local_device_count()`` of the parent process.
        """
        if world_size is None:
            world_size = jax.local_device_count()

        fn_module = test_fn.__module__
        fn_name = test_fn.__qualname__
        assert "." not in fn_name, f"test_fn must be a top-level function, got {fn_module}.{fn_name}"

        coordinator_port = _find_free_port()
        ctx = mp.get_context("spawn")
        error_queue = ctx.Queue()
        processes = []

        for rank in range(world_size):
            p = ctx.Process(
                target=_worker_entry,
                args=(rank, world_size, coordinator_port, fn_module, fn_name, error_queue),
            )
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join(timeout=self.TIMEOUT)

            errors = []
            while True:
                try:
                    errors.append(error_queue.get_nowait())
                except queue.Empty:
                    break

            if errors:
                msg = "\n".join(f"--- Rank {r} ---\n{tb}" for r, tb in errors)
                raise AssertionError(f"Multi-process test failed:\n{msg}")

            for i, p in enumerate(processes):
                if p.is_alive():
                    raise AssertionError(f"Process rank={i} timed out after {self.TIMEOUT}s")
                if p.exitcode != 0:
                    raise AssertionError(f"Process rank={i} exited with code {p.exitcode}")
        finally:
            for p in processes:
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5)
