###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Multi-process (1 GPU per process) MoE dispatch/combine tests using DeepEP
``per_process`` mode.

Tests spawn N child processes (one per local GPU), set
``PRIMUS_TURBO_JAX_DEEPEP_MODE=per_process``, initialise JAX distributed,
call ``primus_turbo.jax.lax.moe.setup`` (strict-freeze contract — required
before the first ``moe_dispatch`` / ``moe_combine``), and exercise
dispatch/combine through the IPC-based buffer path.

Mirrors the verification logic of ``tests/jax/lax/test_dispatch_combine.py``
(the inproc / pmap variant), adapted for multi-process execution.

Run with::

    pytest tests/jax/lax/test_mp_dispatch_combine.py --dist-only
"""

import pytest

from tests.jax.test_utils import JaxMultiProcessTestCase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_TOKENS = 4096
_HIDDEN = 7168
_NUM_TOPK = 8
_EXPERTS_PER_RANK = 32

# ---------------------------------------------------------------------------
# Helpers (top-level for pickling; framework imports are late)
# ---------------------------------------------------------------------------


def _calc_diff(x, y):
    """Symmetric similarity diff (same metric as the inproc test suite)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64) + 1
    y = np.asarray(y, dtype=np.float64) + 1
    denom = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denom
    return float(1 - sim)


def _init_per_process():
    """Set env, initialise JAX distributed, return ``(jax, jnp, np)``.

    DeepEP ``setup()`` is deliberately *not* called here so each worker
    can size buffers from its own per-test hidden_bytes (and call
    ``reset_runtime`` first if the test needs to override the prior
    snapshot).  Every worker that calls ``moe_dispatch`` /
    ``moe_combine`` must invoke ``_setup_deepep_for_worker`` (or its own
    ``setup``) before the first dispatch — the strict-freeze contract
    raises ``RuntimeError`` otherwise.
    """
    import os

    os.environ["PRIMUS_TURBO_JAX_DEEPEP_MODE"] = "per_process"

    import jax
    import jax.numpy as jnp
    import numpy as np

    import primus_turbo.jax  # noqa: F401 – registers FFI targets

    primus_turbo.jax.initialize()
    return jax, jnp, np


def _setup_deepep_for_worker(hidden: int = _HIDDEN) -> None:
    """Per-worker DeepEP ``setup()`` for PER_PROCESS mode.

    Buffer-sizing convention is ``hidden * max(dtype_itemsize, 2)``
    (the same formula primus_turbo uses internally to size the C++ NVL
    / RDMA buffer).  We pick bf16 as the canonical dtype: workers run
    both bf16 (``itemsize=2``) and fp8 e4m3 (``itemsize=1``) over the
    same ``hidden`` dim, and ``max(itemsize, 2)`` collapses both to 2,
    so a bf16-sized buffer covers both dispatches.  The C++ buffer
    grows-only, so re-running with a larger ``hidden`` is fine; tests
    needing a smaller ``hidden`` should still pass their own value
    rather than relying on the leftover from a previous run.

    ``reset_runtime()`` first keeps state clean across re-entries in
    the same subprocess (each worker is its own process, but the
    function is also safe to call twice).
    """
    import jax.numpy as jnp

    from primus_turbo.jax.lax.moe import reset_runtime, setup

    reset_runtime()
    setup(hidden_bytes=hidden * max(jnp.dtype(jnp.bfloat16).itemsize, 2))


def _generate(rank, world_size):
    """Generate per-rank test data (no leading device axis)."""
    import jax
    import jax.numpy as jnp

    num_experts = _EXPERTS_PER_RANK * world_size
    key = jax.random.PRNGKey(rank)

    x = jnp.ones((_NUM_TOKENS, _HIDDEN), dtype=jnp.bfloat16) * rank
    scores = jnp.abs(jax.random.normal(key, (_NUM_TOKENS, num_experts), dtype=jnp.float32)) + 1
    topk_idx = jax.lax.top_k(scores, _NUM_TOPK)[1].astype(jnp.int32)
    topk_weights = jnp.ones((_NUM_TOKENS, _NUM_TOPK), dtype=jnp.float32) * rank

    return x, topk_idx, topk_weights, num_experts


def _per_token_cast_to_fp8(x):
    """BF16 ``[n, h]`` -> ``(FP8 E4M3 [n, h], scales [n, h//128])``."""
    import jax.numpy as jnp

    from primus_turbo.jax.core.low_precision import float8_e4m3

    n, h = x.shape
    assert h % 128 == 0
    fp8_max = jnp.finfo(float8_e4m3).max.astype(jnp.float32)
    x_f32 = x.astype(jnp.float32).reshape(n, -1, 128)
    amax = jnp.abs(x_f32).max(axis=-1)
    scale = jnp.maximum(amax / fp8_max, jnp.finfo(jnp.float32).tiny)
    x_scaled = x_f32 / scale[..., None]
    x_fp8 = x_scaled.reshape(n, h).astype(float8_e4m3)
    return x_fp8, scale


def _per_token_cast_back(x_fp8, x_scales):
    """``(FP8 E4M3 [n, h], scales [n, h//128])`` -> BF16 ``[n, h]``."""
    import jax.numpy as jnp

    n, h = x_fp8.shape
    x_f32 = x_fp8.astype(jnp.float32).reshape(n, -1, 128)
    return (x_f32 * x_scales[..., None]).reshape(n, h).astype(jnp.bfloat16)


def _check_dispatch(cx, cw, rpm, rank, world_size, num_experts):
    """Verify dispatch correctness for a single rank.

    Checks:
      1. Each received row is constant (all elements equal).
      2. Tokens from source *src* carry integer value *src*.
      3. ``recv_topk_idx`` values are in ``[-1, experts_per_rank)``.
      4. ``recv_topk_weights`` follow the same routing pattern.
    """
    import numpy as np

    recv_size = int(rpm[world_size - 1][rank])

    assert np.allclose(
        np.amin(cx[:recv_size], axis=1), np.amax(cx[:recv_size], axis=1)
    ), f"Rank {rank}: received tokens should be constant per row"

    start = 0
    for src in range(world_size):
        end = int(rpm[src][rank])
        assert (
            cx[start:end, :].astype(np.int32) - src
        ).sum() == 0, f"Rank {rank}: tokens from src={src} should carry value {src}"
        assert (
            cw[start:end, :].astype(np.int32) - src
        ).sum() == 0, f"Rank {rank}: weights from src={src} should carry value {src}"
        start = end


def _dispatch_and_check(rank, world_size, use_fp8):
    """Run dispatch + combine forward and return ``(combined_x, x, handle)``."""
    import jax.numpy as jnp
    import numpy as np

    from primus_turbo.jax.lax.moe import moe_combine, moe_dispatch

    x, topk_idx, topk_weights, num_experts = _generate(rank, world_size)

    if use_fp8:
        x_fp8, x_scales = _per_token_cast_to_fp8(x)
        recv_x_tuple, recv_topk_idx, recv_topk_weights, handle = moe_dispatch(
            (x_fp8, x_scales), topk_idx, topk_weights, num_experts
        )
        recv_x = _per_token_cast_back(*recv_x_tuple)
    else:
        recv_x, recv_topk_idx, recv_topk_weights, handle = moe_dispatch(
            x, topk_idx, topk_weights, num_experts
        )

    rank_prefix_matrix = handle[0]
    is_token_in_rank = handle[4]

    cx = np.array(recv_x)
    rpm = np.array(rank_prefix_matrix)
    idx = np.array(recv_topk_idx)

    # --- recv_topk_idx validity ---
    experts_per_rank = num_experts // world_size
    valid = (idx == -1) | ((idx >= 0) & (idx < experts_per_rank))
    assert valid.all(), f"Rank {rank}: recv_topk_idx out of range"

    # --- dispatch routing (replace -1 slots in weights before checking) ---
    cw = np.array(recv_topk_weights)
    amax_cw = np.broadcast_to(np.amax(cw, axis=1, keepdims=True), cw.shape)
    cw = np.where(idx == -1, amax_cw, cw)
    _check_dispatch(cx, cw, rpm, rank, world_size, num_experts)

    # --- combine round-trip ---
    combined_x = moe_combine(recv_x, handle)
    assert combined_x.shape == (_NUM_TOKENS, _HIDDEN)

    scale = jnp.expand_dims(is_token_in_rank.sum(axis=1), axis=1)
    normalised = combined_x.astype(jnp.float32) / scale

    tol = 5e-4 if use_fp8 else 5e-6
    diff = _calc_diff(normalised, x)
    assert diff < tol, f"Rank {rank}: combine round-trip diff={diff} (tol={tol})"


# ---------------------------------------------------------------------------
# Worker functions (top-level for cross-process name resolution)
# ---------------------------------------------------------------------------


def _worker_dispatch_combine_fwd_bf16(rank, world_size):
    """BF16 forward: dispatch correctness + topk_idx validity + combine round-trip."""
    _init_per_process()
    _setup_deepep_for_worker()
    _dispatch_and_check(rank, world_size, use_fp8=False)


def _worker_dispatch_combine_fwd_fp8(rank, world_size):
    """FP8 forward: dispatch correctness + topk_idx validity + combine round-trip."""
    _init_per_process()
    _setup_deepep_for_worker()
    _dispatch_and_check(rank, world_size, use_fp8=True)


def _worker_dispatch_combine_bwd_bf16(rank, world_size):
    """Backward via jax.vjp: round-trip gradient check (BF16).

    ``f(x) = combine(dispatch(x)) / K ≈ identity``, so ``J ≈ I``.
    With cotangent ``v = f(x) ≈ x``, vjp gives ``grad_x = J^T v ≈ x``.
    """
    import jax
    import jax.numpy as jnp

    _init_per_process()
    _setup_deepep_for_worker()
    from primus_turbo.jax.lax.moe import moe_combine, moe_dispatch

    x, topk_idx, topk_weights, num_experts = _generate(rank, world_size)

    def fwd(x):
        recv_x, _, _, handle = moe_dispatch(x, topk_idx, topk_weights, num_experts)
        combined_x = moe_combine(recv_x, handle)
        is_token_in_rank = handle[4]
        scale = jnp.expand_dims(is_token_in_rank.sum(axis=1), axis=1)
        return (combined_x.astype(jnp.float32) / scale).astype(x.dtype)

    output, vjp_fn = jax.vjp(fwd, x)
    (grad_x,) = vjp_fn(output)

    fwd_diff = _calc_diff(output, x)
    assert fwd_diff < 5e-6, f"Rank {rank}: forward diff={fwd_diff}"

    bwd_diff = _calc_diff(grad_x, x)
    assert bwd_diff < 5e-6, f"Rank {rank}: backward diff={bwd_diff}"


def _worker_eval_shape_after_setup(rank, world_size):
    """Verify ``jax.eval_shape`` traces dispatch/combine without crashing.

    Reproduces (in regression form) the ``TracerArrayConversionError`` that
    occurred when ``jax.eval_shape`` traced through
    ``moe_dispatch -> ensure_deepep_runtime -> process_allgather ->
    np.asarray(tracer)``.

    Under the strict-freeze contract, the trace-path call to
    ``ensure_deepep_runtime`` is gone entirely: dispatch/combine read
    only the frozen state from ``setup()``, so there is no path from
    a Tracer to an allgather inside ``moe_dispatch``.  This test pins
    that invariant.
    """
    jax_mod, jnp, np = _init_per_process()
    _setup_deepep_for_worker()
    from primus_turbo.jax.lax.moe import moe_combine, moe_dispatch

    num_experts = _EXPERTS_PER_RANK * world_size

    def model_fn(x, topk_idx, topk_weights):
        recv_x, _, _, handle = moe_dispatch(x, topk_idx, topk_weights, num_experts)
        combined_x = moe_combine(recv_x, handle)
        return combined_x

    x_shape = jax_mod.ShapeDtypeStruct((_NUM_TOKENS, _HIDDEN), jnp.bfloat16)
    idx_shape = jax_mod.ShapeDtypeStruct((_NUM_TOKENS, _NUM_TOPK), jnp.int32)
    weights_shape = jax_mod.ShapeDtypeStruct((_NUM_TOKENS, _NUM_TOPK), jnp.float32)

    out_shape = jax_mod.eval_shape(model_fn, x_shape, idx_shape, weights_shape)
    assert (
        out_shape.shape[1] == _HIDDEN
    ), f"Rank {rank}: expected hidden dim {_HIDDEN}, got {out_shape.shape[1]}"


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestPerProcessDispatchCombine(JaxMultiProcessTestCase):
    """Multi-process DeepEP MoE dispatch/combine tests (1 GPU per process)."""

    def _skip_if_insufficient_gpus(self):
        import jax

        if jax.local_device_count() < 2:
            pytest.skip("Need at least 2 GPUs for per_process mode")

    @pytest.mark.multigpu
    def test_dispatch_combine_fwd_bf16(self):
        """BF16 forward: dispatch correctness + topk_idx + combine round-trip."""
        self._skip_if_insufficient_gpus()
        self.run_multiprocess(_worker_dispatch_combine_fwd_bf16)

    @pytest.mark.multigpu
    def test_dispatch_combine_fwd_fp8(self):
        """FP8 forward: dispatch correctness + topk_idx + combine round-trip."""
        self._skip_if_insufficient_gpus()
        self.run_multiprocess(_worker_dispatch_combine_fwd_fp8)

    @pytest.mark.multigpu
    def test_dispatch_combine_backward(self):
        """BF16 backward: jax.vjp round-trip gradient check."""
        self._skip_if_insufficient_gpus()
        self.run_multiprocess(_worker_dispatch_combine_bwd_bf16)

    @pytest.mark.multigpu
    def test_eval_shape_after_setup(self):
        """``jax.eval_shape`` after ``setup()``: shape inference with no
        Tracer leak into the C++ bootstrap path."""
        self._skip_if_insufficient_gpus()
        self.run_multiprocess(_worker_eval_shape_after_setup)
