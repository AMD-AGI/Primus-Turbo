###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Pre-alloc state-machine tests for ``primus_turbo.jax.lax.moe.setup``.

These tests exercise only the Python-level early-fail validation and the
frozen-state contract; they never reach the C++ buffer allocator and
therefore run on a single CPU process with no GPU required.

The GPU-coupled tests for the actual dispatch/combine round-trip live in
``test_dispatch_combine.py`` and ``test_mp_dispatch_combine.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from primus_turbo.jax.lax.moe import moe_combine, moe_dispatch
from primus_turbo.jax.lax.moe import moe_dispatch_combine as mdc
from primus_turbo.jax.lax.moe import reset_runtime, setup


@pytest.fixture(autouse=True)
def _fresh_runtime():
    """Every test starts and ends with a clean global runtime."""
    reset_runtime()
    yield
    reset_runtime()


# ---------------------------------------------------------------------------
#  Early-fail validation (no C++ touched)
# ---------------------------------------------------------------------------


def test_setup_rejects_mesh_and_ep_ranks_simultaneously():
    """``mesh=`` and ``ep_ranks=`` are mutually exclusive."""
    with pytest.raises(ValueError, match="either `mesh` or `ep_ranks`, not both"):
        setup(
            hidden_bytes=1024,
            mesh=object(),
            ep_ranks=[0, 1],
        )


def test_setup_rejects_unknown_mode_string():
    """Only 'inproc' / 'per_process' / None are accepted for ``mode``."""
    with pytest.raises(ValueError, match="`mode` must be"):
        setup(hidden_bytes=1024, mode="multi_node")


def test_setup_rejects_inproc_mode_under_multiprocess_world(monkeypatch):
    """``mode='inproc'`` is invalid when ``jax.process_count() > 1``."""
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    with pytest.raises(ValueError, match="mode='inproc' is invalid for multi-process"):
        setup(hidden_bytes=1024, mode="inproc")


def test_setup_rejects_per_process_mode_under_single_process(monkeypatch):
    """``mode='per_process'`` is invalid for a single-process multi-GPU launch."""
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    with pytest.raises(ValueError, match="mode='per_process' is invalid for single-process"):
        setup(hidden_bytes=1024, mode="per_process")


# ---------------------------------------------------------------------------
#  Frozen-state contract
# ---------------------------------------------------------------------------


def test_dispatch_without_setup_raises_clear_runtime_error():
    """Forgetting ``setup()`` raises before touching C++."""
    x = jnp.zeros((4, 8), dtype=jnp.bfloat16)
    topk_idx = jnp.zeros((4, 2), dtype=jnp.int32)
    topk_w = jnp.zeros((4, 2), dtype=jnp.float32)
    with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
        moe_dispatch(x, topk_idx, topk_w, num_experts=8)


def test_combine_without_setup_raises_clear_runtime_error():
    """Forgetting ``setup()`` raises on combine too."""
    x = jnp.zeros((4, 8), dtype=jnp.bfloat16)
    fake_handle = (jnp.zeros(1, dtype=jnp.int32),) * 6
    with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
        moe_combine(x, fake_handle)


def test_require_frozen_returns_snapshot_after_manual_freeze():
    """Internal: ``_require_frozen`` returns the cached snapshot."""
    fake = mdc.FrozenRuntimeState(
        mode=mdc.MODE_INPROC,
        launch_mode=int(mdc.MODE_INPROC),
        ep_size=2,
        is_internode=False,
        num_sms=64,
        source_meta_bytes=0,
    )
    mdc._frozen = fake
    try:
        snapshot = mdc._require_frozen()
        assert snapshot is fake
        assert snapshot.ep_size == 2
        assert snapshot.is_internode is False
    finally:
        mdc._frozen = None


def test_frozen_state_is_immutable():
    """``FrozenRuntimeState`` is a frozen dataclass; field writes raise."""
    snap = mdc.FrozenRuntimeState(
        mode=mdc.MODE_INPROC,
        launch_mode=0,
        ep_size=2,
        is_internode=False,
        num_sms=64,
        source_meta_bytes=0,
    )
    with pytest.raises((AttributeError, TypeError)):
        snap.ep_size = 999  # type: ignore[misc]


def test_reset_runtime_clears_frozen_snapshot():
    """``reset_runtime`` returns module-level state to pristine."""
    mdc._frozen = mdc.FrozenRuntimeState(
        mode=mdc.MODE_INPROC,
        launch_mode=0,
        ep_size=2,
        is_internode=False,
        num_sms=64,
        source_meta_bytes=0,
    )
    mdc._default_num_sms = 32
    reset_runtime()
    assert mdc._frozen is None
    assert mdc._default_num_sms == 64


def test_get_ep_size_without_setup_raises():
    """Public ``get_ep_size`` is frozen-state-only; raises before ``setup()``."""
    from primus_turbo.jax.lax.moe import get_ep_size

    with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
        get_ep_size()


def test_is_internode_without_setup_raises():
    """Public ``is_internode`` is frozen-state-only; raises before ``setup()``."""
    from primus_turbo.jax.lax.moe import is_internode

    with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
        is_internode()


def test_is_internode_reads_frozen_state():
    """``is_internode`` returns the value cached at ``setup()`` time."""
    from primus_turbo.jax.lax.moe import is_internode

    mdc._frozen = mdc.FrozenRuntimeState(
        mode=mdc.MODE_PER_PROCESS,
        launch_mode=int(mdc.MODE_PER_PROCESS),
        ep_size=16,
        is_internode=True,
        num_sms=32,
        source_meta_bytes=80,
    )
    try:
        assert is_internode() is True
    finally:
        mdc._frozen = None

    mdc._frozen = mdc.FrozenRuntimeState(
        mode=mdc.MODE_INPROC,
        launch_mode=int(mdc.MODE_INPROC),
        ep_size=8,
        is_internode=False,
        num_sms=64,
        source_meta_bytes=0,
    )
    try:
        assert is_internode() is False
    finally:
        mdc._frozen = None
