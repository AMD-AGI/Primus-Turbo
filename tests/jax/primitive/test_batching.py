###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Shared batching semantics using real JAX transformations and a pure body."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.primitive._batching import (
    broadcast_to_batch,
    enable_x64,
    make_grouped_gemm_spmd_rule,
    make_ipc_batchers,
    spmd_name_or_none,
)


@pytest.mark.parametrize(
    "name,expected",
    [(None, None), ((), None), ([], None), ("stage", "stage"), (("stage",), "stage"), (["stage"], "stage")],
)
def test_spmd_name(name, expected):
    assert spmd_name_or_none(SimpleNamespace(spmd_name=name)) == expected
    assert spmd_name_or_none(object()) is None


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_broadcast_axis(axis):
    x = jnp.arange(6).reshape(2, 3)
    actual = broadcast_to_batch(x, axis, 4)
    expected = np.broadcast_to(np.array(x)[None], (4, 2, 3)) if axis is None else np.moveaxis(x, axis, 0)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("fancy", [False, True])
def test_ipc_scan_preserves_microbatches(axis, fancy):
    primitive = SimpleNamespace(bind=lambda x, y, factor: (factor * x + y, x - y))
    plain, spmd = make_ipc_batchers(primitive, 2)
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    y = jnp.arange(4, dtype=jnp.float32)
    args = (jnp.moveaxis(x, 0, axis), y)
    rule = (
        (lambda args, axes, **kw: spmd(SimpleNamespace(spmd_name=None), args, axes, **kw)) if fancy else plain
    )
    outputs = jax.jit(lambda a, b: rule((a, b), (axis, None), factor=2)[0])(*args)
    np.testing.assert_array_equal(outputs[0], 2 * x + y)
    np.testing.assert_array_equal(outputs[1], x - y)
    assert rule(args, (axis, None), factor=2)[1] == (0, 0)
    unbatched, dims = rule((x[0], y), (None, None), factor=3)
    assert dims == (None, None)
    np.testing.assert_array_equal(unbatched[0], 3 * x[0] + y)


def test_ipc_spmd_selects_local_stage():
    primitive = SimpleNamespace(bind=lambda x, y: (x + y, x - y))
    _, fancy = make_ipc_batchers(primitive, 2)
    x = jnp.arange(12).reshape(3, 4)
    y = jnp.arange(4)
    axis_data = SimpleNamespace(size=3, spmd_name=("stage",))

    def stage(_):
        outputs, dims = fancy(axis_data, (x, y), (0, None))
        assert dims == (0, 0)
        return outputs

    plus, minus = jax.jit(jax.vmap(stage, axis_name="stage"))(jnp.arange(3))
    for i in range(3):
        np.testing.assert_array_equal(plus[i], np.broadcast_to(x[i] + y, (3, 4)))
        np.testing.assert_array_equal(minus[i], np.broadcast_to(x[i] - y, (3, 4)))
    outputs, dims = fancy(axis_data, (x[0], y), (None, None))
    assert dims == (None, None)
    np.testing.assert_array_equal(outputs[0], x[0] + y)


def test_grouped_spmd_int64_and_fallback():
    def bind(a, b, lengths, offsets):
        assert lengths.dtype == offsets.dtype == jnp.int64
        return a + b, jnp.zeros((2,), jnp.uint8)

    plain_calls = []

    def plain(args, axes, **kwargs):
        plain_calls.append(axes)
        return (args[0] + args[1], jnp.zeros((2,), jnp.uint8)), (0, None)

    fancy = make_grouped_gemm_spmd_rule(SimpleNamespace(bind=bind), plain)
    a = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    b = jnp.ones(4)
    lengths = jnp.array([4], jnp.int32)
    offsets = jnp.array([0, 4], jnp.int32)
    with enable_x64(False):

        def stage(_):
            outputs, dims = fancy(
                SimpleNamespace(size=3, spmd_name="stage"), (a, b, lengths, offsets), (0, None, None, None)
            )
            assert dims == (0, 0)
            return outputs

        out, ws = jax.vmap(stage, axis_name="stage")(jnp.arange(3))
    for i in range(3):
        np.testing.assert_array_equal(out[i], np.broadcast_to(a[i] + b, (3, 4)))
    assert ws.shape == (3, 3, 2) and ws.dtype == jnp.uint8
    fancy(SimpleNamespace(spmd_name=None), (a, b, lengths, offsets), (0, None, None, None))
    assert plain_calls == [(0, None, None, None)]
    with enable_x64(True):
        out, dims = fancy(
            SimpleNamespace(spmd_name="stage"),
            (a[0], b, lengths.astype(jnp.int64), offsets.astype(jnp.int64)),
            (None,) * 4,
        )
    assert dims == (None, None)
    np.testing.assert_array_equal(out[0], a[0] + b)


def test_x64_context_legacy_import(monkeypatch):
    """Older JAX exposes the same context manager in jax.experimental."""
    import runpy

    from primus_turbo.jax.primitive import _batching

    monkeypatch.setattr(jax.experimental, "enable_x64", enable_x64, raising=False)
    monkeypatch.delattr(jax, "enable_x64", raising=False)
    namespace = runpy.run_path(_batching.__file__)
    with namespace["enable_x64"](True):
        assert jax.config.x64_enabled
    with namespace["enable_x64"](False):
        assert not jax.config.x64_enabled
