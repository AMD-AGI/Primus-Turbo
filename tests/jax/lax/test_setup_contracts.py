###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Setup/freeze contracts; mock bootstrap to avoid allocating fake peer buffers."""

from unittest.mock import Mock

import jax
import pytest

from primus_turbo.jax.lax.moe import moe_dispatch_combine as mdc


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(
        mdc.deep_ep_runtime, "_get_c_deep_ep", lambda: Mock(is_per_process_buffer_ready=lambda: False)
    )
    mdc.reset_runtime()
    yield
    mdc.reset_runtime()


@pytest.mark.parametrize("world", [2, 8, 16])
def test_setup_freezes_bootstrap_state(monkeypatch, world):
    monkeypatch.setattr(jax, "process_count", lambda: world)
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(jax, "local_device_count", lambda: 1)
    bootstrap = Mock()
    monkeypatch.setattr(mdc.deep_ep_runtime, "ensure_deepep_runtime", bootstrap)
    monkeypatch.setattr(mdc.deep_ep_runtime, "get_source_meta_bytes", lambda: 80)
    mdc.setup(hidden_bytes=256, num_sms=32, ep_ranks=list(range(world)), mode="per_process")
    snap = mdc._require_frozen()
    assert (snap.ep_size, snap.num_sms, snap.launch_mode) == (world, 32, 1)
    assert snap.is_internode == (world > 8)
    assert snap.source_meta_bytes == (80 if world > 8 else 0)
    assert bootstrap.call_args.kwargs["hidden_bytes"] == 256
    assert bootstrap.call_args.kwargs["config"].num_sms == 32
    assert mdc.get_ep_size() == world
    assert mdc.get_combine_config().num_sms == 32
    mdc.setup(hidden_bytes=256, num_sms=32, ep_ranks=list(range(world)))
    assert mdc._require_frozen() == snap
    with pytest.raises(RuntimeError, match="different configuration"):
        mdc.setup(hidden_bytes=256, num_sms=64)
    assert mdc._require_frozen() == snap


def test_setup_mesh_forwarding(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 8)
    monkeypatch.setattr(jax, "local_device_count", lambda: 1)
    pin = Mock()
    monkeypatch.setattr(mdc.deep_ep_runtime, "pin_ep_group_from_jax_mesh", pin)
    monkeypatch.setattr(mdc.deep_ep_runtime, "ensure_deepep_runtime", Mock())
    mesh = object()
    mdc.setup(hidden_bytes=128, mesh=mesh, ep_axis_name="ep")
    pin.assert_called_once_with(mesh, axis_name="ep")


def test_inproc_ep_warning(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 1)
    monkeypatch.setattr(jax, "local_device_count", lambda: 8)
    with pytest.warns(UserWarning, match="ignored under INPROC"):
        mdc.setup(ep_ranks=[0], mode="inproc")
    assert mdc.get_ep_size() == 8


@pytest.mark.parametrize(
    "world,local,mode,match",
    [
        (1, 16, "inproc", "INPROC mode cannot"),
        (9, 1, "per_process", "Internode mode requires"),
        (2, 1, "per_process", "hidden_bytes is required"),
    ],
)
def test_setup_rejects_invalid_bootstrap(monkeypatch, world, local, mode, match):
    monkeypatch.setattr(jax, "process_count", lambda: world)
    monkeypatch.setattr(jax, "local_device_count", lambda: local)
    with pytest.raises(ValueError, match=match):
        mdc.setup(mode=mode)
    assert mdc._frozen is None


def test_config_rejects_unsupported_ep_size(monkeypatch):
    monkeypatch.setattr(mdc.deep_ep_runtime, "get_ep_size", lambda: 3)
    for get_config in (mdc.get_dispatch_config, mdc.get_combine_config):
        with pytest.raises(AssertionError, match="Unsupported number of EP ranks"):
            get_config()
