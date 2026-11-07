###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Device discovery and quantization configuration contracts."""

import subprocess
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from primus_turbo.jax.core import low_precision as lp
from primus_turbo.jax.core import utils
from primus_turbo.jax.primitive._batching import enable_x64


@pytest.mark.parametrize(
    "name,expected",
    [
        ("gfx942", (9, 4)),
        ("gfx950:sramecc+", (9, 5)),
        ("", (0, 0)),
        ("cpu", (0, 0)),
        ("gfx9", (0, 0)),
        ("gfxxy", (0, 0)),
    ],
)
def test_parse_architecture(name, expected):
    assert utils._parse_gfx_string(name) == expected


@pytest.mark.parametrize(
    "result,device,expected",
    [
        (SimpleNamespace(returncode=0, stdout="gfx000\n\ngfx942\ngfx950\n", stderr=""), 1, (9, 5)),
        (SimpleNamespace(returncode=0, stdout="gfx942\n", stderr=""), 4, (0, 0)),
        (SimpleNamespace(returncode=1, stdout="", stderr="failure"), 0, (0, 0)),
        (FileNotFoundError(), 0, (0, 0)),
        (subprocess.TimeoutExpired("rocm_agent_enumerator", 5), 0, (0, 0)),
    ],
)
def test_agent_fallback(monkeypatch, result, device, expected):
    def run(argv, **kwargs):
        assert argv == ["rocm_agent_enumerator", "-t", "GPU"]
        assert kwargs["timeout"] == 5
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(subprocess, "run", run)
    assert utils._get_capability_from_rocm_agent(device) == expected


@pytest.mark.parametrize(
    "hip,agent,expected", [((9, 5), (9, 4), (9, 5)), ((0, 0), (9, 4), (9, 4)), ((0, 0), (0, 0), (0, 0))]
)
def test_discovery_precedence_and_cache(monkeypatch, caplog, hip, agent, expected):
    calls = []
    monkeypatch.setattr(utils, "_get_capability_from_hip", lambda i: calls.append(("hip", i)) or hip)
    monkeypatch.setattr(
        utils, "_get_capability_from_rocm_agent", lambda i: calls.append(("agent", i)) or agent
    )
    utils._get_device_compute_capability.cache_clear()
    try:
        assert utils.get_device_compute_capability(3) == expected
        assert utils.get_device_compute_capability(3) == expected
        assert calls == ([("hip", 3)] if hip != (0, 0) else [("hip", 3), ("agent", 3)])
        if expected == (0, 0):
            assert "Could not determine" in caplog.text
        with pytest.raises(ValueError, match="non-negative"):
            utils.get_device_compute_capability(-1)
    finally:
        utils._get_device_compute_capability.cache_clear()


@pytest.mark.parametrize("cap", [(9, 0), (9, 4), (9, 5), (12, 5)])
def test_architecture_gates(monkeypatch, cap):
    monkeypatch.setattr(lp, "get_device_compute_capability", lambda: cap)
    monkeypatch.setattr(utils, "get_device_compute_capability", lambda: cap)
    assert utils.is_gfx942() == (cap == (9, 4))
    assert utils.is_gfx950() == (cap == (9, 5))
    assert utils.is_gfx1250() == (cap == (12, 5))
    for fn, threshold in [
        (lp.check_fp8_support, (9, 4)),
        (lp.check_fp4_support, (9, 5)),
        (lp.check_fp8_ocp_support, (9, 5)),
        (lp.check_mxfp8_support, (9, 5)),
    ]:
        supported, reason = fn()
        assert supported == (cap >= threshold)
        assert bool(reason) == (not supported)


@pytest.mark.parametrize(
    "dtype,enum",
    [
        (jnp.float32, "kFloat32"),
        (jnp.float16, "kFloat16"),
        (jnp.bfloat16, "kBFloat16"),
        (jnp.int32, "kInt32"),
        (jnp.int64, "kInt64"),
        (jnp.float8_e4m3fn, "kFloat8E4M3FN"),
        (jnp.float8_e4m3fnuz, "kFloat8E4M3FNUZ"),
        (jnp.float8_e5m2, "kFloat8E5M2"),
        (jnp.float8_e5m2fnuz, "kFloat8E5M2FNUZ"),
    ],
)
def test_dtype_mapping(dtype, enum):
    with enable_x64(True):
        assert utils.jnp_dtype_to_turbo_dtype(dtype) == getattr(utils.TurboDType, enum)
    assert lp.is_fp8_dtype(dtype) == ("Float8" in enum)
    assert not lp.is_fp4_dtype(dtype)


def test_unsupported_dtype():
    with pytest.raises(ValueError, match="Unsupported"):
        utils.jnp_dtype_to_turbo_dtype(jnp.bool_)


@pytest.mark.parametrize(
    "config,kwargs,match",
    [
        (lp.Float8QuantConfig, {"granularity": lp.ScalingGranularity.BLOCKWISE}, "block_size must be set"),
        (
            lp.Float8QuantConfig,
            {"granularity": lp.ScalingGranularity.MX_BLOCKWISE, "block_size": 16},
            "block_size should be",
        ),
        (
            lp.Float4QuantConfig,
            {"granularity": lp.ScalingGranularity.TENSORWISE},
            "only supports MX_BLOCKWISE",
        ),
        (lp.Float4QuantConfig, {"block_size": 64}, "block_size should be"),
        (lp.Float4QuantConfig, {"format": lp.Format.E4M3}, "Format must be"),
    ],
)
def test_invalid_quantization_config(config, kwargs, match):
    with pytest.raises(AssertionError, match=match):
        config(**kwargs)


def test_valid_quantization_config():
    assert lp.Float4QuantConfig().block_size == 32
    assert lp.Float4QuantConfig(block_size=32).format == lp.Format.E2M1_X2
    assert lp.Float8QuantConfig().granularity == lp.ScalingGranularity.TENSORWISE
    for granularity, block_size in [
        (lp.ScalingGranularity.BLOCKWISE, 128),
        (lp.ScalingGranularity.MX_BLOCKWISE, 32),
    ]:
        assert lp.Float8QuantConfig(granularity=granularity, block_size=block_size).block_size == block_size


def test_plugin_initialization_is_idempotent(monkeypatch):
    import jax

    import primus_turbo.jax as turbo

    turbo.initialize()
    calls = []
    monkeypatch.setattr(jax.ffi, "register_ffi_target", lambda *args, **kwargs: calls.append(args))
    turbo.initialize()
    assert not calls
