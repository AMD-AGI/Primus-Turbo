###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Schema contract between pinned configs and the kernels that consume them.

A ``backend_config`` is opaque to the framework, so nothing stops a kernel's knobs
from drifting away from the configs an older build tuned. At runtime such a config
degrades to the kernel's default (see ``KernelBackend.execute``), which silently
costs the whole benefit of offline tuning. These tests surface that drift here, in
front of whoever changed the kernel.
"""

import glob
import json
import os

import pytest
import torch

from primus_turbo.pytorch.core.backend import _TUNE_CONFIGS_ROOT, BackendType
from primus_turbo.pytorch.core.low_precision import ScalingGranularity
from primus_turbo.triton.gemm.gemm_fp8_kernel import (
    _DEFAULT_BLOCKWISE_CFG,
    _blockwise_cfg_or_default,
    tune_gemm_fp8_blockwise_triton_kernel,
)

# (backend, granularity) -> the key set a pinned config must carry. Register a pair here
# when a backend starts emitting configs, otherwise the asset test below rejects them.
_PINNED_SCHEMAS = {
    (BackendType.TRITON.name, ScalingGranularity.BLOCKWISE.name): _DEFAULT_BLOCKWISE_CFG.keys(),
}


def _granularity_of(key) -> str:
    """The granularity enum name encoded in a dumped tune-cache key."""
    return next(k["name"] for k in key if isinstance(k, dict) and "__enum__" in k)


def test_packaged_assets_match_kernel_schema():
    """Every pinned config in a packaged asset must still fit the kernel that will run it."""
    paths = sorted(glob.glob(os.path.join(_TUNE_CONFIGS_ROOT, "*", "*", "*.json")))
    if not paths:
        pytest.skip("no packaged tuning assets to check")

    checked = 0
    for path in paths:
        with open(path) as f:
            entries = json.load(f)["entries"]
        for entry in entries:
            cfg = entry.get("backend_config")
            if cfg is None:  # backend has no internal config to pin
                continue
            owner = (entry["backend"], _granularity_of(entry["key"]))
            schema = _PINNED_SCHEMAS.get(owner)
            assert schema is not None, (
                f"{os.path.basename(path)}: {owner[0]}/{owner[1]} pins a config but no schema is "
                f"registered in _PINNED_SCHEMAS"
            )
            assert cfg.keys() == schema, (
                f"{os.path.basename(path)}: {owner[0]}/{owner[1]} config {sorted(cfg)} no longer "
                f"matches the kernel schema {sorted(schema)}; re-run the offline tuner"
            )
            checked += 1
    print(f"validated {checked} pinned configs across {len(paths)} asset(s)")


def test_stale_blockwise_config_degrades_instead_of_raising():
    """The fallback accepts a fitting config and replaces anything else with the default."""
    good = dict(_DEFAULT_BLOCKWISE_CFG)
    assert _blockwise_cfg_or_default(good) is good
    assert _blockwise_cfg_or_default(None) is _DEFAULT_BLOCKWISE_CFG
    assert _blockwise_cfg_or_default({**good, "OBSOLETE_KNOB": 1}) is _DEFAULT_BLOCKWISE_CFG
    assert (
        _blockwise_cfg_or_default({k: v for k, v in good.items() if k != "CHUNK"}) is _DEFAULT_BLOCKWISE_CFG
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tuned_blockwise_config_is_accepted():
    """A freshly tuned config must pass validation, not be swapped for the default.

    The two are compared by key set, so a Triton release that adds a launch meta-param
    would make every tuned config look stale and quietly disable pinning everywhere.
    """
    dev = "cuda:0"
    M = N = K = 256  # blockwise needs N, K divisible by 128
    arch = torch.cuda.get_device_properties(0).gcnArchName
    fp8 = torch.float8_e4m3fnuz if "gfx942" in arch else torch.float8_e4m3fn
    torch.manual_seed(0)
    a = torch.randn(M, K, device=dev).to(fp8)  # NT: a[M,K], b[N,K]
    b = torch.randn(N, K, device=dev).to(fp8)
    a_scale = torch.rand(M, K // 128, device=dev, dtype=torch.float32) + 0.5
    b_scale = torch.rand(N // 128, K // 128, device=dev, dtype=torch.float32) + 0.5

    cfg = tune_gemm_fp8_blockwise_triton_kernel(a, a_scale, b, b_scale, trans_a=False, trans_b=True)

    assert cfg.keys() == _DEFAULT_BLOCKWISE_CFG.keys(), (
        f"tuned config {sorted(cfg)} does not match the schema {sorted(_DEFAULT_BLOCKWISE_CFG)}; "
        f"pinning would silently degrade to the default config"
    )
    assert _blockwise_cfg_or_default(cfg) is cfg
