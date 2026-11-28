###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The blockwise FP8 Triton autotune search spaces hold no duplicate configs.

Pure Python -- no GPU required. A duplicate `triton.Config` compiles once but is
benchmarked once per copy, so every duplicate adds a full benchmark to each new
autotune key (M, N, K, EVEN_K) without adding a candidate.
"""

import sys

import pytest

import primus_turbo.pytorch.core.utils as core_utils

_MOD = "primus_turbo.triton.gemm.gemm_fp8_kernel"

# (gfx950, wrapper kwargs) -> expected number of distinct configs
_EXPECTED = [
    (True, {}, 48),  # NT
    (True, {"extra_matrix_instr_nonkdim": [16]}, 72),  # NN
    (True, {"allow_num_stages_3": False}, 32),  # TN
    (False, {}, 16),
    (False, {"extra_matrix_instr_nonkdim": [16]}, 24),
    (False, {"allow_num_stages_3": False}, 16),
]


def _key(config):
    return (tuple(sorted(config.all_kwargs().items())), config.pre_hook)


@pytest.fixture(scope="module")
def kernel_module():
    if _MOD in sys.modules:
        return sys.modules[_MOD]
    # Importing the module builds its autotune wrappers, which asks the device
    # for its arch; pretend to be gfx950 so the import works without a GPU.
    saved = core_utils.get_device_compute_capability
    core_utils.get_device_compute_capability = lambda: (9, 5)
    try:
        __import__(_MOD)
    finally:
        core_utils.get_device_compute_capability = saved
    return sys.modules[_MOD]


@pytest.mark.parametrize("gfx950, kwargs, expected", _EXPECTED)
def test_blockwise_autotune_configs_unique(kernel_module, monkeypatch, gfx950, kwargs, expected):
    monkeypatch.setattr(kernel_module, "is_gfx950", lambda: gfx950)
    configs = kernel_module._get_blockwise_autotune_configs(**kwargs)
    keys = [_key(c) for c in configs]
    assert len(set(keys)) == len(keys), f"{len(keys) - len(set(keys))} duplicate configs"
    assert len(keys) == expected


@pytest.mark.parametrize(
    "wrapper", ["_blockwise_fp8_nt_kernel", "_blockwise_fp8_nn_kernel", "_blockwise_fp8_tn_kernel"]
)
def test_blockwise_autotune_wrappers_unique(kernel_module, wrapper):
    configs = getattr(kernel_module, wrapper).configs
    keys = [_key(c) for c in configs]
    assert len(set(keys)) == len(keys), f"{wrapper}: {len(keys) - len(set(keys))} duplicate configs"
