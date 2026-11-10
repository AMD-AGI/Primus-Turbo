###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared gates for the DeepSeek-V4 attention kernel tests.

* The V4 kernels and their numerics are validated on MI355X only, so every
  test here is skipped on any other accelerator (or a CPU-only host). Set
  ``PRIMUS_V4_UT_ALLOW_NON_MI355X=1`` to bypass the gate.
* Release-tier shapes (``pytest.mark.slow``: ``head_dim=512``, ``K_topk`` up
  to 1024) are skipped unless ``PRIMUS_TURBO_RUN_SLOW=1``.
* The eager references at release dimensions materialise intermediates of up
  to ~128 GiB; the CUDA cache is emptied after every test so the caching
  allocator does not carry them into the next parametrisation.
* torch >= 2.14 routes outer-product ``bmm`` (inner dim 1) to a Triton
  override in ``torch._native`` that faults on ROCm inside the backward of the
  eager references (H=128 CSA hits it); the override is disabled per test so
  the references run on the ATen ``bmm``.
"""

import functools
import os

import pytest
import torch

try:
    from torch._native import registry as _torch_native_registry
except ImportError:  # torch < 2.14
    _torch_native_registry = None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: DeepSeek-V4 release-tier shape gate; set PRIMUS_TURBO_RUN_SLOW=1 to run",
    )


@functools.lru_cache(maxsize=1)
def _is_mi355x() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0) or ""
    except Exception:  # pragma: no cover - defensive
        return False
    return "mi355x" in name.lower()


@pytest.fixture(autouse=True)
def _require_mi355x():
    if os.environ.get("PRIMUS_V4_UT_ALLOW_NON_MI355X", "0") != "0":
        return
    if not _is_mi355x():
        pytest.skip("DeepSeek-V4 kernel tests require an MI355X GPU")


@pytest.fixture(autouse=True)
def _skip_release_tier(request):
    if request.node.get_closest_marker("slow") and os.environ.get("PRIMUS_TURBO_RUN_SLOW", "0") == "0":
        pytest.skip("release-tier shape (set PRIMUS_TURBO_RUN_SLOW=1 to run)")


@pytest.fixture(autouse=True)
def _aten_bmm_for_eager_references():
    if _torch_native_registry is None:
        yield
        return
    _torch_native_registry.deregister_op_overrides(disable_op_symbols="bmm")
    try:
        yield
    finally:
        _torch_native_registry.reenable_op_overrides(enable_op_symbols="bmm")


@pytest.fixture(autouse=True)
def _release_gpu_cache_per_test():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
