###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Multi-GPU parallel testing support.

Usage:
    # Single-process mode (runs all tests)
    pytest tests/pytorch/

    # Multi-process mode (need both commands for full coverage)
    pytest tests/pytorch/ -n 8        # single-GPU tests in parallel
    pytest tests/pytorch/ --dist-only # distributed tests (skipped by -n)
"""

import os

import pytest

# Assign each xdist worker to a separate GPU
_worker_id = os.environ.get("PYTEST_XDIST_WORKER")
if _worker_id is not None:
    _num_gpus = 8  # TODO: hardcode.
    _gpu_id = int(_worker_id.replace("gw", "")) % _num_gpus
    os.environ["HIP_VISIBLE_DEVICES"] = str(_gpu_id)


# Suites that run on gfx1250 (MI455X). Everything else is skipped there -- see
# pytest_collection_modifyitems. Paths are matched as suffixes of the test file path.
_GFX1250_ENABLED_TESTS = (
    # bf16 grouped GEMM through the backend registry: Triton, plus the FlyDSL
    # WMMA/TDM kernels in grouped_gemm_bf16_kernel_gfx1250.py.
    "tests/pytorch/ops/test_grouped_gemm.py",
)


def pytest_addoption(parser):
    parser.addoption(
        "--dist-only",
        action="store_true",
        default=False,
        help="Only run multi-GPU distributed tests (MultiProcessTestCase)",
    )
    parser.addoption(
        "--deterministic-only",
        action="store_true",
        default=False,
        help="Only run deterministic tests (marked with @pytest.mark.deterministic)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "multigpu: mark test as requiring multiple GPUs")
    config.addinivalue_line("markers", "deterministic: mark test as deterministic-only suite")
    config.addinivalue_line(
        "markers",
        "gfx950: requires gfx950 (MI350X/MI355X); not exercised on CI gfx942 hosts — "
        "run via tools/run_gfx950_tests.sh on an MI355X box",
    )


def _skipif_reason(mark) -> str:
    """Best-effort extract of a skipif reason string."""
    if mark.kwargs.get("reason"):
        return str(mark.kwargs["reason"])
    if len(mark.args) >= 2 and isinstance(mark.args[1], str):
        return mark.args[1]
    return ""


def _looks_like_gfx950_skip(reason: str) -> bool:
    text = reason.lower()
    return any(
        needle in text
        for needle in (
            "gfx950",
            "mi355",
            "mi350",
            "mxfp8",
            "mxfp4",
            "fused_mega_moe",
            "mega moe",
        )
    )


def pytest_collection_modifyitems(config, items):
    """Skip distributed tests in parallel mode, skip single-GPU tests in --dist-only mode."""
    # TODO(ruibin): gfx1250 is not yet fully supported by the Primus-Turbo.
    is_gfx1250 = False
    try:
        from primus_turbo.pytorch.core.utils import is_gfx1250 as is_gfx1250_pytorch

        is_gfx1250 = is_gfx1250_pytorch()
    except ImportError:
        from primus_turbo.jax.core.utils import is_gfx1250 as is_gfx1250_jax

        is_gfx1250 = is_gfx1250_jax()

    if is_gfx1250:
        # The default on gfx1250 is still "skip": most ops have no port yet, and a
        # blanket skip is honest about that. _GFX1250_ENABLED_TESTS is the opt-in list
        # of suites that have been brought up and measured on the part, so a
        # regression in one of those shows up as a failure instead of disappearing
        # behind the skip. Match on a path suffix, not the bare file name -- there is
        # a second test_grouped_gemm.py under tests/jax/.
        skip_gfx1250 = pytest.mark.skip(reason="Not yet supported on gfx1250")
        for item in items:
            path = str(getattr(item, "path", item.fspath)).replace(os.sep, "/")
            if not any(path.endswith(enabled) for enabled in _GFX1250_ENABLED_TESTS):
                item.add_marker(skip_gfx1250)
        # Fall through: the enabled items still need the --dist-only / --deterministic
        # filtering below.

    dist_only = config.getoption("--dist-only", False)
    deterministic_only = config.getoption("--deterministic-only", False)

    if dist_only and deterministic_only:
        raise pytest.UsageError("--dist-only and --deterministic-only cannot be enabled together")

    for item in items:
        # Promote decorator-level skipif reasons that name gfx950/MXFP* into the
        # gfx950 marker so `pytest -m gfx950` / tools/run_gfx950_tests.sh can
        # select them. Prefer explicit @pytest.mark.gfx950 / module pytestmark
        # for new tests; this only backfills existing skipifs.
        if item.get_closest_marker("gfx950") is None:
            for mark in item.iter_markers("skipif"):
                if _looks_like_gfx950_skip(_skipif_reason(mark)):
                    item.add_marker(pytest.mark.gfx950)
                    break

        is_dist = _is_distributed_test(item)
        is_deterministic = item.get_closest_marker("deterministic") is not None
        if _worker_id and is_dist:
            item.add_marker(pytest.mark.skip(reason="Distributed test, run with --dist-only"))
        elif dist_only and not is_dist:
            item.add_marker(pytest.mark.skip(reason="Single-GPU test, run with -n 8"))
        elif deterministic_only and not is_deterministic:
            item.add_marker(
                pytest.mark.skip(reason="Non-deterministic test, run without --deterministic-only")
            )
        elif is_deterministic and not deterministic_only:
            item.add_marker(pytest.mark.skip(reason="Deterministic test, run with --deterministic-only"))


def _is_distributed_test(item):
    """Check if test requires multiple GPUs."""
    if item.get_closest_marker("multigpu"):
        return True
    if hasattr(item, "cls") and item.cls is not None:
        return any(
            cls.__name__
            in [
                "MultiProcessTestCase",
                "MultiProcContinuousTest",
                "JaxMultiProcessTestCase",
            ]
            for cls in item.cls.__mro__
        )
    return False
