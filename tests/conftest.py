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
# pytest_collection_modifyitems. Paths are matched as suffixes of the test file path; an
# optional "::<prefix>" narrows an entry to the test ids starting with that prefix.
_GFX1250_ENABLED_TESTS = (
    # bf16 grouped GEMM through the backend registry: Triton, plus the FlyDSL
    # WMMA/TDM kernels in grouped_gemm_bf16_kernel_gfx1250.py.
    "tests/pytorch/ops/test_grouped_gemm.py",
    # Shape rules and ragged-K coverage for the same FlyDSL bf16 grouped GEMM. Separate
    # from the suite above because most of it is host-only: it checks that the published
    # shape gates and the entry points cannot disagree, which needs no GPU.
    "tests/pytorch/ops/test_grouped_gemm_flydsl_gfx1250_shapes.py",
    # FlyDSL attention at the DeepSeek-V3 MLA head dims (primus_turbo/flydsl/attention/gfx1250_mla).
    "tests/pytorch/ops/test_attention_flydsl_gfx1250_mla.py",
    # Dense FlyDSL attention (primus_turbo/flydsl/attention/gfx1250), bf16 head_dim 128.
    "tests/pytorch/ops/test_attention_flydsl_gfx1250.py",
    # Sparse MLA (primus_turbo/flydsl/attention/gfx1250_sparse_mla); every backend it can pick
    # runs here.
    "tests/pytorch/ops/test_attention.py::test_sparse_mla_op",
    "tests/pytorch/ops/test_attention_flydsl_gfx1250_sparse_mla.py",
)


def _gfx1250_enabled(item) -> bool:
    path = str(getattr(item, "path", item.fspath)).replace(os.sep, "/")
    test_id = item.nodeid.split("::", 1)[-1]
    for enabled in _GFX1250_ENABLED_TESTS:
        suffix, _, prefix = enabled.partition("::")
        if path.endswith(suffix) and test_id.startswith(prefix):
            return True
    return False


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
            if not _gfx1250_enabled(item):
                item.add_marker(skip_gfx1250)
        # Fall through: the enabled items still need the --dist-only / --deterministic
        # filtering below.

    dist_only = config.getoption("--dist-only", False)
    deterministic_only = config.getoption("--deterministic-only", False)

    if dist_only and deterministic_only:
        raise pytest.UsageError("--dist-only and --deterministic-only cannot be enabled together")

    for item in items:
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
