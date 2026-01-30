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
        return any(cls.__name__ == "MultiProcessTestCase" for cls in item.cls.__mro__)
    return False
