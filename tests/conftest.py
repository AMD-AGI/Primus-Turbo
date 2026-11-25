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
import re

import pytest

import primus_turbo.common.envs as envs

# Assign each xdist worker to a separate GPU
_worker_id = os.environ.get("PYTEST_XDIST_WORKER")
if _worker_id is not None:
    _num_gpus = 8  # TODO: hardcode.
    _gpu_id = int(_worker_id.replace("gw", "")) % _num_gpus
    os.environ["HIP_VISIBLE_DEVICES"] = str(_gpu_id)


_TESTS_DIR = os.path.dirname(os.path.realpath(__file__))


def _gfx1250_enabled(item) -> bool:
    """Whether a test runs on gfx1250 (MI455X). Everything else is skipped there -- see
    pytest_collection_modifyitems. A test opts in by its name or by what it says:

    - its file (or a directory above it) carries the architecture tag as a name segment, the way
      the kernels do: ``test_mega_bf16_gfx1250_gemm.py``, ``.../gfx1250/test_x.py``. These are the
      suites of the gfx1250-only kernels;
    - it carries ``pytest.mark.gfx1250`` (a suite, or one test, in a file every architecture
      shares: ``pytestmark = pytest.mark.gfx1250`` or ``@pytest.mark.gfx1250``).

    So a new gfx1250 test needs no edit here.
    """
    # Relative to this file's directory: not to the checkout (one under some .../gfx1250/ must enable
    # nothing) and not to pytest's rootdir (--rootdir or -c elsewhere must not disable the suites).
    # A test that is not below this directory (after resolving links) has no path to judge.
    path = os.path.relpath(os.path.realpath(str(getattr(item, "path", item.fspath))), _TESTS_DIR)
    parts = [] if path.startswith(os.pardir) else path.split(os.sep)
    tagged = any("gfx1250" in re.split(r"[_.]", part) for part in parts)
    return tagged or item.get_closest_marker("gfx1250") is not None


@pytest.fixture(autouse=True)
def _fresh_envs_cache():
    """``primus_turbo.common.envs`` caches each variable at its first read, so a test that sets one
    would otherwise see the value an earlier test left in the cache."""
    envs.reset_cache()
    yield
    envs.reset_cache()


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
        "markers", "gfx1250: runs on gfx1250 (MI455X) too, where unmarked tests without the tag are skipped"
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
        # blanket skip is honest about that. The suites that have been brought up and
        # measured on the part opt in (see _gfx1250_enabled), so a regression in one of
        # those shows up as a failure instead of disappearing behind the skip.
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
