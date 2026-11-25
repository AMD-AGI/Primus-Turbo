###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The TRITON RL attention's bitwise guarantees under other Triton releases.

Bit equality across schedules (BLOCK_M, num_warps, split / fused, paged / contiguous) is a property
of the code a Triton release generates, not only of the kernel source: integrators ship their own
Triton. Each release in PRIMUS_TURBO_RL_TRITON_VERSIONS (default "3.7,3.8") is installed once into
a cache directory (pip, no dependencies; PRIMUS_TURBO_TRITON_CACHE, default
~/.cache/primus_turbo/triton) and the bitwise tests of test_rl_attention.py run against it in a
child process, since one process cannot load two Triton releases. A release that cannot be
installed (no network) is skipped; the running release is skipped too (the suite itself covers it).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import triton

VERSIONS = [v.strip() for v in os.environ.get("PRIMUS_TURBO_RL_TRITON_VERSIONS", "3.7,3.8").split(",") if v.strip()]
REPO = Path(__file__).resolve().parents[3]
SUITE = REPO / "tests" / "pytorch" / "ops" / "test_rl_attention.py"
CACHE = Path(os.environ.get("PRIMUS_TURBO_TRITON_CACHE", Path.home() / ".cache" / "primus_turbo" / "triton"))
# The tests that assert bit equality (modes, launch parameters, layouts, batches, rollout ==
# training, graph replay), split in groups that each fit a CI per-test timeout.
GROUPS = {
    "launch": "test_launch_invariance or test_causal_intervals_equal_builtin",
    "modes": "test_paged_split_fused_contiguous_bitwise or test_visibility_paged_split_fused_bitwise",
    "rollout": "rollout_matches_training",
    "batch": "batch_invariance or test_decode_graph_replay",
}


def _install(version: str) -> Path:
    target = CACHE / version
    if not (target / ".installed").exists():
        res = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", str(target), f"triton=={version}.*"],
            capture_output=True,
            text=True,
        )
        if res.returncode:
            pytest.skip(f"cannot install triton {version}: {res.stderr.strip()[-300:]}")
        (target / ".installed").write_text(version)
    return target


@pytest.mark.parametrize("group", list(GROUPS))
@pytest.mark.parametrize("version", VERSIONS)
def test_bitwise_guarantees_under_triton(version, group):
    if triton.__version__.startswith(version + "."):
        pytest.skip(f"triton {version} is the running release")
    target = _install(version)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(p for p in (str(target), os.environ.get("PYTHONPATH", "")) if p),
        TRITON_CACHE_DIR=str(target / ".kernel_cache"),
    )
    probe = subprocess.run(
        [sys.executable, "-c", "import triton; print(triton.__version__)"], env=env, capture_output=True, text=True
    )
    assert probe.stdout.strip().startswith(version + "."), f"child loads triton {probe.stdout.strip()}"
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(SUITE), "-k", GROUPS[group]],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    tail = "\n".join(res.stdout.strip().splitlines()[-40:])
    assert res.returncode == 0, f"triton {version}, {group} tests:\n{tail}"
