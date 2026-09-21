#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Run gfx950-only PyTorch tests on an MI350X / MI355X host.
#
# CI currently covers gfx942 (MI300X) only. Cases that require gfx950 are
# marked with @pytest.mark.gfx950 (or auto-promoted from matching skipif
# reasons in tests/conftest.py). PR owners who touch Mega MoE / MXFP* /
# FlyDSL / Gluon / HipKittens should run this script before merge.
#
# Usage:
#   ./tests/run_gfx950_tests.sh              # all three CI-equivalent suites
#   ./tests/run_gfx950_tests.sh single       # single-GPU parallel only
#   ./tests/run_gfx950_tests.sh deterministic
#   ./tests/run_gfx950_tests.sh dist
#   EXTRA_PYTEST_ARGS="-k mega_moe" ./tests/run_gfx950_tests.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-all}"
TIMEOUT_ARGS=(--timeout=600 --timeout-method=thread)
# Optional extra flags from the environment (word-split intentionally).
# shellcheck disable=SC2206
EXTRA_ARGS=( ${EXTRA_PYTEST_ARGS:-} )

case "${MODE}" in
  -h|--help|help)
    sed -n '2,25p' "$0"
    exit 0
    ;;
esac

check_gfx950() {
  python3 - <<'PY'
import sys

try:
    from primus_turbo.pytorch.core.utils import is_gfx950
except ImportError:
    print(
        "ERROR: cannot import primus_turbo.pytorch.core.utils.is_gfx950. "
        "Install the package first (e.g. pip install -e '.[pytorch]').",
        file=sys.stderr,
    )
    sys.exit(2)

if not is_gfx950():
    print(
        "ERROR: this host is not gfx950 (MI350X/MI355X). "
        "Refuse to run -m gfx950 on the wrong GPU.",
        file=sys.stderr,
    )
    sys.exit(1)

print("OK: gfx950 detected; running tests marked gfx950.")
PY
}

run_single() {
  echo "==> single-GPU gfx950 tests (parallel)"
  pytest -v tests/pytorch -n 8 -m gfx950 \
    "${TIMEOUT_ARGS[@]}" --max-worker-restart=8 --log-cli-level=INFO \
    "${EXTRA_ARGS[@]}"
}

run_deterministic() {
  echo "==> deterministic gfx950 tests (parallel)"
  pytest -v tests/pytorch -n 8 -m "gfx950 and deterministic" --deterministic-only \
    "${TIMEOUT_ARGS[@]}" --max-worker-restart=8 --log-cli-level=INFO \
    "${EXTRA_ARGS[@]}"
}

run_dist() {
  echo "==> multi-GPU gfx950 tests"
  pytest -v tests/pytorch --dist-only -m gfx950 \
    "${TIMEOUT_ARGS[@]}" --log-cli-level=INFO \
    "${EXTRA_ARGS[@]}"
}

check_gfx950

case "${MODE}" in
  all)
    run_single
    run_deterministic
    run_dist
    ;;
  single)
    run_single
    ;;
  deterministic)
    run_deterministic
    ;;
  dist)
    run_dist
    ;;
  *)
    echo "Unknown mode: ${MODE} (expected: all|single|deterministic|dist)" >&2
    exit 2
    ;;
esac

echo "Done: gfx950 suite (${MODE}) finished."
