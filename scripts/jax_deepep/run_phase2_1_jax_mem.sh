#!/usr/bin/env bash
# Phase 2.1: build UCCL-EP (ROCm JAX makefile) and run JAX GPU memory integration test.
set -euo pipefail

ROOT="/workspace/internode-deepep"
EP="${ROOT}/uccl/ep"

echo "=== Phase 2.1 JAX memory + UCCL-EP ==="
echo "Date: $(date -Is)"
echo "Hostname: $(hostname)"

export PYTHONUNBUFFERED=1

pip install -q nanobind

cd "${EP}"
make -f Makefile.rocm_jax clean
make -f Makefile.rocm_jax -j8

ls -la "${EP}"/ep*.so || true

cd "${ROOT}"
exec python3 uccl/ep/bench/test_jax_memory.py
