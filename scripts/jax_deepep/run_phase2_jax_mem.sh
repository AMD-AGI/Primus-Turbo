#!/bin/bash
set -e
echo "=== Phase 2.1: JAX Memory + Phase 2.2: FP8 ==="
echo "Host: $(hostname), GPU: gfx$(/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null | head -1)"
pip install nanobind 2>&1 | tail -1

echo "=== Build UCCL-EP (with FP8 fix) ==="
cd /workspace/internode-deepep/uccl/ep
make -f Makefile.rocm_jax clean 2>&1 | tail -1
mkdir -p build
make -f Makefile.rocm_jax -j8 2>&1 | tail -5
ls -la ep.abi3.so && echo "BUILD OK" || { echo "BUILD FAILED"; exit 1; }

echo ""
echo "=== Test 1: JAX Memory Integration ==="
python3 -u bench/test_jax_memory.py 2>&1
echo "=== JAX Memory exit code: $? ==="
