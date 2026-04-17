#!/bin/bash
set -e

echo "=== Environment: $(hostname), gfx$(/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null | head -1) ==="
pip install nanobind 2>&1 | tail -1

echo "=== Rebuild UCCL-EP ==="
cd /workspace/internode-deepep/uccl/ep
make -f Makefile.rocm_jax clean 2>&1 | tail -1
make -f Makefile.rocm_jax -j8 2>&1 | grep -E "^(make|error|/opt)" | tail -5
ls -la ep.abi3.so 2>/dev/null && echo "BUILD OK" || { echo "BUILD FAILED"; exit 1; }

echo "=== Test: 8-GPU intranode with destroy() fix ==="
python3 -u bench/test_jax_intranode.py 2>&1
echo "=== Test exit code: $? ==="
