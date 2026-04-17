#!/bin/bash
set -e
echo "=== Phase 1.2: RDMA Proxy Init Test ==="
echo "Host: $(hostname), GPU: gfx$(/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null | head -1)"
pip install nanobind 2>&1 | tail -1

echo "=== ibv_devices ==="
ibv_devices 2>&1

echo "=== Build UCCL-EP ==="
cd /workspace/internode-deepep/uccl/ep
make -f Makefile.rocm_jax clean 2>&1 | tail -1
mkdir -p build
make -f Makefile.rocm_jax -j8 2>&1; echo "MAKE_EXIT=$?"
ls -la ep.abi3.so && echo "BUILD OK" || { echo "BUILD FAILED"; exit 1; }

echo "=== Test: RDMA Proxy Init ==="
python3 -u bench/test_rdma_proxy_init.py 2>&1
echo "=== exit code: $? ==="
