#!/bin/bash
set -e

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null)"
rocminfo 2>/dev/null | grep -E "Name:|Marketing Name:|gfx" | head -10

echo "=== GPU Architecture ==="
/opt/rocm/bin/rocm_agent_enumerator 2>/dev/null

echo "=== ibv_devices ==="
ibv_devices 2>/dev/null

echo "=== Install nanobind ==="
pip install nanobind 2>&1 | tail -2

echo "=== Run POC test with debug ==="
cd /workspace/internode-deepep/uccl/ep
export AMD_LOG_LEVEL=1
export HIP_VISIBLE_DEVICES=0,1
python3 -u bench/test_jax_intranode.py 2>&1 || {
    echo "Test failed with exit code $?"
    echo "=== Trying with just 2 GPUs ==="
}

echo "=== Done ==="
