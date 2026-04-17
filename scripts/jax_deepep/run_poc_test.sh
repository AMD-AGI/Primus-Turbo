#!/bin/bash
set -e

echo "=== Step 0: Verify environment ==="
echo "Host: $(hostname)"
echo "Python: $(python3 --version)"
echo "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
/opt/rocm/bin/hipcc --version 2>&1 | head -2

echo "=== Step 0.1: Check ibv_devices ==="
ibv_devices 2>/dev/null && echo "ibv_devices OK" || echo "ibv_devices FAILED or not found"

echo "=== Step 0.2: Check GPU ==="
rocm-smi --showid 2>/dev/null | head -5

echo "=== Step 0.3: Check nanobind ==="
python3 -c "import nanobind; print('nanobind:', nanobind.__version__)" 2>/dev/null || {
    echo "nanobind not found, installing..."
    pip install nanobind 2>&1 | tail -3
    python3 -c "import nanobind; print('nanobind:', nanobind.__version__)"
}

echo "=== Step 1: Build UCCL-EP ==="
cd /workspace/internode-deepep/uccl/ep
make -f Makefile.rocm_jax clean 2>&1 | tail -2
echo "Building..."
make -f Makefile.rocm_jax -j8 2>&1
echo "Build result: $?"
ls -la ep*.so 2>/dev/null || echo "BUILD FAILED - no .so produced"

echo "=== Step 2: Run POC intranode test ==="
cd /workspace/internode-deepep/uccl/ep
python3 bench/test_jax_intranode.py 2>&1

echo "=== ALL DONE ==="
