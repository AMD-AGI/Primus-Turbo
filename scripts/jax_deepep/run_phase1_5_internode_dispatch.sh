#!/bin/bash
set -e

NODE_IDX=${SLURM_NODEID:-0}
TOTAL_NODES=${SLURM_NNODES:-2}

echo "=== Phase 1.5: Internode Dispatch/Combine (Single-Process Mode) ==="
echo "Host: $(hostname), NODE_IDX=${NODE_IDX}, TOTAL_NODES=${TOTAL_NODES}"

pip install nanobind 2>&1 | tail -1

echo "=== Build UCCL-EP ==="
cd /workspace/internode-deepep/uccl/ep
if [ "$NODE_IDX" = "0" ]; then
    make -f Makefile.rocm_jax clean 2>&1 | tail -1
    mkdir -p build
    make -f Makefile.rocm_jax -j8 2>&1 | tail -5
    echo "MAKE_EXIT=$?"
    ls -la ep.abi3.so && echo "BUILD OK" || { echo "BUILD FAILED"; exit 1; }
    touch /workspace/internode-deepep/_build_done_p15
else
    echo "Waiting for node 0 to finish build..."
    while [ ! -f /workspace/internode-deepep/_build_done_p15 ]; do sleep 1; done
    echo "Build available"
fi

echo "=== ibv_devices ==="
ibv_devices 2>&1

echo "=== Test: Internode Dispatch/Combine ==="
python3 -u bench/test_internode_dispatch.py \
    --node-idx ${NODE_IDX} \
    --num-nodes ${TOTAL_NODES} \
    --num-gpus 8 2>&1
echo "=== exit code: $? ==="
