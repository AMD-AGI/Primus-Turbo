#!/bin/bash
set -e

# Determine node index from SLURM
NODE_IDX=${SLURM_NODEID:-0}
TOTAL_NODES=${SLURM_NNODES:-2}

echo "=== Phase 1.4: Internode Proxy RDMA Connection Test ==="
echo "Host: $(hostname), NODE_IDX=${NODE_IDX}, TOTAL_NODES=${TOTAL_NODES}"
echo "SLURM_PROCID=${SLURM_PROCID}, SLURM_NODEID=${SLURM_NODEID}"

# Clean up old meta files on node 0 only
if [ "$NODE_IDX" = "0" ]; then
    rm -rf /workspace/internode-deepep/_internode_meta
fi

pip install nanobind 2>&1 | tail -1

echo "=== ibv_devices ==="
ibv_devices 2>&1

echo "=== Build UCCL-EP ==="
cd /workspace/internode-deepep/uccl/ep
if [ "$NODE_IDX" = "0" ]; then
    make -f Makefile.rocm_jax clean 2>&1 | tail -1
    mkdir -p build
    make -f Makefile.rocm_jax -j8 2>&1
    echo "MAKE_EXIT=$?"
    ls -la ep.abi3.so && echo "BUILD OK" || { echo "BUILD FAILED"; exit 1; }
    # Signal build complete
    touch /workspace/internode-deepep/_build_done
else
    echo "Waiting for node 0 to finish build..."
    while [ ! -f /workspace/internode-deepep/_build_done ]; do sleep 1; done
    echo "Build available"
    ls -la ep.abi3.so
fi

echo "=== Test: Internode Proxy Connection ==="
python3 -u bench/test_internode_proxy.py \
    --node-idx ${NODE_IDX} \
    --num-nodes ${TOTAL_NODES} \
    --num-gpus 8 2>&1
echo "=== exit code: $? ==="
