#!/bin/bash
# Submit Phase 1.5: 2-node internode dispatch/combine (single-process mode)
PARTITION=${1:-deepep-a66}
WORKSPACE=/mnt/shared/llying/internode-deepep

# Clean up from previous runs
rm -f ${WORKSPACE}/_build_done_p15
rm -rf ${WORKSPACE}/_internode_meta_p15

IONIC_MOUNTS="\
-v /sys/class/infiniband:/sys/class/infiniband \
-v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185 \
-v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1 \
-v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so \
-v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so \
-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver"

echo "Submitting 2-node Phase 1.5 job on partition: ${PARTITION}"

srun -p ${PARTITION} -N 2 --exclusive --gpus-per-task=8 --mem=0 \
  --ntasks-per-node=1 -t 00:30:00 -J internode_p15 \
  bash -c "docker run --rm \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --ipc=host --network=host --privileged \
    ${IONIC_MOUNTS} \
    -v ${WORKSPACE}:${WORKSPACE} \
    -v ${WORKSPACE}:/workspace/internode-deepep \
    -e SLURM_NODEID=\${SLURM_NODEID} \
    -e SLURM_NNODES=\${SLURM_NNODES} \
    -e SLURM_PROCID=\${SLURM_PROCID} \
    rocm/jax-training:maxtext-v26.2 \
    bash /workspace/internode-deepep/run_phase1_5_internode_dispatch.sh" 2>&1
