#!/bin/bash
# 2-node MaxText training on deepep-a66 with host ionic driver mounts
# Tests that IB/RDMA works after mounting host's ionic userspace driver
set -e

IONIC_MOUNTS=(
  -v /sys/class/infiniband:/sys/class/infiniband
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185:/usr/lib/x86_64-linux-gnu/libionic.so.1.1.54.0-185
  -v /usr/lib/x86_64-linux-gnu/libionic.so.1:/usr/lib/x86_64-linux-gnu/libionic.so.1
  -v /usr/lib/x86_64-linux-gnu/libionic.so:/usr/lib/x86_64-linux-gnu/libionic.so
  -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
  -v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver
)

NNODES=2
NODE_RANK=$((SLURM_NODEID))
COORDINATOR_IP=$(scontrol show hostname "$SLURM_NODELIST" | head -1)
COORDINATOR_PORT=29500

echo "=== Node $(hostname), rank=$NODE_RANK/$NNODES, coordinator=$COORDINATOR_IP:$COORDINATOR_PORT ==="

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --device=/dev/infiniband \
  --ipc=host --network=host --privileged \
  "${IONIC_MOUNTS[@]}" \
  -e NNODES=$NNODES \
  -e NODE_RANK=$NODE_RANK \
  -v /mnt/shared/llying/internode-deepep:/workspace/internode-deepep \
  rocm/jax-training:maxtext-v26.2 \
  bash -c '
    set -e
    echo "=== ibv_devices check ==="
    ibv_devices 2>&1

    export RCCL_MSCCL_ENABLE=0
    export NCCL_MIN_NCHANNELS=8
    export NCCL_NCHANNELS_PER_NET_PEER=8
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=NET

    cd /workspace/maxtext
    echo "=== MaxText dir structure ==="
    ls src/MaxText/train.py 2>/dev/null && echo "train.py found" || { echo "train.py NOT found at src/MaxText/"; ls -la; exit 1; }

    mkdir -p /tmp/maxtext_output

    echo "=== Starting 2-node MaxText training ==="
    python3 -u src/MaxText/train.py src/MaxText/configs/base.yml \
      run_name=ib_test_deepep_a66 \
      hardware=gpu \
      steps=5 \
      per_device_batch_size=1 \
      max_target_length=2048 \
      model_name=llama2-7b \
      dataset_type=synthetic \
      use_checkpointing=false \
      enable_profiler=false \
      enable_single_controller=true \
      jax_coordinator_address='"$COORDINATOR_IP:$COORDINATOR_PORT"' \
      num_slices=1 \
      base_num_decoder_layers=4 \
      base_output_directory=/tmp/maxtext_output \
      2>&1
  '
