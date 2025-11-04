#!/bin/bash
set -e
set -x

BASE_IMAGE=${BASE_IMAGE:-rocm/primus:v25.9_gfx942}

IMAGE_NAME=${IMAGE_NAME:-${BASE_IMAGE}_patch}
WHEEL_DIR=${WHEEL_DIR:-"dist"}

mkdir -p "${WHEEL_DIR}"

build_turbo() {
 local WHEEL_DIR=$1

 pip install -r requirements.txt
 export MPI_HOME=/opt/ompi
 export UCX_HOME=/opt/ucx
 export ROCM_HOME=/opt/rocm-7.0-patch
 export ROCSHMEM_HOME=/opt/rocshmem

 export HIP_CLANG_PATH=/opt/llvm-7.0-patch/bin

 python3 setup.py bdist_wheel

 cp dist/*.whl  /workspace/"${WHEEL_DIR}"
}



podman build --build-arg BASE_IMAGE="${BASE_IMAGE}" \
             -t "${IMAGE_NAME}" -f docker/Dockerfile

podman run --rm  \
  --ipc=host \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/infiniband \
  --cap-add=SYS_PTRACE \
  --cap-add=CAP_SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --group-add video \
  --privileged \
  -v "$PWD":/workspace \
  -e WHEEL_DIR="${WHEEL_DIR}" \
  -e FUNCTION_DEF="$(declare -f build_turbo)" \
  "$IMAGE_NAME" /bin/bash -c '
    set -euo pipefail

    eval "$FUNCTION_DEF"

    build_turbo $WHEEL_DIR
    echo "build wheel finish!"
    '
