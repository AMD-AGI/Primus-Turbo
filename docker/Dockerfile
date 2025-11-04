ARG BASE_IMAGE=docker.io/rocm/primus:v25.9_gfx942

FROM ${BASE_IMAGE}

# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# ───── install dependenies ─────
RUN apt-get update && \
        apt install -y  rdma-core libibverbs-dev ibverbs-utils

# ───── patch llvm for rocm7.0 ─────
COPY 7.0-deepep.patch /workspace

RUN git clone https://github.com/ROCm/llvm-project.git
RUN cd llvm-project && \
    git checkout release/rocm-rel-7.0 && \
    git am /workspace/7.0-deepep.patch && \
    mkdir -p /opt/llvm-7.0-patch && mkdir build && cd build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm-7.0-patch \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_ENABLE_RUNTIMES=compiler-rt \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
    -DCLANG_DEFAULT_LINKER=lld \
    -DLLVM_PARALLEL_LINK_JOBS=4 \
    ../llvm && \
    ninja && ninja install

# ───── install rocSHMEM ─────
RUN git clone https://github.com/ROCm/rocSHMEM.git -b develop

ENV HIP_CLANG_PATH=/opt/llvm-7.0-patch/bin

RUN cd rocSHMEM && \
        mkdir build && cd build && \
            INSTALL_PREFIX=/opt/rocshmem \
            MPI_ROOT=/opt/ompi UCX_ROOT=/opt/ucx \
                ../scripts/build_configs/gda_mlx5 -DGDA_BNXT=ON -DUSE_IPC=ON


WORKDIR /workspace

CMD ["bash"]
