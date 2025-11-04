ARG BASE_IMAGE=docker.io/rocm/primus:v25.9_gfx942

FROM ${BASE_IMAGE}

# Non-interactive APT
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# ───── install dependenies ─────
RUN apt-get update && \
        apt install -y  rdma-core libibverbs-dev ibverbs-utils
\
# ───── Clean up apt cache ─────
    rm -rf /var/lib/apt/lists/*

# ───── install rocSHMEM ─────
RUN git clone https://github.com/ROCm/rocSHMEM.git -b develop


ENV PATH="/opt/ompi/bin:/opt/ompi/sbin:${PATH}" \
    LD_LIBRARY_PATH="/opt/ompi/lib:${LD_LIBRARY_PATH}" \
    C_INCLUDE_PATH="/opt/ompi/include:${C_INCLUDE_PATH}" \
    CPLUS_INCLUDE_PATH="/opt/ompi/include:${CPLUS_INCLUDE_PATH}" \
    PKG_CONFIG_PATH="/opt/ompi/lib/pkgconfig:${PKG_CONFIG_PATH}" \
    MANPATH="/opt/ompi/share/man:${MANPATH}" \
    CMAKE_C_COMPILER="/opt/ompi/bin/mpicc" \
    CMAKE_CXX_COMPILER="/opt/ompi/bin/mpicxx"

RUN cd rocrocSHMEM && \
        mkdir build && cd build && \
            INSTALL_PREFIX=/opt/rocshmem MPI_ROOT=/opt/ompi UCX_ROOT=/opt/ucx \
                ../scripts/build_configs/gda_mlx5 -DGDA_BNXT=ON -DUSE_IPC=ON


WORKDIR /workspace

CMD ["bash"]
