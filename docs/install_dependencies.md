# Install Primus-Turbo dependencies

## Install rocSHMEM with IBGDA support

The internode api of Primus-Turbo DeepEP depends on rocSHMEM with IBGDA support. Please follow the steps below to install rocSHMEM with IBGDA.

### Install procedure

rocSHMEM IBGDA now is only available at develop branch, installation package will be released in the future. Now you need to follow the [rocSHMEM-building-and-installation](https://github.com/ROCm/rocSHMEM?tab=readme-ov-file#building-and-installation) document to build and install rocSHMEM.

rocSHMEM supports three types of NIC drivers: `bnxt`,`mlx5` and `ionic`. You should check cluster network card configuration and build rocSHMEM with corresponding compile script.

> **Note: rocSHMEM is under development, no guarantee of full compatibility and performance for bnxt,mlx5 and ionic NIC driver.**

For example, the following shows that NIC used by the cluster is ConnectX-7.
```bash
useocpm2m-401-028:~> lspci | grep Mellanox
0c:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
1f:00.0 Ethernet controller: Mellanox Technologies MT2892 Family [ConnectX-6 Dx]
2a:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
41:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
58:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
86:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
9a:00.0 Ethernet controller: Mellanox Technologies MT2892 Family [ConnectX-6 Dx]
a5:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
bd:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
d5:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
```
You should select rocSHMEM build script which ConnectX-7 driver is supported.
```bash
mkdir build
cd build
../scripts/build_configs/gda_mlx5
```

### Post-installation configuration
After you built and installed rocSHMEM from source, set the following environment variables in your shell configuration:

```bash
export ROCSHMEM_DIR=/path/to/your/dir/to/install # Use for DeepEP installation
export MPI_DIR=/path/to/your/dir/to/install # The dependency of rocSMEM used for DeepEP installation
```
