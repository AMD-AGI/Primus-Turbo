#!/bin/bash

# 1. Set the ROCm paths
# Use hipcc as the compiler wrapper for both C++ and C
# export CXX=/opt/rocm-6.4.1/bin/hipcc
# export CC=/opt/rocm-6.4.1/bin/hipcc
# Tell the build system to use the Cray C++ compiler wrapper
export CXX=CC
# Tell the build system to use the Cray C compiler wrapper
export CC=cc
export HIP_HOME=/opt/rocm-6.4.1

# 2. Specify the GPU architecture
export GPU_ARCHS="gfx90a"

# 3. It's a good practice to clean up previous failed attempts
pip3 cache purge
rm -rf build dist *.egg-info

# 4. Install requirements
# Using --no-build-isolation is crucial for DeepSpeed/FlashAttn on Frontier
pip3 install --no-build-isolation -r requirements.txt


export CXX=hipcc

# Also set the C compiler to hipcc for consistency
export CC=hipcc

# 5. Install the main package
pip3 install --no-build-isolation -e . -v

echo "Setup Complete. Environment is ready."

python -c "import primus_turbo.pytorch as turbo"