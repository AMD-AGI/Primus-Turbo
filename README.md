# Primus-Turbo
**Primus-Turbo** is a high-performance core library for accelerating large-scale model training, inference, and reinforcement learning on AMD GPUs. Built and optimized for the AMD ROCm platform, it targets the full performance stack in Transformer-based models — covering core compute operators (GEMM, Attention, GroupedGEMM), communication primitives, optimizer modules, low-precision (FP8), and compute-communication overlap kernels.

Primus-Turbo is designed to unlock the full potential of AMD GPUs for large-model workloads, providing a comprehensive acceleration foundation for next-generation AI systems.


## 🚀 What's New
...


## 📦 Install & Deployment

### 1. Docker (Recommended)
Use the pre-built AMD ROCm image:
```
rocm/megatron-lm:v25.5_py310
```

### 2. Install from Source
#### Clone Repository
```
git clone https://github.com/AMD-AIG-AIMA/Primus-Turbo.git --recursive
cd Primus-Turbo
```
#### User Install
```
pip3 install -r requirements.txt
pip3 install .
```

#### Developer Install (editable mode)
```
pip3 install -r requirements.txt
pip3 install -e . -v
```

### 3. Build & Deploy Wheel
```
pip3 install -r requirements.txt
python3 -m build --wheel --no-isolation
pip3 install --extra-index-url https://test.pypi.org/simple ./primus_turbo-XXX.whl
```

## 💡 Quick Example
```
...
```


## 📊 Performance
See [Benchmarks](./benchmark/README.md) for detailed performance results and comparisons.



## 📜 License

Primus-Turbo is licensed under the MIT License.

© 2025 Advanced Micro Devices, Inc. All rights reserved.
