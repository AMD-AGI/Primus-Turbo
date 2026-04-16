# Primus-Turbo DeepEP 双路径实现分析报告

> 分析日期：2026-04-13
> 代码库：Primus-Turbo (`primus_turbo/pytorch/deep_ep`)

---

## 一、概述

在 Primus-Turbo 中，PyTorch 的 DeepEP（Expert-Parallel 通信）存在**两条实现路径**：

| 路径 | 名称 | Python 包名 | Backend 标识 | 核心通信机制 |
|------|------|------------|-------------|------------|
| **路径一** | Primus-Turbo 内置 DeepEP | `primus_turbo.pytorch.deep_ep` (别名 `turbo_ep`) | `BackendType.TURBO` | NVLink (intranode) + rocSHMEM RDMA (internode) |
| **路径二** | 外部 UCCL deep_ep 包 | `deep_ep` (独立 pip 包) | `BackendType.DEEP_EP` | UCCL EP 库提供的通信原语 |

两条路径在 MoE dispatch/combine 层面提供了**相同的 Python API 接口**（`Buffer.dispatch()`, `Buffer.combine()` 等），通过 `AutoKernelDispatcher` 机制在运行时选择使用哪条路径。

---

## 二、路径一：rocSHMEM 内置实现（Primus-Turbo 自带 C++ 源码）

### 2.1 实现原理

该路径将 DeepEP 的**全部 C++ 源码内嵌在 Primus-Turbo 代码库中**，包含：

- **节点内通信（intranode）**：通过 HIP IPC（`hipIpcGetMemHandle` / `hipIpcOpenMemHandle`）实现 NVLink 上的 GPU 直接内存访问，无需 rocSHMEM。
- **节点间通信（internode）**：通过 **rocSHMEM**（ROCm 的 SHMEM 实现）实现 RDMA 跨节点通信，使用 IBGDA（InfiniBand GPU Direct Async）技术实现低延迟的 GPU 发起 RDMA 操作。
- **低延迟模式（low-latency）**：所有 rank（无论节点内外）均通过 RDMA/IBGDA 直接通信。

#### 编译时条件控制

`setup.py` 中通过 `find_rocshmem_library()` 检测 rocSHMEM 和 MPI 是否可用：
- **找到 rocSHMEM + MPI**：正常编译，链接 `librocshmem.a`、MPI、IB verbs、mlx5 等库。
- **未找到**：添加 `-DDISABLE_ROCSHMEM` 编译宏，internode 所有代码被条件编译移除，仅保留 intranode 功能。

相关环境变量：
- `ROCSHMEM_HOME` / `ROCSHMEM_PATH` / `ROCSHMEM_DIR`（默认回退 `/opt/rocm/rocshmem`）
- `MPI_HOME` / `MPI_PATH` / `MPI_DIR`（默认回退 `/opt/rocm/ompi`）

### 2.2 源码目录结构

```
csrc/
├── include/primus_turbo/deep_ep/
│   ├── api.h              # C++ API 声明：intranode/internode 所有函数原型
│   ├── config.hpp         # Config 结构体、Buffer 大小计算、LowLatencyLayout
│   └── configs.h          # 常量定义 (NUM_MAX_NVL_PEERS 等)
├── kernels/deep_ep/
│   ├── buffer.cuh         # GPU buffer 内存布局：Buffer, AsymBuffer, SymBuffer
│   ├── utils.cuh          # 工具宏和辅助函数
│   ├── launch.cuh         # Kernel 启动辅助
│   ├── layout.cu          # get_dispatch_layout kernel：计算 token 分发布局
│   ├── intranode.cu       # NVLink 节点内 dispatch/combine kernels
│   ├── internode.cu       # rocSHMEM RDMA 节点间 dispatch/combine kernels
│   │                       # （整个文件在 #ifndef DISABLE_ROCSHMEM 守护下）
│   ├── runtime.cu         # rocSHMEM 初始化/分配/barrier/finalize
│   └── *.hip              # HIP 变体文件
├── pytorch/deep_ep/
│   ├── deep_ep.cpp        # PyTorch 绑定：Buffer 类、dispatch/combine 封装
│   ├── deep_ep.hpp        # 头文件
│   ├── event.hpp          # EventHandle CUDA 事件封装
│   └── *_hip.*            # HIP 变体
└── jax/deep_ep/
    ├── deep_ep.cpp        # JAX FFI 绑定（仅 intranode）
    └── deep_ep.h
```

Python 层面：

```
primus_turbo/pytorch/deep_ep/
├── __init__.py            # 导出 Config, Buffer, EventOverlap
├── buffer.py              # Buffer Python 类，封装 C++ runtime
└── utils.py               # EventOverlap, EventHandle 封装
```

### 2.3 构建流程

1. **`setup.py`** 调用 `find_rocshmem_library()` → `tools/build_utils.py`
2. 搜索 `ROCSHMEM_HOME` + `MPI_HOME` 环境变量或默认路径
3. 如果找到，构建 `Library` 对象，包含：
   - include_dirs: rocSHMEM + MPI 头文件
   - library_dirs: 库搜索路径
   - extra_link_args: `-l:librocshmem.a -fgpu-rdc --hip-link -lamdhip64 -lhsa-runtime64 -l:libmpi.so -libverbs -lmlx5`
4. 构建两个 Extension：
   - **`libprimus_turbo_kernels`**：编译 `csrc/kernels/**/*.cu`（包含 deep_ep kernels）
   - **`primus_turbo.pytorch._C`**：编译 `csrc/pytorch/**/*.cpp`（包含 deep_ep PyTorch 绑定）
5. 如果未找到 rocSHMEM → 添加 `-DDISABLE_ROCSHMEM` 到所有编译命令

### 2.4 代码调用链

#### 2.4.1 初始化流程

```
用户代码
  └→ turbo_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes, ...)
      │ [primus_turbo/pytorch/deep_ep/buffer.py: Buffer.__init__]
      │
      ├→ deep_ep_cpp.Buffer(rank, group_size, num_nvl_bytes, num_rdma_bytes, ...)
      │   [csrc/pytorch/deep_ep/deep_ep.cpp: Buffer 构造函数]
      │   ├→ 分配 NVLink IPC buffer (hipMalloc)
      │   └→ 获取 IPC handle (hipIpcGetMemHandle)
      │
      ├→ dist.all_gather_object(device_ids, ...)     # 同步设备 ID
      ├→ dist.all_gather_object(ipc_handles, ...)    # 同步 IPC handles
      │
      ├→ [如果 num_rdma_ranks > 1 或 low_latency_mode]
      │   ├→ 设置 NVSHMEM 相关环境变量 (IBGDA, QP 深度等)
      │   ├→ runtime.get_local_nvshmem_unique_id()
      │   │   └→ internode::get_unique_id()
      │   │       └→ rocshmem_get_uniqueid()  [csrc/kernels/deep_ep/runtime.cu]
      │   └→ dist.all_gather_object(nvshmem_unique_ids, ...)
      │
      └→ runtime.sync(device_ids, ipc_handles, root_unique_id)
          [csrc/pytorch/deep_ep/deep_ep.cpp: Buffer::sync]
          ├→ hipIpcOpenMemHandle() — 打开 peer 的 NVLink buffer
          └→ internode::init(root_unique_id, rank, num_ranks, low_latency)
              └→ rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, ...)
              └→ rocshmem_team_split_strided() — 创建 cpu_rdma_team
              └→ internode::alloc() → rocshmem_malloc() — 分配 RDMA 对称堆
```

#### 2.4.2 节点内 Dispatch 调用链

```
buffer.dispatch(x, topk_idx=..., num_tokens_per_rank=..., ...)
  │ [buffer.py: Buffer.dispatch, 当 num_rdma_ranks <= 1]
  │
  ├→ buffer.get_dispatch_layout(topk_idx, num_experts, ...)
  │   └→ runtime.get_dispatch_layout(...)
  │       └→ layout::get_dispatch_layout() [csrc/kernels/deep_ep/layout.cu]
  │
  └→ runtime.intranode_dispatch(x, x_scales, topk_idx, topk_weights, ...)
      [csrc/pytorch/deep_ep/deep_ep.cpp: Buffer::intranode_dispatch]
      ├→ intranode::notify_dispatch()  [csrc/kernels/deep_ep/intranode.cu]
      │   └→ 通过 barrier_signal_ptrs 和 buffer_ptrs 进行 NVLink peer 信号同步
      │   └→ 计算 channel_prefix_matrix, rank_prefix_matrix
      └→ intranode::dispatch()  [csrc/kernels/deep_ep/intranode.cu]
          └→ 通过 NVLink IPC buffer 搬运 token 数据
```

#### 2.4.3 节点间 Dispatch 调用链（需要 rocSHMEM）

```
buffer.dispatch(x, topk_idx=..., num_tokens_per_rank=..., ...)
  │ [buffer.py: Buffer.dispatch, 当 num_rdma_ranks > 1]
  │
  └→ buffer.internode_dispatch(x, ...)
      │ [buffer.py: Buffer.internode_dispatch]
      └→ runtime.internode_dispatch(x, x_scales, topk_idx, ...)
          [csrc/pytorch/deep_ep/deep_ep.cpp: Buffer::internode_dispatch]
          ├→ internode::notify_dispatch()  [csrc/kernels/deep_ep/internode.cu]
          │   └→ rocshmem_ctx_int_put_nbi_wave() — RDMA 写元数据
          │   └→ rocshmem_fence() / rocshmem_ctx_quiet()
          │   └→ NVLink barrier 同步本节点内 GPU
          └→ internode::dispatch()  [csrc/kernels/deep_ep/internode.cu]
              ├→ 通过 NVLink IPC buffer 搬运本节点 token
              └→ 通过 rocSHMEM RDMA put/get 搬运跨节点 token
                  └→ rocshmem_ctx_ulong_atomic_add() — RDMA 原子操作
                  └→ rocshmem_int_put_nbi() / rocshmem_ctx_schar_put_nbi_wave()
```

#### 2.4.4 Combine 调用链类似，方向相反

```
buffer.combine(x, handle, topk_weights, ...)
  ├→ [intranode] runtime.intranode_combine() → intranode::combine()
  └→ [internode] runtime.internode_combine() → internode::combine()
```

#### 2.4.5 低延迟模式调用链

```
buffer.low_latency_dispatch(x, topk_idx, ...)
  └→ runtime.low_latency_dispatch(...)
      └→ [通过 IBGDA/rocSHMEM 所有 rank 间直接 RDMA 通信]

buffer.low_latency_combine(x, topk_idx, topk_weights, handle, ...)
  └→ runtime.low_latency_combine(...)
```

### 2.5 rocSHMEM 关键 API 使用汇总

| 文件 | API | 用途 |
|------|-----|------|
| `runtime.cu` | `rocshmem_get_uniqueid()` | 获取初始化用的唯一 ID |
| `runtime.cu` | `rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID)` | 初始化 rocSHMEM |
| `runtime.cu` | `rocshmem_team_split_strided()` | 创建跨节点同 GPU index 的 team |
| `runtime.cu` | `rocshmem_malloc()` / `rocshmem_free()` | 对称堆内存分配/释放 |
| `runtime.cu` | `rocshmem_barrier_all()` | 全局 barrier |
| `runtime.cu` | `rocshmem_finalize()` | 销毁 rocSHMEM |
| `internode.cu` | `rocshmem_wg_ctx_create()` / `destroy()` | workgroup context 管理 |
| `internode.cu` | `rocshmem_ctx_int_put_nbi_wave()` | RDMA 非阻塞 put（wave 粒度） |
| `internode.cu` | `rocshmem_ctx_schar_put_nbi_wave()` | RDMA 非阻塞字节 put |
| `internode.cu` | `rocshmem_ctx_ulong_atomic_add()` | RDMA 原子加 |
| `internode.cu` | `rocshmem_fence()` / `rocshmem_ctx_quiet()` | 内存序保证 |
| `internode.cu` | `rocshmem_ctx_barrier()` | team 级 barrier |

---

## 三、路径二：UCCL deep_ep 外部包实现

### 3.1 实现原理

该路径**不包含任何 C++ 源码在 Primus-Turbo 仓库中**。它依赖于一个**独立安装的 `deep_ep` Python 包**，该包来自 [uccl-project/uccl](https://github.com/uccl-project/uccl) 项目或 [ROCm/DeepEP](https://github.com/ROCm/DeepEP)。

UCCL（Unified Collective Communication Library）项目提供了自己的 EP（Expert Parallel）通信实现：
- 仓库结构中有 `ep/` 目录（EP 核心库）和 `ep/deep_ep_wrapper/`（封装为兼容 DeepEP API 的 Python 包）
- 编译产物为一个独立的 `deep_ep` wheel 包
- 该包提供了与 Primus-Turbo 内置 DeepEP 相同的 Python API 接口（`Buffer`, `Config`, `EventOverlap` 等）

### 3.2 安装方式

在 CI 中的流程（`.github/workflows/ci.yaml`）：

```
1. 检出 uccl-project/uccl 仓库（指定 commit）
2. 安装构建依赖：rdma-core, libibverbs-dev, libnuma-dev, libgoogle-glog-dev
3. 构建 EP 核心：
   cd ep && python3 setup.py build
4. 复制 .so 到 uccl/ 目录：
   cp ep/build/**/*.so uccl/
5. 构建 uccl wheel：
   python3 setup.py bdist_wheel
6. 构建 deep_ep_wrapper wheel：
   cd ep/deep_ep_wrapper && python3 setup.py bdist_wheel
7. 安装 wheel：
   pip3 install ${UCCL_WHEEL_DIR}/*.whl
```

在非 CI 环境中：
- 从 uccl 项目源码构建安装
- 或者从镜像中直接使用预装的 `deep_ep` 包

### 3.3 运行时检测

```python
# primus_turbo/pytorch/core/backend.py
try:
    HAVE_DEEP_EP = True
    import deep_ep
except ImportError:
    HAVE_DEEP_EP = False
```

当用户设置 `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP` 时：
```python
if backend == BackendType.DEEP_EP:
    assert HAVE_DEEP_EP, (
        "DeepEP is required for this module. "
        "Install from https://github.com/uccl-project/uccl "
        "or https://github.com/ROCm/DeepEP"
    )
```

### 3.4 代码调用链

#### 3.4.1 后端注册与选择

```
MoEDispatchKernelDispatcher._backends = {
    BackendType.TURBO:   MoEDispatchTurboBackend,    # 路径一
    BackendType.DEEP_EP: MoEDispatchDeepEPBackend,   # 路径二
}

MoECombineKernelDispatcher._backends = {
    BackendType.TURBO:   MoECombineTurboBackend,     # 路径一
    BackendType.DEEP_EP: MoECombineDeepEPBackend,    # 路径二
}
```

#### 3.4.2 Dispatch 调用链（路径二）

```
moe_dispatch_impl(x, group, topk_idx=..., ...)
  │ [moe_dispatch_combine_impl.py]
  │
  └→ MoEDispatchKernelDispatcher.dispatch(default_backend, user_backend, **kwargs)
      │ [AutoKernelDispatcher 选择后端]
      │
      └→ MoEDispatchDeepEPBackend.execute(x, group, ...)
          │
          ├→ get_buffer(group, hidden_bytes, deep_ep.Buffer, {"is_intranode": group.size() <= 8})
          │   └→ deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes, ...)
          │       └→ [UCCL deep_ep 包自己的初始化逻辑]
          │
          └→ _moe_dispatch_multiple_backends_impl(buffer, deep_ep.utils.EventOverlap, ...)
              ├→ buffer.get_dispatch_layout(topk_idx, num_experts, ...)
              └→ buffer.dispatch(x, topk_idx=..., num_tokens_per_rank=..., ...)
                  └→ [UCCL deep_ep 包内部的 C++ kernel 实现]
```

#### 3.4.3 Combine 调用链（路径二）

```
moe_combine_impl(x, group, handle, ...)
  └→ MoECombineKernelDispatcher.dispatch(...)
      └→ MoECombineDeepEPBackend.execute(x, group, handle, ...)
          ├→ get_buffer(group, hidden_bytes, deep_ep.Buffer, ...)
          └→ _moe_combine_multiple_backends_impl(buffer, deep_ep.utils.EventOverlap, ...)
              └→ buffer.combine(x, handle=handle, topk_weights=..., ...)
                  └→ [UCCL deep_ep 包内部的 C++ kernel 实现]
```

### 3.5 UCCL deep_ep 包特点

- 提供独立的通信后端，不依赖 rocSHMEM
- API 兼容 DeepEP 接口（`Buffer`, `Config`, `EventOverlap`, `EventHandle`）
- 在 `MoEDispatchDeepEPBackend` 中使用 `{"is_intranode": group.size() <= 8}` 参数来区分节点内外
- 由 UCCL 项目维护，包含自己的 RDMA/网络通信优化

---

## 四、两条路径对比

### 4.1 架构差异

| 维度 | 路径一：Turbo 内置 (rocSHMEM) | 路径二：UCCL deep_ep |
|------|------------------------------|---------------------|
| **C++ 源码位置** | 本仓库 `csrc/kernels/deep_ep/` + `csrc/pytorch/deep_ep/` | 外部 `uccl-project/uccl` 仓库 |
| **Python 包名** | `primus_turbo.pytorch.deep_ep` | `deep_ep` |
| **Backend 类型** | `BackendType.TURBO` | `BackendType.DEEP_EP` |
| **编译方式** | 与 Primus-Turbo 一起编译 | 独立 wheel 包安装 |
| **节点内通信** | HIP IPC (NVLink) | UCCL EP 自有实现 |
| **节点间通信** | rocSHMEM (RDMA/IBGDA) | UCCL EP 自有 RDMA 实现 |
| **低延迟模式** | rocSHMEM IBGDA | UCCL EP 自有实现 |
| **依赖** | rocSHMEM + MPI + IB verbs + mlx5 | UCCL EP 库 + rdma-core |

### 4.2 选择逻辑

后端选择通过 `AutoKernelDispatcher` 进行，优先级从高到低：

1. **用户显式指定**：环境变量 `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=DEEP_EP` 或 `TURBO`
2. **自动调优**：`PRIMUS_TURBO_AUTO_TUNE=1` 时，profiling 所有可用后端选最快
3. **代码默认**：`moe_dispatch_impl` 的 `default_backend` 参数（默认 `BackendType.DEEP_EP`）
4. **Fallback**：遍历所有注册的后端，选第一个 `can_handle()` 返回 True 的

注意：**`MoEDispatchDeepEPBackend.can_handle()` 直接返回 `HAVE_DEEP_EP`**，即取决于 `deep_ep` 包是否安装成功。而 **`MoEDispatchTurboBackend.can_handle()` 始终返回 `True`**。

### 4.3 实际默认行为

在 `moe_dispatch_impl` 中，`default_backend` 默认值为 `BackendType.DEEP_EP`。这意味着：

- 如果安装了 `deep_ep` 包（UCCL）→ 默认使用**路径二**
- 如果未安装 `deep_ep` 包 → fallback 到**路径一** (`TURBO`)
- 用户可通过环境变量 `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND=TURBO` 强制使用路径一

---

## 五、CI 中的两条路径

在 `.github/workflows/ci.yaml` 中：

### PyTorch CI (`unittest-pytorch-gfx942`)
1. 先从 UCCL 仓库构建并安装 `deep_ep` wheel（`install-dependencies` job）
2. 然后构建 Primus-Turbo 本身（包含内置 DeepEP kernels）
3. 测试时两条路径都可用，具体用哪条取决于默认值或环境变量

### JAX CI (`unittest-jax-gfx942`)
- 不安装 UCCL wheel
- 仅使用 Primus-Turbo 内置的 intranode DeepEP（JAX 路径暂不支持 internode）

---

## 六、测试覆盖

| 测试文件 | 路径 | 覆盖范围 |
|---------|------|---------|
| `tests/pytorch/deep_ep/test_intranode.py` | 路径一 (`turbo_ep`) | Intranode dispatch/combine |
| `tests/pytorch/ref/deep_ep_ref.py` | 参考实现 | `tune_and_verify_intranode`, `tune_and_verify_internode` (参考逻辑) |

注意：
- 目前**没有** `test_internode.py` 测试文件（internode 需要多节点 + rocSHMEM 环境）
- 测试中使用 `primus_turbo.pytorch.deep_ep` (路径一)，未单独测试路径二的 `deep_ep` 包

---

## 七、总结

```
                    ┌─────────────────────────────────────────────────┐
                    │         MoE Dispatch / Combine 上层调用           │
                    │  (moe_dispatch_impl / moe_combine_impl)         │
                    └────────────────────┬────────────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         │  AutoKernelDispatcher 选择后端  │
                         └───────┬───────────────┬───────┘
                                 │               │
              ┌──────────────────┴──┐     ┌──────┴──────────────────┐
              │  BackendType.TURBO  │     │  BackendType.DEEP_EP    │
              │  MoEDispatchTurbo-  │     │  MoEDispatchDeepEP-     │
              │  Backend            │     │  Backend                │
              └──────────┬─────────┘     └──────────┬──────────────┘
                         │                          │
         ┌───────────────┴───────────────┐   ┌──────┴──────────────────┐
         │  turbo_ep.Buffer              │   │  deep_ep.Buffer         │
         │  (primus_turbo.pytorch.       │   │  (UCCL/ROCm 外部包)      │
         │   deep_ep)                    │   │                         │
         └───────────────┬───────────────┘   └──────┬──────────────────┘
                         │                          │
         ┌───────────────┴───────────────┐   ┌──────┴──────────────────┐
         │  primus_turbo.pytorch._C.     │   │  deep_ep 包内部          │
         │  deep_ep (C++ 扩展)            │   │  C++ 扩展                │
         └───────────────┬───────────────┘   └──────┬──────────────────┘
                         │                          │
         ┌───────────────┴───────────────┐   ┌──────┴──────────────────┐
         │  csrc/kernels/deep_ep/        │   │  UCCL EP 库              │
         │  ├─ intranode.cu (NVLink IPC) │   │  (uccl-project/uccl     │
         │  ├─ internode.cu (rocSHMEM)   │   │   ep/ 目录)              │
         │  └─ runtime.cu (rocSHMEM 初始化)│   │                         │
         └───────────────────────────────┘   └─────────────────────────┘
```

**关键结论**：
1. 路径一（Turbo/rocSHMEM）是完全自包含的，C++ 源码在本仓库，internode 依赖 rocSHMEM。
2. 路径二（UCCL deep_ep）是一个独立外部包，通过 Python `import deep_ep` 按需加载。
3. 两条路径通过 `AutoKernelDispatcher` 统一调度，共享相同的上层 MoE API。
4. 默认优先使用路径二（如果安装了 `deep_ep` 包），否则 fallback 到路径一。
