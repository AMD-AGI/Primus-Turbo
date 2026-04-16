# Primus-Turbo DeepEP：PyTorch vs JAX 实现对比报告

> 分析日期：2026-04-13
> 代码库：Primus-Turbo (`primus_turbo`)
> 对比对象：内置 Turbo-DeepEP 在 PyTorch 和 JAX 两个框架下的实现

---

## 一、总结性概览

### 1.1 一句话总结

PyTorch 和 JAX 的 Turbo-DeepEP 实现**共享同一套底层 GPU kernel**（`csrc/kernels/deep_ep/`），但在框架绑定层、Python API 设计、Buffer 生命周期管理、通信模式支持等方面存在显著差异。**JAX 路径目前仅支持节点内（intranode）通信，而 PyTorch 路径支持完整的节点内+节点间（internode）+ 低延迟（low-latency）模式。**

### 1.2 功能矩阵对比

| 功能特性 | PyTorch | JAX |
|---------|---------|-----|
| **Intranode Dispatch** | ✅ 完整支持 | ✅ 完整支持 |
| **Intranode Combine** | ✅ 完整支持 | ✅ 完整支持 |
| **Cached Dispatch** | ✅ 支持 | ✅ 支持 |
| **Internode Dispatch (rocSHMEM)** | ✅ 完整支持 | ❌ 不支持 (TODO) |
| **Internode Combine (rocSHMEM)** | ✅ 完整支持 | ❌ 不支持 (TODO) |
| **Low-Latency Mode (IBGDA)** | ❌ stub (`PRIMUS_TURBO_CHECK(false)`) | ❌ 不支持 |
| **FP8 数据传输** | ✅ 支持 | ✅ 支持 |
| **Bias 支持** | ✅ combine 支持 | ✅ combine 支持 |
| **异步流管理** | ✅ 双流 (compute + comm) | ❌ 单流（XLA 管理） |
| **CUDA Graph 兼容** | ✅ 部分支持 (worst-case 模式) | ❌ 不涉及 |
| **自动反向传播** | ❌ 需手动（外部 autograd） | ✅ 内置 `custom_vjp` |
| **多后端调度** | ✅ AutoKernelDispatcher | ❌ 仅 Turbo 内置 |
| **IPC Handle 同步** | ✅ `torch.distributed` | ✅ 进程内直接内存访问 |

---

## 二、架构分层对比

### 2.1 整体架构图

```
                     ┌─────────── PyTorch 路径 ──────────┐  ┌───────────── JAX 路径 ──────────────┐
                     │                                    │  │                                      │
用户 API 层          │ primus_turbo.pytorch.deep_ep       │  │ primus_turbo.jax.lax.moe             │
                     │ ├─ buffer.py (Buffer 类)           │  │ ├─ moe_dispatch_combine.py           │
                     │ ├─ utils.py (EventOverlap)         │  │ │  ├─ moe_dispatch()                 │
                     │ └─ __init__.py                     │  │ │  ├─ moe_combine()                  │
                     │                                    │  │ │  └─ custom_vjp（自动微分）           │
                     │                                    │  │ └─ moe_utils.py (Config)             │
                     ├────────────────────────────────────┤  ├──────────────────────────────────────┤
MoE 调度层           │ moe_dispatch_combine_impl.py       │  │ primitive/moe/                       │
                     │ ├─ MoEDispatchTurboBackend          │  │ ├─ moe_dispatch.py (JAX Primitive)   │
                     │ ├─ MoEDispatchDeepEPBackend         │  │ └─ moe_combine.py  (JAX Primitive)   │
                     │ └─ AutoKernelDispatcher             │  │                                      │
                     ├────────────────────────────────────┤  ├──────────────────────────────────────┤
框架绑定层 (C++)     │ csrc/pytorch/deep_ep/              │  │ csrc/jax/deep_ep/                    │
                     │ ├─ deep_ep.cpp (Buffer C++ 类)     │  │ ├─ deep_ep.cpp (Buffer C++ 类)       │
                     │ ├─ deep_ep.hpp                     │  │ ├─ deep_ep.h                         │
                     │ ├─ event.hpp (EventHandle)         │  │ └─ handler.cpp (FFI handlers)        │
                     │ └─ pybind11 注册                    │  │    └─ XLA_FFI_DEFINE_HANDLER_SYMBOL  │
                     ├────────────────────────────────────┤  ├──────────────────────────────────────┤
                     │                    ┌───────────────────────────────────┐                      │
共享 GPU Kernel 层   │                    │  csrc/kernels/deep_ep/            │                      │
                     │                    │  ├─ layout.cu      (布局计算)     │                      │
                     │                    │  ├─ intranode.cu   (NVLink 通信)  │                      │
                     │                    │  ├─ internode.cu   (rocSHMEM RDMA)│                      │
                     │                    │  ├─ runtime.cu     (rocSHMEM 运行时)│                     │
                     │                    │  ├─ buffer.cuh     (Buffer 布局)  │                      │
                     │                    │  └─ utils.cuh      (工具函数)     │                      │
                     │                    └───────────────────────────────────┘                      │
                     └────────────────────────────────────┘  └──────────────────────────────────────┘
```

### 2.2 核心差异点

| 层次 | PyTorch | JAX |
|------|---------|-----|
| **框架绑定** | pybind11 → `primus_turbo.pytorch._C.deep_ep` | XLA FFI → `jax.ffi.register_ffi_target` |
| **Python 封装** | 面向对象 (`Buffer` 类) | 函数式 (`moe_dispatch_p.bind()`) |
| **张量体系** | `torch.Tensor` | `jax.ffi.Buffer<T>` / `AnyBuffer` |
| **流管理** | 双流 (compute + comm) + EventHandle 同步 | 由 XLA runtime 管理 hipStream |
| **IPC 同步** | `torch.distributed.all_gather_object` | 进程内全局 `g_buffer_pool` + `std::barrier` |
| **Buffer 管理** | 用户显式构建 `Buffer(group, ...)` | 全局单例 `get_buffer(rank, ...)` 自动创建 |
| **internode** | 完整 rocSHMEM 支持 | `num_rdma_bytes = 0`，仅 intranode |

---

## 三、PyTorch 实现详细拆解

### 3.1 Python 层（`primus_turbo/pytorch/deep_ep/`）

#### 3.1.1 核心类 `Buffer`（`buffer.py`）

`Buffer` 是一个有状态的 Python 类，封装了 C++ `deep_ep_cpp.Buffer` 对象。核心方法：

| 方法 | 功能 | 调用的 C++ 方法 |
|------|------|----------------|
| `__init__` | 初始化 IPC/RDMA buffer，同步 handle | `deep_ep_cpp.Buffer()` + `sync()` |
| `get_dispatch_layout` | 计算 token 分发布局 | `runtime.get_dispatch_layout()` |
| `dispatch` | Token 分发（自动选择 intra/internode） | `runtime.intranode_dispatch()` 或 `internode_dispatch()` |
| `combine` | Token 聚合（自动选择 intra/internode） | `runtime.intranode_combine()` 或 `internode_combine()` |
| `internode_dispatch` | 跨节点 dispatch | `runtime.internode_dispatch()` |
| `internode_combine` | 跨节点 combine | `runtime.internode_combine()` |
| `low_latency_dispatch` | 低延迟 dispatch（IBGDA） | `runtime.low_latency_dispatch()` |
| `low_latency_combine` | 低延迟 combine（IBGDA） | `runtime.low_latency_combine()` |
| `destroy` | 显式释放资源 | `runtime.destroy()` |

**关键设计**：
- `Buffer` 在 `dispatch()` 中通过 `self.runtime.get_num_rdma_ranks() > 1` 自动路由到 intranode 或 internode 实现
- 支持 **cached mode**：第二次 dispatch 时可复用 handle，跳过 layout 计算
- 支持 **worst-case 模式**（`num_worst_tokens > 0`）：避免 CPU-GPU 同步，与 CUDA Graph 兼容

#### 3.1.2 事件系统（`utils.py`）

`EventOverlap` 类封装 `EventHandle`（C++ CUDA Event 包装器），用于：
- 双流间同步（compute stream ↔ comm stream）
- Python `with` 语法支持计算与通信重叠

#### 3.1.3 MoE 调度层（`moe_dispatch_combine_impl.py`）

通过 `AutoKernelDispatcher` 实现多后端选择：
- `MoEDispatchTurboBackend`：使用 `turbo_ep.Buffer`（内置实现）
- `MoEDispatchDeepEPBackend`：使用 `deep_ep.Buffer`（外部 UCCL 包）
- 通过环境变量 `PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND` 或 auto-tune 切换

### 3.2 C++ 绑定层（`csrc/pytorch/deep_ep/`）

#### 3.2.1 Buffer 构造函数

```cpp
Buffer::Buffer(rank, num_ranks, num_nvl_bytes, num_rdma_bytes, low_latency_mode, ...)
```

关键初始化步骤：
1. **NVLink IPC buffer 分配**：`hipExtMallocWithFlags(..., hipDeviceMallocUncached)` → 分配 uncached GPU 内存
2. **IPC Handle 创建**：`hipIpcGetMemHandle()` → 获取本 GPU 的 IPC handle
3. **Barrier 信号区**：在 NVLink buffer 尾部分配 `NUM_MAX_NVL_PEERS * sizeof(int)` 字节
4. **Buffer 指针区**：在信号区之后存储所有 peer 的 buffer 指针（GPU 可见）
5. **Workspace**：`hipMalloc` 32 MiB 工作空间
6. **MoE 计数器**：`hipHostMalloc` 分配 host-mapped 计数器（CPU 可轮询读取）

#### 3.2.2 `sync()` 方法

IPC handle 和 rocSHMEM 同步：
1. 通过 `hipIpcOpenMemHandle()` 打开远端 GPU 的 NVLink buffer
2. 构造 `barrier_signal_ptrs` 数组指向所有 peer 的信号区
3. 将指针数组拷贝到 GPU（`buffer_ptrs_gpu`, `barrier_signal_ptrs_gpu`）
4. 如果有 RDMA buffer：
   - `internode::init()` → `rocshmem_init_attr()`
   - `internode::alloc()` → `rocshmem_malloc()`
   - `hipMemset(rdma_buffer_ptr, 0, num_rdma_bytes)`

#### 3.2.3 `intranode_dispatch()` 实现

```
intranode_dispatch(x, x_scales, topk_idx, topk_weights, ...)
  │
  ├→ [非 cached mode]
  │   ├→ 重置 moe_recv_counter = -1
  │   ├→ intranode::notify_dispatch(...)
  │   │   // GPU kernel: 写入 token-per-rank 信息到 NVLink buffer
  │   │   // 通过 barrier 同步，对端 GPU 读取
  │   │   // GPU 将 recv_token_count 写入 host-mapped counter
  │   ├→ CPU busy-wait 循环等待 moe_recv_counter >= 0
  │   │   // 这是 CPU-GPU 同步点！
  │   └→ 读取 num_recv_tokens 和 num_recv_tokens_per_expert
  │
  ├→ [worst-case mode] 跳过 CPU 同步，直接用 num_worst_tokens
  │
  ├→ 分配输出 tensor：recv_x, recv_src_idx, ...
  │
  └→ intranode::dispatch(...)
      // GPU kernel: 通过 NVLink IPC buffer 搬运 token 数据
      // 使用 channel prefix matrix 确定每个 channel 负责哪些 token
```

#### 3.2.4 `internode_dispatch()` 实现（需 rocSHMEM）

```
internode_dispatch(x, x_scales, topk_idx, topk_weights, ...)
  │
  ├→ pybind11::gil_scoped_release  // 释放 GIL（CPU busy-wait 可能很久）
  │
  ├→ [非 cached mode]
  │   ├→ internode::notify_dispatch(...)
  │   │   // GPU kernel: 通过 NVLink + rocSHMEM RDMA 发送元数据
  │   │   // rocshmem_ctx_int_put_nbi_wave() 写到远端 RDMA buffer
  │   ├→ CPU busy-wait: 等待 moe_recv_counter, moe_recv_rdma_counter, 以及 per-expert counter
  │   └→ 读取 num_recv_tokens, num_rdma_recv_tokens
  │
  ├→ [cached mode]
  │   └→ internode::cached_notify(...)  // 仅 barrier + clean flags
  │
  ├→ 分配 recv tensors（包括 recv_src_meta 等 internode 特有的）
  │
  └→ internode::dispatch(...)
      // GPU kernel:
      //   NVLink: 同节点内 token 搬运
      //   RDMA:   跨节点 token 搬运
      //   rocshmem_int_put_nbi() / rocshmem_ctx_ulong_atomic_add()
```

#### 3.2.5 双流模型

PyTorch 实现使用 **compute stream + comm stream** 双流：

```
Compute Stream:  ─────────[MoE计算]─────── wait event ───[后续计算]──→
                                             ↑
Comm Stream:     ──── wait ───[dispatch kernel]──── event ─────────→
                      ↑
                  previous_event
```

- `allocate_on_comm_stream`: 在 comm stream 上分配 tensor，避免 compute stream 等待
- `async_finish`: dispatch/combine 不等待完成，返回 `EventHandle` 供后续同步
- `record_stream()`: 确保 tensor 在两个 stream 间安全引用

### 3.3 PyTorch 完整 Dispatch 调用链

```
用户代码: buffer.dispatch(x, topk_idx=..., num_tokens_per_rank=..., ...)
  │
  │ [primus_turbo/pytorch/deep_ep/buffer.py]
  ├→ config = self.get_dispatch_config(self.group_size)
  ├→ if self.runtime.get_num_rdma_ranks() > 1:
  │     └→ self.internode_dispatch(...)    ─── 见上方 3.2.4
  │
  └→ [intranode path]
      ├→ x, x_scales = (x, None) 或 unpack tuple
      │
      ├→ [有 handle (cached mode)]
      │   └→ self.runtime.intranode_dispatch(x, ..., cached_rank_prefix_matrix=..., ...)
      │       │ [csrc/pytorch/deep_ep/deep_ep.cpp: Buffer::intranode_dispatch]
      │       ├→ stream_wait(comm_stream, previous_event)
      │       ├→ intranode::cached_notify_dispatch(...)
      │       │   [csrc/kernels/deep_ep/intranode.cu]
      │       ├→ intranode::dispatch(...)
      │       │   [csrc/kernels/deep_ep/intranode.cu]
      │       └→ EventHandle(comm_stream) → 返回
      │
      └→ [无 handle (首次)]
          └→ self.runtime.intranode_dispatch(x, ..., num_tokens_per_rank=..., ...)
              │ [csrc/pytorch/deep_ep/deep_ep.cpp: Buffer::intranode_dispatch]
              ├→ stream_wait(comm_stream, compute_stream)
              ├→ moe_recv_counter = -1; moe_recv_expert_counter = -1
              ├→ intranode::notify_dispatch(...)
              │   [csrc/kernels/deep_ep/intranode.cu]
              │   └→ 通过 NVLink buffer 广播 token count → 对端 GPU 写入 host-mapped counter
              ├→ CPU busy-wait: while (*moe_recv_counter < 0) { ... timeout check ... }
              ├→ 分配 recv_x, recv_src_idx, recv_topk_idx, recv_topk_weights
              ├→ intranode::dispatch(...)
              │   [csrc/kernels/deep_ep/intranode.cu]
              │   └→ 通过 NVLink IPC buffer 搬运 token data + metadata
              └→ return (recv_x, ..., handle, EventOverlap(event))
```

---

## 四、JAX 实现详细拆解

### 4.1 Python 层

#### 4.1.1 Primitive 定义（`primitive/moe/`）

JAX 使用 **JAX Primitive** 系统将 DeepEP 操作注册为 XLA 自定义算子：

**`moe_dispatch.py`**：
```python
moe_dispatch_p = Primitive("moe_dispatch")           # 首次 dispatch
moe_cached_dispatch_p = Primitive("moe_cached_dispatch")  # cached dispatch
```

每个 Primitive 注册四个表：
- `IMPL_TABLE`: 实现函数（通过 `xla.apply_primitive` 委托）
- `ABSTRACT_EVAL_TABLE`: 形状推断（`_moe_dispatch_abstract_eval`）
- `LOWERING_TABLE`: XLA lowering → FFI（`jax.ffi.ffi_lowering("moe_dispatch")`）
- `TRANSPOSE_TABLE` / `BATCHING_TABLE`: 梯度/batch（TODO）

**`moe_combine.py`**：
```python
moe_combine_p = Primitive("moe_combine")
```

#### 4.1.2 用户 API 层（`lax/moe/moe_dispatch_combine.py`）

提供**函数式 API**，带有内置自动微分：

```python
@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _moe_dispatch(x, topk_idx, topk_weights, num_experts, expert_alignment, config):
    return _moe_dispatch_impl(x, topk_idx=topk_idx, ...)

# VJP 定义
def _moe_dispatch_fwd(x, topk_idx, topk_weights, ...):
    result = _moe_dispatch_impl(x, ...)
    ctx = (handle,)
    return result, ctx

def _moe_dispatch_bwd(num_experts, expert_alignment, config, ctx, grad_output):
    (handle,) = ctx
    grad_x, grad_topk_weights = _moe_combine_impl(grad_x, handle, ...)
    return grad_x, None, grad_topk_weights

# combine 的 VJP 正好反过来：bwd 调用 dispatch
def _moe_combine_bwd(config, ctx, grad_output):
    recv_grad_x, _, _, _ = _moe_dispatch_impl(grad_output, handle=handle, config=config)
    return recv_grad_x, handle_grad
```

**关键差异**：
- JAX 通过 `custom_vjp` 直接定义了 dispatch ↔ combine 的反向传播关系
- `dispatch` 的反向是 `combine`，`combine` 的反向是 `cached_dispatch`
- PyTorch 不在 DeepEP 内部处理 autograd，由外部手动管理

#### 4.1.3 Config

```python
class Config(NamedTuple):
    num_sms: int
    num_max_nvl_chunked_send_tokens: int
    num_max_nvl_chunked_recv_tokens: int
    num_max_rdma_chunked_send_tokens: int
    num_max_rdma_chunked_recv_tokens: int
```

与 PyTorch 的 `Config` 字段完全对应，且默认 config 映射表完全相同。

#### 4.1.4 `num_worst_tokens` 约束

JAX 中 `num_worst_tokens = num_tokens * jax.local_device_count()`，因为 XLA 需要**静态形状**，不能进行 CPU-GPU 同步来获取动态接收 token 数。所有输出 tensor 按最大可能大小分配。

对应的在 C++ 中有检查：
```cpp
PRIMUS_TURBO_CHECK(num_worst_tokens > num_tokens);  // JAX 版特有
```

### 4.2 C++ 绑定层（`csrc/jax/deep_ep/`）

#### 4.2.1 FFI Handler 架构

JAX 不使用 pybind11，而是通过 **XLA FFI (Foreign Function Interface)** 系统：

```cpp
// handler.cpp
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MoEDispatchHandler, MoEDispatchFFI,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()   // 从 XLA 获取 hipStream
        .Arg<ffi::AnyBuffer>()                      // x
        .Arg<ffi::Buffer<ffi::F32>>()               // x_scales
        .Arg<ffi::Buffer<ffi::S32>>()               // topk_idx
        .Arg<ffi::Buffer<ffi::F32>>()               // topk_weights
        .Attr<int64_t>("num_experts")               // 标量属性
        ...
        .Ret<ffi::AnyBuffer>()                      // recv_x (输出)
        ...
);
```

三个 FFI Handler：
- `MoEDispatchHandler` → `MoEDispatchFFI` → 首次 dispatch（含 layout 计算）
- `MoECachedDispatchHandler` → `MoECachedDispatchFFI` → cached dispatch
- `MoECombineHandler` → `MoECombineFFI` → combine

#### 4.2.2 JAX Buffer 类

```cpp
class Buffer {
    // 无 comm stream / compute stream 区分
    // 无 IPC Handle 分发（通过进程内直接指针共享）
    // 无 rocSHMEM（num_rdma_bytes 固定为 0）
    // 无 EventHandle
};
```

关键差异：
- **无独立 comm stream**：JAX 通过 FFI 直接接收 XLA 提供的 `hipStream_t`，所有操作在该 stream 上执行
- **无 IPC Handle 分发**：JAX 使用**进程内共享内存**（同一进程的多个 GPU 线程共享 `g_buffer_pool`）

#### 4.2.3 Buffer 全局池和同步

```cpp
static std::barrier g_barrier_signal(NUM_MAX_NVL_PEERS);
static std::vector<std::unique_ptr<Buffer>> g_buffer_pool(NUM_MAX_NVL_PEERS);

Buffer *get_buffer(int rank, int num_ranks, int64_t hidden_bytes, const Config &config) {
    // 断言: num_ranks <= NUM_MAX_NVL_PEERS（仅 intranode）
    int device_id = hipGetDevice();

    if (需要重建 buffer) {
        g_buffer_pool[device_id] = make_unique<Buffer>(rank, num_ranks, ...);
        g_barrier_signal.arrive_and_wait();  // C++ std::barrier 跨线程同步
        g_buffer_pool[device_id]->Sync();
    }
    return g_buffer_pool[device_id].get();
}
```

**关键设计**：
- `rank` 直接用 `hipGetDevice()` 返回值（即 GPU device ID = rank）
- `num_ranks` 用 `hipGetDeviceCount()` 获取
- Buffer 生命周期由全局池管理，不需要用户显式创建/销毁
- 使用 C++ `std::barrier` 替代 `torch.distributed` 进行跨 GPU 同步

#### 4.2.4 `Sync()` 方法 — 与 PyTorch 的关键差异

```cpp
void Buffer::Sync() {
    for (int i = 0; i < num_nvl_ranks_; ++i) {
        if (i != rank_) {
            // 直接从全局池读取其他 GPU 的 buffer 指针（进程内共享）
            barrier_signal_ptrs_[i] = ...(g_buffer_pool[i]->buffer_ptrs_[i]);
            buffer_ptrs_[i] = g_buffer_pool[i]->buffer_ptrs_[i];
            hipDeviceCanAccessPeer(&can_access_peer, device_id_, i);
        }
    }
    // 复制指针到 GPU
    hipMemcpy(buffer_ptrs_gpu_, buffer_ptrs_, ...);
    hipMemcpy(barrier_signal_ptrs_gpu_, barrier_signal_ptrs_, ...);
}
```

**对比 PyTorch `sync()`**：
- PyTorch：通过 `torch.distributed.all_gather_object` 收集 IPC handle → `hipIpcOpenMemHandle` 打开
- JAX：所有 GPU 在同一进程中，直接通过 `g_buffer_pool` 共享指针，然后 `hipDeviceCanAccessPeer` 验证

#### 4.2.5 `IntranodeDispatch()` vs PyTorch

JAX 的 `IntranodeDispatch()` 与 PyTorch 的 `intranode_dispatch()` **调用相同的底层 kernel**，但有以下差异：

| 方面 | PyTorch `intranode_dispatch()` | JAX `IntranodeDispatch()` |
|------|-------------------------------|--------------------------|
| 输入类型 | `torch::Tensor` | `ffi::AnyBuffer` / `ffi::Buffer<T>` |
| 输出分配 | 函数内部 `torch::empty()` 分配 | 由 XLA 预分配，通过 `ffi::Result<>` 传入 |
| topk_idx 类型 | `int64_t` | `int32_t` |
| 流管理 | 内部切换 comm_stream | 使用传入的 `hipStream_t` |
| 返回值 | `std::tuple<torch::Tensor, ...>` | `void`（通过 `ffi::Result<>` 写入） |
| CPU-GPU 同步 | 有（busy-wait moe_recv_counter） | 有（同样的 busy-wait） |
| async 支持 | ✅ (EventHandle) | ❌ (XLA 管理) |

### 4.3 JAX 完整 Dispatch 调用链

```
用户代码: moe_dispatch(x, topk_idx, topk_weights, num_experts, ...)
  │
  │ [primus_turbo/jax/lax/moe/moe_dispatch_combine.py]
  ├→ _moe_dispatch(x, topk_idx, topk_weights, ...)       # custom_vjp 包装
  │   └→ _moe_dispatch_impl(x, topk_idx=..., ...)
  │       ├→ x_scales = jnp.array([], dtype=jnp.float32)  # 如非 FP8
  │       ├→ num_worst_tokens = num_tokens * jax.local_device_count()
  │       ├→ config = get_dispatch_config()
  │       │
  │       └→ moe_dispatch_p.bind(x, x_scales, topk_idx, topk_weights,
  │               num_experts=..., num_worst_tokens=..., **config._asdict())
  │           │
  │           │ [JAX trace → XLA lowering → FFI call]
  │           │
  │           └→ MoEDispatchFFI(stream, x, x_scales, topk_idx, topk_weights, ...)
  │               │ [csrc/jax/deep_ep/handler.cpp]
  │               │
  │               ├→ rank = hipGetDevice()
  │               ├→ num_ranks = hipGetDeviceCount()
  │               ├→ config = Config(num_sms, ...)
  │               ├→ buffer = get_buffer(rank, num_ranks, hidden_bytes, config)
  │               │   │ [csrc/jax/deep_ep/deep_ep.cpp: get_buffer()]
  │               │   ├→ [首次] Buffer 构造 → g_barrier_signal.arrive_and_wait() → Sync()
  │               │   └→ [后续] 返回已有 buffer
  │               │
  │               ├→ buffer->DispatchLayout(stream, topk_idx, num_experts, ...)
  │               │   └→ layout::get_dispatch_layout(...)
  │               │       [csrc/kernels/deep_ep/layout.cu]  ← 与 PyTorch 共享
  │               │
  │               └→ buffer->IntranodeDispatch(stream, x, ...)
  │                   │ [csrc/jax/deep_ep/deep_ep.cpp: Buffer::IntranodeDispatch]
  │                   │
  │                   ├→ [非 cached mode]
  │                   │   ├→ moe_recv_counter = -1
  │                   │   ├→ intranode::notify_dispatch(...)
  │                   │   │   [csrc/kernels/deep_ep/intranode.cu]  ← 与 PyTorch 共享
  │                   │   ├→ CPU busy-wait: while (*moe_recv_counter < 0) ...
  │                   │   └→ intranode::dispatch(...)
  │                   │       [csrc/kernels/deep_ep/intranode.cu]  ← 与 PyTorch 共享
  │                   │
  │                   └→ [cached mode] 同 PyTorch 逻辑
  │
  └→ 返回 (recv_x, recv_topk_idx, recv_topk_weights, handle)
```

### 4.4 JAX 反向传播调用链

```
JAX autograd: _moe_dispatch_bwd(num_experts, expert_alignment, config, ctx, grad_output)
  │
  ├→ (handle,) = ctx
  ├→ grad_x = grad_output[0]
  │
  └→ _moe_combine_impl(grad_x, handle, topk_weights=grad_topk_weights, config=config)
      └→ moe_combine_p.bind(grad_x, topk_weights, bias_0, bias_1, src_idx,
              rank_prefix_matrix, channel_prefix_matrix, send_head, **config._asdict())
          └→ MoECombineFFI(stream, ...)
              └→ buffer->IntranodeCombine(stream, ...)
                  └→ intranode::combine(...)
                      [csrc/kernels/deep_ep/intranode.cu]  ← 与 PyTorch 共享
```

---

## 五、GPU Kernel 共享层详解

两个框架**共享完全相同的 GPU kernel**，位于 `csrc/kernels/deep_ep/`：

### 5.1 `layout.cu` — 布局计算

```cpp
void get_dispatch_layout(topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank,
                         num_tokens_per_expert, is_token_in_rank, ...)
```
- 输入：`topk_idx [num_tokens, num_topk]`
- 输出：每个 rank/expert/rdma_rank 要接收的 token 数量、is_token_in_rank 布尔矩阵
- PyTorch 使用 `int64_t` topk_idx，JAX 使用 `int32_t`（通过 C++ template 特化）

### 5.2 `intranode.cu` — NVLink 通信

四个核心 kernel：
1. **`notify_dispatch`**：通过 NVLink buffer 广播 token-per-rank 信息，写入 host-mapped counter
2. **`cached_notify_dispatch`**：复用已有 layout，仅 barrier + clean flags
3. **`dispatch`**：使用 channel prefix matrix 将 token data 通过 NVLink IPC buffer 搬运到目标 GPU
4. **`combine`**：反向搬运，聚合 token 数据（支持 BF16/FP8）

通信模型：
- **Channel-based 分块**：每两个 SM 组成一个 channel（一个发送、一个接收）
- **生产者-消费者队列**：每个 channel 对每个 rank 维护一个环形 buffer，通过 head/tail 指针管理
- **Barrier 同步**：通过 `barrier_signal_ptrs` 实现 NVLink 级别的 GPU 间 barrier

### 5.3 `internode.cu` — rocSHMEM RDMA（仅 PyTorch 可达）

整个文件在 `#ifndef DISABLE_ROCSHMEM` 守护下。GPU kernel 使用 rocSHMEM API 进行跨节点 RDMA：
- `rocshmem_ctx_int_put_nbi_wave()` — 写元数据
- `rocshmem_int_put_nbi()` — 写 token 数据
- `rocshmem_ctx_ulong_atomic_add()` — 原子计数器更新
- `rocshmem_fence()` / `rocshmem_ctx_quiet()` — 内存序保证
- `rocshmem_wg_ctx_create()` / `destroy()` — workgroup context

### 5.4 `runtime.cu` — rocSHMEM 生命周期

初始化/销毁 rocSHMEM 环境：
- `get_unique_id()` → `rocshmem_get_uniqueid()`
- `init()` → `rocshmem_init_attr()` + `rocshmem_team_split_strided()`
- `alloc()` / `free()` → `rocshmem_malloc()` / `rocshmem_free()`
- `barrier()` → `rocshmem_barrier_all()`
- `finalize()` → `rocshmem_team_destroy()` + `rocshmem_finalize()`

---

## 六、IPC / 同步机制对比

### 6.1 Buffer 指针共享

| 机制 | PyTorch | JAX |
|------|---------|-----|
| **进程模型** | 多进程（每 GPU 一个进程） | 单进程多线程（JAX multi-device） |
| **指针共享** | HIP IPC Handle 序列化传输 | 进程内 `g_buffer_pool` 直接共享 |
| **同步原语** | `torch.distributed.all_gather_object` | `std::barrier` (C++20) |
| **Handle 打开** | `hipIpcOpenMemHandle()` | 直接指针赋值 + `hipDeviceCanAccessPeer()` |

### 6.2 MoE 计数器同步

两者都使用 **host-mapped GPU memory + CPU busy-wait**：
```cpp
hipHostMalloc(&moe_recv_counter, sizeof(int64_t), hipHostAllocMapped);
hipHostGetDevicePointer(&moe_recv_counter_mapped, moe_recv_counter, 0);

// GPU kernel 写入 moe_recv_counter_mapped
// CPU busy-wait 读取 moe_recv_counter
while (*moe_recv_counter < 0) { ... timeout check ... }
```

但 JAX 必须使用 `num_worst_tokens` 模式以保持 XLA 输出形状静态。

---

## 七、数据类型处理对比

| 方面 | PyTorch | JAX |
|------|---------|-----|
| topk_idx dtype | `int64_t` | `int32_t` |
| topk_weights dtype | `float32` | `float32` |
| x dtype | `bfloat16` / `float8_e4m3fn` | `bfloat16` / `float8_e4m3fn` |
| x_scales | `float32` / `int` | `float32` |
| Tensor 维度检查 | `torch::Tensor::dim()`, `is_contiguous()` | `ffi::Buffer::dimensions().size()` |
| 数据类型转换 | `at::cuda::ScalarTypeToCudaDataType` | `jax::FFIDataTypeToHIPDataType` |

---

## 八、局限性和未来方向

### 8.1 JAX 路径的限制

1. **仅 Intranode**：`num_rdma_bytes = 0`，`internode::*` 函数不可达。`deep_ep.cpp` 中有 TODO 注释。
2. **无独立 comm stream**：无法实现 PyTorch 的计算-通信重叠
3. **无 async 模式**：没有 EventHandle，无法精细控制流同步
4. **静态形状约束**：必须用 `num_worst_tokens` 作为输出大小，可能浪费内存
5. **Batching / transpose 未实现**：`BATCHING_TABLE` 和 `TRANSPOSE_TABLE` 为 TODO
6. **无 CUDA Graph**：XLA 有自己的图机制

### 8.2 PyTorch 路径的限制

1. **Low-Latency 未实现**：`clean_low_latency_buffer`、`low_latency_dispatch`、`low_latency_combine` 都是 stub（`PRIMUS_TURBO_CHECK(false)`）
2. **无内置 autograd**：不像 JAX 有 `custom_vjp`，需外部手动管理

### 8.3 共同限制

1. `intranode.cu` 中 NVLink 最多支持 `NUM_MAX_NVL_PEERS`（通常 8）个 GPU
2. FP8 支持依赖 ROCm 版本

---

## 九、文件路径快速索引

### PyTorch 路径
| 文件 | 作用 |
|------|------|
| `primus_turbo/pytorch/deep_ep/__init__.py` | 导出 Config, Buffer, EventOverlap |
| `primus_turbo/pytorch/deep_ep/buffer.py` | Python Buffer 类（1000 行） |
| `primus_turbo/pytorch/deep_ep/utils.py` | EventOverlap, EventHandle 封装 |
| `primus_turbo/pytorch/kernels/moe/moe_dispatch_combine_impl.py` | MoE 多后端调度 |
| `csrc/pytorch/deep_ep/deep_ep.cpp` | C++ Buffer 实现（1341 行） |
| `csrc/pytorch/deep_ep/deep_ep.hpp` | C++ Buffer 头文件 |
| `csrc/pytorch/deep_ep/event.hpp` | EventHandle + stream 工具 |

### JAX 路径
| 文件 | 作用 |
|------|------|
| `primus_turbo/jax/__init__.py` | FFI 注册入口 |
| `primus_turbo/jax/lax/moe/moe_dispatch_combine.py` | 用户 API + custom_vjp（411 行）|
| `primus_turbo/jax/lax/moe/moe_utils.py` | Config 定义 |
| `primus_turbo/jax/primitive/moe/moe_dispatch.py` | JAX Primitive 定义 |
| `primus_turbo/jax/primitive/moe/moe_combine.py` | JAX Primitive 定义 |
| `csrc/jax/deep_ep/deep_ep.cpp` | C++ Buffer + Dispatch/Combine 实现（505 行） |
| `csrc/jax/deep_ep/deep_ep.h` | C++ Buffer 头文件 |
| `csrc/jax/deep_ep/handler.cpp` | FFI Handler 定义（208 行） |

### 共享 Kernel
| 文件 | 作用 |
|------|------|
| `csrc/kernels/deep_ep/layout.cu` | 布局计算 kernel |
| `csrc/kernels/deep_ep/intranode.cu` | NVLink 通信 kernel |
| `csrc/kernels/deep_ep/internode.cu` | rocSHMEM RDMA kernel（仅 PyTorch 可用） |
| `csrc/kernels/deep_ep/runtime.cu` | rocSHMEM 生命周期管理 |
| `csrc/kernels/deep_ep/buffer.cuh` | GPU Buffer 内存布局定义 |
| `csrc/include/primus_turbo/deep_ep/api.h` | Kernel API 声明 |
| `csrc/include/primus_turbo/deep_ep/config.hpp` | Config + LowLatencyLayout |
| `csrc/include/primus_turbo/deep_ep/configs.h` | 常量定义 |
