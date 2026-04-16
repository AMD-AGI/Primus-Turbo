---
name: UCCL-EP JAX Internode
overview: "Risk-first approach: validate internode RDMA in single-process multi-GPU mode first, then progressively integrate JAX memory → FFI → Turbo. Each phase gates on the previous to avoid wasted effort."
todos:
  - id: p0-build
    content: "Phase 0.1 [DONE]: torch-free Makefile.rocm_jax + cuda_compat headers, build ep.abi3.so"
    status: completed
  - id: p0-sync
    content: "Phase 0.2 [DONE]: sync_same_process() for intranode pointer sharing (bypass hipIpcOpenMemHandle)"
    status: completed
  - id: p0-gil
    content: "Phase 0.3 [DONE]: nb::gil_scoped_release for intranode_prepare/dispatch/combine"
    status: completed
  - id: p0-intranode
    content: "Phase 0.4 [DONE]: Scheme A intranode 8-GPU test PASSED"
    status: completed
  - id: p1-destroy-fix
    content: "Phase 1.1: Fix destroy() crash — add destroy_same_process() that skips IPC close + coordinates barrier via caller-managed sync"
    status: pending
  - id: p1-rdma-proxy-init
    content: "Phase 1.2: Single-process RDMA Proxy init — create Buffer(rdma_bytes>0) + Proxy(is_intranode=False) for 8 GPUs; verify ibv_reg_mr succeeds per-GPU"
    status: pending
  - id: p1-sync-rdma
    content: "Phase 1.3: Extend sync_same_process() to handle RDMA buffer pointers (ipc_rdma_base_ptrs) + atomic buffer pointers"
    status: pending
  - id: p1-loopback-notify
    content: "Phase 1.4: Single-node loopback internode test — 16 virtual ranks on 8 GPUs, validate notify_dispatch + D2H queue → Proxy RDMA path"
    status: pending
  - id: p1-loopback-dispatch
    content: "Phase 1.5: Loopback internode dispatch+combine — full data path through RDMA sender/forwarder/receiver warps"
    status: pending
  - id: p2-jax-mem
    content: "Phase 2.1: JAX memory integration — jax.device_put → dlpack → ep API; validate uncached memory requirement"
    status: pending
  - id: p2-fp8
    content: "Phase 2.2: FP8 dtype on AMD — fix kFloat8E4M3 HIP path, validate E4M3 FNUZ dispatch/combine"
    status: pending
  - id: p3-ffi-handler
    content: "Phase 3.1: XLA FFI handler — C++ dispatch/combine FFI entry points with std::barrier coordination"
    status: pending
  - id: p3-jax-primitive
    content: "Phase 3.2: JAX primitive + abstract_eval — define dispatch_p/combine_p with internode output shapes"
    status: pending
  - id: p3-vjp
    content: "Phase 3.3: custom_vjp — dispatch↔combine gradient rules for internode"
    status: pending
  - id: p4-layer1
    content: "Phase 4.1: Layer 1 — split uccl_ep.cc → libuccl_ep.so (pure C++) + EpManager singleton"
    status: pending
  - id: p4-layer2
    content: "Phase 4.2: Layer 2 — Turbo FFI adapter linking libuccl_ep.so, add internode FFI handlers + Proxy lifecycle"
    status: pending
  - id: p4-layer3
    content: "Phase 4.3: Layer 3 — Turbo Python API: remove num_ranks<=8 limit, extend Config/handle for internode"
    status: pending
  - id: p4-feature-flag
    content: "Phase 4.4: Feature flag — env var to select UCCL-EP internode vs rocSHMEM (Turbo native) vs intranode-only"
    status: pending
  - id: p5-multinode
    content: "Phase 5.1: Multi-node RDMA validation — 2-node 16-GPU internode dispatch/combine"
    status: pending
  - id: p5-maxtext-e2e
    content: "Phase 5.2: MaxText MoE e2e — end-to-end training with internode EP on multi-node cluster"
    status: pending
isProject: false
---

# UCCL-EP JAX Internode DeepEP — 更新方案 (Risk-First)

## 核心策略变更

原方案按 Scheme A → B → C 线性推进，将 internode 验证推迟到集成阶段。
**问题**：internode RDMA 是整个方案的最大技术风险，如果 Phase 4 才发现不可行，前面所有工作白费。

**新策略**：**Risk-First（风险前置）**——先在 Phase 1 用最小代价验证 internode RDMA 核心路径的可行性，再逐步叠加 JAX 集成和 Turbo 集成。

```
Phase 0 (已完成)     Phase 1 (核心风险)    Phase 2 (JAX 内存)   Phase 3 (JAX 原生)   Phase 4 (Turbo)     Phase 5 (生产)
┌──────────────┐  ┌─────────────────┐  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  ┌──────────────┐
│ Build system │  │ destroy() fix   │  │ jax.device_put│   │ XLA FFI      │   │ libuccl_ep.so│  │ 2-node RDMA  │
│ sync_same_   │→ │ RDMA Proxy init │→ │ dlpack interop│ → │ JAX primitive│ → │ Turbo adapter│→ │ MaxText e2e  │
│ process      │  │ RDMA sync       │  │ FP8 on AMD   │   │ custom_vjp   │   │ Feature flag │  │              │
│ Intranode OK │  │ Loopback test   │  │              │   │              │   │              │  │              │
└──────────────┘  └─────────────────┘  └──────────────┘   └──────────────┘   └──────────────┘  └──────────────┘
      DONE           GATE: 能否在单        GATE: JAX 内存       GATE: XLA 并发        GATE: 构建系统
                     进程中跑通 RDMA?      是否兼容 uncached?    调度是否正确?         集成是否通?
```

**每个 Phase 末尾有 Gate**：如果不通过，在单进程约束下寻找替代技术方案（见 Gate Decision 和 Contingency）。

---

## 硬约束：必须使用单进程多 GPU 模式

MaxText 的 GPU 部署虽有 `hardware: 'gpu'`（单进程多 GPU，**默认**）和 `hardware: 'gpu_multiprocess'`（多进程单 GPU）两种模式，但 **必须使用单进程模式**，原因如下：

| 因素 | 分析 |
|------|------|
| MaxText 主路径 | `gpu` 模式是 Google GPU CI 的主测试路径；`gpu_multiprocess` 不被充分保护 |
| ROCm fork 验证 | `ROCm/maxtext` 仅在 `gpu` 模式下验证过，`gpu_multiprocess` 在 AMD 上是否可靠未知 |
| 上游 breakage 风险 | MaxText 迭代极快（Gemma 4, DeepSeek Engram 等），非主路径的模式可能在任何版本更新中被悄悄打破 |
| 路线不可控 | MaxText 由 Google 主导，我们无法影响其测试矩阵和发布节奏 |
| 并行策略耦合 | 切换进程模型影响 MaxText 全部并行原语（TP/DP/PP 的 shard_map/pjit），非局部改动 |

**结论**：所有技术方案必须在单进程多 GPU 约束下设计。Phase 1 Gate 失败时的 contingency 也不允许切换为多进程模式，而是在单进程框架内寻找 UCCL-EP Proxy 层的替代解法。

---

## Environment

| Item | Value |
|------|-------|
| Machine | smc300x-ccs-aus-a17-10 (单节点，8× MI300X) |
| Container | `llying_jax_2601` (rocm/jax-training:maxtext-v26.1) |
| JAX | 0.8.2 |
| UCCL-EP code | `/apps/tas/liying/code/jax-deepep/uccl` |
| UCCL-EP upstream | `/apps/tas/liying/code/adapt_te/uccl_ep_fork/uccl` |
| Primus-Turbo | `/apps/tas/liying/code/adapt_te/Primus-Turbo` |
| 约束 | 目前只有 a17-10 单节点；多节点测试需另行协调 |

---

## Phase 0: Foundation（已完成）

| Step | Status | 关键产出 |
|------|--------|----------|
| 0.1 Makefile.rocm_jax + cuda_compat | DONE | `ep.abi3.so` (17.3MB, gfx942) |
| 0.2 sync_same_process() | DONE | 绕过 hipIpcOpenMemHandle |
| 0.3 GIL release | DONE | nanobind 4 处绑定加 `nb::gil_scoped_release` |
| 0.4 Intranode 8-GPU test | DONE | layout + prepare + dispatch + combine 全通过 |

**关键发现**:
1. `hipIpcOpenMemHandle` 在同进程中失败 → 必须直接指针共享
2. Python GIL + CPU spin-wait → 死锁 → 需 `nb::gil_scoped_release`
3. `destroy()` crash → 待修复（Phase 1.1）

---

## Phase 1: Internode Core Risk Validation（最高优先级）

> **目标**: 在单节点上验证单进程多 GPU 模式下 UCCL-EP 的完整 internode RDMA 路径，包括 Proxy 初始化、RDMA buffer 注册、D2H queue → CPU Proxy → ibverbs 全链路。

### 1.1 Fix destroy() crash

**问题根因分析**:

当前 `Buffer::destroy()` (uccl_ep.cc:560-624) 有两个与单进程模式不兼容的操作：

```
destroy() 流程:
  1. intranode::barrier(...)        ← 需所有 GPU 同时参与，单线程串行调用会死锁
  2. cudaIpcCloseMemHandle(...)     ← sync_same_process 未使用 IPC，close 会出错
  3. cudaFree(buffer_ptrs[nvl_rank])
  4. cudaIpcCloseMemHandle(ipc_rdma_base_ptrs[i])  ← 同上
```

**解决方案**: 新增 `destroy_same_process()` 方法：

```cpp
void destroy_same_process() {
    // 1. 跳过 intranode::barrier — 由调用方（Python threading.Barrier）协调
    // 2. 跳过 cudaIpcCloseMemHandle — 我们没有使用 IPC
    // 3. 直接 cudaFree(buffer_ptrs[nvl_rank]) 释放本 GPU 的 NVL buffer
    // 4. 不 close ipc_rdma_base_ptrs — 同进程直接指针无需 close
    // 5. 其余资源释放与 destroy() 相同（workspace, counters, streams, d2h handles）
    CUDA_CHECK(cudaSetDevice(device_index));
    if (num_nvl_bytes > 0) {
        CUDA_CHECK(cudaFree(buffer_ptrs[nvl_rank]));
    }
    // ... 释放 workspace, moe_recv_counter, d_handle_objs, comm_stream 等
    destroyed = true;
    available = false;
}
```

**验证**: 8-GPU 测试应能正常退出，无 HSA_STATUS_ERROR_EXCEPTION。

**预估工时**: 0.5 天

---

### 1.2 Single-process RDMA Proxy Init

**核心问题**: UCCL-EP 的 Proxy 初始化路径 (`proxy.cpp::init_common()` → `rdma.cpp::per_thread_rdma_init()`) 是否能在同一进程中为不同 GPU 正确工作？

**需要验证的关键点**:

| 关注点 | 原始（多进程）行为 | 单进程中的预期行为 | 风险 |
|--------|-------------------|-------------------|------|
| `cudaSetDevice(gpu_idx)` | 每进程只有 1 个 GPU | 同一进程内切换 device | **低** — HIP 支持 |
| `ibv_open_device(nic)` | 每进程打开 1 个 NIC | 同进程多线程各打开 NIC | **中** — 需验证并发 |
| `ibv_reg_mr(gpu_buf)` | 注册 1 块 GPU memory | 注册 8 块不同 GPU 的 memory | **高** — 核心风险 |
| `ibv_create_qp` × N_peers | 每进程创建 peer 数个 QP | 同进程创建 8×peer 数个 QP | **中** — 资源限制 |
| TCP info exchange | 每进程独立 TCP 连接 | 同进程 32 线程的端口分配 | **中** — 端口冲突 |

**实施步骤**:

```python
# test_rdma_proxy_init.py — 单进程 RDMA Proxy 初始化测试
rdma_bytes = 1 << 24  # 16MB RDMA buffer per GPU

for gpu_id in range(8):
    hip_set_device(gpu_id)
    rdma_buf, is_host = ep.allocate_rdma_buffer(rdma_bytes, gpu_id)  # 分配 RDMA buffer
    
    proxies = []
    for t in range(num_proxy_threads):
        proxy = ep.Proxy(
            thread_idx=t,
            gpu_buffer_addr=rdma_buf_ptr,
            total_size=rdma_bytes,
            rank=gpu_id,
            node_idx=0,
            local_rank=gpu_id,
            num_experts=8,
            num_ranks=16,       # 模拟 2 节点
            num_nodes=2,        # 模拟 2 节点
            use_normal_mode=True,
            is_intranode=False,  # 关键: 启用 RDMA 路径
        )
        proxies.append(proxy)
    ep.register_proxies(gpu_id, proxies)
    
    buf = ep.Buffer(
        rank=gpu_id,
        num_ranks=16,           # 2 nodes × 8 GPUs
        num_nvl_bytes=nvl_bytes,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=False,
        explicitly_destroy=True,
        num_local_ranks=8,
    )
    buf.set_rdma_buffer(rdma_buf_ptr, is_host)
```

**Pass Criteria**: 所有 8 个 GPU 的 Proxy 成功初始化，`ibv_reg_mr` 返回成功，无段错误。

**预估工时**: 1-2 天

---

### 1.3 Extend sync_same_process() for RDMA

当前 `sync_same_process()` 只处理了 NVL 区域指针。Internode 场景需要额外同步：

| 指针 | 用途 | 同步方式 |
|------|------|----------|
| `buffer_ptrs[i]` | NVL data + barrier signal (已实现) | 直接赋值 ✅ |
| `barrier_signal_ptrs[i]` | GPU atomic barrier (已实现) | 从 buffer_ptrs 偏移 ✅ |
| `ipc_rdma_base_ptrs[i]` | 同节点内其他 GPU 的 RDMA buffer | **需新增** |
| `d_ipc_rdma_base_ptrs` | GPU 侧的 RDMA 指针数组 | **需新增** |
| `atomic_buffer_ptr` | Proxy 原子操作 buffer | **需新增** (如果 internode 使用) |

**修改方案**:

```cpp
void sync_same_process(
    std::vector<int> const& device_ids,
    std::vector<std::uintptr_t> const& all_buffer_ptrs,
    std::vector<std::uintptr_t> const& all_rdma_buffer_ptrs = {},   // 新增
    std::vector<std::uintptr_t> const& all_atomic_buffer_ptrs = {}  // 新增
) {
    // ... 现有 NVL 指针同步逻辑 ...
    
    // 新增: RDMA buffer 指针同步
    if (num_rdma_bytes > 0 && !all_rdma_buffer_ptrs.empty()) {
        EP_HOST_ASSERT(all_rdma_buffer_ptrs.size() == num_ranks);
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            int global_rank = offset + i;
            int local_rank_idx = global_rank % max_nvl_peers;
            ipc_rdma_base_ptrs[local_rank_idx] = 
                reinterpret_cast<void*>(all_rdma_buffer_ptrs[global_rank]);
        }
        if (d_ipc_rdma_base_ptrs != nullptr) {
            CUDA_CHECK(cudaMemcpy(d_ipc_rdma_base_ptrs, ipc_rdma_base_ptrs,
                                  sizeof(void*) * max_nvl_peers, cudaMemcpyHostToDevice));
        }
    }
    
    // 新增: Atomic buffer 指针同步 (internode receiver barrier)
    if (!all_atomic_buffer_ptrs.empty()) {
        // ... 类似逻辑 ...
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    available = true;
}
```

**预估工时**: 1 天

---

### 1.4 Single-node Loopback Internode Test — notify_dispatch

**方案**: 在单节点上模拟 2 节点拓扑 (16 ranks = 2 nodes × 8 GPUs)。

虽然物理上只有 1 个节点，但可以让 rank 0-7 视为 node 0，rank 8-15 也映射到相同的 8 个 GPU，形成 "loopback" 拓扑。

**关键约束**:
- `rank 8` 和 `rank 0` 实际在同一个 GPU 上——这意味着需要为同一个 GPU 创建 2 个 Buffer（不同 rank）
- Proxy 的 TCP info exchange 需要与 "远程节点" 握手——loopback 时远程就是自己

**备选方案（如果 loopback 太复杂）**:
仅验证 internode **发送侧** 的完整路径：kernel 写 D2H queue → CPU Proxy 从 queue 读取 → 尝试 `ibv_post_send`。即使没有远程接收端，也能验证：
1. D2H queue 的 GPU→CPU 通信是否正常
2. Proxy 能否正确解码命令
3. `ibv_post_send` 的参数是否正确构造

**具体测试**:

```python
# 方案A: 完整 loopback (优先尝试)
# 16 virtual ranks, 2 virtual nodes, 8 physical GPUs
# rank i 和 rank i+8 共用 GPU i

# 方案B: 发送侧验证 (fallback)
# 8 ranks on node 0, Proxy 初始化但无远端
# 验证 notify_dispatch 能否正确将 metadata 推入 D2H queue
# Proxy 日志确认收到 WRITE 命令
```

**Pass Criteria**:
- 方案A: `notify_dispatch` 完成，各 rank 正确交换 `num_tokens_per_rank` 和 `num_tokens_per_expert`
- 方案B: Proxy 日志显示从 D2H queue 中正确读取到 WRITE 命令

**预估工时**: 3-5 天（含调试）

---

### 1.5 Loopback Internode dispatch + combine

在 1.4 的基础上，运行完整的 internode dispatch 和 combine 内核：

```
get_dispatch_layout → internode_prepare (notify_dispatch) 
    → internode_dispatch (RDMA data send + NVL forward)
    → internode_combine (reverse path)
```

**Pass Criteria**:
- dispatch 后各 rank 收到正确数量的 token
- combine 后数据完整性检查通过（数值正确性）

**预估工时**: 3-5 天

---

### Phase 1 Gate Decision

> **硬约束**：不允许回退到多进程模式（MaxText 的 `hardware: 'gpu'` 单进程多 GPU 是唯一可用路径）。所有 contingency 必须在单进程框架内解决。

| 结果 | 决策 |
|------|------|
| **全部通过** | 进入 Phase 2，单进程 internode 可行 ✅ |
| **1.2 失败: `ibv_reg_mr` 拒绝多 GPU 内存** | → Contingency A: 修改 Proxy，使每个 Proxy 线程在 `cudaSetDevice(own_gpu)` 后独立调用 `ibv_reg_mr` 注册自己 GPU 的 buffer（而非由其他 GPU 的线程注册）。若仍失败 → Contingency B: 使用 `hipMallocManaged` 分配 RDMA buffer（统一虚拟地址，单次 `ibv_reg_mr` 注册整个范围）。若仍失败 → Contingency C: 使用 dmabuf fd 方式注册 GPU memory（`ibv_reg_dmabuf_mr`，需 ROCm 6.x+ 和 OFED 5.5+ 支持） |
| **1.4 失败: loopback 拓扑限制** | → 不阻塞 Gate。标记为 "partial validation"，申请多节点环境在 Phase 5 补全。关键判断点是 1.2（Proxy init + MR 注册）是否成功 |
| **1.4/1.5 失败: D2H queue 或 Proxy 命令解码错误** | → 在单进程内调试 D2H queue 的 GPU→CPU 通路。可能原因：(a) `cudaSetDevice` 竞态——加 per-GPU mutex；(b) D2H channel 地址归属混乱——确保 `collect_d2h_channel_addrs_for_device` 按 device_index 正确分组 |
| **1.5 失败: internode kernel 运算错误但 Proxy 正常** | → kernel 层面问题，与单进程模型无关。参考 upstream UCCL-EP 的 internode 测试用例逐步排查 |

---

## Phase 2: JAX Memory Integration

> **前提**: Phase 1 Gate 通过

### 2.1 JAX 内存 → UCCL-EP 兼容性

**核心问题**: JAX 的 `jax.device_put` 使用 `hipMalloc` 分配标准 GPU 内存，但 UCCL-EP 的 NVL buffer 需要 `hipExtMallocWithFlags(hipDeviceMallocUncached)` 分配 uncached 内存，两者不兼容。

**方案选择**:

| 方案 | 描述 | 可行性 |
|------|------|--------|
| A: UCCL-EP 自行分配 | NVL/RDMA buffer 仍由 UCCL-EP 分配（uncached），JAX 只分配 input/output tensor | **推荐** — 与 Turbo 现有模式一致 |
| B: JAX 自定义 allocator | 在 JAX 中注册自定义 memory allocator 返回 uncached 内存 | 复杂，侵入性大 |
| C: 混合模式 | UCCL-EP 的数据 buffer 用 JAX 内存，barrier signal 用 uncached 内存 | 需验证 atomics 兼容性 |

**推荐方案 A 的实现**:
- NVL/RDMA scratch buffer 由 UCCL-EP Buffer 构造函数分配（已有）
- Input tensor (`x`) 和 output tensor (`recv_x`, `combined_x`) 使用 JAX 分配的标准内存
- 通过 `dlpack` 将 JAX array 的设备指针传递给 ep API
- 验证标准 GPU 内存（非 uncached）作为 input/output 时的 dispatch/combine 正确性

**测试脚本**:

```python
import jax
import jax.numpy as jnp

# JAX 分配 input tensor
x = jax.device_put(jnp.ones((num_tokens, hidden), dtype=jnp.bfloat16), jax.devices()[gpu_id])
x_ptr = x.addressof()  # 或通过 dlpack 获取

# UCCL-EP 分配 scratch buffer (uncached)
buf = ep.Buffer(rank=gpu_id, ..., num_nvl_bytes=nvl_bytes)

# 混合使用
buf.intranode_dispatch(x_ptr, ...)
```

**预估工时**: 2-3 天

### 2.2 FP8 Dtype on AMD

当前 `cuda_dtype_from_code()` 在 AMD 上跳过了 `kFloat8E4M3`。修复：

```cpp
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    case kFloat8E4M3:
      return HIP_R_8F_E4M3_FNUZ;  // MI300X 使用 E4M3 FNUZ 格式
#else
    case kFloat8E4M3:
      return CUDA_R_8F_E4M3;
#endif
```

注意 MI300X 的 E4M3 FNUZ (max=240.0) 与 NVIDIA E4M3 FN (max=448.0) 位格式不兼容，但 EP kernel 内部已处理此差异（upstream `is_sm90_compiled()` 在 HIP 上返回 true）。

**预估工时**: 0.5 天

---

## Phase 3: JAX Native Integration (shard_map + FFI)

> **前提**: Phase 2 Gate 通过

### 3.1 XLA FFI Handler

参考 Turbo 现有的 JAX intranode FFI handler (`csrc/jax/deep_ep/deep_ep.cpp`)，新增 internode 路径。

**架构决策**: 是否复用 Turbo 的 FFI 框架，还是独立实现？

| 方案 | 优点 | 缺点 |
|------|------|------|
| A: 独立 FFI handler (先) | 快速验证，不依赖 Turbo 构建系统 | 后续需要合并到 Turbo |
| B: 直接在 Turbo 中开发 | 一步到位 | 构建系统集成复杂，调试周期长 |

**推荐**: 先走 A（独立验证），再在 Phase 4 合并到 Turbo。

**关键设计**: FFI handler 中的多设备同步

```cpp
// 类似 Turbo 的 g_buffer_pool + std::barrier 模式
static std::unordered_map<int, std::shared_ptr<UcclEpBuffer>> g_uccl_buffer_pool;
static std::barrier g_uccl_barrier(8);  // 动态初始化为 num_local_gpus

XLA_FFI_DEFINE_HANDLER(MoEDispatchHandler, MoEDispatch,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::BF16>>()  // x
        .Arg<ffi::BufferR1<ffi::S64>>()   // topk_idx
        // ... more args
        .Ret<ffi::BufferR2<ffi::BF16>>()  // recv_x
);

ffi::Error MoEDispatch(/* args */) {
    int device_id = /* from XLA context */;
    auto& buf = g_uccl_buffer_pool[device_id];
    
    // XLA 会为每个 GPU 并发调用此 handler
    // std::barrier 确保所有 GPU 都到达后再继续
    g_uccl_barrier.arrive_and_wait();
    
    // 调用 UCCL-EP 的 internode_dispatch / intranode_dispatch
    if (num_ranks > num_local_gpus) {
        buf->internode_dispatch(...);
    } else {
        buf->intranode_dispatch(...);
    }
    
    return ffi::Error::Success();
}
```

**预估工时**: 5-7 天

### 3.2 JAX Primitive + abstract_eval

定义 `dispatch_p` 和 `combine_p` primitive，扩展 abstract_eval 支持 internode 输出 shape：

```python
@dispatch_p.def_abstract_eval
def dispatch_abstract_eval(x, topk_idx, *, num_experts, num_ranks, ...):
    if num_ranks > local_device_count:
        # internode: 需要额外的 rdma_channel_prefix_matrix 等输出
        return (
            ShapedArray((max_recv_tokens, hidden), x.dtype),  # recv_x
            ShapedArray((num_ranks,), jnp.int32),              # num_tokens_per_rank
            ShapedArray((num_rdma_ranks,), jnp.int32),         # num_tokens_per_rdma_rank
            # ... 更多 internode 特有输出
        )
    else:
        # intranode: 现有逻辑
        ...
```

**预估工时**: 3-5 天

### 3.3 custom_vjp

dispatch 的反向即 combine，combine 的反向即 dispatch：

```python
@jax.custom_vjp
def moe_dispatch(x, topk_idx, ...):
    return _dispatch_fwd(x, topk_idx, ...)

def _dispatch_fwd(x, topk_idx, ...):
    recv_x, handle = dispatch_p.bind(x, topk_idx, ...)
    return recv_x, handle  # handle 包含 prefix_matrix 等中间结果

def _dispatch_bwd(handle, g):
    return combine_p.bind(g, handle, ...)
```

**预估工时**: 2-3 天

---

## Phase 4: Primus-Turbo Integration

> **前提**: Phase 3 Gate 通过

### 三层架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3: Turbo Python API                                       │
│  moe_dispatch_combine.py — auto-select intranode/internode       │
│  Remove assert num_ranks <= 8; extend Config for internode       │
└──────────────────────────┬───────────────────────────────────────┘
                           │ jax.ffi.ffi_lowering → XLA FFI
┌──────────────────────────▼───────────────────────────────────────┐
│  Layer 2: Turbo C++ FFI Adapter                                  │
│  handler.cpp — route by num_ranks; manage UCCL-EP lifecycle      │
│  EpManager — singleton, per-GPU Buffer/Proxy pool + std::barrier │
└──────────────────────────┬───────────────────────────────────────┘
                           │ Direct C++ API calls
┌──────────────────────────▼───────────────────────────────────────┐
│  Layer 1: libuccl_ep.so                                          │
│  Pure C++ (no nanobind/Python) — EpManager, Buffer, Proxy        │
│  sync_same_process (NVL + RDMA) + internode kernels + CPU proxy  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.1 Layer 1: libuccl_ep.so

**核心工作**:

1. 拆分 `uccl_ep.cc`:
   - `uccl_ep_core.cc` — Buffer 类 + 所有计算方法（纯 C++，无 nanobind）
   - `uccl_ep_bindings.cc` — nanobind 包装（仅 PoC 使用）
   - `include/uccl_ep_api.h` — 公开 C++ API 头文件

2. 创建 `EpManager` 单例:

```cpp
class EpManager {
public:
    static EpManager& instance();
    
    // 初始化: 为每个 GPU 创建 Proxy + Buffer
    void initialize(int num_local_gpus, int num_nodes, int node_idx,
                    int64_t nvl_bytes, int64_t rdma_bytes, ...);
    
    // 获取指定 GPU 的 Buffer
    Buffer& buffer(int local_gpu_id);
    
    // 同步所有 Buffer（调用 sync_same_process）
    void sync_all();
    
    // 关闭所有 Proxy 和 Buffer
    void shutdown();
    
private:
    std::vector<std::unique_ptr<Buffer>> buffers_;
    std::vector<std::vector<Proxy>> proxies_;
    std::barrier<> barrier_;
};
```

3. 构建系统: 生成 `libuccl_ep.so` (供 Turbo 链接) + `ep.abi3.so` (供 Python PoC 使用)

**预估工时**: 5-7 天

### 4.2 Layer 2: Turbo FFI Adapter

1. Turbo 的 `CMakeLists.txt` / `setup.py` 链接 `libuccl_ep.so`
2. 新增 `csrc/jax/deep_ep/uccl_ep_adapter.h/cpp`:
   - 包装 `EpManager` 调用
   - 提供 `initialize_ep()` / `shutdown_ep()` 的 pybind11 绑定
3. 修改 `handler.cpp`:
   - 当 `num_ranks > NUM_MAX_NVL_PEERS` 时路由到 internode handler
   - Internode handler 调用 `EpManager::instance().buffer(device_id).internode_dispatch(...)`

**预估工时**: 5-7 天

### 4.3 Layer 3: Turbo Python API

1. `moe_dispatch.py`: 移除 `assert num_ranks <= 8`
2. `moe_dispatch_combine.py`: 扩展 handle tuple，增加 internode 特有字段
3. `config.py`: 增加 internode 配置项 (`num_max_rdma_chunked_send/recv_tokens` 等)
4. `__init__.py`: 当检测到 `DEEP_EP_NUM_RANKS > local_device_count` 时调用 `initialize_ep()`

**预估工时**: 3-5 天

### 4.4 Feature Flag

通过环境变量控制通信后端:

```python
# DEEP_EP_BACKEND=uccl     → 使用 UCCL-EP (支持 internode)
# DEEP_EP_BACKEND=rocshmem → 使用 rocSHMEM (Turbo 原生)
# DEEP_EP_BACKEND=auto     → num_ranks <= 8 用 intranode, > 8 用 UCCL-EP
```

**预估工时**: 1 天

---

## Phase 5: Multi-node End-to-End

### 5.1 Multi-node RDMA Validation

- 2 节点 × 8 GPU = 16 ranks
- 验证完整的 internode dispatch/combine 路径
- 验证 RDMA 吞吐和延迟

### 5.2 MaxText MoE E2E

- 在 MaxText 中启用 EP=16 (2 nodes)
- 训练 MoE 模型，验证 loss 收敛和通信正确性
- 性能对比: UCCL-EP internode vs 无 EP (all-to-all)

**预估工时**: 视多节点环境可用性而定

---

## 与原方案的对比

| 维度 | 原方案 | 新方案 |
|------|--------|--------|
| 进程模型约束 | Gate 失败可回退多进程 | **单进程多 GPU 是硬约束**，Gate 失败走 Contingency |
| 风险验证时机 | Phase 4 (集成阶段) | **Phase 1** (最早期) |
| Internode 首次测试 | 多节点才能测 | **单节点 loopback** |
| destroy crash | 标注为 "已知问题，后续处理" | **Phase 1.1 立即修复** |
| RDMA sync | 未涉及 | **Phase 1.3 明确定义** |
| FP8 | 未涉及 | **Phase 2.2 明确解决** |
| Feature flag | 未涉及 | **Phase 4.4 回退机制** |
| Gate decision | 无 | **每个 Phase 有 Go/No-Go + 单进程内 Contingency** |
| Scheme B/C | 线性前进 | 与 Phase 1 **可并行** (JAX 内存验证不依赖 internode) |
| 风险升级路径 | 无 | 每个风险有 A→B→C 逐级技术升级方案 |

---

## 关键风险与缓解

> **原则**: 所有缓解方案均在单进程多 GPU 约束下设计，不涉及切换为多进程模式。

| # | 风险 | 影响 | 缓解 | 升级路径 |
|---|------|------|------|---------|
| R1 | `ibv_reg_mr` 拒绝注册非本 GPU 的内存 | Phase 1.2 失败，internode 不可用 | A: 确保每个 Proxy 线程在自己的 GPU context 下调用 `ibv_reg_mr`；B: 使用 `hipMallocManaged` 分配 RDMA buffer | C: 使用 `ibv_reg_dmabuf_mr` (ROCm 6.x+)；D: 将 RDMA buffer 改为 host memory (`cudaMallocHost`) |
| R2 | 单进程 32 Proxy 线程 + XLA 线程池资源竞争 | 性能下降 | 减少 Proxy 线程数 (4→2)；`pin_thread=True` + NUMA affinity；设置 `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` 减少 XLA CPU 线程 | 动态调整 Proxy 线程数：intranode 时 0 线程，internode 时按需启动 |
| R3 | Loopback 拓扑无法模拟真实 internode | Phase 1.4/1.5 不充分 | 标记为 "partial validation"——Phase 1 重点验证 Proxy init + MR 注册 + D2H queue 通路，完整 RDMA 数据通路在 Phase 5 多节点补全 | 申请多节点环境 |
| R4 | RDMA buffer 脱离 JAX allocator 导致 OOM | 运行时内存不足 | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` 预留 20% 给 UCCL-EP；或 `XLA_PYTHON_CLIENT_ALLOCATOR=platform` 让 JAX 按需分配 | 实现 UCCL-EP buffer 与 JAX allocator 的协调通知 |
| R5 | `hipExtMallocWithFlags(Uncached)` 与 ibverbs page 对齐不兼容 | MR 注册失败 | 手动 `mmap` + `ibv_reg_mr` 验证对齐要求；若不兼容用 `hipMallocManaged` | 参考 UCCL-EP upstream 的 `can_register_gpu_memory_for_rdma()` 检测逻辑 |
| R6 | AMD internode sequential lock 导致高延迟 | 性能不达标 | 评估 upstream `UCCL_EP_ENABLE_AGGRESSIVE_ATOMIC` 优化；分析 lock contention 热点 | 与 upstream 合作优化 AMD 锁策略 |
| R7 | TCP info exchange 端口冲突（同进程 32 线程同时 listen） | Proxy 初始化失败 | 使用递增端口号 `base_port + gpu_id * num_proxy_threads + thread_idx`；或改用共享内存交换 QP info（同节点内不需要 TCP） | 为 Proxy 添加 `port_offset` 配置项 |

---

## 时间线估计

| Phase | 预估工时 | 累计 | 前置依赖 |
|-------|---------|------|---------|
| Phase 1 (Internode 风险) | 8-13 天 | 8-13 天 | Phase 0 ✅ |
| Phase 2 (JAX 内存) | 3-4 天 | 11-17 天 | Phase 1 Gate |
| Phase 3 (JAX FFI) | 10-15 天 | 21-32 天 | Phase 2 Gate |
| Phase 4 (Turbo 集成) | 14-20 天 | 35-52 天 | Phase 3 Gate |
| Phase 5 (多节点 e2e) | TBD | TBD | 多节点环境 + Phase 4 |

**注意**: Phase 2 可与 Phase 1.4/1.5 部分并行（JAX 内存兼容性不依赖 internode 完成）。

---

## 可并行的工作流

```
Timeline:
  Week 1-2:  Phase 1.1 + 1.2 + 1.3 (destroy fix + RDMA init + sync)
             ├── 同时: Phase 2.1 开始 (JAX 内存兼容性预研)
  Week 2-3:  Phase 1.4 + 1.5 (loopback internode test)
             ├── 同时: Phase 2.2 (FP8 fix)
  Week 3:    Phase 1 Gate Decision
  Week 3-4:  Phase 2 完成 + Phase 3.1 开始 (FFI handler)
  Week 4-6:  Phase 3 完成
  Week 6-9:  Phase 4 (Turbo 集成)
  Week 10+:  Phase 5 (多节点，取决于环境)
```

---

## Progress Log

### [2025-03-31] Phase 0 Complete: Intranode 8-GPU PASSED

- All intranode operations validated (layout, prepare, dispatch, combine)
- Known issue: `destroy()` crash → Phase 1.1
- Report: `ep/docs/poc-report-uccl-ep-jax-single-process.md`

### [2025-03-30] Key Discoveries

1. `hipIpcOpenMemHandle` fails within same process → `sync_same_process()`
2. Python GIL + CPU spin-wait → deadlock → `nb::gil_scoped_release`
3. `DISABLE_SM90_FEATURES` mandatory on AMD

### cuda_compat headers consolidation note

POC 创建了独立的 `ep/include/cuda_compat/` (~7 files, ~140 macros)，但 UCCL-EP 上游已有 `include/util/gpu_rt.h`。
**Action**: Phase 4.1 (Layer 1 refactor) 时统一为一套兼容层，优先使用上游 `gpu_rt.h`，仅做增量补充。
