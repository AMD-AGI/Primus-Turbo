# ROCm HIP GPU 执行模型与常见陷阱

> 本文面向从 CUDA 迁移到 AMD ROCm/HIP 平台的开发者，系统介绍 AMD GPU 的执行模型与 NVIDIA GPU 的核心差异，以及由此引发的常见 bug 模式和最佳实践。

## 目录

- [1. 基础概念：GPU 的并行层级](#1-基础概念gpu-的并行层级)
- [2. 锁步执行（Lockstep Execution）](#2-锁步执行lockstep-execution)
- [3. EXEC Mask 机制](#3-exec-mask-机制)
- [4. 分支分化（Branch Divergence）](#4-分支分化branch-divergence)
- [5. AMD vs NVIDIA 线程调度模型对比](#5-amd-vs-nvidia-线程调度模型对比)
- [6. 关键 API 在 AMD 上的行为差异](#6-关键-api-在-amd-上的行为差异)
- [7. 常见 Bug 模式与案例分析](#7-常见-bug-模式与案例分析)
- [8. 最佳实践](#8-最佳实践)

---

## 1. 基础概念：GPU 的并行层级

GPU 程序的并行分为三个层级：

```
Grid（网格）
 └── Block / Workgroup（线程块 / 工作组）
      └── Warp / Wavefront（线程束 / 波前）
           └── Thread / Lane（线程 / 通道）
```

| 概念 | NVIDIA 术语 | AMD 术语 | 大小 |
|------|------------|----------|------|
| 最小调度单位 | Warp | Wavefront | NVIDIA: 32 线程, AMD: **64 线程** |
| 线程块 | Thread Block | Workgroup | 用户指定（通常 128~1024） |
| 线程块内的线程编号 | `threadIdx.x` | `threadIdx.x`（HIP 兼容） | — |
| warp 内的线程编号 | `lane_id`（0~31） | `lane_id`（0~63） | — |

> **重要**：AMD 的 wavefront 大小是 64（CDNA 架构），是 NVIDIA warp 大小的两倍。这意味着所有涉及 warp 大小的常量、mask 都需要调整。

---

## 2. 锁步执行（Lockstep Execution）

### 2.1 什么是锁步执行

**锁步执行 = wavefront 内所有线程共享一个程序计数器（PC），每个时钟周期执行同一条指令。**

可以用队列行进来类比：

- 想象一个 64 人方阵，教官喊口令
- 教官喊"向前走"，64 人同时迈左脚
- 教官喊"举枪"，64 人同时举枪
- **整个方阵永远在同一个动作上**
- 如果命令只针对某些人（如"前排举枪"），后排的人也"执行"这个动作，只是手里没枪（被屏蔽了），不产生实际效果

用表格展示一段代码的执行过程：

```cpp
if (lane_id == 0) {
    a = heavy_computation();   // A
} else {
    b = simple_work();         // B
}
c = a + b;                     // C
```

```
时钟  | PC 指向 | EXEC mask       | 实际发生什么
------+---------+-----------------+----------------------------------
  1   |    A    | 0x000...001     | 只有 lane 0 执行 heavy_computation
      |         | (仅 lane 0)     | 其余 63 个线程"陪跑"，不产生效果
  2   |    A    |      同上        | (heavy_computation 耗多个周期)
  3   |    A    |      同上        | ...
  4   |    B    | 0xFFF...FFE     | lane 1~63 执行 simple_work
      |         | (除 lane 0 外)  | lane 0 "陪跑"
  5   |    C    | 0xFFF...FFF     | 全部 64 线程执行 c = a + b
      |         | (全部)          | 重新汇合
```

### 2.2 锁步执行的关键推论

1. **不存在"某些线程跑在前面"**：wavefront 内所有线程永远在同一条指令上
2. **分支的两侧串行执行**：`if` 和 `else` 不是并行的，而是先执行一侧、再执行另一侧
3. **分支的总耗时 = 两侧之和**，不是取最大值
4. **被屏蔽的线程无法"绕过"当前代码独立前进**：如果 lane 0 在自旋循环里，其他 63 个线程也被困在里面

---

## 3. EXEC Mask 机制

### 3.1 EXEC 寄存器

EXEC 是 AMD GPU 的一个 **64-bit 硬件寄存器**，每一位对应 wavefront 中的一个 lane：

```
EXEC = 0b 1111...1111 1111...1111    → 64 个 lane 全部活跃
EXEC = 0b 0000...0000 0000...0001    → 只有 lane 0 活跃
EXEC = 0b 0000...0000 0000...0000    → 全部屏蔽
```

### 3.2 EXEC 如何控制执行

| 指令类型 | EXEC 的作用 |
|----------|------------|
| 向量 ALU 指令（`v_add_f32` 等） | 只有 EXEC 中为 1 的 lane 写入目标 VGPR |
| 向量访存指令（`global_load_dword` 等） | 只有活跃 lane 发出内存请求 |
| 跨 lane 操作（`ds_bpermute_b32` 等） | **只有活跃 lane 的目标寄存器被写入** |
| 标量指令（`s_barrier` 等） | **不受 EXEC 影响**，wavefront 级别执行 |

### 3.3 EXEC 的保存与恢复

编译器通过 `s_and_saveexec_b64` 和 `s_or_b64` 指令管理 EXEC：

```
// 进入 if (cond) 分支
s_and_saveexec_b64  saved, cond_mask
// saved = 旧 EXEC（保存）
// EXEC  = 旧 EXEC & cond_mask（只保留满足条件的 lane）

... // if 分支体

// 退出 if 分支，恢复 EXEC
s_or_b64  exec, exec, saved
// EXEC = 当前 EXEC | saved = 恢复到原始状态
```

> **关键**：EXEC 的保存/恢复完全由**编译器**管理（LLVM AMDGPU 后端），不是硬件自动完成。这意味着复杂的控制流可能导致编译器生成错误的恢复代码。

---

## 4. 分支分化（Branch Divergence）

### 4.1 简单分化

```cpp
if (lane_id < 32) {
    do_A();
} else {
    do_B();
}
// 汇合点：编译器在此恢复 EXEC
do_C();
```

编译器处理流程：
1. 保存 EXEC，设置 EXEC = 前 32 个 lane
2. 执行 `do_A()`
3. 反转 EXEC = 后 32 个 lane
4. 执行 `do_B()`
5. 恢复 EXEC = 全部 64 lane
6. 执行 `do_C()`

这种简单分化，编译器能正确处理。

### 4.2 嵌套分化（危险模式）

```cpp
if (lane_id == 0) {              // 第一层分化
    while (volatile_load() == 0)  // 第二层：循环分化
        ;
    x = compute();
}
use(x);                           // 期望全部 lane 活跃
```

这种 **`if` 内嵌 `while` 自旋循环** 的模式是 AMD 上的高危模式。原因：

1. 编译器需要同时管理 `if` 的 EXEC 保存/恢复和 `while` 的循环 EXEC 管理
2. 循环的退出条件可能修改 EXEC（将退出循环的 lane 从 EXEC 中移除）
3. 嵌套的 EXEC 保存/恢复可能互相干扰
4. 编译器的结构化分析（Structural Analysis）对这种模式可能生成不正确的恢复代码

### 4.3 扁平化分化（安全模式）

将循环条件与 lane 判断合并，避免嵌套：

```cpp
// 危险：嵌套分化
if (lane_id == 0) {
    while (volatile_load() == 0) ;
}

// 安全：扁平化分化
while (lane_id == 0 && volatile_load() == 0) ;
```

扁平化后只有一层分化（while 循环），编译器能正确处理重汇合。

---

## 5. AMD vs NVIDIA 线程调度模型对比

### 5.1 执行模型对比

| 特性 | NVIDIA Volta+（SM ≥ 7.0） | AMD CDNA |
|------|--------------------------|----------|
| 线程调度 | **独立线程调度**：每个线程有独立 PC | **SIMD 锁步**：wavefront 共享一个 PC |
| warp/wavefront 大小 | 32 | **64** |
| 分化处理 | 线程可独立在不同 PC 位置等待 | 通过 EXEC mask 屏蔽非活跃 lane |
| 重汇合机制 | 硬件自动 + `__shfl_sync` 显式重汇合 | **编译器管理** EXEC mask 保存/恢复 |
| `__shfl_sync` 语义 | 同步 + 数据交换 | **仅数据交换**，"sync" 是空操作 |

### 5.2 一个自旋等待的例子

```cpp
if (lane_id == 0) {
    while (load(ptr) == 0) ;   // lane 0 自旋等待
}
result = __shfl_sync(MASK, value, 0);  // 广播 lane 0 的值
```

**NVIDIA Volta+ 的执行**：

```
lane 0:    进入 while 循环 → 自旋 → 退出 → 到达 __shfl_sync
lane 1~31: 跳过 if → 到达 __shfl_sync → 等待 lane 0
                                         ↓
                              全部线程汇合，shfl 正确执行
```

每个线程有独立 PC，lane 1~31 可以先到 `__shfl_sync` 处等待。`__shfl_sync` 自带重汇合语义。

**AMD CDNA 的执行**：

```
所有 lane:  进入 if → EXEC = lane 0 → while 循环
            lane 0 自旋，lane 1~63 陪跑（被 EXEC 屏蔽）
            → lane 0 退出循环 → 退出 if
            → 编译器应恢复 EXEC = 全部 lane（但可能有 bug）
            → __shfl_sync = ds_bpermute（纯数据操作，不含同步语义）
```

整个 wavefront 始终在同一条指令上。不存在"先到 `__shfl_sync` 等着"这回事。

---

## 6. 关键 API 在 AMD 上的行为差异

### 6.1 `__shfl_sync(mask, var, srcLane)`

| | NVIDIA | AMD |
|---|--------|-----|
| 功能 | 同步指定线程 + 跨 lane 数据交换 | **仅**跨 lane 数据交换 |
| mask 参数 | 指定哪些线程参与同步和交换 | **通常被忽略** |
| 底层指令 | `shfl.sync` | `ds_bpermute_b32` 或 `__shfl` |
| 是否重汇合 | **是**，会等待 mask 中所有线程到达 | **否**，只是读另一个 lane 的寄存器 |
| EXEC 的影响 | 不适用（有独立 PC） | **只有 EXEC 中活跃的 lane 的目标寄存器被写入** |

> 核心差异：在 AMD 上，`__shfl_sync` 不会帮你重汇合线程。如果调用时 EXEC mask 不完整，只有活跃 lane 拿到结果。

### 6.2 `__syncthreads()`

| | NVIDIA | AMD |
|---|--------|-----|
| 功能 | Block 内所有线程同步 | Workgroup 内所有 wavefront 同步 |
| 底层指令 | `bar.sync` | `s_barrier` |
| 同步粒度 | 线程级 | **Wavefront 级** |
| 与 EXEC 的关系 | 每个线程独立到达 | **wavefront 作为整体到达，不看 EXEC** |
| 是否修改 EXEC | 不适用 | **不修改 EXEC**（但编译器通常在其前后插入 EXEC 恢复代码） |

关键行为：

```
// 假设此时 EXEC = 仅 lane 0（其他 lane 被屏蔽）
__syncthreads();
// s_barrier 成功同步了所有 wavefront
// 但 EXEC 仍然是 lane 0 only（s_barrier 不修改 EXEC）
// 编译器可能在 s_barrier 前后插入 EXEC 恢复，也可能不插入
```

`s_barrier` 的计数方式：
- 它计数的是"有多少个 **wavefront** 到达了"
- 一个 wavefront 到达 `s_barrier` = 该 wavefront 中**所有 64 个线程**被视为已到达
- EXEC mask 不影响 wavefront 是否被计入——即使 EXEC = 0，wavefront 仍参与屏障

### 6.3 `__syncwarp()`

| | NVIDIA | AMD |
|---|--------|-----|
| 功能 | Warp 内线程同步 | 映射为 `__builtin_amdgcn_wave_barrier()` 或类似 |
| 意义 | 重汇合分化的线程 | 主要是**编译器屏障**（阻止指令重排） |

因为 AMD 是锁步执行，wavefront 内线程天然同步，`__syncwarp` 的主要价值是作为编译器内存屏障。

### 6.4 对比总结

```
NVIDIA:  __shfl_sync = 同步 + 数据交换     （两件事）
AMD:     __shfl_sync = 数据交换             （一件事）

NVIDIA:  __syncthreads = 线程级全局屏障     （精确到每个线程）
AMD:     __syncthreads = wavefront 级屏障   （以 wavefront 为单位）
```

---

## 7. 常见 Bug 模式与案例分析

### 7.1 Bug 模式：分化代码后直接使用 `__shfl_sync`

**错误代码**：

```cpp
int value;
if (lane_id == 0) {
    while ((value = ld_volatile_global(ptr)) == 0) ;  // 自旋等待
    value = -value - 1;
}
// 编译器可能未正确恢复 EXEC（嵌套 if+while 的缺陷）
value = __shfl_sync(WARP_MASK, value, 0);  // ← EXEC 可能仅 lane 0
```

**现象**：只有 lane 0 拿到正确值，其他 lane 的 `value` 是未初始化的垃圾值。

**原因分析**：
1. `if + while` 嵌套分化导致编译器 EXEC mask 恢复逻辑有缺陷
2. `__shfl_sync` 底层的 `ds_bpermute_b32` 只写入 EXEC 中活跃的 lane
3. 如果 EXEC = lane 0 only，则只有 lane 0 的寄存器被写入广播值

### 7.2 修复方案 A：在 `__shfl_sync` 前加 `__syncthreads()`

```cpp
int value;
if (lane_id == 0) {
    while ((value = ld_volatile_global(ptr)) == 0) ;
    value = -value - 1;
}
__syncthreads();  // ← 强制编译器恢复 EXEC，作为编译器屏障
value = __shfl_sync(WARP_MASK, value, 0);  // 现在 EXEC = 全部 lane
```

`__syncthreads()` 的作用：
- 强制编译器在此处插入 EXEC 恢复代码
- 作为编译器屏障，阻止指令跨 barrier 重排
- 确保后续的 `__shfl_sync` 在完整 EXEC mask 下执行

### 7.3 修复方案 B：扁平化控制流（推荐）

```cpp
int value;
while (lane_id == 0 && (value = ld_volatile_global(ptr)) == 0)
    ;
if (lane_id == 0) {
    value = -value - 1;
}
value = __shfl_sync(WARP_MASK, value, 0);  // 单层分化，编译器正确恢复
```

将 while 循环从 if 块中移出，利用短路求值：
- `lane_id == 0` 为 false 时，短路跳过 `ld_volatile_global`，直接退出循环
- 只有 lane 0 实际执行自旋等待
- 控制流从"嵌套 if+while"变为"顶层 while + 简单 if"
- 编译器对单层循环分化的 EXEC 恢复更可靠

### 7.4 为什么 `__syncthreads()` 放在 `__shfl_sync` 之后无法修复

```cpp
// 错误的修复位置
value = __shfl_sync(WARP_MASK, value, 0);  // ← 已经用错误的 EXEC 执行了
__syncthreads();  // ← 同步成功，但无法回溯修正 __shfl_sync 的结果
```

`__syncthreads()` 是一个同步屏障，不是时光机。它能确保后续代码看到一致的状态，但无法修改已经执行过的指令的结果。

---

## 8. 最佳实践

### 8.1 控制流规则

| 规则 | 说明 |
|------|------|
| **避免嵌套分化控制流** | 不要在 `if (lane_id == X)` 内放 `while` 循环 |
| **用扁平化代替嵌套** | `while (lane_id == 0 && cond)` 代替 `if (lane_id == 0) { while (cond) }` |
| **分化代码后加显式屏障** | 在分化的 `if` 块和集体操作（`__shfl_sync` 等）之间加 `__syncthreads()` 或 `__syncwarp()` |

### 8.2 `__shfl_sync` 使用规则

| 规则 | 说明 |
|------|------|
| **不要依赖 "sync" 语义** | 在 AMD 上它不会重汇合线程，只做数据交换 |
| **确保调用时 EXEC mask 完整** | 在 `__shfl_sync` 前加 barrier，或确保控制流已正确汇合 |
| **mask 参数不可靠** | AMD 上 mask 可能被忽略，不能用它来控制参与的 lane |

### 8.3 `__syncthreads()` 使用规则

| 规则 | 说明 |
|------|------|
| **理解它的同步粒度** | AMD 上是 wavefront 级，不是线程级 |
| **它不会修改 EXEC** | 不要指望它自动恢复 EXEC mask |
| **它的主要附加价值是编译器屏障** | 阻止编译器跨 barrier 重排指令，强制编译器在此处处理 EXEC 恢复 |
| **不要在分化路径中调用** | 如果 block 内有些线程走了 `return` 而不会到达 `__syncthreads()`，会导致死锁或未定义行为 |

### 8.4 平台差异常量

从 CUDA 迁移到 HIP 时需要调整的常量：

```cpp
#if defined(__HIP_PLATFORM_AMD__)
    #define WARP_SIZE 64
    #define WARP_MASK 0xffffffffffffffffULL   // 64-bit
#else
    #define WARP_SIZE 32
    #define WARP_MASK 0xffffffffU              // 32-bit
#endif
```

### 8.5 调试建议

1. **如果 `__shfl_sync` 后只有 lane 0 拿到正确值**：几乎可以确定是 EXEC mask 未恢复。在 `__shfl_sync` 前加 `__syncthreads()` 或重构控制流。

2. **如果 `__syncthreads()` 似乎"没等待"**：它确实等待了（wavefront 级），但问题出在它之前的代码。检查是否有指令在错误的 EXEC mask 下执行。

3. **如果自旋循环导致 hang**：检查是否有跨 wavefront 的依赖导致死锁。锁步执行意味着同一 wavefront 内的线程无法互相通信来推进进度。

---

## 附录：AMD ISA 中的 EXEC 管理指令速查

| 指令 | 作用 |
|------|------|
| `s_and_saveexec_b64 dst, src` | `dst = EXEC; EXEC = EXEC & src`（保存并修改 EXEC） |
| `s_or_b64 exec, exec, src` | `EXEC = EXEC \| src`（恢复 EXEC） |
| `s_mov_b64 exec, src` | `EXEC = src`（直接设置 EXEC） |
| `s_xor_b64 exec, exec, src` | `EXEC = EXEC ^ src`（用于 if/else 翻转） |
| `s_barrier` | wavefront 级屏障，不查看/修改 EXEC |
| `ds_bpermute_b32 dst, addr, src` | 跨 lane 数据交换，**只写入 EXEC 活跃的 lane** |

---

## 9. 实战案例：Atomic Counter MoE Permute Kernel

> 本节基于实际实现 `moe_permute_atomic_kernel` 的设计，展示如何在 AMD CDNA 架构上设计高性能 kernel，以及涉及的 HIP/CUDA 跨平台细节。

### 9.1 问题背景

MoE（Mixture of Experts）推理中，dispatch 阶段通过 NVLink 将 token 发送到目标 rank。接收到的 token 按 **rank 顺序** 排列，但 grouped GEMM 要求 token 按 **expert 分组** 连续排列。permute kernel 的任务是完成这个重排列：

```
输入 (按 rank 排列):   [ rank_0 tokens | rank_1 tokens | ... ]
输出 (按 expert 排列): [ expert_0 tokens | expert_1 tokens | ... ]
```

一个 token 可能被路由到多个 local expert（top-k routing），因此需要 **token 复制**：同一 token 出现在多个 expert 的输出区域。

### 9.2 Atomic Counter 方案

每个 local expert 持有一个全局 atomic counter（初始值 0）。Warp 处理一个 token 时，对每个有效 expert 执行 `atomicAdd` 占 slot，然后将 token 数据写入对应位置。

```
Input:
  recv_x              [num_recv_tokens, hidden]       — 接收到的 token
  recv_topk_idx       [num_recv_tokens, num_topk]     — local expert ID (0..E_r-1, 或 -1)
  recv_topk_weights   [num_recv_tokens, num_topk]     — routing weights
  expert_offsets      [num_local_experts]              — 每个 expert 在输出中的起始偏移 (exclusive prefix sum)

Output:
  permuted_x          [total_expert_tokens, hidden]   — 按 expert 连续排列
  permuted_weights    [total_expert_tokens]            — 对应 weight
  src_row_id          [total_expert_tokens]            — 源 token index (用于 unpermute)
```

### 9.3 Kernel 架构

```
Grid:  num_sms 个 block
Block: 256 threads = 4 wavefronts (AMD wave64)

每个 warp 处理一个 token (round-robin):
  Step 1: lane 0..topk-1 各加载一个 expert ID + weight
  Step 2: 全 warp 将 hidden vector 加载到寄存器缓存 (load once)
  Step 3: ballot + ffs 遍历有效 expert:
          - atomicAdd 占 slot
          - 从寄存器写出 hidden 数据 (write N times)
```

### 9.4 AMD 平台关键设计点

#### 9.4.1 Warp Ballot 跨平台差异

`__ballot_sync` 在 AMD 和 NVIDIA 上的返回类型不同：

| 平台 | Wavefront 大小 | 返回类型 | Find-first-set 函数 |
|------|---------------|----------|---------------------|
| AMD CDNA | 64 | `unsigned long long` (64-bit) | `__ffsll()` |
| NVIDIA | 32 | `unsigned int` (32-bit) | `__ffs()` |

封装跨平台 helper（见 `utils.cuh`）：

```cpp
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
using warp_mask_t = unsigned long long;
#else
using warp_mask_t = unsigned int;
#endif

__device__ __forceinline__ warp_mask_t warp_ballot(bool pred) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __ballot(pred);       // HIP 原生 64-bit ballot
#else
    return __ballot_sync(WARP_MASK, pred);
#endif
}

__device__ __forceinline__ int warp_find_first_set(warp_mask_t mask) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    return __ffsll(static_cast<long long>(mask)) - 1;  // 64-bit
#else
    return __ffs(static_cast<int>(mask)) - 1;           // 32-bit
#endif
}
```

> **注意**：AMD 上 `__ballot(pred)` 不带 mask 参数。HIP 也提供 `__ballot_sync(mask, pred)` 兼容形式，但 mask 参数在 AMD wave64 锁步模型下实际被忽略（参考第 2 节）。

#### 9.4.2 清除最低有效位 — 平台无关技巧

遍历 ballot mask 中的每个有效 bit 时，使用位运算技巧跳过平台差异：

```cpp
valid_mask &= valid_mask - 1;   // 清除最低 set bit
```

此操作对 32-bit 和 64-bit 整数均有效，不需要平台分支。

#### 9.4.3 VGPR 预算与 Register Caching

AMD MI300X (CDNA3) 每 SIMD 有 512 VGPRs。Thread 数决定了每 thread 可用 VGPRs：

| Block Threads | Wavefronts / Block | VGPRs / Thread | 是否支持 Register Cache |
|---------------|-------------------|----------------|----------------------|
| 1024 | 16 | 32 | ✗ (hidden=7168 需 56 VGPRs) |
| 512 | 8 | 64 | 勉强 |
| **256** | **4** | **128** | **✓ 充裕** |

**为什么选 256 threads**：对于 hidden=7168 bf16（hidden_int4=896），每 lane 缓存 `896/64 = 14` 个 int4 = 56 VGPRs。加上控制变量 ~20 VGPRs，共 ~76 VGPRs，在 128 VGPRs 预算内。

Register caching 的核心是 **load once, write N times**：token 可能路由到 1~4 个 local expert，每次写出复用寄存器数据，避免重复 global memory 读取。

```cpp
// 加载阶段 (执行一次)
constexpr int kMaxElemsPerLane = 16;  // 覆盖到 hidden=8192 bf16
int4 cached[kMaxElemsPerLane];
#pragma unroll
for (int j = 0; j < kMaxElemsPerLane; j++) {
    int idx = lane_id + j * WARP_SIZE;
    if (idx < hidden_int4)
        cached[j] = __ldg(src_ptr + idx);   // read-only cache path
}

// 写出阶段 (执行 N 次, N = 有效 expert 数)
#pragma unroll
for (int j = 0; j < kMaxElemsPerLane; j++) {
    int idx = lane_id + j * WARP_SIZE;
    if (idx < hidden_int4)
        st_na_global(dst_ptr + idx, cached[j]);   // non-temporal store
}
```

> **编译器提示**：使用 `#pragma unroll` + 编译时常量 `kMaxElemsPerLane` 作为循环边界，引导编译器将 `cached[]` 数组分配到 VGPRs 而非 local memory（栈溢出）。运行时 `if (idx < hidden_int4)` 保证对未使用的 hidden size 不产生无效访存。

#### 9.4.4 Shared Memory Expert Offsets

`expert_offsets` 是小数组（典型 32 个 int = 128 bytes），被每个 warp 在每个 token 的每个 expert 迭代中访问。放入 shared memory 避免反复 global read：

```cpp
__shared__ int smem_expert_offsets[NUM_MAX_LOCAL_EXPERTS];  // 1024 * 4 = 4KB
for (int i = threadIdx.x; i < num_local_experts; i += kNumThreads)
    smem_expert_offsets[i] = expert_offsets[i];
__syncthreads();
```

AMD MI300X LDS 容量为 64KB per CU，4KB 占比 6.25%，不影响 occupancy。

#### 9.4.5 Atomic Contention 分析

Kernel 中每个 (token, expert) 对触发一次 `atomicAdd`。竞争度分析：

```
场景: 4096 recv tokens, 32 local experts, 64 SMs, 4 warps/block
      → 256 warps 并发
      → 平均每 expert 被 256/32 = 8 个 warp 并发竞争
```

MI300X 的 device-scope `atomicAdd` 对 int32 延迟约 ~100ns。8 路竞争下吞吐量约 80M ops/s，远快于 hidden copy 的带宽瓶颈。**Kernel 是内存带宽受限，不是 atomic 受限。**

### 9.5 性能特征

| 因素 | 分析 |
|------|------|
| **瓶颈** | HBM 写带宽（permuted_x 写出） |
| **读带宽** | recv_x 每 token 读 1 次（register cached），topk_idx/weights 读 1 次（non-temporal） |
| **写带宽** | 每 (token, expert) 对写 hidden bytes + 8 bytes 元数据 |
| **Atomic 开销** | 可忽略（32 expert 分散竞争，远低于带宽瓶颈） |
| **Register 压力** | 76/128 VGPRs（256 threads），无溢出 |

### 9.6 融合 Dispatch + Permute 的方向

当前实现是 **standalone permute**（dispatch 产出 `recv_x` → permute 重排到 `permuted_x`），中间多一次全量 hidden copy。进一步优化可将 permute 融入 dispatch receiver：

```
当前:  NVLink buffer → recv_x (dispatch)  → permuted_x (permute)
                        ↑ 多一次读写
融合:  NVLink buffer → permuted_x (dispatch + permute in one pass)
```

关键变更点在 `fused_dispatch_permute` receiver 循环内，将 `UNROLLED_WARP_COPY` 到 `recv_x` 替换为 register cache + atomic slot + write to `permuted_x`。由于 NVLink buffer 是远端内存（延迟 ~5x vs L2），register caching 在融合版本中价值更大。

### 9.7 调用示例

```python
# expert_offsets = exclusive prefix sum of per-expert token counts
expert_offsets = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
expert_offsets[1:] = moe_recv_expert_counter.cumsum(0)[:-1]
total_permuted_tokens = moe_recv_expert_counter.sum().item()

permuted_x, permuted_weights, src_row_id = cpp_extensions.moe_permute_atomic(
    recv_x, recv_topk_idx, recv_topk_weights,
    expert_offsets, num_local_experts, total_permuted_tokens, num_sms=64)

# permuted_x 可直接作为 grouped GEMM 输入
# group_lens = moe_recv_expert_counter
# group_offs = expert_offsets
```
