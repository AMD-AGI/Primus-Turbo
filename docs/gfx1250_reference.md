# gfx1250 / MI455X (CDNA5) 指令集与内核优化参考

> 整理日期：2026-06-09。资料来源：上游 LLVM/Clang/MLIR 提交、ROCm 文档、IREE 设计讨论、
> 各架构分析文章（见文末 Sources）。**注意**：gfx1250 在 2026 年中仍处于"软件使能进行中"
> 阶段，硬件未公开发售、无第三方实测；本文中大量细节来自编译器提交与数据挖掘，标 ⚠️ 者尤其
> 应在最终量产工具链上复核。本文与 MegaMoE 现有 gfx950(CDNA4) 工作对照，便于未来移植。

---

## 0. TL;DR — gfx1250 是什么

| 项目 | 内容 |
|---|---|
| 架构代号 | `gfx1250` = GFX12.5，社区称 **CDNA5**（实为 CDNA/RDNA 融合，接近 UDNA 思路）|
| 产品 | **Instinct MI450 / MI455X**（AI）；`gfx1251` = **MI430X**（HPC）|
| 工艺 / 内存 | TSMC **N2 (2nm)**；**HBM4 432 GB / 19.6 TB/s**（MI355 是 288GB/8TB/s）|
| 峰值 | **40 PFLOPS FP4 / 20 PFLOPS FP8**（MI355：20.1 PFLOPS FP4=FP6）|
| 互连 | **UALink + Infinity Fabric**（首款支持 UALink），Helios 机柜 = 72 GPU |
| 上市 | 2026 H2；详情见 Advancing AI 2026（7 月）|
| LLVM form factor | 标记为 **APU**，归在 GFX12 大类（与 RDNA4 同主版本）|

**对内核作者最重要的 5 个断裂式变化（相对 gfx950/CDNA4）：**

1. **只有 Wave32**——CDNA1–4 一直是 wave64，gfx1250 彻底移除 wave64；LLVM 在 gfx1250 上用
   wavefront64 会直接报错。所有 wave64 MFMA 内核必须重排到 wave32。
2. **矩阵走 WMMA 路线**（RDNA 风格，固定 M=N=16/32 系列），不再以 MFMA 命名为主；
   但 `scaled_mfma` 仍标 "gfx1250+ 可用"（早期"无 MFMA"的判断后被修正）。
3. **LDS 翻倍到 320 KiB**（CDNA3 64 → CDNA4 160 → gfx1250 320），且只有 CU mode。
4. **新增类 TMA 的 Tensor Data Movement (TDM)** + 直达 LDS 的 async copy + prefetch +
   异步计数器——硬件接管"global→LDS 大块搬运 / bank-conflict 规避"。
5. **Workgroup Clusters**（类 Hopper thread-block cluster）+ 16 个 named barrier +
   cluster barrier / wakeup barrier / cooperative atomics。

---

## 1. gfx950 (CDNA4 / MI355) vs gfx1250 (CDNA5 / MI455) 速查表

| 维度 | gfx950 / MI355X (CDNA4) | gfx1250 / MI455X (CDNA5) |
|---|---|---|
| Wavefront | wave64（也支持 wave32） | **wave32 ONLY**（wave64 报错）⚠️ |
| 矩阵指令 | **MFMA** `v_mfma_*` / `scaled_mfma`，miSIMD + AccVGPR | **WMMA** `v_wmma_*` / `scaled_wmma`（普通 VGPR）；`scaled_mfma` 仍可用 ⚠️ |
| 矩阵 tile | 16x16、32x32（K 多种） | WMMA 固定 16x16x{16,32,64,128}、32x16x128(f4) |
| 低精度 | FP8/FP6/FP4 + E8M0 block scale | FP8/FP6/FP4 (f8f6f4) + block-scale 16/32 |
| LDS / CU | 160 KiB | **320 KiB** |
| CU 组织 | wave64 SIMD | 128 cores/CU (4×32) 固定双发，VOPD3 |
| global→LDS | `buffer_load ... lds`（直达 LDS）| + **TDM tensor_load_to_lds** + `global_load_async_to_lds` + prefetch |
| 同步 | workgroup barrier | + **cluster barrier / 16 named barriers / wakeup / cooperative atomics** |
| Cluster | 无 | **Workgroup Clusters**（cluster sync scope）|
| FP64 | 全速 78.6 TFLOPS | 降配（非全速，HPC 交给 gfx1251/MI430X）|
| Scratch | 普通 | Globally Accessible Scratch（跨 lane 访问栈合法）|
| Buffer rsrc | 48-bit base | **57-bit base / 45-bit NumRecords**（`make.buffer.rsrc` 参数改 i64）⚠️ |
| ECC / packed FP32 | 有 | 保留 |

---

## 2. 逐特性 + 参考代码

> 代码风格：Clang builtin（`__builtin_amdgcn_*`）、LLVM intrinsic（`llvm.amdgcn.*`）、
> MLIR `amdgpu` dialect op。目标特性串：`gfx1250-insts`、`swmmac-gfx1250-insts`、
> `vmem-pref-insts`、并多数要求 `wavefrontsize32`。

### 2.1 Wave32 与目标三元组

```bash
# 编译目标
clang -x hip --offload-arch=gfx1250 ...
# 等价：-mcpu=gfx1250 -mwavefrontsize32（gfx1250 上 wave64 会报错）
# generic 目标：gfx12-5-generic（gfx1250/gfx1251 功能等价）
```

```cpp
// HIP 端查询（运行期）
hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);
// p.warpSize == 32 on gfx1250
// p.gcnArchName 形如 "gfx1250"
```

要点：所有跨 lane 归约、`__shfl`、矩阵 fragment 布局都按 **32 lane** 重新计算；
原 wave64 内核里 `lane = tid & 63`、`__ballot` 掩码宽度等全部要改。

---

### 2.2 WMMA 矩阵指令（主力计算路径）

gfx1250 的 WMMA 是 **M=N=16**（f4 变体 M=32,N=16），K ∈ {16,32,64,128}，依元素类型而定，
**仅 wave32**。Clang builtin（特性串均为 `"gfx1250-insts,wavefrontsize32"`）：

```cpp
// ---- FP16 / BF16（新增更大 K 形状）----
__builtin_amdgcn_wmma_f32_16x16x32_f16(/*A*/, /*B*/, /*C*/);   // f16 累加到 f32
__builtin_amdgcn_wmma_f16_16x16x32_f16(/*A*/, /*B*/, /*C*/);   // f16 累加到 f16

// ---- FP8 / BF8（A、B 类型可独立组合）----
__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(A, B, C);
__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(A, B, C);
__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(A, B, C);
__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(A, B, C);

// ---- 混合微浮点 f8f6f4（一条指令吃 FP8/FP6/FP4 混合块）----
__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(A, B, C);

// ---- 纯 FP4，32x16x128 形状（输出 16xf32）----
__builtin_amdgcn_wmma_f32_32x16x128_f4(A, B, C);

// ---- INT8 / INT4（带 clamp 参数，见 2.3）----
__builtin_amdgcn_wmma_i32_16x16x64_iu8(neg_a, A, neg_b, B, C, /*clamp*/true, ... );
```

对应 LLVM intrinsic：`llvm.amdgcn.wmma.f32.16x16x128.f8f6f4` 等；TableGen 里
`V_WMMA_F32_16X16X128_F8F6F4`，谓词 `SubtargetPredicate=isGFX1250Plus,
WaveSizePredicate=isWave32`。

MLIR：

```mlir
// 16x16x128 f8f6f4 -> vector<8xf32>
%d = amdgpu.wmma 16x16x128 %a * %b + %c : vector<...>, vector<...>, vector<8xf32>
// FP4 16x16x128 / i8 16x16x64 等形状均已暴露
```

**⚠️ Fragment lane 布局陷阱（移植最易踩）**：gfx12 WMMA 是 **列分布**——
`lane % 16 == 列号(N)`，不是行号；`(lane/16)*8 .. +7` 是该 lane 持有的行。
若按"lane%16=行"写，**编译运行都不报错，但 16x16 输出被静默转置**。这一约定与 CDNA MFMA
一致（lane%16 也选列），所以移植逻辑可复用，但务必用 AMD Matrix Instruction Calculator 核对。

**寄存器占用（wave32）**：A_frag/B_frag 各 8 VGPR(fp16/bf16)、4(iu8)、2(iu4)；
C_frag/D_frag 在 wave32 下 **8 VGPR**（wave64 才是 4）。无 AccVGPR（CDNA 的 AGPR 分离消失）。

**gfx12 相对 RDNA3 的改进**：f16→f16 WMMA 结果 16 元素**全部有效**，`subwordOffset` 必须为 0
（RDNA3 上只有 8 个有效、需 subwordOffset 选择）。

---

### 2.3 Scaled WMMA（block-scaling，FP4/FP6/FP8 + microscale）

```cpp
// per-block 缩放，block_size ∈ {16,32}
__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(A, B, C, scaleA, scaleB, ...);
__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(...);   // scale16 变体
__builtin_amdgcn_wmma_scale_f32_32x16x128_f4(...);
__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(...);
```

MLIR `amdgpu.scaled_wmma`：

```mlir
// tile: 16x16x128（混合 f8/f6/f4，输出 8xf32）或 32x16x128（仅 f4，输出 16xf32）
// block_size 16 或 32；num_scales_A = M*K/block_size，num_scales_B = N*K/block_size
// a_first_scale_lane / b_first_scale_lane ∈ {0,16} 选取读 scale 的起始 lane
%d = amdgpu.scaled_wmma ...
```

scale 元素类型：`f8E8M0FNU`（OCP MX 的 E8M0）或 `f8E4M3FN`。
元素格式：FP4=`f4E2M1FN`，FP6=`f6E2M3FN`/`f6E3M2FN`，FP8=`f8E4M3FN`/`f8E5M2`。

配套的解包 op：

```mlir
// 把微浮点矩阵按 scale 展开（scale 可存于其他 lane）
// blockSize 16/32，firstScaleLane 0/16，firstScaleByte 0..3
%ext = amdgpu.scaled_ext_packed_matrix %src scale(%s) ...   // gfx1250+
```

**Clamp 重做**：WMMA clamp 现为 builtin 显式参数（如 `..._iu8(0,a,0,b,c,false,/*clamp*/true,1)`）。

---

### 2.4 MFMA / scaled_mfma（CDNA 兼容路径，仍可用 ⚠️）

早期数据挖掘认为 gfx1250 "不支持 MFMA"，但 MLIR 文档随后显示 `scaled_mfma`
**"Available on gfx1250+"**。结论：以 WMMA 为主，但 scaled_mfma 路径保留。

```mlir
// CDNA mfma 包装；fp4/fp6/fp8，tile = 16x16x128 或 32x32x64
%d = amdgpu.scaled_mfma ... : ...   // gfx1250+
```

对照 gfx950 的同族 builtin（**移植时的来源端**）：

```cpp
// CDNA4 (gfx950) 上的 scaled MFMA，移植到 gfx1250 时优先换 WMMA 等价
__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(A, Ax, B, Bx, C, ...);
// 每线程：A 32 项、Ax 1 项、B 32 项、Bx 1 项、C 16 项
// 前两参数需 256-bit 宽（32×FP4=128bit 数据 + 128bit 补零，定义 fp4x64_t）
// scale 为 E8M0(uint8)，实际倍率 = 2^(scale-127)
```

---

### 2.5 Tensor Data Movement (TDM) —— 类 NVIDIA TMA 的大块搬运

工作流：一个 wave 构造**描述符 D#**（base + size/stride），提交给硬件，硬件自行把大 tile
切块在 global↔LDS 间搬运，无需经过 VGPR。

Clang builtin（特性 `"gfx1250-insts"`）：

```cpp
__builtin_amdgcn_tensor_load_to_lds(desc, ...);       // global -> LDS
__builtin_amdgcn_tensor_load_to_lds_d2(desc, ...);    // 双描述符变体
__builtin_amdgcn_tensor_store_from_lds(desc, ...);    // LDS -> global
__builtin_amdgcn_tensor_store_from_lds_d2(desc, ...);
```

底层指令：`TENSOR_LOAD_TO_LDS`(0xc4) / `TENSOR_STORE_FROM_LDS`(0xc5)，
`VIMAGE_TENSOR_Real_gfx1250`，谓词 `isGFX1250Plus`。tensor store 带 `TH_TYPE_STORE` 时间提示。

MLIR：

```mlir
// 1) 构造 base（LDS<->global 的地址对）
%base = amdgpu.make_dma_base ... : !amdgpu.tdm_base<T>

// 2) 构造描述符：globalSize/globalStride/sharedSize，可选 padShared(bank-conflict)、
//    workgroupMask、atomicBarrier、2D/3D iterate
//    约束：pad 间隔为 2 的幂 ∈ [2,256]；pad 量 ∈ [1,128]；iterate ∈ [1,256]
%desc = amdgpu.make_dma_descriptor %base
          globalSize [...] globalStride [...] sharedSize [...]
          { padShared = ..., iterate = ... } : ...

// 3) 提交
amdgpu.tensor_load_to_lds %desc : ...
amdgpu.tensor_store_from_lds %desc : ...
```

**TDM 能力清单（性能相关）**：
- 维度 **1–5D**；stride 48-bit、size 32-bit（动态维度无需特殊处理）。
- **约束：最内层 stride 必须为 1**（strided conv 不能透明处理）。
- **0 填充**：请求 16×32 而源仅 8×32 → 多出的行补 0（store 时不写）。
- **循环额外步进**：维度足够小时可指定 global/LDS 地址增量，硬件循环累加。
- **gather 模式**：用一组 16/32-bit offset 聚集（见 2.6）。
- **隐式 bank-conflict 规避**：写满 Y 个 word 后让硬件跳过 X 个 32-bit word——
  **取代手写 padding pass**，但要让"LDS 空洞"对读路径可见以免读到空洞。

**Gather DMA**：

```mlir
%gb   = amdgpu.make_gather_dma_base ... : !amdgpu.tdm_gather_base<elem, index>
%gd   = amdgpu.make_gather_dma_descriptor %gb offsets(...) ...
amdgpu.tensor_load_to_lds %gd : ...
```

---

### 2.6 简单 async copy（global→LDS，绕过 VGPR）

不需要描述符的逐线程拷贝，语义最直白（"just copy，无 uniformity / 无 gather"）：

```cpp
// 8/32/64/128-bit，特性 "gfx1250-insts"
__builtin_amdgcn_global_load_async_to_lds_b8 (global_ptr, lds_ptr, /*mask*/);
__builtin_amdgcn_global_load_async_to_lds_b32(global_ptr, lds_ptr, ...);
__builtin_amdgcn_global_load_async_to_lds_b64(global_ptr, lds_ptr, ...);
__builtin_amdgcn_global_load_async_to_lds_b128(global_ptr, lds_ptr, ...);
__builtin_amdgcn_global_store_async_from_lds_b32(global_ptr, lds_ptr, ...);

// cluster 作用域变体（需 "gfx1250-insts,wavefrontsize32"）
__builtin_amdgcn_cluster_load_async_to_lds_b8/ b32 / b64(...);
```

MLIR：

```mlir
// 仅 gfx1250+；类型为 8/32/64/128-bit 标量或向量，可选每线程 $mask
amdgpu.global_load_async_to_lds %global -> %lds [mask %m] : ...
```

**⚠️ 关键性能教训（来自 gfx950 直达 LDS 原型）**：直达 LDS 必须**保留 XOR swizzle 布局**。
去掉 swizzle 后 `ds_write_b128` 虽被消除，却引入 2.01 亿次 LDS bank conflict（基线 0），
吞吐 **-27.9%**（1822 vs 2527 TFLOPS）。Triton ROCm 用 `buffer_load ... lds` 做全 operand
async copy（`AMDGCN_USE_BUFFER_OPS + TRITON_HIP_USE_ASYNC_COPY`），关掉约 -10%。

---

### 2.7 Prefetch

```cpp
// 把数据预取到 GL2；早期目标忽略，gfx1250 上实际发射 flat_prefetch_b8 / global_prefetch_b8
__builtin_prefetch(ptr, /*rw*/0, /*locality*/3);   // 经 llvm.prefetch lowering
// vmem prefetch intrinsics 特性串 "vmem-pref-insts"
```

```mlir
// 时间提示 RT/NT/HT/LU…，cache scope WGP/SE/DEV/SYS
amdgpu.global_prefetch %ptr { ... } : ...   // gfx1250+ 引入
```

---

### 2.8 异步完成跟踪（async counters）

TDM / async copy 是异步的，靠硬件计数器 `asynccnt` / `tensorcnt` 跟踪完成。后端规划
`asyncmark` / `asyncwait` intrinsic 抽象这些计数器，由编译器推导插入正确的等待，避免代码生成器
手数内存操作（且防编译器删/重排/展开）。等待原语：`s_wait_xcnt`。

```cpp
// 概念用法（API 仍在演进 ⚠️）：发起若干 async copy -> mark -> 计算上一批 -> wait
// 实际可能以 memref.dma_start / dma_wait 风格暴露
```

配合 **多缓冲 LDS**（320KiB 足够开多 buffer）做软件流水：DMA 在飞行时计算前一 tile。

---

### 2.9 Workgroup Clusters + 同步原语（类 Hopper TBC）

Barrier 类型枚举扩展：`CLUSTER_TRAP(-4)`、`CLUSTER(-3)`、`TRAP(-2)`、`WORKGROUP(-1)`、
`NAMED_BARRIER_FIRST(1) .. LAST(16)` —— 即 **16 个 named barrier** + cluster 级 barrier。

```cpp
// ---- Cluster barrier / wakeup（gfx1250）----
// s_cluster_barrier：cluster 作用域同步；s_wakeup_barrier：唤醒等待者
__builtin_amdgcn_s_cluster_barrier();          // 概念名，convergent，isGFX1250Plus
__builtin_amdgcn_s_wakeup_barrier(...);

// ---- 16 个 Named barrier（GFX12.5+）----
llvm.amdgcn.s.barrier.init   // 初始化命名屏障 target("amdgcn.named.barrier", 0)
llvm.amdgcn.s.barrier.join
llvm.amdgcn.s.barrier.leave  // barrier drop

// ---- DS atomic barrier arrive（异步到达）----
llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64
llvm.amdgcn.ds.atomic.async.barrier.arrive.b64

// ---- Cooperative atomics（GFX12.5）----
llvm.amdgcn.cooperative.atomic...

// ---- Load monitor（仅 GFX12.5）----
llvm.amdgcn.flat.load.monitor.b32/64/128     // flat_load_monitor_b32...
llvm.amdgcn.global.load.monitor.b32/64/128
```

MLIR 侧：`amdgpu.ds_*`（init / arrive / async_barrier_arrive / poll_state / state 访问）
均标 "only available on gfx1250+"。

同步作用域：`cluster` sync scope 在支持 cluster 的 gfx1250 上跨 cluster 内线程同步
（不支持 cluster 的目标退化为 `agent` scope）。

Triton/Gluon 正在加 split barrier（issue #8420）以利用这些。

---

### 2.10 Transpose Load（转置加载，含 4/6-bit）

```mlir
// (元素 bit, 元素数) -> 指令
// (4,16)  -> global_load_tr4_b64   仅 gfx1250+
// (6,16)  -> global_load_tr6_b96   仅 gfx1250+
// (8,8)   -> ...                    gfx1200+
// (16,8)  -> ...                    gfx1200+
%t = amdgpu.global_transpose_load %ptr : ...
```

4-bit / 6-bit 转置加载是 gfx1250 专属——直接服务 FP4/FP6 GEMM 的 B 矩阵转置喂入。

---

### 2.11 VOPD3 双发射 / 标量与缓冲变化

- **VOPD3** 新编码：放宽 VOPD 双发射约束，即使两条指令在同一 VGPR bank 也能正确双发，
  几乎消除"双 SIMD 无法利用"的情形。128 cores/CU (4×32) 固定双发。
- `s_setprio_inc_wg`、`s_wait_xcnt` 等新标量指令；call 指令从 b64 改名 i64。
- **Buffer resource ⚠️**：base 截断到 **57 bit**，NumRecords **45 bit** →
  `make.buffer.rsrc` 的 NumRecords 参数改为 `i64`。移植手写 buffer descriptor 时注意。
- **Globally Accessible Scratch**（GFX125x 引入）：跨 lane 访问他人栈值变为合法（旧目标 UB）。

---

## 3. 软件 / 开源库支持现状（2026 年中）

| 组件 | 状态 |
|---|---|
| **LLVM / Clang / MLIR** | gfx1250 使能主战场：`gfx1250-insts` / `swmmac-gfx1250-insts` / `vmem-pref-insts` 特性，WMMA/TDM/async/cluster 均已陆续上游。`gfx12-5-generic` 通用目标。 |
| **hipBLASLt** | gfx12 系默认 GEMM 后端；ROCm 7.0.1 起新 GPU（含 FP8）优先用 hipBLASLt 而非 rocBLAS。`extop` / `ExtOp API` 已扩到 gfx12XX。 |
| **rocBLAS / Tensile** | 经 Tensile + hipBLASLt；gfx12 上自动转交 hipBLASLt。 |
| **Composable Kernel** | 模板实例化极重，被戏称"编译器酷刑测试"（单 kernel ~15 min/arch）；新增 arch 构建代价大。权威 lane 布局参考：`wmma_gemm.hpp`(L31–80) 与 `xdlops_gemm.hpp`。 |
| **Triton / Gluon** | ROCm fork 已用 `buffer_load ... lds` async copy；split barrier / cluster 支持在推进（issue #8420）。 |
| **IREE** | 正设计 TDM / LDS-DMA 的 codegen（discussion #23077）：bank-conflict padding 一等公民化、多缓冲流水、TDM 模式匹配、async mark/wait。 |
| **rocWMMA** | WMMA 高层封装（已并入 rocm-libraries 单仓）。 |
| **Matrix Instruction Calculator** | 验证 MFMA/SMFMAC（CDNA）与 WMMA/SWMMAC（RDNA/gfx12）lane 布局的权威工具。 |

组件版本参考（develop）：CK 1.2.0、hipBLAS 3.2.0、hipBLASLt 1.2.2、rocBLAS 5.2.0、Tensile 4.45.0。

---

## 4. gfx950 → gfx1250 移植清单（对 MegaMoE/FlyDSL 内核）

1. **wave64 → wave32**：lane 掩码、跨 lane 归约、`__ballot`/`__shfl` 宽度全改；
   gfx1250 上 wave64 直接编译报错。
2. **MFMA → WMMA**：`v_mfma_f32_32x32x*` 等 32x32 tile 需重排为 WMMA 16x16x{16..128}
   （或 32x16x128 f4）；优先用 WMMA，scaled_mfma 仅作兜底。
3. **核对 fragment lane 布局**：`lane%16 = 列(N)`，写反会静默转置——用 Matrix Calculator 验。
4. **去掉 FP64 矩阵路径**：WMMA 无 F64；FP64 HPC 交给 gfx1251/MI430X。
5. **寄存器预算**：无 AccVGPR；wave32 下 C/D_frag 占 8 VGPR。
6. **用满 320KiB LDS + TDM/async copy** 做 L1→L2 融合的中间态驻留与多缓冲流水
   （直达 LDS 务必保留 XOR swizzle，否则 bank conflict 反吃 ~28% 吞吐）。
7. **同步换原语**：现有跨 block fence 技巧（见 `flydsl-cross-block-rank-sync` 笔记）可换成
   named barrier / cluster barrier / cooperative atomics。
8. **buffer descriptor**：手写 `make.buffer.rsrc` 注意 57-bit base / 45-bit NumRecords(i64)。

---

## 5. Sources

- LLVM AMDGPU Backend User Guide — https://llvm.org/docs/AMDGPUUsage.html
- MLIR `amdgpu` Dialect — https://mlir.llvm.org/docs/Dialects/AMDGPU/
- AMDGPU.td — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPU.td
- LLVM: Initial gfx1250 target (#144965) — https://github.com/llvm/llvm-project/commit/69974658f079cec82a9fc13dd4993ab1e072c811
- LLVM: tensor load/store gfx1250 (#146636) — https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg581056.html
- LLVM: gfx1250 wmma builtins (#148991) — https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg585311.html
- LLVM: wmma f32_16x16x128_f8f6f4 (#149684) — https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg586586.html
- LLVM: wmma_scale f4 32x16x128 (#152194) — https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg592024.html
- LLVM: async loads/stores gfx1250 (#151058) — https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg589665.html
- LLVM: cluster_load_async_to_lds (#156595) — https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20250901/1721416.html
- LLVM: s_cluster_barrier (#159175) — https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg605633.html
- LLVM: s_wakeup_barrier (#170501) — http://www.mail-archive.com/cfe-commits@lists.llvm.org/msg635464.html
- LLVM: vmem prefetch (#150466) — https://www.mail-archive.com/cfe-commits@lists.llvm.org/msg588209.html
- IREE: LDS DMA in gfx1250 (discussion #23077) — https://github.com/iree-org/iree/discussions/23077
- IREE: direct-to-LDS for scaled GEMM (#23765) — https://github.com/iree-org/iree/issues/23765
- Triton: Split Barriers for GFX1250 (#8420) — https://github.com/triton-lang/triton/issues/8420
- Coelacanth's Dream: gfx1250 stub — https://www.coelacanth-dream.com/posts/2025/06/20/gfx1250-stub/
- Coelacanth's Dream: gfx1250 wave32 / 320KiB LDS — https://www.coelacanth-dream.com/posts/2025/08/15/gfx1250-only-wave32/
- Phoronix: GFX1250 support in LLVM — https://www.phoronix.com/news/AMD-GFX1250-LLVM-Start
- RDNA4 WMMA lane mapping guide — https://github.com/JohnTDI-cpu/rdna4-wmma-guide
- AMD Matrix Instruction Calculator — https://github.com/ROCm/amd_matrix_instruction_calculator
- Matrix Core Programming on CDNA3/CDNA4 — https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- Using Matrix Core of RDNA4 — https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
- AMD CDNA4 announcement (Chips and Cheese) — https://chipsandcheese.com/p/amds-cdna-4-architecture-announcement
- ServeTheHome: MI455X/Helios at CES 2026 — https://www.servethehome.com/amds-epyc-venice-instinct-mi455x-helios-hardware-on-display-for-first-time-at-ces-2026/
- Tom's Hardware: MI430X/MI440X/MI455X — https://www.tomshardware.com/tech-industry/artificial-intelligence/amd-touts-instinct-mi430x-mi440x-and-mi455x-ai-accelerators-and-helios-rack-scale-ai-architecture-at-ces-full-mi400-series-family-fulfills-a-broad-range-of-infrastructure-and-customer-requirements
- MI355X datasheet (NEC mirror) — https://www.nec.com/en/global/solutions/hpc/lx/images/AMD/amd-instinct-mi355x-gpu-datasheet.pdf
- ROCm compatibility matrix — https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
