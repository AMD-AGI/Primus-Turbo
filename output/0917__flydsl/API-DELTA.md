# flydsl 0.2.4 vs aiter 所需 API：差集与 wave32 陷阱

日期 2026-09-17 · **全程零 GPU**：所有结论来自 `docker run --rm --network none --entrypoint python3 fa-tune:deps`，
**不挂 `--device /dev/kfd`**，容器物理上碰不到卡。

标注：**实测** = 本次在容器里直接执行得到 / **源码** = 读 0.2.4 包内源码。

---

## 0. 结论先行

| 问题 | 答案 |
|---|---|
| aiter 的 gfx1250 前向能否在 flydsl **0.2.4** 下工作？ | **能，只缺 4 个工具函数**，都是几行的 shim。**不必升级 0.3.2** |
| `T.f8` 在 0.2.4 里还在吗？ | **在**（`T.bf16/f16/f32/f8/i32` 全部可用）。它是 **0.3.0** 移除的，aiter 的 shim 是为 0.3.x 写的，对我们是多余的 |
| gfx1250 的原语在 0.2.4 里齐吗？ | **全齐**，8/8 |
| wave32 误判能用环境变量绕过吗？ | **不能。** 见 §2，这是本文最重要的一节 |

`setup.py:548` 的注释"flydsl 0.2.4 不支持 gfx1250"被包内容推翻。

---

## 1. API 差集

审计范围：`aiter/ops/flydsl/kernels/fmha_gfx1250/*.py` 与 `kernels/kernels_common.py`
里出现的全部 `flydsl` 引用。容器内 `flydsl.__version__ = 0.2.4`，
路径 `/opt/venv/lib/python3.12/site-packages/flydsl`。**实测**

### 1.1 模块：15/15 全部存在

`flydsl.compiler`、`flydsl.expr`、`.expr.{arith,gpu,rocdl,rocdl.tdm_ops,math,typing,primitive}`、
`.expr.utils.arith`、`flydsl._mlir.{ir,dialects.builtin,dialects.gpu,dialects.llvm}`、
`flydsl.runtime.device`。

### 1.2 gfx1250 原语：8/8 全部存在

`rocdl.WMMA`、`rocdl.MFMA`、`rocdl.sched_barrier`、`rocdl.s_wait_asynccnt`、
`rocdl.s_wait_dscnt`、`rocdl.ds_load_tr16_b128`、`rocdl.make_tdm_atom`、
`rocdl.cluster_load_async_to_lds`。

`tdm_ops` 里可用：`TDMDescriptor2D`、`TDMGatherDescriptor`、`make_tensor_descriptor_2d`、
`make_tensor_gather_descriptor`、`make_tensor_gather_dgroup0`、`add_addr_with_carry`、
`l2_prefetch_tile`、`compute_padding_encoding`、`compute_warp_distribution`。

> `add_addr_with_carry` 就是 `optimization-directions.md:172` 那条死路
> （"TDM gather 缺 `addr64` carry-safe update → 大张量硬挂"）所需的那个函数。**它在 0.2.4 里。**

### 1.3 需要 shim 的四个符号

| 符号 | 0.2.4 状态 | aiter 的调用形态 | 影响面 |
|---|---|---|---|
| `fx.ceildiv` | **包内任何地方都没有定义** | `fx.ceildiv(fx.Uint32(max_seqlen_q * gqa_ratio), ...)`（2 处） | 前向 grid 计算 |
| `fx.max` | `flydsl.expr` 上没有。包里只有 `_mlir/dialects/_math_ops_gen.py` 的 **math dialect 浮点** `max`，**不是**这里要的整数/index 版本 | `fx.max(rem, fx.Int32(0))`、`fx.max(valid_rows - fx.Int32(1), ...)`（5 处） | causal / window 边界 |
| `fx.min` | 同上 | `fx.min(clean_hi, n_last)`、`fx.min(kv_len_wg, kv_len)`（5 处） | causal / window 边界 |
| `fx.to_llvm_ptr` | **包内任何地方都没有定义** | `kernels_common.py:90` `fx.to_llvm_ptr(fx.get_iter(memref) + offset)`；`:194` `fx.to_llvm_ptr(fx.inttoptr(pt, value))` | **共享 helper 模块，在关键路径上** |

全部为几行的工具函数，无一涉及 codegen 语义。**结论：写 shim，不升级。**

升级 0.3.2 的代价在另一侧：Primus-Turbo 自己 pin 0.2.4，升级会同时动到
`primus_turbo/flydsl/` 下的 GEMM / quantization / attention 全部代码，破坏面远大于 4 个 shim。

### 1.4 其余子属性：全部存在

`fx.AddressSpace.{Global,Shared}`、`fx.PointerType.get`、`fx.Vector.{filled,from_elements,make_type}`、
`fx.Int32.ir_type`、`fx.Float8E4M3FN{,UZ}.ir_type`、`fx.{add_offset,copy_atom_call,get_iter,inttoptr,
log2,make_layout,make_view,ptr_load,ptrtoint,range_constexpr,as_ir_value}`、
`flyc.{jit,kernel}`、`expr.utils.arith._to_raw`、`expr.primitive.const_expr`。

> 审计脚本第一版把 `T.*` 全报成 MISSING，那是**假失败**：`T.bf16` 会走到
> `BFloat16.ir_type` → `BF16Type.get()`，没有 MLIR Context 就抛 `RuntimeError`。
> 套上 `with Context():` 之后 5/5 全过。这正是 skill §1 那条"先问这个信号会不会说谎"。

---

## 2. wave32 误判 —— 三条实测结论，每一条都推翻一个自然的假设

`flydsl/runtime/device.py:76` `is_rdna_arch()` 的分类规则（**源码**）：

```python
if arch.startswith("gfx10") or arch.startswith("gfx11"):  return True
if arch.startswith("gfx120"):                             return True
```

`gfx1250` 以 `gfx125` 开头，**三条都不匹配**。容器内直接执行（**实测**）：

```
is_rdna_arch('gfx1250') = False   -> warp_size 64      <-- 错
is_rdna_arch('gfx1201') = True    -> warp_size 32
is_rdna_arch('gfx1200') = True    -> warp_size 32
is_rdna_arch('gfx950')  = False   -> warp_size 64
```

### 2.1 它影响的不止 wave size —— buffer descriptor 也错

包内只有 **4 个** wave-size 决策点，**全部经由这一个函数**（**源码**）：

| 位置 | 作用 |
|---|---|
| `compiler/backends/rocm.py:21` | `warp_size = 32 if is_rdna_arch(arch) else 64` → `GPUTarget` |
| `compiler/backends/rocm.py:26` | 同上，第二个重载 |
| `compiler/backends/rocm.py:61` | `"wave64": "false" if is_rdna_arch(chip) else "true"` → LLVM target feature |
| **`expr/buffer_ops.py:68`** | **RDNA 下额外置 `bit 24`（RDNA 上必须为 1）与 `OOB_SELECT = 2`** |

第四条是新发现，而且**比 wave size 更要命**：被误判成 CDNA 之后，buffer descriptor
少了 RDNA 必须置 1 的保留位，OOB_SELECT 也不同。
**而 `fmha_bwd_gfx942` 那份反向模板的全部越界安全（causal 尾块、空 workgroup、每一处 epilogue mask）
正是建立在 buffer descriptor 的 `num_records` 上**（见 `SURVEY.md` §10.2）。
也就是说：这个误判会同时打掉我们打算依赖的那套越界保护。

### 2.2 环境变量**不是**逃生口（**实测**）

```
FLYDSL_GPU_ARCH=gfx1250
  get_rocm_arch() = 'gfx1250'      <-- 字符串设对了
  is_rdna_arch()  = False          <-- 仍然错
```

`FLYDSL_GPU_ARCH` 只决定 arch **字符串**，而错的是对这个字符串的**分类规则**。
`HSA_OVERRIDE_GFX_VERSION` 同理。**不要指望用环境变量绕过去。**

### 2.3 只打 `runtime.device` 的猴子补丁**到不了消费方**（**实测**）

`compiler/backends/rocm.py:6` 和 `expr/buffer_ops.py:33` 都是
`from ...runtime.device import is_rdna_arch` —— **导入时绑定名字**。实测：

```
patch flydsl.runtime.device.is_rdna_arch  ->
  dev.is_rdna_arch('gfx1250')      = True
  rocm_be.is_rdna_arch('gfx1250')  = False   <-- 旧绑定，没被改到
  bops.is_rdna_arch('gfx1250')     = False   <-- 旧绑定，没被改到
```

**所以覆盖必须打在三个命名空间上**（`flydsl.runtime.device`、
`flydsl.compiler.backends.rocm`、`flydsl.expr.buffer_ops`），
或者在这两个模块被 import **之前**打。

这是个典型的"改了、但没生效、而且不报错"——与 skill `gpu-kernel-campaign` §2
"absence is not evidence"同源。**覆盖装上之后必须验证消费方读到的是新值，不能只验证自己改的那个。**

aiter 的做法是完全不信任它：在 `kernels/kernels_common.py:52-69` 自己重判。
如果我们只调用 aiter 的 kernel，走的是 aiter 的判断；
**一旦我们自己写 FlyDSL kernel（Phase D 的反向），就必须自己装覆盖。**

---

## 3. 无设备时会静默回落到 gfx942（**实测**）

```
（容器内，不挂 /dev/kfd，且不设 FLYDSL_GPU_ARCH）
  get_rocm_arch() = 'gfx942'
  is_rdna_arch()  = False
```

**不是空字符串，是 `'gfx942'`。** 所以任何"解析目标"的静态审计，
在无设备环境下会**静默回答一个 gfx942 的问题**，而且看起来完全正常。

**规则**：任何会解析编译目标的离线审计，必须显式设 `FLYDSL_GPU_ARCH` 并
**断言解析出的字符串等于 `gfx1250`**。没有这条断言的"审计干净"当作没有结果。

> 本文档 §1 的符号审计**不受此影响**：它只查符号是否存在，不解析目标，
> 也不依赖目标。这个区分是刻意的。

---

## 4. 对计划的影响

| | 影响 |
|---|---|
| Phase A | 差集已出：4 个 shim，不升级 0.3.2。**本节完成** |
| Phase B（前向对拍） | 调的是 aiter 的 kernel，走 aiter 自己的 wave32 重判，**不需要我们的覆盖**。但仍要在测量里记录实际 warp_size，不能假设 |
| **Phase D（新写反向）** | **必须自己装 wave32 覆盖，打在三个命名空间上，并验证消费方读到新值。**这是发射前置条件，不是文档项 |
| Phase D 的越界安全 | 模板依赖 buffer descriptor `num_records`；wave32 误判会连带打掉它。两件事要一起验 |
| 任何离线审计 | 必须断言 resolved target == `gfx1250`，否则默认答的是 gfx942 |
