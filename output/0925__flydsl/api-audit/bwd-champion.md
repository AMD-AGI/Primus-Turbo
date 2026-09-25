# bwd champion — FlyDSL API 稳定性审计与保守清理

## Stable API usage audit — NOT STABLE-ONLY

- **范围**：`output/0925__flydsl/bwd341/op_clean/{kernels.py,impl.py}`。这是 gfx1250 FA 反向的 7 个 kernel：`k_delta_bshd`、`k_dkdv`、`k_dkdv_sp`、`k_dq`、`k_dq_sp`、`k_redsp`、`k_redsp_q`。
- **只动了 op_clean**：`op0341/`（未改动的对照副本）、`op032/` 和原件都没有碰。
- **判定依据**：`v0.3.4.1` tag 下的 `docs/api_stability.md`。稳定集用 tag 自带的 `scripts/list_stable_apis.py --repo-root <git archive v0.3.4.1>` 生成，共 493 条。当前 main 上跑出 531 条，多出的只有 `flydsl.extension.*` 和 `expr.struct.Empty`，与本 kernel 无关。已安装的 `~/.local/flydsl0341/flydsl` 源码与 tag 的 `python/flydsl` 一致。
- **行号**：一律指 **清理后** 的 `op_clean/kernels.py`（下文简写 `k:`）。清理前的行号见 `op0341/kernels.py`。
- **结论**：清理后仍然 NOT STABLE-ONLY。原因是剩下的 gfx1250 WMMA、tr16 转置读、原始 LDS store、sched_barrier、`rocdl.exp2`，它们在 0.3.4.1 里都没有保证代码生成不变的稳定替代。清理前 kernel 还经 aiter 间接依赖了私有 API，现在已经去掉。

## Stable uses

以下每个路径都已按 `list_stable_apis.py` 的输出（§2.1 / §2.2）逐条核对。类型对象上的方法和属性，依据 §1 的 returned-object 规则判为稳定。

- `flydsl.compiler.kernel`：k:120 599 608 983 992 1054 1115。
- `flydsl.compiler.jit`：k:159 502 513 618 628 880 894 911 1002 1012 1073 1100 1128 1143。
- `flydsl.compiler.compile`：impl.py:68（`_flyc.compile`）。
- `flydsl.expr.numeric.{Int32, Int64, Float32, BFloat16, Index}`：全文件，例如 k:75、k:122、k:505。`Float32.ir_type` 在 k:110、k:115。
- `flydsl.expr.typing.{Tensor, Vector, Stream}`：kernel 签名，以及 k:258–259、k:732–733。用到的 Vector 成员有 `filled`、`from_elements`、`make_type`、`shuffle`、`bitcast`，另有下标取值、`.to()`、`.select()`，都属 §1。
- `flydsl.expr.typing.as_ir_value`：k:116，以及 k:391–1140 共 32 处。**这是新引入的**，替换了原来的 aiter `_to_raw`，见下文“已应用的清理”。
- `flydsl.expr.primitive` 下的 `make_view`、`get_iter`、`make_layout`、`make_copy_atom`、`copy_atom_call`、`slice`、`memref_store_vec`、`ptrtoint`，位置在 k:77–104、k:253、k:731。
- `flydsl.expr.primitive` 下的 `PointerType`、`AddressSpace`、`inttoptr`、`to_llvm_ptr`，在 k:115–116。**同样是新引入的**，替换了原来的 aiter `create_llvm_ptr`。
- `flydsl.expr.primitive.range_constexpr`：k:41 导入，全文件使用。
- `flydsl.expr.primitive.const_expr`：由 AST rewriter 注入 globals，在 k:365 439 498 788 832 872。
- `flydsl.expr.derived.make_rmem_tensor`：k:86 92 98 106。
- `flydsl.expr.gpu` 下的 `thread_idx`、`block_idx`：k:124–125、k:180–195、k:672–711。`shuffle_xor` 在 k:148。`SharedAllocator` 在 k:252 和 k:730，调用链为 `.allocate().peek().ptr`。
- `flydsl.expr.rocdl.make_buffer_tensor` 在 k:77；`flydsl.expr.rocdl.BufferCopy` 在 k:82。
- `flydsl.expr.rocdl`：命名空间本身是稳定的（`_BACKEND_MODULES`），但下面列出的具体成员不在 `rocdl.__all__` 里。

## Private-field writes

- **无**。`kernels.py` 和 `impl.py` 都没有给 FlyDSL 对象的下划线属性赋值，也没有 `setattr` 或 `__dict__` 写入。
- 清理前有两处 **间接的私有读取**，本次都已去掉：
  - aiter `tensor_shim._to_raw` 回退时会读 `ir.Value._CAPICreate(v._CAPIPtr)`。
  - aiter `kernels_common.create_llvm_ptr` 会读 `ptr._value`。
- 两者都在 aiter 源码里（`aiter-src/aiter/ops/flydsl/kernels/tensor_shim.py:312`、`kernels_common.py:188`），不属于本审计范围的文件。但它们原本是本 kernel 的运行时依赖。

## Deprecated-unstable

**DEPRECATED（§3）：无。** 没有用到 `fx.get`、`fx.index_cast`、`fx.constant_vector`、`fx.tdm_ops`、`Numeric.maximumf/exp2/shuffle_xor` 或 `BufferCopyLDS64b`。

**UNSTABLE（清理后仍在）：**

| 路径 | 位置 | 原因 | 0.3.4.1 有无稳定替代 |
|---|---|---|---|
| `flydsl.expr.rocdl.sched_barrier` | k:304（mask=0 的全隔离栅栏）；注释在 k:42 | FlyDSL 包装函数（`rocdl/__init__.py:142`），但不在 `rocdl.__all__` 中 | **无**。稳定的 `sched_mfma/vmem/dsrd/dswr` 都是 `sched_group_barrier`，语义不同。g51 靠这道栅栏把 4 条 b32 钉在 32 条 b128 前面，不能换。 |
| `flydsl.expr.rocdl.{wmma_f32_16x16x32_bf16, ds_load_tr16_b128, exp2}` | 见下一节 | 原始 ODS builder，不在 `__all__` 中，同时属于 UPSTREAM-MLIR | 见下一节 |
| `flydsl._mlir.dialects.llvm` | k:40 导入；注释在 k:39 | `flydsl._mlir.*` 按 §2.5 属于 unstable | 见下一节 |

**清理前有、现已去掉的 UNSTABLE 依赖：**

- 旧 k:42 的 `from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir`：
  - 下划线名，§2 第一条即判为 unstable。
  - 它是 aiter 的 `_to_raw` 副本，也就是 kernel-code-cleanup §1 列为 deprecated 的 `arith._to_raw`。
  - 内部还用了 `ir.Value._CAPICreate`。
  - 共 33 处调用，已全部换成 `fx.as_ir_value`。
- 旧 k:41 的 `from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr`：
  - 这是 aiter 的 helper，内部读 `ptr._value`。
  - 共 9 处调用，已换成本地的 `_lds_ptr`（k:113），只用稳定的 fx 原语拼成。

## Upstream MLIR

以下调用都是直接使用上游 MLIR 方言的 op，按 **§2.5 允许使用但属于 unstable**。FlyDSL 不保证它们的名字、签名和语义在各版本之间不变。每个调用点单独列一条。

- `[UPSTREAM-MLIR]` **`rocdl.wmma_f32_16x16x32_bf16`**：导入 `from flydsl.expr import rocdl`（k:44），是 `_mlir.dialects.rocdl` 的 ODS 经 star 再导出。
  - 调用点：k:427 和 k:430（dkdv 的 S/P GEMM，dt 链拆分）；k:492 和 k:495（dkdv 的 dV/dK GEMM）；k:824 和 k:827（dq 的 S/P GEMM）；k:869（dq 的 dQ GEMM）。
  - 源码 7 处，ISA 中展开为 dkdv 128 条、dq 192 条 `v_wmma`。
- `[UPSTREAM-MLIR]` **`rocdl.ds_load_tr16_b128`**：k:472、k:474（dkdv 的 `tr()` helper）；k:863、k:865（dq 的 `b_k`）。gfx1250 专有。
- `[UPSTREAM-MLIR]` **`rocdl.exp2`**：经 `fx.rocdl.exp2` 调用，位置 k:110（`_exp2`），被 k:448 和 k:841 使用；注释在 k:109。
- `[UPSTREAM-MLIR]` **`rocdl.sched_barrier`**：k:304，经 FlyDSL 包装函数调用，底层是 ODS 的 `_ods_sched_barrier`。
- `[UPSTREAM-MLIR]` **`llvm.store`**：导入 `from flydsl._mlir.dialects import llvm as llvm_dialect`（k:40）。
  - 调用点：k:391 和 k:393（dkdv 把 dO/Q 暂存到 LDS）；k:454 和 k:457（dkdv 的 P/dS 写入 LDS）；k:818（dq 把 K 暂存到 LDS）。
  - 共 5 处，都是 `!llvm.ptr<3>` 上的 16 B store。
- 已不再使用的原始 `fly` 方言绑定：`fx.inttoptr` 和 `fx.to_llvm_ptr` 内部走 `fly.inttoptr` / `fly.to_llvm_ptr`，但调用方拼写的是稳定的 `fx.*` 路径，所以不算。

## Unresolved

- **`for it, carried in range(fx.Index(..), fx.Index(n), 1, init=state)` 加 `final = yield ...`**：出现在 k:502–535、k:880–917、k:1073–1086、k:1128–1138。
  - 这是 AST rewriter 的循环契约：`range` 被改写成 `scf.for`，`yield` 作为 carried state。它不是一个 API 路径，§2 的机械规则没法给它定级。
  - 语法本身由 ast_rewriter 稳定支持，但 “flyc.jit 对单元素 carried state 在返回边界自动解包” 的行为（k:1143 的归一化代码依赖它）没有文档承诺。
- **`const_expr`**：没有 import，靠 rewriter 注入 globals（`compiler/ast_rewriter.py:763`）。它本身是稳定的 `fx.const_expr`，但裸名的来源是动态绑定。
- **`_G07_CLAMP = True`**（k:1023）：这不是死代码。外部筛选脚本通过 `getattr(k, "_G07_CLAMP", False)` 选择调用签名（`0922__flydsl/r9probe/work/screen_mw.py:256`、`0925__flydsl/g1b-4wave/screen4w.py:256`），所以保留。
- **过时注释，未改动**：
  - k:183 的 “h8 PROBE (throwaway, never shipped)” 描述的 x/y 布局，正是现在出货的 g03 布局，注释与现状不符。
  - k:248 的 “740 VGPR” 与本次编译得到的 `k_dkdv` 904 VGPR 不一致。
  - 两者都是性能历史记录，按“保守”原则保留，只在这里标出。

## 已应用的清理（op_clean/kernels.py；impl.py 未改动）

| # | 改动 | 依据 |
|---|---|---|
| 1 | `_to_raw as _ir` 共 33 处换成 `fx.as_ir_value(...)`，并删除 aiter 的导入 | cleanup §1。0.3.4.1 中 `Vector` 是 `ir.Value` 的子类，`_to_raw`、`.ir_value()`、`as_ir_value` 返回的是 **同一个对象**，WMMA `.result` 这类原始 `ir.Value` 也原样透传。选 `as_ir_value` 而不是 `.ir_value()`，是因为循环 carried 列表里混有原始 `ir.Value`。 |
| 2 | `create_llvm_ptr(x, address_space=3)` 共 9 处换成本地 `_lds_ptr(x)`，删除 aiter 导入 | cleanup §3b。`_lds_ptr` 发出的 op 与 aiter 版完全相同：`PointerType.get(i32, Shared, align 4)`、`inttoptr`、`to_llvm_ptr`，但只用稳定 fx 路径，也不再读 `_value`。副作用：`kernels.py` 不再 import 任何 aiter 模块，编译驱动打印 `aiter modules loaded: []`。 |
| 3 | 删除非 PARTIAL 路径里的死值 `sp = fx.Int32(0)`（dkdv、dq 各一处） | cleanup §8。这两处 `sp` 只在 PARTIAL 分支里被读取。 |
| 4 | 删除重复的 r8.i1.g23+g24 注释块：每个 kernel 原有两份，保留前一份 | cleanup §8 |
| 5 | 修正过时注释 | 模块 docstring 里 dq/dk/dv 的 dtype 从 fp32 改为 bf16，并说明 split-K 的 fp32 workspace；`_env` 的版本从 0.3.2 改为 0.3.4.1；k:296 注释里去掉对 `_to_raw` 的说明 |
| 6 | 去掉两处函数体内连续两个空行 | cleanup §8 |
| 7 | 新增 3 行 `# UNSTABLE(gfx1250): ...` 注释，只加在导入或定义处 | k:39（llvm）、k:42（rocdl 的 wmma/tr16/sched_barrier）、k:109（exp2，定义在 k:108） |

## 建议但未应用（需要上卡 A/B）

| 建议 | 稳定路径 | 为什么没直接改 |
|---|---|---|
| 原始 `rocdl.wmma_f32_16x16x32_bf16` 改为 `fx.make_mma_atom(fx.rocdl.WMMA(16,16,32,fx.BFloat16))` 加 `fx.gemm` 或 `mma_atom_call` | 稳定（`rocdl.WMMA` 在 `__all__` 中，支持 gfx1250） | 属于结构性改动。fragment 由 atom 打包，会影响 dt 链拆分（g61）、carried 元组形状和寄存器分配。dkdv 904 VGPR、dq 960 VGPR 都已贴着 1 wave/SIMD 的线，ISA 很可能变化。 |
| `rocdl.exp2` 改为 `fx.math.exp2` | 稳定 | 前者直接发裸 `v_exp_f32`；后者走 `math.exp2` 到 `llvm.exp2`，AMDGPU 会加 denorm 范围缩放，除非带 afn。热循环的 VALU 数会变。 |
| `llvm.store` 写 LDS 改为 `SharedAllocator` 的 struct view 加 `fx.copy` / `memref_store_vec` | 稳定 | 要把整个 LDS 布局（padding 行距 `X_ROW_B`、`S_ROW_B`，以及 +64 KiB 的段分离 g39）迁到 layout view 上。对齐和别名元数据可能不同，`ds_store_b128` 的形态和调度可能跟着变。 |
| `ds_load_tr16_b128` | **无稳定替代** | `rocdl.lds_transpose_load` 不在 `__all__` 中。`fx.rocdl.cdna4.LDSReadTrans` 是 gfx950 的 `ds_read_tr` 系列，按 PROVENANCE 记录在 gfx1250 上无法选指令。 |
| `rocdl.sched_barrier(0)` | **无稳定替代** | 保留。 |
| `range(fx.Index(fx.Int32(0)), ...)` 改为 `range(0, ...)`，以及 `fx.Index(fx.Int32(it))` 一类冗余包装 | 稳定 | 改动的是循环构造，canonicalize 之后大概率一样。但这属于循环契约，本次不动；以后要改，先做编译期 ISA diff 即可，不必上卡。 |

## 验证（只编译，未上卡）

- **编译方式**：
  - 驱动脚本 `api-audit/.bwd_work/compile_bwd.py` 在 `fa-repro` 容器里运行，环境变量为 `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1`，cache 目录各用各的 `/tmp/flycache_agent_bwdclean_*`。
  - 张量全部建在 `device="meta"` 上，stream 传 None，直接 `flyc.compile(launcher, *args)`。
  - 为了不执行 `aiter/__init__`，给 aiter 包链打了桩。
- **shape**：prod，即 b4 s8192 hq32 hkv8 d128 causal bf16，走 delta、dkdv（nsp=1）、dq（nsp_q=1）三个 kernel。split-K 的四个 kernel（dkdv_sp、dq_sp、redsp、redsp_q）按 nsp=2 一并编译；它们所有与 shape 相关的量都是运行时 Int32。
- **三组 dump**（`api-audit/.bwd_work/`）：
  - `before/`：来自未改动的 `op0341`。
  - `control/`：来自改动前的 `op_clean`。它与 before 的 7 个 `22_final_isa.s` 逐字节相同，说明 dump 是确定的，且与源码路径无关。
  - `after/`：来自改动后的 `op_clean`。
- **before 与 after 对比：7 个 kernel 全部逐字节相同**，包括 `22_final_isa.s`、`21_llvm_ir.ll` 和 `00_origin.mlir`。

| kernel | VGPR | SGPR | LDS B | scratch | 溢出 v/s | 指令数（before/after） | v_wmma | 指令直方图 |
|---|---|---|---|---|---|---|---|---|
| k_delta_bshd | 40 | 21 | 0 | 0 | 0/0 | 256/256 | 0 | 相同 |
| k_dkdv | 904 | 79 | 70656 | 0 | 0/0 | 3168/3168 | 128 | 相同 |
| k_dkdv_sp | 910 | 84 | 70656 | 0 | 0/0 | 2851/2851 | 128 | 相同 |
| k_dq | 960 | 89 | 8704 | 0 | 0/0 | 3567/3567 | 192 | 相同 |
| k_dq_sp | 968 | 94 | 8704 | 0 | 0/0 | 3204/3204 | 192 | 相同 |
| k_redsp | 65 | 28 | 0 | 0 | 0/0 | 192/192 | 0 | 相同 |
| k_redsp_q | 41 | 24 | 0 | 0 | 0/0 | 139/139 | 0 | 相同 |

**结论：ISA 完全一致，不需要上卡 A/B。** 上面“建议但未应用”那一节的项目，才需要上卡 A/B。

## Verification（独立对抗复核，只编译，未上卡）

目标是推翻“等价”结论。结论：**没能推翻，没有回退任何 hunk。**

- **impl.py**：与 `op0341/impl.py` 逐字节相同。
- **kernels.py 逐 hunk 检查**（`diff -u op0341 op_clean`，325 行）：
  - `_ir(x)` 改为 `fx.as_ir_value(x)`，共 33 处。aiter 的 `_to_raw` 对 `ir.Value` 原样返回，否则递归调用 `.ir_value()`。0.3.4.1 的 `as_ir_value` 对 `ir.Value` 同样原样返回，对 list/tuple 递归，对有 `__extract_to_ir_values__` 或 `ir_value()` 的对象取值。在这些调用点（`Vector` 以及 wmma 返回的 `ir.Value`），两者产生同一个值。没有 dtype 或符号性变化。
  - `create_llvm_ptr(x, address_space=3)` 改为 `_lds_ptr(x)`，共 9 处。对照 aiter-src 的 `kernels_common.py:188`：`FX_ADDRESS_SPACE[3]` 就是 `fx.AddressSpace.Shared`，`PointerType.get(Int32, alignment=4)`、`inttoptr`、`to_llvm_ptr` 三步完全一致，只是末尾的 `._value` 换成了 `as_ir_value`。地址算式（`lds_* + off`、`base + 16*rowb`、`lds_k + ko + dt*64+u*32`）逐字保留，没有改动，也没有调换顺序。
  - 删掉的 `sp = fx.Int32(0)`（dkdv、dq 各 1 处）：`sp` 的所有使用点（k:561/565/573、k:945/954/958）都在 `if PARTIAL:` 里面。`PARTIAL` 是 Python 常量，所以非 PARTIAL 路径里这是死代码。
  - 删掉的 2 个 r8.i1.g23+g24 注释块：与保留下来的那份逐字相同，而且原位置在函数尾部 `if const_expr(carry)` 之前，本来就放错了地方。纯注释。
  - 新增和修改的注释：模块 docstring 写 dq/dk/dv 为 bf16，对照 impl.py:170/190/213/224（`dtype=q.dtype/k.dtype`，入口 assert bf16）属实；split-K 的 fp32 workspace 对照 impl.py:177/219 属实。`_env` 实际预置的是 `flydsl0341`，所以 k:28 的新注释属实（`_env.py` 自己的 docstring 仍写 0.3.2，但该文件不在改动范围内）。`UNSTABLE` 注释中“不在 `fx.rocdl.__all__` 中”这一点，在容器里核实了：`wmma_f32_16x16x32_bf16`、`ds_load_tr16_b128`、`sched_barrier`、`exp2` 都不在 `__all__` 中，但都能通过属性访问到，且 `rocdl is fx.rocdl`。k:109 说 `fx.math.exp2` 会带范围缩放，这一点本次没有独立验证，属于合理推断，不影响代码。
- **重新编译**：在 `fa-repro` 中进行，参数为 `COMPILE_ONLY=1 FLYDSL_RUNTIME_ENABLE_CACHE=0`，cache 目录为 `/tmp/flycache_agent_verifier*`，沿用同一个 `compile_bwd.py`。当前 `op_clean` 和 `op0341` **各重新编一次**，不依赖上一轮留下的 dump：
  - 当前 `op_clean` 与 `.bwd_work/before/`：7 个 kernel 的 `00_origin.mlir`、`21_llvm_ir.ll`、`22_final_isa.s` 全部相同。
  - 当前 `op_clean` 与刚编出的 `op0341`：**161 个 dump 文件（所有 pass 的中间产物）逐字节相同**。dkdv 的 origin MLIR 有 1.26 MB，ISA 有 164 KB，不是空文件。
- **覆盖面说明**：只编了 causal=1 的 prod shape 和 nsp=2 的 split-K。causal、nsp 以及各个 shape 都是运行时 Int32，所以 IR 与它们的取值无关。编译期常量只有 `PARTIAL` 和 `carry`，两种取值都已覆盖。
