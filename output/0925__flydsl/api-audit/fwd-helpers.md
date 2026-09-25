# FlyDSL API 稳定性审计 —— fwd 辅助模块（tensor_shim / kernels_common / buffer_ops）

## 结论：NOT STABLE-ONLY（可接受）；清理后 ISA 逐字节一致

- 范围：`output/0925__flydsl/fwd341/op_clean/flydsl_fwd/{tensor_shim.py,kernels_common.py,buffer_ops.py}`。
  下文 `op_clean/...:N` 是清理后的行号，`op0341/...:N` 是原始行号（与清理前的 op_clean 相同）。
- 策略依据：FlyDSL `v0.3.4.1` 标签下的 `docs/api_stability.md`（与已安装 wheel 一致：
  `/home/lihuzhan/.local/flydsl0341/flydsl/expr` 和 `git worktree v0.3.4.1:python/flydsl/expr` 的 diff 为空）。
  稳定集合用 `scripts/list_stable_apis.py`（v0.3.4.1）生成，共 493 条路径。0.3.4.1 版的这个脚本
  没有 `--include-deprecated`，§3 那张表是另外手工核对的。
- 清理后，`kernels_common.py` 已经 **stable-only**；`tensor_shim.py` 只剩对
  `flydsl._mlir.ir` 的访问（保留下来的 `_to_raw` 兼容 shim，以及 compile 失败时弹出 Context 的清理逻辑）；
  `buffer_ops.py` 仍然依赖 raw ROCDL/LLVM/arith builder。换成稳定路径必然要换 buffer 存储机制，
  所以这部分只列为"提议"，没有动手。
- 仍在用的 FlyDSL 私有字段写入：**0 处**（原有 2 处，已修掉 1 处、随死代码删掉 1 处）。
- 验证：prod 形状（b4 s8192 hq32 hkv8 d128 causal bf16，return_lse）compile-only。
  清理前后的 `22_final_isa.s`、`21_llvm_ir.ll`、`00_origin.mlir` **逐字节一致**。

## 1. Stable uses（清理后）

| 解析后的路径 | 位置 |
|---|---|
| `flydsl.compiler.compile` | op_clean/tensor_shim.py:30 |
| `flydsl.expr.typing.as_ir_value`（经 `from flydsl.expr import as_ir_value`，等价别名） | op_clean/kernels_common.py:9, :26 |
| `flydsl.expr.primitive.AddressSpace`（`fx.AddressSpace.Global/Shared`） | op_clean/kernels_common.py:17 |
| `flydsl.expr.primitive.PointerType.get`（稳定类型上的公开成员） | op_clean/kernels_common.py:25 |
| `flydsl.expr.primitive.inttoptr`, `flydsl.expr.primitive.to_llvm_ptr` | op_clean/kernels_common.py:26 |
| `flydsl.expr.numeric.Int32.ir_type` | op_clean/kernels_common.py:25 |
| `flydsl.expr.numeric.Int8.ir_type`（替换原来的 `_mlir.extras.types.i8()`） | op_clean/buffer_ops.py:134 |
| `fx.Int64(...).ir_value()` | op_clean/buffer_ops.py:147, :268, :270, :274, :280, :283 |
| `fx.Int32(...)` / `.ir_value()` / 运算符 | op_clean/buffer_ops.py:235, :362, :375, :382-384, :388, :395 |
| `fx.Int16(...).ir_value()` | op_clean/buffer_ops.py:236 |
| `fx.Boolean(...).select(...)`（返回对象规则） | op_clean/buffer_ops.py:379 |
| `flydsl.runtime.device.is_rdna_arch`（§2.4 显式列表） | op_clean/buffer_ops.py:39, :80 |

清理前另有这些稳定用法，所在的代码已作为死代码删除：`flyc.from_c_void_p`（op0341/tensor_shim.py:262-263）、
`flydsl.compiler.protocol.extract_to_ir_values`（:16, :85, :201）、`fx.rocdl.make_buffer_tensor` /
`get_buffer_rsrc` / `BufferCopy{8,16,32,64,128}b`（:65-69, :143, :163, :545）、`fx.make_view` / `fx.make_layout` /
`fx.copy` / `fx.make_rmem_tensor` / `fx.memref_{load,store}_vec`（:133-240, :518-540）、`Vector`、`T`、`ptrtoint`、
`range_constexpr`，以及 `flydsl.runtime.device.get_rocm_arch`（op0341/kernels_common.py:18）。

## 2. Private-field writes

**清理后：无。**

- [PRIVATE-WRITE，已修复] op0341/tensor_shim.py:276 `exe._cf = cf`（读取在 :270 `getattr(exe, "_cf", None)`）。
  这是往稳定对象 `@flyc.jit` 的 `JitFunction` 上挂一个 FlyDSL 本身没有定义的字段；0.3.4.1 源码里
  搜不到 `_cf`。它没有绕过任何校验，但 `_launch_fns` 里的 launcher 是模块级、跨调用共享的对象，
  这个字段也就跟着共享。
  之所以要这样绕，是因为缺一个公开 API：FlyDSL 没有提供"把 CompiledFunction 缓存在 JitFunction 上"的接口。
  → 改成模块级的 `_COMPILED = {}`，以 exe 对象为键（op_clean/tensor_shim.py:18, :25, :31）。
  两种写法语义一致：`JitFunction` 没有自定义 `__eq__`/`__hash__`，按对象身份比较；launcher 本来就被
  `_launch_fns` 永久持有，所以不会多出新的生命周期问题。只影响 host 端逻辑，与 codegen 无关。
- [PRIVATE-WRITE，随死代码删除] op0341/kernels_common.py:232 `jit_func._compiled_cache = compiled_cache`，
  位于 `run_cached()` 内（:211-243），本包里没有调用方。

## 3. Deprecated / Unstable uses（清理后）

**DEPRECATED（§3）：这 3 个文件里没有。** 没有用到 `fx.get`、`fx.index_cast`、`fx.constant_vector`、
`fx.tdm_ops`、`Numeric.{maximumf,minimumf,shrui,addf,exp2,shuffle_xor}` 或 `BufferCopyLDS64b`。
（顺带看了一眼：同包的 fwd kernel 用的是 `from flydsl.expr.rocdl import tdm_ops`，不是已废弃的 `fx.tdm_ops` 别名，
但这条路径本身属于 unstable。那个文件不在本次范围。）

**UNSTABLE：**

- `flydsl._mlir.ir`（§2.5：`_mlir` 属于下划线段）
  - op_clean/tensor_shim.py:13；`ir.Context.current.__exit__` 在 :35-36（compile 失败时弹出泄漏的 Context）；
    `ir.Value` 在 :48、:52。
  - op_clean/buffer_ops.py:35；`ir.Value/IndexType/IntegerType/IntegerAttr/Type.parse/MemRefType/VectorType`
    在 :102, :106, :146, :148, :160, :228, :241, :286, :386。
- `ir.Value._CAPICreate(v._CAPIPtr)`：raw binding 的下划线成员，op_clean/tensor_shim.py:52（`_to_raw` 的兜底分支）。
  fmha_b16_buffer_managers.py:52 还在 `from .tensor_shim import _to_raw as _ir`，所以 `_to_raw` 保留下来，
  docstring 里标注了应改用 `.ir_value()` / `fx.as_ir_value`。
- `flydsl.expr.meta.dsl_loc_tracing`：op_clean/buffer_ops.py:38，作为装饰器用在 :117, :198, :294, :332。
  `meta` 不在 `expr/__init__.py` 的 `from .<m> import *` 聚合列表里，所以即使它自己有 `__all__` 也不算稳定。
  它只影响 MLIR location，没有稳定替代，保留。
- raw FlyDSL `fly` dialect binding：`flydsl._mlir.dialects.fly.extract_aligned_pointer_as_index`，
  op_clean/buffer_ops.py:226, :229（这是 FlyDSL 自家的 raw 绑定，不算 UPSTREAM-MLIR）。
- 鸭子类型读私有字段 `value._value` / `value.value`：op_clean/buffer_ops.py:108, :110（`_unwrap_value`）。
- 靠 introspection 探测 upstream 签名 `inspect.signature(rocdl.RawPtrBufferLoadOp)`：op_clean/buffer_ops.py:43-46。
  0.3.4.1 上 `aux` 是 KEYWORD_ONLY，也就是走 attribute 分支。这个探测存在的原因，正是这个 op 的签名在 upstream 变过。

清理前另有这些 unstable 用法，都在已删除的死代码里：`flydsl._mlir.extras.types`（op0341/buffer_ops.py:39）、
`fly.extract_aligned_pointer_as_index`（op0341/tensor_shim.py:91）、`_to_raw` 在 :163、:172，
`ptr._value` 私有读（op0341/kernels_common.py:195，已改成稳定的 `as_ir_value`）。

## 4. Upstream MLIR operations（清理后）

以下每一处都是：直接调用 upstream MLIR builder，允许使用，但按 docs/api_stability.md §2.5 属于 unstable，
FlyDSL 不保证它们的名字、签名和语义跨版本不变；有稳定封装时应优先用封装。

- [UPSTREAM-MLIR] `rocdl.RawPtrBufferLoadOp`（只用于签名探测，不发射 op）：op_clean/buffer_ops.py:44
- [UPSTREAM-MLIR] `arith.ConstantOp` / `arith.AddIOp`（别名 `std_arith`，函数内导入）：op_clean/buffer_ops.py:157, :162, :165
- [UPSTREAM-MLIR] `llvm.GEPOp`：op_clean/buffer_ops.py:171（`get_element_ptr`，managers 有 8 处调用，热路径上的 LDS 地址计算）
- [UPSTREAM-MLIR] `rocdl.MakeBufferRsrcOp`：op_clean/buffer_ops.py:287（`create_buffer_resource`，O/LSE epilogue 使用）
- [UPSTREAM-MLIR] `rocdl.RawPtrBufferStoreOp`：op_clean/buffer_ops.py:392（`buffer_store`，O/LSE epilogue 使用）
- 导入点：op_clean/buffer_ops.py:37 `from flydsl._mlir.dialects import llvm, rocdl`，已加一行
  `# UNSTABLE(gfx1250): ...` 注释（:36）。本文件只在这一处加注释，没有在每个调用点重复。

随死代码删除的 upstream op：`llvm.call_intrinsic("llvm.amdgcn.s.buffer.load.*")`（op0341/buffer_ops.py:476、
op0341/tensor_shim.py:168）、`llvm.ptrtoint` / `llvm.bitcast`（op0341/buffer_ops.py:130-131、tensor_shim.py:164-166）、
`llvm.IntToPtrOp` 和 from_addr 里的 `rocdl.MakeBufferRsrcOp`（op0341/buffer_ops.py:337, :348）、
`rocdl.RawPtrBufferLoadOp`（:495）、`llvm.PtrToIntOp`（op0341/tensor_shim.py:92）、`llvm.AtomicRMWOp`
（op0341/kernels_common.py:92）、`builtin.UnrealizedConversionCastOp` 和 `gpu.AsyncTokenType`（op0341/kernels_common.py:201-202）。

## 5. Unresolved

- `_to_raw` 的第三个分支（op_clean/tensor_shim.py:52）：能不能走到，取决于 managers 传进来的对象类型，
  而这是运行时的鸭子类型，静态看不出来。prod 形状的 trace 没有暴露问题（ISA 一致），但其它 config
  （has_sink、滑窗、fp16、qk_hdim 192/256、THD）没有覆盖到。
- `_unwrap_value`（op_clean/buffer_ops.py:90-111）：通过 `_value` / `.value` 链逐层解包，具体会收到哪些对象
  同样是运行时决定的。
- `_get_buffer_flags`（op_clean/buffer_ops.py:46-82）：`is_rdna_arch("gfx1250")` 在 0.3.4.1 返回 False
  （已在容器中实测），所以 gfx1250 上 V# 的 flags 走 CDNA 分支 `0x20070`。FlyDSL 自己的
  `make_buffer_tensor`（`expr/rocdl/universal.py:254-260`）用的是同一套判断，两边行为一致。
  但 gfx1250 的 V# 字段本身是否正确，我没有验证；它与 kernels_common 原来那条"gfx1250 被误判为 CDNA"
  的注释（op0341/kernels_common.py:121-126）属于同一个问题。
- `get_element_ptr` 里 `elem_type` 可以传 callable，也可以传 ir.Type（op_clean/buffer_ops.py:133-136），类型是动态的。

## 6. 已应用的清理（全部保持 codegen 不变，ISA 已验证）

| 文件 | 改动 | 依据 |
|---|---|---|
| tensor_shim.py | `exe._cf` 私有写 → 模块级 `_COMPILED` 字典 | PRIVATE-WRITE 修复，只涉及 host |
| tensor_shim.py | 删除死代码：`AITER_FLYDSL_*` 三个环境开关和 `_PRELOAD_COMPILE_LOCK`（op0341:23-41）、`ptr_rsrc`、`buf_load_scalar`（:44-61）、buffer-view 一族 `_BUF_COPY_ATOM`/`BUF_VIEW_MAX_ELEMS`/`buf_base_i64`/`ptr_buf_tensor`/`buf_copy_atom`/`buf_scalar_load`/`_FX_ELEM`/`_fx_elem`/`_fx_value_elem`/`_buf_copy_slice`/`buf_copy_load`/`buf_copy_store`（:64-240）、`wave_size_of`、`ptr_arg`（:243-263）、`_preload_compiled`（:287-309）、`get_dtype_str`、`TensorView`/`TensorBase`/`TorchTensor`/`GTensor`（:321-547） | 包内（加上 impl.py）零引用 |
| tensor_shim.py | `_to_raw` 保留，docstring 标注为 legacy，给出 `.ir_value()` 替代写法 | managers 仍在导入它 |
| kernels_common.py | `create_llvm_ptr`：`ptr._value if hasattr(...) else ptr` → `as_ir_value(...)` | 去掉私有读，改用稳定 API；ISA 一致 |
| kernels_common.py | 删除死代码：`ceildiv`、`format_kernel_name`、`kernel_signature`、`F32_*` 常量和 `ord_signed_f32`、`uint32_to_int32`、`_atomic_rmw_i32`/`atomic_{add,max}_i32`、`get_warp_size`、`default_f8_type`、`dtype_to_elem_type`、`stream_ptr_to_async_token`、`run_cached` 及其锁；导入从 7 行减到 2 行 | 零引用（fwd 里用的是 `fx.ceildiv`，不是这里的 `ceildiv`） |
| buffer_ops.py | 删除死代码：`buffer_load`（含 `is_scalar` 分支）、`_ptr8_to_v4i32`、`create_buffer_resource_from_addr`；同步更新 `__all__` | 唯一调用方是已删除的 tensor_shim 函数 |
| buffer_ops.py | `_mlir.extras.types.i8()` → `fx.Int8.ir_type`（`Int8` 的 `ir_type` 就是 `T.i8`，类型相同）；去掉 `_mlir.extras` 导入 | 稳定替代 |
| buffer_ops.py | 模块 docstring 里写着 "Only published flydsl APIs are used"，与代码不符，已更正；docstring 示例也不再引用已删除的 `buffer_load` | 过时注释 |

行数：tensor_shim 547 → 52，kernels_common 243 → 26，buffer_ops 572 → 398。
被删掉的 helper 在 `op0341/flydsl_fwd/` 和 aiter 原文件里都还在，以后要重新 vendor 时可以从那里取。

## 7. 提议但未应用（可能改变 ISA，需要上卡 A/B）

1. **O/LSE epilogue 改走稳定 buffer 路径**：`buffer_ops.create_buffer_resource` + `buffer_store`
   （fwd kernel op0341/fmha_fwd_prefill_a16w16_m32x8.py:1347, :1368, :1407, :1419, :1424, :1437；
   managers op0341/fmha_b16_buffer_managers.py:1460, :1544）换成 `fx.rocdl.make_buffer_tensor` +
   `BufferCopy*b` atom + `fx.copy`。现成的封装是 aiter 原版的 `ptr_buf_tensor` / `buf_copy_store`
   （op0341/tensor_shim.py:95-240）。这样会改变发射的 op（soffset 折叠、offset 计算、cache policy 的传法），
   属于"换 buffer 机制"，按规定只能提议。
2. `_unwrap_value` → `fx.as_ir_value`：两者对 Python int 的处理不同（as_ir_value 会物化成常量），
   只在 prod 形状上验证不够。
3. managers 里的 6 处 `_ir(...)`（`_to_raw`）→ `.ir_value()`：那个文件不在我的范围；`_to_raw` 能否删掉取决于它。
4. `get_element_ptr` 里的 `std_arith.ConstantOp/AddIOp` → fx 整数运算：只有当宽度能对上 fx 类型时才等价，
   而且 canonicalize 之前的 IR 形状会变。
5. **不建议**把 `_RAW_PTR_BUFFER_AUX_IS_ATTRIBUTE` 折叠成常量：那样会丢掉对 op032（0.3.2）这类旧版本的兼容。

## 8. 验证（compile-only，没有使用 GPU）

- 驱动脚本：`output/0925__flydsl/api-audit/work/compile_fwd_isa.py`。
  - 在容器 `fa-repro` 里运行，环境变量：`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250
    FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache_agent_helpers FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1
    HIP_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1`。
  - 输入全部在 CPU 上：torch CPU 张量，`fx.Stream(None)`。从头到尾没有调用 torch.cuda。
  - 直接调 `_ensure_bshd_kernel(mask_left=False, mask_right=True, return_lse=True, has_sink=False,
    gqa=4, qk_hdim=128, "bf16")`，再对 `_launch_fns[key]` 调 `flyc.compile(...)`。这样用的是 prod 那个
    launcher，compile_hints 相同（expert-scheduling-mode、waves_per_eu=2）。
  - ISA 取自 `FLYDSL_DUMP_IR` 输出的 `22_final_isa.s`。
- "清理后"的输入：op0341 的 5 个文件，只把本次改的 3 个换成 op_clean 版本（目录 `work/after_tree`）。
  这样可以排除并行 agent 对 fwd/managers 的改动干扰。

| 指标 | before (op0341) | after | 
|---|---|---|
| vgpr_count / next_free_vgpr | 445 / 445 | 445 / 445 |
| sgpr_count / next_free_sgpr | 100 / 98 | 100 / 98 |
| LDS（group_segment_fixed_size） | 327680 | 327680 |
| scratch（private_segment） / vgpr_spill / sgpr_spill | 0 / 0 / 0 | 0 / 0 / 0 |
| kernarg_segment_size | 344 | 344 |
| 指令数 / 直方图 | 4341 / — | 4341 / 无差异 |
| `22_final_isa.s` | — | **逐字节一致** |
| `21_llvm_ir.ll`、`00_origin.mlir` | — | **逐字节一致** |

- 确定性对照：op0341 编译两次（`dump_before` 与 `dump_before2`），ISA 逐字节一致。
- 附带检查：op_clean 整体（此刻包含其他 agent 对 fwd/managers 的改动）编译也通过，ISA 同样与 before 一致
  （`work/dump_opclean_combined`）。这只是当时的一个快照。
- 没有覆盖的 config：THD 入口、has_sink、滑窗、fp16、qk_hdim 192/256。本次删除的都是 trace 期间根本
  不会执行到的代码；只有 `create_llvm_ptr` 和 `Int8.ir_type` 两处改动会在所有 config 上被执行到，
  而这两处的等价性不依赖 config。

## Verification（对抗性复核，2026-09-25）

结论：**未能推翻等价性，没有回退任何 hunk**。因为没有回退，也就没有重新编译；我只是复核了已有的 dump。没有使用 GPU。

逐条核对（对照未改动的 `fwd341/op0341/flydsl_fwd/`）：
- `buffer_ops.get_element_ptr`：`T.i8()` 改为 `fx.Int8.ir_type`。在 0.3.4.1 中，`Int8` 定义为 `ir_type=T.i8`（`flydsl/expr/numeric.py:818`），`T` 就是 `flydsl._mlir.extras.types`，`NumericMeta.ir_type` 返回 `cls._ir_type()`，所以两者得到同一个 signless i8。这个改动是 live 的：managers 里所有 `get_element_ptr` 调用都只传了 `static_byte_offset`，走的都是默认 elem_type。GEP 的其余部分、`buffer_store`、`create_buffer_resource` 和 `BufferResourceDescriptor` 与 op0341 逐字相同（已 diff）。
- `kernels_common.create_llvm_ptr`：原写法 `ptr._value if hasattr(ptr,"_value") else ptr` 改为 `as_ir_value(...)`。`fx.to_llvm_ptr` 返回 `fly.to_llvm_ptr(...)` 的结果。如果结果是 `ir.Value`，`as_ir_value` 原样返回，与原写法一致；对 OpView 等其它对象也都原样返回。
- `tensor_shim._run_compiled`：缓存从 `exe._cf` 改为模块级 `_COMPILED[exe]`。`JitFunction`（`jit_function.py:1146`）没有定义 `__eq__`/`__hash__`/`__slots__`/`__setattr__`，所以按对象身份哈希。`exe` 来自 `_launch_fns[key]`，而 `_ensure_*_kernel` 在 key 已存在时直接 return，不会重建，所以同一个 key 始终对应同一个对象。失败路径的 Context 清理没有改动。唯一的语义差别：dict 强引用 exe，但 `_launch_fns` 本来就永久持有它，所以没有新增泄漏。impl.py 按唯一模块名加载每个 arm，每个 arm 各有一份 `_COMPILED`，与原来的逐对象属性等价。
- 删除的代码：在 op_clean 全树以及 op0341 的 impl/_env/__init__ 里 grep 了 `buffer_load`、`_from_addr`、`GTensor`、`buf_scalar_load`、`run_cached`、`get_warp_size`、`_cf`、`_compiled_cache`、`AITER_FLYDSL*`、`getattr(buffer_ops|tensor_shim`。唯一命中是 fwd 文件 :237 一处注释里提到 `buffer_load`，是文字描述，不是调用。被删的模块级代码只有 env 读取和 lock 创建，没有影响其它模块的副作用。`_to_raw` 函数体未变，只改了 docstring。
- ISA 证据复核：`work/after_tree` 中的 3 个文件与当前 op_clean 逐字节相同（cmp），另外 3 个文件与 op0341 逐字节相同。`dump_before` 与 `dump_after` 的 `22_final_isa.s`、`21_llvm_ir.ll`、`00_origin.mlir` 用 cmp 比较结果一致；其中 `00_origin.mlir` 含 14787 个 `loc(`，所以连 location 都一致。

发现的文字问题（均不影响 codegen，也都不是本次 cleanup 引入的）：
1. `buffer_ops._get_buffer_flags` 的 docstring 从上游继承了算错的十六进制值。`(7<<12)|(4<<15)` 等于 **0x27000**，不是 0x20070；RDNA 的值是 **0x21027000**，不是 0x21020070。代码本身是对的。上文"剩余不稳定项"里写的 "`0x20070`" 应读作 **0x27000**。这个 docstring 没有改，因为这些行与 op0341 相同，不在本次 cleanup 的 hunk 里。
2. `PROVENANCE.md` 没有同步本次裁剪：它仍写着 `kernels_common.py` "unmodified, 0 changed lines"，并列出了 tensor_shim 对 `get_warp_size` 的 import 改写，而这个 import 已被删除。新 docstring 里的 "see ../PROVENANCE.md" 因此指向了过时的描述。建议后续补一段 trim 记录。
3. `tensor_shim` 的新 docstring 说 aiter 原版的 buffer-view helper 基于 `fx.rocdl.make_buffer_tensor`，与 op0341 `tensor_shim.py:143` 相符，属实。

覆盖范围的限制没有变：只有 prod bshd config 的 ISA 证据。上面三处 live 改动与 config 无关，所以 THD、sink 和其它 hdim 不需要单独的 A/B。
