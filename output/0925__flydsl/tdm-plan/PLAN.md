# TDM 集成计划 — `k_dkdv`，2026-09-25

**本轮零 GPU、零编译。下列每一个行号与引文都是我今天在文件上核对过的；凡是我没有亲自验证的，写在第 6 节。**

---

## 1. TARGET：**neither（两个 build 都不是）**

四份 mapping 里三份（`prefetch` → 4-wave、`descriptor` → champion、`safety` → 4-wave）都被 2/2 驳回；只有 `target`（判 neither）以 1/2 存活，且驳它的那位评审明确写了「目标选择（"neither"）我不反对……descriptor 的每一个字段值也都算对了」——他驳的是证伪件的字段写法，不是判决。我独立复核了承重的四条，全部成立：

**（a）三个站点全部 dual-use，TDM 在本 op 删不掉任何一条 `buffer_load`。**

冠军 `.../op/current/kernels.py:384-391`、4-wave `/home/lihuzhan/g2work/kernels.py:398-405` 逐字同构：

```python
                    llvm_dialect.store(fx.as_ir_value(dp[dt][u]),
                                       create_llvm_ptr(lds_do + o, address_space=3))
                    llvm_dialect.store(fx.as_ir_value(qp[dt][u]),
                                       create_llvm_ptr(lds_q + o, address_space=3))
            qfr = [qp[dt][0].shuffle(qp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT)]
            dfr = [dp[dt][0].shuffle(dp[dt][1], list(range(16)))
                   for dt in range_constexpr(NDT)]
```

同一批 `qp`/`dp` 寄存器既是 LDS store 的源，又 shuffle 成 `qfr`/`dfr` 喂 S/P WMMA。我另外查了 `k_dq`（冠军 `:820-829`）：`kp = (pre[pi], pre[pi + 1])` 既 `llvm_dialect.store(... lds_k ...)` 又 `kfr = kp[0].shuffle(kp[1], list(range(16)))` 喂 `:828` 的 WMMA；V 根本不进 LDS。**三个站点，一个形状。TDM 没有寄存器目的地，所以它在本 op 的任何位置都只能删 `ds_store_b128`。**

**（b）冠军没有围栏可省。** `staging-split-verdict/VERDICT.md` 的逐条普查：冠军 `.LBB0_8` 779 条指令，`s_barrier_signal` **0**，`s_wait_loadcnt_dscnt 0x0` **0**。TDM 的实测动机在冠军处**字面不存在**；它能做的只有把 40 条 `ds_store_b128` 删到 8 条，同时每 body 多读一遍 16 KB global。这是 P1（删整个 Q/dO LDS 往返，**−6.98%**）的真子集 + P2（把 LDS 往返换成 in-body global 读，**−18.37%**）的附加流量，落在 `dead_ends.md:330` 那句「第六次连续删除失败……**Do not remove anything**」的正中间。

**（c）4-wave 的围栏两半都活得过 TDM——我用 ISA 位置独立确认，不靠论证。**
`output/0925__flydsl/g1b-4wave/isa_4wave_k_dkdv.s`，`.LBB0_8` = 行 1508-2314，**806 条指令**，全文件唯一一条 `s_wait_loadcnt_dscnt 0x0` 在行 2091 = rel-**583**，紧接 barrier-1（2092-2093）。我数出的 body 普查：`ds_store_b128` 40、`ds_load_tr16_b128` 40、`buffer_load_b128` 32、`buffer_load_b32` 4、`v_wmma` 64、`s_barrier_signal` 2。

- **loadcnt 半边**：36 条 carried prefetch 在 rel-**59..63**（4 条 b32）与 rel-**160..194**（32 条 b128）。它们由 `kernels.py:372-373` 的 `nxt = _ldqd(qt_n, gh_n)` 发射、`:513-514` 原样返回，本 body 从不消费，必然横跨 rel-583。TDM 一条都碰不到：32 条是 GEMM 操作数（dual-use），4 条是 LSE/delta 标量、不可 tile，而把它们改回 in-place 就是 `r10.i1.g27`（−14.6%）的坟墓。
- **dscnt 半边**：`:460-466` 计算出来的 8 条 P/dS `ds_store_b128` 仍然跨 barrier-1，`:495-496` 在 barrier 之后读它们，RAW 是活的。

**两半都活。** 对照组同文件：masked body `.LBB0_4`（441-1134）是 `carry=False`，两个 barrier（1026、1107）前都是纯 `s_wait_dscnt 0x0`——我逐行看过。

**（d）4-wave 的天花板就是冠军。** 同一份普查：两 build 每 body 逐条等量，每 CU LDS 写字节同为 65536 B，每 CU global b128 同为 128，waves/CU 同为 4。337.02 / 507.34 需要 +50.5% 才追平，而追平就是上限。

**结论：TDM 作为性能候选，在冠军和 4-wave 上都不开。** 但 TDM 这条线不应该以「论证关闭」收尾——它应该以 **ISA 关闭**。下面是零卡时的关闭路径，以及在它之前应当先做的那件更便宜、更可能赢的事。

---

## 2. 编辑清单（有序）

### S0 — 先修筛子。不修，下面每一道门都是瞎的

**E0.** `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0921__flydsl/bin/compile_only_driver.py:266-269`

当前文本：
```python
OPS = ("v_wmma_f32_16x16x32_bf16", "ds_load_tr16_b128", "ds_read_b128", "ds_write_b128",
       "s_barrier", "buffer_load_dwordx4", "buffer_store_dwordx4", "v_exp_f32",
       "s_set_vgpr_msb", "scratch_", "global_load_lds", "s_wait_dscnt",
       "s_wait_asynccnt", "sched_barrier")
```
改为在元组中追加：`"buffer_load_b128", "buffer_load_b32", "ds_store_b128", "tensor_load_to_lds", "s_wait_tensorcnt", "s_wait_loadcnt_dscnt", "s_wait_loadcnt"`。

**今天在 `isa_4wave_k_dkdv.s` 上实测**：`buffer_load_dwordx4 = 0`、`ds_write_b128 = 0`、`ds_read_b128 = 0`、`global_load_lds = 0`、`s_wait_asynccnt = 0`，而 `buffer_load_b128 = 160`、`ds_store_b128 = 80`。十四项里五项是本 ISA 不存在的助记符，其中**两项正是本计划的填充门禁**。一个 TDM build 会报 `buffer_load_dwordx4 = 0, ds_write_b128 = 0`——与今天的 build 不可区分，也与成功不可区分。

**E0b.** 同文件 `:287`

当前文本：
```python
        rec["total_instr"] = len(re.findall(r"^\s+(?:s_|v_|ds_|buffer_|global_|flat_|scratch_)",
```
在字符类中加入 `tensor_`，否则 `tensor_load_to_lds` 连指令都不算。

---

### S1 — **在写任何 TDM 代码之前先做这个：免费的掩护恢复，零 TDM、零描述符、零挂卡面**

这是我今天读 ISA 时掉出来的，四份 mapping 与八位评审都没有提。

围栏的代价，按战役自己的 round-22 模型（`0925__flydsl/facts.md:22-28`），是 **cover distance**：冠军 ~604，4-wave **389 = rel-583 − rel-194**。但 rel-583 是 barrier-**1**；barrier-2 在 rel-**639**，它前面（行 2146）已经是**纯 `s_wait_dscnt 0x0`**——我核过。

也就是说：36 条 prefetch 今天在 rel-59..194 发射，**在同一迭代内 rel-583 就被排空**。如果把发射点挪到 barrier-2 之后（rel > 639），排空点就落到**下一迭代**的 rel-583，cover 变成 `(806 − 发射点) + 583 ≈ 600`——回到冠军水位，而**一条指令都没删、没加**。

**E1a.** `/home/lihuzhan/g2work/kernels.py:372-376`

当前文本：
```python
        if const_expr(carry):
            nxt = _ldqd(qt_n, gh_n)
        else:
            nxt = None
            pre = _ldqd(qt, gh)
```
改为：
```python
        if const_expr(carry):
            nxt = None
        else:
            nxt = None
            pre = _ldqd(qt, gh)
```

**E1b.** 同文件 `:512-515`

当前文本：
```python
        fx.barrier()
        if const_expr(carry):
            return new + [_ir(v) for v in nxt]
        return new
```
改为：
```python
        fx.barrier()
        if const_expr(carry):
            nxt = _ldqd(qt_n, gh_n)
            return new + [_ir(v) for v in nxt]
        return new
```

`_ldqd` 只依赖 `(qt_n, gh_n)` 与循环不变量，body 内无任何依赖，位置自由。`carry=False` 路径一字不动。语义逐位不变（`r20.i2.g62` 的 depth-2 结构、`r16.i1.g48`/`g51` 的 head-of-FIFO 顺序、`:311` 的 `rocdl.sched_barrier(0)` 全部原样保留）。

**零卡时中止门（读 ISA，不发射）：**
1. `.LBB0_8` 内最后一条 `buffer_load_b128` 的 rel 索引 **> 640**，且 cover `= (N − idx_last) + idx_fence` **≥ 550**。若 < 550，调度器把 clump 又提上去了，**中止，不测卡**。
2. `buffer_load_b128` 仍为 **32**、`buffer_load_b32` 仍为 **4**、`v_wmma` 仍为 **64**、`ds_store_b128` 仍为 **40**、spill 0。任何一项变化说明改的不是发射位置。

**必须说明的反向证据**：`facts.md:29-30` 写着「*any edit that moves the last prefetch load later in the body loses* — five for five」。那五次（`g28/g63/g66/g68/P70`）的全排空点都在**下一迭代的顶部**（idx 13-25），所以发射点后移 = cover 变短；本 build 的全排空点在**本迭代中段**（rel-583），所以发射点越过 rel-639 = cover **变长**。这是那条启发式过滤器与它所代理的变量（cover）**唯一会分歧**的情形，而分歧可以在 ISA 上零成本判定。这一点我没有编译验证，它是本计划最大的单一赌注，中止门就是为它设的。

---

### S2 — TDM 证伪件。**首行必须标死：measurement instrument, COMPILE_ONLY, NEVER LAUNCH**

只有在需要把「TDM 在 4-wave 上关闭」这件事落到 ISA 而不是论证上时才建。预测：**围栏两半都还在，`buffer_load_b128` 仍为 32**。

**E2.** `:43` 之后新增一行

当前文本：
```python
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _ir
```
其后插入：
```python
from aiter.ops.flydsl.kernels import tdm_ops_gfx1250 as tdm_ops
```
**绝不可** `from flydsl.expr.rocdl import tdm_ops`：上游 `tdm_ops.py:359` 是
```python
    lds_base_idx = _ArithValue(memref_dialect.extract_aligned_pointer_as_index(lds_memref))
```
对 Fly shared view 失败（h47 四次失败）。shim 的 `_FlyAwareMemrefDialect.extract_aligned_pointer_as_index`（`tdm_ops_gfx1250.py:52-59`）是唯一让这次调用合法的东西。注意 `global_ptr` 一侧上游本来就走 fly dialect（`tdm_ops.py:349` `glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)`），所以 `Q`/`DO` 这两个 `fx.Tensor` 内核参数可以直接传。

**E3.** `:258` — **不动**。评审提出的 `fx.struct` 重构是多余且更危险的。

当前文本保持：
```python
    smem = fx.SharedAllocator().allocate(LDS_SEG + 2 * 32 * S_ROW_B)
```
理由：描述符只消费 `lds_memref` 的**基址**（`tdm_ops.py:359` 是全函数唯一一次引用），而我们随后用 shim 的 `update_tensor_descriptor_2d_lds_addr` 把整个 LDS 地址字段替换成本 kernel 已有的 i32 地址（`:259-261` 的 `_lds0`/`lds_do`/`lds_q`）。动 `LDS_SEG = 65536` 就要重证 `:249-256` 那整段 bit-16 段分离论证（`r12.i1.g39`）。

**E4.** `:284` 起，`_ldqd` 头部——加绝对下标钳位 + 描述符构造

当前文本：
```python
    def _ldqd(qt, gh):
        qh = hkv * G + gh
        base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
        q0 = qt * fx.Int32(32)
```
改为（`qh`/`base_q` 保持，`q0` 改用钳位后的 `qt_s`）：
```python
    def _ldqd(qt, gh):
        # TDM has NO buffer num_records. r1.i7.g07's clamp (`_bv`, :220) does not
        # cover this path. Clamp the ABSOLUTE tile index once, here: it covers the
        # depth-2 overrun (:542), both non-PARTIAL prologue calls (:596-597) and
        # both PARTIAL prologue calls (:586-587) in one place.
        qt_s = (qt < nqt2).select(qt, nqt2 - fx.Int32(1))
        qh = hkv * G + gh
        base_q = bat * Sq * rs_q + qh * fx.Int32(DV8)
        q0 = qt_s * fx.Int32(32)
```
`nqt2` 定义在 `:333`，晚于 `def` 但早于**全部六个调用点**（`:373`、`:376`、`:586`、`:587`、`:596`、`:597`），闭包成立。**applier 必须验证没有第七个调用点位于 `:333` 之前。**

然后在 `q0` 之后插入描述符构造与发射（字段见第 3 节）：
```python
        def _tdm(T_, lds_base):
            d = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=T_,
                lds_memref=fx.Tensor(fx.make_view(
                    smem.peek().ptr, fx.make_layout((32, D), (D + 8, 1)))),
                global_offset=(fx.Index(bat * Sq + q0), fx.Index(qh * fx.Int32(D))),
                tensor_shape=(32, D),
                strides=(rs_q * fx.Int32(8), 1),
                tile_shape=(32, D),
                elem_bytes=2,
                pad_interval=D,     # 128 ELEMENTS
                pad_amount=8,       # 8 ELEMENTS = 16 B -> 272 B == X_ROW_B
                num_warps=1,
                workgroup_mask=0,
                cache_policy=0,
                pred=1,
                oob_outer_bound=B_ * Sq,
            )
            return tdm_ops.update_tensor_descriptor_2d_lds_addr(d, lds_base)
        rocdl.sched_barrier(0)
        tdm_ops.tensor_load_2d(_tdm(DO, lds_do))
        tdm_ops.tensor_load_2d(_tdm(Q,  lds_q))
        rocdl.sched_barrier(0)
```

**E5.** `:396-401` — 删掉 32 条 Q/dO staging store（TDM 接管这 32 条；`:460-466` 的 8 条 P/dS 保留）

当前文本：
```python
            for dt in range_constexpr(NDT):
                for u in range_constexpr(2):
                    o = xo + fx.Int32(dt * 64 + u * 32)
                    llvm_dialect.store(fx.as_ir_value(dp[dt][u]),
                                       create_llvm_ptr(lds_do + o, address_space=3))
                    llvm_dialect.store(fx.as_ir_value(qp[dt][u]),
                                       create_llvm_ptr(lds_q + o, address_space=3))
```
删除这 7 行。`xo`（`:393-394`）若不再被引用一并删除。

**`:402-405` 的 `qfr`/`dfr` 一个字都不能动**——它们消费同一批 `qp`/`dp`，喂 `:435`/`:438` 的 S/P WMMA。这正是 32 条 `buffer_load_b128` 和 g21/g62 的 tuple 在 TDM 下必须原样存活的原因，也是本证伪件的全部要点。

**E6.** `:473` 之前插入 wait

当前文本：
```python
        fx.barrier()
```
（`r23 -- BARRIER RESTORED` 那一个，barrier-1）之前插入：
```python
        rocdl.sched_barrier(0)
        tdm_ops.tensor_wait(0)
        rocdl.sched_barrier(0)
```
`tensor_wait(0)` 必须在 barrier **之前**。`sched_barrier(0)` 括号是 aiter 的既有惯用法（`gemm_a8w8_256x256_gfx1250.py:491-492` `rocdl.sched_barrier(0)` / `tdm_ops.tensor_wait(0)`），用来阻止 LLVM 把 `ds_load_tr16_b128` 提到 wait 之上——后端**不把 `tensor_load_to_lds` 建模为 LDS 写**，`s_barrier` 不排序它，`s_wait_dscnt` 不计它。

**E7.** `:512` 的 barrier-2 —— **不加安慰剂 wait**（这是评审驳倒上一版的一条，成立）。
此处在飞的 TDM 只有 barrier-1 之前发出、且已在 barrier-1 之前 `tensor_wait(0)` 过的那一次，再加一个 wait 恒为 no-op。真正的危险是下一轮的 `tensor_load_to_lds` 能否被提到 `s_barrier` 之上——这**必须作为 ISA 断言检查**（见第 5 节 A3），而不是靠空 wait。读侧安全，因为 `ds_load_tr16_b128` 的结果直接喂 `:498-505` 的 WMMA，后端必然在 WMMA 前插 dscnt wait。

**E8.** `:545` `_NP = 4 + 32` —— **不动**。32 条 b128 在 TDM 下依然存在、依然被 carry。任何把它改成 `_NP = 4` 的编辑都会让每个 wave 做 K=32 收缩：**能编译、能发射、返回数字的静默错答案**。

---

## 3. 描述符全字段（挂卡面，逐字段附来源）

Q 环；dO 环除 `global_ptr=DO`、`lds_base=lds_do` 外逐字相同。源文件 `/home/lihuzhan/g2work/kernels.py`，API `/home/lihuzhan/.local/flydsl032/flydsl/expr/rocdl/tdm_ops.py`，shim `/home/lihuzhan/code/aiter-src/aiter/ops/flydsl/kernels/tdm_ops_gfx1250.py`。

| 字段 | 表达式 | 来源与理由 |
|---|---|---|
| `global_ptr` | `Q`（原始内核参数，`:616`） | **不是 `g_q`**（`:220` `g_q = _bv(Q, nq_b, fx.BFloat16, 8)`）。`tdm_ops.py:349` 走 `_fly_d.extract_aligned_pointer_as_index` 取裸指针，`_bv`（`:76-79`，`make_buffer_tensor(num_records_bytes=...)`）的边界被整个丢弃。**这是卡安全的转轴。** aiter 先例：`gemm_a8w8_256x256_gfx1250.py:348` `global_ptr=tensor`。 |
| `lds_memref` | `fx.Tensor(fx.make_view(smem.peek().ptr, fx.make_layout((32, D), (D + 8, 1))))`，随后整体被 `update_tensor_descriptor_2d_lds_addr(d, lds_do / lds_q)` 覆盖 | 描述符**只读基址**（`tdm_ops.py:359`，全函数唯一引用），layout 是装饰性的。覆盖写法见 shim `:107-119`：把 dgroup0 lane 1 整个换成 `fx.Int32(new_lds_addr)`。这是把 TDM 接进 `:259-261` 那套 i32 LDS 地址算术（`_lds0 = fx.Int32(fx.ptrtoint(smem.peek().ptr))`）的唯一正确方式。**`lds_byte_offset` 必须为 `None`**：`tdm_ops.py:367-369` 把它**加**到抽出的基址上，与覆盖写法二选一，同时写会双算——aiter 的做法是把偏移烘进指针且**根本不传** `lds_byte_offset`（`gemm:348-352`）。**`num_warps > 1` 时此覆盖会摧毁 per-wave LDS 偏移**（`tdm_ops.py:365-367`），这是本证伪件取 `num_warps=1` 的理由之一。 |
| `global_offset` | `(fx.Index(bat * Sq + q0), fx.Index(qh * fx.Int32(D)))` | `bat` `:194`；`q0` = 钳位后 `qt_s * fx.Int32(32)`（原 `:287`）；`qh = hkv * G + gh`（`:285`）。**必须是 index 型**：`tdm_ops.py:352-354` 做 `(outer_off + warp_off_outer) * outer_stride_idx`，`warp_off_outer` 是 index 型 `ArithValue`（`:342-344`）；`:481` 又做 `arith.index_cast(T.i32, outer_off + warp_off_outer)`。传 `fx.Int32` 是 trace 期类型风险（**响亮失败，不是静默**）。aiter 全部先例都传 `(0, 0)` + `addr64` 推进，运行时 `global_offset` 在本工具链**无先例**（见第 6 节）。 |
| `tensor_shape` | `(32, D)` | **惰性参数，文档用途。** `tdm_ops.py:300` `outer_size, inner_size = tensor_shape` 之后**全函数再无引用**——我做了穷举 grep，其余命中只在 `:236`/`:251` docstring 和 `:1220/:1236/:1249-1252` 的 `l2_prefetch_tile`。真正写进描述符的是 `:397-398` `tdim0 = bpw_inner` / `tdim1 = bpw_outer`，来自 `tile_shape ÷ num_warps`。**传真实张量形状买不到任何边界。** |
| `strides` | `(rs_q * fx.Int32(8), 1)` = `(Hq*D, 1)`，**元素**单位 | `:234` `rs_q = Hq * fx.Int32(DV8)`，`DV8 = D // 8 = 16`（`:45`）是 **vec8 瓦片**单位，×8 才是元素。运行时 i32 受支持（`tdm_ops.py:308-320` → `g1_s5` 在 `:497-500`）。**偏小方向是静默错值；偏大方向是挂卡**（行地址被 `tensor_dim1` 截行数，不截字节）。 |
| `tile_shape` | `(32, D)` = `(32, 128)` | 32 个 query 行（`g07`/`g09` 的 staging 深度），`D = 128`（`:44`）。这才是硬件真正走的 extent。 |
| `elem_bytes` | `2` | bf16。`data_size_code = log2(2) = 1`。 |
| `pad_interval` | `D` = **128 元素** | **必传。** |
| `pad_amount` | `8` = **8 元素**（= 16 B） | **必传。** `compute_padding_encoding(128, 8, 16)`（`tdm_ops.py:85-115`）：`interval_dw = 128*16//32 = 64`（2 的幂，`:110` 断言通过）→ `enc_interval = 5`；`amount_dw = 4` → `enc_amount = 3`；`pad_enable = 1`。`tdm_ops.py:361-362` `lds_inner_stride = inner_tile + pad_amount` = 136 元素 = **272 B = `X_ROW_B`**（`:62`），逐字节复现 `r4.i1.g16` 的 bank 调优。默认值是 `pad_interval=0, pad_amount=0`（`:218-219`），返回 `(0,0)`，产生**未 padding 的 256 B = 64 dwords = 精确的 64 路全冲突步距**——正确但慢，**没有任何正确性门禁抓得到，而锅会记在 TDM 头上**。`pad_amount` 单位是元素不是字节：传 16 会得到 320 B 步距，`compute_padding_encoding(128,16,16)` 返回 `(5,7)`，**全部断言通过**，然后读侧仍按 272 索引 → 静默错答案。 |
| `num_warps` | `1` | `warp_off_outer = warp_off_inner = arith.index(0)`（`tdm_ops.py:342-344`），不读 `rocdl.wave_id()`、不读 TTMP8、无 per-wave 偏移算术。与今天四波复制写 staging 的语义完全一致。`num_warps=4` 见第 6 节，**不在本计划内**。 |
| `oob_outer_bound` | `B_ * Sq`（运行时 i32） | **强制项，不是护栏。** `tdm_ops.py:449` 起：为 `None` 时 `tensor_dim1 == tile_dim1`，**OOB 检查关闭**。它是唯一真界——`tensor_shape` 惰性，`_bv` 的 num_records 在此路径不存在。`:481` `start_i32 = arith.index_cast(T.i32, outer_off + warp_off_outer)` 是**构造时**的偏移，所以描述符必须**逐次重建**（本计划就是），**绝不可** hoist + `update_tensor_descriptor_2d_addr64` 推进：`addr64`（`:1060-1090`）只重写 dgroup0，而 `tensor_dim1` 住在 dgroup1 的 `g1_s2`/`g1_s3`（`:485-493`），钳位会停在第 0 次迭代的值上，越走越松。 |
| `oob_inner_bound` | **不传** | shim `:98-104` 断言 `num_warps == 1` **且** `global_offset[1] == 0`；我们的 inner_off 是 `qh*D`。内维由构造在界内：inner tile = 128 = D，行宽 `Hq*D`，`qh ≤ Hq-1`。但这是**无保护**的，必须写在源码注释里。 |
| `cache_policy` / `pred` / `workgroup_mask` | `0` / `1` / `0` | 无 cluster，无 multicast；字面 Python `int 0` 会折成编译期常量（`tdm_ops.py:438-444`）。 |
| `for_store` / `atomic_barrier_enable` / `early_timeout` | `False` / `False` / `False` | `atomic_barrier_enable` 必须保持 False：helper 把编码的 atomic-barrier 地址保持为零，所有参与波必须同意一个我们没有实现的协议（`tdm_ops.py:272-276`）。 |

**`B_ * Sq` 是展平行轴的界，不是 per-batch 的界。** 一个在 batch `bat` 内越过 `Sq` 但仍小于 `B_*Sq` 的 tile 地址合法、TENSORcnt 正常退休、不 zero-fill，会把 **batch bat+1 的 query 行**读进 batch bat 的 dK/dV——**静默错值**。E4 的 `qt_s` 钳位不是可选项，它和 `oob_outer_bound` 各防一类。

---

## 4. 首次发射协议

**本计划的 S0/S1/S2 全部不发射。** 下面这套协议是 S3 的先决条件；S3 只在 S2 的围栏**消失**（即我的预测被证伪）时才存在。遵循 `gfx1250-card-safety`。

**A 段 — 全部 CPU，零卡时，每次重建都要重跑（含"小改"）：**

- **A1** COMPILE_ONLY（无 `/dev/kfd`、无 `/dev/dri` 的容器）。`private_segment_fixed_size > 0` 或任何 `scratch_` 命中是 **KILL，不是成本**——gfx1250 上一个 spill build 首次发射后挂死，代价是人工断电。
- **A2** ISA 断言（用 E0 修过的筛子），四条同时成立：`tensor_load_to_lds` 每 body **2** 条；`s_wait_tensorcnt` 出现且**支配**每一条 `ds_load_tr16_b128`；`ds_store_b128` **40 → 8**；`buffer_load_b128` **仍为 32**。最后一条是防 K=32 静默错答案的**唯一闸门**。
- **A3** ISA 相对位置断言：`tensor_load_to_lds` 必须出现在本 body 第一个 `s_barrier_signal` **之前**、且 `s_wait_tensorcnt` 在该 barrier 之前。后端不把 TDM 写建模为 LDS 写，`s_barrier` 不排序它；这个位置只能靠 grep 保证。
- **A4** padding 落地断言：GROUP1 sgpr0 的常量应为 `0x07510000`（`data_size 1<<16 | pad_enable 1<<20 | 5<<22 | 3<<25`）。grep 到 `0x10000` 就是那个静默 64 路 bank 冲突。
- **A5** Python 导入期断言：`compute_padding_encoding(D, 8, 16) == (5, 3)`，`X_ROW_B == (D + 8) * 2`。
- **A6** 纸面 extent 穷举，**不推理循环上界，枚举它**：对 `op.shape` 的每个形状、每个可达 `(bat, hkv, gh, qt)`，断言
  `((bat*Sq + qt_s*32 + 31) * Hq + (Hq-1)) * D + 127 < B_*Sq*Hq*D`，其中 `qt_s` 取**钳位后**的值。同时独立断言 `qt_s*32 + 32 <= Sq`（per-batch，防串批静默错值）。dO 同样做一遍。
- **A7** grep diff：任何 `update_tensor_descriptor_2d_addr_lo` 命中是发射阻断项——其 docstring（`tdm_ops.py:874-879`）写明 32 位回绕不传播进 addr_hi，描述符静默别名进错误的 4 GiB 页，GPU 在 `amdgpu_mes_reg_write_reg_wait` 死锁。
- **A8** 装好 wave32 显式覆盖（card-safety row 10：`flydsl/runtime/device.py:76` 把 gfx1250 误判为 CDNA）。`isa_4wave_k_dkdv.s` 今天带 `.amdhsa_wavefront_size32 1`，但这是断言项不是假设项。
- **A9** `export AMD_SERIALIZE_KERNEL=3`。

**B 段 — 首次发射，人在机器旁，绝不并入 sweep，绝不无人值守：**

- **B0** 先跑**独立 smoke kernel**（一个 workgroup，微秒级），A 段覆盖不到它要证的三件事：(i) TDM 写出的 LDS 影像是否与 `tr()` 期望的字节布局一致（A4 只看位，不看字节）；(ii) 一个 under-sized `tensor_dim1` 的描述符是否**真的退休 TENSORcnt 并 zero-fill**（`tdm_ops.py:283-289` 只承诺"on the validated eng-sample"，那是别人的硅）；(iii) `tensor_wait(n>0)` 的语义（aiter 全部调用点都是 `tensor_wait(0)`，`n>0` 零先例）。
  - **SMOKE A**：`R=32 C=128` bf16，`SRC` 填 bf16 精确非重复整数模式，`DST` 主机侧**预填 NaN**（区分"从未写"与"写了零"）。判定是 `DST` 与 `SRC` **逐位相等** + `isfinite` 覆盖 100%。**不是 SQNR。**
  - **SMOKE B**：同上但 `oob_outer_bound = R - 8`、tile outer 仍为 R。判定：kernel **返回**（TENSORcnt 退休），行 `[0, R-8)` 逐位相等，行 `[R-8, R)` **恰好为零**。NaN = 边界什么也没做；挂 = 边界在本硅上不是边界，**整个安全设计作废，TDM 放弃而不是绕过**。
  - **SMOKE B2**：`oob_outer_bound` 使 `tensor_dim1 == 0`（全越界）。这是钳位写错时真 kernel 会撞到的形态，而 `tdm_ops.py` 的承诺只覆盖 partial overhang。**欠尺寸方向不取字节，不会 fault**，是唯一可安全探测的方向。
- **B1** 每次发射前后：`ls /sys/class/kfd/kfd/proc/` 必须为空并点名每一个持有者；`timeout 10 dmesg | tail -60 | grep -iE "MES|page fault|GPU reset|REMOVE_QUEUE|copy_context_work_handler|Mailbox work was idle"`；VRAM 与基线比对。
- **B2** toy 形状 `b=1 s=256 hq=2 causal=False`（走 `qloop_full`、`nmaskp=0`），独立进程，硬墙钟超时 90 s，无 in-process autotune。
- **B3** toy `causal=True`（走 `qloop_mask → qloop_full` 的 prologue 交接）。
- **B4** `op/validation.py`，**不是自写检查**，在任何 benchmark 之前。`g56` 的教训：四个离线筛子全部 `verdict: pass, spill: 0`，而 dk/dv 在 −56/−73 dB。SQNR 靶：dq 52.56 / dk 52.60 / dv 52.71。dv 达标而 dk 不达标 = 两个描述符里恰好一个错。
- **B5** ut 15/15，然后 200 次逐位确定性门，然后**换一个冷进程**跑验收。bring-up 与 acceptance 不同场。
- **挂卡时**：不要 SIGKILL（每一次 kill 多留一个 D-state 持有者）。只读一次有界 dmesg 并分类：`wait for reset ack` / `GPU reset begin` / `ring gfx timeout` = 不可恢复，报告并停；`Queues reset on process` + protection fault 且无上述三者 = 进程死了卡没死，用一次 4096³ bf16 matmul 在硬超时下确认。**绝不 `modprobe -r amdgpu`。**
- **预算**：一次挂卡 = 一个 AC cycle + 一个人。不要和任何其它需要卡的工作排在同一场。

---

## 5. 分阶段序列与中止信号

| 阶段 | 内容 | 卡时 | 第一个数字？ | 中止信号 |
|---|---|---|---|---|
| **S0** | E0 / E0b 修筛子 | 0 | 否 | 修完的筛子无法复现 4-wave 已知值（VGPR 882、LDS 82944、`buffer_load_b128` 160、`ds_store_b128` 80、`s_barrier_signal` 4）→ 仪器本身错，先修它 |
| **S1** | E1a/E1b 发射点下沉，COMPILE_ONLY 读 cover | 0 | **否**（先读 ISA） | cover 未升至 ≥ 550，或 `buffer_load_b128 ≠ 32` / `v_wmma ≠ 64` / spill > 0 → **中止，不上卡**。调度器把 clump 提回去了，本方向零成本关闭 |
| **S1m** | S1 通过 ISA 门后**才**测量 | 小（单次，有人值守） | ✅ **第一个数字在这里** | 对 337.02 无改善 → 围栏的代价不是 cover，round-22 模型在本 build 上失效，**这是比任何 TDM 结果都重要的发现**，记录并关闭 4-wave 方向 |
| **S2** | E2–E8 TDM 证伪件，COMPILE_ONLY，**永不发射** | 0 | 否 | `.LBB0_8` 内 `s_wait_loadcnt_dscnt 0x0` **仍在** → 预测成立，**TDM 在 4-wave 上以 ISA 关闭，最终且不再复探**；配合第 1 节的 (a)(b)，TDM 在本 op 三个站点全部关闭。`buffer_load_b128` 掉到 32 以下 → 编辑错了（K=32 静默错答案），修了再看 |
| **S3** | 仅当 S2 的围栏**消失**才存在：按第 4 节 A 段 + B 段带卡验证 | 高风险 | 是 | 任何 A 段门失败 → 不发射。任何 SQNR / isfinite / 逐位确定性失败 → 停，在独占窗口复现后再相信（争用会造出 42-48 dB 的假失败） |

**S1 与 S2 相互独立，都可以立刻开始。S1 更可能赢，S2 更可能关闭一条线；两者都不碰卡。**

---

## 6. 什么会让这是错的（诚实记账）

**（1）TDM 能不能写出 padded row stride —— 能，这一半不是风险。**
`compute_padding_encoding(128, 8, 16)` → `interval_dw = 64`（2 的幂，`tdm_ops.py:110` 断言通过）、`amount_dw = 4` → `(5, 3)`，`tdm_ops.py:361-362` 给出 136 元素 = **272 B**，与 `X_ROW_B = D * 2 + 16` 逐字节相同。`r4.i1.g16` 的 bank 论证原生存活，不需要替代方案。这条算到位了，**但它是 Python 编码层的推演，不是 ISA 验证**——A4 的 `0x07510000` grep 就是把它变成证据。

**（2）删掉 g21/g62 是不是亏本交易 —— 是，已实测，所以本计划一条都不删。**
`r19.i2.g60` 删掉 tuple 实测 **−17.27% prod**（`dead_ends.md:258-262`），把 `r7.i1.g21` 重新定价到今天 body 上的 **~+21%**，是全战役唯一的正向机制。任何把 `_NP = 4 + 32` 改成 `_NP = 4` 的方案（`prefetch`、`safety` 两份 mapping 都这么做）在本 op 上**既是已测的亏损，又是静默错答案**（dual-use）。本计划的 E8 明确把它标成不动项。

**（3）我最大的赌注是 S1，不是 TDM。**
「把发射点移到 barrier-2 之后能把 cover 从 389 抬回 ~600」是我从 ISA 位置推出来的**预测**，不是观察。`facts.md:29-30` 的过滤器（"移后必输，五比零"）指向相反。我认为那五次的全排空点都在下一迭代顶部而本 build 在中段，因此 cover 的符号相反——但这条区分我没有编译验证过。**S1 的 ISA 门（cover ≥ 550）就是为此设的，它零成本，且在任何卡时之前。**

**（4）本轮没有 GPU、没有编译。** 下列各条我没有验证：
- `make_tensor_descriptor_2d` 是否接受运行时 index 型 `global_offset`。aiter **全部** 2D 描述符都是 `global_offset=(0, 0)` + `update_tensor_descriptor_2d_addr64` 推进（`gemm_a8w8_256x256_gfx1250.py:352`）。本计划传运行时 offset 是因为逐次重建是保持 `oob_outer_bound` 不失效的唯一方式（dgroup1 无法被 `addr64` 更新）。**若 trace 期报类型错，退路是 `(0,0)` + `addr64`，同时接受钳位退化为纯 `qt_s` 的 in-kernel 形式，并把 A6 的纸面枚举升为唯一防线。**
- `elem_bytes=2` + `pad_enable=1` 在本树**无先例**：aiter 全部调用点 `elem_bytes=1`（`gemm:356` 等），int8 下元素与字节退化相等，所以 `strides` 进 `g1_s5` 到底是元素还是字节**从未被一个 element≠byte 的真 kernel 验证过**。A2 必须加这一项。
- `tensor_wait` 的运行时代价、TENSORcnt 的最大深度、`n>0` 的语义：全部未测。TDM 在本机**从未发射过**；`output/0925__flydsl/tdm-screen/tdm_isa.s` 只证明它会 lower（`tensor_load_to_lds s[8:11], s[0:7]` / `s_wait_tensorcnt 0x0`），不证明它会跑。
- `num_warps=4`（把四波复制填充折叠成一次分布式填充，是 TDM 在本 op 唯一能删**冗余**而非**搬移**的工作）**不在本计划内**，因为：它需要 `rocdl.wave_id()`/TTMP8 与 kernel 自己的 `wave = fx.Int32(fx.thread_idx.x) // fx.Int32(WAVE)`（`:180`）编号顺序一致，而这两套来源的等价性**没有任何门禁能检查**，四波合起来仍覆盖 32 行、总字节也对，**不报错不挂卡，只是行内容错位**；而且 LDS 基址覆盖写法会摧毁 per-wave 偏移。它同时受 `beat`/`g2work` 的天花板（= 冠军）约束，所以即便成功也不越顶。
- `k_del` / delta kernel 未评估。结构上它是 elementwise reduction，理论上是本 op 唯一非 dual-use 的站点，但不在热路径预算里。

**（5）一条应当顺手修的过期注释。** 冠军 `.../op/current/kernels.py:862-869` 与 `:880-887` 仍写着 `k_dq` 有 `s_wait_loadcnt_dscnt 0x0` 且把 g21 的掩护截到 299 条指令。我今天在 shipped ISA 上数过：`output/0924__flydsl/bq-probe/k_dq_bq64.s` 的该指令计数为 **0**。这段注释精确地描述了一个不存在的诱人靶子，下一个读到它的 agent 极可能据此把 `k_dq` 选成 TDM 的首个目标。

**（6）如果 S2 的围栏真的消失了。** 那我在第 1 节 (c) 的位置论证错了，4-wave 的 TDM 臂以一个真实的预测重新打开——但它的天花板仍然是冠军的 507.34（第 1 节 (d) 的普查），而 bar 是 713.99。即便如此它也不该抢卡时槽位，除非同一轮里 S1m 已经证明 cover 就是那 33.6%。