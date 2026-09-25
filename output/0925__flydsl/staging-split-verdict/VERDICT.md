## 判决

**拆分不命中实测病因，应当放弃；4-wave 方向本身应当搁置。**

理由不是论证，是两份 ISA 的逐条普查。我把冠军的 `k_dkdv` 反汇编拉出来和 4-wave 的并排数了一遍——**四份 mapping 的八位评审里没有一个人做这件事**：

| 热循环 `.LBB0_8` | 冠军 `rounds/022/1-opt/raw/isa_cur_a_k_dkdv.s` | 4-wave `g1b-4wave/isa_4wave_k_dkdv.s` |
|---|---|---|
| 指令数 | 779 | 806 |
| `ds_store_b128` | **40** | **40** |
| `ds_load_tr16_b128` | 40 | 40 |
| `buffer_load_b128` / `_b32` | 32 / 4 | 32 / 4 |
| `v_wmma_f32_16x16x32_bf16` | 64 | 64 |
| VGPR / spill | 904 / 0 | 882 / 0 |
| waves per CU | 4 (4 WG × 1 wave) | 4 (1 WG × 4 waves) |
| `s_barrier_signal` | **0** | 2 |
| `s_wait_loadcnt_dscnt 0x0` | **0** | 1 |
| prefetch 发射→排空距离 | **~604 指令**（load 在 rel-60..189，消费 wait 在下一迭代 rel-14） | **389 指令**（rel-59..194 → rel-583 全计数器围栏） |

**每个 wave 的工作量逐条相同。每 CU 的 wave 数相同。每 CU 的 LDS 写字节相同（4×32×32×16 = 65536 B 两边都付）。每 CU 的 global b128 相同（128 条）。** 两者的全部差异是：两个 barrier 会合点，加它们拖进来的那条全计数器围栏。

---

## 1. 拆分是否命中实测病因——不命中，定量

### 病因诊断第 1 条（staging 复制）是**不存在的病**

冠军的 staging nest 与本 build 逐字相同，每 lane 32 条 `ds_store_b128`，4 waves/CU，合计 65536 B/CU/body——**和今天一模一样，而它快 50%**。把它砍到 16384 B 是砍到冠军水位以下，不可能补回一个冠军根本没避免的开销。这条诊断是拿本 build 去对比一个从未存在过的"非复制理想态"。

### 病因诊断第 2 条（occupancy 掉到 1 WG/CU）**归因错了**

`kernels.py:256` 自己写着 `Waves per CU is unchanged at 4.`，我按 ISA 验了：冠军 LDS 70656 B → 4 WG/CU，904 VGPR → 1 wave/SIMD，= 4 waves/CU；4-wave 是 1 WG × 4 waves = 4 waves/CU。**冠军也没有第二个 wave 可以切换。** 它的优势不是掩护，是它的 wave 从不等任何别的 wave。

### 拆分能动的量，和它动不了的量

| 量 | 现在 | 拆分后 | 备注 |
|---|---|---|---|
| `ds_store_b128` / wave / body | 40 | 16 | B 环 32→8；A 环 8 条被 `wcol` 分区，切不动 |
| 其中在 barrier-1 真正 outstanding 的 | ~24（v23 块 rel-542..557 + A 环 rel-537..561） | ~12 | v22 块在 rel-122..208，距围栏 380+ 指令，早退休了 |
| barrier-2 的 `s_wait_dscnt 0x0` 深度 | 40 条 `ds_load_tr16` | **40** | 读侧不得带 `wave` 项，结构免疫 |
| `buffer_load_b128` | 32 | **32** | 见下 |
| barrier-1 的 `s_wait_loadcnt_dscnt 0x0` 的 loadcnt 半边 | 36 条在飞 | **36** | 拆分碰不到 |

**任务书的第二个目标 `buffer_load_b128 160 → 40` 是结构不可达，不是调参不到位。** `kernels.py:402-405` 把 `qp[dt]`/`dp[dt]` 全部四个 dt shuffle 成 `qfr`/`dfr`，`:435`/`:438` 对 `dt in range_constexpr(NDT)` 全部喂进 S/P WMMA——每个 wave 都要在**自己的寄存器里**完成 K=128 收缩。那 32 条 load 是 GEMM 操作数，不是 staging load。

另外：任务书的 `80 / 160 / 四对 barrier` 是**整函数静态计数**，不是 per-wave per-body。真实是每 body 40 store / 32 load / **2 对** barrier。按 80→20 设验收门会把一个生效的编辑误判成失效。

### 真正新增的机制

`isa_4wave_k_dkdv.s:2091` 的 `s_wait_loadcnt_dscnt 0x0`——全文件仅此一条，就在 barrier-1 前。它把 prefetch 掩护从 604 指令砍到 389（−36%），并让循环头的两条 `s_wait_loadcnt`（rel-4 `0x21`、rel-14 `0x0`）变成**死指令**（我验过：2093-2314 区间 `buffer_|global_|flat_|scratch_` 命中数 = 0，回边处 loadcnt 已是 0）。

> 原诊断说"`s_wait_loadcnt 0x0` 保持 2 条，所以不是 waitcnt 回归"——那是在读两条 no-op；被它随手放过的"保守全计数器围栏出现了一次"才是回归。

对照组就在同一个文件里：masked body `.LBB0_4` 是 `carry=False`，没有活的 VMEM 跨越 barrier，它的两个 barrier（:1026、:1107）用的都是**纯 `s_wait_dscnt 0x0`**。围栏的 loadcnt 半边，精确地在"carried prefetch 跨越 barrier"时出现。

这正是 `r19.i2.g60`（删掉 prefetch −17.27%）和 `r20.i2.g62`（加深 prefetch +2.00%，全战役唯一的正向机制）指着的那个量。**拆分对它的贡献是零。**

---

## 2. 编辑清单

**不建议现在建。** 但既然要求给，给最省的形式（rotation，非 select tree：零 `v_cndmask`，编译期索引），并附 exactly-once 证明。标注为"测量工具，不是修复"。

### 编辑 1 — `/home/lihuzhan/g2work/kernels.py:57` 之后插入（防静默错答）

当前文本：
```python
WAVES_DKDV = BLOCK_KV // KV_PER_WAVE   # 4
```
追加：
```python
# r24 staging split pins head-dim quarter dt to wave dt. WAVES_DKDV == NDT == 4 is a
# COINCIDENCE of D == BLOCK_KV == 128, the same class as S_ROW_B == X_ROW_B (:246).
assert WAVES_DKDV == NDT, f"staging split needs one wave per head-dim quarter: {WAVES_DKDV} != {NDT}"
```
没有它，`D=64` 时 wave 2/3 写到 `X_ROW_B` 的 padding 和下一行，`BLOCK_KV=64` 时半行永不被写——都是跑得通的错答。

### 编辑 2 — `:203` 之后插入 rotation

当前文本：
```python
    wcol = wave * fx.Int32(KV_PER_WAVE * 2)       # its A-ring byte band: 0/64/128/192
```
追加：
```python
    # r24 -- B-RING byte band of this wave's head-dim quarter. NUMERICALLY equal to wcol
    # today and that is a COINCIDENCE (KV_PER_WAVE*2 == (D//NDT)*2 == 64). Keep separate.
    xcol = wave * fx.Int32((D // NDT) * 2)        # 0/64/128/192, HEAD-DIM bytes
    # r24 -- HEAD-DIM ROTATION. Register list index j holds TRUE head-dim chunk
    # (j + wave) % NDT. K/V and Q/dO rotate IDENTICALLY, so every WMMA still contracts
    # matching d-ranges and the sum over j still covers all four chunks. Its only purpose
    # is to make list index 0 be this wave's own LDS quarter, so the staging store needs a
    # COMPILE-TIME index -- no v_cndmask, no branch.
    rot = [((fx.Int32(j) + wave) % fx.Int32(NDT)) * fx.Int32(4) for j in range(NDT)]
```

### 编辑 3、4 — `:268` 和 `:275`，**按行号匹配，不要全局替换**

两行当前文本完全相同：
```python
        t = base + (r + row) * rs + half + fx.Int32(dt * 4)
```
各改为：
```python
        t = base + (r + row) * rs + half + rot[dt]
```
⚠ 同一字符串在 `:753`、`:760` 还有两处，属于 `k_dq` 的私有 `gfrag`/`gfrag2`，**不得改**（改了是 `NameError`，响亮失败，但会浪费一轮）。

### 编辑 5 — `:395-401` 整个 dt/u 巢

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
替换为（`:388` 的 `for hh`、`:389-392` 的 qp/dp、`:393-394` 的 `xo`、`:402-405` 的 qfr/dfr 全部不动）：
```python
            # r24 -- WAVE-SPLIT STAGING STORE. List index 0 holds TRUE head-dim chunk
            # `wave` (see `rot` at :204), i.e. LDS byte columns [64*wave, 64*wave+64).
            # Wave w commits ONLY that quarter. The LOAD is deliberately NOT split: the
            # S/P WMMA below contracts all of D=128 out of these same registers, so
            # `pre`, `_NP` and the scf.for carried tuple are UNCHANGED.
            for u in range_constexpr(2):
                o = xo + xcol + fx.Int32(u * 32)
                llvm_dialect.store(fx.as_ir_value(dp[0][u]),
                                   create_llvm_ptr(lds_do + o, address_space=3))
                llvm_dialect.store(fx.as_ir_value(qp[0][u]),
                                   create_llvm_ptr(lds_q + o, address_space=3))
```

### 编辑 6 — `:467-472` 注释（**不是装饰，是防删**）

当前文本：
```python
        # r23 -- BARRIER RESTORED. In THIS build the Q/dO staging is REPLICATED: all
        # four waves run _ldqd on wave-uniform (qt, gh) and each writes the entire
        # [32 q][D] image with identical bytes, then reads back only bytes it wrote, so
        # today this is a same-value redundancy rather than a live dependence. It is
        # restored because G1a measured it free at one wave and because it becomes
        # mandatory the moment the staging store is split across waves.
```
改写为：r24 之后这个 barrier 是**活的跨 wave RAW**。wave w 只写 `[64w, 64w+64)`，却在 `:491-492` 读全部 256 B。删掉它不会报错——读到的是**上一个 query pair** 的 3/4 行，形状对、量级合理、答案错。`:473` 的 `fx.barrier()` 本身不动。

### 明确的非编辑（applier 应逐字核对未变）

`:284-319 _ldqd`、`:545 _NP = 4 + 32`、`:586-587`/`:596-597` 两处 prologue、`:311 rocdl.sched_barrier(0)`、`:402-405 qfr/dfr`、`:435/:438` S/P GEMM、`:489-492` 读侧、`:258` LDS 分配。**任务书"拆分会改 carry 宽度和 tuple 布局"的前提是错的**——只有 `pre` 的消费者变，`pre` 本身不动。这是对任务框架最大的一处修正，它把整个"redesign 风险"移除了。

### EXACTLY-ONCE 证明

设一个 buffer（`lds_q`，`lds_do` 同构），image = 32 行 × 256 B 有效载荷，行距 `X_ROW_B = 272`（每行 16 B padding 既不写也不读）。

**地址。** `row = lane % 16 ∈ [0,16)`（:196），`half = lane // 16 ∈ {0,1}`（:197），`lane = fx.lane_id()` 是 **wave 内** 下标 0..31（`flydsl/expr/gpu.py:67`），故四个 wave 呈现完全相同的 `(row, half)` 集合。`wave = thread_idx.x // 32`（:180）在 wave 内均匀、跨 wave 互异。

新地址 = `(hh*16 + row)*272 + half*16 + 64*w + 32*u`，即 `R = 16*hh + row`，`C = 64*w + 32*u + 16*half`。

1. **行覆盖。** `(hh, row) ↦ 16*hh + row` 是 `{0,1}×[0,16) → [0,32)` 的双射（`hh = R>>4`, `row = R&15` 可逆）。32 行全覆盖，每行由恰好两条 lane（`half=0`/`half=1`）贡献。
2. **列覆盖。** `C = 16*(half + 2u + 4w)`：`half` 占 bit0，`u` 占 bit1，`w` 占 bit2-3——**互不相交的位域**，故 `(w,u,half) ↦ C` 是到 `{0,16,...,240}` 的双射。每条 store 宽 16 B，16 个块无缝无叠平铺 `[0,256)`。
3. **恰好一次。** `R` 与 `C` 是地址的独立数字组，故 `(w,hh,row,half,u) ↦ (R,C)` 单射。计数：`4(w)×16(row)×2(hh)×2(half)×2(u) = 512` 块 = `32 行 × 16 块/行`。单射 + 等势 ⇒ 双射。每字节被**恰好一个 wave 的恰好一条 lane 写恰好一次**；`w` 在 wave 内均匀，故两个 wave 永不触碰同一字节。合计 `512 × 16 B = 8192 B` = image，两个 buffer 16384 B。
4. **载荷是对的（这一步和地址必须同时落地，否则静默错答）。** `gfrag2` 的 `t = base + (r+row)*rs + half + rot[dt]`，`g_q`/`g_do` 是 vec8 视图，故 tile 下标 `half + d_j*4` = 字节列 `half*16 + d_j*64`，`t+2` = `+32 B`。取 `j=0` 时 `d_0 = wave`，源字节列 = `half*16 + 64*wave + u*32`，**与目的列逐项相同**。全局列→LDS 列的恒等映射保持不变。
5. **每个 wave 都能读到全部。** 读侧 `:490-492`：`c = (lane_c + dtile*16)*2`，`lane_c ∈ {0,8}`，`dtile ∈ [0,8)` → `c ∈ {0,16,...,240}` 全覆盖；`lane_r = (lane//16)*8 + lane%8 ∈ [0,16)`，`tr` 另取 `+16*X_ROW_B` → 行 `[0,32)`。**无 `wave` 项**，满足 r23 规则（本编辑根本不碰读侧）。由 (3)，`:473` 退休时整幅 image 已驻留。
6. **两个 barrier 都必要且充分。** `:473` 隔开拆分写与全量读（跨 wave RAW，新增）；`:512` 隔开全量读与下一 pair 的拆分写（跨 wave WAR，已存在）。**不需要第三个 barrier**：S/P GEMM（`:432-439`）从本 wave 仍持有的**寄存器**取 `qfr`/`dfr`，不走 LDS。trip count 全部由 `kv0`（block 导出，`:199-201` 明确要求 wave-uniform）导出，四个 wave 执行相同次数的 barrier，不会挂。
7. **rotation 不破坏 GEMM。** `:268` 和 `:275` 用同一个 `rot[]`，故 `wmma(kf[kh][j], qfr[j])` 始终配对相同的 d-range；`j` 跑遍 0..3 时 `d_j` 跑遍 `{0,1,2,3}`，K=128 收缩恰好覆盖一次。`_c = dt % 2` 的双链拆分对每个 `w` 都得到同样的两个集合 `{0,2}`/`{1,3}`，只是链内累加**顺序**变了——fp32 重结合是链内 2 项重排，**SQNR 预期移动 < 0.1 dB**。验收门设 ±0.2 dB；若掉超过 1 dB，说明 `:268`/`:275` 只改了一边。

**最可能的静默错答**（按概率序）：① 只改 `:275` 不改 `:268`（两行文本相同，且 wave 0 恰好正确，单 tile 抽查会通过）；② 改了地址忘了载荷，或反之；③ 改了 `qp` 没改 `dp`（dV 错 dK 对）；④ 凭 `:467-472` 的旧注释删掉 `:473`；⑤ 顺手改了 `:753`/`:760`（这条是响亮失败）。

---

## 3. 预测收益——不能达到平价，上限就是冠军

上界是硬的：**即使 barrier 成本为零，4-wave build 就是冠军的指令流**（40/40/32/64 逐条相同，4 waves/CU 相同）。它的天花板是 ~507，不在其上。这不是估计，是上面那张普查表。

拆分本身的收益：删 24 条指令 / 806 = 3.0% 的 issue；在 barrier-1 真正 outstanding 的 LDS op 从 ~24 降到 ~12，而那条围栏的 loadcnt 半边挂着 36 条 389 指令前发出的 HBM/L2 load，`max()` 的约束项不动；barrier-2 的 40 条读一条不减。

**预测 337.02 → 330-355 TF/s**，点估计 ~342（+1.5%）。下限必须留负：`facts.md:122-133` 记录六次删除六次变慢（g47/g49/g56/g60/P1/P3），其中"WMMA 数不变、删掉全部 80 条 LDS op 是 **−8.19%**"就是在这个 body 上测的。那是单波拓扑，不必然转移，但它是唯一的同 body 实测，符号为负。

**不要按 +2% 设门。** 一个 0% 或 −2% 的结果是把 loadcnt 围栏孤立出来的**有效信息**，不是"编辑没生效"。生效判据只看 ISA：`.LBB0_8` 的 `ds_store_b128` 必须是 16。

对冠军 507.34 的差距：拆分后仍 −30%。对 aiter 713.99：−52%。

---

## 4. 备选，按 (期望值 × 概率) / 成本 排序

**① TDM / LDS-resident B ring —— 唯一天花板在冠军之上的方向。**
aiter 用同一个问题跑到 713.99 TF/s，零 `buffer_load`，26 个 barrier——机制在这块卡上已被证明，只是没在这个 DSL 里证明。`/home/lihuzhan/.local/flydsl032/flydsl/expr/rocdl/tdm_ops.py` 有完整 API（`make_tensor_descriptor_2d`/`tensor_load_2d`/`tensor_wait`/`update_tensor_descriptor_2d_addr_lo`）。
期望值：高——这是唯一能让每个 wave 只读自己那 1/4 global 的路径，也是 BLOCK_KV=128 唯一的存在理由。概率：中（lowering 未验证）。成本：低（独立 kernel，不碰 `k_dkdv`）。
**早杀门（零卡时）**：编译一个最小 TDM kernel，`grep` ISA 里有没有 TDM opcode。不下降就地停，一分钟结论。

**② 退回 BLOCK_KV=32 —— 就是冠军。** 成本 = `git checkout`。期望值 = 保住 507.34。这是默认动作，不是备选。

**③ 搁置 4-wave build。** 见第 6 节。

**④ 2 workgroups/CU —— 结构不可达，判零。**
每 wave 寄存器地板：32 个 v8f32 累加器 = 256（`NST`，:46）+ carried tuple `_NP = 4+32` → `2×(4 + 32×8)` = 520 dword ≈ 260 VGPR + `kf`/`vf` 128 = **644**，还没算任何临时。512 的门槛要求删掉 prefetch（`r7.i1.g21` + `r20.i2.g62`，全战役唯一的正向机制，删它实测 −17.27%）并把 `NKV` 减半，落到 ~492 且两个胜利全丢。**不要花时间定价。**

---

## 5. 下一步建一个：TDM smoke gate

**独立 kernel**，不碰 `k_dkdv`：用 `make_tensor_descriptor_2d` + `tensor_load_2d` 把一块 `[32][128]` bf16 从 global 填进 LDS，读回来对 CPU 参考。它一次回答三个整个战役悬着的问题：(a) FlyDSL 的 TDM 在这块卡上 lower 并落地吗；(b) `s_wait_dscnt 0x0` 真的不等一个 TENSORcnt 退休的填充吗；(c) `tensor_wait` 多贵。

**零卡时筛子（先做，做完再决定要不要上卡）**：只编译，不 launch。`grep` 生成的 `.s` 找 TDM opcode。没有就地停——证明 lowering 不存在，省掉全部卡时。

**上卡时按 `gfx1250-card-safety` 协议**：先用**故意小于张量的 extent**，单次 launch，不进 sweep。`tensor_load_2d` 打在越界 descriptor 上会等一个永不退休的 TENSORcnt——那不是错答，那是楔死的卡和一次人工断电。

---

## 6. 不建的

**拆分本身（任何变体）。** 它退还的是冠军同样在付的开销，天花板等于冠军，而它还**永久封死**当前唯一的一行实验：今天 `:473` 是可证明冗余的（复制式 staging，每个 wave 读回的都是自己写的字节，`:512` 维持迭代锁步），删掉它可以检验"围栏跟着 barrier 还是跟着依赖"。拆分一落地，`:473` 就变成活的跨 wave RAW，这个选项就没了。顺序不可逆：**要么先删 `:473` 测一次，要么不要拆**。

**`(wave == dt).select(real_ptr, dump_ptr)` 门控。** `s_wait_dscnt` 按**已发射的 LDS 操作条数**排空，不按地址域。32 条 store 一条不少地发出去，drain 深度一点不降——它想降的那个量降了**恰好零**——同时把 32 条 lane × 16 B 全砸进一块 32 B 的 scratch，是保证的 32 路 bank 冲突。`:258` 也不需要加宽 32 B。这个问题就此了结。

**拆分 global load。** `:402-405` → `:435`/`:438` 把全部四个 `dt` 喂进收缩 D=128 的 S/P WMMA 链。让 `_ldqd` 的 dt 巢依赖 `wave`，build 照样发射、照样 launch、dK/dV 量级照样合理——而它在算 K=32 而不是 K=128。**这是这个任务里最容易拿到的静默错答。**

**2 WG/CU。** 见 ④。

**继续往 4-wave 上打补丁。** 它的论点是 `BLOCK_KV=128` 买到 4× 的 Q/dO 算术强度。**它没买到**——四个 wave 在 wave-uniform 的 `(qt, gh)` 上复制 `_ldqd`，每 CU 的 global b128 两边都是 128。同 wave 数、同寄存器、同 global 流量、同 LDS 流量、同指令组合，外加两个 barrier 和一条把 prefetch 掩护砍掉 36% 的全计数器围栏。**它是一个纯成本项，没有任何抵消项。** 4-wave 几何只有在每个 wave 真的只读自己那一份 global 时才配得上它的 barrier，而那需要 S/P GEMM 从 LDS 取 B 操作数——那是另一个 kernel，是第 5 节那扇门后面的东西，不是这一轮能补出来的。

**保持冠军在产。** `/home/lihuzhan/g2work` 作为实验树留着，不要合并。