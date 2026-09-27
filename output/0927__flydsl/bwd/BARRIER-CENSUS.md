# 4-wave BLOCK_KV=128 k_dkdv 的 barrier 代价：ISA census（研究 A，纯 CPU）

2026-09-27。按 h53/h54 的规则，这次不提第五个凭空想出来的机制，只用 ISA census 和一个预注册实验。
**没有用 GPU**：只在 `fa-repro` 里做了 compile-only（`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=`），
加上用 llvm-mc/llvm-objdump 取字节大小。

证据标签：**[ISA]** 直接从编译出的 ISA/IR 读出；**[MEAS]** 09-25 已有的上卡测量（引用了文件）；**[INF]** 推断，包括模型、估算和预测。

产物：`output/0927__flydsl/bwd/census/`
- `census.py`：按 barrier 切 region，统计每个 region 的内容，并对 loadcnt/dscnt 做 3 次迭代的稳态计数器模拟。每个 `s_wait_*` 会给出它强制退休了哪些操作、这些操作的 cover，以及到第一次使用的 slack。
- `aiter_census.py`：对 aiter `.co` 的主循环做同样的 region 统计。
- `census032.{json,txt}`：全部数字。`dump032/<arm>/k_dkdv_0/{20_llvm_ir.ll,21_final_isa.s}` 是 flydsl 0.3.2 的输出，这是上卡测量时 pin 的版本；`dump0341/` 是 0.3.4.1 的交叉编译。
- `enc/<arm>.dis`：带字节地址的反汇编。`trees/<arm>/` 是编译用的源码副本。

## 0. 被编译的 arm，以及和上卡测量的对应关系

| arm | 源 | 和测量树是否一致 | prod TF/s（同 session 对照） |
|---|---|---|---|
| champ | `$JC/op/current`（r20） | -- | 507.34 / 511.53 / 513.78 / 521.86（各 session 的 cur）[MEAS] |
| w4 | `/home/lihuzhan/g2work`（= `g1b-4wave/4wave.patch`） | ISA 和 `g1b-4wave/isa_4wave_k_dkdv.s` 逐指令相同 [ISA] | 337.02–342.10 [MEAS] |
| s1 | `s1-cover/kernels.py.s1`（= g3work） | ISA 和 `isa_s1_k_dkdv.s` 相同 [ISA] | 335.96 vs w4 337.59 [MEAS] |
| s2 | `s2-barrier-gap/kernels.py.s2`（= g5work） | ISA 和 `isa_s2_k_dkdv.s` 相同 [ISA] | 332.74 vs w4 337.56 [MEAS] |
| nobar | `barrier-price/kernels.py.nobarrier`（= g4work，**以 S1 为底**删掉两个 barrier，结果按构造就是错的） | kernels.py 字节相同 | 495.56 vs w4 342.10 vs cur 521.86 [MEAS] |
| **x1**（新） | w4 删掉 barrier-1，一行 | 仅 compile | 未测 |
| **x2**（新） | w4 两个 barrier 都删，每个 wave 用私有 B ring（Q/dO 镜像） | 仅 compile | 未测 |

champ、w4、x1、x2 在 0.3.2 和 0.3.4.1 下 ISA 逐指令相同 [ISA]，所以本报告的结论和 flydsl 版本无关。
所有 arm 的 scratch 都是 0，没有 spill [ISA]。

## 1. Census 主表：`qloop_full` 循环体 `.LBB0_8`

这个循环占 prod 迭代数的 99% 以上；masked loop 每个 WG 只有 G·nmaskp≈4–16 次迭代，只在附录列出。
R0/R1/R2 表示被 `s_barrier_signal` 切开的区段。rel 是相对循环头的指令序号。

| arm | VGPR | LDS B | 循环指令 / 字节 | signal→wait 间隔（指令数） | WMMA R0/R1/R2 | R1 内 ds_load_tr / VALU | 每个 signal 前的强制 drain [ISA] |
|---|--:|--:|--:|---|---|---|---|
| champ | 904 | 70656 | 778 / 4716 | 无 barrier | 64 | -- | -- |
| w4 | 882 | 82944 | 806 / 4900 | 1，2（间隔里有 1 个 WMMA） | 32 / **1** / 31 | **40 / 3** | B1：`s_wait_loadcnt_dscnt 0x0` 强制 36 条 vmem load（slack 227–338）和 40 条 ds_store（最后一条 cover 23）。B2：`s_wait_dscnt 0x0` 强制最后 6 条 ds_load_tr（cover 9–14） |
| s1 | 832 | 82944 | 765 / 4728 | 14，3 | 32 / 2 / 30 | 40 / 21 | B1：loadcnt_dscnt 0，强制 3 条 vmem 和 40 条 st。B2：dscnt 0，强制 4 条 ds_load（cover 18–22） |
| s2 | 832 | 82944 | 774 / 4764 | 15，**234** | 32 / **0** / 32 | 40 / 11 | B1：loadcnt_dscnt 0，强制 1 条 vmem 和 40 条 st。B2：**没有 drain**（裸 intrinsic、没有 fence，见 F4） |
| nobar | 808 | 82944 | 800 / 4872 | -- | 64 | -- | -- |
| x1 | 938 | 82944 | 781 / 4792 | 2 | 35 / 29 | -- | 唯一的 B：`s_wait_loadcnt 0x0` 强制 36 条 vmem（slack 160–271）；ds 已被数据 wait 排空 |
| x2 | 938 | 148480 | 774 / 4764 | -- | 64 | -- | -- |

所有 FlyDSL 构建都没有 `s_wait_tensorcnt`（没有 TDM），循环内也没有 `s_wait_storecnt`（没有 vmem store）[ISA]。

对比参照：aiter `bwd_hd128_bf16_causal_br_a32_pssk.co` 的主循环。循环展开 4 次，共 2753 条指令、18116 B、320 WMMA、352 条 ds_load（192 条 tr16_b128 + 160 条 b128）、16 条 ds_store、12 条 TDM、128 条原子操作。每个展开子体有 2 对 barrier [ISA]：

| aiter 区段（每个子体重复一次） | 指令 | WMMA | ds_load | 结尾 | signal 前的 wait |
|---|--:|--:|--:|---|---|
| wait → signal | 381 | 55 | 72 | signal | `s_wait_dscnt 0x14`（**仍有 20 条 LDS 操作在飞**） |
| signal → wait | 18 | **3** | 0 | wait | -- |
| wait → signal | 247 | 16 | 16 | signal | `s_wait_tensorcnt 0x3`、`s_wait_dscnt 0x10` |
| signal → wait | 38 | **6** | 0 | wait | -- |

## 2. 发现

**F1 [ISA] drain 不是硬件 barrier 自带的，而是 `fx.barrier()` 生成的 workgroup fence 带来的。**
`20_llvm_ir.ll` 里每个 `fx.barrier()` 都是
`fence syncscope("workgroup") release; s.barrier.signal(-1); s.barrier.wait(-1); fence syncscope("workgroup") acquire`。
release fence 被 SIInsertWaitcnts 降成 `s_wait_loadcnt 0` 和 `s_wait_dscnt 0`。
s2 里 barrier-2 改用了没有 fence 的 `_mrocdl.s_barrier_signal`，它前面就一条 wait 都没有。
所以回答任务里的具体问题：会，编译器会在 s_barrier 前插满 memory drain。w4 的 barrier-1 在同一迭代内就把 36 条预取 load 强制退休，而这些 load 的第一次使用在 227–338 条指令之后，也就是下一迭代循环头的寄存器轮转。
**但 S1 已经把这一项的上卡代价测成 0**（见 F3）。

**F2 [ISA] w4 的 barrier 之间有一个几乎没有 WMMA 的 LDS 窗口，aiter 没有。**
w4 的 R1 有 55 条指令，其中 40 条是全部的 `ds_load_tr16_b128`，只有 1 个 WMMA。后面紧跟 `dscnt 0` 和 rendezvous。
R2 的 31 个 WMMA 全部依赖 R1 的 load。barrier-1 前面则是 24 条晚发的 ds_store（16 条 Q/dO hh=1，8 条 P/dS），加一个 dscnt 0。
所以每次迭代每个 wave 都要串行走一遍：晚 store 排空 → rendezvous → 40 条转置读 → 排空 → rendezvous，全程没有自己的矩阵工作可以覆盖。
aiter 的 barrier 密度和我们相同（每个 barrier 约 40 WMMA，w4 是 32），也是每个 SIMD 1 个 wave（VGPR 1024）。
但 aiter 的**每个** barrier 间区段都有 16–55 个 WMMA，与 16–72 条 LDS 读交织；**每个** signal→wait 间隔里还有 3–6 个 WMMA 和 3–4 个原子操作；signal 时从不排空到 0（dscnt 0x14/0x10）。
这是 census 里最明显的结构差异。

**F3 [ISA] 纠正 S1（h50）的 cover 数字：S1 实际上缩短了 cover，而不是延长。**
h50 记录的是 cover 389→529，但它只量到 fence。S1 把 36 条预取移到 barrier-2 之后，这些 load 在下一迭代循环头被寄存器轮转的 `v_mov_b64` 提前强制退休了：20 条在 cover 132–166，3 条在 176，7 条在 342–367，4 条 b32 在 262–266，只剩 3 条留给 fence（529）。
所以 min cover 实际是 389→**132**（−66%），而 prod 只变了 0.995x [MEAS]。
结论和 h50 一样（4-wave 体里 vmem cover 不是约束项），只是证据更强，原来那个数字作废。

**F4 [ISA] S2 只拆了 barrier-2，barrier-1 原样保留；拆出来的 barrier-2 也没有 dscnt 保护。**
S2 的 barrier-1 仍是 loadcnt_dscnt 0 加 15 条指令的间隔。barrier-2 在 rel 536 signal，此时 40 条 ds_load_tr 还没完成；它的 WAR 安全性依赖 LDS 按发射顺序服务，架构上没有保证。
S2 的 R2 WMMA 在 signal 之后 8 条指令处的 `dscnt 0x6` 上等数据。所以 S2 并没有给 R1 的 LDS 读提供 WMMA 覆盖。
**barrier-1 单独的作用至今没被测过**：S1、S2 都保留了它，nobar 是两个一起删的。

**F5 [ISA] nobar 和 S1 的单 wave 静态指令流几乎相同，差别只在 barrier 本身。**
两者都是 store 突发 → tr-load 突发 → `dscnt 6` → WMMA（桶状时间线见 `census032.txt`），指令数（800 vs 765）、WMMA、LDS 和 vmem 条数都一样。
区别只有：两次排空到 0、两次 rendezvous、fence 带来的调度边界。
1.449x 的差距用单 wave 的静态延迟解释不了：每次排空最多暴露一次 LDS 往返，加上队尾 ≤6 条。
所以原因只能是**跨 wave 的动态机制**。

**F6 [ISA] 其它被问到的项都排除了。**
- 指令 cache：循环体 w4 4900 B，champ 4716 B，nobar 4872 B，s1 4728 B。快的 nobar 和慢的 w4 只差 28 B；整个 kernel 21.7–22.1 KB，aiter 的循环体是 18.1 KB。
- VGPR 复用：没有 WAW 型 wait（slack 从不为负）。唯一由寄存器复用触发的 wait 是循环头的轮转拷贝，冠军也一样有（F8）。
- 占用率：所有 arm 的 VGPR 都大于 512，都是每个 SIMD 1 个 wave，每个 CU 4 个 wave（[INF]，和 h46 一致）。
- 没有 tensorcnt 或 storecnt 的 drain。

**F7 [MEAS+INF] 代价的量级：每次迭代每个 CU 约 1270 cycle，相当于每个 barrier 约 630 cycle。**
同 session 数据：w4 16.072 ms，nobar 11.095 ms，差 4.977 ms，sclk 1055–1070 MHz [MEAS]。
prod 每个 CU 的串行迭代数：Σ_j 4·(256−4j)·32 / 256 CU ≈ 4160 [INF]（CU 数 256 取自 `pool.md:72`，假设负载均衡；冠军按每 SIMD 算是 ≈4112，两者可比）。
得到 1.197 µs × ~1.06 GHz ≈ **1270 cycle/迭代**。这远大于任何合理的单次 barrier 硬件延迟（几十 cycle 量级）。

**F8 [ISA]（冠军侧的副发现）g62 的"深度 2"预取在下一迭代开头就被强制排空了。**
冠军 `.LBB0_8` 在 rel 3–115 有 67 条 `v_mov` 读取预取 load 的目标寄存器，这是 `pre1→pre0` 的 loop-carried 轮转拷贝。它们在 rel 12 触发 `s_wait_loadcnt 0x0`，强制 33 条 load 退休，cover 603–728 条指令，约 0.8 次迭代。
所以现在的有效预取距离小于 1 次迭代，每次迭代还多付 67 条 mov。
[INF] 把 `qloop_full` 展开 2 次，让 pre0/pre1 靠寄存器改名互换，就能去掉拷贝和这次 drain。预取轴按规则已经关闭，这只是新证据下的线索，需要先 compile-only 筛一遍。

## 3. 最有证据支持的解释 [INF]

**barrier 让 4 个 wave 同相，把每次迭代的 LDS 突发集中到一个没有 WMMA 覆盖的窗口里。**
一个 WG 是 4 个 wave，每个 SIMD 1 个，SIMD 上没有别的 wave 可以切换。
每次迭代，barrier 让 4 个 wave 同时进入 F2 描述的窗口：4×(24 条晚 st + 40 条 tr 读)×512 B ≈ **128 KB 的 LDS 流量，必须在所有 SIMD 都空等的时候被 CU 的 LDS 服务完**，再加两次排空和最慢 wave 的 skew。
在 nobar 和冠军里，同样大小的单 wave 突发落在随机相位上，LDS 服务它的时候另外 3 个 SIMD 正在做 R0/R2 的 580 条 WMMA/VALU，所以被覆盖了。

- 量级：tr 读 80 KB。其中 B ring（Q/dO，64 KB）在同一个 64 KB 段上，按文档写的 256 B/clk 读端口（gfx1250 未实测）约需 256 clk；A ring 16 KB 走另一端口。晚 store 48 KB 的写速率未知，按 128–256 B/clk 算约 190–380 clk。再加两次排空延迟和 skew，大约 700–1000 clk，对应 F7 的 1270 cycle。**量级合理，但没有闭合。**
- 和已被否定的四个机制都相容：
  - 复制 staging（h45）：它只比了单体总数（总数确实相同），没比时间上是否同相。
  - occupancy（h46）：每个 CU 都是 4 个 wave。
  - S1：vmem cover 变短也没影响（F3）。
  - S2：barrier-1 仍在 tr 读突发前对齐 4 个 wave，R2 的 WMMA 仍依赖这次突发（F4）。
- 和 aiter 相容：barrier 密度一样、占用率一样，但它没有"无 WMMA 的 LDS 窗口"，signal 时不排空，store 字节数约少 10 倍（TDM，不复制）（F2）。
- census **排除不了**的另一种解释（H-count）：每次 rendezvous 有一个固定的约 630 cycle 代价，比如硬件 barrier 延迟加 skew，和 LDS 字节数无关。下面的 X1/X5 用来区分这两种解释。

## 4. 上卡要试的修复（预注册；都在 w4 基础上，同 session、回文顺序、prod 排名）

| id | 改动 | 合法性 / 状态 | H-align（本报告）预测 | H-count 预测 |
|---|---|---|---|---|
| **X1**（先做，1 行） | 删 barrier-1（`trees/x1`） | **合法**：Q/dO 是复制 staging，每个 wave 只读自己写的同值字节；A ring 按 wave 分列带；跨迭代 WAR 仍由 barrier-2 保护。compile-only 通过：VGPR 938，scratch 0 | **≤ +8%**：barrier-2 仍在每次迭代对齐 wave，突发仍然同相，只省掉一次排空 | **≥ +15%**（342 → ≥395 TF/s），恢复约一半代价 |
| **X2** | 两个 barrier 都删，每个 wave 用私有 B ring（`trees/x2`，LDS 148480 B，A ring 移到 2×64 KB 处） | **合法**：没有任何跨 wave LDS 共享。compile-only 通过：VGPR 938，scratch 0。ISA 结构和冠军一样（单区段，只有数据驱动的 dscnt） | ≈ nobar（≥1.40x w4，约 0.95x 冠军） | 同左 |
| X5 | 去复制 staging：wave w 只存 dt==w 的 Q/dO（每个 wave 32→8 条 st），barrier-1 变成真正必需 | 需要写代码 + UT | +5–7%（锁相窗口的 LDS 操作数 −19%） | ≈ 0（±floor） |

判读：X1 ≥ +15% 且 X5 ≈ 0，说明 barrier 次数本身是代价，方向应是减少 barrier（双缓冲环）；X1 ≤ +8% 且 X5 ≥ +4%，说明代价来自同相的 LDS 窗口；落在中间算不确定，不再加新机制。
X2 本身就是一个**合法的**无 barrier 4-wave kernel，把 h51"barrier 就是全部代价"从错误构造的探针变成可以过 UT 的构建。

**给 5-GEMM 融合设计的结论 [INF]：** dS 必须跨 wave，所以每次迭代至少要有一个 barrier。
要让它便宜，就照 aiter 的做法：
1. 不用 `fx.barrier()`，改用 `_mrocdl.s_barrier_signal(-1)` / `s_barrier_wait(-1)`（来自 `flydsl._mlir.dialects.rocdl`），配上显式的**部分** `s_wait_dscnt N`，N>0，只等本环的 store。这需要 dS 和 Q/dO 环做双缓冲。
2. barrier 后的 LDS 读要和上一阶段 dK/dV GEMM 的 WMMA 交织（软件流水一级），让每个区段、每个 signal→wait 间隔里都有 WMMA。
3. staging 不复制（TDM 或按 wave 切分）。

风险：裸 intrinsic 对 LLVM 没有内存语义。每次构建都要用 `census.py` 核对 ISA 里没有 ds 操作越过 signal/wait。S2 里没有越过，但这不是保证。
另外：named barrier 在这里帮不上忙，因为 4 个 wave 共享同一批数据；关键在于 signal 前不排空、间隔里有工作。

安全：X1 和 X2 的所有改动只是删 barrier 和改 LDS 偏移。上卡前按规则 4/5 先跑 `screen4w.py` UT（15 个形状）和 validation。X2 的 LDS 148480 < 327680 B。

## 附录：masked loop `.LBB0_4`（w4）[ISA]

692 条指令，R0/R1/R2 = 585/81/26，WMMA 32/17/15。R1 里 40 条 tr 读和 17 个 WMMA 交织，比 full loop 好。但每个 WG 只有约 4–16 次迭代，对 prod 可以忽略。
