# ASM bwd main kernel (aiter `fmha_bwd_hd128_bf16_causal_br_a32_pssk`) 逆向，对照 fly r29

这次只读 ATT 的 csv/json 和 fly 源码，没有碰 GPU。用到的工具有 `0930__bwd/tools/attsum.py`，还有一个新写的只用标准库的脚本 `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/notes/att_wmma_gaps.py`（用法：`<ui_dir> <lo> <hi> [waves]`）。它按 wave 时间线把相邻两条 v_wmma 之间的间隔归类。
缩写：`ASMcsv` = `probe/p2_asm/stats_ui_output_agent_29978_dispatch_11.csv`，`FKcsv` = `probe/p2_r29/..._dispatch_10.csv`（k_dkdv），`FQcsv` = `..._dispatch_11.csv`（k_dqg）。引用写作 文件:行号 (vaddr)。

## 0. 先说方法：csv 的 Latency 列会误导，改用 wave 时间线的 WMMA 间隔

csv 按"本条发射到下一条发射"记账，所以 WMMA 显示约 9.3 cyc/条，VALU 每条 1–2 cyc，看起来 VALU 都没被藏住。用 `se0_sm3_sl0_wv*.json` 的时间戳直接量相邻两条 WMMA 的间隔，结果不一样：**ASM 每步 80 个间隔里有 45 个正好 8.0 cyc**，间隔里塞着 2 VALU + 2 exp + 1 atomic，或者 2 DS + 2 v_nop，全部藏住了。损耗只集中在 4 类同步/访存事件上。下文所有预算都用这种"间隔分解"：下限按 8 cyc/间隔算，超出部分按间隔里的指令归因。

## 1. 循环结构与 trip count

- 被 trace 的是 CU1、SIMD3、slot0 上先后跑的 16 个 wave，一个接一个（`ui_output_agent_29978_dispatch_11/filenames.json`，wv0 71552–612904 … wv15 6747562–7064082）。它们是 16 个 WG 各自的一个 wave，**同一时刻每个 CU 只驻留 1 个 WG（4 wave，每个 SIMD 1 wave）**。
- 外层循环：`ASMcsv:3628 (63552) s_cbranch_scc1` 跳回 10812，prologue 指令 hit=32 = 16 wave × 2。**每个 WG 依次处理 2 个 kv block**，也就是 near/far 配对 (j, 63−j)。
- 内层循环：35248–53356 是一个 4 步展开的循环体，步 A/B/C/D 各约 424 条指令（`ASMcsv:1308` 36460 起，边界是 `s_barrier_signal` 36460/40988/45520/50052）。每步之后有一次提前退出（`ASMcsv:1665 (39768)` 等，都跳到 53360）。回跳在 `ASMcsv:3076 (53356) s_branch`。
- hit 数：A/B/C 各 1040 = 16×65，D 1008 = 16×63，合计 4128 个 wave-step，**每 wave 258 步 = 2 × 129**，再加每个外层迭代在 prologue 里剥出的 1 步 = 130。这和推导对得上：kv block 128 行、每步 32 个 q 行时，(8192−128j)/32 对 (j, 63−j) 求和 = 260 = 2×130。每个外层迭代的步数 ≡ 3 (mod 4)，所以 D 恰好比 A/B/C 少 1 次/外层。
- 因果 mask：每步开头 `s_cmp_lt_i32 s93, 4`（`ASMcsv:1819`），对角线上的 4 步才走 64 条 cmp+cndmask 的 mask 块（hit 32，对应 41156–41696 等）。稳态没有 mask 开销。
- 对角块里多算的量：每对执行 1040 个 wave-step，精确三角形需要 1028，浪费 1.2%。

## 2. Tile 与 4 个 wave 的分工

- **WG 拥有 128 行 kv；wave w 拥有 kv[32w, 32w+32)，它的 K、V 常驻 VGPR；每步处理 32 个 q 行**（名字里的 `a32`）。每 wave 每步是一个 32q×32kv 单元，发 80 条 WMMA = 5 个 GEMM × 16。
- 4 个 wave 跑同一条指令流（hit 完全相同），靠每步 2 个 barrier 保持 lockstep，**没有 ping-pong，也没有生产者/消费者分工**。它们之间共享的是：LDS 里的 Q/dO/LSE/delta tile（每个 WG 用 TDM 取一次，4 个 wave 都读）和 dS（每个 wave 写自己那 32 kv 列，4 个 wave 都读全部 128 列）。K/V **不共享**，每个 wave 各持自己那 32 行。
- dQ 按 d 维切给 4 个 wave：每个 wave 算 32q×**32d** 的 dQ 片，收缩维是全部 128 kv（dS 来自 LDS），所以 atomic 地址不会在 wave 之间重复。

## 3. 一步之内的相位表（以步 B 为例，40988–45520）

一步 80 条 WMMA 分成 5 段，每段 16 条：

| 顺序 | GEMM | A 操作数 | B 操作数 | 累加器 | 同段交错执行的其他指令 |
|---|---|---|---|---|---|
| ① 1994 之前 | 上一步 dP 的最后 3 条，然后 BS/BW | — | — | — | 3 条 atomic |
| ② 41988–42876 | **dVᵀ += dOᵀ·P**（128d×32kv，K=32q），8 个 d tile × 2 个 kv tile，无 Z 初值，跨步常驻 | dOᵀ（`ds_load_tr16`，本段先发射） | P：bf16，由 S 的 C 布局直接 `cvt_pk` 得到（b188/b196） | 常驻 | 16 pk_fma（S·scale−LSE）、16 exp、16 pk_add（dP−δ）、16 atomic |
| ③ 42916–43576 | **dKᵀ += Qᵀ·dS**，同样形状，常驻 | Qᵀ（tr16） | dS bf16（b4/b12），`cvt_pk` 得到 | 常驻 | 16 exp、16 pk_mul（P·(dP−δ)）、8 atomic、3 条 TDM、`s_wait_tensorcnt 0x3`、BS2/BW2、预取 dS 的 `ds_load_b128` |
| ④ 43636–44268 | **dQ = dS·K**（32q×32d，4 条链 × 4 个 k-chunk，从 Z 起步） | 4 个 wave 的 dS，从 LDS 读（`ds_load_b128` offset 13056+） | K 的第二种布局（常驻） | 用完即扔（v204–235） | 32 cvt（新的 P/dS→bf16）、4 `ds_store_b128`（自己的 dS 写到 LDS）、预取下一步 Q |
| ⑤ 44308–44844 | **S = Q·Kᵀ**（32q×32kv，4 条链 × 4） | Q（`ds_load_b128`） | Kᵀ（常驻） | v8–39 | 16 pk_mul（dQ×scale，s[10:11]）、预取 dO 的 tr16 |
| ⑥ 44884–45492（接到下一步 ①） | **dP = dO·Vᵀ**（4 条链 × 4） | dO（tr16） | Vᵀ（常驻） | v140–171 | 8 atomic（dQ）、预取下一步 dOᵀ/Qᵀ（tr16）以及 LSE/δ（`ds_load_b128` offset 192/208） |

**跨步的 3 级软件流水**：第 t 步末尾算 tile t+1 的 S/dP；第 t+1 步里它们的 softmax VALU/exp 穿插在 dV/dK WMMA 之间，而这些 dV/dK 用的是 tile t 的 P/dS；dQ 用的是上一步 barrier 之后交换来的 dS；这一步的 dQ 结果要到下一步才用 atomic 分散写出。于是 **所有 VALU 都和"另一个 tile"的 WMMA 没有依赖**，这就是 45 个间隔能做到 8.0 cyc 的原因。

## 4. 操作数的存放位置

- **VGPR 常驻**：K（给 S 用）、K 的第二种布局（给 dQ 用）、Vᵀ，每份 64 VGPR；dKᵀ、dVᵀ 累加器各 128 VGPR（32kv×128d fp32 / 32 lane）。光常驻就约 448 VGPR。每步的临时量（S/dP/dQ 累加器 96、A 操作数约 128 双缓冲、P/dS bf16 32、LSE/δ 32）叠加后大于 512，与 `0925__flydsl/AITER-5GEMM-STUDY.md:43` 记的 `.vgpr_count 1024` 一致。任务说明写的是 512，我判断不对。ATT 反汇编只有 1 条 `s_set_vgpr_msb`，打印出的寄存器号是隐藏了 MSB bank 之后的，所以 d0–56（dK 累加器）和 b0–56（V）看起来会重名。
- **LDS**：Q、dO（行主序；普通读和 tr16 各读一次，一共两个方向）、LSE/δ、dS 交换区（offset 13056 起）。**P/dS 从不为了 dV/dK 去 LDS 转一圈**：P 和 dS 作为 B 操作数直接从 C 布局的寄存器 `cvt_pk` 得到，因为算的是 dVᵀ/dKᵀ 这个方向。
- **全局 → LDS**：全部走 TDM（反汇编显示为 `image_load_generic`）。每 wave 每步 3 条（`ASMcsv:2022 (43132)` 等）。loop 里 `buffer_load` 为 0。
- DS 流量：每 wave 每步 48 条 `ds_load_tr16_b128` + 40 条 `ds_load_b128` + 4 条 `ds_store_b128`，约 45 KB；每个 WG-step 约 180 KB。按均值 1581 cyc/步算约 114 B/clk/CU，只占 512 B/clk 的 22%（REPORT 的两段 LDS）。

## 5. dQ 累加方式

- 每 wave 每步 32 条 `buffer_atomic_add_f32 ... scope:SCOPE_DEV`，数据寄存器 v204–235，voffset 用 v56/60/64/68，立即数偏移为 0/512/1024/1536 和 4096…5632。加起来是 32×32 lane = 32q×32d fp32，每步每个 WG 16 KB。
- 放在哪里：第 t 步 ④ 段算出 dQ，⑤ 段乘 scale，然后 **每个 WMMA 间隔放 1 条 atomic**，分散在 ⑥（8 条）和下一步的 ②③（24 条）里。最后一条在 `ASMcsv:1993 (42888)`，在下一次 dQ 从 Z 起步之前发完。loop 里没有 `s_wait_storecnt`，完全 fire-and-forget。
- 代价（间隔分解，每步）：28 个含 atomic 的间隔平均 22.4 cyc，比 8 多出 **430 cyc/步**。不同 wave 差别很大：wv11–15 为 208，wv0–5 为 634。**这是 ASM 最大的一项损耗**，而且随全卡负载变化。

## 6. Barrier

- 每步 2 组 split barrier，`s_barrier_signal` 和 `s_barrier_wait` 分开放：
  - BS1 在步首，BW1 在 3 条 WMMA 之后（`ASMcsv:1321`，2 cyc/hit，基本不花钱）；
  - BS2 紧跟在 `s_wait_tensorcnt 0x3` 和 `s_wait_dscnt 0x10` 之后，BW2 在 6 条 WMMA 之后（`ASMcsv:1531`、`:2024`，约 53–60 cyc/hit）。BW2 保护 TDM 刚落地的 Q/dO，以及其他 wave 写的 dS 的读后写依赖。
- 间隔分解：2 个 barrier 间隔合计多出 **208 cyc/步**（wv11–15 为 154，wv0–5 为 261），其中包括步首约 30 条 SALU/mask 判断。
- 对照 bwd-history §5：我们自己的 4-wave 版本（h44–h53）barrier 吃掉了 45%。ASM 靠三点把它压到约 13%：(a) 所有 wave 工作量对称；(b) 访存等待（tensor wait）放在 **signal 之前**，所以 wave 之间的时间差只影响 BW2；(c) signal 到 wait 之间隔着 3–6 条 WMMA。

## 7. DS 调度 / WMMA↔DS 切换

- DS 指令几乎每个 WMMA 间隔都有，每次 **2 条**，经常配 1–2 条 v_nop（有 v_nop 的地方每步 39 条）。**加载至少提前一个相位（16 条 WMMA，约 130 cyc）发出**，比如 ⑤ 段取的 dO 到 ⑥ 才用，⑥ 段取的 dOᵀ/Qᵀ 到下一步 ②③ 才用。
- 每步 6 条 `s_wait_dscnt`，阈值是 0x10/0x10/0x6/0xc/0x16/0x14，**从不等到 0**。合计多出 **68 cyc/步**。"含 DS 的间隔"平均 8.1 cyc。所以 REPORT 里测的约 29 cyc 切换代价，在 ASM 里基本不存在，因为消费它的 WMMA 从来不紧跟在 load 后面。
- 另一点，需要上卡复核：S/dP/dQ 都是**4 条同一累加器、首条 Z 的依赖链背靠背连发**，而时间戳显示间隔是 8 cyc（wv13：v[24:31] 链在 t=32/40/47/55）。这和 `0930__roofline/REPORT.md:13`"依赖链 16 cyc"对不上。可能是 SQ 的发射时间戳不等于矩阵核真正开始的时间，也可能是 srcC==vdst 的原地累加可以前递。建议补一个 mb：同一个 C 的 4 条链中间插 2 DS + 2 nop。

## 8. VALU / exp

- 每 wave 每步：32 v_exp_f32、16 v_pk_fma_f32（S·scale−LSE，一条融合，带 neg_lo/hi）、16 v_pk_add_f32（dP−δ）、40 v_pk_mul_f32（16 算 dS=P·(dP−δ)，16 做 dQ×scale，8 做 LSE 预处理，乘 s[14:15]）、32 v_cvt_pk_bf16_f32。合计 136 条，**每条 WMMA 1.7 条**，加上 nop 是 2.2。exp 每个元素 1 条（1024 元素 / 32 lane）。
- 分布很均匀：每个间隔最多 2 VALU + 2 exp（如 `ASMcsv:1928 (42316)` 前后），正好落在 REPORT §2 说的"每条 WMMA 可藏 4 VALU / 2 exp"以内。归为纯 VALU 的间隔平均 8.9 cyc，每步只多出约 7 cyc。
- 对照 fly：k_dkdv 每轮 218 VALU + 32 exp + 66 v_nop，对 64 WMMA，约 3.9 条/WMMA。其中 64 条 v_mov_b64（csv 延迟 174 cyc/轮）来自 carried prefetch 的回边拷贝（`arms/r29/kernels.py:550` 注释）；另有一大段 `4 add, 27 mul, 28 cvt, 23 st, 40 tr16` 连在一起，后面跟 `s_wait_dscnt 0x6`（`FKcsv:1857`，约 188 cyc/轮）。fly 的 softmax 仍是 mul+sub+mul，`VF_KV = False`（`kernels.py:53`）。

## 9. TDM 的等待怎么藏

- prologue：先发 2 条 TDM（K/V），等到 tensorcnt 0x1/0x0；再发 6 条（2 步的量）等 0x3；再发 6 条等 0x6（`ASMcsv` 14108–16840）。进入 loop 时已有 2–3 步的数据在路上。
- 稳态：③ 段开头 `s_wait_tensorcnt 0x3`（`ASMcsv:1994`），只要求最新一组以外的都已落地，然后马上 BS2，再发本步的 3 条 TDM。
- 即便如此，tensor wait 仍多出 **227 cyc/步**（wv11–15 为 125，wv0–5 为 328），是第二大损耗，和 atomic 一样随全卡负载上升。

## 10. 每步 cycle 预算（每 SIMD，单位 = 一个 32q×32kv 单元）

**ASM**（全部 16 个 wave 的间隔分解，每步 80 条 WMMA）：

| 项 | 全部 | 快 wv11–15 | 慢 wv0–5 |
|---|--:|--:|--:|
| WMMA 下限 80×8 | 640 | 640 | 640 |
| VALU/exp/DS/nop 超出（43+45 个纯间隔） | ≈ 7 | ≈ 0 | ≈ 7 |
| dQ atomic 背压 | **430** | 208 | 634 |
| TDM 等待 | 227 | 125 | 328 |
| 2 个 barrier（含步首 SALU） | 208 | 154 | 261 |
| s_wait_dscnt | 68 | 40 | 98 |
| **合计** | **1581** | **1172** | **1972** |

用 PMC 换算：7.71e6 cyc/SIMD ÷ 4160 步/SIMD = **1853**。4160 = b4·hq32·32 对·260 步·4 wave ÷ 1024 SIMD。

**fly r29**（同样方法）：

| 项 | k_dkdv / 轮（64 WMMA） | k_dqg / 4（192/4 WMMA） | 合计 |
|---|--:|--:|--:|
| WMMA 下限 | 512 | 384 | **896** |
| VALU/SALU 成团没藏住 | 292 | 85 | 377 |
| DS 等待 + 含 DS 的间隔（P/dS 过 LDS、tr16） | 398 + 103 | 114 + 71 | **686** |
| vmem 等待 + 含 vmem 的间隔（buffer_load 暂存） | 335 + 100 | 199 + 43 | **677** |
| 背靠背 WMMA 修正 | −27 | −4 | −31 |
| **合计** | **1714** | **892** | **2606** |

PMC 换算：k_dkdv 7.26e6 + k_dqg 4.00e6，除以 4112 单元/SIMD = **2738**，与 roofline 报告里的 1766 + 973 吻合。
行号：k_dkdv loop 在 `FKcsv:1374–2032`；k_dqg loop 在 `FQcsv:610–1918`，大头是 `s_wait_loadcnt 0x20`（`FQcsv:670`，约 205 cyc/轮）。

**差距（ATT 口径）**：fly 比 ASM 每单元多 **1025 cyc（1.65 倍；PMC 口径 1.48 倍）**。拆开看：
- WMMA 下限多 256，因为 k_dqg 重算了 S 和 dP；
- VALU 多 370；
- DS 多 618；
- 全局访存多 450；
- ASM 反过来要付 atomic 和 barrier 的 638，fly 这一项是 0。

## 11. 设计选择对照

| 维度 | fly k_dkdv + k_dqg (r29) | ASM |
|---|---|---|
| kernel 数 / GEMM 数 | 2（+k_delta）；7 个 GEMM（S、dP 算两遍） | 1（+dq_convert）；5 个 GEMM |
| WG / 每 SIMD 的 wave | 1 wave，BLOCK_KV=32；k_dqg BLOCK_Q=64；每 SIMD 1 wave | 4 wave，BLOCK_KV=128，每 wave 32 kv；每 SIMD 1 wave |
| 每轮 q 行 | 32（k_dkdv）/ 64 kv 步（k_dqg） | 32，4 步展开 |
| S 的方向 | Sᵀ = K·Qᵀ（`kernels.py:411`，A=K） | S = Q·Kᵀ（A=Q 取自 LDS，B=Kᵀ 常驻） |
| dV/dK | 计算 dV = Pᵀ·dO，**P/dS 先写 LDS 再用 tr16 读回**（`kernels.py:451,454`） | 计算 dVᵀ = dOᵀ·P，**P/dS 从 C 布局直接 cvt 成 B**，不经 LDS |
| dQ | k_dqg 重算 S/dP，确定性，fp32 链 | 融合进来：dS 经 LDS 交换，按 d 切给 4 个 wave，32 条 SCOPE_DEV fp32 atomic/步，推迟一步、1 条/间隔 |
| 全局 → 片上 | buffer_load → VGPR →（dual-use）ds_store（每轮 32 load + 40 store） | 每 wave 每步 3 条 TDM 直接进 LDS，0 buffer_load |
| DS 等待 | 3 条（0x6 那条约 188 cyc），dscnt 会等到 0 | 6 条，阈值 6–22，从不等到 0，68 cyc/步 |
| barrier | 0（1 wave） | 2 组 split/步，208 cyc |
| 每 WMMA 的 VALU | 3.9（含 64 v_mov_b64、66 v_nop） | 1.7（加 nop 2.2），fma 融合 |
| 每 WMMA 的 exp | 0.5 / 0.67 | 0.4 |
| softmax 流水 | 同一轮内，成团 | 跨步 3 级（S/dP → softmax → dV/dK/dQ → atomic） |
| 每单元 cycle（ATT / PMC） | 2606 / 2738 | 1581 / 1853 |

## 12. 最值得搬到 FlyDSL 的 5 个 ASM 设计（预期收益按"每单元"算，基线是 fly 合计 2606 ATT）

1. **换成 ASM 的 GEMM 方向：S = Q·Kᵀ，计算 dVᵀ = dOᵀ·P、dKᵀ = Qᵀ·dS，P/dS 从 S/dP 累加器 cvt_pk 后直接当 B 操作数。**
   - 做什么：删掉 P/dS 的 8 条 ds_store 和对应的 tr16 读回，以及那一大段后面的 `s_wait_dscnt 0x6`（`FKcsv:1857`，间隔 363 cyc）。
   - 1-wave 就能做，不涉及 barrier/atomic/TDM，没有已知的挂卡路径。
   - 预期：k_dkdv **−250 到 −400 cyc/轮（−10% 到 −15% 总量）**。
   - 注意：它不是已关闭的"Q/dO LDS round trip"（P1）那条轴，搬动的是 P/dS 这条 round trip。
2. **细粒度交错加跨步 3 级软件流水。** 每个 WMMA 间隔最多 2 VALU + 2 exp，或 2 DS + nop；本 tile 的 softmax 放在上一个 tile 的 dV/dK WMMA 下面；用 pk_fma 算 S·scale−LSE；消掉回边的 64 条 v_mov_b64。
   - 预期：VALU 成团的超出部分 292 + 85 → 约 0，**−250 到 −380 cyc/单元**。
   - 注意：已关闭的"issue roof / sched_barrier"两轴（g60、sched_*）是按指令数砍或按段落 clamp；ASM 的证据说明关键在**分布**，而不是条数。实现时应在 FlyDSL 源码里就用跨步流水把依赖拆开，不要指望 sched_barrier 事后去修。
3. **DS 预取距离至少 1 个相位，dscnt 从不等到 0。** 每条 load 在它被消费前 16 条 WMMA 发出，每间隔 2 条。和第 1 条一起改。
   - 预期：fly 的 DS 等待 + 含 DS 间隔 686 → 约 100，**另外 −150 到 −250**（和第 1 条有重叠）。
4. **（依赖第 1 条）Q/dO/LSE/δ 改用 TDM 直接进 LDS，提前 1–2 步。** 第 1 条做完后 S 的 A 操作数取自 LDS，暂存不再是"双用途"，h49 当年否掉 TDM 的前提就消失了。
   - 删掉每轮 32 条 buffer_load、32 条 ds_store 和 `s_wait_loadcnt 0x23`（`FKcsv:1600`，134 cyc/轮）。
   - 预期：vmem 的 435 → TDM 等待约 125–227，**−200 到 −300 cyc/轮**。
   - 风险：TDM 有两条挂卡路径（gfx1250-card-safety），必须单独、在 toy 上先验，再和其他改动合并。
5. **4-wave BLOCK_KV=128 融合 dQ：dS 经 LDS 交换 + 按 d 切片 + 推迟一步、1 条/间隔的 SCOPE_DEV fp32 atomic + split barrier（等待放在 signal 之前，signal→wait 间隔 3–6 条 WMMA）。**
   - 收益：去掉 k_dqg（892 ATT / 973 PMC 每单元，4.00e6 cyc/SIMD）。
   - 代价：+128（16 条 WMMA）+ atomic 208–634 + barrier 154–261。
   - 预期：**−100 到 −400 cyc/单元**。取决于全卡负载下的 atomic 背压，ASM 自己也在这上面花 27%。
   - 先决条件：`fx.lane_id()` 加独立的 wave 变量；barrier 按 §6 的放法（h51 说 barrier 占 45%，ASM 只花 13%）；atomic 用 `UniversalAtomicAdd(SyncScope.Agent)`。放最后做。

1+2+3+4 只改 k_dkdv，都在 1-wave 形态里做，预计把 k_dkdv 从 1714 压到约 900–1100 cyc/轮，全 bwd 的每单元 cycle 从 2606 降到约 1800–2000，大致和 ASM 的 1581–1853 打平。第 5 条是超过 ASM 的唯一结构性手段。

## 13. 注意事项和待解问题

- ASM 各 wave 的步长差别很大：从 1149 到 2141 cyc/步，前面的 WG 慢、后面的快（`se0_sm3_sl0_wv*.json`）。v_nop/SALU 的时间不变，所以不是时钟变化，而是 atomic/TDM/barrier 随全卡负载上升。fly 做 A/B 时要用全负载的 prod shape，不能用快的 wave 当基准。
- 这个 build 共 400 条 WMMA、192 条 atomic、28 条 TDM，AITER-5GEMM-STUDY 里研究的是 864/514/40 那一版，**不是同一个 binary**，但每步的结构一致。
- 依赖链 8 还是 16 cyc 的冲突（§7）需要一个 GPU 微基准来定。它决定 fly 里同一累加器的 4 链能不能照 ASM 那样背靠背连发。