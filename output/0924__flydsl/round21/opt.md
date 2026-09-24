# Round 21 — gfx1250 FlyDSL attention backward (fast round)

写作方式：边做边写。以下按时间顺序记录，包括失败的构建与被语料库“免费杀掉”的候选。

## 0. 起点

- `op/current` = round 20 的 merge `M6162`，`kernels.py` md5 `51c4bf6077e191b3da3697a92e464b79`，
  与 `rounds/021/op` 逐字节相同。
- round 20 记录：prod 511.42 TF/s vs beat 715.94，ratio 0.7144，score 0.8241。
  **beat 与噪声底本轮重新测量**（`h7` / 噪声底不得跨轮携带）。
- `bound: latency`，MEDIUM。唯一活着的正向机制是 `r7.i1.g21` 的跨迭代 prefetch
  （~+21%），round 20 由 `r20.i2.g62` 把它在 `k_dkdv` 上加深到 depth 2，得到 +1.65%/+1.53%。
  六次“删除式”尝试全部变慢（`g47` `g49` `g56` `g60` `P1` `P3`）。

## 1. 读 findings + round 17 深剖

读毕 `facts.md`(401行) / `dead_ends.md`(332行) / `pool.md`(91行) / `route.md`(1332行)
与 `rounds/017/1-profiling`。本轮最吃劲的几条：

- `h3` **spill 即死**：`private_segment_fixed_size > 0` 或任何 `scratch_` op → 首次 launch 后挂死，
  需人工掉电。`spill 0` 是硬门。
- `h4` **rocprofv3 在本 op 上是死路**（`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`，PC sampling 绝对禁止）。
  因此本轮的 survey **不是** `rocprofv3 --stats`，而是唯二可用的工具：
  **离线静态筛（COMPILE_ONLY ISA dump）** + **带时钟见证的片上 A/B**。这不是偷懒，是 `h4` 明文要求。
- `h7` **只有 prod 能排名候选**；fast/proxy 是哨兵（fast 同代码扩散已测到 6.4–6.5%）。
- 占用率算术：`regs_per_multiprocessor = 131072` → 1 wave/SIMD 的 VGPR 上限 **1024**。
  两个主 kernel 都是 1 wave/SIMD。

## 2. Look —— survey（零卡时间，静态筛）

预期（先说后测）：round 20 已把 `k_dkdv` 的 `s_wait_loadcnt` 从 9 个降到 2 个且都在体首，
所以我**预期** `k_dkdv` 的 loadcnt 侧已经基本盖住，剩下的等待应该在 LDS（dscnt）一侧；
而 `k_dq` 二十轮没被碰过，**预期**它还带着未被覆盖的 loadcnt。

工具：`_scratch/scr/screen.sh`（round 2 `screen.py` 的本轮副本，`COMPILE_ONLY=1 ARCH=gfx1250
FLYDSL_GPU_ARCH=gfx1250` 两个 arch 变量都要设，只设一个会静默产出混合目标 ISA）+ 自写
`scr/blocks.py`（逐 basic-block ISA 普查）与 `scr/pos.py`（块内位置图：每个 `s_wait_*` 在体内的百分比位置）。

### 2.1 热体识别

`k_dq` 的 `.LBB0_2` 与 `.LBB0_6` 是同一循环的两个版本，用 `v_cmp_gt_i32` 计数区分
（`.LBB0_2` = 0，`.LBB0_6` = 56）→ `.LBB0_2` 是无 mask 体，占 prod 迭代的 98.4%。
`k_dkdv` 的热体是 `.LBB0_8`。

### 2.2 结果（incumbent `cur_a`）

`k_dkdv .LBB0_8`（post-`g62`）：778 行 / 64 wmma / 32 `buffer_load_b128` /
40 `ds_store_b128` / 40 `ds_load_tr16` / **2 `s_wait_loadcnt`（`0x21` @0%，`0x0` @1%）** /
3 `s_wait_dscnt` @78–81% / VGPR 904 / spill 0。
→ 预期被证实：loadcnt 侧已收干净，**只剩 LDS 一侧的三个等待**。

`k_dq .LBB0_2`：742 行 / 96 wmma / 32 `buffer_load_b128` / 16 `ds_store_b128` /
16 `ds_load_tr16` / **4 `s_wait_loadcnt`，其中三个很晚：`0xd` @77%、`0x8` @78%、
`0x0` 全排空 @93%** / 8 个 **分级** `s_wait_dscnt`（`0xe…0x0`）@77–92% / VGPR 960 / spill 0。

### 2.3 这一条直接决定了 route row 10 的前置条件

`pool.md` 里 `r21.i1.g63` 自带的反方论据是：*“round 19 的普查说 `k_dq` 热体零停顿……
先看 ISA 里 `k_dq` 的 `s_wait_loadcnt`，**如果已经接近 0，就不要构建它**。”*

**ISA 说不是 0。** 体内 93% 处的一个 `s_wait_loadcnt 0x0` 全排空，强制本迭代 7–13% 处发出的
load 必须在**同一次迭代结束前**落地 —— 也就是说 `r9.i2.g26` 的 depth-1 prefetch
**实际上根本没有跨到下一次迭代**，它的覆盖被截断在一次迭代之内。
`k_dkdv` 在 `g62` 之前是完全相同的形状（9 个站点），之后 2 个且都在体首，值 +1.65%/+1.53% prod。

**前置条件满足，`g63` 活着。** 这是零卡时间拿到的判决。

## 3. 候选来源

### (a) `pool.md` / `route.md`
- `r21.i1.g63` OPEN → **取用，保留原 id**（arm A）。
- `r21.i2.g64` OPEN → **本轮放过**，理由见 §6 与 `route.md`。
- `r16.i3.g50` DEAD（round 20 片上处决）、`r18.i5.g58` CLOSED ×2 → 不动。
- `r3.i6.g15` 已连续第七轮挂账。

### (b) survey
`k_dkdv` 现在只剩 LDS 侧三个 `s_wait_dscnt`；`k_dq` 同时有晚到的 loadcnt 全排空
**和** 八个 dscnt。→ 两条线索，都指向 `k_dq`。

### (c) 语料库（`explored.consulted` 全量列在 YAML 里）

`optimization/routes/1-metrics-to-techniques.md`：Table A 的 A1（spill 使一切结论作废）、
A7（regime/data-swap 判别）。**承重的一行是 Table B 的**：
*“LDS latency exposure —— 是 `ds_read` 上的延迟暴露，不是端口饱和。加宽 write-to-read 距离”*，
但它明确写着只有在 *“由减法探针而非计数器结案”* 之后才可读。

→ **round 20 的 P1/P3 正是那个减法探针**：删掉整条 Q/dO 的 LDS 往返（P1）测得 −6.98%，
再删掉 P/dS 往返（P3，80 个 LDS op 全没了、WMMA 数不变）测得 −8.19%。
**删掉 LDS 流量让 kernel 变慢 → 端口没有饱和 → 这一行的“延迟”读法被解锁。**
这是本轮 arm B 的语料库授权，且它**不是** double-buffering。

`optimization/techniques/3-pipelining-and-scheduling.md`：第 303 行的有序旋钮表
`1 waitcnt distance → 2 barrier count/placement → 3 s_setprio → 4 sched_group_barrier
→ 5 prefetch/staging depth`；第 200 行 “Double-buffer across tiles”；第 150–152 行
buffer distance/phases；第 236 行 paired-wave staggering 在 flydsl hd64 上**输**、在 hipkittens 上**赢**。

### (d) 同一 op 的其它后端
- `backends/aiter/attention/recipes/fmha_v3_bwd_hd128_bf16.md` §6：b1 的 dQ atomic 宽度值 24.6%/23.9%；
  b2 确定性（dK/dV bitwise，dQ 不是，`deterministic=True` 代价 1.61x）。
  **本 job 不可用**：atomics 违反本 job 的确定性输出契约。
- `backends/hipkittens/attention/recipes/gqa_d128.md` §6：mech1 手工分配寄存器 **1.72x**
  （判别式 `v_accvgpr_read/write` 1638→128，`SQ_WAIT_ANY` 8.6x）；mech2 wave 数 1.31x 偏向 8 warps；
  mech3 `RESCALE_THRESHOLD` 5.3%。
  **mech1 不能字面移植**：那是 gfx950 的 AGPR 事实（`HIPCC 不会把 AGPR 当作 MFMA 输入`），
  而 gfx1250 是**扁平 VGPR 文件、无 AGPR**（`r10.i3.g30`）。但它的 *why*（“等待来自寄存器压力驱动的拷贝”）
  正好是 A63 静态筛里 `s_set_vgpr_msb` 142→305、`v_mov_b64` 65→129 的形状 —— 记在这里，作为 §5 风险项的来源。
  **mech2 在本 job 上从四个方向关闭**（`g04` `g14` `g55` `g60`）。

## 4. 被语料库免费杀掉的候选（survey 提出，未构建）

survey 直接建议的 arm B 是 **`k_dkdv` 的 LDS double-buffering**：`g62` 清干净 loadcnt 之后，
`.LBB0_8` 体内只剩三个 `s_wait_dscnt` @78–81%，而 route 表说修法是“加宽 write-to-read 距离”，
且 `r15.i2.g47` 已确立 *“`ds_store -> ds_load -> WMMA` 尾巴是一条真实的 LDS RAW”* ——
看起来只有第二个 buffer 能打断它。

**`knowledge/backends/flydsl/attention/dead-ends.md` §“Double-buffering the streamed K/V through LDS”
把它杀了**：Q-outer 体 **−2.3% ~ −4.7%**，KV-outer 体 **−16.5%**；`k_dkdv` 正是 KV-outer。
根因记作“prefetch 的 LDS 写与 compute 的 LDS 读争用”，且其中一例付出 424 B scratch ——
**scratch 在本 job 上是 `h3` 的即死条款**。叠加本 job 自己在同一轴上的 `g33`/`g39`/`g47`，期望值明确为负。

**不构建。** 两个语料库文件，零卡时间，省下一个 arm。这是本轮语料库赚回工时的地方。

## 5. Arm A —— `r21.i1.g63`（id 沿用 pool，不重新编号）

`k_dq` 的跨迭代 prefetch 由 depth 1 加深到 depth 2，与 `r20.i2.g62` 在 `k_dkdv` 上做的是同一处变换。

补丁 `scr/patch63.py`，四处断言式替换：
1. 非 PARTIAL `kvloop_full`：`jj = ii+1` → `kk = ii + 2`，`kk = (kk < n).select(kk, n-1)`；
   `_NP = NKT*NDT*4`；body 以 `st[NQ*NDO : NQ*NDO+_NP]` 作 `pre0`；
   `final = yield (list(_r[:NQ*NDO]) + list(st[NQ*NDO+_NP:]) + list(_r[NQ*NDO:]))`。
2. PARTIAL `kvloop_full`：同上，clamp 保持相对，`it0` 在 clamp 之后加。
3. PARTIAL 前导：`_pf2 = (_off+1 < nkvt_eff).select(_off+1, nkvt_eff-1)`，追加第二次 `_ldkv`。
4. 非 PARTIAL 前导：`_j1 = (nfull > 1).select(1, 0)` —— **刻意不写 `n-1`**，因为 `nfull == 0` 时
   `n-1` 会是 −1，那是一次活的 OOB 读；`k_dq` 的描述符带 1 GiB 平坦 `num_records`，
   **拿不到 `k_dkdv` 那种来自 buffer extent 的免费钳位**（`r9.i2.g26` 的注释）。追加第二次 `_ldkv`。

### 5.1 离线静态筛结果（零卡时间）

| `k_dq .LBB0_2` | incumbent `cur_a` | **A63** |
|---|---|---|
| VGPR / spill / scratch | 960 / 0 / 0 | **992 / 0 / 0** |
| 行数 | 742 | **1060 (+43%)** |
| `v_wmma` | 96 | 96 |
| `buffer_load_b128` | 32 | 32 |
| `ds_store_b128` / `ds_load_tr16` | 16 / 16 | 16 / 16 |
| **`s_wait_loadcnt`** | **4**（@0%，`0xd`@77%，`0x8`@78%，**`0x0`@93%**） | **2**（`0x2`@0%，`0x8`@8%） |
| `s_wait_dscnt` | 8，分级 `0xe…0x0` @77–92% | **8，全部 `0x0`** @88–95% |
| `v_nop` | 28 | 41 |
| `s_set_vgpr_msb` | 142 | 305 |
| `v_mov_b64` | 65 | 129 |
| `buffer_load` 跨度 | idx 52–100 | idx 52–938 |

**预期的 ISA delta 落地了**：`s_wait_loadcnt` 4→2、都挪到体首、93% 的全排空消失。

**VGPR 恐惧被证伪，但换来一个新风险。** route row 10 与 `h12` 预测约 120 dword 余量对
约 128 dword 成本、并点名 spill 是最可能的死法。实测 **960→992，spill 0，scratch 0**，
在 1024 的 1-wave/SIMD 上限下还剩 32 VGPR。门过了。

**但必须记下三个非预期的 delta**（`h18`/`h16`：ISA delta 是必要不充分条件，静态计数不排名，卡说了算）：
热体 **+43% 行数**、`s_set_vgpr_msb` 142→305、`v_mov_b64` 65→129、
32 个 `buffer_load_b128` 的发射跨度从 idx 52–100 摊到 52–938、
以及七个**分级** `s_wait_dscnt` 塌缩成**八个全排空 `0x0`** @88–95%。
这与“寄存器压力把 LDS 路径串行化了”的故事一致，也正是 (d) 里 hipkittens mech1 的那个 *why*。
**这一项由卡裁决，不由我裁决。**

## 6. Arm B —— `r21.i3.g65`（新 id，从 g64 之后计起）

### 6.1 为什么不是 `r21.i2.g64`
见 §7 与 `route.md` 的一行理由。

### 6.2 机制

把 `k_dq._body` 里的 16 个 `ds_store_b128`（K 的 LDS 镜像）**整体提到 kt 循环之前，成组发射**，
其余一行不动。

现状（`kernels.py` 816–827）：store 嵌在 `kt`/`dt` 双层循环里，与 64 个 S/P WMMA 交错；
最后一个 store 落在 `kt=1,dt=3`，而第一个 `ds_load_tr16_b128`（864–872 行，喂最后 32 个 dQ WMMA）
紧跟在 kt 循环之后。**write→read 距离≈0**，而 tr16 读的是跨全部 32 行的列条，
依赖**全部** 16 个 store —— 这就是体内 8 个 `s_wait_dscnt` 的来源。

提前之后，整个 64-WMMA 的 S/P 突发（约 600 条指令）落在 write 与 read 之间。

**为什么这是免费的**：`kp` 全部来自 `pre`，即上一轮的 carried prefetch ——
**32 个 K/V dword 在 body 入口处按构造已经全部活着**。提前 store 不延长任何值的活跃区间，
不新增寄存器，不改任何算术、地址或顺序 → **逐位相同**。

**语料库授权**：`1-metrics-to-techniques.md` Table B 的 “LDS latency exposure → 加宽
write-to-read 距离”，其可读性前提（“由减法探针结案”）已由 round 20 的 P1/P3 满足（§3c）。
旋钮序上这是 **#1 waitcnt distance**，而 `g62` 上一轮跳到了 **#5**。
它**不是** double-buffering（单 buffer，同一迭代，同一份数据），因此不触 §4 那堵墙。

### 6.3 与 arm A 的独立性
A63 改的是 `kvloop_full` 与两处前导；B65 改的是 `_body` 内的 store 循环。
**行不重叠，可分别归因，可合并。** 两者都落在 `k_dq` —— 这是自觉的选择：
`k_dq` 占 prod 约 37%，二十轮没有一个 arm 落在它上面，而 `k_dkdv` 这一侧的轴要么死要么已经做完。
风险是“如果 `k_dq` 不是问题，两臂同亏”，明写在这里。


### 6.4 `r21.i3.g65` —— 离线筛直接处决，零卡时间

按 §6.2 构建 `arms/B65`（16 个 `ds_store_b128` 整体提到 kt 循环之前）。

**静态筛：`k_dq` 的 ISA 与 incumbent 逐指令相同，只差寄存器编号。**

| `k_dq .LBB0_2` | `cur_a` | `B65` |
|---|---|---|
| 行数/wmma/bufld/dsst/dstr | 742/96/32/16/16 | **742/96/32/16/16** |
| `s_wait_loadcnt` / `s_wait_dscnt` | 4 / 8 | **4 / 8** |
| VGPR / spill | 960 / 0 | **960 / 0** |
| `ds_store_b128` 位置 | idx 4,5,7,8,9,10,16..19,103..108 | **完全相同** |
| `ds_load_tr16` 位置 | idx 110..118, 241..248 | idx 110..119, 242..249 |

把 `.LBB0_2` 的助记符流做寄存器重命名归一化后 diff：**只剩 808 行差异，全是 `v%d`→`v%d` 的编号位移**，
`k_dkdv` 与 `k_delta_bshd` 的 ISA 则**逐字节相同**。

**结论有两层，都记下来：**
1. **后端本来就在做这件事。** 源码里 store 嵌在 kt/dt 循环内，ISA 里它们已经被聚到 idx 4–108，
   tr16 读在 110–249，96 个 WMMA 在其后。这是 `r16.i1.g48` 的教训第二次出现：**源序不是发射序。**
2. **我给 g65 写的前提是错的。** 我以为 write→read 距离≈0；ISA 说 tr16 读（110–249）到它们的
   消费者（体尾 77–92% 的 dQ WMMA）已经有 **300–500 条指令**的距离。
   `k_dq` 的 LDS 路径本来就调度得很好，那 8 个 `s_wait_dscnt` 不是问题所在。

**不上卡。** 两次编译、零卡时间，一个候选结案。`pool.md` 记为 CLOSED。

## 7. `r21.i2.g64` —— 本轮放过（`k_dkdv` 形态），改为 `r21.i4.g66`（`k_dq` 形态）

放过 `k_dkdv` 形态的理由（同一行写进 `route.md`）：
**它的家族在本 job 上三战三败、零胜**——`r15.i2.g47`（`sched_group_barrier` 打在同一个体上）
**−3.12%**、`r16.i2.g49`（prefetch 组后硬栅栏）**−16.25%**、`r17.i2.g53`（弱化栅栏）**−13.24%**；
而 `g64` 正是 `g47` 的工具打在 `g49` 的目标上。
另外 `g64` 的 `s_setprio` 那一半有一条免费且决定性的反证：
**两个主 kernel 都是 1 wave/SIMD，没有 SIMD 同伴可以降优先级**，`s_setprio` 在这里不可能改变任何事。

**保留的那一半，换靶。** 本轮普查给出一个 `g64` 写就时还不存在的具体靶子：
`cur_a` 的 32 个 `buffer_load_b128` 挤在 idx 51–100 的一个 50 条指令的团里，而体尾 idx 694 有
`s_wait_loadcnt 0x0`；A63 把这个团摊到 idx 52–938 —— 但那是 depth 2 的**副作用**，代价是
+32 VGPR 和 +43% 体积。**`g66` 问更便宜的那个问题：付钱的是「摊开」还是「加深」?**
`k_dq` 从未带过任何调度提示（`g47` 打的是 `k_dkdv`，且靶子不同）。

补丁 `scr/patch66.py`：在 `k_dq._body` 的 WMMA 段之前插入 32 组
`sched_group_barrier(VMEM=0x020, 1) / sched_group_barrier(MFMA=0x008, 3)`（96 WMMA / 32 VMEM = 3:1）。

### 7.1 `B66` 静态筛

| `k_dq .LBB0_2` | `cur_a` | `A63` | **`B66`** |
|---|---|---|---|
| VGPR / spill | 960 / 0 | 992 / 0 | **960 / 0** |
| 行数 | 742 | 1060 | **762 (+2.7%)** |
| `buffer_load` 跨度 | idx 51–100 | idx 52–938 | **idx 37–712** |
| `s_wait_loadcnt` | 4 | 2 | **10** |
| `s_wait_dscnt` | 8 | 8 | **5** |

**提示落地了**（`h18` 要求的必要条件）：摊开幅度与 A63 相当，**寄存器零代价**。
但 `s_wait_loadcnt` 4→10，且体内多出 `0x0` 全排空（idx 605/619/660/746）。
**预测（先说后测）：我预期 B66 会输**，因为摊开把等待也摊进了体内；
但这正是把「摊开」与「加深」分开的唯一办法，且它便宜。

## 8. 上卡 —— 第一步：`validation.py`（`h1`，不得与 benchmark 颠倒）

清 `/root/.flydsl` 与 `__pycache__` 后，经 runner (`docker exec fa-repro`) 分离式启动。
前后 `rocm-smi --showpids` = `No KFD PIDs currently running`（设备空闲）。

**正确性：两臂全 PASS。** 15 个 UT 形状全 PASS，dB 与 incumbent 逐项一致（dq/dk/dv 52.4–53.0 dB，
门限 50.0 dB，bitwise x200 通过）。
特别地 A63 改了 prefetch 地址钳位，`nfull == 0` 的 `_j1` 保护有效 —— 没有 OOB。

**速度（framework 自己的 validation 跑的数，原样记录）：**

| | fast TF/s | proxy TF/s | prod TF/s | geomean vs beat |
|---|---|---|---|---|
| A63 | 47.40 | 359.41 | **410.79** | 0.699x → **FAIL** |
| beat（A63 同跑） | 50.79 | 567.44 | 712.10 | |
| B66 | 45.74 | 342.95 | **396.59** | 0.667x → **FAIL** |
| beat（B66 同跑） | 51.12 | 575.74 | 711.69 | |

beat 重测 712.10 / 711.69 对 round 20 的 715.94 = **−0.5%**，机器状态一致，sclk 1051–1057。
round 20 记录的 incumbent prod 是 **511.42**。照此 A63 −19.7%、B66 −22.4%。

**但这还不是判决。** 两臂同时掉约 20% 是一个可疑的共同模式，而 round 20 的 511.42 是**跨轮携带的数**，
按 `h7`/噪声底规则不可直接用来排名。下一步的受控 A/B（`cur_a`/`cur_b` 两份物理独立重建的
incumbent + `beat`，同一 session、palindromic、51 iters、逐 shape 一个进程）才是判决。

## 9. 受控 A/B —— 判决

一次失败的启动，**零卡时间损失**：`benchmark.py` 的 `--arms` 不接受由 `--arm-path` 注册的名字
（`--arm-path` 本身即注册为 arm，`--arms` 只用来追加 `op/` 下的内置 arm）。参数错误在 argparse
阶段返回，没有上卡。修正为 `--arms beat`。

协议：同一 session、同一进程外壳、每个 shape 一个进程、51 iters、median、benchmark 内部
palindromic 序、`cur_a`/`cur_b` 两份**物理独立重建**的 incumbent 作同 session 噪声底、
`beat` 本轮重测、前后 `rocm-smi --showpids` / `--showuse` 见证、每 shape 后取 sclk。

前后见证均为 `No KFD PIDs currently running` / `GPU use (%): 0`，sclk 全程 1100（VR 限频，已知）。
dmesg 无 `amdgpu` 故障行。

### 9.1 prod（唯一能排名的形状，`h7`）

| arm | latency ms | TF/s | vs incumbent |
|---|---|---|---|
| `cur_a` (incumbent) | 10.7766 | **510.20** | — |
| `cur_b` (incumbent, 独立重建) | 10.7974 | **509.22** | — |
| `A63` (`r21.i1.g63`) | 13.3709 | **411.21** | **−19.33%** |
| `B66` (`r21.i4.g66`) | 13.8168 | **397.94** | **−21.93%** |
| `beat` (重测) | 7.7063 | **713.47** | |

**本 session 噪声底 = |510.20 − 509.22| / 509.71 = 0.19%。**
两臂的亏损是噪声底的 **100 倍以上**。没有任何解释空间。

incumbent ratio = 509.71 / 713.47 = **0.7144** —— 与 round 20 记录的 0.7144 **完全一致**。
机器状态、beat、incumbent 三者本轮全部复现，**所以这两个负号是候选的，不是机器的**。

### 9.2 proxy / fast（哨兵）

| arm | proxy TF/s | vs inc | fast TF/s | vs inc |
|---|---|---|---|---|
| `cur_a` | 433.84 | — | 55.59 | — |
| `cur_b` | 429.90 | — | 50.79 | — |
| `A63` | 363.02 | −15.94% | 50.49 | −5.1% |
| `B66` | 347.67 | −19.50% | 49.89 | −6.2% |
| `beat` | 578.70 | | 51.28 | |

proxy 噪声底 0.91%，两臂同号同量级，与 prod 一致。
**fast 噪声底本轮实测 = 8.6%**（`cur_a` 55.59 vs `cur_b` 50.79），两臂的 −5~−6% 完全落在里面 ——
`h7` 的“fast 是哨兵不是裁判”本轮再次自证。

### 9.3 决定

**两臂都输 → 不合并，不出货。** `rounds/021/op` 与 `op/current` 逐字节相同
（`kernels.py` md5 `51c4bf6077e191b3da3697a92e464b79`）。本轮 ship nothing。

## 10. 本轮买到的东西（负结果，但是本 job 二十一轮里最硬的一条）

**`k_dq` 与 `k_dkdv` 在同一条杠杆上符号相反。**

`r20.i2.g62` 把 prefetch depth 1→2 打在 `k_dkdv` 上，**+1.65% / +1.53%**。
`r21.i1.g63` 把**同一处变换**打在 `k_dq` 上，**−19.33%**。
这不是“机制不够强”，是**反向**。

而 `r21.i4.g66` 把这件事钉死了。两臂唯一的共同 ISA 后果是
**32 个 `buffer_load_b128` 从 idx 51–100 的一个 50 条指令紧团被摊开**
（A63 → 52–938，B66 → 37–712）。两者的代价结构完全不同：

| | VGPR | 体积 | 摊开 | prod |
|---|---|---|---|---|
| `A63` | 960→**992** | 742→**1060 (+43%)** | 是 | −19.33% |
| `B66` | 960→**960 (不变)** | 742→**762 (+2.7%)** | 是 | −21.93% |

**B66 是零寄存器代价、几乎零体积代价的对照臂，它输得更多。**
所以亏损**不是**寄存器压力、不是体积、不是 spill（两臂 spill 均为 0）。
**是「摊开」本身。`k_dq` 的那个紧团是承重的。**

机制上讲得通：`k_dq` 的体里有 96 个 WMMA（`k_dkdv` 只有 64）、只有 16+16 个 LDS op
（`k_dkdv` 是 40+40）。它是**计算密度高、访存轻**的那一半。把 32 条 load 背靠背打出去，
它们在一个很短的窗口里全部进入未完成队列；把它们插进 WMMA 流里，发射被 WMMA 串行化，
等待也跟着被摊进体内（B66 的 `s_wait_loadcnt` 4→10，体内多出四个 `0x0` 全排空）。

**这条对下一轮是可执行的**：`k_dq` 想要的是**相反方向** —— 把它的 prefetch 团**收得更紧**
或**缩短**，而不是加深、摊开。这是二十一轮里第一次有一个方向性的、带符号的、经对照臂隔离的
`k_dq` 假设。记进 `pool.md`。

同时，`h18` 又赢了一次：A63 的 ISA delta **完全按预期落地**（`s_wait_loadcnt` 4→2，93% 的全排空消失），
静态筛的每一个门都过了（spill 0、VGPR 992 < 1024、正确性 52+ dB），**然后在卡上输了 19%。**
ISA delta 是必要条件，不是充分条件；静态计数不排名。

### 10.1 本轮记录在案的失败与错误

1. `screen.sh` 经 `launch.sh` 启动时**参数被吞**（`launch.sh` 只透传脚本路径，不透传 argv），
   `screen.py` 报 `--tag: expected one argument`，rc=0 但没产出。改为每个 tag 一个一行的包装脚本。
2. `benchmark.py --arms cur_a,...` 参数错误（见 §9），argparse 阶段失败，零卡时间损失。
3. `r21.i3.g65` 的**前提是我写错的**（见 §6.4）：我以为 `k_dq` 的 LDS write→read 距离≈0，
   ISA 说已经有 300–500 条指令。离线筛在上卡前抓住了它。

## 11. 收尾

- **出货:无。** `rounds/021/op/kernels.py` md5 `51c4bf6077e191b3da3697a92e464b79`,
  与 `job_context/op/current/kernels.py` 逐字节相同。`op/current` 全程未被触碰。
- **`raw/`** 收了实际被解析的东西:三份 `screen_*.json`、五份 `21_final_isa.s`
  （`cur_a`/`A63`/`B65`/`B66` 的 `k_dq` + `cur_a` 的 `k_dkdv`）、`isa_census.txt`
  （逐 basic-block 普查 + `.LBB0_2` 的逐条等待位置图）、`validation_out.txt`、
  `meas_m1_out.txt`、三份 `rows_*.json`,以及全部三个补丁与全部驱动脚本。
- **设备:** 收尾时 `rocm-smi --showpids` = `No KFD PIDs currently running`,
  容器内无残留 `benchmark.py`/`validation.py`/`screen.py` 进程,dmesg 无 `amdgpu` 故障行。
- **`findings/pool.md`** 已改写:`g63` DEAD、`g64` PASSED OVER、`g65` CLOSED(离线筛)、
  `g66` DEAD(承重负结果),并留下一条**未编号的 OPEN**（下一个可用 id 为 `g67`）。
- **`findings/route.md`** 的 `## Route` 表已改写为 4 行(2 `hint · must` 在上,2 `idea` 在下,
  `outcome` 留空),并写入 `g64` 的一行放过理由与 `r3.i6.g15` 第八轮挂账的建议。

## 12. 终测 —— 出货件 + 同 session 冠军重测

**先把一件事摆正:`rounds/021/op` 现在装的是 `r21.i1.g63`,一个输了的 arm,并且它会留在那里。**
revert 会把本轮变成与空轮不可区分(同 hash、`files_changed: []`、`gain` 恰好 1.0000),
而本轮学到的唯一那件事就活在这份 diff 里。`op/current` 全程未被触碰。

清 `/root/.flydsl` 与四棵树的 `__pycache__` 后重建（避免后端端出上一轮的二进制），
`rounds/021/op` 重新走一遍两道门:**UT 15/15 PASS**、`validation.py` correctness **pass**
(52.52–52.84 dB,门 50.0)、determinism ×200 **pass**、speed FAIL(exit 2)。

同一 session、同一块空闲卡、背靠背重测 incumbent 与每个**逐 shape 冠军**
(`rounds/020/op` = incumbent、`rounds/019/op`、`rounds/017/op`),51 iters 中位数,
每 shape 一个进程,前后 `--showpids`/`--showuse` 均空闲,sclk 全程 1100。

| shape | **r021 (g63)** | r020 (incumbent) | r019 | r017 | `beat` | 冠军 | vs 冠军 |
|---|---|---|---|---|---|---|---|
| **prod** | **412.10** | **509.15** | 500.39 | 498.78 | **713.09** | r020 509.15 | **0.8094** |
| **proxy** | **365.71** | 429.00 | **438.99** | 436.67 | 576.05 | r019 438.99 | **0.8331** |
| **fast** | **53.30** | 48.50 | **54.73** | 50.58 | 50.43 | r019 54.73 | **0.9738** |

margin = 1.00 ⇒ target = `beat`。ratio = min(this/target, 1.0):
prod **0.5779**、proxy **0.6349**、fast **1.0000** ⇒ **score = 0.7376**。

**验收算术,照直算:** 吞吐没有改进 best-ever(prod 412.10 < 509.15),
prod 只有自身 best-ever 的 **80.9%**、proxy **83.3%**,两者都低于 95% 门。**本轮不被接受。**

两处值得记下的细节,因为它们会被误读:
- **fast 的 ratio 是 1.0000,不要当成好消息。** r021 的 fast 53.30 确实高过 `beat` 50.43,
  但 fast 的同码地板本轮实测 **8.6%**,而 r020 这一跑只有 48.50(它自己上一跑是 55.59)。
  `h7` 说得很清楚:fast 是哨兵不是裁判。**prod 才是判决,prod 说 −19%。**
- **逐 shape 冠军不是同一轮。** prod 的冠军是 r020,proxy 与 fast 的冠军都是 **r019**。
  这本身就是「冠军按轮追踪而不是按数字追踪」这条规则在本 job 上的具体形态 ——
  r020 的 `M6162` 在 prod 上赢了 r019,在 proxy/fast 上并没有。

## 13. 本轮的结论,一句话

`k_dq` 与 `k_dkdv` 在 prefetch 这条本 op 唯一活着的杠杆上**符号相反**,
而判别式是那 32 条 load 是否保持为一个紧团 —— 这是二十一轮来第一个关于 `k_dq` 的、
带符号的、由零寄存器代价对照臂隔离出来的事实。代价是一个 −19.33% 的出货件和一轮不被接受。
下一轮的第一步不是造 arm,是那个本轮该先做而没做的减法探针。
