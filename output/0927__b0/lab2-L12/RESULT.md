# L12 4-wave 合法化 — GPU 2 / fa-g2，2026-09-27 13:45–14:57

**结论：win（prod）。** 4-wave 可以合法构建，关键不在“每 SIMD 1 个 wave”，而在“WG 减半 + 每个 CU 放 2 个 WG”：

- 基于 r4 冠军的 `w4r2_occ2`：prod 1.0842 / 1.0899 / 1.0856，**均值 1.0865**，与 r4 输出逐位相同。
- 冠军在本 lab 进行中于 14:19:31 换成 r6（投机 stale-max softmax）。同样的改动移植到 r6 上得到 `r6_occ2_lo`：
  prod 1.0665 / 1.0595 / 1.0595，**均值 1.0618**，与 r6 输出逐位相同。同一时段 A/A（`r6_ctrl`）为 1.0012。
- **注意 sentinel**：`r6_occ2_lo` 在 fast 上 +20~25%，在 **proxy 上 -7%**（0.932 / 0.931 / 0.928，A/A 0.995）。
  规则 7 规定按 prod 排名，所以判 win；但 proxy 的回退必须随补丁一起交给 operator。
- 字面意义的“每 SIMD 1 wave、每 wave 算 2 倍 M 行”（`w4r4`，758 VGPR）是 **loss，-12.7%**。retune 后仍全部是 loss（-9% 到 -21%）。

dmesg：129 个卡上进程，新增 dmesg 行 0，所有进程 rc=0。

---

## 1. 阻塞点在哪里，8-wave 假设实际控制了什么

五处 raise，都在 `flydsl_fwd/fmha_b16_buffer_managers.py`（冠军原文行号）：

| 行 | 类 | raise | 背后真正依赖 8 wave 的东西 |
|---|---|---|---|
| :1015-1016 | `QManager16bV2` | `V2 TDM loader assumes 8 waves` | **无**。Q 用每 wave 自己的 TDM（`num_warps=1`），各 wave 的 LDS 区域是 `warp_idx*rows_per_warp*row_bytes`，与 wave 数无关 |
| :1166-1167 | `KManager16bV2` | 同上 | **有**：`_tdm_load_views`（:978）把 TDM 原子写死成 `num_warps=_DEFAULT_NUM_WAVES`=8，一个 64 行 K tile 按 8 行/wave 切分。若只有 4 个 wave，只会加载 0-31 行，32-63 行是上一轮的旧 LDS，结果静默出错。raise 防的就是这个 |
| :1252-1253 | `VManager16bV2` | 同上 | 同 K（V tile 用同一个 `_tdm_load_views`） |
| :1588-1589 | `OManager16bV2` | 同上 | 冠军不用它（`O_VARIANT="v3"`） |
| :1748-1749 | `OManager16bV3` | `V3 assumes 8 waves` | **无**。每 wave 写自己的 `warp_region`，行号用 `warp_base = block_x*block_m + warp_idx*rows_per_warp`，与 wave 数无关 |

内核文件 `fmha_fwd_prefill_a16w16_m32x8.py` 里其余受 8-wave 影响的地方：
- `NUM_WAVES=8`（:112）→ `BLOCK_SIZE`、`BLOCK_M = 16*R*NUM_WAVES`、grid_x、`known_block_size`。这些都由常量推导，改常量即可。
- LO/HI 角色（:1740 `warp_idx // (NUM_WAVES//2)`）：只决定 main_loop 开头“读 K 到寄存器”和“发 prefetch”的先后顺序（:1128-1136）。`_named_barrier_pair` 是空操作，所以**正确性与角色无关**，角色只影响性能。
- 屏障：只有整 WG 的 `gpu.barrier()`（序言 :914 以及每个 tile 的 `_drain_barrier`），不假设 wave 数。
- LDS 环：`_alloc_lds()` 一次性申请整块 320 KB（:689），`MIN_KV_BLK_BYTES=64KB`（:150）把每个 K/V 块垫到 64 KB → 每个 CU 只能放 1 个 WG。这就是“8-wave WG 独占 CU”这一设计所在的位置。
- `_zero_fill_attention`（:1471，只有 THD 路径用）：假设 `BLOCK_SIZE == BLOCK_M`，每个线程写一行 LSE。
- `waves_per_eu=2`（:1871/:1977）：VGPR 上限 512。

**最小合法改动**（`tools/make_arms.py`，每处替换都断言恰好命中一次）：
1. managers：5 处 raise 改成 `num_waves not in (4, 8)`；`_tdm_load_views` 增加 `num_warps` 参数，K/V V2 传入 `self.num_waves`。
   已在 MLIR 中核实：`tdm<rank = 2, warps = 4, ...>`；ISA 中 `s_and_b32 s5, s92, 3`，每个 wave 16 行，LDS 偏移为 `w*0x1100`（K）和 `w*0x1200`（V），valid 为 `valid - 16w`。
2. 内核：`NUM_WAVES`、`WMMA_ROW_PER_WAVE`、`waves_per_eu`、`MIN_KV_BLK_BYTES`、LDS 申请大小、角色划分都做成 arm 参数；
   新增编译期断言 `N_KV_PP*slot_bytes <= 申请量`；把 row-sum 的 `R==2` pk 打包推广到任意偶数 R（逐行结合顺序不变）；zero-fill 的 LSE 改成循环 `BLOCK_M/BLOCK_SIZE` 轮。
3. **控制组 `w8r2_ctrl` / `r6_ctrl`**：走同一条改过的代码路径、取 8 wave，ISA 与冠军**逐字节相同**（`21_llvm_ir.ll` 和 `22_final_isa.s` 的 md5 都相同），说明这些改动在 8-wave 下没有任何副作用。

## 2. Arm 与编译（compile-only，prod 配置；nc_g4 同样 0 spill / 0 scratch）

| arm | 基底 | wave×行/wave | BLOCK_M | LDS 申请 | WG/CU | wpe | 角色 | VGPR | vspill | sspill | scratch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 冠军 r4 | — | 8×32 | 256 | 320K | 1 | 2 | 4LO+4HI | 456 | 0 | 0 | 0 |
| w4r4 | r4 | 4×64 | 256 | 320K | 1 | 1 | 2LO+2HI | **758** | 0 | 0 | 0 |
| w4r4_lo / _hi | r4 | 4×64 | 256 | 320K | 1 | 1 | 全 LO / 全 HI | 758 | 0 | 0 | 0 |
| w4r4_compact | r4 | 4×64 | 256 | 320K（环 136K） | 1 | 1 | split | 758 | 0 | 0 | 0 |
| w4r4_nodefer | r4 | 4×64 | 256 | 320K | 1 | 1 | split，`ENABLE_DEFER_RESCALE=False` | 773 | 0 | 0 | 0 |
| w4r4_r3 | r4 | 4×48 | 192 | 320K | 1 | 1 | split | 598 | 0 | 0 | 0 |
| w4r2 | r4 | 4×32 | 128 | 320K | 1 | 1 | split | 456 | 0 | 0 | 0 |
| **w4r2_occ2** | r4 | 4×32 | 128 | **160K**（环 70K） | **2** | 2 | split | 454 | 0 | 0 | 0 |
| occ2_lo / occ2_hi | r4 | 4×32 | 128 | 160K | 2 | 2 | 全 LO / 全 HI | 454 | 0 | 0 | 0 |
| occ2_kv32 | r4 | 4×32 | 128 | 160K（K/V 垫到 32K） | 2 | 2 | split | 456 | 0 | 0 | 0 |
| w8r1（诊断用） | r4 | 8×16 | 128 | 320K | 1 | 2 | split | 300 | 0 | 3 | 0 |
| 冠军 r6 | — | 8×32 | 256 | 320K | 1 | 2 | split | 448 | 0 | 60 | 0 |
| r6_occ2 | r6 | 4×32 | 128 | 160K | 2 | 2 | split | 446 | 0 | 61 | 0 |
| **r6_occ2_lo** | r6 | 4×32 | 128 | 160K | 2 | 2 | **全 LO** | **446** | **0** | 61 | **0** |
| r6_occ2_lo_kv32 | r6 | 4×32 | 128 | 160K | 2 | 2 | 全 LO | 448 | 0 | 60 | 0 |
| r6_occ2_hi | r6 | 4×32 | 128 | 160K | 2 | 2 | 全 HI | 446 | 0 | 61 | 0 |

sspill 是 SGPR spill 到 VGPR lane，不产生 scratch。r6 冠军自己就有 60，规则的门槛只看 vgpr_spill 和 scratch。
主循环 ISA（`tools/isa_loops.py`）：w4r4 每个 tile WMMA 128、ds_load 64，K/V 的 LDS 读取按 FLOP 摊薄了一半，但卡上仍然是 -13%。这是规则 8“静态指标不能用来排名”的又一个例子。

## 3. 越界证明（CPU，`tools/bounds_proof.py` → `bounds_proof.json`，所有 arm PASS）

用 Python 整数重现内核和 manager 的公式，枚举 12 个 shape（job 的 10 个，再加上 ragged 1000/1037 gqa8 和 333/777 gqa1）× causal/非 causal，覆盖每个 WG、wave、q-tile、lane。断言了以下几点：
packed Q 行恰好是 `[0, grid_x*BLOCK_M)` 的一个划分，且每个有效行只被一个 wave 拥有；
K/V TDM 按 wave 切分后，每个 wave 的 LDS 区落在自己的块内，全局行 `< kv_len`；
Q 每个 wave 的区域和 part2 读取都不超出 slot1；
O V3 每个 wave 的区域和 flush 读取都 `<= slot`；
O 全局行被钳到本 wave 最后一个有效行；
`2*slot <= LDS 申请量`（occ2：环 71680 B，申请 163840 B）。

## 4. 正确性（上卡，一个进程一个 shape，顺序 toy → fast → proxy → prod；toy/fast/proxy 用 AMD_SERIALIZE_KERNEL=3）

`tools/corr_one.py`：cand 的 o/lse 用 NaN 预填，与 job 参考比对（门限 49 dB）；cand 与冠军在 causal 和非 causal 下都做逐位比较；再连续跑 20 次检查确定性。
**所有 arm 全部 PASS**，每个 shape 都与各自的基底冠军逐位相同（causal 和非 causal 都是），确定性 20/20。
prod：o 50.83 dB、lse 89.20 dB；proxy：50.89 / 88.24；fast：51.23 / 85.48；toy（eager 参考）：51.86 / 81.85。
例外只有 `w4r4_nodefer`：它每个 tile 都 rescale，数值本来就会变，o 为 51.00 dB，相对冠军 53.8 dB，仍然 PASS。
证据：`card/<arm>_<shape>.{json,log,dmesg}`。

## 5. 上卡 A/B（prod，n=101，回文顺序，3 个进程分别用 cand,champ / champ,cand / cand,champ，进程里没有 beat；ratio = champ_ms/cand_ms）

驱动沿用 lab2：`measure/run_ab.py` + `measure/runner.sh`（进程内先比较与冠军是否逐位相同）。汇总见 `measure/summary.txt`。

**对照 r4 冠军（14:19:31 之前开始的进程；与 r4 逐位相同，可以作为冠军身份的证明）：**

| arm | p1 | p2 | p3 | mean | 判定 |
|---|---|---|---|---|---|
| **w4r2_occ2**（WG 减半，每 CU 2 个 WG） | 1.0842 | 1.0899 | 1.0856 | **1.0865** | **win** |
| occ2_lo（p1、p2 对 r4） | 1.0870 | 1.0853 | （p3 见下） | 1.0862 | win（2 个进程） |
| w4r4_nodefer | 0.9076 | 0.9072 | 0.9072 | 0.9073 | loss |
| w4r4_lo | 0.8739 | 0.8745 | 0.8731 | 0.8738 | loss |
| w4r4（每 SIMD 1 wave，64 行） | 0.8728 | 0.8721 | 0.8731 | 0.8727 | loss |
| w4r4_compact | 0.8689 | 0.8691 | 0.8687 | 0.8689 | loss |
| w4r4_hi | 0.8327 | 0.8342 | 0.8315 | 0.8328 | loss |
| w4r4_r3 | 0.7914 | 0.7885 | 0.7887 | 0.7895 | loss |
| w4r2（WG 减半，LDS 仍 320K，每 CU 4 wave） | 0.7634 | 0.7636 | 0.7635 | 0.7635 | loss |

**对照 r6 冠军（`base_r6/` 是快照，与 `rounds/006/op` 逐字节相同；runner 在每个进程开始前检查 current 的 md5 = a60bff8e…）：**

| arm | p1 | p2 | p3 | mean | 判定 |
|---|---|---|---|---|---|
| r6_ctrl（A/A，ISA 与 r6 相同） | 1.0030 | 1.0021 | 0.9985 | 1.0012 | 噪声底 |
| **r6_occ2_lo** | 1.0665 | 1.0595 | 1.0595 | **1.0618** | **win**（与 r6 逐位相同） |
| r6_occ2_lo_kv32 | 1.0648 | 1.0652 | 1.0677 | 1.0659 | win；与 occ2_lo 相差 0.4%，在噪声内，视为相同 |
| r6_occ2（split 角色） | 1.0278 | 1.0303 | 1.0256 | 1.0279 | win |
| r6_occ2_hi | 1.0005 | 1.0032 | 1.0021 | 1.0019 | null |

**混杂的测量**（基于 r4 构建，但冠军已经是 r6；与冠军不逐位相同，这恰好能证明当时的冠军是 r6）：
occ2_lo_p3 1.0256、occ2_hi 1.0042/1.0069/0.9971、occ2_kv32 1.0406/1.0382/1.0384、w8r1 0.641/0.643/0.643。
只能当作“r4+L12 相对 r6”的参考，不参与判定。其中 w8r1（8 wave × 16 行，BLOCK_M=128，每 CU 1 个 WG）为 -36%。
这说明收益**不是**来自 grid 变细，而是来自同一个 CU 上两个独立的 4-wave WG：它们只在各自 4 个 wave 内做 barrier，彼此相位错开。

**Sentinel（`r6_occ2_lo` 对 r6，每个进程 1 个 shape，n=101）：**

| shape | 进程 | ratio | A/A |
|---|---|---|---|
| fast | cand,champ / champ,cand | 1.1978 / 1.2551 | — |
| proxy | cand,champ / champ,cand / cand,champ | **0.9320 / 0.9310 / 0.9277** | r6_ctrl 0.9946 |

fast 变快的原因：WG 数翻倍（原来 32 个 WG，放不满 256 个 CU）。proxy 的 -7% 原因未定位：proxy 有 1024 个 WG，对应 512 个 slot；冠军是 512 个 WG，对应 256 个 CU；两者都是 2 轮。
候选解释：K/V 的 L2 流量翻倍（每个 128 行的 WG 都要自己加载完整的 K/V）；或者 2 个 WG 共享 CU 时，LPT 的尾部表现不同。
在 proxy 上修复它，或者按 shape 选择 BLOCK_M，是下一步，不属于本轮。

## 6. 本 lab 的一次测量事故（已排除，写给后来者）

14:27 做原始时间诊断时，A/A 控制组（与冠军 ISA 逐字节相同）在两个顺序下都得到 0.944。排查过程：
- `measure/hostgap_ab.py`：先让 GPU `_sleep` 把 host 开销挡在事件窗口外，差异仍有 4.5%；host 下发耗时两边相同（约 16-20 us）。
- 同一个模块挂两个标签时为 1.00。两份 lab 拷贝互比时也是 1.00。

结论：**job 的 `op/current` 在 14:19:31 被换成了 r6**（kernel 文件 mtime 13:52:49，目录 14:19:31，`rounds/006` 已接受）。
新冠军本身快约 5%，并不是 A/A 出了问题。此后所有 r6 测量都带 md5 守卫（`measure/runner.sh`，`champ_md5.log`）。
教训：**lab 必须对冠军做快照，并在每个进程前校验其 md5**；lab2 的 runner 没有这个守卫。
原始诊断数据在 `measure/raw*.json`、`measure/hg_*.json`，对应日志在 `card/raw*.log`、`card/hg_*.log`。

## 7. 为什么“每 SIMD 1 wave”输了

冠军的 8-wave 设计依靠同一个 SIMD 上的两个 wave 互相掩盖：一个做 softmax 的 VALU/exp，另一个做 WMMA。
每 SIMD 只有 1 个 wave 时，没有其他 wave 来填这些空隙。w4r4 系列无论怎么调角色、LDS 布局或 R，都在 -9% 到 -21% 之间。
最好的是 nodefer（-9%），说明 R=4 时 4 行各自的 ballot 和 rescale 分支代价更高。
如果要走“每 SIMD 1 wave”的路线，必须在 wave 内部做软件流水（L6：QK(i+1) 与 softmax(i) 重叠）。L6 在 8-wave 上输了 15%，但在 1-wave/SIMD 上前提不同。这是多轮的工作，而且属于 A0 的 h21 家族。

## 8. 建议交给 fwd job 的 hint（operator 转交，本 lab 不写 hint.md）

> **h34 (must) -- L12 legal: 4-wave WG x 2 WGs/CU, +6.2% prod on r6 (bitwise), proxy -7% -- land on prod, then fix proxy.**
> Patch `PT/output/0927__b0/lab2-L12/r6_occ2_lo.diff` (tree `lab2-L12/arms/r6_occ2_lo`, on r6 = rounds/006/op):
> managers: 5 `num_waves != 8` raises -> `not in (4, 8)`; `_tdm_load_views(num_warps=...)` fed `self.num_waves`
> by K/V V2 (the real 8-wave assumption: TDM split 8 rows/wave). Kernel: `NUM_WAVES=4`, R=2 (BLOCK_M 128),
> `MIN_KV_BLK_BYTES=0`, `_alloc_lds` 160 KB (2 WGs/CU), `waves_per_eu=2`, ALL waves LO role (split 1.028, all-HI 1.002).
> 446 VGPR / 0 spill / 0 scratch. prod 1.0665/1.0595/1.0595 vs r6 (A/A 1.001); on r4 the same edit was 1.0865.
> fast +20-25%, **proxy 0.932/0.931/0.928** (A/A 0.995) -- the job must decide whether prod-only ranking accepts it,
> or add a shape-dependent BLOCK_M (e.g. 8-wave for grids that already fill 2 WGs/CU at 256 rows).
> Dead: 1 wave/SIMD (4 waves x 64 rows, 758 VGPR) -13%; its retunes -9..-21%; 4-wave WG at 1 WG/CU -24%;
> BLOCK_M=128 with 8 waves -36% (so the gain is WG decoupling, not grid granularity).

## 9. 文件

- 源树：`base/`（r4 快照）、`base_r6/`（r6 快照，md5 在 `base_r6.md5`）、`arms/<arm>/`（每个都有 `L12_ARM.json`）
- 补丁：`w4r2_occ2.diff`（相对 r4）、`r6_occ2_lo.diff`（相对 r6，推荐落地的版本）
- 工具：`tools/make_arms.py`、`tools/bounds_proof.py`、`tools/compile_isa.py`、`tools/run.sh`、`tools/stats.py`、`tools/isa_loops.py`、`tools/corr_one.py`、`tools/seq_corr.sh`、`tools/card.sh`
- ISA：`isa/<arm>/{prod,nc_g4}/*/22_final_isa.s`；`isa/stats.json`
- 越界证明：`bounds_proof.json`
- 正确性：`card/<arm>_{toy,fast,proxy,prod}.{json,log,dmesg}`；sentinel 的结果 json 在 `measure/sent_*.json`，日志在 `card/sent_*.log`
- A/B：`measure/<arm>_p{1,2,3}.{json,log,dmesg}`、`measure/runner{1,2,3,4}.out`、`measure/summary.txt`、`measure/champ_md5.log`
