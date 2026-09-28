# L12 独立复核 — `r6_occ2_lo`（4-wave WG × 每 CU 2 个 WG）— GPU 2 / fa-g2，2026-09-27 15:00–15:10

**结论：确认 win（prod），proxy 回退同样被复现。**

- prod 在 4 个新进程里全部为 win，arm 顺序与原 lab 相反：1.0644 / 1.0613 / 1.0686 / 1.0627，**均值 1.0642**。同进程 A/A 为 1.0000。
- 原 lab 的均值是 1.0618，两者相差 0.2%，在噪声范围内。
- **proxy 0.9322**，与原 lab 的 0.932/0.931/0.928 一致，确认回退 -6.8%。**fast 1.2328**（原 lab 为 +20~25%）。
- 正确性：10 个 job shape 在 job 校验的每种 causal 模式下都 PASS（≥49 dB），与 r6 冠军**逐位相同**（causal 和非 causal 都是）。确定性 20/20。
  对抗大 logit 输入 232 个 case 与 r6 全部逐位相同。
- 编译：6 个 binary 全部 0 vgpr spill、0 scratch。25 个卡上进程 dmesg 新增 0 行，全部 rc=0。

## 1. 被测对象

- cand = `verify/arms/cand`：`arms/r6_occ2_lo` 的快照，已去掉 `__pycache__`，与 `r6_occ2_lo.diff` 一致。
- champ = job 的 `op/current`，只读。每个 perf 进程开始前都校验 kernel 的 md5 `a60bff8e…`，即 r6（`perf/champ_md5.log`）。
- A/A 用的 `verify/arms/champ_copy` 与 `lab2-L12/base_r6` 逐字节相同（`md5.txt`）。
- 我审阅了补丁（`r6_occ2_lo.diff`）：
  - 5 处 raise 改为允许 4 或 8；`_tdm_load_views` 的 `num_warps` 由 K/V 传入 `self.num_waves`。
  - `NUM_WAVES=4`、`MIN_KV_BLK_BYTES=0`、LDS 申请 `min(160K, cap)`，并断言环形缓冲放得下。
  - R 为偶数时行和使用 pk 打包，R=2 时生成的代码与原来相同；zero-fill 的 LSE 改为循环；所有 wave 都走 LO 路径。
  - LO 与 HI 两条路径只在 `_drain_barrier` 之后调换“读 K 到寄存器”和“prefetch”的先后，`_named_barrier_pair` 是空操作，所以全部用 LO 不会引入跨 wave 的依赖。

## 2. 编译检查（compile-only，不占卡，`verify/isa/<cfg>/`）

| cfg | 覆盖的 shape | VGPR | vgpr spill | sgpr spill | scratch | LDS |
|---|---|---|---|---|---|---|
| prod（causal gqa4） | prod / proxy / fast / short_q / gqa4_batch2 / unequal* | 446 | 0 | 61 | 0 | 163840 |
| nc_g4 | 同上，非 causal | 442 | 0 | 2 | 0 | 163840 |
| c_g1 / nc_g1 | mha | 444 / 440 | 0 | 66 / 4 | 0 | 163840 |
| c_g2 / nc_g2（本次新增） | toy | 446 / 442 | 0 | 61 / 2 | 0 | 163840 |

sgpr spill 溢出到 VGPR lane，不产生 scratch；r6 冠军自己就有 60。

## 3. 正确性（上卡，每个进程一个 shape；toy、edge、fast、proxy 设置 AMD_SERIALIZE_KERNEL=3）

`verify/tools/vcorr.py`：
- 用 job 的 `gates.check_correctness`：NaN 毒化分配器，检查全部元素有限，门限 49 dB。参考来自 refcache（causal 的 spec shape），其余情况用 eager 现算。
- 冠军在同一进程里用同样方法测一遍，再比较 cand 与冠军是否逐位相同，最后跑 20 次确定性。

| shape | causal o / lse dB | 非 causal o / lse dB | 与 r6 逐位相同 | 确定性 |
|---|---|---|---|---|
| prod | 50.835 / 89.196（refcache） | — | c 和 nc 都相同 | 20/20 |
| proxy | 50.894 / 88.236（refcache） | — | c 和 nc 都相同 | 20/20 |
| fast | 51.229 / 85.483（refcache） | — | c 和 nc 都相同 | 20/20 |
| toy | 51.859 / 81.851 | 50.183 / 85.169 | c 和 nc 都相同 | 20/20 |
| short_q（sq=128=BLOCK_M） | 49.926 / 86.440 | 49.824 / 86.732 | c 和 nc 都相同 | 20/20 |
| gqa4_batch2 | 51.760 / 81.507 | 50.181 / 85.297 | c 和 nc 都相同 | 20/20 |
| mha | 51.596 / 81.180 | 49.981 / 85.152 | c 和 nc 都相同 | 20/20 |
| unequal_seqlen | 49.984 / 87.380 | 49.984 / 87.874 | c 和 nc 都相同 | 20/20 |
| unequal_seqlen2 | 49.968 / 88.362 | 50.004 / 88.787 | c 和 nc 都相同 | 20/20 |
| sq_gt_skv（只测非 causal） | — | 49.991 / 86.696 | nc 相同 | 20/20 |

每一行的 dB 都与冠军完全相同，因为输出逐位相同。

**对抗大 logit**（`verify/tools/vadv.py`，输入来自 `verify_r6/adv_inputs.py`，要求与 r6 逐位相同）：
- toy、short_q、gqa4 的 causal 和非 causal 各测 32 类输入，proxy 的 causal 也测 32 类，每组都是 32/32 相同。
- prod causal 测 8 类（randn、x16、stair9、ramp1、jump1e3、bnd_sum、diag_neg、neg_early），8/8 相同。
- 合计 232/232。r6 的投机 softmax 在 4-wave 下行为不变。

证据：`verify/card/corr_*.{json,log,dmesg}`、`verify/card/adv_*.{json,log,dmesg}`。

## 4. 性能（job 官方 `benchmark.py`，只用 `--arm-path`，没有 beat，n=101，每个进程一个 shape；ratio = champ_ms / cand_ms）

| 进程 | 顺序 | champ ms | cand ms | ratio |
|---|---|---|---|---|
| prod_p1 | champ,cand | 1.51523 | 1.42362 | **1.0644** |
| prod_p2 | cand,champ | 1.51147 | 1.42422 | **1.0613** |
| prod_p3 | champ,cand | 1.51684 | 1.41945 | **1.0686** |
| prod_p4 | cand,champ | 1.51323 | 1.42398 | **1.0627** |
| prod_aa | champ, champ_copy | 1.46764 | 1.46760 | 1.0000（A/A） |
| proxy_p1 | champ,cand | 0.08593 | 0.09218 | **0.9322** |
| fast_p1 | champ,cand | 0.01827 | 0.01482 | 1.2328 |

- prod 均值为 **1.0642**，4 个进程的极差为 0.7%，每个进程都远高于 0.5% 的噪声门槛。
- cand 在 prod 上约 1545 TF/s，冠军约 1452 TF/s。
- 证据：`verify/perf/*.json`，日志和 dmesg 在 `verify/card/perf_*.{log,dmesg}`，执行脚本是 `verify/seq_perf.sh`。

## 5. 判定与风险

- **win（prod，按规则 7 排名）**，数值与冠军逐位相同。
- **风险：proxy -6.8% 已复现。** 原因仍未定位。原 lab 的候选解释有两个：一是每个 128 行 WG 各自加载完整的 K/V，L2 流量翻倍；二是 2 个 WG 共享一个 CU 时尾部表现不同。
- job 如果把 proxy 当作 sentinel 或 gate，需要 operator 决定是否接受这个回退，或者要求按 shape 选择构型。
- 卡安全：25 个卡上进程（toy 最先，一个进程一个 shape），dmesg 新增 0 行，rc 全部为 0；没有用 rocprof，没有 kill。

## 6. 建议交给 GPU-0 fwd job 的 must-hint（operator 转交；本 lab 不写 hint.md）

> **h34 (must) -- L12 legal 4-wave: 4-wave WG x 2 WGs/CU, prod +6.4% vs r6 (independently verified, bitwise), proxy -6.8%.**
> Land the tree `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab2-L12/arms/r6_occ2_lo`
> (patch `.../lab2-L12/r6_occ2_lo.diff`, on r6 = rounds/006/op, kernel md5 a60bff8e). Two files change:
> - managers: the 5 `num_waves != 8` raises become `not in (4, 8)`; `_tdm_load_views(num_warps=...)` gets `self.num_waves` from K/V V2. This is the only real 8-wave assumption: the TDM split is 8 rows per wave.
> - kernel: `NUM_WAVES=4`, R=2 (BLOCK_M 128), `MIN_KV_BLK_BYTES=0`, `_alloc_lds` 160 KB (2 WGs per CU), `waves_per_eu=2`, every wave on the LO role, zero-fill LSE loops over `BLOCK_M/BLOCK_SIZE`.
>
> 446 VGPR / 0 spill / 0 scratch. Bitwise equal to r6 on all 10 job shapes, causal and non-causal, and on 232 adversarial large-logit cases.
> prod vs r6 (benchmark.py, 4 processes, rotated): 1.0644 / 1.0613 / 1.0686 / 1.0627, A/A 1.0000. fast +23%, **proxy 0.932**.
> Accept on prod ranking. Next lever: a shape-dependent build (keep 8-wave BLOCK_M=256 when the grid already fills 2 WGs per CU, e.g. proxy),
> or find why proxy regresses (K/V L2 traffic doubles per 128-row WG).
> Dead: 1 wave per SIMD (4 waves x 64 rows) -13%; 4-wave WG at 1 WG per CU -24%; 8 waves at BLOCK_M=128 -36%; all-HI roles null; split roles +2.8% only.

## 7. 文件（都在 `lab2-L12/verify/` 下）

`arms/{cand,champ_copy}`、`md5.txt`、`isa/<cfg>/`、`tools/{vcorr.py,vadv.py,compile_isa.py}`、`card.sh`、`seq_corr.sh`、`seq_perf.sh`、
`seq_corr.out`、`seq_perf.out`、`card/`（每个进程的 json、log、dmesg）、`perf/`（benchmark 的 json 和 champ_md5.log）。
