# k_dq 实验室（lab-kdq）：GPU3 / fa-g3，2026-09-28，基线 r19h

## 结论（先看这里）

1. **提 occupancy 这条路是死的，已在卡上测过。** BLOCK_Q 32、不带 prefetch 的 k_dq 编到 445 VGPR（每 SIMD 2 个 wave），prod 上 k_dq 慢 41%，整个 op 慢 13–15%。原因是每个 query 的 K/V load 和 LDS staging 都翻倍了。4 个 GQA 头放进同一个 WG 共享 L0（g4/g2）也更慢，k_dq 分别 +7.4% 和 +3.4%。
2. **k_dq 的瓶颈是发射，不是延迟。** 只减 VALU（VF_Q：softmax 改成一条 fma，scale 挪到 store 时乘一次；循环体 741 → 630 条指令）就让 k_dq 快 6.1%（3.181 → 2.987 ms），整个 op 快 2.4%。这和 profile/REPORT.md 里"k_dq 对时钟不敏感、是延迟 bound"的判断相反。
3. **最好的候选 `qu2ku2`：只做了 compile-only，还没上卡。** 它是 VF_Q 加两处"循环展开 2 次"：DQ_U2 和 KV_U2。展开后，carried prefetch 每次迭代的 64 条 `v_mov_b64` 轮转拷贝没有了。k_dq 循环体按每次迭代算是 741 → 550 条指令，k_dkdv 是 675 → 564 条，k_dkdv 的 VGPR 从 724 降到 638。**我正要做卡上测量时，operator 通知把 GPU3 交还给 bwd job，所以卡上工作就此停止。**
4. **k_delta 融合：null 到略亏，不推荐。** 融进 k_dq 的 prologue 后，k_dq 本身慢 1.7–2.9%，只省下 k_delta 的约 0.05 ms。整个 op 上 n1d −0.2~−0.3%，在噪声以内；kq1d 相对 kq1 +0.7%。低时钟下（k_delta 0.117 ms）可能转正，但没测到。

**本 lab 卡上工作**：7 个进程（5 次正确性、2 次计时筛选），每个进程之后 dmesg 都没有新的 amdgpu 行。GEMM 降频测量（gb）一次都没跑。

## 1. 各 arm 定义（全部基于 r19h；tree 在 `OP/<arm>/`，OP = `oe/artifacts/job/job_context/op`）

新 kernel `k_dqg` 只在不 split 的路径上用（nsp_q == 1，即 proxy/prod）。fast 仍然走 k_dq_sp，保持不变。常量说明：

- NW：每个 WG 的 wave 数。wave w 负责 q head `grp*NW+w`，没有任何跨 wave 通信或 barrier，每个 wave 有自己的 LDS 片。
- BQW：每个 wave 负责的 query 数。
- PF：是否保留 K/V 提前一次迭代的 prefetch。
- DFUSE：在 prologue 里算 delta 并写回，k_dq 先于 k_dkdv 跑，k_delta 不再 launch。
- VF_Q / VF_KV：`exp2(fma(s, scale·log2e, −lse·log2e))`；dS 存储时不乘 scale，scale 在最后写 dQ/dK 时乘一次。
- DQ_U2 / KV_U2：full loop 展开 2 次。k_dq 奇数剩下的那个块交给 mask loop 做（在这个块上 predicate 恒为 false，结果逐位不变）。k_dkdv 奇数剩下的那次迭代交给 qloop_tail 做（G=4 时不会出现）。

| arm | NW/BQW/PF | 其它 | tree md5 | k_dq(g) VGPR | k_dkdv VGPR | spill/scratch |
|---|---|---|---|---|---|---|
| r19h | 1/64/PF（旧 k_dq） | — | a791a047 | 960 | 724 | 0/0 |
| n1 | 1/64/PF | 对照：k_dqg ≡ r19h | fc8c21c5 | 960 | 724 | 0/0 |
| g4 | 4/64/PF | GQA 组共享 L0 | 378a5909 | 974 | 724 | 0/0 |
| g2 | 2/64/PF | | fd15adda | 976 | 724 | 0/0 |
| **o1** | 1/32/无 PF | **2 wave/SIMD** | a0dfbac0 | **445** | 724 | 0/0 |
| g4o | 4/32/无 PF | 2 wave/SIMD + 共享 | 930e1d38 | 445 | 724 | 0/0 |
| o1pf / g4opf | 1或4/32/PF | 未上卡 | f58944c4 / 6e558f95 | 598 / 612 | 724 | 0/0 |
| n1d / g4d | 1或4/64/PF | DFUSE | 6c531e65 / 64fbe351 | 960 / 974 | 724 | 0/0 |
| q1 | 1/64/PF | VF_Q | 0a1d39b5 | 976 | 724 | 0/0 |
| k1 | 1/64/PF | VF_KV | 240551f1 | 960 | 740 | 0/0 |
| kq1 / kq1d | 1/64/PF | VF_KV+VF_Q（+DFUSE） | ac3c5f33 / 2e86f725 | 976 | 740 | 0/0 |
| u2n | 1/64/PF | DQ_U2（只做这一项） | ed57c5aa | 991 | 724 | 0/0（未上卡） |
| u2 | 1/64/PF | VF_Q+DQ_U2 | 3549c98d | 990 | 724 | 0/0（未上卡） |
| ku2 | 1/64/PF | KV_U2（只做这一项） | 890d949a | 960 | **638** | 0/0（未上卡） |
| **qu2ku2** | 1/64/PF | **VF_Q+DQ_U2+KV_U2** | **9895f89e** | 990 | **638** | 0/0（未上卡） |
| all | 同上 | 再加 VF_KV | c03daa62 | 990 | 636 | 0/0（未上卡） |

58 个 kernel 编译结果全部是 0 spill、0 scratch（`compile/isa_summary.txt`）。

**热循环静态计数**（每次迭代；只用来筛选，不能拿来排名）：

| kernel | 指令 | v_mov_b64 | s_set_vgpr_msb | v_nop |
|---|---|---|---|---|
| k_dq r19h | 741 | 65 | 142 | 28 |
| k_dqg q1 | 630 | 65 | 128 | 21 |
| k_dqg u2n | 655 | 0.5 | 140 | 15.5 |
| k_dqg u2 | 550 | 0.5 | 124 | 22.5 |
| k_dkdv r19h | 675 | 64 | 125 | 66 |
| k_dkdv k1 | 624 | 64 | 107 | 67 |
| k_dkdv ku2 | 564 | 0 | 105 | 26.5 |

**风险**：DQ_U2 把 k_dq 的 `s_wait_loadcnt` 从每次迭代 4 条变成 7 条（u2n）或 4.5 条（u2）。h37 说过 k_dq 的 load 聚簇是承重的，所以这一项必须以卡上测量为准。

## 2. 正确性（lab_validate.py，与 validation.py 门槛相同，refcache 不在卡上重算）

| shape | arm | dq / dk / dv dB | dk/dv 确定性 | dq 逐次 | 相对 r19h |
|---|---|---|---|---|---|
| proxy | n1 g4 g2 o1 g4o | 52.52 / 52.57 / 52.67 | 100 次逐位相同 | 逐位相同 | **全部逐位相同** |
| proxy | n1d g4d | 52.52 / 52.57 / 52.67 | 逐位相同 | 逐位相同 | dq 98.9、dk 94.4 dB，dv 逐位相同 |
| proxy | q1 | 52.54 / 52.57 / 52.67 | 逐位相同 | 逐位相同 | dq 49.8 dB，dk/dv 逐位相同 |
| proxy | k1 / kq1 / kq1d | 52.52–52.54 / 52.58 / 52.67 | 逐位相同 | 逐位相同 | dk 50.0、dv 76.0 dB |
| prod | g4 g2 o1 g4o | 52.56 / 52.60 / 52.71 | 50 次逐位相同 | 逐位相同 | 逐位相同 |
| prod | n1d g4d | 52.56 / 52.60 / 52.71 | 逐位相同 | 逐位相同 | dq 96.1、dk 92.7 dB |
| prod | q1 / kq1 / kq1d | 52.54 / 52.59–52.60 / 52.71 | 逐位相同 | 逐位相同 | dq 49.9 dB |
| fast | q1 k1 kq1 kq1d | 52.61 / 52.61–52.65 / 52.83 | 200 次逐位相同 | 逐位相同 | fast 走 split 路径，q1 逐位相同；k1 dk 50.1 dB |

- 上表中"相对 r19h"那一列的 dB，是和 r19h **自身输出**比较的结果，不是和参考比较。VF 只改了舍入位置，对 fp32 参考的 dB 与 r19h 相同，甚至略高（dq 52.54 对 52.52）。
- 边界证明（CPU，`bounds/`）：
  - `bounds_dqg.txt`：k_dqg 的每个新下标，覆盖 9 个 UT shape × causal 模式 × 7 种 (NW,BQW,PF) 配置，越界 0 次，(b,qh,tile) 全覆盖。
  - `bounds_u2.txt`：两处展开对任意 n ∈ [0,4096] 都满足三条：消费顺序等于 0..n-1；所有 prefetch 都在 [0,n-1] 内；carried prefetch 正好被它所预取的那次迭代消费。另外，mask loop 接手的剩余块上，causal predicate 共核对 64,680 次，全部为 false。
  - **U2 系列 arm 还没有在卡上验证正确性**，预期结果：u2n、ku2 与 r19h 逐位相同；u2、qu2ku2 与 q1 逐位相同。

## 3. 计时（prod，`tools/kbench.py blk`：分块尺子 lead 4 + block 9，每轮回文，45 次计时，每次计时前 flush 256 MB L2）

**这只是筛选：每组只有 1 个进程，没有做轮换，不满足 ≥3 进程的要求。**

| arm | k_dq ms（相对 r19h） | 整个 op ms（相对 r19h） | 进程 |
|---|---|---|---|
| r19h | 3.1622 / 3.1811 | 8.5845 / 8.5827 | s1 / s2 |
| n1（A/A 对照） | 1.0023 | 1.0002 | s2 |
| **q1** | **0.9391** | **0.9761** | s2 |
| kq1 | 0.9381 | 0.9781 | s2 |
| k1 | 0.9942（k_dq 与 n1 是同一个 kernel，即噪声） | 1.0017 | s2 |
| kq1d | 0.9645 | 0.9853 | s2 |
| n1d | 1.0196 / 1.0165 | 0.9969 / 0.9978 | s1 / s2 |
| g4d | 1.0301 | 1.0002 | s1 |
| g2 | 1.0339 | 1.0081 | s1 |
| g4 | 1.0744 | 1.0214 | s1 |
| o1（2 wave/SIMD） | **1.4108** | 1.1474 | s1 |
| g4o（2 wave/SIMD） | 1.4091 | 1.1351 | s1 |

sclk 中位数：k_dq 段 1938–2088 MHz，整个 op 段 1845–1947 MHz。原始数据在 `runs/kb_prod_blk_s{1,2}.log`。

## 4. 判定（排名）

| # | arm | 判定 | 证据 |
|---|---|---|---|
| 1 | **qu2ku2** | **待测（最有希望）**：compile-only 通过，边界已证 | 静态：k_dq −26%、k_dkdv −16% 指令/迭代，VGPR 990/638 |
| 2 | q1（VF_Q） | **win（筛选级）**：k_dq −6.1%，op −2.4%，dB 不降 | s2，1 个进程；需要 ≥3 个轮换进程确认 |
| 3 | u2 / u2n / ku2 | 待测 | 用来拆分 qu2ku2 里各项的贡献 |
| 4 | kq1 | 与 q1 相同；VF_KV 本身 null | k1 op +0.17% |
| 5 | n1d / kq1d / g4d（delta 融合） | null 到亏 | op −0.3% 到 +0.7% |
| 6 | g2 / g4（GQA 共享 L0） | loss | k_dq +3.4% / +7.4% |
| 7 | o1 / g4o（2 wave/SIMD） | **loss，方向关闭** | k_dq +41%，op +13–15% |

低时钟（gb）下的测量一次都没做，所以"训练工作点上的收益"目前没有证据。

## 5. 未完成的卡上计划（`tools/plan.sh`，没有运行，交给 operator 排期）

1. `lab_validate` proxy/prod/fast：arm 为 r19h、u2n、ku2、q1、u2、qu2ku2，检查上面列出的逐位预期。
2. `kbench prod blk`：3 个轮换进程，arm 为 r19h、q1、u2、u2n、ku2、qu2ku2。给出只看 k_dq 的时间和整个 op 的时间。
3. `benchmark.py`（官方尺子）：prod/proxy/fast 各 3 个轮换进程，arm 为 r19h、r19h_copy（A/A）、q1、qu2ku2。fast 只有 KV_U2 生效（k_dkdv_sp）。
4. `kbench prod gb`：降频测量。每次计时前跑 4 个 32768×4096×14336 的 bf16 GEMM，每个进程总共约 6 s 的 GEMM，有上限，不是 burn loop。开跑前需要 operator 按 LAB-RULES 第 6 条确认。3 个轮换进程，arm 为 r19h、q1、qu2ku2、kq1d、n1d。

## 6. 建议的 hint（operator 下发；如果卡上确认 qu2ku2 胜出，就用 must refactor）

```
| h69 | must refactor | k_dq is ISSUE-bound, not latency-bound: fma softmax (+scale at store) in k_dq and unroll-by-2 of both carried-prefetch loops (kills the 64 back-edge v_mov_b64 per trip); occupancy (<=512 VGPR, 2 waves/SIMD) measured -41% on k_dq | open |

## h69 -- k_dq: cut issue slots, not latency. Port lab-kdq arm qu2ku2 onto r19h.

Source tree (copy, do not re-derive):
  /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq/oe/artifacts/job/job_context/op/qu2ku2
  tree_md5 (benchmark.py's) 9895f89e; diff vs r19h: lab-kdq/qu2ku2.{kernels,impl}.diff.
  It adds k_dqg (non-split k_dq path, proxy/prod; fast keeps k_dq_sp) with three levers:
  (1) VF_Q: t = fma(s, scale*log2e, -lse*log2e); p = exp2(t); dS = p*(dP-delta) (no
      scale); dQ *= scale once at the store. Loop body 741 -> 630 instr.
  (2) DQ_U2: kvloop_full runs PAIRS of blocks, two _body calls per scf.for trip; the
      first body's prefetch lands in a temporary set, the second one is allocated onto
      the dead carried set, so the 65 back-edge v_mov_b64 disappear (-> 550 instr/iter).
      The odd leftover full block goes to kvloop_mask (predicate provably false there).
  (3) KV_U2: the same unroll in k_dkdv qloop_full (+ qloop_tail for odd G*n, never at
      G=4): 675 -> 564 instr/iter, 64 v_mov_b64 -> 0, VGPR 724 -> 638.
Evidence so far (GPU3, prod, same process, blocked ruler; ONE process -- screen only):
  VF_Q alone (arm q1): k_dq 3.181 -> 2.987 ms (-6.1%), whole op 8.583 -> 8.377 (-2.4%);
  n1 (k_dqg == r19h k_dq) A/A 1.002 / 1.000. dB vs ref dq 52.54 / dk 52.60 / dv 52.71,
  dk/dv bitwise x50, dq bitwise run to run. The U2 parts are COMPILE-ONLY: 0 spill /
  0 scratch, CPU-proved index sequences (lab-kdq/bounds/bounds_u2.txt); expected
  bitwise == q1 (u2, qu2ku2) and == r19h (ku2).
Required before adoption: lab-kdq/tools/plan.sh steps 1-3 (>= 3 rotated processes per
shape vs r19h + A/A copy); if DQ_U2 loses (h37: k_dq's load clump is load-bearing) ship
q1 + ku2 instead. Closed by this lab -- do not rebuild: k_dq occupancy via BLOCK_Q 32
without prefetch (445 VGPR, 2 waves/SIMD) -41% k_dq / -13..15% op; GQA-grouped k_dq WG
(4 q heads of one kv head per WG, no barriers) -7.4% k_dq; delta fused into k_dq's
prologue (op -0.3..+0.7%, null); VF in k_dkdv alone (null, +0.2%).
```

## 文件

- `REPORT.md`：本报告
- `qu2ku2.kernels.diff`、`qu2ku2.impl.diff`：最佳候选相对 r19h 的 diff
- `OP/lab/`：参数化源码，所有 arm 都由它生成：`tools/mkarm.sh NAME NW BQW PF DFUSE [VF_KV VF_Q DQ_U2 KV_U2]`
- `OP/<arm>/`：各 arm 的代码树；`compile/tree_md5.txt` 记录每棵树的 md5
- `tools/dqg_section.py`：k_dqg 源码片段（初版）
- `tools/compile.sh`、`compile/compile_bwd.py`（加了 dqg）、`compile/isa_summary.txt`：compile-only
- `bounds/bounds_dqg.{py,txt}`、`bounds/bounds_u2.{py,txt}`：CPU 边界证明
- `tools/kbench.py`：只看 k_dq 的计时，加整个 op 的计时；有 blk 和 gb 两种模式
- `tools/run1.sh`：单个卡上进程，负责 flock 和 dmesg 检查
- `tools/plan.sh`：未运行的卡上计划
- `runs/val_*`、`runs/val2_*`：正确性检查的原始输出
- `runs/kb_prod_blk_s{1,2}.*`：计时筛选；`.dmesg` / `.dmesg.bad` 全部为空
