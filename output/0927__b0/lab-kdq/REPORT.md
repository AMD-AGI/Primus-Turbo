# k_dq 实验室（lab-kdq），2026-09-28，基线 r19h

卡：先用 GPU3 / fa-g3，06:45 起 GPU3 交还给 bwd job。之后改用 GPU0 / fa-g0。

## 结论

1. **提 occupancy 这个方向，在卡上测过，是死路。**
   - BLOCK_Q=32、不带 prefetch，k_dq 可以编到 445 VGPR，也就是每个 SIMD 2 个 wave（o1、g4o）。
   - 代价是 prod 上 k_dq 慢 41%，整个 op 慢 13–15%。原因是每个 query 的 K/V load 和 LDS staging 都翻倍了。
   - 另一个思路是把 4 个 GQA 头放进同一个 WG，共享 L0，不加任何 barrier（g4、g2）。结果 k_dq 慢 7.4% / 3.4%。
2. **k_dq 实际是发射 bound，不是延迟 bound。** 只减少发射槽位就能让它快 6%。
   - **u2n**：full loop 展开 2 次。carried K/V prefetch 在回边上有 65 条 `v_mov_b64` 的轮转拷贝，展开后全部消失，而且输出与 r19h **逐位相同**。
     - k_dq −6.1%（3.185 → 2.990 ms）；**整个 op −3.06%**（8.580 → 8.317 ms）。
     - 3 个轮换进程分别是 0.9696 / 0.9697 / 0.9688。
   - **q1**：softmax 改成一条 fma，scale 挪到写 dQ 时乘一次。
     - k_dq −5.9%；整个 op −2.37%（3 个进程：0.9748 / 0.9755 / 0.9786）。
     - dq 对参考 52.54 dB，比 r19h 的 52.52 略高。
   - **两者不能叠加**：u2 = q1 + U2，整个 op 只 −1.19%，比单做任何一项都差。
3. **k_dkdv 上同样展开 2 次（ku2）亏 6.5%。** 虽然 VGPR 从 724 降到 638、每次迭代少 16% 的指令，结果照样亏。这是 k_dq 和 k_dkdv 行为不同的第 4 个实例（h37）。k_dkdv 上单独做 VF 是 null（+0.17%）。
4. **k_delta 融进 k_dq 的 prologue：null。** 整个 op −0.3% 到 +0.7%。k_dq 本身多花 1.7–2.9%，只省下 k_delta 的约 0.05 ms。**不推荐。**
5. **没做完的部分：**
   - 低时钟（GEMM 突发）下的测量没有做。我在 GPU0 上启动下一轮卡上筛选时，被 harness 的权限检查拒绝了（"Interfere With Workloads"），因此停止了一切卡上工作。
   - benchmark.py 官方尺子的 3 个进程 A/B 也没有做。
   - 剩下的计划都写在 `tools/plan.sh` 里，没有运行，交给 operator 排期。

**推荐**：用 **u2n** 作为下一个 must refactor。它最快，而且和 r19h **逐位相同**：proxy、prod、fast 三个 shape 的 dq/dk/dv 都逐位相同，所以不需要任何数值上的论证。

**卡上工作量**：GPU3 上 7 个进程，GPU0 上 6 个进程。每个进程之后都查了 dmesg，没有新的 amdgpu 行。GPU0 上出现过 4 条 `ifoe ... MC command 0x1b4 failed`，那是 host fabric 的日志，与 amdgpu 无关。

## 1. arm（全部从 `OP/lab/` 参数化生成，`tools/mkarm.sh`；OP = `oe/artifacts/job/job_context/op`）

新 kernel `k_dqg` 只在不 split 的路径上用（nsp_q == 1，即 proxy/prod）。fast 仍然走 r19h 的 k_dq_sp。

| arm | 内容 | tree md5 | k_dq VGPR | k_dkdv VGPR | spill/scratch | 上卡 |
|---|---|---|---|---|---|---|
| r19h | 基线 | a791a047 | 960 | 724 | 0/0 | ✓ |
| n1 | k_dqg，NW=1，BQW=64，PF（A/A 对照，ISA 与 r19h 基本相同） | fc8c21c5 | 960 | 724 | 0/0 | ✓ |
| g4 / g2 | 同一个 WG 放 4 / 2 个 q head（同一个 kv head、同一个 q tile），不加 barrier | 378a5909 / fd15adda | 974 / 976 | 724 | 0/0 | ✓ |
| o1 / g4o | BQW=32、无 prefetch，**每 SIMD 2 个 wave**；g4o 另加 GQA 组 | a0dfbac0 / 930e1d38 | **445** | 724 | 0/0 | ✓ |
| n1d / g4d | 在 prologue 里算 delta（DFUSE），k_dq 先于 k_dkdv 跑 | 6c531e65 / 64fbe351 | 960 / 974 | 724 | 0/0 | ✓ |
| q1 | VF_Q（fma softmax，scale 在 store 时乘） | 0a1d39b5 | 976 | 724 | 0/0 | ✓ |
| k1 / kq1 / kq1d | VF_KV（k_dkdv 上做同样的改动）/ VF_KV+VF_Q / 再加 DFUSE | 240551f1 / ac3c5f33 / 2e86f725 | 960 / 976 | 740 | 0/0 | ✓ |
| **u2n** | **DQ_U2**：k_dq 的 full loop 展开 2 次 | **ed57c5aa** | 991 | 724 | 0/0 | ✓ |
| u2 | VF_Q + DQ_U2 | 3549c98d | 990 | 724 | 0/0 | ✓ |
| ku2 | KV_U2：k_dkdv 的 qloop_full 展开 2 次 | 890d949a | 960 | **638** | 0/0 | ✓ |
| qu2ku2 / all | VF_Q+DQ_U2+KV_U2 / 再加 VF_KV | 9895f89e / c03daa62 | 990 | 638 / 636 | 0/0 | qu2ku2 ✓ |
| u2nb / u2b | 在 u2n / u2 的两半之间加 `sched_barrier(0)` | f1ac65a5 / 7c2a2486 | **798** / 798 | 724 | 0/0 | ✗ 只编译 |
| u2f / qf | VF 只做 fma，scale 仍在 dS 里（u2 / q1 的变体） | e5a6e7bb / 0a22c0ce | 998 / 978 | 724 | 0/0 | ✗ 只编译 |

`compile/isa_summary.txt` 里全部 kernel 都是 0 spill、0 scratch。

**热循环每次迭代的静态计数**（只用来筛选，不能拿来排名；u2 就是反例）：

| | 指令 | v_mov_b64 | s_set_vgpr_msb | v_nop | s_wait_loadcnt |
|---|---|---|---|---|---|
| k_dq r19h | 741 | 65 | 142 | 28 | 4 |
| q1 | 630 | 65 | 128 | 21 | 4 |
| **u2n** | 655 | 0.5 | 140 | 15.5 | 7 |
| u2 | 550 | 0.5 | 124 | 22.5 | 4.5 |
| k_dkdv r19h | 675 | 64 | 125 | 66 | 9 |
| ku2 | 564 | 0 | 105 | 26.5 | 9 |

## 2. 正确性（`OP/lab_validate.py`，门槛与 validation.py 相同）

refcache 在 06:42 被重新盖了 provenance。lab_validate 现在先按当前的 common_sha 找，再按旧 sha 找，**从来不在卡上重算 fp32 参考**。

| shape | arm | dq / dk / dv dB | dk/dv 确定性 | dq 逐次 | 相对 r19h |
|---|---|---|---|---|---|
| proxy / prod / fast | **u2n、ku2**、n1、g4、g2、o1、g4o | 与 r19h 相同：52.52/52.57/52.67（proxy），52.56/52.60/52.71（prod），52.61/52.65/52.83（fast） | 100 / 50 / 200 次逐位相同 | 逐位相同 | **逐位相同**（u2n/ku2 三个 shape 都验了；其余验了 proxy/prod） |
| proxy / prod | q1、u2、qu2ku2 | 52.54/52.57/52.67；52.54/52.60/52.71 | 逐位相同 | 逐位相同 | dq 49.8–49.9 dB；u2、qu2ku2 与 q1 逐位相同 |
| proxy / prod | k1、kq1、kq1d | dk 52.58–52.59 | 逐位相同 | 逐位相同 | dk 50.0、dv 75–76 dB |
| proxy / prod | n1d、g4d | 与 r19h 相同 | 逐位相同 | 逐位相同 | dq 96–99、dk 92–94 dB |
| fast | q1、u2、qu2ku2、k1、kq1、kq1d | 52.61 / 52.61–52.65 / 52.83 | 200 次逐位相同 | 逐位相同 | fast 走 split 路径 |

原始输出：`runs/val_*`、`val2_*`、`val3_*`。

**CPU 边界证明**（`bounds/`）：
- `bounds_dqg.txt`：k_dqg 的全部新下标，覆盖 9 个 UT shape × causal 模式 × 7 种 (NW,BQW,PF) 配置，越界 0 次，(b,qh,tile) 全覆盖。
- `bounds_u2.txt`：两处展开对任意 n ∈ [0,4096]，消费顺序都等于 0..n-1，prefetch 都在 [0,n-1] 内，carried prefetch 都正好被它所预取的那次迭代消费。mask loop 接手的剩余块上，causal predicate 核对 64,680 次，全部为 false。

## 3. 计时（prod，`tools/kbench.py blk`）

- 方法：分块尺子，lead 4 + block 9，每轮回文，45 次计时，每次计时前 flush 256 MB L2。
- 测两个量：只看 k_dq（直接 launch dq）和整个 op。比值 = arm 中位数 / 同一进程里 r19h 的中位数。

**GPU0，3 个轮换进程**（`runs/kb3_prod_blk_p{1,2,3}.log`，汇总 `runs/kb3_ratios.txt`）：

| arm | k_dq p1/p2/p3（均值） | 整个 op p1/p2/p3（均值） | 判定 |
|---|---|---|---|
| **u2n** | 0.9391/0.9399/0.9372（**0.9387**） | 0.9696/0.9697/0.9688（**0.9694**） | **win** |
| q1 | 0.9448/0.9392/0.9403（0.9414） | 0.9748/0.9755/0.9786（0.9763） | win |
| u2 | 0.9650/0.9570/0.9613（0.9611） | 0.9886/0.9880/0.9876（0.9881） | win，但不如单做任何一项 |
| qu2ku2 | 0.9609/0.9570/0.9533 | 1.0515/1.0522/1.0511（1.0516） | loss（被 ku2 拖累） |
| ku2 | 0.9978/1.0011/0.9981（k_dq 不变） | 1.0647/1.0662/1.0646（**1.0652**） | **loss** |

- r19h 绝对值：k_dq 3.181–3.188 ms，整个 op 8.579–8.583 ms。
- sclk 中位数：k_dq 段 1899–1940 MHz，整个 op 段 1823–1965 MHz。

**GPU3 筛选，每组 1 个进程**（`runs/kb_prod_blk_s{1,2}.log`）：

| arm | k_dq | 整个 op |
|---|---|---|
| n1（A/A） | 1.0023 | 1.0002 |
| k1 | 0.9942（同一个 kernel，即噪声） | 1.0017 |
| kq1 | 0.9381 | 0.9781 |
| kq1d | 0.9645 | 0.9853 |
| n1d | 1.0196 / 1.0165 | 0.9969 / 0.9978 |
| g4d | 1.0301 | 1.0002 |
| g2 | 1.0339 | 1.0081 |
| g4 | 1.0744 | 1.0214 |
| o1 | **1.4108** | 1.1474 |
| g4o | 1.4091 | 1.1351 |

**缺口**：
- u2n / q1 在 proxy 上还没有计时。它们在 proxy 上也走 k_dqg，所以需要测。fast 走的是 r19h 的代码，预期约 1.00。
- 官方 benchmark.py 和低时钟 gb 两种测量都没做（见第 5 节）。

## 4. 排名

| # | arm | 整个 op 相对 r19h（prod） | VGPR（dq/dkdv） | 最低 dB | 判定 |
|---|---|---|---|---|---|
| 1 | **u2n** | **−3.06%**（3 个进程，离散度 0.09%） | 991 / 724 | 52.52（逐位 = r19h） | **win** |
| 2 | q1 | −2.37%（离散度 0.4%） | 976 / 724 | 52.54 | win（数值有变化） |
| 3 | kq1 | −2.2%（1 个进程） | 976 / 740 | 52.54 | 等于 q1；VF_KV 是 null |
| 4 | u2 | −1.19% | 990 / 724 | 52.54 | 叠加反而更差 |
| 5 | n1d / kq1d / g4d | −0.3% 到 +0.7% | 960 / 724 | 52.52 | null |
| 6 | g2 / g4 | +0.8% / +2.1% | ~975 / 724 | 52.52 | loss |
| 7 | qu2ku2 / ku2 | +5.2% / +6.5% | 990 或 960 / 638 | 52.52 | loss |
| 8 | o1 / g4o（2 wave/SIMD） | +14.7% / +13.5% | **445** / 724 | 52.52 | **loss，方向关闭** |
| — | u2nb / u2b / u2f / qf | 没测 | 798 / 798 / 998 / 978 | — | 只编译，待测 |

- **最佳 tree**：`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq/oe/artifacts/job/job_context/op/u2n`
  - tree_md5（benchmark.py 的算法）`ed57c5aa`
  - `kernels.py` md5 `1fda18284a431c890bdae5da180d37ed`，`impl.py` md5 `08cb8533d75e82198fabb9514fd18ba5`
  - 相对 r19h 的 diff：`u2n.kernels.diff`、`u2n.impl.diff`
- **注意**：u2n 的 kernels.py 里带着 lab 的全部开关代码（DFUSE、VF、KV_U2、U2B 等），但它们都是关的，ISA 不受影响。移植到 job 时只需要保留 k_dqg 和 DQ_U2 两部分。

## 5. 没跑的卡上计划（`tools/plan.sh`）

1. u2nb、u2b、u2f、qf 的筛选。u2nb 是 u2n 加 `sched_barrier`，VGPR 798，可能更好。
2. 上述 arm 的正确性检查。
3. benchmark.py：prod/proxy/fast 各 3 个轮换进程，arm 为 r19h、r19h_copy（A/A）、u2n、q1。
4. 低时钟 gb 测量：每次计时前跑 4 个 32768×4096×14336 的 GEMM，每个进程总共约 6 s，有上限。开跑前需要 operator 按 LAB-RULES 第 6 条确认。arm 为 r19h、u2n、q1、n1d、kq1d，3 个进程。

## 6. 建议的 hint（operator 下发）

```
| h69 | must refactor | k_dq is ISSUE-bound: unroll k_dq's kvloop_full by 2 (u2n) -- the 65 back-edge v_mov_b64 of the carried K/V prefetch vanish; prod op -3.06% (3 procs), output BITWISE identical to r19h | open |

## h69 -- Port lab-kdq arm u2n onto the champion (r19h): unroll k_dq's full kv loop by 2

Source (copy, do not re-derive):
  /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-kdq/oe/artifacts/job/job_context/op/u2n
  tree_md5 ed57c5aa; diff vs r19h: lab-kdq/u2n.{kernels,impl}.diff. Keep only k_dqg +
  launch_dqg with DQ_NW=1, DQ_BQW=64, DQ_PF=True, DQ_U2=True (all other lab switches are off
  and may be deleted), and impl.py's use_g routing (nsp_q == 1 -> launch_dqg; fast keeps k_dq_sp).
Mechanism: kvloop_full runs PAIRS of kv blocks, two _body calls per scf.for trip. The first
  body's prefetch goes to a temporary register set; the second body's prefetch is issued
  after the carried set is dead and is allocated onto it, so the 65 v_mov_b64 rotation
  copies at the back-edge disappear (loop 741 instr/iter -> 655). npair = nfull // 2; the
  odd leftover full block runs in kvloop_mask, where the causal predicate is provably
  false (bounds_u2.txt, 64,680 checks) -- same blocks, same order, same arithmetic.
Evidence (GPU0, prod, blocked ruler, 3 rotated processes, per-kernel + full op):
  k_dq  0.9391 / 0.9399 / 0.9372 of r19h (3.185 -> 2.990 ms)
  op    0.9696 / 0.9697 / 0.9688 of r19h (8.580 -> 8.317 ms)
  dq/dk/dv BITWISE identical to r19h at fast, proxy and prod; dk/dv bitwise x200/x100/x50.
  Compile: k_dqg 991 VGPR, 0 spill, 0 scratch. CPU-proved index sequences.
Still to measure before adoption: benchmark.py 3x3 rotated vs r19h + A/A (proxy is also
  affected -- it takes the non-split path), lowered-clock condition.
Do NOT combine with (all measured on the same card, same ruler):
  - fma softmax + scale-at-store in k_dq (q1): alone op -2.37%, but WITH the unroll -1.19%
    (the gains do not add; u2 is worse than either);
  - the same unroll in k_dkdv qloop_full (ku2): op +6.5% despite 724 -> 638 VGPR and -16%
    instructions -- k_dq and k_dkdv are different machines (h37);
  - k_dq occupancy: BLOCK_Q 32 without prefetch reaches 445 VGPR = 2 waves/SIMD and costs
    k_dq +41%, op +13..15% (per-query K/V load + staging doubles). Closed;
  - GQA-grouped k_dq workgroup (4 q heads of one kv head, same q tile, no barriers): k_dq +7.4%;
  - delta fused into k_dq's prologue (k_delta removed, k_dq before k_dkdv): op null (-0.3..+0.7%).
Follow-ups prepared compile-only: u2nb (u2n + sched_barrier(0) between the halves, 798 VGPR).
```

## 文件

- `REPORT.md`：本报告
- `u2n.kernels.diff`、`u2n.impl.diff`：最佳 arm 相对 r19h 的 diff
- `OP/lab/`：参数化源码
- `OP/<arm>/`：各 arm 的代码树；`compile/tree_md5.txt` 记录每棵树的 md5
- 另有 `OP/q1r/__pycache__` 残留，属主是 root，删不掉，可以忽略
- `tools/mkarm.sh`：生成 arm
- `tools/compile.sh`、`compile/compile_bwd.py`、`compile/isa_summary.txt`：compile-only
- `tools/kbench.py`：只看 k_dq 的计时，加整个 op 的计时；blk 和 gb 两种模式，gb 从未运行
- `tools/kbratio.sh`：汇总 kbench 的比值
- `tools/run1.sh`：当前指向 fa-g0 / gpu0 lock
- `tools/step2.sh`：已运行的 3 进程计时脚本
- `tools/plan.sh`：未运行的卡上计划
- `bounds/bounds_dqg.*`、`bounds/bounds_u2.*`：CPU 边界证明
- `runs/`：所有卡上进程的日志、`.dmesg` 和 `.dmesg.bad`。`.dmesg.bad` 全部为空
