# k_dq 实验室（lab-kdq），2026-09-28

## 第二轮（GPU0 / fa-g0，用户已放行）：以新冠军 r29（= r19h + u2n + g86）为基线

- 基线树 `OP/cur` 是 job 的 `op/current` 的只读拷贝，tree_md5 为 `c6b95c2c`，其中 kernels.py 的 md5 是 `37f37052…`。
- 所有新 arm 都从 `OP/lab2` 生成。`OP/lab2` 就是冠军代码加上 lab 的开关，开关默认值与冠军一致。我按 compile-only 核对过：`c0` 与 `cur` 在 k_dqg、k_dkdv、k_dkdv_sp 三个 kernel 上 ISA 字节完全相同。
- arm 用 `tools/mkarm2.sh NAME KEY=VAL…` 生成。

### 结论

1. **训练工作点（低时钟）确认了 u2n 的收益，而且比高时钟下更大。**
   - 条件：每次计时前跑 10 个 bf16 GEMM，形状 32768×4096×14336，每次约 25 ms、约 1.5 PF/s，使用 e2e 所用的 IMAGE hipBLASLt 库。kernel 执行期间的 sclk 为 **1272–1352 MHz**，与训练里 1250–1430 MHz 相符。
   - 结果（cur 对 r19h，3 个轮换进程）：**k_dq 0.9224 / 0.9234 / 0.9212（−7.8%）；整个 op 0.9667 / 0.9680 / 0.9688（−3.2%）**。
   - 对照：约 1840 MHz 下，k_dq −6.0%，整个 op −3.05%。
   - 有一处与 profile/REPORT.md 不同：在这个低时钟状态下，k_dq 是 4.05 ms，而约 1950 MHz 时是 2.98 ms，**对时钟敏感（约 1.36×）**。profile 里说"k_dq 对时钟不敏感"，这一点这里没有复现。
   - **方法注意**：GEMM 必须使用 IMAGE 库（`/opt/venv/.../_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`）。第一次测量（`kb6_prod_gb_*`）用的是 job 的 `~/.local` 库，一个 GEMM 要 47 ms（约 80 TF/s），时钟不降反升到 2155 MHz。那组数据作废，只作为"高时钟、无 L2 flush"条件下的旁证：cur/r19h = 0.9647。
2. **operator 点名的 4 个后续 arm，外加 5 个新 arm，全部没有超过冠军。**
   - u2nb / u2b：加 `sched_barrier(0)` 把展开的两半隔开。VGPR 从 991 降到 798，但 k_dq **慢 16–21%**，整个 op 慢 6–8%。h37 说过 k_dq 的 load 聚簇是承重的，这里又得到一次印证。
   - u2f / u2fb：只把 softmax 改成 fma，scale 仍在 dS 里。整个 op 分别慢 2.0% 和 6.8%。
   - qf：在已经展开 2 次的基线上，它和 u2f 等价，所以没有单独建。
3. **k_dq 的 epilogue 经 LDS 转置（c_epi，把 k_dkdv 的 g86 移植过来）：null。**
   - 输出与冠军逐位相同；256 条 `buffer_store_b16` 变成 32 条 `buffer_store_b128`。
   - prod：整个 op 0.9992 / 0.9989 / 1.0008，k_dq +0.4%。
   - proxy：k_dq −3.2%，整个 op −0.5%，但 proxy 的 op 噪声约 ±2%。
   - 低时钟：整个 op 0.9999。
   - 判定为 null。每个 WG 只执行一次 epilogue，在 prod 的约 64 次循环迭代面前占比太小。
4. **k_dkdv 减发射槽位（operator 的第 3 点），全部 null 或 loss。**
   - c_k1（VF_KV：fma softmax，scale 挪到 dK store）：循环体 659 → 609 条指令，v_pk 96 → 41。高时钟 +0.43%，**低时钟 +0.39%**。这个 kernel 对时钟敏感，但在低时钟下同样没有收益。
   - c_k1f（只做 fma）：+0.86%。
   - c_ku2（k_dkdv 展开 2 次，VGPR 729 → 644，消掉 64 条 v_mov_b64）：**+7.3%**。
   - c_ku2b（再加 sched_barrier）：+8.7%。
   - 结论：k_dkdv 在循环体内减 VALU 和轮转 mov 都拿不到时间。这与 h16/g60 的"发射槽位不是这个 kernel 的约束"一致。
5. **这一轮没有新的冠军候选。** 最佳树仍然是冠军本身（u2n）。

### 各 arm 的定义与结果

| arm | 相对冠军的改动 | tree md5 | k_dq VGPR | k_dkdv VGPR | spill/scratch | 相对冠军的 dB | prod 整个 op（3 进程） | prod k_dq | proxy 整个 op / k_dq（3 进程） | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| cur | 冠军 | c6b95c2c | 991 | 729 | 0/0 | — | 1 | 1 | 1 | — |
| r19h | 上一任冠军 | a791a047 | 960 | 724 | 0/0 | 逐位相同 | 1.0359/1.0300/1.0287 | 1.064 | 1.030 / 1.059 | 对照 |
| c_u2nb | + sched_barrier | 480af77f | 798 | 729 | 0/0 | **逐位相同** | 1.0846/1.0800/1.0763 | **1.207** | 1.087 / 1.231 | loss |
| c_u2b | + barrier + VF_Q | 41fff126 | 798 | 729 | 0/0 | dq 49.9 dB（参考 52.54） | 1.0633/1.0590/1.0560 | 1.161 | 1.067 / 1.181 | loss |
| c_u2f | VF_Q 只做 fma | e3c7c318 | 998 | 729 | 0/0 | dq 80 dB（参考 52.56） | 1.0247/1.0197/1.0165 | 1.034 | 1.029 / 1.078 | loss |
| c_u2fb | 同上 + barrier | 50ebc975 | 806 | 729 | 0/0 | dq 80 dB | 1.0717/1.0665/1.0656 | 1.181 | 1.072 / 1.209 | loss |
| **c_epi** | dQ epilogue 经 LDS | 03afacee | 1000 | 729 | 0/0（LDS 24576） | **逐位相同** | 0.9992/0.9989/1.0008 | 1.004 | 0.995 / **0.968** | null |
| c_ku2b | k_dkdv 展开 + barrier | c5f6d2cc | 991 | 644 | 0/0 | 逐位相同 | 1.0900/1.0878/1.0822 | 1.001 | 1.101 / 1.004 | loss |
| c_ku2 | k_dkdv 展开 | 88079a87 | 991 | 644 | 0/0 | 逐位相同 | 1.0670/1.0751/1.0758 | 1.001 | — | loss |
| c_k1 | VF_KV | 0e504495 | 991 | 746 | 0/0 | dk 50 dB（参考 52.59） | 1.0003/1.0063/1.0062 | 1.001 | — | null |
| c_k1f | VF_KV 只做 fma | f8266569 | 991 | 748 | 0/0 | dk 75 dB | 1.0033/1.0111/1.0115 | 1.000 | — | null/loss |

- 比值 = arm 的中位数 / 同一进程里 cur 的中位数。
- 计时方法：`tools/kbench.py blk`，分块尺子，lead 4 + block 9，每轮回文，45 次计时，每次计时前 flush 256 MB L2。每个 shape 3 个轮换进程。
- c_k1、c_k1f、c_ku2 的计时在单独的 3 个进程里做（`kb6_prod_blk_*`），arm 为 cur、c_k1、c_k1f、c_ku2。
- 正确性（`runs/val5_*`、`val6_*`）：全部通过 fast ×200、proxy ×100、prod ×50 的门槛。对参考的 dB 全部 ≥ 52.52，dk/dv 逐位确定，dq 逐次逐位相同。
- 编译：`compile/isa_summary_round2.txt` 里 33 个 kernel 全部是 0 spill、0 scratch。
- 边界证明：
  - `bounds/bounds_epi.txt`：LDS 每个 wave 占 24576 B，写入和 tr16 读回全部在界内，每个 (q, d) 恰好覆盖一次，违规 0 次。
  - 全局下标与 bounds_dqg 已证的 Q 片段下标同形。
  - KV_U2B 和 DQ_U2B 只加了 sched_barrier，不引入新的下标。

**低时钟**（`runs/kb7_prod_gbimg_p{1,2,3}.log`，IMAGE 库，10 个 GEMM，sclk 由 0.5 ms 的 sysfs 轮询线程在 kernel 的 host 窗口内采样）：

| arm | 整个 op 相对 cur | k_dq 相对 cur | 整个 op ms | sclk |
|---|---|---|---|---|
| cur | 1 | 1 | 9.304–9.309 | 1344–1346 |
| r19h | 1.0344 / 1.0330 / 1.0322 | 1.0842 / 1.0830 / 1.0855 | 9.61–9.63 | 1352 |
| c_epi | 1.0001 / 1.0008 / 0.9989 | 1.000 | 9.30–9.31 | 1346 |
| c_k1 | 1.0048 / 1.0039 / 1.0031 | 1.000 | 9.34–9.35 | 1348–1352 |

**外来负载**：09:00:31–约 09:10 UTC，别的用户在 GPU0 上跑了 hipblaslt-bench。与这段时间重叠的 `kb5_prod_blk_p1` 和 `p2` 已排除，并用 `p1r`、`p2r` 重测，arm 顺序相同。两组数据的比值差异在 0.5% 以内。之后的所有进程都在停掉外来负载之后运行。汇总见 `runs/round2_ratios.txt`。

**卡上工作量**：第二轮在 GPU0 上共 18 个进程。每个进程之后都查了 dmesg，没有新的 amdgpu 故障行。唯一出现的 amdgpu 行是 `0002:04:00.0 MES ring buffer full`，那是已宕的 GPU1，按规则忽略。

### 建议的 hint（第二轮；不是 must refactor，而是一条关闭记录）

```
| h70 | closed | k_dq/k_dkdv issue-count follow-ups on the r29 champion are all null or losses; u2n's gain is confirmed at the training clock (-3.2% op, -7.8% k_dq at 1272-1352 MHz) | closed |

## h70 -- lab-kdq round 2 (GPU0, blocked ruler, 3 rotated processes per shape, vs r29 = op/current)
Confirmed: r29 (u2n) vs r19h at the TRAINING operating point (10 x bf16 32768x4096x14336 GEMM
  with the IMAGE hipBLASLt lib before every timed call; sclk 1272-1352 MHz): whole op
  0.9667/0.9680/0.9688, k_dq 0.9224/0.9234/0.9212. At ~1840 MHz: op 0.970, k_dq 0.940.
  Note: k_dq is clock-SENSITIVE in this state (4.05 ms at ~1280 MHz vs 2.98 at ~1950).
  A GEMM burst with the ~/.local hipBLASLt lib runs at ~80 TF/s and does NOT lower the clock.
Dead, do not rebuild (all 0 spill; all gates pass):
  - sched_barrier(0) between the two unrolled k_dq bodies (u2nb): VGPR 991 -> 798 but k_dq
    +20.7%, op +8.0% -- the load clump (h37) again. With VF_Q (u2b) +6%.
  - fma-only softmax in k_dq keeping the scale in dS (u2f): op +2.0%; any VF_Q on top of the
    unroll loses (the u2 result of round 1 holds).
  - dQ epilogue through LDS (g86 ported to k_dqg: 256 buffer_store_b16 -> 32 b128, bitwise):
    prod op 0.9996, low clock 0.9999, proxy k_dq -3% / op -0.5% (within proxy noise). Null.
  - k_dkdv: VF_KV (fma softmax + scale at dK store, loop 659 -> 609 instr) +0.4% at both
    ~1840 MHz and ~1345 MHz; unroll-by-2 (729 -> 644 VGPR, no rotation movs) +7.3%, with
    sched_barrier +8.7%. Cutting VALU/mov slots in k_dkdv's loop does not buy time.
```

---

# 第一轮（GPU3，之后改用 GPU0），基线 r19h


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

## 第二轮新增文件

- `OP/cur`：冠军 r29 的拷贝
- `OP/lab2`：冠军代码加 lab 开关，默认值与冠军一致
- `OP/c_*`：第二轮各 arm 的代码树
- `tools/mkarm2.sh`：生成第二轮 arm
- `tools/step5.sh`、`step5r.sh`、`step6.sh`、`step7.sh`：已运行的卡上脚本
- `tools/kbench.py`：新增 sclk 轮询线程、`sclk_pre` 和 `pre_ms` 字段
- `tools/run1.sh`：新增 `KB_BLASLIB`，用于覆盖 hipBLASLt 库路径
- `runs/val5_*`、`val6_*`、`kb5_*`、`kb6_*`、`kb7_*`：第二轮原始数据；汇总在 `runs/round2_ratios.txt`
- `compile/isa_summary_round2.txt`、`bounds/bounds_epi.{py,txt}`
- `tools/plan.sh`：第一轮留下的计划，其中所有项目都已经在第二轮完成或被取代
