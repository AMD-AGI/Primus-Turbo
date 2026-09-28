# bwd lab：r19 / r19h 和冠军 r20 对比（hint h67）。GPU3 / fa-g3，2026-09-28

## 结论

**r19h 应该取代 r20 成为冠军。** 判定为 **win**。

- prod 快 1.73%，proxy 快 2.6%，fast 快约 1.4%。三个 shape 上都超过 min_gain 0.7%，fast/proxy 也没有跌出 band。
- 正确性：dq/dk/dv 的 SQNR 和 r20 相同，都在 52.5 dB 左右。确定性门槛全部通过。三个 shape 上 r19h 的输出都和 r19 **逐位相同**，说明 clamp 确实不改变数值。
- 编译：0 spill，0 scratch。k_dkdv 用 724 个 VGPR，r20 是 904。
- 按 validation 的计速方式（candidate 和 beat 同进程，分块计时），r19h 对 ASM 的几何平均是 **1.022x**，r20 是 **0.9999x**。也就是说，这张卡上 r20 刚好差一点过不了 speed gate，r19h 能过。

## 1. 测量设置

- harness：job 的 `job_context/op/benchmark.py`，BLOCKED 版（每轮 lead 4 + block 9，轮次按回文顺序），在拷贝 `oe/artifacts/job/job_context/op/` 里运行；`refcache` 是指向 job 的只读 symlink。
- 环境：`TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250`，卡是 fa-g3（物理 GPU 3），每个进程只跑一个 shape。
- 每个进程有 4 个 arm：`current`（r20）、`current_copy`（r20 的 A/A 对照）、`r019`、`r19h`。arm 的顺序在进程之间轮换。
  - prod 和 proxy 各 3 个进程，`--iters 101`（实际 108）。
  - fast 共 6 个进程：3 个 `--iters 101`，另外 3 个 `--iters 505`（实际 513）。
- tree md5：current/current_copy `5c499890`（= rounds/020/op = job op/current），r019 `d7abe9e9`，r19h `a791a047`，beat `2802a918`。
- 每个卡上进程结束后都查了 dmesg：共 27 个进程，新增 amdgpu 故障行都是 0（0002:04:00.0 已忽略）。

## 2. 第 1 步：r19 对 r20

比值 = arm 的 median / 同一进程里 current 的 median。小于 1 表示比 r20 快。

| shape | A/A current_copy | r019 | r19h |
|---|---|---|---|
| prod p1/p2/p3 | 1.0007 / 0.9996 / 1.0004 | 0.9835 / 0.9830 / 0.9817 | 0.9838 / 0.9826 / 0.9820 |
| **prod 均值** | 1.0002 | **0.9827** | **0.9828** |
| proxy p1/p2/p3 | 0.9965 / 0.9996 / 0.9957 | 0.9718 / 0.9684 / 0.9677 | 0.9707 / 0.9780 / 0.9701 |
| **proxy 均值** | 0.9973 | **0.9693** | **0.9729** |
| fast p1..p6 | 1.0063 / 0.9982 / 0.9985 / 0.9970 / 0.9985 / 0.9809 | 0.9859 / 0.9904 / 0.9846 / 0.9808 / 0.9814 / 0.9706 | 0.9851 / 0.9848 / 0.9993 / 0.9874 / 0.9836 / 0.9684 |
| **fast 均值** | 0.9966 | **0.9823** | **0.9848** |

fast 的 p6 里 current 本身偏慢（A/A 读到 0.981）。如果改用两份 r20 的平均值做分母（`ratios_pair.txt`），fast 上 r019 是 0.9840（0.9800–0.9913），r19h 是 0.9864（0.9777–1.0000）。结论不变。

- prod：和 ruler 在 GPU0 上的结果（0.9827）完全一致。r19 在 prod 上快 1.7%，3 个进程的离散度 0.2%，A/A 在 ±0.07% 以内。
- proxy：r19 快 3.1%，离散度 0.6%，A/A 在 0.43% 以内。
- fast：r19 快 1.6–1.8%，但 fast 的 A/A 本身有 ±0.6%（有一次到 1.9%），所以这里只能说"明确不慢，大约快 1.5%"。
- round 20 当年用三个 shape 的几何平均选了 r20。在分块计时下，r19 在三个 shape 上都比 r20 快。

## 3. 第 2 步：构建 r19h

**来源**：`r19h` = `rounds/019/op` 加上 h33 的三处修复。修复的写法照搬 `rounds/024/op`，由 `diff -r rounds/020/op rounds/024/op` 提取，完整 diff 见 `r19h.diff`。

- **defect a**（`impl.py`）：原来的 `assert sq % 32` 改成 `assert sq % _k.BLOCK_Q == 0`（BLOCK_Q = 64），注释照搬。
- **defect b**（`kernels.py`，k_dq_sp 的 `g_dq` extent）：改成 Int64 连乘，和 r24 逐字相同。
- **defect c**（`kernels.py`，k_dkdv 预取）：
  - 加入 `_clampqt` helper，和 r24 逐字相同。
  - r19 的预取深度是 1（`jj = ii+1`；r20 是 2，`kk = ii+2`），所以用 r24 同样的式子把 `jj` clamp 到 `n-1`：`jj = (jj < n).select(jj, n - 1)`。
  - prologue 在 r19 里只发一条 `_ldqd`（r20 发两条），split 和非 split 两条路径上都套 `_clampqt`。r24 里 `_pc1` 那一条在 r19 中不存在，因此没有移植。

**compile-only**（fa-g3，不上卡，flydsl 0.3.2，prod 参数，7 个 kernel 全部编译；`compile/isa_summary.txt`）：

| kernel | r019 VGPR | r19h VGPR | r20 VGPR | spill/scratch（三个 arm） |
|---|---|---|---|---|
| k_dkdv | 724 | 724 | 904 | 0 / 0 |
| k_dkdv_sp | 724 | 724 | 910 | 0 / 0 |
| k_dq | 960 | 960 | 960 | 0 / 0 |
| k_dq_sp | 968 | 968 | 968 | 0 / 0 |
| delta / redsp / redsp_q | 40 / 65 / 41 | 相同 | 相同 | 0 / 0 |

- r19 和 r19h 之间，k_dq、delta、redsp、redsp_q 的 ISA 字节完全相同。
- k_dkdv、k_dkdv_sp、k_dq_sp 的 WMMA/VMEM/LDS 条数不变，只多了少量 SALU（s_min/s_max/s_cselect）和几条地址计算 VALU（`isa_mnemonic_diff.txt`）。k_dkdv 主循环 .LBB0_8 只多一行。

**CPU 边界检查**（`bounds/bounds_clamps.py` 和 `.txt`）：

- 覆盖 `ut/common.py` 里的全部 9 个 shape，causal 和 non-causal 都算，按 impl 的规则推出 nsp。
- 方法：逐个枚举 workgroup 和 split，检查每一次 Q/dO 预取的 query pair 是否落在 [0, nqt2-1] 内。
- 结果：共 194,624 次预取。r19 越界 1,580 次（prod 256 次，都是最后一次迭代预取了不存在的 pair，被 descriptor 挡住）；r19h 越界 0 次。
- 所有真正会被消费的预取，clamp 前后的下标都相同（nonneutral = 0），所以 clamp 不改变数值。
- 9 个 shape 都满足 `sq % 64 == 0`。在 fast 的 nsp_q = 8 下，dq extent 不会发生 Int32 溢出；prod 走的是 k_dq（nsp_q = 1），defect b 在那里本来就是潜在问题。

**正确性和确定性**（`runs/val_{fast,proxy,prod}.log`，每个 shape 一个进程，用 `op/lab_validate.py`，门槛和 validation.py 相同：NaN 预填，先查 isfinite 覆盖，再算 SQNR 对 op/eager，≥ 50 dB）：

| shape | arm | dq / dk / dv SQNR | dk/dv 确定性 | dq 逐次 SQNR | 和 r19h 逐位对比 |
|---|---|---|---|---|---|
| fast | r19h | 52.61 / 52.65 / 52.83 | 200 次逐位相同 | 逐位相同 | -- |
| fast | r019 | 52.61 / 52.65 / 52.83 | 200 次逐位相同 | 逐位相同 | **dq/dk/dv 都逐位相同** |
| fast | r20 | 52.61 / 52.64 / 52.84 | 200 次逐位相同 | 逐位相同 | dk 81.2 dB，dv 77.7 dB |
| proxy | r19h | 52.52 / 52.57 / 52.67 | 100 次逐位相同 | 逐位相同 | -- |
| proxy | r019 | 相同 | 100 次逐位相同 | 逐位相同 | **逐位相同** |
| proxy | r20 | 52.52 / 52.57 / 52.67 | 100 次逐位相同 | 逐位相同 | dk 80.3 dB，dv 81.0 dB |
| prod | r19h | 52.56 / 52.60 / 52.71 | 50 次逐位相同 | 逐位相同 | -- |
| prod | r019 | 相同 | 50 次逐位相同 | 逐位相同 | **逐位相同** |
| prod | r20 | 52.56 / 52.60 / 52.71 | 50 次逐位相同 | 逐位相同 | dk 78.9 dB，dv 80.5 dB |

r20 的 dB 和 job 的 rounds/024–026 gate.log 完全相同（52.61/52.64/52.84 等），说明 lab 用的 refcache 参考没有问题（原因见第 6 节第 1 条）。r20 的 dk/dv 和 r19 不逐位相同，这是预期的：r20 的 g61 chain split 改变了 fp32 的累加结合顺序。

**计时**：见第 2 节表格的 r19h 一列。r19h 和 r019 同进程直接对比：prod 1.0001（0.9996–1.0003），proxy 1.0037（0.9989–1.0099），fast 1.0026（0.9944–1.0148），都在各自 shape 的 A/A 噪声以内。clamp 的开销在 prod 上测不出来。

## 4. 相对 ASM（beat）

**(a) beat 单独一个进程**（`runs/beat_*.log`），FlyDSL 取第 2 节各进程的平均 TF/s。注意这是跨进程比较：

| shape | beat ms / TF/s | r20 TF/s（% ASM） | r19h TF/s（% ASM） |
|---|---|---|---|
| fast | 0.0890 / 60.4 | 99.5（164.7%） | 101.1（167.4%） |
| proxy | 0.4595 / 748.0 | 574.1（76.8%） | 590.1（78.9%） |
| prod | 6.722 / 818.0 | 629.2（76.9%） | 640.2（78.3%） |
| 几何平均 | | 0.991x | 1.011x |

**(b) 按 validation 的方式**：candidate 在前、beat 在后，同一进程，每个 shape 一个进程，默认 51 次迭代（`gate_style.txt`，`runs/g_*`）：

| shape | r20 x beat | r19h x beat |
|---|---|---|
| fast | 1.6951 | 1.7350 |
| proxy | 0.7681 | 0.7880 |
| prod | 0.7679 | 0.7809 |
| **几何平均** | **0.9999** | **1.0221** |

GPU3 上同进程 prod 的 beat/r20 是 0.768。ruler 在 GPU0 上测的是 0.7878，两张卡相差约 2.5%，所以不同卡之间的 x-beat 不能直接比较。本报告的判定只依赖同进程比值。

## 5. 判定

| arm | prod | proxy | fast | VGPR（dkdv） | 最低 dB | 判定 |
|---|---|---|---|---|---|---|
| r019 | -1.73% | -3.07% | -1.8% | 724 | 52.52 | win，但带着 h33 defect c/a/b |
| **r19h** | **-1.72%** | **-2.71%** | **-1.5%** | **724** | **52.52** | **win，建议作为新冠军** |

prod 排第一，r19h 的增益 1.72% 远大于 min_gain 0.7%，3 个进程离散度 0.2%，A/A 在 ±0.07% 以内。proxy 和 fast 也都更快，没有 band 问题。r19h 同时保留了 r24 那三处 h33 修复。

## 6. 发现的问题

1. **job 的 validation 从 round 24 起在卡上重算 fp32 参考。** `ut/common.py` 在 refcache 建成（09-22）之后被改过，sha 从 `988c14caed5d9a80` 变成 `d441e55aba5c0b89`。因此 `load_reference` 对三个 shape 都报 "IGNORED ... recomputing"（rounds/024、025、026 的 gate.log），prod 每一轮都会在卡上跑 fp32 reference。按 hint 记录，这正是 09-22 那次需要断电重启的触发条件。
   - `make_inputs` 没有变：重算得到的 dB 和缓存时期逐位一致（52.61/52.64/52.84 …）。
   - 建议 operator 二选一：更新 refcache 的 provenance（只改 common_sha），或者让 load_reference 接受旧的 sha。lab 里采用了后者（`lab_validate.py`），全程没有在卡上跑 fp32 reference。
2. fast 的 A/A 离散度大（最大 1.9%），这个 shape 的单进程比值不能用来判定 1% 级别的差异，至少需要 6 个进程。

## 7. 建议的 hint（operator 下发，lab 不写 hint.md）

```
| h68 | must refactor | Adopt r19h verbatim as the new champion: r19 + the three h33 fixes; prod -1.72%, proxy -2.7%, fast -1.5% vs r20, bitwise = r19 | open |

## h68 -- Adopt r19h (round 19 + h33 fixes) verbatim as op/current

Source tree (copy it, do not re-derive it):
  /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/lab-bwd-r19/oe/artifacts/job/job_context/op/r19h
  tree_md5 (benchmark.py's) a791a047. It is rounds/019/op plus exactly the round-24 h33 fixes
  (diff: lab-bwd-r19/r19h.diff): (a) impl.py assert sq % BLOCK_Q; (b) k_dq_sp g_dq extent in
  Int64; (c) k_dkdv _clampqt helper, prologue _ldqd pair clamped on the split and non-split
  paths, and qloop_full's prefetch jj = ii+1 clamped to n-1 (r19 has prefetch depth 1, so
  round 24's `_pc1` second prologue load has no counterpart and is not added).
Evidence (GPU3, blocked harness, one shape per process, rotated arm orders, A/A in each):
  r19h / r20 median ratio  prod 0.9838/0.9826/0.9820 (A/A <= 0.07%)
                           proxy 0.9707/0.9780/0.9701 (A/A <= 0.43%)
                           fast 6 processes mean 0.985 (A/A up to 1.9% -- fast is noisy)
  r19h vs r19 same process: prod 1.0001, i.e. the clamps cost nothing measurable.
  validation-style x beat geomean: r19h 1.022, r20 0.9999 (same card, same method).
  Correctness: dq/dk/dv 52.61/52.65/52.83 (fast), 52.52/52.57/52.67 (proxy),
  52.56/52.60/52.71 (prod) dB; dk/dv bitwise x200/x100/x50, dq bitwise run to run;
  outputs BITWISE IDENTICAL to r19 on all three shapes (clamps are value-neutral).
  Compile: 0 spill / 0 scratch everywhere; k_dkdv 724 VGPR (r20: 904).
  CPU bounds: 194,624 prefetches enumerated over all 9 UT shapes x causal modes: r19 has
  1,580 out-of-range query-pair prefetches, r19h 0; every consumed prefetch unchanged.
Action: install the tree as op/current (and the next round's parent), run validation.py once
to record the gate, and treat r20's g61 chain split / g62 prefetch-depth-2 as NOT part of the
champion -- on the blocked ruler they cost 1.7-3% at every shape. Any future lever is measured
against r19h, not r20.
```

## 文件

- `REPORT.md`：本报告
- `r19h.diff`：r19 到 r19h 的完整 diff
- `oe/artifacts/job/job_context/op/{current,current_copy,r019,r19h,beat,eager,ut}`：代码树。`benchmark.py` 是 job 原件；`lab_validate.py` 是 lab 的门槛脚本；`refcache` 是 symlink，未提交
- `compile/compile_bwd.py`、`compile/{r019,r19h,current}.log`、`compile/isa_summary.txt`、`compile/isa_mnemonic_diff.txt`：compile-only 的脚本、日志和汇总。`compile/dump/` 未提交
- `bounds/bounds_clamps.py`、`bounds/bounds_clamps.txt`：CPU 边界检查
- `runs/val_*`：正确性和确定性检查的原始输出
- `runs/t_*`：计时，r19 / r19h 对 r20
- `runs/beat_*`：beat 单独一个进程
- `runs/g_*`：validation 方式的 candidate + beat 同进程计时
- `.dmesg` / `.dmesg.bad`：每个进程的 dmesg，全部为空
- `ratios.txt`、`ratios_pair.txt`、`tflops.txt`、`gate_style.txt`、`batch_timing.out`：汇总
- `run1.sh`、`batch_timing.sh`、`ratios.sh`：工具脚本
