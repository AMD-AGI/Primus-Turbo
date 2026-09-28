# lab2 上卡测量 — GPU 2 / fa-g2，2026-09-27 13:14–13:43

方法：每个 arm 跑 3 个独立进程，每个进程只跑 prod 一个 shape，只有 cand 和 champ 两个 arm（没有 beat）。顺序依次为 cand,champ / champ,cand / cand,champ。
计时用 job 自带的 `benchmark.measure()`：回文顺序，n=101，取中位数，每个 arm 先连续预热 8 s，每次调用前刷 256 MB L2。
同一进程内先做正确性检查（`gates.check_correctness`：NaN 预填，fp32 参考取自 refcache，门限 49 dB），再逐位比较 cand 和 champ。
ratio = champ_ms / cand_ms。判定规则：win 要求 mean >= 1.005 且每个进程都 > 1.0；loss 为 mean <= 0.995；其余为 null。
champ = `job_context/op/current`（只读）。驱动脚本 `measure/run_ab.py`，串行调度 `measure/runner.sh`，汇总 `measure/summarize.py`。

**dmesg：45 个进程跑完后 dmesg 都是 0 条新增行，所有进程 rc=0。** 每个进程的记录在 `measure/<arm>_pN.{log,json,dmesg}`。

| arm | lever | p1 | p2 | p3 | mean | 判定 | 正确性 |
|---|---|---|---|---|---|---|---|
| ctrl (A/A) | L4 | 0.9992 | 0.9995 | 0.9988 | 0.9992 | null（噪声底） | 与 champ 逐位一致 |
| itilp | L24 | 0.9336 | 0.9349 | 0.9334 | **0.9339** | loss（-6.6%） | 逐位一致 |
| nosm2 | L24 | 0.9722 | 0.9730 | 0.9742 | 0.9731 | loss | 逐位一致 |
| mmc_nopm | L24 | 0.9774 | 0.9777 | 0.9765 | 0.9772 | loss | 逐位一致 |
| mmc | L24 | 0.9794 | 0.9808 | 0.9795 | 0.9799 | loss | 逐位一致 |
| L22_L23 | L22_L23 | 0.9883 | 0.9841 | 0.9869 | 0.9864 | loss | 逐位一致 |
| ilp | L24 | 0.9877 | 0.9862 | 0.9857 | 0.9866 | loss | 逐位一致 |
| L23 | L22_L23 | 0.9869 | 0.9880 | 0.9882 | 0.9877 | loss | 逐位一致 |
| nopm | L24 | 0.9933 | 0.9945 | 0.9935 | 0.9938 | loss（-0.6%） | 逐位一致 |
| nosink | L24 | 0.9978 | 0.9994 | 1.0010 | 0.9994 | null | 逐位一致 |
| L22 | L22_L23 | 0.9992 | 1.0009 | 0.9991 | 0.9997 | null | 逐位一致 |
| bmajor | L4 | 0.9993 | 0.9991 | 1.0020 | 1.0001 | null | 逐位一致 |
| gate | L5 | 1.0006 | 1.0004 | 1.0002 | 1.0004 | null（prod 上跑的就是 champ 内核） | 逐位一致 |
| rot | L5 | 1.0012 | 1.0006 | 1.0006 | 1.0008 | null | 逐位一致 |
| spread | L4 | 1.0028 | 1.0027 | 1.0036 | 1.0031 | null（< 0.5%，见下） | 逐位一致 |
| clause32 / clause4 / relaxocc | L24 | — | — | — | — | 未上卡（ISA 与 champ 逐字节相同，属 no-op） | — |

正确性：所有 arm（包括 champ）在 prod seed 0 下 o 为 50.83 dB，lse 为 89.20 dB，o 全部 finite，参考来源 ref=refcache。
每个进程里 cand 的 o 和 lse 都与 champ 逐位相同（max abs diff 0）。VGPR/spill 取 lab 的编译数据：L22/L23/L22_L23/L24 各 arm 为 456（itilp 439，ilp 462），bmajor/spread/ctrl/gate 为 456，rot 为 454/456 双峰；所有 arm spill 0，scratch 0。

要点：
- **本轮没有 win。** L24 调度选项全部是 loss 或 null。静态指标最好看的 itilp（VGPR 456->439，SGPR spill 为 0，loop dscnt 10->3）在卡上是最大的 loss，-6.6%，再一次说明静态 ISA 指标不能用来排名（rule 8）。
- L22 单独用时为 null，L23 单独用时为 -1.2%，两者合用也是 -1.4%。所以 loss 来自 L23，即去掉 `_drain_barrier` 外面那对 sched_barrier(0)；L22 的 dscnt 分级在 prod 上不起作用。
- nosm2（关掉 ENABLE_SCHED_MODE2）为 -2.7%，说明 champ 的 expert-scheduling-mode 选择是对的。
- L4：spread（反局部性对照）三个进程都在 +0.3% 左右，ctrl A/A 为 -0.08%，两者差约 0.4%，仍在 0.5% 噪声线以内，判 null。spread 的 3 个进程里 champ 本身慢了约 2%（1.52 ms 对比其它进程的 1.49 ms），sclk 起点也更低，当时有邻卡负载。bmajor 为 null。结论：prod 上 XCD 分组局部性不影响速度。
- L5：prod 上 pad=0，rot 和 gate 都只增加索引运算，结果符合预期，为 null（+0.08% / +0.04%）。这个 lever 的收益只可能出现在 pad!=0 的 shape 上，prod 测不到。
