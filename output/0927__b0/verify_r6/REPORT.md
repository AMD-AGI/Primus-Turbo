# r6 冠军（投机 softmax）数值正确性对抗验证 —— B0 GPU3 / fa-g3，2026-09-27

## 结论：正确，未发现缺陷

在 32 类对抗输入 × 8 个 (shape, causal) 组合共 256 个 case 上：0 个 FAIL。r6 与 r4 的 o SQNR 相差最多 0.0004 dB，lse SQNR 相差最多 0.0005 dB。
r4 输出有限的地方，r6 也全部有限。更强的一条证据：r6 与“每个 tile 都强制走慢路径”的变体 `r6_as` 在 **256/256 个 case 上 o 和 lse 都逐位相同**。
所以快路径加 ballot 触发器，与每个 tile 都跑精确 softmax 的结果逐位等价，包括慢路径触发率 99-100% 的输入（stair9/stair30/stair100/ramp1）。
r6 与 r4 本身不逐位相同（126/256 相同），差别来自 round 6 有意改动的 d 累加顺序（`fadd_t` 不再重结合）：最大 |Δlse| 为 1 ULP（lse≈1e6 时为 0.0625），最大 |Δo| 为 0.0156（1 个 bf16 ULP）。
确定性：proxy causal 下 stair9 / bnd_sum / ramp01 / diag_pos 各跑 20 次，全部 20/20 逐位相同。
整个验证期间 dmesg 没有新的 amdgpu 事件（仅有开机前的旧 13468 s 故障记录）。

## 方法

- **待测臂**（同进程运行，输入相同，均为 `arms/` 下的只读拷贝）：
  - `r4`：rounds/004/op。
  - `r6`：job_context/op/current，与 fwd/champion_r6 逐字节相同。
  - `r6_as`：r6，仅把 `SPEC_TRIGGER` 改为 -1.0，每个 tile 都走慢路径（精确 softmax）。
  - `r6_nvs`：r6，仅把 `SPEC_TRIGGER` 改为 3e38。注意：由于 OGT 对 inf 成立，这个变体只在 p 溢出为 inf 时才触发，所以它是“仅 inf 触发”，并非“永不触发”。
  - 编译检查（prod 形状，compile-only）：r4 VGPR 456；r6、r6_as、r6_nvs 均为 VGPR 448，vgpr spill 0，scratch 0（`compile/descriptors_prod.txt`）。r6 的 sgpr_spill_count=60，溢出到 VGPR lane，不产生 scratch，属于冠军本身的属性。
- **参考**：CPU fp64 的精确 softmax（容器内 CPU，不占卡）。o 不做 bf16 截断。toy/short_q/gqa4 全量计算；proxy 取 4 个 q 头的全部行；prod 取 batch{0,3} × 头{0,13,31} × 行[0,128)∪[4032,4160)∪[8064,8192)。
- **判据**：r4 ≥49 dB 的 case 要求 r6 ≥49 dB。任何 case 若 r6 比 r4 差 >1 dB（o 或 lse），或 r4 有限而 r6 出现非有限值，判 FAIL。r4 本身 <49 dB 的记为 PASS(shared<49)，此时只要求 r6 不比 r4 更差。
- **触发率**：由 CPU 行级模拟估计（m 按 r4 的行级规则更新，统计 tile≥1 上任一半行和 > e^7 的比例）。kernel 中 ballot 把同一 wave 的多行耦合在一起，真实 wave 级触发率 ≥ 该估计。另一个旁证是 `r6_nvs`：它与 r6 输出不同的行数，说明慢路径确实改变了这些行的结果。
- **形状**：toy (1,256,256,2,1)、short_q (1,128,512,4,1)、gqa4 (2,256,256,8,2)，causal 与 full 各一；proxy (1,4096,4096,32,8) causal；prod (4,8192,8192,32,8) causal。每个进程只跑一个 (shape, causal)。

## 按输入类型汇总（8 个 shape 组合取最小值；括号内为 prod causal 的值）

| kind | r6 o dB min (prod) | r4 o dB min (prod) | r6 lse dB min | r4 lse dB min | r6 finite | r6==always-exact | est. trigger rate | status |
|---|---|---|---|---|---|---|---|---|
| randn | 50.87 (52.65) | 50.87 (52.65) | 80.97 | 80.97 | Y | 8/8 | 0% | PASS x8 |
| x4 | 42.78 (42.78) | 42.78 (42.78) | 63.43 | 63.43 | Y | 8/8 | 4% | PASS(shared<49) x8 |
| x16 | 29.55 (30.59) | 29.55 (30.59) | 62.66 | 62.66 | Y | 8/8 | 8% | PASS(shared<49) x8 |
| x64 | 21.89 (23.77) | 21.89 (23.77) | 62.98 | 62.98 | Y | 8/8 | 9% | PASS(shared<49) x8 |
| stair9 | 51.54 (52.35) | 51.54 (52.35) | 74.73 | 74.73 | Y | 8/8 | 99% | PASS x8 |
| stair30 | 51.51 (53.02) | 51.51 (53.02) | 74.71 | 74.71 | Y | 8/8 | 100% | PASS x8 |
| ramp02 | 52.14 (52.46) | 52.14 (52.46) | 74.72 | 74.72 | Y | 8/8 | 98% | PASS x8 |
| ramp01 | 51.44 (51.95) | 51.44 (51.95) | 74.74 | 74.74 | Y | 8/8 | 87% | PASS x8 |
| ramp1 | 53.81 (53.86) | 53.81 (53.86) | 74.71 | 74.71 | Y | 8/8 | 100% | PASS x8 |
| late_jump | 58.97 (61.75) | 58.97 (61.75) | 74.22 | 74.22 | Y | 8/8 | 2% | PASS x8 |
| periodic_jump30 | 56.31 (60.26) | 56.31 (60.26) | 74.54 | 74.54 | Y | 8/8 | 5% | PASS x8 |
| jump88 | 50.86 (52.80) | 50.86 (52.80) | 75.06 | 75.06 | Y | 8/8 | 3% | PASS x8 |
| jump89 | 50.91 (52.87) | 50.91 (52.87) | 75.05 | 75.05 | Y | 8/8 | 3% | PASS x8 |
| jump1e3 | 51.05 (52.90) | 51.05 (52.90) | 74.74 | 74.74 | Y | 8/8 | 3% | PASS x8 |
| stair100 | 51.49 (52.93) | 51.49 (52.93) | 74.71 | 74.71 | Y | 8/8 | 100% | PASS x8 |
| stair1e4 | 25.23 (25.23) | 25.23 (25.23) | 74.70 | 74.70 | Y | 8/8 | 100% | PASS x4, PASS(shared<49) x4 |
| huge_off1e4 | 50.33 (52.35) | 50.33 (52.35) | 74.71 | 74.71 | Y | 8/8 | 0% | PASS x8 |
| huge_off1e6 | 24.79 (26.85) | 24.79 (26.85) | 74.70 | 74.70 | Y | 8/8 | 0% | PASS(shared<49) x8 |
| neg1e4 | 50.23 (52.42) | 50.23 (52.42) | 74.71 | 74.71 | Y | 8/8 | 0% | PASS x8 |
| neg1e6 | 24.85 (26.88) | 24.85 (26.88) | 74.71 | 74.71 | Y | 8/8 | 0% | PASS(shared<49) x8 |
| neg_late | 51.43 (52.16) | 51.43 (52.16) | 78.63 | 78.63 | Y | 8/8 | 0% | PASS x8 |
| neg_early | 50.85 (52.98) | 50.85 (52.98) | 74.71 | 74.71 | Y | 8/8 | 4% | PASS x8 |
| dom_first | 173.80 (173.80) | 173.80 (173.80) | 74.31 | 74.31 | Y | 8/8 | 0% | PASS x8 |
| dom_mid | 63.18 (65.32) | 63.18 (65.32) | 74.27 | 74.27 | Y | 8/8 | 2% | PASS x8 |
| dom_last | 52.09 (53.01) | 52.09 (53.01) | 74.29 | 74.29 | Y | 8/8 | 1% | PASS x8 |
| diag_pos | 100.53 (100.53) | 100.53 (100.53) | 70.27 | 70.27 | Y | 8/8 | 4% | PASS x8 |
| diag_neg | 48.00 (49.66) | 48.00 (49.66) | 69.17 | 69.17 | Y | 8/8 | 0% | PASS x3, PASS(shared<49) x5 |
| bnd_sum | 51.89 (54.34) | 51.89 (54.34) | 61.57 | 61.57 | Y | 8/8 | 45% | PASS x8 |
| bnd_one7 | 50.88 (55.24) | 50.88 (55.24) | 54.53 | 54.53 | Y | 8/8 | 38% | PASS x8 |
| bnd_one8 | 53.87 (53.87) | 53.87 (53.87) | 55.30 | 55.30 | Y | 8/8 | 63% | PASS x8 |
| ext_pos1e30 | -inf (-inf) | -inf (-inf) | -inf | -inf | N (r4 same) | 8/8 | 0% | INFO x8 |
| ext_neg1e32 | 0.00 (0.00) | 0.00 (0.00) | -inf | -inf | N (r4 same) | 8/8 | 0% | INFO x8 |

每个 case 的完整表格（256 行，含 r6/r4 的 o dB、lse dB、逐位比较、|Δlse|、r6_nvs 结果、触发率）见 `results.md` / `results.json`。

## 输入类型说明

logit 结构通过 head dim 的第 0/1 维注入，c=128^0.25，使 logit 增量恰为 A_i·B_j。各类型如下：

- 随机缩放：randn；x4/x16/x64（q、k 各乘 f）。
- 行最大值随 KV tile 单调上升，每个 tile 都让过时的 m_prev 失效：
  - stair9/30/100/1e4：每个 tile +9/+30/+100/+1e4；stair100 以上在快路径中 exp2 溢出为 inf。
  - ramp02/ramp01/ramp1：每个 key +0.2/+0.1/+1。ramp01 的特点是每 tile +6.4：r4 仍走延迟路径，但 r6 的半行和会越过 e^7。
- 最大值在行内后段跳变：
  - late_jump：最后一个 key +40，倒数第 70 个 key +20。
  - periodic_jump30：每 256 个 key 出现一次 +30，3/4 处 +60。
  - jump88/jump89/jump1e3：行中点跳 +88（e^88 仍为有限值）/ +89.5（溢出为 inf）/ +1e3。
- 极大、极小 logit：
  - huge_off1e4/1e6：全体 logit 约 +1e4 / +1e6。
  - neg1e4/1e6：全体约 -1e4 / -1e6。
  - neg_late：tile 0 正常，其余 tile 为 -1e4，全部下溢为 0。
  - neg_early：tile 0 为 -1e4，后续 tile 回到 0，快路径中 exp 溢出。
- 单个主导 key：dom_first / dom_mid / dom_last（位置 0 / 中间 / 最后，+30）。
- 对角线：diag_pos 中 q_i ∝ +k_{i+shift}，对角 logit 约 +37。diag_neg 中 q_i ∝ -k_{i+shift}，causal 首行唯一可见的 key 是最低值。
- 触发边界：
  - bnd_sum：tile≥1 的 64 个 key 全部取 δ∈[3.40,3.70]，使半行和 32e^δ 跨过 e^7（δ*=3.534）。
  - bnd_one7：每 tile 一个 key 取 δ∈[6.8,7.2]，跨过触发阈值 e^7。
  - bnd_one8：同上，δ∈[7.8,8.2]，跨过 r4 的 rescale 阈值 8。

## r4 与 r6 共有的限制（不是 r6 缺陷，不计分，列为 INFO / shared）

- **ext_neg1e32**：logit 约 -1e32，低于 kernel 的 running-max 种子 BIG_NEG=-1e30。所有 tile 的 p 都下溢为 0，d=0，结果 o=0、lse=-inf，参考值为有限。r4、r6、r6_as 行为完全一致，源于基线 kernel 的种子设计。
- **ext_pos1e30**：logit 约 +1e30。fp32 的 fma(s·log2e − m·log2e) 在这个量级抵消误差约 1e23，结果为 NaN。r4 与 r6 相同。
- **o <49 dB 的 case**：x4/x16/x64、huge_off1e6、neg1e6、stair1e4（short_q/proxy/prod），以及 diag_neg（48.0-49.6 dB）。原因是近 one-hot softmax 下 bf16 P 的量化，以及 1e6 量级 logit 的 fp32 间距（0.06）。r6 与 r4 的差 ≤0.0004 dB。
- `r6_nvs`（仅 inf 触发）在 ramp02/ramp01/stair30/x16/x64/jump88(prod) 等 case 上产生 NaN（-inf dB），在 late_jump/dom_mid/diag_pos 上比 r6 低 5-140 dB。这说明套件确实击中了 e^7 触发器负责的区域，而 r6 在这些区域全部正确。

## 结构审查（读代码）

- 慢路径重读的是 `k_curr`/`v_curr` 槽。t+1 预取写入另一个槽；t+2 的预取要等 tile t+1 顶部的 `gpu.barrier()` 之后才能发出，慢路径 wave 尚未到达这个 barrier，所以不存在 WAR 竞争。O epilogue 使用非当前槽，同样安全。
- 触发器是充分条件：如果不触发，每个 p ≤ 半行和 ≤ e^7，所以 s−m_prev ≤ 7 < 8，r4 在这种情况下也必然走延迟路径，m 与 p 完全相同。p=inf 时 OGT 成立，会触发。首个 tile 由种子 -1e30 保证触发。

## 文件

- 输入生成：`adv_inputs.py`
- 卡上运行（每进程一个 shape）：`card_run.py`、`run_card.sh`、`run_all.sh`
- CPU fp64 参考与触发模拟：`ref_eval.py`
- 汇总：`summarize.py`
- 确定性：`determinism.py`、`logs/determinism_proxy.log`
- 结果：`results.md`（256 行完整表）、`results.json`、`kind_table.md`、`data/<tag>/{card_summary,eval}.json`
- 日志：`logs/card_*.log`、`logs/eval_*.txt`、`logs/dmesg_after_*.txt`、`logs/run_all.log`
- 编译资源：`compile/descriptors_prod.txt`
- 待测臂拷贝：`arms/{r4,r6,r6_as,r6_nvs}`（仅 r6_as、r6_nvs 改了 `SPEC_TRIGGER` 一行）
- 采样行的 .pt 中间文件（2.2 GB）已删除，重跑 `run_all.sh` 即可重新生成。
