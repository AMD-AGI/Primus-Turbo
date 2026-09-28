# fwd r13 关掉投机（r13ns）：op 级三种条件 A/B + e2e（2026-09-28，GPU0 / fa-g0）

## 0. 结论

**判定：win（按训练工作点和真实数据）；在 job 的 randn 尺子上是 loss（prod −4.8%）。建议 fwd job 直接采纳 r13ns。**

- **r13ns 是什么**：fwd job 第 13 轮冠军 `op/current`，只把两处 `SPEC_STALE_MAX = True` 改成 `False`，其余不变。
  - 改动位置：`m32x8`（prod/proxy 走这条）和 `m32x2`（小 grid，fast 走这条）。
  - `m16x8` 里没有投机开关，而且 impl 已经不调用它。
  - 没有找到别的投机开关。`ENABLE_DEFER_RESCALE` 是 deferred rescale，不是投机，保持不变。
- **正确性**：
  - job 的 gate 全部通过：o ≥ 49 dB，lse 全覆盖，fast 上 200 次确定性 200/200。
  - 在 randn（全部 gate 形状）和 6 份真实 dump 上，**输出与 r13 逐位相同**。
- **op 级 prod，同进程比较，3 个进程轮换 arm 顺序**：

| 条件 | r13ns / r13 | r13 / ASM | r13ns / ASM |
|---|---|---|---|
| (i) randn，分块（job 的尺子） | **1.048**（慢 4.8%） | 1.030–1.033 | 1.080–1.084 |
| (ii) 真实 dump，分块 | **0.846–0.919**（快 8–15%） | 1.17–1.29 | 1.073–1.093 |
| (iii) 真实 dump，紧跟 GEMM 突发（训练工作点） | **0.774–0.880**（快 12–23%） | 1.44–1.66 | 1.253–1.288 |

- **e2e，32 层，62 步**：
  - 训练中 r13 的 fwd 每步 50.3 → 55.3 → 58.6/61.0 ms，随训练步增长，因为投机触发得越来越多。
  - r13ns 基本不变：48.3 → 48.9 → 49.3 ms。到第 50–60 步，每步省 **9–12 ms**。
  - 同进程直接配对 r13ns/r13 的单步时间比是 **0.9966**（5 组全部 <1，0.9949–0.9977，并且随步数下降）。
  - fly/ASM 的配对比从 1.0447–1.0454（r13）降到 1.040–1.041（r13ns）。
  - fwd 只占一步的约 3%，所以 e2e 的收益是 0.3–0.5%，而且步数越多收益越大。
- **剩下的差距**：r13ns 在训练里仍然比 ASM 慢，fwd 每步 49 ms 对 39 ms。原因是对时钟敏感（profile §2.1）。这是下一步要做的，不在本 lab 范围内。

## 1. 分支与编译（交付 1）

- 来源：`job_context/op/current`，06:05 拷贝，与 `rounds/013/op` 逐文件相同；本报告写完时 job 的 current 仍然没变。
  - 拷贝：`arms/fwd_r13/`，md5 在 `arms/fwd_r13.md5`。
- r13ns：`arms/fwd_r13ns/`。
  - 相对 r13 的 diff 在 `arms/fwd_r13ns.diff`，只有 2 行。
  - md5 在 `arms/fwd_r13ns.md5`；清单文件本身的 md5 是 `2a4a887bc49b50034c16573a12324331`。
  - 改动的两个文件：
    - `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`：`370769c94c0f951655f66e10c39947d8`
    - `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x2.py`：`b09721562d696d5c1edc7eba29857a88`
- 只编译检查：`tools/compile_arm.py`、`tools/build_all.sh`，结果在 `isa/isa_table.txt`。16 个配置 = 2 个 arm × {m32x8, m32x2} × {gqa4 causal, gqa4 non-causal, gqa1, gqa2}。
  - 全部 **vgpr_spill 0，scratch 0**。
  - m32x8 non-causal 在 r13 和 r13ns 上都有 4 个 SGPR spill，spill 到 VGPR lane，不是 scratch。

| kernel（gqa4 causal） | r13 VGPR / 指令数 | r13ns VGPR / 指令数 |
|---|---|---|
| m32x8（prod） | 454 / 5024 | 456 / **4130** |
| m32x2（fast） | 348 / 4779 | 350 / **4147** |

- **正确性**（`tools/gate.py` 调用 job 的 `gates.py`，用的是拷贝，见 `harness/`；refcache 的来源校验一致）：
  - fast / proxy / prod（causal）：o 51.23 / 50.89 / 50.83 dB，lse 85–89 dB。
  - 7 个边界形状 × 两种 mask：o 49.82–51.86 dB。
  - r13 在每一项上的读数都与 r13ns 完全相同，逐位一致。
  - 确定性：fast 上 200/200 逐位相同。
  - 日志：`runs/gate_edges_fast.log`、`runs/gate_proxy.log`、`runs/gate_prod.log`。
- **真实 dump**（`tools/realcheck.py`，`runs/realcheck.log`），第 0/1/2/8/16/31 层：
  - r13ns 与 r13 逐位相同，max|Δo| = max|Δlse| = 0；
  - 对 fp32 参考（batch 0，head 0/13/31）：o 52.3–54.9 dB。
  - ASM 对参考是 56–61 dB；fly 对 ASM 是 52.7–56.6 dB。

## 2. op 级 A/B（交付 2）

- 形状：prod（4×8192，Hq 32 / Hkv 8，causal）。
- 每个条件跑 3 个进程，arm 顺序依次是 p1 = r13,r13ns,asm；p2 = r13ns,asm,r13；p3 = asm,r13,r13ns。
- 每个进程之后都查了 dmesg，全部干净。
- 汇总在 `ratios_per_process.txt` 和 `ratios.txt`。
- 三种条件：
  - (i)：job 自己的 `benchmark.py`，用的是拷贝，只把 TOOLS 路径改成绝对路径。分块 9+4，L2 flush，101 次。
  - (ii)：`tools/ab.py AB_COND=blk`，同样的分块方法，每个 arm、每组输入 108 次。
  - (iii)：`tools/ab.py AB_COND=gb`，与 profile 的 replay 相同。每次计时前做一次 GEMM 突发：10 次 32768×4096×14336，镜像 hipBLASLt，约 25 ms。每个 arm、每组输入 20 次。
- **坑**：impl 的 `_env.py` 会把 `HIPBLASLT_TENSILE_LIBPATH` 改指向宿主库。宿主库上 GEMM 突发慢约 20 倍，时钟压不下来，(iii) 就会读成 r13/asm 0.85。ab.py 在加载 arm 之后把镜像库的环境变量恢复回来，profile/replay.py 也是这么做的。

**(i) randn 分块，job 的尺子，ms（p1 / p2 / p3）**

| arm | prod | proxy | fast |
|---|---|---|---|
| r13 | 1.4795 / 1.4799 / 1.4796 | 0.0886 / 0.0881 / 0.0874 | 0.0133 / 0.0135 / 0.0135 |
| r13ns | 1.5515 / 1.5512 / 1.5507 | 0.0909 / 0.0926 / 0.0896 | 0.0132 / 0.0133 / 0.0132 |
| ASM | 1.4317 / 1.4336 / 1.4363 | 0.0835 / 0.0836 / 0.0840 | 0.0141 / 0.0142 / 0.0142 |

- r13ns/r13：prod **1.0487 / 1.0482 / 1.0481**，proxy 1.026 / 1.051 / 1.026，fast 0.992 / 0.988 / 0.982。
- 在 job 的尺子上，geomean 约慢 2.3%，其中 prod 慢 4.8%。

**(ii) 真实 dump 分块 / (iii) 真实 dump + GEMM 突发。每格是 r13ns/r13 的 p1 / p2 / p3，括号里是 r13ns/ASM 的范围**

| 输入 | (ii) 分块 | (iii) GEMM 突发 |
|---|---|---|
| L00 | 0.857 / 0.857 / 0.855（1.081–1.085） | 0.819 / 0.816 / 0.816（1.259–1.270） |
| L01 | 0.869 / 0.872 / 0.865（1.083–1.086） | 0.812 / 0.807 / 0.813（1.270–1.281） |
| L02 | 0.883 / 0.885 / 0.882（1.073–1.074） | 0.826 / 0.824 / 0.821（1.253–1.269） |
| L08 | 0.852 / 0.852 / 0.846（1.089–1.093） | 0.777 / 0.776 / 0.774（1.281–1.288） |
| L16 | 0.919 / 0.919 / 0.917（1.074–1.078） | 0.878 / 0.880 / 0.877（1.259–1.269） |
| L31 | 0.875 / 0.876 / 0.872（1.087–1.091） | 0.817 / 0.813 / 0.812（1.271–1.285） |
| randn | 1.046 / 1.048 / 1.049（1.078–1.084） | 1.055 / 1.065 / 1.058（1.254–1.258） |

- 绝对值（3 个进程的中位数）：
  - 真实数据分块：r13 1.71–1.86 ms，r13ns 1.55–1.59，ASM 1.42–1.47。
  - GEMM 突发后：r13 1.85–2.11 ms，r13ns **1.62–1.64**，ASM 1.27–1.30。
- r13ns 的耗时与输入无关，randn 和真实数据只差 ±2%。r13 在真实数据上慢 15–40%。

## 3. e2e（交付 3）

配置：`E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1`，32 层，62 步，每 10 步 profile 一次；bwd 仍用 `e2e/arms/bwd_r20_0341`。
fwd 树由 `E2E_FLY_TREES` 指定，已经在训练进程的 `/proc/<pid>/environ` 里核对过。

| run | 顺序 | fly 的 fwd | fly tps | ASM tps | 配对单步比 fly/asm（5 组） | fly fwd ms/步（profile 步） | ASM fwd | fly FA path ms | ASM FA path |
|---|---|---|--:|--:|---|---|---|---|---|
| `ns_p2a_asmfly` | ABBA | r13ns | 19,929 | 20,761 | **1.0403**（1.0382–1.0431） | 48.4 / 48.8 / 49.3（s10/30/50） | 38.8–39.3 | 349.5–353.0 | 272.9–274.7 |
| `ns_p2b_flyasm` | BAAB | r13ns | 19,951 | 20,760 | **1.0408**（1.0373–1.0425） | 48.4 / 48.9 / 49.2（s20/40/60） | 38.9–39.1 | 348.7–355.3 | 272.7–275.5 |
| `r13_p2a_asmfly` | ABBA | r13 | 19,822 | 20,721 | 1.0454（1.0394–1.0469） | 50.3 / 55.3 / 58.6 | 38.9–39.4 | 353.3–361.9 | 273.4–275.1 |
| `r13_p2b_flyasm` | BAAB | r13 | 19,887 | 20,794 | 1.0447（1.0401–1.0457） | 53.3 / 59.1 / 61.0 | 38.8–39.0 | 353.5–365.3 | 271.6–275.7 |
| `nsr13_p2c`（直接比较） | fly13,fly,fly,fly13 | r13ns 对 r13 | 19,956 对 19,873 | — | **r13ns/r13 0.9966**（0.9949–0.9977，依次 0.9977 / 0.9967 / 0.9966 / 0.9958 / 0.9949） | ns 48.3 / 48.9 / 49.1；r13 53.1 / 58.0 / 60.2 | — | ns 350.1–350.9；r13 353.8–364.8 | — |

- 5 次运行都是 rc=0，nan/inf 步数为 0，dmesg 干净，memguard 和 watchdog 都没有触发。
- 数据来源：
  - 日志：`../e2e/logs/e2e.<tag>.log`；
  - trace：`../e2e/traces/<tag>/`；
  - 分析结果：`runs/e2e.<tag>.steady.txt` 和 `runs/e2e.<tag>.breakdown.txt`。
- r13 的 fwd 从 s10 到 s60 增加了约 10 ms，r13ns 只增加约 1 ms。profile 报告里提到的"fwd 差距随训练步增长"，就是投机造成的。

## 4. 建议给 fwd job 的 hint（由 operator 投递）

**must refactor**（建议编号 h44）：

```
## h44 -- Adopt r13ns verbatim as op/current: speculation OFF (SPEC_STALE_MAX=False on m32x8 and m32x2)

Operator decision (2026-09-28, evidence PT/output/0927__b0/fwd-nospec/REPORT.md). r13ns = round-13 op/current with the
two `SPEC_STALE_MAX = True` lines set to False (m32x8 line 173, m32x2 line 178); nothing else changed. Output is
BITWISE identical to r13 on every gate shape (randn) and on 6 real training q/k/v dumps. Gate passed: o >= 49 dB on
fast/proxy/prod + all edge shapes both masks, lse covered, 200/200 determinism; 0 VGPR spill / 0 scratch on all 16
compiled configs; prod m32x8 5024 -> 4130 instructions.
Measured (prod, same process, 3 rotated orders): real inputs blocked r13ns/r13 0.846-0.919; real inputs after a GEMM
burst (training operating point) 0.774-0.880; randn blocked (this job's ruler) 1.048. In 32-layer e2e training the
fwd costs 48-49 ms/step flat vs r13 50 -> 61 ms/step growing with steps; direct same-process pairing r13ns/r13 0.9966.

**Refactor task**: copy `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0927__b0/fwd-nospec/arms/fwd_r13ns/`
into the working copy VERBATIM -- no edits (md5: m32x8 370769c94c0f951655f66e10c39947d8, m32x2
b09721562d696d5c1edc7eba29857a88, full list in ../fwd_r13ns.md5) -- and let the correctness gate run. Do not change any
kernel code. Expect the randn speed gate to read ~2% geomean / ~4.8% prod BELOW r13: that is the known ruler bias
(h43/h45), not a regression; promotion must not be blocked by it.
```

**must standing**（建议编号 h45）：

```
## h45 -- STANDING: speculation is dead on real data; the randn ruler mis-ranks it

Real attention inputs have score std 21-53 (randn ~1). The stale-max speculation (g14 SPEC_STALE_MAX and any variant
that computes P against a guessed/stale max and redoes the tile on a trigger) re-computes on 13-25% of tile steps on
real data and 0% on randn, so this job's randn ruler rewards it (+4.8% prod) while it loses 8-15% (blocked) to 12-23%
(training clock) on real inputs and grows slower as training proceeds (e2e fwd 50 -> 61 ms/step vs 48-49 flat).
Rules from now on: (1) do NOT build or re-land speculative / stale-max / guessed-max / trigger-and-redo softmax
variants, and do not re-enable SPEC_STALE_MAX; (2) any candidate whose cost depends on the data (early exits, skips,
triggers, max guesses) must also be measured on the real dumps /home/lihuzhan/_prof_dump/qkv_call0{672,673,674,680,688,703}.pt
([B,S,H,D] bf16, prod shape) against the champion in the same process, and it loses if it loses there, whatever randn
says; (3) prefer levers that cut issued work and clock sensitivity (v_nop, SALU, barriers, permlane): at the training
clock our fwd is still 1.25-1.29x ASM (1.62 vs 1.28 ms) while it is ~1.08x under the blocked ruler.
```

## 5. 文件

- `arms/`：
  - `fwd_r13/`、`fwd_r13ns/`：两棵树；
  - `*.md5`：文件 md5 清单；
  - `fwd_r13ns.diff`：r13ns 相对 r13 的改动。
- `harness/`：job 测试框架的只读拷贝，包括 gates、ut、eager、beat、benchmark.py 和 refcache。refcache 约 295 MB，已写进 .gitignore。
- `tools/`：
  - `compile_arm.py`、`build_all.sh`、`isa_table.sh`：只编译检查；
  - `gate.py`、`realcheck.py`：正确性；
  - `ab.py`、`ab_all.sh`：op 级 A/B；
  - `run_op.sh`：一个卡上进程，负责 lock、时钟采样和 dmesg 检查；
  - `e2e_runs.sh`、`e2e_runs2.sh`、`e2e_ana.sh`、`pair2.py`：e2e。
- `isa/`：编译日志和 ISA 表。
- `runs/`：每个卡上进程的 `.log` 和 `.clk`，以及 e2e 的 steady/breakdown 结果。
- 卡上工作量：op 级 21 个进程，e2e 5 次启动，每次各约 2 分钟，GPU0 总共约 30 分钟。dmesg 始终干净：没有新的 amdgpu 行，只有 NIC ifoe 的 MC 行。
- e2e 运行按 run_e2e.sh 的设计，写入 `../e2e/logs`、`../e2e/traces`、`../e2e/configs` 和 `../gemm/logs/nkfix.<tag>.txt`。
