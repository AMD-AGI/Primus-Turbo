# B0 e2e 终版：今天的冠军（fwd r16 = r13ns，bwd r29 = r19h + u2n）对 aiter ASM（2026-09-28，GPU0 / fa-g0）

Llama-3.1-8B BF16，32 层，MBS=GBS=4，seq 8192，1 卡，AC none，compile 关，seed 1234，GEMM 修复路径开着
（`E2E_NKFIX=1 E2E_MEM_STOP=89.5 NKFIX_CHECK=1`）。P2 同一进程内按步交替 ASM / FlyDSL，62 步，每 10 步 profile 一次。
两种顺序各跑一个进程：`fin_p2a_asmfly`（ABBA）和 `fin_p2b_flyasm`（BAAB）。

## 0. 汇总表（可以直接放进最终报告）

| FlyDSL arm（fwd + bwd） | 进程 | fly/asm 单步时间比（ABBA 周期，5 组） | 相邻步配对（15 对） | 稳态 tps fly / asm | step ms fly / asm | FA path ms/步 fly / asm | 其中 fwd fly / asm | 其中 bwd fly / asm（含 GQA 求和） | attention 占单步 fly / asm |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| r6 + r20（修复 GEMM 后第一版） | n5 / n6 | 1.0475 / 1.0500 | — | 19,762 / 20,718；19,809 / 20,792 | 1,658 / 1,582；1,654 / 1,576 | 352–364 / 270–273 | 约 52–59 / 39 | — | 21.4–21.8% / 17.0–17.3% |
| r13ns + r20 | ns_p2a / ns_p2b | 1.0403 / 1.0408 | 1.0400 / 1.0415 | 19,929 / 20,761；19,951 / 20,760 | 1,644 / 1,578；1,642 / 1,578 | 348.7–355.3 / 272.6–275.5 | 48.4–49.3 / 38.8–39.3 | 300.3–306.2 / 233.6–236.5 | 21.2–21.4% / 17.2–17.4% |
| **r16（= r13ns）+ r29（今天的冠军）** | **fin_p2a / fin_p2b** | **1.0323 / 1.0328** | **1.0324 / 1.0333** | **20,091 / 20,747；20,070 / 20,697** | **1,631 / 1,579；1,633 / 1,583** | **333.7–339.3 / 272.1–275.7** | **48.0–49.3 / 38.8–39.3** | **285.4–290.0 / 233.1–236.7** | **20.5–20.6% / 17.2–17.4%** |

- **结论：FlyDSL 在 e2e 里仍然比 ASM 慢，但差距从 4.8–5.0% → 4.0–4.1% → 现在 3.2–3.3%**（约 51–52 ms/步，tps 低 3.0–3.2%）。
  两种顺序结果一致（1.0323 / 1.0328），相邻步配对的方法也给出同样的数（1.0324 / 1.0333）。同一方法重算 r13ns+r20 的旧日志得到
  1.0400 / 1.0415，与之前报告的 1.0403 / 1.0408 一致，所以新旧数字可以直接比。
- **这一轮的收益全部来自 bwd**：fly 的 bwd 从 300–306 ms/步降到 285–290 ms/步（约 −15 ms，−5%）。其中
  `k_dq` 106.4–109.0 → `k_dqg` 96.0–98.3 ms（−10%，u2n），`k_dkdv` 190.2–193.4 → 185.7–188.0 ms（−2%，r19h），`k_delta` 3.8 不变。
  fwd 没变（48.0–49.3 ms，和 r13ns 那次一样，本来就是同一份源码）。单步时间的配对差从约 64 ms 降到约 51 ms，与 trace 一致。
- **剩下的差距在哪**：FA path 差 58–65 ms/步。bwd fly 287 对 ASM 235（ASM 含 GQA 求和 3.9 ms、dq_convert 2.3、odo 1.6），
  **约 52 ms，是大头**；fwd fly 48.7 对 ASM 39.1，**约 10 ms**。单步 fly/ASM 比：bwd 约 1.22，fwd 约 1.25。
- **attention 占单步**：fly 20.5–20.6%，ASM 17.2–17.4%。GEMM 主体 845–862 ms（+ nkfix 辅助 98–100 ms），两个 arm 相同。
- **loss 正常**：step 1 和之前逐位一致（ASM 先 12.25951，fly 先 12.25957）。step 1 的 grad_norm 在 fly 先的进程里是 42.3893，
  之前 r20 是 42.3894（bwd kernel 换了，只差 1e-4 级，符合预期）。之后的轨迹在 bf16 噪声内：step 62 loss 3.53003 / 3.51697，
  旧 r13ns+r20 运行是 3.52833 / 3.51920。没有 nan/inf，nkfix 的 `nonfinite_events: []`，watchdog 和 memguard 都没触发，峰值显存 88.99%。
- **卡的安全**：本轮卡上共 4 个进程（opcheck fast、opcheck prod、2 次 e2e），每次之后 dmesg 在 0001:04:00.0 上都没有新行。

## 1. FlyDSL arm 是怎么指过去的

- 机制和 `fwd-nospec/REPORT.md` 相同：`E2E_FLY_TREES={"fly":{"fwd":<dir>,"bwd":<dir>}}` 通过 `E2E_ENV` 传进容器，
  由 `attn_backends/e2e_attn/arms.py` 读取。已在训练进程的 `/proc/<pid>/environ` 里核对了 `E2E_FLY_TREES` 和
  `FLYDSL_RUNTIME_CACHE_DIR`。trace 里 bwd kernel 名是 `k_dqg_0`（r20 是 `k_dq_0`），这也证明加载的是新树。
- 两棵树（只从 job 目录拷出，不在 job 目录里写任何东西；`__pycache__` 没有拷）：
  - `arms/fwd_r16/` ← `gfx1250-flydsl-attn-fwd-b0-20260927/job_context/op/current`。和 `../fwd-nospec/arms/fwd_r13ns/` 源码逐字节相同。
  - `arms/bwd_r29_0341/` ← `gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/current`，只把 `_env.py` 换成
    `arms/bwd_r20_0341/_env.py`（pin 到 flydsl 0.3.4.1，做法与 BUILD.md 里的 r20 相同）。未改动的原树留在 `arms/bwd_r29_032/`。
  - md5 清单：`arms/SNAPSHOT2.md5`，时间：`arms/SNAPSHOT2.date`。
- **0.3.4.1 下的 ISA 与 job 的 0.3.2 逐条相同**（只编译，不上卡，`tools/compile_bwd_ver.py`，prod 形状，结果在 `recon/r29isa/isa_compare.txt`）：

| kernel | 0.3.2 VGPR / 指令数 | 0.3.4.1 VGPR / 指令数 | 指令序列 diff | spill / scratch |
|---|--:|--:|--:|--:|
| k_delta_bshd | 40 / 256 | 40 / 256 | 0 行 | 0 / 0 |
| k_dkdv | 729 / 2,339 | 729 / 2,339 | 0 行 | 0 / 0 |
| k_dqg | 991 / 4,232 | 991 / 4,232 | 0 行 | 0 / 0 |

  VGPR 数和 lab-kdq 第二轮里 r29 冠军（`cur`：729 / 991）一致。
- **0.3.4.1 下的正确性**（`run_e2e.sh opcheck fly`，经 E2EAttention 模块）：
  - fast（`AMD_SERIALIZE_KERNEL=3`）：bwd 用参考 o/lse 时 dq/dk/dv 52.61 / 52.65 / 52.83 dB，与 lab-kdq 在 fast 上的数逐位一致。
  - prod：52.56 / 52.60 / 52.71 dB（与 lab-kdq prod 一致），全链路 o/dq/dk/dv 50.83 / 51.90 / 51.99 / 52.39 dB，适配层拷贝 0 次。
  - prod 计时（单进程，只作参考）：k_bwd 8.56 ms（BUILD.md 里 r20 是 9.02 ms），k_fwd 1.21 ms。
  - 日志：`opcheck/opcheck.fly.fast.0928_103846.log`、`opcheck/opcheck.fly.prod.0928_103903.log`。
- **JIT 缓存**：`run_e2e.sh` 新增了 `E2E_FLYCACHE`（默认仍是 `/tmp/flycache_e2e`）。每个进程都用一个新的目录：
  `/tmp/flycache_fin_opc_fast`、`/tmp/flycache_fin_opc_prod`、`/tmp/flycache_fin_p2a_asmfly_<hhmmss>`、`/tmp/flycache_fin_p2b_flyasm_<hhmmss>`。
  修改前的脚本备份在 `logs/run_e2e.sh.pre-flycache.bak`。
- 注意：fwd 树的 `_env.py` 会把 `HIPBLASLT_TENSILE_LIBPATH` 改指向宿主库。它在第一次 attention fwd 时才被加载，这时 hipBLASLt
  已经初始化过了。r13ns 那次运行也是这样。两个 arm 的 GEMM 时间相同（845–862 ms），所以这次没有影响。

## 2. 方法

- 稳态窗口和以前一样（`tools/steady_arms.py`）：去掉 step 1–7，以及每个 profile 步 F 的 F−1、F、F+1。每个 arm 18–19 步。
- 配对：(a) `steady_arms.py` 的 ABBA 4 步周期（5 组）；(b) 相邻两步、arm 不同、且都在稳态窗口里（15 对）。取 step 时间比的中位数。
- trace：每 10 步 1 个，按 `e2e::attn_*` 的 CPU 祖先归类（`tools/trace_breakdown.py`）。每个进程 3 个 fly 步、3 个 ASM 步。
- 运行时 sclk（busy>50% 的样本平均）：1342 / 1371 MHz，旧 ns 运行 1335 / 1330 MHz，工作点相同。

## 3. 文件

- 驱动：`tools/final_e2e.sh`、`tools/final_trees.sh`；输出 `runs_final/driver.log`、`runs_final/e2e.<tag>.out`。
- 日志：`logs/e2e.fin_p2a_asmfly.log`、`logs/e2e.fin_p2b_flyasm.log`（以及 `clk.*.csv`、`tree.*.sha256`）。
- 稳态和 trace 拆分：`runs_final/e2e.<tag>.steady.txt`、`runs_final/e2e.<tag>.breakdown.txt`；trace：`traces/fin_p2*/iteration_*`。
- nkfix 统计：`../gemm/logs/nkfix.fin_p2a_asmfly.txt`、`../gemm/logs/nkfix.fin_p2b_flyasm.txt`。
- 占卡时间：约 5 分钟（10:38–10:44）。
