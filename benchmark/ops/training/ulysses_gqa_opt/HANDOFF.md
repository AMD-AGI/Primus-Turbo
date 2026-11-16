# 任务：Ulysses GQA 注意力块（四个融合算子 + 整层）的性能优化

用中文和我对话。先读仓库里的 `.cursor/rules/iteration_rules.mdc` 和 `.cursor/skills/kernel-optimize/SKILL.md`，按里面的迭代规则做（每轮只改一个变量、正确性优先、每轮跑完整 quick set、accept / rollback、每轮 summary + kernel 快照 + trend 行、日志只追加；连续 2 次 rollback 重新 profile，3 次看 ISA，4 次以上换根本不同的方案）。

## 代码和分支

| 仓库 | 分支 | 说明 |
|---|---|---|
| Primus-Turbo `git@github.com:AMD-AGI/Primus-Turbo.git` | **`cursor/gqa-ulysses-opt-43f1`**（你在这上面开发） | 基于 `dev/xiaoming/flydsl_mla_ulysses_latent` 的 `d5e28371`，多一个 commit，见下 |
| Primus `git@github.com:AMD-AGI/Primus.git` | `cursor/ulysses-mla-gqa-proxy-configs-43f1`（只读） | Megatron 接入（`--turbo_ulysses_fused_a2a 1/-1/0`）和 GQA 模型 / 代理配置，只在端到端检查时用 |

`cursor/gqa-ulysses-opt-43f1` 比 `d5e28371` 多的那个 commit 包含：
1. **MLA 出口 campaign 已经 accept、但原来没 commit 的 kernel 改动**（R2–R6），也就是端到端 GQA 矩阵实际跑的代码，是你的 baseline：
   - `primus_turbo/pytorch/ops/attention/ulysses_fused_a2a.py`：R2 bwd o_proj dgrad 默认 `BM=256`；R5 `num_pull_cu=8 if n == 2 else 16`。
   - `primus_turbo/flydsl/ulysses/a2a_gemm_kernel.py`：R3 push flag 和 exit 改 relaxed；R4 `_all_landed` 快路径（tile 开工时数据已全部到达就走无等待 K walk）；R6 GEMM-bound 时 n≥4 用 16 个 pusher。
   - `primus_turbo/flydsl/ulysses/gemm_a2a_kernel.py`：R3 tile flag 和 exit 改 relaxed。
2. `benchmark/ops/training/bench_ulysses_overlap.py`：`MODELS` 加了 `qwen3-235b-a22b`、`glm-4.7`、`minimax-m2.5`。
3. `benchmark/ops/training/bench_ulysses_mla_layer.py`、`check_mla_entry_variants.py`、`bench_ulysses_mla_latent.py` 的改动（MLA 用，顺带带上）。
4. **`benchmark/ops/training/ulysses_gqa_opt/`：本任务的工具和基线**（本文件也在这里）：

   | 文件 | 用途 |
   |---|---|
   | `gqa_quick_bench.py` | GQA 四个 op：先做 fp32 参考正确性检查，再测融合时间和当场测出的下限 |
   | `score_gqa.py` | quick set 打分：每个形状每个 op 的"融合 / 下限"，以及 step_us 的 geomean |
   | `run_quick.sh` | 整个 quick set（含 MLA 回归保护），`CP8=1` 加跑 CP8 |
   | `trace_a2a_gemm.py` | A2AGemm 的 kernel 内 trace，按 CU 时间拆分（tile / wgrad / push / 空闲、尾巴、纯 tile 耗时 vs hipBLASLt） |
   | `trace_gemm_a2a.py` | GemmA2A 的 trace（来自 MLA campaign，默认参数是 DSv3，注意传 `--model`） |
   | `sweep_a2a_gemm.py` | A2AGemm 启动参数扫描（BM × num_push × push_split × wgrad 位置） |
   | `quick_test_bench.py`、`check_mla_exit.py` | MLA 出口的计时和正确性（回归保护） |
   | `run_e2e_gqa.sh`、`summarize_gqa.py` | 端到端训练检查和解析 |
   | `baseline/` | 我在 MI355X 上测的基线原始输出：`{model}_n{4,8}.txt`（16K–128K 的 gemm / wgrad / a2a / pack / serial / unfused / overlap / fused 各列）、`sweep_*.txt`、`trace_*.txt` |

   所有脚本默认把仓库根目录（`ulysses_gqa_opt/` 往上 4 级）当 `TURBO_ROOT`，也可以用环境变量覆盖。**`gqa_quick_bench.py`、`score_gqa.py`、`run_quick.sh`、`run_e2e_gqa.sh` 是新写的，还没在 GPU 上跑过**，第 1 轮先把它们跑通（见下文"第 1 轮"）。

## 环境搭建（MI355X × 8，gfx950）

- 容器镜像：`tasimage/primus:pr-1275`（torch 2.14.0+rocm10.1.0、flydsl 0.3.4.1、amd-aiter 0.1.24.post1）。用别的镜像时，这几个版本要对齐，否则基线不可比。
- 编译 Primus-Turbo（C++ 扩展 `primus_turbo/pytorch/_C*.so` 和 `primus_turbo/lib/libprimus_turbo_kernels.so` 是编译产物，不在 git 里）：
  ```bash
  git clone git@github.com:AMD-AGI/Primus-Turbo.git && cd Primus-Turbo
  git checkout cursor/gqa-ulysses-opt-43f1
  git submodule update --init --recursive
  pip3 install -r requirements.txt
  GPU_ARCHS="gfx950" pip3 install --no-build-isolation -e ".[pytorch]" -v
  ```
  FlyDSL kernel 是 Python 源码，改 kernel 不需要重新编译。运行时设 `PYTHONPATH=<repo>`、`FLYDSL_EXTRA_SOURCE_DIRS=<repo>/primus_turbo/flydsl`（`run_quick.sh` 已经设了）。
- 端到端检查需要 Primus：`git clone git@github.com:AMD-AGI/Primus.git && git -C Primus checkout cursor/ulysses-mla-gqa-proxy-configs-43f1 && git -C Primus submodule update --init --recursive`。用 mock 数据，不需要数据集；tokenizer 可能要从 HuggingFace 下载，网络不通时需要 `HF_TOKEN` 或本地缓存，缺了就找我要。
- 所有运行都加 `ulimit -c 0`：持久 kernel 卡死时，一个 GPU core dump 就有 4 GB，几次就能把磁盘写满。

## 背景

Ulysses 上下文并行下，GQA/MHA 的注意力块在 attention 两侧各做一次 all-to-all（序列切分 ⇄ head 切分）。Primus-Turbo 把它们和相邻的投影 GEMM 融合在同一个 FlyDSL 持久 kernel 里（单 stream，tile 级 flag 驱动）：

| op | 位置 | 做什么 | kernel |
|---|---|---|---|
| fwd QKV（GemmA2A） | 入口前向 | `x[S/n, hidden] × W_qkv` → 按 head push 给各 rank | `gemm_a2a_kernel.py` |
| fwd o_proj（A2AGemm） | 出口前向 | all-to-all o（head → 序列）进 K 维 → `× W_o` + 残差 | `a2a_gemm_kernel.py` |
| bwd o_proj（GemmA2A + wgrad） | 出口反向 | dgrad `dy × W_o` → all-to-all 回 head 切分；wgrad 同 kernel | `gemm_a2a_kernel.py` |
| bwd QKV（A2AGemm + wgrad） | 入口反向 | all-to-all dq/dk/dv 进 K 维 → `× W_qkv`；wgrad 同 kernel | `a2a_gemm_kernel.py` |

GQA 是论文（Shuttle）的主线：Qwen3、GLM-4.7、MiniMax-M2.5 都是 GQA。端到端（GBS 16、unfused 和融合用同一个 attention kernel）CP4 32K 时融合比 unfused 快 5.6–8.6%，128K 降到 2.7–3.7%。目标是把算子做到接近 overlap 下限，并把收益带到整层和端到端。

**范围不只是 A2AGemm**：四个融合算子都要看，另外还有整层里融合算子之外的开销（阶段 0b）、通信下限（阶段 2）和 CP8 编译时间（阶段 3）。

## 当前性能（本分支，MI355X，us，max over ranks）

"下限" = max(hipBLASLt GEMM + wgrad, RCCL all-to-all)。注意融合 kernel 自己的 push 比 RCCL 快 15–20%，所以通信主导时融合可以略低于下限。CP4 每 rank 对外带宽约 154 GB/s（只用到 3 个对端），CP8 约 340 GB/s。

每层四个算子求和（32K）：

| 模型 | hidden / hq / hkv | CP | unfused | 融合 | 下限 | 融合 / 下限 |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 2048 / 32 / 4 | 4 | 2992 | 1634 | 1592 | 1.03 |
| Qwen3-235B-A22B | 4096 / 64 / 4 | 4 | 6663 | 4090 | 3326 | 1.23 |
| GLM-4.7 | 5120 / 96 / 8 | 4 | 11168 | 7027 | 5753 | 1.22 |
| MiniMax-M2.5 | 3072 / 48 / 8 | 4 | 5062 | 3073 | 2447 | 1.26 |
| GLM-4.7 | 5120 / 96 / 8 | 8 | 4839 | 3564 | 2789 | 1.28 |

每个 op 的"融合 / 下限"（32K）：

| 模型 | CP | fwd QKV | fwd o_proj | bwd o_proj | bwd QKV |
|---|---|---|---|---|---|
| Qwen3-30B | 4 | 0.96 | 1.01 | 1.03 | 1.10 |
| Qwen3-235B | 4 | 1.07 | 1.28 | 1.12 | 1.42 |
| GLM-4.7 | 4 | 1.10 | 1.51 | 1.05 | 1.29 |
| MiniMax-M2.5 | 4 | 1.09 | 1.10 | 1.18 | 1.60 |
| GLM-4.7 | 8 | 1.34 | **1.71** | 1.08 | 1.23 |

GLM CP8 的 fwd o_proj 融合后（744）和串行（739）一样，16K–128K 都比串行慢 1–11%，是最坏的点。16K–128K 的完整数据在 `baseline/`；CP8 只有 GLM。

## 已知结论（不要重复走）

1. **trace 定位**（`trace_a2a_gemm.py`，`baseline/trace_*.txt`）：GEMM 主导时，数据交换早就结束了，后面是 kernel 内 GEMM 在拖。

   | 形状（32K） | push 结束 | kernel 结束 | 128 行 tile 耗时 / hipBLASLt | GEMM 效率 | push 占 CU |
   |---|---|---|---|---|---|
   | GLM CP4 fwd o_proj（BM=128） | 870 | 1503 | 1.46 倍 | 0.66 | 7% |
   | GLM CP8 fwd o_proj（BM=128） | 255 | 726 | 1.36 倍 | 0.71 | 4% |
   | GLM CP4 bwd QKV（BM=256） | 1226 | 2522 | 0.84 倍 | 0.81 | 6% |

   "耗时 / hipBLASLt"是数据已全部到达时开工的 tile 的中位耗时，和 hipBLASLt 按 256 个 CU 折算到单个 tile 的耗时之比；"GEMM 效率"包含 K walk 里的等待。trace 本身让计时变慢 1–10%，只看比例。
2. **fwd o_proj 的 `BM=128` 是主要损失之一。** `ulysses_a2a_gemm` 的启发式在 `gemm_bound` 不成立、tile 又放不进一波时选 128 行。GLM（`N·n/(n−1)` = 6827 < 8192）因此被判成通信主导。阈值是按 hipBLASLt 的 GEMM 速度定的，而融合 kernel 里的 GEMM 只有它的 0.66–0.8。
3. **调参扫描**（`baseline/sweep_*.txt`，CP4 32K）：只有 GLM fwd o_proj 收益明显（1532 → 1337，BM=256、32 个 push），其他算子最好的配置只比默认快 0–5%。纯调参不够，要改结构。`push_split` 4 和 8 没有区别。
4. **MLA campaign 的结论**（同一组 kernel）：
   - 256×256 tile 单独跑时和 hipBLASLt 只差 3–6%，tile 本身不是问题；128 行 tile 慢 24%。
   - K walk 里每段的 flag 等待（即使 flag 已就绪）会在完全展开的 K 循环里插入 `s_waitcnt vmcnt(0)` 排空预取，重新加载 spill 掉的 flag 地址，还让累加器 spill 到 scratch。R4 的 `_all_landed` 快路径只解决了"开工时已全部到达"的 tile。
   - **fwd QKV 的 GemmA2A 仍是 BM=128**（R2 只改了 bwd o_proj dgrad）。
   - 无效、不要再试：按 XCD 领取 tile（L2 复用）、每个 block 只做一次 sys acquire、nt push loads / nt pull stores、调整 GemmA2A 内 wgrad 的调度顺序、调 push block 数量（16 / 32 / 64 差别在波动内）。
5. **SDMA（拷贝引擎）做交换已经试过并放弃**：最好的配置比 CU push 慢 10%。原因有三：IPC 指针上的 `hipMemcpy2DAsync` 会串行化，因此要换成 source-major 的接收布局，A 的 loader 也要改；8 B 的 flag 拷贝会在 CU 上以 blit kernel 形式运行，持久 kernel 占满 CU 时会卡死（要用 NoCU 拷贝）；Python 侧发起拷贝每次要 140–380 us。除非你有针对这三点的新方案，否则别重做。

## 优化计划

### 第 1 轮：跑通工具 + baseline

1. 在 4 卡上用小形状跑通 `gqa_quick_bench.py`（例如 `--shapes qwen3-30b-a3b:16384 glm-4.7:16384`），确认正确性检查是 PASS，计时和 `baseline/` 里同形状的 fused 列相差在 5% 以内。如果差得多，先查环境（镜像版本、GPU 型号、XGMI 拓扑），不要急着优化。
2. 跑完整的 `run_quick.sh <out> --unfused`，再跑一次 `CP8=1`，作为 baseline。

### 阶段 0a：四个 op 的 GEMM 主导损失拆分（不改 kernel，或只加临时调试开关）

代表形状：GLM-4.7 CP4 32K、GLM-4.7 CP8 32K、MiniMax-M2.5 CP4 32K（bwd QKV 1.60）、Qwen3-235B CP4 32K。
1. 用 trace 拆开 bwd QKV 里 dgrad tile 和 wgrad tile 的耗时，以及 wgrad 放前 / 放后对尾巴的影响。MiniMax 和 Qwen3-235B 的 bwd QKV 是所有 op 里离下限最远的。
2. GemmA2A 那一侧（fwd QKV、bwd o_proj）用 `trace_gemm_a2a.py` 拆，重点看 fwd QKV 的 BM=128 在 CP8 和大 hidden 下损失多少。
3. 回答这些问题，给出实测数字：
   - (a) tile 配置（BM=128 vs 256）各贡献多少；
   - (b) push / pull block 占的 CU 贡献多少；
   - (c) K walk 里的等待、未到达 tile 的空转贡献多少；
   - (d) 最后一波的波次量化（尾巴）贡献多少。

### 阶段 0b：整层 profile（找融合算子之外的开销）

用 torch profiler（或 rocprofv3）抓 GQA 注意力块一层的 fwd + bwd：入口 QKV、RoPE、q/k norm（Qwen3 有 qk-layernorm）、attention、出口 o_proj。看 fused 路径上融合 kernel 之外还有哪些 kernel 和拷贝（`.contiguous()`、view/reshape 引起的拷贝、RoPE 和 norm 的单独 kernel、dtype 转换），各占多少。Megatron 侧的接入代码在 Primus 的 `primus/backends/megatron/patches/ulysses_fused_a2a_patches.py`（模式 1 = 融合，-1 = unfused）。可以用 `run_e2e_gqa.sh` 截一个 step 来 profile，也可以写一个只跑一层 attention 的 bench。改 Primus 侧需要另开分支，先找我确认。

阶段 0 结束时单独汇报定位结论，再进入阶段 1。

### 阶段 1：GEMM 主导时四个 op 的吞吐（主目标）

根据阶段 0 选方向，每轮一个改动。候选（按我估计的收益排序，以阶段 0 结论为准）：
1. fwd o_proj：在 GQA 形状上用 256 行 tile，同时处理最后一波的尾巴（例如把最后到达的 m-tile 拆成更小的块、混合 BM、stream-K）。
2. 修正 `gemm_bound` 的判定：把融合 kernel 里 GEMM 的实际效率算进去，或者直接按测得的 GEMM / 交换比例来判；push 数要随之重新扫。
3. fwd QKV GemmA2A：BM=256（参照 R2 对 bwd o_proj 的做法），重扫 `num_pull_cu` / `num_lead`。
4. bwd QKV：wgrad tile 和 dgrad tile 的调度（wgrad 放前 / 放后的阈值、wgrad 和 push 阶段的重叠）、K walk 里每段等待的开销（R4 只覆盖了已全部到达的 tile）。

目标：quick set 上四个 op 的"融合 / 下限"降到 1.1 以内；GLM CP8 fwd o_proj 至少要明显快于串行。Qwen3-30B（通信主导，已在下限）和 MLA 出口（DSv3）不能回退。

### 阶段 2：通信下限（次目标，探索性）

Qwen3-30B 这类小 hidden 模型已经贴着通信下限，纯交换时间约占端到端 step 的 7–10%。只能靠提高带宽或减少字节：
- CP4 只用到 7 个 XGMI 对端里的 3 个。可以研究经节点内另一个 CP 组的 GPU 中转（两跳），用上空闲的跨组链路。代价是中转 GPU 也在跑自己的 op，要评估 CU 和 HBM 开销。
- FP8 交换 QKV / dO：需要先做精度验证，做之前先找我确认。

先做可行性测量（例如两跳 push 的实测带宽），有把握再改 kernel。

### 阶段 3：CP8 编译时间

GLM-4.7 CP8 的 A2AGemm 第一次编译约 25 分钟：py-spy 显示 8 个 rank 都在 FlyDSL 前端 trace `_tile_loop` → `_gemm_tile` → `gemm_bf16_tile`（K walk 按 8 段展开）。这会撞上端到端每个 run 的超时，也会让每轮迭代都要等很久。查清楚是哪部分展开导致的，在不损失性能的前提下降下来（例如把逐段展开改成循环，或减少 constexpr 特化）。这一项可以和阶段 1 交替做。

## Quick set 和检查

`run_quick.sh OUT_DIR [--unfused] [--no-check]`（在容器里跑，`BASE=<上一轮的 results.jsonl>` 可以直接对比）：

| n | 形状 | 作用 |
|---|---|---|
| 4 | glm-4.7:32768, glm-4.7:131072 | 主目标（fwd o_proj、bwd QKV） |
| 4 | minimax-m2.5:32768, qwen3-235b-a22b:32768 | 主目标（bwd QKV） |
| 4 | qwen3-30b-a3b:32768 | 通信主导，不能回退 |
| 4 | deepseek-v3:32768（`quick_test_bench.py`） | MLA 出口回归保护（共用 A2AGemm / GemmA2A） |
| 8（`CP8=1`） | glm-4.7:32768, minimax-m2.5:32768 | CP8；编译慢，每 3 轮、每次 accept 前和阶段结束时跑。阶段 3 把编译时间降下来后放回每轮 |

主指标：GQA 形状上 step_us（四个融合 op 之和）的 geomean。每轮同时报每个形状每个 op 的"融合 / 下限"（下限在同一次运行里现测）。

正确性（每轮先跑，失败直接 rollback）：
- `gqa_quick_bench.py` 自带：每个形状先用 `tests/pytorch/ops/test_ulysses_fused_a2a.py` 的 `check_qkv` / `check_o_gemm` 在模型形状上对比 fp32 参考，覆盖 q/k/v、dx、dW_qkv、y、do、dW_o、d_residual。
- `torchrun --nproc_per_node 8 tests/pytorch/ops/test_ulysses_fused_a2a.py --quick`；accept 前跑一次完整版（不带 `--quick`）。
- `check_mla_exit.py --long`（n=4），MLA 出口回归保护。

下限 / 不融合参考的完整表（16K–128K）：`benchmark/ops/training/bench_ulysses_overlap.py --model <m> --seq 16384 32768 65536 131072`（cwd 要在 `benchmark/ops/training`）。

端到端迁移检查（accept 的改动要能在训练里体现；每个阶段结束时至少跑一次，需要 8 卡）：
```bash
export PRIMUS_ROOT=<Primus checkout> OUT=<campaign>/e2e
benchmark/ops/training/ulysses_gqa_opt/run_e2e_gqa.sh g_glm4_7_4layer_proxy_cp4_s32768_u1 glm4_7_4layer_proxy 4 8 32768 1
benchmark/ops/training/ulysses_gqa_opt/run_e2e_gqa.sh g_glm4_7_4layer_proxy_cp4_s32768_u-1 glm4_7_4layer_proxy 4 8 32768 -1
python3 benchmark/ops/training/ulysses_gqa_opt/summarize_gqa.py $OUT
```
tag 的格式（`g_<model>_cp<n>_s<seq>_u<mode>`）是 `summarize_gqa.py` 解析用的，不要改。参考基线（MI355X，GBS 16，12 iter，取第 3–12 步中位数）：

| 配置 | unfused | 融合 |
|---|---|---|
| GLM 4 层代理 CP4 32K | 2087 ms | 1957 ms |
| Qwen3-30B CP4 32K（`qwen3_30B_A3B 4 8 32768`） | 6407 ms | 5856 ms |

先在你的机器上跑一次 unfused 和融合作为本地基线。第 12 步的 loss 要和 unfused 一致（目前差 < 1e-4）。

## 相关代码

- kernel：`primus_turbo/flydsl/ulysses/a2a_gemm_kernel.py`（A2AGemm：`_tile_loop`、`_gemm_tile`、`_claimed_tile`、`_arrival_k_seg`、`_all_landed`、`_push_loop`、`_wgrad_tile`、`_wgrad_k_hook`），`gemm_a2a_kernel.py`（GemmA2A），`common.py`，`workspace.py`；GEMM tile 来自 `primus_turbo/flydsl/mega/bf16/gemm_bf16_kernel.py`（`gemm_bf16_tile`），wgrad 用 `grouped_gemm_bf16_variable_k_tile`。
- launch 启发式：`a2a_gemm_kernel.py::ulysses_a2a_gemm`（`gemm_bound`、`wgrad_last`、`num_push`、BM、stagger），`primus_turbo/pytorch/ops/attention/ulysses_fused_a2a.py`（`_qkv_gemm_a2a_forward`、`_qkv_backward`、`_o_gemm_forward`、`_o_gemm_dgrad`，以及 `GemmA2ATuning` / `A2AGemmTuning`）。

## 硬性约束

- 只在 `cursor/gqa-ulysses-opt-43f1` 上开发；accept 的轮次可以在本地 commit，**只有我要求时才 push**。commit 用 `git -c user.name=xiaoming-amd -c user.email=xiaompen@amd.com`。
- 只能用单 stream，不能用多 stream。
- 跑之前确认没有残留的 torchrun（`pgrep -af torchrun`）。性能测量不要并行跑多个任务（会抢链路）；正确性检查可以分到不相交的 GPU 上并行。
- `ruff format` / `ruff check` 用 `--line-length 110`。
- 调试开关提交前必须删掉。
- 日志必须落盘（campaign 目录自己建，例如 `<repo>/agent_tmp/gqa_opt/`，不要提交进 git）；结论用实测数据，相对下限报告，不只和不融合比。
- 图用 PNG。不要用 Llama 作为模型（`baseline/` 里有 Llama 的数据，只作参考）。
- 汇报时：每轮结束简要说明改了什么、正确性、quick set 结果、accept 还是 rollback；阶段 0 结束时单独汇报定位结论，再进入阶段 1。
