# 交接：Ulysses GQA 注意力块（四个融合算子 + 整层）性能优化（2026-11-17 06:30 UTC）

用中文和用户交流。开始前先读 `.cursor/rules/iteration_rules.mdc` 和 `.cursor/skills/kernel-optimize/SKILL.md` 并遵守。

## 0. 硬约束（用户原话的要求）
- 只在 `cursor/gqa-ulysses-opt-43f1` 上开发；accept 的轮次可以本地 commit，**只有用户要求时才 push**。
  commit 用 `git -c user.name=xiaoming-amd -c user.email=xiaompen@amd.com`。
- 只能用单 stream。跑之前 `pgrep -af torchrun` 确认没有残留。性能测量不要并行跑（抢链路）；正确性检查可以分到
  不相交的 GPU 上并行。所有运行加 `ulimit -c 0`。
- `ruff format` / `ruff check` 用 `--line-length 110`。调试开关提交前必须删掉。
- 日志落盘在 `<repo>/agent_tmp/gqa_opt/`（不进 git）；结论用实测数据，相对下限（floor）报告。图用 PNG。不用 Llama。
- 改 Primus 侧要另开分支并先找用户确认；FP8 交换先找用户确认；需要 HF_TOKEN 找用户要。
  Primus 分支 `cursor/ulysses-mla-gqa-proxy-configs-43f1` 只读。
- 不要重试已知死路：XCD tile claiming、每 block 一次 sys acquire、nt push loads / nt pull stores、GemmA2A wgrad
  排序、push block 数（指 bwd 的那次结论）、SDMA。
- 每轮：一个变量；正确性先行；跑完整 quick 集；accept 或 rollback；写 summary、kernel snapshot、trend 行；日志只追加。
  连续 2 次 rollback 重新 profile，3 次看 ISA，4 次以上换根本不同的思路。每轮结束简短汇报：改了什么、正确性、
  quick 集结果、accept/rollback。
- CP8 节奏：每 3 轮、每次 accept 前、每阶段结束跑一次 CP8（`CP8=1`）。
- 每阶段结束做 e2e：`run_e2e_gqa.sh g_glm4_7_4layer_proxy_cp4_s32768_u1 glm4_7_4layer_proxy 4 8 32768 1`，同样再跑
  mode -1，然后 `summarize_gqa.py`；参考 GLM unfused 2087 ms / fused 1957 ms，Qwen3-30B 6407 / 5856 ms；
  step-12 loss 和 unfused 一致（< 1e-4）。本机 e2e 基线还没跑过。

## 1. 代码状态
- 仓库 Primus-Turbo，分支 `cursor/gqa-ulysses-opt-43f1`。基线 c8b054cc（handoff commit）。
- **本地 commit（未 push）**：`dff7cc51 ulysses GemmA2A fwd QKV: 256-row tiles from 3 waves of tiles`
  （第 3 轮，只改 `primus_turbo/pytorch/ops/attention/ulysses_fused_a2a.py`）。
- **未提交的工作**（在 `wip.patch` 里，基于 dff7cc51）：
  - `primus_turbo/flydsl/ulysses/a2a_gemm_kernel.py`：
    1. **临时调试开关 `dbg_serial`**（`_make_kernel` / `_launch` / `ulysses_a2a_gemm` 参数，kernel 里在 `_tile_loop`
       前等全部 m-tile 到达）。**commit 前必须删掉**（trace 脚本 `--dbg_serial` 依赖它）。
    2. **split-K 最后一波**（默认关闭，`split_m=0`）：`_split_k_store_c`（非 owner 写 fp32 partial + flag，owner 等
       flag、按 part 顺序求和后走 residual store）；`_arrival_k_seg(seg_offset, num_segs)`；`_gemm_tile(split=...)`；
       `_tile_loop` 把最后 `split_m` 个 m-tile 的 tile 拆成 `split_k` 份（owner 最后领）；`split_bytes` /
       `_split_buffer`（按 workspace state 缓存的 zero 缓冲，flag 存 epoch）；kernel 多一个 `SPLIT` tensor 参数。
    3. **`opaque_tid` / `opaque_base`** 用在 `_gemm_tile` 的 `gemm_tile(...)`（`tid=`、W/Y base），和 GemmA2A /
       AgGemm 一样，防止 claim loop 里多种 tile body 的 lane offset 被 LICM 提出来 spill。
  - `ulysses_fused_a2a.py`：`A2AGemmTuning` 增加 `split_m` / `split_k` 字段。
  - benchmark 工具（可以提交）：
    - `sweep_a2a_gemm.py`：`--only fwd|bwd --bm --push --split_m --split_k --wat`；fwd 每个配置对比默认配置输出
      （`maxdiff`、`OG-MISMATCH`）。
    - `trace_a2a_gemm.py`：`--ops fwd_o bwd_qkv --BM --num_push --wgrad_at --dbg_serial`，每 rank 打印
      `kernel = hipWork + tileCfg tileWait wgCfg wgWait push tail gaps noCU | lastFull wgPure`。
    - `trace_gemm_a2a.py`：`--op bwd_o|fwd_qkv --tokens --BM --num_pull_cu --num_tail_pull --num_lead --wgrad_at`。
    - 新文件 `bench_gqa_layer.py`：单层 GQA fwd+bwd（按 Primus patch），`--mode 1/-1 --trace`，按 kernel 类别汇总。

## 2. 环境（旧机器 smci355-ccs-aus-m11-13，MI355X x8，256 CU）
- 容器 `tasimage/primus:pr-1275`（torch 2.14.0+rocm10.1.0，flydsl 0.3.4.1，aiter 0.1.24.post1），home 挂载同路径。
- in-place 编译：`GPU_ARCHS=gfx950 python3 setup.py build_ext --inplace`；运行时
  `PYTHONPATH=<repo> FLYDSL_EXTRA_SOURCE_DIRS=<repo>/primus_turbo/flydsl`（旧机器上的封装 `agent_tmp/dx.sh "<cmd>"`）。
- 调用 `run_quick.sh` 时命令行里不要出现 "torchrun" 字样（它的 `pgrep -f torchrun` 会匹配到自己）。
- 首次编译慢：CP8 每个新 kernel ~25 min；完整 `test_ulysses_fused_a2a.py` ~70 min（多数是编译）。
  sweep 每个新配置 ~1 min（n=4）。sweep 的 stdout 经过 grep 会缓冲到结束才写文件。

## 3. 检查与测量命令（在 repo 根目录、容器内）
- quick 集：`CP8=1 BASE=agent_tmp/gqa_opt/rounds/round-1/artifacts/quick_cp8/results.jsonl
  benchmark/ops/training/ulysses_gqa_opt/run_quick.sh agent_tmp/gqa_opt/rounds/round-N/artifacts/quick_cp8 --unfused`
  （去掉 `CP8=1` 只跑 n4 五个形状 + MLA guard）。
- 正确性：`HIP_VISIBLE_DEVICES=0..7 torchrun --nproc_per_node 8 tests/pytorch/ops/test_ulysses_fused_a2a.py --quick`
  （accept 前跑不带 `--quick` 的完整版）；`torchrun --nproc_per_node 4
  benchmark/ops/training/ulysses_gqa_opt/check_mla_exit.py --long`。
- sweep 示例：`torchrun --nproc_per_node 4 benchmark/ops/training/ulysses_gqa_opt/sweep_a2a_gemm.py --model glm-4.7
  --only fwd --bm 128 256 --push 16 32 --split_m 0 1 --split_k 4`。

## 4. 结果汇总（us，max over ranks；floor = max(hipBLASLt GEMM + wgrad, RCCL a2a)，同一次运行测得；噪声 ~1 %）
| 轮 | 状态 | 改动 | n4 geomean step | all-7 geomean | MLA guard fwd/bwd |
|---|---|---|---|---|---|
| 1 | BASELINE | c8b054cc | 5228.1 | 4093.8 | 1994 / 3154 |
| 2 | ROLLBACK | fwd QKV BM=256 一律 | 5224.4 | 4081.5 | 2014 / 3149 |
| 3 | ACCEPT dff7cc51 | fwd QKV BM=256 当 256 行 tile ≥ 3 波 | 5198.1 | 4059.0 | 1994 / 3142 |

第 3 轮各形状 fused/floor（fwd QKV / fwd o / bwd o / bwd QKV）：GLM 32K n4 1.01/1.47/1.07/1.39；GLM 128K
0.95/1.43/1.22/1.48；MiniMax 1.02/1.09/1.17/1.59；Qwen3-235B 1.01/1.27/1.18/1.55；Qwen3-30B 0.94/1.00/1.00/1.06；
GLM CP8 1.17/1.80/1.10/1.23；MiniMax CP8 1.14/1.58/1.00/1.71。

## 5. 阶段 0 定位结论（详见 `profiles/phase0_report.md`）
- fwd o_proj（A2AGemm push）：BM=128 tile 是 hipBLASLt 的 1.22–1.31 倍（BM=256 1.02–1.06）；push 并发让 tile 再慢
  ~25 %；尾巴 43–165 us；GLM CP8 GEMM 主导（push 260 us 结束，kernel 746 us），BM=256 只有 1.25 波。
- bwd QKV（A2AGemm + wgrad）：tile 速度正常；dgrad 在 push 阶段被到达速度拖住 264–355 us（kernel 的 23–30 %）；
  wgrad 放前面更差（405–785 us 等待）。
- GemmA2A：BM=128 tile 慢 9–20 %；comm-bound 形状 pull 在第一波 GEMM 后才开始（~70 us），GEMM 后 pull 尾巴
  166–260 us。
- 整层（32K n4）：attention 73–78 %，融合算子 17–22 %，其他 3–5 %（RoPE、`F.rms_norm` bwd、qk-norm、dW 累加；
  MiniMax 非融合 partial RoPE）——这些要改 Primus，需用户确认。MiniMax trace 里的大段 "空闲" 是 profiler 丢事件。

## 6. 第 4 轮进行中（fwd o_proj，A2AGemm）：sweep 实测（untraced，n=4 32K，`rounds/round-4/artifacts/`）
| 形状 | 默认（BM=128 push=32） | BM=128 push=16 | BM=256 push=16 | BM=256 push=32 |
|---|---|---|---|---|
| GLM-4.7 | 1551 | 未测 | 1719 | **1346** |
| Qwen3-235B | 891 | **802** | 831 | 851 |
| MiniMax-M2.5 | 591 | **575** | 679 | 606 |
| Qwen3-30B | 372 | 378 | 439 | 430 |

- split-K：正确（y maxdiff = 1 bf16 ulp，o_gathered 完全一致），但**都更慢**：GLM BM=256 push=32 上 split_m=1/2/4、
  split_k=2/4 是 1407–1542（对比不拆 1346）。连 split_m=1 也慢 ~60 us，说明是多一种 tile body 拖慢了主循环。
- 加 `opaque_tid` / `opaque_base` 之后：GLM 默认 1551→1515，BM=256 push=32 1346→1322，split_m=1 k=4 1411→1361
  （差距从 +65 缩到 +39，仍然更慢）；Qwen3-235B 默认 893、BM=128 push=16 813（基本不变）。
- **第 4 轮下一步**：
  1. 把 `opaque_tid` 单独作为一轮（split_m=0 时 split 代码不实例化），跑正确性 + quick 集，看能否 accept（GLM fwd o −2 %）。
  2. 再一轮改 fwd o_proj 的 BM / num_push 启发式（需要补 GLM BM=128 push=16/24、Qwen3-235B / MiniMax 的
     push 8/12/24，以及 GLM CP8 / MiniMax CP8）。现状启发式在 `a2a_gemm_kernel.py` 的 `ulysses_a2a_gemm`：
     `gemm_bound = wgrad is None and N * n >= 8192 * (n - 1)`，不 gemm_bound 时 32 pusher、BM=128（除非 fits）。
     数据显示 GLM 需要 BM=256 + 32 pusher，Qwen3-235B / MiniMax 需要 16 pusher，Qwen3-30B 保持默认。
  3. split-K 要变快需要先查清楚为什么多一种 body 会拖慢（看 ISA：VGPR/spill/scratch），或者让 split part 复用同一个
     tile body（例如 K 循环长度不变，part 只换 chunk_base 并跳过不属于自己的段——需要 gemm_bf16_tile 支持运行时
     跳段），否则放弃。
- 之后的候选（按阶段 0 收益排序）：GemmA2A pull 提前起步 / 缩短 pull 尾巴；bwd QKV 调度（本 rank 数据先拷进
  recv bank、空闲 CU 领本地 source 的 wgrad tile）；push 并发 tile 慢 25 % 的原因；阶段 2（两跳 relay 可行性）、
  阶段 3（CP8 编译时间：`_tile_loop` → `_gemm_tile` → `gemm_bf16_tile` K 循环全展开）。

## 7. campaign 目录 `agent_tmp/gqa_opt/`
`manifest.yaml`、`logs/performance_trend.md`、`logs/optimize.md`（只追加）、`profiles/phase0_report.md`、
`profiles/phase0a/*.txt`、`profiles/phase0b/*`、`rounds/round-{1,2,3}/{summary.md,kernel_snapshot/,artifacts/}`、
`rounds/round-4/artifacts/sweep_*.txt`、`handoff/`（本文件、`dff7cc51.patch`、`wip.patch`）。
