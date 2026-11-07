# Ulysses GemmA2A / A2AGemm：多模型单节点 proxy 端到端测试 prompt

> 把下面整段交给 agent 作为任务描述，在一台 8×MI355X 节点上执行。实现说明和 Qwen3-30B-A3B 的结果见 `docs/ulysses_gemm_a2a.md`；环境搭建和 2 节点测试见 `docs/ulysses_2node_e2e_prompt.md`。

---

在单节点（8×MI355X）上，用若干模型的 proxy（层数减到 4 层，attention / MoE 结构不变）跑 Megatron 预训练，测 Ulysses CP 三种实现的每步耗时，产出一份对比报告，用来决定 GemmA2A / A2AGemm 融合对外发布时主推哪个模型。全程用中文汇报，不用 CUDA graph，跑完清理遗留进程。

## 代码

| 仓库 | 分支 | 用到的内容 |
|---|---|---|
| Primus-Turbo（`git@github.com:AMD-AGI/Primus-Turbo.git`） | `feat/ulysses-gemm-a2a` | `ulysses_qkv_gemm_a2a` / `ulysses_a2a_o_gemm` |
| Primus（`git@github.com:AMD-AGI/Primus.git`） | `feat/ulysses-fused-a2a` | patch `primus/backends/megatron/patches/ulysses_fused_a2a_patches.py`；训练配置 `examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-cp4-pretrain.yaml`（名字是 Qwen3，但模型由 `PRIMUS_MODEL` 选，CP / EP 由 `PRIMUS_CP` / `PRIMUS_EP` 选）；proxy 模型配置在 `primus/configs/models/megatron/*_4layer_proxy.yaml` |

## 环境

和 `docs/ulysses_2node_e2e_prompt.md` 的“环境”一节完全相同：镜像 `rocm/primus:v26.8`，编译 Turbo 分支，`flydsl==0.3.4.1`，并**确认训练进程加载的是这个分支的 `primus_turbo`**（Primus 的 `base_env.sh` 会把 site-packages 排到 PYTHONPATH 最前，需要 `sitecustomize.py` 或 `pip install -e` 解决）。baseline 和 fused 必须用同一份 Turbo。

## 测试矩阵

所有组：TP=1，PP=1，`micro_batch_size=1`，`global_batch_size=16`，mock 数据，12 步，序列长度 16K 和 32K。

| 前缀 | 模型（`PRIMUS_MODEL`） | CP | EP | 为什么测它 |
|---|---|---|---|---|
| `q30b` | `qwen3_30B_A3B`（完整 48 层） | 4 | 8 | 复现参考，应与下方单节点结果一致 |
| `q235` | `qwen3_235B_A22B_4layer_proxy` | 4 | 8 | 首推发布模型，结构与 30B 相同 |
| `glm_cp4` | `glm4_7_4layer_proxy` | 4 | 8 | 96 Q head / 8 KV head，attention 占比高 |
| `glm_cp8` | `glm4_7_4layer_proxy` | 8 | 8 | 唯一能用 CP=8 的 MoE 候选；CP=8 的 kernel 参数还没调过 |
| `mm25` | `minimax_m2.5_4layer_proxy` | 4 | 8 | partial RoPE（`rotary_percent: 0.5`）+ qk norm |
| `l8b` | `llama3.1_8B`（完整 32 层，dense） | 4 | 1 | dense 长上下文代表 |

每个组合跑三种实现：

| 实现 | 额外环境变量 | 含义 |
|---|---|---|
| `base` | 无 | 原路径：zigzag CP 切分 + Turbo `flash_attn_usp_func` |
| `serial` | `PRIMUS_ULYSSES_FUSED_A2A=1 PRIMUS_ULYSSES_FUSED_A2A_IMPL=serial` | 消融：contiguous 切分 + 本地 flash-attn，GEMM 与 RCCL all-to-all 串行 |
| `fused` | `PRIMUS_ULYSSES_FUSED_A2A=1` | GemmA2A / A2AGemm 融合 |

单次运行（容器内，路径按实际修改）：

```bash
export PYTHONPATH=<sitecustomize 目录>:<flydsl0341 目录>:<Primus-Turbo 目录>${PYTHONPATH:+:$PYTHONPATH}
export PRIMUS_MODEL=<模型> PRIMUS_CP=<cp> PRIMUS_EP=<ep>
export PRIMUS_SEQ_LENGTH=16384 PRIMUS_MAX_POSITION_EMBEDDINGS=16384   # 32K 时两个都改成 32768
export PRIMUS_TRAIN_ITERS=12 PRIMUS_GBS=16 PRIMUS_EXP_NAME=<前缀>_<实现><16k|32k>
# 加上实现对应的额外环境变量
cd <Primus 目录>
./runner/primus-cli direct -- train pretrain \
    --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-cp4-pretrain.yaml > <前缀>_<实现><16k|32k>.log 2>&1
```

共 6 × 2 × 3 = 36 次运行，串行执行，节点上不要有别的任务。

## 每次运行都要检查

- 日志里打印的模型参数（`num_layers`、`hidden_size`、`num_attention_heads`、`num_query_groups`、`num_experts`、`rotary_percent`、`qk_layernorm`）和预期一致。`minimax_m2.5.yaml` 用的是 `bases:` 而 Primus loader 只处理 `extends:`，要特别核对它是否拿到了 `language_model.yaml` 的默认值。
- fused / serial 日志里必须有 `[Patch] ✓ Applied: megatron.ulysses_fused_a2a`，并且**不能**出现 `keeping the stock path` 警告；base 里不能有这个 patch。patch 遇到不支持的结构（MLA、attention output gate、projection bias、packed seq）会让该层回退到原路径，这时 fused 和 serial 的耗时会和 base 很接近，要在报告里指出。
- loss 健全性：三种实现第 12 步 `lm loss` 应在 1e-3 量级内一致（serial 和 fused 的数据切分相同，应该更接近）；不能有 NaN、跳过的步。偏差明显时先停下来查，不要继续跑后面的组合。
- `llama3.1_8B` 是 dense 模型，训练配置里的 MoE 相关项（`moe_router_force_load_balancing`、`use_turbo_deepep`、`turbo_sync_free_moe_stage` 等）如果导致启动失败，复制一份配置去掉这些项再跑，并在报告里写明。
- OOM 或其他失败如实记录，不要为了跑通而改并行配置以外的东西。

## 指标

- 每步耗时：最后一步日志里 `elapsed time per iteration (ms): <本步>/<平均>` 的平均值（前 2 步 warmup 已被排除）。
- 同时记录 `TFLOP/s/GPU`、`tokens/s/GPU`、显存占用、第 12 步 loss。
- 收益拆分：`fused / base - 1` 是总收益；`fused / serial - 1` 是 GEMM 与 a2a 重叠本身的收益；`serial / base - 1` 是路径重构（contiguous 切分 + 本地 flash-attn）的收益。

## 参考数据（1×8 MI355X，ms/iter）

| 组合 | 长度 | base | serial | fused | fused 对 base |
|---|---|---|---|---|---|
| Qwen3-30B-A3B 完整，CP4 EP8 | 16K | 3147 | 3094 | 2897 | -7.9% |
| Qwen3-30B-A3B 完整，CP4 EP8 | 32K | 6583 | 6534 | 6103 | -7.3% |
| Qwen3-235B-A22B 4 层 proxy，CP4 EP8 | 16K | 634 | 632 | 597 | -5.8% |

理论估算（按每层 FLOPs / 通信字节，用 Qwen3-30B-A3B 实测校准，长序列偏保守）给出的 32K 完整模型收益：Qwen3-235B-A22B 约 7%，GLM-4.x CP4 约 8%、CP8 约 9%，MiniMax-M2.5 约 7%，Llama3.1-8B 约 9%。

注意：4 层 proxy 里 embedding / output layer / 优化器的占比比完整模型高，相对收益会比完整模型低（235B proxy 16K 是 -5.8%，估算完整模型约 -9%）。报告里要换算出“每层每 micro-batch 节省的时间”（`(base − fused) / (层数 × 每个 DP rank 的 micro-batch 数)`），并用它推算完整模型的收益：完整模型每步节省 ≈ 每层节省 × 完整层数 × micro-batch 数，再除以完整模型的 base 每步耗时（可以用 proxy 的每层耗时乘层数近似，或者如果单节点放得下就直接跑完整模型）。

## 报告

写到 `docs/ulysses_proxy_single_node_report.md`（Primus-Turbo 仓库），包含：

1. 结果总表：每个组合 × 长度的 base / serial / fused ms/iter、fused 对 base、fused 对 serial、第 12 步 loss（三种实现）、显存。
2. 每层每 micro-batch 节省时间，以及推算的完整模型收益。
3. 结论：哪个模型 / CP 配置最适合作为发布主推，理由；CP=8 是否值得先调 kernel 参数；有没有模型因为结构原因回退或数值异常。
4. 附录：每次运行的日志路径、Turbo 和 Primus 的 commit、`flydsl` 版本、节点名。

提交报告并推送到 `feat/ulysses-gemm-a2a`。
