# MegaAttn：下一步任务的 prompt（留档，暂不做）

> 把下面整段交给 agent 作为任务描述。前置工作在 `feat/ulysses-gemm-a2a` 分支（Ulysses GemmA2A / A2AGemm，见 `docs/ulysses_gemm_a2a.md`）。

---

在 Primus-Turbo 里用 FlyDSL 实现 **MegaAttn**：把 Ulysses attention 层的 attention 本身也放进 GemmA2A / A2AGemm 的融合流水线，做到 “QKV GEMM → a2a → attention → a2a → o_proj GEMM” 之间没有整段的阶段边界。

## 背景

- 现有实现（`primus_turbo/flydsl/ulysses/`，`primus_turbo/pytorch/ops/attention/ulysses_fused_a2a.py`）：
  - `ulysses_qkv_gemm_a2a`：QKV GEMM 的 tile 直接 pull 到 a2a 之后的 `[b, s, h/n, d]` 布局（GemmA2A）。
  - `ulysses_a2a_o_gemm`：attention 输出 O 由发送方 push 进目标 rank 的 recv bank，o_proj GEMM 按 m-tile 等 flag 后计算，epilogue 里加 residual（A2AGemm）。
  - 反向复用两者；flag 用 epoch + 双 bank，无全局 barrier，无 D2H 同步，可 CUDA graph capture。
- 现在 attention 是单独的 `flash_attn_func`：它必须等 QKV 全部到齐才开始，o_proj 必须等 attention 全部算完才开始收数据。在 Qwen3-30B-A3B n=4 32K 上，两侧交换阶段 fused 后仍有约 400–700 µs，其中大量是等最慢 rank 的时间。

## 目标

1. **attention 消费端流水**：attention 按 (head, q-tile) 粒度等对应的 Q 行以及所需的 K/V 行的 flag（GemmA2A 写入时按 tile 置位），到齐就算，不等整个 QKV a2a 结束。causal 下 q-tile i 只需要 K/V 的前 i 个块，天然可以先算前面的块。
2. **attention 生产端流水**：attention 每算完一个 (head, q-tile) 的 O，就把这一块直接 push 到序列所属 rank 的 o_proj recv bank 并置 flag（替代现在 A2AGemm 的 push 阶段）；o_proj GEMM 按 m-tile 等齐所有来源后计算，epilogue 加 residual。
3. 反向对称：dO 由 o_proj dgrad（GemmA2A）按 tile 送达，attention backward 按 tile 消费；dq/dk/dv 按 tile push 给 QKV dgrad（A2AGemm）。
4. 保持现有约束：op 返回后无常驻或自旋 kernel；无全局 barrier；无 D2H 同步；可 CUDA graph capture；每个自旋都有超时诊断；容忍 rank skew；结果按位确定（固定累加顺序）。

## 要先回答的问题（不确定的先问我）

- attention kernel 用哪个实现做底座（Turbo 的 FlyDSL / Triton / CK flash-attn），能否在 kernel 内插入 “等 flag” 和 “push O 块” 的钩子；如果不能，是否拆成多个 kernel + 细粒度 flag 也能拿到大部分收益。
- 一个 kernel 里同时跑 GEMM tile、attention tile、通信 block 的 CU 划分和死锁分析（参照现有 `num_lead / num_pull_cu / num_tail_pull` 的排布原则：被等待的工作必须先派发）。
- causal 负载不均：attention tile 的完成顺序和 o_proj 的 m-tile 顺序如何对齐，避免 o_proj 等最后一个 q-tile。
- 显存：O 不再整块落到普通张量，但 backward 仍需要 O 和 LSE；保存策略。

## 验证

- 模型：Qwen3-30B-A3B，n=4，seq 32K（主），16K / 128K 抽查。
- 正确性：沿用 `tests/pytorch/ops/test_ulysses_fused_a2a.py` 的方式，与 fp32 参考比较 fwd 输出和 dx、dW_qkv、dW_o、d_residual；多层、CUDA graph、非默认 stream、buffer poisoning。
- 性能：`benchmark/ops/training/bench_ulysses_a2a.py --fused`（整层逐阶段，关键 rank / 平均 / 最大）和 `bench_ulysses_overlap.py`（串行 vs overlap vs roofline）。目标是整层 fwd+bwd 相对当前 GemmA2A / A2AGemm 再降低，并给出 attention 本身的耗时变化。
