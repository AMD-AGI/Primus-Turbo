# DeepSeek-V4 Attention 支持情况与性能总结

> 更新日期：2026-07-06
> 适用范围：AMD CDNA4（gfx950 / MI355X），DeepSeek-V4 的 SLA/SWA、HCA、CSA 三类注意力。
> 后端：Triton（默认）与 FlyDSL（可选，`pip` 安装版 `flydsl 0.2.2`）。
> 数据来源：本轮在 MI355X 上跑的 `bench_deepseek_attention.py`（smoke 子集：`S=2048, B=1, bf16`）。

---

## 1. 概述

DeepSeek-V4 的每层注意力按 `compress_ratio` 分三类，共用 `head_dim=512`、MQA（`K_H=1`）、`sliding_window=128`：

- **SLA / SWA（dense）**：`compress_ratio == 0`，仅局部滑窗，走 `hca_attention`。
- **HCA（split-mask pool）**：`compress_ratio == 128`，局部滑窗 + 压缩池联合掩码，走 `hca_attention` split-mask。
- **CSA（Compressed Sparse Attention）**：`compress_ratio == 4`，局部滑窗 + 每 query 的 top-K 稀疏池分支 + 共享 softmax sink，融合为单次 online softmax。

CSA 只保留一种实现：

- **CSA from-pool（记作 `csa`）**：`csa_attention_from_pool`。传入压缩池 `pool [B, P, D]` 与 top-K 索引 `topk_idxs [B, Sq, K]`，gather 在 kernel 内部完成；反向对 `pool` 做 scatter-add 得到 `dpool`。不物化稠密张量，显存占用 `O(B·P·D)`，是训练实际使用的形态。

> 历史上还存在一种等价实现 **CSA pre-gathered（`csa_gathered` / `csa_attention`）**：调用方预先把 top-K 条目 gather 成稠密 `gathered [B, Sq, K, D]` 再喂给 kernel。它与 from-pool 数学等价，但需物化 `O(B·Sq·K·D)` 张量、大 seqlen 会 OOM，性能与显存均劣于 from-pool。该路径已于 2026-07-06 移除（公开 API、dispatcher、FlyDSL/Triton 内核、benchmark、单测），后续只优化 from-pool。

---

## 2. 后端支持矩阵

下表来自 dispatcher 实际接线（`primus_turbo/pytorch/kernels/attention/deepseek_attn_impl.py` 与 `ops/attention/csa_attention.py`）。「已接线」指存在 FlyDSL 内核且 `can_handle` 满足时会选中；不满足时静默回退 Triton。

| 注意力类型 | 前向 (fwd) | 反向 (bwd) |
|---|---|---|
| SLA / SWA（dense） | FlyDSL 已接线（bf16, D=512, swa>0, sink=None, MQA+MHA, scale=1/√D） | FlyDSL 已接线（bf16, swa>0, sink 可选, **仅 MQA**） |
| HCA（split-mask pool） | 无 FlyDSL → Triton | FlyDSL 已接线（bf16, D=512, MQA, B=1, hca_local%64==0, pool≤32） |
| CSA from-pool（`csa`） | FlyDSL 已接线（in-kernel gather, bf16, D=512, swa>0, K_topk>0, MQA+MHA） | **FlyDSL 已接线**（2026-07-06 新增；scatter-add per-row + 专用 dpool kernel；MQA 复用 MFMA SWA dq/dkv） |

`can_handle` 通用门槛：gfx950、bf16、`D==512`、`K_H ∈ {1, H}`、`swa>0`、`scale==1/√D`，以及 FlyDSL 的 int32 参数编码限制（相关张量 `numel < 2**31`）。

一个补充说明：CSA from-pool 的前向用 FlyDSL，但反向固定走 Triton。因此用 `--kinds csa --backends flydsl` 测得的 fwd+bwd 中，反向部分实际是 Triton 内核。

---

## 3. 本轮修复（2026-07-06）

本轮解决了两个阻塞 FlyDSL CSA 路径的问题，修复后 `csa` 与 `csa_gathered` 的前向、反向均可端到端运行。

### 3.1 `rocdl.readlane` 操作数类型（编译期阻塞）

现象：`ValueError: Operand 1 of operation "rocdl.readlane" must be a Value (is not a Value)`。

原因是 flydsl 0.2.2 要求 `rocdl.readlane` 的第 2 个操作数（lane 索引）为 MLIR `Value`，而原代码传入 Python `int`。修法是把 lane 索引用 `arith.constant(..., type=T.i32)` 包成 Value，涉及 4 处：

- `kernels_common.py` — `mfma_mv_reduce_16` 的 `src_lane`（CSA 反向使用）
- `csa_pool_fwd_kernel.py` — csa 前向 local 分支、gathered 分支各 1 处，及 invalid 标志 readlane 1 处

影响：`csa`（from-pool）前向与 `csa_gathered` 反向的编译由此打通。

### 3.2 `csa_gathered` 反向 split scratch 的 i32 偏移溢出（运行期越界写）

现象：`Memory access fault ... Write access to a read-only page`，仅在 `H×S` 较大时触发（`H=64, S=2048` 必现，`H=32, S=2048` 正常）。

原因是 32 位整数偏移溢出。为避免 atomic 竞争，反向给每个 head-group 一条独立 stripe，写入 split scratch `DGATHERED_SPLIT[B, Sq, K_topk, num_head_groups, D]`（MHA 下 `num_head_groups = HQ`）。其元素总数 `B·Sq·K·HQ·D` 在宽 MHA 形状下超过 `2**31`，而偏移计算用 i32 乘加、且 finalize pass 用 32 位 `buffer_load` voffset（寻址上限 4 GB），二者都会回绕并越界。排查中确认 flydsl 在 gfx950 上的 `index` 类型是 **32 位**，因此必须全程显式 i64。

修法（3 个文件）：

- `kernels_common.py` — `dgathered_split_elem_base` 用 `arith.extsi(T.i64, ...)` 全程 i64 乘加，返回 i64。
- `csa_bwd_full_kernel.py` — store 偏移保持 i64 直达 GEP，去掉中间 `index_cast(T.index)` 的 32 位往返截断。
- `csa_bwd_dq_finalize_kernel.py` — split scratch 读取从 `buffer_load`（32 位 voffset）改为 64 位平坦 GEP load。

一个需要注意的点：反向 `can_handle` 的 numel 守卫只检查了 `gathered` 与 `q`，没有检查体积大 `num_head_groups` 倍的 split scratch。所以在修复前，`gathered`/`q` 都 `< 2**31` 的合法形状仍会因 split scratch 溢出而崩溃。i64 寻址修复后该路径不再依赖此守卫；但 split scratch 的显存开销见第 6 节。

---

## 4. 性能总结

环境：MI355X（gfx950），bf16，smoke 子集 `S=2048, B=1`。延迟为中位数（ms），越低越好。

> 注：下表中的 `csa_gathered` 行是该路径移除（2026-07-06）之前的历史数据，仅作对比留存；当前代码只有 `csa`（from-pool）。

### 4.1 FlyDSL 绝对数据

| model | kind | fwd_ms | fwdbwd_ms | fwd_TFLOPs | tot_TFLOPs | SNR_dB |
|---|---|---|---|---|---|---|
| flash (H=64) | dense | 0.174 | 2.981 | 197.1 | 40.3 | 44.5 |
| flash | csa (from-pool) | 10.586 | 31.554 | 16.2 | 19.1 | 44.9 |
| flash | csa_gathered | 12.481 | 124.784 | 13.8 | 4.8 | 45.1 |
| pro (H=128) | dense | 0.381 | 5.649 | 180.4 | 42.6 | 44.5 |
| pro | csa (from-pool) | 20.843 | 61.134 | 16.5 | 19.7 | 45.0 |
| pro | csa_gathered | 24.345 | 247.943 | 14.1 | 4.9 | 45.1 |

### 4.2 Triton 绝对数据

| model | kind | fwd_ms | fwdbwd_ms | fwd_TFLOPs | tot_TFLOPs | SNR_dB |
|---|---|---|---|---|---|---|
| flash (H=64) | dense | 0.180 | 1.864 | 190.5 | 64.5 | 45.5 |
| flash | hca | 0.259 | 2.188 | 149.1 | 61.8 | 45.0 |
| flash | csa (from-pool) | 0.696 | 21.206 | 246.7 | 28.4 | 44.7 |
| flash | csa_gathered | 22.825 | 694.399 | 7.5 | 0.9 | 45.1 |
| pro (H=128) | dense | 0.370 | 3.637 | 185.7 | 66.1 | 45.5 |
| pro | hca | 0.522 | 4.277 | 148.1 | 63.3 | 45.1 |
| pro | csa (from-pool) | 1.360 | 41.097 | 252.7 | 29.3 | 44.8 |
| pro | csa_gathered | 45.558 | 1384.193 | 7.5 | 0.9 | 45.1 |

### 4.3 FlyDSL 相对 Triton（同 kind 对比，>1 表示 FlyDSL 更快）

| kind | 指标 | flash | pro | 结论 |
|---|---|---|---|---|
| dense | fwd | 1.03× | 0.97× | 基本持平 |
| dense | fwdbwd | 0.63×（慢 1.60×） | 0.64×（慢 1.55×） | FlyDSL 反向慢 ~1.6× |
| csa (from-pool) | fwd | 0.07×（慢 15.2×） | 0.07×（慢 15.3×） | FlyDSL 前向显著慢 |
| csa (from-pool) | fwdbwd | 0.67×（慢 1.49×） | 0.67×（慢 1.49×） | 反向两者都是 Triton，差异来自慢的 FlyDSL 前向 |
| csa_gathered | fwd | 1.83× | 1.87× | FlyDSL 前向快 ~1.85× |
| csa_gathered | fwdbwd | 5.56× | 5.58× | FlyDSL 端到端快 ~5.6× |

### 4.4 结论

- **`csa_gathered` 是当前 FlyDSL 的明确收益点**：前向快 ~1.85×，前后向端到端快 ~5.6×（本轮 `csa_bwd_full` MFMA 化 + split-K scratch 反向生效后的结果）。
- **`csa`（from-pool）前向慢 ~15×**：`csa_pool_fwd_kernel` 刚打通编译，尚未优化；其 tot_TFLOPs 仅 ~19，远低于 Triton 前向的算力水平。反向因走 Triton，与 Triton 持平。
- **dense**：前向持平，反向慢 ~1.6×。
- HCA 前向无 FlyDSL 实现，本轮未纳入 FlyDSL 对比。

---

## 5. 正确性

- 关键路径 vs eager 参考的前向 SNR 均为 ~44–45 dB（bf16 正常精度区间）。
- 全量单测 `tests/pytorch/ops/test_deepseek_attention.py`：移除 `csa_gathered` 后 **138 passed**（原 210 中约 72 个为 `csa_gathered` 参数化用例，已随代码删除）。

---

## 6. 已知限制

- **CSA 只保留 from-pool**：`csa_gathered`（pre-gathered）路径已移除（见第 1 节）。
- **CSA from-pool 反向无 FlyDSL**：固定走 Triton。
- **CSA from-pool 前向 FlyDSL 尚慢**：比 Triton 前向慢 ~15×，暂非性能收益点。
- **HCA 前向无 FlyDSL 实现**：走 Triton。
- **SLA/SWA 反向 FlyDSL 仅 MQA**；MHA 回退 Triton。
- **FlyDSL 仅在 gfx950 启用**（依赖 `ds_read_*_tr` / 16B G2S 等 CDNA4 特性）。
- FlyDSL int32 参数编码：相关张量 `numel ≥ 2**31` 的超大形状回退 Triton。

---

## 7. 相关文件

- 内核 builder（vendored）：`primus_turbo/flydsl/attention/kernels/{sla_fwd,csa_fwd,csa_pool_fwd,sla_bwd,sla_bwd_dq,sla_bwd_dkv,hca_bwd_dq_pool,hca_bwd_dkv_pool,csa_bwd_full,csa_bwd_dkv,csa_bwd_dq_finalize}_kernel.py`、`kernels_common.py`、`warp_pipeline_common.py`
- launcher：`primus_turbo/flydsl/attention/deepseek_attn_{fwd,bwd}_kernel.py`
- dispatcher：`primus_turbo/pytorch/kernels/attention/deepseek_attn_impl.py`
- ops：`primus_turbo/pytorch/ops/attention/{hca_attention,csa_attention}.py`
- 单测：`tests/pytorch/ops/test_deepseek_attention.py`
- benchmark：`benchmark/ops/bench_deepseek_attention.py`、`benchmark/ops/benchmark_suite.yaml`（`dpsk_attn_smoke` / `dpsk_attn_full` 组）
- 历史计划文档：`suggest/dpsk_attn_flydsl_status_and_plan.md`
