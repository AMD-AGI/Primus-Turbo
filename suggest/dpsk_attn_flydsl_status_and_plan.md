# DeepSeek-V4 Attention FlyDSL 后端：现状与计划

> 适用范围：AMD CDNA4（gfx950 / MI355X），DeepSeek-V4 的 SLA/SWA、HCA、CSA 三类注意力，FlyDSL 后端（pip 安装版 `flydsl 0.2.0`，与参考 Primus 所基于的源码版本存在 API 差异）。
> 默认后端始终是 Triton；FlyDSL 为可选后端，`can_handle` 不满足时静默回退 Triton。

> **补全进展（2026-06-23，深度=接线+正确性达标，性能优化留后续，改动未提交）**：
> P1 CSA 前向 HG=2 已修复（根因是 banked 路径 MHA 误用 head0 的 local K/V，非 0.2.0 drift；
> 修复后 MHA/MQA 均 ~45 dB，默认仍 HG=1，HG=2 经 env 开启，性能 sweep 待做）。
> P2 HCA 反向已接线（local SWA + pool 双流分解，新增 fp32 参考单测 `test_v4_hca_attention_bwd`）。
> P3 CSA 反向 MFMA 化按深度推迟（已正确，纯性能；方案见 campaign 日志）。
> P4 CSA(from-pool) 前向 FlyDSL in-kernel gather 已实现并接线（`csa_pool_fwd_kernel`）。
> P5 benchmark suite 拆为 `dpsk_attn_smoke` / `dpsk_attn_full`。
> 全量 `test_deepseek_attention.py` = 210 passed。
> 详见 `agent/workspace/dpskv4_attn_flydsl_complete_gfx950_20260623/logs/progress.md`。

---

## 1. 现状总览

参考来源：`/apps/tas/yaoc/agent_work/mi355x/Primus` 的 `v4_attention_kernels/_flydsl`。其内核 builder 已 vendored 到本库 `primus_turbo/flydsl/attention/kernels/`（去掉了 `v4` 前缀，子包从 `v4/` 改名为 `kernels/`）。launcher 适配层在 `primus_turbo/flydsl/attention/deepseek_attn_fwd_kernel.py` 与 `deepseek_attn_bwd_kernel.py`；dispatcher 在 `primus_turbo/pytorch/kernels/attention/deepseek_attn_impl.py`。

### 1.1 FlyDSL 支持矩阵

| 注意力类型 | 前向 (fwd) | 反向 (bwd) |
|---|---|---|
| SLA / SWA（dense） | 已接线（bf16, D=512, swa>0, sink=None, MQA+MHA, scale=1/√D） | 已接线（bf16, swa>0, sink 可选, 仅 MQA, dense） |
| HCA（split-mask pool） | 未接线 → Triton（FlyDSL HCA 前向无实现） | **已接线**（P2：bf16, D=512, MQA, B=1, hca_local%64==0, pool≤32） |
| CSA（pre-gathered） | 已接线（bf16, D=512, swa>0, K_topk>0, MQA+MHA；HG=2 P1 已修，默认 HG=1） | 已接线（bf16, swa>0, K_topk>0, MQA+MHA） |
| CSA（from-pool） | **已接线**（P4：in-kernel gather, bf16, D=512, swa>0, K_topk>0, MQA+MHA） | 未接线 → Triton |

### 1.2 性能（MI355X，bf16，相对 Triton；已接线项）

| 项 | FlyDSL vs Triton | 说明 |
|---|---|---|
| SLA/SWA 前向 | 持平（~1.0x） | fwd 与 Triton 同级 |
| SLA/SWA 反向 | 慢 ~1.6x | FlyDSL dense 反向慢于 Triton |
| CSA(pre-gathered) 前向 | 快 ~1.9x | 唯一明确收益点（HG=1） |
| CSA(pre-gathered) 反向 | 慢 ~2x | 标量 per-row 内核，未 MFMA 化 |

结论：FlyDSL 当前唯一明确的性能收益是 **CSA(pre-gathered) 前向 ~1.9x**（适合推理/eval）。其余前向持平、反向更慢，因此训练端到端默认 Triton 更优。

### 1.3 正确性

- 全量单测 `tests/pytorch/ops/test_deepseek_attention.py`：**206 passed**（含 FlyDSL 维度约 99 个 + Triton 维度）。
- 关键内核 vs fp32 真值（SNR）：SWA 前向 ~49 dB；CSA 前向 ~49 dB；SWA 反向 dq/dk/dv 46–51 dB；CSA 反向（MHA+MQA）dq/dk/dv/dgathered ~55.6 dB、dsink 53.3 dB。

---

## 2. 移植过程中处理的关键问题（已修复，记录备查）

### 2.1 flydsl 0.2.0 API 差异（参考内核所基于版本与安装版不一致）

1. **编译期 `if` 作用域**：安装版 AST rewriter 把每个 `if` 体包成函数，跨块共享的编译期常量/闭包会丢失。修法：所有编译期 `if/elif` 条件用 `const_expr(...)` 包裹（运行期分支用显式 `scf.IfOp`，保持原样）。
2. **MFMA 调用约定**：改为 `rocdl.mfma_xxx(result_type, [a, b, c])`（已自动 `.result`，cbsz/abid/blgp 默认 0）。
3. **`range_constexpr` 内 `continue`**：rewriter 不支持，重构为 if/else（默认关闭的 prefetch 分支）。
4. **`kernels.kernels_common`**：源码树依赖，本库重建 `dtype_to_elem_type` 于 `kernels/kernels_common.py`。

### 2.2 CSA 前向 banked HG=2 数值错误

参考的 `HEAD_GROUP=2`（banked，对应其 2.79x）在 flydsl 0.2.0 下结果错误（SNR ~2 dB）。已按正确性降级默认 `HG=1`（~49 dB，~1.9x）。env 旋钮 `PRIMUS_TURBO_CSA_HEAD_GROUP` 保留。

### 2.3 FlyDSL int32 arg 打包上限

flydsl 把展平张量长度按 int32 打包，单张量 numel ≥ 2³¹ 会崩溃（如 Pro / S=4096 / K_topk=1024 的 `gathered`）。已在 CSA/SWA 前向与 CSA 反向的 `can_handle` 加 `numel < 2**31` 守卫，超大形状回退 Triton。

### 2.4 Triton CSA(from-pool) 反向的 dpool bug（生产级，已修）

`_csa_attention_pool_sparse_bwd_partial_kernel` 默认 segreduce 路径把 `dpool_partial[b,m,k,:]`（无 head-block 维）按 `cdiv(HQ, BLOCK_H)` 个 head-block 用 `tl.store` 写。`HQ > BLOCK_H=32`（即 HQ=64/128，生产区间）时多个 head-block 互相覆盖，dpool 只保留部分 head 贡献（vs fp32 真值 3 dB）。修法：`HQ > BLOCK_H` 时改用 `tl.atomic_add` 累加到 fp32 zero-init 的 `dpool_partial`（HQ≤32 保持原 `tl.store`+bf16 快路径）。修复后 vs fp32 真值 50 dB。对应单测 `test_v4_csa_from_pool_real_shape` 的 pool 梯度参考同步改为 fp32（bf16 参考在 K_topk=512 重用下自身仅 ~31 dB，低于阈值）。

---

## 3. 已知限制

- HCA 前向无 FlyDSL 实现（参考也没有），走 Triton。
- HCA 反向、CSA(from-pool) 反向未接线（内核 builder 已 vendored，HCA 反向有 B=1/pool≤32 等紧约束）。
- SLA/SWA 反向 FlyDSL 仅 MQA；MHA 回退 Triton。
- FlyDSL 仅在 gfx950 启用（`ds_read_*_tr` / 16B G2S）。
- 反向 FlyDSL（SWA、CSA）当前均慢于 Triton；不是训练收益点。

---

## 4. 计划（按收益/成本排序）

### P1：CSA 前向 banked HG=2 修复（纯前向收益，目标拉回接近 2.79x）
- 现状：HG=1 给 ~1.9x；参考 HG=2 给 2.79x 但在 0.2.0 下数值错误。
- 任务：定位 `csa_fwd_kernel` 的 head-group 融合在 0.2.0 下的错误点（疑似 head-block 间的规约/索引与某个 op 行为漂移），修正后用 `verify-accuracy` 校验 SNR ≥ 40 dB，再 benchmark 确认加速。
- 验收：HG=2 SNR 通过 + CSA 前向加速 ≥ HG=1。

### P2：HCA 反向接线（参考里唯一比 Triton 快的反向：HCA dkv_pool）
- 现状：`build_hca_bwd_dq_pool_module` / `build_hca_bwd_dkv_pool_module` 已 vendored，未接线。
- 任务：写 launcher（local 用 SWA bwd dq/dkv + pool 用 hca pool dq/dkv），按约束 gating（B=1、pool_size≤32/64、hca_local_seqlen%64==0、MQA），接到 `HCAAttentionFn.backward` 的 dispatcher（dense bwd dispatcher 已有，需扩展 HCA 分支或新增）。
- 风险：约束紧；HCA 反向当前无单测覆盖，需要新增 HCA 反向用例（fp32 参考）。
- 验收：HCA 反向 SNR 通过 + HCA dkv_pool 段相对 Triton 有加速。

### P3：CSA 反向 MFMA 化（把已接线但慢 ~2x 的反向做快）
- 现状：FlyDSL CSA 反向是 per-row 标量内核（correctness-only），慢 ~2x。
- 任务：参照前向 CSA 内核（tile + MFMA）重写反向，QK/dP/dS/dQ/dK/dV/dgathered 用 MFMA atom，sparse 分支 head-block 分块。属内核优化，非简单移植。
- 验收：CSA 反向不慢于 Triton 且 SNR 通过。

### P4：CSA(from-pool) 前向 FlyDSL（in-kernel gather）
- 现状：无 FlyDSL pool-gather 前向内核（参考只有 pre-gathered）。
- 任务：在 CSA pre-gathered 前向基础上加 in-kernel gather（按 topk_idxs 从 pool 取行），接 `deepseek_csa_pool_attn_fwd` dispatcher。
- 价值：避免物化 `[B,Sq,K_topk,D]`，省显存带宽；前向有望接近 pre-gathered 的加速。

### P5：benchmark suite 常规化
- 现状：`benchmark_suite.yaml` 已加 `dpsk_attn` 组（triton/flydsl）；默认 seqlen 含 8192，csa_gathered 在大形状物化开销大、Pro/S=4096 会 OOM。
- 任务：为 CI 常规跑设更小的 `--seqlen` 子集；或拆出 `dpsk_attn_smoke` 与 `dpsk_attn_full` 两组。

---

## 5. 相关文件

- 内核 builder（vendored）：`primus_turbo/flydsl/attention/kernels/{sla_fwd,csa_fwd,sla_bwd,sla_bwd_dq,sla_bwd_dkv,hca_bwd_dq_pool,hca_bwd_dkv_pool,csa_bwd_full,csa_bwd_dq}_kernel.py`、`kernels_common.py`
- launcher：`primus_turbo/flydsl/attention/deepseek_attn_{fwd,bwd}_kernel.py`
- dispatcher：`primus_turbo/pytorch/kernels/attention/deepseek_attn_impl.py`
- ops：`primus_turbo/pytorch/ops/attention/{hca_attention,csa_attention}.py`
- Triton CSA(from-pool) 反向（本次修复）：`primus_turbo/triton/attention/deepseek/csa_attention_bwd.py`
- 单测：`tests/pytorch/ops/test_deepseek_attention.py`
- benchmark：`benchmark/ops/bench_deepseek_attention.py`、`benchmark/ops/benchmark_suite.yaml`（`dpsk_attn` 组）
