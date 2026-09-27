# L6 原型：QK(i+1) 与 softmax(i) 软件流水（仅编译，未上卡）

builder 的完整报告没有落盘（harness 拦下了写文件），见 workflow 输出。本文件只有对抗性 review。

## Review（2026-09-27，仅编译，未碰 GPU）

**结论：needs-fix，已就地修复。修复后可以上卡（ready-for-card）。**

### 1. 阻断性问题：非 causal 和 gqa=1 会 spill（已修）
builder 只编译了 causal + gqa=4，但 validation.py 的 edge shape 还会覆盖非 causal（toy/short_q/gqa4_batch2/mha/unequal*/sq_gt_skv）以及 gqa=1（mha）和 gqa=2（toy）。
原型把 `PIPELINE_QK_SOFTMAX` 设成了全局开关，这些配置下的结果如下（prod 形状，bf16，d128，`review/isa/`）：

| 配置 | 冠军 VGPR/spill | 原型（修复前） |
|---|---|---|
| causal gqa4 | 445/0 | 508/0 |
| causal gqa2 | 445/0 | 508/0 |
| causal gqa1 | 443/0 | **512/85 spill，scratch 344 B** |
| 非 causal gqa1/2/4 | 432/0 | **512/92 spill，scratch 372 B** |

validation 一跑这些 kernel 就会用到 scratch spill，按项目规则有挂卡风险。
**修复**：在 `_core_attention` 里加了编译期门控 `_PIPE`（`op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py` 约第 757 行）。只有同时满足 `mask_right and not mask_left and gqa_ratio in PIPE_GQA_OK(2,4) and qk_hdim==128 and v_hdim==128` 时才走流水，其它配置一律回到冠军主体。原来所有用 `PIPELINE_QK_SOFTMAX` 的地方都改成了 `_PIPE`。

修复后重新编译验证：
- causal gqa4：508/0，ISA 与 builder 的 `isa/pipe_prod` 逐字节相同；
- causal gqa2：508/0；
- 非 causal gqa1/2/4、causal gqa1、thd 非 causal：ISA 与冠军**逐字节相同**（diff 0 行）；
- thd causal：508/0；
- fp16 causal gqa4：511/0（只剩 1 个 VGPR 余量，job 不测 fp16）。

代价：win_sink（mask_left）现在也回到冠军（455 VGPR，4 个 SGPR spill，与冠军相同）。job 不测 window。qk_hdim 192/256 同样不走流水，没有编译验证过。

### 2. LDS 环正确性（推演 + ISA 数据流检查，未发现问题）
- **slot 相位**：第 i 步从 cur=(i-start+1)%2 读 [K(i+1)|V(i)]，向 (i-start)%2 预取 [K(i+2)|V(i+1)]。这个 slot 的上一批读者是第 i-1 步（读 K(i)、V(i-1)），第一步时是 prologue 里 QK(start) 读 slot 0。这些读都在本步顶部 WG barrier 之前。carried 指针的交换与 `nxt_pp` 的奇偶一致。
- **n_iter=1**：三段循环都是 0 次，peel 从 slot 1 读 V(start)。O 写到 slot 0，slot 0 只被 prologue 的 QK 读过。K 的重复拷贝进 slot 1，没有人读，并由 peel 的 tensor_wait+barrier 排空。
- **n_iter=2**：第 start 步把 V(start+1) 写进 slot 0，K(start+2) 按 `i+2<n_tiles` 跳过。peel 读 slot 0，O 写到 slot 1，slot 1 最后一次被读是在第 start 步，在 peel 的 barrier 之前。
- **Q 与 slot 1**：新增的 `s_wait_dscnt 0` 在 ISA 里确实位于 prologue barrier 之前（LO 第 450 行，HI 第 1684 行）。
- **ISA 级验证**（`review/dscnt_cfg.py`）：在 CFG 上对所有路径做数据流，统计每个 `s_barrier_signal` 处最多还有几条未完成的 LDS 操作，gqa4 causal 和 thd 的 7 个 barrier 点全部是 0。
  - 这很重要，因为 `gpu.barrier` 本身不带 dscnt fence。
  - 每个 barrier 前都有 `s_wait_tensorcnt 0`。
  - LO/HI 两条路径上的 WG barrier 次数都是 1+n_iter，与冠军相同。LLVM 把两种 warp 的 peel 合并成了一段代码（第 3809 行），barrier 数不受影响。
- **越界**：循环里 V(i+1) 满足 i+1≤n_last。K(i+2) 按条件发出。prologue 的 K(start+1) 用 select 限制在界内。kv_len==0 在外层走 `_zero_fill_attention`，不会进入 core，所以 n_tiles≥1，peel 总是有效的。

### 3. 数值与分段
- mask 标志跟着做 softmax 的 tile i 走。分段 [start,clean_lo)、[clean_lo,clean_hi)、[clean_hi,n_last) 加上 peel n_last，与冠军的 [.., n_tiles) 划分完全一致，kv_len 尾部 mask 也在同样的 tile 上。
- 每个 tile 的 QK/softmax/rescale/PV 序列不变，预期 O/LSE 与冠军逐位相同。
- 没有原子操作，结果是确定性的。
- GQA 头映射（`_packed_tile_indices`）没有改动。
- 状态布局 `_S_BASE`/`_PTR_BASE` 与 yield 顺序一致：m,d,O×R 在前，然后是 S×R×4，最后是指针。

### 4. ISA 数据复核
在 fa-repro 里用 compile-only 重跑，builder 的数字全部复现：
- VGPR 508、SGPR 96、spill 0；
- clean-LO 694 条 / clean-HI 751 条，masked 950/977 条；
- v_exp 在 WMMA 之后 8 条以内：59/66、54/66、0/66、0/66；
- 开关设为 False 时与冠军逐字节相同（`isa/off_*`）。

### 5. 上卡时仍需确认的
- 与冠军逐位对比 O/LSE：必须覆盖 toy（gqa2 走流水）、短序列 n_iter=1/2、unequal_seqlen*、short_q、kv_len%64≠0，同一 shape 重复 80 次（mode-2 历史）。
- 非 causal 和 mha 现在应该与冠军逐位相同，因为走的是同一个 kernel。
- VGPR 508/512，叠加任何新 lever 都要先重跑 `review/run.sh` 的 6 个配置矩阵（gqa 1/2/4 × causal 0/1）再上卡。

### 文件
- `op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`：已加门控。修复前的版本在 `review/fmha_pre_review.py`。
- `pipeline.patch`：已重新生成。旧版在 `review/pipeline.pre_review.patch`。
- `review/rc.py`、`review/run.sh`：编译矩阵，环境变量 GQA/CAUSAL/D/DT。
- `review/dscnt_cfg.py`：检查 barrier 处未完成的 LDS 操作。
- `review/isa/<impl>_g<gqa>_c<causal>`、`op_thd`、`op_thd_c0`、`op_ws`、`*_fp16_*`：各配置的 ISA。
