# HANDOFF — 根治 MegaMoE STEP3(fc1 dgrad + combine)fp8-PUSH combine 跨 rank 死锁

> **When**: 2026-07-22
> **给谁**: 换机器继续这项工作的 agent。所有路径用**相对 repo root**（容器内路径与 repo 路径一致）。
> **分支**: `feat/xiaompen/mega_moe_flydsl_mxfp8`
> **注**: 详细 perf notes 在本机 `slab/`（一个指向 repo 外 `../slab/` 的 symlink，**不随 repo 分发**）；本文件已把关键信息全部内联，是自包含的。

---

## 任务
根治 MegaMoE STEP3(fc1 dgrad + combine)的 **fp8-PUSH combine 跨 rank reduce-flag 死锁**，让 STEP3 能稳定 bench 并给出 fp8-vs-bf16 数 / 打通 e2e@8192。

## 背景
本 repo(`MegaMoE`）从 `Primus-Turbo` vendored 了一套自包含的 MXFP8 mega MoE 栈(`primus_turbo/flydsl/mega/fp8/`），与 bf16 栈分离。前向(L1/L2/fwd）+ 反向(dispatch_fc2_dgrad / fc2_wgrad / fc1_wgrad / fc1_dgrad_combine）共 **8 个 stage** 已接入训练 bench `benchmark/ops/training/bench_mega_moe_fp8.py`。**除 STEP3 外，其余 7 个 stage 都能稳定给出 fp8-vs-bf16 数**。

**唯一未解决的是 STEP3 的 fp8-PUSH combine 跨 rank 死锁**：独立 bench 和 training bench 都会 hang，源 `Primus-Turbo` 的 `test_step3_bench` 同样 hang → 是继承自源的 kernel 内在竞争，非移植引入。

## 死锁定位（已排查）
- 融合 kernel：`primus_turbo/flydsl/mega/fp8/grouped_gemm_combine_fp8_kernel.py` 的 `_compile_bwd` / `grouped_gemm_combine_fp8_bwd`。一个 grid 里 3 个 role：
  - **COMBINE**（block `[0,ncomb)`）：spin `sb_l2`（等 GEMM 完成）→ 读本地 fp8 dx → push fp8 payload+E8M0 到 peer `comb[slot]` → 置 `barrier_local[slot]` sys flag（+ backward 的 gate scatter）。
  - **REDUCE**（dedicated `[ncomb, ncomb+nreduce)` + tail 空 GEMM block）：自旋等 `barrier_local[slot]`（等 peer 的 push 落地）→ fp8 dequant 加权 topk 求和 → `output`。
  - **GEMM**（`[gemm_base,...)`）：mxfp8 NT tile + CShuffle mxfp8-quant epilogue → 本地 fp8 dx pool + `atomic_add sb_l2`。
- **根因**：REDUCE 的 warp 自旋等 `barrier_local` flag，但产生该 flag 的 GEMM/combine tile 可能还排在这个 reduce block **后面未被调度** → co-scheduling 死锁；跨 rank 上 reduce 等 peer 的 push、peer 的 push 等本地 GEMM，占 CU 后互相等 → 跨 rank spin-deadlock。
- 已排除：gate-scatter、`num_combine_cu`、专用 reduce 都不是根因。`PT_COMBINE_NO_REDUCE`（仅 GEMM+push）**稳**；前向 L2 fp8 combine(`grouped_gemm_combine_fp8`，同结构）**8/8 稳**。反向更易触发，原因见代码注释：反向 mxfp8 GEMM `K=2I` 更重 + occupancy 更低，窗口更宽。

## 修法方向
**拆「融合 GEMM+push → host barrier → 独立 reduce」**，让 reduce 不再和 GEMM co-schedule：
1. **Launch A**（GEMM + combine push，无 reduce）：直接复用 `_compile_bwd` 里已有的 `PT_COMBINE_NO_REDUCE` 路径（只跑 COMBINE role 读本地 dx→push peer comb+置 barrier flags+gate scatter，和 GEMM role 产本地 fp8 dx+bump sb_l2）。
2. **Host barrier**：`torch.cuda.synchronize(); group.barrier()`（保证所有 rank 的 push payload+E8M0+flags 全部落地并跨 rank 可见）。
3. **Launch B**（独立 reduce）：新起一个纯 reduce kernel（把 `_make_topk_reduce_fp8_bwd` 包成 `@flyc.kernel`，grid 全是 reduce block），读 `comb`+`barrier_local` → `output` + `d_topk_w`。因 push 已全部落地，不再有 co-scheduling 等待 → 无死锁。
- 在 op 层 `primus_turbo/pytorch/ops/moe/mega_moe_fused_fp8.py::_mxfp8_step3_fc1_dgrad_combine` 里把单次调用改成「launch A → `_host_rendezvous` → launch B」。
- 代价：多一次 kernel launch + 一次 host barrier（STEP3 本就每次 reset+rendezvous，可接受），换稳定。
- 前向 L2 目前稳，先**只改 STEP3 backward**；如需再同法处理前向。

## 关键文件（相对 repo root）
- `primus_turbo/flydsl/mega/fp8/grouped_gemm_combine_fp8_kernel.py` — 核心修改点（`_compile_bwd`、`grouped_gemm_combine_fp8_bwd`、`_make_topk_reduce_fp8_bwd`、`combine_copy_fp8_tile`）
- `primus_turbo/pytorch/ops/moe/mega_moe_fused_fp8.py` — op 层（`_mxfp8_step3_fc1_dgrad_combine`、`_host_rendezvous`）
- `primus_turbo/flydsl/mega/fp8/sym_layout.py`、`primus_turbo/flydsl/mega/fp8/symm_buffer.py` — 跨 rank buffer 协议（`sb_l2` / `barrier_local` / `comb` / `combine_gate`）
- `benchmark/ops/bench_step3_fp8.py` — STEP3 独立 bench（smoke+延迟，复现死锁最快）
- `benchmark/ops/training/bench_mega_moe_fp8.py` — 训练 bench，`--stage fc1_dgrad_combine`
- 参考源（若该机也有）：`../Primus-Turbo/tests/pytorch/modules/test_mega_moe_mxfp8.py`（`test_step3_bench` / `test_step3_fp8push_bench`）
- 本机 perf notes（若可访问）：`slab/notes/MegaMoeFlydsl/2026-07-21_1010_mega_fp8_backward_perf_t8192_mi355x.md`（反向，含死锁分析 + 全阶段数）、`.../2026-07-20_1520_mega_fp8_forward_perf_mi355x.md`（前向）

先读 workspace 规则 `.cursor/rules/iteration_rules.mdc` 和 skill `.cursor/skills/kernel-optimize/SKILL.md`（GPU kernel 优化硬约束：单变量迭代、正确性先于性能、快照/回滚）。

## 复现 / 验证命令（container 内，repo root）
```bash
# 复现死锁（最快）：独立 STEP3 bench，每次换全新 MASTER_PORT
PYTHONPATH=$(pwd) MASTER_PORT=<fresh> MEGA_BENCH_TIMEOUT_S=90 \
  python3 benchmark/ops/bench_step3_fp8.py --num-processes 8 --num-tokens 8192
# 训练 bench 里的 stage
PYTHONPATH=$(pwd) MASTER_PORT=<fresh> \
  python3 benchmark/ops/training/bench_mega_moe_fp8.py --num-tokens 8192 --stage fc1_dgrad_combine --mode load_balanced
```

**验收**：
1. 正确性 —— dx SNR ≥ 15 dB（可用源 `test_step3_fp8push` 的 finite-SNR 口径，或 e2e gradcheck）。
2. 稳定性 —— **T=8192 连跑 30–50 次零死锁**。
3. 性能 —— 记录 fp8 STEP3 ms/TFLOPS（之前侥幸跑通一次为 ~3.26 ms）。

## 环境坑（务必遵守）
- 8×MI355X（gfx950），容器 `xiaoming-dev`（`rocm/primus:v26.3`）。**新容器先** `bash slab/notes/MegaMoeFlydsl/_repro/fix_flydsl.sh`（把 flydsl 升到 0.2.4；该脚本也在本机 symlink 下）。
- 每次跑换**全新 MASTER_PORT**（TCPStore TIME_WAIT 会导致 `TCPStore` 创建失败）。
- **死锁 hang 后必须彻底清残留**：`pkill -9 python3` 常抓不到自旋的 combine worker（cmdline 不匹配）。用 `rocm-smi --showpids` 找占 GPU 的 `UNKNOWN` PID，`kill -9 <pid>`，再 `rocm-smi --showmeminfo vram` 确认 8 卡回到 ~298 MB 基线。**不清干净会导致后续 run 反复 hang / 抢不到 GPU**。

## 已完成的进展（本次会话）
- training bench `benchmark/ops/training/bench_mega_moe_fp8.py` 加齐 8 个 stage：前向 `l1`/`l2`/`fwd`、反向 `dispatch_fc2_dgrad`/`fc2_wgrad`/`fc1_wgrad`/`fc1_dgrad_combine`。反向 stage 不含在 `--stage both` 里。
- 各 stage 与源 `test_*_bench` 逐项对齐（尤其 dW2：同 E 下 GEMM/M_pool/breakdown 完全一致；「1.4ms vs 1.77ms」差异 = 源 `_E=32` vs DSv3 真实 `E=256`，非 bug）。
- STEP3 的 bf16 对比腿代码就绪（分离 bf16 stack + nt 路径），但受 fp8-PUSH combine 死锁阻塞，拿不到稳定同 run 对比 → 本任务解锁后即可。
