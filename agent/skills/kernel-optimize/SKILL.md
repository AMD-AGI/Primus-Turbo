---
name: kernel-optimize
description: AI 驱动的算子性能优化闭环。定义优化流程、知识引用与日志约定，驱动 agent 自主迭代逼近硬件极限。
---

# 算子性能优化

本 skill 驱动 AI agent 对算子进行持续性能优化。
agent 按闭环流程运行：改 kernel → 验正确性 → 量性能 → 记录 → 决策下一步。

## 输入参数

启动优化前，需明确以下参数（由用户指令或配置提供）：

| 参数 | 说明 | 示例 |
|------|------|------|
| `target_op` | 目标算子 | `gemm_fp8_blockwise` |
| `target_lang` | 实现语言 | `TRITON` / `HIP` |
| `target_gpu` | 目标 GPU 架构 | `gfx942` / `gfx950` |
| `target_shapes` | 测试形状（可选，默认用 benchmark suite） | `4096x4096x4096` |
| `performance_target` | 性能目标（可选） | `>500 TFLOPS` 或 `>60% 峰值效率` |

## 优化流程（高层）

```
BASELINE → ANALYZE → OPTIMIZE → VALIDATE → COMMIT → REPORT
                ▲                    │                  │
                │    失败/退步：回滚  │                  │
                ├────────────────────┘                  │
                │             继续优化                   │
                └───────────────────────────────────────┘
```

| 阶段 | 做什么 |
|------|--------|
| **BASELINE** | 跑单测 + benchmark，记录起点 |
| **ANALYZE** | 读代码 + profile + 查 skill + 查优化日志 → 生成优化假设列表 |
| **OPTIMIZE** | 选假设 → 改 kernel → 构建 |
| **VALIDATE** | 跑单测（硬门控）+ benchmark → 对比当前最佳 |
| **COMMIT** | 通过门控且有提升 → 写日志 + git commit |
| **REPORT** | 汇总成果 + 未尝试方向 → 决定继续或停止 |

详细步骤、门控规则、日志格式、停滞检测与回滚策略见 **[workflow/optimize-loop.md](workflow/optimize-loop.md)**。

## 知识引用表

根据 `target_lang` 和 `target_gpu` 按需读取对应 skill。**不要一次全读**，按当前阶段需要的信息查阅。

| 需要什么 | 读哪里 |
|---------|--------|
| 详细优化工作流 | [workflow/optimize-loop.md](workflow/optimize-loop.md) |
| Triton 通用优化技巧 | [triton/SKILL.md](triton/SKILL.md) |
| Triton 算子专项策略 | [triton/ops/\<op\>.md](triton/ops/)（如 `gemm.md`、`attention.md`） |
| HIP/CK 通用优化技巧 | [hip/SKILL.md](hip/SKILL.md) |
| HIP 算子专项策略 | [hip/ops/\<op\>.md](hip/ops/)（如 `gemm.md`） |
| 硬件参数与优化策略 | [../hardware/\<arch\>/SKILL.md](../hardware/) + `optimization-guide.md` |
| Profiling 方法 | [../tool-rocprof/SKILL.md](../tool-rocprof/SKILL.md) |
| 构建 / 测试 / benchmark | [../primus-turbo-develop/SKILL.md](../primus-turbo-develop/SKILL.md) |
| 参考实现与论文 | [references/](references/)（CK、AITER、hipkittens、关键论文） |
| 历史优化案例 | [examples.md](examples.md) |

### 典型查阅路径

**优化 Triton FP8 GEMM on MI300X**：
1. `workflow/optimize-loop.md` — 流程步骤
2. `../primus-turbo-develop/SKILL.md` — 构建和测试命令
3. `../tool-rocprof/SKILL.md` — profile 方法
4. `../hardware/gfx942/SKILL.md` + `optimization-guide.md` — MI300X 硬件约束与策略
5. `triton/SKILL.md` + `triton/ops/gemm.md` — Triton GEMM 优化技巧
6. `references/aiter.md` — 参考 AMD 官方实现

## 优化日志

Agent 在优化过程中维护结构化日志，存放于 `agent/logs/`（运行时自动创建，不纳入 git）。

- 命名：`<op>_<lang>.md`（如 `gemm_fp8_blockwise_triton.md`）
- 作用：对人可随时查看进展；对 agent 可回溯历史避免重复尝试
- 格式：见 [workflow/optimize-loop.md](workflow/optimize-loop.md) 中的日志格式模板

## 相关 skill

| Skill | 说明 |
|-------|------|
| `primus-turbo-develop` | 构建、测试、benchmark、后端系统 |
| `hardware/gfx942` | MI300X/MI325X 硬件参数与优化策略 |
| `hardware/gfx950` | MI350X/MI355X 硬件参数与优化策略 |
| `tool-rocprof` | rocprof profiling 工具用法 |
