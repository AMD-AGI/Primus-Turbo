---
name: kernel-optimize
description: AI 驱动的算子性能优化框架。定义优化闭环、执行环境选择、知识路由与日志约定，驱动 agent 自主迭代逼近硬件极限。
---

# 算子性能优化框架

本 skill 定义**通用的算子优化闭环**，驱动 agent 自主迭代逼近硬件极限。

本 skill 负责定义：
- 优化所需的前置信息（接口契约）
- 优化 campaign 的参数确认和目录结构
- 高层优化闭环（参考 AVO: Agentic Variation Operators）
- 知识路由方式
- 日志、lineage、停滞处理的总原则

本 skill **不负责**：
- 硬编码某个项目的源码路径、测试命令、benchmark 命令（由项目 skill 提供）
- 替代语言专项、硬件专项、profiling 专项文档

## 前置信息需求

**本节是 kernel-optimize 对项目 skill 的接口契约。** 在启动优化前，agent 必须从对应项目 skill 收集以下信息：

| 需求项 | 说明 | 到项目 skill 哪里找 |
|--------|------|---------------------|
| **kernel 源文件路径** | 要优化的 kernel 代码位置 | 代码结构 / 文件映射表 |
| **focused test 命令** | 限定目标算子 + backend 的正确性测试（全量） | Testing 节 |
| **focused benchmark 命令** | 限定目标 backend 的性能测试（全量） | Benchmark 节 |
| **quick 验证脚本模板** | 直接调用算子 API 对代表性 shape 做 correctness + benchmark 的自包含脚本模板；PREPARE_ENVIRONMENT 阶段生成到 campaign 目录（shapes 暂留空），BASELINE 后填入代表性 shapes | 项目 skill 的 Quick 验证节 |
| **benchmark 输出格式** | CSV 列名，哪些列是可用性能指标（`Forward TFLOPS`、`Backward TFLOPS` 等），哪一列是正确性门控（`Check`） | Benchmark 输出说明 |
| **评分规则** | 如何从 benchmark 输出计算 `aggregate score`（如几何平均） | 项目 skill 的算子优化评分节 |
| **execution_mode 建议** | `repo-mode` 还是 `workspace-mode`，以及对应的构建/rebuild 方式 | 项目 skill 的算子优化环境节 |
| **rebuild 要求** | 改完代码后是否需要重新构建，构建命令是什么 | Build 节 |

agent 收齐以上信息后，再回到本文件执行 DEFINE_TARGET。

## 输入参数

DEFINE_TARGET 阶段需要将用户指令 + 前置信息整理为以下结构化参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `target_op` | 目标算子 | `gemm_fp8_blockwise` |
| `target_backend` | 目标 backend | `TRITON` |
| `target_lang` | 实现语言 | `TRITON` / `HIP` |
| `target_gpu` | 目标 GPU 架构 | `gfx942` / `gfx950` |
| `target_shapes` | 目标测试 / benchmark 配置 | 若干代表性 shape 或 `all`（使用 benchmark 默认 shape 集） |
| `performance_target` | 性能目标 | `>500 TFLOPS` 或 `>60% 峰值效率`；未指定时默认"相对 baseline 有可测量提升" |
| `primary_metric` | 主性能指标（可多个，取决于算子类型） | GEMM: `"Forward TFLOPS"` / `"Forward TFLOPS, Backward TFLOPS"`；elementwise: `"Forward GB/s"` |
| `project_skill` | 对应项目 skill | `primus-turbo-develop` |
| `execution_mode` | 执行环境，参考项目 skill 建议，由 agent 判断 | `repo` / `workspace` |
| `git_commit` | 是否对 accepted version 做 git commit | `true`（默认）/ `false` |
| `git_branch` | 优化分支策略 | `auto`（默认，自动创建 `optimize/<campaign>` 分支）/ `none` / `<自定义分支名>` |
| `max_iterations` | 最大迭代轮数（可选） | `10`；未指定时由 agent 根据终止条件判断 |
| `max_duration` | 最大运行时长（可选） | `"4h"`、`"1.5h"`；未指定时不限时 |

## 总体闭环

```text
DEFINE_TARGET
  -> PREPARE_ENVIRONMENT
  -> BASELINE
  -> [ANALYZE -> OPTIMIZE -> VALIDATE]  (迭代循环)
  -> REPORT
```

| 阶段 | 做什么 |
|------|--------|
| **DEFINE_TARGET** | 将用户指令 + 项目 skill 已收集的信息整理为结构化参数，确认齐全，**向用户确认目标后再开始** |
| **PREPARE_ENVIRONMENT** | 建立 campaign 目录，记录元信息，生成 quick 验证脚本（shapes 暂留空） |
| **BASELINE** | 记录起点 correctness 与 performance |
| **ANALYZE** | 读代码、profile、查技能知识、生成优化假设 |
| **OPTIMIZE** | 实施单个主假设，对实现做小步修改 |
| **VALIDATE** | correctness 硬门控 + benchmark 比较，通过则 accept（+ git commit，若 `git_commit=true`），不通过则 rollback |
| **REPORT** | 汇总最佳版本、有效方向、失败方向和下一步建议，交还项目 skill 做最终验收 |

详细优化过程、门控规则、回滚、停滞检测、日志模板见 [`workflow/optimize-loop.md`](workflow/optimize-loop.md)。

## DEFINE_TARGET

agent 到达此处时，应已按「前置信息需求」从项目 skill 收齐了所需信息。本阶段将用户指令 + 前置信息整理为结构化参数。

**Step 1: 填充参数**

| 参数 | 提取方式 |
|------|---------|
| `target_op` | 从用户指令中识别算子名称和精度 |
| `target_backend` | 从用户指令中识别；若未指定，从项目 skill 的 backend 表中选择 |
| `target_lang` | 由 `target_backend` 决定：`TRITON` → Triton，`CK` / `HIPBLASLT` / `TURBO` → HIP |
| `target_gpu` | 从用户指令中识别 GPU 型号，映射到架构代号（如 MI300X → `gfx942`，MI355X → `gfx950`） |
| `target_shapes` | 若用户指定则使用；否则使用 benchmark 默认 shape 集 |
| `performance_target` | 若用户指定则使用；否则默认"相对 baseline 有可测量提升" |
| `primary_metric` | 从项目 skill 的评分节获取可用指标；若用户指定则使用 |
| `execution_mode` | 参考项目 skill 的建议，由 agent 根据任务特点判断 |
| `git_commit` | 默认 `true`；若用户指定不 commit 则设为 `false` |
| `git_branch` | 默认 `auto`；若用户指定则使用 |
| `max_iterations` | 若用户指定则使用；否则留空，由终止条件控制 |
| `max_duration` | 若用户指定则使用；否则留空，不限时 |

**Step 2: 确认前置信息齐全**

对照「前置信息需求」做最终检查：
- [ ] kernel 源文件路径
- [ ] focused test 命令
- [ ] focused benchmark 命令
- [ ] quick 验证脚本模板
- [ ] benchmark 输出格式和可用性能指标列
- [ ] 评分规则（如几何平均）
- [ ] `execution_mode` 决策
- [ ] 改动后是否需要 rebuild，rebuild 命令

若有缺失，回查项目 skill 补齐。

**Step 3: 向用户确认目标**

将 agent 推断的关键参数列出，向用户确认后再开始。至少包括：

- `target_op`、`target_backend`、`target_gpu`
- `primary_metric`：只优化 forward？还是 forward + backward？还是自定义指标？
- `performance_target`：有具体数字还是"尽量提升"？
- `execution_mode`：repo 还是 workspace？
- `git_commit` / `git_branch`
- `max_iterations` / `max_duration`（如有）
- 特殊约束（如不能改动某些接口）

用户可以直接确认，也可以调整参数。确认后进入 PREPARE_ENVIRONMENT。

## PREPARE_ENVIRONMENT

建立本轮优化的 campaign 目录，并根据 `git_branch` 参数创建优化分支。

**Step 1: 创建优化分支**（若 `git_branch` 不为 `none`）

- `git_branch=auto`：`git checkout -b optimize/<campaign_name>`
- `git_branch=<自定义>`：`git checkout -b <自定义分支名>`
- `git_branch=none`：不切分支，在当前分支工作

**Step 2: 建立 campaign 目录**

```text
agent/workspace/<campaign_name>/
├── logs/
│   └── optimize.md       # 优化日志（主文件）
├── profiles/              # profiler 输出
├── results/               # benchmark 结果（.md 格式）
└── manifest.yaml          # 元信息
```

campaign 命名约定：`<op>_<backend>_<gpu>_<date>`，如 `gemm_fp8_blockwise_triton_gfx942_20260412`。

**Step 3: 写入 manifest.yaml**

```yaml
target_op: <target_op>
target_backend: <target_backend>
target_lang: <target_lang>
target_gpu: <target_gpu>
execution_mode: <repo | workspace>
project_skill: <project_skill_name>
performance_target: "<性能目标描述>"
primary_metric: "<主性能指标，可逗号分隔多个>"
target_shapes: <all | shape list>
kernel_source: <kernel 源文件路径>
test_command: "<focused test 命令>"
benchmark_command: "<focused benchmark 命令>"
quick_command: "<BASELINE 阶段填入>"
representative_shapes: <BASELINE 阶段选出>
git_commit: <true | false>
git_branch: <分支名 | none>
max_iterations: <数字 | null>
max_duration: <"Nh" | null>
created: <YYYY-MM-DD HH:MM>  # 必须精确到分钟，如 "2026-04-13 14:35"，禁止只写日期
```

> **时间精度规则**：campaign 全程所有时间戳（manifest `created`、日志中的"开始时间"/"时间"、结果文件的"时间"）都必须精确到分钟，格式 `YYYY-MM-DD HH:MM`。

**Step 4: 生成 quick 验证脚本**

根据项目 skill 提供的 quick 验证脚本模板，在 campaign 目录下生成 `quick_test_bench.py`。此时 agent 刚从项目 skill 收集完信息，项目 API 上下文最完整，是生成脚本的最佳时机。

- 脚本中 `SHAPES` 列表暂留空（或填入全部 shapes 作为占位）
- BASELINE 完成后再选出代表性 shapes 并更新 `SHAPES` 列表和 manifest 中的 `quick_command` / `representative_shapes`

## 执行环境

优化可在两种模式下进行：

- **`repo-mode`**：直接在上游项目中修改和验证。代码改动、测试、benchmark 都在主仓进行。
- **`workspace-mode`**：先搭建最小开发环境，在其中高频试错，优化达标后再集成回上游项目。

项目 skill 会给出倾向性建议，但最终由 agent 根据任务特点判断。一般规律：
- 改动面小、调参为主、构建快 → `repo-mode`
- 需要大规模试错、从头写新 kernel、主仓构建链路重 → `workspace-mode`

### workspace-mode 最小开发环境

当 agent 选择 `workspace-mode` 时，在 campaign 目录下搭建最小开发环境，至少包含：

```text
agent/workspace/<campaign_name>/
├── src/                   # 从上游项目抽取的最小 kernel 实现
├── tests/                 # 定向 correctness 测试
├── bench/                 # 定向 benchmark
├── logs/
│   └── optimize.md
├── profiles/
├── results/
└── manifest.yaml
```

搭建原则：
- **最小化**：只抽取目标 kernel 及其直接依赖，不搬整个项目
- **可复现**：记录从上游项目哪个 commit 抽取、哪些文件被抽出
- **忠实性**：测试和 benchmark 必须与上游项目的对应环节等价，确保结果可信
- **回灌路径明确**：优化达标后，只把 kernel 核心改动同步回上游项目，不搬脚手架和临时代码

如何从具体项目中抽取代码、构建最小测试和 benchmark，由对应项目 skill 指导。

无论哪种模式，优化产物（日志、profile、benchmark 结果）统一沉淀在 campaign 目录。

## 工作方式

agent 的完整路径：**项目 skill → 本文件（了解需求）→ 项目 skill（收集信息）→ 本文件（执行优化）→ optimize-loop.md → 项目 skill（验收）**。

典型交互序列：

1. agent 从项目 skill 被引导到本文件。
2. 读「前置信息需求」，了解优化框架需要什么。
3. 回到项目 skill，按需求清单收集对应的项目信息。
4. 带着信息回到本文件，执行 DEFINE_TARGET → PREPARE_ENVIRONMENT。
5. 读取 [`workflow/optimize-loop.md`](workflow/optimize-loop.md)，执行 BASELINE → ANALYZE → OPTIMIZE → VALIDATE 循环。
6. 在 ANALYZE / OPTIMIZE 阶段按需读取语言、硬件、profiling、reference 文档。
7. 输出 REPORT，交还项目 skill 做最终验收。

## 知识引用表

根据 `target_lang`、`target_gpu` 和当前阶段按需读取对应 skill。**不要一次全读**。

| 需要什么 | 读哪里 |
|---------|--------|
| 优化过程与门控规则 | [workflow/optimize-loop.md](workflow/optimize-loop.md) |
| Triton 通用优化技巧 | [triton/SKILL.md](triton/SKILL.md) |
| Triton 算子专项策略 | [triton/ops/\<op\>.md](triton/ops/) |
| HIP/CK 通用优化技巧 | [hip/SKILL.md](hip/SKILL.md) |
| HIP 算子专项策略 | [hip/ops/\<op\>.md](hip/ops/) |
| 硬件参数与优化策略 | [../hardware/\<arch\>/SKILL.md](../hardware/) + `optimization-guide.md` |
| Profiling 方法 | [../tool-rocprof/SKILL.md](../tool-rocprof/SKILL.md) |
| 项目代码结构 / 构建 / 测试 / benchmark / 集成 | 对应项目 skill，如 [../primus-turbo-develop/SKILL.md](../primus-turbo-develop/SKILL.md) |
| 参考实现与论文 | [references/](references/) |
| 历史优化案例 | [examples.md](examples.md) |

## 日志与 lineage 总原则

优化过程中需要保留结构化历史（参考 AVO 的 lineage 设计），以支撑长期迭代。

- 当 `git_commit=true` 时，每个 accepted version 对应一个 git commit，构成 lineage
- 当 `git_commit=false` 时，仍在日志中记录 accepted version，但不做 git commit
- 失败尝试记录在 campaign 日志中，但不进入 accepted lineage
- 被接受的版本必须有明确假设、验证结果和接受理由
- 日志既服务于人（随时查看进展），也服务于 agent（回溯历史避免重复尝试）
- **每轮 VALIDATE 结束后必须立即更新 `logs/optimize.md`**，不得延迟或留 placeholder
- 日志和 profiling 结果沉淀在 `agent/workspace/<campaign_name>/`
- 详细日志格式见 [`workflow/optimize-loop.md`](workflow/optimize-loop.md)

## 停滞处理总原则

参考 AVO 的 continuous evolution 机制：**停滞不是停止信号，而是触发策略切换的信号。连续 rollback 意味着当前方向走不通，应该换方向，而不是写报告终止。**

- 连续 2 次 rollback 即触发 stagnation review
- 必须回顾失败尝试、做 profiling（若之前没做）、生成本质不同的新方向
- 切换方向后回到 ANALYZE 继续迭代
- 只有满足终止条件（见 `optimize-loop.md`）才能停止

详细停滞检测和换向规则见 [`workflow/optimize-loop.md`](workflow/optimize-loop.md)。

## 端到端示例

**用户指令**："请优化 Primus-Turbo 里的 blockwise FP8 GEMM Triton 实现，目标 GPU 是 MI300X。"

**Step 1: 了解需求**

agent 从 `primus-turbo-develop/SKILL.md` 被引导到本文件，读「前置信息需求」，得知优化框架需要：kernel 源文件、focused test、focused benchmark、quick 验证脚本模板、benchmark 输出格式、评分规则、execution_mode 建议、rebuild 要求。

**Step 2: 收集项目信息**

agent 回到 `primus-turbo-develop/SKILL.md`，按需求清单收集：
- Kernel: `primus_turbo/triton/gemm/gemm_fp8_kernel.py`
- Focused test: `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"`
- Focused benchmark: `PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise`
- Quick 验证脚本模板: PREPARE_ENVIRONMENT 时生成 `quick_test_bench.py`（shapes 暂留空，BASELINE 后填入）
- 评分: `Forward TFLOPS` 几何平均，`Check` 为正确性门控
- 环境建议: Triton → `repo-mode`，无需 rebuild

**Step 3: DEFINE_TARGET**

整理参数：`target_op=gemm_fp8_blockwise`, `target_backend=TRITON`, `target_gpu=gfx942`, `execution_mode=repo`, `performance_target=相对 baseline 有可测量提升`。对照前置信息需求 checklist 确认齐全。

向用户确认：

> 确认优化目标：
> - 算子: blockwise FP8 GEMM, backend: TRITON, GPU: MI300X (gfx942)
> - 主性能指标: Forward TFLOPS（是否也需要优化 Backward？）
> - 性能目标: 相对 baseline 有可测量提升
> - 执行模式: repo-mode, git_commit=true, git_branch=auto
> - 请确认或调整。

用户确认后进入 PREPARE_ENVIRONMENT。

**Step 4: PREPARE_ENVIRONMENT**

1. 创建优化分支：`git checkout -b optimize/gemm_fp8_blockwise_triton_gfx942_20260412`
2. 创建 `agent/workspace/gemm_fp8_blockwise_triton_gfx942_20260412/`
3. 创建子目录 `logs/`, `profiles/`, `results/`
4. 写入 `manifest.yaml`
5. 根据项目 skill 的 quick 验证脚本模板，生成 `quick_test_bench.py`（shapes 暂留空，BASELINE 后填入）

**Step 5: 进入优化循环**

读 `workflow/optimize-loop.md`，按其定义执行：
1. BASELINE：执行 focused test（确认 PASS）→ 执行 focused benchmark → 记录 baseline TFLOPS
2. ANALYZE：读 kernel 源码、查 `triton/SKILL.md` 和 `hardware/gfx942/`，生成优化假设
3. OPTIMIZE → VALIDATE 循环：改 kernel → 跑 test → 跑 benchmark → 对比 → accept 或 rollback
4. 达到目标或穷尽方向后，输出 REPORT

**Step 6: 验收**（回到项目 skill）

项目 skill 跑完整测试、审查报告、确认 commit。

## 相关 skill

| Skill | 说明 |
|-------|------|
| `workflow/optimize-loop` | 详细优化过程、门控、回滚、停滞处理 |
| `triton` | Triton 通用优化技巧 |
| `hip` | HIP/CK 通用优化技巧 |
| `hardware/gfx942` | MI300X/MI325X 硬件参数与优化策略 |
| `hardware/gfx950` | MI350X/MI355X 硬件参数与优化策略 |
| `tool-rocprof` | rocprof profiling 工具用法 |
| `primus-turbo-develop` | Primus-Turbo 项目的代码结构、构建、测试、benchmark 与集成方式 |
