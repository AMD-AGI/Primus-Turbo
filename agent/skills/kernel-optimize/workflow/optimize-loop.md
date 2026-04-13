# 优化过程执行规范

本文件定义 `kernel-optimize` 的**详细优化过程**。

默认前提是（agent 依次经过：项目 skill → `../SKILL.md`（了解需求）→ 项目 skill（收集信息）→ `../SKILL.md`（执行）→ 本文件）：
- 已按 `../SKILL.md` 的「前置信息需求」从项目 skill 收齐了所需信息
- 已在 `../SKILL.md` 完成 DEFINE_TARGET（参数结构化）和 PREPARE_ENVIRONMENT（campaign 目录建立）
- 已建立本轮 campaign 目录，`manifest.yaml` 已填写
- 已明确 `target_op`、`target_backend`、`target_lang`、`target_gpu`、`execution_mode`

因此，本文件**不负责**：
- 解释如何从用户指令提取参数（见 [`../SKILL.md`](../SKILL.md) 的 DEFINE_TARGET）
- 规定 campaign 目录结构（见 [`../SKILL.md`](../SKILL.md) 的 PREPARE_ENVIRONMENT）
- 提供某个具体项目的构建、测试、benchmark 命令（由项目 skill 提供）

## 输入参数

开始执行优化过程前，至少明确：

| 参数 | 含义 |
|------|------|
| `target_op` | 目标算子 |
| `target_backend` | 目标 backend |
| `target_lang` | 实现语言 |
| `target_gpu` | 目标 GPU 架构 |
| `execution_mode` | `repo` / `workspace` |
| `campaign_dir` | 本轮 campaign 目录 |
| `target_shapes` | 本轮关注的 shape 集 |
| `performance_target` | 目标性能 |
| `primary_metric` | 主比较指标，可多个；取决于算子类型（如 GEMM: `Forward TFLOPS`，elementwise: `Forward GB/s`） |
| `git_commit` | 是否对 accepted version 做 git commit |
| `git_branch` | 当前优化分支（或 `none`） |
| `max_iterations` | 最大迭代轮数（可选） |
| `project_skill` | 对应项目 skill |

## 确认已获取的信息

以下信息应在项目 skill 阶段和 DEFINE_TARGET 阶段已获取，开始优化循环前做最终确认：

| 信息 | 用途 | 示例 |
|------|------|------|
| **kernel 源文件路径** | ANALYZE 阶段读代码 | `primus_turbo/triton/gemm/gemm_fp8_kernel.py` |
| **focused test 命令** | full correctness 验证 | `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON"` |
| **focused benchmark 命令** | full performance 评估 | `PRIMUS_TURBO_GEMM_BACKEND=TRITON python benchmark/ops/bench_gemm_turbo.py --dtype fp8 --granularity blockwise` |
| **quick test 命令** | 每轮快速 correctness 验证 | `pytest tests/pytorch/ops/test_gemm_fp8.py -v -k "blockwise and TRITON" --maxfail=3` |
| **quick benchmark 命令** | 每轮快速 performance 评估 | 跑 full benchmark 后提取 `representative_shapes` 的数据；或使用脚本的 shape 过滤参数（如支持） |
| **benchmark 输出列名** | 解析结果、计算分数 | `Forward TFLOPS`、`Backward TFLOPS`（primary_metric）、`Check`（correctness gate） |
| **rebuild 要求** | 改完代码后是否需要重新构建 | Triton: 无需 rebuild；HIP: 需增量 rebuild |

若有缺失，回查项目 skill 或 `../SKILL.md` 的 manifest 补齐。

## 两级验证

优化循环中有两级验证：

| 级别 | 何时使用 | 内容 |
|------|---------|------|
| **quick** | 每轮 VALIDATE 默认使用 | 代表性 shape subset（3-5 个）的 test + benchmark，快速反馈 |
| **full** | BASELINE、最终验收、或 agent 判断需要时 | 全量 shape 的 test + benchmark |

quick 和 full 的具体命令由项目 skill 提供，记录在 manifest 中。

quick validation 的 aggregate score 可用于快速接受/拒绝决策。当 quick 结果接近临界（提升 < 5%）或涉及 loop 结构等高风险改动时，agent 应主动升级到 full validation 确认。

## 总流程

```text
ENVIRONMENT_BASELINE
  -> [ANALYZE -> OPTIMIZE -> VALIDATE -> ACCEPT/ROLLBACK] (迭代循环)
  -> REPORT
```

在 `repo-mode` 下，VALIDATE 即直接在主仓验证，无 SYNC_BACK 步骤。
在 `workspace-mode` 下，VALIDATE 分 local gate 和 integration gate，中间有 SYNC_BACK。

## 评分操作规范

### 从 benchmark 输出到 aggregate score

**Step 1**: 执行 benchmark（quick 或 full），输出结果

**Step 2**: 从输出中提取每一行的 `primary_metric`（如 `Forward TFLOPS` 或 `Forward GB/s`，取决于算子类型；若多指标则分别提取）和 `Check` 列

**Step 3**: correctness 门控
- 任一行 `Check = FAIL` → 该候选 `aggregate score = 0`，直接拒绝

**Step 4**: 计算 aggregate score
- 若 `target_shapes` 只有 1 个配置：`aggregate score = 该配置的 primary_metric`
- 若 `target_shapes` 为多个配置：`aggregate score = geometric_mean(所有配置的 primary_metric)`

```
geometric_mean = (x1 * x2 * ... * xn) ^ (1/n)
```

**Step 5**: 保留 score vector（每个 shape 的原始值），避免总分掩盖局部退化

**多指标处理**：当 `primary_metric` 包含多个指标（如 `Forward TFLOPS, Backward TFLOPS`）时，分别计算每个指标的 aggregate score。接受条件：所有指标的 aggregate score 均不退步，且至少一个指标有提升。

### 噪声判断

- 提升幅度 < 2% 时，视为接近噪声区间
- 此时必须复测至少 3 次，取均值和标准差
- 若均值提升 > 1% 且标准差 < 提升幅度的一半，视为有效提升
- 否则视为噪声，不接受

### 接受规则

- `aggregate score` 不得低于当前最佳
- 若只是持平，则必须有明确附加价值（如更广适用范围、更稳定结果）
- 若某个核心 shape 退步 > 3%，默认拒绝；除非本轮明确是定向 shape 优化

### 计算示例

Benchmark 结果（3 个 shape）:

| M | N | K | Forward TFLOPS | Check |
|---|---|---|---------------|-------|
| 4096 | 4096 | 4096 | 320 | PASS |
| 2048 | 8192 | 4096 | 305 | PASS |
| 1024 | 4096 | 8192 | 290 | PASS |

```
aggregate score = (320 * 305 * 290) ^ (1/3) = 304.8
```

Baseline aggregate score = 285.0 → 提升 = (304.8 - 285.0) / 285.0 = +6.9% → 接受。

## 阶段说明

### 1. ENVIRONMENT_BASELINE

**目标**：冻结起点，建立统一比较标尺。

**"focused" 的含义**：
- focused test = 限定 `target_op` + `target_backend` 的测试子集（如 `-k "blockwise and TRITON"`）
- focused benchmark = 限定 `target_backend` 的 benchmark，覆盖所有 `target_shapes`

**操作步骤**：
1. 确认当前环境可正常构建和运行
2. 执行 **full** focused test，确认全部 PASS
3. 执行 **full** focused benchmark，结果写入 `<campaign_dir>/results/baseline.md`
4. 按评分规范计算 baseline `score vector` 和 `aggregate score`
5. 记录 backend 配置和关键环境状态

BASELINE 始终使用 full validation，确保起点数据完整可靠。

**baseline 记录模板**（写入 `<campaign_dir>/logs/optimize.md`）：

```markdown
## Baseline
- 时间: <timestamp>
- Backend: <target_backend>
- GPU: <target_gpu>
- Commit: <git_hash>
- 验证级别: full

- Aggregate score (geomean): 278.0
- 所有 Check: PASS
- 详细数据: results/baseline.md
```

**输出物**：
- `results/baseline.md`（详细数据） + baseline aggregate score
- 当前 best = baseline

### 2. ANALYZE

**目标**：找到下一轮最值得尝试的方向，而不是盲目调参。

**必做项**：
- 阅读当前 best 版本的核心实现
- 回看最近几轮 accepted version 与失败尝试（从 campaign 日志获取）
- 按需做 profiling 或分析 benchmark 指标，定位当前主要瓶颈
- 按需读取语言/硬件/profiling skill

**瓶颈分类与优化方向映射**：

当有 profiler 数据时，按以下框架判断瓶颈类型：

| 瓶颈信号 | 分类 | 优化方向 |
|---------|------|---------|
| ALU utilization 低、MFMA 指令占比低 | Compute bound | tile size 调整、指令选择（如 MFMA vs WMMA）、算法简化、减少冗余计算 |
| Memory throughput 接近硬件峰值、大量 global load/store stall | Memory bound | 数据布局优化、prefetch / software pipelining、减少冗余访存、LDS 利用 |
| Occupancy 低、register 或 LDS 用量过高 | Resource bound | 减少 register 压力、调整 LDS 分配、降低 tile size 换取更多 wave |
| Kernel launch 开销占比高、总计算量小 | Launch/overhead bound | Persistent kernel、batch 多个小 kernel、减少 dispatch 次数 |

当没有 profiler 时，可从 benchmark 结果间接推断：
- 如果 TFLOPS 远低于理论峰值且增大 shape 后效率显著提升 → 可能是 launch overhead 或 occupancy 问题
- 如果 TFLOPS 随 K 增大而显著提升 → 可能是 memory bound（更高的计算访存比改善效率）
- 如果不同 shape 之间效率差异小且整体偏低 → 可能是 compute bound

**每个候选方向至少要回答**：
- 当前瓶颈是什么
- 这轮准备改什么
- 预计收益是什么
- 风险是什么
- 用什么信号来验证成败

**输出物**：
- 有优先级的假设列表
- 本轮首选假设

### 3. OPTIMIZE

**目标**：实施一次可归因、可回滚的小步改动。

**必做项**：
- 每轮只推进一个主假设
- 修改后能够明确回答"这轮到底改了什么"
- 若涉及编译产物，按项目 skill 说明 rebuild
- 记录本轮修改文件、关键参数、预期影响

**约束**：
- 不要把无关清理混入优化尝试
- 不要同时引入多个正交的大改动
- 不要破坏与上游项目约定的关键接口语义

**输出物**：
- candidate diff
- build 状态

### 4. VALIDATE

**目标**：决定候选是否通过验证。

**`repo-mode` 下**（常用路径，无 SYNC_BACK）：

1. 如有需要，确保 backend 设置正确（env var 或 reset）
2. 执行 **quick** test
3. correctness 全部 PASS 后，执行 **quick** benchmark
4. 按评分规范计算 `score vector` 和 `aggregate score`
5. 与当前 best 对比
6. 当结果接近临界（提升 < 5%）或涉及高风险改动时，升级到 **full** validation 确认

结果写入 `<campaign_dir>/results/v<N>.md`（标注验证级别）。

**硬门控**：
- build 失败 → 直接拒绝
- correctness 失败 → 直接拒绝
- benchmark 中 `Check = FAIL` → 直接拒绝
- aggregate score 退步 → 默认拒绝
- 核心 shape 退步 > 3% → 默认拒绝

**通过后**：
- 更新当前 best
- 若 `git_commit=true`：git commit（见下方 git 集成规范）
- 将本轮结果写入 campaign 日志

**`workspace-mode` 下**：
验证分 local gate（最小环境内）和 integration gate（回灌主仓后）。
只有通过 integration gate 才算真正的 accepted version。
SYNC_BACK 步骤：只同步被接受的核心改动，不搬脚手架和临时代码。

### 5. ACCEPT / REPORT

**目标**：更新 lineage，并为下一轮留下可复用上下文。

**ACCEPT 必做项**：
- 更新 campaign 日志中的 accepted history
- 记录相对 baseline 的累计提升
- 标记哪些方向有效、无效、待复查
- 产出下一轮候选方向

**REPORT**（campaign 终止时输出，写入 `logs/optimize.md` 末尾的 `## Final Report` section）：
- baseline 与最终 best 对比（含 full validation 数据）
- 总累计提升
- 关键有效优化列表
- 已验证无效方向列表
- 若继续优化，优先建议的下三步
- 详细数据指向 `results/` 下的对应 `.md` 文件

## 回滚规则

以下情况必须回滚本轮 candidate：
- build 失败
- correctness 失败
- benchmark 中 `Check` 失败
- 相比当前 best 明显退步
- 结果波动过大，暂时无法确认提升是否真实

回滚操作：
- `repo-mode`: `git checkout -- <modified_files>` 或 `git revert <commit>`
- `workspace-mode`: 回滚本轮 local 改动，不影响主仓
- 回滚原因必须写入日志，避免重复踩坑

## Git 集成规范

当 `git_commit=true` 时，每个 accepted version 对应一个 git commit，构成 lineage（参考 AVO 的设计）。当 `git_commit=false` 时，跳过 commit，但仍在日志中记录 accepted version。

**commit 时机**：VALIDATE 通过后，检查 `git_commit` 开关，若为 `true` 则立即 commit。

**commit message 格式**：

```
[optimize] <target_op> <target_backend> v<N>: <一句话摘要>

假设: <本轮优化假设>
结果: <aggregate score 变化>
详见: <campaign_dir>/logs/optimize.md
```

示例：

```
[optimize] gemm_fp8_blockwise TRITON v3: increase num_stages from 2 to 3

假设: 增加 software pipelining 深度，隐藏访存延迟
结果: geomean 301 -> 319 TFLOPS (+6.0%)
详见: agent/workspace/gemm_fp8_blockwise_triton_gfx942_20260412/logs/optimize.md
```

**回滚**：`git revert <commit>` 回滚单个版本。

**注意**：campaign 目录（`agent/workspace/`）默认不纳入 git，只有 kernel 代码的改动进入 git lineage。results/ 下的 `.md` 文件仅沉淀在 campaign 目录中。

## 优化日志模板

每个 campaign 维护一个 `logs/optimize.md`，实时更新，人随时可查看进展。

```markdown
# <target_op> <target_backend> 优化日志

## 基本信息
- 目标算子: <target_op>
- 实现语言: <target_lang>
- Backend: <target_backend>
- 目标 GPU: <target_gpu>
- Campaign: <campaign_dir>
- 开始时间: <timestamp>
- 当前状态: 优化中 (v<N>)

## Baseline
| Shape (MxNxK) | Forward TFLOPS | Check |
|---------------|---------------|-------|
| ... | ... | ... |
- Aggregate score: <baseline_score>

## 优化历史

### v1 — <一句话描述改动>
- 时间: <timestamp>
- 验证级别: quick / full
- 假设: <为什么做这个改动>
- 改动: <改了哪些文件、哪些参数>
- 结果: <aggregate score 变化> ✅/❌
- 单测: PASS/FAIL
- 决策: accept / rollback
- 详细数据: results/v1.md
- 备注: <失败原因或关键观察>

### v2 — ...

## 当前最佳
| Shape (MxNxK) | Baseline | 当前最佳 | 提升 |
|---------------|----------|---------|------|
| ... | ... | ... | ... |
- Aggregate score: baseline <X> → 当前 <Y> (+Z%)

## 待尝试方向
- [ ] <方向 1>
- [ ] <方向 2>
- [x] ~~<已验证无效的方向>~~ (vN 已验证)

## 已验证无效方向
| 方向 | 版本 | 失败原因 |
|------|------|---------|
| ... | v<N> | ... |

## Final Report
（campaign 终止时填写）
- Baseline aggregate score: <X>
- Final best aggregate score: <Y> (+Z%)
- 总迭代轮数: <N>（accepted: <A>, rollback: <B>）
- 关键有效优化: ...
- 已验证无效方向: ...
- 若继续优化，建议下三步: ...
```

## 结果文件模板

每轮 VALIDATE 的详细数据写入 `results/v<N>.md`（BASELINE 写入 `results/baseline.md`）。

```markdown
# v<N> — <一句话描述改动>

- 时间: <timestamp>
- 验证级别: quick / full
- 假设: <本轮优化假设>

## Correctness
- 命令: <test 命令>
- 结果: <X passed, Y failed>

## Benchmark
- 命令: <benchmark 命令>

| M | N | K | Forward TFLOPS | Backward TFLOPS | Check |
|---|---|---|---------------|----------------|-------|
| ... | ... | ... | ... | ... | ... |

## Score
- Forward aggregate (geomean): <score>
- Backward aggregate (geomean): <score>（若适用）
- vs baseline: +X%
- vs current best: +Y%
```

## 停滞检测与条件性干预

借鉴 AVO 的 continuous evolution 思路，停滞不是停止信号，而是触发干预的信号。

满足任一条件即可触发：
- 连续 `N` 次候选都未被接受，默认 `N = 5`
- 连续多轮都在同一类方向微调，但没有可测得的提升
- profiler 指向的瓶颈长期未变化
- 最近几轮只有参数抖动，没有结构性新假设

一旦触发，必须执行一次 `stagnation review`：

1. 回顾最近若干 accepted version 的收益曲线
2. 回顾失败尝试，找出已证明无效的方向
3. 重新查看 profiler、参考实现和硬件文档
4. 生成至少 3 个本质不同的新方向
5. 优先尝试近期没有覆盖过的方向

推荐换向类别：
- tile / launch 参数
- memory layout / 数据搬运
- software pipelining / overlap
- occupancy / register / LDS 资源分配
- backend 切换或参考实现对比
- 算法级重排、分支消除、kernel fusion

## 终止条件

满足任一条件即可终止本次优化 campaign：
- 达到 `performance_target`
- 达到可接受的硬件效率区间
- 最近若干 accepted version 的增益已低于噪声水平
- 假设池已基本穷尽
- 剩余方向风险过高、收益过低
- 达到 `max_iterations` 上限（若已设置）
- 用户要求停止，或时间 / 算力预算耗尽

终止时必须输出 REPORT（见 ACCEPT / REPORT 阶段）。

## 跨 skill 查阅路径

| 阶段 | 需要什么 | 去哪里读 |
|------|----------|----------|
| 全流程起点 | 项目信息收集 → DEFINE_TARGET → PREPARE_ENVIRONMENT | 项目 skill → [`../SKILL.md`](../SKILL.md) |
| ENVIRONMENT_BASELINE / VALIDATE | 项目测试命令、benchmark 命令 | 对应项目 skill |
| ANALYZE | profiling 方法 | [`../../tool-rocprof/SKILL.md`](../../tool-rocprof/SKILL.md) |
| ANALYZE | 架构约束 | `../../hardware/<arch>/SKILL.md` + `optimization-guide.md` |
| ANALYZE / OPTIMIZE | 语言级优化技巧 | [`../triton/SKILL.md`](../triton/SKILL.md) 或 [`../hip/SKILL.md`](../hip/SKILL.md) |
| ANALYZE / OPTIMIZE | 算子专项策略 | `../triton/ops/*.md` 或 `../hip/ops/*.md` |
| ANALYZE | 参考实现与论文 | `../references/*.md` |

## 执行提醒

- 先过 correctness，再谈性能。
- 每轮只推进一个主假设。
- 当 `git_commit=true` 时，accepted version 必须 git commit，构成可追溯的 lineage。
- 停滞时必须换向，不能无限做同方向微调。
- 日志实时更新，人随时可查看当前进展和历史决策。
