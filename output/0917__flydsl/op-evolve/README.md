# Phase C：op-evolve 作业规格（已写好，**尚未运行**）

两个文件的原件放在同事的仓里 `~/code/2026_0910__op-evolve/op-evolve/jobs/`
（那是另一个 git 仓，我没有在那边提交）。这里是版本化的副本。

## 为什么现在这个作业值得跑

时序结果反而让它成了理想对象：基线**慢 93×**，搜索空间完全未开采。
这和上一次那个作业正相反 —— `gfx1250-attn-llama31-8b-bwd`（0914）只接受了 1 轮就停，
因为它的 round 1 把 launch-config 那根轴挖干了，后面三轮在同一个空间里重搜。

## 针对上次失败的三处防护

| 上次的死因 | 这次 |
|---|---|
| round 2 的两条有用发现没写进 `facts.md`/`dead_ends.md` → `FastError`，轮次作废 | 两条发现**已预先写进 `HINTS.md`**，setup 要据此预填 `facts.md`；HINTS 里明写「每轮都要 append，门失败也要 append」 |
| round 4 `validation.py` exit **137**（OOM 被当成回归记分） | HINTS 要求参考实现分块、子进程、形状间 `empty_cache()`，并保留退出码语义（137/124/143 = 基础设施故障，重试并标注，不记为候选失败） |
| `stop` 之后再没人 resume | 用 `tools/supervise_job.sh` 挂起；它只在真正的 `finished` 事件才停，且拒绝重启进死卡 |

## 单卡带来的两处改动

- `runtime.gpu_pool: []` —— **不是占位符**。deep round 的并行 profiling 用不了，会显著变慢。
- 所以 `fast_rounds: 12` / `fast_per_deep: 5`，把预算前压到 fast round。

## 跑之前需要你做的两件事

1. **补 `~/.op_evolve_openai`**：`ln -s ~/.op_evolve_anthropic ~/.op_evolve_openai`
   （OpenAI 的两个变量本来就在那个文件里），或把 `agent.roles.reviewer` 设成 `null` 走单 SDK。
   倾向前者 —— deep round 的质量来自 planner(Claude) 与 reviewer(Codex) 互相辩论。
2. **授权长时间独占卡**，并在此期间停掉一切交互式 GPU 工作。

然后：

```bash
cd ~/code/2026_0910__op-evolve/op-evolve
.venv/bin/python tools/check_llm_account.py --config jobs/gfx1250-flydsl-attn-bwd.yaml
.venv/bin/python tools/check_runner.py      --config jobs/gfx1250-flydsl-attn-bwd.yaml
setsid nohup tools/supervise_job.sh jobs/gfx1250-flydsl-attn-bwd.yaml >/dev/null 2>&1 &
```

**成本**：deep round 约 2.4 h，optimize round 约 $13–20，`max_rounds` 是唯一的花费上限
（框架没有成本记账）。**安全提示**：它派出的 agent 是 `bypassPermissions`，不会再请求授权。

## 一个坦率的判断

作业说明里写明了：**这件事的真实问题不是「FlyDSL 能不能超过 ASM」，而是「能不能追平」。**
前向那个 1.51×（两边都成熟）是这套惯用法竞争力的最好估计，持平就是反向的现实上限。
所以作业的 `beat` margin 设成 **0%** 而不是惯例的 20% —— 追平就算赢。
一个测得干净的否定结论同样是结果，而且十轮拿到比四十轮拿到好。
