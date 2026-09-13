# 用 op-evolve 跑自动优化 + 交叉验证

`op-evolve` 在 `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`。
它的核心设计是 **「成功是退出码，不是意见」**：`op/validation.py` 判定目标是否达成，框架
自己通过 runner 执行它并读退出码，**agent 不给自己的工作打分**。每一轮是否被保留，是对
同一会话里前后脚测出的数字做算术，不是由 agent 声明。

---

## 1. 先搞清楚你不需要写什么

**`op/` 目录不是手写的。** 框架有一个 **Op Setup** 阶段，由 agent 从 job spec 生成：

```
op/baseline/     种子实现（round 0）
op/current/      rounds/<best_round>/op/ 的副本，act 阶段改的就是它
op/beat/         对照实现（可选）
op/eager/        参考实现
op/ut/           单元测试
op/benchmark.py  计时
op/validation.py 判定，退出码即结论
```

生成完之后**人审批**，不满意就让它重做。你要写的只有 **job spec 一个 yaml**。

---

## 2. 落地缺口（现在还差的）

| 项 | 状态 | 怎么补 |
|---|---|---|
| op-evolve 安装 | ⚠ `pip install -e .` 被 PEP 668 挡住 | 从 checkout 目录跑，或 `pip install -e . --break-system-packages`，或建 venv |
| `~/.op_evolve_anthropic` | ✅ 存在 | — |
| `~/.op_evolve_openai` | ❌ **不存在** | 见下 |
| job spec | ✅ 已写 `phase0/gfx1250-attn-llama31-8b-bwd.yaml` | 校验通过 |
| GPU | ❌ **wedged** | 重启 + `sudo modprobe amdgpu` |

**openai 凭据缺失是唯一的硬缺口。** job spec 里 `reviewer` 角色配的是 `sdk: codex`，
这正是「交叉验证」的关键——**用另一个模型来复核 planner 的方案**。两个选择：

- **保留交叉验证**（推荐）：创建 `~/.op_evolve_openai`，格式和 anthropic 那个一样，
  是会被 source 的 shell 文件：
  ```bash
  cat > ~/.op_evolve_openai <<'ENV'
  export OPENAI_BASE_URL="https://your-gateway/openai"
  export OPENAI_API_KEY="..."
  ENV
  ```
- **放弃交叉验证**：把 spec 里 `reviewer: {sdk: codex, model: gpt-5.6-sol}` 改成
  `{sdk: claude, model: claude-opus-5}`。**但这就失去了「两个不同模型互相争论」的价值**，
  而那恰恰是 deep round 的核心。

---

## 3. 跑起来

```bash
cd /home/lihuzhan/code/2026_0910__op-evolve/op-evolve

# 检查账号和机器（文档说每个失败都会在第 7 轮的某个工具里才暴露，所以先查）
python3 -m op_evolve.cli run \
  --config /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0913__opt_plan__claude/phase0/gfx1250-attn-llama31-8b-bwd.yaml
```

其它子命令：

```bash
python3 -m op_evolve.cli status --job <job_name>-<job_id>   # 看账本
python3 -m op_evolve.cli stop   --job <job_name>-<job_id>   # 在下个模块边界停
python3 -m op_evolve.cli resume --job <job_name>-<job_id>   # 续跑
python3 -m op_evolve.cli tune   --beat-margin <N>           # 改相对门槛
```

状态全在 `artifacts/<job>/` 下的文件里——**这是它可停、可续、事后可读的原因**。

---

## 4. 轮次调度（我们这个 job 该怎么配）

| 设置 | 含义 |
|---|---|
| 都不设 | 每轮都是 fast |
| `fast_rounds: N` | 前 N 轮 fast，之后全 deep |
| `fast_per_deep: K` | K 轮 fast 配 1 轮 deep，循环 |
| 两者都设 | N 轮 fast，然后 K:1 循环 |

**fast round** ≈ 半小时，1 个 agent：看、选一个改动、实现、测量。
**deep round** ≈ 2.4 小时：6 项 profiling 分析 + **2 个不同模型的 agent 争论方案**。

我们 spec 里设的是 `fast_rounds: 12, fast_per_deep: 5`（约 34 fast / 6 deep）。
**理由是这张卡的 profiling 面几乎是瞎的**：51 个硬件计数器、0 个 derived metric、
**没有任何访存或矩阵计数器**，deep round 的 6 项分析里 2 项可用、2 项严重退化、2 项不可能。
而 fast round 是几秒的设备时间，且已知的收益形态就是「改两个整数」，适合 A/B。

**如果明天换到 4 卡健康机器，这个判断要重估**——profiling 面可能不同，deep round 的
性价比会回升。

---

## 5. 必须写进 spec 的几条（否则会静默失效）

这些是我读 `loop.py` 和踩坑得到的，不是推测：

1. **`target.tflops` 和 `target.roofline_pct` 必须留 `null`**。任一被设置，
   `loop.py::_margin()` 直接返回 None，**相对门槛被静默关掉**。
2. `beat:` 字符串里必须**字面出现** `beaten by N%`，正则是
   `beaten by\s+(\d+(\.\d+)?)\s*%`。
3. **anchor 必须同轮实测**，不能用存档值——这张卡的时钟状态正是存疑的东西。
4. **`precision_sqnr_db` 的检查必须分别覆盖 out/dq/dk/dv**。我们实测抓到过
   `out 53.67 ✓ / dq 52.24 ✓ / dk −inf ✗ / dv −inf ✗` 的配置——只看输出会直接放过。

---

## 6. 我们已经有的东西可以直接喂给它

不必从零开始，Op Setup 可以直接用：

- **种子**：`primus_turbo/triton/attention/fused_mha_bwd_kernel.py`（已 vendor 在树内，
  当前冠军 316 TFLOP/s），而不是 spec 里原先写的 aiter 出厂配置。
- **测量台**：`tools/gfx1250/tune_attention.py` —— 四张量 SQNR 门、fp32 分块参考、
  逐位确定性模式、launch 级配置验证，全都现成。
- **对照**：`torch flex`，31.337 ms（同会话实测可复现，和 Phase 1 的 31.508 差 0.5%）。
- **禁区清单**：`PROGRESS.md` 的「不要重做」6 条，直接进 spec 的 description，
  省掉 agent 重新推导并重新踩一遍。

---

## 7. 最大的风险

**op-evolve 没有 GPU 锁，也没有成本核算。** 文档写明：被别人抢走的 GPU 不会报错，
它只会记录一个偏低的数字，**而那个数字会成为下一轮必须打败的冠军**。4 卡机上尤其要注意
——必须用 `HIP_VISIBLE_DEVICES` 做硬隔离，spec 的 `runtime.gpu_id` / `gpu_pool` 就是干这个的。

成本方面：`cost_usd` 从 SDK 读出来但既不累加也不持久化，**`--rounds` 是唯一的花费上限**，
每个 optimize round 约 $13-20。
