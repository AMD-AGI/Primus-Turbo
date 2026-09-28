# B0 攻关（2026-09-27/28）交接清单：已推到 GitHub 的内容 + 本地备份包

机器：B0 = `ctheliosp-1b112-a37-1`（4× gfx1250，GPU1 目前 wedged）。工作于 2026-09-28 12:45 UTC 结束，机器交还。
这台机器的磁盘随时可能被清空。**本目录下的 tar.gz 也在同一块盘上**，要在清盘前拷到别处（A0 或个人存储），见 §3。

## 1. 已推到 GitHub 的内容

### 1.1 Primus-Turbo（主要成果）
| 项 | 值 |
|---|---|
| 仓库 | `git@github.com:AMD-AGI/Primus-Turbo.git` |
| 分支 | **`dev/lhz/flydsl-attn-b0`**（B0 专用，已 push） |
| 基点 | `d7ee84d5`（A0 分支 `dev/lhz/flydsl-attn` 上最后一个 commit，"fwd job paused at round 13"） |
| commit 范围 | **`d7ee84d5..35464852`**，共 32 个 commit；第一个 `90508b9b`（把 09-27 的 B0 工作压成一个 squash commit），最后一个 `35464852`（收工） |
| 改动 | 1792 个文件，全部在 `output/0927__b0/` 下；另有 `.claude/skills/gfx1250-attn-campaign/SKILL.md` 的 §3 入口 |
| 与 A0 分支的关系 | 在 `dev/lhz/flydsl-attn` 之上 fast-forward，可以直接 merge |

分支里的关键内容（都在 `output/0927__b0/` 下）：
- `HANDOFF-A0.md`：A0 无缝衔接指南（英文，给 agent 看），最终状态在第 8、9 节
- `REPORT-0928.html`：两天的中文总结报告
- `champions/`：两个冠军的代码树逐字节快照，带 MD5SUMS
  - fwd r16：r13ns
  - bwd r29：r19h + u2n
- `patches/`：对 op-evolve job 目录的全部手工改动（benchmark 分块计时、sclk、validation 判据、gates 49 dB、final.yaml 等）
- `fwd-hint.md`、`bwd-hint.md`：两个 job 的 hint.md 最终拷贝（fwd h28–h50，bwd h57–h73）
- `fwd/`、`bwd/`：每轮的 progress 表和各轮 opt/act 文档
- 各实验的报告：
  - `ruler/`、`ruler/bwd/`：计时偏差审计
  - `gemm/`：nkfix 相关代码和报告
  - `e2e/`：启动脚本、attn 后端、结果（`RESULT.md`、`RESULT-final.md`）
  - `profile/`、`fwd-nospec/`
  - `lab*/`、`verify_r6/`
  - `OP-EVOLVE-SUGGESTIONS.md`
- `LAB-RULES.md`、`interference.md`、`mon/watch.sh`、`iso/`、`probe/`：多卡隔离与监控

### 1.2 没有推送的 git 内容（原因见说明）
| 仓库/分支 | 状态 | 处理 |
|---|---|---|
| Primus-Turbo 本地分支 `b0/0927`（09-27 的原始历史，27 个 commit） | **有意不推**：历史里混有约 40 MB 的 lab bulk（ISA dump、代码树拷贝）；内容已经以 squash 形式包含在 `90508b9b` 里 | 打成 bundle `Primus-Turbo_b0-0927.bundle` 放进备份 |
| op-evolve 本地分支 `lhz/gfx1250`（HEAD `4e0ad31`，含 A0 时期的 route/acceptance/deep_loop 修改） | origin（wenxie-amd/op-evolve）上没有这个分支；**B0 期间没有新 commit** | 打成 bundle `op-evolve_lhz-gfx1250.bundle` 放进备份 |
| op-evolve 的 job 目录（`artifacts/…`）和 LOG | 不在任何 git 仓库里（体积大，含 refcache） | 放进备份 `b0_opevolve_jobs.tar.gz` |

没有改动过的仓库：Primus（main，只有 A0 时期留下的未跟踪 e2e 配置，已放进 misc 备份）、aiter-src、wt-bakeoff、lhz-devsync。

## 2. 本地备份包（`output/0928__bak_b0/`，不进 git）
由 `make_bak.sh` 生成，校验和见 `SHA256SUMS`。

| 文件 | 内容 | 用途 |
|---|---|---|
| `b0_opevolve_jobs.tar.gz` | 两个 op-evolve job 的完整目录：`gfx1250-flydsl-attn-fwd-b0-20260927`（1.4 GB）、`gfx1250-flydsl-attn-bwd-20260917-115934`（3.2 GB，含 refcache、各轮 rounds、state.yaml、改过的 harness、hint.md），以及 `LOG.fwd`、`LOG.fwd-b0`、`LOG.bwd`、`jobs/gfx1250-flydsl-attn-fwd.yaml` | 在 A0 上 `op-evolve resume` 两个 job，保留历史、冠军记录和池子 |
| `b0_output_0927b0_full.tar.gz` | `output/0927__b0` 全量，包括被 gitignore 的 bulk：lab/arm 代码树、ISA dump、e2e trace、各次运行的原始日志。refcache 的重复拷贝已排除 | 重新分析或复现 lab 结论、e2e trace |
| `b0_real_qkv_dumps.tar.gz` | `/home/lihuzhan/_prof_dump`（2.3 GB）：从 e2e 训练里 dump 出的 6 层真实 q/k/v，[B,S,H,D] bf16 | fwd 第二把尺子（h45/h47）必需；解包到 `/home/lihuzhan/_prof_dump` |
| `b0_claude_sessions.tar.gz` | Claude Code session：本项目目录下全部 session jsonl、subagent 与 workflow 记录、memory；09-27 的 session `1215a290`；本次的 plan 文件 | 追溯决策过程；memory 可以合并到 A0 |
| `b0_misc.tar.gz` | Primus-Turbo 里 `0927__b0` 之外的未跟踪文件（09-27 早上的 `output/0925__flydsl`、`0927__flydsl` 残留）、Primus 的 e2e 配置（`examples/torchtitan/configs/MI455X`、`benchmark/kernel/attention`）、e2e rank 日志 `_dbg_l8b`、09-27 的 A0/B0 性能对比数据（`2026_0927__bak/perf_b0`、`PERF-A0-vs-B0.md`） | 杂项 |
| `op-evolve_lhz-gfx1250.bundle`、`Primus-Turbo_b0-0927.bundle` | 见 §1.2 | `git clone <bundle>` 或 `git fetch <bundle> <branch>` |

没有备份的东西：
- docker 镜像 `fa-tune:b0-snap`（47 GB）：它是 fa-repro 的快照，可以用 `fa-tune:deps` 重建。
- 宿主机上的 flydsl/hipblaslt 安装：在 `~/.local`，09-27 的 devsync 备份（`2026_0927__bak/home_local.tar.gz`）里已有。

## 3. 拷出去（清盘前必须做）
```bash
# 在 A0 上执行（B0 可达时）：
rsync -av ctheliosp-1b112-a37-1:/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0928__bak_b0/ ~/b0_bak/
cd ~/b0_bak && sha256sum -c SHA256SUMS
# 恢复（A0 上，路径和用户相同）：
tar -C /home/lihuzhan/code/2026_0910__op-evolve/op-evolve -xzf b0_opevolve_jobs.tar.gz   # A0 旧的 bwd job 目录先改名为 *.a0-pre-b0
tar -C /home/lihuzhan -xzf b0_real_qkv_dumps.tar.gz
# 其余按需解包；然后按 output/0927__b0/HANDOFF-A0.md 第 2-3 节修改 final.yaml（容器 fa-repro、host、gpu_id）和 host hint（fwd h49 / bwd h74）
```

## 4. 备份包大小与校验（2026-09-28 生成，全部通过 pigz -t 和 tar tzf 检查）

| 文件 | 大小 | 条目数 | sha256（前 16 位） |
|---|--:|--:|---|
| `b0_claude_sessions.tar.gz` | 1928.0 MB | d5a8067987d48f4d | `` |
| `b0_misc.tar.gz` | 1620.0 MB | aa0d3d9c7ac17b74 | `` |
| `b0_opevolve_jobs.tar.gz` | 11072.0 MB | 9fe9ccc756236e54 | `` |
| `b0_output_0927b0_full.tar.gz` | 12167.0 MB | 81081a6c6fdb4985 | `` |
| `b0_real_qkv_dumps.tar.gz` | 7.0 MB | 38252482e0eb3903 | `` |
| `op-evolve_lhz-gfx1250.bundle` | 0.0 MB | 873e7ea25e8b490a | `` |
| `Primus-Turbo_b0-0927.bundle` | 0.0 MB | 82f3a698a2cb6580 | `` |

说明：
- 会话备份是 12:55 UTC 左右的快照，之后的对话（包括本清单的生成）不在里面。
- `misc.err` 为空，说明 misc 包没有漏掉文件。
