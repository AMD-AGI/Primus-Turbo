# A6 / A7：一个修掉的禁令违反，和一个我判断错了的 cherry-pick

2026-09-21 · **零 GPU**

## A6 `sweep_attention._reap()` —— 它的注释和它的行为不是一回事

计划里我写它「按进程组杀」。**核实后是错的**，而且错在两头：

- 它的 docstring 本来就写着 "Kill the process group"；
- 它的实现是 `pkill -f <harness>` → `pgrep -f` → `pkill -9 -f <harness>`。

这两个动作都在本机的禁令清单上：

1. **`pkill -9` 打正在跑 kernel 的进程**。每杀一次多留一个 D 态不可杀进程，
   把一个可恢复的状态往「需要人去 AC-cycle」推。
2. **用 `pgrep` 决定要不要升级**。`pgrep` 走 `/proc`，在挂卡时会阻塞在卡死于驱动里的进程上——
   它在这台机器上 hang 过，和 `ps`、`rocm-smi` 一样。
   于是这个为了防挂而存在的清理路径，自己会挂在里面。

还有一点：按模式匹配本身就是错的工具。`-f <harness 路径>` 会匹配**同一个 harness 的每一个并发运行**，
所以清理一个候选可能顺手打掉旁边的 sweep；而且一个模式可以匹配到杀手自己的命令行。

**改法**：候选用 `start_new_session=True` 起，放进自己的进程组；
`_reap` 对进程组发 SIGTERM，用 `os.killpg(pgid, 0)` 做存活检查（**单个信号检查，不会阻塞在驱动上**），
只有在仍然存活时才升级到对**同一个进程组**发 SIGKILL。

### 自测抓到的第二个问题

第一版写完跑自测，结果是「有幸存者、耗时 13 秒、每次都升级到 SIGKILL」。
根因不在信号，在**僵尸**：`communicate()` 抛了 `TimeoutExpired`，所以我们的直接子进程从没被 wait 过，
一个死了但没被回收的子进程**仍然属于那个进程组**，于是 `os.killpg(pgid, 0)` 在一个所有成员都已退出的
组上持续成功，循环白等满 10 秒然后升级。**那等于把「优先 SIGTERM」这件事悄悄废掉了。**

修法是在轮询之前先 `proc.wait(timeout=10)` 把自己的子进程收掉。

自测（纯 CPU，一个 fork 了孙进程的父进程）：

| | 修之前 | 修之后 |
|---|---|---|
| 判定 | `kill` | **`term`** |
| 耗时 | 13.0 s | **0.0 s** |
| 组内残留 | 1 个 | **空** |

---

## A7 cherry-pick `70607aa9` —— 我在计划里的判断是错的，已中止

计划 §A7 写着「已核实零重叠，不预期冲突」。**实际 13 个文件冲突，已 `--abort` 并把树恢复干净。**

**我错在哪**：我用的比较是
`git diff --name-only origin/main...HEAD`（三点，即从 merge-base 到 HEAD），
那只列出**我们自己 29 个提交改过的文件**，然后拿它跟 compat 的 21 个文件取交集，得到空集。

但 cherry-pick 的基线不是 `origin/main`，是**那个提交自己的父提交 `33d9f30e`**。
正确的检查是「这 21 个路径在 `33d9f30e` 和 HEAD 之间有没有差异」——实测 **19 个有差异**。
差异不是我们造成的，是 **main 在 `33d9f30e` 之后自己往前走了**
（`primus_turbo/flydsl/mega/prims.py` 在我们 HEAD 里甚至已经不存在了）。

**「父提交是 HEAD 的祖先」和「cherry-pick 干净」是两件事。** 前者只说明历史相连。

### 所以现在怎么办（按计划的 B 计划，并调整了时机）

这个提交的价值是「让 turbo 在 flydsl 0.3.2 下可 import」，它服务于两件事：
pytest 能收集、以及 Phase E 的派发接线。

**Phase B 和 Phase C 都不需要它**：静态筛跑的是 `op/baseline`，
而 op-evolve 那套内核 `_env.py` 里明写 "NEVER import primus_turbo in this process"。

而计划 §九.4 专门警告过：在一个正在持卡的作业底下改主工作树，
会让锚和候选一起移动，作业报出一个漂亮的「无变化」。
**在发车前去解 13 个我不拥有的 MoE / GEMM / quantization 内核文件的冲突，正是那个风险。**

**决定**：这件事挪到 Phase E 开始时做，在**独立的 git worktree 和独立分支**上做，
验收是 `pytest --collect-only` 能收集，做完再合回来。
cherry-pick 的残留（`buffer_ops.py`、`prims.py`）已移到 `/tmp/cp-residue/`，没有留在树里。
