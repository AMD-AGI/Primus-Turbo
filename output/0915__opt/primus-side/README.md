# Primus 侧的改动备份（不属于 Primus-Turbo，也不该推给上游）

Primus 仓库（`/home/lihuzhan/code/2026_0828__primus/Primus`）的 remote 是上游
`AMD-AGI/Primus`，所以今天在那边的改动**没有也不应该**推上去。备份在这里，
以免同事清理工作区时丢失。

## `train_runtime-nkfix-hook.patch`

14 行，在 `_run_trainer_lifecycle()` 开头按 `NKFIX_ENABLE=1` 安装 `bin/nkfix.py`。

**为什么在这里而不是 sitecustomize**：`sitecustomize` 在解释器启动时就跑，那时 torch
还不存在，`torch.ops.aten.mm.default` 求值直接崩；而且它会在每个无关的 python 子进程里
触发。放在这里时 torch 已完全加载，且只在真正跑训练循环的那个进程里生效。

重放：`cd <Primus>; git apply <此 patch>`

## 配置文件

这四个 yaml 在 Primus 树里是 **untracked**（整个 `examples/torchtitan/configs/MI455X/`
都不在版本控制里），所以只存在于那台机器的工作区。

| 文件 | 用途 |
|---|---|
| `repro_l8b_turbo_conv.yaml` | 32 层生产配置，显存 88%，今天两次挂卡都与它有关 |
| `repro_l8b_turbo_conv_8L.yaml` | 8 层，20 步，峰值显存 141.00 GiB（32.64%）—— 绝大多数 A/B 用它 |
| `repro_l8b_turbo_conv_8L_fast.yaml` | 8 层，10 步（已验证 step 4–8 中位与 8–20 差 ≤0.22%），每次省 53% |
| `repro_l8b_turbo_conv_8L_prof.yaml` | 8 层 + torch profiler，抓 step 5 |

三个带 converter 的配置里都加了警告：**不要把 `converters:` 改成 `[]`** ——
那会退回 Flex FA，其内部强制 `torch.compile`，在本机会打死 GPU（0915 挂过一次）。

## 0916 核对

五个 yaml 与权威副本
`/home/lihuzhan/code/2026_0828__primus/Primus/examples/torchtitan/configs/MI455X/`
逐字节一致，`train_runtime-nkfix-hook.patch` 与该仓库工作区的实际 diff 一致。
**这里是唯一的备份** —— Primus 那个仓库的 remote 是上游 `AMD-AGI/Primus`，这些改动不推上去。

同时删掉了 `output/0915__opt/e2e/` 下的三份同名旧副本。它们是陷阱：缺 `debug.seed`
（照着跑出来的 A/B 两臂模型初始化不同，0915 第一轮 8L A/B 就是这么作废的），
也缺 `converters: []` 会打死这张卡的警告块。**要用配置，只看这个目录。**
