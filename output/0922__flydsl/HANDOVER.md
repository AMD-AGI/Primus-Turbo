# 节点交接 —— 2026-09-22 16:0x

**卡处于挂死状态，需要一次 AC-cycle。操作者已知悉并会执行。之后节点交给他人使用。**

## 我方状态：已全部停止，不会有任何东西自动起来

| 项 | 状态 |
|---|---|
| op-evolve 循环 | 已写 `.stop` 模块边界哨兵（文件，非信号）|
| supervisor | **从未启动过**（全程直接跑 `op-evolve resume`，不挂 supervisor）|
| release_guard | **未武装**（`PATROL_ARM_GUARD` 是 opt-in，默认关）|
| patrol 定时任务 | 今日早些时候已删除 |
| dmesg 监控 / 哨兵 | 已全部停止 |
| 持卡令牌 `~/gfx1250.owner` | **已删除** |
| 容器 `fa-repro` | 仍在（AC-cycle 后需 `docker start`）|

**在途的 op-evolve 进程卡在挂死的卡上（D-state），没有对它们发送任何信号**——
每次 `kill -9` 都会多留一个不可杀的 D-state 进程，把可恢复推向不可恢复。AC-cycle 会清掉它们。

## 给接手这台机器的人

- 卡需要 AC-cycle 才能用。重启后 `amdgpu` **不会自动加载**（内核命令行里被 blacklist）：
  `sudo modprobe amdgpu && sleep 6 && ls /sys/class/kfd/kfd/proc/`（后者必须为空）
- 重启后 VR 限频会回来，DPM 天花板 1100 MHz——**每次都如此**
- 挂卡时唯一安全的探测是带 `timeout` 的独立 `dmesg` 读和 `/sys` 读。
  `rocm-smi` / `pgrep` / `ps -o wchan` / `docker stop` 都会自己挂住
- **磁盘根分区 99% 满**（53G 可用）。Docker 里约 780 GB 可回收
  （镜像 442 GB + 28 个停止的容器 284 GB + 构建缓存 60 GB），但那些容器可能属于其他用户，
  需要确认后再清。core dump 会写进每个 run 的 CWD 且同名覆盖（`core_pattern=core`），
  每次 faulting round 约 +1.1 GB

## 断电后必查（有前科）

```bash
cd /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo && git fsck --no-dangling
```
2026-09-16 那次断电损坏过六个 loose object 且分支 ref 指向其中之一，
`git status`/`log`/`diff` 全部报 `fatal: bad object HEAD`。

## 作业账目（完好，可恢复）

`gfx1250-flydsl-attn-bwd-20260917-115934`，`best_round 8`，`op/current` 是 round 8 的码。

round 9 **未记账**，但它的产物全部落盘且证据完整（见 `ROUND3-CLOSEOUT.md` 的 round 9 段）。
恢复后只差一次同会话验收测量即可手工收口。
