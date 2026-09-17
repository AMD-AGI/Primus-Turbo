# GQA 越界写绕法的代价：第一次实测

日期 2026-09-17 · `heliosr-1b114-c07-1` 单卡 @1100 MHz · 脚本 `bin/gqa_workaround_cost.py`
原始数据 `gqa_workaround_cost.json` · 形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`

这个绕法（aiter gfx1250 ASM 反向按 q head 索引 kv 尺寸缓冲区，我们改为按 q head 分配 + host 规约）
**从 0915 引入起就没有单独计过量**，而它在当前冠军的关键路径上。现在有数字了。

## 结果

### 显存

| | |
|---|--:|
| 正确内核需要的 dk/dv（`[B,S,Hkv,D]`） | **0.125 GiB** |
| 绕法实际持有（`[B,S,Hq,D]`，实测 `dk.shape=(4,8192,32,128)`） | **0.500 GiB** |
| 多出来的 | **0.375 GiB（4.0×）** |
| **整次反向发射的峰值增量** | **1.254 GiB** |

此前文档里反复出现的"1 GiB 常驻 scratch"对应的是**峰值增量 1.254 GiB**，
不是 dk/dv 张量本身（那是 0.500 GiB）。两个数都记下来，别再混用。
4.0× 正是 GQA ratio，符合预期。

### 时间

ABAB 交替、CUDA event、每次 256 MiB L2 flush、n=15：

| 臂 | median | best | sd |
|---|--:|--:|--:|
| `q`（正确发射，不含规约） | 9.195 ms | 9.101 ms | **28.96%** |
| `q+reduce`（实际交付的） | 9.677 ms | 9.584 ms | 0.75% |

**host 规约的代价 = +0.482 ms，占反向的 +5.2%。**

**关于那个 28.96%**：不掩饰。`q` 臂有少数大离群点（median 9.195 与 best 9.101 很近，
说明多数样本聚在 9.1–9.2），而 `q+reduce` 臂只有 0.75%，两臂交替进行，
所以这不是时钟漂移而是该臂自身的分配抖动（每次调用新分配 dk/dv，与另一臂释放的块交错）。
**结论对统计量不敏感**：用 best-of-N 算是 `9.584 − 9.101 = +0.483 ms`，
与中位数的 +0.482 ms 相差 0.2%。

时钟见证：窗口内 1100 → 1017 MHz，两臂等量承受。

## 这值多少

反向在冠军里是 10.160 ms / 总 11.726 ms。省掉 0.482 ms 相当于**总时间 −4.1%**。
再加上 0.375 GiB 的常驻显存 —— 在 32 层、显存已到 88% 的配置里，那不是零。

所以 `PLAN.md` 把它排在 op-evolve 目标的第 2 位、并注明"这是一天的手工修复，不是 40 轮搜索"
是对的：**收益明确、上界已知、不需要搜索**。

## 过程中挂了一次卡（可恢复，未花 AC-cycle），原因是我的实验设计

第一版脚本把 `dkdv_heads="kv"`（**已知越界写的那条路径**）当作"时间下界"来测。它在 s=8192 上：

```
HW Exception by GPU node-2 ... reason: GPU Hang
GCVM_L2_PROTECTION_FAULT_STATUS_LO32: 0x00D040A1
  Faulty UTCL2 client ID: TCP (0x8)   PERMISSION_FAULTS: 0x5   RW: 0x1
MES(0,0) failed to respond to msg=REMOVE_QUEUE / SUSPEND
queue id 0x1 at pasid 43714 is reset
Queues reset on process python3
```

**分类：可恢复。** 零 `wait for reset ack`、零 `GPU reset begin`、零 `ring gfx timeout` ——
驱动做的是**按进程的队列重置**，卡本身没坏。事后直接测量确认：
无 KFD 持有者，4096³ bf16 matmul **5.08 ms**（故障前 5.09 ms），结果有限。
**代价是一个进程，零次 AC-cycle。**

教训，而且是最扎心的那种 —— **我自己一小时前写的上报文档里就说了这条路径在 s=256 会 fault、
s=1024 会静默损坏**，然后我让它在 s=8192 上跑。

**给一条已知越界写的路径计时，得到的不是下界，是一次故障。**
那个"下界"本来想得到的分配差值是**算术**，不需要发射任何内核就能算出来 —— 上表就是这么来的。

与 0916 那次的同源之处：把已经写下来的假设当成了背景知识而不是行动约束。
已写进 `~/.claude/skills/gfx1250-card-safety`。
