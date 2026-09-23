# 第四次挂卡 —— 2026-09-23 约 09:21（round 11 的 d_s4）

## 一句话

**触发是 `INVALIDATE_TLBS` 超时，不是内存越界。我先前建立的归因不完整，h15 的判别式覆盖不到这一类。**

## 故障链（完整抓到，这次 dmesg 没丢）

开机 2155 秒到 34444 秒之间 **零 amdgpu 异常**，九个多小时干净。然后十秒内级联：

```
[34444.85] MES(0,0) failed to respond to msg=INVALIDATE_TLBS   ← 第一条，之前没有任何故障
[34447.33] MES(0,0) failed to respond to msg=REMOVE_QUEUE
[34449.80] MES(0,0) failed to respond to msg=SUSPEND
           failed to suspend all gangs
[34452.28] MES(0,0) failed to respond to msg=RESET
           Failed to detect hung queues
[34454.76] MES(0,0) failed to respond to msg=RESUME
           failed to resume all gangs
           failed to remove queue from MES, doorbell=0x2a2c
[34454.79] MES might be in unrecoverable state, issue a GPU reset
           Failed to evict queue 2
[34454.80] GPU reset begin!. Source: 3
           Failed to evict process queues
```

随后是 14 次 `REMOVE_QUEUE` + `failed to unmap legacy queue` 的无界级联。

## 这次**没有**出现的东西（这才是关键）

| h15 判别式里的特征 | 前三次 | 本次 |
|---|---|---|
| 4 GB 以下的截断故障地址 | 有 | **无** |
| `PERMISSION_FAULTS: 0x5, RW: 0x1` | 有 | **无** |
| `GCVM_L2_PROTECTION_FAULT` / 页故障 | 有 | **无** |
| `Cijk_*` Tensile kernel 名 | 有 | **无** |
| `copy_context_work_handler hogged CPU`（早停信号）| 有 | **无** |

**我在 h15 里立的早停信号，这次一条都不会触发。**

## 对先前归因的更正

我建立的说法是「Tensile 的 fp32 GEMM 先损伤卡状态，下一个重派发压垮它」。

- 对 **round 3** 成立：有文档记载的 aperture violation 签名、`Cijk_` kernel 名。
- 对 **round 8** 和**本次**不成立：两次的**第一条**都是 `INVALIDATE_TLBS` 超时，
  **没有任何前置内存故障**。round 8 那次我看到的通知第一行同样是 INVALIDATE_TLBS。

所以**至少存在两类挂卡机制**，而我把它们并成了一类：

1. **内存越界类**（round 3）：Tensile 的越界写 → L2 保护故障 → MES 降级 → 升级
2. **TLB/队列类**（round 8、本次）：`INVALIDATE_TLBS` 超时 → 队列管理全线失效 → MES 不可恢复

第二类的直接原因**尚不清楚**。已知的只有：不是内存故障、没有前兆、卡在此前长时间干净运行。

## 待查（下次有卡时）

- `INVALIDATE_TLBS` 超时是否与**并发的 KFD 上下文数**相关？本次是 round 11 的第四个 GPU
  session（d_s4），前三个 rc=0。round 8 也是在一轮的后段。
- `doorbell=0x2a2c` 这个具体的 doorbell 是否可追溯到某个队列/进程。
- 两类是否共享一个更上游的原因（例如队列创建/销毁的累积）。
- h15 的早停信号需要补一条：`failed to suspend all gangs` 或 `Failed to detect hung queues`
  出现即中止——它们在本次出现在**不可恢复之前 5 秒**，是这一类唯一可用的窗口。

## round 11 的状态

`d_s1/d_s2/d_s3` 全部 rc=0，`opt.md` 与 `act.yaml` 已写出（09:20），挂在 `d_s4`。
候选 `r11.i4.g36`，`outcome: delivered`，但 `shipped: None`。

**静态读数已经给出负面信号**：两个臂的 bank 切换数都比基线**涨了**——
`cur` 323 条 / 740 VGPR，`armA` 339 / 772，`armB` 355 / 780。
g30 想减的那个税，两个臂都把它做大了。
