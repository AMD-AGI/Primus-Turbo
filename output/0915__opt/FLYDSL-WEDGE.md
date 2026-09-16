# 规则 3 第一次上机就把卡搞挂了：原因分析与改法

日期 2026-09-16 · 运行 `fly1-fly` · 8 层配置 · **需要一次 AC-cycle**

## 事实

```
step:  1  loss: 12.13813  memory: 125.34GiB(29.01%)  tps: 1,313
terminate called after throwing an instance of 'c10::AcceleratorError'
  what():  CUDA error: unspecified launch failure   (hipErrorLaunchFailure)
Signal 6 (SIGABRT)
```

随后 dmesg：`MES(0,0) failed to respond to INVALIDATE_TLBS / REMOVE_QUEUE / SUSPEND / RESET /
RESUME` → `MES might be in unrecoverable state` → `GPU reset begin!` → 未完成。

**step 1 跑完了，step 2 死。** 同一配置带规则 2（两次拷贝）今天已经跑过几十次，零故障。
所以嫌疑集中在规则 3，但这不是证明 —— 只有一臂跑过，没有对照。

**注意：规则 3 默认关闭**（`NKFIX_FLYDSL_WGRAD` 未设即不启用），所以默认路径不受影响。

## 我先排除掉的一个猜测

第一反应是"我漏了 `can_run()`"——NN 的基准里调了，规则 3 里没调。**这个猜测不成立**：
`gemm_gfx1250` 内部会调 `_resolve`，不合法的形状会抛 `ValueError`，而我的 `except Exception`
接得住。校验本身是完整的（K % tile_k、LDS 预算、累加器寄存器、转置块整除等）。

## 两条确实存在的问题

### 1. `autotune` 的异常处理救不回已经损坏的 HIP context

`autotune` 的测量循环是：

```python
try:
    for _ in range(3): call()
    torch.cuda.synchronize()          # <-- 发射失败在这里抛出
    ...
except Exception:
    continue                          # <-- 被吞掉，继续试下一个配置
```

一个候选配置若发射失败，`synchronize()` 抛异常、被吞、循环前进。但 **HIP context 一旦吃到
`unspecified launch failure` 就永久损坏**，捕获 Python 异常不能让它复活。之后第一个真正的
GEMM 报错，而此时真正的肇事者已经在 30 个配置之前。

**它把"上下文已经死了"静默转成了"那个配置不算候选"。** 这一条与"谁先 fault"无关 ——
**在训练步里跑 autotune 本身就不安全**。

### 2. 调优缓存的 key 漏掉 M，而给出的理由在 TN 上正好反过来

```python
# Tuned configs, keyed on (kind, layout, N, K). M is left out: it is the token
# count, varies per step, and moves throughput far less than N and K do.
```

对 NT（Linear 前向）和 NN（dgrad）成立。但 TN（wgrad）的定义是
`A[K,M]^T @ B[K,N]`——**M 是 out_features，K 才是 token 数**，正好反了。

后果：o_proj 的 wgrad（M=4096, N=4096, K=32768）与 kv 的（M=1024, N=4096, K=32768）
**共用同一个调优配置**；而 `feasible_configs` 的筛选条件里**完全没有 M**
（只筛 N 和 K），所以换了 M 之后没有任何一处复核。

**这是分支作者需要知道的问题。它是不是杀死卡的那一刀，我没有证据，不这么写。**

## 改法

规则 3 重写为**离线表驱动，训练内绝不调优**：

- `output/0915__opt/bin/flydsl_table.py` 在**空闲卡**上生成配置表，
  **每个形状一个子进程**（一个形状 fault 只损失那个形状），带硬超时，
  并对每个形状**单独过 SQNR ≥ 50 dB 门**——一个又快又错的配置绝不能写进训练要读的表。
- `nkfix.py` 只读表，key 是**完整的 (M, N, K)**，表里没有的形状**退回规则 2**，
  计入 `flydsl_no_config`。训练里不再有任何 `autotune` 调用。

## 代价与教训

这次冒险花掉一次 AC-cycle。可以更早避免：**我自己在估算文档里写过"kv (M=1024) 用
qkv (M=6144) 的实测值代入，仍是估算"**——即我知道有一个形状没实测过，却还是让训练去
现场调优它。标注了假设，却没有在行动上尊重那个标注。

写入全局 skill：**离线能做的调优，绝不放进在线路径**；以及**一个吞掉异常后继续的循环，
前提是异常可恢复 —— GPU 发射失败不可恢复。**
