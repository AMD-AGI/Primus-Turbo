# wgrad 也在坏 tile 上，而且要两个操作数一起改才有用

承接 `GEMM-NN-FINDING.md`。那一轮修好了 dgrad，端到端 1.894×。
修完后的 profile 显示**还有一半没修**：57 次调用、2185.6 ms、占 GEMM 时间 89.8%、
占单步 77%，全是 wgrad。

本轮把它们也修好：**端到端 6.21×（对无 patch），3.28×（对上一轮）**，n=6。

## 结论

| | tps | 跨运行 sd | 单步 | 真实 TFLOP/s |
|---|--:|--:|--:|--:|
| 无 patch（n=9） | 6,128 | 3.91%（三模态） | 5.35 s | ~106 |
| dgrad 修好（n=9） | 11,604 | 0.42% | 2.82 s | ~204 |
| **dgrad + wgrad 修好（n=6）** | **38,043** | 1.97% | **861 ms** | **640.7** |

去掉首次运行（可能受缓存状态影响）后 n=5 是 37,738，sd 0.09%。
五次连跑零卡故障、零 KFD 残留。

正确性：step-1 loss 与前几版**逐位相同**（同一 seed），loss 正常下降，
**峰值显存 141.00 GiB 不变** —— 尽管 lm_head 那次要拷贝 7.83 GiB 的 A，
缓存分配器把它复用掉了，没有推高峰值。

## 走了两段弯路，两段都值得记

### 弯路一：微基准构造了错误的布局，得出 16× 的假象

第一次给 wgrad 写规则时，我按"A 和 B 都是转置 view"构造微基准，测出
`A.contiguous()` 能到 1160 TF/s、净收益 8.4–11.2×。据此写的 rule 2 **端到端零收益**。

原因：**真实的 wgrad 调用里 B 是连续的**，不是我构造的非连续。读命中统计才看出来：

```
stats: {'dgrad': 1140, 'wgrad': 0, 'miss': 570}
    160 x  A(14336, 32768) Av   B(32768, 4096) Bc    <- wgrad：A 是 view，B【连续】
    160 x  A(32768, 4096)  Ac   B(4096, 14336) Bv    <- 前向：A 连续，B 是 view
```

（表里记的是**传入时**的布局，在判据之前采样。所以 `Bc` 的那些会被 rule 1 认领 ——
包括本该归 wgrad 的那些；而 `Bv` 的那些两条规则都不碰。）

**rule 2 命中 0 次。** 它的判据 `not b.is_contiguous()` 永远不成立，
因为 wgrad 的 B 本来就是连续的，而 rule 1（`b.is_contiguous()`）先把它们拦走了。

### 弯路二：以为 rule 1 在帮倒忙，也错了

既然 rule 1 把连续的 B 改成 N-major view，而 profile 里的坏 tile 正是 `Alik_Bjlk`
（A view + B N-major），我一度推断 **rule 1 在主动把 wgrad 推进坑**。用真实布局测，证伪：

| | untouched (Av+Bc) | rule1 (B→N-major) | rule2 (A→contig) | **两个都改** |
|---|--:|--:|--:|--:|
| mlp wgrad | 54.3 TF/s | 69.3 | 62.7 | **1145.8** |
| lm_head | 49.9 | 67.2 | 62.6 | **1485.3** |
| qkv wgrad | 56.5 | 69.2 | 66.7 | **1627.0** |
| 含拷贝净收益 | — | 1.26× | 1.10× | **9.36× / 14.74× / 7.76×** |

rule 1 是有帮助的（1.16–1.34×），只是远不够。

**任何单独一个都只到 60–70 TF/s，两个一起才跳到 1145–1627。**
所以之前那个"A.contiguous() → 1160 TF/s"数值没错，但功劳归错了：
当时构造的 B 恰好已经是 N-major view，实际是"两个都改"的效果。

## 修法

```python
# 顺序很重要：wgrad 判据必须在前。wgrad 的 B 也是连续的，
# 所以 dgrad 规则会先认领它，然后把它留在坏 tile 上 —— 上一版正是这样静默失败的。
if not a.is_contiguous() and b.is_contiguous() and _big(a):
    return func(a.contiguous(), b.t().contiguous().t())     # wgrad：两个都改
if b.is_contiguous() and _big(b):
    return func(a, b.t().contiguous().t())                  # dgrad：只改 B
```

命中统计验证：`{'dgrad': 570, 'wgrad': 570, 'miss': 570}` —— wgrad 从 0 次变成 570 次。

数值零代价：四种变体对 fp32 参考的 SQNR 全部一致（55.60–55.62 dB）。

## 顺带查出：所有 MFU 数字都是错的

`torchtitan/tools/utils.py::get_peak_flops()` 的兜底分支：

```python
else:  # for other GPU types, assume A100
    return 312e12
```

本机 `device_name` 是 `AMD Radeon Graphics`，不匹配任何已知型号（容器里也没有 `lspci`），
**所以 MFU 的分母一直是 A100 的 312 TFLOP/s**。
验证：本轮 tflops 672.76 ÷ 312 = 2.156 = 日志报的 215.63%。

**`tflops` 列是对的**（FLOP÷时间），**`mfu` 列一律不可用**。
此前材料里的「MFU 34% → 65.4%」应读作 TFLOP/s「~106 → ~204」。
若改用本 campaign 实测的 Triton GEMM roof（1002.7 TF/s @1100 MHz）作分母，
本轮的 640.7 TF/s 是 **63.9%** —— 那才是有意义的利用率。

## 仍未解决

- **32 层生产配置未测。** wgrad 规则要拷贝 A（lm_head 是 7.83 GiB）。
  8 层配置下峰值显存没变，但 32 层已在 88%，不能据此推断。
- **`MES ring buffer is full` 仍未归因。** 本轮 5 次连跑零故障；
  昨天那次出现在第 4 次连续运行，前天那次在第 4 次。样本仍不足以定性。

---

## 32 层验证：SIGBUS，根因明确，已加自适应保护

**结果：32L + b=4 + wgrad 规则 = SIGBUS，第 1 步之前就死。**

| 配置 | 显存 | 结果 |
|---|--:|---|
| 32L, b=2, wgrad 规则 | **56.41%** | ✅ 10,646 tps，跑完 10 步 |
| 32L, b=4, wgrad 规则 | — | ❌ **SIGBUS**，0 步 |
| 32L, b=4, 无 patch（历史） | 87.98% | ✅ 跑完 20 步 |

原因不神秘：32L b=4 本来就在 **87.98%**，而 wgrad 规则要为 lm_head 拷贝 **7.83 GiB** 的 A。
没有那个余量。

### 这次故障的表象具有误导性，值得单独记

dmesg 的表现是：

```
MES(0, 0) failed to respond to msg=INVALIDATE_TLBS      (反复)
MES might be in unrecoverable state, issue a GPU reset
GPU reset begin!. Source: 3                             ← 没有对应的 reset end
```

读起来像是本周追了三次的那个 MES wedge。**实际根因在训练日志里，一行**：

```
Signal 7 (SIGBUS) received by PID 1098
```

进程 SIGBUS 死掉后不释放 KFD 上下文，驱动才去尝试 reset，而 reset 没能完成。
**MES 报错是次生现象。** 这与 0915 那次 scratch 常驻导致的 SIGBUS 是同一个模式。

**教训：先读训练日志，再对驱动下结论。** 前几次 MES 故障也该按这个顺序重看一遍。

### 修法：问一句能不能装得下，而不是假设配置有余量

```python
def _headroom_ok(t):
    free, _total = torch.cuda.mem_get_info()
    need = t.numel() * t.element_size() * 2   # contiguous() 同时持有源和目标
    return need < free * _HEADROOM            # 默认 0.5
```

同一条规则在 32% 占用的配置上免费、在 88% 的配置上致命，所以**判据必须是运行时的可用显存，
不是固定的尺寸阈值**。

显存不足时**不放弃，而是落回 dgrad 规则** —— 它不需要额外分配，在 wgrad 调用上仍值
1.16–1.34×，远好于吃一个 SIGBUS。统计里单列 `wgrad_skipped_oom`，这样"规则没生效"
和"规则生效了但没效果"永远能分辨。

### 预期

按 profile，lm_head wgrad 是 510.7 ms / 1 次，其余 wgrad 合计 1674.9 ms / 56 次 ——
**跳过最大的那一个仍能拿到 wgrad 收益的 77%**。32L 上的实际数字待测。

## 32 层生产配置跑通：5.85×，但成功的机制未定

AC-cycle 后带 headroom 保护重跑，**20 步完整完成**：

| | tps | 单步 | 峰值显存 |
|---|--:|--:|--:|
| 32L 无 patch，ASM-bwd 关 | 1,984 | 16.5 s | 87.98% |
| 32L 无 patch，ASM-bwd 开 | 1,858 | 17.6 s | 87.98% |
| **32L + nkfix** | **11,608** | **2,823 ms** | **88.32%** |

**5.85×（对 1,984）/ 6.25×（对 1,858）**，稳态 n=13 步，逐步抖动 <0.4%。
显存只比无 patch 高 0.34 个百分点。

### 但不能把这次成功归功于 headroom 保护

```
stats: {'dgrad': 4500, 'wgrad': 4500, 'wgrad_skipped_oom': 0, 'miss': 4500}
```

**`wgrad_skipped_oom` 是 0 —— 保护一次都没触发。** 算一下就知道为什么：
`torch.cuda.mem_get_info()` 返回的是**设备级** free（缓存分配器已占的不计入），
88% 占用时约 51 GiB；lm_head 的 A 拷贝需要 7.83×2 = 15.66 GiB，
对阈值 `51 × 0.5 = 25.9` 从来没触到过。

**所以带保护和不带保护的代码路径，在这个配置上是一样的。** 上一次 SIGBUS 而这一次没有，
差别在别处：

- **AC-cycle 前后的卡状态不同。** 上次是在一连串实验之后跑的，这次是 AC-cycle 后的干净卡。
- **`mem_get_info` 可能引入了隐式同步**，间接改变了分配时序 —— 这是推测，未验证。

**诚实的结论：32L 现在能跑通，5.85× 是实测；上次 SIGBUS 的确切触发条件仍未定。**
在搞清楚之前，32L 上的每次运行都应当监控显存并保留 dmesg。

要定它，最便宜的实验是：**同一块干净卡上，连续跑 32L 三次**，看是否稳定复现成功。
若三次都成，说明上次是脏卡状态；若再次 SIGBUS，说明有别的随机因素。

### 顺带修正前文的一个预期

前文推测"跳过 lm_head 仍能拿到 wgrad 收益的 77%"—— **这个前提不成立**，
因为根本没有跳过任何调用。8 层的 38,043 tps 与 32 层的 11,608 tps 之间的差距，
是层数带来的（32 层每 token 的工作量是 8 层的四倍），不是保护跳过造成的。
两者都不该拿来互相换算。
