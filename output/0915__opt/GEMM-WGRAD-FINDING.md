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
    160 x  A(14336, 32768) Av   B(32768, 4096) Bc     <- wgrad：A 是 view，B 连续
    160 x  A(32768, 4096)  Ac   B(4096, 14336) Bv     <- dgrad
```

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
