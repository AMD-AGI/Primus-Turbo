# 两 kernel Triton 反向：配置空间确实没扫过，18% 也确实在那里，但这条路还是死的

日期 2026-09-16 · 形状 `llama31-8b-s4096`（b=4 s=4096 hq=32 hkv=8 d=128，causal，bf16）

## 起因

`primus_turbo/triton/attention/attention_kernel.py` 的 `_build_configs`（:190-213）默认只返回
**单元素列表**，文件自注 *"a compile cache, not a search — this space has never been swept on
this arch"*。而 0915 那次 op-evolve 唯一被接受的一轮，赢家旋钮之一正是 `waves_per_eu 1→0`，
但它**不在**内置的 opt-in 网格里。所以这里预期有未采的收益。

先解决两个会让结果作废的前置问题：

1. **`sweep_attention.py` 没有任何超时保护。** 已知 `num_warps=16` 在本卡上是**挂死**而非编译
   失败 —— 子进程会无限等待，看门狗也不会介入，最后只能 AC-cycle。现已加 `CANDIDATE_TIMEOUT_S`
   （默认 600 s，可用 `SWEEP_TIMEOUT_S` 覆盖）并在超时后 `pkill` 回收，**同时把 `num_warps=16`
   排除在网格外** —— 为一个格点冒钉住卡的风险不值得。超时记为终局判定而非重试：挂死是可复现的，
   重试只会再花一次超时。
2. **`waves_per_eu` 是否真的生效。** 这个版本的 `triton.Config.__init__` **不接受**
   `waves_per_eu` 参数，而解析器却接受这个 key，于是它被留在 kwargs 里当 constexpr 传给一个
   没有声明该参数的 kernel —— 典型的"扫出一条平线，因为覆盖从未生效"。实测确认它走的是
   backend option 通道：`Config({'waves_per_eu': 3}, ...)` 后 `best_config` 里带着它，
   单 kernel 试验里 `compiled.metadata.waves_per_eu` 也确实是 3。**这条路是通的。**

## 结果：33 点全部 ok，0 个 SQNR 不过，0 个失败

| 配置 | bwd ms | total ms |
|---|--:|--:|
| **num_warps=2, waves_per_eu=1, num_stages=1** | **7.507** | **8.950** |
| num_warps=2, waves_per_eu=0, num_stages=1 | 7.533 | 8.987 |
| `<默认>` (num_warps=4, ns=1) | 9.158 | 10.604 |
| …（最差）num_warps=1, waves_per_eu=4, ns=2 | 71.158 | 72.599 |

赢家在 `num_warps` 与 `waves_per_eu` 两个轴上都是**内点**（不是区间边缘，不需要扩展区间），
`num_stages=1` 是该旋钮的硬下界，已收敛。

交替 3 对复现：

```
赢家   7.475 / 7.471 / 7.482   sd 0.07%
默认   9.097 / 9.104 / 9.116   sd 0.10%
效应   −17.8% (bwd) / −15.5% (total)   四张量 SQNR 均为 52.2，不变
```

**曲面极度非单调**：`waves_per_eu` 从 1 走到 4，同样的 `num_warps=1` 下从 15.9 ms 变成 63.8 ms
（4 倍）。这正是"默认值 = 编译缓存"这种做法的代价：没人扫过，就没人知道默认值落在哪。

## 但这条路仍然是死的

把三条反向路径放到**同一形状、同一次会话**里量：

| 反向路径 | bwd ms | 相对最优 |
|---|--:|--:|
| 两 kernel · 默认 | 9.10 | 3.32× |
| 两 kernel · **本轮调优赢家** | 7.47 | 2.73× |
| 融合反向（`auto`，e2e 的旧默认） | 5.60 | 2.05× |
| **ASM 反向**（e2e 现用） | **2.740** | 1.00× |

调优把两 kernel 路径从落后 3.3× 拉到落后 2.7×。**18% 是真的，但它赢在一条 e2e 不走的路上，
而且这条路的起点比现用路径慢 2.7 倍 —— 靠调配置追不上。**

结论：**两 kernel Triton 反向路径就此关闭**，与 0915 已经关闭的 vendored 融合反向配置面并列。
Triton 在本卡这个形状上的反向，两条路都到头了。

修改后的 sweep 超时保护与 `waves_per_eu` 验证保留，它们对以后任何一次 sweep 都有效。

## 顺带得到的下一个目标

反向压到 2.740 ms 后，**前向占到总时间的 34%**（1.424 / 4.164 ms）。而且实测发现
**ASM 前向在这个形状上根本没生效** —— `--asm-fwd auto` 与 `--asm-fwd off` 分别是
1.426 与 1.416 ms，关掉反而略快，说明跑的一直是 Triton 前向（384 TFLOP/s）。
前向的配置空间同样从未扫过，已开扫（72 点，含 `PRE_LOAD_V` 与 `waves_per_eu`）。

## 一个需要记录的观察

开 ASM 反向后四张量 SQNR 从 52.2 降到 **dk 50.56 / dv 50.84** —— 仍然过 50 dB 门，
但**余量只有 0.5 dB**。融合反向的余量是 2.2 dB。这不是问题，但如果将来门线上浮或换形状，
ASM 反向会是先撞线的那个，应当记住。
