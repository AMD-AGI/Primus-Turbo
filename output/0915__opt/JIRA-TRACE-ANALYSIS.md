# 解 JIRA 的 trace：数字对上了，而 fwd/bwd 的拆分是 JIRA 没给的那一半

日期 2026-09-17 · trace 来自 `/home/lihuzhan/traces/`（两个 zip，共 5 份 rank0 trace）· **全程未使用 GPU**

## 一、先验证能不能对上 JIRA 的数

`jira_1` 给的是 "AITER fmha" 一行：MI455 **414.3 ms** / MI355 **294.6 ms**。
按 kernel 名把 trace 里所有 attention kernel 求和（每个都是 `n=32`，即 32 层 × 每层一次，
所以就是**单步**）：

| trace | attention 合计 | JIRA | 差 |
|---|--:|--:|--:|
| `mi455-case3_20260901_130748`（27,891 tks） | **414.3 ms** | **414.3** | **0.0%** |
| `mi355-case4_20260831_114405`（21,521 tks） | 293.7 ms | 294.6 | 0.3% |
| `case6_20260910_063635`（MI355 BF16，21,502 tks） | **294.2 ms** | 294.6 | 0.1% |

MI355 那 0.3% 的差不是解析误差 —— **JIRA 的 294.6 对应的是 0910 那次运行**（294.2），
而不是 0831 那次（293.7）。三次独立的 MI355 运行给出 291.9 / 293.7 / 294.2，彼此相差 0.8%。

**口径确认**：自带的对比报告写明 `ProfilerStep#8` = 日志 step 9，
`traceName` 为 `iteration_9/rank0_trace.json`，与 `jira_2` 的 "profile step 9" 一致。

一个命名上的坑：`jira_1` 那一行写的是 "AITER fmha"，但 **MI455 一侧跑的根本不是 AITER**，
是 inductor 的 Triton Flex Attention（`triton_tem_fused_flex_attention*`）——
这一点 `jira_2` 的 "FA path: Triton flex vs AITER fmha (gfx950)" 说清楚了，
但 `jira_1` 的表头容易让人误读成两边跑同一个实现。

## 二、JIRA 没拆的 fwd/bwd：差距**全部**在反向

| 单步（32 层合计） | MI455（Flex） | MI355（AITER） | MI455 相对 |
|---|--:|--:|---|
| **前向** | **105.9 ms** | 121.5 ms | **快 1.15×** |
| **反向** | **308.4 ms** | 172.2 ms | **慢 1.79×** |
| 合计 | 414.3 ms | 293.7 ms | 慢 1.41× |

**MI455 的 attention 前向比 MI355 还快 13%。** JIRA 的结论"MI455 慢在 FA 软件栈上"方向没错，
但它掩盖了前向其实已经赢了 —— **能省的 120 ms 全部在反向那一侧。**

反向/前向比值也很能说明问题：MI455 **2.91×**，MI355 **1.42×**。
一个结构良好的 FA 反向大约是前向的 2–2.5 倍工作量，MI355 的 1.42 说明它的反向实现效率很高，
而 MI455 的 2.91 说明 Flex 的反向是拖后腿的那个。

另外两个旁证：

- **compile + autotune 对 attention 几乎无效**。MI455 两次运行（eager 24,651 tks 与
  compile+autotune 27,891 tks）的 attention 分别是 410.1 与 414.3 ms —— **基本没动**。
  E2E 提升了 13%，全部来自别处。
- **MI455 的 GEMM 本来就赢**：534.8 ms vs MI355 的 997.9 ms，**1.87×**。
  与 `jira_2` 的 "GEMM is already 1.51× faster" 同向（那次是另一组配置）。

## 三、和我们自己的数对比

**形状完全一致** —— 我们 `RESULTS.md` 的阶梯用的就是 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`，
正是 32 层里的一层。JIRA 的数除以 32 即得单层：

| 单次调用（一层） | fwd | bwd | total |
|---|--:|--:|--:|
| 我们 · turbo 出厂 | 10.651 | 45.134 | 55.785 |
| JIRA · MI455 Flex | 3.309 | 9.638 | 12.947 |
| JIRA · MI355 AITER | 3.797 | 5.381 | 9.178 |
| **我们 · 今日冠军（ASM 前向 + ASM 反向）** | **1.572** | **10.160** | **11.726** |

### 三条结论

1. **我们的"出厂"55.785 ms 不是一个硬件结论。** 它比 JIRA 的 MI455 差 4.3×，
   但两者跑的是**不同的软件路径**：我们是 turbo 的 Triton，JIRA 的 MI455 是 inductor Flex。
   拿它说"这块卡慢"是错的。
2. **我们的前向已经大幅领先两边**：1.572 ms vs MI455 的 3.309（**2.10×**）、
   MI355 的 3.797（**2.42×**）。
3. **剩下的差距全在反向**：我们 10.160 ms —— 与 MI455 的 Flex 反向基本持平（对方 9.638，
   我们慢 5%），但**比 MI355 的 AITER 反向慢 1.89×**。

合计上我们 11.726 ms 已经**优于 JIRA 的 MI455 基线 1.10×**，但**仍落后 MI355 1.28×**，
而这 1.28 完全由反向贡献。

### 必须同时说明的四条限定

- **我们的卡在限频**：标称 1100 MHz，实测负载下 **1001 MHz（91%）**。
  JIRA 那台 MI455 的时钟未知。若对方跑在满频，我们的数是偏悲观的 —— 真实优势可能更大。
- **硅版本不同**（我们是 A0-c07-1），软件栈版本也不同。
- **测量方式不同**：我们的是**隔离的算子微基准**，JIRA 的是**训练中 trace 出的 kernel 时间**。
  两者都是 GPU kernel 时间，但缓存状态与并发情况不同。
- **不要拿我们的 E2E tps 和 JIRA 的并列**。我们 32 层最好是 14,050 tps，JIRA 的 MI455 是
  27,891 —— 差距来自限频、来自我们的 GEMM 还走在 nkfix 绕路上（而不是一个正常的库），
  以及软件栈差异。**我们的 7.08× 是相对自己的出厂基线，不是相对 JIRA。**

## 四、一条可执行的线索：我们和 MI355 跑的是**不同的 ASM 反向变体**

| | 反向 kernel |
|---|---|
| **我们**（e2e 实测） | `fmha_bwd_hd128_bf16_causal_br_**a32**_pssk` —— **单个 kernel** |
| **MI355**（trace 实测） | `fmha_bwd_hd128_bf16_causal_br_**a16**_pssk**ddv**` 164.9 ms<br>+ `fmha_bwd_hd128_odo_bf16` 4.6 ms<br>+ `fmha_bwd_hd128_dq_shuffle` 2.7 ms —— **三个 kernel 的分解** |

三处差别：**`a32` vs `a16`**、我们缺 **`ddv`** 后缀、以及 MI355 把 `odo`（dO·O 预处理）
与 `dq_shuffle` 拆成了独立 kernel 而我们是一个融合 kernel。

而这正是差距所在的那一半：MI355 这三个加起来 172.2 ms，我们 32 层折算约 325 ms。

**下一步该查的**（无需 GPU 即可开始）：本机 aiter 的预编译 `.co` 集合里是否**存在 a16 变体**，
以及选择 a32 的判据是什么。如果 a16 在 gfx1250 上可用，这是一条比继续调 Triton 便宜得多的路。
**注意这是一条线索，不是结论** —— `a16/a32` 具体含义、以及它在 gfx1250 上是否合法，都还没查。

## 五、复现方式

```bash
mkdir -p /tmp/tr && cd /tmp/tr
unzip -oq /home/lihuzhan/traces/MI455-vs-MI355-BF16-amdprimus_20260730.zip -x "*.pickle"
unzip -oq /home/lihuzhan/traces/MI355-BF16_FP8-amdprimus_20260910.zip
python3 attn.py *.json.gz          # 脚本见 output/0915__opt/bin/trace_attn.py
```
