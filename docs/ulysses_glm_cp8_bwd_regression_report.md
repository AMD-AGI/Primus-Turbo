# Ulysses 融合 kernel：GLM-4.7 反向性能回退排查报告

## 1. 结论

- **回退属实，由 R9 引入，R10 又加重了一些。** R9（`789d51d1`，把两个 wgrad 放进融合 kernel）之后，按 layer bench，GLM-4.7 的 bwd QKV（A2AGemm）在 n=8 慢了 8–17%，在 n=4 慢了 28–35%；bwd o_proj（GemmA2A）在 n=8 16K 慢了 15%，在 n=8 32K 和 n=4 16K 慢了 6%。R10（`5d3e68e6`，有 wgrad 时 push block 从 32 降到 16）在 bwd QKV 上再慢 2–4%。前向和 unfused 路径都没有变化（差异在 1% 以内）。
- **不只出现在 n=8。** n=4 的回退幅度更大，因为 n=4 时每个 rank 的 wgrad K 维更长，wgrad tile 更多。
- **根因是 launch 参数，不是 wgrad tile 本身慢。** GLM-4.7 的 wgrad tile 很多（QKV 1120 个，o_proj 960 个，MI355X 只有 256 个 CU）。现有默认参数是按 Qwen3-30B 这类 wgrad tile 少于一波的形状调的：
  - A2AGemm 先领 wgrad tile（`wgrad_at=0`），GLM 的 1120 个 wgrad tile 要占满 CU 四波多，dgrad GEMM tile 只能排在后面，等它们开始时数据交换早就结束了，GEMM 和交换变成串行；
  - BM 自动选择：一波放不下就选 128 行，GLM 的 dgrad tile 因此翻倍、单 tile 效率降低；
  - R10 的 16 个 push block 让交换本身更慢；
  - GemmA2A 把 wgrad tile 放在 GEMM 的前 1/8 之后，960 个 wgrad tile 把后面的 GEMM tile 推迟了几波，pull 也跟着推迟。
- **修复：按 wgrad tile 数切换 launch 参数，只影响 wgrad tile 很多的形状。**
  - A2AGemm：wgrad tile 超过 2 × CU 数（512）时，用 BM=256、32 个 push block，并把 wgrad tile 排在全部 dgrad tile 之后。目前只有 GLM-4.7 触发。
  - GemmA2A：wgrad tile 超过 CU 数（256）时，把 wgrad tile 排在全部 GEMM tile 之后。GLM-4.7 和 MiniMax-M2.5 触发。
- **修复效果：**
  - GLM 的 bwd QKV 从 R10 的 990 / 1702 / 1987 / 3850 µs（n8 16K / n8 32K / n4 16K / n4 32K）降到 718 / 1289 / 1474 / 2614 µs，比 old 还快；
  - layer bench 的 bwd 整层时间在 4 个配置下都比 old 快 0.4–2.3%；
  - Llama-3.1-8B、Qwen3-30B 的 bwd QKV 与 R10 完全相同，MiniMax-M2.5 的 bwd o_proj 快了 1–10%；
  - 正确性测试全部通过（n=4 完整，n=2/4/8 quick），并新增了一个 GLM 宽度的用例覆盖新路径。
  - 端到端（GLM-4.7 4 层 proxy，12 步）：CP8 fused 每步 R10 → 修复为 1375 → 1363 ms（16K）、2336 → 2312 ms（32K），与 old 的 1361 / 2320 ms 持平；CP4 比 R10 快 1.8–2.0%。Qwen3-30B CP4 与 R10 持平（±0.2%）。
- **遗留问题：** MiniMax-M2.5 的 bwd QKV 在 R9 之后也比 old 慢（n4 16K 632 → 743 µs，n4 32K 1160 → 1356 µs，n8 32K 587 → 670 µs），但扫描的 env 参数里没有一个能让它回到 old 的水平，只有关掉融合 wgrad 才行。这一点见第 8 节的建议。

## 2. 环境与方法

- 节点：`smci355-ccs-aus-n12-29`，8×MI355X（256 CU），全程独占（开跑前 `amd-smi metric -u` 确认无他人任务）。
- 软件：flydsl 0.3.4.1（`/home/xiaompen/pydeps/flydsl0341`），每份代码用独立的 `FLYDSL_RUNTIME_CACHE_DIR`；Primus `feat/ulysses-fused-a2a` @ `f9a298a49`。
- 三份 worktree：old = `183450cc`，R9 = `789d51d1`，R10 = `5d3e68e6`。`183450cc` → `789d51d1` 之间只多了一个文档提交，`5d3e68e6` → HEAD（`69991df7`）没有代码改动。
- A/B 按 A B C A B C 的顺序跑两轮，layer bench 和 overlap bench 都用 prompt 里给的 `uly_bench.py`。overlap bench 取各 rank 的最大值，layer bench 的单算子取各 rank 均值，整层总时间取最慢 rank。
- 本节点峰值：bf16 GEMM 1242 TFLOP/s，单向链路 321 GB/s（n=8）和 145 GB/s（n=4）。
- 不用 CUDA graph，不用多 stream。

## 3. 同节点 A/B

### 3.1 overlap bench（µs，fused，各 rank 最大值，第一轮 ; 第二轮）

| 模型 / n / seq | 算子 | old `183450cc` | R9 `789d51d1` | R10 `5d3e68e6` | R10 / old |
|---|---|---|---|---|---|
| GLM-4.7 n8 16K | bwd QKV | 849 ; 847 | 958 ; 956 | 990 ; 989 | +17% |
| | bwd o_proj | 573 ; 572 | 635 ; 632 | 630 ; 634 | +10% |
| GLM-4.7 n8 32K | bwd QKV | 1414 ; 1405 | 1648 ; 1657 | 1706 ; 1703 | +21% |
| | bwd o_proj | 1109 ; 1108 | 1154 ; 1157 | 1157 ; 1161 | +5% |
| GLM-4.7 n4 16K | bwd QKV | 1461 ; 1466 | 1905 ; 1910 | 1984 ; 1985 | +36% |
| | bwd o_proj | 1142 ; 1143 | 1198 ; 1189 | 1194 ; 1205 | +5% |
| GLM-4.7 n4 32K | bwd QKV | 2675 ; 2709 | 3713 ; 3694 | 3834 ; 3851 | +43% |
| | bwd o_proj | 2362 ; 2375 | 2301 ; 2299 | 2309 ; 2311 | −2% |
| Qwen3-30B n4 16K | bwd QKV | 376 | 324 | 300 | −20% |
| Qwen3-30B n4 32K | bwd QKV | 659 | 550 | 515 | −22% |

所有配置的 unfused 列在三份代码之间相差不到 1%。按判定标准（两轮都慢 5% 以上，且 unfused 相差不到 2%），GLM 在 n=8 和 n=4 都判定为回退。

### 3.2 layer bench（µs，unfused / fused，第一轮 ; 第二轮）

| 模型 / n / seq | 项目 | old | R9 | R10 |
|---|---|---|---|---|
| GLM-4.7 n8 16K | bwd QKV | 1005/899 ; 1002/894 | ~/974 ; ~/968 | 1007/1002 ; 1000/993 |
| | bwd o_proj | 812/654 ; 809/659 | ~/755 ; ~/760 | 804/764 ; 813/776 |
| | bwd 整层 | 6414/6178 ; 6411/6176 | ~/6337 ; ~/6337 | 6410/6377 ; 6428/6364 |
| GLM-4.7 n8 32K | bwd QKV | 1803/1441 ; 1802/1439 | ~/1692 ; ~/1673 | 1802/1728 ; 1798/1713 |
| | bwd 整层 | 16577/16091 ; 16578/16082 | ~/16417 | 16586/16436 ; 16588/16431 |
| GLM-4.7 n4 16K | bwd QKV | 2056/1502 ; 2040/1500 | ~/1928 | 2040/2003 ; 2049/1992 |
| | bwd 整层 | 12161/11280 ; 12146/11240 | ~/11788 | 12173/11821 ; 12148/11799 |
| GLM-4.7 n4 32K | bwd QKV | 3975/2723 ; 3954/2734 | ~/3676 | 3951/3810 ; 3976/3821 |
| | bwd 整层 | 33292/31636 ; 33233/31636 | ~/32614 | 33236/32694 ; 33272/32703 |
| Llama-3.1-8B n8 16K | bwd 整层 | 4639/4507 ; 4638/4501 | ~/4402 | 4646/4405 ; 4632/4463 |
| Llama-3.1-8B n8 32K | bwd 整层 | 8683/8468 ; 8703/8494 | ~/8471 | 8696/8482 ; 8701/8499 |
| Qwen3-30B n4 16K | bwd 整层 | 3226/2890 ; 3241/2895 | ~/2801 | 3259/2789 ; 3265/2799 |
| Qwen3-30B n4 32K | bwd 整层 | 8631/7933 ; 8630/7988 | ~/7805 | 8665/7783 ; 8656/7794 |

（R9 的 unfused 与另外两份相同，表中略为 `~`。）前向在三份代码之间没有变化，完整表格见附录的 `tables/ab_tables.txt`。

**回答 prompt 的三个问题：** 回退由 R9 引入，R10 在 bwd QKV 上再加 2–4%；涉及 bwd QKV（A2AGemm，主要部分）和 bwd o_proj（GemmA2A）两个算子；n=8 和 n=4 都有，n=4 的 QKV 更严重。

## 4. 拆因（R10 `5d3e68e6`，只改 env 参数，overlap bench，µs）

参数说明：`p32` = `ULY_QKV_PUSH=32`，`bm256` = `ULY_QKV_BM=256`，`atX` = `ULY_QKV_WGRAD_AT=X`，`oatX` = `ULY_O_WGRAD_AT=X`；`noqkv` / `noo` / `noboth` 是在 scratch worktree 里关掉融合 wgrad（回到 R9 之前的独立 hipBLASLt wgrad）。BM=256 时 dgrad tile 数为：n8 16K 160，n8 32K 320，n4 16K 320，n4 32K 640。

### 4.1 GLM-4.7 bwd QKV（A2AGemm）

| 配置 | n8 16K | n8 32K | n4 16K | n4 32K |
|---|---|---|---|---|
| 默认（BM128，push16，at=0） | 987 | 1709 | 1976 | 3826 |
| p32 | 956 | 1657 | 1908 | 3737 |
| bm256 | 878 | 1618 | 2079 | 3739 |
| at = BM128 的半数 / 全部 dgrad tile | 890 / 886 | 1641 / 1626 | 1738 / 1715 | 3253 / 3310 |
| bm256 + p32 | 790 | 1454 | 1763 | 3169 |
| bm256 + at = 全部 dgrad tile | 753 | 1347 | 1672 | 3063 |
| **bm256 + p32 + at = 全部 dgrad tile** | **720** | **1295** | **1481** | **2621** |
| noqkv（独立 hipBLASLt wgrad） | 836 | 1396 | 1461 | 2697 |
| 参考：old `183450cc` | 849 | 1414 | 1461 | 2675 |

三个参数单独调都只能找回一小部分，组合起来才有效：BM 选 256、32 个 push block、wgrad tile 排在全部 dgrad tile 之后。组合后在 n=8 比 wgrad-off 还快 7–14%，在 n=4 与 wgrad-off 持平或更快。

### 4.2 GLM-4.7 bwd o_proj（GemmA2A）

| 配置 | n8 16K | n8 32K | n4 16K | n4 32K |
|---|---|---|---|---|
| 默认（wgrad 在 GEMM 前 1/8 之后） | 634 | 1164 | 1201 | 2305 |
| oat256 / oat384 | 674 / 638 | 1171 / 1162 | 1203 / 1206 | 2312 / – |
| oat576 / oat768 | 621 / 592 | – / 1168 | – / 1203 | – / 2337 |
| oat = GEMM tile 数（n8 32K 1536，n4 16K 1536，n4 32K 3072） | – | 1118 | 1118 | 2222 |
| oat999999（排在最后） | 594 | 1122 | 1131 | 2229 |
| noo（独立 hipBLASLt wgrad） | 605 | 1091 | 1100 | 2207 |
| 参考：old | 573 | 1109 | 1142 | 2362 |

wgrad 位置只要还在 GEMM tile 中间就没有收益，排到全部 GEMM tile 之后才能找回来，与 wgrad-off 的差距缩到 1–3%。

### 4.3 对照模型（同一组参数，不能变差）

| 模型 / 形状 | 默认 | p32 | bm256 + p32 | bm256 + at = dgrad | bm256 + p32 + at = dgrad | noqkv |
|---|---|---|---|---|---|---|
| Llama-3.1-8B n8 16K（wgrad tile 384） | 318 | 310 | 302 | 281 | 278 | 366 |
| Llama-3.1-8B n8 32K | 574 | 548 | 468 | 517 | 503 | 555 |
| Qwen3-30B n4 16K（wgrad tile 160） | **303** | 323 | 334 | 357 | 406 | 370 |
| Qwen3-30B n4 32K | **515** | 551 | 544 | 697 | 749 | 662 |
| MiniMax-M2.5 n8 16K（wgrad tile 384） | 378 | 357 | 393 | – | 431 | 421 |
| MiniMax-M2.5 n8 32K | 663 | 616 | 700 | – | 810 | 588 |
| MiniMax-M2.5 n4 16K | 745 | 767 | 876 | – | 956 | 634 |
| MiniMax-M2.5 n4 32K | 1352 | 1399 | 1510 | – | 1843 | 1160 |

Qwen3-30B 用 GLM 的最优参数会慢 34–45%，所以不能全局改默认值。MiniMax-M2.5 没有参与调参，用来检验门槛。它的 wgrad tile 数（384）与 Llama 相同，但套用 GLM 参数会变慢，所以 A2AGemm 的门槛定在 2 × CU（512），只有 GLM 的 1120 会触发。

Qwen3-30B 的 bwd o_proj：默认 232 / 429 µs（16K / 32K），排在最后是 262 / 494 µs，关掉融合 wgrad 是 305 / 530 µs，所以 GemmA2A 的门槛定在 CU 数（256）。Qwen 的 128 个 wgrad tile 不触发，GLM（960）和 MiniMax（288）触发。

## 5. wgrad GEMM 单独对比：FlyDSL variable-K tile 与 hipBLASLt

单卡、一个 group（`offs=[0, K]`），`grouped_gemm_bf16_variable_k_flydsl_kernel` 与 `torch.mm(a.t(), b)`（hipBLASLt）比较，µs：

| 形状（dW） | K | tile 数 | FlyDSL | hipBLASLt | FlyDSL / hipBLASLt |
|---|---|---|---|---|---|
| GLM dW_qkv [14336, 5120] | 2048 / 4096 / 8192 | 1120 | 302 / 541 / 1037 | 285 / 499 / 938 | 1.06 / 1.08 / 1.11 |
| GLM dW_o [5120, 12288] | 2048 / 4096 / 8192 | 960 | 256 / 442 / 843 | 234 / 428 / 825 | 1.09 / 1.03 / 1.02 |
| Qwen3-30B dW_qkv [5120, 2048] | 2048 / 4096 / 8192 | 160 | 66 / 107 / 188 | 58 / 91 / 158 | 1.12 / 1.17 / 1.19 |
| Qwen3-30B dW_o [2048, 4096] | 2048 / 4096 / 8192 | 128 | 64 / 103 / 181 | 50 / 81 / 141 | 1.28 / 1.28 / 1.28 |
| Llama-8B dW_qkv [6144, 4096] | 2048 / 4096 / 8192 | 384 | 124 / 219 / 396 | 106 / 179 / 342 | 1.16 / 1.22 / 1.16 |
| Llama-8B dW_o [4096, 4096] | 2048 / 4096 / 8192 | 256 | 77 / 129 / 232 | 71 / 122 / 225 | 1.08 / 1.05 / 1.03 |
| MiniMax dW_qkv [8192, 3072] | 2048 / 4096 / 8192 | 384 | 124 / 214 / 399 | 112 / 191 / 357 | 1.11 / 1.12 / 1.12 |
| MiniMax dW_o [3072, 6144] | 2048 / 4096 / 8192 | 288 | 120 / 205 / 382 | 92 / 155 / 282 | 1.29 / 1.33 / 1.36 |

结果与 fp32 参考的相对误差都在 5e-5 以内。FlyDSL 的 tile 比 hipBLASLt 慢 2–36%，GLM 的形状差距最小（2–11%）。所以 GLM 的回退主要不是 tile 慢造成的；第 4 节里，同样的 wgrad tile 换个顺序就比 hipBLASLt 方案快。MiniMax dW_o 和 Qwen dW_o 差距最大，这也是 MiniMax bwd QKV 遗留问题的一个来源。

## 6. 根因

融合 kernel 里，CU 按 claim 顺序领取 tile：先是 push / pull 交换 block，然后按 `wgrad_at` 把 wgrad tile 插进 GEMM tile 序列。wgrad tile 不依赖交换的数据，用来填补 GEMM tile 等数据时空闲的 CU。

- **A2AGemm（bwd QKV）：** R9 默认 `wgrad_at=0`，即先领 wgrad tile。Qwen3-30B 只有 160 个 wgrad tile，不到一波，正好填满等数据的时间，所以 R9/R10 让 Qwen 快了 20%。GLM-4.7 有 1120 个 wgrad tile，占满 256 个 CU 四波多，dgrad GEMM tile 要等它们全部领完才开始，此时数据交换已经结束，GEMM 和交换串行。BM 自动选择（`fits` 不成立就选 128）进一步把 GLM 的 dgrad tile 数翻倍，R10 又把 push block 从 32 降到 16，交换更慢。
- **GemmA2A（bwd o_proj）：** wgrad tile 插在 GEMM 前 1/8 之后。GLM 的 960 个 wgrad tile 把剩下 7/8 的 GEMM tile 推迟几波，而 pull 要等 GEMM tile 写完才能开始，交换尾部变长。

wgrad tile 少于一波时，原来的顺序是对的；超过几波以后，wgrad 应该排在最后，让 GEMM 和交换先重叠完，wgrad tile 去填尾部。

## 7. 修复与验证

### 7.1 修改

- `primus_turbo/flydsl/ulysses/a2a_gemm_kernel.py`（`ulysses_a2a_gemm`）：`wgrad_at` 默认值从 0 改为 `None`（按形状选择）。wgrad tile 数大于 2 × CU 数时，`BM=256`、`num_push=32`、`wgrad_at` = dgrad tile 数；否则行为与 R10 相同。
- `primus_turbo/flydsl/ulysses/gemm_a2a_kernel.py`（`ulysses_gemm_a2a`）：wgrad tile 数大于 CU 数时，`wgrad_at` = GEMM tile 数（排在最后）；否则行为与 R10 相同。
- `tests/pytorch/ops/test_ulysses_fused_a2a.py`：新增 GLM 宽度用例（hq 96、hkv 8、d 128、hidden 5120、s/n 512），QKV 1120 个、o_proj 960 个 wgrad tile，覆盖两条新路径。原有用例的 hidden 是 1024，wgrad tile 最多 288 个，触发不了新规则。
- 显式传入的 env 参数（`ULY_QKV_*`、`ULY_O_WGRAD_AT`）仍然优先。

### 7.2 overlap bench（µs，fused，各 rank 最大值）

| 模型 / n / seq | 算子 | old | R10 | 修复 | 修复 / old |
|---|---|---|---|---|---|
| GLM-4.7 n8 16K | bwd QKV | 845 | 990 | **718** | −15% |
| | bwd o_proj | 573 | 633 | 600 | +5% |
| GLM-4.7 n8 32K | bwd QKV | 1410 | 1702 | **1289** | −9% |
| | bwd o_proj | 1111 | 1159 | 1123 | +1% |
| GLM-4.7 n4 16K | bwd QKV | 1461 | 1987 | **1474** | +1% |
| | bwd o_proj | 1144 | 1198 | 1134 | −1% |
| GLM-4.7 n4 32K | bwd QKV | 2669 | 3850 | **2614** | −2% |
| | bwd o_proj | 2380 | 2312 | 2230 | −6% |
| Llama-3.1-8B n8 16K | bwd QKV / bwd o_proj | 371 / 188 | 321 / 195 | 319 / 195 | 不变 |
| Llama-3.1-8B n8 32K | bwd QKV / bwd o_proj | 552 / 337 | 576 / 362 | 576 / 362 | 不变 |
| Qwen3-30B n4 16K | bwd QKV / bwd o_proj | 374 / 245 | 302 / 229 | 302 / 237 | 不变（o_proj 在噪声内） |
| Qwen3-30B n4 32K | bwd QKV / bwd o_proj | 658 / 437 | 515 / 420 | 514 / 426 | 不变 |
| MiniMax-M2.5 n8 16K | bwd QKV / bwd o_proj | 422 / 238 | 376 / 206 | 378 / **186** | o_proj −10% |
| MiniMax-M2.5 n8 32K | bwd QKV / bwd o_proj | 587 / 405 | 668 / 388 | 670 / 385 | 不变 |
| MiniMax-M2.5 n4 16K | bwd QKV / bwd o_proj | 632 / 423 | 746 / 423 | 743 / **390** | o_proj −8% |
| MiniMax-M2.5 n4 32K | bwd QKV / bwd o_proj | 1160 / 817 | 1356 / 787 | 1356 / **751** | o_proj −5% |

所有配置的 unfused 列与 old / R10 相差不到 1%。GLM n8 16K 的 bwd o_proj 仍比 old 高 5%（关掉融合 wgrad 也是 605 µs，说明这一部分不是 wgrad 顺序造成的），但同一配置下 bwd QKV 快了 127 µs，足以抵消。

第一版门槛曾把 A2AGemm 的门槛设在 CU 数（256），MiniMax 的 bwd QKV 因此变慢（n8 16K 376 → 432，n4 16K 746 → 958），所以改成 2 × CU。

### 7.3 layer bench（µs，unfused / fused，两轮）

| 模型 / n / seq | 项目 | old | R10 | 修复 |
|---|---|---|---|---|
| GLM-4.7 n8 16K | bwd QKV | 1005/899 ; 1002/894 | 1007/1002 ; 1000/993 | 1000/799 ; 1002/806 |
| | bwd o_proj | 812/654 ; 809/659 | 804/764 ; 813/776 | 810/609 ; 807/607 |
| | **bwd 整层** | 6414/**6178** ; 6411/**6176** | 6410/6377 ; 6428/6364 | 6414/**6030** ; 6407/**6038** |
| GLM-4.7 n8 32K | bwd QKV | 1803/1441 ; 1802/1439 | 1802/1728 ; 1798/1713 | 1799/1495 ; 1794/1503 |
| | bwd o_proj | 1504/1329 ; 1505/1337 | 1532/1399 ; 1537/1409 | 1530/1143 ; 1529/1140 |
| | **bwd 整层** | 16577/**16091** ; 16578/**16082** | 16586/16436 ; 16588/16431 | 16601/**15946** ; 16588/**15953** |
| GLM-4.7 n4 16K | bwd QKV | 2056/1502 ; 2040/1500 | 2040/2003 ; 2049/1992 | 2061/1610 ; 2057/1625 |
| | bwd o_proj | 1743/1343 ; 1745/1321 | 1757/1397 ; 1741/1380 | 1743/1147 ; 1718/1155 |
| | **bwd 整层** | 12161/**11280** ; 12146/**11240** | 12173/11821 ; 12148/11799 | 12177/**11202** ; 12157/**11224** |
| GLM-4.7 n4 32K | bwd QKV | 3975/2723 ; 3954/2734 | 3951/3810 ; 3976/3821 | 3982/2958 ; 3980/2983 |
| | bwd o_proj | 3328/2792 ; 3304/2785 | 3318/2772 ; 3299/2805 | 3331/2322 ; 3320/2322 |
| | **bwd 整层** | 33292/**31636** ; 33233/**31636** | 33236/32694 ; 33272/32703 | 33288/**31399** ; 33262/**31406** |
| Llama-3.1-8B n8 16K | bwd 整层 | 4639/4507 ; 4638/4501 | 4646/4405 ; 4632/4463 | 4636/4390 ; 4649/4439 |
| Llama-3.1-8B n8 32K | bwd 整层 | 8683/8468 ; 8703/8494 | 8696/8482 ; 8701/8499 | 8686/8456 ; 8732/8508 |
| Qwen3-30B n4 16K | bwd 整层 | 3226/2890 ; 3241/2895 | 3259/2789 ; 3265/2799 | 3239/2790 ; 3228/2786 |
| Qwen3-30B n4 32K | bwd 整层 | 8631/7933 ; 8630/7988 | 8665/7783 ; 8656/7794 | 8626/7792 ; 8652/7788 |

修复后 GLM 的 bwd 整层时间比 R10 快 2.9–5.3%，比 old 快 0.4–2.3%。Llama、Qwen 与 R10 持平，前向不变。layer bench 里单看 bwd QKV，修复版在 n8 32K 和 n=4 下仍比 old 高 4–9%（layer bench 的 GEMM 和交换与 attention 交替，比 overlap bench 更挤），但 bwd o_proj 比 old 快 7–17%，整层是净收益。

### 7.4 正确性

在修复版 worktree 上运行 `tests/pytorch/ops/test_ulysses_fused_a2a.py`（包含新增的 GLM 宽度用例）：

| 规模 | 模式 | 结果 | 耗时 |
|---|---|---|---|
| n=4 | 完整 | all cases passed | 29 min |
| n=2 | `--quick` | all cases passed | 3 min |
| n=4 | `--quick` | all cases passed | 3 min |
| n=8 | `--quick` | all cases passed | 3 min |

### 7.5 端到端

Primus `f9a298a49`，GLM-4.7 4 层 proxy（`glm4_7_4layer_proxy`），EP8，GBS 16，12 步。每步耗时取第 4–12 步的中位数（ms），括号里是 fused 相对同一份代码 base 的变化。

| 配置 | 实现 | old `183450cc` | R10 `5d3e68e6` | 修复 | 修复 / R10 | 修复 / old |
|---|---|---|---|---|---|---|
| GLM CP8 16K | base | 1386.9 | 1395.7 | 1390.4 | | |
| | fused | 1360.9（−1.9%） | 1375.3（−1.5%） | **1363.3**（−1.9%） | −0.9% | +0.2% |
| GLM CP8 32K | base | 2406.0 | 2406.5 | 2401.8 | | |
| | fused | 2319.8（−3.6%） | 2336.1（−2.9%） | **2312.2**（−3.7%） | −1.0% | −0.3% |
| GLM CP4 16K | base | 1080.8 | 1079.1 | 1080.7 | | |
| | fused | 1019.6（−5.7%） | 1028.7（−4.7%） | **1008.0**（−6.7%） | −2.0% | −1.1% |
| GLM CP4 32K | base | 2129.0 | 2127.9 | 2132.6 | | |
| | fused（两次） | 1997.4 ; 1996.2（−6.2%） | 2037.6 ; 2033.2（−4.3%） | **2000.5 ; 2000.7**（−6.2%） | −1.8% | +0.2% |

Qwen3-30B-A3B（完整 48 层）CP4 EP8，确认修复没有让它变慢：

| 配置 | base（修复） | fused R10 | fused 修复 | 修复 / R10 |
|---|---|---|---|---|
| 16K | 3157.1 | 2821.5（−10.6%） | 2814.6（−10.8%） | −0.2% |
| 32K | 6581.1 | 5973.2（−9.2%） | 5978.6（−9.2%） | +0.1%（噪声内） |

检查项：

- 每个 fused 日志都有 `[Patch] ✓ Applied: megatron.ulysses_fused_a2a`，没有 `keeping the stock path`；base 日志里该补丁都是 `Skipped`。
- 每个日志的 `[pyboot] sys.path head` 都指向对应的 worktree（`turbo_183450cc` / `turbo_5d3e68e6` / `turbo_fix`）。
- 第 12 步 loss：同一配置下三份代码一致（GLM 相差不超过 1e-5，fused 与 base 相差不超过 2e-4）；Qwen3-30B 的 fused R10 与 fused 修复相差 5e-4，与 base 相差 2e-3，在 MoE 路由不确定性的范围内。
- GLM CP4 32K fused R10 第一次运行有两步异常（3448 / 3409 ms，其余 2037 ms 左右），三份代码都重跑了一次，中位数与第一次一致。

GLM CP8 上，R10 的 fused 比 old 慢 0.7–1.1%（整步里 attention 投影只占一小部分，每层 bwd 回退约 200–350 µs）；修复后与 old 持平（16K +0.2%，32K −0.3%），比 R10 快 0.9–1.0%。GLM CP4 修复后比 R10 快 1.8–2.0%，与 old 持平或更快。

## 8. 遗留问题与建议

1. **MiniMax-M2.5 的 bwd QKV 仍比 old 慢**（n8 32K +14%，n4 16K +18%，n4 32K +17%；n8 16K 反而快 10%）。扫描的 env 参数都找不回来，只有关掉融合 wgrad（回到独立 hipBLASLt）才能回到 old 的水平。原因是 MiniMax 的 wgrad tile（384）介于一波和两波之间，FlyDSL 的 dW tile 又比 hipBLASLt 慢 11–36%，放进 kernel 节省的重叠时间抵不过 tile 本身变慢。建议：
   - 按形状选择融合 wgrad 还是独立 hipBLASLt wgrad（例如 wgrad tile 在 1–2 波之间时退回独立 wgrad），需要 MiniMax E2E 确认收益；
   - 或者做一次 autotune，各 rank 的计时取 all-reduce 后的最大值来选配置（所有 rank 必须选同一套参数，因为 flag 布局依赖 BM）；
   - 长期：提升 FlyDSL variable-K tile 在窄 N（2048–6144）上的效率。
2. **Llama-3.1-8B 还有空间：** BM=256 + 32 个 push block 在 n8 下 bwd QKV 能再快 5–18%（16K 318 → 302，32K 574 → 468），但 Llama 的 wgrad tile（384）与 MiniMax 相同，而 MiniMax 用这组参数会变慢，单靠 wgrad tile 数区分不开。这次没有改，需要更多形状特征（如 dgrad tile 数与 wgrad tile 数之比）再定。
3. **GLM n8 16K 的 bwd o_proj 比 old 高 5%**，关掉融合 wgrad 也一样，问题在 R9 对 GemmA2A 的其他改动（如 flag / tile 布局），可以单独排查。

## 附录

- 节点：`smci355-ccs-aus-n12-29`（8×MI355X，256 CU），容器 `xiaompen_a2a_bench`。
- 代码：Primus-Turbo old `183450cc`、R9 `789d51d1`、R10 `5d3e68e6`，修复基于 `69991df7`（与 `5d3e68e6` 代码相同）；Primus `feat/ulysses-fused-a2a` @ `f9a298a49`；flydsl 0.3.4.1。
- 日志都在 `/home/xiaompen/uly_reg/`：
  - A/B：`logs/ab_r{1,2}_<commit>_{L8,L4,O8g,O4g,O4q}.log`，汇总表 `tables/ab_tables.txt`；
  - 拆因扫描：`logs/sw_<tag>_<cfg>.log`，汇总表 `tables/sweep_glm.txt`（脚本 `sweep.sh`、`sweep2.sh`、`sweep3.sh`，解析 `parse_sw.py`）；
  - wgrad 单独对比：`logs/wgrad_bench.log`（脚本 `wgrad_bench.py`）；
  - 修复验证：`logs/val_{183450cc,5d3e68e6}_*.log`、`logs/val2_fix_*.log`、`logs/ab_r{1,2}_fix2_{L8,L4}.log`（第一版门槛是 `logs/val_fix_*.log`、`logs/ab_r{1,2}_fix_*.log`）；
  - 正确性：`logs/test_n{4_full,2_quick,4_quick,8_quick}.log`；
  - 端到端：`e2e/<tag>.log`（脚本 `e2e/run_e2e.sh`、`e2e/e2e_all.sh`、`e2e/e2e_qwen.sh`，解析 `e2e/parse_e2e.py`）。
- 训练进程加载哪份 `primus_turbo`：Primus 的 `base_env.sh` 会重排 PYTHONPATH，所以每份代码用一个 `e2e/pyboot_<commit>/sitecustomize.py` 把对应 worktree 和 flydsl 0.3.4.1 放到 `sys.path` 最前面，rank 0 打印 `[pyboot] sys.path head`，每个日志都已核对。
