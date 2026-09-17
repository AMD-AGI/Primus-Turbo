# 反向移植成本：模板的「免费操作数」技巧，dq 成立、dkdv 不成立

日期 2026-09-17 · 实测 · 脚本 `kernels/accumulator_operand_probe.py`、`kernels/dq_operand_probe.py`

## 结论

**`SURVEY.md` §10.2 里那句"`Pᵀ`/`dSᵀ` 从累加器里免费得到"是 gfx942/MFMA 的性质，不是 gfx1250/WMMA 的。**
在 gfx1250 上它**不成立**，dkdv 需要为 `Pᵀ` 和 `dSᵀ` 多做一次 LDS 往返或跨 lane 转置 ——
而 dkdv 是反向里最贵的那个内核。

这条在动手写 700 行之前查出来，代价是一个 80 行的探针。

## 为什么 gfx942 上是免费的

gfx942 用 `v_mfma_f32_16x16x16_bf16_1k`，累加器和操作数**每 lane 都是 4 个值、映射相同**。
所以模板里 `_pack4_trunc(vals)` —— "4 fp32 -> one v4bf16 MFMA operand fragment" ——
是一个**纯 lane 内**的位操作，零跨 lane 移动。这正是那句"免费"的来源。

## 为什么 gfx1250 上不成立

gfx1250 用 `v_wmma_f32_16x16x32_bf16`（wave32）。两个布局（均已对 torch 验证，
见 `kernels/wmma_layout_probe.py`）：

```
累加器      lane l 持有 C[(l//16)*8 + si, l%16]，si = 0..7
            -> 一列的八行
A 操作数    lane l 持有 A[l%16, (l//16)*8 + (e%8) + (e//8)*16]，e = 0..15
            -> 一行的十六列
```

**这是可证的，不只是实测**：要让 lane `l` 拿到 C 的第 `l%16` **行**，
那一行的 16 个值分散在全部 32 个 lane 上。**不在本 lane 里的数据，lane 内变不出来。**
必须跨 lane 移动。

### 实测确认

探针做 `C1 = A @ B1`，然后把 C1 的累加器**原地**打包成 bf16 当作下一次 WMMA 的 A 操作数
（8 个值填 16 槽，重复填满 —— 这是对"免费"最宽容的读法），与真实的 `C1 @ B2` 比：

| | SQNR |
|---|--:|
| 第一级（健全性检查） | **150.34 dB** |
| 第二级 vs "免费"读法 | **−4.28 dB** |

第一级通过说明探针本身是对的，第二级 −4.28 dB 说明那条路不通。

## 对计划的影响

`PLAN.md` Stage 3 把 dkdv 的工作描述为"把 `MFMA(16,16,16)` 换成 `WMMA(16,16,32)`，
并相应调整 k 循环步长与 LDS staging"。**这个描述现在是不完整的。** 还要加上：

- **`Pᵀ` 和 `dSᵀ` 各需要一次 accumulator → operand 的转置**，在 kv×q 的每个 tile 上，
  每个 k 循环迭代一次。gfx942 那份是零成本的地方，这里要么走 LDS（写 fp32/bf16 出去再按
  operand 布局读回来，加 barrier），要么用跨 lane 原语（gfx1250 有
  `ds_load_tr16_b128`，前向用了 12 处，正是为转置读准备的）。
- **LDS 预算和 barrier 数都要重算。** gfx942 那份的 LDS arena 是按"只 stage `dOᵀ` 和 `Qᵀ`"
  算的；现在还要为 `Pᵀ`/`dSᵀ` 的中转留空间，或者把它们挤进同一块 arena 并加同步。
- **`dq` job 不受影响 —— 已实测确认，见下一节。**

## dq 那一路是免费的（实测 151.32 dB）

`kernels/dq_operand_probe.py`。模板对 dq 的说法和对 dkdv 的不同：它把 score 算成**转置的**
（`Sᵀ = K·Qᵀ`），于是 `dSᵀ` 片段免费就是 `dQ = dS·K` 需要的 `dS` 操作数。
在 gfx1250 上**这一条成立**，理由清楚：

```
kv-tile j 的累加器    lane l 持有 S^T[j*16 + (l//16)*8 + si, l%16]，si = 0..7
dS 的 A 操作数        lane l 需要 dS[l%16, (l//16)*8 + (e%8) + (e//8)*16]
```

`dS[q, kv] == S^T[kv, q]`，而 **`l%16` 两边都是 `q`** —— lane 已经拿着对的那个 q。
lane 内：`e = 0..7` 要 `kv = half*8 + 0..7`，正好是 tile `j` 的八个值；
`e = 8..15` 要 `kv = half*8 + 16..23`，正好是 tile `j+1` 的八个。
**两个相邻 kv-tile 的累加器在 lane 内直接拼接成一个 v16 操作数。**

| | SQNR |
|---|--:|
| `S^T` tile 0（健全性） | 149.25 dB |
| `S^T` tile 1（健全性） | 148.43 dB |
| **`dQ`，用 lane 内拼接的操作数** | **151.32 dB** |

对照用的 `dS` 同样截断到 bf16，把布局问题和 bf16 舍入分开。

**所以额外成本精确地只落在 dkdv 一个 job 上。** 两者方向不同：dkdv 沿 query 收缩，
`P^T` 的 `q` 落在累加器的 N 轴（`l%16`）而操作数要它在 M 轴；dq 沿 kv 收缩，
两边的 `q` 都在 `l%16` 上。**不能从一个推另一个** —— 这也是为什么两个都要单独验。

## dkdv 的转置怎么做：机制存在，而且两半都是现成惯用法

不用另发明。前向里已经有这条路的两半（`fmha_b16_buffer_managers.py` 的类清单）：

| 需要的动作 | 前向里的现成实现 |
|---|---|
| WMMA 累加器 → LDS | `OManager16bV1`："WMMA accumulator -> swizzled LDS -> coalesced buffer_store" |
| 按转置布局从 LDS 读回 | `VManager16bV1/V2`：`rocdl.ds_load_tr16_b128(v8_ty, ptr)`，返回 v8 bf16 |

所以 dkdv 里 `P^T` / `dS^T` 每个 tile、每次 k 迭代要多付的是：

1. 累加器 fp32 → bf16，**lane 内**（前向的 epilogue 就这么做）；
2. 按 O 风格的 XOR swizzle 存进 LDS；
3. 一次 barrier；
4. 两次 `ds_load_tr16_b128` 拼成一个 v16 操作数。

**swizzle 是 bank 冲突优化，不是正确性要求。** 前向的注释写明：V block 按 32(kv)×d 子块堆放，
每块切成 32×32 tile、再切成 4(kv)×16(d) 子 tile，子 tile 列号按行号 `& 1` 做 XOR
——"to make the transpose load (`ds_load_tr16_b128`) bank-conflict-free"。
先不 swizzle 也能对，只是慢。

**还有一条硬要求**（前向注释里用 55% NaN 的代价换来的）：LDS 读**必须**用
plain intrinsic，**绝不能用不透明的 inline asm** —— 否则 LLVM 看不见它与异步
global→LDS 写之间的 RAW 依赖，在 `DEP_MODE=2` 下会错序，表现为
**16384 causal 下 55% 的静默 NaN**。写 dkdv 时照抄这一条，不要自己发明内存 op 封装。

### `ds_load_tr16_b128` 的语义（已实测反解）

`kernels/tr16_semantics_probe.py`：往 16×16 的 LDS tile 填 `value = row*16 + col`
（0..255，bf16 精确可表示），用 aiter 文档化的那个 lane 寻址
（`kv = (l//16)*8 + l%8`、`d = ((l//8)%2)*8`）读回来，反解出：

```
ds_load_tr16_b128  ->  lane l, 元素 e  得到  src[row = (l//16)*8 + e, col = l%16]
```

**每个 lane 拿到的是源 tile 的一整列的八行**（32/32 个 lane 都如此，0/32 是同一行）。

### 于是 dkdv 的转置路径是（已端到端验证 148.00 dB）

`kernels/tr16_operand_e2e.py`。把 `P` 按 **`[q][kv]` 行主序**存进 LDS，然后：

| 操作数要的 | tr16 给的 |
|---|---|
| lane l, e → `A[l%16, (l//16)*8 + (e%8) + (e//8)*16]` | 偏移 0 读：`P[q=(l//16)*8+e, kv=l%16]` = `A[kv=l%16, q=(l//16)*8+e]`，对上 `e=0..7` |
| | 再在 **+16 行**处读一次，对上 `e=8..15` |

**而存这一侧是便宜的**：累加器交给 lane 的八个值，对固定的 `q` 在 `kv` 上是连续的，
所以按 `P[q][kv]` 行主序写是**一次 b128 存**，不是八次 strided 存。

**每个 16×32 操作数的总成本：一次 b128 存 + 一次 barrier + 两次 `ds_load_tr16_b128`。**
这和 `VManager16bV2.load_v_to_reg` 的 `(dt, kt, half)` 循环是同一个形状 —— 前向已经在用了。

| 端到端验证 | |
|---|--:|
| isfinite | 256/256 |
| SQNR vs `P.T @ B` | **148.00 dB** |
| max abs error | 9.54e-07 |

**dkdv 写之前的未知量到此清零。**

## 与已有证据合起来看

到今天为止，FlyDSL gfx1250 反向这件事的账目是：

| | |
|---|---|
| 前向惯用法竞争力 | **证伪**：比 ASM 慢 1.51×（`STAGE1-FWD.md`） |
| 反向结构模板 | 存在；**dq 的免费操作数可移植（151.32 dB 实测）**，**dkdv 的不可移植**（本文） |
| 版本可共存 | **否**：0.3.2 与 turbo 的 0.2.4 树装不进一个进程 |
| 工具链可行性 | **是**：`odo` 一次通过 159 dB，WMMA 布局已验证，累加链已验证 |
| 剩下的立项理由 | gfx1250 **没有 varlen ASM 反向**（有无问题）；ASM 反向的 GQA 越界写（已量化：峰值 1.254 GiB + 反向 5.2%）；ASM 资产缺口 1/25 |

**建议**：在投入 dkdv 之前，把这三条摆给决策人——前两条是新增的负面证据，第三条没变。
`PLAN.md` 的 Stage 2 决策门原本只考虑前向速度，现在应当把"模板技巧不可移植"也计入。
