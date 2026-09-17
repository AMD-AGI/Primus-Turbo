# FlyDSL attention on gfx1250：调研报告

日期 2026-09-17 · 分支 `dev/lhz/attn`（HEAD 已合入 `origin/main` 13 条提交）· **全程只读，未使用 GPU**

标注：**实测** = 本次直接验证 / **文档记载** = 来自 commit、注释或 in-repo 文档 / **推算** / **未知**

---

## 0. 结论先行

**不要移植 Primus-Turbo 那份 gfx950 的 FlyDSL attention。**
它依赖的三个 CDNA4 原语在 gfx1250 上**三中三 "Cannot select"**，`warp_size = 64` 是硬编码的结构常量，
移植等于重写调度表。0914 的 LLVM 探针已经给出 18–36 工程日的估价，那个估价成立。

**但方向不是"FlyDSL 在 gfx1250 上不可行"** —— 恰恰相反：

| | gfx1250 的 FlyDSL 实现 | 位置 | 状态 |
|---|---|---|---|
| **前向** | **已存在，且是 aiter 在 gfx1250 上的默认路径** | `aiter/ops/flydsl/kernels/fmha_gfx1250/`（3 文件 4176 行） | 现成 |
| **反向** | **aiter 与 turbo 里都不存在** | — | 全新开发 |

而我们**缺的正是反向**（前向已经是全表最快）。所以这件事的真实形状是：
**用 aiter 那份 gfx1250 前向做模板，新写一个 gfx1250 FlyDSL 反向** ——
不是移植 CDNA4 的那份。

---

## 1. 四条前提纠正（都是承重的，且其中一条是我自己文档的错）

调研开始时我给出的背景有三处与仓库实际不符，另有一处是我 0916 写的文档有误：

| 说法 | 实际 | 证据 |
|---|---|---|
| gfx1250 FlyDSL GEMM「已部分进入 main」 | **完全没有**。main 的 `primus_turbo/flydsl/gemm/` 只有 `__init__` / bf16 / fp8 / mxfp4 / mxfp8 五个文件 | `git ls-tree origin/main -- primus_turbo/flydsl/gemm/`。**实测** |
| 「FlyDSL 编译器支持 gfx1250 WMMA」 | 成立但要说准：`primus_turbo/flydsl/` 整棵树里 **`gfx1250` 出现 0 次** | `grep -rc gfx1250 primus_turbo/flydsl/` 无输出。**实测** |
| 障碍是 `ds_read_tr16_b64` 的寄存器布局差异 | **障碍大一圈**，见 §2 | `output/0914__campaign/RESULTS.md:173-184`。**实测**（LLVM 探针） |
| 我在 `BACKEND-STRATEGY.md` 写「反向五个 kernel 各断言一次」 | **是六个**，漏了 **dkdv**（`:1343`），而它是 2777 行、最贵的那个 | `grep -c "targets gfx950" flash_attn_bwd.py` = 6。**实测**，原文档已更正 |

---

## 2. 真正的障碍：不是换指令，是重写调度表

`output/0914__campaign/RESULTS.md:175-179` 的容器内 LLVM 探针，**实测**：

| intrinsic | gfx950 | gfx1250 |
|---|---|---|
| `llvm.amdgcn.mfma.f32.32x32x16.bf16` | 选中 `v_mfma_f32_32x32x16_bf16` | **Cannot select** |
| `llvm.amdgcn.ds.read.tr16.b64` | 选中 `ds_read_b64_tr_b16` | **Cannot select** |
| `llvm.amdgcn.permlane32.swap` | 选中 `v_permlane32_swap_b32_e64` | **Cannot select** |

**gfx1250 根本没有 MFMA。** 而 turbo 那份 attention 的数学、累加器宽度、
以及每一个手推的 LDS→VGPR 立即数 stride，全部表达在 **MFMA 32x32x16 wave64 的 fragment 布局**上；
`warp_size = 64` 硬编码在 `primus_turbo/flydsl/utils/attn_helper.py:382`，
并往下传进 `BLOCK_SIZE` / `NUM_WAVES` / LDS line stride / DMA split ——
**它不是一个可调旋钮**。

七处 arch gate（**实测**逐一确认）：

```
flash_attn_fwd.py:72-74   raise "... requires gfx950+ (uses ds_read_tr16_b64) ..."
flash_attn_bwd.py:296     assert "odo kernel targets gfx950"
                 :492     assert "lse transpose kernel targets gfx950"
                 :683     assert "dq reduce kernel targets gfx950"
                 :1079    assert "slot reduce kernel targets gfx950"
                 :1161    assert "a16 un-permute kernel targets gfx950"
                 :1343    assert "bwd dkdv kernel targets gfx950"      ← 2777 行，最贵
```

另外 `sparse_mla_fwd.py` / `sparse_mla_bwd.py` **完全没有 arch gate**，
却直接用 `ds_read_tr16_b64` / `mfma` / `permlane16/32_swap` ——
在 gfx1250 上导入它们会是**编译器报错而不是干净拒绝**。**实测**

---

## 3. 转折：aiter 已经有一份 gfx1250 的 FlyDSL 前向

`aiter/ops/flydsl/kernels/fmha_gfx1250/`，3 个文件 **4176 行**（对照 `fmha_gfx950/` 是 8 文件 2600 行）。

`fmha_fwd_prefill_a16w16_m32x8.py` 的模块 docstring（**实测**原文）：

> MHA Forward Prefill kernel — `m32x8` design, **gfx1250 (MI400 / mi450)**.
> `m32x8` names the threadgroup shape: **8 waves per threadgroup**, each wave owning a **32-row** Q span.
> **gfx1250 runs wave32**, so a threadgroup is `8 * 32 = 256` threads and `BLOCK_M = 256`.
> Scope — `qk_hdim in {128, 192, 256}`, `v_hdim == 128`, `n_block == 64`；dtype bf16；
> **grouped-query attention (GQA)**；**causal and non-causal**。

**我们的形状（b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal）完全在覆盖范围内。**

而且它是**默认路径**，不是实验分支 —— `aiter/ops/flydsl/fmha_kernels.py:381` 原文：

> Routing (D_v=128, bf16, gfx1250): **our m32x8 kernel is the DEFAULT for qk_hdim 128 and 192.**

派发门（`:393` 起）：`get_gfx() == "gfx1250"` and `qk_hdim in (128,192)` and `v.shape[-1] == 128`
and dtype 为 bf16/fp16。入口函数 `flydsl_flash_attn_func`（`:91`）、
`flydsl_flash_attn_varlen_func`（`:305`）、`flydsl_flash_attn_batch_func`（`:435`）。

### 它用的原语与 turbo 那份完全不同

`grep -o` 统计（**实测**）：

```
12  rocdl.s_wait_dscnt          9  rocdl.sched_barrier
12  ds_load_tr16_b128           6  rocdl.s_wait_asynccnt
 4  rocdl.make_tdm_atom         1  rocdl.cluster_load_async_to_lds
 1  rocdl.wave_id               1  rocdl.ballot / exp2
```

**零 MFMA、零 `permlane32_swap`、零 `ds_read_tr16_b64`。**
用的是 WMMA + `ds_load_tr16_b128` + **TDM 异步拷贝族**（gfx1250 独有，CDNA4 没有对应物）。

**这就是 gfx1250 上写 FlyDSL attention 的模板。**

---

## 4. 反向：任何地方都没有 gfx1250 版本

`fmha_kernels.py:527` 原文（**实测**）：

> No `not deterministic` gate: it is a **backward-only flag** (this forward is atomic-free …)

即这份实现**只有前向**。全 aiter flydsl 树里找不到任何 gfx1250 的反向；
`fmha_gfx950/` 那份是 gfx950 fp8 的 dual-wave。

**所以反向是全新开发。** 好消息是它不是从零：前向那 4176 行已经把
wave32 布局、WMMA atom、`ds_load_tr16_b128` 的 lane 映射、TDM 异步拷贝与等待计数器
这一整套 gfx1250 惯用法建立起来了。

---

## 5. 包与版本：一条先前的说法被包内容推翻

| 事实 | 证据 | 标记 |
|---|---|---|
| `fa-repro` 镜像里装的是 **flydsl 0.2.4** | 无 GPU 一次性容器内 `ls site-packages/flydsl-*.dist-info`。**实测** |
| aiter pin 的是 **flydsl 0.3.2** | `aiter-src/setup.py:17 FLYDSL_VERSION = "flydsl==0.3.2"`。**文档记载** |
| `setup.py:548` 注释称「flydsl 0.2.4 does not support gfx1250」 | **与包内容矛盾**，见下 | **实测推翻** |

**0.2.4 已经带有全部 gfx1250 原语**（无 GPU 容器内 grep，**实测**）：

| 原语 | py 文件数 | `.so` 符号数 |
|---|--:|--:|
| `ds_load_tr16_b128` | 5 | — |
| `make_tdm_atom` / `tdm_ops` | 2 / 4 | — |
| `s_wait_asynccnt` | 5 | — |
| **`MmaOpGFX1250`** | 5 | **12** |
| `wmma_f32_16x16x32` | 3 | — |

提到 gfx1250 的文件：`_mlir/_mlir_libs/libFlyPythonCAPI.so`、`_mlir/dialects/_rocdl_ops_gen.py`、
`utils/smem_allocator.py`、`expr/rocdl/__init__.py`、`expr/rocdl/tdm_ops.py`、`expr/rocdl/cluster.py`。

**含义**：编译器侧的能力在我们已有的版本里就有。是否仍需升到 0.3.2，取决于
aiter 那份前向具体用了哪些 0.3.x API（已知断裂：`T.f8` 在 0.3.0 从 `flydsl.expr.typing` 移除）。
**这是 Stage 0 要静态查清的第一件事。**

### 一个必须防的静默错误

`flydsl/runtime/device.py:76` 的 `is_rdna_arch`（**实测**原文）：

```python
if arch.startswith("gfx10") or arch.startswith("gfx11"): return True
if arch.startswith("gfx120"): return True
return False
```

**`gfx1250` 不以 `gfx120` 开头**（是 `gfx125`）→ 返回 `False` → 被判为 **CDNA → wave64**。
aiter 拒绝使用它并在 `kernels_common.py:52-69` 自己重判，注释原文：

> Building a kernel for wave64 while gfx1250 dispatches wave32 **corrupts every >1-warp-per-block kernel**
> (the phantom upper lanes **silently drop their work**).

**任何用到 FlyDSL 自带 wave size 判定的地方都会静默错。** 这是本次调研里最危险的一条。

---

## 6. 分支景观：没有任何一行 gfx1250 attention 代码

| 分支 / PR | 状态 | 对我们的价值 |
|---|---|---|
| `feat/flydsl-attn-fwd-opt` (PR #500) | **已合入 main** `d3592bba` | gfx950 前向优化。**所有数字都是 head_dim 64 测的**，对 D=128 的影响未测 |
| `dev/kyle/flydsl-attn-bwd-nd` (PR #480) | 主体已合入 main | main 的反向已经是调优过的那一版 |
| `dev/kyle/flydsl-attn-gqa4` | 未合，2026-08-21 | **我们引用的 2.10×/1.13× 出自这里**。kernel 侧两行已随 #480 进 main，只剩 dispatcher 一行 |
| `dev/sukylasa/llama8b-flydsl-fa-pair` | 未合 | 价值在**警告**：fixed-max dualwave 在 D=128 上会 overflow/underflow BF16 行。作者自述 *"this is not a training keep"* |
| **`dev/sukylasa/llama8b-flydsl-hybrid-attn` (PR #516)** | 未合，2026-09-16 | **结构上最像我们的处境**：FlyDSL 前向 + AITER 反向 + packed-LSE adapter。**但 commit body 里一个性能数字都没有** |
| `feat/gemm/gfx1250-flydsl-gemm` (PR #499) | 未合 | **唯一的 gfx950→gfx1250 FlyDSL 移植先例**，1230 行 |

**穷举 grep 的结论**：每一条 attention 性能断言都带 gfx950 / MI355X 限定。
**整个分支景观里没有任何 gfx1250 attention 性能数字。**

### GEMM 分支里可复用的三样（但第一样有重大折扣）

1. `ds_load_tr16_b128` 的 **lane 映射**（`gemm_gfx1250_kernel.py:429-432`），commit `67beab6f` 明说该映射**无文档、是测出来的**
2. **wave32 写法范式**（`:71-73, :594-595`），并确认 `readfirstlane` 在 gfx1250 上存活
3. 518 行**测试模板** `tests/pytorch/ops/test_gemm_gfx1250.py`

**折扣**：GEMM 用 `ds_load_tr16_b128` 做的是 **LDS→LDS staging 转置**，MMA 随后用普通 copy 读 operand；
attention 需要的是 **LDS→寄存器直接喂进 MMA fragment**。`67beab6f` 明说 fragment-read 形式**没有做**：

> Doing it at staging rather than in the fragment read keeps the compute path byte-identical …
> at the cost of an LDS round trip and a smaller tile: **NN reaches 33% of roofline and TN 28%**,
> against NT's 59%. **Transposing in the fragment read instead would recover most of that.**

**attention 需要的正是这个被明确放弃的形式。** 而 aiter 的 gfx1250 前向做了 ——
所以模板应当以 aiter 那份为主，GEMM 分支为辅。

---

## 7. 预期收益：能说什么，不能说什么

### 现状（单层，`b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`）

| | fwd | bwd | total | 来源 |
|---|--:|--:|--:|---|
| 我们 · A0 冠军（ASM+ASM，1001 MHz） | 1.572 | 10.160 | 11.726 | **实测** |
| 我们 · B0 最好（满频 2133–2244 MHz） | 1.406 | 8.835 | 10.236 | **实测** |
| JIRA · MI455 Flex | 3.309 | 9.638 | 12.947 | **实测**（trace 解析） |
| JIRA · MI355 AITER | 3.797 | 5.381 | 9.178 | **实测**（trace 解析） |

### 上限的三种算法，结论都指向同一处

**（a）拿 gfx950 上的 FlyDSL/AITER 比值外推** —— `dev/kyle/flydsl-attn-gqa4`，B=2，**文档记载**：
前向 2.10×、**反向 1.13×**。若同比值成立，我们的反向 10.160 → **约 9.0 ms**。**推算**，而且
**这个外推的前提（比值跨架构成立）完全未验证**，gfx1250 连 MFMA 都没有。

**（b）拿 MI355 的 AITER 反向当目标** —— 5.381 ms。我们即便满频也是 8.835，差 **1.64×**。
一个 gfx1250 FlyDSL 反向若能达到 MI355 AITER 的水平，收益是 **约 3.5 ms/层**。**推算**

**（c）端到端折算** —— attention 目前占单步 **11–13%**（**实测**，profile）。
反向从 10.160 降到 9.0（算法 a）折合端到端约 **+1.2%**；降到 5.4（算法 b）约 **+4.5%**。**推算**

### 必须同时说清楚的

- **算法 (a) 的 1.13× 是 B=2 测的，我们是 B=4**，且 gfx950 与 gfx1250 的 MMA 形状完全不同 ——
  这个外推**不应当用于决策**，只能用于说明"FlyDSL 的反向优势本来就不大"。
- 同一份数据里 FlyDSL 的**前向**优势是 2.10×，**而前向我们已经是全表最快**（1.572 ms）。
  **FlyDSL attention 的价值集中在我们已经赢了的那一半。**
- 因此：**若只按"FlyDSL 反向能比 AITER 快多少"来立项，这个项目的预期收益是 1.13× 量级，很弱。**
  它值得做的理由只能是另一个 —— 见 §8。

---

## 8. 这件事真正值得做的理由（以及不值得做的条件）

**值得做的理由不是"FlyDSL 比 AITER 快"，而是三条结构性的：**

1. **我们现在的反向是一个会静默算错的预编译二进制。** aiter 的 gfx1250 ASM 反向在 GQA 下
   按 q head 索引写进 kv 尺寸缓冲区（**越界写**），我们靠 per-q-head + host 规约绕过，
   代价是 1 GiB 常驻 scratch 和一次规约。**源码级的 FlyDSL 实现可以从根上避免这类问题，并且我们能改。**
2. **gfx1250 的 ASM 资产只有 gfx950 的 1/25**（52 个 `.co` vs 1466；反向 6 vs 124）。
   我们被锁在别人发布节奏上；FlyDSL 是唯一能自己补齐变体的路。
3. **前向的模板已经有人写好了**（aiter 4176 行），反向可以复用它的全部 gfx1250 惯用法。

**不值得做的条件（任何一条成立就应当停）：**

- Stage 1 测出 aiter 的 gfx1250 FlyDSL **前向** 显著慢于我们现有的 ASM 前向（1.572 ms）——
  说明这套 gfx1250 FlyDSL 惯用法在本卡上竞争力不足，反向没有理由更好；
- aiter 那份前向在 flydsl 0.2.4 下**无法运行**，而升级 0.3.2 会破坏 turbo 自己的 FlyDSL 代码；
- nkfix 的 **21% nan 率**未解 —— 它决定当前 7.08× 能否交付，**优先级高于本项目**。

---

## 9. 无需 GPU 就能立刻做的部分

全部在 §10 的 Stage 0，见 `PLAN.md`。要点：aiter 那份前向对 flydsl 0.3.x API 的依赖是
**纯静态可查**的，不需要卡。

---

## 10. 两份模板：新写反向的实际材料

> 本节是 `PLAN.md:24` 和 `:104` 引用的那一节。它在 0917 那次断电里丢了 ——
> `SURVEY.md` 被填充到 16384 字节、尾部 1969 个 NUL，正文停在 §9。
> 现已按源码重写。分阶段计划本身在 `PLAN.md`，不在这里。

新写 gfx1250 反向不是从零开始，也不是移植 CDNA4 那份。手上有两份互补的模板：
一份给**原语与布局**，一份给**结构**。

### 10.1 `fmha_gfx1250/` —— 提供 gfx1250 的原语与布局（**实测存在**）

`/home/lihuzhan/code/aiter-src/aiter/ops/flydsl/kernels/fmha_gfx1250/`，4147 行 2 文件，
是 aiter 在 gfx1250 上的默认前向路径。可直接借用的惯用法（均为在这张卡上跑通过的）：

`ds_load_tr16_b128`（12 处，注意是 `b128`，不是 gfx950 的 `ds_read_tr16_b64`）、
`make_tdm_atom` ×4 + `tdm_ops` + `cluster_load_async_to_lds`（TDM 异步拷贝）、
`s_wait_dscnt` ×12 / `s_wait_asynccnt` ×6、`sched_barrier` ×9、WMMA `16x16x32` atom、wave32 布局。
**零 MFMA，零 `permlane32_swap`。**

### 10.2 `fmha_bwd_gfx942/` —— 提供反向的结构

`/home/lihuzhan/code/aiter-src/aiter/ops/flydsl/kernels/fmha_bwd_gfx942/fmha_bwd_core.py`，
1042 行。一份**高层写法**的完整 FlyDSL FA 反向。

**它的结构（来自源码 docstring 与常量，非转述）：**

`k_bwd` 在一个 grid 上承载**两种 job**，**每个输出元素恰好被一个 workgroup 写一次 ——
无 atomic、全确定性**，代价是每种 job 各算一次 score 矩阵。

| job | 分工 | 关键技巧 |
|---|---|---|
| **dK/dV** | 一个 workgroup 管 (128 个 key, seq, head)，流式吃 32 个 query 的 tile | `S = Q·Kᵀ` 和 `dP = dO·Vᵀ` 沿 d 收缩，操作数直接从显存来；`dV = Pᵀ·dO` 和 `dK = dSᵀ·Q` 沿 query 收缩，所以 **dOᵀ 和 Qᵀ 在 LDS 里转置暂存**，而 `Pᵀ`/`dSᵀ` 从累加器里**免费**得到 |
| **dQ** | 一个 workgroup 管 (128 个 query, seq, head)，流式吃 32 个 key 的 tile | **把 score 算成转置的**（`Sᵀ = K·Qᵀ`），于是 `dSᵀ` 片段免费就是 `dQ = dS·K` 需要的那个 `dS` 操作数；LDS 里只需暂存 `Kᵀ` |
| **pre** | `k_delta`：`D = rowsum(dO*O)`，fp32，布局 `[H, T]` 与 lse 一致 | 最简单，可独立验证 |

**越界安全靠 buffer descriptor，不靠 predication。** 每个全局张量都通过 `_buffer_view`
访问，descriptor 带精确的 `num_records`：越界读返回 0，越界写被硬件丢弃。
causal 尾块、空 workgroup、每一处 epilogue mask 都依赖这个机制。

**关键常量**（`KEYS_PER_WG=128` / `QUERY_TILE=32` / `QUERIES_PER_WG=128` / `KEY_TILE=32` /
`NUM_THREADS=512`），以及一条值得抄走的教训：

> LDS 行 stride 上的 `+4` 是**承重的**。未填充的 192 元素 stride = 96 dword = 3×32，
> 会把一个 MFMA A-operand 读的全部 16 个 lane 压到同一个 bank 上（16 路冲突），
> 曾经**主导整个内核的耗时**。

`build(n_split=1)` 还提供一个 split-K 变体：`n_split > 1` 独立编译第二份，只用于小负载
—— 那种情况下所有 workgroup 同时驻留，makespan 由**最长的单个 workgroup**决定，而不是总工作量。

### 10.3 合成时必须自己解决的三件事（模板没有给）

诚实地记下模板的边界，避免下一个会话以为这三件也是现成的：

1. **头维不同。** 这份模板是 **MLA 形状 `d_qk=192 / d_v=128`**，不是我们的 `d=128`。
   `DQK` / `DV` / `N_TILES_QK` / `N_TILES_V` / `FRAGS_PER_ROW_*` / `LD_QK` / `LD_V`
   以及那个 `+4` 的 bank 冲突分析，**全部要按 d=128 重推**。
2. **GQA 不在这份模板里。** 它的 grid 是 per head，dK/dV job 一个 workgroup 管一个 head 的
   key 块 —— 没有 q head → kv head 的规约路径。
   `_buffer_view` 的 `num_records` 是**越界保护**，不是 GQA 正确性机制：
   一个按 q head 索引 kv 尺寸缓冲区的写法，即使越界部分被硬件丢弃，dk/dv 仍然是错的。
   **新实现要从一开始就按 q head 分配或做 workgroup 内规约**，这是自己的设计，不是抄来的。
3. **指令替换不是机械的。** `fx.rocdl.MFMA(16,16,16,bf16)`（`C[16m,16n] += A[16m,16k]·B[16n,16k]ᵀ`）
   → `fx.rocdl.WMMA(16,16,32,bf16,fx.Float32)`：**k 从 16 变成 32**，
   所以 k 循环步长、LDS staging 的分块、以及"Pᵀ/dSᵀ 从累加器免费得到"所依赖的 fragment 布局
   都要跟着变。这一条是整个合成工作里最不确定的部分。

### 10.4 为什么这个结构正好对上我们的痛点

- **无 atomic、每个输出写一次** —— aiter 的 gfx1250 ASM 反向正是在 dkdv 上按 q head 索引
  kv 尺寸缓冲区造成越界写，我们用 1 GiB 常驻 scratch + host 规约绕过。
  一个从设计上"每个输出恰好写一次"的结构不会有这个问题。
- **varlen THD** —— gfx1250 **完全没有 varlen ASM 反向**（派发表只有 `mode=0`，
  host 对所有 seqstart 指针传 nullptr）。这份模板本身就是 varlen 的。
- **全确定性** —— 逐位可复现，`--determinism-reps` 那条验收路径直接可用。
