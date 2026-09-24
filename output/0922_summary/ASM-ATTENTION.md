# 基于 ASM 的 attention 优化总结

数据截止 2026-09-22 · 机器 A0 `heliosr-1b114-c07-1`（单卡 gfx1250 / MI455X，VR 限频 1100 MHz）
· 形状 `b=4 s=8192 hq=32 hkv=8 d=128 bf16 causal`（Llama-3.1-8B，GQA ratio=4）
· 分支 `dev/lhz/attn`（从 main `c1325c7e` 起）· 镜像 `amdprimus/amdprimus:gfx1250-20260910`
（容器内派生 `fa-tune:deps`）· aiter `6963ae9d`

> **口径提示**：A0 是限频单卡（rocm-smi 报 1100 MHz，持续负载下 sysfs 采到 943–967 MHz），
> 绝对值不可与满频的 B0（2133–2244 MHz）直接比较，本文不做任何时钟折算。
>
> **但限频的幅度要说准**：B0 时钟是 A0 的 2.13–2.24×，而同代码同形状只快
> **1.118× / 1.150× / 1.146×**（fwd / bwd / total）。**gfx1250 的 attention 几乎不受核心时钟限制**，
> 它受限于访存与固定延迟，而 VR 限的是 sclk 不是 HBM。此前"限频所以不可比"方向对、幅度错：
> **只值约 13%**。A0 的 11.726 ms 换到健康卡上约 **10.2 ms**。

---

## 0. 一句话

我们把 AMD 已经发布但**在 Python 侧完全够不着、且在 GQA 下算错**的 gfx1250 预编译 ASM
attention 内核，做成了 Primus-Turbo 里可派发、可 A/B、正确的一条生产路径：
单层 fwd+bwd **55.785 ms → 11.726 ms（4.76×）**，端到端训练在其它瓶颈被搬开之后再贡献
**+14.40%**。**我们一行汇编都没有改过** —— 价值全部在适配、纠错与工程化上。

---

## 1. 我们做了哪些适配

### 1.1 从零写发射器：aiter 没有给 gfx1250 任何可调用的入口

aiter 的 C++ host 有五处 gfx1250 特判，但 **Python wrapper 没有**：
`can_impl_fmha_v3_bwd` 从 `get_gfx() == "gfx942"` 起判，只放宽到 gfx950，
**gfx1250 永远选不中这些内核**。所以只能自己发射。

做法：离线从 ELF metadata + 反汇编反推 kernarg 布局，写出完整调用契约，再做发射器。

| 文件 | 行数 | 作用 |
|---|--:|---|
| `tools/gfx1250/asm_bwd_abi.py` | 136 | 从 ELF 推出的 kernarg 字段表 + 无 GPU/无 torch 自检 |
| `tools/gfx1250/asm_bwd_launcher.py` | 185 | 手工发射器（bring-up / ABI 自检入口） |
| `primus_turbo/.../attention/_asm_bwd_kernargs.py` | 335 | 产品侧 kernarg 打包 + `HipModule` |
| `primus_turbo/.../attention/attention_asm_bwd_impl.py` | 353 | 资格门 + adapter |
| `primus_turbo/.../attention/attention_asm_fwd_impl.py` | 199 | 前向资格门 + adapter |

主要的坑：**两种 packing 约定并存** —— `dqdkdv` 每字段 16 B 对齐，`odo` 紧凑打包。
ABI 自检（无 torch、无 GPU）当场抓到一张手抄字段表悄悄丢了最后七个字段（含 `mask_x`/`mask_y`）。

反向不是一次发射而是**三次**：
`bwd_hd128_odo_bf16`（delta=rowsum(dO*O)）→ `bwd_hd128_bf16_causal_br_a32_pssk`（主体，
514 条 `buffer_atomic_add_f32` 累加进 fp32 dq_acc）→ `bwd_hd128_dq_convert_bf16`（fp32→bf16）。

### 1.2 找到并绕过了 aiter 这批资产里的一个正确性 bug（最实质的一项）

**主体内核的 grid 是 `(kv_tiles, nhead_q, batch)`，却按 q head 去索引 kv 尺寸的 dk/dv 缓冲区
—— 越界写。** GQA ratio=4 下 4 个 workgroup 无同步地争同一块 dk/dv tile。

两副面孔，取决于越界落在哪：

| seqlen | 表现 |
|---|---|
| 1024 | **静默损坏**：dk/dv 掉到约 −0.3 dB，而 dq 仍然正确 52.24 dB |
| 256 | 跨页 → 进程 fault（dmesg 零记录，**不伤卡**） |

三个实验钉死它，并排除了"累加 vs 赋值"假设：

| 配置 | dq | dk | dv |
|---|--:|--:|--:|
| ratio=1（无 GQA） | 52.26 | 52.27 | 52.76 |
| ratio=4，dk/dv 按 kv head | 52.24 | **−0.94** | **−0.65** |
| ratio=4，dk/dv 按 q head + host 规约 | 52.24 | 50.55 | 51.09 |

**绕法**：dk/dv 按 q head 分配再在 host 侧规约。128–8192 全 seqlen 正确。
**代价（0917 首次单独计量）**：dk/dv 张量 0.125 → 0.500 GiB（4.0× = GQA ratio），
**反向发射峰值增量 1.254 GiB**；host 规约 **+0.482 ms = 反向的 +5.2%**（总时间 −4.1% 的空间）。

> 不做这一步，这个内核对 Llama-3.1-8B 这类 GQA 模型**直接不可用**。
> 已连同完整复现与定位写成 vendor report 提交 aiter（`output/0917__flydsl/VENDOR-REPORT-aiter-gfx1250.md`）。

### 1.3 接进派发层，而不是做成一个脚本

- 前向：可用时替换为 aiter 预编译 ASM 前向，并留 `PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD` 硬开关
  —— 没有开关就只能靠"让 aiter 不可 import"来做 A/B，那会同时改掉反向，对比作废。
- 反向：`PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD`（0915 起**默认 OFF**，见 §3）。
- 补了 **non-causal** 发射（换 `mask=0` 的 `.co`，causal 才半 x-grid）。
- 修了 `enable_gqa` 不被接受 —— 这才是 e2e 里 turbo attention 根本没被调用的原因。
- 资格门：**seqlen ≥ 2048**。依据是内核自身时间几乎不随工作量变化
  （0.88 / 0.91 / 1.00 / 1.34 ms，seqlen 1024→8192，16× 工作量），即约 0.85 ms 固定开销，
  低于 2048 摊不平。
  *（这条我们错过两次：先从单样本外推成"`b*hq ≥ 32`"，后又差点整个拿掉而那组数据是被污染的。）*

### 1.4 两个每调用开销

1. **每次反向都重新 `hipModuleLoad` 三个 `.co`** —— 32 层就是每步 96 次加载。改成进程级加载一次。
2. **scratch 按形状缓存**，进程级共 1.000 GiB，不随层数增长（此前一个 45× 的外推错误已纠正）。

### 1.5 ASM 之前的铺垫（同一条 attention 线上的工作）

- 为 gfx1250 vendored 一个**融合 Triton MHA 反向**（316 TFLOP/s），并按并行度而非序列长度设门
  —— 这是 ASM 反向接进来之前的冠军，也是现在的默认反向。
- B0 上九轴扫描 + 离线 ISA 预筛：`num_warps=2`（fwd）+ `waves_per_eu` 不设（bwd）
  合计 **+4.28%**，并把 16 个配置**在没有 GPU 的情况下判死**。
  被证伪的重要一条：离线 ISA 预测 `num_warps=8` 有 VGPR 收益，实测 spill 代价压倒 —— 寄存器压力假设不成立。

---

## 2. 算子级结果（A0，单层 fwd+bwd）

| # | fwd 实现 | bwd 实现 | fwd ms | fwd TF/s | bwd ms | bwd TF/s | total ms | total TF/s | vs 出厂 |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | Triton（turbo 出厂 `1cb2e183`） | Triton fused（出厂） | 10.651 | 206.5 | 45.134 | 121.8 | 55.785 | 138.0 | 1.00× |
| 2 | Triton（强制） | Triton fused（vendored） | 4.166 | 527.9 | 17.835 | 308.2 | 22.001 | 349.8 | 2.54× |
| 3 | **aiter ASM** | Triton fused | 1.561 | 1408.7 | 17.686 | 310.8 | 19.239 | 400.1 | **2.90×**（当前默认） |
| 4 | **aiter ASM** | **aiter ASM** | 1.572 | 1398.9 | 10.160 | 541.1 | 11.726 | 656.4 | **4.76×**（opt-in） |

**对出厂的分方向增益**：fwd **10.651 → 1.572 ms（6.78×，206.5 → 1398.9 TF/s）**，
bwd **45.134 → 10.160 ms（4.44×，121.8 → 541.1 TF/s）**。
第 3 行的收益**全部来自前向**（6.82×），反向未动；第 4 行才把反向也换掉。

FLOP 约定（causal）：fwd = `2·b·hq·s²·d`、bwd = `5·b·hq·s²·d`、合计 `7·b·hq·s²·d`；
生产形状下分别是 2.199e12 / 5.498e12 / 7.697e12 FLOP。
SQNR 全程 ≥ 50 dB 闸门
（第 4 行 out/dq/dk/dv = 53.62 / 52.24 / 50.61 / 50.83 —— dk/dv 这个指纹正是"ASM 反向确实跑了"的证据）。
第 1–2 行 n=3、事故前；第 3–4 行 AC-cycle 后 n=5，且第 3 行复现了事故前的值到 **0.03%**
（19.239 vs 19.233），这是机器回到同一状态的证据。

---

## 3. 端到端：同一个内核，从 −0.28% 变成 +12.72%

llama-3.1-8B，b=4 s=8192，1 GPU，seed 固定：

| 时点 | ASM 反向的 e2e 效应 | 原因 |
|---|---|---|
| 0915 | **−0.28%**（0.15× sem，不显著）→ **默认 OFF** | 当时 94% 的 step 是 hipBLASLt GEMM，attention 只占 3.3% |
| 0916（GEMM 缺陷绕开后） | 原报 **+14.40%** | attention 占比升到 20.8%，同一个内核、同一台机器、同样的算子级 1.74× |
| **0917 更正** | **+12.72%**（ON 42,534 / OFF 37,734） | 全量 74 次运行的正确性筛查发现两臂各有一次 `loss: nan`（`asm1-on` nan×6，且是 ON 臂最快的一次；`asm3-off` nan×1），各自剔除后重算。**效应依然真实且显著，只是小了 1.7 个百分点。** |

**结论写进了 skill**：一个优化的端到端价值是它与当前瓶颈关系的函数，不是它自己的属性。
叠加三项修法后全链（8 层，干净运行）**6,128 → 46,374 tps（7.57×）**，step 5.35 s → 707 ms；
**32 层生产配置 1,984 → 14,050 tps（7.08×）**，同会话对照 12,340。

> **默认仍然关闭**。理由已经不是"测不出收益"，而是它依赖的 GEMM 绕路（`nkfix.py`）
> 是运行时补丁而非产品代码，且该绕路本身有 21% 的 nan 率待解决。

## 4. 这条路还剩什么

| 顺位 | 动作 | 依据 |
|---|---|---|
| 1 | **向 aiter 报 GQA 越界写、申请 gfx1250 的 `psskddv` / a16 变体** —— 成本近零，杠杆最高 | gfx1250 只有 6 个反向 `.co`，gfx950 有 124 |
| 2 | 测 `_perf` 变体并修它的 dq pass | 已测：dk/dv 与 shipped 逐位相同，但 dq 只有 5.84 dB（"部分正确"，大概率要不同 grid/split） |
| 3 | 去掉 GQA 绕法 → 直接省 0.482 ms（总时间 −4.1%） | §1.2 |
| **不建议** | **自己改汇编** —— 内核 VGPR 已用满 1024、LDS 320 KiB、workgroup 1024、wave32，资源饱和 | 技能门槛最高、杠杆最低 |

**存在性缺口（无绕法）**：gfx1250 **没有 varlen ASM 反向**，dispatch 表只有 `mode=0`，
host 对每个 `seqstart` 指针都传 `nullptr`。这是 FlyDSL 反向立项的最强理由（见 `FLYDSL-ATTENTION.md`）。
