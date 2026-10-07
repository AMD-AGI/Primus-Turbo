## (b) 适配后各后端的性能

**算子形状**：Llama-3.1-8B attention，b4 s8192 hq32 hkv8 d128 bf16，causal（Sq=Skv，bottom-right 与 top-left 等价），BSHD，GQA 4。

**FLOP 口径**：表中 TF/s 统一按 op-evolve `tools/op_flops.py` 计算，fwd 2.199292e12，bwd 5.498229e12（bwd 按 5-GEMM 名义计数）。09-13..09-15 的源文档只报 fwd+bwd 合计（7.697e12 / 总耗时），表中带 `*` 的单向 TF/s 由单向耗时换算。树内 vendored 融合反向实际发 7 趟 GEMM，issued = 7.6975e12 = 名义的 1.400×（7/5）；我们的 FlyDSL bwd 同为 7-GEMM 结构，再加对角块粒度（×1.00546），issued = 7.7395e12 = 名义的 1.40764×。本节一律不用 issued 口径（`0914__repro__c07/RESULTS.md:121-122`；`0923__flydsl/hint.md:2104-2110`）。

**相对 ASM** = 本行耗时 / 同一 epoch 的 aiter ASM 耗时，>1 表示比 ASM 慢。
- "同进程"：与 ASM 在同一进程里交替测得，最可信。
- "≈"：同一台机器、同一时钟状态，但跨会话或跨尺子，只能看量级。
- 不同 epoch 之间的绝对值不可比，只能比同 epoch 内的比值。

**来源路径**：默认相对 `Primus-Turbo/output/`。
- `OE:` = `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`
- `skill:` = `Primus-Turbo/.claude/skills/gfx1250-attn-campaign/`
- `WT:` = worktree `wt-llama31`（分支 `dev/lhz/llama31_attn_opt`）

**测量 epoch**（下文按此分组）：

| 代号 | 机器 / 时钟状态 | 日期 | 主要尺子 |
|---|---|---|---|
| E1 | A0 `heliosr-1b114-c07-1` 单卡，VR 限频（DPM 上限 1100 MHz，负载约 943–1069 MHz） | 09-10..09-17 | `tools/gfx1250/tune_attention.py`：iters 20 / warmup 5，CUDA event 取中位，每 rep 冲刷 256 MiB L2，四张量 SQNR ≥ 50 dB；计时含 autograd 管路 |
| E2 | B0 `ctheliosp-1b112-a37-1`，4 卡同时负载，2133–2244 MHz | 09-14 | 同 E1 的 tune_attention.py |
| E3 | A0 仍限频（prod 计时窗口 989–1069 MHz） | 09-17..09-27 | op-evolve `benchmark.py`：只计 kernel 发射，median n=51，回文交错，3 s 预热；`fwdab.py` ABBA |
| E3′ | A0 刷固件前最后一次测量，仍 VR 限频（op 级 sclk 1020–1061 MHz）；VBIOS 630A、SMU 125.7.1、驱动 amdgpu-dkms 7.1.1-2397345 | 09-28 | 分块 randn 尺子（prod 每种顺序一个进程，A/A 副本同进程）；e2e 3 arm 同进程按步交替 |
| E4 | B0（空闲 2356 MHz；op 级没有逐条记录时钟；训练中 1.25–1.45 GHz） | 09-27..09-28 | 先用逐次交错；09-28 起改分块计时（fwd 约 03:40、bwd 约 05:00 UTC，lead 4 + block 9，A/A ±0.2%）；另有 `opcheck.py` |
| E5 | A0 09-29 刷固件后：VBIOS 700E、SMU 125.12.0，op 级 1.74–2.03 GHz，训练约 1.5 GHz；驱动 09-29 为 7.1.0-2411946，10-01 起 7.1.0-2412954 | 09-30..10-05 | 分块 randn 同进程；10-02 另有真实数据 blk / gb 两把尺子；`bench_attention_turbo.py` |

E1/E3/E3′ 期间的 A0 绝对数在 09-29 刷固件后作废。刷固件后时钟约提高 1.75×，但各 kernel 的提速并不一致（ASM fwd 1.24×，FlyDSL bwd 1.67×），所以旧数据不能按一个系数换算（`0928__a0_repro/REPORT.md` §6.3）。同一把分块尺子、同一批 arm，r13ns/ASM 从 E3′ 的 1.29 降到 E5 的 1.08，r29/ASM 从 1.44 降到 1.20，比值本身也随时钟 epoch 变化。刷固件和换驱动在同一时段（7.1.1-2397345 → 7.1.0-2411946，10-01 又换成 -2412954），两者不能严格分开归因（同上 §6.3）。e2e 也一样：E3′ 的 ASM 臂 1,947 ms 到 E5 的 1,350.6 ms（≈1.44×）不纯是平台差异——除了固件，两次运行之间驱动换了两次；而且 E3′ 那个进程（a0_p3a）在 FlyDSL 首次调用（step 2）之后，`HIPBLASLT_TENSILE_LIBPATH` 已被 fwd 树的 `_env.py` 改指到宿主库，hipBLASLt 是否会中途重读这个变量没有证实（`0928__a0_repro/REPORT.md` §5.3）。

### b.1 做了哪些适配才让这些后端跑起来

| 后端 | 关键适配 | commit / 路径 |
|---|---|---|
| Primus-Turbo Triton dense 后端（main #481 `c1325c7e`，gfx1250 上唯一能跑的树内后端） | <ul><li>torch 2.11 下 `low_precision.py` 的 `register_opaque_type` 让 import 失败，先在本地打补丁</li><li>默认派发选不到 Triton：AITER 先注册，且对任何架构都 `can_handle`，结果落到 CDNA-only 的 CK，反向报 invalid argument。修法是让 AITER dense 后端在 gfx1250 上不参与</li><li>FlyDSL 门写成 `>=(9,5)`，gfx1250 也会通过，改成精确判定 gfx950；flydsl 的模块级 import 加保护</li><li>conftest 在 gfx1250 上整体 skip，改为 opt-out 标记；能跑的测试 0 → 36 → 51</li><li>autotune 列表只有一个 CDNA 照搬的 config，改为可调（`PRIMUS_TURBO_ATTN_TRITON_TUNE`）：fwd `num_stages=2`、bwd `num_warps=2`</li></ul> | <ul><li>上游 `f5e1f18b`（#501）</li><li>`f8c45dee`、`df20f1d6`、`5eba2cf4`、`d00ba261`</li><li>产品分支 `c36cc124`</li></ul> |
| 树内 vendored aiter onekernel 融合 Triton 反向 | <ul><li>vendor aiter-src @`ffa945f9`（MIT，约 1790 行），按并行度而不是序列长度设门</li><li>修 torch.compile 下不可用的问题</li><li>fwd `num_warps=2` 加 bwd 不设 `waves_per_eu`：合计 +4.28%</li><li>`TRITON_HIP_USE_IN_THREAD_TRANSPOSE`：反向 −9%</li></ul> | `1cb2e183`、`03a76f61`→`8dd2fd32`、`ad67a2cc`、`566e7798`、`a009599b` |
| aiter Triton MHA（只用于测量，不进树） | <ul><li>amdprimus 镜像里没有 aiter，另外 git clone aiter-src（09-13 @`ffa945f9`，09-17 起 @`6963ae9d`）</li><li>aiter 的 CK/HIP JIT 在 gfx1250 上 import 时编译失败，但 Triton ops 仍可用</li></ul> | `tune_attention.py --impl aiter`；`OE: output/0911__fa_gfx1250_phase1/E2.md` §3 |
| torch flex | <ul><li>torchtitan 自带，不需要改</li><li>在 B0 上必须做 autotune（num_warps=4）。用默认启发式（bwd num_warps=8）测得的 fwd+bwd 合计 37.122 ms（fwd 6.894 / bwd 30.228，B0 GPU0 2356 MHz）已撤回，autotune 后合计 15.26–15.31 ms</li></ul> | `0914__repro__c07/flex_ledger.jsonl:1-3`；`0914__repro__c07/RESULTS.md:89-98` |
| aiter ASM fwd（`fmha_bf16_pertokenBf16_hd128_128x256_mask.co`） | <ul><li>aiter 可 import 时由派发层替换；ASM fwd 必须和融合 bwd 同时启用（合取门）</li><li>加硬关开关 `PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD`</li><li>写了直调 `.co` 的发射器</li><li>容器补装 psutil；首次调用会 JIT 编译 C++，需要 hipcc</li></ul> | `0e5cb743`、`c93057b6`、`20f2ccde`、`e4efebe2`（均在 `dev/lhz/attn`，未上游） |
| aiter ASM bwd（odo → `bwd_hd128_bf16_causal_br_a32_pssk` → dq_convert） | <ul><li>aiter Python 入口不选 gfx1250，自己写了约 674 行发射器，kernarg 字段表从 ELF 生成并做自检</li><li>`dkdv_heads='q'` 加 host 规约，绕开 GQA 越界写</li><li>资格门 seqlen ≥ 2048；放行非因果</li><li>scratch 复用；HipModule 改为进程级单例</li><li>默认关闭（`PRIMUS_TURBO_ATTN_ENABLE_ASM_BWD=1` 才开）</li><li>op-evolve 的 beat 臂按文件路径加载 `_asm_bwd_kernargs.py`，不 import primus_turbo</li></ul> | <ul><li>`c871cd47`、`adc367f0`、`c765edd4`、`46b1774a`、`f8155ae9`、`b281b46b`、`514e4a21`、`cb3ed519`</li><li>`tools/gfx1250/asm_bwd_{abi,launcher}.py`</li><li>`OE: …/job_context/op/beat/impl.py`</li></ul> |
| aiter FlyDSL fwd m32x8 | <ul><li>必须用 flydsl 0.3.2：0.2.4 的 ast_rewriter 不接受 list 作为状态变量</li><li>0.3.2 侧装到 `~/.local/flydsl032`。0.3.2 删除了 `flydsl.expr.buffer_ops`，因此不能和 turbo 的 0.2.4 树在同一进程</li><li>09-25 起改用 0.3.4.1，性能不变</li></ul> | `234ec33a`（`--impl flydsl`）、`963e7d7d`；`0917__flydsl/STAGE1-FWD.md` §4 |
| 我们的 FlyDSL fwd / bwd | <ul><li>op-evolve job：bwd 固定 0.3.2，fwd 用 0.3.4.1</li><li>产品分支 `dev/lhz/llama31_attn_opt`（基于 origin/main `8cda13c7`）移植到仓库固定的 flydsl 0.2.4</li><li>0.2.4 的 LLVM 把 async LDS→global O store 的地址编错，改用 `buffer_store`，代价 fwd −1.3%、bwd −1.8%</li></ul> | `5658a443`、`e05b67fd`、`5a8446b5`、`c36cc124`；`primus_turbo/flydsl/attention/gfx1250/` |
| e2e 接入（各后端共用） | <ul><li>接受 `enable_gqa`。此前训练配置用 `converters: []` 绕开 TypeError，导致 turbo attention 从未进入 e2e</li><li>容器内 `pip uninstall primus_turbo`：镜像里的 editable 0.4.1.dev12 遮蔽了 checkout</li><li>`HIPBLASLT_TENSILE_LIBPATH` 指向镜像库（`library/gfx1250` 子目录）</li><li>nkfix 绕开 hipBLASLt NN/wgrad 坏 tile，B0 上重写为 `nkfix_b0.py` + `transpose_triton.py`</li><li>e2e 后端切换 shim（`E2E_ATTN`，不改 Primus / Turbo 源码）</li></ul> | `1babf4ce`、`4fec7712`、`23fdc0bc`、`f08ccffe`；`0927__b0/gemm/`、`0927__b0/e2e/`、`1002__e2e/`；`WT: docs/gfx1250_llama31_8b_e2e/` |

### 总览：各后端的最新可比读数

每个后端只列它最近一个能和 ASM 对照的 epoch（epoch 定义见上方 E1–E5 表）。各行的绝对值来自不同 epoch，行与行之间不能直接比，只看每行自己的"相对 ASM"。TF/s 为 op_flops 名义口径，带 `*` 的由耗时换算；o1 / o2 是分块尺子的两种 arm 顺序；e2e 默认 32 层，带 † 的 ms 由 tps 换算；标"跨工具""跨尺子""跨进程"的比值只看量级。没有任何读数的格子写"未测"。主要缺口：
- E5（刷固件后）只测过 ASM、三种 FlyDSL（aiter m32x8、我们的冠军、产品移植）和 main Triton；flex、aiter Triton、vendored 融合 Triton 都没有 E5 数字。
- nkfix 之后，flex 和 aiter Triton 没有 e2e 数字；turbo Triton 在 A0 上只有估算，实测只有 B0 E4 一次。

| 后端 | 最新可比 epoch | fwd ms / TF/s | bwd ms / TF/s | e2e ms/step / tok/s（及 GEMM 路径） | 相对 ASM | 备注 |
|---|---|---|---|---|---|---|
| Primus-Turbo Triton dense 后端，出厂单 config（main；E1、E4 测的是 dev 分支出厂 `1cb2e183`，B0 上它与 main `c1325c7e` 的 fwd+bwd 合计差 7.6%） | op：E5（09-30，`bench_attention_turbo.py`）；e2e：E4（B0 09-28） | 5.11 / 430.45 | 24.49 / 224.50 | A0：只有估算，main + TRITON + nkfix ~1,900–2,050 / ~16,000–17,000；B0 E4：2,250 / 14,563（nkfix_b0 + transpose_triton，`NKFIX_CHECK=1`，`1cb2e183`） | fwd ≈4.1–4.3×、bwd ≈4.5×（跨工具）；e2e ≈1.42×（B0，跨进程） | E4 opcheck：fwd 5.090、bwd 24.049 ms（≈4.5× / ≈3.4×，跨进程）；E1：10.673 / 48.969 ms（≈6.8× / ≈6.4×） |
| Primus-Turbo Triton 调优 + 树内 vendored 融合 bwd + ITT（09-14 冠军 `a009599b`；冠军本身的 fwd 走 ASM，fwd 列给的是树内调优 Triton fwd） | fwd：E2（B0 09-14）；bwd：E1（A0-VR，对 E3 bar）；e2e：E1（A0-VR 8L，09-16） | 2.734 / 804.4*（E2） | 17.835 / 308.3*（E1，fwd 走 Triton；fwd 走 ASM 时 17.686 / 310.9*）；E2 8.840 / 622.0*（当天无 ASM bwd） | A0-VR 8L：≈868† / 37,734（ASM fwd + 融合 bwd，镜像库 + nkfix v3）；32L 只有 nkfix 前：16,500 / 1,984 | fwd ≈1.94×（E2）；bwd ≈2.30×（E1 对 E3 bar，跨尺子），同进程对 bring-up ASM 8.661 ms 为 2.01×（该 ASM 读数含逐次 hipModuleLoad，偏慢）；e2e 8L 1.127×（换成 ASM bwd 后 +12.72% tps） | E3–E5 未测（09-16 之后没有读数） |
| aiter Triton MHA（只用于测量，不进树） | fwd：E2；bwd：E2（当天无 ASM bwd）/ E1（对 E3 bar） | 出厂 3.366 / 653.4*；调优上界 2.095 / 1049.8*（E2） | 出厂 13.838 / 397.3*；调优上界 9.174 / 599.3*（E2）；E1 调优冠军 18.416 / 298.6* | 未测 | fwd ≈2.39×（出厂）/ ≈1.49×（调优上界），E2；bwd 调优冠军 ≈2.40×（E1 对 E3 bar，跨尺子） | E5 未测。E1 的两次出厂读数互相矛盾（fwd 6.676 vs 3.725 ms），未定；`mha_fused_bwd` 数值错误，已淘汰 |
| torch flex（torchtitan 现役 attention） | op：E2；e2e：只有 nkfix 前（E2 B0、E1 A0） | 3.771 / 583.2*（E2，inductor autotune，num_warps=4） | 11.536 / 476.6*（E2）；E1 23.831 / 230.7* | B0 E2：eager 13,700 / 2,392（镜像 hipBLASLt，nkfix 前）；compile + Triton GEMM 3,410 / 9,602；A0 E1：133,700 / 244（rocBLAS）；nkfix 之后未测 | fwd ≈2.67×（E2）；bwd ≈3.1×（E1 对 E3 bar，跨尺子）；e2e 无同条件 ASM 对照（A0 rocBLAS 下，09-15 真正走 turbo、ASM fwd 的运行也是约 244 tok/s，差异被 GEMM 掩盖） | E5 未测。A0 上关掉 converter 走 flex 必经 inductor autotune，09-15 因此挂卡一次，所以 A0 上测不了不用 turbo 的 flex 基线 |
| torch SDPA FLASH（AOTriton） | E1 | 未拆分：只有 fwd+bwd 合计 97.063 ms / 约 79（合计口径） | 未拆分（见左） | 未测 | 没有单向比值；E1 合计最慢 | `c1325c7e` 提交信息里 b2 形状 161.4 TF/s（测量条件不明） |
| aiter ASM（fwd `fmha_bf16_pertokenBf16_hd128_128x256_mask.co`；bwd odo + `bwd_hd128_bf16_causal_br_a32_pssk` + dq_convert，自写发射器 + `dkdv_heads=q` host 规约） | E5（op 09-30；e2e 10-02） | 1.257 / 1750（o2 1.253 / 1755） | 5.503 / 999（o2 5.494 / 1001） | 1,350.6 / 24,262（镜像库 + nkfix，`NKFIX_CHECK=1`） | 1 | 训练工作点 gb 尺子：fwd 1.137、bwd 5.530 ms（10-02）；A0-VR 时 bar 为 fwd 1.5724 / bwd 7.6766 ms（E3） |
| aiter FlyDSL fwd m32x8（aiter 自带的 gfx1250 prefill，我们 fwd 的起点） | E5 | 1.414 / 1555*（o2 1.413） | 无（aiter 只有 fwd） | 未测 | 1.125 / 1.128（同进程） | A0-VR 下为 1.50–1.53×（E1、E3、E3′） |
| 我们的 FlyDSL 冠军（flydsl 0.3.2 / 0.3.4.1）：fwd r16（= r13ns），bwd s6（另列 r29） | E5（op 09-30；e2e 10-02） | r16：1.353 / 1626（o2 1.356 / 1622） | s6：5.295–5.300 / ≈1037–1038；r29：6.583 / 835 | r16 + s6：1,349.4–1,349.9 / 24,275–24,283；r16 + r29：1,388.5–1,389.6 / 23,581–23,599（镜像库 + nkfix，`NKFIX_CHECK=1`，与 ASM 同进程交替） | fwd 1.076 / 1.082；bwd s6 0.963（ASM 的 103.8%），r29 1.196 / 1.201；e2e 0.999 / 1.027（同进程） | 训练工作点 gb 尺子：fwd 1.349×，s6 1.038×，r29 1.342×；op-evolve bwd 冠军 r24 与 s6 等价（≈0.962×） |
| 产品分支移植（`dev/lhz/llama31_attn_opt`，flydsl 0.2.4，fwd r16 + bwd r29） | E5（op 09-30；e2e 10-05） | 1.292 / 1702*（同进程 ASM 1.187） | 6.618 / 830.8*（同进程 ASM 5.360） | 教程 1,361.9 / 24,061（镜像库 + nkfix + transpose_triton，`NKFIX_CHECK=0`） | fwd 1.088、bwd 1.235（同进程）；e2e 约 1.03×（推算：ASM 行是 CHECK=1，按约 +33 ms/步折算到 CHECK=0 约 1,318 ms；未实测） | s6 未移植；O 改用 `buffer_store` 写回，代价 fwd −1.3%、bwd −1.8%；对 main Triton 的 causal 几何平均 fwd 4.13× / bwd 3.37× |
| CK fmha / HipKittens / Gluon / Primus-Turbo 自带的 gfx950 FlyDSL | — | 未测（CK fwd 能跑通，但没有计时；其余编不过，或只认 gfx950） | 跑不起来（CK 在训练中途报 `invalid argument for fmha_bwd`；HipKittens 卡在三道编译期硬墙，GPU gate 256 个元素错 126 个；Gluon 只有 fwd） | 跑不通（CK 的 bwd 在训练中途报错） | — | 见 b.2.3 |

### b.2 算子性能

#### b.2.1 fwd

| 后端 | 版本/变体 | 耗时 ms | TF/s | 相对 ASM | 机器/频率 | 口径（FLOP，尺子） | 日期 | 来源 |
|---|---|---|---|---|---|---|---|---|
| **E1 · A0 VR 限频** | | | | | | | | |
| Primus-Turbo Triton（dev 分支出厂配置 @`1cb2e183`，即 `wt-bakeoff`；main `c1325c7e` 在 A0 上从未测过，B0 上两棵树 fwd+bwd 合计差 7.6%） | 出厂，单 config：num_stages=1，num_warps=4 | 10.673（09-15 复测 10.651，同为 @`1cb2e183`） | 206.1* | ≈6.8× | A0 ≤1100 MHz | 由合计口径拆分；tune_attention.py | 09-13 | `0913__opt_plan__claude/phase1/BAKEOFF.md:15-21`；`0915__repro__c07/RESULTS.md:13`；树的对应关系见 `0914__repro__c07/FINAL-TABLE.md:15`、`RESULTS.md:44-64` |
| 同上 | 调优：fwd num_stages=2 | 4.152（09-15 强制 Triton fwd 4.166） | 529.7* | ≈2.66× | 同上 | 同上 | 09-13/15 | `BAKEOFF.md:20`；`0915__opt/RESULTS.md:12` |
| aiter Triton MHA（测量用） | 出厂 | 6.676；09-15 复测 3.725 | 329.4* / 590.4* | ≈4.3× / ≈2.4× | 同上 | 同上 | 09-13/15 | `BAKEOFF.md:19`；`0915__repro__c07/RESULTS.md:16,49-60`（0915 判定 0913 这一行偏差 +23%、不可信，原因未查清；(a) a.2 #12 反过来认为更可能异常的是 09-15 那次，其 fwd 3.725 像调优过的前向。未复测，未定） |
| 同上 | +fwd num_stages=2 | 3.260 | 674.6* | ≈2.09× | 同上 | 同上 | 09-13 | `BAKEOFF.md:15` |
| torch flex（torchtitan compile 选项） | dense causal BlockMask，enable_gqa | 7.506 | 293.0* | ≈4.8× | 同上 | 同上 | 09-13 | `BAKEOFF.md:16` |
| aiter ASM fwd（入口 `fmha_fwd_with_sink_asm`） | 经派发层替换 | 1.561（n=5）；09-17 ABAB 1.5691（n=20） | 1408.9* / 1401.6* | 1 | A0，09-17 窗口 1100→967 MHz | tune_attention.py / stage1_fwd_ab.py | 09-15/17 | `0915__opt/RESULTS.md:14`；`0917__flydsl/STAGE1-FWD.md` §1 |
| aiter FlyDSL fwd m32x8（flydsl 0.3.2） | aiter 自带的 gfx1250 默认 prefill | 2.3732（n=20，sd 2.68%） | 926.7* | 1.51×（同进程） | A0 1100→967 MHz | stage1_fwd_ab.py ABAB | 09-17 | `0917__flydsl/STAGE1-FWD.md` §0-1 |
| **E2 · B0 09-14** | | | | | | | | |
| Primus-Turbo Triton | 出厂 @`c1325c7e`（wt-main）/ @`1cb2e183` | 5.226 / 5.179 | 420.8* / 424.7* | ≈3.7× | B0 2133–2244 MHz | 合计口径拆分；tune_attention.py，n=3–4 | 09-14 | `0914__repro__c07/FINAL-TABLE.md:7-8` |
| 同上 | 只调配置 | 2.738 | 803.2* | ≈1.94× | 同上 | 同上 | 09-14 | `FINAL-TABLE.md:9` |
| 树内 vendored 融合路径用的 Triton fwd | 当时冠军的配置 | 2.734 | 804.4* | ≈1.94× | 同上 | 同上 | 09-14 | `FINAL-TABLE.md:12` |
| aiter Triton MHA | 出厂 / 调优上界 | 3.366 / 2.095 | 653.4* / 1049.8* | ≈2.39× / ≈1.49× | 同上 | 同上 | 09-14 | `FINAL-TABLE.md:10,13`（出厂行 0915 建议重测） |
| torch flex | inductor autotune（num_warps=4） | 3.771 | 583.2* | ≈2.67× | 同上 | flex_anchor.py | 09-14 | `FINAL-TABLE.md:11` |
| aiter ASM fwd | T7 探针（与冠军交替 3 次）/ 收尾复测（GPU1 较空闲） | 1.410 / 1.396 | 1559.8* / 1575.4* | 1 | 同上 | tune_attention.py `--impl asm` | 09-14 | `0914__campaign/RESULTS.md:101-112`；`HANDOFF.md:436-448` |
| 同上 vs 树内 Triton fwd（soak 10 轮） | — | ASM 1.408–1.423 / Triton 2.591–2.634 | ~1556 / ~847 | Triton 为 ASM 的 1.83–1.86× | 冷却后 | `t7/asm_fwd_soak.py` | 09-14 | `0914__campaign/t7/soak_cooled.jsonl` |
| **E3 · A0 09-17..09-27（仍限频）** | | | | | | | | |
| aiter ASM fwd | bar 普查 | 1.5724（n=51）；同窗 fwd+bwd anchor 中为 1.6061 | 1398.67 | 1 | sclk 1011→989 | op_flops；同进程 median | 09-24 | `0924__flydsl/bar-census/fwd_anchor.json`、`fwdbwd_anchor.json` |
| aiter FlyDSL fwd m32x8 | flydsl 0.3.2 | 2.4005 | 916.16 | 1.527×（同进程） | 同上 | 同上 | 09-24 | `fwd_anchor.json` |
| 同上 | 0.3.2 对 0.3.4.1，ABBA×3 | 2.332–2.343（同进程 ASM 1.539–1.553） | 938.5–942.9 | 1.50–1.52×（两个版本无差别） | sclk 1038–1049 | `fwdab.py`，n=52，一个进程只跑一个形状 | 09-25 | `0925__flydsl/fwd341/ab_prod.log` |
| 我们的 FlyDSL fwd（仅作参考） | A0 fwd job r11 冠军 | 1.979* | 1111.15 | 1.27×（同 session 的 beat 为 1407.8 TF/s） | ~1.04 GHz | benchmark.py（h28 尺子，beat 单独一个进程） | 09-27 | `OE: artifacts/gfx1250-flydsl-attn-fwd-20260925-114644/rounds/011/1-opt/act.yaml` |
| **E3′ · A0 09-28（刷固件前最后一次，仍 VR 限频；A0 上唯一一次刷固件前的 FlyDSL/ASM 同进程分块测量）** | | | | | | | | |
| aiter ASM fwd | prod，两种顺序 o1 / o2 | 1.564 / 1.543 | 1406.2* / 1425.3* | 1 | A0 sclk 1020–1061 MHz | 分块 randn，每种顺序一个进程，A/A 0.15% / 0.01% | 09-28 | `0928__a0_repro/REPORT.md` §1（:12-19）、:162 |
| aiter FlyDSL fwd m32x8（vendored base） | 同上 | 2.395 / 2.329 | 918.3* / 944.3* | 1.531 / 1.509 | 同上 | 同上 | 09-28 | 同上 |
| 我们的 FlyDSL fwd r13ns（= r16） | 同上 | 2.023 / 1.992 | 1087.1* / 1104.1* | 1.293 / 1.291（刷固件后同尺子降到 1.076 / 1.082，见 E5） | 同上 | 同上 | 09-28 | 同上；`§6.1` |
| **E4 · B0 09-27/28** | | | | | | | | |
| Primus-Turbo Triton | 出厂 @`1cb2e183`，opcheck：kernel / 经模块 | 5.090 / 5.200 | 432.1* | ≈4.5×（对 ASM opcheck 1.129，不同进程） | B0 GPU0 | opcheck.py，单进程 | 09-28 | `0927__b0/e2e/BUILD.md` §0 |
| aiter ASM fwd | job 同会话 beat（分块）/ 尺子审计稳态 / opcheck | ≈1.41 / 1.43–1.46 / 1.129 | 1552.7–1557.2 | 1 | B0 | 分块；opcheck | 09-28 | `0927__b0/fwd/progress.md`（r10–r13）；`ruler/REPORT.md`；`e2e/BUILD.md` |
| 同上 | 旧交错尺子（PERF 表，2 次均值） | 1.144 | 1922* | 1 | B0 | 交错（09-28 03:40 前的默认尺子） | 09-27 | `0927__b0/REPORT-0928.html` §2 |
| aiter FlyDSL fwd m32x8（vendored base） | 交错尺子 | 1.737（约 1.78） | ≈1236 | ≈1.52×（= 1/0.658，分母为同尺子 ASM 1.144） | B0 | 交错（旧尺子；该尺子下 FlyDSL 对 ASM 的比值最多偏差约 25%） | 09-27 | `0927__b0/fwd/progress.md:4`；`REPORT-0928.html` §2 |
| 我们的 FlyDSL fwd r13（投机 softmax，randn 上最快；仅作参考） | job 分块 / 同进程 randn 分块 / 真实 dump 分块 / 真实 dump 紧跟 GEMM 突发 | 1.463 / 1.4795–1.4799（ASM 1.4317–1.4363）/ 1.71–1.86 / 1.85–2.11 | 1502.8（job 尺子，beat 的 96.5%）/ 1486.1–1486.5* | — / 1.030–1.033 / 1.17–1.29 / **1.44–1.66** | B0 GPU0 | 分块；h47 真实数据尺子 | 09-28 | `REPORT-0928.html` §2；`0927__b0/fwd-nospec/REPORT.md:16-20,74-97`。randn 上最快、训练工作点下最慢，所以改采纳关掉投机的 r13ns（= r16） |
| 我们的 FlyDSL fwd r16（= r13ns，B0 冠军） | randn，分块 | 1.5507–1.5515（ASM 1.4317–1.4363） | 1418*（job 尺子 1443–1455） | 1.080–1.084（同进程） | B0 GPU0 | 分块，3 个进程轮换 arm 顺序 | 09-28 | `0927__b0/fwd-nospec/REPORT.md` §0、`:77-78` |
| 同上 | 真实 q/k/v dump 分块 / dump 紧跟 GEMM 突发（训练工作点，sclk 约 1.25–1.69 GHz，未逐次记录） | 1.55–1.59（ASM 1.42–1.47）/ 1.62–1.64（ASM 1.27–1.30） | 1383–1419* / 1341–1358* | 1.073–1.093 / 1.253–1.288 | B0 GPU0 | h47 真实数据尺子，3 个进程的中位数 | 09-28 | `fwd-nospec/REPORT.md` §0、`:95-97`；时钟见 `REPORT-0928.html` §2 |
| **E5 · A0 刷固件后** | | | | | | | | |
| aiter ASM fwd | randn 分块，两种顺序 o1 / o2 | 1.257 / 1.253 | 1750 / 1755 | 1 | A0 sclk 1757–1992 MHz（o1 1757–1928，o2 1992–1902） | 分块，同进程，A/A 0.07% / 0.02% | 09-30 | `0928__a0_repro/REPORT.md` §6.1 |
| aiter FlyDSL fwd m32x8（vendored base） | 同上 | 1.414 / 1.413 | 1555* | 1.125 / 1.128 | 同上 | 同上 | 09-30 | 同上 |
| 我们的 FlyDSL fwd r13ns（= r16，0.3.4.1） | 同上 | 1.353 / 1.356 | 1626 / 1622 | 1.076 / 1.082 | 同上 | 同上 | 09-30 | 同上 |
| 同上 | 10-02 randn（驱动 2412954） | 1.362（ASM 1.262） | 1614.8* | 1.079 | A0 | 分块（base1002） | 10-02 | `1002__e2e/E2E-PLAN.md:56` |
| 同上 | 10-02 真实数据（6 层几何平均）：blk / gb | 1.308 / 1.535（ASM 1.209 / 1.137） | 1681* / 1433* | 1.081 / **1.349** | blk ~1.45–1.5 GHz；gb ~1.28 GHz | blk = 分块；gb = 每次计时前跑 10 个 32768×4096×14336 GEMM | 10-02 | `1002__e2e/RESULT-realab.md` |
| 产品移植 FlyDSL fwd（flydsl 0.2.4，buffer_store 写 O） | prod | 1.292（ASM 1.187） | 1702* | 1.088 | A0 | 分块 randn，同进程 | 09-30 | `0930__port/PR_BODY.md:31-33` |
| Primus-Turbo main Triton dense 后端 | 出厂（产品分支 bench，TestID 54） | 5.11（同一张表里 FlyDSL 为 1.29） | 430.45 | ≈4.1–4.3×（跨工具，对 ASM 1.19–1.26） | A0 | `bench_attention_turbo.py`，FLOP 与 op_flops 一致 | 09-30 | `0930__port/runs/bench_turbo_{triton,flydsl}.csv` |

#### b.2.2 bwd

| 后端 | 版本/变体 | 耗时 ms | TF/s | 相对 ASM | 机器/频率 | 口径（FLOP，尺子） | 日期 | 来源 |
|---|---|---|---|---|---|---|---|---|
| **E1 · A0 VR 限频（计时含 autograd 管路；E1 期间唯一干净的 ASM bwd 数在 E3，对它的比值标 ≈）** | | | | | | | | |
| Primus-Turbo Triton 两内核 bwd（dev 分支 @`1cb2e183`，同 fwd 表） | 出厂，num_warps=4 | 48.969（09-15 复测 45.134，同为 @`1cb2e183`） | 112.3* / 121.8* | ≈6.4× / ≈5.9× | A0 ≤1100 MHz | 由合计口径拆分；tune_attention.py | 09-13/15 | `BAKEOFF.md:15-21`；`0915__repro__c07/RESULTS.md:13` |
| 同上 | 调优：num_warps=2 | 32.252 | 170.5* | ≈4.2× | 同上 | 同上 | 09-13 | `BAKEOFF.md:20` |
| aiter Triton onekernel bwd | 出厂 / 加 2 个旋钮 | 27.888 / 27.947（09-15 出厂复测 24.310） | 197.2* / 196.7*（226.2*） | ≈3.6×（≈3.2×） | 同上 | 同上 | 09-13/15 | `BAKEOFF.md`；`0915__repro__c07/RESULTS.md:16`。另：09-11 记录的"aiter + 2 knobs"24.347 ms / 316.1 TF/s 是 **fwd+bwd 合计**（7.697e12 口径，其中 bwd 21.075 ms），来自更旧的 aiter 版本，09-13 未复现，**已撤回**（`0913__opt_plan__claude/phase1/RESULTS.md:462-484`；`OE: output/0911__fa_gfx1250_phase1/E2.md:239`） |
| 同上，调优冠军（非树内） | BLOCK_M1=32, N1=256, M2=256, N2=32, BSF=1 | 18.416 | 298.6* | ≈2.40× | 同上 | 同上；launch 处核对 kernel 名 | 09-13 | `0913__opt_plan__claude/phase2/DECISIONS.md` D9 |
| torch flex | — | 23.831 | 230.7* | ≈3.1× | 同上 | 同上 | 09-13 | `BAKEOFF.md:16` |
| 树内 vendored 融合 Triton bwd | `1cb2e183` | 20.201 | 272.2* | ≈2.63× | 同上 | 同上 | 09-13 | session 541e7bc3 2026-09-13T09:43Z；`0913__opt_plan__claude/PROGRESS.md:70` |
| 同上，加 `566e7798`、`a009599b`（ITT） | fwd 走 Triton / fwd 走 ASM；另与 ASM bring-up 同进程测一次 | 17.835 / 17.686；同进程 17.407 | 308.3* / 310.9* | 对同进程 ASM 8.661 为 2.01×；对 bar 7.68 ≈2.30× | 同上 | 同上 | 09-15 | `0915__opt/RESULTS.md:12-14,72-79` |
| aiter ASM bwd（3 个 .co + dkdv_heads=q + host 规约） | t2_bringup 独立对拍：不经 autograd，未传 hip= / scratch= | 8.661 | 634.8* | 本会话参照（偏慢，含每次 hipModuleLoad） | 同上 | 独立脚本 | 09-15 | `0915__opt/RESULTS.md:72-79` |
| 同上 | ~~tune_attention.py `--impl asmbwd`（autograd shim）~~ | ~~10.160~~ | ~~541~~ | **已撤回**：shim 每次调用都做 3 次 hipModuleLoad、新分配约 1 GiB scratch，另含约 1.4 ms autograd 开销。修复后的路径重测为 8.13–8.68 ms，干净值见 E3 的 7.68 ms。同一 shim 下"强制 Triton fwd + ASM bwd"测得的 10.418 也一并作废 | 同上 | — | 09-15（09-24 撤回） | `0915__opt/status.json:34-38`；`0915__opt/E2E-AB.md:284-302`；`skill: references/baselines.md` §5 |
| **E2 · B0 09-14（当日没有 ASM bwd 实测）** | | | | | | | | |
| Primus-Turbo Triton 两内核 | 出厂 @`c1325c7e` / @`1cb2e183` | 25.025 / 22.942 | 219.7* / 239.7* | — | B0 2133–2244 MHz | 合计口径拆分；tune_attention.py | 09-14 | `0914__repro__c07/FINAL-TABLE.md:7-8` |
| 同上 | 只调配置 | 16.195 | 339.5* | — | 同上 | 同上 | 09-14 | `FINAL-TABLE.md:9` |
| aiter Triton onekernel bwd | 出厂 / 调优上界 | 13.838 / 9.174 | 397.3* / 599.3* | — | 同上 | 同上 | 09-14 | `FINAL-TABLE.md:10,13` |
| torch flex | autotune | 11.536 | 476.6* | — | 同上 | 同上 | 09-14 | `FINAL-TABLE.md:11` |
| 树内 vendored 融合 Triton bwd | 当日冠军 | 10.270 | 535.4*（源文档记名义 534.6 / 7-GEMM 实发 748.4，两者对应约 10.284 ms 的读数） | — | 同上 | 同上 | 09-14 | `FINAL-TABLE.md:12`；`0914__repro__c07/RESULTS.md:121-122` |
| 同上，加 ITT（`a009599b`，09-14 终版） | GPU1 较空闲，n=1 | 8.840（四卡满载时 fwd+bwd 合计 10.299） | 622.0* | — | 同上 | `bin/final_champion.sh` | 09-14 | `0914__campaign/HANDOFF.md:436-448` |
| **E3 · A0 09-17..09-25（仍限频；op-evolve benchmark.py）** | | | | | | | | |
| aiter ASM bwd（beat 臂） | 每次调用 6 个 dispatch：odo、pssk、dq_convert、dq_acc 清零、2 次 GQA reduce | job setup：7.6134 | 722.2 | 1 | sclk 1051–1100 | op_flops；只计 kernel 发射 | 09-17 | `0917__flydsl/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/.runs/bench_v000.json` |
| 同上 | bar 普查 | **7.6766**（n=224 原始 / n=146 去重 7.6769，sd 0.47–0.50%，没有 ≥7.8 ms 的记录）；同窗 fwd+bwd anchor 中为 7.7636 | ~716 / 708.2 | 1 | prod 窗口 998–1029 MHz | 同上 | 09-24 | `0924__flydsl/DAY-SUMMARY.md:12`；`bar-census/fwdbwd_anchor.json` |
| 我们的 FlyDSL bwd r0（从零手写，单 wave32 WG） | job baseline | 96.05 | 57.24 | 12.6×（同会话） | 同上 | 同上 | 09-17 | `OE: artifacts/gfx1250-flydsl-attn-bwd-20260917-115934.a0-stale-0930/job_context/state.yaml` rounds[0] |
| 我们的 FlyDSL bwd r20（A0 冠军，仅作参考） | — | 10.751* | 511.42（r22 复测 516.87） | ≈1.39×（速度为 ASM 的 0.72） | 同上 | 同上 | 09-24 | `0924__flydsl/DAY-SUMMARY.md:9-15` |
| **E3′ · A0 09-28（刷固件前最后一次，仍 VR 限频；分块 randn，同进程）** | | | | | | | | |
| aiter ASM bwd | prod，两种顺序 o1 / o2 | 7.654 / 7.666 | 718.3* / 717.2* | 1 | A0 sclk 1020–1061 MHz | 分块 randn，每种顺序一个进程，A/A 0.08% / 0.01% | 09-28 | `0928__a0_repro/REPORT.md` §1（:32-38） |
| 我们的 FlyDSL bwd r20 | 同上 | 10.600 / 10.605 | 518.7* / 518.5* | 1.385 / 1.383 | 同上 | 同上 | 09-28 | 同上 |
| 我们的 FlyDSL bwd r29（= r19h + u2n，B0 冠军） | 同上 | 11.010 / 11.015 | 499.4* / 499.2* | 1.438 / 1.437（比 r20 慢 3.9%，与 B0 方向相反；刷固件后同尺子为 1.196 / 1.201，与 r20 持平） | 同上 | 同上 | 09-28 | 同上；`§6.2` |
| **E4 · B0 09-27/28** | | | | | | | | |
| Primus-Turbo Triton 两内核 | 出厂 @`1cb2e183`，opcheck：kernel / 经模块 | 24.049 / 24.160 | 228.6* | ≈3.4×（对 ASM opcheck 7.107，不同进程） | B0 GPU0 | opcheck.py | 09-28 | `0927__b0/e2e/BUILD.md` §0 |
| aiter ASM bwd | 分块（尺子审计）/ job beat / opcheck kernel、经模块 / 旧交错尺子 | 6.911 / ≈6.70–6.73 / 7.107、7.223 / 6.504–6.763（PERF 表 2 次均值 6.504） | 795.6* / 816.8–820.7 / — / 845*（交错 6.504） | 1 | B0 GPU0/GPU3 | 经模块多出的约 0.12 ms 基本就是 GQA 求和（0.124 ms） | 09-27/28 | `0927__b0/ruler/bwd/REPORT.md`；`bwd/progress.md`；`e2e/BUILD.md`；`REPORT-0928.html` §2 |
| 我们的 FlyDSL bwd r20（B0 起点） | 分块 | 8.773 | 626.7* | 1.27×（同进程；旧交错尺子下 1.31×） | B0 GPU0 | 分块 | 09-28 | `ruler/bwd/REPORT.md` |
| 我们的 FlyDSL bwd r29（= r19h + u2n，B0 冠军） | job 分块 | 8.375 | 656.5 | 1.244×（ASM 的 80.4%，beat 817.0 TF/s） | B0 GPU3 | 分块 | 09-28 | `0927__b0/bwd/progress.md:15` |
| **E5 · A0 刷固件后** | | | | | | | | |
| aiter ASM bwd | randn 分块，o1 / o2 | 5.503 / 5.494（当天其它同进程复测 5.476–5.517） | 999 / 1001 | 1 | sclk 1872–2031 | 分块，同进程 | 09-30 | `0928__a0_repro/REPORT.md` §6.2；`0930__bwd/PROGRESS.md` |
| 我们的 FlyDSL bwd r29 | 同上 | 6.583 / 6.596 | 835 / 834 | 1.196 / 1.201 | 同上 | 同上 | 09-30 | 同上 |
| 我们的 FlyDSL bwd s6（手工 campaign 冠军，0.3.2；在 0.3.4.1 下 ISA 逐字节相同） | randn 分块，与 r29 A/A、ASM 同进程 | **5.295–5.300**（ASM 5.497–5.500） | ≈1037–1038 | **0.963**（ASM 的 103.8%） | ~1.4–1.8 GHz（受功耗墙限制） | 同上 | 09-30 | `0930__bwd/REPORT.md` §1 |
| 同上 | 10-02 randn：s6 / r29 | 5.295 / 6.567（ASM 5.513） | 1038.4* / 837.3* | 0.960 / 1.191 | A0，驱动 2412954 | 分块 | 10-02 | `1002__e2e/E2E-PLAN.md:56` |
| 同上 | 10-02 真实数据：s6 blk / s6 gb / r29 gb | 5.324 / 5.743 / 7.421（ASM 5.475 / 5.530 / 5.530） | 1032.7* / 957.4* / 740.9* | 0.972 / **1.038** / 1.342 | blk ~1.45–1.5 GHz；gb ~1.28 GHz | blk / gb | 10-02 | `1002__e2e/RESULT-realab.md` |
| A0 op-evolve bwd job 冠军 r24（= s6 + `DQT_VT_KEEP`，`armsrc/job_r24`） | job 分块尺子，与 beat 臂（ASM）同会话 | 5.292–5.295（job 记 1038.3 TF/s；同会话 beat 999.0 TF/s ≈ 5.504 ms） | 1038.4–1039.0* | ≈0.962（ASM 的 104.0%） | A0 | op-evolve benchmark.py（blocked 补丁） | 09-30 | `0930__bwd/REPORT.md:13,32`；`0930__bwd/PROGRESS.md:50`。job 自报的"1.4263×"增益是 fast 形状中位数噪声造成的假阳性，prod 实为 0.9986，树与 s6 等价 |
| 产品移植 FlyDSL bwd r29（flydsl 0.2.4） | prod | 6.618（ASM 5.360） | 830.8* | 1.235 | A0 | 分块 randn，同进程 | 09-30 | `0930__port/PR_BODY.md:31-34` |
| Primus-Turbo main Triton dense 后端 | 出厂（bench，TestID 54） | 24.49（同一张表里 FlyDSL r29 为 6.78） | 224.50 | ≈4.5×（跨工具，对 ASM 5.36–5.50） | A0 | `bench_attention_turbo.py` | 09-30 | `0930__port/runs/bench_turbo_triton.csv` |

**产品分支对 main Triton dense 后端的整体加速**（E5，A0，flydsl 0.2.4，`bench_attention_turbo.py`，09-30）：在 FlyDSL 门放行的 36 个形状上（causal 18 + 非因果 18）取几何平均，causal fwd **4.13×** / bwd **3.37×**，非因果 fwd **3.47×** / bwd **3.58×**（`0930__port/PR_BODY.md:36-41`；用两份 CSV 复算结果一致）。

**引用这两份 bench CSV 时要注意**：bench 自带的参考值在非因果 s8192 上本身有问题，所有后端都报 dq ≈12 dB 并标 FAIL（TRITON 15 行；FLYDSL 9 行，FLYDSL 另有 54 行是门不放行的 ERROR），而 CPU fp32 参考给出 52 dB。这些 FAIL 是参考值的问题，不是 kernel 错（`0930__port/runs/bench_turbo_triton.log:48`、`bench_turbo_flydsl.log:72`；session fdc2534d .txt:474）。

#### b.2.3 没跑起来或被淘汰的后端

| 后端 | 结果 | 原因 | 来源 |
|---|---|---|---|
| aiter CK fmha（main 默认派发会落到这里） | 失败 | <ul><li>CK 只支持 CDNA：fwd 能跑通，bwd 在训练中途报 `invalid argument for fmha_bwd`</li><li>aiter 的 CK/HIP JIT 在 gfx1250 上 import 时编译失败（`ck_tile/core/config.hpp:576` "Only one target architecture can be defined"，共 9 个错误），随后打印 "CK and HIP ops are disabled"</li></ul> | `git show c1325c7e`；`OE: output/0911__fa_gfx1250_phase1/E2.md` §3 |
| HipKittens（main #462 只支持 gfx950；另有 udna1 的 gfx1250 移植） | 失败，判 RED | 卡在三道编译期硬墙：<ul><li>`reductions.cuh` 用了 gfx950 独有的 `permlane32_swap`（7 处），编不过</li><li>`mma_AB` / `mma_AtB` 仍派发到 wave64 MFMA，只有 `mma_ABt` 能编</li><li>`swap_layout` 在 wave32 下只覆盖一半寄存器</li></ul>GPU gate 256 个元素里错 126 个（worst abs 2.73）。72 个头文件中 53 个与 cdna4 逐字节相同 | `0914__campaign/RESULTS.md` §8；`0914__hk_udna1_gate/` |
| Gluon（main #469） | 不适用 | 只支持 gfx950，且只有前向（`DenseAttnFwdGluonBackend` 的门只认 gfx950）；从未在 gfx1250 上跑过 | `git show e18cf0a1`；session fdc2534d |
| Primus-Turbo 自带的 FlyDSL attention（gfx950） | 失败 | 依赖 MFMA 32x32x16、`ds_read_tr16_b64`、`permlane32_swap`，这三条在 gfx1250 上都报 "Cannot select"；`warp_size=64` 写死。另外原来的 `>=(9,5)` 门会把它派发给 gfx1250，只因 G=4 恰好被 `_gqa_group_ok` 拒掉才没炸 | `0914__campaign/RESULTS.md` §9；`f8c45dee` |
| torch SDPA FLASH（AOTriton） | 能跑，但最慢 | A0 限频下 fwd+bwd 合计 97.063 ms（约 79 TF/s，合计口径）。`c1325c7e` 的提交信息称在 B2 形状上测得 161.4 TF/s，Triton 为 220.6 TF/s（测量条件不明，见 a.2 #11） | `0913__opt_plan__claude/phase1/RESULTS.md` §1；`git show c1325c7e` |
| aiter Triton `mha_fused_bwd` | 数值错误，淘汰 | 合计 49.051 ms，但 dk 只有 −0.22 dB。另外 onekernel bwd 只改 `BLOCK_N1=256` 时 dq 只有 9.59 dB，属于静默错误 | `OE: output/0911__fa_gfx1250_phase1/E2.md` §4-6 |
| aiter ASM `*_pssk_perf.co` | 不可用 | dk/dv 与出货版逐位相同，但 dq 只有 5.84 dB；它也不在 dispatch CSV 里 | `0915__opt/PERF-CO-CLOSED.md` |

**要点**：
- **Triton 系**：最好的 Triton bwd（aiter 调优版、vendored 融合版加 ITT）在 A0 限频下仍是 ASM 的约 2–2.4 倍。其中 ≈2.30–2.40× 是跨尺子的比值：E1 tune 尺子（含约 1.4 ms autograd）对 E3 benchmark.py 的 bar 7.68 ms；2.01× 是同进程比值，但分母是 bring-up 的 ASM 读数 8.661 ms，这个读数偏慢。fwd 最好约 1.5 倍（E2 aiter Triton 调优上界 ≈1.49×，同机跨会话）。差距是结构性的：ASM 单个 workgroup 用满 320 KiB LDS 和 1024 VGPR，bwd 有 864 条 `v_wmma`，而 Triton 只分配 64 KB LDS。调 Triton 的参数补不上这个差距（`0913__opt_plan__claude/phase2/T2-ASM-BACKWARD-SPEC.md`）。
- **fwd 对时钟的敏感度**：A0 限频时代的 fwd 差距被时钟放大了。aiter FlyDSL fwd 对 ASM 的比值在 A0 限频下是 1.51–1.53×，刷固件后降到 1.125×。原因是 ASM fwd 几乎不随 sclk 变化，而 FlyDSL 随 sclk 变化。
- **op 级领先不一定能带到训练里**：在训练工作点（GEMM 突发之后，约 1.28 GHz），s6 从领先 2.8% 变成落后 3.8%，fwd r16 从落后 8% 变成落后 35%。

### b.3 e2e 性能（Llama-3.1-8B）

**统一配置**（除注明外）：BF16、单卡、MBS=GBS=4、seq 8192（每步 32,768 token）、AC none、torch.compile 关、32 层。"8L" 是 8 层裁剪版，只用于 A0 09-15/16 的 GEMM 阶梯。

**口径**：
- tokens/s 为单卡吞吐。
- TFLOP/s 取 torchtitan 自报值；带 † 的是按 torchtitan 口径由 tps 换算的（32L 约 57.9 GFLOP/token，8L 约 16.84）。
- torchtitan 的 MFU 用 A100 312 TF/s 作分母，在这块卡上全部不可用。

| 配置/后端 | GEMM 路径（nkfix 前/后） | step ms | tokens/s | TFLOP/s | 机器 | 日期 | 来源 |
|---|---|---|---|---|---|---|---|
| **E1 · A0 VR 限频** | | | | | | | |
| 名义上是 turbo Triton 调优，很可能实际是 flex（`converters: []`，事后推断，未逐条重验）；对照组 flex；torchtitan v0.2.2（`73a0e6979`；`0914__campaign/RESULTS.md:65` 记作 0.1.0，是笔误，见 a.1） | rocBLAS（`TORCH_BLAS_PREFER_HIPBLASLT=0`），nkfix 前 | 133,700 | 245（flex 244）；09-15 打开 converter 后真正走 turbo（ASM fwd，ASM bwd 开/关）两臂同为 244 | 14.17 | A0 ≤1100 MHz | 09-13 / 09-15 | `0913__opt_plan__claude/phase1/RESULTS.md` §10；`0914__campaign/HANDOFF.md` §4；`0915__opt/RESULTS.md:150-176` |
| turbo（ASM fwd + 融合 Triton bwd），3 步，n=1 | hipBLASLt 镜像库（修了 LIBPATH，反向仍落在 MT32x16x32），nkfix 前 | ≈17,000 | 2,027 | 117.4 | 同上 | 09-15 | `0915__opt/BLAS-FINDING.md:44-58` |
| 同上，20 步：ASM bwd 关 / 开（未固定种子，n=1） | 同上 | 16,500 / 17,600 | 1,984 / 1,858 | 114.9† / 107.6† | 同上 | 09-15 | `0915__opt/E2E-AB.md:20-51`（−6.8% 后来被证明落在 e2e 多模态噪声的模态间距之内，"来自显存足迹"的归因也已撤回，见 `E2E-AB.md:180-282`） |
| 8L：ASM bwd 关 / 开，seed 1234，每臂 n=9 | 同上（三模态，sd 3.9%） | ≈5,347 | 6,128 / 6,111（−0.28%，不显著） | 103† | 同上 | 09-15 | `0915__opt/E2E-AB.md:172-260` |
| 8L：ASM bwd 关 | nkfix v1（dgrad 改成 TN） | 2,824 | 11,604（n=9，sd 0.42%） | 195† | 同上 | 09-15 | `0915__opt/GEMM-NN-FINDING.md:288-327` |
| 8L：ASM bwd 关 / 开 | nkfix v3（dgrad 和 wgrad 两个操作数都改） | ≈868† / ≈770† | 37,734 / 42,534（+12.72%，由 +14.40% 更正）；GEMM 阶梯值 37,746（n=5） | 635.4† / 716.3† | 同上 | 09-16（09-17 更正） | `0915__opt/BOTTLENECK-SHIFT.md`；`PROGRESS-REPORT-0917.html`；`0915__opt/GEMM-WGRAD-FINDING.md:9-22` |
| 8L：ASM fwd + ASM bwd | nkfix 规则 1 + 规则 3（wgrad 用 FlyDSL GEMM，离线表） | 707 | 46,374（n=3；同会话规则 2 拷贝法为 42,445） | 781† | 同上 | 09-16 | `0915__opt/FLYDSL-AB.md`；`NAN-FINDING.md:34-41`（49,878 那次含 NaN，已撤回） |
| 32L 生产配置：ASM fwd + ASM bwd | 规则 1+3 / 规则 1+2 | ≈2,332† / ≈2,655† | 14,050 / 12,340（n=1；相对 1,984 分别为 7.08× / 6.22×） | 813.5† / 714.5† | A0，负载下 1001 MHz | 09-16 | `0915__opt/RESULT-32L.md`（11,608 / 5.85× 那次含 NaN，已作废） |
| **E2 · B0 09-14（torchtitan 0.2.2，20 步取后 8 步中位）** | | | | | | | |
| flex eager | hipBLASLt 镜像库（torch.mm 113 TF/s），nkfix 前 | 13,700 | 2,392 | 138.5 | B0 GPU3 | 09-14 | `0914__campaign/RESULTS.md` §10 |
| flex + compile 默认 | inductor 把 matmul 留在 aten.mm | 14,280 | 2,294 | 132.8 | 同上 | 09-14 | 同上 |
| flex + compile + Triton GEMM | `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON` | 3,410 | 9,602 | 556.1 | 同上 | 09-14 | 同上 |
| ~~"turbo attention" + compile + Triton GEMM~~ | 同上 | 2,480 | ~~13,204~~（三臂 A/B 的 13,928 vs flex 9,543 也一并撤回） | 764.7 | 同上 | 09-14 | `0914__campaign/RESULTS.md` §14、§17；`HANDOFF.md` §4。**已撤回**：训练 import 的是镜像里的 0.4.1.dev12，且 `converters: []` 使 attention 仍是 flex，差异来自其它 turbo patch（float8 / mx linear） |
| **E3′ · A0 09-28（刷固件前，仍 VR 限频；op 级同日 sclk 1020–1061 MHz，e2e 未记录 sclk。与 E1 不是同一棵树、同一套 nkfix）** | | | | | | | |
| 32L，3 个 arm 同进程按步交替（a0_p3a，90 步）：ASM / FlyDSL r6+r20 / r16+r29。平台状态已作废 | nkfix（`NKFIX_CHECK=1`）+ 镜像库（fwd 树的 `_env.py` 在 import 时把 `HIPBLASLT_TENSILE_LIBPATH` 改指到宿主库；本进程 FlyDSL 从 step 2 起才调用） | 1,947 / 2,070 / 2,071（对 ASM 1.067 / 1.065；new/old 0.999） | 16,830† / 15,830† / 15,822† | 974† / 917† / 916† | A0，驱动 7.1.1-2397345（E5 e2e 为 7.1.0-2412954） | 09-28 | `0928__a0_repro/REPORT.md` §2（同批另有 1 次 NaN、2 次挂卡，见 §5.3） |
| 24L，2x2 分解（a0_p4a，100 步，无 profile）：ASM / r6+r20 / r16+r29 / r16+r20 / r6+r29 | 同上 | 1,483.4 / 1,578.8 / 1,579.7 / 1,569.9 / 1,590.5（对 ASM 1.064 / 1.065 / 1.058 / 1.072） | — | — | A0 | 09-28 | `0928__a0_repro/REPORT.md` §5.1：fwd r6→r16 每步 −8.9 / −10.8 ms（与 B0 一致），bwd r20→r29 每步 +11.7 / +9.8 ms（与 B0 相反） |
| **E4 · B0 09-28** | | | | | | | |
| turbo Triton 出厂 @`1cb2e183` / aiter ASM / FlyDSL r6+r20 | nkfix 前：镜像 hipBLASLt，反向落在 MT32x16x32，GEMM 占每步 15.6–16.5 s（96%） | 17,655 / 17,049 / 17,120–17,125 | 1,856 / 1,922 / 1,914（低档） | 107.5† / 111.3† / 110.8† | B0 GPU0，训练 sclk 约 1,811–1,826 MHz；ASM / FlyDSL 同进程（P2），turbo 是单独进程 P1，跨进程差里约 100 ms 是 GEMM 波动 | 09-28 | `0927__b0/e2e/RESULT.md` §0（:21-32） |
| 同上三者 | nkfix_b0 + transpose_triton（`NKFIX_CHECK=1`），每步 997 个 GEMM 全部落在 MT256x256x128 | 2,250 / 1,582 / 1,658 | 14,563 / 20,718 / 19,762 | 843† / 1,200† / 1,144† | B0 GPU0；ASM / FlyDSL 为同进程 n5（n6 为反序），训练 sclk 1,383 / 1,394 MHz；turbo 是单独进程 n7，对 ASM 是跨进程比较 | 09-28 | `0927__b0/gemm/REPORT.md` §0；`e2e/RESULT.md:1-7`；时钟见 `profile/REPORT.md:138` |
| 终版：FlyDSL fwd r16（r13ns）+ bwd r29 对 ASM，同进程 ABBA / BAAB 交替 | 同上 | 1,631 / 1,633 对 1,579 / 1,583 | 20,091 / 20,070 对 20,747 / 20,697（fly/asm 1.0323 / 1.0328） | 1,163† 对 1,201† | 同上（1,342 / 1,371 MHz） | 09-28 | `0927__b0/e2e/RESULT-final.md` §0 |
| **E5 · A0 刷固件后** | | | | | | | |
| 3 个 arm 交替（2 个进程 × 92 步）：ASM / FlyDSL fwd r16 + bwd **s6** / FlyDSL r16 + r29（0.3.4.1） | 镜像库 + nkfix（`NKFIX_CHECK=1`，约多 33 ms/步；BLAS guard 记录 0 次改写） | 1,350.6 / 1,349.4–1,349.9 / 1,388.5–1,389.6 | 24,262 / 24,275–24,283 / 23,581–23,599（对 ASM 为 **0.999** / 1.027） | 1,405† / 1,406† / 1,366† | A0，训练 sclk 中位约 1.50 GHz，驱动 7.1.0-2412954 | 10-02 | `1002__e2e/RESULT-e2e.md`；`e2e/runs/TABLE.1002_095504.md` |
| 教程：产品分支 FlyDSL（fwd r16 + bwd r29，flydsl 0.2.4），20 步 | 镜像库 + nkfix + transpose_triton（`NKFIX_CHECK=0`；上一行 3 个 arm 都是 `NKFIX_CHECK=1`，检查约占 33 ms/步，不能拿 1,361.9 直接对 ASM 的 1,350.6，折算后分支约为 ASM 的 1.03×，见下文要点） | 1,361.9（按提交版本复跑 1,360） | 24,061（24,099） | 1,393† | A0，sclk 最高 2.36 GHz | 10-05 | `WT: docs/gfx1250_llama31_8b_e2e/README.md` §0、§7 |
| 同上，但缺 `transpose_triton.py` | nkfix 退回 torch copy | 1,689 | 19,406 | 1,124† | A0 | 10-05 | 同上 §7 |
| main + `PRIMUS_TURBO_ATTN_BACKEND=TRITON` + nkfix（**估算，未实测**） | 同上 | ~1,900–2,050 | ~16,000–17,000 | — | A0 | 10-05 | 同上 §7 |
| 参考，非我们的机器：JIRA MI455X flex eager / compile+autotune；MI355X AITER | 未知 | FA path 755 ms/步（eager） | 19,795（工单正文；trace 中 eager 为 24,651）/ 27,891；MI355X 21,351（trace 中 21,521） | — | JIRA 环境 | 09-01 前后 | `0927__b0/e2e/RECON.md:39-41`；`0915__opt/JIRA-TRACE-ANALYSIS.md:13-14,45` |

**e2e 要点**：
- **nkfix 之前**，e2e 被 GEMM 卡死，attention 在每步里只占很小一块，后端差异基本看不出来：
  - A0 32L rocBLAS：不到 1%。09-13 那次（名义 turbo Triton，很可能实为 flex）按 35.7 ms/层 × 32 = 1.14 s 算，约 0.85%（`0913__opt_plan__claude/phase1/RESULTS.md:266`）；09-15 ASM 路径按 11.7 ms/层 × 32 ≈ 374 ms 算为 0.28%（`0915__opt/RESULTS.md:169`），但这个 11.7 ms 里的 ASM bwd 用的是后来撤回的 shim 读数 10.160 ms，实际占比更低；
  - A0 8L hipBLASLt 镜像库：3.3%（profile 实测，`0915__opt/BOTTLENECK-SHIFT.md:14`）；
  - B0 32L：1.4–5.4%（`0927__b0/e2e/RESULT.md:37`）；
  - A0 09-13 turbo 与 flex 相差 0.4%；B0 09-28 FlyDSL 对 ASM 为 1.0045。
- **nkfix 之后**，attention 在每步中的占比：
  - turbo Triton 43.9%（B0）；
  - ASM 17.0–17.4%（B0）、14.9%（A0）；
  - FlyDSL 20.5–21.8%（B0）、15.3%（A0，s6）。
- **FlyDSL 对 ASM 的单步比演进**：B0 上 r6+r20 为 1.0475–1.0500，r13ns+r20 为 1.0403–1.0408，r16+r29 为 1.0323–1.0328；A0 10-02 上 r16+r29 为 1.027，r16+s6 为 **0.999**。在 r16+s6 这组里：
  - bwd 比 ASM 每步快约 4 ms：相邻配对的中位差为 4.50 / 4.56 ms（两个进程），即 `RESULT-e2e.md` 正文的 4.5；按两个 arm 各自的中位数相减为 3.8–4.4 ms（`RESULT-e2e.md` 表中 166.5 对 162.2 / 162.6，即 3.9–4.3）。
  - fwd 每步慢 9.2 ms（相邻配对 9.18 / 9.24）。只看 attention，剩下的差距在 fwd。
  - 对不上的地方：attention 合计 fly 每步多约 4.3–5.2 ms，单步却快 0.7–2.0 ms。也就是说，attention 以外的部分在 fly arm 上每步少约 6 ms（`e2e/runs/analysis.1002_095504.txt` 的 step − FA 两个进程都是 −6.3 ms；flyr29 对 asm 为 −21 / −22 ms）。这部分差异出在 attention 以外的哪里、有多少是噪声，没有拆解。
- 产品分支 `dev/lhz/llama31_attn_opt` 交付的仍是 bwd r29（flydsl 0.2.4 下 op 级为 ASM 的 1.235×），s6 还没有移植进去。
- **教程的 e2e 数不能直接对 ASM**：教程的 1,361.9 ms/步是 `NKFIX_CHECK=0`，ASM 的 1,350.6 ms 是 10-02 的 `NKFIX_CHECK=1`。两者只差 0.8%，但这不等于分支只比 ASM 慢 0.8%。按 CHECK=1 约多 33 ms/步折算（这是 B0 trace 实测值，教程沿用；`0927__b0/gemm/REPORT.md` §5），ASM 在 CHECK=0 下约 1,318 ms，分支约为它的 1.03×，即约慢 3%（推算，未实测）。这与同为 CHECK=1 的 10-02 r16+r29（flydsl 0.3.4.1）对 ASM 的 1.027 一致。
- **PR 正文引用的是跨机器的旧数**：`0930__port/PR_BODY.md:43` 写"step time is 1.032x that of the ASM kernels"，没有注明机器。这是 B0 09-28 的 r16+r29 读数（1.0323 / 1.0328，flydsl 0.3.4.1 树）；A0 10-02 上同一组合为 1.027，r16+s6 为 0.999（`1002__e2e/RESULT-e2e.md`）。PR 正文需要更新。
- **跨机器不可比**：B0（ASM 1,579 ms）和 A0 刷固件后（ASM 1,351 ms）的 e2e 绝对值不能直接比，两边平台和训练时钟都不同（B0 终版 r16+r29 运行为 1,342 / 1,371 MHz，n5/n6 运行为 1,383 / 1,394 MHz；A0 10-02 中位约 1.50 GHz）。
- **NaN 更正**：nkfix 在 A0 09-16 有 21% 的运行出现 NaN，因此几个数字被更正或作废：38,043 → 37,746，+14.40% → +12.72%，49,878 和 11,608 作废。B0 重写版跑了 8 次、486 步，0 次 NaN。

### b.4 性能最优的 aiter ASM 有哪些问题

| 问题 | 现象/影响 | 我们的处理 | 来源 |
|---|---|---|---|
| bwd 没有可调用的 gfx1250 Python 入口 | <ul><li>`can_impl_fmha_v3_bwd` 只认 gfx942/gfx950；C++ host 虽然有 5 处 gfx1250 特判，但经 aiter 正常入口调用会静默拿不到 ASM bwd</li><li>kernarg：`dqdkdv` 为 704 B、44 个字段、每字段 16 B 对齐；`odo` 紧凑打包且走 kernarg 预加载，无法反汇编确证；`dq_convert` 头文件写 192 B，ELF 是 208 B；所有 stride 都是字节</li></ul> | <ul><li>离线从 ELF 读出调用契约，自写约 674 行发射器</li><li>ABI 自检抓到手抄字段表漏了最后 7 个字段（含 `mask_x/mask_y`）。这两个字段 aiter 自己在 mt=2 路径上传的是未初始化栈值，我们传 0，安全</li><li>`odo` 用 NaN 预填后验证</li></ul> | `0922_summary/ASM-ATTENTION.md` §1.1；`0913__opt_plan__claude/phase2/T2-ASM-BACKWARD-SPEC.md`；`skill: references/baselines.md` §3.2 |
| GQA 越界写（最实质的正确性 bug） | pssk 的 grid 是 `(kv_tiles, nhead_q, batch)`，却按 q head 去索引 kv 尺寸的 dk/dv，G=4 时 4 个 WG 无同步地争同一个 tile：<ul><li>s=1024：dk/dv 静默损坏，SQNR 约 −0.3 dB，dq 仍为 52.24 dB</li><li>s=256：越界跨页，进程 fault（dmesg 无记录，不伤卡）</li><li>定位实验（ratio=4，dk/dv 按 kv head 索引）：dk −0.94 / dv −0.65 dB；ratio=1 时全对</li><li>s=8192 用 `dkdv_heads=kv`：触发 `GCVM_L2_PROTECTION_FAULT`，按进程重置队列，未花 AC-cycle</li></ul> | `dkdv_heads='q'` 加 host 规约：<ul><li>dk/dv 显存 0.125 → 0.500 GiB，峰值多 1.254 GiB</li><li>耗时：fp32 求和版 +0.482 ms（bwd 的 5.2%）；bf16 reduce 版 0.2225 ms（bar 的 2.88%）</li><li>dk/dv 精度下降 1.7–1.9 dB（50.55 / 51.09）</li></ul>已提交 vendor report | `0917__flydsl/VENDOR-REPORT-aiter-gfx1250.md:15-38`；`0922_summary/ASM-ATTENTION.md:52-67`；`GQA-WORKAROUND-COST.md:50-65`；`skill: references/baselines.md` §3.3 |
| 不是 drop-in：3 个 kernel，fp32 dq_acc 每次都要清零 | <ul><li>odo → pssk（514 条 `buffer_atomic_add_f32` 写 fp32 dq_acc）→ dq_convert（ts_dq=64）</li><li>不清零就得到静默垃圾 dq</li><li>每次调用 6 个 dispatch</li><li>每层 dq_acc 536 MB，dk/dv 各 268 MB</li><li>B0 e2e 上适配开销 9.2 ms/步（GQA 求和 3.84、dq_convert 2.27、清零 1.62、odo 1.44），另需 1 GiB 常驻 scratch</li></ul> | `asm_backward()` 内部分配并清零；scratch 按形状缓存（进程级 1.000 GiB）。FlyDSL bwd 直接输出 hkv 个头的 bf16，没有这些开销 | `T2-ASM-BACKWARD-SPEC.md`；`0927__b0/e2e/RESULT.md` §0；`e2e/BUILD.md` §0 |
| 只有预编译 .co：资产稀薄、不可调、改不了 | <ul><li>gfx1250 共 52 个 .co（gfx950 有 1466）；bwd 只有 6 个（gfx950 有 124）</li><li>bwd 只有 a32 + pssk、bf16、hd128、batch mode。MI355 用的 `a16_psskddv` 在 gfx1250 上不存在；没有 varlen bwd（dispatch 只有 mode=0，seqstart 传 nullptr）；没有 fp16 / swa / group</li><li>fwd 有 hd64、hd128 和 varlen，但 hd128 没有 sink 变体</li><li>VGPR 1024、LDS 327,680 B、wg 1024 全部用满</li></ul> | 只调用和包装，不改汇编。建议向 aiter 申请 psskddv / a16 变体（未见回复）。缺 varlen 和不可维护是 FlyDSL 立项的直接理由 | `0922_summary/ASM-ATTENTION.md` §4；`VENDOR-REPORT-aiter-gfx1250.md` §2；`0915__opt/BACKEND-STRATEGY.md:24-48` |
| 测量伪影：10.160 ms（已撤回） | <ul><li>09-15 `tune_attention.py` 里的 `_AsmFwdAsmBwd` autograd shim 调用 `asm_backward` 时没传 `hip=` / `scratch=`，每次 3× hipModuleLoad，新分配约 1 GiB；另含约 1.4 ms autograd</li><li>该值被误标为"产品路径"，一直用到 09-24</li><li>由此推出的 541 TF/s、11.726 ms/层、4.757×、1.74×、"FlyDSL 已到 0.92× bar" 全部偏高</li></ul> | 以 op-evolve beat 臂同会话实测为准：09-17 为 7.6134 ms，09-24 普查为 7.6766 ms（n=224）/ 约 716 TF/s。FlyDSL 当时的真实进度是 0.70–0.72× | `skill: references/baselines.md` §0、§5；`0924__flydsl/DAY-SUMMARY.md`；`0915__opt/E2E-AB.md:284-302` |
| 产品集成：每次反向都重新加载，scratch 反复分配，显存紧张 | <ul><li>产品路径 `asm_dense_backward` 没传 `HipModule`，32L 每步 96 次 hipModuleLoad 且从不卸载；另外每次调用都按调用分配 dq_acc（fp32）和按 q head 的 dk/dv。op 级 harness 看不到这两项</li><li>~~每层新分配约 1 GiB，使 32L e2e 慢 6.78%（n=1）~~ **已撤回**：scratch 的 key 不含层索引，缓存后是进程级 1.000 GiB，不随层数增长；32L 不缓存时两臂 reserved 逐位相同（380.06 GiB）；6.78%（1,858 vs 1,984 tps）是 n=1，落在后来测出的 e2e 多模态噪声的模态间距之内（簇间 1.8% / 6.8%，跨运行 sd 约 3.9%）</li><li>加了 scratch 缓存后，32L（显存 88%）在第 3 步 SIGBUS 挂卡（+1.38 GiB 贴着 88.30% 的上限），花了 1 次 AC-cycle</li></ul> | HipModule 改为进程级单例（`514e4a21`）；scratch 复用（`b281b46b`）。两项修复后 8L、seed 1234、每臂 n=9：ON 对 OFF 为 −0.28%（0.15× sem），e2e 无可测差异；资格门因此维持默认关闭 | `0915__opt/E2E-AB.md:94-133,180-282`；`ASM-ATTENTION.md` §1.4 |
| 约 0.85 ms 固定开销，短序列反而更慢 | ASM bwd 在 s1024→8192（工作量 16×）上耗时 0.88 / 0.91 / 1.00 / 1.34 ms；s1024 在任何并行度下都输（0.537–0.660×） | 资格门定为 seqlen ≥ 2048（`f8155ae9`）。但这个开销里有一部分就是逐次 hipModuleLoad，修复后 9 个支撑点需要重测，一直没做 | `0915__opt/RESULTS.md:99-124`；`E2E-AB.md:296-302` |
| dq 用 fp32 原子累加，非确定 | <ul><li>514 条 `buffer_atomic_add_f32 SCOPE_DEV`；dq 的 run-to-run SQNR 在 fast 为 113.0 dB、prod 为 98.0 dB，不是逐位一致</li><li>请求 `is_deterministic` 时 aiter 返回 −1，退回 CK</li><li>这与我们"200 次逐位一致"的合同冲突，迫使 FlyDSL 走 7-GEMM，结构因子 1.386–1.408×</li></ul> | 09-25 拆分确定性门（`c741fa8d`）：dk/dv 仍要求逐位一致，dq 改为 run-to-run SQNR ≥ 70 dB。FlyDSL s6 的 dq run-to-run 逐位一致 | `0925__flydsl/AITER-5GEMM-STUDY.md` §2；`0923__flydsl/hint.md` h18 / h41；`0930__bwd/REPORT.md` |
| fwd 的 LSE 布局与树内两内核 bwd 不兼容 | ASM fwd 输出自然对数 LSE，形状 [B,Hq,Sq]，`return_lse=False` 时也照写；树内两内核 bwd 读的是打包的 [B,Hq,2*Sq]。混用不报错，梯度会"平滑地错" | 设合取门：ASM fwd 只与融合 bwd 一起启用，决策存在 ctx 里（`0e5cb743`） | `0913__opt_plan__claude/phase2/T7-ASM-FORWARD.md` (b)；`0914__campaign/RESULTS.md:124-126` |
| fwd 入口的依赖链脆弱 | <ul><li>`fmha_fwd_with_sink_asm` 首次调用 JIT 编译 C++（需要 hipcc）；架构探测靠 rocminfo</li><li>能否启用取决于无关的 psutil：e2e 容器缺 psutil 时会静默退回 Triton fwd，op 级损失 9.8%，51/51 测试照样通过</li><li>`AITER_LOG_LEVEL=ERROR` 把横幅吞掉，导致 B0 09-14 训练里 ASM fwd 实际从未被调用</li><li>09-13 在 A0 上撞到 `import jax`，09-14 在 B0 上不存在，原因未核实</li></ul> | 惰性一次性 import，失败则永久关闭；容器补装 psutil；由门自己写 trace 文件证明被调用；加硬关开关 `20f2ccde` 供 A/B | `0914__campaign/RESULTS.md:116-130,267-293`；`T7-ASM-FORWARD.md` (a) |
| 版本冲突，无法作为上游产品后端 | <ul><li>aiter 固定 flydsl 0.3.2，Primus-Turbo 固定 0.2.4，两者不能在同一进程</li><li>aiter 的 gfx1250 dispatch 走不到 ASM，只能 ctypes 手工发射</li><li>测试镜像里没有 aiter：产品分支上回落 aiter 的 28 个测试失败</li></ul> | 侧装 `~/.local/flydsl032` 和 `flydsl0341`；beat 臂按文件路径加载。产品分支 `dev/lhz/llama31_attn_opt` 改为交付 FlyDSL，ASM 只保留为对照 arm | `0917__flydsl/STAGE1-FWD.md` §4；`0930__port/PR_BODY.md` Testing；session fdc2534d 2026-10-05 |
| causal / 布局 / 形状限制 | <ul><li>bwd 只有 causal bottom-right（`causal_br`，CSV mask=2）和非因果两种 .co</li><li>门只放行 self-attention：Sq=Skv 时两种掩码语义等价，cross-attention 被拒</li><li>要求 BSHD、最后一维连续、hq%hkv==0，不支持 sink、滑窗、bias、alibi、dropout、fp16（fwd 门共 12 个拒绝条件）</li><li>aiter 在 hd128 上只测过 gqa=8</li></ul> | 按能力设门。gqa=4 在 B0 上验证：32 个 q head 的 SQNR 都在 53.54–53.72 dB。非因果放行，提速 1.71× | `0914__campaign/RESULTS.md:118-128,273-275`；`0915__opt/PROGRESS-REPORT.html` Q3 |
| dk/dv 精度余量小 | host 规约时每个 partial 先舍入成 bf16，dk/dv 为 50.2–51.0 dB（prod 50.61 / 50.83），门槛 50 dB；确定性两内核为 52.29 / 52.70 | 接受。FlyDSL 在 kernel 内规约，达到 52.4–52.6 dB | `AITER-5GEMM-STUDY.md` §2；`0915__opt/TWOKERNEL-SWEEP.md:76-78` |
| 时钟特性：低频下 bar 显得更硬，训练工作点下 ASM 不降速 | <ul><li>ASM fwd 几乎不随 sclk 变化：A0 1100 MHz 时 1.552 ms，B0 时 1.396 ms，只差 1.11×（同代码的 Triton bwd 差 2.01×）；B0 上 1350/2350 MHz 为 0.97×；刷固件后时钟 ×1.75，ASM fwd 只快 1.24×</li><li>PMC 显示 ASM fwd 跑在功耗墙上，kernel 内有效时钟只有 1.36 GHz</li><li>因此限频时代 FlyDSL fwd 显得慢 1.29–1.53×，刷固件后只慢 1.08×</li><li>在训练工作点，ASM 基本不变而 FlyDSL 变慢</li></ul> | 只用同进程比值；新增 gb 尺子（GEMM 突发后测）；以 e2e 为准；把降低 FlyDSL 对时钟的敏感度列为下一步 | `0915__repro__c07/RESULTS.md`；`0927__b0/profile/REPORT.md` §2.1；`0930__roofline/REPORT.md`；`1002__e2e/RESULT-realab.md` |
| 读数随发射方式剧烈变化，尺子会选错 | <ul><li>同进程 ASM fwd：连续跑 20 次为 1.438 ms，每次 sync 为 1.266，fwd 后紧跟 bwd 为 1.225；profile 下 iso / blk / eburst / layer 分别为 1.22 / 1.45 / 1.13 / 1.00 ms</li><li>逐次交错计时让 FlyDSL 对 ASM 系统性偏低：fwd 约 25%，bwd 约 3.4%</li><li>fast 形状是 launch-bound（bwd beat 0.089 ms），几何平均被它拉高，导致 B0 bwd r27 被误判为 target_met</li></ul> | 改为分块计时（h40 / h66）；判据改为 proxy 和 prod 各自 ≥ beat（h69） | `0927__b0/e2e/BUILD.md` §3；`ruler/REPORT.md`；`OP-EVOLVE-SUGGESTIONS.md` #13 |
| 工具链看不全 | <ul><li>llvm-objdump 解不出 TDM 指令（opcode 0x31，显示为 `.long 0xd031…`），ASM fwd 的预取深度只能推断</li><li>rocprofv3 的 VGPR_Count 只显示一半（512，实际 1024）</li><li>gfx1250 .co 一度"反汇编不出"</li></ul> | <ul><li>按 dword 解码描述符；从 ISA 读 `.vgpr_count`</li><li>09-25 用 `--mcpu=gfx1250` 拿到 bwd 反汇编：864 条 v_wmma、514 条原子、26 个 barrier</li><li>09-29 刷固件后 ATT 可用，能看到逐指令 stall</li></ul> | `0925__flydsl/fwd-isa/REPORT.md`；`0930__bwd/notes/report_asm.md`；`AITER-5GEMM-STUDY.md` §1 |
| ASM 本身也离 roofline 很远（这意味着可以超越它） | <ul><li>PMC：bwd 主 kernel 7.71e6 cyc/SIMD，只到 5-GEMM 矩阵下限的 34%</li><li>每个 SIMD 只有 1 个 wave；每步 TDM 等待 227 cyc、split barrier 208 cyc、原子背压 430 cyc</li><li>fwd 到下限的 72%，但有效时钟只有 1.36 GHz</li></ul> | s6 借鉴 TDM 3 级 LDS ring，并提前把下一轮 B 操作数读进寄存器，bwd 达到 ASM 的 103.8% | `0930__roofline/REPORT.md`；`0930__bwd/notes/report_asm.md`；`0930__bwd/REPORT.md` §2 |
| 一次未诊断的卡状态退化 | 09-14 T7 期间，ASM fwd 一度完全丢掉优势（turbo 只慢 13%），之后没有复现，也没有诊断 | 训练负载下的稳定性没有单独确认 | `0914__campaign/RESULTS.md:128-130` |
| 文档里流传的错误归因 | <ul><li>"B0 上 ASM bwd 对 fused 无优势（8.840 vs 8.848）"是错的：两臂都是 fused Triton bwd，B0 09-14 从未跑过 ASM bwd</li><li>"A0/B0 时钟差 2.1×、attention 只差 1.15×，所以几乎不受时钟限制"只对 ASM fwd 成立：bwd 那一项比的是 A0 的 shim 10.160 和 B0 的 fused 8.835</li><li>0914 的 `report.html` 把 aiter Triton 行标成了"手写 ASM"</li></ul> | 本节一律以同代码、同进程的比值为准。注意 `skill: references/baselines.md` §8 **还没有更正**：仍写着"On B0 ASM bwd showed no edge over fused Triton (8.840 vs 8.848)"，并把 B0 的 8.835 当作带 shim 开销的 ASM bwd，需要更新 | `skill: references/baselines.md:272-293`（错误原文）；`0914__campaign/HANDOFF.md:436-450`（§15：`--impl asm` 与 `--impl fused` 两臂本是同一条路径）；`0914__campaign/RESULTS.md:103`（`--impl asm` = ASM 前向 + 融合反向）；`0915__repro__c07/RESULTS.md` |

**小结**

1. **op 级排名**：倍数只在同一 epoch 内可比，下面按 epoch 分开给，不把不同 epoch 的倍数排在同一条线上。
   - bwd，E5（A0 刷固件后，分块，同进程）：FlyDSL s6 0.963–0.972×（op-evolve job 冠军 r24 在 job 尺子上 ≈0.962×；gb 训练工作点 1.038×）≈ aiter ASM > FlyDSL r29 1.20×（gb 1.342×）。同期 bench 中 main Triton dense 出厂约 4.5×（跨工具）。
   - bwd，E1（A0 限频，tune_attention.py，都是对 E3 bar 7.68 ms 的 ≈ 比值）：树内 vendored 融合 Triton 加 ITT ≈2.30×（`1cb2e183` 时 ≈2.63×）≈ aiter Triton 调优冠军 ≈2.40× > flex ≈3.1× > aiter Triton 出厂 ≈3.2–3.6× > turbo Triton 调优 ≈4.2× > turbo Triton 出厂 ≈5.9–6.4× > SDPA（只有 fwd+bwd 合计 97.063 ms）。
   - 两组之间用 A0 限频下的 FlyDSL 读数衔接：E3 的 r20 ≈1.39×、E3′ 的 r29 1.44×，都快于 E1 最好的 Triton（≈2.3×）。时钟状态相同、尺子不同，但两者相差约 6–7 ms，远大于 E1 尺子里约 1.4 ms 的 autograd 开销。B0 E4 的 turbo 出厂 ≈3.4× 来自跨进程 opcheck，不并入上面的排序。
   - fwd：aiter ASM > FlyDSL r13ns（E5 1.08×，gb 训练工作点 1.35×；E4 1.08×；E3′ 1.29×）> aiter FlyDSL m32x8（E5 1.13×；E1/E3/E3′ 1.50–1.53×）> aiter Triton 调优 > turbo Triton 调优 > flex > turbo Triton 出厂。后四者在 E2（B0 满频）/ E1（A0 限频）分别为 1.49× / 2.09×、1.94× / 2.66×、2.67× / 4.8×、3.7× / 6.8×，两个 epoch 内顺序相同。aiter Triton 出厂在 E1 有两次互相矛盾的读数（≈4.3× / ≈2.4×），不参与排序。
2. **e2e**：A0 10-02 上 FlyDSL r16+s6 与 ASM 持平（单步 0.999），r16+r29 为 1.027（同进程）；B0 上出厂 turbo Triton 的单步时间约为 ASM 的 1.42×（2,250 vs 1,582 ms，14,563 vs 20,718 tps），但这是跨进程比值（turbo 为单独进程 n7，ASM 为 n5），不像 fly/asm 那样是同进程配对。nkfix 之前任何 attention 后端的差异都被 GEMM 掩盖。
3. **为什么自研 FlyDSL**：ASM 最快，但它是黑盒——没有 Python 入口、有 GQA 越界写、bwd 只有 6 个 .co、没有 varlen、dq 非确定、与 flydsl 版本冲突、改不了也调不了。Triton 受 LDS 和寄存器结构限制，最好也只到 ASM 的约 2 倍时间。aiter 的 FlyDSL 只有 fwd，且慢 1.5×（09-17 立项时 A0 限频下的同进程读数；刷固件后为 1.13×）。所以 09-17 按"可维护、补 varlen、确定性、可上游"立项。
4. **结果**：bwd 已反超 ASM（s6），fwd 在 op 级仍差约 8%，在 e2e 中每步约差 9 ms。
