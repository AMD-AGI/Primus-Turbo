# Llama-3.1-8B attention 在 MI455X (gfx1250) 上的优化进展总结（截至 2026-10-07）

- **范围**：Llama-3.1-8B 训练用的 attention。算子形状 b4 s8192 hq32 hkv8 d128 bf16，causal（bottom-right；Sq=Skv 时与 top-left 等价），BSHD，GQA 4。e2e 为 Llama-3.1-8B 32 层（09-15/16 部分实验为 8 层）、BF16、单卡、MBS=GBS=4、seq 8192（每步 32,768 token）、AC none、torch.compile 关。
- **FLOP 约定**：现行口径为 op-evolve `tools/op_flops.py`，每次调用 fwd 2.199292e12、bwd 5.498229e12（5-GEMM 名义计数），fwd+bwd 7.697e12。09-13..09-16 的文档只报 fwd+bwd 合计（7.697e12 ÷ 总耗时）；7-GEMM 实发口径、旧 harness 的 fwd ≈2.233e12 等口径本报告不混用，见附录 A.3。
- **数据截止**：最后一次实测是 2026-10-05 的产品分支 e2e 教程（(d) 另记 10-06 一次其它项目的挂卡，仅作风险参考）。本报告写于 2026-10-07，同日经一次完整性审查后修订过一次（审查前的全文暂存于 `_work/SUMMARY.v1.md`）。
- **声明**：本报告只整理已有记录（`output/` 下的各日报告、op-evolve 账本、会话记录、git 历史），未运行任何 GPU 任务。A0 的 GPU 当前由同事使用。
- **文件清单**（都在 `Primus-Turbo/output/1007__summary/` 下）：
  - `SUMMARY.md`：本报告，定稿；与下面任何文件不一致时以 SUMMARY.md 为准。
  - `rounds.csv`：每一轮一行的明细，198 行，不合并（与 `parts/rounds.csv` 相同）。
  - `REPORT-1007.html`：本报告的 HTML 版，自包含，浏览器直接打开即可。
  - `parts/`：分节草稿。SUMMARY.md 由 `0_head.md`（标题、说明、一页结论、时间线）、`a_mainline.md`、`b_backends.md`、`c_rounds.md`、`d_problems.md`、`z_appendix.md`（附录 A/B/C）按此顺序直接拼接而成；另有 `rounds.csv`。内容以 SUMMARY.md 为准。
  - `_work/`：草稿区：会话文本导出（`sessions/`）、各子任务的中间文件与修改前备份（`scratch/`）、v1 分节稿（`v1/`）、审查前全文 `SUMMARY.v1.md`、会话导出脚本（`tools/`）。可以整个删除，不影响上面的文件（`_work/.gitignore` 已忽略其全部内容）。
  - `tools/`：`build.sh` 把 `parts/` 拼成 SUMMARY.md、复制 rounds.csv 并重新生成 HTML（只用 CPU）；`md2html.py` 是 HTML 转换脚本。改了 `parts/` 之后跑 `tools/build.sh` 即可。
- **记号**：A0 = `heliosr-1b114-c07-1`（本机，单卡）；B0 = `ctheliosp-1b112-a37-1`（4 卡）。A0 在 09-29 刷固件之前处于 VR 限频（sclk 上限 1100 MHz），**09-29 之前 A0 的绝对数既不能和之后的 A0 比，也不能和 B0 比**。四节对同一时钟阶段的叫法不同（A0-VR / E1 / A0-T / A0-限频 等），对照见附录 A.1。时间一律为 UTC。

## 0. 一页结论

1. **起点：main 原样在 gfx1250 上默认跑不起来。** upstream main `c1325c7e`（#481）下，torch 2.11 一 import 就崩（`register_opaque_type`）；默认派发落到只支持 CDNA 的 CK（镜像里没有 aiter 时第一个 step 就 ImportError，装了 aiter 则 bwd 在训练中途报 `invalid argument for fmha_bwd`）；gfx1250 上的测试全部被 skip。唯一能手动 pin 的 Triton 后端照搬 CDNA 的单一 config：A0-VR 上 fwd+bwd 58.424 ms / 131.7 TF/s（bench0910，合计口径；分支树 tune 尺子 59.642 ms / 129.0），比 flex 慢约 1.9×，其中 dkdv 一个 kernel 占 61.5%。e2e 方面，A0 此前从没跑完过一个 step；09-13 跑通后只有 245 tok/s、133.7 s/step（GEMM 退到 rocBLAS，27.4 TF/s），而且 `converters: []` 让 turbo attention 根本没进 e2e。详见 (a)。
2. **适配后的后端（只在同一时钟阶段内比较）。** A0-VR 09-13 的 bake-off（fwd+bwd 合计，耗时由短到长）：aiter Triton 加 2 个旋钮 31.207 ms、flex 31.337、aiter Triton 出厂 34.565、turbo Triton 调优 36.405、turbo Triton 出厂 59.642、SDPA FLASH 97.063。Triton 系最好的 bwd 耗时约为 ASM 的 2–2.4 倍（≈，跨尺子：Triton 是 E1 `tune_attention.py` 的读数，计时含 autograd 管路；ASM 取 E3 benchmark.py 的 bar 7.68 ms，或偏慢的同进程 bring-up 读数 8.661 ms），fwd 最好约 1.5 倍（≈，B0 09-14）；这是结构性差距（ASM 单个 WG 用满 320 KiB LDS 和 1024 VGPR）。aiter 预编译 ASM 是最快的现成实现：A0-VR 上 fwd 1.5724 ms / 1398.67 TF/s、bwd 7.6766 ms / ≈716 TF/s（09-24 普查）；A0 刷固件后 fwd 1.257 ms / 1750、bwd 5.503 ms / 999 TF/s（09-30，分块 randn）。详见 (b)。
3. **最快的 ASM 不能直接拿来用。** bwd 没有 gfx1250 的 Python 入口（`can_impl_fmha_v3_bwd` 不认 gfx1250），只能自写约 674 行发射器；GQA 下按 q head 越界写 dk/dv（ratio=4 时 dk −0.94 / dv −0.65 dB），绕法是 `dkdv_heads=q` 加 host 规约，峰值多占 1.254 GiB，反向多 0.482 ms（A0-VR，+5.2%）；gfx1250 只发布了 52 个 `.co`（gfx950 有 1466 个），bwd 只有 6 个，没有 varlen；dq 用 fp32 原子累加，结果不确定；一次调用要 3 个 kernel、6 个 dispatch；aiter 固定 flydsl 0.3.2，Primus-Turbo 固定 0.2.4，两者不能同进程，做不了上游的产品后端。另外，09-15 测得的 ASM bwd 10.160 ms / 541 TF/s 是 autograd shim 造成的伪影，09-24 才撤回；在此之前 FlyDSL 的进度被报成 ASM 的 0.92×，实际是 0.70–0.72×。这些问题是 09-17 决定自研 FlyDSL 的理由。
4. **FlyDSL bwd：从 ASM 的 1/12.6 做到 103.8%（两端跨了固件阶段）。** 09-17 从零手写，A0-VR 上 r0 为 96.05 ms / 57.24 TF/s（同会话 ASM 的 1/12.6）；A0-VR op-evolve 20 轮后，r20 到 511.42 TF/s（10.75 ms，约为 ASM 的 0.71–0.72×）；B0 上 r29（= r19h + u2n）656.5 TF/s，为 ASM 的 80.4%。同一份 r29 代码、同一把分块尺子，A0-VR 上（09-28）只有 ASM 的 69.5%，A0 刷固件后（09-30）为 83.6%：按 r29 估算，标题里的进步有约 14 个百分点来自刷固件（FlyDSL 比 ASM 更吃时钟），不是代码。09-30 手工攻关 s1–s6，核心是 TDM 3 级 LDS ring 加上把下一轮 B 操作数提前读回寄存器，把 bwd 做到 **5.295–5.300 ms / ≈1037–1038 TF/s，为 ASM（5.497–5.500 ms）的 103.8%**（A0 刷固件后，分块 randn，同进程，逐位确定）。10-02 用真实数据复核，s6/ASM 时间比在 blk 尺子下为 0.972，在训练工作点的 gb 尺子下为 1.038。s6 仍发 7 个 GEMM，比 ASM 多做约 39% 的矩阵运算。
5. **FlyDSL fwd：仍慢约 8%，在训练工作点下慢 35%。** A0-VR fwd job 从 r0 的 935.04 TF/s 做到 r11 的 1111.15 TF/s（+18.8%；为同会话 beat 1407.8 TF/s 的 78.9%，对 bar 1398.67 TF/s 为 79.4%）。B0 上投机 softmax 在 randn 上快 4.6%，但在真实数据和训练时钟下反而更慢；r16 把投机关掉（即 r13ns），在真实数据加训练时钟下快 12–23%。当前 fwd 冠军 r16 = r13ns：A0 刷固件后 09-30 测得 1.353 ms / 1626 TF/s，ASM 1.257 ms，时间比 1.076（ASM 的 92.9%）；10-02 真实数据下 blk 1.081、gb **1.349**。fwd 是现在剩下的主要差距。
6. **e2e：FlyDSL 已与 ASM 持平。** A0 10-02（驱动 7.1.0-2412954，32 层，nkfix，`NKFIX_CHECK=1`，三个 arm 同进程交替，训练 sclk 中位约 1.50 GHz）：fwd r16 + bwd s6 单步 1,349.4 / 1,349.9 ms、24,283 / 24,275 tok/s，为 ASM（1,350.6 ms、24,262 tok/s）的 **0.999**；fwd r16 + bwd r29 为 1.027。按 CUDA event 拆分（A0 上 kineto trace 无效），attention bwd 每步比 ASM 快约 4 ms（相邻配对差 4.5 ms，两 arm 中位数之差 3.8–4.4 ms），fwd 慢约 9.2 ms，attention 合计多约 4.3–5.2 ms，单步却快 0.7–2.0 ms；这约 6 ms 的出入（两进程 step − FA 均为 −6.3 ms）落在 attention 以外的部分或计时噪声里，没有拆解。s6 比 r29 每步省下 attention bwd 约 53 ms，单步快 37–39 ms，同样有约 14–16 ms 未拆解。此前 B0 09-28 的终版 r16 + r29 为 1.0323 / 1.0328（20,091 对 20,747 tok/s）。产品分支（r16 + r29，flydsl 0.2.4）在 10-05 的教程里实测 1,361.9 ms/step、24,061 tok/s，但那是 `NKFIX_CHECK=0`，不能和上面 `NKFIX_CHECK=1` 的 ASM 1,350.6 ms 直接比（直接比像是只慢 0.8%）：按 `NKFIX_CHECK=1` 每步约 +33 ms（B0 实测）折算，分支约比 ASM 慢 3%（推算，未实测），与同为 r16 + r29 的 flyr29 臂的 1.027 相符。
7. **e2e 的头号杠杆其实是 GEMM（hipBLASLt）。** 第一个问题是库路径错位：402 个文件放在 `library/gfx1250/` 子目录里，设 `HIPBLASLT_TENSILE_LIBPATH` 之后 A0-VR 32 层从 244 到 2,027 tok/s（8.3×）。第二个问题是 NN（dgrad）和 wgrad 布局缺纯 bf16 调优库，落到 GEMV 用的 tile `MT32x16x32`，只有 50–80 TF/s，GEMM 占单步 94–96%。nkfix（用 `TorchDispatchMode` 改写 `aten::mm` 的操作数布局，命中 `MT256x256x128`）让 A0-VR 8 层从 6,128 到 37,746 tok/s（6.16×），B0 32 层 ASM 臂从 2,016–2,022 到 20,718 tok/s（10.3×）。GEMM 修好之后，ASM/FlyDSL 的 attention 才占到单步的约 15–22%，op 级的改进才能在 e2e 里看到。代价是 A0 上带 nkfix 的运行有 21% 出 NaN（8/39 对 0/35，p=0.004），因此撤回了 49,878、11,608/5.85×、38,043/6.21×、+14.40% 等数字；B0 重写版跑了 8 次、486 步，0 次 NaN。`TORCH_BLAS_PREFER_HIPBLASLT` 至少在三处被静默改写；宿主 hipBLASLt 库与多起 NaN 和挂卡相关，10-02 起（h85）只用镜像库。
8. **挂卡（wedge）和 AC 断电是最大的时间成本。** A0 在 09-11..10-02 因挂卡做了 27 次人工断电或重启（(d) d.1.4 跨来源去重；09-11 的两次若是热重启则为 25；算上 09-29 驱动/固件不匹配时那次无效 AC 则为 28），09-16 一天就有 9 次；09-21..22 五轮的墙钟时间里有 82% 耗在挂卡和等 AC 上。按 dmesg 首行分三类：A 类是卡上 fp32 Tensile 参考 GEMM 越界，B 类以 `INVALIDATE_TLBS` 超时起头，C 类是候选 kernel 越界读引发的故障风暴。先后提出的四个前兆判别式全被证伪，结论是没有可用的预测指标，只对挂卡本身告警。真正定位并消除的诱因有：卡上的 fp32 参考 GEMM（refcache、h50）、单进程跑多个 shape（改成一 shape 一进程）、训练步内 autotune（改成离线建表）、`converters: []` 导致 inductor autotune、PC sampling（永久禁用）、新 kernel 直接上 prod 和宿主 hipBLASLt 库（h85：先 toy 后 prod，读 VGPR 只用 compile-only，只用镜像库）。另外，风险按启动次数计价（09-16 启动期失败 3/14，即 21%）。断电还两次截断了 git 对象（6 个和 255 个），由此定下"每轮结束都 push"。
9. **尺子和判据本身多次改写了结论。** 逐次交错计时让 FlyDSL/ASM 比值系统性偏低（fwd 约 25%，bwd 约 3.4%），改成分块计时后撤回了两个假赢；randn 与真实数据、op 时钟与训练工作点（GEMM 突发后约 1.28 GHz）下的结论可以相反；框架判据出过三次错（A0 r15 靠算术平均被晋升、B0 r27 的 geomean 误判 target_met、A0 r24 的 fast 中位数假阳性），都是手工纠正的。A0 09-29 刷固件（VBIOS 630A→700E，SMU 125.7.1→125.12.0，驱动同时从 7.1.1-2397345 换成 7.1.0-2411946）后，同一份代码 op 级快了 1.24–1.67×，09-29 之前 A0 的绝对数全部作废。教程引用的 e2e 1.44×（1,947→1,351 ms）不纯是刷固件的效应：1,947 ms 出自 09-28 那批后来整组作废的运行（a0_p3a），该进程在 step 2 首次加载 FlyDSL fwd 树时，树里的 `_env.py` 把 `HIPBLASLT_TENSILE_LIBPATH` 改指宿主库（之后的 GEMM 是否受影响未证实）；1,351 ms 是 10-02 测的，其间驱动又在 10-01 换成 7.1.0-2412954。
10. **还没做完的事和下一步。** ① fwd：op 级仍慢约 8%，训练工作点下 1.349×，e2e 每步约 9 ms。② s6 还没移植进产品分支 `dev/lhz/llama31_attn_opt`（分支里是 r29，op 级为 ASM 的 1.235×）；PR 还没开，`PR_BODY.md` 引用的 1.032 是 B0 09-28 的跨机器旧数，需要更新。③ 训练工作点的 gb 计分尺子已设计、未安装；fwd job 已从 B0 备份恢复、未上线；两者都需要用户批准。④ 5-GEMM 融合版 w4f 受整芯片原子吞吐限制，已暂停。⑤ 还没解释的现象：e2e 里 ASM 比两把 op 尺子测得都快（怀疑输入在 MALL 里是热的，没测）；B 类挂卡和启动期 MES 故障的原因；A0 上 nkfix 出 NaN 的根因。另有若干 skill 文档需要更正，见 (d) 节末尾。

## 时间线

| 日期 | 机器 | 阶段 | 关键结果 |
|---|---|---|---|
| 09-10..09-13 | A0（VR 限频，sclk ≤1100 MHz） | main 原样 bring-up、后端 bake-off、Triton 调优 | main 出厂 Triton 58.424 ms（bench0910）/ 59.642 ms（tune），比 flex 慢约 1.9×；两个 config 旋钮加 vendored aiter 融合 bwd，做到 24.435 ms（2.44×）；e2e 首次跑通：245 tok/s、133.7 s/step（rocBLAS 27.4 TF/s，attention 实际是 flex）；09-11 两次挂卡重启（E3 外积、E4 PC sampling）；09-13 第 5 次 wedge（从 09-04 起累计），转到 B0 |
| 09-14 | B0（4 卡，2133–2244 MHz） | B0 复现阶梯、接入 ASM fwd、ITT | 出厂 28.121 → 10.235 ms / 752 TF/s（2.75×）；ASM fwd 1.410 ms，比 Triton fwd 快 1.82×；hipBLASLt 113 对 Triton GEMM 1190 TF/s，compile 加 Triton GEMM 让 e2e 到 9,602 tok/s（4.01×）；当天的 turbo attention e2e 数字（13,204 tok/s、1.459×）事后撤回；HipKittens udna1 判 RED |
| 09-15 | A0（VR） | 回 A0 复现、接入 ASM bwd、修正 hipBLASLt 路径、nkfix v1 | 出厂 55.785 → 19.285 ms（2.893×）；自写 ASM bwd 发射器并绕开 GQA 越界写（当时的 10.160 ms 后来撤回）；`HIPBLASLT_TENSILE_LIBPATH` 让 32 层从 244 到 2,027 tok/s；nkfix v1 让 8 层从 6,128 到 11,604 tok/s；当天 5 次 AC |
| 09-16 | A0（VR） | nkfix v3 与规则 3、NaN 普查 | 8 层 37,746 tok/s（6.16×）→ 46,374 tok/s（7.57×）；32 层 14,050 tok/s（7.08×，n=1）；GEMM 修好后 ASM bwd 的 e2e 收益为 +12.72%；带 nkfix 的运行 21% 出 NaN，撤回 4 个数字；当天 9 次 AC |
| 09-17 | A0（VR） | 转向 FlyDSL：调研、Stage1、bwd 三个 kernel、bwd job 发车 | aiter FlyDSL fwd 2.373 ms 对 ASM 1.569 ms（慢 1.51×）；odo/dkdv/dq 三个 kernel 首跑即通过（141–159 dB）；bwd job r0 96.05 ms / 57.24 TF/s（同会话 ASM 7.6134 ms / 722.2）；当天交机，r1 中断 |
| 09-21..09-25 | A0（VR） | bwd op-evolve r1–r23 | r12 499.15 TF/s；r20 冠军 511.42 TF/s（10.75 ms，ASM 的 0.71–0.72×）；09-24 ASM bar 普查 7.6766 ms / ≈716 TF/s，撤回 10.160 ms；7 次挂卡 AC（A/B/C 三类）；09-25 dq 确定性门放宽为 run-to-run ≥70 dB；4-wave 骨架只有 0.664×，搁置 |
| 09-23..09-27 | A0（VR） | fwd stage 2 扫参、fwd op-evolve r0–r13 | fwd bar：ASM 1.5724 ms / 1398.67 TF/s，aiter FlyDSL fwd 2.4005 ms / 916.16；fwd job 从 r0 的 935.04 做到 r11 冠军 1111.15 TF/s（1.979 ms，+18.8%；为同会话 beat 1407.8 TF/s 的 78.9%，按上面的 bar 1398.67 为 79.4%） |
| 09-27..09-28 | B0（4 卡；GPU1 09-28 起挂死） | 四卡攻关：fwd r5–r20、bwd B0 r24–r32、尺子审计、nkfix e2e | 交错尺子改为分块尺子，撤回假赢；fwd 冠军 r16 = r13ns（randn，job 尺子 1443–1455 TF/s，约为 ASM 的 93%）；bwd 冠军 r29 656.5 TF/s（ASM 的 80.4%）；nkfix_b0 让 e2e ASM 臂到 20,718 tok/s（10.3×）；终版 fly(r16+r29)/ASM 单步 1.0323 |
| 09-28 | A0（VR，最后一次） | 在 A0 复现 B0 冠军 | fwd r13ns/ASM 时间比 1.29，bwd r29 为 ASM 的 69.5%；e2e 5 个训练进程里 1 个 NaN、2 次挂卡（各 1 次 AC）；09-30 复测后整组作废 |
| 09-28..09-29 | A0 | 管理员换驱动、刷固件 | 换驱动后 PSP `LOAD_TOC` 失败，卡无法初始化，AC 也无效；随后刷 VBIOS 630A→700E、SMU 125.7.1→125.12.0、dkms 7.1.0-2411946，sclk 档位变为 500/2356/2400；09-29 之前 A0 的绝对数作废 |
| 09-30 | A0（新固件，op 级 1.74–2.03 GHz） | 同尺子复测、roofline、bwd 手工攻关 s1–s6、A0 bwd job r24/r25、产品移植 | fwd r13ns 1.353 对 ASM 1.257 ms（1.08）；bwd r29 6.583 对 5.503 ms（83.6%）→ s6 5.295–5.300 ms（ASM 的 103.8%）；产品分支（flydsl 0.2.4）fwd 1.292 ms、bwd r29 6.618 ms，对 main Triton 的 causal 几何平均加速 fwd 4.13×、bwd 3.37× |
| 10-02 | A0（驱动 7.1.0-2412954） | 真实数据 op A/B、e2e 三臂对比、A0 bwd r26/r27、gb 尺子设计 | s6/ASM blk 0.972、gb 1.038；r16/ASM blk 1.081、gb 1.349；e2e（`NKFIX_CHECK=1`）fly(r16+s6) 1,349.4 ms、24,283 tok/s，为 ASM 的 0.999；r27 的 opt agent 只为读 VGPR，就用宿主 hipBLASLt 库把两个从未上过卡的 s6 基底 cluster multicast 变体（A_g74 改 k_dkdv、B_g82 改 k_dqg；1ee0dd59 提交信息和 WEDGE-1002.md 误记为 w4f）直接跑在 prod 上，11:40 挂在 A_g74 上，新增 h85 |
| 10-05 | A0（新固件） | 产品分支 e2e 教程 | 分支版 fwd r16 + bwd r29（`NKFIX_CHECK=0`）：1,361.9 ms/step、24,061 tok/s（按提交版本复跑 1,360 / 24,099），与 10-02 `NKFIX_CHECK=1` 的 ASM 1,350.6 ms 口径不同，按 +33 ms/步折算约慢 3%（推算）；漏掉 `transpose_triton.py` 时 1,689 ms；教程写的固件更新前后 ASM e2e 1,947→1,351 ms（1.44×）不纯是刷固件的效应（1,947 出自 09-28 作废批次，进程中途 LIBPATH 被改指宿主库，驱动也换过），见结论第 9 条 |
| 10-07 | — | 本总结 | 只整理已有记录，未运行任何 GPU 任务；同日经一次完整性审查后修订 |

## (a) 优化前：直接用 main 分支跑有哪些问题

范围：2026-09-10..09-14 的起步阶段。本节的 "main" 指当时 upstream main 的头 `c1325c7e`（#481，2026-09-09，第一次给 gfx1250 加 Triton dense attention 后端）。第一个工作分支 `gfx1250-attn-dispatch-and-tuning` 就是从它分出来的（`git merge-base f8c45dee origin/main` = `c1325c7e`）。表中"现状"一栏都以本地 `origin/main` = `8cda13c7`（09-30）为准核对过。目标算子是 Llama-3.1-8B attention：b4 s8192 hq32 hkv8 d128 bf16 causal（bottom-right），BSHD，GQA G=4。

记号与口径：
- **A0-VR**：A0（`heliosr-1b114-c07-1`）在 09-29 刷固件之前的 VR 限频态，sclk 上限 1100 MHz，负载下约 0.97-1.07 GHz。**A0-RF**：09-29 刷固件之后，op 级约 1.74-2.03 GHz，训练中约 1.5 GHz。**B0**：`ctheliosp-1b112-a37-1`，负载下 2133-2244 MHz。三种状态的绝对数互相不可比，几种"倍数"也不能互相顶替：① 跨机器 A0-VR→B0，同一份 Triton 类代码 5 行 like-for-like 为 1.88-2.01x（中位 1.93x；FINAL-TABLE 全 7 行为 1.88-2.12x，中位 1.97x）。这是两张卡之间的比值，B0 自己的 dmesg 也有 VR 限频警告（0914 repro 估其代价约 9%），所以它不是 A0 自身的限频代价（0914__repro__c07/RESULTS.md:32-38, 124-130）。② 同样跨机器的 ASM attention，只有 fwd 的 1.118x 成立（A0-VR 09-15 1.572 ms 对 B0 09-14 安静窗口 1.406 ms，两边都是 ASM fwd、tune 尺子）。0915__opt/JIRA-TRACE-ANALYSIS.md:148-156 写的"attention 跨机只差 1.12-1.15x"（bwd 1.150x；合计 11.726 对 10.236 ms，1.146x）不成立：bwd 一项比的是 A0 的 shim 伪影 10.160 ms（09-24 撤回）和 B0 的融合 Triton bwd 8.835 ms（09-14 B0 还没有 ASM bwd，`--impl asm` 是 ASM 前向 + 融合反向），不是同一个 kernel。两边都用融合 Triton bwd 时，bwd 为 17.735/8.835 ≈ 2.01x，合计 19.285/10.235 = 1.88x（0915__repro__c07/RESULTS.md:17；0914__campaign/RESULTS.md:102-103；0914__campaign/HANDOFF.md:246；(d) d.3）。③ A0 自身刷固件前→后：op 级同代码同方法为 1.24-1.67x（两种顺序的均值）。e2e 的 1.44x 不是干净的同方法对照，不纯是平台差异：前值出自已作废的 09-28 运行，那个进程中途被 FlyDSL fwd 树把 `HIPBLASLT_TENSILE_LIBPATH` 改指到宿主库；两次运行之间驱动和 e2e 套件也都换了。它只作参考（见 a.1 表后的说明）。
- FLOP 口径：fwd 2.199292e12，bwd 5.498229e12，fwd+bwd 7.697e12 / call。当时的文档只报 fwd+bwd 合计，这个合计就等于两者之和。表里"分方向 TF/s（换算）"一栏是本节按现行口径算出来的。
- 尺子：**tune** = `tools/gfx1250/tune_attention.py`（取 CUDA event 中位数，20 iters / 5 warmup，每个 rep 冲刷 256 MiB L2，out/dq/dk/dv 四个张量分别对 fp32 参考算 SQNR）。**bench0910** = Primus 侧 attention bench 在 09-10 的日志（`Primus:output/0910__fa_opt/attn_bench_gfx1250.log`）。**E2** = op-evolve phase-1 的 `E2_bench.py`（warmup 5，iters 20，重复 3 次）。**bat** = `benchmark/ops/training/bench_attention_turbo.py`（09-30）。
- 来源路径：不带前缀的相对路径都在 `Primus-Turbo/output/` 下。`0913/` = `0913__opt_plan__claude/`；`ph1/` = `op-evolve:output/0911__fa_gfx1250_phase1/`（op-evolve = `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/`）；`Primus:` = `/home/lihuzhan/code/2026_0828__primus/Primus/`；`wt-llama31 README` = `wt-llama31/docs/gfx1250_llama31_8b_e2e/README.md`（10-05 教程）。

### a.1 当时的平台状态

| 项 | 优化前状态（09-10..09-14） | 后续变化 / 备注 | 来源 |
|---|---|---|---|
| 机器 | A0 = `heliosr-1b114-c07-1`，独占 1× gfx1250（MI455X，DID `0x75C1`，SKU `M4500001`）。256 CU，wave32，每 CU 320 KB LDS，432 GiB HBM，PPT0 功耗上限 2500 W | 09-14 起加用 B0 = `ctheliosp-1b112-a37-1`（4× gfx1250，镜像相同）。09-14 当天 B0 的 GPU0 有 120 条 MES 超时（故障始于 uptime 14647，早于当天工作），当天把它从调度中摘除，只用 GPU1-3。09-27/28 的 B0 campaign 又用回了 GPU0：09-28 终版 e2e 就跑在 GPU0（容器 fa-g0） | ph1/HARDWARE-ISSUE.md Environment；0914__campaign/HANDOFF.md:285,422；0927__b0/e2e/RESULT-final.md:1 |
| 时钟（最关键的变量） | VR 限频。每次 amdgpu init 都打印 `WARN: GPU is throttled, expect performance decrease. VR.`。sclk DPM 只剩 500 / 1100 MHz 两档，MAX_CLK = 1100。同一张卡 09-04 负载下还能跑 1699-1703 MHz（上限 2400）。限频最早见于 09-10 22:38 那次开机，之后重启也清不掉。负载实测：attention 1067 MHz / 1227 W（功耗只用了 49%），Triton GEMM 1008 MHz / 1770 W，说明卡被时钟顶住，而不是被功耗墙挡住 | "限频代价"有三种口径，量的不是同一件事，不能互相顶替：① 09-13 按时钟比（09-04 的 1699-1703 MHz 对 1100 MHz 上限）估约 1.65x（原文没给算式，按这两个数算约 1.55x）；② 09-14 文档用 B0 上 5 行同代码对照得 1.88-2.01x（中位 1.93x），写成"1.65x → ~1.95x"，但那是 A0-VR→B0 的跨卡比值（B0 自身也有 VR 警告，代价约 9%），不是 A0 自己的限频代价；③ A0 自身刷固件前→后同代码对照：op 级 1.24-1.67x（两种顺序的均值）；e2e 1.44x 不纯是平台差异，还混有 BLAS 库环境和 e2e 套件的差异，驱动也多换了一次（见下表及表后说明）。09-29 管理员刷 VBIOS（630A→700E）、SMU（125.7.1→125.12.0）后解除：开机仍打印 VR 警告，但 DPM 表不再截断到 1100 MHz。**09-29 之前 A0 的绝对数，既不能和之后的 A0 比，也不能和 B0 比** | ph1/E1.md §2, §4；0913/phase0/PLATFORM-ESCALATION.md「The finding」；0914__repro__c07/RESULTS.md:32-38, 124-130；0928__a0_repro/REPORT.md §6.3；wt-llama31 README §7 |
| 固件 / 驱动 / 内核 | VBIOS `113-M4500001-630A`。amdgpu `7.1.1.31300009`（drm 3.65.0），SMU fw `125.7.1`，IP 为 `gfx_v12_1_0` / `mes_v12_1_0` / `psp_v15_0_8` / `smu_v15_0_0`。内核 `6.14.0-37-generic`，cmdline 里有 `modprobe.blacklist=amdgpu`：每次开机都要手动 `sudo modprobe amdgpu`，否则没有 `/dev/kfd`，torch 报 "No CUDA GPUs are available" | 09-28/29 换驱动并刷 VBIOS（amdgpu-dkms 7.1.0-2411946）；10-01 又换成 7.1.0-2412954 | ph1/HARDWARE-ISSUE.md Environment；0913/PROGRESS.md:36-47；0928__a0_repro/REPORT.md §6.3 |
| 稳定性 | 09-04 到 09-13 共 wedge 5 次，签名都一样：MES failed to respond → 驱动 reset 卡在 `wait for reset ack` → D 状态任务堆积 → 只能重启主机，重载驱动不行。前两次与本项目无关：09-04 是 HipKittens GEMM；09-06/07 是 grouped-GEMM job，`SMU: No response` 857 次。09-11 两次是我们触发的：E3 在 GPU 上对 8192² 全量 mask_mod 求值 → GPU Hang；E4 用 rocprofv3 PC sampling → memory fault → MES 挂死。第 5 次在 09-13 约 13:30 UTC，当时开着 4 个 worker 的队列，又反复用 `pkill -9` 打断正在跑的内核，疑为诱因。卡 wedge 后，`rocm-smi`、`ps … wchan`、`pgrep`、`docker exec` 都会挂住，只有加 timeout 的 dmesg 能用 | 整个 campaign 期间（09-11..10-02），A0 因挂卡而做的人工断电或重启合计 27 次。算上 09-29 驱动/固件不匹配时那次无效 AC，是 28 次；09-11 的两次如果是热重启，则为 25 次。按日分布见 (d) d.1.4 | ph1/HARDWARE-ISSUE.md §2.1；0913/phase2/INCIDENT-2026-09-13-wedge.md；0913/PROGRESS.md |
| 镜像 / 软件栈 | 用过两个镜像。`primus-turbo:gfx1250-20260831-extended-v2`：09-11 的 E1/E2 用它，含 amd-aiter 0.1.14.post1。`amdprimus/amdprimus:gfx1250-20260910`：09-11 的 E3/E4 和 09-13 之后都用它，image id `6e656de79e6c`，不带 aiter。两个镜像都是 torch `2.11.0+rocm7.14.0a20260625`、triton 3.6.0、HIP 7.14（7.14.60850）、python 3.12.3。09-13 把 torchtitan 依赖装进去后 `docker commit` 成 `fa-tune:deps` | 09-17 起把 flydsl 0.3.2 侧装到 `~/.local/flydsl032`，后来升到 0.3.4.1 | ph1/E1.md、E2.md、E3.md 头部；0913/phase1/RESULTS.md 头部；0914__repro__c07/RESULTS.md「复现环境」 |
| Primus-Turbo | upstream main 头 = `c1325c7e`。它新加的 Triton dense 后端只在 `is_gfx1250` 时接受调用。镜像里另有一份 `/workspace/Primus-Turbo` 的 editable 安装 `primus_turbo 0.4.1.dev12`（09-09 构建），09-10 的第一份 bench 测的就是它 | 它会遮蔽自己的 checkout，见 a.3 #10 | git log；bench0910；0914__campaign/RESULTS.md:277-300 |
| aiter | `amdprimus` 镜像不带 aiter，09-13 起改用 `git clone` 的 `aiter-src @ffa945f9`（09-13 08:41 UTC clone）。09-11 的 E2（`extended-v2` 镜像里的 amd-aiter 0.1.14.post1）观察到：aiter 的 CK/HIP JIT 在 gfx1250 上 import 时就编译失败（`Only one target architecture can be defined` 等 9 个错误，约 130 行 stderr，每次起进程约 7 s），然后自动降级为只剩 Triton ops。`aiter-src @ffa945f9` 下是否同样如此，没有找到独立记录 | 09-17 07:34 UTC 更新到 `6963ae9d` | ph1/E2.md:3-4, §3（:99-105）；aiter-src reflog |
| FlyDSL | 宿主机没装 flydsl；镜像里有 flydsl（09-16 确认已装，09-17 SURVEY 核实版本为 0.2.4）。Primus-Turbo 的 `setup.py:529-535` 在 gfx1250 构建时跳过 flydsl，TODO 写的理由是 "Triton 3.7.0 and flydsl 0.2.4 does not support gfx1250"。aiter 则 pin `flydsl==0.3.2` | 09-16 首次验证推翻了这个 TODO：在 A0 的 `fa-repro` 容器（镜像 `fa-tune:deps`）里 `import MmaOpGFX1250_WMMAType` 成功（session 11052787，09-16T00:48Z；HANDOFF 的 0916 补充）。09-17 的 SURVEY §5 查包内容成文：0.2.4 已经带 `MmaOpGFX1250`、`ds_load_tr16_b128`、`tdm_ops`。upstream #508（09-21）改成在所有 arch 上都装 0.2.4 | git show c1325c7e:setup.py；0914__campaign/HANDOFF.md:133-141；session 11052787；0917__flydsl/SURVEY.md §5；git show origin/main:setup.py:531-535 |
| torchtitan / e2e 配置 | Primus 的 `third_party/torchtitan` 是 v0.2.2（`73a0e6979`）。09-13 的 e2e 配置是 `examples/torchtitan/configs/MI455X/repro_l8b_bf16_mbs4_seq8k_{v5,turbo,turbo_ac}.yaml`：flavor `8B_flex`，`converters: []`，`compile.enable: false` | 09-13 A0 与 09-14 B0 都是 torchtitan v0.2.2。`0914__campaign/RESULTS.md:65` 把 09-13 那次 A0 运行记成 "torchtitan 0.1.0"，判为笔误：这个说法只此一处、后来的引用都抄自它；09-13 当天改过的配置（`_turbo.yaml` 09-13 05:31 UTC、`_turbo_ac.yaml` 06:51）第 28 行注释写的是 "torchtitan 0.2.2 会传 enable_gqa"；子模块 09-10 03:54 checkout 到 `73a0e6979`（tag v0.2.2）后 reflog 再没动过；session 541e7bc3 引用的也是 `third_party/torchtitan` 的源码 | Primus git（`git log -- third_party/torchtitan` = `f96b1fbb`，07-17 升级到 v0.2.2；子模块 reflog）；上述配置文件 :28 |
| BLAS | `amdprimus` 镜像里，任何 matmul 都抛 `HIPBLAS_STATUS_INVALID_VALUE`（`Cannot read …/library/TensileLibrary_lazy_gfx1250.dat`），只能设 `TORCH_BLAS_PREFER_HIPBLASLT=0` 退到 rocBLAS。`extended-v2` 镜像设了 `HIPBLASLT_TENSILE_LIBPATH` 后 hipBLASLt 能跑，但各种布局都只有 60-92 TF/s。09-11 E1 测得 Triton（inductor）bf16 GEMM 1002.7 TF/s @1008 MHz（4 个形状中最好），当时被当作这张卡的 compute roof。HBM 实测 6.46-6.48 TB/s（4 种 PyTorch elementwise 模式中最好），只有 amd-smi 标称 23,347 GB/s 的 28%，原因未定：VR 限频、不是调优过的 STREAM、标称值本身不可达，三者分不开 | 根因前后改写了两次，见 a.3 #3。1002.7 后来被证明不是 roof：09-15 同卡同时钟（A0-VR）hipBLASLt 的 NT 路径 8192³ 实测 1502 TF/s，nkfix 改布局后单形状最高 1628（dgrad mlp）/ 1742 TF/s（lm_head，含转置开销） | ph1/E1.md §5（:200-222）；ph1/HARDWARE-ISSUE.md S2；0913/phase1/RESULTS.md §5, §11；0915__opt/GEMM-NN-FINDING.md:70, 84-86, 118 |
| Profiling | 容器里 ROCm 7.14 的 rocprofv3 运行时只暴露 51 个计数器。申请 27 个，只有 8 个可用，没有字节、矩阵单元、stall、占用率这几类计数器。混进未知计数器时 exit 0，但对应的列被静默丢掉。PC sampling 每次尝试都会引发 GPU 故障（E4 记了 2 次，09-13 的文档记为 3 次），其中一次导致 wedge，之后全面禁用 | 09-29 刷固件后 ATT 可以用了 | ph1/E4.md Headline, §6；0913/phase0/PLATFORM-ESCALATION.md |

**A0 自身刷固件前后的同代码对照**（A0 自己的限频代价。op 行的命令、arm 目录和分块尺子都不变，变的只有平台：固件，连同在同一时段换的驱动，两者分不开（REPORT §6.3）。e2e 一行例外，见表后说明。这些是优化后期的代码，列在这里只为给出这个倍数）：

| 测量（op：b4 s8192 hq32 hkv8 d128 causal，prod，ms 按 o1 / o2 两种顺序；e2e：Llama-3.1-8B 32L） | 前：A0-VR（09-28，负载 sclk 1010-1060 MHz，驱动 7.1.1-2397345） | 后：A0-RF（op 09-30，负载 1740-2030 MHz，驱动 7.1.0-2411946；e2e 10-02，驱动已换成 7.1.0-2412954） | 倍数（op 行取两种顺序的均值，括号内为 o1 / o2） | 来源 |
|---|--:|--:|--:|---|
| FlyDSL fwd（r13ns，op） | 2.023 / 1.992 ms | 1.353 / 1.356 ms | 1.48x（1.50 / 1.47） | 0928__a0_repro/REPORT.md §6.1、§6.3；wt-llama31 README §7 只取 o1，写作 1.50x |
| aiter ASM fwd（op） | 1.564 / 1.543 ms | 1.257 / 1.253 ms | 1.24x（1.24 / 1.23） | 0928__a0_repro/REPORT.md §6.1、§6.3；README §7 |
| FlyDSL bwd r29（op） | 11.010 / 11.015 ms | 6.583 / 6.596 ms | 1.67x（1.67 / 1.67） | 0928__a0_repro/REPORT.md §6.2；README §7 |
| aiter ASM bwd（op） | 7.654 / 7.666 ms | 5.503 / 5.494 ms | 1.39x（1.39 / 1.40） | 同上 |
| e2e 单步，ASM attention + nkfix（`NKFIX_CHECK=1`） | 1,947 ms（09-28 a0_p3a，已作废的那批运行，见下） | 1,351 ms（1350.6；10-02，新 e2e 套件，训练中 sclk 中位约 1.50 GHz） | 1.44x（不纯是平台差异，见下） | 0928__a0_repro/REPORT.md §2、§5.3、§6.4；1002__e2e/RESULT-e2e.md、E2E-PLAN.md §1-§3；README §7 |

负载时钟比约 1.75x，实测提速都低于它：FlyDSL 对 sclk 更敏感，ASM fwd 基本不随 sclk 变。同一节里其余 FlyDSL fwd arm 逐顺序为 1.45-1.69x（0928__a0_repro/REPORT.md §6.1、§6.3）。FlyDSL fwd 的提速按顺序是 1.47-1.50x：REPORT §6.3 汇总写的 1.48x 是两种顺序的均值，wt-llama31 README §7 的 1.50x 只取了 o1。

e2e 一行不是干净的同代码同方法对照，1.44x 只能作参考：
- 前值 1,947 ms 是 09-28 a0_p3a 里 asm arm 的单步中位数，只有这一个进程（反序的 a0_p3b 出 NaN 作废）。REPORT §6.4 已把 09-28 的 A0 绝对数整体作废，E2E-PLAN.md §1 也记为"作废数字，只取教训"。这个进程在 step 2 首次调用 FlyDSL fwd，fwd 树的 `_env.py:26` 在 import 时把 `HIPBLASLT_TENSILE_LIBPATH` 改指宿主机库 `~/.local/hipblaslt-gfx1250`，之后的 asm 步有没有受影响没有验证（REPORT §5.3）。
- 后值来自 10-02 另写的套件 `1002__e2e/e2e/`：fwd 用的是删掉那两行改指的副本 `fwd_r16_imglib`，适配层加了 BLAS guard，step 1-2 只跑 ASM（E2E-PLAN.md §2-§3）。
- 两次运行之间，驱动先在 09-28/29 经 -2410994、-2411826 换成 7.1.0-2411946（09-30 op 复测时就是它），10-01 又换成 7.1.0-2412954，比 op 行多换了一次。
- 两边相同的只有：nkfix（`0927__b0/gemm/nkfix_b0.py`，git 记录里 09-28 05:12 之后没改过）加 `NKFIX_CHECK=1`、同一份 32L 配置模板（`0927__b0/e2e/configs/l8b_e2e.template.yaml`），以及 3 个 arm 按步交替、约 90 步的做法。wt-llama31 README §7 把这一行也算作 "Same code and method"，不准确。

### a.2 算子层面的问题

| # | 问题 | 现象 / 数字 | 原因 | 影响 | 后来怎么解决 | 来源 |
|---|---|---|---|---|---|---|
| 1 | 在镜像的 torch 2.11 下，main 一 import 就崩 | `import primus_turbo.pytorch` 抛 `TypeError: Opaque type ... must subclass torch._opaque_base.OpaqueBase`。4 个候选镜像的 torch 都是 `2.11.0+rocm7.14.0a20260625`，全部中招。测试 collection 也跟着失败 | `core/low_precision.py` 在模块级对 `Float8QuantConfig`、`Float4QuantConfig`、`ScalingRecipe` 调用 `register_opaque_type`（#483 引入）。torch 2.11 要求这些类用 `OpaqueBaseMeta` 元类。这个问题和 attention 无关 | 整个包都不能用，op、e2e、测试一个也起不来 | 09-13 从镜像里较新的 `/workspace/Primus-Turbo` 手工移植了 shim。09-14 在 B0 上复现 main 时，也得单独打这个补丁。upstream 在 `f5e1f18b`（#501，09-14）修复。另外，`3rdparty/hipify_torch` 子模块不初始化的话，`build_ext` 跑不起来 | 0913/phase1/RESULTS.md §5；0914__repro__c07/RESULTS.md「复现环境」；git show f5e1f18b |
| 2 | 默认派发选不到 gfx1250 上唯一能跑的后端 | 不 pin 后端时，`flash_attn_func`（BSHD b4）的派发过程是：先试 FLYDSL，G=4 被拒；再按插入顺序扫 `_DENSE_FWD_BACKENDS`（FLYDSL→AITER→HIPKITTENS→GLUON→TRITON），结果 AITER 先命中。`amdprimus` 镜像里没有 aiter，所以第一个训练 step 就报 `ModuleNotFoundError: No module named 'aiter'` → `ImportError: Primus-Turbo requires amd-aiter==0.1.14.post1`。如果装了 aiter，fwd 能过，bwd 会在训练 step 中途报 `invalid argument for fmha_bwd` | `DenseAttnFwdAiterBackend.can_handle` 只要是 4-D 的 fp16/bf16 就返回 True，不看架构。AITER 这条路最终落到 CK，而 CK 的 fmha 只有 CDNA 内核 | gfx1250 上唯一能跑的是 `c1325c7e` 加的 Triton 后端，它的 `can_handle` 只认 `is_gfx1250`，但默认派发到不了它，只能手动设 `PRIMUS_TURBO_ATTN_BACKEND=TRITON`。而且出错的时刻是训练中途，不是派发时 | 分支上 `f8c45dee`（09-13）让 AITER dense 在 gfx1250 上拒绝调用。09-30 的 port 分支 `dev/lhz/llama31_attn_opt` 让 FLYDSL 在 gfx1250 上解析到新内核（G∈{1,2,4,8,16}、d128、bf16）。origin/main `8cda13c7` 至今未修 | 0913/phase0/PHASE0-STATUS.md:11-22；git show c1325c7e（commit message；`attention_impl.py:122-127, 431-436, 621-626`）；session 541e7bc3（aiter ImportError）；0930__port/PR_BODY.md |
| 3 | FlyDSL 架构门 `>= (9,5)` 对 gfx1250 是开的（潜伏 bug） | gfx1250 报 cc (12,5)，`_flydsl_common_ok` 会放行。可所有 gfx950 FlyDSL FA builder 在非 gfx950 上都会硬 raise | 门写成了 `get_device_compute_capability() >= _GFX950`，应该是相等判断 | 09-13 没炸，只是因为 `_gqa_group_ok` 恰好拒掉了 G=4。upstream `9f05cd85`（#509，09-17）把 GQA 门放宽到 [1,256] 内的 2 的幂之后，两道互相掩护的门在 main 上同时打开了：SBHD 或 b=1 BSHD 的 Llama 调用会进 gfx950 JIT，报 `requires gfx950+ (uses ds_read_tr16_b64)`。BSHD b4 仍然落到 AITER/CK，见 #2 | 分支上 `f8c45dee`、`d00ba261` 改成 `is_gfx950()`；port 分支 `c36cc124`（09-30）改成相等判断。origin/main `8cda13c7` 仍然是 `>= _GFX950`。0914__campaign/HANDOFF.md 的 0916 补充说"main 上已修"，那里的 main 其实指我们的分支 | PHASE0-STATUS.md:24-32；git show origin/main:primus_turbo/pytorch/kernels/attention/attention_impl.py:49,79-106；git show 9f05cd85；wt-llama31 README §2 |
| 4 | flydsl 名义上是可选依赖，却在模块级无条件 import | `c1325c7e` 的 `setup.py:529-535` 在 gfx1250 构建时不装 flydsl。但 `attention_flydsl_impl.py` 在模块级 import flydsl，`attention_impl` 和 `flash_attn_interface` 又在模块级 import 它。所以在没有 flydsl 的环境里，`from primus_turbo.pytorch.ops.attention import flash_attn_func` 直接失败。`sparse_mla_impl.py` 有同样的问题，而且没有 arch gate。MoE、GEMM、quantization 下还有 10 处同类的模块级 import | 没有 try/except 保护；安装策略和 import 策略互相矛盾 | gfx1250 上唯一可用的 Triton 路径也跟着挂掉。现在能跑，只是因为镜像碰巧预装了 flydsl 0.2.4（宿主机没有） | 分支上 `f8c45dee` 加了 `FLYDSL_AVAILABLE` 保护，`d00ba261`（09-17）补上 sparse_mla。upstream `eeb9e8d7`（#508，09-21）改为在所有 arch 上都装 `flydsl==0.2.4`。这只是绕开，不是修复：origin/main 的模块级 import 仍然没有保护 | PHASE0-STATUS.md:34-43；git show d00ba261；git show origin/main:setup.py:531-535 |
| 5 | flydsl 版本互斥：aiter 要 0.3.2，Primus-Turbo 要 0.2.4（09-17 发现） | aiter pin `flydsl==0.3.2`，因为它的 gfx1250 FlyDSL fwd 在 0.2.4 下编不过：0.2.4 的 `ast_rewriter` 不接受用 list 作 stateful dynamic-if 的状态变量。可 0.3.2 删掉了 `flydsl.expr.buffer_ops`，而 Primus-Turbo 的 FlyDSL 代码无条件 import 它，结果是 `ImportError: cannot import name 'buffer_ops' from 'flydsl.expr'` | 同一个包，在同一个进程里被 pin 成两个不兼容的 `==` 版本 | aiter 的 gfx1250 FlyDSL fwd 和 primus_turbo 不能在同一个进程里用。09-28 B0 e2e 的 turbo 基线因此只能单独开一个进程 | 09-17 用 `pip install --target ~/.local/flydsl032` 侧装，再调整 `sys.path` 顺序；op-evolve 作业只 import torch 和 aiter。09-30 port 分支把 fwd r16 和 bwd r29 移植回 0.2.4。0.2.4 自带的 LLVM 把 async LDS→global 的 O store 地址编错了，上卡就 fault，所以改用 `buffer_store`，代价是 fwd 约 -1.3%、bwd 约 -1.8% | 0917__flydsl/VENDOR-REPORT-aiter-gfx1250.md §3；0917__flydsl/STAGE1-FWD.md §3-4；0927__b0/e2e/RECON.md §3.3；0930__port/PR_BODY.md |
| 6 | gfx950 的 FlyDSL attention 移植不过来，而且 G=4 被拒 | fwd 的 `flash_attn_fwd.py:71-74` 直接 raise "requires gfx950+ (uses ds_read_tr16_b64)"（`c1325c7e` 中 71 行是 arch 判断，72-74 行是 raise；HANDOFF 记 :71，SURVEY 记 :72-74）。bwd 有 6 个 kernel（odo、lse-transpose、dq-reduce、slot-reduce、a16-unpermute、dkdv），每个都 assert gfx950。用容器里的 LLVM 直接探测，`llvm.amdgcn.mfma.f32.32x32x16.bf16`、`ds.read.tr16.b64`、`permlane32.swap` 在 gfx1250 上全部 Cannot select。`warp_size=64` 硬编码在 `flydsl/utils/attn_helper.py:382`。main 的 `_gqa_group_ok` 要求 G 是 [8,256] 内的 2 的幂 | 这些内核是按 CDNA4 wave64 MFMA 的 fragment 布局手工排的。gfx1250 是 wave32 + WMMA + 320 KB LDS + TDM，根本没有 MFMA | 对 gfx1250 上的 Llama-3.1-8B，FlyDSL 后端有两重不可用。09-14 估算移植要 18-36 个工程日（"FlyDSL Gate A BLOCKED"）。0916 撤回的只是"FlyDSL 这条路整体走不通"的结论；"现成的 gfx950 内核移植不过来"这一条仍然成立 | 不移植。09-17 起自己写 gfx1250 FlyDSL bwd（op-evolve）；fwd 以 aiter 的 gfx1250 FlyDSL m32x8 为种子。09-30 合入 port 分支的 `primus_turbo/flydsl/attention/gfx1250/` | 0914__campaign/RESULTS.md §9；0914__campaign/HANDOFF.md（0916 补充）；0917__flydsl/SURVEY.md §2；git show c1325c7e:primus_turbo/pytorch/kernels/attention/attention_impl.py:79-87 |
| 7 | gfx1250 上一个测试都不跑 | `tests/conftest.py` 在 gfx1250 上给所有用例打 skip（"Not yet supported on gfx1250"），连专门为 gfx1250 写的测试也不跑。`c1325c7e` 说自己有 19 个测试通过，那是"临时去掉 skip"之后跑的。torch 2.11 下，conftest 里的 `except ImportError` 接不住 #1 的 TypeError，collection 直接失败。0914 报告把这个状态记为"gfx1250 测试 0/51" | 全局 skip（`TODO(ruibin): gfx1250 is not yet fully supported`） | #2-#4 这类问题完全没有 CI 信号 | 分支上 `f8c45dee` 改为用 `@pytest.mark.gfx1250` 显式放行：通过数 36 → 44（`03a76f61`）→ 51（`8dd2fd32`）。`963e7d7d`（09-17）在新基线上恢复了这个 marker。upstream #508（09-21）加了 allow-list，但只放行 `test_grouped_gemm.py`，attention 测试仍然全部 skip。09-30 在 origin/main 上去掉 skip 跑 `test_attention_16bit` 子集，有 54 个失败；port 分支是 28 个，全部是 FlyDSL 按设计拒绝后回退到 aiter，而测试镜像里没装 aiter，没有新增失败 | git show c1325c7e:tests/conftest.py:51-68；f8c45dee、03a76f61、8dd2fd32 的 commit message；git show origin/main:tests/conftest.py:31-37；0914__campaign/report.html §3.1；0930__port/PR_BODY.md「Testing」 |
| 8 | Triton 后端的出厂配置很慢，比现役的 flex 慢约 1.9x | A0-VR 上，bench0910（镜像自带的 `0.4.1.dev12`）测得 fwd 10.366 / bwd 48.059 / 合计 58.424 ms = 131.7 TF/s，flex 是 31.508 ms，慢 1.85x。09-13 的 tune bake-off 是在分支树上测的（对应 `@1cb2e183` 的出厂配置，不是 main 原样）：10.673 / 48.969 / 59.642 ms = 129.0 TF/s，flex 31.337 ms，慢 1.90x。这两个代码状态不能混为一谈：B0 上同尺子，`@1cb2e183` 出厂 28.121 ms（bwd 22.942），`@c1325c7e` 原样 30.258 ms（bwd 25.025），差 7.6%（FINAL-TABLE 脚注 †）。逐内核（bench0910）：`_bwd_kernel_dkdv` 35.92 ms（61.5%），`_bwd_kernel_dq` 11.75 ms（20.1%），`attn_fwd` 9.94 ms（17.0%），preprocess 0.06 ms。B0（09-14，tune，main `c1325c7e` 原样）为 30.258 ms = 254.4 TF/s，flex 15.307 ms，慢 1.98x | fwd 和 bwd 的 `@triton.autotune` 都只有 1 个 config（`num_stages=1, num_warps=4`），`_bwd_preprocess_use_o` 连装饰器都没有，`FIXED_BLOCK_M = FIXED_BLOCK_N = 64` 是模块常量。整套参数照搬自面向 CDNA（wave64、MFMA、64 KB LDS）的 perf-kernel，从没在 gfx1250（wave32、WMMA、320 KB LDS）上搜索过，autotune 实际上只起编译缓存的作用。另外，`dense_backward` 把 dq/dk/dv 按 fp32 分配，autograd 再转回 bf16（`bfloat16_copy` ×3 共 0.182 ms，Fill ×5 共 0.144 ms）。这是内存和 cast 开销，不是精度 bug | 在 op 层，打开 turbo attention 反而是倒退；dkdv 一个内核就占了 61.5% | 只改两个整数（fwd `num_stages` 1→2，bwd `num_warps` 4→2），A0-VR 的分支树上就到了 35.704-36.405 ms，快 1.64-1.66x，SQNR 不变，1000 次运行逐位一致。之后是 vendored aiter 融合反向（`1cb2e183`）、ASM fwd（`0e5cb743`）等，见 (c)。origin/main 的 Triton kernel 到 `8cda13c7` 都没动过：09-30 在 A0-RF 上用 bat 测得 fwd 5.11 ms / 430.45 TF/s，bwd 24.49 ms / 224.50 TF/s。同日同尺子，port 分支的 FlyDSL 对这个 Triton 后端在 36 个门内形状上的几何平均提速：causal fwd 4.13x / bwd 3.37x，non-causal fwd 3.47x / bwd 3.58x（A0-RF，各 18 个形状） | bench0910；0913/phase1/BAKEOFF.md §1；0913/phase1/RESULTS.md §3, §12；PHASE0-STATUS.md:54-67 与「Corrections」节；git show c1325c7e:primus_turbo/triton/attention/attention_kernel.py:452-480, 1095-1118；0914__repro__c07/FINAL-TABLE.md；0930__port/runs/bench_turbo_triton.csv:55；0930__port/PR_BODY.md「Performance」（已用 runs/bench_turbo_{flydsl,triton}.csv 复算） |
| 9 | 反向 autotune 的 key 写错了（潜伏 bug） | bwd 的 key 列表里有 `BLOCK_DMODEL`，可两个反向内核都没有这个参数；同时漏掉了 HQ/HK 和 seqlen | Triton 会静默丢弃未知的 key，而且只在 `len(configs)>1` 时才计算 key | 只有一个 config 时无害；一旦出多个 config，就会按错误的 key 复用调优结果 | 分支上 `df20f1d6`（09-13）修正；origin/main 没改（这个 kernel 文件从 `c1325c7e` 起就没动过） | 0913/phase1/RESULTS.md §9；git show c1325c7e:primus_turbo/triton/attention/attention_kernel.py:1095-1110 |
| 10 | Triton 两内核反向里有"静默算错"的陷阱 | `FIXED_BLOCK_M` 实际上是跨内核的 LSE/delta ABI：`attn_fwd` 在 `m*BLOCK_M*2` 处写 LSE，preprocess 在 `+BLOCK_M` 处写 delta，host 端由 `_lse_delta_views` 重建，dkdv 用 `2*start_m+arange(0,2*BLOCK_M)` 加两次 `tl.gather` 拆开。只改 dkdv 的 BLOCK_M 不会报错，但 dk/dv 会平滑地算错。`sequence_parallel=False` 会把 grid dim1 设成 1，只算 1/128 的梯度，其余保持 zeros 初值 | ABI 是隐式的，没有任何断言 | 只看计时的 sweep 会把错误配置当成大胜。这种结构也表达不了 aiter 那样的非对称 tile（dkdv 32×128、dq 128×32），这正是 09-13 bake-off 输给 aiter 的结构性原因 | 不动这个 ABI，改为 vendor aiter 的单内核融合反向（`1cb2e183`）；tune 加上四张量 SQNR 门 | 0913/phase1/RESULTS.md §8, §13；0913/phase1/BAKEOFF.md §3 |
| 11 | 仓库自带的测量和校验不可信 | 现有 bench 的正确性检查对所有后端都返回同一个 max-abs 值 `8.011e-03`，而且不看梯度。`bench_attention.py --mask block_causal` 其实不构建 document mask，和 causal 完全一样（31.008 vs 31.012 ms）。`c1325c7e` 的提交信息称达到 220.6 TF/s（b2 s8192，SDPA flash 为 161.4），而在 A0-VR 上同一内核实测只有 b2 129.42 / b4 127.83 TF/s | max-abs 只看 out；mask 参数只改了 BlockMask 的 batch 维。220.6 的测量条件不明：提交信息只写 "seq 8192, B2 H32 D128, fwd+bwd"，没写 hkv、是否 causal、时钟。220.6/129.4 = 1.70x，09-13 拿它对照当时估的 1.65x 时钟比，判定 220.6 是在没限频的卡上测的。这只是推断：220.6 远高于 A0-VR 的实测，方向大概率对（B0 上 main 原样 b4 是 254.4），但它是在什么卡、什么时钟下测的，仍然不明。1.65x 只是时钟估算；A0-VR→B0 是两张卡之间的跨卡比（5 行中位 1.93x），A0 自身刷固件前后 op 级是 1.24-1.67x，1.70x 和哪个比值相符都没有定论 | 既验不出错误的梯度，也很容易把不同时钟下的数放在一起比 | 自己写了 tune：四个张量分别算 SQNR（门槛 50 dB），在 launch 处断言实际用的 config，rc=139 记为重试，健康检查只读 dmesg | ph1/E2.md §4；ph1/E3.md Headline；0913/phase1/RESULTS.md §2；git show -s c1325c7e；PHASE0-STATUS.md「Tooling」；0914__repro__c07/RESULTS.md:124-130；wt-llama31 README §7 |
| 12 | 备选后端 aiter Triton MHA 在 gfx1250 上没调过，还有数值陷阱 | `gfx1250-MHA-DEFAULT.json` 和 gfx950 版 md5 相同（`dd06bae7…`），里面还有 CDNA 专用旋钮 `matrix_instr_nonkdim`；fwd 是 `num_stages=1`。出厂性能：34.111 ms（E2，09-11，extended-v2 + amd-aiter 0.1.14.post1，225.6 TF/s）；34.565 ms（tune，09-13，`aiter-src @ffa945f9`，222.7 TF/s）。`mha_fused_bwd` 数值是坏的（dk −0.22 dB，dq 46.61，dv 49.03），而且更慢（42.294 ms）。`BLOCK_N1=256` 不配对 `BLOCK_M2` 时，dq 只有 9.59 dB，却快了 1.31x。`BLOCK_M2=64,BLOCK_N2=128` 报 CompilationError。还有 7x 的性能悬崖，例如 `BLK_SLICE_FACTOR=4` 151.6 ms、`waves_per_eu=4` 120.6 ms。`_get_config` 带 `@functools.lru_cache`，不先 `cache_clear()` 的话 override 静默无效，因此作废过一整轮 sweep | 配置是从 gfx950 拷过来的，从没在 gfx1250 上搜索过。单内核反向里 dkdv 和 dq 两半共用同一个 launch grid，所以 `BLOCK_N1` 与 `BLOCK_M2` 耦合 | 它不在 turbo main 的默认路径上（需要另装 aiter）。但它赢了 09-13 的 bake-off，后来树内的融合反向也是从它来的 | 09-11 的"aiter + 2 个旋钮 = 24.347 ms / 316.1 TF/s"**已撤回**：09-13 用同一口径重测是 31.207 ms / 246.6 TF/s。原因是上游 aiter 改了反向的出厂配置，bwd `num_warps` 4→2 不再有效。纯 aiter 调优的上界是 21.672 ms / 355.1 TF/s（N1=M2=256，BSF=1；DECISIONS D9）。PROGRESS.md 和 0914 FINAL-TABLE 引用的是 21.684 / 354.9；`phase2/ledgers/wq.jsonl` 里同配置的 5 次重复（tag 为 rep 1-5）为 21.527-21.713 ms，两个数都在其中，彼此差 0.06%。0915__repro__c07 以 A0/B0 = 1.63x 为由，判 09-13 的 34.565 "不可信"，但这个 1.63x 是拿 09-15 新测的 28.035 算的。09-13 的值对 B0 的 17.204 是 2.01x，和 FINAL-TABLE 其他行的 1.88-2.12x 一致，而且 09-11 在另一个镜像上独立测得 34.111。所以更可能异常的是 09-15 那次（它的 fwd 3.725 ms 像是调优过的前向）。本节仍用 09-13 的值；这一点未复测 | ph1/E2.md §2-§6；0913/phase1/RESULTS.md §13；0913/phase2/DECISIONS.md D9；0913/PROGRESS.md:71；0913/phase2/ledgers/wq.jsonl；0915__repro__c07/RESULTS.md:49-60；0914__repro__c07/FINAL-TABLE.md |
| 13 | aiter 为 gfx1250 预编译的 ASM 内核"有但够不着"，反向在 GQA 下越界写（09-14..09-17 发现，属于上游现状） | gfx1250 只发布了 52 个 `.co`（gfx950 有 1466 个，VENDOR-REPORT 的数；本地 aiter-src @6963ae9 用 `find` 数 gfx950 为 1459，差异来源未查）。反向只有 6 个 `.co`（gfx950 有 124 个）：`bwd_hd128_bf16_{a32_pssk,a32_pssk_perf,causal_br_a32_pssk,causal_br_a32_pssk_perf}` 加 `bwd_hd128_dq_convert_bf16`、`bwd_hd128_odo_bf16`。gfx950 同目录的文件名里有 psskddv（52 个）、a16（26）、fp16（45）、swa（4）、group（50）变体，gfx1250 一个都没有。也没有 varlen 反向（dispatch 表只有 mode=0）。aiter Python 侧的 `can_impl_fmha_v3_bwd` 从 gfx942 判起，只放宽到 gfx950，gfx1250 永远选不中。反向主 kernel 的 grid 是 (kv_tiles, nhead_q, batch)，却按 q head 去索引 kv 尺寸的 dk/dv：ratio=4 时 dq 52.24 / dk −0.94 / dv −0.65 dB（A0-VR）。s=1024 时是静默损坏；s=256 时进程 fault，但卡不受损 | 上游的资产和 wrapper 只为 CDNA 做全了。GQA 下有 ratio 个 workgroup 无同步地写同一个 dk/dv tile | aiter 在 gfx1250 上最快的内核不能直接调用，只能离线从 ELF metadata 反推 kernarg、自己写发射器。绕法是 dk/dv 按 q head 分配、再在 host 端规约，代价是峰值多占 1.254 GiB，外加规约 0.482 ms（A0-VR，反向时间 +5.2%） | `0e5cb743`（09-14）接上 ASM fwd。09-15 接上 ASM bwd：自写发射器约 674 行，用上面的绕法，资格门为 seqlen>=2048。09-17 写了给 AITER 的 vendor report。性能和其余问题见 (b) | 0917__flydsl/VENDOR-REPORT-aiter-gfx1250.md §1-§2；`/home/lihuzhan/code/aiter-src/hsa/{gfx1250,gfx950}/fmha_v3_bwd/*.co`（@6963ae9 文件清单）；0917__flydsl/GQA-WORKAROUND-COST.md；0915__opt/BACKEND-STRATEGY.md:57,73；0922_summary/ASM-ATTENTION.md §1.1 |
| 14 | 其余后端在 gfx1250 + G=4 上都用不了 | HipKittens 的 attention 硬编码 gfx950：`setup.py` 只在 gfx950 上开它，而且无条件加 `-DKITTENS_CDNA4`。仓库里的 udna1（gfx1250）移植没有 attention kernel，09-14 的 udna1 gate 判为 RED：`reductions.cuh` 里 7 处 `permlane32_swap` 编不过；`mma_AB`/`mma_AtB` 仍然派发到 wave64 MFMA；`swap_layout` 在 wave32 下可以证明是错的；GPU gate 256 个元素错 126 个。Gluon 只有前向，且只支持 gfx950。CK 的 fmha 只支持 CDNA。gfx1250 上 varlen 没有 Triton 路径（Triton fwd 不收 cu_seqlens，`c1325c7e` 明说没覆盖）。torch SDPA FLASH（AOTriton）在 A0-VR 上 96.58-97.06 ms（bench0910 / tune） | 上游只为 CDNA 实现过 | 起步时可选的只有 Triton，也就是 turbo 自带的和 aiter 移植过来的 | 主线先走 Triton，之后转向 aiter ASM 和自写 FlyDSL，见 (b)(c) | 0913/index.html §05；0913/phase2/PLAN-4GPU-TOMORROW.md:138；0914__campaign/RESULTS.md §8；0913/phase1/RESULTS.md §1；git show c1325c7e |
| 15 | 现役 flex 自身有可靠性问题 | flex 反向在 B=4 的 BlockMask 下会间歇出现 `Memory access fault ... Page not present`：12 次里 2 次，B=1 时 6 次里 0 次。rc=139，进程被杀，但卡不 wedge。最小化复现全部通过 | 未归因 | 训练形态下 block_causal 必然是 B=4，在这种形态下做 fitness 评估，大约会误杀 1/6 的好候选 | harness 把 rc=139 记为重试，每个候选单独一个进程 | ph1/E3.md §8 |
| 16 | Triton 反向出现过一次 dk/dv = −inf（未解释） | 09-13 在 A0-VR 上，配置 `fwd:num_stages=2; bwd:num_warps=2,waves_per_eu=1` 下出过一次：out/dq 正常，dk/dv = −inf dB（约 1/100 次运行）。之后 2,300 次调用都没复现（进程内 1000+1000 次，加 150 个新进程各 2 次），出厂配置和调优配置都逐位一致。早先"出厂配置 dk 不确定"的信号，其实是 fp32 参考自己不确定造成的（`torch.matmul` 走 rocBLAS），已更正 | 不明。候选解释是这张卡已有记录的瞬态硬件故障 | 单次检查放过去的偶发错梯度，恰恰是最容易被发布出去的那类错误 | 保留四张量门。确定性检查改为和自身逐位比较，不再对每次重算的参考比 SQNR | 0913/phase1/RESULTS.md §4, §12 |

### a.3 e2e 层面的问题

| # | 问题 | 现象 / 数字 | 原因 | 影响 | 后来怎么解决 | 来源 |
|---|---|---|---|---|---|---|
| 1 | A0 上从来没有跑完过一个训练 step | 09-10 的 run1、run3、run_smoke 三份日志里，step 行一条也没有。JIRA 里的 19,795 tok/s 不是这台机器测的 | 镜像缺 torchtitan 的依赖（tyro、torchdata、datasets、tabulate、tokenizers、safetensors、einops、pillow、tensorboard、wandb）；`_hfassets` 没挂载；再加上 #2 那条 patch / enable_gqa 链 | 没有任何本地 e2e 基线可以对照 | 09-13 装好依赖后 `docker commit` 成 `fa-tune:deps`，06:31 UTC 跑出这台机器历史上第一个 `step: 1`。注：`primus-cli … train pretrain` 在能连 PyPI 时每次启动都会重装 torchtitan 依赖，所以这个问题只在直接用 `torchrun` 或离线时出现 | 0913/phase1/RESULTS.md §10；session 541e7bc3 2026-09-13T06:27-06:31Z；wt-llama31 README §1 |
| 2 | turbo attention 实际从没进入过 e2e（patch、enable_gqa、converter 三个问题连在一起） | ① 默认配置下，turbo_attention patch 其实是应用了的（12/12），然后在第一个 step 死于 `No module named 'aiter'`。JIRA 说的"⊘ Skipped"，来自后来手工加上的 `use_turbo_attention: false`，那是规避手段，不是原因。② 缺 tensorboard 时，patch runner 把它报成"patch failed"然后继续往下跑，于是改调用点的 patch 没生效；torchtitan 自己的 Attention 把 `enable_gqa` 传给 `TurboAttention.forward(q,k,v,bias)`，抛 `TypeError: ... unexpected keyword argument 'enable_gqa'`。③ 复现配置于是写成了 `converters: []` | patch 只替换 Attention 类；真正把 `inner_attention` 换掉的是模型 converter（`primus_turbo_converter.py:22`）。converter 关掉后，`inner_attention` 一直是 torchtitan 的 `FlexAttentionWrapper`。而 `TurboAttention.forward` 不接受 `enable_gqa` | 09-13 的"turbo:TRITON 调优 245 tps vs flex 244 tps"，两边实际都是 flex：当天的 `repro_l8b_bf16_mbs4_seq8k_turbo.yaml`（和 `_turbo_ac.yaml`）里本身就写着 `converters: []`。09-14 B0 的 13,204 tps 和"turbo vs flex 1.459x"同样测的不是 attention（已撤回）：两臂的 inner attention 都是 flex。HANDOFF 把差别归到 turbo 配置一并打开的 `turbo_float8_linear` + `turbo_mx_linear`（见 #11）；但按本机 Primus @`e7968675` 的代码，这两个 patch 只替换 float8/mx converter 的实现，不往 `converters` 列表里加东西，`converters: []` 下看不出它们怎么生效。09-14 B0 用的 Primus 版本和配置文件不在本机，无法核对，所以 1.459x 的真正来源未定，能确定的只是它不来自 attention。同日 B0 训练里 ASM 前向也从未被调用：让门自己写 trace 文件后，21 步训练中这个文件始终没被创建，即 `asm_forward_eligible` 根本没被调用到；此前三轮诊断依据的"日志里没有 aiter 横幅"这个判据本身是坏的，因为 `AITER_LOG_LEVEL=ERROR` 把横幅压掉了。"converter 开着、compile 开着、融合反向在跑"这三个条件，在此之前从没同时满足过 | 09-14 打开 converter 后，依次撞到并修掉 6 层问题。其中"`os.environ` 写在启动路径上"和"`num_ctas` 直传 kernel"两处，是我们自己早期补丁里的 bug（`a009599b` 和 vendored 反向），由 `ad67a2cc` 修复，不属于 main。09-15 在 `dev/lhz/attn` 上让 `TurboAttention.forward` 接受并校验 `enable_gqa`，并新建打开 converter 的配置 `repro_l8b_turbo_conv.yaml`。origin/main `8cda13c7` 的 `forward` 仍然没有这个参数 | 0913/phase1/RESULTS.md §10；0913/index.html §02（更正一）；0914__campaign/HANDOFF.md §4（:78-79）；0914__campaign/RESULTS.md §16-§17；Primus:examples/torchtitan/configs/MI455X/repro_l8b_bf16_mbs4_seq8k_turbo.yaml:27-30；0915__opt/RESULTS.md:145-150；git show ad67a2cc；git show origin/main:primus_turbo/pytorch/modules/attention.py:53-58 |
| 3 | e2e 被 GEMM/BLAS 卡死 | 09-13 在 A0-VR 上跑通后：245 tps，133.7 s/step，14.17 TFLOPS，GPU 一直 100% 忙。整条 attention 路径是 35.7 ms × 32 = 1.14 s，不到 step 的 1%。`amdprimus` 镜像里 hipBLASLt 一调用就报 `HIPBLAS_STATUS_INVALID_VALUE`，只能 `TORCH_BLAS_PREFER_HIPBLASLT=0` 退到 rocBLAS。rocBLAS 在空闲卡上：bf16 8192³ 40.12 ms = 27.4 TF/s；qkv 32768×4096×6144 24.0、mlp up 23.9、mlp down 26.1 TF/s。同一张卡上 Triton GEMM 实测 1002.7 TF/s（09-11 E1，当时被当作 roof；09-15 被同卡同时钟的 hipBLASLt NT 1502 TF/s 超过，见右栏） | 当时的归因是"镜像缺 gfx1250 的 Tensile 库"。这个归因后来改写了两次，见右栏 | 优化之前的任何 e2e 数都既验证不了、也证伪不了 attention 的改动。JIRA 里 MI455X 的数字（GEMM 655 ms/step、19,795 tok/s）不可能出自 A0 这个镜像 | 第一次改写在 09-14：B0 上（容器 fa-e2e）库是在的，hipBLASLt 能用，但只有 112.98 TF/s，朴素的 Triton GEMM 有 1190.07 TF/s，差 10.5x。第二次改写在 09-15：查明 A0 镜像的问题是路径错位，402 个文件在 `library/gfx1250/` 子目录里，加载器却在上一层找。设 `HIPBLASLT_TENSILE_LIBPATH=…/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250` 后，8192³ NN 到 68.74 TF/s，32L e2e 244→2,027 tps（8.3x，3 步，n=1，A0-VR）。这条修法有两个环境陷阱：`_rocm_sdk_devel/lib/hipblaslt/library/gfx1250` 那份库不完整，指过去会 core dump / SIGSEGV，只能用 `_rocm_sdk_libraries_gfx1250` 那份；`bash -lc` 登录 shell 会读到一个强制 `TORCH_BLAS_PREFER_HIPBLASLT=0` 的 profile 脚本（我们的容器里就有），要用 `bash -c`、显式设两个变量，再在进程内查 `torch.backends.cuda.preferred_blas_library()`。同一天又查出真正的大头：hipBLASLt 只给前向的 TN 布局带了纯 bf16 调优库（`BB_BB_UA_Type_BB_HPA`），dgrad（NN，`Ailk_Bljk`）和 wgrad 按 GridBased 最近邻查表，落到 GEMV 用的 tile `MT32x16x32`，只有 50-80 TF/s，同形状比 TN 慢 16.5-21.5x；而同卡同时钟，torch 层的 NT 路径（`a @ bt.T` / `F.linear`，落到 hipBLASLt 的 TN 调优库）8192³ 实测 1502 TF/s，比 09-11 当作 roof 的 Triton 1002.7 还快。GEMM-NN 没修时的 e2e 代价：09-28 B0 上（hipBLASLt 路径已修、无 nkfix），无论用哪种 attention，32L 都只有 1,856-1,922 tok/s（低档中位；高档 2,008-2,022），17,049-17,655 ms/step，其中反向 `MT32x16x32` GEMM 占 15.3-16.1 s/step。修法：B0 09-14 用 compile 加强制 inductor Triton GEMM，快 4.01x；09-15/16 做 nkfix（用 TorchDispatchMode 改写 `aten::mm` 的操作数布局）；09-28 在 B0 上移植 nkfix_b0 + transpose_triton，tps 约为修复前的 10 倍。详见 (d) | 0913/phase1/RESULTS.md §10-§11；0913/phase0/PLATFORM-ESCALATION.md Problem 3；0914__campaign/RESULTS.md §3；0915__opt/BLAS-FINDING.md「修法」；0915__opt/GEMM-NN-FINDING.md:70, 118；0915__opt/VENDOR-REPORT-hipblaslt.md；session 11052787（8192³ NT 1502.0 TF/s）；0927__b0/e2e/RESULT.md 开头更新、§0、§3；wt-llama31 README §1 第 4 行、§2 第 3 条、§7 |
| 4 | 显存贴着上限 | 32L、mbs4、s8192、AC none 时峰值 379.50 GiB（87.85%），flex 是 378.00 GiB。开 AC full 后降到 174.79 GiB（40.5%），吞吐却是 215 tps，说明显存不是瓶颈。默认 flavor `8B` 走 SDPA：CUDNN 和 FLASH 在 gfx1250 上不可用，回退到 MATH 后会物化 [4,32,8192,8192] 的矩阵，显存冲到 442 GB 后 SIGBUS，所以配置改用 `8B_flex`。排查时有一次，前一个训练进程还没退出，仍占着 379.5 GiB，被误判成"泄漏" | 8B 参数 + Adam + 全量激活本身就要约 378 GiB | 之后任何 +0.3% 的显存变动都可能触发 SIGBUS：09-15 加了常驻 scratch 缓存后峰值到 381.44 GiB（88.30%），就 SIGBUS 了 | 改用只改了层数的 8L flavor 做 A/B；启动前确认 GPU 上没有残留进程。见 (d) | 0913/phase1/RESULTS.md §10-§11；Primus:…/repro_l8b_bf16_mbs4_seq8k_turbo.yaml:20-24（注释）；0915__opt/E2E-AB.md:80-92 |
| 5 | Tensile split-K GEMM 内存故障，留下巨型 core dump | 09-13 第 5 次 e2e 尝试时，GPU memory fault 落在 `Cijk_Ailk_Bjlk_BBS_BH_MT64x1x64_…_GSU7`（Tensile split-K，形状像 LM head），写出 311 GB 的 GPU coredump。同一天 op 级进程异常退出时，还写了两个 8 GB 和一个 7.8 GB 的 core dump（root 所有，落在仓库目录里），磁盘一度用到 80% | 未归因。单独测主要的训练 GEMM 都能过；可能是这张卡已有记录的间歇 page fault | 一次故障就能把磁盘写满 | 关掉 coredump 后重试，跑通了；core dump 加进 gitignore。后来规定卡上不跑 fp32 GEMM，只用镜像自带的 hipBLASLt，见 (d) | session 541e7bc3 2026-09-13T05:25:48Z、06:26:24Z-06:28:33Z、08:17:58Z |
| 6 | torch.compile：关着是因为会挂卡，单独打开又是负收益 | 复现配置里 `compile.enable: false`，注释说本机上 inductor 对 TransformerBlock 做 triton autotune 时会触发 `hipErrorLaunchFailure`，把 GPU 打死。`FlexAttentionWrapper._compiled_flex_attn` 是 ClassVar，在类定义时就 compile 了，`compile.enable: false` 管不到它。B0 09-14：只开 compile，2,394→2,295 tps（−4%），峰值显存 380.1→260.7 GiB | inductor 默认把 matmul 留在 `aten.mm`（也就是 hipBLASLt）上，只融合 elementwise | compile 单独解决不了 GEMM 问题。09-15 晚一次 `converters: []` 的 8L 运行，在 flex 内部 compile 的 inductor autotune（`flex_decoding`）阶段把 A0 wedge 了，要 AC 断电才能恢复（n=1，见 #9） | B0 09-14 加上 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` 和 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON` 后到 9,602 tps（4.01x，attention 仍是 flex）。A0 上 compile 一直关着，GEMM 改走 nkfix | Primus:…/repro_l8b_bf16_mbs4_seq8k_turbo.yaml:58-65；ph1/E3.md §5；0914__campaign/RESULTS.md §4, §7；0915__opt/E2E-AB.md:307-335 |
| 7 | torchtitan 开箱即用时的度量和语义陷阱（后来才发现） | ① `get_peak_flops()` 遇到不认识的 device_name（本机报 "AMD Radeon Graphics"）时，兜底用 A100 的 312 TF/s，所以日志里的 MFU 一列全部无效：09-13 的 4.54% = 14.17/312，B0 的 44.44% = 138.64/312，后来还出现过 178%、215.63%、245%。② 单卡且 `debug.seed` 为 None 时，`set_determinism` 直接 return，不播种：5 次运行给出 5 个不同的 step-1 loss，A/B 两臂的初始化也不一样。③ `8B_flex` 每一步都调用一次被 torch.compile 过的 `create_block_mask` 来构建 block_causal mask，而 Primus llama3 的 `Attention.forward` 调 `inner_attention(xq,xk,xv)` 时把 mask 直接丢掉了，实际算的是纯 causal | torchtitan 只认识 NVIDIA 的设备名；单卡分支提前返回；Primus 与 torchtitan 的接口不一致 | MFU 不能引用，只能看 tps 和 tflops。不播种的 A/B 里混进了初始化噪声。算的东西和 JIRA 里的 flex（block_causal）不是同一个计算：E3 测得 block_causal 的密度是 99.25%，性能上没差别，但 loss 不能直接对照 | 只报 tps/tflops，需要时用实测峰值作分母（09-11 的 1002.7 不是 roof，见 a.1 BLAS 行）；09-15 起固定 `debug.seed: 1234`；mask 问题至今未修 | 0915__opt/GEMM-WGRAD-FINDING.md:78-94；0915__opt/E2E-AB.md:189-201；0927__b0/e2e/VERIFY.md §1-§3；ph1/E3.md Headline |
| 8 | JIRA 的起点数字和归因里，有几处不成立 | JIRA（jira_2，seq 8192）：MI455X 19,795 tok/s/GPU，MI355X 21,351；FA path 755 vs 315 ms/step；GEMM 655 ms/step，比 MI355X 快 1.51x；目标约 30,000 tok/s。① "turbo_attention patch 被 Skipped"不对，见 #2。② "在 gfx1250 上启用 AITER fmha"走不通，因为 CK 只支持 CDNA。③ 245 ms 的 elementwise 不是 attention：flex fwd+bwd 里的 elementwise 只有 0.399/31.857 ms = 1.25%，折合每步约 12.8 ms，差了 19 倍（推导）。09-13 进一步推测它是 compile 关闭时没融合的 eager RoPE 和 layout 转置，但原文自己注明"这个拆分是推导，不是实测"（trace 不在仓库，"492" 在磁盘上搜不到）。这个推测**已被取代**，实测归因见右栏。④ JIRA 里训练中 flex 15.656 ms/call 在 A0 上复现不出来（standalone 是 31.0-31.5 ms）。mask 稀疏的假说也被否定了：mock 数据每篇 24,327 token，block_causal 的密度是 99.25%，实测 31.133 vs 31.012 ms | JIRA 的数字来自另一套环境：BLAS 正常，而且没有限频 | 目标得重定。09-13 的论证是：A0-VR 下 30k tok/s 要求 attention 跑到 1282.8 TF/s，是 Triton GEMM "roof" 1002.7 的 128%，"by construction 不可能"。这个推论依赖的 roof 后来不成立：09-15 同卡同时钟 hipBLASLt NT 实测 1502 TF/s（1282.8 是它的 85%），所以"不可能"也不成立。09-14 另把 1002.7 按时钟线性换算到 2350 MHz（2142 TF/s，换算值，非实测），修正为 59.9%，"基本就压在结构墙上"，这个分母同样出自那个 Triton 数。另外，JIRA 原始 trace 显示 JIRA 环境里 MI455 的 GEMM 和 attention 前向都比 MI355 快，慢只慢在 attention 反向（见右栏） | 09-17 解析了原始 trace（和 jira_1 表里 414.3 ms 那一行完全对上）：MI455 一侧跑的是 inductor 的 Triton Flex，单步 attention 414.3 ms = fwd 105.9 + bwd 308.4；MI355 的 AITER 是 293.7 = 121.5 + 172.2。MI455 前向反而快 1.15x，反向慢 1.79x，差距全在反向。trace 里还有：MI455 eager 24,651 tks、compile+autotune 27,891 tks，两次的 attention 分别是 410.1 / 414.3 ms，几乎没变，e2e 的 +13% 全来自别处；GEMM MI455 534.8 ms 对 MI355 997.9 ms，MI455 快 1.87x（与 jira_2 "GEMM 快 1.51x" 同向，配置不同）；MI355 两次运行 21,521 / 21,502 tks。这些 trace 吞吐和 jira_2 表里的 19,795 不是同一次运行。09-28 B0 自己的 e2e profile（同样 compile 关）实测：elementwise 约 265 ms/step，其中 AMP 每步把 fp32 权重 cast 成 bf16 共 492 次（27 ms，对上了 JIRA 的 "×492"），bf16 梯度累加到 fp32（21 ms），rms_norm 因 dtype 不匹配走非融合路径（pow/mean/rsqrt/mul 约 25 ms），silu fwd/bwd 28 ms，softmax/CE 9 ms | 0913/phase0/PLATFORM-ESCALATION.md；0913/index.html §02-§03；0913/phase0/PHASE0-STATUS.md:138-140；ph1/E3.md Headline, §5；0914__repro__c07/RESULTS.md:100-133；0915__opt/GEMM-NN-FINDING.md:70；0915__opt/JIRA-TRACE-ANALYSIS.md:7-48；0927__b0/e2e/RESULT.md §3（:108-114） |
| 9 | 平台 wedge 和恢复成本（从 e2e 角度看） | 优化前 A0 共 wedge 5 次（见 a.1）。09-13 的 e2e 本身没有触发 wedge，但 rocBLAS 的 Tensile 路径出过 page fault（#5）。后来（09-15/16）又查出：训练启动阶段有一类 MES 故障，约 21% 的启动会失败（3/14）；09-15 晚一次 `converters: []` 的 8L 运行（`repro_l8b_noconv_8L_fast.yaml`，事后已删）停在 inductor 的 `AUTOTUNE flex_decoding`，卡 wedge（n=1）。这不是必然的：09-13 A0-VR 上同样 `converters: []` 的 32L 运行（245/244 tps、AC full）和 09-14 B0 的运行都跑完了，触发条件没定；`modprobe -r amdgpu` 会让整机失联 | MES 固件 / 驱动的 reset 路径无法恢复，根因一直没定 | 每次 wedge 都要人工 AC 断电或重启，再 `sudo modprobe amdgpu`、重建容器 | 见 (d) | ph1/HARDWARE-ISSUE.md §2；0915__opt/MES-WEDGE.md；0915__opt/E2E-AB.md:307-335（单次事故）；0913/phase1/RESULTS.md §10-§11 与 Primus `repro_l8b_bf16_mbs4_seq8k_turbo{,_ac}.yaml:30`（`converters: []` 跑通）；0914__campaign/HANDOFF.md §4；`~/.claude/skills/gfx1250-card-safety/SKILL.md` |
| 10 | 镜像自带的 primus_turbo 会遮蔽自己的 checkout | 镜像里 `/workspace/Primus-Turbo` 的 editable 安装（`primus_turbo 0.4.1.dev12`，09-09 构建）通过 `.pth` 注册了 MetaPathFinder，优先级比 `PYTHONPATH` 高。09-13 最初几次 harness 静默测的是镜像里的版本，而不是正在改的 checkout；09-14 B0 e2e 训练 import 的也是镜像版本 | `sys.meta_path` 的查找先于 `sys.path`；加进 sys.path 的是脚本所在目录，不是 repo 根目录 | 造成"改动没有效果"的假象；09-14 那批 turbo e2e 数据作废，这是原因之一 | 在容器里 `pip uninstall -y primus_turbo`（并补装 psutil）；harness 每一行都记录 `turbo_path`。不要用 Primus 的 `REBUILD_PRIMUS_TURBO=1`：它会按 gfx942;gfx950 构建 main，还会拉进不支持 gfx1250 的 `triton>=3.7.0` | 0913/phase1/RESULTS.md §5；0914__campaign/RESULTS.md:277-300；wt-llama31 README §2 第 4 条 |
| 11 | Primus 的 turbo 默认配置顺带打开 FP8 / MX linear | Primus `primus/configs/modules/torchtitan/pre_trainer.yaml:154-161` 的 `primus_turbo` 段默认 `use_turbo_async_tp: true`、`use_turbo_mx_linear: true`、`use_turbo_float8_linear: true`。只想换 attention 的 BF16 实验，如果不显式关掉，这几个 patch 也会被应用（09-14 B0 日志里 patch 13/13 全上）。按本机 Primus @`e7968675` 的代码，float8/mx 两个 patch 只替换对应 converter 的实现，`converters` 里列了 float8/mx 才会真正改 Linear | Primus 模块级默认就是 true（`config_extension.py` 的 dataclass 默认是 false，但 `pre_trainer.yaml` 覆盖成 true）。本机 09-13 的 `_turbo.yaml` 和 09-16 的 `repro_l8b_turbo_conv.yaml` 都只设了 `enable_primus_turbo` / `use_turbo_attention`，没关这几项。它们从没在 gfx1250 上验证过 | 实验里混进了未经验证、也不是本意的 patch。HANDOFF 认为 09-14 B0 的"turbo vs flex 1.459x"量到的正是这两个 linear；这个归因存疑（见 #2），但 1.459x 不来自 attention 这一点是确定的，结论已撤回。A0 上那些运行里 FP8/MX linear 是否真的生效，没有查到记录 | BF16 配置里显式设 `use_turbo_float8_linear` / `use_turbo_mx_linear` / `use_turbo_async_tp: false`（10-05 教程 `l8b_mi455x.yaml.in:83-85` 的做法） | 0914__campaign/HANDOFF.md:78-79；0914__campaign/RESULTS.md §10；Primus `primus/configs/modules/torchtitan/pre_trainer.yaml:154-161`、`primus/backends/torchtitan/patches/turbo/{fp8,mx}_linear_patches.py`、`primus_turbo_extensions/config_extension.py:19-29`；Primus `examples/torchtitan/configs/MI455X/repro_l8b_turbo_conv.yaml:94-97`；wt-llama31 README §2 第 5 条 |

### a.4 优化前基线数字

**a.4.1 算子（单层 fwd+bwd，b4 s8192 hq32 hkv8 d128 bf16 causal）**

| 后端（代码状态） | 机器 / 时钟 | 日期 | 尺子 | fwd ms | bwd ms | fwd+bwd ms | TF/s（fwd+bwd，7.697e12） | 分方向 TF/s（换算） | 来源 / 备注 |
|---|---|---|---|--:|--:|--:|--:|---|---|
| turbo Triton 出厂单 config（镜像自带的 `primus_turbo 0.4.1.dev12`，09-09 构建；推断与 main `c1325c7e` 是同一个 kernel，未在源码层核对） | A0-VR | 09-10 | bench0910 | 10.366 | 48.059 | 58.424 | 131.7 | fwd 212.2 / bwd 114.4 | bench0910。A0-VR 上能代表 main 状态的只有这一行。09-13 04:53 UTC 在分支工作树上复测为 60.209 ms，相差 3.0%（0913/phase1/RESULTS.md §1） |
| turbo Triton 出厂配置，09-13 分支工作树（FINAL-TABLE 把它对应到 `@1cb2e183`；**不等于 main**） | A0-VR | 09-13 | tune（bake-off） | 10.673 | 48.969 | 59.642 | 129.0 | fwd 206.1 / bwd 112.3 | 0913/phase1/BAKEOFF.md §1（08:45 UTC）。同日另两次测得 59.319（129.7，05:00 UTC）和 60.209（127.8，04:53 UTC），也都是分支工作树；09-15 复测 55.785（138.0，漂移 +6.9%）。FINAL-TABLE 脚注 †：`c1325c7e` 这个代码状态 09-13 从没测过；B0 上同尺子两者出厂数差 7.6%（下两行） |
| turbo Triton 出厂，main `c1325c7e` 原样（另打 low_precision 补丁） | B0（4 卡同时有负载） | 09-14 | tune，n=4，离散 1.35% | 5.226 | 25.025 | 30.258 | 254.4 | fwd 420.8 / bwd 219.7 | 0914__repro__c07/FINAL-TABLE.md |
| turbo Triton 出厂，`@1cb2e183`（09-13 分支树） | B0（4 卡同时有负载） | 09-14 | tune，n=3，离散 1.20% | 5.179 | 22.942 | 28.121 | 273.7 | fwd 424.7 / bwd 239.7 | 0914__repro__c07/FINAL-TABLE.md。对 09-13 A0-VR 的 59.642 为 2.12x |
| 同一个 Triton kernel（origin/main 到 `8cda13c7` 都没改） | A0-RF | 09-30 | bat | 5.11 | 24.49 | 29.60（相加） | — | fwd 430.45 / bwd 224.50（bat 自报，口径相同） | 0930__port/runs/bench_turbo_triton.csv:55。尺子不同，只作"刷固件后的 main"参考 |
| torch flex（torchtitan 的 compile 选项；e2e 现役） | A0-VR | 09-13 | tune | 7.506 | 23.831 | 31.337 | 245.6 | fwd 293.0 / bwd 230.7 | BAKEOFF.md。bench0910 测 31.508（244.3），E2 测 31.185（246.8） |
| torch flex | B0 | 09-14 | tune（自写 flex_anchor，autotune 后 bwd num_warps=4） | 3.771 | 11.536 | 15.307 | 502.8 | fwd 583.2 / bwd 476.6 | FINAL-TABLE。最初用 inductor 默认（bwd num_warps=8）测得 37.122 ms，已撤回 |
| aiter Triton MHA 出厂（不是 turbo 的默认路径） | A0-VR | 09-11 | E2 | 6.652 | 27.456 | 34.111 | 225.6 | fwd 330.6 / bwd 200.3 | ph1/E2.md §5（extended-v2，amd-aiter 0.1.14.post1） |
| 同上 | A0-VR | 09-13 | tune | 6.676 | 27.888 | 34.565 | 222.7 | fwd 329.4 / bwd 197.2 | BAKEOFF.md（`aiter-src @ffa945f9`）。09-15 的 28.035 存疑，见 a.2 #12 |
| 同上 | B0 | 09-14 | tune | 3.366 | 13.838 | 17.204 | 447.4 | fwd 653.4 / bwd 397.3 | FINAL-TABLE |
| torch SDPA FLASH（AOTriton） | A0-VR | 09-10 / 09-13 | bench0910 / tune | 41.849 | 54.733 | 96.581 / 97.063 | 79.7 | fwd 52.6 / bwd 100.5 | bench0910；0913/phase1/RESULTS.md §1 |
| 参照（非本机）：JIRA 的 MI455X inductor Flex / MI355X AITER | JIRA 环境，时钟未知 | 09-17 解析 | 原始 trace 按 kernel 求和 ÷32 | 3.309 / 3.797 | 9.638 / 5.381 | 12.947 / 9.178 | — | — | 0915__opt/JIRA-TRACE-ANALYSIS.md。MI455 用的是 compile+autotune 那次 trace（`mi455-case3`，27,891 tks），MI355 是 `mi355-case4`（21,521 tks）。另一口径是 JIRA 的"FA path"755 / 315 ms/step，÷32 后为 23.594 / 9.844 ms/call（bench0910 的 reference 行） |

**a.4.2 GEMM（决定 e2e 天花板）**

| GEMM 路径 | 机器 / 时钟 | 日期 | 形状 | ms | TF/s（2MNK） | 来源 / 备注 |
|---|---|---|---|--:|--:|---|
| rocBLAS（`TORCH_BLAS_PREFER_HIPBLASLT=0`，`amdprimus` 镜像） | A0-VR，空闲卡 | 09-13 | bf16 8192³ | 40.12 | 27.4 | 0913/phase1/RESULTS.md §11；09-15 复测 39.78 ms / 27.64 |
| hipBLASLt（`extended-v2`，已设 LIBPATH） | A0-VR | 09-11 | bf16 8192³，NN / NN(c) / NT / TN | 11.896-18.253 | 60.2-92.4 | ph1/E1.gemmchk.log |
| hipBLASLt（`amdprimus` 镜像，容器 fa-e2e） | B0 GPU3 | 09-14 | bf16 8192³ | 9.71 | 112.98 | 0914__campaign/RESULTS.md §3 |
| Triton（inductor）bf16 GEMM（当时被当作 roof，后被下面的 hipBLASLt NT 推翻） | A0-VR（1008 MHz / 1770 W） | 09-11 | 8192³ NT，持续 6 s × 3 次 | 1.096-1.097 | 1002.7 | ph1/E1.md §5 |
| 朴素 Triton GEMM | B0 GPU3 | 09-14 | bf16 8192³ | — | 1190.07 | 0914__campaign/RESULTS.md §3（和 hipBLASLt 同进程、同张量） |
| Triton GEMM（BM128/BN256/BK32/w8/s3） | A0-VR | 09-15 | bf16 8192³ | 1.225 | 897.53 | 0915__opt/BLAS-FINDING.md「实测」 |
| hipBLASLt（`fa-tune:deps`，修正 LIBPATH），torch 层 NN（`torch.mm(a, b)`） | A0-VR | 09-15 | bf16 8192³ | 15.99 | 68.74 | 0915__opt/BLAS-FINDING.md；VENDOR-REPORT-hipblaslt.md Defect 1 |
| 同上，torch 层 NT（`a @ bt.T` / `F.linear`，落到 hipBLASLt 的 TN 调优库） | A0-VR（1100 MHz 上限，负载采样 943 MHz） | 09-15 | bf16 8192³；模型真实形状 | 0.732；0.79-3.06 | 1502.0；1222-1626 | 0915__opt/GEMM-NN-FINDING.md:70, 118；session 11052787。和 09-11 `extended-v2` 那行（NT 91.3）镜像不同、库也不同，两边的布局标签定义是否一致未核对 |

**a.4.3 e2e（Llama-3.1-8B 32L，MBS=GBS=4，seq 8192，单卡，AC none，compile 关）**

| 配置 | attention（实际在跑的） | GEMM 路径 | 机器 / 时钟 | 日期 | tok/s | step | TFLOP/s（torchtitan） | 日志 MFU（A100 分母，无效） | 峰值显存 | 来源 |
|---|---|---|---|---|--:|---|--:|--:|---|---|
| `repro_l8b_bf16_mbs4_seq8k_turbo.yaml`（标称"turbo:TRITON 调优"，`converters: []`） | flex | rocBLAS | A0-VR | 09-13 | 245 | 133.7 s | 14.17 | 4.54% | 379.50 GiB（87.85%） | 0913/phase1/RESULTS.md §10 |
| flex 基线配置 | flex | rocBLAS | A0-VR | 09-13 | 244 | 约 134 s | 14.12 | 4.53% | 378.00 GiB | 同上 |
| turbo 配置 + AC full | flex | rocBLAS | A0-VR | 09-13 | 215 | — | — | — | 174.79 GiB（40.5%） | 0913/phase1/RESULTS.md §11 |
| torchtitan v0.2.2，20 步，取后 8 步中位数 | flex | hipBLASLt（约 113 TF/s） | B0 GPU3 | 09-14 | 2,392-2,394 | 13.7 s | 138.5-138.64 | 44.44% | 380.1 GiB | 0914__campaign/RESULTS.md §4, §10 |
| 同上，再开 compile（inductor 默认） | flex | hipBLASLt（`aten.mm`） | B0 GPU3 | 09-14 | 2,293-2,295 | 14.28 s | 132.78-132.9 | 42.61% | 260.6-260.7 GiB | 0914__campaign/RESULTS.md §4、§7、§10 |
| 参照（非本机）：JIRA（jira_2，seq 8192） | MI455X 是 flex，MI355X 是 AITER | — | JIRA 环境 | — | 19,795 / 21,351 | FA path 755 / 315 ms | — | — | — | 0913/phase0/PLATFORM-ESCALATION.md |
| 参照（非本机）：JIRA 原始 trace（09-17 解析） | MI455X inductor Flex / MI355X AITER | MI455 GEMM 534.8 ms/step，MI355 997.9 | JIRA 环境，时钟未知 | 08-31..09-10 | MI455 eager 24,651、compile+autotune 27,891 / MI355 21,521、21,502 | attention 410.1（eager）、414.3（compile）/ 293.7、294.2 ms | — | — | — | 0915__opt/JIRA-TRACE-ANALYSIS.md:7-48。和 jira_2 的 19,795 不是同一次运行 |

说明：
- "main + `PRIMUS_TURBO_ATTN_BACKEND=TRITON`"的 e2e 从来没实测过。10-05 教程用 op 级时间和训练中的 attention 时间估算过，约 1.9-2.05 s/step（约 16-17k tok/s）。这个估算的前提是 nkfix 等 GEMM 修法已经到位，并且是在 A0-RF 上，不属于"优化前"的状态（wt-llama31 README §2、§7）。
- 后续阶梯用的 32L e2e 基线 1,984 tok/s，来自 09-15 的 A0-VR，当时 hipBLASLt 路径已修正，attention 也已经是 ASM fwd + 融合 bwd，所以它不是 main 的数。见 (b)(c)。
- 09-13 A0 的 245 tok/s 和 09-14 B0 的 2,394 tok/s 不能当作同一配置下的对比：时钟不同（A0-VR 对 B0），BLAS 路径也不同（A0 走 rocBLAS，B0 走 hipBLASLt）。两机的 torchtitan 都是 v0.2.2（见 a.1）。两者都被 BLAS 卡住，attention 只占不到 1%（A0）或很小一部分（B0）。
- GEMM-NN 问题没修时的 e2e 基线（hipBLASLt 路径已修、无 nkfix）：09-15 A0-VR 32L 2,027 tok/s（3 步，n=1）；09-28 B0 32L 1,856-1,922 tok/s（17.0-17.7 s/step，attention 已是 Triton / ASM / FlyDSL 三种之一），见 a.3 #3。

**小结。** 优化前，main 在 gfx1250 上跑 Llama-3.1-8B attention 有三层问题：
1. 默认根本跑不起来。torch 2.11 下 import 就崩；默认派发落到只支持 CDNA 的 CK；测试全被 skip，所以没有任何信号。
2. 唯一能手动 pin 的 Triton 后端用的是照搬 CDNA 的单一配置，比现役 flex 慢约 1.9x，其中 dkdv 占 61.5%。
3. e2e 测不出任何 attention 改动。首先，镜像的 BLAS 坏了或者极慢（rocBLAS 27.4 TF/s，后来查明是路径错位，以及 NN 布局缺调优库）。其次，`converters: []` 让 e2e 里跑的始终是 flex。再次，Primus 的 turbo 默认配置把 FP8/MX linear 一起打开，09-14 那次"turbo vs flex 1.459x"两臂都是 flex，差别的真正来源至今未定。

另外，A0 上的所有数字都处在 VR 限频（1100 MHz）这个前提下，而且这张卡 09-04..09-13 平均约两天 wedge 一次（共 5 次）。所以本节每个绝对数都必须带上机器和时钟标签；跨机器、跨时钟的倍数也要说清是哪一种（见"记号与口径"）。

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

## (c) 优化过程每一轮的进展

本节先用一张里程碑速览（c.2a）给出主线，再把 09-10 到 10-05 的每个冠军变化点按时间排成一张总表（c.2），最后给当前最优版本（c.3）和进展要点（c.4）。每一轮单独成行的明细见 `parts/rounds.csv`（198 行，不合并）。路径若无前缀，均相对 `Primus-Turbo/output/`；`OE:` 表示 `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/`。

### c.1 口径说明

**机器与时钟阶段。** 绝对值只在同一阶段内比较；跨阶段只比较同进程比值。

| 代号 | 时间 | 机器 / 固件 / 驱动 | 负载时钟 | 说明 |
|---|---|---|---|---|
| A0-H | 09-04 及之前 | A0 未限频（这一阶段的固件/驱动版本没有单独记录） | 09-04 负载下 1699–1703 MHz | 本节没有这一阶段的 attention 测量。c1325c7e 提交信息里的 220.6 TF/s（b=2 s=8192）测量条件不明：提交信息只写 "seq 8192, B2 H32 D128, fwd+bwd"，没写是哪台机器、hkv、是否 causal 和时钟。09-13 同 kernel 在限频卡上 b2 实测 129.4，比值 1.70×；当天按估算的约 1.65× 时钟比判定它是在未限频的卡上测的，这个判定没有定论（见 a.2 #11）[0913__opt_plan__claude/phase1/RESULTS.md §0,§2；git show -s c1325c7e] |
| A0-T | 09-10..09-28 | A0 = heliosr-1b114-c07-1，单卡；VBIOS 113-M4500001-630A，SMU 125.7.1，amdgpu-dkms 7.1.1.31300009-2397345 | sclk DPM 只剩 500/1100 MHz（VR 限频）；op 级约 0.94–1.07 GHz（09-24 prod 窗口实测 998–1029 MHz） | 最早的限频记录是 09-10 22:38，09-11 由 op-evolve Phase-1 探针写成 HARDWARE-ISSUE.md；09-05..09-09 没有记录 [phase0/PLATFORM-ESCALATION.md:11-28,98-101]。09-29 之后，这一阶段的绝对数全部作废 |
| B0-14 | 09-14 | B0 = ctheliosp-1b112-a37-1，4×gfx1250（GPU0 降级，只用 GPU1–3）；驱动 / VBIOS / SMU 版本未记录 | 2133–2244 MHz | 与 A0 用同一镜像；e2e 与 A0 09-13 一样用 torchtitan v0.2.2（Primus 子模块 09-10 起固定在 73a0e6979）。`0914__campaign/RESULTS.md:65` 把 A0 09-13 记成 0.1.0，是笔误（见 a.1） |
| B0-28 | 09-27..09-28 | B0，每卡一个快照容器 fa-g0..g3；09-28 GPU1 wedge，交还时仍未恢复；驱动 / VBIOS / SMU 版本未记录 | op 级（分块尺子）约 2.15 GHz；nkfix 后训练中约 1.34–1.39 GHz（整板功耗墙约 2.13 kW） | 09-27/28 的 B0 会话不在本机，数字只来自文档和账本 |
| A0-R | 09-29..10-01 | A0 刷固件：VBIOS 700E，SMU 125.12.0，amdgpu-dkms 7.1.0-2411946 | sclk DPM 500/2356/2400；op 级 1.74–2.03 GHz；满载 WMMA 约 1.4–1.8 GHz（2500 W 功耗墙）；训练中约 1.5 GHz | 开机仍打印 VR 告警，但 DPM 不再截到 1100 MHz。09-30 开机后 dmesg 有 56 次 pcie_pl 可纠正错误 [0928__a0_repro/REPORT.md §6.3] |
| A0-D | 10-01 起 | A0，amdgpu-dkms 7.1.0-2412954 | 同 A0-R；GEMM 突发后（训练工作点）约 1.28 GHz | randn 基线与 A0-R 一致（bwd s6 = ASM 的 104.1%，fwd 时间比 1.079）。10-02 GPU RAS 在 07:58:50、08:09:46 各报一批 pcie_pl 可纠正错误（57+58 条）。主机 CPU L3 取指致命错误（BERT）导致的重启跨阶段出现：09-28 两次、10-01 20:41，与 GPU 无关 [1002__e2e/E2E-PLAN.md §1,§7；0928__a0_repro/REPORT.md §5.3] |

软件栈：容器 fa-repro（镜像 fa-tune:deps，即 amdprimus/amdprimus:gfx1250-20260910 加 torchtitan 依赖），torch 2.11.0+rocm7.14.0a20260625，triton 3.6.0，aiter-src @6963ae9，torchtitan v0.2.2（73a0e6979）。flydsl 有三个版本并存：
- 0.2.4：镜像自带，产品分支固定用它；
- 0.3.2：A0 bwd job 冠军和手工 s1–s6 用它；
- 0.3.4.1：09-25 起 fwd job、B0、e2e 用它。

**尺子**
1. **tune_attention.py**（09-13..09-16）：一层 fwd+bwd，CUDA event 取中位数，iters 20，每次冲刷 256 MiB L2；四张量 SQNR 门（out/dq/dk/dv ≥50 dB）。报 fwd / bwd / total ms，TF/s 按 fwd+bwd 合计计算。
2. **op-evolve benchmark.py**，逐次交错（A0 两个 job 09-17..09-27；B0 fwd r5–r9；B0 bwd r24–r26）：
   - 每个 arm 取 51 次中位数，palindromic 交错，ASM beat 与候选在同一进程。
   - 三个形状：fast (1,1024,1024,8,2,128)、proxy (1,4096,4096,32,8,128)、prod (4,8192,8192,32,8,128)；**只按 prod 排名**。
   - gain = 本轮代码与历史最佳代码在同一会话重测后的比值。min_gain：bwd 先是 0（B0 09-28、A0 09-30 改为 0.007），fwd 为 0.007。
   - fwd job 从 r9 起不再与 beat 同进程（h28）。r10 因三形状算术平均 1.0022 被否后，operator 加了 gain_weights（prod 1 / proxy 0.25 / fast 0），r11 起生效 [a8fa47bf]。A0 bwd job 从 r25 起也加 gain_weights，fast 改按 min 计。
3. **分块计时**（blocked，h40/h41/h66）：B0 fwd 从 r10 的 act 起（act 在 09-28 03:16 UTC 已按 blocked 9 + lead 4 计时 [0927__b0/fwd/rounds/010/act.md]，而 ruler/REPORT.md 记 harness 约 03:40 UTC 换装）；B0 bwd 从 r27 起（progress.md 的 r27 已报分块 beat 817.8，而 ruler/bwd/REPORT.md 记安装时间约 05:00 UTC）。两处的安装时间记录都比实际使用晚，以轮次为准。之后 A0 一直沿用。每个 arm 先跑 4 次不计时，再连续计时 9 次，轮间 palindromic。审计发现逐次交错会把 FlyDSL/ASM 比值压低：fwd 约 25%，bwd 约 3.4%。改为分块后 A/A 误差为 fwd ±0.19%、bwd ±0.05%。
4. **真实数据与训练工作点**（h47）：输入用 B0 step 43 导出的 6 层真实 q/k/v。“gb”指每次计时前先跑 10 个 bf16 32768×4096×14336 GEMM，把时钟压到训练工作点（A0 约 1.28 GHz）。FlyDSL 对时钟敏感，ASM 几乎不敏感，所以 randn/blk 和 real/gb 两种条件下的 FlyDSL/ASM 比值可以差 25% 以上。
5. **e2e**：Llama-3.1-8B BF16，32 层（09-15/16 部分实验为 8 层），MBS=GBS=4，seq 8192，每步 32,768 token，单卡，不开 activation checkpoint，不开 compile；tps = tokens/s。
   - 09-16 起，比较 attention 的前提是 nkfix（运行时改写 aten::mm 的 GEMM 布局）。
   - NKFIX_CHECK=1 每步约多 33 ms（10-02 的 e2e 开着它，10-05 的教程关掉了它）。
   - torchtitan 的 mfu 列按 A100 峰值计算，不可用。

**FLOP 口径**
- 当前口径（op-evolve `tools/op_flops.py`）：fwd 2.199292e12，bwd 5.498229e12（5-GEMM 名义计数，causal 取 (s+1)/(2s)），fwd+bwd 7.697e12；ms = FLOP / TF/s。表中标 derived 的值都是这样换算的。
- 09-13..09-16 文档的 TF/s 是 fwd+bwd 合计（7.697e12 / total ms）。0913/0914 文档另按 7-GEMM 实发口径报过 bwd（例如 B0 vendored 融合 bwd 748.4 vs 534.6 TF/s），本节不用这个口径。09-14 B0 Triton op-evolve job 只计 bwd。
- 09-23 的 STAGE2-FWD-SWEEP 用 fwd ≈2.233e12（旧 harness，TF/s 偏高约 1.5%）。09-17 DKDV-FIRST-TIMING 的 5.8 TF/s 是自定义的 useful 口径（单头），不可比。
- 我方 1-wave bwd（r0 到 s6）实际发 7 个 GEMM，是算法量的 1.4076×；aiter ASM 只发 1.0155×。本节一律按算法量计。所以 s6 的“占 ASM 103.8%”，是在多做约 39% 矩阵运算的情况下达到的。

**“占 ASM”**：ASM 时间 ÷ 本行时间（等于本行 TF/s ÷ ASM TF/s）。ASM 优先取同进程或同会话的测量；大于 100% 表示比 ASM 快。09-14 之前还没有 ASM，记“—”。op-evolve 的轮次用该轮同会话的 beat。

**为什么跨阶段不可比（实测例子）**
- **同代码、同尺子，只换平台**（A0-T 09-28 → A0-R 09-30）：
  - FlyDSL fwd r13ns 2.023→1.353 ms（o1 顺序；反序 o2 为 1.992→1.356），提速 1.47–1.50×（两种顺序，均值约 1.48×）；ASM fwd 1.564→1.257（1.24×；o2 为 1.23×）；
  - FlyDSL bwd r29 11.010→6.583（1.67×），ASM bwd 7.654→5.503（1.39×）。
  - 连比值都变了：bwd r29 从 ASM 的 69.5% 变成 83.6%。
  - e2e 层面没有干净的同口径倍数。教程 §7 的 ASM attention + nkfix 1,947→1,351 ms（1.44×）不纯是平台差异：1,947 出自 09-28 已作废的 a0_p3a 进程，同进程的 FlyDSL arm 从 step 2 起加载，fwd 树的 `_env.py` 随即把 `HIPBLASLT_TENSILE_LIBPATH` 改指宿主库（hipBLASLt 是否中途重读这个变量未证实）；1,351 是 10-02 的读数（两边都是 NKFIX_CHECK=1），两次之间驱动也从 7.1.1-2397345 换成 7.1.0-2412954（中间经过 -2410994、-2411826、-2411946）[0928__a0_repro/REPORT.md §2,§5.3,§6.3；1002__e2e/E2E-PLAN.md §1]。
- **同机同代码，只换尺子**：B0 fwd 从交错改为分块后，占 ASM 从约 85% 跳到约 96%，kernel 没有变。r16 起冠军 r13ns 在 randn 上按设计读数低约 4.8%。
- **同代码，只换输入和工作点**：A0 10-02，bwd s6/ASM 时间比 blk 0.972、gb 1.038；fwd r16/ASM blk 1.081、gb 1.349。
- **AC-cycle 前后会漂**：A0 09-22 同一份 r7 代码，404.06 vs 382.73 TF/s（+5.6%）。同会话噪声地板 0.24–0.66%，跨会话漂移约 1.5%。

**各阶段的 ASM 基准（bar）**

| 阶段 | ASM fwd（prod） | ASM bwd（prod） | 尺子 | 来源 |
|---|---|---|---|---|
| A0-T 09-15 | 1.549–1.572 ms | 10.160 ms（**已撤回**：autograd shim，每次调用都 hipModuleLoad 并新分配约 1 GiB scratch） | tune_attention.py | 0915__opt/RESULTS.md；baselines.md §5 |
| A0-T 09-17 | 1.5691 ms（n=20，967 MHz） | 7.6134 ms / 722.2 TF/s（op-evolve setup 同会话） | stage1_fwd_ab / benchmark.py | 0917__flydsl/STAGE1-FWD.md；OE:…a0-stale-0930/job_context/history/review/op_setup_v000.md |
| **A0-T 09-24（引用值）** | **1.5724 ms / 1398.67 TF/s**（n=51） | **7.6766 ms / ≈716 TF/s**（n=224 raw；7.6769 n=146） | benchmark.py | 0924__flydsl/bar-census/fwd_anchor.json；0924__flydsl/DAY-SUMMARY.md |
| A0-T job 每轮 beat | 1395–1416 TF/s（fwd r1–r12） | 704–720 TF/s（bwd r1–r23） | 同会话 | OE: 两个 job 的 state.yaml |
| A0-T 09-28 | 1.543–1.564 ms | 7.654–7.666 ms | 分块 | 0928__a0_repro/REPORT.md §1 |
| B0-14 | 1.410 ms（安静 1.396） | 当天未测 | tune_attention.py | 0914__campaign/RESULTS.md |
| B0-28 交错 | 1751–1797 TF/s（1.22–1.26 ms，偏快） | 852 TF/s（6.45 ms，r24） | 交错（有偏差） | 0927__b0/fwd,bwd/progress.md；bwd/rounds/024/act.yaml |
| B0-28 分块 | 1549–1562 TF/s（1.41–1.42 ms） | 816.8–820.7 TF/s（6.70–6.73 ms） | 分块 | 同上 |
| A0-R 09-30 | 1.253–1.257 ms（1750–1755 TF/s） | 5.476–5.517 ms（约 1000 TF/s） | 分块 randn | 0928__a0_repro/REPORT.md §6；0930__bwd/PROGRESS.md |
| A0-R 09-30 产品 bench | 1.187 ms | 5.360 ms | 产品 bench，同进程 | 0930__port/PR_BODY.md |
| A0-D 10-02 | randn blk 1.190–1.262；真实 blk 1.209；gb 1.137 | randn blk 5.480–5.513；真实 blk 5.475；gb 5.530 | 分块 / 真实数据 / gb | 1002__e2e/RESULT-realab.md；1002__e2e/E2E-PLAN.md §1 |
| A0-D 10-02 e2e（训练中，每步 32 层） | 34.7 ms/步 | 166.5 ms/步 | CUDA event | 1002__e2e/RESULT-e2e.md |

**轮次编号**
- bwd 的 round 24–27 有两套。一套是 B0 在 09-27/28 跑的 r24–r32，记作“B0 r24”等，账本只在 `0927__b0/bwd/`。另一套是 A0 在 09-30 和 10-02 于 live 目录续跑的 r24–r27，记作“A0 r24”等。
- A0 bwd job r0–r23 的账本在 `OE:gfx1250-flydsl-attn-bwd-20260917-115934.a0-stale-0930/` 和 live 目录中逐项相同；r23 的 09-30 收口只记在 live 目录。
- fwd 有两个 job：A0 的 `gfx1250-flydsl-attn-fwd-20260925-114644`（r0–r13），以及 B0 的 `gfx1250-flydsl-attn-fwd-b0-20260927`（克隆自 A0 的 r4，在 B0 上跑了 r5–r20）。
- state.yaml 里按形状记的 champions 指针与 best_round 不一定相同。A0 bwd job：best_round=24，champions={prod: 25, proxy: 25, fast: 24}，r25 虽被否，却是 prod/proxy 的 best-ever 记录。A0 fwd job：best_round=11，champions={prod: 12, proxy: 11, fast: 11}。续跑时，这些被否轮次的读数会被当作 best-ever 门槛；B0 fwd 的 r9、r19 就是这样被挡住的。

### c.2a 里程碑速览

下表从 c.2 里挑出主线上的 14 个节点。序号就是 c.2 的行号，数字都取自 c.2 对应行，口径与 c.2 相同；跨机器、跨固件阶段的绝对值不能直接比（见 c.1）。“—”表示该行没有这一项。

| 序号 | 日期 | 里程碑 | fwd ms / TF/s | bwd ms / TF/s | 占 ASM | e2e | 机器/口径 |
|---|---|---|---|---|---|---|---|
| 2 | 09-13 | 优化前基线：树内 Triton 出厂（R0；main 原样 58.424 ms 见 #1） | 10.673 / — | 48.969 / — | — | 32L 133.7 s/步，245 tps（#9；GEMM 走 rocBLAS，attention 很可能没进 e2e） | A0-T，tune_attention；合计 59.642 ms，129.0 TF/s（合计） |
| 7–8 | 09-13 | A0 Day-1 Triton 终点：两个 config 旋钮 + vendor aiter 融合 bwd + shape gate | 4.148 / — | 20.201 / — | — | — | A0-T，tune_attention；合计 24.349–24.435 ms，315.0–316.1 TF/s（合计），相对出厂 2.44× |
| 14 | 09-14 | B0 09-14 冠军：ASM fwd + Triton 融合 bwd（in-thread transpose） | 1.396 / —（ASM） | 8.840 / — | fwd = ASM | — | B0-14 GPU1，tune_attention；合计 10.235 ms，752.0 TF/s（合计），相对出厂 2.75× |
| 17、43、47 | 09-15..09-24 | aiter ASM bwd 接入（opt-in，默认关；ASM fwd 已在 09-14 #12 接入）；09-24 定 ASM bar | 1.5724 / 1398.67（ASM bar） | 7.6766 / ≈716（ASM bar；09-15 的 10.160 ms 是 shim 伪数，已撤回） | 100% | nkfix 后 8L ASM bwd 开/关 +12.72%（#21） | A0-T，op-evolve benchmark.py；fwd n=51，bwd n=224 普查 |
| 18–22 | 09-15..09-16 | GEMM 修复：hipBLASLt 路径 + nkfix（e2e 的头号杠杆） | — | — | — | A0 32L 244→2,027 tps（#18）；A0 8L nkfix v3 37,746 tps，6.16×（#20）；B0 32L nkfix 后 ASM 臂 20,718 tps（#64） | A0-T 8L / 32L；B0-28 32L |
| 24、26 | 09-17 | 转向 FlyDSL：fwd 首测 aiter FlyDSL m32x8；bwd 从零手写，A0 bwd job r0 | 2.3732 / 926.7 | 96.05 / 57.24 | fwd 66.1%；bwd 7.9% | — | A0-T；fwd 用 stage1_fwd_ab n=20（同会话 ASM 1.5691 ms），bwd 用 benchmark.py 逐次交错（同会话 beat 7.6134 ms） |
| 42、52 | 09-24 / 09-27 | 限频期冠军：bwd A0 r20，fwd A0 r11 | 1.979 / 1111.15（r11） | 10.75 / 511.42（r20） | fwd 78.9%（对同会话 beat 1407.8；对 09-24 bar 为 79.4%）；bwd 71.4%（对 09-24 bar 为 0.71–0.72×） | — | A0-T，op-evolve benchmark.py 逐次交错，同会话 beat |
| 58、62 | 09-28 | B0 冠军：fwd r16 = r13ns（关投机 softmax），bwd r29（u2n） | randn ≈1.51–1.52 / ≈1443–1455 | 8.375 / 656.5 | fwd randn ≈93%；bwd 80.4% | 见 #66 | B0-28，分块尺子，randn |
| 66 | 09-28 | B0 e2e 终版：fwd r16 + bwd r29 | — | — | — | fly/asm 1.0323 / 1.0328（fly 1,631/1,633 ms，ASM 1,579/1,583 ms） | B0-28 GPU0，32L，nkfix，NKFIX_CHECK=1 |
| 69 | 09-30 | A0 刷固件后同尺子复测（代码未变，新基线） | r13ns 1.353 / 1626（ASM 1.257 / 1750） | r29 6.583 / 835（ASM 5.503 / 999） | fwd 92.9%；bwd 83.6%（同代码 09-28 为 77.3% / 69.5%，见 #67，已作废） | — | A0-R，分块，randn |
| 77 | 09-30 | s6：当前 bwd 冠军（TDM 3 级 ring + 下一轮 B 操作数预读 + VALU 精简） | — | 5.300（反序 5.295）/ 1037–1038 | bwd 103.8% | 见 #81 | A0-R，分块，randn，同进程 ASM |
| 78 | 09-30 | 产品分支移植：fwd r16 + bwd r29 → flydsl 0.2.4 | 1.292 / 1702 | 6.618 / 831 | fwd 91.9%；bwd 81.0% | 见 #87 | A0-R，产品 bench，同进程 ASM 1.187 / 5.360 ms |
| 81 | 10-02 | A0 e2e：fwd r16 + bwd s6，单步与 ASM 持平 | 每步慢 9.2 ms | 每步快约 4 ms | 单步 0.9990 / 0.9985× | fly 1349.4/1349.9 ms，24,283/24,275 tps；ASM 1350.6 ms，24,262 tps | A0-D，32L，nkfix，NKFIX_CHECK=1，训练 sclk 约 1.50 GHz |
| 87 | 10-05 | 产品分支教程 e2e：fwd r16 + bwd r29（flydsl 0.2.4） | — | — | 折算到同口径约慢 3%（推算，未实测） | 1,361.9 ms/步，24,061 tps（NKFIX_CHECK=0） | A0-D；与 #81 的 NKFIX_CHECK=1 不同口径 |

### c.2 总表

单行“算子耗时 ms”里写成 a / b / c 的，依次是 fwd / bwd / total。TF/s 列的每个数后面都注明口径：“（合计）”= 7.697e12 ÷ fwd+bwd 总 ms，是 tune_attention 和 Primus bench 的报法，阶段 1–2 除 #13 外都是它；“（bwd）”“（fwd）”是单方向，分别按 5.498229e12、2.199292e12 计，#13 的 B0 Triton job 和阶段 3 起都是单方向。合计口径的数与单方向的数不能直接比。被否决的连续轮次合并成一行，每轮明细见 rounds.csv。

| 序号 | 日期 | 方向 | 轮次 | 算子耗时 ms | TF/s | 占 ASM | e2e 耗时/吞吐 | 机器 | 优化内容 | 备注（驱动/固件/口径） |
|---|---|---|---|---|---|---|---|---|---|---|
| **阶段 1 · Triton 基线与调优（A0-T，09-10..09-13；TF/s 为 fwd+bwd 合计 7.697e12）** | | | | | | | | | | |
| 1 | 09-10/11 | fwd+bwd | 优化前基线：main 的 turbo:TRITON 后端（c1325c7e，#481） | 58.424（同次 flex 31.508，sdpa FLASH 96.581） | 131.7（合计，bench 自报） | — | — | A0-T | 无（main 原样，需显式指定 Triton 后端） | 测时卡已在 VR 限频下（sclk 500/1100 MHz；最早的限频记录是 09-10 22:38，09-11 才写成报告）。默认派发会落到 CK，bwd 报 invalid argument。Primus bench_attention.py，log 为 Primus 仓库 `output/0910__fa_opt/attn_bench_gfx1250.log`（目录名 0910，文件时间 09-11 06:29 UTC）。09-13 同法复测 60.209 [0913__opt_plan__claude/phase1/RESULTS.md §1；phase0/PLATFORM-ESCALATION.md] |
| 2 | 09-13 | fwd+bwd | R0 出厂（tune_attention 基线） | 10.673 / 48.969 / 59.642 | 129.0（合计） | — | — | A0-T | 树内 Triton 的出厂单一 config（num_stages=1、num_warps=4，照搬 CDNA） | 同会话 flex 31.337 ms，turbo 慢 1.9×。镜像 amdprimus gfx1250-20260910 [0913__opt_plan__claude/phase1/BAKEOFF.md] |
| 3 | 09-13 | fwd | R1 | fwd 10.781→4.227（total ≈52.8） | ≈146（合计） | — | — | A0-T | fwd num_stages 1→2：fwd 快 2.42–2.55×，SQNR 逐位不变 | [phase1/RESULTS.md §3] |
| 4 | 09-13 | bwd | R2（f8c45dee） | 4.227 / 31.477 / 35.704（bake-off 复测 36.405） | 215.6；bake-off 复测 211.4（合计） | — | — | A0-T | bwd num_warps 4→2，bwd 快 1.54×；waves_per_eu=2/4 时掉入性能悬崖 | 只改 config，1000 次逐位确定。R3 的三处 dkdv 源码改动被否（mask-skip 反而慢 9%）[phase1/RESULTS.md §3,§7] |
| 5 | 09-13 | fwd+bwd | R4 Round-0 bake-off | aiter+2 knobs 31.207；flex 31.337；aiter 出厂 34.565；turbo 36.405 | 246.6 / 245.6 / 222.7 / 211.4（合计） | — | — | A0-T | 同会话、同门槛对比：树内实现输在 bwd（32.252 vs 27.888），于是转向 aiter 融合 bwd 结构 | 09-11 的 aiter 24.347 ms / 316.1 已撤回：上游 aiter 改过 bwd 配置，复测为 31.207 [phase1/BAKEOFF.md §1,§4] |
| 6 | 09-13 | bwd | R5–R7（纯 aiter 调参，作参考上界） | 25.283→21.672（bwd 18.416） | 304.4→355.1（合计） | — | — | A0-T | LSE-ABI 解耦（5eba2cf4）；非对称 tile 被证伪（64×64 已经最优）；aiter 融合 bwd 的 N1/M2 128→256（1.26×），BLK_SLICE_FACTOR 2→1（1.20×） | 首轮 sweep 因 import 绑定，override 没生效，作废。这个上界没有入树 [phase2/DECISIONS.md D1–D9] |
| 7 | 09-13 | bwd | R8：vendor 融合 bwd（1cb2e183） | 4.148 / 20.201 / 24.349 | 316.1（合计） | — | — | A0-T | vendor aiter 的 mha_onekernel_bwd（pin ffa945f9），配置硬编码，由 turbo 自己的 fwd 驱动 | [commit 1cb2e183；session 541e7bc3] |
| 8 | 09-13 | bwd | R9 shape gate（03a76f61→8dd2fd32），Day-1 终点 | 24.435 | 315.0（合计） | — | — | A0-T | 按并行工作量决定是否走融合 bwd，按 seqlen 选 tile；51 个 gfx1250 测试通过（main 上一个都不运行） | 相对出厂 2.44×，相对 flex 1.28× [0913__opt_plan__claude/PROGRESS.md] |
| 9 | 09-13 | e2e | 首次跑通 e2e | — | — | — | 32L 133.7 s/step，245 tps（flex 244） | A0-T | 补齐 torchtitan 依赖后，第一次跑完 step | GEMM 走 rocBLAS（27.4 TF/s）。converters: [] 导致 turbo attention 没进 e2e，两臂很可能都是 flex。当日误判为“缺 gfx1250 Tensile 库”，09-15 更正为路径错位 [phase1/RESULTS.md §10；0914__campaign/HANDOFF.md §4]。整条 attention 路径 35.7 ms × 32 = 1.14 s，不到单步的 1%（约 0.85%）[phase1/RESULTS.md:266]。当天约 13:19 UTC（13:30 发现）第 5 次 wedge（前 4 次在此前八天内，见 HARDWARE-ISSUE.md），需要重启主机 [0913__opt_plan__claude/phase2/INCIDENT-2026-09-13-wedge.md；PROGRESS.md:36-38] |
| **阶段 2 · ASM 接入（B0-14 09-14；A0-T 09-15..09-17；TF/s 除 #13 只计 bwd 外，都是 fwd+bwd 合计）** | | | | | | | | | | |
| 10 | 09-14 | fwd+bwd | B0 复现 Day-1 阶梯 | main@c1325c7e 30.258；出厂@1cb2e183 28.121；配置调优 18.916；vendored 融合 12.994；纯 aiter 上界 11.269 | 254.4 / 273.7 / 406.9 / 592.3 / 683.0（合计） | — | — | B0-14 | 无新代码，换到满频机重测 | 换机到 B0。七行加速 1.88–2.12×（报告按五行 like-for-like 取中位 1.93×），名次和 SQNR 都一致。这是 A0-VR→B0 的跨卡比，不是 A0 自身的限频代价：A0 自身刷固件前后，同代码同尺子 op 为 1.24–1.67×（o1 顺序），e2e 1.44×（不纯，见 c.1）。09-13 按时钟估的约 1.65× 与它不是同一个量，不再称作废 [0914__repro__c07/FINAL-TABLE.md；0914__repro__c07/RESULTS.md:126-130] |
| 11 | 09-14 | fwd+bwd | B0 step1（566e7798） | 2.579 / 9.867 / 12.412 | 620.5（合计） | fwd 54.7% | — | B0-14 | 融合 bwd 不设 waves_per_eu，fwd num_warps=2（+4.28%）；修掉 harness 把 0 当非法值的 bug | 0915 xlsx 按 ledger 中位数改记为 12.454 [0914__campaign/RESULTS.md] |
| 12 | 09-14 | fwd | B0 step2：T7 ASM fwd（0e5cb743） | 1.410 / 9.786 / 11.198 | 687.3（合计） | fwd = ASM | — | B0-14 | fwd 换成 aiter 预编译的 fmha_fwd_with_sink_asm（.co），fwd 快 1.82×，越过纯 aiter 上界 11.269 | 四卡同时负载；200 次逐位确定 [0914__campaign/RESULTS.md] |
| 13 | 09-14 | bwd | B0 Triton op-evolve job r0→r1（…bwd-20260914-015415） | bwd 10.017→8.683 | 548.9→633.2（bwd） | — | — | B0-14 GPU1 | r1：waves_per_eu 1→0，加 TRITON_HIP_USE_IN_THREAD_TRANSPOSE，同会话 1.153× | r2 gain 0.934（act.yaml 记 vs_champion 0.9345；另因 FastError 记 failed）、r4 gain 0.925 被否；r3 没有产出候选就放弃。09:44 UTC 用户中断，job 停在 r4，best_round=1。beat = flex bwd 11.68 ms [0914__campaign/op-evolve-artifacts/…/job_context/state.yaml、progress.md] |
| 14 | 09-14 | bwd | **B0 step3：ITT 入树（a009599b），09-14 冠军** | 1.396 / 8.840 / 10.235（四卡满载 10.299） | 752.0（合计） | fwd = ASM | — | B0-14 GPU1 | 融合 bwd 打开 in-thread transpose，bwd −9.7%；相对出厂 2.75×，相对 flex 1.50× | 安静条件复测，n=1 [0914__campaign/HANDOFF.md §15] |
| 15 | 09-14 | e2e | B0 e2e：GEMM 绕行 | — | — | — | flex eager 13.70 s，2,392 tps → compile + Triton GEMM 3.41 s，9,602 tps（4.01×） | B0-14 GPU3 | 让 inductor 强制走 Triton GEMM，绕开 hipBLASLt（113 vs 1190 TF/s） | torchtitan 0.2.2。当日的“turbo attention 13,204 tps / 1.459×”已撤回：镜像包遮蔽了 checkout，加上 converters: []，attention 实际是 flex [0914__campaign/RESULTS.md §10,§17] |
| 16 | 09-15 | fwd+bwd | A0 复现 B0 冠军 | 出厂 55.785 → 1.552 / 17.735 / 19.285 | 138.0→399.1（合计） | fwd = ASM | — | A0-T | 同一份代码回 A0：2.893×（B0 为 2.748×），SQNR 吻合到 9 位有效数字；ASM fwd 单独值 2.74×（4.253→1.549） | 回到 A0。新增 PRIMUS_TURBO_ATTN_DISABLE_ASM_FWD 开关 [0915__repro__c07/RESULTS.md；0915__opt/RESULTS.md] |
| 17 | 09-15 | bwd | T2：aiter ASM bwd 接入（opt-in） | 1.572 / 10.160 / 11.726 | 656.4（合计，作废） | bwd = ASM | ASM bwd 开相对关：32L 开了反而慢 6.78%（1,858 vs 1,984 tps，n=1）；8L tps −0.28%（n=9，0.15× sem，不显著）。32L 的 6.78% 先归因于显存压力，当晚 n=9 复测后撤回这个归因：本机 e2e 跨运行噪声约 3.9% 且多模态，6.78% 落在模态间距内 [0915__opt/E2E-AB.md:101,182,232] | A0-T | 手写 ASM bwd launcher。发现 aiter gfx1250 ASM bwd 在 GQA 下按 q head 越界写 dk/dv，改为 dkdv_heads=q 加 host 规约；资格门 seqlen≥2048，默认关 | **10.160 ms / 541 TF/s 已撤回**（autograd shim 没传 hip= 和 scratch=），真实 ASM bwd bar 是 7.6766 ms（09-24）。当日 modprobe -r 导致整机失联，做了 AC-cycle [0915__opt/RESULTS.md；baselines.md §5] |
| 18 | 09-15 | e2e | e2e 打通，修正 hipBLASLt 路径 | — | — | — | 32L 134 s→17 s/step，244→2,027 tps（8.3×，3 步，n=1）；20 步稳态 1,984 tps | A0-T | TurboAttention.forward 接受 enable_gqa，第一次真正用上 turbo attention；HIPBLASLT_TENSILE_LIBPATH 指向镜像的 library/gfx1250（GEMM 27.64→68.74 TF/s） | GEMM 路径从 rocBLAS 改为 hipBLASLt [0915__opt/BLAS-FINDING.md；0915__opt/E2E-AB.md] |
| 19 | 09-15 | e2e | nkfix v1（dgrad） | — | — | — | 8L 5.35→2.82 s/step，6,128→11,604 tps（1.894×，n=9） | A0-T | 查明 hipBLASLt 的 NN（dgrad）和 wgrad 布局缺纯 bf16 调优库，落到 MT32x16x32 GEMV tile；用 TorchDispatchMode 把 B 改成 N-major | e2e 方差 3.91%→0.42%，原来的三模态噪声也来自这里 [0915__opt/GEMM-NN-FINDING.md] |
| 20 | 09-16 | e2e | nkfix v3（wgrad） | — | — | — | 8L 867 ms/step，37,746 tps（6.16×，n=5） | A0-T | wgrad 的 A 改连续、B 改 N-major（rule 2 只改 A，命中 0 次，被否） | 原报 38,043 / 6.21× 含一次 NaN 运行，0917 更正 [0915__opt/GEMM-WGRAD-FINDING.md；PROGRESS-REPORT-0917.html] |
| 21 | 09-16 | e2e | ASM bwd e2e 重做 A/B | — | — | — | 8L ON 42,534 / OFF 37,734 tps（+12.72%） | A0-T | GEMM 修好后，attention 占单步从 3.3% 升到 20.8%，ASM bwd 才有 e2e 收益 | 原报 +14.40% 已更正；仍默认关 [0915__opt/BOTTLENECK-SHIFT.md] |
| 22 | 09-16 | e2e | nkfix rule 3：wgrad 改走 FlyDSL GEMM | — | — | — | 8L 707 ms，46,374 tps（+9.26%，累计 7.57×）；32L ≈2.33 s，14,050 vs 12,340 tps（+13.9%，相对 1,984 为 7.08×，n=1） | A0-T | wgrad 改调同事的 FlyDSL gfx1250 WMMA GEMM（离线建表），省掉每步 137 ms 的转置拷贝 | 带 nkfix 的运行有 21% 出现 loss NaN（p=0.004），没解决，rule 3 默认关。8L 49,878 和 32L 11,608 / 5.85× 都是 NaN 运行，已作废 [0915__opt/NAN-FINDING.md；0915__opt/RESULT-32L.md]。AC 断电按 d.1.4 去重统计：09-15 共 5 次（07:18 前整机失联、~08:20 SIGBUS、09:00 8L worker、11:16 noconv、13:07 nk4），09-16 共 9 次（02:26、07:35、07:56、08:20、09:18、11:17、11:57、12:44 有用户确认，~13:15–13:30 为推断）。DAY-0916-SUMMARY 记的“09-16 五次”（14 次启动有 3 次死在启动阶段，21%）只是 08:20 之后那一段 [d.1.4；0915__opt/DAY-0916-SUMMARY.md §四；0915__opt/INCIDENT-2026-09-15-machine-death.md；session b596bddb、11052787] |
| 23 | 09-16 | fwd/bwd | Triton 侧收尾扫描 | bwd 两 kernel 9.10→7.47（−17.8%）；fwd 1.424→1.371（−3.73%） | — | bwd 36.7% | — | A0-T | proxy s4096 口径。Triton bwd 仍比 ASM 慢 2.73×，这条路线关闭 | 09-17 BACKEND-STRATEGY 建议不移植 FlyDSL，用户同日决定转向 FlyDSL [0915__opt/TWOKERNEL-SWEEP.md；FWD-SWEEP.md；BACKEND-STRATEGY.md] |
| **阶段 3 · FlyDSL 移植（A0-T，09-17；从这里起 TF/s 分方向）** | | | | | | | | | | |
| 24 | 09-17 | fwd | Stage1：aiter FlyDSL m32x8 fwd | FlyDSL 2.3732 / ASM 1.5691 | 926.7 / 1401.6（fwd） | 66.1% | — | A0-T（窗口 967 MHz） | flydsl 0.3.2 不加任何 shim 就能跑通。“前向模板有竞争力”被证伪：比 ASM 慢 1.51× | flydsl 0.3.2 侧装（与镜像的 0.2.4 不能同进程）；n=20，ABAB 交替 [0917__flydsl/STAGE1-FWD.md；530dd77e] |
| 25 | 09-17 | bwd | Stage3 bring-up 和 dkdv 首测 | dkdv_loop 单头 S=4096：1.475 | 5.8（bwd，自定义 useful 口径） | “1/93” 已撤回 | — | A0-T | 用 6 个探针排除未知量；odo/dkdv/dq 三个 kernel 首跑即通过（141–159 dB） | “慢 93×”已撤回：单头的 grid 填不满卡，同一 kernel 在 prod 形状是 57.2 TF/s [0917__flydsl/DKDV-FIRST-TIMING.md；HANDOFF-0917.md] |
| 26 | 09-17 | bwd | A0 bwd job r0（setup 基线） | 96.05 | 57.24（bwd） | 7.9% | — | A0-T | 从零手写 k_delta_bshd + k_dkdv + k_dq：每个 WG 一个 wave32，GQA 在寄存器里规约，无 atomic，结果确定 | job gfx1250-flydsl-attn-bwd-20260917-115934 于 11:59 发车，同会话 beat 7.6134 ms / 722.2 [OE:…a0-stale-0930/job_context/history/review/op_setup_v000.md]。r1 首次尝试已测到 armA（causal tile 跳过）prod 58.272 ms / 94.4 TF/s（同次 cur 57.4、beat 7.608 ms / 722.7，13.1%），armB（32 深收缩）64.8 TF/s；13:17 交机，r1 未判即中断，09-21 重跑 [OE:gfx1250-flydsl-attn-bwd-20260917-115934/rounds/001/_scratch/benchAB.json] |
| **阶段 4 · bwd op-evolve（A0-T，r1–r23，09-21..09-25；ms 按 5.498229e12/TF/s 换算；占 ASM 用同会话 beat）** | | | | | | | | | | |
| 27 | 09-21 | bwd | A0 r1 | 59.36 | 92.62（bwd） | 12.9% | — | A0-T | g01：causal tile 跳过（gain 1.342） | sclk 1055–1100 [OE:gfx1250-flydsl-attn-bwd-20260917-115934/job_context/state.yaml（下同）] |
| 28 | 09-21 | bwd | A0 r2 | 38.39 | 143.23（bwd） | 19.9% | — | A0-T | g07 32 深收缩，buffer descriptor 的 num_records 改用真实 extent（修越界读）；g06 BLOCK_KV 16→32（gain 1.921） | |
| 29 | 09-21/22 | bwd | A0 r3（手工收口） | 26.50 | 207.47（bwd） | 29.4% | — | A0-T | g10/g13 k_dq BLOCK_Q 16→32→64；g11 直接存 bf16（gain 1.297） | 09-21 两次 wedge（约 14:12、16:01:54），09-22 AC 后补测；本轮 beat 704.69 偏低 |
| 30 | 09-22 | bwd | A0 r4 | 16.61 | 330.98（bwd） | 46.3% | — | A0-T | g09 删掉 Q/dO 的 staging，直接写 LDS；g16 LDS padding（行距 256→272 B），消除 64 路 bank 冲突（gain 1.512） | |
| 31 | 09-22 | bwd | A0 r5 | 16.39 | 335.42（bwd） | 46.8% | — | A0-T | g12 删掉 k_dq 的 K staging；g17 改用 flyc.compile() 发射（gain 1.154） | gain 主要来自 fast（三形状算术平均），prod 实际约 +1.3%（act.yaml vs_champion 1.0132；335.42 对同会话 incumbent 331.07） |
| 32 | 09-22 | bwd | A0 r6 | 15.52 | 354.29（bwd） | 49.5% | — | A0-T | g19/g20 把循环拆成 mask-free 段和 masked 段；g03 交换 grid 轴（gain 1.049） | |
| 33 | 09-22 | bwd | A0 r7 | 14.37 | 382.73（bwd） | 53.4% | — | A0-T | g21：k_dkdv 跨迭代预取 Q/dO（gain 1.093；r19 删掉它会 −17.27%） | |
| 34 | 09-22 | bwd | A0 r8（手工收口） | 12.80 | 429.41（bwd） | 60.9% | — | A0-T | g23/g24 删掉单 wave 的 fx.barrier()，连带去掉保守的 s_wait_loadcnt_dscnt 0x0（gain 1.030） | 14:26 wedge 后 AC。同一份代码 AC 前后 382.73→404.06（+5.6%），绝对值不跨这条边界比较 |
| 35 | 09-22/23 | bwd | A0 r9（手工收口） | 11.43 | 480.96（bwd） | 66.8% | — | A0-T | g25 合并 carried-state；g26 k_dq 预取（k_dq 5.17→3.99 ms）（gain 1.119） | 16:01 在 d_final wedge。断电截断了 255 个 git 对象，由此定下“每轮结束都 push”的规则 |
| 36 | 09-23 | bwd | A0 r10–r11，均未接受 | 12.77 / 13.08 | 430.61 / 420.23（bwd） | 59.8% / 58.4% | — | A0-T | 指令级调度（g27/g28/g32）；s_set_vgpr_msb 的 bank 税（g36/g37）（gain 0.959 / 0.929） | r11 09:21 wedge（B 类） |
| 37 | 09-23 | bwd | A0 r12 | 11.02 | 499.15（bwd） | 69.9% | — | A0-T | g40 k_dq longest-first 派发（prod +4.91%）；g39 LDS 段分离（null）（gain 1.091） | max_rounds 12 用完，提到 40。当天查明 hipBLASLt 的真因：HIPBLASLT_TENSILE_LIBPATH 从没设过，sitecustomize 又把 PREFER_HIPBLASLT 钉成 0（_env.py 的 setdefault 空转了 12 轮）；改为显式赋值并指向宿主库（fp32 参考此前已改用 refcache）[0924__flydsl/REFCACHE-PREMISE-GONE.md] |
| 38 | 09-23 | bwd | A0 r13–r14 | 11.05 / 11.02 | 497.39 / 499.14（bwd） | 70.0% / 69.5% | — | A0-T | g42 k_dkdv 在 q 轴做确定性 split-K（fp32 partial，按固定顺序归约）；g44 nsp 规则；g45 k_redsp 归约 kernel（gain 1.370 / 1.111）。只提升 fast/proxy，prod 走 nsp=1，没动 | r14 首次尝试时 13:35 wedge：五个变体之一越界读，触发 fault storm（IH ring overflow）后 MES 不可恢复（C 类），这次尝试没有留下读数，AC 后重跑；之后 dmesg_restrict 改为 0 [0923__flydsl/hint.md h22] |
| 39 | 09-24 | bwd | A0 r15（被误晋升后回滚） | 11.57 | 475.14（bwd） | 66.5% | — | A0-T | g47 sched_group_barrier：prod −3.1%（vs_champion 0.9688），被 fast +6% 抵消 | 三形状算术平均 gain 1.0035 加 min_gain 0，让它被晋升为冠军；框架另因 FastError（g46/g47 未写进 facts/dead_ends）把本轮记为 failed，但晋升已经发生。用 20baa45e 手工回滚到 r14（best_round 和三个形状的 champions 都改回 14）。回滚后的 state.yaml 记 accepted:false / outcome failed，只看账本已看不出它曾被晋升 |
| 40 | 09-24 | bwd | A0 r16–r17（r17 是第一个 deep 轮） | 10.95 / 10.83 | 502.03 / 507.79（bwd） | 69.8% / 70.6% | — | A0-T | g51 把 LSE/delta 载入并入 g21 预取，加 sched_barrier(0)；g52 k_dq nsp_q split-K，只在 grid 欠填时启用（gain 1.005 / 1.159） | r17 的 PC sampling 在 prod 形状 fault（卡存活），此后永久禁用 |
| 41 | 09-24 | bwd | A0 r18–r19，均未接受 | 11.71 / 10.88 | 469.56 / 505.29（bwd） | 65.3% / 70.2% | — | A0-T | g56 改用 async LDS 载入 −6.84%；g59 WMMA reuse null；g60 删掉 g21 预取 −17.27% | r18 08:58 在 validation 单进程跑多形状时 wedge，之后改为一个形状一个进程 |
| 42 | 09-24 | bwd | **A0 r20：限频期 bwd 冠军** | 10.75 | 511.42（bwd；r22 同会话复测 516.87） | 71.4%；对 09-24 bar 为 0.71–0.72× | — | A0-T | g62 k_dkdv 的 Q/dO 预取深度 1→2（+1.65%）；g61 拆分累加链（null）；VGPR 904（gain 1.0016） | 靠 min_gain 0 才过；shape key 改名导致 champions 分叉（h35） |
| 43 | 09-24 | bwd | ASM bar 普查 | ASM 7.6766（n=224 raw；7.6769 n=146） | ≈716（bwd） | 100% | — | A0-T | 撤回 10.160 ms / 541 TF/s。剩余差距 = 7 vs 5 GEMM 的结构因子 1.386× × 调度因子 1.00–1.01× | 之前说的“0.92×”作废，实际是 0.70–0.72× [0924__flydsl/DAY-SUMMARY.md] |
| 44 | 09-24/25 | bwd | A0 r21–r23，均未接受 | 13.34 / 15.20 / 10.94 | 412.10 / 361.73 / 502.44（bwd） | 57.8% / 50.3% / 70.1% | — | A0-T | k_dq 预取 1→2：−19.33%；打散载入：−21.93% / −30.0%；g71 删回 g62 的第二级预取 −1.74%，反证 g62 是真实机制 | r23 于 09-30 在 A0 手工收口，判为 REJECTED |
| 45 | 09-25 | bwd | 手工 4-wave 探索（job 外） | G2 4-wave 16.31 vs 冠军 10.84 | 337.02 vs 507.34（bwd） | 47.2% | — | A0-T | G0 原子 scope；G1b 4-wave BLOCK_KV=128 骨架（UT 15/15）；S1/S2 均 null；barrier-free 探针 1.449×（就算 barrier 免费，上限也只到冠军的 0.950×） | 当日 dq 合同放宽为 run-to-run ≥70 dB，dk/dv 仍要 200 次逐位 [bwd-history.md §5,§6] |
| **阶段 5 · fwd op-evolve（A0-T，09-23..09-27；fwd FLOP 2.199292e12）** | | | | | | | | | | |
| 46 | 09-23 | fwd | Stage2 扫参（aiter FlyDSL fwd） | 出货点 2.356–2.400 | 930–948（fwd，旧口径 ≈2.233e12） | 65.7–66.1% | — | A0-T | n_block 64→128/256 分别慢 3.2×/9.9×（VGPR 溢出）；(BLOCK_M, n_block) 联扫证明出货点是局部最优；4-wave 在 aiter 代码里走不通；O_VARIANT v1 +1.90%（09-25 没复现） | [0923__flydsl/STAGE2-FWD-SWEEP.md] |
| 47 | 09-24 | fwd | fwd bar 锚定 | FlyDSL 2.4005 / ASM 1.5724 | 916.16 / 1398.67（fwd） | 65.5% | — | A0-T（sclk 1011→989） | 1.5266× 的差距全部是效率问题 | [0924__flydsl/bar-census/fwd_anchor.json] |
| 48 | 09-25 | fwd | A0 fwd job r0（20260925-114644） | 2.352 | 935.04（fwd） | 66.9% | — | A0-T | 基线是 vendored 的 aiter FlyDSL fwd（op0341）。flydsl 0.3.2→0.3.4.1 的 A/B 结果中性（fly/asm 0.664–0.666 vs 0.657–0.664），ISA 逐字节一致 | fwd 改用 flydsl 0.3.4.1（bwd 仍用 0.3.2）；r0 的数取自 r1 同会话的 incumbent [0925__flydsl/fwd341/ab_prod.log；OE: fwd state.yaml] |
| 49 | 09-25 | fwd | A0 r1（被否）→ r2（接受） | 2.166→2.160 | 1015.37→1017.97（fwd） | 72.7%→73.0% | — | A0-T | g01：因果 q-tile 按 longest-first 调度（prod +8.6–8.9%） | r1 被否只因两个框架 bug：50 dB 精度门（基线自己才 49.82–49.99）和 fast band 0.993。h20 重新提交后接受 [d64933d4] |
| 50 | 09-25/27 | fwd | A0 r3（被否）→ r4（接受） | 2.170→2.032 | 1013.60→1082.09（fwd） | 72.5%→77.4% | — | A0-T | r3：s_prefetch_inst / s_setprio 为 null，但探针证明 softmax 的 exp 占 prod 13.4%。r4：exp 参数用 v_pk_fma_f32，row-sum 用 v_pk_add_f32（两者叠加超过单独之和，+6.3%） | r4 在 09-25 交机时中断，09-27 重跑 [53ca89c9] |
| 51 | 09-27 | fwd | A0 r5–r10，均未接受 | 2.409 / 2.064 / 2.104 / 2.012 / 2.072 / 2.001 | 912.99 / 1065.79 / 1045.27 / 1093.10 / 1061.58 / 1099.24（fwd） | 65.4–77.8%（逐轮 65.4 / 76.1 / 74.7 / 77.8 / 75.2 / 77.6，各用本轮同会话 beat） | — | A0-T | r5（deep）QK/softmax 软件流水 −15.2%；r6 per-lane row-sum −1.9%，发现 post-ASM 的 I$ 惩罚；r7 barrier 减半原型 −3.5%（删 barrier 的上限是 +4.5–5.2%）；r8 nodelay +0.6%，被 post-beat 伪影否掉；r9 合并提交，输；r10 nodelay 单独 +0.83%，被三形状平均稀释，又被硬编码的 GATE_DB=50 挡住 | 尺子变了：r9 起不与 beat 同进程（h28）；r10 被否后加 gain_weights，r11 起生效 [a8fa47bf]；r9 之后的 fast/proxy 绝对值不能与 r1–r8 比 |
| 52 | 09-27 | fwd | **A0 r11：限频期 fwd 冠军** | 1.979 | 1111.15（fwd） | 78.9%（对同会话 beat 1407.8；对 09-24 bar 1398.67 为 79.4%） | — | A0-T | nodelay（amdgpu-enable-delay-alu=False），加 g30 按形状切到 R=1（BLOCK_M=128），fast 55.2→65.4，超过 ASM 的 64.5（gain 1.0074） | 相对 r0 +18.8% [e5d57d49] |
| 53 | 09-27 | fwd | A0 r12（被否）、r13（中断），以及 CPU 原型 | 1.979 | 1111.36（fwd） | 79.0% | — | A0-T | r12：R=1 路径做 split-KV（fast 0.987），关闭。另有 h21–h25 的 compile-only 原型，以及 ASM 结构 P2 设计（800 VGPR、0 spill，预测 +8–20%，没上卡） | 15:21 交机，fwd 迁到 B0 [0927__flydsl/asm-structure/DESIGN.md]。r12 被否，但 state.yaml 的 champions 记 prod=12（1111.36，与 r11 持平），best_round=11 [OE: fwd state.yaml] |
| **阶段 6 · B0 四卡攻关（B0-28，09-27..09-28）** | | | | | | | | | | |
| 54 | 09-27 | fwd | B0 fwd r5（失败）→ r6（接受）（job fwd-b0-20260927，克隆自 A0 r4） | 1.443 | 1524.1（fwd） | 84.8%（交错，偏低） | — | B0-28 GPU0 | 投机 softmax（SPEC_STALE_MAX）：跳过 row-max 和 rescale，用 ballot 回退到精确 softmax；+5.7%（分块复核为 +4.6%） | 四卡隔离成 fa-g0..g3；r5 的 reviewer 报 401（B0 上没有 codex 账号）；r4 在 B0 上为 1441.7 / 80.2%。这个投机后来在真实数据上是负收益，r16 关掉了 [0927__b0/fwd/progress.md] |
| 55 | 09-27/28 | fwd | B0 fwd r7–r9（被否）→ r10 refactor h39 | 1.453 / 1.457 / 1.466 | 1513.7 / 1509.8 / 1500.6（fwd） | 84.3–85.7%（交错） | — | B0-28 GPU0→GPU2 | 把 r9 的 BG 提为冠军：m16x8 小 grid gate（fast +27.5%）加 g18（masked tile 不做投机） | r9 只是被未发布 arm 的 best-ever 记录挡住。09-27 15:43 容器被他人删除，停机；09-28 GPU1 wedge，fwd 移到 GPU2 |
| 56 | 09-28 | fwd+bwd | 尺子审计：交错改为分块（h40/h41/h66） | — | — | 交错让 FlyDSL/ASM 比偏低：fwd 约 25%，bwd 约 3.4% | — | B0-28 GPU0 | 撤回两个假赢：lab2-L12 +6.4%→−0.4%（分块下 r6/L12 0.9964）；lab3-L21 +6.3%→约 +0.3%（lab3 VERIFY；分块下 bnegg 对 r6 +0.6%，在噪声边缘）。fwd r6 的 +5.7% 下修为 +4.6%，仍是真实收益（OP-EVOLVE-SUGGESTIONS 把这次高估也算作第三个“假赢”） | A/A：fwd ±0.19%，bwd ±0.05%；新旧尺子的绝对 ms 不可比。审计报告称“r6/ASM 稳态 1.035（r6 更快）”，与 job 分块尺子的 96.5%、A0 09-30 的 r6/ASM 1.028 矛盾，未解决 [0927__b0/ruler/REPORT.md] |
| 57 | 09-28 | fwd | B0 fwd r10 deep（被否）→ r11、r13 接受 | 1.474 / 1.466 / 1.463 | 1491.8 / 1500.7 / 1502.8（fwd） | 95.8 / 96.4 / 96.5%（分块 beat ≈1557） | — | B0-28 GPU2 | r10：L12 occ2 4-wave −0.7%，判死；r11：欠填 grid 改用 2-wave m32x2（fast +8.4%）；r13：小 grid 改 4-wave 并切分 d（fast +5.6%，fast 达到 ASM 的 105.5%） | prod 基本没动，增益都在 fast。占 ASM 从 85% 到 96% 是尺子变化。r12/r14/r15 被否（host 路径 null；TDM multicast 使 proxy −5.4%；分 lane row-sum null） |
| 58 | 09-28 | fwd | **B0 fwd r16 refactor h44：r13ns，当前 fwd 冠军** | randn ≈1.51–1.52 | randn ≈1443–1455（fwd） | randn ≈93%；真实数据 + GEMM 突发下 r13ns/ASM 时间比 1.25–1.29 | e2e fwd 50–61 → 48–49 ms/步 | B0-28 | 关掉投机 softmax（SPEC_STALE_MAX=False）：randn 慢 4.8%（预期内），真实 dump 快 8–15%，真实数据 + GEMM 突发快 12–23% | 线索来自 profile 和 fwd-nospec lab。r16 起 randn 读数比 r13 低约 4%，前后不能直接比 [0927__b0/fwd-nospec/REPORT.md；df9aa3e8] |
| 59 | 09-28 | fwd | B0 fwd r16opt–r19 均未接受；r20 中断 | 1.518 / 1.510 / 1.498 / 1.500 | 1448.9 / 1456.9 / 1467.8 / 1466.6（fwd） | 93.1–94.8%（randn） | — | B0-28 GPU2 | 逐行独立的 deferred rescale；用 ones-WMMA 算 row-sum；row-sum 重构（只在 randn 上有收益）；r19 = r18 的 g64 加 RESCALE_THRESHOLD 8→24（对冠军：h47 尺子即真实 dump 加每次计时前 GEMM 突发，+1.5%，18/18 读数高于 A/A；randn +1.7%），只被 r18 的 best-ever 记录挡住，refactor h48 没执行 [0927__b0/fwd/rounds/019/opt.md] | r18 起同时报真实 dump 尺子（h47）。r20 的 act 在卡上算 fp32 hipBLASLt 参考 GEMM，导致 GPU2 page fault，硬停，补 h50 |
| 60 | 09-27/28 | bwd | B0 r24–r27 均未接受（B0 编号） | 8.607 / 8.931 / 8.591 / 8.573 | 638.8 / 615.6 / 640.0 / 641.3（bwd） | 75.0 / 74.4 / 76.5 / 78.4% | — | B0-28 GPU1→GPU3 | unroll −4.5%；4-wave 的 X1/X2 判死；反向扫描做 L2 对齐 −1.9%；h33 clamp null。r27 的 geomean 被 fast 的 1.66× 拉高，误判 target_met，job 自行结束 | 起点是 A0 r20，在 B0 上约为 ASM 的 0.74–0.76×。r24–r26 用交错尺子。h69：validation 改为 proxy 和 prod 各自 ≥ beat |
| 61 | 09-28 | bwd | B0 r28 refactor h68：r19h | 8.583 | 640.6（bwd） | ≈78% | — | B0-28 GPU3 | lab-bwd-r19：r19 加 h33 的三处 clamp（越界预取 1,580→0，VGPR 904→724）；r19 在 prod 上比 r20 快 1.7% | r28 opt 为 null（0.9967）；min_gain 0→0.007 |
| 62 | 09-28 | bwd | **B0 r29：u2n，B0 bwd 冠军** | 8.375 | 656.5（bwd） | 80.4%（分块 beat 817.0） | 见 #66 | B0-28 GPU3 | lab-kdq：k_dq 的 full kv 循环展开 ×2（u2n），去掉回边上 65 条 v_mov_b64 的寄存器轮转（k_dq 3.185→2.990 ms）；加 g86，dK/dV 的 epilogue 经 LDS ring | prod +2.7%，proxy +2.8%；训练工作点下 r29/r19h −3.2% [0927__b0/bwd/progress.md；9aa21d96] |
| 63 | 09-28 | bwd | B0 r30–r32 均未接受；FUSED5 判死 | 8.377 / 8.419 / 8.641 | 656.4 / 653.1 / 636.3（bwd） | 80.0 / 79.8 / 77.8% | — | B0-28 GPU3 | dQ epilogue 经 LDS、常数外提、寄存器承载 dV/dK，全是 null 或变慢；lab3 FUSED5（把 dQ 的 fp32 原子累加融进 k_dkdv）21.9 ms，只有冠军的 0.39× | 12:45 UTC 两个 loop 停止，冠军快照 5188c9da 交接给 A0 |
| 64 | 09-28 | e2e | B0 e2e：修复前 → nkfix_b0 | — | — | — | 修复前：ASM 17,049 ms，低档中位数 1,922 tps（高档 2,016–2,022），fly(r6+r20) 1,914（1.0045）。nkfix 后：ASM 1,582/1,576 ms，20,718/20,792 tps；fly 19,762/19,809（1.0475/1.0500）；turbo Triton 2,250 ms / 14,563 | B0-28 GPU0 | 移植并重写 nkfix，加 transpose_triton：997 个 GEMM 全走 MT256x256x128。GEMM 每步 15.6–16.5 s（占 96%）→0.83–0.86 s（约 53%），约 18–20×。ASM 臂 20,718 tps 是修复前高档 2,016–2,022 的 10.3×（对低档中位数 1,922 为 10.8×） | 8 次运行 0 NaN [0927__b0/gemm/REPORT.md §0；0927__b0/e2e/RESULT.md §0]。nkfix 后训练 sclk 均值约 1,820→1,390 MHz（整板约 2.13 kW 功耗墙），attention kernel 慢 10–16%（k_dkdv +18%，ASM bwd 主 kernel +17%，k_dq +2%）[0927__b0/profile/REPORT.md:11,21,§3] |
| 65 | 09-28 | fwd+bwd | B0 训练中 attention 剖析（e0e42ad7） | 训练中每层（nkfix 后）：fly fwd 1.57–1.84 / ASM fwd 1.21–1.23；k_dkdv 5.95–5.99 + k_dq 3.30–3.35 / ASM bwd 主 kernel 7.00–7.09 | — | — | fly（fwd r6 + bwd r20）每步比 ASM 慢 87 ms：fwd +19.9、bwd +71.0、ASM 独有的 GQA 求和 −3.9 ms | B0-28 GPU0（训练中 attention 跑在 1250–1690 MHz，整板约 2.13 kW） | 解释 op 级结论为何在训练里反转，两个因子相乘。①时钟：randn 下 FlyDSL fwd 在 2350 MHz 为 ASM 的 0.84×，GEMM 刚跑完（约 1300 MHz）时为 1.20×，ASM fwd 基本不随时钟变。②数据：step 43 导出的真实 q/k/v，score 标准差 21–53，r6 的投机 softmax 在 13–25% 的 tile 步回退重算（randn 上为 0%）。真实输入 + GEMM 后时钟下，op 级 fly/asm 为 1.45–1.67。bwd 差距主要在 k_dq（3.33 ms/层，约 107 ms/步，ASM 融合 kernel 没有这份工作） | 由此出了 fwd 的 nospec lab → h44（r16 = r13ns）；bwd 随后的 lab-kdq 以 k_dq 为对象 → u2n（r29）。profile 对 k_dq 的定性（延迟/访存 bound、对时钟不敏感）后被 lab-kdq 修正：第一轮判为发射 bound，第二轮在 1272–1352 MHz 下测得 k_dq 约 1.36× 时钟敏感；这与 profile 里训练中 k_dq 只慢 2% 的读数仍未对上 [0927__b0/profile/REPORT.md §0,§3；0927__b0/lab-kdq/REPORT.md] |
| 66 | 09-28 | e2e | **B0 e2e 终版：fwd r16 + bwd r29** | — | — | — | fly 1,631/1,633 ms、20,091/20,070 tps，对 ASM 1,579/1,583 ms、20,747/20,697 tps，为 1.0323/1.0328。各版同进程 fly/asm：r6+r20 1.0475/1.0500 → r13（投机）+r20 1.0454/1.0447 → r13ns+r20 1.0403/1.0408 → 本版 [0927__b0/fwd-nospec/REPORT.md §3] | B0-28 GPU0 | 这一步的收益全部来自 bwd（每步 −15 ms）；剩余差距 bwd 约 52 ms、fwd 约 10 ms/步 | NKFIX_CHECK=1；sclk 1342/1371 MHz。PR 正文引用的 e2e 1.032 就是这个数 [0927__b0/e2e/RESULT-final.md；64d59269] |
| **阶段 7 · A0 换固件前后复测（09-28..09-30）** | | | | | | | | | | |
| 67 | 09-28 | fwd+bwd | A0 复现 B0 冠军（作废） | fwd r13ns 2.023 / ASM 1.564；bwd r29 11.010 / ASM 7.654 | 1087（fwd）/ 499（bwd） | fwd 77.3%；bwd 69.5% | 32L asm/old/new 1947 / 2070 / 2071 ms；24L（第二次开机）ASM 1483.4 / r16+r20 1569.9 ms（均作废） | A0-T（最后一次） | 同样的分块尺子；B0 上的 bwd 提升在 A0 上方向相反（r19h −2.2%，r29 −3.9%） | 两次开机共 5 个训练进程：开机 1 的 p3a 正常、p3b 从 step 4 起 NaN、p4a-32L 11:55 挂卡；AC 后开机 2 的 p4a-24L 正常，p4b-24L 在 12:32:50 第 1 步挂卡。两次挂卡各要一次 AC。最可疑的原因（未证实）：FlyDSL fwd 树的 _env.py 在 import 时把 HIPBLASLT_TENSILE_LIBPATH 改指宿主库 ~/.local/hipblaslt-gfx1250；第 1 步就用 FlyDSL 的两个进程一个 NaN、一个挂卡。10-02 的 h85“只用镜像 hipBLASLt”沿用了这条嫌疑。09-30 复测后整组作废 [0928__a0_repro/REPORT.md §1,§2,§5.3,§6.4；1002__oe/incident/WEDGE-1002.md:13] |
| 68 | 09-29 | — | A0 刷固件、换驱动（管理员操作） | — | — | — | — | A0 | — | VBIOS 630A→700E；SMU 125.7.1→125.12.0；amdgpu-dkms 7.1.1-2397345→7.1.0-2411946；sclk DPM 500/1100→500/2356/2400；fclk 1100→1250/1900。过程：09-28 19:39 UTC 另一用户卸掉 7.1.1-2397345，19:40 装上 7.1.0-2410994（21:40–21:42 又重装同版本）；重启后 PSP `ID_LOAD_TOC failed (0x11)`，probe 报 −62，卡无法初始化，09-29 06:40 AC 断电后依旧。09-29 16:09 换成 -2411826，21:28 换成 -2411946（21:41、22:13 又重装）；VBIOS 在 09-29 06:25 之后刷成 700E；19:08–22:19 主机重启约 10 次 [0928__a0_repro/REPORT.md §6.3；session 2dafe0d2] |
| 69 | 09-30 | fwd+bwd | **A0-R 同尺子复测（新基线）** | fwd r13ns 1.353 / ASM 1.257；bwd r29 6.583 / ASM 5.503 | 1626 / 1750（fwd）；835 / 999（bwd） | fwd 92.9%（反序 o2 92.4%）；bwd 83.6%（o2 83.3%，5.494/6.596） | — | A0-R（1.74–2.03 GHz） | 代码没变：FlyDSL fwd 快 1.47–1.50×（两种顺序，均值约 1.48×），ASM fwd 1.23–1.24×，FlyDSL bwd 1.67×，ASM bwd 1.39×；r20、r19h、r29 三者相差 <0.3% | 9b8f7397（提交标题写 bwd 83.5%，即两种顺序的均值）。同日跑了 roofline 微基准（W=8 cyc/WMMA；LDS 256 B/clk/段；WMMA↔DS 切换约 29 cyc；满载约 1.7 GHz）和 M8 fwd 消融（骨架开销 28%，softmax 19%）；新固件下 ATT 可用 [0930__roofline/REPORT.md；0930__bwd/probe/P1-RESULTS.md] |
| **阶段 8 · bwd 手工攻关（A0-R，09-30；同进程跑 r29 + r29_aa + ASM，分块尺子，randn，判定阈值 0.5%）** | | | | | | | | | | |
| 70 | 09-30 | bwd | base r29 → a01 c1 | 6.579→6.504 | 835.7→845（bwd） | 83.6%→84.2% | — | A0-R | c1：DQ_U2=False（新固件上 k_dq 展开 ×2 反而是负收益，关掉后 −0.85%） | A/A 0.01% [0930__bwd/PROGRESS.md（下同）] |
| 71 | 09-30 | bwd | a02–a08（被否 / 仅信息 / 待叠加） | ku2 7.099；tdm 6.628；tdm2 6.503 | — | 77.3–86.2% | — | A0-R | 消融 A2（去掉 Q/dO 全局加载，上限 −18.5%）指出主杠杆；trorder −0.89%、tailpf −3.33%、streams −0.89%、soffset −0.65% 留待叠加；只上 TDM ring、不把操作数读回寄存器反而 +1.7%；abl_l2 证明 L2 带宽不是瓶颈 | 每一步都由 ATT 指导 |
| 72 | 09-30 | bwd | a09 s1 → a10 s2 | 6.329→6.304 | 869→872（bwd） | 86.9%→87.4% | — | A0-R | s1 = tailpf + trorder + streams（k_dqg 放到 side stream，与 k_dkdv 并发）；s2 = s1 + soffset | 结果逐位等价 [7f42a5da] |
| 73 | 09-30 | bwd | a11 dkdv_tdm3 → a12 s3 | 5.856→5.769 | 939→953（bwd） | 94.0%→95.4% | — | A0-R | k_dkdv 用 TDM 3 级 LDS ring 装 Q/dO，并把下一轮的 B 操作数提前读回寄存器（每轮 WMMA↔DS 切换 7→2）；s3 = tdm3 + dqg_ts + side stream | 逐位 [12bbd6c9；4080e84a] |
| 74 | 09-30 | bwd | a14 s3_trim → a16 s4 | 5.508→5.532（复测 5.494） | 998→994（bwd） | 99.8%→99.7% | — | A0-R | 精简 VALU、v_nop 和地址计算（DS 段地址 VALU 20→2，v_nop 23→9）；计数器去掉除法（热循环 SALU 105→60） | [552dfb4d] |
| 75 | 09-30 | bwd | a13/a15/a17/a18/a20：w4f 4-wave 融合 dQ（均未接受） | 15.857→9.529→6.630→6.509（r4b）；relax_noatom 4.241（noatom 4.341；两者都不写 dQ 原子，输出错误，只作上限） | 347→845（bwd） | 34.6%→84.5% | — | A0-R | 5 个 GEMM，split barrier，dQ 用 fp32 SCOPE_DEV 原子累加；结果正确。dQ 原子写吃掉约 5 ms；按 ASM 的 fragment 顺序写满 128 B line 后，9.27→6.63 ms | 受整芯片原子吞吐限制，暂停 |
| 76 | 09-30 | bwd | a19 s5（dqg_tdm） | 5.338 | 1030（bwd） | 103.1%（proxy 112.8%） | — | A0-R | k_dqg 也改走 TDM 3 级 ring 并在寄存器里预取（VGPR 1020→878）；1-wave 路径第一次快过 ASM | 200 次逐位一致 [a94ccebe] |
| 77 | 09-30 | bwd | **a21 s6：当前 bwd 冠军** | 5.300（反序复核 5.295） | 1037–1038（bwd） | 103.8% | 见 #81 | A0-R | s5 再精简 k_dqg 的 VALU（热循环 VALU 290→194） | 相对 r29 −19.5%。dq 与 s5 差 81 dB（packed 运算顺序不同），SQNR 仍为 52.5 dB；只测过 randn [7cd15aef] |
| **阶段 9 · A0 e2e 与产品移植（09-30..10-02）** | | | | | | | | | | |
| 78 | 09-30 | fwd+bwd | 产品分支移植（dev/lhz/llama31_attn_opt，基于 origin/main 8cda13c7） | fwd 1.292 / ASM 1.187；bwd 6.618 / ASM 5.360 | 1702（fwd）/ 831（bwd） | fwd 91.9%（1.088×）；bwd 81.0%（1.235×） | — | A0-R | 把 fwd r16 + bwd r29 移植到 main 固定的 flydsl 0.2.4：c36cc124（gfx950 gate 的 >= 改为 ==）、5658a443（kernel 包 + 兼容层）、e05b67fd（FLYDSL 后端路由）、5a8446b5（测试）；API 清理 1f74e662/58d5d798（同日）：11 个 kernel 在两个 flydsl 版本下指令流都不变，60 个 UT 通过，prod 复测 fwd 1.283 / bwd 6.620 ms（清理前 1.292 / 6.618；这两个数只见于会话记录，没有对应的 ASM 读数）[session fdc2534d]。对 Triton 的 36 个形状几何平均：causal fwd 4.13×，bwd 3.37× | 0.2.4 把 async LDS→global 的 O store 地址编码错了，一上卡就 fault，改用 buffer_store，代价 fwd 约 1.3%、bwd 约 1.8%。已 push，PR 未开 [0930__port/PR_BODY.md] |
| 79 | 10-02 | fwd+bwd | 新驱动下 randn 复测；s6 移植到 0.3.4.1 | bwd s6 5.295 / ASM 5.513；fwd r13ns 1.362 / ASM 1.262 | 1038（bwd）/ 1615（fwd） | bwd 104.1%；fwd 92.7% | — | A0-D | s6 在 0.3.2 和 0.3.4.1 下的 ISA、LLVM IR、gpu.binary 逐字节相同（fd6dce02） | amdgpu-dkms 7.1.0-2412954（10-01 换）；上一次开机在 10-01 20:41 以 CPU L3 取指致命错误（BERT）重启结束；当天 GPU RAS 在 07:58:50、08:09:46 各报一批 pcie_pl 可纠正错误（57+58 条）[1002__e2e/E2E-PLAN.md §1] |
| 80 | 10-02 | fwd+bwd | 真实数据 op A/B（realab） | blk：fwd r16 1.308 / ASM 1.209，bwd s6 5.324 / ASM 5.475；gb：fwd 1.535 / 1.137，bwd 5.743 / 5.530 | — | blk：fwd 92.5%，bwd 102.9%；gb：fwd 74.1%，bwd 96.3% | — | A0-D（gb ≈1.28 GHz） | 用 B0 step 43 导出的 6 层真实 q/k/v。在训练工作点上 ASM 几乎不变，FlyDSL 两个方向都变慢 | gb 下 s6/r29 为 0.774 [1002__e2e/RESULT-realab.md；9108fd7c] |
| 81 | 10-02 | e2e | **A0 e2e 三臂对比：fwd r16 + bwd s6，当前 e2e 最佳** | — | — | 每步 bwd 快约 4 ms（相邻配对 4.50 / 4.56，即正文的 4.5；两臂中位数相减 3.8–4.4，RESULT-e2e.md 表四舍五入后为 3.9–4.3），fwd 慢 9.2 ms | fly 1349.4/1349.9 ms，24,283/24,275 tps（ASM 的 0.9990/0.9985×）；ASM 1350.6 ms，24,262 tps；flyr29 1389.6/1388.5（1.027） | A0-D | s6 相对 r29 每步省下 attention bwd 约 53 ms，与 op 级预测一致 | 32L，nkfix，NKFIX_CHECK=1，2 个进程 × 92 步；训练 sclk 中位数约 1.50 GHz；0 次非有限值。attention 合计 fly 每步多约 4.3–5.2 ms（相邻配对 +4.89 / +4.27；两臂中位数相减 +4.6 / +5.2），单步却快 0.7–2.0 ms（相邻配对 1.4 / 2.0；两臂中位数 1.2 / 0.7）：两进程的 Δ单步−ΔFA 都是 −6.3 ms，超出 E2E-PLAN §4 判定 2 的 5 ms 门限（三对比较都超出；s6 对 r29 是 attention −53 ms、单步 −37.0 至 −39.1 ms）。这部分在 attention 以外或是噪声，RESULT-e2e.md 没有讨论，也没有拆解（A0 的 kineto trace 无效）[1002__e2e/RESULT-e2e.md；1002__e2e/e2e/runs/analysis.1002_095504.txt §2；5497446d] |
| **阶段 10 · A0 bwd job r24–r27（A0 编号）、gb 尺子设计与 10-02 挂卡** | | | | | | | | | | |
| 82 | 09-30 | bwd | A0 r24（deep；refactor h75 采用 s6） | 5.295 | 1038.34（bwd） | 104.0% | — | A0-R | 候选 g77 DQT_VT_KEEP（只改寄存器分配），在 prod/proxy 上 null；deep profiling 第一次跑通 ATT，判定瓶颈是功耗墙（操作数全零时间 −34%） | 被接受的“1.4263×”是 fast 中位数噪声造成的假阳性（prod 0.9986），随后 fast 改按 min 计分，加 gain_weights 1/0.25/0（4d61867f）。best_round=24，即 s6 + DQT_VT_KEEP |
| 83 | 09-30 | bwd | A0 r25 | 5.292 | 1039.02（bwd） | 103.9% | — | A0-R | g75 DQT_DSADDR prod −0.19%；g73（dQ 用 packed bf16 原子）在 CPU 数值预筛中判死 | 加权 gain 1.0033 < 1.007，被否；14:34 停在 r26 开始前，交机 [070c8895]。r25 虽被否，state.yaml 的 champions 仍把 prod/proxy 记为 25（1039.02 / 1002.18），fast 记为 24，best_round=24；续跑时 r25 的读数就是 best-ever 门槛 |
| 84 | 10-02 | bwd | A0 r26 | 6.486（w4f_r4b） | 847.73（bwd） | 84.7% | — | A0-D | 改走融合 w4f：noatom 上限 4.23 ms，dQ 原子池占 2.24 ms（34.7%）；F=2 加宽在编译门判死（spill 616） | 被否，又因 FastError 记为 failed |
| 85 | 10-02 | fwd+bwd | gb 尺子设计；fwd job 从 B0 备份恢复（5696ae53，两者都没上线） | — | — | — | — | A0-D（只做 CPU 侧准备，没有用卡） | gb 尺子（训练工作点）：每次计时前跑 10 个 bf16 32768×4096×14336 GEMM，proxy/prod 按 gb 中位数计分，fast 仍用 blk。依据是同日 A0 的时间比（blk / gb / e2e）：bwd s6/r29 0.806 / 0.774 / 0.754，r29/ASM 1.206 / 1.341 / 1.290，fwd r16/ASM 1.081 / 1.349 / 1.263，这三项 gb 更接近训练；例外是 s6/ASM（0.972 / 1.038 / 0.974），gb 偏悲观约 6%。fwd job（B0 的 gfx1250-flydsl-attn-fwd-b0-20260927）从 0928 备份解出：best_round 16（r13ns），round 20 的 act 待在 A0 上重做，h48（采用 r19 的 op）在 round 21 开头执行 | 两者都只是备好，上线需用户批准。A0 上 fwd 的 yaml 没有 gain_weights，验收仍是三形状算术平均。同一时段 bwd job 被另一会话在 10:21:42 resume 跑 r26 [1002__oe/RULER.md §0,§1；1002__oe/FWDJOB.md §0] |
| 86 | 10-02 | bwd | A0 r27（未完成） | — | — | — | — | A0-D | opt agent 只为读 VGPR，就在 prod 形状直接跑了两个从未上过卡的变体（A_g74、B_g82），还用了宿主 hipBLASLt 库。WEDGE-1002.md 和 1ee0dd59 的提交标题都把它们记为 w4f 融合变体，是误记。核对 rounds/027/_scratch/arms 源码，两者都是 s6 原样拷贝后加 cluster multicast：A_g74 在 k_dkdv 里让同一 (bat, hkv) 的 4 个相邻 KV band 共享 Q/dO tile（DKDV_CLUSTER=4），B_g82 在 k_dqg 里让同一 GQA 组的 4 个 q head 共享 K/V tile（DQG_CLUSTER=4）。探针脚本是沿用 r26 w4f 的 build_probe.py | 11:40:59 UTC MES 不可恢复，挂卡，需 AC 断电；新增 h85：先 toy 后 prod，读 VGPR 一律 compile-only，只用镜像自带的 hipBLASLt [1002__oe/incident/WEDGE-1002.md；1ee0dd59] |
| **阶段 11 · 教程（10-05）** | | | | | | | | | | |
| 87 | 10-05 | e2e | 产品分支教程 e2e（docs/gfx1250_llama31_8b_e2e） | — | — | — | 1,361.9 ms/step，24,061 tps（按提交版本复跑 1,360 / 24,099）；漏掉 transpose_triton 时 1,689 / 19,406 | A0-D（sclk 最高 2.36 GHz） | 分支版 fwd r16 + bwd r29（flydsl 0.2.4），加 nkfix（NKFIX_CHECK=0）和镜像 hipBLASLt；附 Dockerfile、启动脚本、Primus hook patch | 20 步墙钟 53 s。这是 NKFIX_CHECK=0，不能直接对 #81 的 ASM 1,350.6 ms（NKFIX_CHECK=1）读成“只慢 0.8%”：按 CHECK=1 约 +33 ms/步（B0 实测，教程沿用）折算，ASM 在 CHECK=0 下约 1,318 ms，分支约慢 3%（推算，未实测）；bwd 版本（r29 对 s6）也不同。教程 §7 称固件更新前后同代码同方法，ASM attention + nkfix 的 e2e 1,947→1,351 ms（1.44×），这个倍数不纯是平台差异（见 c.1）[wt-llama31/docs/gfx1250_llama31_8b_e2e/README.md §3.2,§7；0927__b0/gemm/REPORT.md:83] |

### c.3 当前最优版本

| 项 | 版本 | 算子 ms | TF/s | 对 ASM | e2e | 机器 · 尺子 | 代码位置 / commit | 日期 |
|---|---|---|---|---|---|---|---|---|
| fwd 冠军 | r16 = r13ns（B0 fwd job 在 round 16 由 refactor h44 晋升；即 B0 r13 关掉投机 softmax） | 1.353 / 1.356 | 1626 / 1622 | 92.9% / 92.4%（时间比 1.076 / 1.082）；10-02 真实数据 blk 1.081，训练工作点 gb 1.349 | 在 #81 中每步 fwd 比 ASM 慢 9.2 ms | A0-R 09-30，分块，randn，同进程 ASM 1.257 | `0927__b0/champions/fwd_r16_r13ns`（5188c9da，df9aa3e8）；产品分支 `primus_turbo/flydsl/attention/gfx1250/`（5658a443，flydsl 0.2.4 下 1.292 ms，1.088×） | 09-28 晋升，09-30 复测 |
| bwd 冠军 | s6 = s5 + k_dqg VALU 精简（手工攻关；A0 job r24 = s6 + DQT_VT_KEEP，1038.34 TF/s） | 5.295–5.300 | 1037–1038 | 103.8%（ASM 5.497–5.500）；10-02 randn 104.1%；真实 blk 0.972（≈102.9%），gb 1.038（≈96.3%） | 在 #81 中每步 bwd 比 ASM 快约 4 ms（相邻配对 4.50 / 4.56，即正文的 4.5；两臂中位数相减 3.8–4.4），比 r29 快约 53 ms | A0-R 09-30 / A0-D 10-02，分块 | `0930__bwd/armsrc/s6`（7cd15aef，flydsl 0.3.2）；0.3.4.1 版 `1002__e2e/arms_src/bwd_s6_0341`（fd6dce02）。**还没进产品分支**，分支里是 r29：6.618 ms，81.0% | 09-30 |
| e2e 最佳 | fwd r16 + bwd s6（flydsl 0.3.4.1）+ nkfix（NKFIX_CHECK=1） | — | — | 单步为 ASM 的 0.9990 / 0.9985× | 1349.4 / 1349.9 ms/step，24,283 / 24,275 tps（ASM 1350.6 ms，24,262 tps） | A0-D 10-02，32L，训练 sclk 约 1.50 GHz | `1002__e2e/RESULT-e2e.md`（5497446d） | 10-02 |
| 产品分支现状 | fwd r16 + bwd r29（flydsl 0.2.4）+ nkfix（NKFIX_CHECK=0） | fwd 1.292 / bwd 6.618 | 1702 / 831 | fwd 91.9%，bwd 81.0% | 1,361.9 ms/step，24,061 tps（10-05，NKFIX_CHECK=0）。不能直接对 ASM 的 1,350.6 ms（10-02，NKFIX_CHECK=1）读成“只慢 0.8%”：按 CHECK=1 约 +33 ms/步（B0 实测，教程沿用）折算，ASM 在 CHECK=0 下约 1,318 ms，分支约慢 3%（推算，未实测）。#81 的 flyr29 臂同为 fwd r16 + bwd r29，为 ASM 的 1.027×，但那是 flydsl 0.3.4.1 构建、NKFIX_CHECK=1，不是分支本身；分支在 0.2.4 上用 buffer_store O writer，相对 0.3.4.1 约慢 fwd 1.3%、bwd 1.8%，与约 3% 的推算大体相符 [1002__e2e/RESULT-e2e.md；0930__port/PR_BODY.md；wt-llama31 README §3.2,§7] | A0-D | `dev/lhz/llama31_attn_opt`（c36cc124..58d5d798；教程 7e88d891/28e09a01/b8c543a9） | 09-30 / 10-05 |

### c.4 进展曲线要点

- **e2e 的头号杠杆是 GEMM，不是 attention。**
  - hipBLASLt 路径修正让 A0 32L 从 244 到 2,027 tps（8.3×）。nkfix 又带来 A0 8L 6.16×；B0 32L 上 ASM 臂 20,718 tps，是修复前高档 2,016–2,022 tps 的 10.3×（对低档中位数 1,922 为 10.8×），其中 GEMM 每步 15.6–16.5 s→0.83–0.86 s，约 18–20×。
  - 修好之前，attention 只占单步不到 1% 到 3.3%：A0 09-13 32L 走 rocBLAS 时整条 attention 路径 35.7 ms × 32 = 1.14 s，约 0.85%（phase1/RESULTS.md:266）；A0 8L 走 hipBLASLt、无 nkfix 时 3.3%（BOTTLENECK-SHIFT）；B0 09-28 nkfix 前为 1.4–5.4%（0927__b0/e2e/RESULT.md:37）。ASM bwd 的 e2e A/B 因此被 e2e 噪声淹没（8L n=9，−0.28%，不显著）。修好之后 attention 占 15–22%（A0 8L 20.8%，B0 17–22%，A0-D 10-02 约 15%），op 级改进才在 e2e 里看得见。
- **Triton 阶段：A0 一天 2.44×，B0 一天 2.75×。**
  - A0-T 上 59.642→24.435 ms，靠两个 config 旋钮加 vendor aiter 融合 bwd。
  - B0-14 上先复现 Day-1 阶梯（28.121→12.994 ms），再靠 waves_per_eu、ASM fwd（fwd 1.82×）和 in-thread transpose 降到 10.235 ms，相对出厂 2.75×。
  - waves_per_eu 在限频机上测不出收益，满频下值 4.2%：机器状态本身决定哪些杠杆有效。
- **ASM fwd 是 fwd 方向单次最大的一跳**（A0 4.253→1.549 ms，2.74×），此后一直作为 bar。ASM bwd 的 bar 先被 10.160 ms 的 shim 伪数把耗时高估了 32%（等于低估了 ASM 的速度）；09-24 普查定为 7.6766 ms，FlyDSL 的“0.92×”随之更正为“0.70–0.72×”。
- **FlyDSL bwd 从零做到约 0.71×**（A0-T，20 轮，57.24→511.42 TF/s，8.9×；对 09-24 bar 716 TF/s 为 0.714，按 r22 同会话复测 516.87 为 0.72）。
  - 按 prod 冠军 TF/s 逐轮相比（各轮在自己会话里的读数），最大的四跳是 r1 1.62×（57.24→92.62，causal tile 跳过）、r4 1.60×（207.47→330.98，LDS padding 消除 64 路 bank 冲突）、r2 1.55×（地址钳位 + 32 深收缩 + BLOCK_KV 32）、r3 1.45×（k_dq BLOCK_Q 64）。
  - 框架的三形状 gain 排序不同：r2 1.921、r4 1.512、r13 1.370、r1 1.342、r3 1.297。r13 的 gain 全部来自 fast/proxy，prod 没动。
  - r12 之后进入效率期，上限被 7 vs 5 GEMM 的结构因子 1.386× 卡住，r13–r23 在 prod 上只多出约 2–3%。
- **fwd 的收益主要来自调度和 softmax 指令。**
  - A0-T 上 935→1111 TF/s（+18.8%），由 longest-first 调度、packed exp/row-sum、nodelay 三项组成。
  - B0 上投机 softmax 在 randn 上 +4.6%，但真实数据和训练时钟下反而更慢。r16 关掉投机（r13ns）后，真实数据快 8–15%，真实数据加训练时钟快 12–23%。同一个改动，在两把尺子上方向相反。
- **尺子和判据的修正，本身“改写”了一部分进度。**
  - B0 从交错改为分块后，fwd 占 ASM 从约 85% 变成约 96%。同时撤回了两个假赢（lab2-L12 +6.4%→−0.4%，lab3-L21 +6.3%→约 +0.3%），fwd r6 的 +5.7% 下修为 +4.6%（仍为真实收益）。
  - r15 的算术平均、B0 r27 的 geomean 误判 target_met、A0 r24 的 fast 中位数假阳性，三次框架判据错误都是手工纠正的。
- **A0 刷固件让同一份代码快了 1.24–1.67×（o1 顺序；反序 o2 下 ASM fwd 1.23×、FlyDSL fwd 1.47×），FlyDSL 受益更多**：bwd r29 从 ASM 的 69.5% 升到 83.6%，fwd r13ns/ASM 从 1.29 降到 1.08。09-29 之前 A0 的绝对数全部作废。e2e 没有干净的同口径倍数（1.44× 不纯，见 c.1）。
- **bwd 手工攻关一天从 83.6% 做到 103.8%**（09-30，r29 6.58→s6 5.295 ms，−19.5%）。
  - 核心杠杆是 TDM 3 级 LDS ring，加上把下一轮 B 操作数提前读回寄存器。两步都是整个 op 的变化：只换 k_dkdv 的 dkdv_tdm3 使整个 op 相对 r29 −11.17%（5.856 vs 6.592 ms）；k_dqg 也换成 TDM（s5）后，整个 op 相对 s4 再 −2.1%。消融 A2 先给出了上限，ATT（刷固件后才可用）再定位到具体指令。
  - s6 仍是 7-GEMM 结构，多做约 39% 矩阵运算，依然快过 ASM。
  - 5-GEMM 融合版 w4f 不做 dQ 原子累加时（relax_noatom）只要 4.241 ms（ASM 的 129.6%；输出错误，只作上限），但 dQ 原子写要吃掉约 5 ms，所以暂停。
- **e2e 从落后 4.8–5.0% 追到持平。**
  - B0 上 fly/asm 1.0475（r6+r20）→1.0454（换 fwd r13，仍带投机）→1.0403（换 fwd r13ns）→1.0323（换 bwd r29）。A0-D 再换成 s6 后为 0.999，每步 bwd 省 53 ms。B0 和 A0-D 的绝对 ms 不可比，这里只比同进程比值。
  - 剩余差距在 fwd：训练工作点下 r16/ASM 1.349，每步慢 9.2 ms。
  - 产品分支仍是 r29，s6 还没移植进去。同组合（fwd r16 + bwd r29）在 #81 中为 ASM 的 1.027×，但那是 flydsl 0.3.4.1 构建；分支本身（0.2.4）只测过教程 e2e 1,361.9 ms/步，那是 NKFIX_CHECK=0，按 +33 ms/步折算到同口径约比 ASM 慢 3%（推算，未实测）。下一步是 fwd，以及把计分尺子换成训练工作点的 gb 尺子（10-02 已设计，未安装）。

## (d) 过程中遇到的问题及解决方案

**阅读约定**

- 机器与时钟态（不同态的绝对数不能直接比较，凡并列处均另行说明）：
  - **A0-限频**：heliosr-1b114-c07-1（单卡），2026-09-29 刷固件之前，VR 限频，DPM 只有 500/1100 MHz 两档，空闲 1100 MHz，负载约 0.95–1.07 GHz。
  - **A0-新固件**：同一张卡，09-29 刷 VBIOS 630A→700E、SMU 125.7.1→125.12.0 之后。op 级 1.74–2.03 GHz，训练中约 1.5 GHz（2,500 W 功耗墙）。10-01 起驱动为 amdgpu-dkms 7.1.0-2412954。
  - **B0**：ctheliosp-1b112-a37-1，4×gfx1250，负载 2,133–2,244 MHz。09-28 nkfix 之后训练中约 1.25–1.69 GHz。
- 来源写法：
  - 路径省略前缀 `output/`。
  - `s:<会话id前8位> <时间Z>` 指会话记录。
  - `OE:` 指 op-evolve 仓库 `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`。
  - `skill:` 指 `.claude/skills/gfx1250-attn-campaign/references/*.md` 或 `~/.claude/skills/*`。
  - 8 位十六进制是 Primus-Turbo 的 commit。
  - `hNN` 指 hint 编号：A0 bwd 在 `0923__flydsl/hint.md`，A0 fwd 在 `0925__flydsl/fwd-job/hint.md`，B0 在 `0927__b0/{fwd,bwd}-hint.md`，A0 09-30 之后（h83/h85）在 bwd job 的 `job_context/hint.md`。
- 时间一律 UTC。bwd 轮次编号 r24–r27 有两套：B0 09-27/28 一套，A0 09-30/10-02 一套。本节都会标明机器。

---

### d.1 GPU wedge 与 AC power cycle

在这两台机器上，卡挂死（wedge）之后，软件手段都救不回来：驱动自己的 reset 完不成，容器里有不可杀的 D 状态进程，`/dev/kfd` 的引用计数也不会归零。唯一的恢复办法是用户本人到机器前给整机断电（AC power cycle）。因此挂卡首先是排程约束，其次才是技术问题。

#### d.1.1 事件表（跨来源去重）

类别图例：

- **A**：参考 GEMM（fp32 Tensile）地址越界写，即"内存越界类"。
- **B**：TLB/队列类。首条是 `INVALIDATE_TLBS` 超时，此前没有任何内存故障。
- **C**：候选 kernel 越界读引发故障风暴，伴随 `IH ring buffer overflow`。
- **S**：启动期 MES 故障族，未归因。
- **K**：已知操作诱因，包括在线调优、inductor、坏候选、profiler、新 kernel 直接上 prod。
- **R**：资源或进程问题，如 SIGBUS、KFD 残留、占卡。
- **F**：进程级故障，可恢复，不算挂卡。
- **D**：降级，不是挂死。
- **X**：非 kernel 原因，如驱动、固件、协作。

| 日期 UTC | 机器 | 当时在跑什么 | 触发/根因 | dmesg 特征 | 类别 | 恢复方式 | 代价 | 来源 |
|---|---|---|---|---|---|---|---|---|
| 09-04（活动前） | A0 | HipKittens bf16 GEMM 阶梯第 16/25 轮，没开 profiler | 未知，普通 GEMM 负载 | MES 无响应，reset 失败 | 平台 | 主机重启 | 1 次重启 | OE:`output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md` §2.1-A |
| 09-06 20:33→09-07 07:42（活动前） | A0 | 无人值守的 op-evolve grouped-GEMM job | 未知 | 11 h 内 `SMU: No response` 857 次 | 平台 | 主机重启 | 1 次重启 | 同上 §2.1-B |
| 09-11 ~08:31 | A0-限频 | E3 探针：在卡上对 FlexAttention `mask_mod` 求 8192×8192 外积 | 在卡上物化超大布尔外积 | `GPU Hang` | K | 运维重启 | 1 次重启；该分析改在 CPU 上用 numpy 做 | 同上 §2.2 |
| 09-11 10:32→10:48 | A0-限频 | E4 探针：rocprofv3 PC sampling（同系列 PMC/ATT 都正常） | PC sampling | 10:32 `GCVM_L2_PROTECTION_FAULT` PF 0x3 RW 0x0 TCP，涉及 3 个 XCD，进程死、卡活；10:48 重试后 `GPU Hang` → MES 无响应 → `wait for reset ack` | K（profiler） | ~11:10 运维重启 | 1 次重启；PC sampling 永久禁用 | 同上 §2.3 |
| 09-13 ~13:19（13:30 发现） | A0-限频 | `forever_queue.sh` 4 个 worker，加独占窗口探针 | 卡本身脆弱；探针半小时内多次 `pkill -9` 正在跑 kernel 的队列，这是可避免的诱因 | `MES(0,0) … INVALIDATE_TLBS` → `MES(7,0) REMOVE_QUEUE` → `wait for reset ack`；随后 `Mode2 reset failed`、`ASIC reset failed -62` | B | 主机重启（时间未记录）；用 `docker cp` 抢救出 1063 行账本 | 下午起无卡可用；09-14 转到 B0 | `0913__opt_plan__claude/phase2/INCIDENT-2026-09-13-wedge.md` |
| 09-14 | B0 GPU0 | 当天工作开始前已坏（uptime 14647 s 起） | 非我方触发 | 120 条 `MES failed to respond`，0 条 `wait for reset ack` | D | 从调度中摘除，改用 GPU1–3；不重启 | 0 次 AC；当天 4 次误判卡状态；一个孤儿循环给坏卡喂了 2 h 15 min | `0914__campaign/HANDOFF.md:274-360` |
| 09-15 04:47→06:07 | A0-限频 | 两臂 e2e 相隔 65 s 启动；随后执行 `docker kill` + `modprobe -r amdgpu` | 没等 KFD 清空；`timeout` 只杀了父进程，torchrun 孤儿仍占卡和 1234 端口；反复探测；在已经不能计算的卡上卸驱动 | 首臂第一次 attention fwd 100% 占用 24 min，dmesg 0 故障；`torch.cuda.init` 45.2 s；之后整机 SSH 失联 | R + 操作 | 操作者 AC（07:18 前） | ~1 h 卡时 + ~2 h 整机不可达；助手曾误报"驱动重载修好了"，已更正 | `0915__opt/INCIDENT-2026-09-15-machine-death.md`；457593d8 |
| 09-15 ~08:08 | A0-限频 | 32L e2e + ASM bwd scratch 常驻缓存 | 显存 88.3% 的配置上再加 1.38 GiB | step 3 SIGBUS；死进程泄漏 KFD 上下文和 411 GB 显存；dmesg 干净 | R | AC（~08:20，08:25 时 up 4 min） | 改用 8L 配置 | s:b596bddb 08:08–08:25Z；`0915__opt/E2E-AB.md` |
| 09-15 08:38 | A0-限频 | AC 后的第一个 8L run | 未定 | worker 首次 attention 后静默死亡；3 个陈旧 KFD 条目；`docker exec` 报 setns 失败；dmesg 0 | 未归因 | AC（09:00） | 当天剩余 GPU 工作暂停 | s:b596bddb 08:38–09:00Z |
| 09-15 ~10:47 | A0-限频 | noconv 对照（`converters: []`） | 退回 torchtitan Flex → 被强制走 inductor → `flex_decoding` autotune | `MES(6,0)/(7,0) REMOVE_QUEUE` 反复约 30 s → `Suspending ip block ih_v7_0` → `wait for reset ack` | K（inductor） | AC（11:16） | 结论：A0 上测不了"不用 turbo attention"的基线 | s:11052787 10:47–11:16Z；8ad25967 |
| 09-15 12:35–12:54 | A0-限频 | nkfix ON 第 4 次连跑（cooldown 10 s） | 未归因 | 38 条 `MES(0,0) ring buffer is full` → `MES(4,0) failed to respond … WAIT_REG_MEM` + `reg_write_reg_wait` 失败；进程 Zsl | S | AC（13:07） | cooldown 改 20 s 后连跑 6 次都干净（不构成安全结论） | `0915__opt/MES-WEDGE.md`；s:11052787 12:51Z |
| 09-16 01:19 | A0-限频 | 第 3 次连续 profile run（v2prof） | 未归因 | 故障计数 0→72；`REMOVE_QUEUE` 反复 + `wait for reset ack` + hung task；rocm-smi 全部 N/A | S | AC（02:26） | — | s:11052787 01:19–02:26Z |
| 09-16 03:16 | A0-限频 | 32L 基线启动 | 并行 workflow 中一个"只做 CPU"的 agent 自写了 GPU 探针 `warps16.py`；其中 num_warps=16 候选挂起，两个实例各占 385 MB，把卡占死 | `ring buffer is full` 持续；`gpu_health.sh` 仍报 HEALTHY；D 状态持有者，无 reset ack | R（编排） | AC（07:35） | "挂卡与 nkfix 无关"这条证据被污染，已撤回（4e03bd57） | `0915__opt/MES-WEDGE.md:11-29` |
| 09-16 07:38 | A0-限频 | 32L b=4 + nkfix（显存 88%） | 起初归因为 SIGBUS/显存，为此加了 `_headroom_ok`，后来证实它从未触发 | 首条 `INVALIDATE_TLBS`；`GPU reset begin` 之后没有 end；训练日志 SIGBUS | S | AC（07:56） | — | s:11052787 07:38–07:57Z |
| 09-16 08:01 | A0-限频 | 32r1（显存仅 6.93%）。07:59 刚有一次 32L+nkfix 跑完 20 步（11,608 tps，5.85×），08:00 起连跑三次做定性，第一次就挂 | 启动期族；把两份日志对齐到同一时钟后，MES 失败比 SIGBUS 早 27 s，说明 SIGBUS 只是症状 | MES 17 条 + SIGBUS | S | AC（08:20） | 推翻上一条归因；此后只用 8L。07:59 那次 run 在 13:13 查出从 step 3 起 NaN（20 步里 17 步），11,608/5.85× 已撤回（见 d.2.1） | s:11052787 07:59–08:20Z、13:13Z |
| 09-16 09:02 | A0-限频 | Triton fwd sweep 第 58 个候选 | 候选 `num_warps=8, waves_per_eu=0, num_stages=2, PRE_LOAD_V=0` | `MES(7,0) REMOVE_QUEUE` → `wait for reset ack`；内核线程卡在 `svm_migrate_to_ram` | K（坏候选） | AC（09:18） | 账本记为 `card-wedge` 终态，不再重试 | e4730158；`0915__opt/FWD-SWEEP.md:42-51` |
| 09-16 09:35 | A0-限频 | e2e 规则 3（FlyDSL wgrad GEMM）第一次真跑 | 在训练步内做 autotune，`except Exception: continue` 吞掉了 launch failure | step 2 `hipErrorLaunchFailure` → MES 对 INVALIDATE_TLBS/REMOVE_QUEUE/SUSPEND/RESET/RESUME 全部无响应 → `GPU reset begin` 未完成 | K（在线调优） | AC（11:17） | 规则 3 改为离线建表 | f08ccffe；`0915__opt/FLYDSL-WEDGE.md` |
| 09-16 11:23、12:28、13:12 | A0-限频 | fly3-fly、copy4（不经过 FlyDSL 的基线臂）、nanhunt | 启动期族，与规则 3 无关 | 三次都是 0 步；nanhunt 为启动期 SIGBUS | S | AC 11:57、12:44、~13:15–13:30（最后一次为推断：13:30 读到的 dmesg 已是新开机） | 当日 14 次启动有 3 次启动期失败（21%） | 6d0cc872；`0915__opt/MES-WEDGE.md:84-103`；s:11052787 13:30Z |
| 09-17 09:12 | A0-限频 | 给已知会越界写的 `dkdv_heads='kv'` 臂计时，当作"下界" | 实验设计错误 | `GCVM_L2_PROTECTION_FAULT` PF 0x5 RW 0x1 TCP → MES REMOVE_QUEUE/SUSPEND 无响应 → `Queues reset on process python3`；没有 GPU reset | F | 不需要 AC；KFD 为空，matmul 5.08 ms（故障前 5.09） | 损失 1 个进程 | `0917__flydsl/GQA-WORKAROUND-COST.md`；s:c4b79aa6 09:13Z |
| 09-21 ~14:12（14:35 报警） | A0-限频 | bwd job r3 opt | 闸门里的 eager fp32 参考走 Tensile GEMM（`Cijk_*`）越界 | 2 条不可恢复签名（reset ack / ring gfx timeout / GPU reset begin 一类） | A | AC（~14:55 开机） | 被打断的 opt 归档为 `1-opt.stale-20260921T141238` | s:a9a96fef 14:35–15:09Z；`0923__flydsl/wedge4/ANALYSIS.md:45-51` |
| 09-21 16:01:54 | A0-限频 | r3 验收 `gpu5.sh`（detach 运行） | 同属 r3 路径，事后按 A 类归档；`patrol.sh` 告警不锁存，`release_guard` 从未被拉起 | 会话判为不可恢复；4547915c 称这次 dmesg 没有任何 reset 签名，supervisor 停在 `gpu_ok()` 的 900 s 退避里 | A?（签名矛盾） | AC 09-22 00:49:22 | 整夜无卡；09-21..22 五轮墙钟里有 17.8 h（82%）耗在挂卡和等 AC | 617481d7；4547915c；s:a9a96fef 09-22T00:59Z |
| 09-22 14:26 | A0-限频 | bwd r8 armB（`bitwise.py` 在 prod 形状直接调 fp32 参考） | 首条是 INVALIDATE_TLBS，此前没有内存故障 | INVALIDATE_TLBS→REMOVE_QUEUE→SUSPEND→RESET→RESUME→`MES might be in unrecoverable state`→`GPU reset begin!. Source: 3`→MES(0..6) REMOVE_QUEUE 无界级联 | B | AC 14:40:38 | r8 与 incumbent 逐位相同，以此继承正确性，手工收口 | `0922__flydsl/ROUND3-CLOSEOUT.md` Round 8 |
| 09-22 16:02 | A0-限频 | bwd r9 `d_final` 验收 sweep；挂前约 2 min `bitwise.py` 在 prod 跑过约 2000 次 fp32 Tensile | 首行签名没保存（ring buffer 随断电丢失） | 会话读到 `unrecoverable` + `GPU reset begin!. Source: 3`；`ps` 诊断命令本身挂住 | 未定 | 节点交接，夜间 AC（~23:47 开机） | 断电截断 255 个 git object，丢失 70efc407；ROUND3-CLOSEOUT.md 被截在 16384 B | `0922__flydsl/HANDOVER.md`；s:a9a96fef 09-23T07:13Z |
| 09-23 09:21 | A0-限频 | bwd r11 `d_s4`（开机后 9.5 h 零异常） | 直接原因不明 | INVALIDATE_TLBS→`failed to suspend all gangs`→`Failed to detect hung queues`→unrecoverable→GPU reset begin→14 次 REMOVE_QUEUE；没有 sub-4GB 地址，没有 `Cijk_` | B | AC（09:42 前） | 新增早停信号（出现在不可恢复前约 5 s） | 98eaa09e；`0923__flydsl/wedge4/ANALYSIS.md` |
| 09-23 13:35 | A0-限频 | bwd r14 的 scratch 候选臂 | 候选 kernel 越界读到已映射但不可读的页 | `GC_UTCL2` 故障风暴：TCP、RW 0x0、PF 0x3、高地址、MORE_FAULTS 0x1、3 个 XCD、6 次 `IH ring buffer overflow`，1 s 后 MES 不可恢复 | C | AC（~14:00） | 当天 `dmesg_restrict=1`，全天的"零故障"检查都在读空输出；证据最后从 `/var/log/kern.log` 找回 | d2bb6a87；`0923__flydsl/hint.md` h22 |
| 09-24 ~05:24 | A0-限频 | bwd r17 deep 轮在 prod 形状跑 PC sampling | PC sampling | 131 行 page-fault burst；`GPU core dump skipped because PC Sampling active`；`Timeout while waiting for queue sync` | K（profiler），卡存活 | 不需要 AC | 离挂卡只差一步 | 82420bdf |
| 09-24 08:58:53 | A0-限频 | bwd r18 `validation.py` 在单进程里跑 fast,proxy,prod | 单进程跨 shape 运行。"一 shape 一进程"在 `facts.md:241` 早有记录，但没进 `measure()`；写 fault 本身仍未解释 | RW 0x1 PF 0x5 写 fault，31 s 后 MES 死（MES(6)→MES(7)），没有 IH overflow | A 型写 fault | AC（09:14:30 开机） | 改成一 shape 一进程后，6 次完整测量 0 故障 | bf58bec0；`0924__flydsl/DAY-SUMMARY.md:82-104` |
| 09-28（Day2 开局发现） | B0 GPU1 | 夜间没有我方任务 | 未知；uptime 69813 s 时有一个非我方 python3 在 libamdhip64 里 GPF | 持续刷 `MES(0,0) failed to respond` / `ring buffer full` | 未知 | 弃用 GPU1（fwd 移到 GPU2，bwd 移到 GPU3）；交还时注明需要 AC（同一 hive 的 4 卡会一起断电） | B0 一直少一张卡到收工 | `0927__b0/LAB-RULES.md`；`0927__b0/HANDOFF-A0.md` §9 |
| 09-28 ~11:55 | B0 GPU2 | fwd job r20 act 的 `adv_m32x8.py` | 在卡上跑 fp32 hipBLASLt 参考 GEMM | TCP PF 0x3（AID1.XCD2），page not present 0x280000，rc=134；无 MES hang | A（未升级为挂卡） | 硬停 fwd loop | r20 act 作废；h50 禁止在卡上算 fp32 参考 | d13160ba；`0927__b0/fwd-hint.md:862-870` |
| 09-28 11:55 | A0-限频 | e2e `a0_p4a` 32L（ASM 先跑），紧接在 p3b 的 86 步 NaN 运行之后 | 怀疑上一个进程已把卡弄坏，加上显存 89% | step 2 `hipErrorLaunchFailure` → MES unrecoverable → GPU reset begin | S? | AC（12:10 开机） | 改为 24L，每次开机只跑一个训练进程 | `0928__a0_repro/REPORT.md` §3 |
| 09-28 12:32 | A0-限频 | e2e `a0_p4b` 24L，FlyDSL 在 step 1 运行 | fwd 树的 `_env.py` 把 `HIPBLASLT_TENSILE_LIBPATH` 改到宿主库（与证据最吻合，但未在卡上证实） | INVALIDATE_TLBS → unrecoverable → GPU reset begin → `wait for reset ack`；KFD 残留 2 个 | 宿主库 | AC（13:01 开机） | 当天 5 次训练启动：2 次有效、1 次 NaN、2 次挂卡 | `0928__a0_repro/REPORT.md` §5.2–5.3 |
| 09-28 19:39 → 09-29 22:19 | A0 | 无（管理员 asierrag 更换驱动和固件包） | dkms 7.1.1-2397345 → 7.1.0-2410994；推断新固件的 TOC 与 VBIOS 630A 不匹配 | PSP `LOAD_TOC failed (0x11)`、`LOAD_IP_FW failed (0xFFFF0006)`、`SMU: No response`、`hw_init of IP block <smu> failed -62` | X | 06:35 用户做了 AC，无效；管理员刷 VBIOS 700E + dkms 7.1.0-2411946（主机约 10 次重启） | 约 1.5 天不可用；09-29 之前 A0 的绝对数全部作废 | s:2dafe0d2 06:25–06:50Z；`0928__a0_repro/REPORT.md` §6.3 |
| 10-02 11:40:59 | A0-新固件 | bwd job r27 opt agent 生成的 `build_cmd.sh` | 在 prod 形状直接调用两个从未上过卡的 s6 基底变体，目的只是读 `vgpr_count`（compile-only 就能拿到）：A_g74 让 k_dkdv 按 KV band 组成 (1,4,1) cluster、多播 Q/dO；B_g82 让 k_dqg 按 GQA 组组成 (4,1,1) cluster、多播 K/V。卡挂在第一个臂 A_g74 上，B_g82 没跑到。探针 `build_probe.py` 与 r26 读 w4f VGPR 的 `vgpr_probe.py` 逐字节相同（docstring 仍写 w4f）；1ee0dd59 提交信息、WEDGE-1002.md（以及 h85 正文、`gfx1250-card-safety` §1 #13）把这两个变体记成"w4f 融合变体（4-wave + split barrier + dQ 原子）"，是误记。没有 toy 先行、没有 `AMD_SERIALIZE_KERNEL`、没有锁/KFD 包装，LIBPATH 指向宿主库 | REMOVE_QUEUE→SUSPEND 失败→queue reset→RESET 失败→unrecoverable→`GPU reset begin!. Source: 3`→ADD_QUEUE 失败 | K（新 kernel 直接上 prod） | 只写 `.stop`，不 kill；AC（时间未记录） | r27 作废；新增 h85 | 1ee0dd59；`1002__oe/incident/WEDGE-1002.md`；OE bwd job `rounds/027/_scratch/arms/{A_g74,B_g82}/kernels.py`、`_scratch/build_run/out`、`1-opt/raw/build_probe.py`（= `rounds/026/1-opt/raw/vgpr_probe.py`） |
| 10-06（窗口外，其它项目） | A0-新固件 | `hipblaslt-bench --algo_method all`，bf16 TN 7168×8192×576 | 预编译 Tensile 解（约第 122/133 个）本身是坏的 | `unspecified launch failure` → MES REMOVE_QUEUE/SUSPEND/RESET/RESUME 无响应 → GPU reset begin，未完成 | 坏 Tensile 解 | AC | 仅作为 hipBLASLt 的风险参考 | skill:`gfx1250-card-safety` §1 #14 |

还有几次可存活的故障，它们不需要 AC，但都曾被误判：

- 09-17 r1 首次尝试时的越界读，PF 0x3 RW 0x0。
- 09-23 ~14:15 那批，以及 09-24 03:58、07:58、10:51 的 shape 切换处故障。

这些都只损失进程。来源：s:c4b79aa6 09-17；0e9ca37c；`0924__flydsl/DAY-SUMMARY.md:84-87`。

#### d.1.2 挂卡类别与认识的演进

1. **09-04..09-13（A0-限频）：先当成平台问题。**
   - 四种互不相关的负载都触发过同一种挂法：MES 无响应、reset 完不成。这四种负载是 HipKittens GEMM、grouped-GEMM、E3 mask 外积、E4 PC sampling。
   - 当时的结论：卡本身脆弱；PC sampling 必挂；驱动重载不可行，因为 D 状态进程不释放 `/dev/kfd`、amdgpu refcount 不归零。
   - 同时发现挂卡时 `rocm-smi`、`ps -eo …wchan`、`pgrep`、`docker exec` 都会自己挂住。
   - 处理：提交 HARDWARE-ISSUE / PLATFORM-ESCALATION，09-14 换到 B0。
2. **09-14..09-17：学会区分三种状态。**
   - "降级"与"挂死"不同。B0 GPU0 有 120 条 `MES failed to respond`，但仍能跑完 kernel。只有 `wait for reset ack`、`ring gfx timeout`、`GPU reset begin` 才是不可恢复。
   - "进程级故障"与"挂卡"不同。09-17 那次 `Queues reset on process` 不需要 AC。
   - e2e 期间启动期挂卡很多，09-16 当天 14 次启动有 3 次挂（21%），由此得出：**风险按启动次数计价，不按运行时长**。
   - 识别出的确定诱因：训练步内 autotune、`converters: []`→inductor、Triton `num_warps=16`、坏候选 #58、在 KFD 残留时启动新 run、`modprobe -r`、反复 SIGKILL 正在跑的 kernel、"只做 CPU"的 agent 私自上卡。
   - 另一个结论：SIGBUS 是症状，不是病因。
3. **09-21..09-24（op-evolve bwd job）：按 dmesg 首行分出 A/B/C 三类，判别式被逐个证伪。**
   - 先后提出过四个"挂卡前兆"判别式，全部被实测推翻：
     - h15：sub-4GB 截断地址 + PF 0x5/RW 0x1 + `copy_context_work_handler`。B 类挂卡完全不触发它。
     - h22：`IH ring buffer overflow`。09-24 那次开机出现 8 次 overflow，卡仍然活着；而 08:58 的挂卡反而没有 overflow。
     - MORE_FAULTS / 多 XCD。可存活的批次里 8/8 都有 MORE_FAULTS。
     - 故障到达率"6000×"。这个数字其实是驱动约 97 ms 的节流（7436cd29）。
   - 结论（3fd48059）：**不存在可用的挂卡预测指标**，只对挂卡本身告警。
   - 这期间真正定位到的触发器有两个：一是闸门在卡上跑的 fp32 Tensile 参考，用 refcache 解决；二是单进程跑多个 shape，约 44%/进程会出故障，改成一 shape 一进程后连续 6 次 0 故障。
   - 同期还发现 `dmesg_restrict=1` 使此前的"零故障"检查全部无效。
4. **09-28..10-06：宿主库与新 kernel 成为主要诱因，刷固件没有让卡变结实。**
   - 宿主 hipBLASLt 库 `~/.local/hipblaslt-gfx1250` 与多起事件相关：A0 09-28 一次 NaN、一次挂卡；B0 step 1 卡住；10-02 挂卡。
   - 在卡上跑 fp32 参考 GEMM 仍会出故障（B0 GPU2）。
   - 09-29 刷固件后时钟恢复，ATT 也能用了，但 10-02 一个新 kernel 直接上 prod 照样挂卡。10-06 `hipblaslt-bench` 枚举到坏的 Tensile 解也会挂。
   - 规则收敛为 h85：新 kernel 按 toy→prod 逐级放大；只读 ISA/VGPR 的探针一律 compile-only；只用镜像自带的 hipBLASLt 库。
   - 另一组数据：B0 09-27/28 共 ≥12 次训练启动，0 次挂卡。把各段合起来，启动期挂卡率约为 ≤5/31≈16%（`1002__e2e/E2E-PLAN.md:377`）。

#### d.1.3 有效的对策

| 措施 | 针对的失败 | 引入 | 证据/效果 | 来源 |
|---|---|---|---|---|
| 不做驱动级恢复：从恢复阶梯里删掉 `modprobe -r amdgpu`。挂卡时只读一次带超时的 `timeout 20 sudo -n dmesg`（再 tail）和有界的 `/sys`，不给 D 状态进程发信号，写 `.stop` 哨兵后转做 CPU 工作 | 09-15 整机失联；反复 kill 留下更多不可杀进程 | 09-15 起 | 此后再没有因恢复操作而加重的事件 | 457593d8；skill:`gfx1250-card-safety` §4–§5 |
| 风险按启动计价：多个配置合进一个进程；用 60 步长 run 代替多次短 run；两次运行之间等 KFD 清空，再冷却 25–45 s；单卡上 fwd/bwd job 不并行；每卡一把 flock 锁 | 启动期族（21%/启动） | 09-16 起 | B0 ≥12 次启动 0 次挂卡；10-02 e2e 0 故障 | skill:`gfx1250-card-safety` §0；`1002__e2e/E2E-PLAN.md` |
| compile-only 先行（`COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 HIP_VISIBLE_DEVICES=-1`）；只要 spill>0 或 `private_segment_fixed_size>0` 就直接淘汰 | spill 的 build 首次 launch 即挂起（代价是一次断电） | 09-21（b1572877） | VGPR ≤951 从不 spill；=1024 必 spill | skill:`env-and-pitfalls` §2b §6 |
| 新地址算术先在 CPU 上做越界证明，全枚举而不是抽样 | C 类越界读 | 09-23..09-24 | k_dq 全枚举 134,217,728 元组，结论为双射；另找到 3 个潜在缺陷（见 d.6） | 41c916b7；h22/h33 |
| 新 kernel 上卡的顺序：toy 形状 → 独立进程 → `AMD_SERIALIZE_KERNEL=3` → 不做进程内 autotune → 经锁/KFD 包装 → 只用镜像 hipBLASLt。h85 必须写成 hint 表格行 | 10-02 挂卡 | 09-17 起写进 skill，10-02 升级为 h85 | 09-30 手工 campaign s1–s6 期间 0 挂卡（w4f 首次上卡只用 toy、nsp=1、串行单进程） | 1ee0dd59；skill:`gfx1250-card-safety` §1 #13、§2 |
| 一 shape 一进程，首个 rc≠0 即中止 | 09-24 挂卡；shape 边界故障约 44%/进程 | 09-24（bf58bec0） | 之后 6 次完整测量 0 故障 | `0924__flydsl/wedge-rootcause/` |
| 参考值离开 GPU（refcache），并且只用镜像 hipBLASLt 库 | A 类挂卡 | 09-22（376b9fd3 等），10-02（h85） | prod 参考在卡上直接 FAULT，改在 CPU 上算，10m26s，产出 1.1 GB | 376b9fd3（commit message）；`0922__flydsl/gate-patch/README.md` |
| 禁用 PC sampling | 09-11 挂卡；09-24 差一点挂卡 | 09-11 起；09-24 重申（82420bdf） | 旧固件上 3 次尝试全部 fault，其中 1 次挂卡（09-11 10:48），2 次卡存活（09-11 10:32、09-24 ~05:24）。`0930__bwd/PLAN.md` D2 和 skill:`gfx1250-attn-campaign/SKILL.md:106` 写成"3/3 挂卡"，说重了。刷固件后 rocprofv3 1.3.2 启动时直接拒绝该配置 | skill:`env-and-pitfalls` §2c；OE:`output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md` §2.3；82420bdf |
| 只按显式 PID 杀进程，绝不用 `pkill -f`（会匹配到自己，已杀过 4 次以上自己的 shell）；长循环用 `setsid nohup` 启动，停止用 `op-evolve stop` 或 `kill -TERM -<PGID>`；watchdog 绑定 `E2E_RUN_MARKER` 或 PID；定期查 ppid=1 的孤儿 | 误杀、孤儿占卡、监控盯错 run | 09-14..09-25 | 6ad9bda0（按进程组回收）；6029b430 | skill:`gfx1250-card-safety` §6；`op-evolve-ops` §3 |
| 监控与守护：`patrol.sh` 挂卡告警改为锁存并与开机时间比对（617481d7）；`release_guard` 写 pidfile、由 patrol 拉起，新增"supervisor 停在退避""state.yaml 冻结"检查（dd1aec29、4547915c）；`card_ok.sh` 看 KFD 持有者、`/sys` VRAM，并跑一次带硬超时的玩具 matmul（ac95f1c7）；免 root 的活性探针 `timeout 20 docker exec fa-repro true`；只对 MES failed/unrecoverable 和 docker exec 超时报警 | 告警只响一次；守护进程从未启动；健康脚本误报 HEALTHY | 09-21..09-24 | 09-21 起每次挂卡都在一个巡检周期内报出 | 617481d7；3fd48059 |
| 在线调优改为离线建表，坏候选记为终态 | 训练步内 autotune；sweep 候选挂卡 | 09-16 | `flydsl_table.py`：每形状一个子进程 + 硬超时 + SQNR 门；`num_warps=16` 和 #58 永久排除 | f08ccffe；e4730158 |
| e2e 防线：禁止 `converters: []`；`compile.enable: false`；留显存余量（8L 或 `E2E_MEM_STOP=89.5`）；前两步先跑 ASM；BLAS guard（exit 94）；非有限值 watchdog（exit 96）；外来 KFD 进程（exit 91） | 09-15/16、09-28 的 e2e 挂卡 | 09-15..10-02 | 10-02 e2e 两个 92 步进程 0 故障 | `1002__e2e/E2E-PLAN.md` §3 |
| 委派任务时明确禁止上卡，并在委派前后检查 KFD 持有者 | 09-16 agent 私自上卡 | 09-16 | 此后再没有私自上卡 | 4e03bd57 |
| 每轮结束都 push；每次 AC 后跑 `git fsck --no-dangling` | 断电截断 git 对象（见 d.7） | 09-23 07:36 | 09-23 09:42 那次 AC 没有损坏仓库 | memory `push-after-every-round` |

**AC 之后的恢复步骤**（gfx1250-card-safety §5、`1002__oe/incident/WEDGE-1002.md`）：

```bash
# 0. 用户 AC 断电（只有这一步能恢复 wedge；绝不 modprobe -r）
sudo modprobe amdgpu                    # 内核 cmdline 里 modprobe.blacklist=amdgpu，不会自动加载
sleep 6
sudo sysctl -w kernel.dmesg_restrict=0  # 每次重启都会回到 1；确认 dmesg | wc -l 是几千行
cat /sys/class/drm/card*/device/pp_dpm_sclk   # 检查 DPM 表/VBIOS/dkms 是否又被人改过
ls /sys/class/kfd/kfd/proc/             # 必须为空
docker start fa-repro                   # 复用，不重建；确认容器内没有镜像版 primus_turbo
# 一次带超时的 4096^3 bf16 matmul（A0-限频 约 5.07 ms）——这才算"卡能用"，rocm-smi 不算
git fsck --no-dangling; find .git/objects -size 0   # 断电可能截断 git 对象
# 先补 hint（如 h85）再 op-evolve resume（不带 --config）
```

#### d.1.4 AC 断电 / 重启次数统计

| 日期（UTC） | A0 因挂卡引发的人工断电/重启 | 说明 |
|---|---|---|
| 09-11 | 2 | E3 外积、E4 PC sampling；运维执行，是 AC 还是热重启没有记录 |
| 09-13 | 1 | 第 5 次 wedge（从 09-04 起累计）；重启时间未记录 |
| 09-15 | 5 | 机器失联（07:18 前）、SIGBUS（~08:20）、8L worker（09:00）、noconv（11:16）、nk4（13:07） |
| 09-16 | 9 | 02:26、07:35、07:56、08:20、09:18、11:17、11:57、12:44 有用户确认；~13:15–13:30 为推断 |
| 09-21 | 2 | ~14:55、09-22 00:49（都是 r3） |
| 09-22 | 2 | 14:40（r8）、~23:47（r9） |
| 09-23 | 2 | 09:42 前（r11）、~14:00（r14） |
| 09-24 | 1 | 09:14（r18） |
| 09-28 | 2 | 12:10（p4a）、13:01（p4b），都在刷固件之前 |
| 10-02 | 1 | r27，刷固件之后；AC 时间未记录 |
| **合计** | **27（区间约 25–28）** | 下界 25：09-11 两次若是热重启则不计。上界 28：再加 09-29 那次无效 AC |

- 另有 1 次：09-29 驱动/固件不匹配，用户 AC 无效，随后管理员刷固件，期间主机约 10 次重启。把它算进去，人工断电合计 **28 次**（即上表的上界；不含它为 27 次）。全报告统一用这两个数。
- 活动开始前 A0 还有 2 次挂卡重启（09-04、09-06/07）。窗口外 10-06 有 1 次（其它项目）。
- B0：我方没有发起过 AC。GPU1 在交还时仍需 AC；GPU0 只是降级。
- 文档口径要更正：skill `gfx1250-card-safety` 写"09-16 一天五次"，DAY-0916-SUMMARY 也只记了 5 次，那只是 08:20 之后那一段。按 UTC 算，09-16 实际是 9 次。
- 另有约 13 次原因不明或由他人造成的主机重启，不计入上面的统计：09-13 ~04:11、09-17 ~07:46（造成 6 个 git 对象被截断，见 d.7）、09-22 06:47:43、09-23 ~18:15、09-24 ~21:55、09-25 03:10、09-25 10:10:51、09-28 ~02:10 和 08:13 前（这两次是 CPU BERT 致命错误导致主机崩溃）、09-28 21:50（管理员）、09-29 01:47、09-30 01:31、10-01 20:41（CPU L3 致命错误）。其中部分时间取自 journalctl 首条记录，可能与实际开机时间有偏差。
- 时间代价举例：09-21..22 五轮墙钟 21.65 h，其中 17.8 h（82%）花在挂卡和等待 AC 上，实际工作只有 3.85 h（s:a9a96fef 09-22）。

---

### d.2 GEMM / hipBLASLt 问题与优化

#### d.2.1 时间线

| 日期 | 认识/现象 | 数字（机器/时钟） | 处理 | 来源 |
|---|---|---|---|---|
| 09-11 | 镜像 `primus-turbo:gfx1250-20260831-extended-v2` 上 hipBLASLt 能用，但很慢 | A0-限频：bf16 8192³ 91.5 TF/s（各布局 60–92），Triton GEMM 1002.7，差 10.96× | 不用 hipBLASLt，改用 Triton GEMM 作 roof | OE:`output/0911__fa_gfx1250_phase1/E1.md` §5 §8 |
| 09-13 | 换成 `amdprimus/amdprimus:gfx1250-20260910`（fa-tune:deps）后，任何 matmul 都报 `HIPBLAS_STATUS_INVALID_VALUE` 和 `Cannot read …/TensileLibrary_lazy_gfx1250.dat`。当时误判为"镜像缺 gfx1250 Tensile 库"，按"~37×"写进了 PLATFORM-ESCALATION | A0-限频：`TORCH_BLAS_PREFER_HIPBLASLT=0` 走 rocBLAS，8192³ 27.4 TF/s；e2e 32L 245 tps、133.7 s/step | harness 默认 PREFER=0，只保证参考能跑；得出"e2e 无法验证 attention" | `0913__opt_plan__claude/phase1/RESULTS.md` §5 §11 |
| 09-14 | B0 上更正：库是在的（46 个 bf16 解）；换环境变量的三种组合结果相差 <1%；当时结论是"hipBLASLt 在 gfx1250 上本来就慢" | B0：torch.mm 112.98 TF/s，朴素 Triton 1190.07（差 10.5×）；flex eager e2e 2,392 tps，compile + 强制 Triton GEMM 后 9,602（4.01×）；只开 compile 反而 −4% | 设 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` 和 `TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=TRITON`（后来证明在 A0 上开 compile 会挂卡） | `0914__campaign/RESULTS.md:39-57,132-146` |
| 09-15 | 真因①：路径错位。加载器在 `library/` 下找索引，402 个文件实际放在 `library/gfx1250/` 子目录（镜像打包缺陷）。指向 `_rocm_sdk_devel` 那份会 core dump | A0-限频：8192³ 从 rocBLAS 27.64 到 68.74 TF/s（2.49×）；e2e 32L 单步 135/134 s → 17/17 s（7.9×），吞吐 244 → 2,027 tps（8.3×；两臂各 3 步、非稳态、n=1） | 设 `HIPBLASLT_TENSILE_LIBPATH=<镜像>/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`；写 VENDOR-REPORT-hipblaslt.md 上报 | `0915__opt/BLAS-FINDING.md:15-31,49-58`；`0915__opt/VENDOR-REPORT-hipblaslt.md:58-61` |
| 09-15 | 真因②：开 `HIPBLASLT_LOG_LEVEL=4` 和 `TENSILE_DB=0x6` 后看到，NN（`Ailk_Bljk`，即 dgrad）只有稀疏的 GridBased 表，缺纯 bf16 GEMM 库 `BB_BB_UA_Type_BB_HPA`（TN 有 3 份，含 CU96/CU192 变体；文件数 TN 68、NN 40）。查表最近邻落到 N=1，选中 GEMV 用的 tile `MT32x16x32`。这也是 e2e 三模态噪声的来源 | A0-限频：GEMM 占单步 94%，其中 97% 在 MT32x16x32；单次 592–681 ms，grid 6567 万 WG；同形状走 TN 快 16.5–21.5×；同一张卡 torch NT（库内为 TN）跑到 1502 TF/s | nkfix v1（改 dgrad 的 B 布局）：8L 6,128 → 11,604 tps（1.894×，n=9）；跨运行 sd 从 3.91%（三模态）降到 0.42% | `0915__opt/GEMM-NN-FINDING.md`；`0915__opt/VENDOR-REPORT-hipblaslt.md` |
| 09-16 | wgrad（`Ailk_Bjlk`）仍然落在 MT32x16x32，57 次调用共 2185.6 ms，占单步 77%。v2 规则的微基准把布局造错了，e2e 里命中 0 次。实测只改 A 或只改 B 都只能到 60–70 TF/s | A0-限频：两个操作数都改后 1145.8–1627.0 TF/s；8L 37,746 tps（6.16×，n=5）。原报 38,043/6.21×（n=6），其中混入了一次 NaN 运行，已更正 | nkfix v3：wgrad 判据放在前面，A 改为连续、B 改为 N-major；统计写到 `NKFIX_STATS_FILE`（atexit 的输出会被启动器吞掉） | `0915__opt/PROFILE-POST-NKFIX.md`；`0915__opt/GEMM-WGRAD-FINDING.md:24-76`；`PROGRESS-REPORT-0917.html` |
| 09-16 | 规则 3：用同事的 FlyDSL gfx1250 WMMA GEMM 接管 wgrad，去掉约 137 ms/步的转置拷贝。第一次把 autotune 放在训练步里跑，卡挂了 | A0-限频：8L 46,374 tps，同会话对照 42,445（+9.26%，n=3，累计 7.57×）；32L 1,984 → 14,050（7.08×，n=1；同会话对照 12,340） | 改成离线建表 `flydsl_table.py`，表的 key 用完整 (M,N,K)；写 VENDOR-REPORT-flydsl-gemm.md | `0915__opt/DAY-0916-SUMMARY.md`；`0915__opt/FLYDSL-WEDGE.md` |
| 09-16/17 | 扫描 74 次 e2e：带 nkfix 的 39 次里有 8 次 loss NaN，不带的 35 次为 0（Fisher 单尾 p=0.004）。NaN 运行都是各自组里最快的 | A0-限频：撤回 49,878、32L 11,608/5.85×、38,043/6.21×；ASM bwd 的 +14.40% 改为 +12.72% | `e2e.sh` 自动统计 NaN 并打印 DISCARD；加累加器式有限性检查 `NKFIX_CHECK`；规则 3 默认关闭；nkfix 判为不可交付 | `0915__opt/NKFIX-NAN-RATE.md`；`0915__opt/NAN-FINDING.md` |
| 09-17 | 为了绕开 INVALID_VALUE，在 op-evolve 发车前往 fa-repro 的 `/usr/lib/python3.12/sitecustomize.py` 追加了 `setdefault("TORCH_BLAS_PREFER_HIPBLASLT","0")` | — | 后果：接下来 12 轮里，job 进程中 `_env.py` 的 setdefault 全部无效，torch GEMM 一直走回落路径 | s:c4b79aa6 09-17T12:01Z；`0923__flydsl/STAGE2-S0-PROBE.md` S0-b |
| 09-21..22 | 闸门、benchmark、bitwise 里的 eager fp32 参考在卡上走 Tensile（`Cijk_Ailk_Bljk_SB_MT128x64x8`），触发 aperture violation，造成 A 类挂卡 | A0-限频：r3 挂卡两次；r8/r9 挂卡前也都跑过这条路径 | refcache（376b9fd3），并补上漏掉的 4 个调用点（00128018、a5693e9c、93ad3b89） | `0922__flydsl/gate-patch/README.md` |
| 09-23 | S0-b：确认"缺库"的诊断不成立，并列出容器里三个库目录：`/opt/rocm/lib/hipblaslt/library` 是空的（容器的 ROCm 不在 `/opt/rocm`）；默认搜索路径 `_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library` 这一层没有 gfx1250 载荷，是 INVALID_VALUE 的真因；`_rocm_sdk_devel/lib/hipblaslt/library/gfx1250` 有 326 个文件但残缺（无 `Kernels.so-000`、无 extop/transform，`.dat` 未压缩），指向它会 SIGSEGV（exit 139）。表里没有 09-15 已验证可用的 `library/gfx1250/` 子目录（402 个文件），于是判为"只有宿主库完整"，把宿主 ROCm 10.1.0 的 328 个文件拷到 `~/.local/hipblaslt-gfx1250/gfx1250` 使用，而且只验证了 512/2048/4096 的方阵。此后镜像子目录库（09-15 方案，09-28 B0 e2e 也用它）与宿主库两条路径并存，直到 10-02 h85 才统一为镜像库 | A0-限频：fp32/bf16 方阵和当年 fault 过的 fp32 QK^T 都跑干净 | `TORCH_BLAS_PREFER_HIPBLASLT=1` 改为显式赋值，保留 `OPEVOLVE_KEEP_BLAS_ENV=1` 作逃生口 | d9160519；`0923__flydsl/STAGE2-S0-PROBE.md` S0-b；`0915__opt/BLAS-FINDING.md:15-31` |
| 09-24 | LIBPATH 设对之后，refcache 的前提不复存在 | A0-限频：prod 上 `forward_reference` 0.88 s + `eager_attn_bwd` 0.89 s = 1.77 s，跨进程逐位一致 | 仍保留 refcache，因为偶尔还有非致命 page fault | 1b19fbe4；`0924__flydsl/REFCACHE-PREMISE-GONE.md` |
| 09-28 | B0 e2e 的反向 GEMM 同样全部落在 MT32x16x32。另外 A0 的 `0915__opt/bin/nkfix.py` 在 HEAD 上已经坏了（`_MM/stats/_big/_headroom_ok` 未定义） | B0：修复前 GEMM 15.6–16.5 s/步（占 96%），ASM e2e 2,016–2,022 tps（高档）；op 级 dgrad/wgrad 50–79 TF/s，lm_head 50–55 TF/s | 以 1255557f 为基础重写 `nkfix_b0.py`。修复后每步 997 次 GEMM 全部落在 `Alik_Bljk MT256x256x128`，GEMM 0.83–0.86 s，转置和检查另占 0.1 s；ASM 20,718 tps（10.3×），FlyDSL 19,762，Triton@1cb2e183 14,563；8 次运行共 486 步，0 次 NaN | `0927__b0/gemm/REPORT.md` §0–§4 |
| 09-28 | GEMM 修好后整板功耗顶到约 2.13 kW，bwd 阶段 sclk 从约 1,820 降到约 1,390 MHz。单步拆分里 attention kernel 本身慢了 10–16%：gemm/REPORT 记为"原因没查"，profile/REPORT §0.3 归因于时钟。同进程的配对比值不受影响 | B0：attn bwd ASM 197 → 229、FlyDSL 269 → 300 ms/步；k_dkdv 5.049 → 5.95–5.99 ms（+18%）；ASM bwd 主 kernel 6.008 → 7.00–7.09 ms（+17%）；k_dq +2% | 绝对 ms 一律用修复后的数；建议设 `NKFIX_CHECK=0` 降低 GEMM 功耗 | `0927__b0/profile/REPORT.md` §0.3 §3；`0927__b0/gemm/REPORT.md` §5 |
| 09-28 | 宿主库在 B0 上出问题：p1a_turbo 用宿主库，step 1 卡住 14 min（gdb 显示主线程在等 AQL 槽位，dmesg 干净）。GEMM 突发尺子用宿主库时，一次突发 47 ms（约 80 TF/s，慢约 20×），时钟也不降 | B0：`kb6_prod_gb_*` 数据作废 | launcher 在 `bash -c` 里给镜像库赋值；尺子加载 arm 之后再赋一次值（fwd 树的 `_env.py` 会改写它）。VERIFY 起初认为归因证据不足（同时变的还有 roctracer、每步 inductor block mask）；RESULT.md 随后用 p1a/p1b 单变量对照（只差 BLAS 库，p1b 正常跑完）排除了这两个嫌疑，归因到宿主库，每边 n=1 | `0927__b0/e2e/RESULT.md:50-59`；`0927__b0/fwd-nospec/REPORT.md` §2；`0927__b0/e2e/VERIFY.md:28-38` |
| 09-28 | 宿主库在 A0 上出问题：fwd 树的 `_env.py` 在首次 attention import 时把 LIBPATH 改到宿主库 | A0-限频：a0_p3b 在 step 4 出现 grad inf，之后 86 步 NaN，非有限值全部出在 bwd GEMM 输出；a0_p4b 在 step 1 挂卡；两个"ASM 先跑"的进程都正常（只是相关性，没有在卡上证实） | e2e 副本里删掉这两行（`fwd_r16_imglib`） | `0928__a0_repro/REPORT.md` §5.3 |
| 10-02 | A0-新固件 e2e 加了几道 BLAS 防线：preflight 用 grep 拒绝任何 BLAS 赋值、setdefault、putenv；运行时 guard 发现改写就打 `!! BLAS-REPOINT` 并 exit 94；前两步先跑 ASM；`NKFIX_CHECK=1` 遇到非有限值就停（exit 96）。同一天 bwd r27 的 opt agent 仍用宿主库在 prod 跑新 kernel，卡挂了 | A0-新固件，`NKFIX_CHECK=1`：0 次 BLAS 改写，0 故障；ASM+nkfix 1,350.6 ms/步、24,262 tps；fly（r16+s6）1,349.4/1,349.9 ms（24,283 tps） | h85：禁止使用 `~/.local/hipblaslt-gfx1250`，只用镜像库 | `1002__e2e/E2E-PLAN.md` §3；`1002__e2e/RESULT-e2e.md`；`1002__oe/incident/WEDGE-1002.md`；wt-llama31 README:189-190（tps） |
| 10-05 | 教程把配置固定下来：镜像库，`bash -c` 里显式赋值，nkfix hook，transpose_triton | A0-新固件：分支 FlyDSL attention + nkfix（CHECK=0）1,361.9 ms/步、24,061 tps；按提交版本重跑 1,360 ms、24,099 tps；漏了 `transpose_triton.py` 时 1,689 ms、19,406 tps。这三个数都是 CHECK=0，不能和 10-02 的 CHECK=1 数直接比：按 CHECK=1 约 +33 ms/步（B0 实测值，教程沿用）折算，ASM 在 CHECK=0 下约 1,318 ms，分支 FlyDSL 约慢 3%（推算，未实测）。固件更新前后 ASM+nkfix 1,947 → 1,351 ms（1.44×；两边都是 CHECK=1）。教程称前后"同代码同方法"，但这个倍数不纯是平台差异：1,947 来自 09-28 a0_p3a，该进程 step 2 起加载的 FlyDSL fwd 树 `_env.py` 把 `HIPBLASLT_TENSILE_LIBPATH` 改指到宿主库；两次之间驱动也从 7.1.1-2397345 换成了 7.1.0-2412954（见附录 A.2） | — | wt-llama31：`docs/gfx1250_llama31_8b_e2e/README.md` §3.2 §6–§7 |
| 10-06（窗口外） | 其它项目用 `hipblaslt-bench --algo_method all` 枚举所有解，碰到坏的 Tensile 解后挂卡 | A0-新固件 | 不再枚举全部解，只测预先筛过的短名单 | skill:`gfx1250-card-safety` §1 #14 |

#### d.2.2 nkfix 是什么、怎么装

- **问题**：镜像里的 hipBLASLt 只给前向 Linear 用到的 TN 布局（`Alik_Bljk`）发布了纯 bf16 调优库。反向的 dgrad（`Ailk_Bljk`）和 wgrad（`Ailk_Bjlk`）只能查稀疏的 GridBased 表，最近邻落在 N=1，结果选到 GEMV 用的 tile `MT32x16x32`，只有 50–80 TF/s。前向能到 1.5–1.9 PF/s，单次调用差 11–32×。
- **做法**：用 `TorchDispatchMode` 拦截 `aten::mm`，把反向 GEMM 的操作数物理布局改成前向那种，让它命中 `MT256x256x128`：
  - dgrad 在 B 连续时改为 `mm(A, B.t().contiguous().t())`。
  - wgrad 改为 `mm(A.contiguous(), B.t().contiguous().t())`。wgrad 的判据必须先于 dgrad 判断。
- **试过不行的方法**：
  - 零拷贝写法 `(BᵀAᵀ)ᵀ` 仍然落到坏 tile，只有 0.83–0.95×，因为这份库只有 TN 一种布局有好 tile。
  - 用 `torch.library` 覆盖 `aten::mm` 会无限递归。
  - 放进 `sitecustomize` 时 torch 还没加载，装不上。
- **数值**：op 级 SQNR 与原调用相同，都是 55.59–55.63 dB；99.996% 以上的元素差在 1 ulp 以内；3 次重复逐位确定。
- **版本**：
  - A0 v1（09-15）只改 dgrad。
  - A0 v3（09-16）加上 wgrad；规则 3 为 FlyDSL wgrad，走离线表，默认关闭。
  - B0 `nkfix_b0.py`（09-28）在 1255557f 基础上重写：
    - scratch 常驻，按 (device, stream) 区分。
    - 沿输出维分块，块大小 `NKFIX_CHUNK_BYTES` 默认 256 MiB，不拆 K 归约。
    - 转置用 Triton 分块 kernel `transpose_triton.py`，与 torch 拷贝逐位相同，快 5–7×。
    - `NKFIX_CHECK=0/1/2` 检查非有限值；`NKFIX_SHADOW` 在真实训练中做影子对照。
  - 教程使用的就是 B0 这一版。
- **安装**：
  - 教程方式：`primus-nkfix-hook.patch`，在 Primus 的 `primus/core/runtime/train_runtime.py` 的 `_run_trainer_lifecycle` 里加 14 行，`NKFIX_ENABLE=1` 时 import 并 `install()`。
  - B0 e2e kit 方式：shim 的 `primus_turbo/__init__.py`，配合 `E2E_NKFIX=1`。
  - A0 09-15/16 的 Primus 侧 hook 只备份在 `0915__opt/primus-side/`，没有推送。
  - `transpose_triton.py` 必须和 `nkfix.py` 放在同一目录。
- **开销**：每步约 70 ms；开 `NKFIX_CHECK=1` 再加约 33 ms/步。

#### d.2.3 e2e 效果（不同机器、时钟、层数，不可横比）

| 机器/时钟 | 配置 | 修复前 | 修复后 | 说明 | 来源 |
|---|---|---|---|---|---|
| A0-限频（负载约 1.0 GHz） | 8L，b4 s8192 | 6,128 tps（n=9，三模态，sd 3.91%） | v1 11,604（1.894×）→ v3 37,746（6.16×，n=5）→ v3+规则 3 46,374（7.57×，n=3） | 规则 3 有在线调优挂卡史，默认关闭 | `0915__opt/GEMM-NN-FINDING.md`；`0915__opt/DAY-0916-SUMMARY.md` |
| A0-限频 | 32L 生产配置（含 ASM bwd） | 1,984 tps（16.5 s/步，n=1） | 14,050 tps（约 2.33 s/步，7.08×，n=1；同会话对照 12,340） | 32L 的复现率从未有效测量过 | `0915__opt/RESULT-32L.md` |
| B0（满频；修复后训练中约 1.4 GHz） | 32L | ASM 2,016–2,022 tps | ASM 20,718（10.3×）；FlyDSL 19,762；Triton 14,563 | GEMM 占比从 96% 降到约 53%，另有 6% 是转置和检查 | `0927__b0/gemm/REPORT.md` |
| A0-新固件 | 32L | 刷固件前（A0-限频）ASM+nkfix 1,947 ms/步（09-28 a0_p3a；与修复后一行之间还差了 LIBPATH 改写和驱动版本，见附录 A.2） | ASM+nkfix 1,351 ms/步、24,262 tps（10-02，`NKFIX_CHECK=1`）；fly r16+s6 1,349 ms/步、24,283 tps（10-02，CHECK=1，未进分支）；分支 FlyDSL 1,362 ms/步、24,061 tps（10-05，CHECK=0） | 不开 nkfix 约 16 s/步、约 2k tps（B0 实测）。CHECK=0 与 CHECK=1 的行不能直接比（CHECK=1 约 +33 ms/步）；同为 CHECK=1 时 fly 对 ASM 是 0.999；按 +33 ms 折算，分支 FlyDSL 约比 ASM 慢 3%（推算，见 d.2.1） | wt-llama31 README §3.2 §7（:88-89、:184-191）；`1002__e2e/RESULT-e2e.md`；`0928__a0_repro/REPORT.md` §2、§5.3 |

#### d.2.4 NaN、`HIPBLASLT_TENSILE_LIBPATH` 与宿主库的风险

1. **nkfix 与 NaN（A0-限频，09-16）**
   - 带 nkfix 的运行 8/39 出现 NaN，不带的 0/35（p=0.004）。另有统计：带 nkfix 40 次中 7 次只出现 `grad_norm=inf`，不带的 36 次中 3 次。
   - 所有 NaN 都发生在 step ≤10，而且和当天的 MES TLB 故障混在一起。根因没有找到。
   - B0 重写版共 8 次 nkfix 运行、486 步（150 个不稳定步），0 次 NaN。上界只按其中 32 层的 7 次运行、466 步（138 个不稳定步）计算：按步 95% 上界是 2.2%/步；按次是 35%/次，还不足以排除 A0 的 21%/次，要压到 21% 以下约需 13 次干净运行（`0927__b0/gemm/REPORT.md` §0 §4）。
   - 因此 nkfix 保持 opt-in；长跑或排查时开 `NKFIX_CHECK=1`；只要出现 NaN 或 inf，这次运行的速度数据整次作废。
2. **库路径**
   - 默认搜索路径高了一层目录：`library/` 下只有一个 `gfx1250/` 子目录，402 个文件都在里面，结果是 `INVALID_VALUE`。09-23 S0-b 的目录表没有列这个子目录，因此转用了宿主库（见 d.2.1）。
   - `_rocm_sdk_devel/.../library/gfx1250` 是残缺版（326 个文件，无 `Kernels.so-000`、无 extop/transform），会 SIGSEGV，exit 139。
   - `/opt/rocm/lib/hipblaslt` 是空目录，容器的 ROCm 根本不在 `/opt/rocm`。
   - 回落到 rocBLAS 时 e2e 慢 8×，它的 fp32 Tensile kernel 还曾让卡 page fault。
3. **`TORCH_BLAS_PREFER_HIPBLASLT` 会被静默改写，至少有三处**
   - `sitecustomize.py` 用 `setdefault("0")`，是 09-17 我们自己加的。
   - `/etc/profile.d/zz-gfx1250.sh` 里 export 为 0，只在 login shell 生效，所以 `bash -lc` 会中招。
   - `tune_attention.py` 在 import torch 之前自己设成 0。
   - 规则：只用赋值，不用 `setdefault`；用 `bash -c`；在进程内打印 `torch.backends.cuda.preferred_blas_library()` 确认。
   - 另外，op-evolve spec 里的 `runtime.env` 实际不生效，必须写进 op 自己的代码或 exec 命令行。
4. **宿主库 `~/.local/hipblaslt-gfx1250`（宿主 ROCm 10.1.0 的拷贝，328 个文件）**
   - 只验证过方阵。
   - 与这些事件相关：B0 step 1 卡住（p1a/p1b 单变量对照，每边 n=1）；GEMM 突发慢约 20×；A0 09-28 的 NaN 和挂卡；10-02 挂卡。除 p1a/p1b 外，都只是相关性证据。
   - 10-02 起由 h85 禁用。
   - `skill:env-and-pitfalls` §1b 至今仍推荐宿主库，已过时，需要更新。
5. **卡上的 fp32 GEMM 参考**：会导致 A 类挂卡（refcache）和 B0 GPU2 故障（h50）。测试和闸门的参考一律放在 CPU 上算或走 refcache；refcache 的 sha 不一致时直接判 FAIL（见 d.5）。
6. **`hipblaslt-bench --algo_method all`**：会碰到坏的预编译解而挂卡，不要使用。

#### d.2.5 当前推荐配置（10-05 教程，A0-新固件实测）

```bash
# Primus 打 nkfix hook：git apply docs/gfx1250_llama31_8b_e2e/primus-nkfix-hook.patch
# 容器内：pip uninstall -y primus_turbo（镜像 editable 安装会遮蔽 checkout）；不要用 REBUILD_PRIMUS_TURBO=1
docker exec -w $PWD fa-repro bash -c '            # bash -c，不用 bash -lc
export TORCH_BLAS_PREFER_HIPBLASLT=1              # 显式赋值，绝不 setdefault
export HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250   # 只用镜像库
export NKFIX_ENABLE=1 NKFIX_CHECK=0               # 长跑/排查 NaN 时 NKFIX_CHECK=1（+~33 ms/步）
export PYTHONPATH=$TURBO/docs/gfx1250_llama31_8b_e2e/nkfix:$TURBO   # nkfix.py 与 transpose_triton.py 同目录
...'
```

- 检查点：
  - 日志里要有 `[nkfix] installed`，并且不能出现 `[nkfix] triton transpose unavailable`。
  - 稳态约 24k tps（A0-新固件）。稳定在约 19.4k 说明缺 `transpose_triton.py`；只有约 2k 说明 nkfix 没生效。
- yaml 配置：`flavor: 8B_flex`，`converters: ["primus_turbo"]`，`use_turbo_attention: true`，`compile.enable: false`，设置 `debug.seed`。
- 根本解决还得靠镜像方：补 NN/wgrad 的 bf16 调优库，修正库路径（`0915__opt/VENDOR-REPORT-hipblaslt.md`）。

---

### d.3 计时与尺子问题

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| ASM bwd 标杆 10.160 ms 是 shim 伪影 | 09-15..09-24 一直把 10.160 ms / 541 TF/s 当作 ASM bwd 标杆（A0-限频），FlyDSL bwd 因此被报为 ASM 的 0.92× | 09-15 那次测的是 `_AsmFwdAsmBwd` autograd shim，调用 `asm_backward` 时没传 `hip=`/`scratch=`，每次都做 3 次 hipModuleLoad 并新分配约 1 GiB scratch，还算进了 autograd 管路。修正后的路径（8.13–8.68 ms）当天就有，但没有传播开；这个数还被误标为"产品路径" | 改用 op-evolve beat 在同进程里普查：7.6766 ms（n=224 raw）/ 7.6769（n=146 去重），约 716 TF/s。n=177/7.6721 那一版也已撤回。规定只引用同进程 beat 的比值 | 进度被高估：0.92× 实际是 0.70–0.72×。11.726 ms/层、4.76×、1.74× 一并作废。开销拆分（+1.15 / +1.42 / +0.01 ms）只有文字描述，只能引总值 | skill:`baselines.md` §0 §5；77963d7e；`0924__flydsl/DAY-SUMMARY.md` |
| 逐次交错计时与分块计时（B0，09-28） | 同一份代码，换个位置或换个同进程 arm 组合，读数就差 2–10%。r6/beat 交错测 1.29×、稳态 1.035×；bwd beat/current 交错测 0.7618、分块测 0.7878 | 每次调用都继承前一个 arm 留下的功耗/时钟状态，2 GB flush 也消不掉，结果偏向功耗高的 arm | 改为分块计时：每 arm 先跑 4 次再计 9 次，回文排序（fwd 03:40 起用 h40/h41，bwd 约 05:00 起用 h66）。每个排名进程带一个 A/A 副本（fwd ±0.19%，bwd ±0.05%），偏离超过 0.5% 整个进程作废（h35） | 三个假赢：r6 报 +5.7%（实际 +4.6%）；L12 报 +6.4%（实际 −0.4%）；L21 报 +6.3%（实际 +0.3–0.6%，两份文档口径不同，都在噪声边缘）。r6 之前的 %ASM 系统性偏低（fwd 约 25%，bwd 约 3.4%）。新旧 harness 的绝对 ms 不能比 | `0927__b0/ruler/REPORT.md`；`0927__b0/ruler/bwd/REPORT.md`；`0927__b0/REPORT-0928.html` §6 |
| ASM 之后的 I-cache 惩罚（A0-限频 fwd job，09-25..27） | 同进程里先跑 aiter ASM，再跑 FlyDSL kernel，I$ miss 从 0 变成 808，冷启动固定多 25–42 µs；fast/proxy 的中位数随臂的位置漂移 | beat 与候选在同一进程里回文排列（A B C C B A），紧挨着 beat 的臂吃亏。这和功耗、循环次数、代码体积都无关。B0 审计后把同类现象解释为"状态继承"，两种解释对应的修法相同 | h28：候选和冠军在不含 beat 的进程里比较，至少轮转 3 个 session；beat 放到单独进程测 | r3/r6/r8 的判定都受了影响；r8 nodelay 的 proxy 0.956（去掉 beat 后为 1.004）被错误拒绝 | `0925__flydsl/fwd-job/hint.md` h28；skill:`fwd.md` |
| 用 randn 输入还是真实数据（B0 09-28；A0 10-02 复核） | B0：r6 投机 softmax 在 randn 上快 4.6%；但训练中 FlyDSL fwd（r6）对 ASM 逐层是 1.19（第 0 层）到 1.72（第 7 层），而且越训越慢（50.2→55.3→59.2 ms/步）。op 级用真实输入、在刚跑完 GEMM 的时钟下是 1.45–1.67；同一批数据用分块尺子测是 1.15–1.30，randn 分块是 1.02 | 真实 q/k/v 的 score 标准差为 21–53（randn 约为 1），13–25% 的 tile 步会触发重算（randn 为 0%） | 用 step 43 的 6 层真实 dump（`/home/lihuzhan/_prof_dump/qkv_call0*.pt`）作第二把尺子（h45/h47），r18 起两把尺子都报。关掉投机的 r13ns 经 refactor h44 采用：randn 上慢 4.8%，真实数据加训练时钟下快 12–23% | fwd r16 起，job 的 randn 读数按设计会偏低约 4.8%，r16 前后的 %ASM 不能比。A0 09-30 的 s1–s6 只用 randn 测（当时以为 dump 只在 B0，见 d.7）。10-02 在 A0-新固件上用同一批真实 dump 补测：真实数据与 randn 一致（bwd s6/ASM blk 0.972，randn 0.957；fwd r16/ASM 1.081，randn 1.080），r16/s6 的结论在真实输入上成立。拉开差距的是工作点：gb 下 ASM 几乎不动，FlyDSL 两个方向都变慢，bwd 从领先 2.8% 变成落后 3.8%，fwd 从落后 8% 变成落后 35%（见下一行） | `0927__b0/profile/REPORT.md` §0 §1 §2.2；`0927__b0/fwd-nospec/REPORT.md`；`1002__e2e/RESULT-realab.md` |
| 训练工作点的时钟（gb 尺子） | B0 训练中 attention 跑在 1,250–1,690 MHz（空闲 2,356），整板约 2.13 kW。B0 拟合的 1350/2350 MHz 时间比：FlyDSL fwd r6 1.25（r²=0.28）；ASM fwd 0.97（r²=0.01），基本不随 sclk 变。A0-新固件 10-02（真实数据）三种尺子的比值（blk / gb / e2e）：fwd r16/ASM 1.081 / 1.349 / 1.263；bwd s6/ASM 0.972 / 1.038 / 0.974；bwd r29/ASM 1.206 / 1.341 / 1.290；s6/r29 0.806 / 0.774 / 0.754 | A0-新固件上，blk 尺子的调用窗口约 1.45–1.5 GHz；gb 尺子按"GEMM 刚跑完"的工作点设计，调用窗口约 1.26–1.39 GHz（realab 记约 1.28 GHz）；A0 e2e 整步 sclk 中位数约 1.50 GHz。A0 上没测过训练中 attention 窗口的时钟，1,250–1,690 MHz 是 B0 的读数。在功耗墙下，耗时还随数据变化：A0 r24 全零操作数时间少 34%，cycle 数反而多 7.7% | 设计了 gb 尺子：每次计时前跑 10 个镜像库 bf16 GEMM 突发，proxy/prod 按 gb 计分。尚未安装，需要用户批准。另用 cycle（GRBM_GUI_ACTIVE/8）作第二把尺子 | 4 项比值里 3 项（s6/r29、r29/ASM、r16/ASM）是 gb 更接近 e2e；s6/ASM 是例外，gb 偏悲观约 6%（训练中 s6 比 ASM 快 2.6%）。单一 op 尺子代表不了训练 | `1002__oe/RULER.md` §0–§1；`1002__e2e/RESULT-realab.md`；`1002__e2e/RESULT-e2e.md`；`0927__b0/profile/REPORT.md` §0 §2.1；`0930__roofline/REPORT.md` |
| e2e 里 ASM 比两把 op 尺子测的都快（未解释） | A0-新固件 10-02：e2e 中 ASM 每层 fwd 1.07–1.11 ms、bwd 5.11–5.29 ms，比 blk 和 gb 两把尺子的读数都短（realab 6 层真实数据：fwd blk 1.209 / gb 1.137 ms，bwd blk 5.475 / gb 5.530 ms）。B0 09-28 有类似迹象：ASM fwd 不随 sclk 变，却随条件变（iso 1.22、blk 1.45、eburst 1.13、layer 1.00 ms） | 时钟解释不了。一种可能：训练中 q/k/v 刚由前面的 GEMM/RoPE 写出，还热在 MALL 里，而两把尺子每次调用前都会冲掉缓存。没测过 | 无；RULER §8.1 列为未决 | op 尺子和训练之间除了时钟还有缓存热度这个变量，ASM 的绝对 ms 不能直接外推到 e2e | `1002__oe/RULER.md` §1 末条、§8.1；`1002__e2e/RESULT-realab.md`；`0927__b0/profile/REPORT.md` §2.1 |
| fast 形状中位数噪声导致假接受 | A0-新固件 bwd r24（09-30）被判为 1.4263×（score 1.32077 对 0.91238），其中 fast 2.275×，prod 只有 0.9986。09-17 A0 上，两份相同目录在 fast 迭代 20 次时中位数差 7.8% | fast 每次约 55 µs，受 launch 开销主导，中位数被调度噪声左右：两个臂跑的是同一个 k_dq_sp，min 只差 0.05%，median 却差 56%。验收 gain 是三个 shape 的等权算术平均：(2.275+1.0051+0.9986)/3=1.426（几何平均只有约 1.317）。`0930__bwd/PLAN.md` D7、bwd hint h83 和 s:69864fc9 都写成"几何平均"，写错了；框架里只有 target_met 用几何平均（见 d.5） | D7：fast 只取 min，gain_weights 设为 prod 1 / proxy 0.25 / fast 0（4d61867f，h83）；迭代次数 20 → 51/101 | 新冠军在 prod 上与 s6 等价，没有造成损失。h83 因为缺索引行，到 10-02 才真正生效（见 d.5） | 4d61867f；20baa45e（commit message：gain 为三 shape 算术平均）；OE bwd job `rounds/024/3-act/act.yaml`、`job_context/state.yaml`（round 24 gain 1.4263）；s:69864fc9 09-30T13:41Z |
| 噪声地板 | 曾把 1.57% 当噪声地板。同一份代码跨 session 从 382.73 跳到 404.06（+5.6%）。B0 e2e 进程中途整体掉档约 5%（2,015 → 1,918 tps）。A0-限频 e2e 呈三模态，sd 3.91% | 1.57% 是 HipKittens bf16 GEMM 阶梯的地板。本算子同 session 的地板是 0.24–0.66%，fwd 跨 session 漂移约 1.5%。e2e 三模态来自 hipBLASLt NN 的 GridBased 选解，nkfix 后 sd 0.42%。B0 掉档原因未查 | 只用同 session 回文比值；min_gain 设 0.007。e2e 只用同进程 ABBA/BAAB 配对，剔除 step 1–7 和 profile 步 | 多轮判定要回头重看 | skill:`env-and-pitfalls` §3；`0923__flydsl/hint.md` h38-CORRECTION/h56；`0927__b0/e2e/RESULT.md` §2 |
| 同 session 回文 A/B 的槽位偏差与 beat 异常 | A0-限频 fwd r3：同一个 A 臂的 proxy 只因槽位不同就读到 858/759/718（单 session 虚涨 15%）。bwd r8 那个 session 里，beat 在 fast 只有 21.83（正常约 50.70），proxy 461.57（正常 573.31），score 因此从 0.42321 跳到 0.60244 | 5 个臂回文排列时，fast/proxy 随槽位有 ±8% 的变化；beat 在小形状上偶尔异常 | 臂的顺序轮转 3 个 session 后取均值；排名只看 prod 和同 session 的 gain；score 不跨 session 画图 | 这次 score 跳升不是真进步 | fwd job `rounds/003/1-opt/act.yaml`；`0922__flydsl/ROUND3-CLOSEOUT.md` Round 8 |
| JIT/配置缓存导致"改了没生效" | B0 r16 refactor 只改了模块常量 `SPEC_STALE_MAX`，`/tmp/flycache` 仍返回 r13 的二进制，逐字节相同的 A/A 副本差了 6.6%。09-23 n_block 扫描三个臂的时间和 SQNR 完全一样，构建耗时 0.0 s。09-11/13 aiter bwd 配置扫描结果平坦（27.45–27.49 ms） | FlyDSL 0.3.4.1 的 `_jit_function_cache_key` 只哈希函数源码和闭包标量（h46/h72）。n_block 是 def 时绑定的 kwdefault，又被 `functools.cache` 缓存。aiter 的 `_get_config` 带 lru_cache，wrapper 又用 `from … import` 绑定了旧引用 | 每进程、每 arm 用新的 `FLYDSL_RUNTIME_CACHE_DIR`，或清掉 `/root/.flydsl` 和 `~/.cache/comgr`。实验开关写成参数。在 launch 处验证：看 kernel 名、`--pmc` 描述符里的 VGPR、构建 <0.5 s 就报 SUSPICIOUS。扫描结果平坦就说明覆盖没生效 | 受影响的结果整批丢弃。陈旧缓存会返回旧的、正确的 SQNR，是最危险的假阴性 | 6c798efd；`0923__flydsl/STAGE2-FWD-SWEEP.md:24-38`；`0913__opt_plan__claude/phase2/DECISIONS.md` D7 |
| 时钟、功耗"已排除"的结论被撤回；限频代价的几种口径被混用 | 09-24 宣布 sclk 一直是 1100 MHz、853 W，排除了限频。09-13 估限频代价为 1.65×，09-14 又改称"实测约 1.93×，1.65× 作废"。09-15 的 JIRA-TRACE-ANALYSIS（后来进了 `PROGRESS-REPORT-0917.html` 和 `0922_summary/ASM-ATTENTION.md`）说 A0 与 B0 的 attention 只差约 13%（A0/B0：fwd 1.118、bwd 1.150、total 1.146），并称两边是"同一个 ASM 反向实现" | pw.sh 传了 `--arms cur_a`，benchmark.py 以 exit 2 退出，输出进了 /dev/null，wait 却返回 0，那 60 行其实是空载数据。真实 264 个样本显示 prod 窗口 998–1029 MHz，1100 只是空闲时钟。1.65× 是按 MAX_CLK 之比做的估算，不是实测。1.93× 是 B0 五行同版本对照（total）1.88–2.01×（中位 1.93×），即 A0-VR→B0 的跨卡比（B0 自身也有 VR 告警，代价约 9%），不是 A0 自身的限频代价；A0 自身刷固件前后同代码实测为 op 1.24–1.67×、e2e 1.44×（e2e 另有附加差异，见附录 A.2）。所以 1.65× 不算作废，只是估算。"13%"有两处错：A0 的 bwd 10.160 是被污染的 shim 数（见本表第 1 行）；B0 的 8.835 是融合 Triton bwd（09-14 B0 还没有 ASM bwd，HANDOFF 把它列为待办 T2，ledger 记录也都标 vendored fused backward）。fwd 1.118×（两边都是干净的 ASM fwd）成立；同代码的 bwd（两边都是融合 Triton bwd）由两份 RESULTS 推算为 17.735/8.835≈2.01× | 测量脚本要断言被测对象真的跑了（逐 shape echo RC）。每次计时附 sclk 见证。不做跨机换算。hwmon sclk 是 10–14 ms 平均值，看不到 kernel 内部时钟，改用 PMC 的 GRBM_GUI_ACTIVE/8 | 一条假结论进过 facts.md；"13%"和"A0 11.726 ms 在健康卡上约 10.2 ms"一起进过 0917 进度报告和 0922 总结 | `0924__flydsl/DAY-SUMMARY.md`（RETRACTED 段）；`0915__opt/JIRA-TRACE-ANALYSIS.md` §五；`0922_summary/ASM-ATTENTION.md` 页首；skill:`baselines.md` §8；`0914__campaign/HANDOFF.md:246`；`0914__campaign/ledgers/*.jsonl`；`0914__repro__c07/RESULTS.md:126-129`；`0915__repro__c07/RESULTS.md:17`；`0930__roofline/REPORT.md` |
| e2e 测到的不是想测的东西 | 09-13 turbo 245 tps 对 flex 244 tps。09-14 turbo_asm 13,970 对 turbo_noasm 13,930，并据此得出"turbo 比 flex 快 1.459×"。实际上 ASM fwd 在训练中从未被调用 | `converters: []` 让 turbo attention 根本没进 e2e，两边其实都是 flex。镜像里 editable 安装的 primus_turbo 通过 `.pth` MetaPathFinder 压过了 PYTHONPATH。`AITER_LOG_LEVEL=ERROR` 把横幅压掉了，"日志里没有 aiter"这个判据本身无效 | 容器内 `pip uninstall -y primus_turbo` 并打印 `__file__`；打开 converter；让门在入口处写 trace 文件 | 09-13/14 所有 turbo attention 的 e2e 结论作废（13,204 tps、1.459×、"38% 杠杆"） | `0914__campaign/RESULTS.md:277-318`；`0914__campaign/HANDOFF.md` §4 |
| e2e 统计方法 | "ASM bwd 让训练慢 6.78%（n=1）"、"快 3.42%"、"OFF 很稳定，0.06%"先后被推翻。MFU 显示 178–249% | 把单次运行内的步间抖动当成了跨运行方差；没有固定随机种子；三模态。torchtitan 的 `get_peak_flops` 对 gfx1250 兜底成 A100 的 312 TF/s | 设 `debug.seed 1234`；用 `bin/ab_replicate.sh` 交替跑 n≥9，再用 `ab_summary.py` 汇总；步数 20 → 10（step 4–8 的中位数差 ≤0.22%）；只引用 tps/tflops | 所有 MFU 列都不可引用 | `0915__opt/E2E-AB.md:180-260`；`0915__opt/GEMM-WGRAD-FINDING.md:78-94` |
| 卡上争用 | B0 09-14 四卡满载时，单卡计时漂移最高 50%（11.2–17.1 ms），还出现了假 SQNR 失败。B0 09-27 邻卡跑持续 GEMM 时，被测卡 attention 慢 3–11×，fwd/bwd 的比值甚至反转。A0 r23 遗留的 meas2 与框架的 validation 同时占卡约 2 min | 整板共享功耗和温度，同一 hive；孤儿进程 | 用独占窗口加交替测量；禁止在任何卡上跑持续 GEMM 压测；`rocm-smi --showpids` 和 KFD 持有者必须能一一说出来；放弃一个 session 时同步按 PID kill | — | `0914__campaign/RESULTS.md:84-97,217-233`；`0927__b0/interference.md`；6029b430 |
| 测量驱动/harness 自身的 bug | "turbo 仅配置调优"记成 25.695 ms（实际 18.916，错 1.36×）。flex 锚点测成 37.2 ms（应为 15.31，错 2.43×）。waves_per_eu=0 被判为无效旋钮。harness 测到的是镜像里的 Primus-Turbo。ledger 的 impl_note 是硬编码。clock_probe 9 s 的窗口变成约 55 min 的卡时。前向有 0.9 ms 差距 | `bash -lc "… $*"` 遇到分号截断了 `--tune`。flex 走 inductor 默认启发式，选了 num_warps=8。harness 把 0 当非法值拒掉。子进程的 sys.path 不对。impl_note 在运行前就写死了。enqueue 没有背压（多发 372×）。那 0.9 ms 里约 0.7 ms 是 host 侧开销 | 改成 `bash -lc 'cd "$0" && exec … "$@"'` 并回显实际收到的配置；flex 用 max-autotune；加 `_ZERO_MEANS_UNSET`（满频下 +4.2%）；ledger 记录 turbo_path；用 SQNR 指纹审计（ASM dk 50.61 / dv 50.83，融合 bwd 52.31 / 52.71）；加 Throttle 包装；两种测法结果不一致时，视为差距未确立 | 多行阶梯数据返工 | `0914__repro__c07/RESULTS.md:81-113`；`0915__opt/LEDGER-AUDIT.md`；OE:`output/0911__fa_gfx1250_phase1/E1.md` §3 |
| FLOP 口径不统一 | 同一个 2.356 ms 有时读成 947.8 TF/s，有时读成 933.5。早期 TF/s 用的是 fwd+bwd 合计 7.697e12。bwd 有"7-GEMM 实发"和"5-GEMM 名义"两种算法 | 旧 harness 的 fwd FLOP 约 2.233e12；`op_flops.py` 用 fwd 2.199292e12、bwd 5.498229e12 | 统一用 `op_flops.py`；引用时同时给 ms；issued TF/s 和算法 TF/s 不混在一起比 | — | `0925__flydsl/fwd-job/NOTES.md:299-304`；skill:`baselines.md` §1 |

---

### d.4 工具链问题

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| FlyDSL 三个版本并存 | 镜像自带 0.2.4（Primus-Turbo main 固定这个版本，用于 gfx950 树）；0.3.2 是 aiter 固定的版本（`~/.local/flydsl032`，bwd job 用）；0.3.4.1 在 `~/.local/flydsl0341`（fwd job 和 e2e 用）。0.3.2 在 PYTHONPATH 上时，`import primus_turbo` 报 `cannot import name 'buffer_ops'`。B0 smoke 报 `AssertionError: flydsl 0.2.4 @ /opt/venv` | 0.3.x 删掉了 `flydsl.expr.buffer_ops`。Primus 的 base_env 把镜像 site-packages 排在前面，而 `_ensure_flydsl0341()` 看到路径已在 sys.path 里就不再前移。用 pip --target 装 matplotlib 时，numpy 2.5.3 覆盖了 torch 依赖的 2.4.1 | `_env.py` 先 `sys.path.insert(0, dir)` 再 import，并断言版本和 `__file__`；job 进程里永远不 import primus_turbo。0.3.2 与 0.3.4.1 在同 session A/B 下性能持平：fly/asm 分别为 0.657–0.664 和 0.664–0.666（A0-限频）；r29/s6 在两个版本下 ISA 逐字节相同 | Stage 4 的派发接线必须重新设计。cherry-pick 70607aa9（兼容补丁）出现 13 个文件冲突，已中止 | skill:`env-and-pitfalls` §1a；`0917__flydsl/STAGE1-FWD.md` §4；`0925__flydsl/fwd341/ab_prod.log` |
| 0.2.4 能力不足（产品移植） | 09-17 "0.2.4 加 4 个 shim 即可"的结论被推翻，报错 `state variable 'result' is list`。09-30 产品 fwd 在 0.2.4 上，toy 和中等 shape 都出现 memory access fault（进程级） | 0.2.4 的 ast_rewriter 不接受 list 作为 stateful-if 的状态变量。0.2.4 缺 `fx.ceildiv`、`to_llvm_ptr`、max/min、`shuffle_xor`、`global_store_async_from_lds_b128`。手写 intrinsic 时，0.2.4 的 LLVM 把 LDS 操作数编成了全局地址，fault 地址就是 LDS 偏移。0.2.4 的 rocm.py:61 对 gfx1250 传了 wave64。COMPILE_ONLY 下 0.3.4.1 的 `flyc.compile` 会返回 None，0.2.4 没有这个提前返回 | 加兼容层 `_flydsl_compat/common.py`。dynamic-if 的状态改成具名变量。用特性检测 `HAS_ASYNC_LDS_STORE`：没有该 op 时改用 buffer_store 版 O writer v1，flydsl ≥0.3.4 时自动回到 v3。ISA 门断言 wave32、0 spill | fwd 慢约 1.3%（0.3.4.1 下 v3 1.279 ms，v1 1.296 ms，A0-新固件），bwd 慢约 1.8% | `0917__flydsl/API-DELTA.md`（RETRACTION）；`0930__port/PR_BODY.md:45-53`；s:fdc2534d 09-30T03:57–04:25Z |
| FlyDSL API 稳定性 | 维护者指出 Primus-Turbo 用了内部 API。审计发现：私有 `_raw/_ir` 约 90 处，llvm load/store，raw fastmath，对 `exe._cf` 的私有写入，已废弃的 shuffle_xor | 09-25 的清理只做在副本 op_clean 上，冠军是从未清理的 op0341 演进来的（fwd 差 181 行，bwd 差 590 行）。v0.2.4 本身没有 api_stability 策略 | 按 0.3.4.1 的策略重新审计（`API_AUDIT_SUMMARY.md`）。T1 已落地（1f74e662、58d5d798），两个版本下 ISA 都不变。T2 要等升级到 ≥0.3.4。T3（WMMA、TDM、ds_load_tr16、s_wait_dscnt、sched_barrier）没有稳定替代，标为 UNSTABLE(gfx1250)。`global_store_async_from_lds` 的 wrapper 在两个版本里都有 mask/cpol 参数顺序 bug | 在 gfx1250 上，FlyDSL 的部分功能只能依赖不稳定 API | `0930__port/API_AUDIT_SUMMARY.md` |
| gfx1250 上的 API 陷阱 | 能编译，但运行时静默出错或挂起 | `is_rdna_arch("gfx1250")` 返回 False：0.2.4 会因此误判成 wave64；0.3.x 中它只影响 V# flags，导致缺 OOB_SELECT，buffer descriptor 成了 CDNA 形态 0x00027000。`rocdl.s_waitcnt` 在 gfx1250 上会抛错（计数器是分开的）。`BufferAtomicAdd` 生成 SCOPE_CU，8 个 XCD 之间会静默丢更新。V# 的 num_records 以 128 B 为单位，int32 计算在 prod 形状下会回绕成 0。TDM 不加 pad 会有 64 路 bank 冲突。`make_tdm_atom(num_warps=8)` 用在 4-wave 下会静默漏载。aiter fwd 的 buffer manager 在 `num_waves != 8` 时抛 NotImplementedError。`fastmath=fast` 让 LLVM 删掉了因果 −inf mask。Python 的 `if` 被改写成 `scf.if` 分支函数后出现 NameError（需要 `const_expr`）。语句级 `for range` 里不能往 list append。`exec()` 会破坏源码内省 | 用 `UniversalAtomicAdd(Float32, SyncScope.Agent)` 得到 SCOPE_DEV；用裸 `fx.barrier()` 或 `rocdl.s_wait_dscnt`；用显式谓词和钳位，不依赖 descriptor 截断；extent 用 i64 计算；用 aiter 的 `tdm_ops_gfx1250`，并显式传 pad；把 `num_waves!=8` 记为路障，而不是记成"4-wave 更慢" | 造成多轮 build failure；h41 误判 atomic scope，由 4e02d3ae 更正 | skill:`env-and-pitfalls` §7；skill:`flydsl-api.md`；`0921__flydsl/A0-A1-ZERO-GPU-COMPILE.md` §3 |
| COMPILE_ONLY 的 arch 变量 | 只设一个 arch 变量时，会静默生成 target 混杂的 ISA。没有设备也没有 env 时，`get_rocm_arch()` 回落到 gfx942。fwd r5 漏设 `FLYDSL_DUMP_IR=1`，grep 改读 stdin，600 s 超时 | `ARCH` 决定编译后端，`FLYDSL_GPU_ARCH` 决定 buffer descriptor 的写法 | `COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 HIP_VISIBLE_DEVICES=-1 FLYDSL_RUNTIME_CACHE_DIR=…`，并断言结果是 gfx1250；从 `*_final_isa.s` 读 vgpr、spill、LDS。bwd 每次 build 约 4 min | `rounds/002/_scratch/screen.py` 写死了三个 kernel 签名和 block=(32,1,1)，新 kernel 会被静默漏检 | b1572877；skill:`env-and-pitfalls` §6 |
| profiler 能用到什么程度 | rocprofv3 PC sampling：09-11 两次 fault、一次挂卡；09-24 r17 在 prod 上 fault；刷固件后 1.3.2 启动时直接拒绝。`--kernel-trace/--runtime-trace/--hip-trace` 拿到 0 条 dispatch，刷固件前后都一样。`--pmc` 能用，但 51 个计数器只有 9 个可信，`SQ_VALU_WMMA_FLOP_*` 恒为 0，没有 stall 和字节计数器。VGPR 列是 ISA 实际值的一半（ASM 显示 512，实际 1024）。ATT 在旧固件下抓不到 FlyDSL JIT，09-29 刷固件后能给出逐指令 Hitcount/Latency/Stall（FlyDSL 和 ASM .co 都行，但 csv 的 Latency 列会把共发射指令重复计入）。A0 上 kineto 每个 profile 步只记录到 4–15 个 kernel。`--stats` 在 teardown 时报 `corrupted double-linked list`。`--pmc` 配 1 s 不同步的 warmup 会积压几万次 dispatch，看起来像卡死。rocm-smi 在空闲卡上显示 13% busy。hwmon `power1_input` 在负载下偏低约 800 W。llvm-objdump 解不出 ASM 里的 TDM 指令（显示为 `.long 0xd031…`） | 工具链对 gfx1250 的支持不完整，再加上旧固件的限制 | 计时用 CUDA event，kernel 名和时间戳用 PMC csv；VGPR 从 ISA 读；按 ATT 配方采集（`--att-library-path …/_rocm_sdk_devel/lib --att-target-cu 1 --kernel-include-regex`）；`--warmup-seconds 0`；kernel 数 <1500 的 trace 判为 INVALID；功耗和 sclk 改为 100 Hz 读 raw gpu_metrics | PC sampling 永久禁用；静态 ISA 指标只能当 build gate，不能用来排序（d.5） | skill:`env-and-pitfalls` §2c；`0930__bwd/probe/P1-RESULTS.md`；`0930__roofline/REPORT.md:158-160`；`1002__e2e/RESULT-e2e.md` |
| Triton、inductor、torch 的问题 | Triton `num_warps=16` 会挂起，日志 0 字节。inductor autotune 在本卡上报 `hipErrorLaunchFailure`。融合 bwd 在 torch.compile 下先后撞上 `num_ctas` kwarg 被拒、inductor 生成的 f-string SyntaxError。upstream main 在 torch 2.11 下 import 失败（`register_opaque_type`）。8B flavor 关掉 converter 后，SDPA MATH 物化 [4,32,8192,8192]，导致 SIGBUS | 编译器和运行时对 gfx1250 支持不足；AOTAutograd 编译前向时会顺带追踪反向 | 网格里排除 num_warps=16；`compile.enable: false`；ad67a2cc 改为 num_ctas=1 时不传该参数；从分支拷 `low_precision.py` 过来；用 `8B_flex` 并打开 converter | 51 个测试都不开 compile，e2e 又因 `converters: []` 走不到这条路径，两道门的盲区正好错开 | skill:`gfx1250-card-safety` §1；`0914__campaign/HANDOFF.md:86-101`；wt-llama31 README §1 |
| 其它路线的判定反复 | FlyDSL Gate A 在 09-14 判为 BLOCKED（gfx950 的 MFMA、ds_read_tr16_b64、permlane32_swap 在 gfx1250 上都报 `Cannot select`），09-16 撤回。09-13 曾判断"没有任何 FlyDSL 版本支持 gfx1250"，因为只看了 0.2.4。HipKittens udna1 门判 RED：256 个元素错 126 个，原因是 3 条编译期障碍 | 探针回答的是"gfx950 的 wave64 MFMA kernel 能不能直接 retarget"，不是"FlyDSL 能不能用于 gfx1250" | 改为逐 kernel 移植到 gfx1250 的 WMMA 和 `ds_load_tr16_b128` | 18–36 工程日的估算先被"确证"、后又撤回 | `0914__campaign/RESULTS.md:148-184`；`0914__campaign/HANDOFF.md:133-202` |

---

### d.5 op-evolve 框架问题

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| 精度门设得连 baseline 都过不了 | fwd 的 50 dB 门连未修改的 baseline 都过不了：7 个边界 case 在 49.82–49.99 dB。r1 longest-first 在 prod 上快 8.6%，因此被拒。另外 `op/ut/gates.py` 里硬编码了 `GATE_DB=50.0`。0923 版 fwd spec 的 `precision_gate` 被并进了 `refcache: >` 块标量，读出来是 None | 门槛没先拿 baseline 校准；同一个门在两处定义 | 在 resolved yaml 里改为 49，不 bump spec；`GATE_DB` 手改为 49.0；修好 spec | r1 被拒；r5–r10 在 state 里记成 passed=false，账面失真 | skill:`fwd.md:107,112`；skill:`op-evolve-ops` §10 |
| UT 门一直空转 | pytest 输出"no tests ran in 0.72s" | `seed=hash(shape)&0xFFFF` 受 PYTHONHASHSEED 加盐影响，每个进程都不同，缓存永远命中不了；文件里也没有 `test_` 函数 | 93ad3b89 改成 seed=0；r11 是 UT 第一次真正检查到东西的一轮 | — | 93ad3b89；`0923__flydsl/CAMPAIGN-FINAL.md` |
| 验收规则：算术平均 + min_gain 0 + 固定 band | A0-限频 bwd r15 g47 在 prod 上慢 3.1%（0.9688），却因 gain 1.0035（由 fast 撑起）被晋升为冠军。r5 账面 1.15×，其实是 fast 拉起来的，prod 只有 +1.6%。fwd r10 nodelay 在 prod 上 4/4 次 +0.83%，平均后只剩 1.0022，被拒。fwd r1 fast 0.934 低于 band 0.993，否决了 prod 的 +8.6% | acceptance 对三个 shape 等权平均；band = max(0.95, 1−min_gain)，对每个 shape 一视同仁；min_gain 不是 tune 能改的参数 | 20baa45e 手工回滚冠军；h7/h24 规定按 prod 排名、prod 低于本轮地板就不 ship；加 `evolve.gain_weights {prod 1, proxy 0.25, fast 0}` 和 `shape_band`（fast 0.90，proxy 0.98）；min_gain 设 0.007 | 改规则要手改 final.yaml 再普通 resume | 20baa45e；skill:`op-evolve-ops` §5；skill:`fwd.md:101-112` |
| 冠军记录混入未晋升的 arm | B0 fwd r9 BG 在三个 shape 上都胜过 r6，却得 0.9984 被拒。r19 在真实 dump 上 +1.5%，被 r18 一个未发布 arm 的记录挡住（0.9998）。bwd champions 里 fast/proxy=30 是被拒的 r30 的记录 | 每个 shape 的 best-ever 在被拒的轮次也会更新 | operator 用 refactor hint 手工晋升（h39 → r10，h44，h48/h68）；建议 champions 只记已晋升的轮次 | — | `0927__b0/OP-EVOLVE-SUGGESTIONS.md` #1 |
| target_met 用几何平均判定 | B0 bwd r27：fast 比 beat 快 1.66×（ASM 在小 shape 上慢），盖住了 prod/proxy 约 0.78× 的落后，geomean 1.007，job 自行结束 | 达标判据是三个 shape 的几何平均 | 经用户批准改为 h69：proxy 和 prod 各自 ≥ 同进程 beat，fast 只报告。A0 之后又设 `BEAT_MARGIN=20`（目标 1.20×），避免第一个达标的轮次就结束 job | — | `0927__b0/bwd-hint.md` h69；`0930__bwd/oejob/README.md` |
| hint 送不到 | fwd hint 用 L#/f# 编号，而 `core/hints.py` 只解析首格为 `h<N>` 的表格行，结果一条都没读到；hint.md 还是在 r1 开始之后才放进去的。deep 轮的 profiling/plan/act 根本不读 hint.md（ae6dd218）。deep_loop-trim 补丁（OE 58b2134）把 bwd 专用的 CAMPAIGN CORRECTIONS 写死在 3 个 preamble 里，所以 fwd 的 deep 轮（A0 r5/r10、B0 r15/r20）读到的都是 bwd 的内容。h83 只有段落、没有索引行，从 09-30 到 10-02 一直没生效。`core/route.py` 在整行里匹配类型，条件列里的 "must be predicated" 被当成了 must，fwd r1 因此报 RouteError | 解析器只认索引表；deep 模块不 import hints；preamble 被所有 job 共用 | 改成 id、type、title、status 四列的索引表；每个 hint 都必须有索引行，并用 `core/hints.py` 自检；route.py 只读 type 列；准备了 fwd 版 deep prompt 补丁和切换脚本 | 无法确认哪些轮次真的读到了哪些规则 | skill:`fwd.md:34-35,106-109`；s:69864fc9 10-02T11:41Z；`1002__oe/FWDJOB.md` |
| deep 轮"盲飞" | h4（09-22）以 rocprofv3 下出现 APERTURE_VIOLATION 为由，禁了 profiling 和 deep 轮；09-23 S0-c 查明那是 eager Tensile 造成的混淆。stock 的 01_select 强制用 `--kernel-trace`，拿到 0 行。r17 deep 轮的 PC sampling 在 prod 上 fault。deep act 只允许一个候选 | 禁令建立在被混淆的证据上；框架默认的 profiler 用法在 gfx1250 上无效 | af33a8fd 启用 deep 轮（ATT 已测）；82420bdf 禁 PC sampling；09-30 刷固件后把 ATT 接入 04_thread_trace；fast 轮改用 PMC csv（h80） | — | af33a8fd；82420bdf；`0930__bwd/REPORT.md:35` |
| `resume --config` 会重建一切 | `apply_spec` 会 bump spec_version，导致两个 setup 阶段重跑（约 58 min，op_setup 实测 2850 s），并重新生成 `op/{baseline,eager,ut,benchmark.py,validation.py}`，把 refcache 和手工补丁全部冲掉。min_gain 不是 tune 的选项；tune 不改 state.yaml。bwd 的 `validation.py` 硬编码了 `BEAT_MARGIN=0.0`，所以 `tune --beat-margin` 只改了文字，不改门 | 框架设计如此 | 只用 `op-evolve tune`，或者先备份 final.yaml（`.bak.<原因>`）再手改，然后普通 `resume` | 用这个办法，bwd 手改三次后 spec 仍是 v000 | skill:`op-evolve-ops` §4 |
| refcache 与在卡上算参考 | 闸门、benchmark、check_determinism、bitwise.py、test_correctness.py 共 5 处在卡上跑 fp32 Tensile 参考，这是 A 类挂卡的来源。B0 r24 起，`ut/common.py` 只改了一行 toy 形状，common_sha 就对不上了，于是 gate 每轮都打出"IGNORED … recomputing"，在卡上重算 prod 参考（正是 09-22 那次断电的触发路径）；A0 09-30 也一样。B0 fwd r20 act 自写的脚本也在卡上跑 fp32 参考，GPU2 因此 fault | 参考计算设计在 GPU 上；sha 不一致时静默回退到重算 | 376b9fd3、00128018、a5693e9c、93ad3b89 把所有调用点改为走 refcache（prod 参考在 CPU 上算，10m26s，1.1 GB）。B0 把 provenance 从 988c14ca 改为 d441e55a；A0 接受已记录的 sha。cache miss 直接判 FAIL。h50 禁止在卡上算 fp32 参考 | 09-21..22 那条路径占了当时墙钟的 82% | 376b9fd3；`0927__b0/bwd-hint.md` h69；`0930__bwd/oejob/README.md` |
| 进程管理与监管 | `supervise_job.sh` 里的 VENV 写死成别人的路径，找不到 bin 就 exit 127，然后无限重启。MAX_RESTARTS=60，backoff 只增不减。supervisor 只看进程是否死亡，不看挂卡和停滞。用 `kill <pid>` 杀 loop 会留下孤儿 agent，继续写 job_context。`op-evolve stop` 打断的 act 下次从头重跑（B0 bwd r25 因此跑了两遍）。stop 生效前框架已经建好了下一轮的空壳。09-25 resume 没用 setsid，日志放在 /tmp，10:10:51 重启后丢失。r23 遗留的 meas2 与 validation 并发占卡。09-24 `.stop` 晚了几十秒，下一轮 opt 已经启动 | 框架只有"进程死亡"这一个监控维度；手工操作习惯不对 | 用 `setsid nohup … resume`；停止用 `op-evolve stop` 或 `kill -TERM -<PGID>`；部署 card_ok.sh、patrol、release_guard；监控里用 `[o]p-evolve` 这种括号模式避免匹配到自己 | — | skill:`op-evolve-ops` §3 §11；`0917__flydsl/op-evolve/LAUNCHED.md` |
| 账本和状态的缺陷 | state.yaml 的 flops 键错位（A0 从 r18 起，B0 从 r20 起），progress.md 的 ms 列因此是错的，TF/s 本身没错。改了 shape 名之后 champions 分叉（A0 r20 同时存在 `prod:19` 和长名 `:20`）。refactor 会清空全部 champions。refactor 之后同一轮仍复用 `rounds/N/op`：fwd job 的 `rounds/016/op` 其实是被拒的 g61，r17–19 都把 g61 当成冠军重测。FastError：一个想法没写进 facts/dead_ends，整轮就被记成 failed（A0 r22、r26）。setup 不给 round 0 的速度 | 框架实现缺陷 | ms 一律用 FLOP ÷ TF/s 推导；act.yaml 里只用 fast/proxy/prod 三个名字（h35）；用 `--arms current` 测 incumbent（h51）；state_edit.py 做 rebase 前先核对 `rounds/<best>/op` 与 `op/current` 逐字节相同 | 根治要改框架，需要用户批准，这次没做 | `1002__oe/RULER.md` §3.3–3.4；`0923__flydsl/hint.md` h35 |
| 多卡调度与多棵代码树 | gpu_id 只写进 prompt，gate 的 docker exec 不带设备变量，结果总是跑在 device 0。preflight 要求整机空闲，power_wall 因此永远被跳过。09-14 同一进程里加载多棵 vendored primus_turbo 树：sys.modules 守卫被触发；opaque-type 注册表是进程全局的；两棵树都声明 `primus_turbo::attention_triton_forward_impl`，第二个 torch.library 片段析构时把 schema 从第一个的活句柄下抽走（use-after-free），表现为间歇性的内部断言、bad_alloc、SIGSEGV 139 | 框架默认单卡、单树 | B0 每卡一个容器（用 docker commit 做快照，只挂 /dev/kfd 和一个 renderD）；模块改用私有前缀 `_op_evolve_<n>.`，算子注册到 `primus_turbo_opevolve::`；每个臂一个进程 | use-after-free 吃掉了 4 次验证，约 2 h | `0927__b0/README.md`；`0927__b0/OP-EVOLVE-SUGGESTIONS.md` #4 #6；`0914__campaign/op-evolve-artifacts/…/dead_ends.md` |
| agent 基础设施与安全 | `.venv` 里的 claude-agent-sdk 0.2.152 自带的 CLI 不支持 claude-opus-5-5。B0 上没有 `~/.op_evolve_openai`，codex reviewer 返回 401，deep 轮失败。spec 里的 `runtime.env` 静默无效。review 10 min 后自动批准。opt agent 自己写探针，在 prod 上跑从未上过卡的 kernel，导致 10-02 挂卡 | 依赖版本、账号配置、框架默认值不合适；agent 不知道卡的安全规则 | 升级到 claude-agent-sdk ≥0.2.159；reviewer 改为 claude；环境变量写进 op 代码；h85 写进 hint 表和 deep 轮的 CAMPAIGN CORRECTIONS | agent 的权限要靠 hint 约束，没写成表格行就等于没写 | skill:`fwd.md:110`；`0927__b0/HANDOFF-A0.md` §3；1ee0dd59 |

---

### d.6 正确性与精度门

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| 需要四张量 SQNR 门 | 09-13（A0-限频）出现过一次 dk/dv = −inf，而 out 53.67、dq 52.24 dB 都正常（当时跑的是 `fwd:num_stages=2; bwd:num_warps=2,waves_per_eu=1`）。这是一次性事件：约 1/100 次，立即重复 3 次都通过，之后 2,300 次调用（进程内 1000+1000，150 个新进程共 300 次）都没复现，原文判为"不是任何配置的属性"，疑为瞬时硬件故障；但它说明只看 out/dq 的门会漏检。aiter 只改 `BLOCK_N1=256` 快 1.31×，但 dq 只有 9.59 dB。dkdv 的 BLOCK_M 改成 128 后结果"平滑地错"。num_block_m 没跟着 BLOCK_M 变时，dk/dv 尾部一直停在 zeros 初值。`sequence_parallel=False` 只算了 1/128 | 只看 out 的门看不到反向错误；grid 和切分参数之间相互耦合 | out/dq/dk/dv 各自 ≥50 dB；派发前断言 `BLOCK_N1==BLOCK_M2`、`BLOCK_M1==BLOCK_N2`（拒掉了 3 个配置）；输出预填 NaN，并检查 isfinite 覆盖率；fwd 门对 o 和 lse 分别要求（后来降到 49 dB，见 d.5） | 错的配置往往看起来更快 | `0913__opt_plan__claude/phase1/RESULTS.md` §4 §12（:89-95、:380-393）；`0913__opt_plan__claude/phase2/DECISIONS.md` D1 D4 D8 |
| 确定性测试测到了自己的尺子 | 09-13 默认配置跑 25 次，dk 出现 3 个不同的 SQNR，被报告为"dk 非确定" | 每次都重算 fp32 参考，而参考走的 rocBLAS 本身就不确定 | 改成与第 0 次结果逐位比较（`--determinism-reps`，每次 0.47 s，原来 30 s）；冠军和出厂各跑 1000 次，再各开 150 个新进程跑两轮，0 mismatch | 不要拿新导出的参考来测确定性 | `0913__opt_plan__claude/phase1/RESULTS.md` §4 §12 |
| 确定性门拆分：dk/dv 逐位，dq 用 SQNR | spec 写着"引入 split-k 或 atomic 即失败"，可 validation.py 只做 200 次逐位比对，没有检测 split-k 的手段，固定顺序的 split-k 从 r13 起每轮都在 ship。check_determinism 只跑 fast 一个 shape。aiter ASM 的 dq 本身就是 fp32 atomic 累加（514 条 `buffer_atomic_add_f32`），过不了我们自己的门；7-GEMM 对 5-GEMM 的结构差距（1.386–1.408×）因此在旧规则下无法缩小 | 规则的本意是"每个输出元素只写一次"，200 次逐位只是一个便宜的观测手段 | 09-25 c741fa8d：dk/dv 保持 200 次逐位一致；dq 改为同输入多次运行之间 SQNR ≥70 dB，用 aiter 实测的 dq（fast 113.0 / prod 98.0 dB）校准。冠军不改代码即可通过 | 遗留：确定性测试还要扩到 (q_split, BLOCK_KV) × {fast, prod} | c741fa8d；`0925__flydsl/PARITY-STRATEGY.md` R0；`0925__flydsl/AITER-5GEMM-STUDY.md` §5 |
| `bench_attention_turbo` 的参考值有 bug | 09-30 跑产品分支 bench：non-causal s8192 下所有 backend 的 dq 都只有 12.0–12.4 dB，判 FAIL（triton 那几行 out 60.6、dk 61.0；flydsl 那几行 out 48.8、dk 50.0） | benchmark 自带的参考值在这个形状上不准；用 CPU fp32 参考实测是 52 dB | 记进 memory，未修；PR 里说明这不是 kernel 错误 | 很容易误判为产品 kernel 出错 | `0930__port/runs/bench_turbo_triton.log`；`0930__port/runs/bench_turbo_flydsl.log`；s:fdc2534d 09-30T05:29Z |
| 毒化分配器（poison allocator） | 一个候选越界写冲掉了中断环，poison_allocator 却没抓到"没写过的元素" | 最大毒块只有 64 MiB，而 prod 的 dq 是 256 MiB，caching allocator 永远不会把毒块分给 dq，新页面全是 0。填充值是 fp32 NaN 0x7FC00000，按 bf16 读时有一半是 0.0。自己的修复里 0x7FC07FC0−2^32 又越出了 int32。另外，离线表的平均 SQNR 门会放过 NaN（`nan < 50` 为 False） | 改用 0x7FC07FC0（三种 dtype 下都是 NaN），通过 int32 视图写入，块大小从最大张量往下覆盖；放到 `op/poison_util.py`（686c9927），不改 ut/common.py，免得 refcache 失效；预填 NaN 并检查 isfinite 覆盖（720 次调用 0 个非有限值） | — | cba2abc2；686c9927；`0923__flydsl/poison-fix/README.md`；`0915__opt/NAN-FINDING.md:66-73` |
| 争用造成假 SQNR 失败 | B0 09-14 孤儿扫描进程抢卡时：llama31-8b-s4096 fused 的 out 43.23，gate-s2048 42.88，gate-s1024 asm 48.48（门限 50）。在独占窗口里重测是 53.74/53.87/53.93 | 共用一张卡 | 任何 SQNR 失败，都要先在独占窗口里复现一次才算数 | 差点误杀正确的配置 | `0914__campaign/RESULTS.md:217-233`；skill:`gfx1250-card-safety` §7 |
| SQNR 余量薄、区分度低 | bf16 梯度的 SQNR 上限约 52–53 dB，离 50 dB 门只有约 2.5 dB。ASM bwd 的 dk/dv 只有 50.2–51.0 dB（s4096 下 dk 50.56、dv 50.84），原因是 host 端把 g=4 个部分和先各自舍入成 bf16 再相加。armN 改了 fp32 累加顺序，SQNR 仍读 52.61/52.65/52.83，看不出区别。causal 被跳过的区域真值接近 0，off-by-one 或竞态可能照样过门 | SQNR 是全局平均量 | 以逐元素逐位比较为主：armAB 与 incumbent 的 134,217,728 个元素零差异。SQNR 只作辅助；用 SQNR 指纹审计实际跑的是哪个实现 | 门槛上调或换形状时，ASM 会最先撞线 | `0922_summary`（xlsx NOTES 4/12）；`0925__flydsl/AITER-5GEMM-STUDY.md` §2 |
| 用证明找越界 | 09-24 对 prod 的 134,217,728 个元组做全枚举，证明 k_dq 和 k_dq_sp 的读写都是到真实张量的双射。同时找到 3 个潜在缺陷：(1) impl.py:115 只断言 sq%32，launcher 向下取整而 kernel 向上取整，sq%64==32 时会越界写 262,144 B；(2) dqp 的 num_records 用 int32 计算，nsp≥8 时回绕成 0，所有 dQ 写入被丢弃；(3) k_dkdv 的预取没有钳位，最多读过末尾 982,272 B，只是被真实 extent 挡住了。B0 r19 在 9 个 UT 形状上发生 1,580 次越界预取（prod 上 256 次） | 地址算术改动没有配套的边界证明 | B0 用 h33 的三处修复得到 r19h：0 次越界，输出逐位相同，开销测不出来（1.0001）。w4f 的证明模型修正后（之前把部分等待误当成整批 TDM 已退休）重新证明。注意：不要把 k_dq 的描述符改成真实 extent，那会丢掉 75% 的 dQ | — | 41c916b7；`0924__flydsl/kdq-enumeration/SYNTHESIS.md`；`0927__b0/lab-bwd-r19/REPORT.md` §3–§4 |
| 产品移植的门 | 回归测试发现 gate 放行了 G=6/7，fwd 在构建时 assert；G=32 时 n_seq=0。对抗式 review 又确认了 3 处问题：apply 路径的 window 检查、autotune 缓存 key、COMPILE_ONLY 下的 launch memo | fwd 的 O store 要求 G 整除 32 且 ≤16 | gate 收紧到 G∈{1,2,4,8,16}，补上 gqa16 的精度测试和 G=7/32 的拒绝测试；3 处 review 问题修完后才提交 | — | s:fdc2534d 09-30T04:37–04:56Z |
| e2e 正确性：NaN 运行反而最快 | 74 次 e2e 里 8 次 loss NaN，每次都是所在组最快的：fly 49,487 对 46.5k（+6.4%），asm-on 44,294 对 42.5k，v3 38,713 对 37.7k，nkfix2-on 12,101 对 11,650 | 只按 tps 打分，从来不看 loss；算坏的计算往往更快 | `e2e.sh` 统计 nan_steps 并打印 DISCARD；先判正确性再看速度；一次 run 出问题就重新审查所有 run；后续 e2e 的 watchdog 遇到非有限值即退出（exit 96） | 已报出的 4 个数字被撤回（见 d.2） | `PROGRESS-REPORT-0917.html`；skill:`gfx1250-card-safety` §8；`1002__e2e/E2E-PLAN.md` |

ASM 自身的正确性问题，见 (b) 节：GQA 下 dk/dv 越界写（−0.94/−0.65 dB，绕法是 `dkdv_heads=q` 加 host 规约），`_perf.co` 的 dq 只有 5.84 dB，ASM fwd 的 LSE 布局与两 kernel 版 bwd 不兼容（混用会静默算出错梯度）。

---

### d.7 基础设施与协作

| 问题 | 现象 | 根因 | 解决方案 | 代价/教训 | 来源 |
|---|---|---|---|---|---|
| 断电截断 git 对象 | 09-17 约 07:46 主机重启（原因未记录）：dev/lhz/attn 的 HEAD 3412b2b 等 6 个 loose object 变成 0 字节，`git status` 报 `fatal: bad object HEAD`；SURVEY.md 被填到 16384 B，尾部 1969 个 NUL。09-22 夜间那次 AC：255 个 object 被截断，分支 ref 指向已损坏的 70efc407（无法恢复）；ROUND3-CLOSEOUT.md 也被截在 16384 B（尾部 2277 个 NUL），而且这个损坏状态还被提交过一次。当时有 4 个提交从未 push | 断电时 commit 正写到一半 | 先隔离 0 字节对象，再 `git update-ref <branch> <最后一个完好提交>`，然后 `rm .git/index && git reset`，最后 `git fsck --no-dangling`。09-23 07:36 起规定每轮结束必须 push（memory `push-after-every-round`）；每次 AC 后都查 fsck；op-evolve 仓库的改动另存成 patch | 丢了 2 个提交（内容还在工作区）和若干文件尾部。skill 里把第一次记成"2026-09-16 cycle"，按会话记录实际是 09-17 | s:c4b79aa6 09-17T08:02Z；s:a9a96fef 09-23T07:13–07:29Z；43eb5d89；f940fac6 |
| 推送通道 | 09-15 本机没有 GitHub 凭据（gh 未登录，remote 用 HTTPS）。09-17 13:25 的 `git push` 被 auto mode 分类器判为数据外泄，29 个提交在 4 天里只有本地一份 | 环境未配置；安全分类器拦截 | 09-15 把 remote 改为 SSH（`git@github.com`）。09-21 用户在 `~/.claude/settings.json` 里加了 Bash 权限规则，09:22 推送成功 | 4 天没有异地备份 | s:b596bddb 09-15T03:18Z；s:c4b79aa6 09-17T13:25Z、09-21T09:22Z |
| A0 与 B0 之间来回切换 | 时间线：09-10..13 在 A0（限频单卡）；09-14 去 B0（4 卡，GPU0 降级）；09-15..17 回 A0（B0 打包时做了 sha256 校验，落成分支 b0-20260914=3742d591 并快进，SSH 只通 B0→A0 方向）；09-17..25 在 A0，与同事分时用；09-27..28 在 B0（4 卡隔离，fwd job 从 A0 r4 克隆，bwd 从 r24 续跑，于是出现 B0 的 r24–r32，与 A0 09-30 之后的 r24–r27 重名）；09-28 12:45 交还 B0，留下冠军快照 5188c9da、HANDOFF-A0.md、备份 `0928__bak_b0`，B0 历史 squash 成 90508b9b 推送，约 40 MB 的 bulk 没推；此后都在 A0 | 两台机器交替可用 | 用 devsync 做 backup/restore（skill xmachine-sync）；跨机文件进 git；Primus 侧的配置在 Primus-Turbo 里另存一份 | 绝对数不能跨机比较，只能迁移同进程比值。Primus 侧配置漏备过一次。09-30 B0 的 ssh host key 变了（会话推测被重装过），按 D1 不绕过校验，于是认定 B0 bwd job 的 r24–r32 历史和真实 dump 都拿不到：r29 代码改从 git 里的 `0927__b0/champions/` 取，09-30 的 s1–s6 只用 randn 测。其实 B0 交还时打的备份包 `/home/lihuzhan/code/0928__bak_b0.tar.gz`（5.3 GB）09-28 14:05 起就在 A0 上（stat），里面有真实 dump 和两个 op-evolve job 的完整目录（含 B0 bwd 各轮）。10-02 07:52–07:55 从中解出真实 dump 到 `/home/lihuzhan/_prof_dump`（sha256 已核对），当天 realab 用它在 A0 上补测（见 d.3）；fwd job 也从这个包恢复。B0 bwd job 目录至今没有解包 | s:b596bddb 09-15T02:20–04:07Z；`0927__b0/HANDOFF-A0.md`；`0928__bak_b0/MANIFEST.md` §2–§4；`0930__bwd/PLAN.md:16,20`；`0930__bwd/REPORT.md:16`；s:69864fc9 10-02T07:51–07:55Z；`1002__oe/FWDJOB.md:11`；`1002__e2e/RESULT-realab.md:3` |
| 固件、驱动与限频 | 09-04 A0 负载下还能跑到 1,699–1,703 MHz。09-10 起出现 VR 限频，DPM 只剩 500/1100 MHz（当时 VBIOS 630A、SMU 125.7.1、dkms 7.1.1-2397345），重启也不消失，负载下约 1.0 GHz。09-28 19:39 有人换了驱动和固件包，09-29 卡无法初始化，AC 也没用。管理员随后刷 VBIOS 700E、SMU 125.12.0、dkms 7.1.0-2411946（主机约 10 次重启），之后 sclk 档位为 500/2356/2400，op 级 1.74–2.03 GHz，训练中约 1.5 GHz。10-01 dkms 又换成 7.1.0-2412954。开机仍会打印 VR 警告，但 DPM 表不再被截断 | 平台侧状态，我方无法控制 | 每次开机检查 pp_dpm_sclk、VBIOS、dkms；所有数字都标上时钟态；时钟变化的问题提交给平台方 | 09-29 之前 A0 的绝对数全部作废。刷固件后（同代码、同分块尺子）fwd FlyDSL r13ns 提速 1.47–1.50×（o1 2.023→1.353 ms 为 1.50×，o2 1.992→1.356 ms 为 1.47×，均值约 1.48×，即 REPORT §6.3 汇总写的 1.48×；教程 README §7 只取 o1，写作 1.50×）、ASM 1.24×（o2 1.23×）；bwd r29 1.67×、ASM 1.39×；e2e 1,947→1,351 ms（1.44×，不纯是平台差异：1,947 那个进程里 fwd 树 `_env.py` 改写过 LIBPATH，驱动也从 7.1.1-2397345 换成了 7.1.0-2412954，见附录 A.2）。09-28 得出的"u2n 在 A0 上退化"结论作废。ATT 恢复可用 | OE:`output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md` §1；`0928__a0_repro/REPORT.md` §2、§5.3、§6.1–§6.4；`0930__bwd/REPORT.md` §6；`1002__e2e/RESULT-e2e.md`（驱动）；wt-llama31 README §7 |
| 容器 | fa-repro（镜像 fa-tune:deps）标为 `owned: false`，不能重建，不能 pip install，只挂载 /home/lihuzhan。具体问题：镜像里 editable 安装的 primus_turbo 会遮蔽 checkout；容器的 /tmp 与宿主不通；sitecustomize.py 被追加过 BLAS 设置；容器里以 root 生成的文件和 core dump 在宿主上删不掉（每次 fault 约 1.1 GB core；09-13 仓库里出现过两个 8 GB 的 GPU core，e2e 还写出过 311 GB 的 coredump）；AC 后不会自动启动。B0 09-14 的 HIP 初始化要 58–172 s（推测 KFD 上下文创建被串行化）。B0 每卡一个容器 fa-g0..g3，09-27 15:43 被他人删除 | 共享镜像，不归我们管 | 改名或移走过期的 core 并关闭 coredump；重新生成产物前先确认旧的已删（必要时 chown）；AC 后用 docker start 复用容器；镜像库路径和补丁写进教程 | 有过磁盘被 core dump 撑满的风险 | skill:`env-and-pitfalls` §1；`0927__b0/README.md`；`0927__b0/STOPPED.md`；s:541e7bc3 09-13T05:25Z |
| 和同事共用卡与机器 | A0 只有一张卡，交接记录：09-17 13:20 交出（TERM 进程组、归档、停 docker）；09-21 发现他人容器 triton-7ff97e-20260910；09-22 16:10 交出；09-24 08:34 linxwang 的 jolly_easley 占着卡（用户授权以后直接停 linxwang 的容器）；09-25 14:00、09-27 15:21、09-30 14:34 分别交出；10-07 同事仍在用。B0：09-27 容器被删；09-28 有他人的 hipblaslt-bench 和容器（weihuan、andyye12）；邻卡 GEMM 会让被测卡慢 3–11× | 共享机器 | 交接流程：`op-evolve stop`、写当日总结、push、`docker stop fa-repro`；交接之后遇到挂卡、重启、容器被停，都不自动重启。KFD 持有者必须能一一说出来；每卡一把 flock 锁；B0 禁止跑持续 GEMM 压测 | 单卡时 fwd 和 bwd job 不能同时跑，op-evolve 和交互工作只能分时；本次总结全程不碰 GPU | skill:`env-and-pitfalls` §11；`0927__b0/LAB-RULES.md`；`0927__b0/REPORT-0928.html` |
| 宿主机硬件与资源 | A0 的 CPU 8 MC60 bank 持续报 corrected L3 MCE（09-25 那次开机报了 163 次）。09-28 02:15 和 08:13 两次开机都带有上一次的 BERT fatal L3 取指错误，说明是主机崩溃；10-01 20:41 也是同类重启。pcie_pl 可纠正 RAS 错误：09-30 共 68 行，10-02 先后 57 条和 58 条。根分区 99% 满（09-22 只剩 53 GB，09-23 有人清理 docker 后剩 828 GB）。`dmesg_restrict` 每次重启都会回到 1。amdgpu 被 blacklist | 主机 CPU 硬件问题，与 GPU 无关 | 报给机器管理员；dmesg 过滤分成 INFO 和 FAULT 两类；上卡前先 push | 主机崩溃同样会截断 git | `0928__a0_repro/REPORT.md` §5.3；`1002__e2e/E2E-PLAN.md` |
| 无人值守与编排 | 09-13 GPU 空转了 2 h 45 min，因为把"提交并汇报"当成了停止点；cron 只在 REPL 空闲时才触发，而 cron 的提示里还写着"GPU busy 就什么也不做"。用户多次要求"不要停"。并行 agent 默认会自己上卡。监控输出刷屏 | 编排设计不当 | 用 `forever_queue.sh` 加 supervisor；PROGRESS.md 作为唤醒入口；委派时明确禁止上卡；监控约 30 s 轮询一次，正常时不输出 | — | s:541e7bc3 09-13T12:29Z；skill:`env-and-pitfalls` §11；memory `e2e-monitoring-style` |

---

#### 仍未解决或待核实（d 节范围）

- B 类挂卡（INVALIDATE_TLBS 起头）和启动期 MES 故障族的直接原因都还不知道。也没有可用的前兆指标。
- nkfix 在 A0 上 21% 运行出 NaN 的根因没有找到。宿主 hipBLASLt 库与 NaN 和挂卡之间，除了 B0 p1a/p1b 这一对单变量对照（每边 n=1，B0 VERIFY 的质疑已由它回应）外，只有相关性证据，没在卡上反复证实。
- 09-21 16:01:54 那次挂卡，会话判为不可恢复，但 4547915c 说 dmesg 里没有任何 reset 签名。09-22 16:02 那次的首行签名没保存下来。这两次的类别都定不了。
- AC 计数中，09-11 的两次（是 AC 还是热重启）和 09-16 第 9 次（推断）不确定。09-13 和 10-02 的重启时间没有记录。
- 需要更正的文档：
  - `env-and-pitfalls` §1b 仍推荐宿主库，与 h85 和教程冲突。
  - `gfx1250-card-safety` 写"09-16 一天五次 AC"，实际是 9 次。
  - 同一份 skill 写"09-16 cycle 截断 6 个 object"，实际是 09-17。
  - skill:`gfx1250-attn-campaign/SKILL.md:106` 和 `0930__bwd/PLAN.md` D2 写 PC sampling"3/3 挂卡"；原始记录是 3 次都 fault、1 次挂卡（`env-and-pitfalls` §2c 的"3 attempts 3 faults"是准确的）。
  - `0930__bwd/PLAN.md` D7 和 bwd hint h83 说 A0 r24 是被"几何平均"接受的，实际是算术平均（20baa45e）。
  - `0930__bwd/PLAN.md` P4/D1 说 B0 bwd job 的 r24–r32 历史拿不到，`0930__bwd/REPORT.md:16` 说真实 dump 只在 B0；其实两者 09-28 起都在 A0 的 `/home/lihuzhan/code/0928__bak_b0.tar.gz` 里（见 d.7）。
  - 1ee0dd59 提交信息、`1002__oe/incident/WEDGE-1002.md`、h85 正文（`0930__bwd/oejob/hint.md:4806`）和 `gfx1250-card-safety` §1 #13 把 10-02 r27 挂卡的两个变体记成 w4f 融合 kernel；按 `rounds/027/_scratch/arms` 源码，实为 s6 基底的 cluster multicast（A_g74 改 k_dkdv，B_g82 改 k_dqg），见 d.1.1。
- gb 尺子已经设计好但没有安装（需要用户批准；可选先花约 6 min 卡时校准 burst 长度）。s6 相对 ASM 的幅度已在 10-02 实测（A0-新固件、真实数据）：blk 0.972、gb 1.038、e2e 内 attention bwd 每步耗时比 0.974（p1：162.15 / 166.54 ms；这是 bwd 耗时比，不是单步比。bwd 每步比 ASM 快约 4 ms：相邻配对差 4.50 / 4.56 ms，即 `1002__e2e/RESULT-e2e.md` 正文的 4.5；两臂中位数相减 3.8–4.4 ms），gb 在这一项上偏悲观约 6%。
- 同一次 e2e 里，fly 的 attention 合计每步比 ASM 多约 4.3–5.2 ms（fwd 43.9 对 34.7，+9.2；bwd 约 −4），单步却快 0.7–2.0 ms（1,349.4 / 1,349.9 对 1,350.6）。两进程的 step − FA 都是 −6.3 ms，超出 E2E-PLAN §4 判定 2 的 5 ms 门限；s6 对 r29 同样如此（attention −53.0 ms，单步只快 37.0–39.1 ms）。差额来自 attention 以外的部分或噪声，没有拆解 [`1002__e2e/e2e/runs/analysis.1002_095504.txt`]。
- e2e 里 ASM 每层（fwd 1.07–1.11 ms、bwd 5.11–5.29 ms）比两把 op 尺子测的都短，时钟解释不了；"输入在 MALL 里是热的"这一假设没测（RULER §8.1）。
- L21 的真实增益两份文档说法不一：REPORT-0928.html 写 +0.3%，ruler/REPORT.md 的分块读数是 +0.6%。

## 附录 A 口径与可比性

### A.1 机器、时钟阶段与驱动/固件

四节对同一阶段的叫法不同：(a) 用 A0-VR / A0-RF / B0，(b) 用 E1–E5，(c) 用 A0-H / A0-T / B0-14 / B0-28 / A0-R / A0-D，(d) 用 A0-限频 / A0-新固件 / B0。对照如下。

| 阶段 | 各节记号 | 机器 | 日期（UTC） | VBIOS / SMU / 驱动 | 负载时钟 | 可比性与备注 |
|---|---|---|---|---|---|---|
| A0 未限频 | (c) A0-H | A0 `heliosr-1b114-c07-1`，独占 1× gfx1250（MI455X，256 CU，wave32，每 CU 320 KB LDS，432 GiB HBM，PPT0 功耗上限 2500 W） | 09-04 及之前 | 未单独记录 | 09-04 负载下 1699–1703 MHz（上限 2400） | 这一阶段没有本项目的 attention 测量 |
| A0 VR 限频 | (a) A0-VR；(b) E1（09-10..09-17）、E3（09-17..09-27）、E3′（09-28）；(c) A0-T；(d) A0-限频 | A0 | 09-10..09-28（最早的限频记录是 09-10 22:38） | VBIOS `113-M4500001-630A`；SMU fw 125.7.1；amdgpu 7.1.1.31300009（dkms 7.1.1-2397345）；内核 6.14.0-37-generic，cmdline 带 `modprobe.blacklist=amdgpu` | sclk DPM 只剩 500/1100 MHz；负载约 0.94–1.07 GHz（09-24 prod 计时窗口实测 998–1029 MHz） | 09-29 之后，这一阶段的绝对数全部作废，只能引用同进程比值 |
| B0 09-14 | (b) E2；(c) B0-14 | B0 `ctheliosp-1b112-a37-1`，4× gfx1250（GPU0 降级，当天只用 GPU1–3） | 09-14 | 未记录（与 A0 同镜像） | 2133–2244 MHz | 四卡同时负载时，单卡计时漂移最高 50% |
| B0 09-27/28 | (b) E4；(c) B0-28；(d) B0 | B0，每卡一个快照容器 fa-g0..g3；09-28 GPU1 挂死，交还时仍未恢复 | 09-27..09-28 | 未记录 | 空闲 2356 MHz；op 级（分块尺子）约 2.15 GHz；nkfix 后训练整步约 1.34–1.39 GHz（终版运行 1,342 / 1,371 MHz），训练中 attention 窗口 1.25–1.69 GHz（整板约 2.13 kW 功耗墙） | B0 09-27/28 的会话不在本机，数字只来自文档和账本 |
| A0 刷固件后 | (a) A0-RF；(b) E5；(c) A0-R；(d) A0-新固件 | A0 | 09-29..10-01 | VBIOS 700E；SMU 125.12.0；amdgpu-dkms 7.1.0-2411946 | sclk DPM 500/2356/2400；op 级 1.74–2.03 GHz；满载 WMMA 约 1.4–1.8 GHz（2500 W 功耗墙）；训练中约 1.5 GHz | 开机仍打印 VR 告警，但 DPM 不再截到 1100 MHz；ATT 可用 |
| A0 新驱动 | (b) E5（10-01 起）；(c) A0-D；(d) A0-新固件 | A0 | 10-01 起 | VBIOS 700E 不变；amdgpu-dkms 7.1.0-2412954 | 同上；GEMM 突发后（训练工作点）约 1.28 GHz；e2e 整步 sclk 中位约 1.50 GHz | randn 基线与 09-30 一致（bwd s6 为 ASM 的 104.1%，fwd 时间比 1.079） |

**软件栈**：容器 `fa-repro`（镜像 `fa-tune:deps`，即 `amdprimus/amdprimus:gfx1250-20260910` 加 torchtitan 依赖），torch `2.11.0+rocm7.14.0a20260625`，triton 3.6.0，HIP 7.14，python 3.12.3；aiter-src 09-13 为 @`ffa945f9`，09-17 起为 @`6963ae9d`；torchtitan v0.2.2（`73a0e6979`）。flydsl 有三个版本并存：
- 0.2.4：镜像自带，Primus-Turbo main 和产品分支固定用它；
- 0.3.2：aiter 固定的版本（`~/.local/flydsl032`），A0 bwd job 和手工 s1–s6 用它；
- 0.3.4.1：`~/.local/flydsl0341`，09-25 起 fwd job、B0、e2e 用它。
0.3.x 删掉了 `flydsl.expr.buffer_ops`，所以和 primus_turbo 不能在同一个进程里用。

### A.2 跨阶段只能比比值，而且比值本身也会变

| 对照 | 倍数 | 说明 | 出处 |
|---|---|---|---|
| A0-VR → B0，同一份 Triton 类代码 | 5 行 like-for-like 1.88–2.01×（中位 1.93×）；FINAL-TABLE 全 7 行 1.88–2.12×（中位 1.97×） | 两张卡之间的比值。B0 自身也有 VR 警告（代价约 9%），所以这不是 A0 自己的限频代价；09-14 文档把它写成"VR 限频代价实测约 1.93×，1.65× 作废"，这个说法不成立 | (a) 记号与口径 ①；(d) d.3；`0914__repro__c07/RESULTS.md:32-38、124-130`；`0914__repro__c07/FINAL-TABLE.md` |
| A0-VR → B0，ASM fwd | 1.118×（1.572 对 1.406 ms） | 两边都是干净的 ASM fwd，成立；ASM fwd 几乎不随 sclk 变 | (d) d.3；`0915__opt/JIRA-TRACE-ANALYSIS.md` §五 |
| A0-VR → B0，"ASM attention 合计" | 1.12–1.15×（bwd 1.150、total 1.146；A0 11.726 ms 对 B0 10.236 ms） | **不成立**：bwd 一项比的是 A0 的 shim 伪影 10.160 ms 和 B0 的融合 Triton bwd 8.835 ms（09-14 B0 还没有 ASM bwd）；同代码的融合 Triton bwd 约为 2.01×。三项里只有上一行 fwd 的 1.118× 成立 | `0915__opt/JIRA-TRACE-ANALYSIS.md` §五；(a) 记号与口径 ②；(b) b.4 末行；(d) d.3 |
| A0 刷固件前 → 后，同代码同尺子（op） | FlyDSL fwd r13ns 1.47–1.50×（o1 2.023→1.353 ms 为 1.50×，o2 1.992→1.356 ms 为 1.47×，均值约 1.48×）；ASM fwd 1.24×（o2 1.23×）；FlyDSL bwd r29 1.67×；ASM bwd 1.39× | 负载时钟比约 1.75×，实测提速都低于它；FlyDSL 对 sclk 更敏感。驱动也同时变了（7.1.1-2397345 → 7.1.0-2411946），不能与固件严格分开 | (a) a.1；(c) c.1；`0928__a0_repro/REPORT.md` §6.1–§6.3 |
| A0 刷固件前 → 后（e2e，ASM + nkfix） | 1,947 → 1,351 ms，1.44× | 教程称"同代码同方法"，两边也都是 `NKFIX_CHECK=1`，但这个倍数不纯是平台（固件/时钟）差异：① 1,947 ms 是 09-28 a0_p3a（ASM / r6+r20 / r16+r29 三臂按步交替，90 步）里 ASM 臂的中位数；该进程 step 2 起加载的 FlyDSL fwd 树 `_env.py` 把 `HIPBLASLT_TENSILE_LIBPATH` 改指到宿主库（hipBLASLt 会不会中途重读没有证实），同批另有 1 次 NaN、2 次挂卡，09-28 的 A0 绝对数已整批作废；② 两次之间驱动从 7.1.1-2397345（09-28）经刷固件时的 7.1.0-2411946，换成了 10-02 的 7.1.0-2412954；③ 10-02 用的是另写的套件 `1002__e2e/e2e/`（fwd 树副本 `fwd_r16_imglib` 删掉了 LIBPATH 改写，加 BLAS guard）。两边相同的是 nkfix_b0 + `NKFIX_CHECK=1`、32L 配置模板和三臂按步交替的做法 | (a) a.1；(d) d.2.1；`0928__a0_repro/REPORT.md` §2、§5.3、§6.3–§6.4；`1002__e2e/RESULT-e2e.md`、`E2E-PLAN.md` §1–§3；wt-llama31 README §7 |
| 同一把分块尺子、同一批 arm，跨时钟阶段 | r13ns/ASM 1.29（E3′）→ 1.08（E5）；r29/ASM 1.44 → 1.20；bwd r29 占 ASM 69.5% → 83.6% | 比值本身也不能跨时钟阶段迁移 | (b) 开头；(c) c.1 |
| 同机同代码，只换尺子 | B0 fwd 交错 → 分块：占 ASM 约 85% → 约 96% | kernel 没有变 | (c) c.1 |
| 同代码，只换输入和工作点（A0 10-02） | bwd s6/ASM blk 0.972 → gb 1.038；fwd r16/ASM blk 1.081 → gb 1.349 | FlyDSL 对时钟敏感，ASM 几乎不敏感 | (b) b.2；(c) c.1；(d) d.3 |
| 09-13 按 MAX_CLK 估算的限频代价 | 1.65× | 时钟比估算，不是实测（原文以 MAX_CLK 1100 MHz 对 09-04 负载 1699–1703 MHz 得出约 1.65，没给算式；按这两个数算约 1.55）。09-14 拿跨卡比 1.93× 判它"作废"不成立，本报告不再称作废；A0 自身的实测见上面两行（op 1.24–1.67×，e2e 1.44×） | (a) a.1；(d) d.3；`0913__opt_plan__claude/phase0/PLATFORM-ESCALATION.md` |

### A.3 FLOP 口径

- **现行口径**（op-evolve `tools/op_flops.py`）：每次调用 fwd 2.199292e12、bwd 5.498229e12（5-GEMM 名义计数，causal 取 (s+1)/(2s)），fwd+bwd 7.697e12；ms = FLOP ÷ TF/s。(b) 表中带 `*`、(c) 表中标 derived 的值都是这样换算的。
- **09-13..09-16 的文档**：只报 fwd+bwd 合计（7.697e12 ÷ 总耗时）。(a)(b) 表里的分方向 TF/s 是按单向耗时换算的。
- **7-GEMM 实发口径**（本报告不用）：树内 vendored 融合 bwd 发 7 趟 GEMM，issued 7.6975e12，是名义的 1.400×；我们的 FlyDSL bwd（r0 到 s6）issued 7.7395e12，是名义的 1.40764×（含对角块粒度 ×1.00546）；aiter ASM 只发名义的 1.0155×。所以 s6 的"占 ASM 103.8%"是在多做约 39% 矩阵运算的情况下达到的。0913/0914 文档曾按实发口径报 bwd（例如 B0 vendored 融合 bwd 748.4 对名义 534.6 TF/s）。
- **其它旧口径**：09-23 STAGE2-FWD-SWEEP 用 fwd ≈2.233e12（旧 harness，TF/s 偏高约 1.5%）；09-17 DKDV-FIRST-TIMING 的 5.8 TF/s 是单头的自定义 useful 口径，不可比；09-14 B0 的 Triton op-evolve job 只计 bwd。
- **GEMM**：TF/s = 2MNK ÷ 耗时。
- **e2e**：tokens/s 是单卡吞吐；TFLOP/s 取 torchtitan 自报值，或按 torchtitan 口径由 tps 换算（32 层约 57.9 GFLOP/token，8 层约 16.84）；torchtitan 的 MFU 以 A100 的 312 TF/s 为分母，全部不可用。
- **比值方向**：(b) 的"相对 ASM" = 本行耗时 ÷ ASM 耗时（>1 表示比 ASM 慢）；(c) 的"占 ASM" = ASM 耗时 ÷ 本行耗时（>100% 表示比 ASM 快）。两者互为倒数，引用时要注意。

### A.4 尺子

| 尺子 | 用在哪里 | 做法 | 已知偏差 |
|---|---|---|---|
| tune（`tools/gfx1250/tune_attention.py`） | 09-13..09-16 的 op 阶梯（(b) 的 E1/E2） | 一层 fwd+bwd；CUDA event 取中位；iters 20 / warmup 5；每个 rep 冲刷 256 MiB L2；out/dq/dk/dv 四张量 SQNR ≥50 dB | 计时含 autograd 管路；ASM bwd 走 autograd shim 时的读数（10.160 ms）是伪影 |
| bench0910 / bat | 09-10 Primus 侧 bench；09-30 产品分支的 `bench_attention_turbo.py` | 仓库自带 bench | 自带的正确性检查不可信：0910 版对所有后端都报同一个 max-abs；0930 版在非因果 s8192 上 dq 参考只有约 12 dB，CPU fp32 参考实为 52 dB |
| E2_bench（op-evolve phase-1） | 09-11 | warmup 5，iters 20，重复 3 次 | — |
| op-evolve benchmark.py，逐次交错 | A0 两个 job（09-17..09-27）；B0 fwd r5–r9；B0 bwd r24–r26 | 只计 kernel 发射；每个 arm 51 次取中位；回文交错；ASM beat 与候选同进程；三个形状 fast/proxy/prod，只按 prod 排名 | FlyDSL/ASM 比值偏低（fwd 约 25%，bwd 约 3.4%）；ASM 之后的 I-cache 惩罚（h28 起 beat 单独一个进程） |
| 分块（blocked，h40/h41/h66） | B0 fwd 从 r10 的 act 起，B0 bwd 从 r27 起；之后 A0 一直沿用 | 每个 arm 先跑 4 次不计时，再连续计时 9 次；块间回文；每个排名进程带一个 A/A 副本 | A/A 误差 fwd ±0.19%、bwd ±0.05%；A/A 偏离超过 0.5% 整个进程作废；新旧尺子的绝对 ms 不可比 |
| 真实数据（h45/h47） | B0 fwd r18 起；A0 10-02 的 realab | B0 训练 step 43 导出的 6 层真实 q/k/v（`/home/lihuzhan/_prof_dump/qkv_call0*.pt`） | 真实数据的 score 标准差为 21–53（randn 约为 1）；投机 softmax 在 13–25% 的 tile 步回退重算 |
| gb（训练工作点） | A0 10-02 的 realab；op-evolve 计分尚未安装 | 每次计时前先跑 10 个 bf16 32768×4096×14336 GEMM，把时钟压到训练工作点（A0 约 1.28 GHz） | 4 项比值里有 3 项比 blk 更接近 e2e；s6/ASM 是例外，gb 偏悲观约 6% |
| e2e | 09-13 起 | Llama-3.1-8B BF16 32 层（09-15/16 部分为 8 层），MBS=GBS=4，seq 8192，单卡，AC none，compile 关；09-16 起以 nkfix 为前提；同进程 ABBA/BAAB 或按步交替；剔除 step 1–7 和 profile 步 | `NKFIX_CHECK=1` 每步约多 33 ms；出现 NaN/inf 的运行整次作废；e2e 里 ASM 每层比两把 op 尺子测得都快（未解释） |

### A.5 噪声地板

- **op**：本算子同 session 的噪声地板为 0.24–0.66%（09-24 之前曾误用 HipKittens GEMM 阶梯的 1.57%）；fwd 跨 session 漂移约 1.5%；同一份 r7 代码在 AC 断电前后读到 382.73 → 404.06 TF/s（+5.6%）；分块 A/A 为 fwd ±0.19%、bwd ±0.05%；验收的 min_gain 为 0.007（bwd 早期为 0）。
- **fast 形状**：每次约 55 µs，受 launch 开销主导。跑同一个 kernel 的两个臂，min 只差 0.05%，median 可以差 56%，所以后来 fast 改按 min 计分，权重设为 0。
- **e2e**：A0-VR 上 nkfix 之前呈三模态，跨运行 sd 3.91%（簇间 1.8% / 6.8%）；nkfix 之后 sd 0.42%；B0 有一个进程中途整体掉档约 5%（2,015 → 1,918 tps，原因没查）；步数从 20 减到 10 时，step 4–8 的中位数差 ≤0.22%。
- **卡上争用**：B0 09-14 四卡满载时单卡计时漂移最高 50%；邻卡跑持续 GEMM 会让被测卡的 attention 慢 3–11×。

### A.6 各节之间的不一致：已统一与未定

第一版在这里列了 11 条四节之间互相矛盾的记录。本版把其中能核实的几条改进了正文，列在"已统一"下（若某处仍残留旧说法，以这里为准）；其余的要么还没核实，要么是口径不同、算不上矛盾，列在"未定或口径不同"下。

**已统一**

- **A0 09-13 e2e 的 torchtitan 版本**：v0.2.2（Primus `third_party/torchtitan` 固定在 `73a0e6979`）。`0914__campaign/RESULTS.md:65` 写的 "0.1.0" 是笔误。
- **"VR 限频代价实测约 1.93×"**：1.88–2.01×（中位 1.93×）是 A0-VR→B0 的跨卡比（B0 自身也有约 9% 的 VR 代价），不是 A0 自身的限频代价。A0 自身刷固件前后为 op 1.24–1.67×、e2e 1.44×（后者另有附加差异）。09-13 的 1.65× 是时钟比估算，不再称作废（见 A.2）。
- **"ASM attention 跨机只差 1.12–1.15×"**：只有 fwd 的 1.118×（1.572 对 1.406 ms）成立；bwd 一项比的是 shim 伪影 10.160 ms 对融合 Triton 8.835 ms，不成立（见 A.2、d.3）。
- **`c1325c7e` 提交信息里的 220.6 TF/s**：测量条件不明（提交信息没写 hkv、是否 causal、时钟），220.6 / 129.4 = 1.70× 和哪种时钟比相符都没有定论。不再说"是在未限频的卡上测的"。
- **AC 次数**：以 d.1.4 为准。A0 因挂卡人工断电/重启 27 次，算上 09-29 那次无效 AC 为 28 次；其中 09-15 五次、09-16 九次（UTC）。
- **刷固件后 FlyDSL fwd 的提速**：1.47–1.50×，o1 顺序 2.023→1.353 ms 为 1.50×，o2 顺序 1.992→1.356 ms 为 1.47×，均值约 1.48×（`0928__a0_repro/REPORT.md` §6.1）。

**未定或口径不同**

1. **09-14 B0 "turbo vs flex 1.459×" 的来源**：(b) b.3 的撤回说明写"差异来自其它 turbo patch（float8 / mx linear）"；(a) a.3 #2、#11 认为这个归因存疑（按本机 Primus 代码，`converters: []` 下看不出这两个 patch 怎么生效），能确定的只是它不来自 attention。未定。
2. **aiter Triton MHA 出厂读数（09-13 34.565 ms 对 09-15 28.035 ms）**：(b) b.2.1 引用 0915 的判断，认为 09-13 那一行偏差 +23%、不可信；(a) a.2 #12 认为更可能异常的是 09-15 那次（它的 fwd 3.725 ms 像是调优过的前向）。未复测，未定。
3. **B0 训练中的时钟**：(b) E4 写 1.25–1.45 GHz，(c) B0-28 写 1.34–1.39 GHz，(d) 写 1.25–1.69 GHz。前者偏向整步 sclk 均值，后者是训练中 attention 窗口的读数，口径不同，未逐条核对。
4. **A0-VR 的负载时钟**：各节分别写 0.94 / 0.95 / 0.97–1.07 GHz，另有单点采样 943 MHz。都指 VR 限频下的负载时钟，差异来自测量窗口不同。
5. **L21 的真实增益**：REPORT-0928.html 写 +0.3%，ruler/REPORT.md 的分块读数是 +0.6%（(d) 末尾已列）。未定。
6. **10-02 e2e 的 fwd/bwd 拆分（已统一写法，仍有一项未拆解）**：`1002__e2e/RESULT-e2e.md` 正文的"bwd 每步快 4.5 ms"是相邻配对差的中位数（两进程 −4.50 / −4.56 ms）；两臂各自中位数相减是 3.8–4.4 ms（166.54−162.15、166.41−162.62；RESULT-e2e.md 表四舍五入后为 3.9–4.3）。两者不矛盾，本报告统一写"约 4 ms"。仍未拆解的是：fly 的 attention 合计每步比 ASM 多约 4.3–5.2 ms，单步却快 0.7–2.0 ms，两进程的 step − FA 都是 −6.3 ms，超出 E2E-PLAN §4 判定 2 的 5 ms 门限；s6 对 r29 同样如此（attention −53.0 ms，单步只快 37.0–39.1 ms）。这部分差额落在 attention 以外或计时噪声里 [`1002__e2e/e2e/runs/analysis.1002_095504.txt`]。

## 附录 B 术语表

| 术语 | 含义 | 见 |
|---|---|---|
| A0 / B0 | A0 = `heliosr-1b114-c07-1`（本机，单卡 gfx1250）；B0 = `ctheliosp-1b112-a37-1`（4× gfx1250，同一 XGMI hive） | 附录 A.1 |
| 时钟阶段代号 | A0-VR / A0-T / A0-限频 / E1·E3·E3′ = A0 刷固件前；A0-RF / A0-R / A0-新固件 / E5 = 刷固件后；A0-D = 10-01 换驱动之后；E2 / B0-14 = B0 09-14；E4 / B0-28 = B0 09-27/28 | 附录 A.1 |
| VR 限频 | A0 在 09-10..09-28 期间每次 amdgpu init 都打印 `GPU is throttled ... VR`，sclk DPM 只剩 500/1100 MHz；09-29 刷 VBIOS/SMU 后解除 | (a) a.1 |
| ASM（aiter ASM） | aiter 为 gfx1250 预编译的汇编 kernel（`.co`）。fwd 为 `fmha_bf16_pertokenBf16_hd128_128x256_mask.co`（入口 `fmha_fwd_with_sink_asm`）；bwd 为 odo → `bwd_hd128_bf16_causal_br_a32_pssk` → dq_convert 三个 kernel。本项目所有"相对 ASM / 占 ASM"的基准 | (b) b.1、b.4 |
| beat / bar | beat：op-evolve 里与候选同会话测的 aiter ASM 臂；bar：引用的 ASM 基准值（A0-VR 09-24 普查 fwd 1.5724 ms、bwd 7.6766 ms；各阶段的 bar 见 (c) c.1） | (c) c.1 |
| 10.160 ms（shim 伪影） | 09-15 用 autograd shim 测的 ASM bwd：每次调用做 3 次 hipModuleLoad、新分配约 1 GiB scratch，另含约 1.4 ms autograd 开销。09-24 撤回，真实 bar 为 7.6766 ms | (b) b.4；(d) d.3 |
| GQA 越界写 / `dkdv_heads=q` | aiter gfx1250 ASM bwd 的 grid 是 (kv_tiles, nhead_q, batch)，却按 q head 去索引 kv 尺寸的 dk/dv，G=4 时 4 个 WG 无同步地写同一个 tile。绕法是按 q head 分配 dk/dv，再在 host 端规约 | (a) a.2 #13；(b) b.4 |
| 7-GEMM / 5-GEMM | 我们的 1-wave bwd 为了 dq 确定（不用原子）要发 7 个 GEMM 当量，aiter ASM 用 dq 原子累加只发 5 个；结构因子 1.386–1.408× | (b) b.4；附录 A.3 |
| SQNR 门 | 输出与 fp32 参考的信噪比。out/dq/dk/dv 四张量各自 ≥50 dB（fwd 门后来降到 49 dB）；dk/dv 要求 200 次逐位一致，dq 自 09-25 起改为 run-to-run ≥70 dB | (d) d.5、d.6 |
| nkfix | e2e 中绕开 hipBLASLt 坏 tile 的运行时补丁：用 `TorchDispatchMode` 拦截 `aten::mm`，把反向 GEMM 的操作数改成前向那种物理布局（dgrad：`mm(A, B.t().contiguous().t())`；wgrad：`mm(A.contiguous(), B.t().contiguous().t())`，wgrad 的判据要先于 dgrad），从而命中 `MT256x256x128`，而不是 GEMV tile `MT32x16x32`。版本：A0 v1（09-15，只改 dgrad）；v3（09-16，加 wgrad；规则 3 让 wgrad 走 FlyDSL GEMM 离线表，默认关）；B0 `nkfix_b0.py`（09-28 重写，Triton 分块转置 `transpose_triton.py`、按输出维分块、`NKFIX_CHECK` 非有限值检查，教程用这一版）。开销每步约 70 ms，`NKFIX_CHECK=1` 再加约 33 ms | (d) d.2.2 |
| MT32x16x32 / MT256x256x128 | hipBLASLt（Tensile）kernel 的 macro tile。前者是 GEMV 用的小 tile，反向 GEMM 落到它时只有 50–80 TF/s；后者是 nkfix 之后全部命中的大 tile | (d) d.2 |
| refcache | 把精度门和 benchmark 用的 fp32 参考预先在 CPU 上算好并缓存（prod 10m26s、1.1 GB），避免在卡上跑 fp32 Tensile 参考（A 类挂卡的来源）；sha 对不上时直接判 FAIL | (d) d.1.3、d.5 |
| prod / proxy / fast | op-evolve 的三个计分形状 (B, Sq, Skv, Hq, Hkv, D)：prod (4,8192,8192,32,8,128)，即 Llama-3.1-8B 训练形状；proxy (1,4096,4096,32,8,128)；fast (1,1024,1024,8,2,128)。只按 prod 排名（h7）；fast 是 launch-bound，后来权重设为 0 | (c) c.1 |
| 逐次交错尺子 | op-evolve benchmark.py 原来的计时法：各 arm 逐次交替、回文排列，每个 arm 51 次取中位。每次调用继承上一个 arm 留下的功耗/时钟状态，FlyDSL/ASM 比值系统性偏低（fwd 约 25%，bwd 约 3.4%） | (d) d.3 |
| 分块尺子（blocked ruler，blk） | 09-28 起的计时法（h40/h41/h66）：每个 arm 先跑 4 次不计时，再连续计时 9 次，块间回文；每个排名进程带 A/A 副本（fwd ±0.19%、bwd ±0.05%），A/A 偏离超过 0.5% 整个进程作废。blk 也用来指"分块尺子下的读数"，与 gb 相对 | (c) c.1；(d) d.3 |
| 真实数据尺子（real-dump ruler） | 用 B0 训练 step 43 导出的 6 层真实 q/k/v 代替 randn（h45/h47）。真实 score 标准差 21–53，randn 约为 1 | (d) d.3 |
| 训练工作点（training clock）/ gb 尺子 | 训练中 GEMM 把整板功耗顶满，attention 跑在比 op 测试低得多的时钟上（B0 训练中 1.25–1.69 GHz；A0 在 GEMM 突发后约 1.28 GHz）。gb 尺子在每次计时前先跑 10 个 bf16 32768×4096×14336 GEMM 来复现这个工作点。ASM 几乎不随时钟变，FlyDSL 会变慢。gb 已设计，op-evolve 计分尚未安装 | (d) d.3；`1002__oe/RULER.md` |
| wedge（挂卡）类别 A/B/C | 卡挂死后驱动 reset 完不成，只能 AC 断电恢复。按 dmesg 首行分：**A** 是卡上 fp32 Tensile 参考 GEMM 的地址越界写（sub-4GB 截断地址、PF 0x5 / RW 0x1）；**B** 是首条为 `INVALIDATE_TLBS` 超时、此前没有任何内存故障；**C** 是候选 kernel 越界读引发 `GC_UTCL2` 故障风暴，伴随 `IH ring buffer overflow`。另有 S（启动期 MES 故障族）、K（已知操作诱因）、R（资源/进程）、F（进程级可恢复故障，不算挂卡）、D（降级）、X（驱动/固件等非 kernel 原因） | (d) d.1.1 |
| AC cycle（AC 断电） | 用户本人到机器前给整机断电再上电，是挂卡后唯一的恢复手段（`modprobe -r amdgpu` 会让整机失联）。恢复后要 `sudo modprobe amdgpu`、把 `dmesg_restrict` 设回 0、确认 KFD 为空、`git fsck` | (d) d.1.3 |
| MES / KFD | MES 是 GPU 上的硬件调度微码，挂卡签名多为 `MES failed to respond`、`wait for reset ack`；KFD 是 `/dev/kfd` 计算驱动接口，挂卡后进程的引用不会归零 | (d) d.1 |
| TDM | Tensor Data Mover：gfx1250 的异步张量搬运单元，按描述符把 global 数据搬进 LDS，FlyDSL 里对应 `make_tdm_atom` / `tdm_ops`。s3–s6 的主杠杆就是"TDM 3 级 LDS ring + 把下一轮 B 操作数提前读回寄存器" | (c) #73、#76 |
| WMMA | gfx1250（wave32）的矩阵乘累加指令，取代 CDNA 的 MFMA。roofline 实测每条 WMMA 8 cycle，WMMA↔DS 切换约 29 cycle | (c) #69；`0930__roofline/REPORT.md` |
| MFMA / `ds_read_tr16_b64` / `permlane32_swap` | CDNA（gfx950）专用指令，在 gfx1250 上都 "Cannot select"，所以 gfx950 的 FlyDSL / HipKittens attention 移植不过来 | (a) a.2 #6；(b) b.2.3 |
| ATT | rocprofv3 的线程追踪，能给出逐指令的 Hitcount / Latency / Stall。A0 在 09-29 刷固件之后才对 FlyDSL JIT 和 ASM `.co` 可用 | (d) d.4 |
| ITT | `TRITON_HIP_USE_IN_THREAD_TRANSPOSE`，09-14 op-evolve 找到的 Triton 编译开关，让融合 bwd 快约 9.7%（`a009599b`） | (c) #13、#14 |
| k_dkdv / k_dqg | 我们的 FlyDSL bwd 由几个 kernel 组成：k_delta（odo 预处理）；k_dkdv（按 KV tile 计算 dK/dV，GQA 在寄存器里规约，不用原子）；k_dq（计算 dQ；r29/s6 系列里 proxy/prod 形状走 k_dqg，toy/fast 形状走 split-K 版 k_dq_sp）；k_redsp 是 split-K 的归约 kernel。s1 起 k_dqg 放在 side stream 上，与 k_dkdv 并发 | (c) #26、#38、#72 |
| r20 | A0-VR bwd job 的冠军（k_dkdv 的 Q/dO 预取深度 1→2），511.42 TF/s；也是 B0 bwd 的起点 | (c) #42 |
| r19h | r19 加上 h33 的三处 clamp（越界预取 1,580→0，VGPR 904→724），由 B0 r28 的 refactor h68 晋升 | (c) #61 |
| u2n | k_dq 的 full KV 循环展开 ×2，去掉回边上 65 条 `v_mov_b64` 的寄存器轮转（k_dq 3.185→2.990 ms，h70）。A0 刷固件后反而是负收益，s 系列从 c1 起关掉（`DQ_U2=False`，−0.85%） | (c) #62、#70 |
| r29 | B0 bwd 冠军 = r19h + u2n + g86（dK/dV 的 epilogue 经 LDS ring），B0 上 656.5 TF/s（ASM 的 80.4%），A0 刷固件后为 ASM 的 83.6%。产品分支里的 bwd 就是它 | (c) #62、#69 |
| r13ns / r16 | B0 fwd job 第 13 轮的 kernel 关掉投机 softmax（`SPEC_STALE_MAX=False`）得到 r13ns，在第 16 轮由 refactor h44 晋升为冠军，所以 r16 = r13ns。它是当前的 fwd 冠军，也是产品分支里的 fwd | (c) #58 |
| s1–s6 / s6 | 09-30 A0 手工攻关的阶段冠军：s1 = tailpf + trorder + streams；s2 = s1 + soffset；s3 = k_dkdv 的 TDM 3 级 ring（dkdv_tdm3）+ dqg_ts + side stream；s4 = 精简 VALU、v_nop、地址计算，计数器去掉除法；s5 = k_dqg 也走 TDM ring（dqg_tdm）；s6 = s5 + k_dqg 的 VALU 精简。s6 为 5.295–5.300 ms、ASM 的 103.8%，是当前 bwd 冠军（flydsl 0.3.2；在 0.3.4.1 下 ISA 逐字节相同），还没进产品分支 | (c) #70–#77 |
| w4f | 4-wave 融合 bwd：5 个 GEMM、split barrier、dQ 用 fp32 `SCOPE_DEV` 原子累加。结果正确，但受整芯片原子吞吐限制，最好 6.509 ms（r4b，09-30），A0 r26 为 6.486 ms；不写 dQ 原子的 relax_noatom 上限为 4.241 ms（输出错误，只作上限）。已暂停。10-02 r27 挂卡的两个变体常被误记为 w4f，实为 s6 基底的 cluster multicast（A_g74、B_g82） | (c) #75、#84；(d) d.1.1 |
| h-hints（hNN） | operator 写给 op-evolve agent 的规则或指令，放在 job 的 `hint.md`，按 `h<N>` 编号。解析器只读首格为 `h<N>` 的索引表行，所以 h83 因为缺索引行，到 10-02 才生效。**编号按文件独立**，不同文件里的同号 hint 不是同一条（例如 A0 bwd 的 h41 讲 atomic scope，B0 的 h41 讲分块计时）。文件位置：A0 bwd 在 `0923__flydsl/hint.md`，A0 fwd 在 `0925__flydsl/fwd-job/hint.md`，B0 在 `0927__b0/{fwd,bwd}-hint.md`，A0 09-30 之后（h83/h85）在 bwd job 的 `job_context/hint.md`。常被引用的有：h7（只按 prod 排名）、h18（aiter 的 dq 原子过不了确定性门）、h28（beat 单独一个进程）、h40/h41/h66（分块计时）、h44（采用 r13ns）、h46/h72（FlyDSL JIT 缓存 key 漏掉模块常量）、h47（真实数据尺子）、h50（禁止在卡上算 fp32 参考）、h68/h70（r19h、u2n）、h69（proxy 和 prod 各自 ≥ beat）、h83（fast 按 min 计分，gain_weights 1/0.25/0）、h85（新 kernel 先 toy 后 prod，读 VGPR 只用 compile-only，只用镜像 hipBLASLt） | (d) 阅读约定、d.5 |
| op-evolve | 同事开发的自动持续优化算子框架（`/home/lihuzhan/code/2026_0910__op-evolve/op-evolve`）。一个 job 按轮推进，账本在 `artifacts/<job>/rounds/NNN/` 和 `job_context/state.yaml`。普通（fast）轮为 `1-opt`（agent 提出并实现候选）→ `2-reflect`；deep 轮为 `1-profiling` → `2-plan` → `3-act` → `4-reflect`；operator 的 refactor 记在 `0-refactor`。gain 是本轮代码与历史最佳在同一会话重测后的比值，验收看 min_gain（0.007）和 gain_weights；FastError 指某个想法没写进 facts/dead_ends 时整轮被记为 failed | (c) c.1；(d) d.5 |
| champion（冠军） | 一个 job 当前被接受的最佳代码（`state.yaml` 的 best_round，代码在 `op/current`）。注意 state.yaml 按形状记的 champions 指针会记下被否轮次的 best-ever 读数，与 best_round 不一定相同，续跑时会挡住后面的轮次 | (c) c.1；(d) d.5 |

## 附录 C 主要来源索引

路径若无前缀，都相对 `Primus-Turbo/output/`；`OE:` = `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/`；`skill:` = `Primus-Turbo/.claude/skills/gfx1250-attn-campaign/`；`WT:` = worktree `/home/lihuzhan/code/2026_0903__turbo/wt-llama31`（分支 `dev/lhz/llama31_attn_opt`）；`Primus:` = `/home/lihuzhan/code/2026_0828__primus/Primus/`。

**总入口与状态**
- skill:`SKILL.md`，以及 skill:`references/{baselines,bwd-history,fwd,env-and-pitfalls,op-evolve-ops,flydsl-api,kyle-learnings}.md`；`~/.claude/skills/gfx1250-card-safety/SKILL.md`；memory 目录 `~/.claude/projects/-home-lihuzhan-code-2026-0903--turbo-Primus-Turbo/memory/`。
- op-evolve job 目录：
  - A0 bwd：`OE:artifacts/gfx1250-flydsl-attn-bwd-20260917-115934.a0-stale-0930/`（r0–r23）；live 目录 `OE:artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/`（r0–r27，其中 r24–r27 是 A0 09-30 之后的编号）。
  - A0 fwd：`OE:artifacts/gfx1250-flydsl-attn-fwd-20260925-114644/`（r0–r13）。
  - B0 fwd：`OE:artifacts/gfx1250-flydsl-attn-fwd-b0-20260927/`（r0–r20，10-02 从 B0 备份恢复）。
  - B0 bwd（B0 编号 r24–r32）：账本在 `0927__b0/bwd/`；完整目录只在 B0 备份包 `/home/lihuzhan/code/0928__bak_b0.tar.gz` 里，至今没有解包（见 `0928__bak_b0/MANIFEST.md`）。
  - B0 09-14 Triton job：`0914__campaign/op-evolve-artifacts/`。
- 各日总报告：`0913__opt_plan__claude/index.html`、`0914__campaign/report.html`、`0915__opt/PROGRESS-REPORT.html`、`PROGRESS-REPORT-0917.html`、`0922_summary/ASM-ATTENTION.md`、`0927__b0/REPORT-0928.html`、`0928__a0_repro/REPORT.md`、`0930__bwd/REPORT.md`、`0930__roofline/REPORT.md`、`1002__e2e/RESULT-e2e.md`、WT:`docs/gfx1250_llama31_8b_e2e/README.md`。

**(a) 优化前 main 的问题**
- `0913__opt_plan__claude/`：`PROGRESS.md`、`phase0/PHASE0-STATUS.md`、`phase0/PLATFORM-ESCALATION.md`、`phase1/RESULTS.md`、`phase1/BAKEOFF.md`、`phase2/DECISIONS.md`、`phase2/INCIDENT-2026-09-13-wedge.md`、`phase2/ledgers/wq.jsonl`。
- `OE:output/0911__fa_gfx1250_phase1/`：`E1.md`、`E2.md`、`E3.md`、`E4.md`、`HARDWARE-ISSUE.md`、`E1.gemmchk.log`。
- `0914__campaign/{RESULTS,HANDOFF}.md`；`0914__repro__c07/{RESULTS,FINAL-TABLE}.md`；`0915__repro__c07/RESULTS.md`。
- `0915__opt/`：`BLAS-FINDING.md`、`GEMM-NN-FINDING.md`、`GEMM-WGRAD-FINDING.md`、`E2E-AB.md`、`JIRA-TRACE-ANALYSIS.md`、`VENDOR-REPORT-hipblaslt.md`、`BACKEND-STRATEGY.md`。
- `0917__flydsl/{SURVEY,VENDOR-REPORT-aiter-gfx1250,STAGE1-FWD,GQA-WORKAROUND-COST}.md`；`0927__b0/e2e/{RESULT,RESULT-final,RECON,VERIFY}.md`；`0928__a0_repro/REPORT.md`；`0930__port/PR_BODY.md`、`0930__port/runs/bench_turbo_{triton,flydsl}.csv`；WT:`docs/gfx1250_llama31_8b_e2e/README.md`。
- Primus 侧：`Primus:output/0910__fa_opt/attn_bench_gfx1250.log`、`Primus:examples/torchtitan/configs/MI455X/repro_l8b_*.yaml`、`Primus:primus/configs/modules/torchtitan/pre_trainer.yaml`。
- git：`c1325c7e`（#481）、origin/main `8cda13c7`、`f5e1f18b`（#501）、`9f05cd85`（#509）、`eeb9e8d7`（#508）。

**(b) 各后端性能与 ASM 的问题**
- `0913__opt_plan__claude/phase1/BAKEOFF.md`、`phase2/{DECISIONS,T2-ASM-BACKWARD-SPEC,T7-ASM-FORWARD}.md`。
- `0914__repro__c07/{FINAL-TABLE.md,RESULTS.md,flex_ledger.jsonl}`；`0914__campaign/{RESULTS.md,HANDOFF.md,t7/soak_cooled.jsonl}`；`0914__hk_udna1_gate/`。
- `0915__opt/`：`RESULTS.md`、`E2E-AB.md`、`BOTTLENECK-SHIFT.md`、`FLYDSL-AB.md`、`RESULT-32L.md`、`NAN-FINDING.md`、`PERF-CO-CLOSED.md`、`TWOKERNEL-SWEEP.md`、`status.json`；`0915__repro__c07/RESULTS.md`。
- `0917__flydsl/{STAGE1-FWD,VENDOR-REPORT-aiter-gfx1250,GQA-WORKAROUND-COST}.md`；`0922_summary/ASM-ATTENTION.md`；`0924__flydsl/DAY-SUMMARY.md`、`0924__flydsl/bar-census/{fwd_anchor,fwdbwd_anchor}.json`；`0925__flydsl/{AITER-5GEMM-STUDY.md,fwd-isa/REPORT.md,fwd341/ab_prod.log}`。
- `0927__b0/`：`REPORT-0928.html`、`fwd/progress.md`、`bwd/progress.md`、`ruler/REPORT.md`、`ruler/bwd/REPORT.md`、`fwd-nospec/REPORT.md`、`e2e/{BUILD,RESULT,RESULT-final,RECON}.md`、`gemm/REPORT.md`、`profile/REPORT.md`、`OP-EVOLVE-SUGGESTIONS.md`。
- `0928__a0_repro/REPORT.md`；`0930__bwd/{REPORT.md,PROGRESS.md,notes/report_asm.md}`；`0930__roofline/REPORT.md`；`0930__port/{PR_BODY.md,runs/}`；`1002__e2e/{E2E-PLAN.md,RESULT-realab.md,RESULT-e2e.md}`、`1002__e2e/e2e/runs/TABLE.1002_095504.md`；skill:`references/baselines.md`。

**(c) 每一轮的进展**
- `rounds.csv`（= `parts/rounds.csv`）；上面列出的各 op-evolve job 的 `job_context/state.yaml`、`rounds/NNN/{1-opt,3-act}/act.yaml`、`timing.yaml`、`opt.md`。
- `0923__flydsl/{hint.md,STAGE2-FWD-SWEEP.md}`；`0924__flydsl/DAY-SUMMARY.md`；`0925__flydsl/fwd-job/hint.md`；`0927__flydsl/asm-structure/DESIGN.md`。
- `0927__b0/{fwd,bwd}/progress.md`、`0927__b0/fwd/rounds/`、`0927__b0/lab-kdq/REPORT.md`、`0927__b0/lab-bwd-r19/REPORT.md`、`0927__b0/champions/`。
- `0930__bwd/{PROGRESS.md,REPORT.md,armsrc/}`、`0930__bwd/probe/P1-RESULTS.md`；`0930__roofline/REPORT.md`；`1002__oe/{RULER.md,FWDJOB.md,incident/WEDGE-1002.md}`；`1002__e2e/arms_src/bwd_s6_0341`；skill:`references/{bwd-history,fwd}.md`。

**(d) 问题与解决方案**
- 挂卡：`OE:output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md`；`0913__opt_plan__claude/phase2/INCIDENT-2026-09-13-wedge.md`；`0915__opt/{MES-WEDGE,INCIDENT-2026-09-15-machine-death,FLYDSL-WEDGE,DAY-0916-SUMMARY,FWD-SWEEP}.md`；`0922__flydsl/{ROUND3-CLOSEOUT,HANDOVER}.md`；`0923__flydsl/wedge4/ANALYSIS.md`；`0924__flydsl/wedge-rootcause/`；`0927__b0/{LAB-RULES,HANDOFF-A0}.md`；`1002__oe/incident/WEDGE-1002.md`、`OE:artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/rounds/027/{_scratch/arms,_scratch/build_run,1-opt/raw}/`（10-02 挂卡的实际变体与探针）；`~/.claude/skills/gfx1250-card-safety/SKILL.md`。
- GEMM / hipBLASLt：`0915__opt/{BLAS-FINDING,GEMM-NN-FINDING,GEMM-WGRAD-FINDING,PROFILE-POST-NKFIX,NKFIX-NAN-RATE,NAN-FINDING,VENDOR-REPORT-hipblaslt}.md`；`0922__flydsl/gate-patch/README.md`；`0923__flydsl/STAGE2-S0-PROBE.md`；`0924__flydsl/REFCACHE-PREMISE-GONE.md`；`0927__b0/gemm/REPORT.md`、`0927__b0/profile/REPORT.md`、`0927__b0/e2e/{RESULT,VERIFY}.md`；`1002__e2e/E2E-PLAN.md`。
- 尺子、框架、工具链、正确性：`0927__b0/ruler/REPORT.md`、`0927__b0/OP-EVOLVE-SUGGESTIONS.md`、`0927__b0/{fwd,bwd}-hint.md`、`0927__b0/interference.md`；`0925__flydsl/{PARITY-STRATEGY,AITER-5GEMM-STUDY}.md`、`0925__flydsl/fwd-job/{hint,NOTES}.md`；`0923__flydsl/{hint,CAMPAIGN-FINAL}.md`、`0923__flydsl/poison-fix/README.md`；`0924__flydsl/kdq-enumeration/SYNTHESIS.md`；`0917__flydsl/API-DELTA.md`；`0921__flydsl/A0-A1-ZERO-GPU-COMPILE.md`；`0930__port/API_AUDIT_SUMMARY.md`；`0930__bwd/{PLAN.md,oejob/README.md}`；`1002__oe/{RULER,FWDJOB}.md`；skill:`references/{env-and-pitfalls,op-evolve-ops,flydsl-api}.md`。
- 基础设施与协作：`0927__b0/{README,STOPPED}.md`；`0928__bak_b0/MANIFEST.md`；memory `push-after-every-round.md`、`e2e-monitoring-style.md`。

**会话记录**（原始 JSONL 在 `~/.claude/projects/-home-lihuzhan-code-2026-0903--turbo-Primus-Turbo/<id>.jsonl`；去掉工具输出的文本版在 `_work/sessions/`。B0 09-27/28 的会话不在本机）

| 会话 id | 时间范围（UTC） | 主要内容 |
|---|---|---|
| 541e7bc3 | 09-13 02:24 – 09-14 03:03 | 制定自动多轮优化计划；A0 Day-1：main bring-up、bake-off、Triton 调优、e2e 首次跑通 |
| b596bddb | 09-15 01:15 – 09:04 | 把 B0 09-14 的工作备份到 A0；A0 复现 B0 冠军；接入 ASM bwd；09-15 早上整机失联 |
| 11052787 | 09-15 09:05 – 09-17 07:37 | e2e A/B；hipBLASLt 路径与 nkfix；NaN 普查；09-16 多次挂卡 |
| e8ec014a | 09-17 07:52 – 08:00 | FlyDSL attn 开发计划（很短，随后由 c4b79aa6 接续） |
| c4b79aa6 | 09-17 08:01 – 09-21 09:57 | FlyDSL 调研、Stage1、bwd kernel bring-up、op-evolve bwd job 发车 |
| a9a96fef | 09-21 09:58 – 09-25 10:28 | bwd op-evolve r1–r23 无人值守运行；挂卡分类与防护 |
| 03a497e0 | 09-22 00:57 – 00:58 | 一次 deep-research workflow 调用，没有实质内容 |
| 1406f036 | 09-22 12:13 – 09-24 00:49 | 组会汇报用的 0922 总结（`0922_summary/`） |
| 0a07052f | 09-25 10:29 – 09-27 15:25 | flydsl 0.3.4.1；fwd op-evolve job；把经验整理进 skill |
| d2b7f402 | 09-28 11:02 – 12:44 | 在 A0 复现 B0 当天的 fwd/bwd 相对提升和 e2e |
| 2dafe0d2 | 09-29 06:24 – 06:53 | 换驱动后卡在 rocm-smi 里看不到、无法初始化 |
| c28b7a9f | 09-30 02:23 – 07:22 | roofline 分析（多机对比、M8 消融） |
| fdc2534d | 09-30 03:23 – 10-05 05:06 | 产品分支 `dev/lhz/llama31_attn_opt` 移植、API 审计、e2e 教程 |
| 69864fc9 | 09-30 07:37 – 10-05 01:09 | bwd 手工攻关 s1–s6；A0 bwd job r24–r27；10-02 解包真实 dump、处理挂卡 |
