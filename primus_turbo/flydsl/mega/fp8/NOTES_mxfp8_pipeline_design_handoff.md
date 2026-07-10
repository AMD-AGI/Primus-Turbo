# 交接:Primus-Turbo mega-MoE 的 MXFP8 融合流程设计

> 用途:新 session 直接 `@` 这个文件即可接续。

## ★ 已解决(2026-07-09):L2「combine fp8 化」结论 = 不做,收在 L1 fp8 + L2 bf16

核心问题「combine 能否 fp8 化」已彻底探明,详见 `NOTES_mxfp8_fused_gemm_combine_perf.md`:
- combine **可以** fp8 化且正确(cos 0.9996),隔离下字节杠杆真实(combine 2.70→1.44ms,1.85×)。
- 但在**融合核里赢不了**:mxfp8 量化 L2 GEMM 输出是重计算(~1–2ms),放哪都吃掉字节收益。三种放法
  全实现+实测(都正确、都更慢):quant-in-combine **0.99×** / 独立 quant role **0.76×** / GEMM epilogue **0.76×**。
- 顺带纠正:旧 note 的 combine「43 GB/s / 117MB」是**错的**(那是输出大小,非传输量);实测 combine
  是**带宽 bound ~300 GB/s**(≈ L1 dispatch 的 377)。
- **落地方案:L1 用 `comm="fp8_fused"`(1.4×),L2 用 `grouped_gemm_combine_bf16`(2.7–2.9ms)。**
- 实验性 dead-end 代码:`grouped_gemm_combine_fp8_kernel.py`(正确但更慢,未接入,仅参考)。

下面为历史背景(问题已闭环,无需再讨论 L2 combine fp8 化)。

## 背景
在 AMD MI355X (gfx950) 上,给 Primus-Turbo 的 mega fused MoE 加 MXFP8 (per-1×32 E8M0 block scale)
低精度训练支持。目标:用「通信+计算融合」的单核 pipeline,让 fp8 前向比 bf16 融合核和 fp8 解耦路径都快。
前向分三段:**L1(dispatch + up/gate GEMM)→ SwiGLU(bf16)→ L2(down GEMM + combine + reduce)**。

代码根目录(远程):`/mnt/shared/xiaoming/Primus-Turbo`

关键文件:
- 前向: `primus_turbo/flydsl/mega/fp8/mega_moe_fused_mxfp8.py`
  - `mega_moe_fused_mxfp8_forward(comm="fp8_fused"|"fp8"|"bf16")`
  - 流程: L1 dispatch+GEMM → `swiglu(l1)` → 重新量化 act 到 mxfp8 → L2 `grouped_gemm_combine_mxfp8`
- L1 融合核: `primus_turbo/flydsl/mega/fp8/dispatch_grouped_gemm_mxfp8_kernel.py`(3 角色)
- L1 helper: `primus_turbo/flydsl/mega/fp8/ep_fp8.py`
- L2 融合核: `primus_turbo/flydsl/mega/fp8/grouped_gemm_combine_mxfp8_kernel.py`(3 角色)
- bf16 L2 参考核: `primus_turbo/flydsl/mega/grouped_gemm_combine_bf16_kernel.py`
- mxfp8 GEMM tile: `primus_turbo/flydsl/mega/fp8/gemm_mxfp8_tile.py`(`gemm_mxfp8_nt_tile`)
- 共享 C store: `primus_turbo/flydsl/utils/gemm_helper.py`(`StoreCPerTensor`,已加 `cache_modifier` plumbing,默认关)
- 量化: `primus_turbo/flydsl/mega/fp8/quant.py`
- benchmark: `benchmark/ops/bench_dispatch_grouped_gemm_mxfp8.py`(L1)、`benchmark/ops/bench_grouped_gemm_combine_mxfp8.py`(L2)
- e2e 测试: `tests/pytorch/ops/test_mega_mxfp8_ffn_e2e.py`、`tests/pytorch/modules/test_mega_moe_mxfp8.py`
- 性能笔记: `NOTES_mxfp8_fused_dispatch_gemm_perf.md`(L1)、`NOTES_mxfp8_fused_gemm_combine_perf.md`(L2)、`NOTES_mxfp8_grouped_gemm_perf.md`

## 当前状态(已完成)
两级融合核都写完、正确性验证通过(e2e ~22.97 dB;kernel cos=1.0 vs 解耦 mxfp8 参考)。

**L1(dispatch+GEMM)—— fp8 融合明确获胜** (DSv3 L1: T=8192,H=7168,I=2048,E=256,K=8,EP8,BM=BN=256):
- fp8 fused **2.56ms**(LB)/ **2.29ms**(RR),= 1.4× vs bf16 fused(3.58),1.12× vs fp8 decoupled(2.87)
- 3 角色: COMM(clean-push 预量化 fp8 + raw E8M0 scale)→ PRESHUFFLE(raw→broadcast,每 pool-block 一次)→ GEMM(preshuffled mxfp8),scoreboard sys-scope + L2 fence 做跨 rank/跨 XCD 可见性,省掉 host sync。
- 关键:E8M0 scale 有两种布局 —— 快速 push 要 raw `[row,K/32]`(每 token coalesced),快速 MMA scale loader 要 broadcast;两者互为转置,必须有一次 preshuffle。最优放法 = 「每 block 一次、本地、单独作为一个 pipeline stage」。

**L2(GEMM+combine+reduce)—— fp8 拿不到收益(核心待讨论问题)**:

| 项 | 延迟 |
|---|---|
| fp8 gemm_only(down-proj)| 1.12ms / 1828 TF(不是瓶颈)|
| fp8 combine_only(bf16 reduce-scatter push)| **2.71ms**(~117MB/rank XGMI,墙)|
| fp8 reduce_only(bf16 top-k)| 1.05ms |
| fp8 decoupled(串行)| 4.84ms |
| **fp8 fused**(l2_writeback 一致性)| **4.68ms**(只比 decoupled 快 1.03×)|
| **bf16 fused**(参考)| **2.80ms** |

## 核心设计发现 / 张力(重新讨论的重点)
1. **为什么 L1 赢、L2 不赢**:L1 的通信是 **fp8**(dispatch push 字节减半)且 push(1.13ms)≲ gemm(1.74ms),两个 fp8 量级的成本重叠 → 融合有收益。L2 的 combine 是 **bf16**(MoE 最终输出必须 bf16),是 `[tokens,H]` bf16 的 reduce-scatter,`combine 2.71ms ≫ fp8 gemm 1.12ms` → L2 是 combine(通信)-bound,fp8 GEMM 的算力优势被完全掩盖。fused 天花板 ≈ max(gemm,combine) ≈ 2.7–2.8ms = **和 bf16 fused 一样**。
2. **当前 fp8 fused L2 甚至比 bf16 慢**:因为它用了「每个 gemm block 全设备 `l2_writeback`」做跨 XCD 一致性,把 gemm 角色串行化、毁了 overlap。bf16 核用的是免费的逐-store **write-through (sc1)**。已尝试给 mxfp8 也上 write-through 两次都失败:①row-band+cache_modifier miscompile 挂死;②改成全 C resource(匹配 bf16 机制)能编译,但 mxfp8 的**列跨步 store 写穿 HBM 太慢**→ gemm 角色信号晚 → reduce flag 超时。已回退到可用的 l2_writeback。要真正修好需要 coalesced(CShuffle 风格)write-through store,而且**即便修好也只是追平 bf16,不会更快**。

## 需要在新 session 重新讨论的设计问题
1. **L2 的 combine 能不能 fp8 化?**(最关键)—— 若在 push 前把每个 expert 输出行量化成 fp8、reduce 里 dequant + 加权求和,combine 字节减半(2.71→~1.4ms),L2 才可能 fp8-win。代价是精度(fp8 combine + 最多 K 个 fp8 值的加权和)。这是能否让 L2 真正受益的唯一杠杆,需评估精度可行性。
2. **L2 到底该不该做 fp8?** 还是 L1 用 fp8 fused、L2 直接用 bf16 fused(2.80,已验证)的混合最优?(除非 combine 能 fp8 化)
3. **compute-heavy régime**:tokens/expert 更大时 gemm 会 ≳ combine,那时 fp8 L2 才开始有用 —— 是否值得为那种 shape 保留 fp8 L2?
4. **一致性方案**:coalesced write-through vs l2_writeback vs 其它跨 XCD 可见性手段,哪个是通用最优?
5. **SwiGLU**:目前是 L1 GEMM 输出(bf16)后单独 kernel,bf16 精度,再量化进 L2。是否要把它和 L1 epilogue 或 L2 量化融合?

## 复现环境 & 命令
- 远程: `ssh chi2761` → `docker exec xiaoming-dev-slime bash -lc '...'`,`cd /mnt/shared/xiaoming/Primus-Turbo`
- L2 bench: `MEGA_BENCH_TIMEOUT_S=150 PYTHONPATH=$PWD:$PWD/benchmark/ops python benchmark/ops/bench_grouped_gemm_combine_mxfp8.py --num-processes 8 --mode load_balanced --iters 20`
- L1 bench: `PYTHONPATH=$PWD:$PWD/benchmark/ops python benchmark/ops/bench_dispatch_grouped_gemm_mxfp8.py --num-processes 8 --iters 20`
- 8×GPU 分布式;跑挂了用 `ps -eo pid,cmd | grep '[s]pawn_main'` 找到 worker `kill -9`
- 遵循 kernel-optimize skill 和 `.cursor/rules/iteration_rules.mdc`:**正确性优先于性能**,单变量线性迭代,每次改动跑完整验证 + benchmark,accept/rollback 记录。

## 请先做
先读 `NOTES_mxfp8_fused_gemm_combine_perf.md` 和 L2 融合核 + bf16 参考核,然后和我讨论「combine fp8 化」的精度/收益可行性,再决定 L2 的流程设计(**不要直接开写**)。
