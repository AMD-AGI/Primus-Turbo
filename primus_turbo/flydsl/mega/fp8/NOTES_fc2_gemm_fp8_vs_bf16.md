# fc2(L2 down-proj)GEMM:fp8 vs bf16 性能分析(含量化开销)

> MI355X (gfx950)。DeepSeek-V3 MoE L2 形状:`H=7168, I=2048`,EP8 `G=32` experts/rank。
> fc2 = `act[pool, I] @ w2[G, H, I]^T → l2y[pool, H]`,grouped NT,K=I=2048,N=H=7168。
> 测试脚本思路见 `_fc2_quant_bench.py`(GEMM-only 见 `NOTES_mxfp8_fused_gemm_combine_perf` 同批)。

## 结论(TL;DR)

**fc2 用 fp8(mxfp8, per-1×32 E8M0)GEMM,即使把 act 量化开销算进去,仍比 bf16 快 ~1.7×。**
量化开销很小(act 量化只占 fp8 总时间的 ~5–6%),w2 量化是静态权重、可缓存摊销(每步 ~0)。

## 数据

### GEMM-only(输入已预量化,只测 GEMM 本身)

| 实现                                   | pool=8192                            | pool=4096                 |
| -------------------------------------- | ------------------------------------ | ------------------------- |
| bf16 grouped(Triton)                   | 0.344 ms / 700 TF                    | 0.305 ms / 394 TF         |
| **mxfp8 FlyDSL**(mega,per-1×32) | 0.184 ms / 1306 TF(**1.87×**) | 0.169 ms / 712 TF(1.81×) |
| mxfp8 Triton(main)                     | 0.228 ms / 1054 TF(1.51×)           | 0.220 ms / 548 TF(1.39×) |
| tensorwise FlyDSL(#407, 8-wave)        | 0.155 ms / 1550 TF(2.22×)           | 0.148 ms / 812 TF(2.06×) |

### 含量化(fp8 路径 = act 量化 + fp8 GEMM;w2 量化静态缓存,不计入每步)

| 项                            | pool=8192          | pool=4096          |
| ----------------------------- | ------------------ | ------------------ |
| bf16 GEMM                     | 0.341 ms / 706 TF  | 0.304 ms / 396 TF  |
| act 量化(rowwise mxfp8)       | 0.012 ms           | 0.009 ms           |
| fp8 GEMM(mxfp8 FlyDSL)        | 0.186 ms / 1293 TF | 0.173 ms / 695 TF  |
| **fp8 总(量化 + GEMM)** | **0.198 ms** | **0.182 ms** |
| **fp8 总 vs bf16**      | **1.72×**   | **1.67×**   |
| 量化占 fp8 总的比例           | 6%                 | 5%                 |

## 量化开销分析

- **act 量化**(每次调用都要做):`act[pool, I]` rowwise mxfp8,pool=8192 时仅 **0.012 ms**,占 fp8 总时间 **~6%**。它是纯访存 kernel(读 bf16 写 fp8+E8M0),开销随 pool 线性但绝对值极小。
- **w2 量化**(权重,静态):`w2[G, H, I]` grouped mxfp8。权重在训练一步内不变,量化结果按 `(id(weight), _version)` 缓存、跨 fwd/bwd 复用,每步摊销 ~0(参考 iteration_rules Rule 11 的 W1 bounded weight cache)。因此**不计入每步 fp8 成本**。
- 结论:量化把 GEMM-only 的 1.81–1.87× 降到含量化的 1.67–1.72×,**代价很小,fp8 依然大幅领先**。

## ★ 重要:这个 1.7× 是 GEMM-only —— 在融合 L2 里未必有 step 收益

**fc2 mxfp8 GEMM 的输出是 bf16**(scale 折进 f32 累加器,epilogue 存 bf16)。所以如果
**输出留 bf16、combine 也走 bf16**,那么:

- 融合 L2 是 **combine 带宽 bound(~2.7 ms)**;fc2 GEMM(fp8 0.19 / bf16 0.34 ms)在融合核里
  与 combine **重叠**(3-role,floor = max(gemm_role, combine_role) = combine)。
- fp8 GEMM 省的 ~0.15 ms 被藏在 combine 后面 → **L2 step 层面收益 ~0**。GEMM 从来不是 L2 瓶颈。
- 实测佐证:fc2 GEMM 只占 combine-bound L2 的 ~6–13%(见
  `NOTES_mxfp8_fused_gemm_combine_perf.md`)。

**结论:单独把 fc2 GEMM 换成 fp8、但输出/传输仍是 bf16 —— 没有优势。** fp8 GEMM 的价值只有在
**整条 L2 链都 fp8** 时才兑现:fp8 GEMM → **输出量化成 fp8(CShuffle epilogue,而非 bf16)** →
**fp8 combine(字节减半,combine 2.7→1.44 ms)** → fp8 reduce。也就是说,fp8 GEMM 必须和
**fp8-combine** 一起做才有意义(见 `grouped_gemm_combine_fp8_kernel.py` 的
`PT_FP8_COMBINE_GEMM=mxfp8` + `PT_FP8_COMBINE_CSHUF`)。EP8 实测:加上 fp8 GEMM 后,fp8-combine
从 0.92× 追到 ~1.00× vs bf16(见 performance_trend.md)—— 这才是 fp8 GEMM 起作用的地方。

（前向 L1 dispatch+fc1 不同:那里 fp8 GEMM 的输出直接喂 SwiGLU、且 dispatch 推的是 fp8 输入,
comm+compute 平衡,所以 fp8 L1 稳赢 1.4×。fc2/L2 是 combine-bound,规律相反。）

## ★★ L2 端到端实测:bf16 全链 vs fp8 全链(为什么"只换 GEMM"没用)

把 fc2 放回融合 L2(GEMM + 量化 + combine 传输 + reduce)来看。分两层:

### (a) 组件时间(isolation,未含 barrier)

| 环节                | bf16          | fp8                            | 性质                       |
| ------------------- | ------------- | ------------------------------ | -------------------------- |
| fc2 GEMM            | 0.34 ms       | 0.19 ms(含 act 量化 0.20 ms) | **compute-bound**,fp8 1.7× |
| combine 传输 + reduce | 2.7 ms        | 1.44 ms                        | **bandwidth-bound**,fp8 半字节 |
| **融合 L2 floor**   | max(0.34, 2.7) = **2.7** | max(0.19, 1.44) = **1.44** | GEMM 与 combine 在融合核里**重叠** |

关键:融合核里 GEMM 与 combine 是重叠的 role,step floor = max(两者)。bf16 下 combine(2.7)完全
盖住 GEMM(0.34);所以**动 GEMM 不动 combine → floor 仍是 2.7,step 不变**。

### (b) 端到端 step(EP8 T=8192 DSv3,chi2868,forward 延迟含内部 barrier,同批可比)

| # | 配置                                        | GEMM | 输出/传输 | combine | step ms | SNR      | vs bf16       |
| - | ------------------------------------------- | ---- | --------- | ------- | ------- | -------- | ------------- |
| ① | **bf16 全链**(production ref)              | bf16 | bf16      | bf16    | 8.35    | 24.9 dB  | 1.00×         |
| ② | **只换 GEMM=fp8**,输出/combine 仍 bf16     | fp8  | bf16      | bf16    | ~8.35*  | —        | **~1.00×(无收益)** |
| ③ | fp8 combine,bf16 GEMM(CShuffle 量化输出 fp8) | bf16 | **fp8**   | **fp8** | 9.05    | 23.96 dB | 0.92×         |
| ④ | **全链 fp8**(fp8 GEMM + fp8 输出 + fp8 combine) | fp8  | **fp8**   | **fp8** | 8.32    | 22.4 dB  | ~1.00×(tie) |

\* ② 未单独实测,是分析推算:L2 是 combine-bound、GEMM 与 combine 重叠、fc2 GEMM 仅占 L2 的
~6–13%,换 fp8 省的 0.15 ms 落在 combine floor(2.7 ms)以下 → step 不变。

### 读表结论(正是你的判断)

- **② vs ①:只把 fc2 换成 fp8、输出/传输还留 bf16 → step ≈ combine floor 不变 → 无收益。** GEMM 省的
  被 combine 藏住,fp8 GEMM 的 1.7× 作用在被重叠的非瓶颈环节上。
- **③:先把输出量化成 fp8 + combine 走 fp8(半字节)** → combine 2.7→1.44 ms,L2 由 combine-bound
  转成 **GEMM-role-bound**(CU sweep 证实:此时加 combine CU 反而更慢)。但 bf16 GEMM + 量化/reduce
  成本吃掉了字节收益 → 9.05 ms,0.92×(仍输 bf16)。
- **④:在③已变 GEMM-role-bound 的基础上,再把 GEMM 也换 fp8** → 这才是唯一有效的杠杆 → 8.32 ms,
  把 0.92× 追到 ~1.00×(tie)。**这就是 fp8 GEMM 真正起作用的地方 —— 必须整条链先 fp8。**

轨迹:0.76×(死路 butterfly 量化)→ 0.92×(CShuffle 量化 epilogue)→ ~1.00×(+ fp8 GEMM)。

**最终:全链 fp8(④)在 DSv3 EP8 打平优化后的 bf16(①),SNR 低 ~2.5 dB。** 这些形状下 bf16 仍是务实
选择(更简单、+2.5 dB);fp8 全链已可用,会在更 compute/comm-bound 的 regime(更大 tokens/expert)
或更少 barrier 开销下反超。**单独换 GEMM(②)在任何情形下都无用 —— 这就是"整条链 fp8 才有意义"的实测证据。**

## ★★★ 修正(隔离 L2-kernel 实测):fp8 全链其实是真 ~1.37× 净胜,不是"打平"

上面 (b) 的 ~1.00× "tie" 是 **whole-forward(L1+L2+共享大 barrier floor)** 口径 —— L2 的收益被
~8.3 ms 的 barrier-dominated 整前向稀释掉了,才看着像打平。为把 L2 单独量出来,给两个 combine kernel
加了编译期开关 `PT_COMBINE_NO_REDUCE`(把 reduce role 编译掉,只留 GEMM 产 L2Y + combine PUSH),
在**同一 harness、同一 L1 输出、同形状、每模式独立进程**下测 L2 本体:

| L2 段 | bf16 | fp8(mxfp8 GEMM + fp8 PUSH) | fp8 speedup |
| ------------------------------------ | ------ | -------------------------- | ----------- |
| **GEMM + PUSH**(no reduce,back-to-back 无 barrier) | 2.7205 ms | 1.9909 ms | **1.37×** |
| **FULL L2**(GEMM + PUSH + REDUCE,+1 barrier/iter) | 2.9834 ms | 2.1649 ms | **1.38×** |

(EP8 T=8192 DSv3,chi2868,max over ranks。FULL L2 每 iter 加一次 host barrier —— reduce 是跨 rank
producer-consumer,不加 barrier 会触发已知 reduce-flag liveness stall;barrier 对两模式等量,ratio 抵消。)

**结论(修正之前的说法):**
- **byte-lever 在 PUSH 上确实兑现:2.72→1.99 ms = 1.37×**(用户判断正确)。fp8 GEMM 输出直接量化成 fp8、
  XGMI 半字节 —— 产+传这半程是真赢。
- **reduce/dequant 不吃收益:** FULL 相比 PUSH 只多 ~0.26 ms(bf16)/ ~0.17 ms(fp8,含那次 barrier),
  reduce 侧 fp8 反而更省。之前"接收端 dequant 吃掉收益"的说法**是错的,dequant 几乎免费**(几条 cvt +
  折进 gate 权重的 scale 乘 + 只多读 1/32 scale 流)。
- **所以 fp8 L2 在 kernel 层面是真 ~1.37× / 省 ~0.73–0.82 ms**,push 和 full 都成立。之前的 "0.92×→~1.0× tie"
  只是 whole-forward barrier 稀释后的表观值,不是 L2 本身的真实比。

**这修正了 (b) 的判断:结论 ① fp8 全链不是"打平 bf16",而是在 L2 本体上净胜 ~1.37×。**之所以端到端(整前向)
看不出来,是因为 L2 只占 barrier-dominated 整前向的一小段。要把这 ~0.8 ms 兑现到 step,需要**削掉 L1↔L2
之间的 host barrier floor**(fp8_fused 前向里有多处 `sync + group.barrier()`),让 L2 的 kernel 收益不被
稀释。bench:`agent/workspace/mxfp8_nn_step1/rounds/round-4/_pushonly_bench.py`
(`PT_COMBINE_NO_REDUCE` / `BENCH_NO_REDUCE`)。

## 为什么 fp8 在 fc2 上稳赢(而反向 wgrad 不赢)

- fc2 是**大 N(H=7168)、大 K(I=2048)的 compute-bound GEMM** → fp8 的 ~2× 算力直接兑现(1293 vs 706 TF)。
- 对比:反向 wgrad(variable-K,tokens/expert~256,输出写主导 / small-K)是 memory-bound,fp8(bf16 输出)拿不到算力收益,mxfp8-Triton 反而 ~0.85× 输给 bf16。所以 **fp8 GEMM 的收益取决于是否 compute-bound**。

## 实现选择建议

- **fc2 用 mxfp8 FlyDSL(mega-local)**:1.87×(GEMM-only)/ 1.72×(含量化),per-1×32 精度稳(抗离群/padding),且**比 main 的 mxfp8 Triton 快**(Triton 只有它的 0.81×,换过去反而退化)。
- tensorwise FlyDSL(#407)更快(2.22×)但 per-tensor,遇 padded pool / 离群值精度崩(实测真实 pool dW2 = 0 dB),**不适合直接用在 MoE pool 上**。

## 相关文件

- fc2 mxfp8 GEMM 核:`primus_turbo/flydsl/mega/fp8/grouped_gemm_mxfp8_kernel.py`(`grouped_gemm_mxfp8_flydsl_kernel`)
- 量化:`primus_turbo/flydsl/mega/fp8/quant.py`(`quantize_rowwise_mxfp8` / `quantize_grouped_weight_mxfp8`)
- bf16 grouped GEMM:`grouped_gemm_impl`(Triton)
- 对比数据来源:`agent/workspace/mxfp8_nn_step1/logs/performance_trend.md`
