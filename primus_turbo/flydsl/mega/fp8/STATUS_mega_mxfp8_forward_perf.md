# Mega MoE mxfp8 前向 —— 性能现状 (living doc,持续更新)

> **口径**: MI355X (gfx950),EP8,DeepSeek-V3 MoE (T=8192, H=7168, I=2048, E=256, K=8, BM=BN=256)。
> 除非注明,数字为 8 rank max-over-ranks。SNR 是 vs fp32 dense MoE 的 min-over-ranks(fp8 gate > 15 dB)。
> 机器: GPU 节点 **chi2761**(容器 `xiaoming-dev`,镜像 `tasimage/primus:pr-867`;之前用 chi2868)。
> 更新方式: 每次有新测量,在末尾 Changelog 追加一行,并同步更新对应表格 + "结论 TL;DR"。

---

## 结论 TL;DR (2026-07-12)

- **真·全链整前向:fp8(mxfp8 L1 + mxfp8 fc2 + fp8 combine)比 bf16(全 bf16)快 1.24× / 省 1.319 ms/step**
  (单步,chi2761 EP8 T=8192:bf16 6.791 ms / fp8 5.472 ms,SNR 31.63 vs 18.28 dB > 15 gate)。L1 1.4× +
  L2 1.34× 复合。见"整前向 真·全链对比"表。
- (旧口径,只差 combine、两边都用 fp8 L1:combine=fp8 比 combine=bf16 ~快 0.6ms —— L1 收益被抵消,数值偏小。)
- 之前 NOTES 里的"打平/略慢"是 **`_ep8_combine_bench` 连发 harness 的测量假象**(20 前向流水线连发、步间无同步、复用同批张量 → 跨 rank 争用专门拖累 fp8)。真实训练是一步一前向、步间有 comm/梯度同步 → 用单步口径。
- **combine 最优 = `fp8`**;`mxfp8`(fp8 GEMM + bf16 combine)和 `bf16` 已从前向删除。
- host barrier floor 很小(~0.27 ms),**不是**收益稀释源;`PT_MEGA_BARRIER_MODE=reduced` 可再省 ~0.15–0.2 ms(opt-in)。
- fc2+combine 的实现 = **mxfp8 GEMM + fp8 combine PUSH**,现在是**默认**(`PT_FP8_COMBINE_GEMM` 默认已从
  `bf16` 改为 `mxfp8`;设 `=bf16` 才回退到慢的 bf16-GEMM 对比路径)。之前的 footgun(不设 env 走 0.92× 慢变体)已消除。

---

## 1. 单步前向延迟(训练相关口径,最可信)

`_step_latency_bench.py`: global barrier 统一起点 → 1 前向逐 rank cuda events 计时(计时区内无尾 barrier)→ 50 trials 取 median。chi2868, 2026-07-10。

| barrier 模式 | fp8 (fp8 L1 + fp8 L2) | bf16 (fp8 L1 + bf16 L2) | **fp8 净胜** | fp8 SNR | bf16 SNR | straggler spread |
|---|---|---|---|---|---|---|
| full (shipped) | 10.671 ms | 11.346 ms | **−0.675 ms** | 20.32 dB | 24.94 dB | 0.065 / 0.111 ms |
| reduced        | 10.542 ms | 11.160 ms | **−0.618 ms** | 20.25 dB | 24.92 dB | 0.066 / 0.057 ms |

straggler spread(max−min over ranks)可忽略 → 没有跨 rank 掉队问题。fp8 win 在 full/reduced 下都稳。

## 2. 段拆解(in-context, drained;佐证 L2 kernel 真赢)

`_seg_time_bench.py`: 前向内 cuda events 分 L1 段(quant+dispatch+swiglu)/ L2 段(reset+barrier+combine),drained,max-over-ranks median。chi2868, reduced barrier。

| 模式 | L1 段 | **L2 段** | 合计 |
|---|---|---|---|
| fp8  | 5.102 ms | **2.263 ms** | 7.365 ms |
| bf16 | 5.233 ms | **2.991 ms** | 8.224 ms |

L2 段 fp8 比 bf16 快 **0.728 ms**,和隔离 L2 实测(下表)一致。

## 3. 隔离 L2 kernel(round-4,`_pushonly_bench.py`,PT_COMBINE_NO_REDUCE)

| L2 段 | bf16 | fp8 (mxfp8 GEMM + fp8 PUSH) | fp8 speedup |
|---|---|---|---|
| GEMM+PUSH (no reduce) | 2.7205 ms | 1.9909 ms | **1.37×** |
| FULL L2 (+reduce)     | 2.9834 ms | 2.1649 ms | **1.38×** |

## 4. 连发 harness(`_ep8_combine_bench.py`)—— ⚠️ 有假象,仅存档

20 前向流水线连发、步间无同步。**fp8 被跨 iter 争用拖累,不代表单步真值。** chi2868, 2026-07-10。

| barrier | fp8 | bf16 | mxfp8 combine (旧) |
|---|---|---|---|
| full    | 8.412 ms | 8.311 ms | ~13.132 ms |
| reduced | 8.261 ms | 8.116 ms | — |

## 5. host barrier floor 大小(`_rendezvous_cost_bench.py`,纯 rendezvous)

| 配置 | µs/forward |
|---|---|
| 4× rdv (+resets) 当前 full | 270.5 |
| 2× rdv (+resets) reduced   | 157.0 |
| 1× rdv                      | 50.8 |
| resets only (0 rdv)         | 13.7 |

整个 floor ~0.27 ms(非之前假设的 ~0.8 ms);reduced 省 ~0.11–0.15 ms。

---

## 完整前向实测(真实 op `mega_moe_fused_mxfp8`,含 x/w1 量化 + swiglu + 两个 mega-kernel)

`_function_fwd_bench.py`: common start + 逐 rank cuda events + 50 trials,max over ranks。
chi2761, full barrier, `PT_FP8_COMBINE_GEMM=mxfp8`, EP8 T=8192, 2026-07-12。

| 口径 | max-rank | median | straggler | SNR |
|---|---|---|---|---|
| inference (no_grad) | **6.671 ms** | 6.656 | 0.064 | 18.40 dB |
| training fwd (+ctx save_for_backward) | 6.621 ms | 6.604 | 0.058 | — |

- 完整前向 ~**6.67 ms**(2026-07-12 fused grouped 权重量化后;之前是 7.425 ms,fused-quant 省 ~0.75 ms/step)。
- `save_for_backward` 开销 ~0。x + w1 量化**每步都跑**(权重每 `optim.step()` 更新,真实训练不可缓存;fused kernel 是可兑现的省法,见已知问题)。

## 代码现状

- **module** `primus_turbo/pytorch/modules/moe/mega_moe_fp8.py`:`class MegaMoEFP8(MegaMoE)` ——
  继承 `MegaMoE` 的路由/权重(bf16 存储)/shared-expert/`forward`,覆写 `expert_compute` 调
  `mega_moe_fused_mxfp8`。**权重量化维护归 module**:`_weight_fp8` 把 w1/w2 的 grouped mxfp8 量化
  存在 module 上(`_w1_fp8`/`_w2_fp8`),按 `w._version` 失效(optim.step 后才重量),预量化结果传给 op
  (op 的 `w1_fp8`/`w2_fp8` 参数);bf16 权重仍作可导输入给 backward。只 token 相关的活(act 量化/prologue/
  dispatch+GEMM/swiglu/combine)每次 forward 跑。已导出。EP8 module fwd+bwd smoke 通过
  (y + dx/dW1/dW2/gate grad 均 finite,0 stall)。
- **真实 op** `primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py`(`MegaMoEFusedMxfp8Function` +
  `mega_moe_fused_mxfp8(group, x, ...)`):**前向流程已 inline 进 Function.forward**(镜像 bf16 的
  `MegaMoEFusedFunction`),不再套 flydsl helper。流程:quant(x/w1)→ prologue → scoreboard 清零
  (±barrier)→ `dispatch_grouped_gemm_mxfp8`(L1)→ swiglu → sb_l2/flags 复位(±barrier)→
  `grouped_gemm_combine_fp8`(L2)→ save_for_backward。`PT_MEGA_BARRIER_MODE` 开关搬到这里。
- **已删** `primus_turbo/flydsl/mega/fp8/mega_moe_fused_mxfp8.py`(旧 flydsl helper
  `mega_moe_fused_mxfp8_forward`);调用方(test/bench)已迁到真实 op。备份:
  `agent/workspace/mxfp8_nn_step1/rounds/round-5/kernel_snapshot/mega_moe_fused_mxfp8.py.flydsl-helper.bak`。
  - L1 = `dispatch_grouped_gemm_mxfp8`(comm 只剩 fused,decoupled/bf16 dispatch 已删)。
  - **L2 只有 fp8 combine**(`grouped_gemm_combine_fp8`);mxfp8/bf16 combine 已删。
  - `PT_MEGA_BARRIER_MODE ∈ {full(默认,与原始逐字节一致), reduced(opt-in ~0.15ms), none(不安全,仅探测)}`。
  - `PT_MEGA_TIME_SEG=1`: 段计时诊断。
- **L2 kernel** `grouped_gemm_combine_fp8_kernel.py`: `PT_FP8_COMBINE_GEMM`(**默认 `mxfp8`** = mxfp8 GEMM + fp8 PUSH;`=bf16` 回退对比)、`PT_FP8_COMBINE_CSHUF`(默认 on)、`PT_COMBINE_NO_REDUCE`(隔离用)。
- **量化 epilogue** `gemm_helper.py`: `StoreCQuantMxfp8CShuffle32`(Mfma32x32x16 layout,in-register mxfp8 量化)。

## 已知问题 / 待办

1. ~~`PT_FP8_COMBINE_GEMM` 默认 bf16~~ **已修(2026-07-12 k)**:默认改成 `mxfp8` → fc2+combine 默认走
   mxfp8 GEMM + fp8 PUSH,无需 env。默认(无 env)module smoke 通过,0 stall。
2. `combine="fp8"` 前向 SNR ~18–20 dB(比 bf16 低 ~4–6 dB),仍过 15 gate。若要更高,可查 CShuffle 量化精度。
3. 纯连发(步间无 barrier,如 400 前向)会触发**既有的 L1 scoreboard liveness stall**(`MEGA mxfp8 GEMM gate timeout`),full/reduced 都有,与 barrier 削减无关。是既有间歇 bug,barrier-free pipelining 前需修(scoreboard generation counter)。
4. 反向仍 bf16(见 `NOTES_mxfp8_backward_handoff.md`:backward comm/wgrad-bound,fp8 无收益)。
5. autograd Function `primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py` 走 `comm="fp8_fused"`(不传 combine → 现在固定 fp8 combine)。需端到端 e2e 精度门 + 训练 loss 对齐(见 `logs/optimize.md` 的 remaining work)。

## 整前向 真·全链对比(bf16 op vs fp8 op,单步)

`_whole_fwd_compare_bench.py`(global-barrier common start + 逐 rank cuda events,max over ranks,
no_grad)。chi2761 EP8 T=8192 DSv3, 2026-07-12。**真正两边 L1+L2 都不同**(不是之前只差 combine 的口径)。

| 整前向 | 实现 | 单步延迟 | SNR |
|---|---|---|---|
| bf16 | bf16 dispatch+fc1 + bf16 fc2+combine (`mega_moe_fused`) | 6.791 ms | 31.63 dB |
| **fp8** | mxfp8 dispatch+fc1 + mxfp8 fc2 + fp8 combine (`mega_moe_fused_mxfp8`) | **5.472 ms** | 18.28 dB |
| | **fp8 净胜** | **−1.319 ms (1.24×)** | −13 dB(> 15 gate) |

拆解一致:L1 fp8 fused 2.56 vs bf16 3.55(−1.0),L2 fp8 1.97 vs bf16 2.63(−0.66)→ ~1.66ms,和整前向
−1.32ms 同量级(重叠/overhead 差)。fp8 那次曾触发间歇 reduce-flag stall,重试即过(待根治)。

## Changelog

- **2026-07-12 (m)**: 真·全链整前向对比(`_whole_fwd_compare_bench.py`):bf16 op 6.791 ms/31.63 dB vs
  fp8 op **5.472 ms/18.28 dB → fp8 1.24× / 省 1.319 ms/step**(单步,chi2761 EP8 T=8192)。这是两边 L1+L2
  都不同的口径(区别于之前只差 combine 的~打平)。fp8 首跑触发间歇 reduce-flag liveness stall(8 worker
  101% 自旋),重试 0 stall 通过 —— 稳定性待根治。
- **2026-07-12 (l)**: 修一个我引入的临时退化 + 确认 L2 数。之前把 w2 改成 module 传 `w2_fp8=(w2q,w2s)`(只量化)
  导致 combine 每次调用重跑 `preshuffle_b_scale`(~1ms,原本和量化一起缓存)→ fp8 L2 GEMM+PUSH 从 1.99→3.02ms。
  **修法**:新增 `prepare_w2_fp8`(量化+preshuffle+int8 flat 一次产出),module 版本缓存**整份 prepared w2**
  `(weight_flat, b_sp)`;combine 的 `w2_fp8` 改收 prepared 形态,不传时走内部 version-keyed 缓存(缓存
  quant+preshuffle)。重测(chi2761 EP8 T=8192,GEMM+PUSH no-reduce):**bf16 2.630 / fp8 1.966 ms(fp8 1.34×)**,
  回到 round-4 水平(1.37×)。module smoke(prepared 路径)finite,0 stall。
- **2026-07-12 (k)**: fc2+combine 的实现定为 **mxfp8 GEMM + fp8 combine PUSH** 并设为默认:
  `grouped_gemm_combine_fp8_kernel.py` 的 `PT_FP8_COMBINE_GEMM` 默认 `bf16`→`mxfp8`(两处:`_compile` +
  host wrapper)。不设任何 env,`MegaMoEFP8`/op 就走 mxfp8 GEMM + fp8 PUSH(最优);`=bf16` 才回退慢路径对比。
  消除了之前"不设 env 走 0.92× 慢变体"的 footgun。默认(无 env)EP8 module smoke 通过,y+grad finite,0 stall。
- **2026-07-12 (j)**: L1 的 x(activation)rowwise 量化也换成 FlyDSL(`quantize_rowwise_mxfp8_flydsl`,
  scale 视图 e8m0 匹配契约)。至此 `MegaMoEFP8` 全路径量化都走 FlyDSL(x + act + w1 + w2);act 早已是。
  x 量化很小(~0.05ms),主要为一致性。module smoke:finite,0 stall。
- **2026-07-12 (i)**: FlyDSL 权重量化。新增 `quantize_grouped_weight_mxfp8_flydsl`(reshape [G,N,K]→
  [G*N,K] + 复用 `quantize_rowwise_mxfp8_flydsl`,scale 视图成 e8m0 匹配返回类型),**bit-identical**
  于通用版但 **~2.6-2.9x 快**:w1 [32,4096,7168] 1249→486 µs,w2 [32,7168,2048] 661→230 µs
  (~2.3→5.9 TB/s,近 HBM 峰值)。`MegaMoEFP8._weight_fp8` + `quantize_grouped_weight_mxfp8_cached`
  (E4M3)改走它。端到端(module,accum=8,EP8 T=8192,chi2761):micro-1 权重量化段 ~2.08→0.74 ms,
  effective per-fwd 8.869→8.736 ms,SNR/grad finite,0 stall。通用 `quantize_fp8` 只 ~2.3 TB/s 是主要空间。
- **2026-07-12 (h)**: 梯度累积口径验证(`_gradaccum_fwd_bench.py`,MegaMoEFP8,accum=8=GBS16/MBS2/DP1,
  EP8 T=8192,chi2761,module 整前向含 routing):micro-1(含权重量化)10.691 ms,micro-2..8(命中,不重量)
  8.609 ms,有效每-fwd 8.869 ms。权重量化 ~2.08 ms 只在每 step 第 1 个 micro 发生 → 摊销到 1/8(省
  ~1.8 ms/fwd vs 每 fwd 都量)。证实"权重每 optim.step 量一次、跨 micro-batch 复用"。0 stall。
- **2026-07-12 (g)**: 权重量化维护上提到 `MegaMoEFP8`(module 拥有)。module 的 `_weight_fp8` 按
  `w._version` 维护 w1/w2 的 grouped mxfp8 量化(存 `self`,version 变才重量),经 op 新增的
  `w1_fp8`/`w2_fp8` 参数传入;`MegaMoEFusedMxfp8Function.forward` 收到则跳过内部量化(backward 返回值
  相应加 2 个 None),`grouped_gemm_combine_fp8` 新增 `w2_fp8`(给了就跳量化,仅保留 kernel 内的
  preshuffle 布局)。op 独立使用时(不传)仍内部量化,向后兼容。EP8 module fwd+bwd smoke 通过,0 stall。
  好处:每 step 权重量一次、跨 grad-accum micro-batch 复用;不变量不再散在 op/kernel。
- **2026-07-12 (f)**: 新增 `MegaMoEFP8(MegaMoE)` 模块(`modules/moe/mega_moe_fp8.py`)—— 单独一个
  子类,只覆写 `expert_compute` 走 `mega_moe_fused_mxfp8`(bf16→fp8 的区分用"独立类"而非 flag)。
  EP8 module fwd+bwd smoke 通过(y + dx/dW1/dW2/gate grad 均 finite,0 stall)。symm 走全局单例复用、
  权重量化走 on-tensor version cache,所以不变量已不重复;module 持有 symm 的进一步托管待需要时再加。
- **2026-07-12 (e)**: 权重量化处理(完整 fp8 方案铺垫)。**#1**:新增 `quantize_grouped_weight_mxfp8_cached`
  —— **把 fp8 量化结果直接缓存在权重张量上**(属性 `_mxfp8_grouped_q`,key = `w._version`):版本变了才
  重量化,没变直接复用;第一次必量。per-weight 存储 → 自动按层扩展、无全局 dict/LRU/上限、权重释放时一起释放。
  w1 量化改走它。验证:bit-identical、同版本 HIT、in-place 更新→版本 bump→MISS 重量化(无 stale)。
  **#2**:修 `grouped_gemm_combine_fp8_kernel.py` 的 `_L2_BSP_CACHE` —— key 从 `data_ptr` 加上
  `_version`,消除 in-place 更新权重后返回 stale fp8 w2 的正确性隐患。chi2761 smoke:SNR 18.31 dB,
  0 stall。⚠️ **诚实口径**:这两个 cache 在**标准训练(无梯度累积)里每步都 miss**(权重每 optim.step
  更新),所以标准训练完整前向仍是 ~6.67 ms(fused kernel 才是可兑现的每步省点);cache 只在**梯度累积**
  或**等 backward 转 mxfp8 后 fwd/bwd 共用**时兑现。**未做**(待 fp8 backward):权重第二布局(转置)+
  `ctx.save_for_backward` 让 fwd/bwd 真正共用同一份量化权重。forward-only bench 会因 cache 命中显示更低
  (~5.5 ms),那是 benchmark/grad-accum 口径,勿当标准训练收益。
- **2026-07-12 (d)**: `quantize_grouped_weight_mxfp8` 从 `G=32` Python 循环 + 2×`torch.stack` 换成
  **单 kernel**(`[G,N,K]→[G*N,K]` 一次 rowwise 量化再 reshape,rowwise 沿 K 与 group 边界无关 →
  bit-identical)。单 GPU:w1 [32,4096,7168] 1.94→1.19 ms(1.6×),w2 [32,7168,2048] 1.00→0.67 ms
  (1.5×),q/s 均 bit-identical。完整前向(chi2761 EP8 T=8192 full PT_FP8_COMBINE_GEMM=mxfp8):
  inference 7.425→**6.671 ms**(省 ~0.75 ms/step),SNR 18.40 dB,0 stall。⚠️ 权重每 `optim.step()`
  更新→真实训练每步重量化不可缓存(除梯度累积);fused kernel 是可 1:1 兑现的省法。完整 fp8 方案里
  权重应量化两布局(fwd 沿一轴 + bwd-dgrad 沿转置轴),用 `ctx.save_for_backward` 让 fwd/bwd 共用。
- **2026-07-12 (c)**: 把前向流程从旧 flydsl helper `mega_moe_fused_mxfp8_forward` **inline 进
  `MegaMoEFusedMxfp8Function.forward`**(镜像 bf16),删除该 helper,迁移 test/bench 到真实 op
  `mega_moe_fused_mxfp8`。新增 `_function_fwd_bench.py` 测完整前向:chi2761 EP8 T=8192 full barrier
  PT_FP8_COMBINE_GEMM=mxfp8 → inference 7.425 ms / training-fwd 7.427 ms(save_for_backward ~0)/
  SNR 18.12 dB。与 drained 段拆解和(7.365)一致。备份 `*.flydsl-helper.bak` / `*_Function.py.pre-inline.bak`。
- **2026-07-12 (b)**: 前向收敛为**单一 all-fp8 路径**:删 decoupled `comm="fp8"` 分支 + `comm` 参数(只剩融合 L1),删随之变死的 fallback L1(`grouped_gemm_mxfp8_flydsl_kernel`)+ `dispatch_fp8_push`/`l2_invalidate_all` import。同步更新调用方(autograd Function / `test_mega_moe_mxfp8.py` 去 comm 参数化 / `bench_mega_moe_mxfp8.py`)。chi2761 smoke:EP8 T=8192 finite,min SNR 18.45 dB > 15。备份 `*.pre-fusedonly.bak`。前向从 236 → 175 行。
- **2026-07-12 (a)**: 前向收敛为 fp8-only combine(删 `combine="mxfp8"` 分支 + `combine` 参数;之前已删 bf16 dispatch/combine)。chi2761 smoke 验证:EP8 T=8192 单前向 finite,min SNR 18.18 dB > 15。备份于 `agent/workspace/mxfp8_nn_step1/rounds/round-5/kernel_snapshot/*.bak`。
- **2026-07-10**: 单步口径定论 —— fp8 前向净胜 bf16 ~0.6–0.7 ms/step(full+reduced);"打平"是连发 harness 假象。测得 barrier floor ~0.27 ms(非 ~0.8 ms);段拆解证实 L2 段 fp8 −0.73 ms。加 `PT_MEGA_BARRIER_MODE`/`PT_MEGA_TIME_SEG` 诊断开关。详见 `agent/workspace/mxfp8_nn_step1/rounds/round-5/summary.md`。
