# 设计文档：FlyDSL 原生多波 dKV-intermediate kernel

> 目标：把 DeepSeek-V4 sparse-MLA backward 里唯一还在用 Triton 的核心 kernel
> `_bwd_compute_dkv_intermediate` 换成原生 FlyDSL，做到 ≥ gluon 的 dkv-interm。
> 背景：反向由 dQ(flydsl) + **dkv-interm(triton)** + dkv-gather(triton) 三段组成；
> dkv-interm 占 pro cr=4 反向的 ~39%（3.1ms/8.0ms kernel），且它随 triton wheel
> 版本剧烈波动（换 wheel 后 nw8 灾难 3.5x）——换成 flydsl 可去掉这个版本依赖。
> 平台：MI355X (gfx950, CDNA4)，bf16，flydsl 0.2.2。

---

## 0. 术语与维度（pro cr=4 为例）
- `T = B*S = 4096`（query token 数，也是 grid 主维）
- `H = 128`（heads；flash=64）
- `D_V = 512`（latent value dim）、`D_QK = 576`（含 64 rope pad，V4 里 rope=0 跳过）
- `TOPK = 1152`（padded；flash cr=4=640；小 topk case=128~192）
- 输出 `interm[T, TOPK, D_V] bf16`；pro cr=4 = 5.4GB > 2^31 bytes → **必须按 T 分块 launch**

---

## 1. kernel 要算什么（数学）
对每个 query token `t`，contract 掉 head 轴 H：
```
interm[t, key, d] = Σ_h ( Q[t,h,d] * dS[t,h,key] + dO[t,h,d] * P[t,h,key] )
                    key ∈ [0,TOPK)， d ∈ [0,D_V)
```
即两个 head-contraction GEMM 累加：
- `dKV_lora[d, key] = Qᵀ[d,h] @ dS[h,key] + dOᵀ[d,h] @ P[h,key]`
- 收缩维 = H（128），输出 [D_V=512, TOPK]。

`dS/P` 由 dQ kernel 已经算好并存在 HBM（[T,H,TOPK] bf16），**本 kernel 只读不重算**。
下游 `_bwd_dkv_gather_acc`（CSR 反向散射累加，保持 triton 不动）把 interm 按
inverted-topk 聚合到 `dkv[num_kv, D_V]`。

---

## 2. 为什么旧 flydsl port 失败（必须避开的坑）
`ca2a8f3` 的 port：**grid=(T,)，每 token 单 wave（64 lane）**，串行 stage 整个
qT/doT 转置（128 head × 512 d）到 LDS，用**标量 store**；D 分块（DB=128×4）还要
重复 re-stage dS/P。结果 22.6ms（triton 9.25ms 的 2.4x）。
根因（record 已证）：**单波 + 标量转置 staging 带宽不足**，D_BLOCK sweep 证实是
staging-bound 不是 compute-bound。

结论：**必胜关键 = 多波 + 向量化转置 staging + 用硬件 ds_read_tr 免手工转置**。
这正是 gluon 做的（4 warps + async_copy + ds_read_tr），也是我们 tr16 fwd / M=16 dq
已经验证过的同一套 recipe。本设计 = 把那套 recipe 套到 head-contraction 上。

---

## 3. gluon 参考做法（我们要对标/超越的基线）
`dsa_bwd_dkv_interm_gluon.py` 关键点：
- MFMA `instr_shape=[16,16,16] transposed=True, warps_per_cta=[4,1]` → **4 warp/CTA**，
  输出 [D_V, TILE_K]，收缩 BLOCK_H heads。
- Q/dO `[BLOCK_H, D_V]` 用 `async_copy.buffer_load_to_shared`（HBM→LDS DMA，**绕过 VGPR**），
  再 `smem.permute([1,0]).load()`（= ds_read_tr 硬件转置）读成 A-operand `[D_V, BLOCK_H]`。
- dS/P `[BLOCK_H, TILE_K]` 是 opIdx-1，直接寄存器 load + convert（**不转置**）。
- LDS 用 `PaddedSharedLayout with_identity_for([[512,16]])`（= 我们的 +pad 破 bank 冲突）。
- config：BLOCK_H=64/TILE_K=64/单缓冲/nw4。

我们的优势：tr16/dq 已经把 ds_read_tr 的 lane-map、+16 pad 破冲突、QK_PF 流水都调好了，
可直接复用；且 flydsl 能手工控制 waitcnt/prefetch 深度，理论上能压到 ≥ gluon。

---

## 4. 提议设计

### 4.1 网格与波映射（waves/SIMD 分析）
- **Grid = (T, )**：每 token 一个 workgroup，token 间完全独立（无 atomic，interm 每
  (t,key,d) 唯一 writer）。这点保持不变（正确且简单）。
- **每 workgroup 用 NW=4 waves**（对齐 gluon 4-warp；BLOCK_SIZE=256 threads）。
  - 4 waves 沿 **输出 D_V 维**切分：D_V=512 = 4 waves × 128 d each（每 wave 出 [128, TILE_K]）。
    → 各 wave 的输出 tile 不相交，无跨 wave reduction；收缩维 H 在 wave 内部做。
  - 备选切法：沿 **TILE_K（key）维**切 —— 但 D_V 切法让每 wave 的 A-operand(qT/doT)只需
    自己那 128 d 的列，LDS staging 量 /4，更省。**选 D_V 切分。**
- occupancy 估算：M=16 acc 一个 wave 的输出 [128 d, TILE_K] 若 TILE_K=64 →
  128×64 f32 = 8192/lane÷64lane… 实际 acc 按 MFMA 分块累加，见 4.4。目标 occ≥2（LDS 允许）。

### 4.2 MFMA 选择
- **`mfma_f32_16x16x32_bf16`**（我们全项目已用、已验证的指令，K=32/step）。
  - 对比 gluon 的 16x16x16：**16x16x32 每指令收缩 32 而非 16**，H=128 → 只需 128/32=**4 次
    MFMA**（gluon 8 次）。指令数减半是我们相对 gluon 的结构性优势（同 fwd QK 用的招）。
  - A-operand = qT/doT `[d=16, h=32]`（转置后，d 作 M，h 作 K）
  - B-operand = dS/P `[h=32, key=16]`（h 作 K，key 作 N）
  - C/acc = `[d=16, key=16]` f32，对应输出 interm[d,key]。
- rope：V4 全零 pad，`HAS_ROPE=False` → 跳过 Q_rope 的 MFMA 和 rope 列 store（同 triton/gluon）。

### 4.3 LDS 流水线设计
每 tile（一个 TILE_K 段 × 所有 head）需要：
1. **qT/doT staging**：Q/dO `[BLOCK_H_tile, D_V_wave=128]` 从 HBM → LDS。
   - **用 `raw_ptr_buffer_load_lds`（async DMA，绕 VGPR）**——这是旧 port 缺的关键。
     旧 port 用标量 VGPR store 卡带宽；DMA 直写 LDS 释放 staging 寄存器且异步。
   - LDS 布局 **行主序 [key_or_head, d] + pad 破 bank**：复用 tr16 的 **+16 pad**
     （V_STRIDE=D_wave+16）已验证把 ds_read_tr bank 冲突 52%→28%。
   - 读回用 `ds_read_tr16_b64` 硬件 4×4 转置 → 直接得到 A-operand `[d, h]` 布局，
     **免手工转置**（旧 port 最大的坑）。lane-map 直接抄 tr16 的 `_ds_read_tr_v4`。
2. **dS/P load**：`[BLOCK_H, TILE_K]` opIdx-1，**普通 vec load + convert，不进 LDS、不转置**
   （和 gluon 一样，B-operand 天然布局）。可直接从 HBM vec8 读到寄存器。
3. **软件流水**（gluon 是单缓冲；我们做双缓冲争取超越）：
   - 双缓冲 qT/doT 的 LDS tile：DMA(head-group g+1) 与 MFMA(head-group g) 重叠。
   - 因为是 async DMA（`s_waitcnt` 手控），MFMA 链能在 DMA 在途时推进——这正是 fwd
     里 DMA 对 M=32「有效」的同款场景（这里每 wave 计算量足够大能盖住延迟）。
   - prefetch 深度：H=128/BLOCK_H。dS/P 的寄存器 prefetch 复用 dq 的 QK_PF 全提前招式。
- LDS 预算（pro，每 wave D=128）：qT+doT 双缓冲 = 2buf × 2tensor × (BLOCK_H×(128+16)) × 2B。
  BLOCK_H=32 → 2×2×32×144×2 = 36.8KB/wave ×… 需按 NW 与 occ 复核 ≤160KB（见 4.6 风险）。

### 4.4 指令级并行（ILP）
- **两个 GEMM（qT@dS 和 doT@P）累加进同一个 acc**：它们无依赖，可交错发射
  （A-operand 不同、B-operand 不同、同一 C）——MFMA 之间填满，隐藏单条 MFMA 延迟。
- **head-group 循环展开**：H=128 分成 BLOCK_H 组，各组的 A-operand LDS 读 + MFMA
  用 range_constexpr 展开，A-operand 读全提前（同 fwd QK_PF=6 / dq full-hoist QK_PF=16 招）。
- **acc 分块**：输出 [D_wave=128, TILE_K] 拆成 (128/16)×(TILE_K/16) = 8×(TILE_K/16) 个
  16×16 acc；每个 acc 独立累加 → 天然多个 in-flight MFMA，ILP 充足。
- DMA(t+1) ‖ MFMA(t) ‖ dS/P-load(t+1) 三路重叠（fwd pipeline 已验证可行的三段式）。

### 4.5 正确性/落地约束（flydsl 特有坑，已知）
- **i64 偏移**：q/do/dS `T*H*TOPK`、interm `T*TOPK*D_QK` 都超 i32 元素范围 → 全用 i64 GEP
  （旧 port 已解决，抄过来）。
- **2^31 字节 tensor 上限**：interm 5.4GB → **按 T 分块 launch**（grid=(T_chunk,)，每块
  <2^31 B），kernel per-token 独立所以分块无副作用（旧 port 已实现该 launcher，复用）。
- **单行三元 / const_expr**：flydsl AST rewriter 不传播多行 if 重绑定 → 配置分支用单行
  三元或 const_expr（fwd 踩过）。
- **scf.IfOp 需自带 YieldOp**（fwd hoist-gather 踩过）。
- dS/P 的 `-1`(invalid key) 已在 dq kernel 里 mask 成 0，interm 端 valid mask 沿用。

### 4.6 风险与回退
| 风险 | 说明 | 缓解 |
|---|---|---|
| LDS 超预算 | 双缓冲 qT/doT + 多 wave 可能 >160KB | 先单缓冲（=gluon）跑通，再加双缓冲；BLOCK_H/TILE_K 可调 |
| DMA scatter 无收益 | fwd 里 DMA 对 M=16 无效（计算太小盖不住） | interm 每 wave 计算量大（H=128 收缩），类似 M=32「有效」区；但**必须实测**，无效则退回 coop vec-load 多波 staging（仍比旧单波强） |
| occ 仍=1 | acc 太大占满 VGPR | acc 是 fp32 输出必须驻 arch-VGPR，但输出按 D_V 切到 4 wave 后每 wave 只 128 d，acc 压力 /4，occ≥2 可期 |
| 小 topk 仍慢 | TOPK=128 时固定开销摊不开 | interm 对小 topk 本就便宜；保留 triton 版本做 fallback，按 TOPK 门控 |
| 换 wheel 波动 | —— | flydsl 手写不吃 triton autotune，**根除版本依赖**（本项目动机之一） |

---

## 5. 实施阶段（增量、每步可测）
1. **Stage A（对齐 gluon，先正确后快）**：grid=(T,)，NW=4 沿 D_V 切，单缓冲，
   `raw_ptr_buffer_load_lds` DMA staging + `ds_read_tr16_b64` 转置读 + 16x16x32 双 GEMM。
   目标：dkv SNR ≥54dB（对齐 triton 参考），速度 ≥ 旧 triton 默认。复用 tr16 的
   lane-map / pad / mfma helper。
2. **Stage B（双缓冲流水）**：qT/doT LDS 双缓冲，DMA(g+1)‖MFMA(g)；A-operand 全提前。
   目标：追平/超过 gluon interm（pro K1152 ~3.1ms → 目标 <2.8ms）。
3. **Stage C（调参 + 门控）**：BLOCK_H/TILE_K/NW 扫；按 (num_heads, TOPK) 选 flydsl vs
   triton fallback（小 topk 用 triton）。full-bwd 端到端验证 pro cr=4 ≥ gluon 8.5ms。
4. **Stage D（可选）**：与 dkv-gather 融合探索（interm 直接散射，省一趟 HBM 往返 5.4GB）——
   高天花板高复杂度，单列。

## 6. 验证
- 正确性：新 kernel vs triton_v2 参考 dkv SNR ≥50dB（复用已加的 fused-path 单测，
  再补一个直接测 interm 输出的单测）。全 6 个 fused case + 138 CSA 测试。
- 性能：`bench_v4_attention.py`（官方，含 pro cr=4 topk=1152）+ 逐 phase profiler
  （dq/interm/gather 分解）。**每 stage 前后都要在当前 triton wheel 上测**（gluon baseline
  会随 wheel 变）。
- PMC：LDS-wait/busy、bank-conflict%、occupancy、VALUBusy —— 对标 gluon interm。

## 7. 关键文件
- 新建：`primus_turbo/flydsl/attention/kernels/sparse_mla_v2/dsa_bwd_dkv_interm_kernel.py`
  （从 `ca2a8f3` 的旧 port 起手，但按本设计多波重写 staging）
- 改：`dsa_bwd.py`（`_get_interm_kernel` + 按 (H,TOPK) 门控 flydsl/triton + T-chunk launch）
- 参考：gluon `dsa_bwd_dkv_interm_gluon.py`；tr16 `dsa_fwd_tr16_kernel.py`（lane-map/pad/DMA）；
  dq `dsa_bwd_dq_m16_kernel.py`（full-hoist prefetch）；旧 port（i64/chunk launcher）。
- 相关 record：`dsa_bwd_dkv_interm_20260707.md`（失败根因）、`OPT_REPORT_dsa_flydsl_20260707.md`。

## 8. 预期与判据
- 成功 = pro cr=4 interm ≤ gluon 对应分量 且 full-bwd 追回 ≥ gluon（当前落后 0.90x）；
  同时**消除 triton 版本依赖**（换 wheel 不再波动）。
- 失败可能 = DMA scatter 对 interm 也无throughput 优势（同 fwd M=16 结论）→ 退回多波
  coop vec-load staging，仍应显著超旧单波 port，能否超 gluon 需实测定夺。
