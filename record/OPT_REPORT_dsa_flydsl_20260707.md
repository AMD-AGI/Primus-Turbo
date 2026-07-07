# FlyDSL DeepSeek-V4 sparse-MLA attention 优化报告

> 目标：优化 Primus-Turbo 中 FlyDSL 后端的 DeepSeek-V4 CSA (Compressed Sparse
> Attention / fused single-latent sparse-MLA) fwd+bwd，大幅超过 gluon backend。
> 平台：MI355X (gfx950, CDNA4)，bf16，flydsl 0.2.2。
> 测量：`output/bench_3way.py`（真实 adapter 结构化 topk，sink on，gluon 已 warm），
> 三个生产 shape：H128/K512 (V4-Pro)、H64/K512 (V4-Flash)、H128/K2048。

---

## 一、最终结果（vs 全 warm 的 gluon_v2，也全面超 triton_v2）

### Forward (TFLOP/s，越高越好)
| shape | flydsl(tr16) | triton_v2 | gluon_v2 | flydsl/gluon |
|---|---:|---:|---:|---:|
| H128 K512  | 539 | 381 | 416 | **1.30x** |
| H64  K512  | 492 | 377 | 414 | **1.19x** |
| H128 K2048 | 578 | 403 | 440 | **1.31x** |

### Backward (ms，越低越好)
| shape | flydsl | triton_v2 | gluon_v2 | flydsl/gluon |
|---|---:|---:|---:|---:|
| H128 K512  | 8.30  | 9.28  | 8.45  | **1.02x** |
| H64  K512  | 5.25  | 6.85  | 6.93  | **1.32x** |
| H128 K2048 | 19.75 | 26.55 | 24.00 | **1.22x** |

### Combined fwd+bwd (ms)
| shape | flydsl | gluon | combined/gluon |
|---|---:|---:|---:|
| H128 K512  | 10.17 | 10.88 | **1.07x** |
| H64  K512  | 6.28  | 8.15  | **1.30x** |
| H128 K2048 | 24.06 | 29.67 | **1.24x** |

正确性：所有路径 out SNR vs M32 参考 99 dB / vs gluon 81-82 dB；144 CSA 测试全通过。

**结论：forward / backward / combined 三个维度、全部 3 个生产 shape 上都反超 gluon。**

---

## 一·B、官方 Primus benchmark 交叉验证（bench_v4_attention.py，2026-07-07）

上表用的是自建 `output/bench_3way.py`（K512/K2048 随机 topk）。用 **Primus 官方
`deepseek-v4/benchmark/bench_v4_attention.py`**（结构化 `[SWA window ++ pool]` topk，
flash H64/pro H128，cr=0 SWA / cr=4 CSA / cr=128 HCA）复测，把我们的 Primus-Turbo
后端接成 `flydsl_turbo` 列（S=4096, warmup15/iters40, CUDA events）。结果：

| case | topk | fwd turbo | fwd gluon_v2 | bwd turbo | bwd gluon_v2 | 判定 |
|---|--:|--:|--:|--:|--:|---|
| flash cr=0 (SWA) | 128  | 0.31/222 | 0.30/226 | 1.35/127 | 1.19/145 | fwd平 / **bwd 输** |
| flash cr=4 (CSA) | 640  | **0.73/468** | 0.90/383 | **3.71/232** | 5.16/167 | **全赢** |
| flash cr=128(HCA)| 160  | 0.36/236 | 0.38/227 | 1.91/113 | 1.74/124 | fwd赢 / **bwd 输** |
| pro cr=0 (SWA)   | 128  | 0.54/254 | 0.56/247 | 2.15/160 | 1.78/193 | fwd赢 / **bwd 输** |
| pro cr=4 (CSA)   | 1152 | **2.09/591** | 2.73/453 | **9.34/331** | 9.47/327 | **全赢** |
| pro cr=128 (HCA) | 160  | 0.65/264 | 0.68/254 | 2.95/146 | 2.50/172 | fwd赢 / **bwd 输** |

**关键结论（重要修正）**：
1. **CSA（cr=4）是本项目的主目标，turbo 在 flash/pro 上 fwd+bwd 全面反超 gluon_v2**
   （pro fwd 1.30x，flash bwd 1.39x）。前向所有 case 都 ≥ gluon。
2. **但小 topk 的 backward（cr=0/128，topk 128-192）turbo 落后 gluon ~10-20%**。
   根因：dq kernel + dkv-interm 在 topk 极小时固定开销摊不开（gluon 结构对小 topk 更友好）。
   这是自建 bench 没覆盖到的短板（它只测大 topk）。
3. **正确性坐实**：turbo 与**同仓库 Triton v2 参考**在全部 shape SNR 51-63 dB 一致
   （含 pro cr=4 topk=1152，dkv 54 dB）。曾出现的 "dkv vs gluon 5 dB" 是 **gluon dkv
   累加约定不同**（norm 1522 vs 887，triton 也同样 -4.5 dB 对不上 gluon）——是 gluon 的
   convention artifact，不是 turbo bug。已加 6 个 fused-path 单测固化（144 测试全过）。

**用户先前看到的表格（turbo bwd pro cr=4 = 17.08/181 落后）是本会话 bwd 修复之前的旧状态**；
当前代码 turbo bwd pro cr=4 = 9.34/331，已反超。fwd 数字与旧表一致（fwd win 更早已提交）。

**遗留短板**：小-topk（SWA/HCA）backward 落后。若要全 case 反超，需针对 topk≤192
优化 dq/dkv-interm 的固定开销（当前调优点是为大 topk 选的）。

---

## 二、演进轨迹（起点 → 终点）

| 里程碑 | fwd (H128K512) | bwd (H128K512) | vs gluon |
|---|---|---|---|
| 起点 M=32 default | 278 TF | 0.58-0.64x | 全面落后 |
| tr16 shared-gather (基础架构) | 375 TF | — | fwd 0.90x |
| M=16 dq + interm + delta 系列 | — | 2/3 shape 反超 | — |
| **本报告的 5 个 win** | **539 TF** | **8.30ms (1.02x)** | **全面反超** |

---

## 三、核心架构（前置基础，非本次新增但支撑一切）

**tr16 kernel** = M=16 (16x16x32 MFMA) + **per-workgroup 共享 kv tile** + **ds_read_tr
硬件转置**（第二个 GEMM）。
- 关键洞察：topk 只依赖 token，同一 CTA 内所有 wave gather 相同 kv 行 →
  只 gather 一次到共享 LDS，QK A-operand 从 LDS 读（消除冗余 per-wave HBM）。
  这一条把 naked-M16 从 131 → 373 TFLOP/s。**冗余 HBM 流量才是当初真瓶颈，不是占用率。**
- 第二个 GEMM（fwd 的 PV / bwd 的 dQ=dS·K）读同一行主序共享 tile，走 `ds_read_tr16_b64`
  硬件 4x4 转置。
- 文件：`primus_turbo/flydsl/attention/kernels/sparse_mla_v2/dsa_fwd_tr16_kernel.py`（fwd）
  与 `dsa_bwd_dq_m16_kernel.py`（bwd dq）。均为**生产默认**（heads%16==0 && topk%32==0）。

---

## 四、本次会话的 5 个优化（按 commit 顺序）

### 1. `67a7d44` — fwd QK_PF 3→6（H128 前向 +7%）
- **做了什么**：QK GEMM 的 A-operand LDS 预取深度从 3 提到 6。
- **为什么有效**：QK_PF=3 时寄存器分配溢出 74 VGPR（occ=1）；QK_PF=6 时 spill=0，
  更深的预取填满 LDS-load→MFMA 流水（H128 是 VGPR-bound、occ=1）。
- **效果**：H128 K512 376→404，K2048 388→420（~+7%）；H64 持平；QK_PF≥8 因 VGPR 压力回退 H64。
- **文件**：`dsa_fwd_tr16_kernel.py`（`_QK_PF` 默认）。

### 2. `59e46aa` — bwd dkv-interm 调参 BH32/TK64 → BH16/TK128/nw8（bwd +14-25%）
- **做了什么**：flydsl bwd 复用的 Triton dkv-intermediate kernel（占 bwd ~43%）的 launch 参数
  从 BLOCK_H=32/TILE_K=64 改为 BLOCK_H=16/TILE_K=128/num_warps=8。
- **为什么有效**：扫参发现原默认非最优；小 BLOCK_H + 大 TILE_K 更契合该 kernel 的
  head-contraction 结构。TK128 需 TOPK%128==0（K512/K2048 都满足），否则 fallback TK64。
- **效果**：interm 单 kernel H128K512 1669→1338us，H64 1059→800us，K2048 6630→5730us。
  全 bwd：H128K512 8.99→8.28ms（**0.94x→1.02x，反超 gluon**），H64 5.75→5.25，K2048 20.9→19.75。
  **backward 由此从 2/3 反超变为 3/3 全反超。**
- **文件**：`dsa_bwd.py`（`_tk_default` / `BH_DKV` / `_NW_DKV`）。

### 3. `532859b` — fwd 两阶段 coop gather（H64 前向 +13%）
- **做了什么**：共享 kv gather 从"每次迭代 topk读→kv读→LDS存"的依赖链，拆成
  "先发射全部 VMEM load，再做全部 LDS store"两阶段。
- **为什么有效**：H64 的 workgroup 只有 4 waves（256 线程），但 gather 是固定 2048 元素
  tile → GATHER_ITERS=8，per-iter 依赖链串行化 VMEM 延迟。两阶段暴露大量在途 load。
- **效果**：H64 前向 331→374 TFLOP/s（0.80x→0.90x gluon）。
- **门控**：NUM_WAVES≤4（H128 是 8 waves 已线程饱和，hoist 反因活跃 VGPR 增多回退 404→396）。
- **坑**：hoist 的 store IfOp 必须自带 `scf.YieldOp([])`，否则 MLIR verifier 报 no terminator。
- **文件**：`dsa_fwd_tr16_kernel.py`（`_HOIST_GATHER`）。

### 4. `22d3f06` — fwd p-region LDS stride pad +4（全 shape +2-3%）
- **做了什么**：softmax 结果 p 在 LDS 的行 stride 从 BLOCK_K=32 pad 到 36。
- **为什么有效**：32 是 2 的幂，相邻 head（lane_mod_16 行）别名同 LDS bank。
- **效果**：H64 376→386，H128 404→413，K2048 420→429。冲突率 52.5%→49.1%（p 非主因）。
- **文件**：`dsa_fwd_tr16_kernel.py`（`P_STRIDE`）。

### 5. `24e4eea` — fwd kv-tile LDS stride pad +4→+16（前向 +30%，THE 关键 win）
- **做了什么**：共享 kv tile 的行 stride 从 HEAD_DIM+4 (516) 改为 HEAD_DIM+16 (528)。
- **为什么有效**：**推翻了历史上"fwd occ=1 结构性天花板"的错误结论。** rocprofv3 PMC 显示
  H64 前向是 **LDS-stall-bound**（LDS-wait/busy=2.0 vs gluon 0.37，bank 冲突率 52.5%）。
  原 +4 pad 对 ds_read_tr 转置读的访存 stride 远远不够；+16 把冲突降到 27.6%、
  LDS-wait/busy 降到 1.19。
- **效果**：H64 385→499，H128 412→542，K2048 430→578（**~+30%，一举反超 gluon**）。
  扫参：+16 最优（与 +48 平但省 LDS）；+64（2 的幂）大幅回退；+0 崩到 223（全冲突）。
  kv tile @+16 = 32×528×2 = 33.8KB << 160KB LDS。
- **文件**：`dsa_fwd_tr16_kernel.py`（`_v_pad` / `V_STRIDE`）。

### 附：`f5cd293` — chore，把 `output/` 开发脚本从 git index 移除（已在 .gitignore）
无功能影响；`output/` 只是手动跑 benchmark 的 harness，无任何生产代码 import。

---

## 五、关键方法论 / 经验

1. **占用率不是万能解释。** 多个历史会话把 fwd 慢归因为"occ=1 天花板"并放弃，
   实际用 PMC 一测才发现是 **LDS bank 冲突**（52% 冲突率）。VGPR 一直是 256/occ=1，
   但那从来不是瓶颈。**先测 PMC，别信旧 handoff 的归因。**
2. **LDS stride 的 pad 量要按访存 pattern 扫，不能拍脑袋。** +4 曾被认为够了，
   实际对 ds_read_tr 转置读要 +16 才有效；2 的幂 stride（32/64）必然冲突。
3. **门控要看 workgroup 结构。** 两阶段 gather 对 H64（4 wave 线程饥饿）+13%，
   对 H128（8 wave 饱和）反而回退——同一改动按 NUM_WAVES 门控。
4. **复用的 Triton kernel 也要自己扫参。** dkv-interm 沿用上游默认参数，实测非最优，
   一次调参拿到 14-25%。
5. **验证可信度**：gluon 用 @triton.autotune，必须充分 warm（否则虚低）。已确认其
   autotune 收敛到 BLOCK_H=64/TILE_K=32/stages=3（= PR#853 记录的最优 config），
   并用 CUDA events + 重预热 + 5×100 迭代取中位独立复测，方差 <0.3%。

---

## 六、剩余机会（未做，递减价值）

- **fwd 残留 LDS 冲突**：LDS-wait/busy 仍 1.19 vs gluon 0.37，冲突率 27.6%。残留疑为
  QK A-operand 的 vec8 读；可试对 kv 列做 XOR swizzle 进一步消除（但 fwd 已 1.19-1.31x，收益递减）。
- **bwd H128K512 仅 1.02x**：dQ + dkv-interm 都已在调优点；再进一步需 gluon 式
  BLOCK_H=64 async-K 流水重写（大改）。
- **已确认死路**（勿重试）：fwd DMA gather（raw_ptr_buffer_load_lds，topk-scattered 无吞吐优势）、
  softmax‖QK 2-buffer pipeline（同步 coop gather 的 barrier 抵消收益）、
  occ=2（Q-in-LDS / waves-per-eu，acc[16,512]f32 必须常驻 arch-VGPR）、BLOCK_K=64（崩）。

---

## 七、环境与复现

```bash
pip install --force-reinstall --no-deps flydsl==0.2.2   # 必须，否则不编译
export HIP_VISIBLE_DEVICES=0                             # 隔离，避免其他 workspace 争 GPU
python output/bench_3way.py                              # 三方 fwd+bwd 对比（自动 warm gluon）
```
生产默认已开启全部 win（无需 env）。调参 env：`PRIMUS_DSA_TR16_V_PAD`(默认16)、
`PRIMUS_DSA_TR16_P_STRIDE`(36)、`PRIMUS_DSA_TR16_QK_PF`(6)、`PRIMUS_DSA_TR16_HOIST_GATHER`
(NUM_WAVES≤4 自动)、`PRIMUS_DSA_DKV_BH/TK/NW`(16/128/8)。
