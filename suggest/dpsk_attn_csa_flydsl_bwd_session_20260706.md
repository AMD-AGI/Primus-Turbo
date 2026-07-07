# DeepSeek-V4 CSA FlyDSL 反向实现与优化 —— 会话工作记录（2026-07-06）

> 范围：AMD CDNA4（gfx950 / MI355X），DeepSeek-V4 CSA from-pool 注意力，FlyDSL 后端（`flydsl 0.2.2`）。
> 目标：为 CSA from-pool 补齐 FlyDSL 反向（此前只有前向，反向回退 Triton），并优化以逼近/超过 Triton。
> 结论：**反向已从无到有，正确且接入，本轮优化 2.7–8.9x；尚未整体超过 Triton**（瓶颈与后续方案见 §5）。

---

## 1. 本轮完成的工作

### 1.1 新增 FlyDSL CSA from-pool 反向（此前完全缺失 → 回退 Triton）

- **新内核** `primus_turbo/flydsl/attention/kernels/csa_pool_bwd_kernel.py`
  - `build_csa_pool_bwd_module`：per-row 反向内核（一个 wave / program，grid `(Sq, B*H)`）。
    从保存的 fp32 LSE（raw `qk*scale` 域）+ preprocess delta 重建联合 online-softmax 行，
    产出 `dq`（直接 fp32 store）、`dk_local` / `dv_local` / `dpool` / `dsink`（原子 scatter-add）。
    原子用 `llvm.AtomicRMWOp(fadd, monotonic)` 走 i64-GEP 指针（避开 buffer 2GB voffset 上限）。
    带 `has_local` / `has_sparse` / `store_dpool` const_expr 开关，支持只编译某个分支。
  - `build_csa_pool_bwd_dpool_module`：专用 dpool-only 内核（见 §1.3）。
- **launcher** `csa_pool_attention_bwd_flydsl_kernel`（`deepseek_attn_bwd_kernel.py`）。
- **dispatcher** `DeepSeekCSAPoolAttnBwdDispatcher` + `deepseek_csa_pool_attn_bwd`
  （`deepseek_attn_impl.py`）；autograd `CSAPoolAttentionFn.backward` 改为经此分发
  （`csa_attention.py`，保存 `ctx.backend_override`）。
  `can_handle` 门槛：gfx950 / bf16 / D==512 / K_H∈{1,H} / swa>0 / K_topk>0 / scale==1/√D / numel<2³¹。

### 1.2 反向拆分优化（复用现成快内核）

CSA 联合 softmax 可按 local SWA 流 + sparse pool 流分解（二者共享联合 LSE / delta，
与现有 `_hca_split_mask_bwd` 同构）：

- **MQA 路径**（K_H==1，V4 生产布局）：
  - local 流复用**已有的 MFMA SWA 反向内核** `build_swa_bwd_dq_module`（+dsink）与
    `build_swa_bwd_dkv_module`，喂联合 LSE → dq_local / dk_local / dv_local / dsink。
  - sparse dq 用 per-row 内核（`store_dpool=False`，无原子，~7ms）。
- **MHA 路径**（K_H==H，benchmark 布局）：SWA 内核仅支持 MQA，故 dq/dk/dv 用 per-row
  monolithic 内核（`store_dpool=False`）。

### 1.3 dpool 原子竞争的关键修复（反向最大瓶颈）

- **诊断**：dpool 的 scatter-add 是全部开销来源。`Sq*H*K ≈ 67M` 次原子加全部砸向
  `[B,P,D]` 里 P=512 个池行 → 严重串行。实测关掉 dpool 原子后内核从 2040ms → 7ms。
- **方案**（对应 Triton 的 partial+segreduce 思路）：新增专用 dpool 内核
  `build_csa_pool_bwd_dpool_module`，grid `(K/KB, Sq, B)`，**在寄存器内对所有 head 求和**
  单个 (b,m,k) 的 dpool 贡献，然后**无原子**平铺 store 到 partial buffer `[B,Sq,K,D]`
  （每个 (b,m,k) 唯一属主），host 侧用 `index_add` 归约到 `dpool[B,P,D]`。
  → dpool 从 2040ms 降到 **296ms**（kernel 本体），归约仅 ~1.8ms。

### 1.4 前向 head-as-M MFMA 构件（已验证正确，未接线）

- **新内核** `primus_turbo/flydsl/attention/kernels/csa_pool_sparse_fwd_kernel.py`
  - 关键洞察：`topk_idxs` 是 `[B,Sq,K]`，**与 head 无关** → 同一 query 的所有 head gather
    相同的池行，于是 query-head 轴是真正的 MFMA M 维。`qk[BLOCK_H,BLOCK_K]=Q@gathered^T`
    是满利用率矩阵核 GEMM（不像 per-row 前向用 1 query 广播浪费 15/16 行）。
  - QK 已用 head-as-M MFMA，**正确**（out 55dB / lse 146dB，覆盖 H=8/16/64、K=17/48/512、-1 pad）。
  - 未接线原因见 §5：AV 步骤仍是 scalar VALU（reload gathered 16 次），11.7ms 尚不快于 per-row 7.5ms。

---

## 2. 正确性

全量 `tests/pytorch/ops/test_deepseek_attention.py`：**138 passed**（含 FlyDSL/Triton 两维）。

| 项 | SNR (dB) | 阈值 | 结果 |
|---|---|---|---|
| 前向 | ~45 | 40 | ✅ |
| dq / dk / dv | 44–45 | 35 | ✅ |
| dpool | 37 | 35 | ✅ |
| dsink | 46 | 35 | ✅ |

---

## 3. 性能记录（MI355X gfx950, bf16, flash H=64 D=512 S=2048 B=1 smoke，中位数 ms，越低越好）

### 3.1 前向 (fwd_ms) —— 本轮未改动

| 布局 | Triton | FlyDSL | 比值 |
|---|---|---|---|
| MHA (benchmark 布局) | 0.70 | 10.6 | FlyDSL 慢 ~15x |

### 3.2 前向+反向 (fwdbwd_ms)

| 布局 | Triton | FlyDSL 本轮前(naive) | FlyDSL 本轮后 | 本轮改善 |
|---|---|---|---|---|
| MHA (benchmark 布局，与 Triton 可比) | 21.0 | 2751 | **~1000** | 2.7x |
| MQA (V4 生产布局) | — * | 2747 | **~310** | 8.9x |

\* Triton CSA 路径要求 `k_local=[B,H,S,D]`（MHA 展开，见 `csa_attention_fwd.py` 的 shape 检查），
不支持 MQA 直传，故 MQA 行无 Triton 对照。V4 生产模型是 MQA。

### 3.3 反向内部分解（MQA S=2048）

| 组件 | naive | 优化后 | 手段 |
|---|---|---|---|
| dpool scatter | 2040 | **296** | head-summed 无原子 partial kernel + index_add |
| local 分支 (dq/dk/dv) | 811 | 快（复用） | 复用现成 MFMA SWA bwd 内核 |
| sparse dq | — | ~7 | `store_dpool=False`，无原子 |

---

## 4. 改动 / 新增文件清单

新增：
- `primus_turbo/flydsl/attention/kernels/csa_pool_bwd_kernel.py`（反向内核 + 专用 dpool 内核）
- `primus_turbo/flydsl/attention/kernels/csa_pool_sparse_fwd_kernel.py`（head-as-M MFMA 前向 QK 构件，未接线）

改动：
- `primus_turbo/flydsl/attention/deepseek_attn_bwd_kernel.py`（launcher + 拆分调度）
- `primus_turbo/pytorch/kernels/attention/deepseek_attn_impl.py`（反向 dispatcher）
- `primus_turbo/pytorch/ops/attention/csa_attention.py`（autograd 接反向 dispatcher）
- `suggest/dpsk_attn_support_and_perf_summary.md`（支持矩阵：CSA 反向改为"FlyDSL 已接线"）

---

## 5. 尚未超过 Triton 的原因与后续方案

per-row（1 query / 1 head）架构是性能天花板；Triton 靠 2D-MFMA tiling 取胜。两处待办：

1. **前向仍慢 ~15x**、**dpool / sparse 的 AV 仍是 scalar**。共同缺口是 **AV 步骤需要第二个 MFMA**：
   `acc[head,d] = p[head,key] @ gathered[key,d]`（A=p[head,key]，B=gathered[key,d]，需把 p 排成
   A-operand 布局 + 对 d 做 N-tiling）。这一步做完，sparse 前向与 dpool 反向都能吃满矩阵核。
2. **前向接线**：local 复用 SLA MFMA 前向 `build_swa_fwd_module` → (out_local, lse_local)；
   sparse 用 §1.4 内核（补上 AV-MFMA）→ (out_sparse, lse_sparse)；host 侧稳定 merge：
   `lse=ln(exp(lse_local)+exp(lse_sparse)+exp(sink)); out=out_local*exp(lse_local-lse)+out_sparse*exp(lse_sparse-lse)`
   （sink 只进 lse）。数学已核对：未归一化全局分子 = out_i * exp(lse_i)。

### FlyDSL 0.2.2 踩坑备忘（本轮新增）
- 原子：`llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, ptr, val, llvm.AtomicOrdering.monotonic)`
  或 `rocdl.raw_ptr_buffer_atomic_fadd`（buffer 有 2GB voffset 限制）。
- 无 `T.i1`；布尔用 `arith.cmpi/cmpf` + `AndIOp/OrIOp` 组合。float 相等要用 `arith.cmpf(CmpFPredicate.OEQ,...)`（不是 cmpi）。
- `rocdl.readlane` 的 lane 索引必须 **warp-uniform**；跨 lane 变化的广播要用 `rocdl.ds_bpermute`
  （index = 4*target_lane，操作数须 i32 → f32 需 bitcast 往返）。
- MFMA `v_mfma_f32_16x16x32` C-layout：lane L 持有 `C[(L//16)*4 + r, L%16]`。
