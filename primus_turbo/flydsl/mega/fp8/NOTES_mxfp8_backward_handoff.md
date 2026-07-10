# 交接:把 Primus-Turbo mega-MoE 的**反向(backward)** MXFP8 化

> 用途:新 session 直接 `@` 这个文件接续。目标是给 mega fused MoE 的**反向**加 MXFP8 支持。
> **先读本文件 + 下面列的关键文件和性能笔记,再和我讨论方向,不要直接开写。**

## 背景(前向已完成,反向目前是 bf16)

MI355X (gfx950) 上,Primus-Turbo mega fused MoE 的 **MXFP8 前向已做完**:
- **L1(dispatch + up/gate GEMM = fc1)用 fp8 fused**(`comm="fp8_fused"`,实测 ~1.4× vs bf16 fused),✅ 赢。
- **L2(down GEMM = fc2 + combine + reduce)用 bf16 fused**(~2.7–2.9ms)。fp8-combine 试过三种量化放法
  都赢不了(见 `NOTES_mxfp8_fused_gemm_combine_perf.md`),combine 是 bf16 输出、带宽 bound,量化开销吃掉字节。
- **反向目前整条是 bf16**(`mega_moe_fused_mxfp8` docstring:backward handled separately in bf16)。

本任务:**把反向 fp8 化,优先做能赢的那一半**。

## 关键结构:Dispatch ↔ Combine 对偶(理解这个就理解了反向)

`dispatch`(scatter,token→K 个 expert 行)和 `combine`(reduce,K 行→token)互为**转置**。所以反向
复用前向的核、对调 + 换 layout。现有反向在 `primus_turbo/pytorch/ops/moe/mega_moe_fused.py`
`MegaMoEFusedFunction.backward`(注释:"Conjugate of forward via Dispatch<->Combine duality"):

| 前向 | 结构 | 反向 step | 复用的核 | layout |
|---|---|---|---|---|
| dispatch + fc1 | **comm→compute** | **STEP1**: dispatch(dy) + fc2 dgrad | `dispatch_grouped_gemm_impl` | nn |
| fc2 + combine | **compute→comm** | **STEP3**: fc1 dgrad + dispatch_bwd(=combine) | `grouped_gemm_combine_impl` | nn |
| — | — | dW1/dW2 wgrad(variable-K）| `grouped_gemm_variable_k_impl` / 再 dispatch x（tn）| tn |

反向要点(已在代码里):reduce 不加权(权重经 `swiglu_backward(scale=dispatch_weights)` 注回 `grad_l1`);
`grad_gate` 单独 scatter 出 `grad_topk_weights`;dW1 把 `saved_x` 重新 dispatch 一遍(省显存)。

## ★ 这轮学到的教训 —— 决定反向 fp8 的优先级(务必先看,别重复踩坑)

反向和前向的 fp8 收益结构**完全对称**:

1. **STEP1(fc2+combine 的反向 = dispatch(dy) + fc2 dgrad,comm→compute)== 前向 L1 的结构 → fp8 会赢。**
   - dy 的 dispatch push 是 fp8 → 字节减半(和 L1 dispatch 一样,~377 GB/s、~1.1ms 级);dgrad GEMM 用 fp8
     → ~1.6–2× 算力;两个 fp8 量级成本可重叠。**这是反向最该做、最可能赢的一段(优先级 1)。**
   - 直接复用/扩展 L1 的 fp8 fused dispatch+GEMM 核到 **NN(dgrad)layout**(现在 L1 fp8 只做了 NT)。

2. **STEP3(fc1 dgrad + combine/dispatch_bwd,compute→comm)== 前向 L2 的结构 → 大概率赢不了。**
   - 走 combine(把 grad_pool reduce 回 grad_x),和前向 L2 一样是 combine-bound;fp8-combine 的量化开销会
     吃掉字节收益(前向已证三种放法都输)。**不要重复前向 L2 那三条死路**(quant-in-combine / 独立
     quant role / GEMM epilogue quant 都试过,0.99× / 0.76× / 0.76×)。这段建议保持 bf16,或只做 dgrad 的
     fp8 而 combine 留 bf16。

3. **wgrad(dW1/dW2,TN variable-K)**:输入是激活/梯度,可 fp8 化 GEMM(有现成 fp8 grouped GEMM 参考)。
   收益取决于 wgrad 是否在关键路径(反向 GEMM 量大,值得评估)。

**结论建议**:反向的主攻是 **STEP1(dispatch(dy)+fc2 dgrad)fp8 化** + 可选 **wgrad fp8**;STEP3 的
combine 侧沿用前向 L2 的结论(bf16)。先把 STEP1 做出来量收益,别一上来碰 combine。

## 精度(反向比前向更需要小心)

- 梯度张量动态范围大,fp8 常用 **E5M2**(range 优先)而非 fwd 的 E4M3——dy / grad_l1 量化时要评估 E4M3 vs E5M2。
- dgrad/wgrad 的 GEMM 累加在 f32;量化只在 GEMM 输入(dy、w、act)。
- gate:反向要看 **dx / dW1 / dW2 / grad_topk_weights 的 SNR**(fp8 用 SNR 而非 allclose),并跑 e2e /
  几步训练 loss 确认端到端可训。参考 `verify-accuracy` skill(SNR/tolerance/FP8 编码)。

## 关键文件

- 反向编排(autograd Function):`primus_turbo/pytorch/ops/moe/mega_moe_fused.py`(`backward` 三步 + wgrad)
- L1 fused 核:`primus_turbo/flydsl/mega/fp8/dispatch_grouped_gemm_mxfp8_kernel.py`(fp8,NT)、
  `primus_turbo/flydsl/mega/dispatch_grouped_gemm_bf16_kernel.py`(bf16,支持 nt/nn/tn)
- impl 封装(前反向都走这里):`primus_turbo/pytorch/kernels/mega_moe/dispatch_grouped_gemm_impl.py`、
  `.../grouped_gemm_combine_impl.py`
- L2 核:`grouped_gemm_combine_bf16_kernel.py`(nt/nn/tn + wgrad)、`fp8/grouped_gemm_combine_mxfp8_kernel.py`
- wgrad:`gemm_bf16_kernel.py`(`grouped_gemm_tn_wgrad_bf16`)、`grouped_gemm/gemm_fp8_grouped_kernel.py`(fp8 参考)、
  `grouped_gemm_variable_k_impl`
- 量化:`fp8/quant.py`、`fp8/quant_flydsl.py`(in-kernel rowwise mxfp8,bit-exact,含 preshuffle)
- fp8 GEMM tile:`fp8/gemm_mxfp8_tile.py`(`gemm_mxfp8_nt_tile`;NN/TN dgrad/wgrad 需新增或参考 bf16 的 _nn_tn_tile)
- 性能笔记:`NOTES_mxfp8_fused_dispatch_gemm_perf.md`(L1,反向 STEP1 的模板)、
  `NOTES_mxfp8_fused_gemm_combine_perf.md`(L2,STEP3 为什么不做 combine fp8)、`NOTES_mxfp8_grouped_gemm_perf.md`
- e2e / 单测:`tests/pytorch/ops/test_mega_mxfp8_ffn_e2e.py`、`tests/pytorch/modules/test_mega_moe_mxfp8.py`

## 复现环境 & 命令

- 远程可用机器之一:`ssh chi2798` → `docker exec xiaoming-dev-slime bash -lc '...'`,
  `cd /mnt/shared/xiaoming/Primus-Turbo`(其它机器/容器同名:chi2761/chi2878 上也有 `xiaoming-dev-slime`;
  跑前先确认 GPU 空闲 + 磁盘够 + 无残留 worker)。
- 分布式跑挂/清理:**spawn worker 的 cmdline 是 `python -c from multiprocessing.spawn ...`,不含脚本名**,
  `pkill -f <script>` 杀不掉!必须 `kill -9 $(ps -eo pid,cmd | grep '[s]pawn_main' | awk '{print $1}')`,
  否则残留 worker 会占 GPU + 污染对称内存,导致后续跑 combine/reduce 的 flag 超时/挂死。
- 每次分布式 bench 用**不同 MASTER_PORT**(上一轮 TCPStore 可能没释放 → EADDRINUSE)。
- 遵循 `.cursor/rules/iteration_rules.mdc` + kernel-optimize skill:**正确性优先、单变量线性迭代、每次改动
  跑完整验证 + benchmark、accept/rollback 记录、收益要能落到真实训练 step**。

## 请先做

先读 `mega_moe_fused.py` 的 `backward` + `NOTES_mxfp8_fused_dispatch_gemm_perf.md`(L1 fp8 模板)+
`dispatch_grouped_gemm_mxfp8_kernel.py`,然后和我讨论:**STEP1(dispatch(dy)+fc2 dgrad)fp8 化的方案
(NN dgrad layout + dy 的 fp8 dispatch + E4M3/E5M2 选择 + 精度 gate)**,以及 wgrad 是否一起做。再决定,不要直接开写。
```
一句话启动 prompt(粘到新 session):
把 mega-MoE 反向 MXFP8 化,先读 @primus_turbo/flydsl/mega/fp8/NOTES_mxfp8_backward_handoff.md 再讨论方案。
```
