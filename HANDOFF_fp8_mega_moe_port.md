# 交接:把 mega-MoE 的 fp8(MXFP8)版本移植进 MegaMoE

> 用途:新 session / 新机器直接 `@HANDOFF_fp8_mega_moe_port.md` 即可接续。
> 一句话启动 prompt:
> **「继续把 mega-MoE 的 fp8 版本移植进 MegaMoE,先读 @HANDOFF_fp8_mega_moe_port.md。上一步刚把 token quant 挪进 dispatch_grouped_gemm_mxfp8(bf16 分支),还没在 GPU 上验证,先跑 L1 bench 确认,再继续移植 L2 combine + SwiGLU。」**

---

## 0. 任务

把 fp8/MXFP8 版的 mega fused MoE 从源 repo `/perf_apps/xiaoming/Primus-Turbo` 移植进
`/perf_apps/xiaoming/MegaMoE`,设计好顶层维护方式。bf16 版已在 MegaMoE(`mega_moe_fused.py` +
`mega_moe_{forward,backward}_impl.py` + `flydsl/mega/*`),**不要动 bf16**。

目标平台:AMD MI355X (gfx950),容器 `rocm/primus:v26.3`。遵循
`.cursor/rules/iteration_rules.mdc` + kernel-optimize skill:**正确性优先、单变量线性迭代、每次改动
跑完整验证 + benchmark、accept/rollback 记录、收益要能落到真实训练 step**。

---

## 1. 已定的关键设计决策(不要推翻,除非有新证据)

1. **fp8 跳过 `pytorch/kernels/mega_moe` 的 custom_op / AutoKernelDispatcher 层**。直接在
   `pytorch/ops/moe/mega_moe_fused_fp8.py` 的 autograd Function 里 inline 编排(和源 repo 一致)。
   原因:fp8 前向/反向带着 custom_op schema 装不下的状态(可选权重预量化 tuple、复用 forward 的
   live symm buffer、host `synchronize()+barrier()` rendezvous)。
2. **权重量化状态由有状态模块 `MegaMoEFP8` 持有**(`pytorch/modules/moe/mega_moe_fp8.py`),按
   `w._version` 缓存 `w1_fp8/w2_fp8/w2t_fp8/w1t_fp8`,每 `optim.step` 量化一次、跨 grad-accum 复用。
   MegaMoE 没有 bf16 的 `MegaMoE` 基类模块,所以 `MegaMoEFP8` 做成**独立精简模块**(只持有
   w1/w2/ep_group + 量化缓存 + `expert_compute`;routing/shared-expert 交给上层框架)。
3. **移植路线 A(vendoring):把源 repo 的对称内存底座整套 vendored 进自包含的
   `flydsl/mega/fp8/` 子包**,fp8 用这套、bf16 继续用它现有的那套(两套并存)。
   原因:两个 repo 的对称内存架构已彻底分叉 —— 源 fp8 站在 `SymLayout`+scoreboard+**双堆**上,
   MegaMoE 的 bf16 用的是 `SymBuffer`+`Workspace`+**双 bank flag/奇偶** + 单堆。硬改 kernel 到
   bf16 那套(路线 B)风险太大。路线 A 保真、re-sync 容易,代价是暂时维护两套 symm。
   (可行性已确认:MegaMoE 自己的 `pytorch/core/symm_mem.py::SymmetricMemory` 支持双堆
   signal pad,源 symm 栈直接跑在上面。)

---

## 2. 已完成(可用 + 已验证)

### 顶层骨架(三个落点,bf16 全未动)
- `primus_turbo/flydsl/mega/fp8/__init__.py` — 导出 L1 入口。
- `primus_turbo/pytorch/ops/moe/mega_moe_fused_fp8.py` — `MegaMoEFusedFP8Function` +
  `mega_moe_fused_fp8()` + `prepare_w1t_dgrad_fp8` / `prepare_w2t_dgrad_fp8`。
  **forward/backward 目前仍是 `NotImplementedError`(占位),等 L2/反向移植完再接线。**
- `primus_turbo/pytorch/modules/moe/mega_moe_fp8.py` — 独立 `MegaMoEFP8`,已实现通用
  `_cached_weight`(版本键控);`expert_compute` 里权重预量化 + op 调用是注释模板,待放开。
- 已把 `from .mega_moe_fp8 import *` 加进 `pytorch/modules/moe/__init__.py`。

### L1(fused mxfp8 dispatch + fc1)—— 已移植 + GPU 验证通过
- Vendored 进 `flydsl/mega/fp8/` 的 12 个文件(`ls` 确认):
  底座 `prims.py` `sym_layout.py` `barrier.py` `symm_buffer.py` `dispatch_prologue.py`
  `gemm_helper.py`;fp8 核/量化 `ep_fp8.py` `gemm_mxfp8_tile.py` `quant.py` `quant_flydsl.py`
  `dispatch_grouped_gemm_mxfp8_kernel.py`;加 `__init__.py`。
  **内部 import 已全部重写成 `primus_turbo.flydsl.mega.fp8.*`**,子包只对外依赖共享的
  `primus_turbo.pytorch.core`(SymmetricMemory / low_precision / quantize_fp8)+ 外部 `flydsl`。
- Benchmark:`benchmark/ops/bench_dispatch_grouped_gemm_mxfp8_l1.py`(自包含,只用 vendored fp8 栈 +
  纯 torch dequant-GEMM 参考;**没用源 `mega_utils`**,因为它会拖进整套源 bf16 栈)。
- 验证结果(8×MI355X,EP8,DSv3 H=7168 I=2048 E=256 K=8):
  - import 冒烟全过(`import primus_turbo.flydsl.mega.fp8` / `mega_moe_fused_fp8` / `MegaMoEFP8`)。
  - **正确性 cos=1.00000 rel=0.0017 PASS**(T=2048 与 T=8192)。
  - 性能:T=8192 fused ≈ 2.22–2.52 ms @ ~1600–1840 TFLOPS(和源 repo 报的 fp8 fused ~2.56ms 一致);
    T=2048 fused ≈ 1.0 ms。token quant ≈ 0.04–0.06 ms(~L1 的 2.5%)。

### ★★ fc2 (L2 fp8 combine) 的 NaN 竞争已修复 —— 根因是 signal pad 没用 uncached 内存
移植 L2 后,完整 fp8 前向在部分 rank 随机出 NaN。根因**不是** kernel bug,而是**对称内存的 port gap**:
源 repo 的 `SymmetricMemory` 用 **`hipMallocUncached`** 分配 signal pad(存跨 rank flags + fp8 combine
buffer `comb`);MegaMoE 之前用普通(cached)`hipMalloc`(bf16 走单堆 `signal_pad_size=0`,从没用过
signal pad,所以没暴露)。cached 下 peer PUSH 过来的 payload/E8M0 留在 stale L2 line → fp8 reduce 读到
垃圾 E8M0 指数(=+inf)→ 非有限输出。**修复**:`primus_turbo/pytorch/core/symm_mem.py` 里 signal pad 改用
`hipMallocUncached`(bf16 不受影响,它 `signal_pad_size=0`;wrapper 里 `hipMallocUncached` 本就存在)。
验证(n03-33,2026-07-20):3 次跑(2×T=2048、1×T=8192)全 rank 无 NaN,fp8 vs bf16 **SNR 20.8–22.3 dB
cos ~0.996 PASS**,fp8 前向 T=8192 = 5.22ms(**1.35× vs bf16 7.06ms**)。

### token quant 已挪进 `dispatch_grouped_gemm_mxfp8`(已 GPU 验证)
`dispatch_grouped_gemm_mxfp8_kernel.py` 函数顶部(约 line 431 `if xq.dtype == torch.bfloat16:` 分支):
传 **bf16 x + `xs=None`** 时,op 内部做**一次全局** rowwise mxfp8 quant(`quantize_rowwise_mxfp8_flydsl`,
单独 launch、同流天然 ordered,无需显式 sync)再跑 clean-push 流水线;传预量化 fp8 `xq`+`xs` 则跳过
(源签名兼容,re-sync 友好)。benchmark 用 bf16 x 调用(`_l1_step` 里 `dispatch_grouped_gemm_mxfp8(x, None, ...)`)。
**已在 n03-33 GPU 验证(2026-07-20)**:T=2048 L1 0.94ms、T=8192 L1 2.35ms(fused 2.29ms @ 1782 TFLOPS,
token_quant 0.059ms ~2.5%),cos=1.00000 PASS。

> 设计结论(已讨论定):token quant **不要** fuse 进 device kernel 的每个 push(会变成 per-push、
> K× 重复量化,还把 bandwidth-bound 的 COMM 角色变 compute-bound,毁掉 comm/gemm overlap)。
> 现在这种"launcher 里一次全局 quant"是对的。想再省那 0.05ms 就往上游 fuse(router/上一层
> activation epilogue 产出 x 时顺带量化),别往下 fuse 进 dispatch。

---

## 3. 接续步骤(按优先级)

**✅ 前向已全部完成并验证**(L1 fp8 + SwiGLU + L2 fp8 combine)。L2 vendored 的文件:`gemm_bf16_kernel.py`、
`grouped_gemm_combine_bf16_kernel.py`、`swiglu_kernel.py`、`grouped_gemm_combine_fp8_kernel.py`(都在
`flydsl/mega/fp8/`,import 已重写)。op 前向在 `mega_moe_fused_fp8.py::MegaMoEFusedFP8Function.forward`
(standalone,权重内部量化;`_host_rendezvous` 做 L1/L2 scoreboard/flag 复位);e2e bench
`benchmark/ops/bench_mega_moe_fused_fp8.py`(fp8 vs bf16 SNR gate)。**下一步是反向。**

**✅ 反向 STEP1 已完成并验证**(dispatch(dy)+fc2 dgrad,fp8)。**没有独立的 bwd kernel** —— STEP1 直接
复用前向的 `dispatch_grouped_gemm_mxfp8`(泛型 dispatch PUSH + 分组 mxfp8 NT GEMM:A=dy bf16 内部量化+push、
weight=w2^T `[G,I,H]` → grad_swiglu `[P,I]`),只是 CU 划分不同(STEP1 用 `ndcu=24/pscu=8`,前向 16/16;见
`_STEP1_NUM_DISPATCH_CU`)。源 repo 的 `dispatch_grouped_gemm_mxfp8_bwd_kernel.py` 是前向 kernel 的 fork
(只多一堆 net-negative 的实验旋钮 diag/two_stage/...,默认路径逐字等价)—— **已确认不需要,没 vendor**。
op 层:`prepare_w2t_dgrad_fp8`(=`quantize_grouped_weight_mxfp8(w2.T)`)+ 版本缓存 `_w2t_fp8_cached` +
`_mxfp8_step1_dispatch_dgrad`。验证 bench `benchmark/ops/bench_step1_dispatch_dgrad_fp8.py`:DSv3 T=8192
STEP1 = **1.87ms @ 1093 TFLOPS,SNR 30.9 dB PASS**;vs bf16 dgrad(nn) 2.40ms(bench_mega_moe.py)= **1.27×**。

1. **[下一步] 移植反向剩余段**:STEP2(SwiGLU^T,`swiglu_backward`,bf16)→ dW2/dW1 mxfp8 variable-K wgrad →
   STEP3(fc1 dgrad fp8 + combine)。源 `mega_moe_fused_mxfp8.py::backward` + 源核
   `grouped_gemm_combine_mxfp8_kernel.py`(或 `grouped_gemm_combine_fp8_bwd`)、`quant_colwise_trans_flydsl.py`;
   依赖 `grouped_gemm_fp8_variable_k_impl`(MegaMoE 已有)。见源 `NOTES_mxfp8_backward_handoff.md`。
   forward 目前**没 save_for_backward**,接完整反向时要补 ctx 保存(handle/l1/dispatch_weights/pool_x_fp8 等)。
   反向的 fp8 combine PUSH 同样吃 uncached signal pad(已修)。
2. **(已完成)移植 L2(fp8 combine)+ SwiGLU**:combine 保持 bf16 输出(带宽 bound)
   (见源 `NOTES_mxfp8_fused_gemm_combine_perf.md`);
   落地方案 = L1 fp8 fused + L2 用 `grouped_gemm_combine_fp8`(GEMM fp8、combine/reduce bf16)。
   vendor 进 `flydsl/mega/fp8/`,同样把 import 重写成 `mega.fp8.*`,补 `__init__` 导出。
   ⚠️ 命名漂移:源 op 用 `swiglu`/`swiglu_backward`;MegaMoE bf16 侧叫 `swiglu_flydsl_kernel` 等 ——
   vendor 源的 swiglu 到 fp8 子包即可,别混用 bf16 侧的。
3. **接线前向** `mega_moe_fused_fp8.py::MegaMoEFusedFP8Function.forward`:参照源
   `Primus-Turbo/primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py` 的 forward(L1 → swiglu → L2)。
4. **移植反向**:STEP1(dispatch(dy)+fc2 dgrad,NN,fp8,**会赢**)优先;dW2/dW1 mxfp8 variable-K
   wgrad;STEP3(fc1 dgrad fp8 + combine/reduce bf16)。源文件
   `dispatch_grouped_gemm_mxfp8_bwd_kernel.py`、`grouped_gemm_combine_mxfp8_kernel.py`、
   `quant_colwise_trans_flydsl.py`;依赖 `grouped_gemm_fp8_variable_k_impl`(MegaMoE 已有
   `pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`)。见源
   `NOTES_mxfp8_backward_handoff.md`。
5. **接线 `MegaMoEFP8.expert_compute`**(放开注释模板,接 §2 的权重预量化 producers)。
6. **精度测试**:fp8 用 **SNR gate 而非 allclose**;看 dx/dW1/dW2/grad_topk 的 SNR + 几步训练 loss。
   参考源 `tests/pytorch/ops/test_mega_mxfp8_ffn_e2e.py`、`tests/pytorch/modules/test_mega_moe_mxfp8.py`
   和 verify-accuracy skill。

---

## 4. 环境 & 命令(重要:构建/运行必须在 GPU 机器的容器里)

- **★ 新容器先修 flydsl 版本**:stock `rocm/primus:v26.3` 容器自带 flydsl `0.1.1.dev409`(egg),
  缺 `#412` mega 核需要的符号(`TargetAddressSpace` / `extract_base_index`)→ `import primus_turbo.flydsl.mega`
  会挂在 **bf16** 文件上(不是 fp8 的问题)。修复:在容器里跑
  `bash slab/notes/MegaMoeFlydsl/_repro/fix_flydsl.sh`(卸 dev409 egg → 装 `flydsl==0.2.4` → 清 `~/.flydsl`)。
  `amd-aiter` 的依赖冲突警告无害。每台新起的容器都要先做这步。
- **我的编辑 shell 和 GPU 机器不是同一台**。`/perf_apps` 是共享盘(改文件在哪都行),但
  flydsl/GPU 的 import、build、bench 必须在有 GPU 的机器的容器 `xiaoming-dev`(`rocm/primus:v26.3`)里跑。
- 之前的 GPU 机是 `smci355-ccs-aus-n01-25`,容器 `xiaoming-dev`。**换机器后先确认容器/GPU 位置**:
  `docker ps`(找 `rocm/primus` 的 `xiaoming-dev`);GPU 空闲:`rocm-smi --showuse`;`torch.cuda.device_count()`。
  若从非 GPU 机操作:`ssh <gpu-host> 'docker exec xiaoming-dev bash -lc "..."'`。
- **L1 bench(验证 + 计时)**,在容器里、repo 根目录:
  ```bash
  cd /perf_apps/xiaoming/MegaMoE
  MASTER_PORT=8585 MEGA_BENCH_TIMEOUT_S=300 PYTHONPATH=$PWD \
    python benchmark/ops/bench_dispatch_grouped_gemm_mxfp8_l1.py \
    --num-processes 8 --num-tokens 8192 --warmup 8 --iters 20
  # 快速 smoke 用 --num-tokens 2048
  ```
  期望输出:`token_quant` / `fused` / `L1 total` 三行 + `[acc] ... cos=1.00000 ... PASS`。
- import 冒烟:
  ```bash
  PYTHONPATH=$PWD python -c "import primus_turbo.flydsl.mega.fp8 as m; print(m.__all__)"
  ```

### 分布式跑挂了怎么清理(务必)
- spawn worker 的 cmdline 是 `python -c from multiprocessing.spawn ...`,**不含脚本名**,`pkill -f <script>`
  杀不掉!必须:`kill -9 $(ps -eo pid,cmd | grep '[s]pawn_main' | awk '{print $1}')`。
  残留 worker 会占 GPU + 污染对称内存,导致后续 flag 超时/挂死。
- **每次 bench 换不同 `MASTER_PORT`**(上轮 TCPStore 可能没释放 → EADDRINUSE)。

### 已知坑(踩过)
- **bench 只用一个含 fused 的循环**:大 T 下背靠背 fused launch 会和跨 rank scoreboard reset 抢跑
  (上一轮 XGMI 写在 zero-barrier 之后才落地)→ `MEGA mxfp8 GEMM gate timeout`。现在 harness 已收成
  单个 `t_l1` 循环(quant+fused)+ 一个纯本地 quant 循环。
- 大 T(8192)torch 参考的 fp32 临时张量约 8GB,已在计时前 `del` 掉。
- flydsl 对 `c_n: int` 注解的 UserWarning 无害(源 repo 也有),grep 掉即可。
- L1 参考读的是 kernel 自己 dispatch 出来的 pool → 强验证 GEMM/preshuffle/scale,但**没独立复算跨 rank
  dispatch**(那部分是 verbatim vendored,源已验证)。要独立 dispatch parity 可后续对 bf16 加对照。

---

## 5. 关键源文件对照(源 repo 根:`/perf_apps/xiaoming/Primus-Turbo`)

- 前向编排参考:`primus_turbo/pytorch/ops/moe/mega_moe_fused_mxfp8.py`
- 模块参考:`primus_turbo/pytorch/modules/moe/mega_moe_fp8.py`(源里继承 bf16 `MegaMoE`,我们改成独立)
- L1 核(已移植):`primus_turbo/flydsl/mega/fp8/dispatch_grouped_gemm_mxfp8_kernel.py`
- L2 核(待移植):`.../fp8/grouped_gemm_combine_fp8_kernel.py`(+ `prepare_w2_fp8`)
- 反向核(待移植):`.../fp8/dispatch_grouped_gemm_mxfp8_bwd_kernel.py`、
  `.../fp8/grouped_gemm_combine_mxfp8_kernel.py`、`.../fp8/quant_colwise_trans_flydsl.py`
- 设计/性能笔记:`.../fp8/NOTES_mxfp8_pipeline_design_handoff.md`、
  `.../fp8/NOTES_mxfp8_backward_handoff.md`、`.../fp8/NOTES_mxfp8_fused_dispatch_gemm_perf.md`、
  `.../fp8/NOTES_mxfp8_fused_gemm_combine_perf.md`
- 源 L1 bench(参考,别直接用——依赖源 bf16 栈):`benchmark/ops/bench_dispatch_grouped_gemm_mxfp8.py`
