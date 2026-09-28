# VERIFY：对 RESULT.md（B0 e2e，2026-09-28）的对抗性复核

复核人：无卡、只读（只写本文件）。核对对象：RESULT.md、logs/*.p1a_turbo.*、rank-0 debug.log、run_e2e.sh、
configs/run.p1a_turbo.yaml、BUILD.md、attn_backends/e2e_attn/*.py、Primus/torchtitan 源码。

## 0. 结论

- **主结论成立：没有任何 tps 数据，三个 arm 都没有 e2e 测量。** 没有可以复核的 step 时间、稳态窗口、统计量或 loss，
  所以"每个 arm 真的跑了它声称的 kernel""稳态窗口""开销归因加起来等于 step 时间"这几项**全部无法检验**
  （不是通过，是没有对象）。汇总 JSON 里 arm 1 的 `step_ms: 0` 应写成 null / "—"，0 会被误读成测量值。
- **op 级 attn_ms 算术正确**：32×(5.200+24.160)=939.5，32×(1.146+7.223)=267.8，32×(1.146+9.008)=324.9，
  与 BUILD.md 一致；+3.5%（0.61/17 s）与 0.3%（57 ms/17 s）的先验算术也对。但这是 A0 的 17 s/步外推，不是 B0 数据。
- **arm 1 身份核实**：wt-bakeoff HEAD = 1cb2e183；debug.log 有 `P1: primus_turbo -> wt-bakeoff/primus_turbo`、
  `TurboAttention -> e2e_attn.E2EAttention`、`E2E_ATTN='turbo'`，主日志有 `fallback backend TRITON is selected`。
  这只证明 **fwd 走了 Triton**；bwd 和 GEMM 从未完成过一次，没有 trace，**kernel 名没有任何 profile 证据**。

## 1. 需要更正 / 证据不支持的说法

1. **"GPU 100% busy、sclk 钉在 2355"不是负载证据。** `clk.p1a_turbo.csv` 只记 sclk/功耗/温度，没有 busy%。
   复核时（02:59，run 已结束、GPU0 空闲）实测 card0 = 0001:04:00.0：`gpu_busy_percent=0`、sclk `1: 2355Mhz *`、
   power1_average = 1.134 kW —— **与 hang 期间的读数完全相同**。而且 02:37-02:41（RESULT 说是"纯 CPU 编译"）
   的读数也是 2355 MHz / 1.13 kW。所以这条 csv 既不能证明 GPU 在干活，也不能区分"卡死（GPU 空转等待）"和"极慢"。
   "100% busy" 在 logs/ 里找不到出处。
2. **"两次 gdb 采样隔 50 s，停在同一处"没有留存证据。** `hang.p1a_turbo.gdb.txt` 只有一个样本（`Thread 1` 仅出现一次，
   146 行）。gdb 自己还警告 "different PID namespaces; thread lists ... likely unreliable"。
3. **"compile 仍然是关的 / 不是 autotune"具有误导性。** `compile.enable=false` 只管 model/loss。torchtitan
   `models/attention.py:299` 是 `_compiled_create_block_mask = torch.compile(create_block_mask)`，
   `train.py:463` 在 attn_type=flex 时**每一步**都调用 `model.get_attention_masks()`。也就是说
   **inductor 生成的 Triton kernel 每步都在这张卡上跑**。config 里的注释说 converter "保护这张卡不进 inductor 路径"，
   在 8B_flex 下已经不成立。卡安全记录里挂卡两次都和 inductor 路径有关，所以 RESULT §4 第 3 条的嫌疑
   （inductor block-mask kernel）**不应排在 BLAS 之后**，至少应当并列。
4. **profiler 混杂因素没有提到。** 主线程栈 #3 是 `roctracer::hsa_support::hsa_signal_wait_scacquire_callback`：
   step 1（不在 profile 窗口内）的 HSA signal wait 已经被 roctracer 拦截。`enable_profiling: true` 从第 1 步起就装上了
   roctracer 钩子。这是 A0 可跑配置之外的又一个变量（A0 17 s/步那次是否开了 profiling 需确认），可能影响
   hang/极慢。
5. **"首要嫌疑是宿主 hipBLASLt"证据不足。** 支持它的只有"未在训练形状上验证过"这一条先验。卡住位置在 grad clip 的
   `_foreach_mul_` → `hipLaunchKernel` 等 AQL 槽位，只说明 GPU 积压了整步的工作，**不指向任何具体 kernel**。
   相对 A0 的已知可跑组合，B0 这次至少有 3 个同时变化的量：宿主 BLAS 库、8B_flex 每步 inductor block mask、
   第 1 步就挂上的 roctracer。一次运行无法归因到其中任何一个。
6. RESULT 说 dmesg 干净：`hang.p1a_turbo.dmesg_mark.txt` 只有 GPU1（0002:04:00.0）的旧行；ruler f13 的 dmesg tail
   里 pid 2015598 的 `DQM create queue failed / general protection` 发生在 uptime ≈69813 s（约 01:44），早于本次运行，
   BUILD.md 也已报告过。**此项成立。**

## 2. 同配置性（same-config）检查

- P1 与 P2 用同一模板，差别只在 PYTHONPATH 和 E2E_ATTN：成立（run_e2e.sh）。
- 但 **P1 加载的是真 primus_turbo（170 个模块），P2 只加载 shim（5 个模块）**。float8/mx/async_tp 已在两边都关掉，
  理论上等价；这点只能靠 trace 里非 attention kernel 的集合与耗时一致来确认（trace_breakdown 的 GEMM/elementwise
  两栏应在 P1 与 P2 之间相等，否则 P1 vs P2 的 tps 差不能全算给 attention）。
- **mask 语义**：8B_flex 的 attn_mask_type=block_causal，但 Primus 的 llama3 `Attention.forward` 调用
  `self.inner_attention(xq, xk, xv)`，**完全丢掉了 attention_masks**，三个 arm 都算纯 causal。三臂之间一致，
  但和截图里的 flex（block_causal）不是同一个计算；mock data 若含 EOS，loss 不能与截图 flex 直接对照。
- arm 2 vs arm 3 只能靠同进程 ABBA 分辨（0.3% << 跨进程噪声 3.9%）：同意 RESULT 的判断。

## 3. FlyDSL 路径里应当去掉的 e2e 开销（remove_list）

按预期收益排序。1、2 对三个 arm 相同，但它们都在 FlyDSL 路径上，而且会稀释 attention 的收益。

1. **每步的 block-mask 构建（最高优先级）**：`train.py:463` → `get_attention_masks` → 被 torch.compile 的
   `create_block_mask`，产物被 Primus `Attention.forward` 丢弃，纯浪费；首启还有约 5 分钟的 inductor 编译，并把 inductor
   Triton kernel 带上卡（安全风险）。做法：用一个 attn_type 不是 flex/varlen 的 flavor（与 8B_flex 参数相同），
   前提是先确认 converter 在第一次 forward 之前就替换掉 inner_attention（这样 SDPA/MATH 永远不会执行，
   不会复现 442 GB 的 SIGBUS）；或者在 converter 生效时让 `get_attention_masks` 返回 None。
2. **未融合的 rms_norm**：日志 `Mismatch dtype between input and weight: input bf16, weight float, Cannot dispatch to fused
   implementation`。每层 2 次、共 65 次 norm 走了非融合路径（多个 elementwise kernel 加 fp32 中间结果）。
   做法：norm 权重用 bf16，或在调用处 cast，让它走 fused kernel。需要用 trace 的 elementwise 栏确认数值。
3. **profiler 常驻钩子**：`enable_profiling: true` 让 roctracer 从第 1 步起就拦截 HSA 调用。测 tps 的运行用 pfreq=0，
   profile 放到单独的运行里；或者至少证明非 profile 步的 step 时间与 pfreq=0 的运行一致。
4. **FlyDSL 每次调用的 host 端分发**（有条件，需要 trace 确认）：op 级测量里"经模块"与 kernel 的差在噪声内，
   但 e2e 里 32 层 × (fwd+bwd) 的 Python 发射、flydsl JIT 缓存查找和 lse/中间 buffer 分配是否造成 GPU idle
   间隙，需要看 trace 里 `e2e::attn_*` 范围内的 idle。bwd 内部是否有独立的 dq_acc 清零 / dq convert / odo 预处理 kernel
   也要从 trace 里数出来（ASM 的对应部分已在 kernel 时间里）。有的话就合进主 kernel，或者复用常驻 buffer。
5. **层 0 的调试日志和 record_function 包装**：开销很小。确认 tps 窗口不包含 step 1-3（它们带 stderr 打印）。
   `e2e::*` ranges 保留，它们是归因需要的。

**不需要去掉的**：适配层拷贝（实测 0 次）、GQA 求和（FlyDSL 没有，只 ASM 有）、`lse.contiguous().float()`（no-op）。

## 4. 建议的下一步顺序（修改 RESULT §4）

1. GEMM 形状探针（RESULT 原方案，两份库各一个进程），**同时**准备第 3 步的 flavor 改动（CPU 上就能做）。
2. 如果探针正常：第一次重试应当 **pfreq=0、去掉 per-step block mask、镜像 BLAS 库**，也就是尽量接近 A0 的已知
   可跑组合，只改一个变量；成功以后再逐项加回来。不要直接带着 profiling 和 flex mask 重跑。
3. 稳态口径（执行前就写死）：丢掉 step 1-5 和所有 profile warmup/active 步，每个 arm 至少 15 个稳态步，
   报 median 和 IQR；loss 要逐步对照（同 seed 1234，各 arm 第 1 步 loss 应一致到 bf16 噪声以内，且没有 nan）。
