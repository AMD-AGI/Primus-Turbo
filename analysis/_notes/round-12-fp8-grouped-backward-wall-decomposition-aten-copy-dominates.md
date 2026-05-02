# Round 12 — FP8 grouped backward wall decomposition: `aten::copy_` dominates, not kernel spill

## TL;DR

- **R12 metric (forward) = 963** — unchanged from R11 plateau (best=962, +1 from noise band top).
  Forward kernel binary bit-identical to R11 commit `be2e7a30` (verified post-rebuild).
- **R11 round-note pivot delivered**: switch to backward optimization scout. Per
  `bench_grouped_gemm_turbo.py --dtype fp8` (24 shapes), gpt_oss subset shows
  bwd/fwd wall ratio **3.0× vs expected 2.0×** (other models ~2.0× as expected).
- **Wall decomposition via torch.profiler** on the worst case
  (gpt_oss-Down B32-M2048: bwd 3.01 ms, 723 TFLOPS) reveals the bottleneck is
  **NOT** in any HK kernel:

| Pipeline component | per-iter μs | % wall | what it is |
|---|--:|--:|---|
| `aten::copy_` (`elementwise_kernel_manual_unroll<12,...>`) | 1078 | **36 %** | `b.transpose(-2,-1).contiguous()` triggered by `K_RRR % 128 != 0` H4 reroute (R14+R18) |
| `grouped_var_k_kernel_fp8<0>` | 669 | 22 % | dB path (CRR variable-K) |
| `grouped_rcr_kernel<0,T,T>` (dA reroute) | 589 | 20 % | dA after transpose, reuses forward kernel |
| `quantize_fp8_tensorwise` ×3 | 208×3 = 624 | 21 % | input quantize (a, b, grad_out) per iter |
| `aten::add_` ×2 | 226×2 = 452 | 15 % | scale accumulation |
| `unary_kernel<bfloat16>`, `reduce_row_kernel`, ... | ~250 | 8 % | scalar finalize |

(rows sum > 100 % because they overlap on the GPU; profiler `Self CUDA Time` is per-stream.)

- **R12 deliverable**: this round note + the falsification matrix below
  (no HK kernel changes; metric unchanged at 963).
- **R13 lever priority** (in order): (H) drop the `b.contiguous()` round-trip —
  this is *the single highest-leverage* backward win, larger than every kernel
  spill lever combined.

---

## Falsification logic (why R12 ends as docs)

R10/R11 already established the forward FP8 grouped path is at an architectural
ceiling: A/B/C/D/E levers all falsified, score plateau at 962-964. The R11 plan
switched the agenda to backward, with the working hypothesis that
`grouped_var_k_kernel_fp8` (R3 "secondary cluster" of 162 NumVR / 52 dw spill)
was the dominant bottleneck.

**That hypothesis is wrong.** Profiler data above shows:

1. The single largest backward wall component on the gpt_oss subset (the
   *only* subset with anomalous bwd/fwd ratio) is the **PyTorch
   `aten::copy_`** issued by the H4 reroute path
   (`grouped_gemm_fp8_impl.py:365`): `b.transpose(-2,-1).contiguous()`.
2. `grouped_var_k_kernel_fp8` is only 22 % of bwd wall — even cutting its
   spill in half (the most aggressive plausible refactor) would buy 11 % bwd
   wall reduction. The transpose lever is 3× larger.
3. The dA path (after reroute) uses `grouped_rcr_kernel<0,T,T>` — already at
   the forward architectural ceiling.

This inverts the R11 plan: instead of attacking the kernel spill (low
ceiling, high refactor cost), the next round should attack the wall
breakdown directly — the `aten::copy_` and the redundant quantize/elementwise
glue.

---

## Forward metric baseline (R12 starting point)

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1877 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1233 (n=24)
[metric_grouped_only] score=963  weights=grpBF16:1 grpFP8:1
```

Worst 5 grpFP8 (all gpt_oss K=2880, all forward-architectural-ceiling per
R3..R11 falsifications):

| shape | ratio |
|---|--:|
| gpt_oss-GateUP-B32-M4096 | 1.020 |
| gpt_oss-GateUP-B4-M4096  | 1.058 |
| gpt_oss-GateUP-B32-M2048 | 1.064 |
| gpt_oss-Down-B32-M4096   | 1.068 |
| gpt_oss-Down-B32-M2048   | 1.074 |

These remain frozen (forward path).

## Backward bench baseline (24 shapes, FP8 tensorwise, HIPKITTEN)

```
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8
```

All 24 cases `Correctness Check: PASS (out, da, db SNR ≥ 28.4)` vs threshold 25.

**Average forward TFLOPS = 1337.07**, **Average backward TFLOPS = 1238.55**
(naive expectation: bwd flops = 2× fwd flops, so for matched throughput
bwd_TFLOPS ≈ fwd_TFLOPS; we currently sit at 0.93×).

**bwd/fwd wall ratio by subset**:

| subset | n | bwd/fwd wall (median) | vs expected 2.0× |
|---|--:|--:|--:|
| DeepSeek-V3 | 8 | 2.00× | matches |
| gpt_oss_20B (K=2880) | 8 | **3.00×** | **+50 % overhead** |
| Qwen3-235B-A22B | 8 | 1.94× | matches |

The gpt_oss subset is the entire backward bottleneck. All 8 cases share
`K=2880` (== `2 * 1408 + 64`, i.e. `K_RRR % 128 == 64`, misaligned). The
H4 reroute (R14 + R18) gates on exactly that condition.

## Per-kernel resource usage (-Rpass-analysis=kernel-resource-usage)

```bash
cd HipKittens/analysis/fp8_gemm/mi350x && touch kernel_fp8_layouts.cpp && make -B
```

| Kernel | Spill (dw) | Scratch (B/lane) | LDS (B) | Loop S+R | Notes |
|---|--:|--:|--:|--:|---|
| `gemm_kernel<CRR,0>` (dense fwd CRR) | 8 | 36 | 135168 | ~0 | dense baseline |
| `gemm_kernel<RCR,0>` (dense fwd RCR) | 12 | 48 | 139264 | ~0 | dense baseline |
| `gemm_kernel<RRR,0>` (dense fwd RRR) | 24 | 100 | 139264 | ~0 | dense baseline |
| `grouped_rcr<0,F,T>` (DSV3+Qwen fwd) | 32 | 132 | 139796 | 58 | grouped fwd, FUSED_KTAIL=true |
| `grouped_rcr<0,T,T>` (gpt_oss fwd) | 39 | 160 | 139796 | 72 | grouped fwd, N_MASK + FUSED_KTAIL |
| **`grouped_rrr<0>`** (dA path) | 76 | **308** | 139796 | 58 | grouped backward dA (DSV3 only) |
| **`grouped_var_k<0>`** (dB path) | 52 | 212 | 139796 | **161** | grouped backward dB |

`Loop S+R` is sum of `LoopSpillReloadCopies.NumSpills + .NumReloads` parsed
from `kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.opt.yaml`, attributed to
the outer `for (int gt = pid; gt < total_tiles; gt += NUM_CUS)` loop.

Two observations:

1. `grouped_var_k<0>` outer-loop spill (161) is **2.78× higher** than any
   forward kernel (max 72). This is anomalous — the inner K-iter body is
   byte-identical to dense `gemm_kernel<CRR,0>` (line 1924-1949 vs line
   5746-5771), and dense CRR has ~0 loop S+R. The 161 must come from the
   outer-loop `binary search → group_idx → m_start_g → ki_g → tile-coord
   branch → lambda capture` chain.
2. `grouped_rrr<0>` stack 308 B is **the highest of any FP8 kernel**, but its
   loop S+R is only 58 (== forward grouped_rcr level). The 308 B is therefore
   *function-scope* spill (epilog + K-tail / N-tail handlers + RRR
   FP8_RRR_FUSE_PROBE dead-but-allocated branch from R28).

## Backward profile decomposition (gpt_oss-Down B32-M2048)

10 fwd + 10 bwd iters under `torch.profiler` (CPU+CUDA, ROCTracer):

| Kernel / op | Self CUDA | # Calls | per call | % bwd-only wall |
|---|--:|--:|--:|--:|
| `grouped_rcr_kernel<0,T,T>` | 11.781 ms | 20 | 589 μs | (10× fwd, 10× dA via reroute) |
| `aten::copy_` → `elementwise_kernel_manual_unroll<12>` | 10.781 ms | 10 | **1078 μs** | **35.8 %** |
| `grouped_var_k_kernel_fp8<0>` | 6.688 ms | 10 | 669 μs | 22.2 % |
| `quantize_fp8_tensorwise` | 6.251 ms | 30 | 208 μs | (2× fwd + 1× bwd) |
| `aten::add_` (`vectorized_elementwise_kernel<8>`) | 4.513 ms | 20 | 226 μs | 15.0 % |
| `unary_kernel<512,8,bfloat16>` | 3.254 ms | 30 | 108 μs | 10.8 % |
| `reduce_row_kernel<...>` (×2 variants) | 2.625 ms + 0.258 ms | 30+60 | 87/4 μs | 9.6 % |
| `compute_scale_from_amax_kernel` | 0.114 ms | 30 | 3.8 μs | 0.4 % |

`bwd-only wall` = (per-iter contribution to backward only), computed by
subtracting the forward iter share. The `aten::copy_` is the
`b.transpose(-2,-1).contiguous()` triggered by R14+R18 H4 reroute on
K_RRR=2880 (= K_BLOCK-misaligned, fast_k=2816, K_REM=64).

`b.numel() * 2 (rd+wr) * 1 byte = 32 * 2880 * 2880 * 2 * 1 = 530 MB rd+wr =
1060 MB total`. Effective HBM ~1.0 TB/s during this op (530 MB / 1078 μs ×
1e9 / 1e12 = 0.49 TB/s × 2 = 0.98 TB/s combined rd+wr) — only 28 % of MI350X
peak 3.4 TB/s. The non-contiguous transpose stride is the throughput killer.

## R13+ lever priority (re-ranked vs R11 plan)

### **Lever H (PRIORITY 1)** — drop the `b.contiguous()` round-trip on dA
- **Where**: `grouped_gemm_fp8_impl.py:363-366`
  ```python
  if not trans_b and ((a.shape[1] % K_BLOCK) != 0
                      or (b.shape[-1] % BLOCK_SIZE) != 0):
      b = b.transpose(-2, -1).contiguous()  # ← 1078 μs at 1 TB/s
      trans_b = True
  ```
- **Direction A**: replace `b.transpose(-2,-1).contiguous()` with a fused
  HK transpose-only helper (no quantize, no scale change — b is already fp8
  here). A coalesced transpose at peak HBM should run in ~310 μs (3.4 TB/s
  effective) vs current 1078 μs at 1 TB/s. Estimated **win: ~770 μs / iter
  on gpt_oss subset**, i.e. backward wall 3.01 → 2.24 ms (+34 % bwd
  TFLOPS) on the worst case. Not bench-visible until full 24-shape rerun.
- **Direction B**: write a fused `dispatch_grouped_rcr_btranspose` that
  consumes `b: [B, K, N]` (RRR layout) directly and produces grad_a inside
  the RCR fuse epilog without external transpose. Higher cost / risk; only
  if Direction A is insufficient.
- **Compliance**: layout transpose helper, NOT host-pad / per-group launch
  / CPU-sync. Bindings .so changes (new function), but no dispatcher fallback
  semantics change.
- **Risk**: medium. R28 docs already noted that native K_RRR-misaligned
  RRR fuse falsified (5 attempts, SNR 15 dB), so the alternative path is
  bounded.

### Lever I (PRIORITY 2) — quantize / elementwise overhead trim
- The 30 `quantize_fp8_tensorwise` calls (208 μs each = 6.25 ms across 10
  fwd+10 bwd = ~625 μs / iter) include a redundant pair: forward iter
  quantizes `a, b`; backward iter quantizes `grad_out` (and arguably
  `transposed_b` post-reroute, depending on dispatch order).
- The `aten::add_` and `unary_kernel` calls are inside the dscale path
  (R11 host-overhead trim noted ~4.8 µs / iter Python overhead, but the
  HBM-side cost was not measured). Each `aten::add_` 226 μs at 600 KB
  numel = sub-peak; likely compute-bound on small reductions.
- **Estimated win: ~200-400 μs / iter** if redundant quantize pair can be
  collapsed. Needs deeper rocprof to localize.

### Lever J (PRIORITY 3) — `grouped_var_k_kernel_fp8` outer-loop spill
- Loop S+R 161 vs forward floor 58. Net 103 unnecessary spills per iter,
  per CU. At ~80 cy / spill-reload pair × 161 / 2 / 64 lanes ≈ 100 cy / lane / K-iter
  outer iteration. 11 K-iter (gpt_oss-Down K=2880 → 2816/256 = 11 K-block /
  outer iter) × 100 cy ≈ 1100 cy per gt iter, ~1.5 % of dB wall.
- **Estimated win: 1-3 %** of bwd wall on gpt_oss subset.
- **Defer until Lever H has been quantified** — even fully eliminated, this
  is < 1/15 of Lever H's potential.

### Lever K (Lever G in R11 plan) — `grouped_rrr_kernel` function-scope spill
- 76 dw spill, 308 stack. But `grouped_rrr` is **only used on K_RRR-aligned
  shapes** (DSV3 8 / 24 cases), where the dA wall is already in the same
  ratio as forward (~2.0×). No measured ratio gap.
- **Defer indefinitely** unless a future H4 reroute removal exposes the
  RRR path on misaligned shapes, in which case re-evaluate.

---

## R11 backward hypotheses (now falsified)

| R11-postulated bottleneck | R12 measurement | verdict |
|---|---|---|
| `grouped_var_k_kernel_fp8` 52 dw spill is dominant | 22 % of bwd wall, capped lever | **falsified** |
| `grouped_rrr_kernel` 76 dw spill is dominant | DSV3 only; dA reroute on gpt_oss never enters this kernel | **falsified** |
| Backward kernel spill is the architectural ceiling | `aten::copy_` is 36 % wall, kernel spill is < 25 % | **falsified** |
| 162-NumVR secondary cluster (R3) is the lever | Secondary cluster does map to outer-loop bookkeeping (binary search + lambda capture), but its absolute wall contribution is < 3 % | **falsified** (in scale) |

The R3 spill data was *correct* (and the R12 rebuild reproduces it exactly:
52 dw / 212 B / 161 loop S+R), but the R11 conclusion *that fixing the spill
would close the bwd/fwd ratio gap* was wrong. The ratio gap is 99 %
explained by the H4 reroute's `aten::copy_`.

## Cumulative lever-falsification matrix (forward + backward, R1..R12)

| Lever | Round | Path | Verdict |
|---|---|---|---|
| A: async global→LDS | R1 | fwd | already shipped (`buffer_load_dwordx4 ... offen lds`) |
| B: dual/triple LDS ping-pong | R2 | fwd | already shipped (`__shared__ ST_rcr As[2][2]; Bs[2][2]`); LDS capacity bound |
| C: VGPR live-range reduction | R3..R5 | fwd | falsified — at architectural VGPR cap (256) |
| D: 32×32×64 MFMA cell | R5 | fwd | microbench ≈ 0 % delta on per-warp throughput |
| E: ASM software pipelining | R11 | fwd | microbench -7.28 % × 5 trials (LLVM scheduler is already optimal) |
| F: per-shape dispatcher rules | R6..R10 | fwd | landed 5 rules; forward score 962-964 plateau |
| G: `grouped_rrr` spill compression | R12 | dA bwd | de-prioritized (kernel only used on DSV3, no ratio gap) |
| H: drop `b.contiguous()` round-trip | R12 | dA bwd | **untested, R13 PRIORITY 1** |
| I: quantize/elementwise glue trim | R12 | bwd | **untested, R13 PRIORITY 2** |
| J: `grouped_var_k` outer-loop spill | R12 | dB bwd | de-prioritized (1-3 % wall) |

Forward path: 6 levers exhausted, plateau accepted.
Backward path: 1 net-positive lever identified (Lever H), 3 dead ends documented.

---

## R12 reproducibility notes

- Rebuild: `THUNDERKITTENS_ROOT=/workspace/code/HipKittens ROCM_PATH=/opt/rocm
  make -B` in `HipKittens/analysis/fp8_gemm/mi350x/`. Stderr captured to
  `/tmp/build_R12.log`. Resource-usage table parsed from
  `kernel_fp8_layouts-hip-amdgcn-amd-amdhsa-gfx950.opt.yaml`.
- Bench: `PRIMUS_TURBO_HIPKITTEN_PATH=... PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN
  python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8 --output
  /tmp/bench_fp8_round12_baseline.csv`. All 24 PASS, avg fwd 1337 / bwd 1239 TFLOPS.
- Profile: 20-iter `torch.profiler` on gpt_oss-Down B32-M2048 with
  `ProfilerActivity.CUDA + CPU`, `record_shapes=False`. ROCTracer back-end.

## Decision

R12 commits the round note only. HK kernel files are untouched. Forward metric
remains at 963 (post-rebuild verification confirmed bit-identical
forward-kernel codegen relative to R11 commit `be2e7a30`). R13 will
implement Lever H Direction A (fused HK transpose helper) per the plan above.

---

## Round-end summary (本轮目标 / 改了什么 / before-after metric / commit SHA / 下一轮建议)

- **本轮目标**: 接 R11 的 backward agenda — 跑 backward bench baseline，
  做 spill 数据 + wall decomposition，找下一个 net-positive lever。
- **改了什么**: 没动 HK kernel；写本 round note，记录:
  (a) backward bench 24-shape baseline (avg fwd 1337 / bwd 1239 TFLOPS, 全 24 PASS);
  (b) per-kernel resource usage 表 (R3 数据 reproduce 一致);
  (c) torch.profiler decomposition on worst case (gpt_oss-Down B32-M2048 bwd 3.01 ms);
  (d) R11 hypotheses 全部 falsified (kernel spill 不是 dominant);
  (e) R13 lever 重排 — Lever H (drop `b.contiguous()`) 是 priority 1，
      预期 +34 % bwd TFLOPS on gpt_oss subset.
- **Before-after metric**: 963 → 963 (forward unchanged; bwd 不计 score).
- **Commit SHA**: filled at commit time.
- **下一轮建议**: 实施 Lever H Direction A — 在 HK 中加 fused transpose helper
  (input `b: [B, K, N]` fp8, output `b_t: [B, N, K]` fp8, single launch peak HBM)，
  替换 `grouped_gemm_fp8_impl.py:365` 的 PyTorch `b.transpose(-2,-1).contiguous()`。
  必须跑 `bench_grouped_gemm_turbo.py --dtype fp8` 验证 bwd TFLOPS 提升 +
  全 24 PASS, bench output 贴 commit message。
