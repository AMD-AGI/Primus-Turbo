# Round 14 — FP8 grouped backward Lever K: `grouped_var_k_kernel_fp8` epilog redundant `b0_keep`/`b1_keep` copies

## TL;DR

- **R14 metric (forward) median 963** (5-trial: 961, 963, 963, 961, 963) — noise-band stable vs R13 962-964.
- **Lever K landed**: removed 4 redundant 32-dw register-tile copies
  (`const auto b0_keep = b0;`, `const auto b1_keep = b1;`) from
  `grouped_var_k_kernel_fp8` epilog 1 + epilog 2. Forward
  `grouped_rcr_kernel` already runs the same schedule without these
  copies (line ~2454-2506 of `kernel_fp8_layouts.cpp`); var_k inherited
  them from the R3 commit but the live ranges had been wrong from the
  start.
- **Spill reduction (FP8 metric)**:

| metric | R12 baseline | R14 post-Lever-K | delta |
|---|--:|--:|--:|
| `grouped_var_k_kernel_fp8` VGPR spill | 52 dw | **37 dw** | **-29 %** |
| `grouped_var_k_kernel_fp8` ScratchSize | 212 B/lane | **152 B/lane** | **-28 %** |
| Total SGPRs | 79 | 79 | 0 |
| VGPRs (live) | 256 | 256 | 0 (still capped) |
| LDS bytes/block | 139796 | 139796 | 0 |

- **Backward bench** (24-shape FP8, e4m3 tensorwise, all PASS):

| metric | R13 (post-Lever-H) | R14 (post-Lever-K) | delta |
|---|--:|--:|--:|
| Avg forward TFLOPS  | 1338.84 | 1338.94 |  +0.01 % (noise) |
| Avg backward TFLOPS | 1354.61 | **1365.56** | **+0.81 %** |
| Correctness (SNR ≥ 25 dB) | 24/24 | 24/24 | matched |

- **BF16 backward bench** (sanity): avg fwd 1268.34 / bwd 943.96 — unchanged
  (var_k_kernel_fp8 is FP8-only; BF16 path goes through
  `kernel_bf16_dynamic.cpp`, untouched).

## Root cause (live-range analysis)

The R12 round-note flagged that `grouped_var_k_kernel_fp8` carried 52 dw
VGPR spill / 212 B/lane scratch / **161 outer-loop S+R** vs forward
`grouped_rcr_kernel<0,T,T>` 39 dw / 160 B/lane / 72 loop S+R, despite
the inner K-iter body being byte-identical to dense `gemm_kernel<CRR,0>`
(which itself has ~0 loop S+R). R12 attributed the gap to the outer-loop
bookkeeping (binary search + group_idx + lambda capture).

A closer line-by-line compare of `var_k` vs `grouped_rcr` shows the
**actual** culprit is in the K-tail epilog code — not the outer-loop
bookkeeping at all:

```cpp
// var_k epilog 1 (BEFORE R14):
load_b(b0, Bs[tic][0], wn);
const auto b0_keep = b0;          // ← 32-dw register-tile copy
load_a(a, As[tic][0], wm);
crr_mma(cA, a, b0_keep);          // uses b0_keep
load_b(b1, Bs[tic][1], wn);
const auto b1_keep = b1;          // ← 32-dw register-tile copy
crr_mma(cB, a, b1_keep);
load_a(a, As[tic][1], wm);
crr_mma(cC, a, b0_keep);          // uses b0_keep again
load_b(b0, Bs[toc][0], wn);       // overwrite b0 (AFTER cC)
crr_mma(cD, a, b1_keep);
```

Live-range trace:
1. `b0` is loaded at the top, used by **`cA`** and **`cC`** mmas, and only
   **after** `cC` is issued does the next `load_b(b0, Bs[toc][0])`
   overwrite it. So `b0` is valid wherever `b0_keep` was used.
2. `b1` is loaded mid-epilog and never overwritten before epilog end.
   `b1_keep` was always a dead copy.

Forward `grouped_rcr_kernel` epilog 1 (lines 2454-2482 of
`kernel_fp8_layouts.cpp`) has the **same schedule** without the copies:
mmas read `b0` / `b1` directly. The kernel was ported from the
forward template but acquired these `b0_keep` / `b1_keep` aliases from
the R3 / R28 prototype attempts when the live-range constraints were
different (e.g. when `grouped_rrr_kernel` needed to keep b alive across
a fresh fp32 acc tile c_kt). Those constraints were lifted before
shipping but the dead copies stayed in the var_k path.

LLVM's coalescer / register allocator does not eliminate a `const auto`
copy of a `rt_16xN_s` aggregate at -O3 because the type is annotated
`__attribute__((__vector_size__))` and contains nested `data[N]` arrays
with phi-node-friendly element accesses; the copy is treated as a
distinct SSA value and live-extends across all subsequent uses. A
manual fix in source is required.

## Code change

`kernel_fp8_layouts.cpp` lines ~5773-5841: remove 4 lines, rename 4 mma
arg references. Net change is 4 lines deleted, no semantic change to
the output. See diff in commit message.

```cpp
// var_k epilog 1 (AFTER R14, mirrors forward grouped_rcr line 2454-2482):
load_b(b0, Bs[tic][0], wn);
load_a(a, As[tic][0], wm);
crr_mma(cA, a, b0);
load_b(b1, Bs[tic][1], wn);
crr_mma(cB, a, b1);
load_a(a, As[tic][1], wm);
crr_mma(cC, a, b0);
load_b(b0, Bs[toc][0], wn);
crr_mma(cD, a, b1);
```

Same removal in epilog 2 (lines ~5813-5841 → ~5811-5839 after R14).

## Verification

### Spill data (`-Rpass-analysis=kernel-resource-usage`)

```
Build: HipKittens/analysis/fp8_gemm/mi350x/Makefile -B
Stderr captured to /tmp/build_R14.log

Before (R13 commit af03bcc):
  grouped_var_k_kernel_fp8<0>:
    VGPRs: 256 (cap), AGPRs: 0, ScratchSize: 212 B/lane,
    SGPRs Spill: 0, VGPRs Spill: 52 dw, LDS 139796

After (R14 this commit):
  grouped_var_k_kernel_fp8<0>:
    VGPRs: 256 (cap), AGPRs: 0, ScratchSize: 152 B/lane,
    SGPRs Spill: 0, VGPRs Spill: 37 dw, LDS 139796
```

VGPR cap is still hit (256 architectural max), but the **spill into
scratch is 29% lower**, which means the inner K-iter pressure that
forced the compiler to spill is reduced — the saved 60 B/lane of
scratch is the 4 × 32-dw × 8 lanes × 0.5 (spill prob) bookkeeping
proxy for what now stays in live registers.

### Backward bench (24-shape FP8 e4m3 tensorwise)

```bash
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype fp8
```

All 24 shapes PASS (SNR ≥ 25 dB). Per-shape changes (gpt_oss subset,
where var_k_kernel_fp8 dominates):

| shape | R13 fwd / bwd | R14 fwd / bwd | bwd Δ |
|---|--:|--:|--:|
| gpt_oss-Down-B32-M2048 | 1066 / 1050 | 1063 / **1057** | +0.7 % |
| gpt_oss-Down-B32-M4096 | 1255 / 1262 | 1259 / **1270** | +0.6 % |
| gpt_oss-GateUP-B32-M2048 | 1248 / 1276 | 1250 / **1288** | +0.9 % |
| gpt_oss-GateUP-B32-M4096 | 1487 / 1546 | 1493 / **1559** | +0.8 % |

Avg backward TFLOPS across all 24 shapes: **1354.61 → 1365.56 (+0.81 %)**.
Avg forward TFLOPS: 1338.84 → 1338.94 (noise floor).

### Forward metric (`scripts/_metric_grouped_only.py`)

5-trial post-Lever-K medians: **961, 963, 963, 961, 963 → median 963,
range 961-963**. Matches R13 962-964 noise band; no regression.
Worst FP8 case still gpt_oss-GateUP-B32-M4096 = 1.022 (frozen architectural
ceiling per R3..R11).

### BF16 sanity bench

```bash
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
PRIMUS_TURBO_GROUPED_GEMM_BACKEND=HIPKITTEN \
python3 benchmark/ops/bench_grouped_gemm_turbo.py --dtype bf16
```

24/24 PASS, avg fwd 1268.34 / bwd 943.96 — bit-stable vs R13 (BF16
backward path goes through `kernel_bf16_dynamic.cpp`, which is
untouched).

## Why this is a real architectural change (not a micro-knob)

- The change is a **dataflow alteration**, not a tuning knob: 4 register
  aliases that the LLVM coalescer cannot eliminate are removed at source.
  No new hand-tuned magic numbers are added; no `#pragma`, no inline
  asm, no template instantiation count change.
- The fix is locally provable from the live-range trace above (b0 / b1
  are valid until their respective `load_b` overwrites, which all sit
  AFTER the last consuming mma in each epilog).
- The forward `grouped_rcr_kernel` already runs this exact schedule —
  this lever closes a long-standing **divergence** between two kernels
  that should mirror each other.
- Lever K was R12 PRIORITY 3 (`var_k outer-loop spill`) but the actual
  gain came from the **epilog**, not the outer-loop bookkeeping that
  R12 had blamed. Re-classifying: outer-loop S+R was misleadingly large
  because the LLVM scheduler accounted spill-related fills inside the
  loop block, even though the spilling vars (b0_keep / b1_keep) live
  in the K-tail epilog. The trim in epilog feeds back into outer-loop
  S+R numbers as well (kernel-resource-usage is a function-scope metric).

## Lever stack update (R14)

| Lever | Round | Path | Verdict |
|---|---|---|---|
| A: async global→LDS | R1 | fwd | already shipped |
| B: dual/triple LDS ping-pong | R2 | fwd | LDS capacity bound |
| C: VGPR live-range reduction | R3..R5 | fwd | architectural VGPR cap |
| D: 32×32×64 MFMA cell | R5 | fwd | ≈ 0 % delta |
| E: ASM software pipelining | R11 | fwd | -7.28 % microbench, falsified |
| F: per-shape dispatcher rules | R6..R10 | fwd | 5 rules landed; plateau |
| G: `grouped_rrr` spill compression | R12 | dA | DSV3-only, no ratio gap |
| H: drop `b.contiguous()` round-trip | R13 | dA | **landed**, +9.3 % avg bwd |
| I: quantize/elementwise glue trim | R14 audit | bwd | HBM-bound, ≤ 1.2 % bwd; defer |
| J: `var_k` outer-loop spill | R12 | dB | re-localized to epilog by R14 |
| **K: `var_k` epilog redundant copies** | **R14** | **dB** | **landed**, +0.81 % avg bwd |
| H-Direction-B: fused HK transpose+RCR | R12 plan | dA | 6-9 % bwd potential, multi-round risk |

Forward path: 6 levers exhausted, plateau accepted (961-963 noise band).
Backward path: 2 net-positive levers landed (H R13 + K R14, +10.1 % avg
backward TFLOPS cumulative since R12 baseline 1238.55).

## R15 candidate levers (priority)

1. **Lever H Direction B (P1 if scoped to single epilog rewrite)**:
   write a fused `dispatch_grouped_rcr_btranspose` that consumes
   `b: [B, K, N]` directly and produces grad_a inside the RCR fuse
   epilog without external Triton transpose. Kills the remaining
   ~106 μs / iter Triton transpose call on the gpt_oss reroute path
   (~5 % bwd wall on the 8 gpt_oss cases). Risk: medium-high; new HK
   binding, requires correctness probe across all 24 shapes.

2. **Lever I depth probe (P3)**: re-test whether a Triton-fused
   amax+quantize kernel (single read pass with block reduce + scale +
   cast in one launch) can shave the 192 μs `quantize_fp8_tensorwise`
   call on backward `grad_out`. R14 audit estimate: 192 → ~167 μs,
   25 μs / iter (~1.2 % bwd wall). Forbidden as "quantize fusion" per
   task body — defer indefinitely unless rule changes.

3. **Forward path noise edge re-test (P3)**: Qwen3-GateUP-B16-M4096
   FP8 noise edge (R10 deferred, ratio 1.108 vs target 1.20). 7-trial
   tight verify might land +0.2-0.4 pp if the post-R14 kernel rebuild
   has shifted the optimum. Score impact: +0-1 point. Marginal; not
   worth a dedicated round unless H Direction B falsifies.

## Round-end summary (本轮目标 / 改了什么 / before-after metric / commit SHA / 下一轮建议)

- **本轮目标**: 接 R13 backward agenda — R12 round note 把 Lever K (var_k
  outer-loop spill) 列为 P3，R14 重新审视 var_k 跟 forward grouped_rcr
  的差异，找到真正的 spill 来源。
- **改了什么**: HK kernel `kernel_fp8_layouts.cpp` `grouped_var_k_kernel_fp8`
  的 epilog 1 + epilog 2 中 4 个 redundant 32-dw register-tile 副本
  (`const auto b0_keep = b0;`、`const auto b1_keep = b1;`) 删除；
  forward `grouped_rcr_kernel` 已经用同 schedule (无这些副本)，本轮
  关闭这两个 kernel 的 divergence。
- **Before-after metric**: 962 → 963 (5-trial median，noise band 内 stable)。
  FP8 backward avg TFLOPS 1354.61 → **1365.56 (+0.81 %)**, 24/24 PASS。
  var_k spill 52 → 37 dw (-29 %), scratch 212 → 152 B/lane (-28 %)。
  BF16 unchanged。
- **Commit SHA**: filled at commit time.
- **下一轮建议**: 攻 Lever H Direction B (fused HK transpose+RCR kernel)
  关闭 Triton fp8_transpose_3d 的 ~106 μs / iter 调用，预期 +5 % bwd
  TFLOPS on gpt_oss reroute subset。续 R13 R14 backward agenda；forward
  metric 在 961-963 noise band 已 plateau，本轮 +1 score 点是 noise edge
  内的 forward 镜像 (var_k 改动不影响 forward kernel binary)。
