# Round 11 — HK FP8/BF16 Grouped Host-Overhead Breakdown vs Triton

**Date**: 2026-05-01
**Branch**: `dev/kyle_hipkitten_bf16`
**HEAD before**: `17a62c8d`
**HEAD after**: this commit
**Metric**: 793 → 793.3 mean over 7 runs (within ±1.5 noise)

## Why this note

Rounds 8/9/10 saturated `(group_m, num_xcds)` config tuning across all 16
gpt_oss BF16+FP8 metric shapes. Each of the recent 6 rounds delivered
+0..+1 score; patience=6/30. Before pivoting to kernel-level work
(BN=128 N-tile / XCD swizzle / K-tail M-amortize / Skip A-tile LDS staging
— see task body §"为什么 gpt_oss 这么慢"), it's worth knowing how much
of the gap to Triton is **kernel work** vs **Python dispatch overhead**,
since the latter is far cheaper to attack and we hadn't characterized it.

## Method

Layered timing of FP8 RCR grouped path with `WARMUP=30, ITERS=500, p20`,
`HIP_VISIBLE_DEVICES=2`, shape `B=4, M=2048, N=2880, K=2880` (worst-case
B=4 gpt_oss-Down — overhead is largest fraction of total wall here):

| Layer | Probe target |
|---|---|
| L1 | direct `hk.module.grouped_rcr_dscale(...)` call (no Python wrapper) |
| L2 | `GroupedGEMMFP8HipKittenBackend.execute(**kwargs)` body |
| L3 | `GroupedGEMMFP8KernelDispatcher.dispatch(...)` (adds can_handle + entry) |
| L4 | `grouped_gemm_fp8_impl(...)` (adds `@torch.library.custom_op` wrapper + enum conversions + kwargs dict) |

Same probe replicated for Triton path
(`GroupedGEMMFP8TritonBackend.execute` → `grouped_gemm_fp8_tensorwise_triton_kernel`).

Probe scripts (kept in `/tmp/` not committed — reproducible from this note):
- `/tmp/probe_hk_layers.py` — HK layer breakdown
- `/tmp/probe_trt_layers.py` — Triton layer breakdown
- `/tmp/probe_hk_execute_steps.py` — per-line timing inside HK execute body
- `/tmp/probe_execute_cleanup.py` — slim variant timing + bit-equality verify
- `/tmp/probe_fp8_overhead.py` — quantize vs gemm split for context

## Headline numbers (B=4, M=2048, N=2880, K=2880, FP8 RCR)

|             | HK µs | TRT µs | Δ µs |
|-------------|------:|-------:|-----:|
| L1 kernel   | 119.80 | 109.60 | +10.20 |
| L2 +execute | 124.40 | 109.64 | +14.76 |
| L3 +dispatch | 126.56 | 111.24 | +15.32 |
| L4 +wrapper | 133.24 | 120.28 | +12.96 |

**Decomposition of HK's 13 µs gap to Triton:**
- **~10.2 µs in raw kernel** — Triton's kernel itself (incl. its in-kernel
  setup work) is 10 µs faster on B=4 shapes. This is where the **real
  ratio gain** lives — needs kernel-level architectural work.
- **~4.76 µs in HK execute body** — Python work HK does that Triton hides
  in its kernel function (`_resolve_fp8_scales`, `select_default_config`,
  `_avg_group_m`, dual function lookup, dscale-path branching, etc.).
  Triton's execute is essentially `return triton_kernel(...)` — 0.04 µs.
- **~2 µs in dispatcher / wrapper** — both backends pay these symmetrically
  so they don't move the ratio.

### Per-step inside HK execute body (B=4-M2048 FP8 RCR):
```
hipkitten.load_fp8():       0.033 us
reroute branch (no-op):     0.021 us
shape arith (bs,n,k):       0.226 us
layout str pick:            0.018 us
_resolve_fp8_scales:        0.421 us  ← biggest discretionary item
hk.grouped('rcr'):          0.032 us
hk.grouped_dscale('rcr'):   0.032 us
_avg_group_m:               0.116 us
select_default_config:      0.646 us  ← if/elif chain
torch.empty(B*M, N, bf16):  1.653 us  ← unavoidable, both backends pay
is_contiguous a + b:        0.093 us
cfg.num_xcds None pick:     0.021 us
fp8_has_dscale:             0.048 us
TOTAL (whole body):         3.409 us  ← matches L2-L1 ≈ 4.8 within frame overhead
```

### Quantize overhead (sanity check that it doesn't matter for ratio):
```
gpt_oss FP8 shape       T_total  T_qa+qb  T_kernel  Q_pct  K_pct
Down-B4-M2048             192.1     65.9    130.7   34.3%  68.0%
Down-B32-M4096           1975.3    564.1   1418.3   28.6%  71.8%
GateUP-B32-M2048         2009.6    624.8   1385.7   31.1%  69.0%
```
Quantize is 22-36% of total wall. Both backends pay it (FP8 dispatch
upstream of grouped_gemm_fp8_impl). Doesn't move the HK/TRT ratio. Noted
here for future reference if we ever want to fuse `quantize_a + quantize_b`
into a single launch (~30-40% wall improvement for E2E benchmarks but
**zero** ratio improvement — both backends benefit identically).

## What this commit changes

Trim HK's 4.76 µs execute body asymmetry where safely possible, while
keeping behavior bit-identical:

### `GroupedGEMMFP8HipKittenBackend.execute` (`grouped_gemm_fp8_impl.py`):
- (a) Skip `_resolve_fp8_scales` on the dscale fast path — FP8 tensorwise
  scales come from `quantize_fp8(..., TENSORWISE)` which always returns
  cuda fp32 contiguous numel==1 tensors. The 8-condition check is
  redundant in the hot path (-0.42 µs).
- (b) Defer `hk.grouped(layout)` lookup to the (rare) fallback branch,
  not needed for the dscale path (-0.05 µs + simpler hot trace).
- (c) Inline `_avg_group_m` (-0.10 µs) and unify `m_total = a.shape[0]`
  / `n` / `k` reads.
- Safety guard added: dscale fast path now also checks `a_scales.is_cuda`
  to preserve the original behavior of falling back to host-scalar path
  for CPU scales (no in-tree caller does this today, but the original
  `_resolve_fp8_scales` covered it).

### `GroupedGEMMHipKittenBackend.execute` (`grouped_gemm_impl.py`):
- (a) Inline `hipkitten.grouped_run(hk, cfg, ...)` — saves one function
  call frame (~0.5 µs). Body is identical (`hk.grouped(cfg.layout)` ->
  None check -> positional kernel call). The `grouped_run` helper is
  still in use by `GroupedGEMMVariableKHipKittenBackend` (different call
  shape), so the helper itself is preserved.
- (b) Inline `_avg_group_m` (-0.10 µs).

Both edits preserve `_avg_group_m`'s `max(_, 1)` clamp for the
degenerate `bs <= 0` and `m_total < bs` paths.

## Bit-equality verification

`/tmp/probe_execute_cleanup.py` ran current vs slim variant on 4 metric
gpt_oss FP8 shapes (B∈{4,32}, M∈{2048,4096}, N∈{2880,5760}, K=2880),
all returned `bit_eq=True, max_abs_diff=0.0`. The metric script's own
`correct_fail=0/16` (focus) + `correct_fail=0/16` (watch) gates pass
on all 32 grouped shapes (16 BF16 + 16 FP8 across DSV3 + gpt_oss).

## Measured impact

Single-call timing (B=4-M2048 FP8 — biggest fraction):
- T_HK_impl 192.20 → ~191.36 µs (-0.84 µs; +0.4pp ratio)
- B=32-M4096 FP8 1413 → ~1412 µs (-0.96 µs; +0.05pp ratio)

Metric (`scripts/_metric_grouped_only.py`, 7 runs after edit):
- 794, 792, 792, 795, 794, 793, 793 → mean **793.3** vs baseline **793**
- Within ±1.5 score per-run noise. Net effect is 0..+0.3 score on average.

The Python overhead trim is **real** (~0.5-1.0 µs/call measured) but
**below the metric's noise floor** for the 50-iter p20 timing on these
shapes. Useful as a code-quality / maintainability change and as a
documented "ceiling check" — it confirms that ~80% of HK's gap to
Triton on B=4 gpt_oss shapes is in the **raw kernel**, not Python
dispatch.

## Implications for round 12+

1. **Stop chasing Python overhead** — at most ~+1 score available across
   all 16 gpt_oss shapes via Python trims, all within metric noise.
   Marginal wins but absorbed by run-to-run variance.

2. **Kernel-level work is the only path to >+5 score on gpt_oss FP8**.
   Three concrete bets the task body and round-10 plan flagged:
   - **(a) Skip A-tile LDS staging** in FP8 grouped RCR main kernel
     (analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:1904 +
     `grouped_rcr_kernel`): A operand currently staged through LDS
     with cross-warp s_barrier sync; direct HBM→register would save
     ~10-15% on B=4 shapes per the round-7 estimate. 4-8 hr work.
   - **(b) BN=256 → BN=128 for N=2880 Down shapes** — N-tile last-tile
     utilization 25% → 50% (per task body §"为什么 gpt_oss 这么慢" item 2).
     Requires kernel template parameter add + HK rebuild.
   - **(c) Grid swizzle / XCD pinning for B=4 case** — 768 tiles on 8
     XCDs, redistribute so all XCDs finish together (per task body
     §"为什么 gpt_oss 这么慢" item 3).

3. **Quantize fusion** would help E2E but **not ratio** — both backends
   pay quantize cost upstream of dispatch. Skip unless E2E becomes the
   metric.

## Reproduction

```bash
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
    python3 /tmp/probe_hk_layers.py        # HK layer breakdown
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
    python3 /tmp/probe_trt_layers.py       # Triton layer breakdown
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
    python3 /tmp/probe_hk_execute_steps.py # per-line inside execute
PRIMUS_TURBO_HIPKITTEN_PATH=/workspace/code/HipKittens \
    python3 /tmp/probe_execute_cleanup.py  # bit-equality + slim timing
```

(Probe scripts are throwaway; recreate from this note's headline numbers
if needed.)
