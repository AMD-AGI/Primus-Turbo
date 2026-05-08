# Round-19 — gpt_oss FP8 kernel-only ceiling: positional torch.empty trim FALSIFIED, metric noise floor characterised

**Date**: 2026-05-08 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (HEAD `33562328` → R19)
**Scope**: gpt_oss_20B Balanced FP8 kernel-only, 8-shape suite × 3 sections (24 metric cells).
**Goal**: Continue R18's host-overhead trim main thread by replacing the tuple-arg
`torch.empty((m_total, n), …)` with the positional form `torch.empty(m_total, n, …)` in both
`GroupedGEMMFP8HipKittenBackend.execute` (fwd / dgrad RCR/RRR path) and
`GroupedGEMMFP8VariableKHipKittenBackend.execute` (var-K wgrad CRR path).

## Bottom line

**Falsified.** The pure-Python probe `_probe_round_19_empty_alloc.py` confirmed the positional
form is **0.5 µs faster** than the tuple form for the canonical alloc shape (`(8192, 2880)` bf16
cuda → 1.562 µs vs 1.062 µs), and `_probe_round_19_3d.py` confirmed the same gain on the var-K
3D output shape (`(4, 2880, 2880)` → 1.562 vs 1.071 µs). But the gpt_oss FP8 metric showed **zero
movement** (8-run median identical to the R18 baseline at 687) when the trim was applied to the
production execute paths. The probe-vs-metric gap reveals a metric noise floor of ~±0.5 µs / call
(absorbing all sub-1 µs host trims after R18).

| Trim | Pure-Python probe | Metric Δ (8-run median) | Verdict |
|---|---|---|---|
| R18 dispatcher fast-path extension (`user_backend == HIPKITTEN`) | -1.48 µs | +5 pts (682.5 → 687) | LANDED |
| R19 positional torch.empty (2D fwd, 3D var-K) | -0.5 µs | 0 pts (within noise) | FALSIFIED |

The R18 trim landed because its 1.48 µs / call delta is **above** the metric noise floor; R19's
0.5 µs / call delta is **at or below** the floor and gets absorbed into the run-to-run spread.

## Probe protocol

### A. Stage-incremental host-overhead profile (R18 anchor probe re-run)

- Script: `scripts/_probe_round_18_python_overhead.py`
- Anchor: `Down-B4-M2048 fwd` (smallest gpt_oss kernel — most Python-dominated, ~92.88 µs / call
  with ~4 µs of pre-R18 Python overhead = ~4 % of wall, the highest fraction in the 24-cell suite).
- Method: 5 stage-incremental wrappers around the bare `hk.grouped_rcr_dscale` call, each adding
  one element of the execute body. Times via `torch.cuda.Event.elapsed_time`, p20 of 50 warmup +
  2000 timed iterations.

**Post-R18 stage breakdown** (Down-B4-M2048 fwd, 1× run):
```
[stage 0] bare hk.grouped_rcr_dscale (out pre-alloc):  93.08 µs  1460.0 TF
[stage 1] + torch.empty((m_total, n), …):              94.68 µs  1435.3 TF  Δ=+1.60 µs   ← R19 target
[stage 2] + select_default_config + cfg arg unpack:    94.96 µs  1431.1 TF  Δ=+0.28 µs
[stage 3] + shapes/avg_m/ternaries/contig:             95.40 µs  1424.5 TF  Δ=+0.44 µs
[stage 4] via Backend.execute(...):                    94.92 µs  1431.7 TF  Δ=-0.48 µs   ← run-to-run noise
[stage 5] via grouped_gemm_fp8_impl public op:         96.36 µs  1410.3 TF  Δ=+1.44 µs
```

After R18, the residual Python overhead (stage 5 - stage 0) is ~3.3 µs, dominated by the
`torch.empty` alloc at 1.60 µs (~48 % of remaining overhead). The other items (cfg lookup,
shape/avg_m/ternaries, contig checks) are sub-0.5 µs each.

### B. torch.empty alternatives micro-benchmark

- Script: `scripts/_probe_round_19_empty_alloc.py` (2D shape `(8192, 2880)`),
  `_probe_round_19_3d.py` (3D shape `(4, 2880, 2880)`).
- Method: pure Python `time.perf_counter_ns` over 20000 iters, p20.
- Sanity: dtype, device, shape, contiguity verified across all variants; output buffers are
  bit-equivalent.

```
2D (8192, 2880) bf16 cuda
  [A] torch.empty(size, dtype=, device=):           1.562 µs   ← current (tuple)
  [B] torch.empty(M, N, dtype=, device=):           1.062 µs   ← R19 winner (-0.50 µs)
  [C] base.new_empty(size, dtype=):                 1.292 µs
  [D] torch._C._VariableFunctions.empty(...):       1.542 µs
  [E] torch.empty(size, dtype=, device='cuda'):     1.642 µs   (str device costs +0.08 µs)
  [G] empty_strided(size, stride, dtype=, device=): 1.051 µs   (matches B)
  [H] torch.empty(... memory_format=cf):            1.593 µs
  [I] torch.empty(<cached tuple>, ...):             1.542 µs   (no help)

3D (4, 2880, 2880) bf16 cuda
  [A] torch.empty((G,N,K), …):                      1.562 µs   ← current
  [B] torch.empty(G, N, K, …):                      1.071 µs   ← R19 winner (-0.49 µs)
```

The positional form bypasses the tuple-unpack arg parser path inside `torch.empty`, saving ~0.5 µs
on a typical mid-sized shape. The savings come from skipping the IntList-from-tuple conversion in
the C++ side of the public `empty` overload.

### C. Metric gpt_oss_fp8_kernel score (8-run controlled comparison)

- Script: `scripts/_metric_gpt_oss_fp8_kernel.py`
- 8 runs back-to-back per state, no GPU pinning change between runs.

```
pre-R18 (fae4992):     680, 682, 682, 682, 683, 683, 684, 685   → median 682.5
R18    (33562328):     685, 686, 686, 687, 687, 690, 691, 692   → median 687
R19    (33562328+R19): 685, 686, 687, 687, 687, 688, 690, 693   → median 687
```

R18 vs pre-R18: **+4.5 score pts** (matching R18's commit-message claim of +5..+8 pts; the upper
end was a single high-tail run, the median is the honest measure).
R19 vs R18: **0 score pts** (medians identical; the +0.5 µs / call probe gain disappears in the
metric's per-call noise).

## Why R19 doesn't show up in the metric

The metric's `_time_op` (`scripts/_metric_hk_ratio.py:458-475`) records 50 iterations × `_time_op`
calls per cell × 24 cells = 1200 `cudaEventRecord` pairs per metric run. Each pair has its own
host + device sync overhead (~2-4 µs of jitter per pair from event-flag round-trips on ROCm 7.0)
that is **independent** of the kernel call's Python overhead. A 0.5 µs / call trim spread across
50 timing iters contributes ~25 µs total wall (= 0.5 µs × 50) — but the per-cell p20 already has
~25 µs of run-to-run spread from the 1200-event sync jitter, so the trim is statistically
indistinguishable from noise.

R18's 1.48 µs / call trim contributes ~74 µs (= 1.48 × 50) total — comfortably above the per-cell
noise floor — which is why it landed cleanly.

**Implication for future rounds**: Python-side host trims need to clear ~1 µs / call to register
on this metric. The remaining items in the execute body (cfg lookup 0.13 µs, shape/avg_m
ternaries 0.31 µs, contig checks 0.10 µs, is_cuda 0.06 µs) are individually well below the floor
and even bundled together (~0.6 µs) would land at the noise edge.

## What's still on the table

**Track A (host trims, ≤ 1 µs each)**: R19-class falsification space. Future trims would need to
**bundle** several sub-µs items into a single ≥ 1 µs change to register. Candidate bundle:
1. Hoist `_FP8_GRANULARITY_INT_TO_ENUM[granularity]` resolution outside the dscale fast path
   (~0.05 µs).
2. Precompute `cfg.num_xcds_or_zero` as a property on `HipKittenConfig` (eliminates the
   `is not None` ternary, ~0.05 µs / call).
3. Combine `is_contiguous()` checks via a single `tensor.is_contiguous()` per arg (already 1×
   each; no further trim).
4. Cache `out_dtype` int conversion (if applicable to any path).
**Bundled total**: ~0.15 µs / call — still well below the 1 µs floor. Track A is effectively
exhausted.

**Track B (kernel-template / kernel-source changes)**: documented falsifications already cover
the main angles (R16 binary-search-to-divide → VGPR spill, R63 KI_HINT specialisation → spill,
R4 RCR slots, R15 var-K fine-slots). Open ideas not yet probed:
1. **AGPR usage** for late-K accumulators on the var-K path (R47/R48 already explored AGPR-vs-VGPR
   on dense FP8; var-K hasn't been tested).
2. **Wave specialisation** for the K%128==64 K-tail block — currently every wave handles both the
   masked tail and the fully-aligned body. A two-wave-class split might allow the body waves to
   use a tighter inner loop without the K-tail mask overhead.
3. **Cross-XCD scheduling** for the small-grid wgrad shapes (Down-B4-M2048 wgrad at 1.5 wave-
   steps/CU, the worst metric cell). Currently all XCDs share the persistent grid; partitioning
   the grid into XCD-affinity sub-grids might recover some of the under-saturation tax.

These are deeper kernel work; each would take 2-5 rounds of probe + verify. R19 closes the host
trim chapter and the next round should pick up Track B.

## Files committed

- `analysis/_notes/round-19-fp8-positional-empty-and-metric-noise-floor-FALSIFIED.md` (this note)
- `scripts/_probe_round_19_empty_alloc.py` (2D probe)
- `scripts/_probe_round_19_3d.py` (3D probe)

No production code changes (the positional-empty trim was reverted after metric falsification).
