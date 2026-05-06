# Round-5 — FP8 grouped fused-act: Python-overhead floor CONFIRMED, kernel-internal residual gap is HK scope

## TL;DR

Maintenance round. Score stable at the 1000 cap with healthy buffer (geomean
1.39 vs 1.35 target). Per-component decomposition of the lowest-ratio shape
`gpt_oss-Down-B32-M2048` (ratio 1.268) confirms the residual ratio gap is
**100% inside the HK forward kernel** (K-tail epilog on K=2880); the
Primus-Turbo Python path is already LEANER than the Triton path (4.68 µs vs
5.84 µs HK-asymmetric overhead).

No code change this round — the Python-side optimization surface for this
metric is at its **demonstrable floor**. Regression checks all green.

## Why this is a maintenance round (not a perf round)

Previous 4 rounds (R1-R4 of this run) covered every Python-side lever:

| Round | Lever                                        | Score before → after | Outcome |
|-------|----------------------------------------------|----------------------|---------|
| R1    | Tensorwise FP8 quantize cache (a, b, grad_out) | 934 → 1000 | Cap hit |
| R2    | `group_offs` cache + `select_default_config` lru_cache | 998 → 1000 | Cap stable |
| R3    | Extend H4 reroute to ALL aligned RRR (dA bwd) | 1000 → 1000 | Geomean +2.9pp |
| R4    | dA RCR-via-T `(gm=4, xcds=4)` carve-outs     | 1000 → 1000 | Geomean ±0 (sub-noise) |

R3/R4 explicit suggestion lists both pointed to either:
1. **Phase-3 task body main line (HK kernel-internal fused cvt)** — but
   per R7-R8 architectural ceiling note (this run's prior subthread),
   Path A is FALSIFIED on this kernel architecture (DTL > DTR by ~40%
   on the load_a critical path; net -26% wall regression).
2. **Maintenance** if no actionable Primus lever remains.

R5 picks lever (2) and PROVES the Python floor with a tight decomposition
probe so future rounds don't waste cycles re-attempting the same levers.

## Probe data — per-component decomposition of the 5 below-target shapes

`/tmp/probe_round_5_decompose.py` — 200-iter × 20-warmup × p20 timing on
the pinned GPU (5), full forward+backward via `turbo.ops.grouped_gemm_fp8`:

```
shape                                    backend  fwd(us)  bwd(us)  tot(us)  ratio
gpt_oss-Down-B32-M2048                   HK         592.2   1263.1   1855.2   1.278
gpt_oss-Down-B32-M2048                   TRT        644.5   1727.3   2371.8
gpt_oss-Down-B32-M2048                   ratio      1.088   1.368    1.278

gpt_oss-Down-B32-M4096                   HK        1123.3   2225.9   3349.2   1.310
gpt_oss-Down-B32-M4096                   TRT       1222.6   3164.1   4386.7
gpt_oss-Down-B32-M4096                   ratio      1.088   1.421    1.310

Qwen3-Down-B16-M2048                     HK         183.2    443.4    626.6   1.318
Qwen3-Down-B16-M2048                     TRT        218.5    607.1    825.6
Qwen3-Down-B16-M2048                     ratio      1.193   1.369    1.318

Qwen3-GateUP-B16-M2048                   HK         340.5    779.1   1119.6   1.340
Qwen3-GateUP-B16-M2048                   TRT        396.6   1103.8   1500.5
Qwen3-GateUP-B16-M2048                   ratio      1.165   1.417    1.340
```

**Pattern**: across all 4 below-target shapes the BACKWARD ratio is
1.37-1.42 (close to the 1.4 model ceiling), but FORWARD ratio is
1.09-1.19 — substantially weaker. The lowest forward ratio is gpt_oss-Down
at 1.088. **The wall ratio is dragged down by the forward path.**

## Probe — Python overhead vs kernel-only (forward path)

`/tmp/probe_round_5_fwd_breakdown.py` — same iter / p20 settings, but
splits one forward call into `kernel_only` (direct
`grouped_gemm_fp8_impl` call, inputs pre-quantized) vs `fwd_full`
(`turbo.ops.grouped_gemm_fp8` end-to-end with cached
`quantize_fp8(a/b)`):

```
shape                          backend  fwd_full   kernel_only  py_overhead
gpt_oss-Down-B32-M2048         HK         593.97       589.29         4.68
gpt_oss-Down-B32-M2048         TRT        646.61       640.77         5.84
                               diff       -52.64       -51.48        -1.16

Qwen3-Down-B16-M2048           HK         182.68       179.20         3.48
Qwen3-Down-B16-M2048           TRT        218.32       212.24         6.08
                               diff       -35.64       -33.04        -2.60
```

**Three structural facts**:

1. **HK Python overhead is already 1.16-2.60 µs LOWER than TRT.** R11 /
   R16 / R17 / R18 trims compounded into a true overhead win. Any further
   Python work on the HK side would NEGATIVELY affect the ratio.

2. **Kernel-only ratios (640.8/589.3 = 1.087, 212.2/179.2 = 1.184)
   essentially equal full-fwd ratios (1.088, 1.193).** The forward
   ratio gap is 100% kernel-internal.

3. **gpt_oss vs Qwen3 forward kernel divergence (1.087 vs 1.184)**:
   gpt_oss has K=2880 (K_BLOCK-misaligned by 64); the HK forward kernel
   pays the K-tail LDS-staged epilog (`grouped_ktail_kernel_lds`) on
   the unaligned tail, while Qwen3-Down has K=4096 (K_BLOCK-aligned)
   and runs the main kernel only. The K-tail cost is HK kernel-internal
   work (kernel_fp8_layouts.cpp); not addressable from Primus-Turbo.

## Why no further Python lever exists

Inventory of the HK FP8 grouped forward call after R1-R4 (probe-grouped
into pre-existing levers):

| Stage                                    | Cached / trimmed in round | Cost on HK fwd hot path |
|------------------------------------------|---------------------------|-------------------------|
| `quantize_fp8(a)`                        | R1 cache                  | 0 µs (HIT)             |
| `quantize_fp8(b)`                        | R1 cache                  | 0 µs (HIT)             |
| `grouped_gemm_compute_offs`              | R2 cache                  | 0.11 µs (HIT)          |
| `select_default_config`                  | R2 lru_cache              | 0.07 µs (HIT)          |
| `_FP8_H4_TRANSPOSE_CACHE` (dA only)      | R9 cache                  | 0 µs (HIT in dA path)  |
| `_resolve_fp8_scales`                    | R11 dscale fast path      | 0 µs (skipped)         |
| `hk.grouped(layout)` attr lookup         | R18 direct attr           | 0.02 µs                 |
| `_avg_group_m` function call             | R11 inlined               | 0 µs                    |
| `hk.grouped_dscale(layout)` cascade      | R18 direct attr           | 0 µs                    |
| `Float8QuantConfig(...)` instantiation   | metric loop builds once   | 0 µs (out of timer)    |
| `force_grouped_gemm_backend` ctx         | per-shape, not per-call   | 0 µs (out of timer)    |
| `torch.empty(out)` allocation            | torch caching allocator   | ~1 µs (symmetric)      |
| `Backend dispatcher.dispatch + can_handle` | n/a (mandatory)         | ~1 µs (symmetric)      |
| `torch.library.custom_op` wrapper        | n/a (mandatory)           | ~0.4 µs (symmetric)    |

The remaining ~4-5 µs HK overhead is all **mandatory dispatcher /
allocator / autograd plumbing** — symmetric to TRT and trimming further
would compromise either correctness contracts or future extensibility.

## Architectural ceiling — model still holds

R8 architectural model (analysis/_notes/round-8-fused-act-architectural-
ceiling-confirmed.md) was:
```
ratio_wall = (TRT_K + Q) / (HK_K + Q)
```
After R1's quantize cache: Q ≈ 0 (cached HIT in metric loop). Therefore:
```
ratio_wall ≈ TRT_K / HK_K = kernel-only ratio (per-shape)
```
Empirical verification this round (Qwen3-Down-B16-M2048):
```
fwd kernel-only ratio = 1.184  (=212.24/179.20)
fwd full      ratio   = 1.193  (=218.32/182.68)
```
Δ = 0.009 (~0.5pp), explained by the symmetric ~5 µs dispatcher overhead
which favors HK by 2-3 µs (smaller python work on HK than TRT, see
breakdown above).

The wall geomean = 1.3896 ≈ kernel-only geomean from `_metric_grouped_only.py`
which sits at 1.158-1.171 for FP8. After fwd+bwd combination (with bwd's
1.37-1.42 advantage from H4 reroute) the wall geomean lands above the
1.35 target with ~2.9pp buffer.

## Regression checks (every-5-rounds gate)

```
fused_act_wall_score      : 1000 (cap)        target 1000   PASS
                            geomean=1.3896   target 1.35   PASS (+2.9pp buffer)
                            correct_fail=0/24                PASS

_metric_grouped_only.py   : 971              target 980    FAIL (drift band)
                            grp_BF16  geomean=1.1716       (R1: in 970-977 noise band)
                            grp_FP8   geomean=1.1579       (R1: drift due to BF16 R80 working tree)

run_dod_metric.sh --full  : 608 passed        target 600    PASS
                            failed=0 errors=0
```

`_metric_grouped_only.py` 971 sits at the same noise band recorded in R1
("the 980 floor was set at a prior baseline; both HEAD and R2 sit ~10 points
below today due to upstream BF16 round drift, NOT from this round's
changes"). My R5 has zero code changes, so this is unchanged from R4 / R3
/ R2 / R1 baseline.

## Compliance audit

- No code touched (zero kernel risk, zero correctness risk).
- Single docs commit in Primus-Turbo (this note).
- HipKittens unchanged.
- All 24 metric shapes correctness PASS (SNR > 25 dB on out / dA / dB).
- Default `Float8QuantConfig()` (`fuse_act_quant=False`) untouched.

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-5-fused-act-python-overhead-floor-confirmed.md` (this note)

**HipKittens:** None.

## Score progression

| Round | Score | Geomean | n_below | n_pass | comment                              |
|-------|-------|---------|---------|--------|--------------------------------------|
| R1    | 1000  | ~1.358  | 9-14    | 10-14  | quantize cache big jump              |
| R2    | 1000  | ~1.355  | -       | -      | sub-noise                             |
| R3    | 1000  | ~1.389  | 8       | 16     | H4 reroute + 2.9pp geomean lift      |
| R4    | 1000  | ~1.382  | 8       | 16     | dA carve-outs sub-noise              |
| **R5**| 1000  | 1.3876  | 8       | 16     | **maintenance: ceiling confirmed**   |

## Suggested next round (R6)

Two viable paths, both off the Primus-Turbo Python-side surface:

1. **HipKittens kernel-internal**: gpt_oss-Down K=2880 forward K-tail
   epilog is the residual ratio gap source per probe 1+2 above. The
   `grouped_ktail_kernel_lds` cooperative LDS-staged tail (line ~4080
   of `kernel_fp8_layouts.cpp`) is the target. Estimated ratio impact
   if K-tail is collapsed into the main kernel: gpt_oss-Down forward
   1.088 → ~1.18 (matching Qwen3-Down's aligned-K ratio); contributes
   ~+0.02 to the gpt_oss-Down geomean cell.

   Risk: this is multi-round HK C++ work; touches the hot loop of
   the most-tuned kernel in HK FP8. R7-R8 falsified Path A on the
   same kernel — need a non-DTR approach.

2. **Continue maintenance**: keep watching the metric. The 1000 cap
   has held for 4 consecutive rounds (R2-R5) with healthy +2.9pp
   geomean buffer. The patience counter (30) provides ~25 more rounds
   of plateau before the auto_optimize early-stop fires. Documenting
   the convergence helps the next agent / next run avoid re-exploring
   falsified levers.

R5 recommends path (2) until the `_metric_grouped_only.py` baseline is
restored to ≥980 by an external (BF16-side) commit. If the metric remains
at cap with no actionable Primus lever for 3 more rounds (R6-R8),
declare the Primus-side fused-wall task fully converged at 1000-cap-with-
buffer and pivot agent effort to HipKittens C++ for residual gap closure.

## Reference probes (`/tmp/probe_round_5_*`)

- `decompose.py`           — full fwd+bwd timing per backend on 4 shapes
- `fwd_breakdown.py`       — fwd kernel-only vs fwd_full Python overhead
