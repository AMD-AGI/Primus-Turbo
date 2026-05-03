# Round-41 BF16 grouped — DIAGNOSTIC: fwd vs bwd wall split

## Context

R38-R40 all attacked the BF16 forward path (FUSE K-tail, KI specialization,
fuse epilog reorder) and all FALSIFIED with sub-noise / negative deltas.
The metric has been stuck at ~885 ± 5 for 11 rounds. R40's note
explicitly recommended a wall-decomposition diagnostic round before
committing to another speculative kernel attack.

R41 is that diagnostic. **No code change** this round; the deliverable
is the data + the redirected attack surface for R42+.

## Data — fwd vs bwd torch.cuda.Event timing

`/tmp/probe_r41_wall_decomp.py` — for each shape, time HK and Triton
separately on:
1. fwd-only: `out = grouped_gemm(a, b, ..., trans_b=True)` (50-iter median).
2. bwd-only: `out.backward(grad_data)` against a pre-cached fwd `out`.
3. wall = fwd + bwd.

DSV3 first to warm the K%128==0 path (cold-start sync-fault workaround,
same as the metric script's canonical suite order).

```
                                   HK              Triton            ratios
shape                          fwd_ms  bwd_ms    fwd_ms  bwd_ms    fwd     bwd     wall   metric
DSV3-GateUP-B32-M2048           4.644  10.558    5.536  11.061   1.192  1.048  1.092  1.130
DSV3-Down-B32-M2048             1.401   3.185    1.631   3.466   1.164  1.088  1.112  1.113
gpt_oss-GateUP-B32-M2048 *worst 1.760   3.934    2.001   3.998   1.137  1.016  1.054  1.044
gpt_oss-Down-B32-M2048          0.916   2.108    1.062   2.126   1.159  1.009  1.054  1.058
gpt_oss-GateUP-B32-M4096        3.448   7.180    3.877   7.707   1.124  1.073  1.090  1.084
gpt_oss-GateUP-B4-M2048         0.222   0.551    0.252   0.585   1.135  1.060  1.082  1.100
Qwen3-Down-B32-M2048            0.748   1.353    0.865   1.447   1.157  1.069  1.100  1.105
```

(`metric` column is the ratio reported by `_metric_grouped_bf16_weighted_wall.py`
on the same shape — the small delta vs the probe's `wall` is run-to-run
noise; the trend matches.)

## Key finding 1: bwd is the dominant component AND the limiting ratio

**Wall share** (HK time, fwd / wall):
- DSV3-GateUP-B32-M2048: fwd 30.5 %, bwd 69.5 %.
- DSV3-Down-B32-M2048:   fwd 30.6 %, bwd 69.4 %.
- gpt_oss-GateUP-B32:    fwd 30.9 %, bwd 69.1 %.
- gpt_oss-Down-B32:      fwd 30.3 %, bwd 69.7 %.
- gpt_oss-GateUP-B4:     fwd 28.7 %, bwd 71.3 %.
- Qwen3-Down-B32:        fwd 35.6 %, bwd 64.4 %.

**bwd is always ≥ 64 % of wall.** Triton's distribution is similar
(fwd 30-37 %, bwd 63-70 %), so the bwd-dominated wall structure is
real, not a metric artifact.

**Per-component ratios** (HK_TFLOPS / TR_TFLOPS):
- fwd ratio: **1.124-1.192** — HK is 12-19 % faster than Triton on fwd.
- bwd ratio: **1.009-1.088** — HK is 1-9 % faster on bwd, but
  approaching parity / sometimes regressing on the worst shapes.
- wall ratio = weighted geomean ≈ (fwd_r^0.30 * bwd_r^0.70).

The wall-ratio gap to 1.25 is **dominated by bwd**, not fwd. Even if
fwd was lifted to 1.30, gpt_oss-GateUP-B32-M2048 wall ratio would
only reach ~1.30^0.31 * 1.016^0.69 ≈ 1.0866 — still capped well below
1.25. Bwd MUST move for wall to move.

## Key finding 2: R38-R40 all attacked the wrong axis

| Round | Lever                                         | Target axis | Outcome |
|-------|-----------------------------------------------|-------------|---------|
| R38   | Port FP8 R37 K-tail [b0,b1,a,a_kt1] reorder    | fwd RCR     | sub-noise |
| R39   | fuse KI=44 specialization                      | fwd RCR     | -41 (spill) |
| R40   | non-fuse KI={24,32,48,88}                      | fwd RCR     | -8 (spill) |

All three rounds were attacks on **`grouped_kernel<L, KI, FUSED_KTAIL>`**,
which is the fwd path (and also the H4-rerouted dA RCR fuse path —
which IS bwd, but a small dA fraction of bwd). They left the dB var-K
path completely untouched.

Even on dA via H4 RCR (a "bwd" attack via the fwd kernel's path), the
unroll savings can't help the wall ratio because the dA wall fraction
× the achievable per-kernel speedup × weight is below the metric noise
floor (R40 v2 ki=88 specifically showed +0.5 score on a clean-compile
dA-only specialization — exactly as the diagnostic data here predicts).

## Key finding 3: R40's case 88 was the wrong ki value (kernel never hit it)

`g.ki = g.fast_k / K_STEP` with `K_STEP = 64` (line 7 of
`kernel_bf16_dynamic.cpp`). For gpt_oss-GateUP dA H4 RCR, kernel-side
K = N_fwd = 5760 → fast_k = 5760 (already 128-aligned, K_REM=0) →
ki = 5760 / 64 = **90**, NOT 88.

R40 added `INSTANTIATE_K_GRP(88) + case 88: launch_one_grouped<L, 88>`
expecting it to capture this shape — but the runtime dispatch never
hits case 88 because `g.ki == 90` falls through to default (KI=0).

The 4-run mean +0.5 score in R40 v2 was therefore noise (the new ki=88
explicit instantiation in the .so is dead code). This doesn't change
the falsification (sub-noise stays sub-noise) but it does mean the
ki=88-as-dA-H4 mechanism story in the R40 commit message was wrong.

For future reference, the actual ki values for K%128==0 metric paths
(fast_k = K, ki = K / 64):

```
shape                         layout  K (kernel)   ki    case in switch?
DSV3-Down fwd                 RCR     2048         32    no  → default
DSV3-GateUP fwd               RCR     7168         112   YES (case 112)
DSV3-Down dA RRR              RRR     N_fwd=2048   32    no  → default
DSV3-GateUP dA RRR            RRR     N_fwd=4096   64    YES (case 64)
Qwen3-Down fwd                RCR     1536         24    no  → default
Qwen3-GateUP fwd              RCR     4096         64    YES (case 64)
Qwen3-Down dA RRR             RRR     N_fwd=4096   64    YES (case 64)
Qwen3-GateUP dA RRR           RRR     N_fwd=3072   48    no  → default
gpt_oss-Down fwd RCR FUSE     RCR     fast_k=2816  44    fuse path (KI=0)
gpt_oss-Down dA via H4 fuse   RCR     fast_k=2816  44    fuse path (KI=0)
gpt_oss-GateUP fwd RCR FUSE   RCR     fast_k=2816  44    fuse path (KI=0)
gpt_oss-GateUP dA via H4      RCR     5760         90    no  → default  *R40 added wrong 88
```

## Key finding 4: dA-only / dB-only torch probe is broken

Attempted a second probe `/tmp/probe_r41_dadb_decomp.py` to split dA
vs dB by toggling `requires_grad` on `a` only / `b` only. Result:
dA-only time ≈ dB-only time ≈ full bwd time on every shape. The custom
autograd Function for `turbo.ops.grouped_gemm` always computes BOTH
grad_a and grad_b in its backward; PyTorch returns only the requested
output but the kernel work is unchanged.

Splitting dA from dB requires either:
- Calling the lower-level dispatch (`hipkitten.grouped_run` for dA,
  `hipkitten.grouped_run_var_k` for dB) directly, bypassing the
  autograd Function. ~1 hour of code archaeology to replicate the
  dispatch correctly.
- Inserting per-call cuda events inside the autograd Function's
  backward. Cleaner; ~30 min of Primus-Turbo edit.

Neither was done this round. R42 if pursuing the bwd attack should add
this instrumentation as the first step before kernel work.

## R42 attack surface

Confirmed by R41 data, ranked by expected leverage:

### Lever V (var-K dB kernel) — HIGHEST expected leverage

`grouped_var_k_kernel` (line 4511) is the dB path. Currently:
- Only KI=0 instantiated (line 4687: "round-2 can add specializations
  once the numerics are validated and bench data identifies hot ki
  values").
- 256 VGPRs / 0 VGPR spill / 2 occupancy (clean) — verified in R40
  build log line 4531.
- Untouched for many rounds; presumably has unexplored optimization
  surface.

This kernel handles **half of the bwd time** on every shape. If the
HK_dB / TR_dB ratio is the same as the overall bwd ratio (~1.02-1.07),
lifting it to 1.15 would close ~50 % of the wall gap — score bump
estimate of +20-50.

### Lever D (dA RRR fuse) — uncertain

`grouped_kernel<L=RRR, KI, true>` (RRR fuse) is currently disabled by
`BF16_RRR_FUSE_PROBE=0` due to documented phantom-read bugs (R3-R29
notes). gpt_oss-Down dA H4 reroutes to RCR fuse to bypass this; DSV3
Down dA goes to native non-fuse RRR. Re-enabling RRR fuse with a
working numerics path would simplify dispatch and possibly speed up
dA backward, but the bug is subtle (~25 % cells phantom-read) and
likely needs a full kernel rewrite.

### Lever K (dA RCR via H4 — already there) — exhausted

R40 + R41 confirmed this is metric-detectable but sub-noise even when
the kernel speedup is real (small wall fraction × small per-kernel
delta).

### Lever PROFILE (rocprofv3 var-K MfmaUtil) — free info

If pursuing Lever V, first capture `rocprofv3 valuMfmaUtil` on the
var-K dB kernel for gpt_oss-GateUP-B32-M2048. R9's PMC report was
forward-only. Knowing var-K's MFMA utilisation (compute-bound vs
HBM-bound) is the precondition for picking the right var-K
optimisation.

## Suggestion for R42

1. **Step 1 (free)**: rocprofv3 var-K kernel MfmaUtil on gpt_oss-GateUP-
   B32-M2048 (single-shape PMC capture). 5 min.
2. **Step 2**: instrument autograd Function backward with cuda events
   to split dA/dB times. Probe gpt_oss-GateUP-B32-M2048 to confirm dB
   wall fraction and HK_dB / TR_dB ratio.
3. **Step 3 (if dB ratio < dA ratio)**: var-K kernel KI specialization
   for the hot ki values OR small kernel-body change (load coalescing,
   barrier reduction, MFMA scheduling). Expected +5-15 score per
   focused win.

If step 1+2 reveal dB is NOT the bottleneck (e.g., dA dominates with
its own slow ratio), pivot to dA RRR work-stealing or a different
attack on the legacy `grouped_ktail_kernel_lds_rrr` path that R37
identified as 2-launch-bound.
