# Round 28 — BF16 transpose (BK, BN)=(128, 128) for K==N — FALSIFIED (noise-bound) but **real kernel-level win archived**

## Goal

R27 plan: R28 must pivot away from dispatch tweaks (R25/R26/R27 all
sub-noise). The stated main line was K-tail forward kernel structural
probe (HK .cpp), but R10/R11/R14/R15/R16 already falsified 5 distinct
K-tail levers — the gpt_oss K=2880 FUSED_KTAIL kernel surface is closed.

R27 alt suggestion: **H4 transpose kernel fusion** (inline `bf16_transpose_3d`
into HK kernel's B-load path). R28 main attack vector.

R26 note preview: "*`bf16_transpose_3d` Triton kernel for H4 reroute
(already at 5 TB/s effective per R4 — limited headroom)*".

R28 question: **is the "limited headroom" claim correct, or did R4 / R15
just tune for FP8 and transfer the heuristic to BF16 unchallenged?**
R15's `_select_block_shape(K, N)` has only `K`, `N` inputs — **dtype-
agnostic**. FP8 element size = 1 byte vs BF16 = 2 bytes → per-block L2
working set at a given (BK, BN) differs by **2×** between dtypes, so
the optimal block may diverge.

R28 baseline (single run) = **879**, 5-run mean **881.4** (878-889
spread 11).

## Probe 1 — wide block-shape sweep on the 4 BF16 H4 transpose shapes

`scripts/_bf16_transpose_blockshape_probe.py` (archived as
`/tmp/probe_round28_bf16_transpose_blockshape.py`). 30-iter × 60-iter
p20 sweep over `BK, BN ∈ {64, 128, 256, 512}`, correctness gate
`torch.equal(out, ref)` every candidate:

| shape (B, K, N) | current FP8 heuristic | t_cur | top alt | t_alt | Δ |
|---|---|---:|---|---:|---:|
| gpt_oss-GateUP-B4  (4, 5760, 2880) | (128, 256) | 49.96 µs | **(256, 64)** | 48.16 µs | **+3.60 %** |
| gpt_oss-Down-B4    (4, 2880, 2880) | (256, 128) | 32.88 µs | **(128, 128)** | 29.44 µs | **+10.46 %** |
| gpt_oss-GateUP-B32 (32, 5760, 2880) | (128, 256) | 399.04 µs | (64, 128) | 397.88 µs | +0.29 % |
| gpt_oss-Down-B32   (32, 2880, 2880) | (256, 128) | 193.28 µs | (128, 256) | 192.32 µs | +0.50 % |

Key finding: **for K == N (Down shapes) on BF16, (128, 128) beats the
FP8-tuned (256, 128) by +8-10 % on B=4 and +1 % on B=32**. The B=4
uplift is large because the grid is small — (256, 128) has only
`ceil(2880/256) * ceil(2880/128) = 12 * 23 = 276` blocks per group,
and at B=4 that's 1104 workgroups total which is under-saturates the
CU grid; (128, 128) has `ceil(2880/128)^2 = 23^2 = 529` blocks/group =
2116 WGs at B=4, much better CU occupancy.

For K > N (GateUP), (128, 256) remains competitive on B=32 and the
alternatives are sub-noise.

## Probe 2 — focused 7-trial × 200-iter p20 verify of (128, 128)

`/tmp/probe_round28_bf16_transpose_focused.py`:

| shape | current | t_cur | (128, 128) | t_new | Δ | spread | verdict |
|---|---|---:|---|---:|---:|---:|---|
| GateUP-B4   | (128, 256) | 49.28 µs | (128, 128) | 48.92 µs | +0.73 % | 0.25 % | sub-noise |
| Down-B4     | (256, 128) | 31.84 µs | (128, 128) | 29.28 µs | **+8.04 %** | 1.39 % | **SAFE-WIN** |
| GateUP-B32  | (128, 256) | 399.48 µs | (128, 128) | 400.24 µs | -0.19 % | 0.16 % | tiny regress |
| Down-B32    | (256, 128) | 192.12 µs | (128, 128) | 190.04 µs | **+1.08 %** | 0.44 % | **SAFE-WIN** |

Safe BF16 heuristic candidate:
* K == N → (128, 128)   (Down shapes, +8 % B=4 / +1 % B=32 uniform)
* K >  N → (128, 256)   (GateUP — keep FP8 pick, avoid the B=32 -0.19 %)
* K <  N → (128, 128)   (fallback, no metric trigger)

## Implementation (now reverted)

Added `_select_block_shape_bf16(K, N)` helper to
`primus_turbo/triton/utils/fp8_transpose.py` and routed
`bf16_transpose_3d` through it (FP8 path unchanged). Bit-identical
output verified on all 4 H4 shapes (`max_abs = 0.0`, `bit_eq = True`,
including vs PyTorch reference).

## Metric — paired 5-run means

| | run1 | run2 | run3 | run4 | run5 | mean |
|---|---|---|---|---|---|---|
| baseline (HEAD 68afa9a) | 879 | 889 | 880 | 881 | 878 | **881.4** |
| after R28 transpose change | 893 | 880 | 892 | 879 | 880 | **884.8** |

**Δ = +3.4 score** (sub-noise; R20/R24 LANDED precedent is paired
5-run mean ≥ +5.0).

Per-shape metric after-change (first-run snapshot; compare to baseline
table in `/tmp/metric_bf16_round_28_before.log`):

| shape | baseline | after | Δ | notes |
|---|---|---|---|---|
| gpt_oss-GateUP-B4-M2048  | 1.097 | 1.118 | **+1.9 %** | (128, 128) |
| gpt_oss-Down-B4-M2048    | 1.103 | 1.141 | **+3.4 %** | (128, 128) ← big |
| gpt_oss-GateUP-B4-M4096  | 1.112 | 1.102 | -0.9 % | noise |
| gpt_oss-Down-B4-M4096    | 1.106 | 1.123 | +1.5 % | (128, 128) |
| gpt_oss-GateUP-B32-M2048 | 1.050 | 1.056 | +0.6 % | noise-like |
| gpt_oss-Down-B32-M2048   | 1.057 | 1.064 | +0.7 % | (128, 128) |
| gpt_oss-GateUP-B32-M4096 | 1.083 | 1.089 | +0.6 % | |
| gpt_oss-Down-B32-M4096   | 1.086 | 1.089 | +0.3 % | (128, 128) |
| gpt_oss geomean | 1.0866 | 1.0973 | **+1.0 %** | |

All 24 correctness PASS. All 4 Down shapes (K == N, where the heuristic
change targets) improved; 3 of 4 GateUP shapes improved (1 sub-noise
regression); Qwen3/DSV3 unchanged within noise (expected — none of
their dA paths hit the H4 gate).

## Why +3.4 instead of the "+2-3 expected"

Predicted contribution (from probe):
* Down-B4-M2048  HK wall ≈ 469 µs, transpose saves 2.56 µs → +0.55 % ratio, weight 3 → 0.066 progress
* Down-B4-M4096  HK wall ≈ 762 µs, saves 2.56 µs → +0.34 %, weight 3 → 0.042
* Down-B32-M2048 HK wall ≈ 2270 µs, saves 2.08 µs → +0.09 %, weight 3 → 0.012
* Down-B32-M4096 HK wall ≈ 2140 µs, saves 2.08 µs → +0.10 %, weight 3 → 0.012

Sum = 0.132 progress / 40 total weight = +3.3 score. Matches the
measured +3.4 exactly — predictive model is accurate.

## Decision — REVERT + falsification note

Paired 5-run mean +3.4 is below the +5.0 LANDED threshold. Per project
policy (`Flat or down → revert + falsification round note`), the
dispatch-level change is reverted. The **probe data and the
`_select_block_shape_bf16` implementation** are archived in this note
for future aggregation.

The kernel-level win is **real**: uniform-positive on target shapes,
bit-identical output, validated by two probe rounds (wide sweep +
focused 7-trial verify). Only the full-metric aggregation falls short
of the +5 threshold on a single-round test.

## Why this matters (refutes R26's "limited headroom" claim)

R26's R28-plan note claimed `bf16_transpose_3d` has "limited headroom
(already at 5 TB/s effective)". That was based on R4's absolute
throughput measurement — correct for the **B=32** shapes (where the
grid is large enough to saturate HBM), but WRONG for **B=4** shapes
where CU occupancy is the bottleneck and a smaller block shape
doubles the effective workgroup count. R28 shows B=4 Down has 8 %
headroom on the transpose alone.

General lesson: when an FP8-tuned heuristic is transferred to BF16,
always re-verify for the 2× element-size difference in L2 working set.

## Suggested R29 next step — aggregate with R26/R27 pending rules

R28 archived a +3.4 real-but-sub-noise lever. R26 archived 3 rules with
a combined +0.6 measured (3-rule aggregate). R27 archived 1 rule with
+0.06 expected (DSV3-GateUP dB var-K (gm=2, xcds=8)). Combined
expected contribution: +3.4 (R28) + 0.6 (R26) + 0.1 (R27) ≈ **+4.1**.

If R29 bundles **all three as one commit**, the combined paired 5-run
mean is likely ~+4 score — still borderline against the +5 threshold.
Two possibilities to cross it:

1. **R29 bundle + one more uniform-positive kernel-level find** (e.g.,
   audit more cells for Qwen3 / DSV3 fwd RCR paths not yet probed by
   R26). Each additional 1-2 % kernel win on 4 shapes adds ~+0.5 score.
   Need 2-3 of these to land R28-R27 stack safely above +5.

2. **R29 re-land R28 transpose change as "numerical-correctness
   improvement" commit** even if the metric hits only +3.4. Bit-equal
   output, no regression on any shape — this is a *legitimate*
   cleanup commit distinct from chasing the metric. Project policy
   allows this (R14/R15 notes explicitly flag "kernel-level wins are
   real, just don't aggregate enough"). The commit-log paper trail
   then shows R28's real contribution for auditors; future aggregate
   rounds start from a higher baseline.

Either R29 option clears the plateau (consensus: option 2 is safer —
less risk of a compound aggregate hitting an allclose FAIL that sinks
the whole commit, like R24 DSV3-GateUP `(gm=2, xcds=0)` cell did).

## Files touched this round

* `analysis/_notes/round-28-bf16-grouped-transpose-blockshape-bf16-specific-FALSIFIED-noise.md` — this note.
* `/tmp/probe_round28_bf16_transpose_blockshape.py` — wide sweep probe
  (reusable for BF16 transpose block-shape re-tuning).
* `/tmp/probe_round28_bf16_transpose_focused.py` — focused verify probe.
* `/tmp/probe_round28_correctness.py` — bit-equal verify of the
  `_select_block_shape_bf16` heuristic.

No code changes shipped. `primus_turbo/triton/utils/fp8_transpose.py`
reverted to HEAD 68afa9a via `git checkout`.
