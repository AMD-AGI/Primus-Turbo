# Round 30 — H4 wall-breakdown profile + R28 transpose change non-reproducible in paired 5-run

## Goal

R29 concluded: "dispatch surface exhausted, next R30 main line is H4
fusion into HK kernel B-load". R30 opens that investigation by first
profiling the H4 transpose wall-fraction on the lowest-progress shapes
and by testing whether the R28-found `bf16_transpose_3d (128, 128) for
K==N` heuristic still moves the metric when applied fresh on this
round's baseline.

## H4 wall-breakdown profile

`/tmp/probe_round30_wall_breakdown.py` with a DSV3-GateUP warmup to
dodge the HK K-tail cold-start sync-fault. fwd-only path (trans_b=True)
does NOT trigger H4 (gate is `if not trans_b`); H4 fires only on bwd
dA where the impl is called with `trans_b=not ctx.trans_b=False`.

Results on idle MI355X (p20, 30-50 iter):

| shape | H4 transpose (standalone) | fwd-only | full wall (fwd+bwd, from metric TFLOPS) | H4 share of wall |
|---|---|---|---|---|
| gpt_oss-GateUP-B32-M2048 | 393 µs | 1747 µs | 5710 µs | **6.9 %** |
| gpt_oss-Down-B32-M2048   | 215 µs |  905 µs | 3020 µs | **7.1 %** |

Full wall computed from metric's `hk_tflops` and FLOPs = 6·(B·M)·N·K:
  * GateUP-B32-M2048: hk=1142 TF → wall = 6.52e12 / (1142 · 1e9) ≈ 5710 µs.
  * Down-B32-M2048:   hk=1080 TF → wall = 3.26e12 / (1080 · 1e9) ≈ 3020 µs.

**Headroom if H4 fully eliminated (via HK kernel fusion):**

| shape | current ratio | post-elim ratio | progress Δ | score Δ (weight 3 ÷ 40 · 1000) |
|---|---|---|---|---|
| GateUP-B32-M2048 | 1.049 | 5990/(5710-393) = 1.127 | +0.062 | **+4.6** |
| Down-B32-M2048   | 1.053 | 3180/(3020-215) = 1.134 | +0.065 | **+4.9** |
| GateUP-B32-M4096 | 1.085 | est 1.15            | est +0.05 | ~+3.7 |
| Down-B32-M4096   | 1.076 | est 1.14            | est +0.05 | ~+3.7 |

B=32 sum: **~+16-17 score** if H4 fully eliminated. B=4 shapes have
smaller H4 wall (55-60 µs vs 215-416 µs) but the ratio gap is similar
(~10 %) and weight is 3x, so another ~+3-5 score. **Total potential: +20-22 score.**

This confirms R27/R29's claim: "H4 fusion into HK kernel B-load is
the last structural lever that can move the BF16 24-shape wall metric
past the 892 plateau toward ~900-912."

## R28 (128, 128) heuristic re-tested in paired 5-run

R28 single-lever paired 5-run measured **+3.4** (sub-+5). R30 re-runs
the same paired-5 against this round's baseline (which has drifted
after R29's docs-only commit that didn't change behavior).

Same test harness (`/tmp/r30_paired_5run.sh`, git-stash the applied,
5 baseline runs, stash-pop, 5 applied runs, same metric):

```
baseline: 878, 880, 884, 884, 886   →  mean 882.4  (range 8, σ ~3.3)
applied:  879, 885, 885, 884, 878   →  mean 882.2  (range 7, σ ~3.4)
Δ = -0.2
```

**R28's +3.4 is NOT reproducible** in R30 paired 5-run. The difference
(R28 +3.4 vs R30 -0.2 ≈ 3.6 Δ) exceeds the 5-run σ, indicating R28's
paired-5 sample picked up session-warmup or thermal bias rather than
a real improvement. **The R28 finding is formally demoted: it is
within metric noise, not a sub-threshold positive.**

Kernel-level `bf16_transpose_3d` µs measurement is still real and
uniform-positive (3-run stable):

| shape | prod(256,128,nw=4) | new(128,128,nw=4) | Δ % |
|---|---|---|---|
| Down-B4  (K=N=2880) | 22.9 µs | 21.5 µs | **+5.94 %** |
| Down-B32 (K=N=2880) | 214.2 µs | 211.8 µs | +1.08 % |

But 2-3 µs saved out of 3000-5700 µs wall = < 0.1 % wall, well below
the metric's single-shot noise of ~1 %. Transpose µs isn't the
bottleneck; the **full H4 pass (394-215 µs)** is — and eliminating
that requires kernel fusion, not transpose-block tuning.

## num_warps=8 swept, first-probe artifact only

First probe showed +15 % on GateUP-B32 for `num_warps=8`. 3-run
stability check revealed:

| shape | nw=4 baseline | nw=8 (3-run mean) | 3-run Δ % |
|---|---|---|---|
| GateUP-B4 (128,256)  | 42.0 µs | 42.0 µs | +0.16 % |
| GateUP-B32 (128,256) | 419.6 µs | 419.1 µs | +0.14 % |
| Down-B4  (128,128)   | 22.9 µs  | 21.6 µs  | +5.88 % (is really just (256,128)→(128,128) gain, coincident with the nw switch at B<16) |
| Down-B32 (128,128)   | 214.1 µs | 211.8 µs | +1.05 % |

The initial "+15 %" for GateUP-B32 at nw=8 was first-run thermal /
cold-cache artifact; 3 independent runs show nw=8 matches nw=4
within noise on the (128, 256) block. **num_warps=8 is NOT a real
lever for BF16 transpose.**

## Decision

**No production change this round.** Evidence:

1. R28's (128, 128) K==N heuristic no longer moves the metric in a
   paired 5-run (-0.2 measured, vs R28's +3.4). The kernel-µs gain
   is real but too small relative to wall to surface.
2. num_warps=8 is not a real gain; first-probe showed +15 % on
   GateUP-B32 but 3-run mean is +0.14 %.
3. The only remaining structural lever (H4 elimination via HK kernel
   fusion) requires multi-round C++ work and cannot be attempted
   within a single session window.

Archive the profile findings for R31+.

## Files

* `analysis/_notes/round-30-bf16-grouped-h4-wall-breakdown-and-r28-nonreprod.md` — this note.
* `/tmp/probe_round30_wall_breakdown.py`, `/tmp/probe_round30_triton_numwarps.py`,
  `/tmp/r30_paired_5run.sh` — probe artefacts (not committed).

## Suggested R31 next step

Stop chasing dispatch-rule and block-shape heuristics. The remaining
score headroom is gated on eliminating the H4 transpose via HK kernel
fusion — specifically:

**R31 main line: investigate the RRR-layout B-load path in**
`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
(lines 917-1000 contain `layout == Layout::RRR` B-load with the
Round-7 "path A hybrid" comment; the phantom-read bug that forced
the H4 workaround is in this block). Goal: produce a fix that makes
RRR + K-tail + N-tail numerically correct, or add a new compile-time
path that handles one of those gate conditions (K%128 != 0 OR
N%256 != 0) without needing the standalone Triton transpose pass.

Concrete investigation steps for R31:

1. `git log --all --oneline /workspace/code/HipKittens/analysis/bf16_gemm/mi350x/` —
   enumerate the Rounds 3-7 "RRR K-tail fuse" attempts and their
   falsification rationale to avoid re-treading.
2. Read the LDS-staged `b_load` for RRR (kernel_bf16_dynamic.cpp
   `Layout::RRR` branches around lines 540-555 for LDS warp-subtile
   setup and 917-950 for the K-tail load) and compare to RCR's
   (which is known to work correctly on K-tail shapes).
3. If a 30-min read identifies a concrete bug location, attempt a
   patch + validate against `gpt_oss-GateUP-B32-M2048` downsized
   allclose. If PASS, time `bf16_transpose_3d` bypass: remove the
   H4 gate check for that shape and see how the metric ratio
   responds.

**R31 alt line (lower risk, smaller upside):** try to reduce the
Triton launch overhead of `bf16_transpose_3d`. R30 wall profile
shows 393 µs per H4 call on GateUP-B32; kernel-proper is ~340 µs
and the rest (~50 µs) is Python + Triton dispatch. Shaving 30 µs
(e.g., by switching `_fp8_transpose_3d_kernel` from JIT to pre-
compiled AOT) would save 30 µs × 4 shapes × weight 3 = ~+1.5 score.
Marginal but bounded-risk.

**R31 last resort (if C++ work blocks):** accept the 880-892 plateau
and pivot to DoD regression reduction (failed=608 last run). Not
score-positive on this metric but reduces cross-backend risk.
