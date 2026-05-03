# Round 44 — BF16 grouped, dispatch tune `(group_m, num_xcds)` for gpt_oss-GateUP dA H4 RCR — FALSIFIED

## Goal coming in

R43 falsification note (vmcnt drop on var-K epilog, sub-noise) closed
two surfaces and recommended a pivot:

> Pivot from kernel-body micro-opts (cheap to try, all sub-noise to
> date) to **dispatch / config tuning**, since:
> - KI specialization closed (R39/R40/R42).
> - Per-tile epilog wait closed (R43).
> - The 4 worst metric shapes — `gpt_oss-{GateUP,Down}-{B32-M2048,
>   B32-M4096}` — span the same `select_default_config` route.

R44 followed that pivot — but with the dA/dB-decomposed picture from
R41, not just "dispatch" in general. R41 showed bwd is the bottleneck
(64-71% of wall) and the lowest dA ratio in R42 was on `gpt_oss-Down-
B32-M2048` (0.942) — but R44 metric-row review re-targeted to
`gpt_oss-GateUP` instead, since GateUP has 2 of the 4 worst shapes
(B32-M2048: 1.049 wall; B32-M4096: 1.088 wall).

## Hypothesis (R44)

The `select_default_config` Round-7 rule for `tiles_n == 11` and
`k == 5760` (the dispatch coordinates of `gpt_oss-GateUP dA H4 RCR`
where H4 transposes the original `b=[B, N=5760, K=2880]` and dispatches
as `m=2048, n=K=2880, k=N=5760`) currently picks:

```
tiles_m == 8  + m_total >= 65536:  (group_m=16, num_xcds=4)
tiles_m == 16:                     (group_m=8,  num_xcds=4)
tiles_m == 8  + default:           (group_m=4,  num_xcds=8)
```

The hypothesis: a single-block grouping `(group_m=1, num_xcds=4)`
might be better for the 4 GateUP H4 dA shapes given:
- the dispatch problem size on dA is `m=2048, n=2880, k=5760` —
  smaller `n` than fwd RCR (`m=2048, n=5760, k=2880`);
- with smaller n, the `group_m`-driven swizzle gain may diminish
  (fewer tiles per group → less reuse benefit);
- the K=5760 K-loop dominates wall time, so the M-N tile-launch
  pattern matters less than for the fwd path.

## Probe (corrected)

The first sweep used the wrong `b_orig` shape (`[B, K_fwd, N_fwd]`
instead of `[B, N_fwd, K_fwd]`), which produced a misleading
"(1, 4) wins on B=32" signal. After fixing `make_inputs` in
`/tmp/probe_r44_all_gateup_da.py` to mirror the actual fwd-RCR `b`
layout (so H4 transposes b correctly inside `grouped_gemm_impl`), the
sweep across all 4 GateUP H4 dA shapes — paired-randomized,
5 trials × 80 iters per cfg — gave:

| Shape | Current cfg | Current p20 (ms) | (1,4) p20 (ms) | Δ vs (1,4) |
|---|---|---|---|---|
| B4-M2048  (m_total=8192,   tiles_m=8,  default fall-through) | (4, 8) | 0.278 | 0.307 | **−10.24 %** |
| B4-M4096  (m_total=16384,  tiles_m=16, R7 rule)              | (8, 4) | 0.436 | 0.432 | +0.95 % |
| B32-M2048 (m_total=65536,  tiles_m=8,  R7 rule)              | (16,4) | 1.953 | 1.960 | −0.37 % |
| B32-M4096 (m_total=131072, tiles_m=16, R7 rule)              | (8, 4) | 3.518 | 3.512 | +0.16 % |

**Uniform-positive for (1, 4): False.** The −10.24% on B4-M2048 is
the killer — B=4 shapes are weight 3 in the metric (gpt_oss family).

The two "wins" (+0.95 %, +0.16 %) sit at the sweep noise floor
(spread% column ≈ 0.3-1.8 % per cfg per shape).

## v1 attempt (already discarded)

Even though uniform-sign failed, R44 v1 tried a restricted rule:
`m_total >= 65536` route picks (1, 4); `tiles_m == 16` else (8, 4);
B=4 M=2048 falls through to the default (4, 8) — preserving the loss-
direction shape's existing cfg.

```
config.py rule diff (DISCARDED):
        if tiles_n == 11 and k == 5760:
-            if tiles_m == 16:
-                return HipKittenConfig(layout=layout, group_m=8, num_xcds=4, kernel=None)
-            if tiles_m == 8 and m_total >= 65536:
-                return HipKittenConfig(layout=layout, group_m=16, num_xcds=4, kernel=None)
+            if m_total >= 65536:
+                # cluster (1,4) attempt
+                return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
+            if tiles_m == 16:
+                return HipKittenConfig(layout=layout, group_m=8, num_xcds=4, kernel=None)
```

Metric (single run on pinned MI355X GPU 3, no other workload):

```
baseline (00772059):           879
R44 v1 (cluster-(1,4)):        880   Δ = +1   (sub-noise)
post-revert recheck:           890          (run-to-run thermal noise)
```

The revert run scoring 890 (not 879) is the real story: the metric's
single-run noise floor on this GPU is ~±10 score points, dominated by
boost-clock variance. Comparing v1's 880 against baseline's 879 is
within noise; comparing baseline's 879 against revert's 890 is also
within noise (they share the same kernel + dispatch).

Per-shape ratio diffs across all 24 shapes between baseline (879) and
revert (890) span ±2.7-5.1 pp on `gpt_oss-Down-B4-M{2048,4096}` — but
both HK and Triton TFLOPS dropped on the revert run (e.g.
`gpt_oss-Down-B4-M2048` HK: 867 → 798 TF; Triton: 782 → 688 TF), so
the +5.1 pp ratio shift is "Triton lost more clock than HK", not "HK
got faster". Pure thermal/state variance.

## Mechanism — why dispatch tuning didn't move

The R7 rule already picks the local optimum on the 4 dA H4 RCR
shapes: every alternative cfg in the sweep is within ±2 % of the
current cfg's p20 (except (4, 8) on B4-M2048, where (4, 8) IS the
current default and dominates everything else by ≥9 %). With that
spread, no general predicate can cherry-pick a winner big enough to
clear the metric's ±10-point single-run noise.

Concretely:
- B4-M2048 default (4, 8) wins by 9.8-10.4 % over every other cfg.
  Cannot replace.
- B32-M2048 R7 (16, 4) is the literal winner of its sweep
  (1.953 ms, next best (12, 4) at 1.959 ms = +0.30 % loss).
  Replacement candidates are at noise.
- B4-M4096 + B32-M4096 (R7 (8, 4)) sit on a flat plateau —
  3 candidates within 0.95 % of each other on B4-M4096; 4 candidates
  within 0.55 % on B32-M4096. (1, 4) is nominal winner on both, by
  +0.95 % and +0.16 % respectively — both well below metric noise.

The Round-7 rule generation (autotuned over a wider sweep months
ago) has already nailed this surface to within ~1 % of optimal on all
4 shapes. Single-knob `(group_m, num_xcds)` tuning cannot extract more.

## Falsification consequence

R44 closes:
- `(group_m, num_xcds)` dispatch tuning for the **gpt_oss-GateUP dA H4
  RCR** route (4 shapes). Round-7 rule is at local optimum across the
  feasible cfg lattice (verified with 7-cfg × 5-trial × 80-iter sweep
  per shape).
- The R43 R44-pivot suggestion is now downgraded: dispatch tuning at
  `(group_m, num_xcds)` granularity is exhausted for this shape family.

R44 does NOT close:
- `gpt_oss-Down dA H4 RCR` family — different `tiles_n` route
  (`n=2880, n_per_block=256` → `tiles_n = 11` shared with GateUP, but
  different K-direction layout). The R32-R34 sweeps were on GateUP only.
- `gpt_oss-{GateUP,Down} dB CRR` family — var-K kernel, different
  dispatch route in `select_default_config` (`crr` branch, not `rcr`).
  R43 closed one micro-opt inside the var-K kernel but the cfg surface
  was untouched.
- `select_default_config` parameter dimensions OTHER than
  `(group_m, num_xcds)` — e.g., `kernel=` slot (specialised
  templated launch) is unset on all gpt_oss routes and could be a
  separate lever (would require a HK kernel-template addition).

## Action

- HipKittens kernel: no change.
- Primus-Turbo `config.py`: no diff after revert (verified clean tree).
- Primus-Turbo: 1 commit (this falsification note).

## R45 next-action surface

Three candidate vectors, ordered by expected leverage:

1. **Dispatch surface for `gpt_oss-Down dB CRR`** (var-K kernel route).
   The `crr` branch in `select_default_config` is much shorter than
   `rcr` and may have un-tuned cfgs for the K=2048 / k_per_block=128
   k-major layout. R32-R34 never touched it. Probe: same
   monkey-patch sweep approach as R44, 7-cfg × 4-shape, B={4,32} ×
   M={2048,4096} on `gpt_oss-Down`. If any cfg uniformly beats the
   current by ≥3 % across all 4 shapes (the metric noise threshold),
   it's a real lever.
2. **Var-K kernel structural opts beyond R43's epilog drop**.
   R43 listed 4 candidates: persistent loop overhead amortisation,
   conditional-barrier semantics audit, store batching (4 → 1 store),
   var-K KI specialisation. Item 1 (per-group cache for `k_offset_tiles`,
   etc.) is the cheapest scout — read 30 LOC of `grouped_var_k_kernel`,
   check if the per-tile arithmetic depends only on `gt` or also on
   loop-invariant per-group quantities. If the latter, hoisting
   could save SGPR ops per tile.
3. **HK fwd `grouped_kernel` audit** for the gpt_oss-K=2880 K-tail
   path. R39-R42 closed KI specialisation, but the fwd-tail branch
   structure (after the K-main loop ends) was not directly examined.
   Lower priority than (1) and (2) — fwd ratio on `gpt_oss-GateUP-
   B32-M2048` is already 1.124 (R41 split), only bwd is below 1.0.

Recommended: start R45 with (1). dB CRR is the largest unexplored
dispatch surface and is the var-K kernel side of `gpt_oss`'s problem.
