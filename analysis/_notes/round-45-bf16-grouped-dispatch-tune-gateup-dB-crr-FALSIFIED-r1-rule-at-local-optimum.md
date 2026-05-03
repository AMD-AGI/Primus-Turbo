# Round 45 — BF16 grouped, dispatch tune `(group_m, num_xcds)` for gpt_oss-GateUP dB CRR — FALSIFIED

## Goal coming in

R44 falsification (R7 dispatch rule for gpt_oss-GateUP dA H4 RCR is at
local optimum) recommended R45 attack the parallel surface on the **dB
CRR var-K** route:

> R45 surface ranked: (1) gpt_oss-Down dB CRR var-K dispatch sweep;
> (2) var-K kernel structural opts beyond R43 epilog; (3) fwd K-tail
> branch audit.

R45 picked (1) but pivoted within it: the gpt_oss-**Down** dB var-K
route is already covered by the R24 rule
(`tiles_m=11, tiles_n=11, k<=4096 → (gm=1, xcds=4)` at line 720) and
was uniformity-validated when added. The **un-tuned** sub-family is
gpt_oss-**GateUP** dB var-K (`tiles_m=22, tiles_n=11, k<=4096`),
which falls through R24's Down carve-out into the R1-vintage family
rule at line 1203 returning `(gm=4, xcds=4)`.

The R1 rule was a single-cfg pick across BOTH Down and GateUP at a
time when the gpt_oss family geomean was 0.870 (today: 1.085 — kernel
~25 % faster). Per-cfg sensitivity often diminishes as a kernel
matures; today's GateUP-B32-M2048 dB ratio sits at 1.049 (R1's was
0.753), so the (4, 4) pick deserved a re-sweep on the modern kernel.

## Hypothesis (R45)

The R1 single-cfg `(gm=4, xcds=4)` for GateUP dB var-K (carved out
from the family rule via a `tiles_m == 22` guard before it falls
through) might be sub-optimal on the modern kernel. Re-sweep the
`(group_m, num_xcds)` lattice with `xcds=4` (R44 confirmed `xcds=8`
is consistently dominated by `xcds=4` on this geometry) and pick a
cell that uniformly beats `(4, 4)` across all 4 GateUP shapes.

## Probe

Mirror the metric's exact call pattern: full
`turbo.ops.grouped_gemm(a, b, group_lens, trans_b=True)` fwd + bwd
under `force_grouped_gemm_backend(HIPKITTEN, BF16)`, monkey-patched
`select_default_config` overriding only when
`layout=='crr' and tiles_m==22 and tiles_n==11`.

DSV3-GateUP-B16-M2048 fwd+bwd warmup first (mirrors metric's
DSV3-first iteration order — works around the HK BF16 K-tail
cold-start sync-fault bug; without it raw gpt_oss-first probes crash
with a memory access fault).

Candidate set: `(gm, xcds) ∈ {(1,4), (2,4), (4,4), (8,4), (16,4)}`
(5 cells × 4 trials × HK metric `_time_op` per trial; randomized
order per trial). Probe at `/tmp/probe_r45_gateup_db_crr_sweep.py`.

### Sweep results (3 of 4 shapes — see footnote on B=32 M=4096)

| Shape | (4,4) baseline TF | top-1 cfg | top-1 Δ | (1,4) Δ | (8,4) Δ | (16,4) Δ |
|---|---:|---|---:|---:|---:|---:|
| B=4-M2048   k_var=2048 | 1014.1 | (8, 4) | +0.39 % | −0.10 % | +0.39 % | +0.31 % |
| B=4-M4096   k_var=4096 | 1160.2 | (8, 4) | +0.43 % | +0.17 % | +0.43 % | +0.29 % |
| B=32-M2048  k_var=2048 | 1153.5 | (1, 4) | +0.34 % | +0.34 % | +0.01 % | −0.21 % |
| B=32-M4096  k_var=4096 | (probe crashed; metric run is the verify) |

`(8, 4)` is uniform-positive on the 3 probed shapes (avg +0.28 %,
min +0.01 %, max +0.43 %). `(1, 4)` is non-uniform (loses on
B=4-M2048 by −0.10 %). `(16, 4)` is non-uniform (loses on B=32-M2048
by −0.21 %).

Footnote on B=32-M4096: the probe consistently crashes with
"Memory access fault by GPU" on this single shape regardless of
isolation / ordering; unrelated to the cfg under test (occurs on
the first iteration of the per-shape benchmark loop, before any
non-(4,4) cfg is exercised). The metric — which runs the canonical
suite order — runs this shape successfully (current ratio 1.089),
so the metric run substitutes for the probe verify on this
sub-shape.

## v1 attempt

Carved out a tiles_m==22 specific rule before the R1 family rule:

```python
if (layout == "crr"
        and tiles_m == 22
        and tiles_n == 11
        and k <= 4096
        and m_total is not None):
    return HipKittenConfig(layout=layout, group_m=8, num_xcds=4, kernel=None)
if layout == "crr" and tiles_n == 11 and 8 <= tiles_m <= 24 and k <= 4096:
    return HipKittenConfig(layout=layout, group_m=4, num_xcds=4, kernel=None)
```

The carve-out preserves Down's R24 rule (which fires earlier at line
720) and the R1 family rule (which still covers
`tiles_m ∈ {8, 9, 10, 12..21, 23, 24}` for any DoD smoke shapes).

### Dispatch verify

```
gpt_oss-GateUP-B32-M2048 dB CRR (m=5760, n=2880, k=2048) -> gm=8, xcds=4   ✓ rule fires
gpt_oss-Down-B32-M2048   dB CRR (m=2880, n=2880, k=2048) -> gm=1, xcds=4   (R24 unchanged)
gpt_oss-GateUP-B32-M4096 dB CRR (m=5760, n=2880, k=4096) -> gm=8, xcds=4   ✓ rule fires
gpt_oss-Down-B32-M4096   dB CRR (m=2880, n=2880, k=4096) -> gm=1, xcds=4   (R24 unchanged)
```

### Metric

```
baseline (1d2d3bb):       879
R45 v1 (gm=8 carve):      880   Δ = +1   (sub-noise)
R45 v1 (rerun):           878   Δ = -1
```

Per-family geomean drift between baseline and R45 v1 run-1 vs run-2:

```
                       baseline   v1 run-1   v1 run-2
gpt_oss_20B            1.0849     1.0888     1.0860
DeepSeek-V3            1.1230     1.1233     1.1195
Qwen3-235B-A22B        1.1122     1.1124     1.1113
```

Per-shape gpt_oss diffs (baseline → v1 run-1):

```
GateUP-B4-M2048    1.107  →  1.105   −0.2 pp   (rule applies, mild noise)
GateUP-B4-M4096    1.095  →  1.116   +2.1 pp   (rule applies, noise-magnitude positive)
GateUP-B32-M2048   1.049  →  1.048   −0.1 pp   (rule applies, flat)
GateUP-B32-M4096   1.089  →  1.085   −0.4 pp   (rule applies, mild noise)
Down-B4-M2048      1.111  →  1.104   −0.7 pp   (rule does NOT apply, baseline noise)
Down-B4-M4096      1.103  →  1.110   +0.7 pp   (rule does NOT apply, baseline noise)
Down-B32-M2048     1.051  →  1.062   +1.1 pp   (rule does NOT apply, baseline noise)
Down-B32-M4096     1.077  →  1.083   +0.6 pp   (rule does NOT apply, baseline noise)
```

Down shapes (rule unchanged) move ±0.7-1.1 pp purely from run-to-run
variance — that IS the noise floor for any single-shape rule
attribution. GateUP shapes (rule applied) move within the same
band (−0.4 to +2.1 pp) with no coherent direction. The +2.1 pp on
GateUP-B4-M4096 is the largest individual move but is dwarfed by
the +1.1 pp Down-B32-M2048 swing under no rule change.

The v1-run-2 metric (878) confirms the +1 from run-1 was noise:
gpt_oss geomean drifted back to 1.0860 (within 0.001 of baseline
1.0849) and DSV3 + Qwen3 both shifted ~−0.4 pp (also pure noise).

## Mechanism — why dispatch tuning didn't move (again)

Same as R44, but on the dB CRR side:

The R1 (4, 4) rule is at local optimum on the modern kernel for the
GateUP dB var-K route. Sweep top-1 cells beat (4, 4) by **0.01 to
0.43 %** on the 3 probed shapes — well below the metric's ±10-point
single-run noise floor. Even applied uniformly across the 4 GateUP
shapes (which carry 4 × 3 = 12 / 40 = 30 % of the metric weight),
the expected score lift is:

```
+0.28 % avg dB-kernel speedup
× 25 % dB share of wall (per R41 split)
× 30 % GateUP weight share
≈ +0.02 % weighted progress
≈ +0.2 score
```

— literally below the noise floor of every other cause of
run-to-run variance.

## Falsification consequence

R45 closes:

* `(group_m, num_xcds)` dispatch tuning for the **gpt_oss-GateUP dB
  CRR var-K** route (4 shapes). The R1-vintage `(4, 4)` rule is at
  local optimum across the 5-cell × `xcds=4` lattice (3 shapes
  probe-verified; B=32-M4096 metric-verified).

R45 does NOT close:

* `gpt_oss-{GateUP, Down} dA` and `dB` kernel structural opts. R44
  closed the dA H4 RCR dispatch surface; R45 closes the dB CRR
  dispatch surface for GateUP (Down was R24). The kernel itself
  (var-K + grouped fwd + dA H4 RCR) is the only remaining lever
  for gpt_oss family progress.
* `select_default_config` `kernel=` slot — unset on all gpt_oss
  routes. A specialised templated launch (FUSED_KTAIL, alternate
  swizzle, etc.) would require a HK kernel-template addition, which
  is the only un-explored part of the dispatch surface.
* DSV3 / Qwen3 dispatch tuning. R24 covered Qwen3-Down, Qwen3-GateUP,
  DSV3-Down dB var-K. DSV3-GateUP dB var-K was tested at R24 but
  dropped due to xcds=0 allclose drift — could be re-attempted with
  a different (gm, xcds) cell that doesn't use xcds=0.

## Action

* HipKittens kernel: no change.
* Primus-Turbo `config.py`: no diff after revert (verified clean
  tree).
* Primus-Turbo: 1 commit (this falsification note).

## R46 next-action surface

Three candidate vectors, ordered by expected leverage:

1. **HK var-K kernel structural opts beyond R43 epilog drop**.
   R43's note listed 4 candidates: persistent loop overhead
   amortisation (per-tile arithmetic hoisting), conditional-barrier
   semantics audit (`if (warp_row == 0) s_barrier()` is suspicious),
   store batching (4 → 1 store), var-K KI specialisation. Item 1
   (per-group cache for `k_offset_tiles` and friends) is the cheapest
   scout. With dispatch tuning closed for both dA H4 RCR (R44) and
   dB CRR GateUP (R45), the kernel itself is the only remaining
   lever inside the gpt_oss family.
2. **DSV3-GateUP dB var-K dispatch retry**. R24 dropped this family
   due to `xcds=0` allclose drift on a specific candidate. Re-sweep
   with `xcds ∈ {1, 2, 4, 8}` (no xcds=0) on the (tiles_m=16,
   tiles_n=28) geometry. Smaller upside than (1) since DSV3-GateUP
   ratios are already 1.13-1.15 (well above gpt_oss), but cleaner
   dispatch surface.
3. **Fused K-tail kernel (HK template addition)**. The original
   task body's "Lever A1": in-kernel masked K-tail loop
   (`FUSED_KTAIL` template; load-side mask on the last K iteration)
   to eliminate the 2-launch K-main + K-tail synchronization. This
   is the highest-upside lever (Triton's K=2880 win comes from this)
   but the highest-cost (HK kernel template addition + extensive
   correctness validation). Defer to a multi-round attack window.

Recommended: start R46 with (1). The var-K kernel persistent loop
overhead has had only the R43 vmcnt drop investigation (one
micro-opt, falsified). The other 3 R43-listed candidates remain
unexplored.
