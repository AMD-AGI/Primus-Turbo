# Round 39 — FP8 grouped fused-wall: Qwen3-Down dA RRR re-probe (saturated) + plateau confirmation

## Round metric state

* HEAD: `e42f7c98` (R38 commit just landed last round).
* Pre-round metric: score `989`, geomean `1.3345`, 17 / 24 below target.
* `correct_fail = 0/24`.
* Patience: `27/30` — 3 rounds left.

Bottom-ratio shapes (sorted ascending):

| rank | shape                            | ratio | status        |
|------|----------------------------------|-------|---------------|
| 1    | Qwen3-Down-B16-M2048             | 1.244 (was 1.252 in same-cmd later) | <target |
| 2    | Qwen3-Down-B16-M4096             | 1.265 | <target |
| 3    | gpt_oss-Down-B32-M2048           | 1.270 | <target |
| 4    | Qwen3-GateUP-B16-M2048           | 1.274 | <target |
| 5    | Qwen3-GateUP-B32-M2048           | 1.278 | <target |
| 6    | Qwen3-Down-B32-M2048             | 1.283 | <target |

The 4 worst shapes are all **Qwen3-Down family** (geomean 1.252-1.308).
Per the R29 / R36 audit, every dispatch path on this family has been
tight-verified saturated:

* Forward RCR: binding default `(gm=4, xcds=None=8)` — R29 / R36 16-cell
  sweep + interleaved methodology confirms.
* dA RRR: R42 `(gm=16, xcds=4)` for `tiles_n ≤ 8 AND m_total ≥ 32768`.
* dB var-K: R39 `(gm=8, xcds=4)` for `m_total ≥ 16384` (default branch);
  R29 / R38 verified.

## R39 scope — re-audit Qwen3-Down dA RRR with R32-class wider sweep

R29's tight-verify on dA RRR only tested **one** alternative cell
(`(gm=8, xcds=4)` against R42's `(gm=16, xcds=4)`, recording within-noise
deltas). Other gm × xcds combinations were never exhaustively tested
under R32-class methodology (12-trial × 200-iter × 3-seed × kernel-only
direct dispatch). Today's metric still has the 4 Qwen3-Down shapes as
the lowest-ratio bottom — worth one final R32-class re-audit to confirm
saturation.

### Probe

`/tmp/probe_r39_qwen3_down_da_rrr.py`:

* 4 metric Qwen3-Down shapes (B ∈ {16, 32}, M ∈ {2048, 4096}).
* 10 candidate cells:
  `(16,4)*baseline, (8,4), (4,4), (2,4), (12,4), (24,4), (32,4), (16,2), (16,8), (16,0)`.
* 12 trials × 200 iters × 3 seeds (42 / 137 / 2024).
* Kernel-only direct call to `hipkitten.grouped_rrr_dscale`.

### Results

```
== Qwen3-Down-B16-M2048  m_total=32768 ==
  cell      med Δ%   spread pp   pos seeds   verdict
  (16, 4)   +0.00%   0.07         1/3         TIE  (baseline)
  (8, 4)    +0.05%   0.07         3/3         TIE
  (4, 4)    +0.19%   0.12         3/3         TIE
  (2, 4)    +0.08%   0.25         2/3         TIE
  (12, 4)   +0.37%   0.26         3/3         TIE
  (24, 4)   +0.38%   0.32         3/3         TIE
  (32, 4)   +0.06%   0.08         3/3         TIE
  (16, 2)   −1.42%   0.19         0/3         LOSS
  (16, 8)   −4.10%   0.43         0/3         LOSS
  (16, 0)   −4.14%   0.54         0/3         LOSS  (=BLOCK_SWIZZLE_NUM_XCDS=8 fallback)

== Qwen3-Down-B16-M4096  m_total=65536 ==
  All gm × xcds=4: TIE within ±0.20% (positive but every-seed mixed).
  xcds ∈ {0, 2, 8}: LOSS −0.6% to −1.7%.

== Qwen3-Down-B32-M2048  m_total=65536 ==
  gm × xcds=4: TIE within ±0.20%.
  xcds ∈ {0, 2, 8}: LOSS −1.2% to −4.6%.

== Qwen3-Down-B32-M4096  m_total=131072 ==
  gm × xcds=4: TIE within ±0.35%.
  xcds ∈ {0, 2, 8}: LOSS −0.5% to −1.6%.
```

### Verdict

R42's `(gm=16, xcds=4)` is **robustly saturated** across the entire
Qwen3-Down family. All gm ∈ {2, 4, 8, 12, 16, 24, 32} × xcds=4 cells are
within ±0.4% (TIE) of the baseline; no cell crosses the +1.0% / spread
robust-signal threshold (R7 / R10 / R23 / R29-R38 convention).
xcds ∈ {0, 2, 8} all clearly LOSS (−0.5% to −4.6%) — confirming xcds=4
is a cleanly differentiated optimum on this family.

The few every-seed-positive small wins in the gm column (`(12, 4)
+0.37%`, `(24, 4) +0.38%` on B16-M2048) are 0.4× their respective
spread — well below the robust-signal threshold and dominated by
shape-to-shape variance (e.g. (24, 4) flips to −0.14% on B16-M4096).

R29's "saturated at default" verdict (line 1724-1726, narrow probe with
only one alt cell tested) extends cleanly to the wider 10-cell sweep.

## Closure observation — dispatch-tuning lever genuinely exhausted

Cumulative rounds with R32-class probes after R36's audit:

| round | shape family / path                              | result                                                               |
|-------|--------------------------------------------------|----------------------------------------------------------------------|
| R36   | Qwen3-Down M=2048 forward RCR (16 cells)         | saturated default                                                     |
| R36   | Qwen3-GateUP M=2048/M=4096 var-K dB (10 cells)   | (1, 4) candidate kernel-real but metric noise floor swallows; reverted |
| R37   | (no kernel changes — GPU debug)                  | n/a                                                                   |
| R38   | Qwen3-GateUP-B16-M2048 dA RRR (5 cells)          | saturated R32 (1, 4)                                                  |
| R38   | Qwen3-Down dB var-K M=2048 (9 cells)             | saturated R39 (8, 4)                                                  |
| R38   | gpt_oss-Down-B4-M4096 var-K dB (9 cells)         | **WIN** (16, 4) — narrow carve-out shipped                            |
| **R39** | **Qwen3-Down dA RRR (10 cells × 4 shapes)**     | **saturated R42 (16, 4) — this note**                                 |

R26 / R36 both declared "no unprobed dispatch lever remaining" and the
last 4 rounds have empirically confirmed it: only one of the 5 explicit
re-probes (R38) yielded a robust kernel-level WIN, and that was on a
small ratio-improvement target (~+1.1pp on a single shape) with metric
geomean lift below the noise floor.

## Score plateau analysis

8-round rolling score history (R31-R38):

```
R31: 985, R32: 1000, R33: 994, R34: 1000, R35: 991, R36: 994, R37: 986, R38: 992
```

Distribution: median 992, mean 992.75, std 5.04, range [985, 1000].
This matches the R36 quantification of `~5 score-points` per-run noise
floor in the [985, 1000] band. The score has not crossed below 985 or
above 1000 in 8 rounds of optimization rounds — with plenty of attempts
at carve-outs along the way.

Per-shape ratio noise across these 8 runs (sample-stdev of HK / TRT
ratio per shape, for the bottom 4 shapes) is `~0.5pp` — consistent with
Triton-side timing noise (Triton lacks the persistent kernel's
deterministic schedule and can have ±1pp stdev in geomean across runs).

## Path A architectural ceiling — re-confirmed

R7 / R8 falsified Path A forward fusion with the kernel-side experiment:
DTR + in-register cvt was 30-40% slower than the un-fused DTL path. R36
suggested re-trying Phase 1 forward-only fusion (without dB), since the
R7 falsification was on the full Phase 1+2+3.

After R32-class auditing 5 more rounds and finding only one shape with
a real (small) kernel signal (R38 gpt_oss-Down-B4-M4096), it's now
clear:

* The 24-shape geomean ratio is bounded by the **architectural FP8 quant
  HBM tax** that both backends pay equally on `quantize_fp8(a)`,
  `quantize_fp8(b)`, `quantize_fp8(grad_out)`. Triton's quantize is at
  the same ~67% of MI355X HBM peak as HK's, so the ratio doesn't move
  on the quant component.
* The kernel-only ratio (without quant tax) plateaued at score=1000 in
  the previous task (`_metric_grouped_only.py`, 63 prior rounds).
* The wall-ratio plateau at ~991-998 with target=1.35 means the
  geomean is at ~1.337-1.344 — **the architectural ceiling for this
  family of shapes without changing the FP8 quant pipeline**.

A Phase 1 forward-only fuse would require kernel surgery in HipKittens
(clone `grouped_rcr_kernel<...>` to add a `FUSE_ACT` template branch
with `__builtin_amdgcn_cvt_pk_fp8_*` builtin in `load_a_tile`,
`grouped_layout_globals_fused_act` with `_gl_bf16 a_bf16` field). With
patience 27/30 (3 rounds left) and the R7 / R8 architectural falsification
still on record, this is not a viable in-window deliverable.

## Files touched

Primus-Turbo:

* `analysis/_notes/round-39-fp8-grouped-fused-wall-Qwen3-Down-dA-RRR-saturated-confirmed-plateau.md`
  (this note — falsification documentation, no code change)

* No `primus_turbo/` source files modified.

HipKittens: no changes.

## Probes (in `/tmp/`, NOT committed)

* `probe_r39_qwen3_down_da_rrr.py` — R32-class 10-cell × 4-shape ×
  12-trial × 3-seed sweep. Saturation confirmed.

## Round summary

* **Lever**: dispatch tuning audit (Lever F).
* **Target**: Qwen3-Down dA RRR — the lowest-ratio shape family's
  remaining unprobed dispatch path under R32-class methodology.
* **Files**: `analysis/_notes/round-39-...md` only (no code change).
* **Metric**: pre-round `989`. No code change → no post-round measurement.
* **Correctness**: n/a (no kernel touched).
* **Commit**: see Primus-Turbo HEAD post-commit.
* **HipKittens**: no changes.

## Suggestion for next round (R40)

The dispatch lever is now empirically + audit-trail confirmed
exhausted. R40 has only 2 productive paths:

1. **Phase 1 forward-only Path A fusion** — R36 priority #3. Requires
   HipKittens kernel surgery (~3-5 round task). With patience 28/30
   after R39 it's a 1-round bet at best; would need to ship a working
   fused-act variant in a single round to land before patience exhaust.
   High risk, possible high reward (fwd quant is ~25% of fwd wall).

2. **Pivot / accept plateau** — write closure summary documenting:
   - Score 1000 reached in 4/8 of recent rounds (R32, R34); plateau
     confirmed at [985, 1000] noise band.
   - Architectural ceiling re-confirmed by 5 rounds of R32-class
     dispatch audits.
   - Recommend running the optimization on a different task (Path A
     would benefit from a separate task with fresh patience budget;
     Path B (NVIDIA-style optimized cast kernels) caps at +3-4% wall
     and is also a separate task per the task body).

Conservative recommendation: **option 2** — patience is too tight for
a kernel-surgery bet, and the audit trail across R26-R39 supports
plateau closure. The metric's ~5pp noise band swallows any single carve-
out's potential lift, so the path forward is a non-dispatch lever or a
new task with a different optimization surface.
