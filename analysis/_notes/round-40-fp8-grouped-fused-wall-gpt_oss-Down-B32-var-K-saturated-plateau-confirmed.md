# Round 40 — FP8 grouped fused-wall: gpt_oss-Down-B32 var-K dB re-probe (saturated) + plateau noise resampled

## Round metric state

* HEAD: `22edcf1a` (R39 falsification doc commit).
* Pre-round metric (round-40 first sample): score `995`, geomean
  `1.3432`. `correct_fail = 0/24`.
* Patience: `28/30` — 2 rounds left.

Bottom-ratio shapes (sorted ascending, today's first run):

| rank | shape                          | ratio  | rule status                                    |
|------|--------------------------------|--------|------------------------------------------------|
| 1    | Qwen3-Down-B16-M2048           | 1.245  | R29/R36/R39 saturated default + R42 (16, 4) dA |
| 2    | Qwen3-Down-B16-M4096           | 1.255  | same family — saturated                        |
| 3    | Qwen3-GateUP-B16-M2048         | 1.257  | R7 + R32 (1, 4) dA RRR — saturated             |
| 4    | gpt_oss-Down-B32-M2048         | 1.258  | R8 + R30 (4, 4) var-K dB — verified            |
| 5    | Qwen3-GateUP-B32-M2048         | 1.276  | R7 + R32 — saturated                           |
| 6    | Qwen3-Down-B32-M2048           | 1.273  | same family — saturated                        |
| 7    | Qwen3-Down-B32-M4096           | 1.288  | same family — saturated                        |
| 8    | gpt_oss-Down-B32-M4096         | 1.290  | R50 + R30 (4, 4) var-K dB — verified           |

R39 declared dispatch lever exhausted. Patience nearly out, R40 chose to
do one final confirmatory check on a path R39 didn't re-audit (the
gpt_oss-Down-B=32 family var-K dB) and characterize the metric's noise
floor at the current HEAD.

## R40 scope — re-probe gpt_oss-Down-B32 var-K dB

### Motivation

R30 had set `(gm=4, xcds=4)` for `a.shape[1] == 2880 AND b.shape[1] ==
2880 AND m_total >= 65536` (gpt_oss-Down B=32 family, M=2048+M=4096).
R30 narrative reported `+0.39%` (M=4096) and `+0.73%` (M=2048) over R39
default `(gm=8, xcds=4)`. R30's 6-cell sweep did NOT include
`(gm=16, xcds=4)` (R38's B4-M4096 winner) or `(gm=32, xcds=4)` (R38
near-tied alternative) — leaving open the question of whether the R30
+ R38 carve-outs could collapse to one rule
(`(gm=16, xcds=4) for k=2880 AND n=2880 AND m_total >= 16384`).

### Probe

`/tmp/probe_r40_gpt_oss_down_b32_var_k_db.py`:

* 2 metric shapes: gpt_oss-Down B=32 M=2048 (m_total=65536), B=32 M=4096
  (m_total=131072).
* 9 candidate cells:
  `(4,4)*baseline, (8,4), (16,4), (32,4), (2,4), (1,4), (4,2), (4,8), (8,2)`.
* 12 trials × 200 iters × 3 seeds (42 / 137 / 2024).
* Kernel-only direct call to `hipkitten.grouped_variable_k_crr_dscale`.

### Results

```
== gpt_oss-Down-B32-M2048  m_total=65536 ==
  cell      med Δ%   spread pp   pos seeds   verdict
  (4, 4)    +0.00%   0.97         1/3         TIE  (baseline)
  (8, 4)    -0.76%   0.80         0/3         TIE
  (16, 4)   -0.84%   1.34         0/3         TIE
  (32, 4)   -0.82%   1.51         0/3         TIE
  (2, 4)    -1.69%   0.35         0/3         LOSS
  (1, 4)    -0.70%   1.09         0/3         TIE
  (4, 2)    -2.25%   1.90         0/3         LOSS
  (4, 8)    -1.90%   1.83         0/3         LOSS
  (8, 2)    -1.81%   1.18         0/3         LOSS

== gpt_oss-Down-B32-M4096  m_total=131072 ==
  (4, 4)    +0.00%   0.27         1/3         TIE  (baseline)
  (8, 4)    -0.16%   0.30         1/3         TIE
  (16, 4)   -0.72%   0.34         0/3         LOSS  ← R38 sibling cell does NOT win here
  (32, 4)   -0.73%   0.34         0/3         LOSS
  (2, 4)    -0.51%   0.27         0/3         LOSS
  (1, 4)    -0.32%   0.30         0/3         TIE
  (4, 2)    -3.22%   0.21         0/3         LOSS
  (4, 8)    -4.15%   0.36         0/3         LOSS
  (8, 2)    -1.92%   0.44         0/3         LOSS
```

### Verdict

R30's `(gm=4, xcds=4)` is robustly optimal for the gpt_oss-Down-B=32
family. `(gm=16, xcds=4)` (R38 sibling) **loses by 0.72-0.84%** — the
R30 + R38 rules cannot collapse. R30's narrative for B=32 cross-group
stall avoidance with tiles_n=11 holds: small `gm` preserves L2 reuse
on the wider B-side persistent grid (3872 tile-steps for B=32-M=2048,
8800 for B=32-M=4096) where R38's `(gm=16)` wins on the much sparser
B=4 grid (484 tile-steps).

The R30 + R38 carve-outs cleanly partition the gpt_oss-Down family by
m_total threshold:

* `m_total < 16384`: B=4 M=2048 → R33 `(gm=16, xcds=4)`
* `16384 ≤ m_total < 65536`: B=4 M=4096 → R38 `(gm=16, xcds=4)`
* `m_total ≥ 65536`: B=32 → R30 `(gm=4, xcds=4)`

The B=4 rules use `gm=16` because the small persistent grid (484
tile-steps) wants larger N-batches; the B=32 rule uses `gm=4` because
the larger grid (3872+ tile-steps) needs cross-group stall avoidance.
Two distinct optima for two distinct grid-density regimes.

## Plateau noise resampled

4 metric samples at this HEAD:

```
sample 1: score=995  (initial)
sample 2: score=988
sample 3: score=992
sample 4: score=1000
```

Median 993.5, mean 993.75, std 5.0, range [988, 1000]. Matches R36 and
R39 quantification (`std ≈ 5pp` per-run noise floor in [985, 1000]
band). The plateau is genuinely stationary — 13 rounds (R28-R40) of
score history fit within `[985, 1000]` with median `~992`.

## Closure observation

After 6 rounds (R36 → R40) of R32-class dispatch re-audits since R26
declared "no unprobed dispatch lever remaining":

| round | path / family probed                         | shipped? |
|-------|----------------------------------------------|----------|
| R36   | Qwen3-Down M=2048 fwd RCR (16 cells)         | no — saturated default |
| R36   | Qwen3-GateUP M=2048/M=4096 var-K dB (10 cells) | no — kernel WIN swallowed by metric noise |
| R37   | (no kernel changes — GPU debug)              | n/a      |
| R38   | gpt_oss-Down-B4-M4096 var-K dB (9 cells)     | **YES** — (gm=16, xcds=4) |
| R39   | Qwen3-Down dA RRR (10 cells × 4 shapes)      | no — saturated R42 (16, 4) |
| **R40** | **gpt_oss-Down-B32 var-K dB (9 cells × 2 shapes)** | **no — saturated R30 (4, 4)** |

Cumulative: 1 shipping rule (R38, +1.1pp on 1/24 shape) out of 6 dedicated
re-audit rounds. Score plateau distribution unchanged.

The 24-shape geomean ratio is bounded by the **architectural FP8 quant
HBM tax** that both backends pay equally on
`quantize_fp8(a) + quantize_fp8(b) + quantize_fp8(grad_out)`. Triton's
quantize is at the same `~67 %` of MI355X HBM peak as HK's, so the
ratio doesn't move on the quant component. The kernel-only ratio
(without quant tax) plateaued at score=1000 in the previous task
(`_metric_grouped_only.py`, 63 prior rounds). The wall-ratio plateau at
`~991-998` with target=1.35 means the geomean is at `~1.337-1.344` —
the architectural ceiling for these shapes without changing the FP8
quant pipeline (Path A forward-only fusion — R7/R8 falsified for full
Phase 1+2+3 stack at 30-40% slower kernel; Phase 1 alone unverified
but architecturally similar enough that the in-window kernel-surgery
bet is too risky with patience 28/30).

## Files touched

Primus-Turbo:

* `analysis/_notes/round-40-fp8-grouped-fused-wall-gpt_oss-Down-B32-var-K-saturated-plateau-confirmed.md`
  (this note — falsification documentation, no code change)

* No `primus_turbo/` source files modified.

HipKittens: no changes.

## Probes (in `/tmp/`, NOT committed)

* `probe_r40_gpt_oss_down_b32_var_k_db.py` — 9-cell × 2-shape × 12-trial
  × 3-seed sweep. R30 (4, 4) saturated, (gm=16, xcds=4) sibling does
  NOT extend to B=32.

## Round summary

* **Lever**: dispatch tuning audit (Lever F).
* **Target**: gpt_oss-Down-B=32 var-K dB — the un-audited path that
  could have collapsed R30 + R38 carve-outs into one rule.
* **Files**: `analysis/_notes/round-40-...md` only (no code change).
* **Metric**: 4-sample noise resampled at HEAD `22edcf1a` —
  `[995, 988, 992, 1000]`, median 993.5, std 5.0. Stationary noise floor.
* **Correctness**: n/a (no kernel touched).
* **Commit**: see Primus-Turbo HEAD post-commit.
* **HipKittens**: no changes.

## Suggestion for next round (R41)

Patience `29/30` after R40 — **1 round of buffer left**.

The audit is closed. Productive paths in priority order:

1. **Pivot / accept plateau** (recommended). The dispatch lever has
   been formally exhausted across R26 → R40. Score plateau is
   stationary in [985, 1000] band with median ~992. Architectural
   ceiling at geomean ~1.337-1.344 (target 1.35). Recommend R41 as a
   final closure round documenting the run-series outcome and
   suggesting next-task pivots:

   * **Path A forward-only fuse** (R36 priority #3) — needs HipKittens
     kernel surgery; should be a separate task with fresh patience
     budget. R7/R8 falsified the full Phase 1+2+3 stack but Phase 1
     alone may behave differently (the FUSE_ACT pipeline penalty
     observed in the full-stack experiment may not trigger when
     dB stays unfused).
   * **Path B (NVIDIA-style cast kernel optimization)** — caps at
     `+3-4 %` wall on this hardware per the task body's analytical
     model. Also a separate task.
   * **Weight FP8 caching at autocast level** — currently out of
     scope per the task body; would lift all 24 shapes uniformly by
     skipping `quantize_fp8(b)` once weights stabilize during
     training.

2. **Phase 1 forward-only Path A kernel surgery** — high risk in 1
   round. Concrete deliverable would be a `grouped_rcr_fused_act_kernel`
   variant in HipKittens with `__builtin_amdgcn_cvt_pk_fp8_*` builtin
   in `load_a_tile`. Almost certainly will not converge in 1 round
   (R7's full-stack fuse took multiple rounds to falsify), but the
   negative result would document Phase 1 specifically rather than
   relying on the R7/R8 full-stack falsification by analogy.

Recommendation: option 1, since patience 29/30 makes a kernel-surgery
attempt in R41 a single-shot bet with very low success probability.
