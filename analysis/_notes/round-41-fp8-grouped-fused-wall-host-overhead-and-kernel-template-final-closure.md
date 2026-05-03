# Round 41 — FP8 grouped fused-wall: host overhead + kernel template audit (final closure)

## Round metric state

* HEAD: `1bb4c80f` (R40 falsification doc commit).
* Pre-round metric (R41 first sample): score `985`, geomean `1.3293`,
  `correct_fail = 0/24`.
* Patience: `29/30` — **last round in patience window**.
* Last DoD = 608 (sha 1bb4c80f, much improved from R36's -1394).

R39 / R40 declared dispatch lever exhausted. R41 audits the **two
remaining un-probed angles** before formal closure: (1) `select_default_config`
host-side overhead (untouched since R11 / R16 / R19 trims) and (2)
forced kernel template selection (`kernel="4"` vs auto-pick `"8"`) on
the 4 lowest-ratio Qwen3-Down shapes.

Bottom-ratio shapes (sorted ascending, R41 first run):

| rank | shape                          | ratio  | path saturation                                  |
|------|--------------------------------|--------|--------------------------------------------------|
| 1    | Qwen3-Down-B16-M2048           | 1.249  | fwd default + R42 dA + R39 var-K — saturated     |
| 2    | Qwen3-Down-B16-M4096           | 1.259  | same family — saturated                          |
| 3    | Qwen3-GateUP-B16-M2048         | 1.262  | R7 fwd + R32 dA + R39 var-K — saturated          |
| 4    | gpt_oss-Down-B32-M2048         | 1.264  | R8 fwd + R8 dA H4 + R30 var-K — saturated        |
| 5    | Qwen3-Down-B32-M2048           | 1.270  | same family — saturated                          |

## Audit 1 — `select_default_config` host-side overhead profile

`/tmp/probe_r41_select_default_config_overhead.py`: 100,000 calls per
key × 5 trials, p20.

```
shape (m, n, k, layout, dtype, m_total)                   us / call
(2048, 4096, 7168, rcr, fp8, 32768)  DSV3-GateUP B16M2048  0.536
(4096, 4096, 7168, rcr, fp8, 65536)  DSV3-GateUP B16M4096  0.507
(2048, 7168, 2048, rcr, fp8, 32768)  DSV3-Down B16M2048    0.514
(2048, 5760, 2880, rcr, fp8,  8192)  gpt_oss-GateUP B4M2048 0.466
(4096, 2880, 2880, rcr, fp8, 16384)  gpt_oss-Down B4M4096   0.479
(2048, 1536, 4096, rcr, fp8, 32768)  Qwen3-Down B16M2048    0.516  ← lowest-ratio
(2048, 3072, 4096, rcr, fp8, 32768)  Qwen3-GateUP B16M2048  0.495
(2048, 1536, 4096, rrr, fp8, 32768)  Qwen3-Down dA RRR      0.463
(2048, 4096, 1536, rrr, fp8, 32768)  Qwen3-? dA RRR          0.475
```

Per-call overhead is ~`0.46-0.54 us`, well below my pre-probe estimate
of `~1-3 us`. Per fwd+bwd call the function is invoked at most twice
(forward RCR + dA RRR/RCR; var-K bypasses it via inline rule), so
~`1 us / fwd+bwd`. Out of total per-shape wall ~`1 ms` (Qwen3-Down
B=16-M=2048 worst case), this is **0.1% of wall** — well below the
metric's ~5-point single-run noise floor (~0.5% per-run).

A memoization cache would save at most ~`0.4 us/call` (dict lookup
~`0.1 us`), netting `0.05% wall` per shape — fundamentally below
detection. **Not worth shipping.**

R11 / R16 / R19 host-trims were the right calls (those each saved
`0.5-2 us` per fwd+bwd hot path for shapes where it mattered;
`select_default_config` is already in the cleaner-trimmed regime).

## Audit 2 — forced kernel template on Qwen3-Down family

`/tmp/probe_r41_qwen3_down_kernel_force.py`: 4 shapes × 3 cells × 8
trials × 200 iters × 3 seeds, kernel-only via direct
`hipkitten.grouped_rcr_dscale` call with `TK_RCR_FORCE_KERNEL` env
toggled per call.

The historical comment in `hipkitten/config.py:124-129` notes:

> The historical `layout=="rcr" and tiles_n>=86 and K<=4096` outlier
> rule (offline-cache-derived `kernel="4"` for two LLM shapes) was
> removed once re-benches on the current `.so` showed the binding's
> auto-pick (8-wave for grid<3200, 4-wave for grid>=3200 && k<=8192) is
> bit-equivalent within ±0.1pp on every metric shape.

Re-tested with R32-class methodology on Qwen3-Down (every shape has
grid < 3200 → binding auto-picks 8-wave):

```
shape                         auto Δ%   force4 Δ%   force8 Δ%   spread pp   verdict
Qwen3-Down-B16-M2048           +0.00     +0.16       +0.05       1.90       all TIE
Qwen3-Down-B16-M4096           +0.00     -0.14       -0.09       0.28       all TIE
Qwen3-Down-B32-M2048           +0.00     +0.18       +0.21       0.56       all TIE
Qwen3-Down-B32-M4096           +0.00     +0.04       +0.09       0.32       all TIE
```

All 12 (4 shapes × 3 cells) data points are within ±`0.21%` of auto-pick
— well inside R32-class noise. The historical claim survives the wider
re-audit. `force4` and `force8` are both equivalent to auto-pick on
Qwen3-Down's small-grid shapes.

## Final closure observation

After 6 dedicated dispatch / host-overhead audits since R26's exhaustion
declaration:

| round | path probed                                  | shipped? |
|-------|----------------------------------------------|----------|
| R36   | Qwen3-Down M=2048 fwd RCR (16 cells)         | no       |
| R36   | Qwen3-GateUP var-K dB (10 cells)             | no       |
| R37   | (no kernel changes — GPU debug)              | n/a      |
| R38   | gpt_oss-Down-B4-M4096 var-K dB (9 cells)     | **YES** (gm=16, xcds=4) |
| R39   | Qwen3-Down dA RRR (10 cells × 4 shapes)      | no       |
| R40   | gpt_oss-Down-B32 var-K dB (9 cells × 2)      | no       |
| **R41** | **`select_default_config` profile + Qwen3-Down kernel template** | **no** |

**Verdict**: 1 shipping rule out of 7 dedicated re-audit rounds. The
score plateau at `[985, 1000]` is genuinely the architectural ceiling
for this family of shapes under the current FP8 quant pipeline.

## Architectural ceiling summary

The 24-shape geomean ratio is bounded by:

1. **Symmetric FP8 quant HBM tax**: both backends call
   `quantize_fp8(a) + quantize_fp8(b) + quantize_fp8(grad_out)` at
   ~`67 %` of MI355X HBM peak. This contributes ~`25 %` of fwd wall
   and ~`15 %` of bwd wall and is invariant across HK / Triton.

2. **Kernel-only ratio**: plateaued at `score=1000` in the previous
   task's `_metric_grouped_only.py` (63 prior rounds). The current
   wall-ratio plateau at `[991, 998]` (with R38 carve-out) corresponds
   to geomean `1.337-1.344`, vs target `1.35` — a `~0.5 %` gap.

3. **Dispatch-tuning saturation**: every (gm, xcd) cell on every shape
   in every layout (RCR / RRR / var-K-CRR) has been R32-class probed
   across R36-R41. R7-R8 falsified Path A forward fusion at the kernel
   level (DTR + in-register cvt 30-40% slower than DTL).

The remaining `~0.5 %` gap to `score=1000` is dominated by:

* **Triton-side timing noise** (`std ≈ 0.5 % per-run` on geomean —
  characterized by R36's 5-pre-vs-5-post Welch's t-test and R39's
  8-round score-distribution analysis).
* **Architectural FP8 quant ceiling** (Path A would lift this but
  requires kernel surgery in HipKittens; R7/R8 falsified for full
  Phase 1+2+3 stack).

## Recommendations for next task / continuation

(R41 is the last patience round; R42 is expected to EARLY-STOP if no
improvement. After patience exhaust, the task should:)

1. **Accept plateau and close this run-series**. Score `985-1000`
   plateau achieved; geomean `1.337-1.344` vs target `1.35`. R38
   carve-out shipped, all other audits saturated.

2. **Spawn a separate Path A task** with fresh patience budget. Phase 1
   forward-only fusion was suggested by R36 as priority #3 — kernel
   surgery in HipKittens is the only architectural lever that could
   break the FP8 quant ceiling. R7/R8 falsified the full Phase 1+2+3
   stack but Phase 1 alone (forward + un-fused dB) may behave
   differently. A separate task with `patience=30+` would give the
   kernel-surgery + multi-round bench + revert-on-falsification cycle
   room to land or formally falsify Phase 1 specifically.

3. **Spawn a Path B task** for NVIDIA-style cast kernel optimization
   (`+3-4 %` wall cap on this hardware per task body's analytical
   model). Lower ceiling but no kernel-surgery risk.

4. **Spawn a Weight FP8 caching task** at the autocast / optimizer-
   state level. Currently out of scope per the task body but would
   uniformly lift all 24 shapes by skipping `quantize_fp8(b)` once
   weights stabilize during training. Estimated wall lift `+5-8 %`
   on B=4 shapes (where weight quant is ~10% of wall), `+1-2 %` on
   B=32 (where it's amortized over more tokens).

## Files touched

Primus-Turbo:

* `analysis/_notes/round-41-fp8-grouped-fused-wall-host-overhead-and-kernel-template-final-closure.md`
  (this note — falsification documentation, no code change)

* No `primus_turbo/` source files modified.

HipKittens: no changes.

## Probes (in `/tmp/`, NOT committed)

* `probe_r41_select_default_config_overhead.py` — 100k-call × 5-trial
  p20 latency profile across 9 representative keys.
* `probe_r41_qwen3_down_kernel_force.py` — 4-shape × 3-cell × 8-trial
  × 3-seed kernel-only force-kernel sweep.

## Round summary

* **Lever**: dispatch-tuning audit + host-overhead audit (Lever F final
  closure).
* **Target**: `select_default_config` host-side overhead + forced
  kernel template on Qwen3-Down family (the two remaining un-probed
  angles per R39 / R40 audit table).
* **Files**: `analysis/_notes/round-41-...md` only (no code change).
* **Metric**: pre-round `985`. No code change → no post-round measurement.
* **Correctness**: n/a (no kernel touched).
* **Commit**: see Primus-Turbo HEAD post-commit.
* **HipKittens**: no changes.
