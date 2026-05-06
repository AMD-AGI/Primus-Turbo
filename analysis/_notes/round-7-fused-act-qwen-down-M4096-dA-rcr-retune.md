# Round 7 — fused-act: Qwen3-Down dA RCR M=4096 rule (R4 sibling, post-H4-reroute follow-up #2)

**Date**: 2026-05-05
**Primus-Turbo HEAD before**: `60307120efe8e92190e9be877d508a893599156a`
**HipKittens HEAD**: unchanged (no kernel work this round)
**Lever**: dispatch / config — extend R4 H4-reroute follow-up to the missing `tiles_m=16` gap
**Target shapes**: Qwen3-235B-A22B-Down-B{16,32}-M4096 dA backward (post-H4 RCR kernel)
**Continuation of R6 main line** — auditing all dispatch rules pre-dating R3's H4 reroute extension

## Summary

| metric          | pre-R7 (5-run, 1 outlier) | post-R7 (3-run) | Δ        |
| --------------- | ------------------------- | --------------- | -------- |
| score           | 1000 (cap)                | 1000 (cap)      | 0        |
| geomean (incl. outlier) | mean 1.3761       | 1.3890          | +0.013   |
| geomean (excl. outlier) | mean 1.3873       | 1.3890          | +0.0017  |
| Qwen3-Down-B16-M4096 ratio | 1.337          | 1.338           | +0.001   |
| Qwen3-Down-B32-M4096 ratio | 1.362          | 1.363           | +0.001   |

Score capped at 1000. Wall lift in the noise band (the metric noise floor
spans ~1.331 to 1.396 across 5 runs at unchanged HEAD). Per-shape kernel
signal **is** robust (+0.77% / +1.01% all-positive per seed at the kernel
boundary); wall noise from non-kernel sources (Triton launch jitter,
quantize_fp8 throughput variance, thermal) drowns it for the 2 affected
shapes. Score is capped so the gain is buffer rather than headline.

## Why this rule was missing

R4 (`perf bf2146f`, 2026-05-04) added the post-R3 H4-reroute carve-out for
Qwen3-Down dA at `config.py:2531` with predicate
`tiles_n == 6 AND tiles_m == 8 AND k == 4096 AND m_total >= 32768`,
giving `(group_m=4, num_xcds=4)` for the M_per_group=2048 family
(B={16,32}). The `tiles_m == 8` clause was correct for M=2048 but
**implicitly excluded M=4096** (tiles_m=16). The 2 M=4096 shapes
(B16-M4096, B32-M4096) fell through to the binding default
`(gm=4, num_xcds=None=kernel 8)` — neither a probed optimum nor R4's
verified `(4, 4)` cell.

Same H4-reroute follow-up class as R6 (Qwen3-GateUP M=4096 dA): rules
written for the FP8 RRR kernel pre-R3 are dead code for the metric
post-R3 (since all aligned `trans_b=False` calls reroute to RCR), and
the live RCR rules sometimes have gaps in their tile-geometry coverage.

## Probe data

`/tmp/probe_round_7_qwen_down_dA_rcr.py` — 200-iter × 7-trial × p20 × 3 seeds,
direct call of `hk.grouped_rcr_dscale` with post-H4 inputs:
`a_fp8=[B*M, k=N_fwd=4096]` (grad_out), `b_fp8=[B, n=K_fwd=1536, k=N_fwd=4096]`
(b_T after H4).

Cross-shape mean Δ vs default `(gm=4, xcds=None=kernel 8)` on the 4 shapes:

| cell        | B16-M2048 (R4)        | B16-M4096 (GAP)       | B32-M2048 (R4)        | B32-M4096 (GAP)       |
| ----------- | --------------------- | --------------------- | --------------------- | --------------------- |
| `(4, 8)` def | +0.00%               | +0.00%                | +0.00%                | +0.00%                |
| `(4, 4)` R4  | **+2.27%** ← R4 wins | +0.51% mixed*        | **+5.03%** ← R4 wins  | +1.05% all-pos        |
| `(8, 4)` NEW | +1.13%               | **+0.77%** all-pos   | +2.65%                | **+1.01%** all-pos    |
| `(2, 4)`    | +1.52%                | +0.10%                | +3.12%                | +1.06%                |
| `(1, 4)`    | +1.16%                | +0.35%                | +3.17%                | +0.25%                |
| `(16, 4)`   | +1.15%                | +0.38%                | +2.63%                | +0.65%                |
| `(2, 8)`    | -3.16%                | -4.56%                | -3.05%                | -2.54%                |
| `(1, 8)`    | -19.68%               | -9.21%                | -17.74%               | -6.67%                |

* `(4, 4)` per-seed on B16-M4096: -0.23% / +0.78% / +0.99% (one negative seed,
  spread 1.22pp). `(8, 4)` per-seed on B16-M4096: +0.46% / +1.04% / +0.80%
  (all-positive, spread 0.58pp = 2× tighter).

`(gm=8, xcds=4)` is the **all-positive per-seed** winner on both M=4096
shapes; `(gm=4, xcds=4)` would also work but has a -0.23% seed-42 outlier
on B16-M4096 (sub-noise but mixed-sign). `(8, 4)` chosen for robustness.

The R4 rule's `(gm=4, xcds=4)` for M=2048 is independently confirmed as
the per-shape winner (+2.27% / +5.03%); not changed.

## DSV3-GateUP dA cross-check (negative result)

Also probed DSV3-GateUP dA after H4 (tiles_n=28, tiles_m=8 or 16, k=4096,
m_total ∈ {32768, 65536, 131072}) at `/tmp/probe_round_7_dsv3_gateup_dA_rcr.py`.
This dispatch hits the Round-20/58/67 rule at `config.py:2410`
(`tiles_n == 28 AND 8 <= tiles_m <= 16 AND k <= 4096 → (gm=32, num_xcds=2)`)
which was tuned for DSV3-Down **forward** (k=K_fwd=2048).

Result: **(gm=32, num_xcds=2) is optimal for DSV3-GateUP dA RCR too**.
No probed cell beats it by more than +0.24% on any shape; most regress
0.5-50%. The Round-20/58/67 rule happens to be stable across the
forward (k=2048) → dA (k=4096) transition for `tiles_n=28`. **No lever
here**; documented as audit-complete.

## Sibling shape sanity (rule MUST add `tiles_m == 16` clause)

| sibling                                       | predicate | catches first?               |
| --------------------------------------------- | --------- | ---------------------------- |
| Qwen3-Down dA M=2048 (`tiles_m=8`)            | R4 above  | YES — R4 catches first       |
| Qwen3-Down fwd (`n=N_fwd=4096, tiles_n=16`)   | excluded by `tiles_n==6` | N/A           |
| DSV3-Down dA after H4 (`tiles_n=8`)           | excluded by `tiles_n==6` | N/A           |
| Qwen3-GateUP dA after H4 (`tiles_n=16`)       | excluded                 | N/A           |
| DSV3-GateUP dA after H4 (`tiles_n=28`)        | excluded                 | N/A           |
| gpt_oss dA after H4 (`tiles_n=11`)            | excluded                 | N/A           |
| Dense FP8 (`m_total=None`)                    | excluded by `m_total is not None` | N/A  |
| DoD smoke grouped FP8 (`tiles_n in {8, 28}`)  | excluded                 | N/A           |

The `k == 4096` clause keeps the rule uniquely tied to Qwen3-Down dA after
H4 reroute (`k=N_fwd=4096` unique to this path among `tiles_n=6` shapes
in the 24-shape MoE suite).

## Correctness

`/tmp/probe_round_7_correctness.py` — full fwd+bwd vs torch-native ref:

| shape                   | SNR(out)  | SNR(dA)   | SNR(dB)   | status |
| ----------------------- | --------- | --------- | --------- | ------ |
| Qwen3-Down-B16-M4096    | 28.5 dB   | 28.5 dB   | 28.5 dB   | PASS   |
| Qwen3-Down-B32-M4096    | 28.5 dB   | 28.4 dB   | 28.4 dB   | PASS   |
| Qwen3-Down-B16-M2048    | 28.5 dB   | 28.5 dB   | 28.5 dB   | PASS   |
| Qwen3-Down-B32-M2048    | 28.5 dB   | 28.5 dB   | 28.5 dB   | PASS   |

All ≥ 25 dB. `(gm, xcds)` are pure persistent-grid scheduling knobs.

## Files touched

| repo         | file                                                          | change                  |
| ------------ | ------------------------------------------------------------- | ----------------------- |
| Primus-Turbo | `primus_turbo/pytorch/kernels/hipkitten/config.py`            | Add R7 narrow rule for `tiles_n=6, tiles_m=16, k=4096, m_total>=32768 → (gm=8, num_xcds=4)` after the R4 rule |
| Primus-Turbo | `analysis/_notes/round-7-fused-act-qwen-down-M4096-dA-rcr-retune.md` | new doc note    |

## Next round suggestion

The R6+R7 H4-reroute follow-up audit is now substantially complete:
- R6: Qwen3-GateUP dA RCR M=4096 (`tiles_n=16, tiles_m=16, k=3072`) → `(gm=2, None)`
- R7: Qwen3-Down dA RCR M=4096 (`tiles_n=6, tiles_m=16, k=4096`) → `(gm=8, 4)`
- R7 audit-complete: DSV3-GateUP dA RCR (`tiles_n=28, k=4096`) — already optimal at R20/58/67's `(32, 2)` (no lever)
- R4: Qwen3-Down dA RCR M=2048 + DSV3-Down dA RCR — already done

The only remaining un-audited dA RCR family is **gpt_oss dA after H4**
(`tiles_n=11, tiles_m={8,16,22}`). gpt_oss had been entering RCR PRE-R3
via the K_RRR%128!=0 clause (R14/R18), so its RCR-side tunings were
written when only gpt_oss was on this path. The R3 widening doesn't
change which kernel runs for gpt_oss (it was already RCR), but it's
worth re-verifying the gpt_oss RCR cells haven't drifted vs newer
HipKittens builds.

Lower-priority maintenance:

1. The wall metric noise band is wide (5-run pre-R7: 1.3313 - 1.3962,
   one outlier at 1.3313 with progress=0.986 FAIL). This single-run
   variance makes per-rule wall validation hard; consider running the
   3-run-mean methodology by default instead of single-run.
2. Continue HK kernel-side K-tail epilog work for `gpt_oss-Down`
   (the persistent lowest-ratio shape at 1.27-1.28, kernel-bound per
   R5 decomposition).
3. Round 10 will trigger automatic DoD checkpoint — ensure the
   accumulated R6+R7 dispatch rule additions don't regress DoD.
