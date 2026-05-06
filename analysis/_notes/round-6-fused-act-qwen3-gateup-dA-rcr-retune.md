# Round 6 — fused-act: Qwen3-GateUP dA RCR config re-tuning (post-H4-reroute follow-up)

**Date**: 2026-05-05
**Primus-Turbo HEAD before**: `ed91619b7f38bf1dbf45e552771a878cd9a54205`
**HipKittens HEAD**: unchanged (no kernel work this round)
**Lever**: dispatch / config re-tune (H4 reroute follow-up)
**Target shape**: Qwen3-235B-A22B-GateUP-B{16,32}-M4096 dA backward
**Metric**: `fused_act_wall_score` (grouped FP8 fwd+bwd wall, HK fused vs Triton un-fused)

## Summary

| metric          | pre-R6 (5-run mean) | post-R6 (5-run mean) | Δ        |
| --------------- | ------------------- | -------------------- | -------- |
| score           | 1000 (capped)       | 1000 (capped)        | 0        |
| geomean         | ~1.3867             | ~1.3884              | +0.0017  |
| below_target    | 8 / 24              | 7 / 24               | −1       |
| Qwen3-GateUP-B16-M4096 ratio | 1.345  | 1.360 | **+0.015** |
| Qwen3-GateUP-B32-M4096 ratio | 1.393  | 1.406 | **+0.013** |

Score is at the cap (1000) so the geomean lift is buffer rather than headline,
but the rule fix is **structurally correct** (R32 was tuned for the wrong
kernel) and the per-shape kernel signal is robust.

## Why R32 was stale

The `tiles_n == 16 and m_total >= 32768` FP8 dispatch rule (`config.py:2793`)
is annotated as the "Round-32 (2026-05-02) Qwen3-GateUP dA RRR re-verify" rule.
It picked `(group_m=1, num_xcds=4)` after a 12-trial × 200-iter × 3-seed sweep
on the **FP8 RRR grouped kernel**.

Current Primus run **R3** (`perf 3ed7a402`, 2026-05-04) extended the FP8 H4
reroute (`grouped_gemm_fp8_impl.py: GroupedGEMMFP8HipKittenBackend.execute`)
to **all aligned `trans_b=False` RRR dA calls**: `b → fp8_transpose_3d(b)`,
then `trans_b=True` (RCR) — caching the transposed `b` in
`_FP8_H4_TRANSPOSE_CACHE`. This means **the FP8 RRR grouped kernel is no
longer the live path for any aligned dA call**; the RCR kernel runs instead.

R32's `(1, 4)` was a RRR optimum applied to the RCR kernel — same shape
parameters but a different kernel implementation, different optimal
`(group_m, num_xcds)`.

## Probe data

`/tmp/probe_round_6_qwen_gateup_dA_rcr.py` — 200-iter × 7-trial × p20 × 3 seeds
direct call of `hk.grouped_rcr_dscale` with the post-H4 inputs:
`a_fp8=[B*M, k=3072]` (grad_out), `b_fp8=[B, n=4096, k=3072]` (b_T after H4).

Cross-shape mean Δ vs default `(gm=4, xcds=None=kernel 8)`:

| cell        | B16-M2048 | B16-M4096 | B32-M2048 | B32-M4096 | avg    |
| ----------- | --------- | --------- | --------- | --------- | ------ |
| `(1, 4)` R32 | -0.18%   | +0.08%    | -0.87%    | +0.04%    | -0.23% |
| `(2, 8)` NEW | -2.32%   | **+1.52%**| -2.33%    | **+1.69%**| -0.36% |
| `(16, 4)`   | -0.61%    | -0.87%    | -0.88%    | -0.96%    | -0.83% |
| `(32, 4)`   | -0.76%    | -0.75%    | -0.64%    | -0.95%    | -0.78% |
| `(8, 4)`    | -1.25%    | -0.29%    | -1.30%    | -0.40%    | -0.81% |

`(gm=2, xcds=8)` wins both M=4096 shapes by **+1.5..+1.7%** over the binding
default. Per-seed deltas all-positive on both M=4096 shapes (B16: +1.45% /
+1.49% / +1.61%; B32: +1.70% / +1.63% / +1.73%), well above the 0.5pp
run-to-run spread. M=2048 shapes regress -2.3% on `(2, 8)` — the rule MUST
gate on `tiles_m == 16`.

R32's `(1, 4)` is essentially **tied with default** on the 3 shapes it caught
(B16-M2048 -0.18%, B16-M4096 +0.08%, B32-M4096 +0.04%) — sub-noise, no help.

## xcds=None ≡ xcds=8 equivalence

`/tmp/probe_round_6_xcds_equiv.py` confirmed:
- bit-identical output (`max_abs_diff=0.0`) between `num_xcds=0` (config
  `None` → `xcds_arg=0` → kernel `BLOCK_SWIZZLE_NUM_XCDS=8` default) and
  explicit `num_xcds=8` on both M=4096 shapes.
- perf within 0.2-0.7% (well below kernel-only noise band).

So the new rule uses `num_xcds=None` for config consistency (matches the
sibling Qwen3-Down M=4096 fwd rule at `config.py:1886`).

## Sibling shape sanity (rule MUST add `k == 3072` clause)

| sibling | predicate | catches first? |
| ------- | --------- | -------------- |
| DSV3-GateUP fwd M=4096 (`tiles_n=16, tiles_m=16, k=7168`) | R8 rule `config.py:2205` (`k == 7168`) | YES — R8 catches first; `k==3072` clause excludes anyway (defensive) |
| Qwen3-Down fwd M=4096 (`tiles_n=16, tiles_m=16, k=1536`) | R6 sibling rule `config.py:1886` (`k == 1536`) | YES — that rule catches first |
| Qwen3-GateUP fwd (`n=N_fwd=3072, tiles_n=12`) | excluded by `tiles_n==16` |  N/A |
| DSV3-GateUP dA after H4 (`n=K_fwd=7168, tiles_n=28`) | excluded by `tiles_n==16` | N/A |
| Dense FP8 callers (`m_total=None`) | excluded by `m_total is not None` guard | N/A |
| DoD smoke grouped FP8 fwdbwd shapes (per R32 audit) | only `tiles_n ∈ {8, 28}`, neither matches | N/A |

The `k == 3072` clause keeps the rule uniquely tied to Qwen3-GateUP dA after
H4 reroute (`k = N_fwd = 3072` is unique to this path in the 24-shape MoE
suite).

## Correctness

`/tmp/probe_round_6_correctness.py` — full fwd+bwd vs torch-native ref
(per-group BF16 GEMM):

| shape                       | SNR(out)  | SNR(dA)   | SNR(dB)   | status |
| --------------------------- | --------- | --------- | --------- | ------ |
| Qwen3-GateUP-B16-M4096      | 28.5 dB   | 28.4 dB   | 28.4 dB   | PASS   |
| Qwen3-GateUP-B32-M4096      | 28.4 dB   | 28.4 dB   | 28.5 dB   | PASS   |
| Qwen3-GateUP-B16-M2048      | 28.5 dB   | 28.5 dB   | 28.5 dB   | PASS   |
| Qwen3-GateUP-B32-M2048      | 28.4 dB   | 28.5 dB   | 28.5 dB   | PASS   |

All ≥ 25 dB (E4M3 noise floor). `(gm, xcds)` are pure persistent-grid
scheduling knobs; arithmetic and FP8 quantization rounding invariant —
same property documented for R6 / R7 / R8 / R10 / R27 / R32 / R39 / R42 /
R43 / R44 / R45 in `config.py`.

## Regression checks

### `_metric_grouped_only.py` (un-fused path, target ≥ 980)
- Pre-R6 baseline (5-run): 970 (HEAD ed91619)
- Post-R6 (4-run): 968, 975, 970, 970 → mean 970.75
- **No regression** (within noise band of 5 points). Pre-R6 was already at
  970 (below the 980 target — pre-existing, not caused by this round).

### Per-shape fused metric Qwen3-GateUP (5-run, post-R6 vs pre-R6)
| shape         | pre-R6 mean | post-R6 mean | Δ        |
| ------------- | ----------- | ------------ | -------- |
| B16-M2048     | 1.333       | 1.333        | 0.000    |
| B16-M4096     | 1.345       | 1.360        | **+0.015** |
| B32-M2048     | 1.353       | 1.350        | -0.003   |
| B32-M4096     | 1.393       | 1.406        | **+0.013** |

The two M=4096 shapes (which the new rule actually catches) lift by +0.013
and +0.015 ratio — both consistent with the kernel-only +1.5-1.7% projection
× 23% wall fraction = +0.34..+0.39% wall (observed +0.9-1.1%, on the
favorable side of projection). M=2048 shapes (which fall to default) are
unchanged within noise — confirms the rule scope is correct.

## Files touched

| repo         | file                                                          | change                  |
| ------------ | ------------------------------------------------------------- | ----------------------- |
| Primus-Turbo | `primus_turbo/pytorch/kernels/hipkitten/config.py`            | Replace R32 rule with R6 narrow rule (preserves R27/R32 historical comment block) |
| Primus-Turbo | `analysis/_notes/round-6-fused-act-qwen3-gateup-dA-rcr-retune.md` | new doc note            |

## Next round suggestion

The R32 rule fix mechanically maps to a class of opportunity: **all rules
in `config.py` that were tuned BEFORE the R3 H4 reroute (commit `3ed7a402`,
2026-05-04) and still match RRR-time inputs are potentially stale**. R3
collapsed many RRR aligned dA calls into RCR. Candidates to audit next:

- The R42 narrow-N RRR rule at `config.py:2657` (`tiles_n <= 8 and
  m_total >= 32768` → `(gm=16, xcds=4)`) — covers Qwen3-Down dA + DSV3-Down
  dA after H4. Re-probe with the post-H4 RCR kernel to see if a different
  cell wins.
- The R43/R44 wide-N RRR rule at `config.py:2862` (`tiles_n == 28 and
  m_total >= 32768` → `(gm=16, xcds=4)`) — covers DSV3-GateUP dA after H4.
  Re-probe similarly.

Same methodology: 200-iter × 7-trial × p20 × 3 seeds direct call of
`grouped_rcr_dscale` with post-H4 inputs. Look for >1% per-seed all-positive
deltas; commit narrow rule with `k == <unique>` clause to avoid catching
sibling shapes.

Lower-priority follow-ups (in maintenance band):

- Run another `bash scripts/run_dod_metric.sh --full` after R10 (next
  scheduled DoD checkpoint) to confirm DoD stays healthy under the
  recent R3-R6 dispatch changes.
- Continue monitoring the metric noise band (currently ~1.383-1.393
  geomean, score robustly capped at 1000).
- Consider HK kernel-side K-tail epilog work for `gpt_oss-Down` shapes
  (the lowest-ratio shape at 1.27, kernel-bound per R5 decomposition).
  Multi-round investment, small score impact (capped) but documents
  the architectural path.
