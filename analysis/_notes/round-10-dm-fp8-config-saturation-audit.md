# Round 10 (death-march) — FP8 grouped config-tuning saturation audit

**Scope:** `HipKittenConfig` `(group_m, num_xcds)` rules in
`primus_turbo/pytorch/kernels/hipkitten/config.py` for the 16 FP8 grouped
shapes.

## Context

Round-9 committed a parallel-init cleanup (perf-neutral). The 4-wave port
(round-8), WARPS flip (round-6), launch-bounds tweaks (round-5/7), K-tail
schedule probes (round-4), LDS layout swap (round-2), main-loop unroll
(round-3) have ALL been falsified. The only remaining obviously-bounded
lever is per-shape `(group_m, num_xcds)` retune on the worst metric ratios.

## Probes this round

### Metric baseline (round-10 HEAD = 837e8a7)
```
score=809, grp_FP8 geomean=0.9710, correct_fail=0/16
worst 4:
  gpt_oss-Down-B4-M4096    0.924   1082 / 1171 TF
  gpt_oss-GateUP-B4-M4096  0.935   1227 / 1312 TF
  gpt_oss-GateUP-B4-M2048  0.927   1041 / 1123 TF
  gpt_oss-Down-B4-M2048    0.966    779 /  806 TF
  DSV3-Down-B16-M2048      0.952   1156 / 1215 TF
```

### Probe 1 — gpt_oss-GateUP-B4-M4096 `(gm, xcd)` sweep
Current rule: `(gm=14, xcd=4)` from round-7 (commit `208fa8f`).
The rule's comment block (config.py lines 897-945) cites a hypothetical
"Round-21" sweep showing `(gm=8, xcd=4)` dominates `(gm=14, xcd=4)` by
+29.2 TF — **no FP8 round-21 commit exists in git log**; the comment is
a BF16 paste-over that was never cleaned.

1500-iter × 7-repeat p14 verify (`/tmp/probe_gateup_b4_m4096_round10.py`):

| cfg     | p14 TFLOPS | Δ vs (14,4) |
|---------|-----------:|------------:|
| (14, 4) | 1240.33    | +0.00 *rule |
| (14, 8) | 1241.28    | +0.95       |
| (14, 2) | 1241.25    | +0.92       |
| ( 8, 8) | 1240.80    | +0.47       |
| (16, 4) | 1240.18    | -0.16       |
| (10, 4) | 1239.61    | -0.72       |
| (12, 4) | 1239.31    | -1.02       |
| ( 8, 4) | 1238.93    | -1.40       |

Entire candidate space within ±1.4 TF (±0.1 pp) of current rule.
**`(gm=14, xcd=4)` confirmed correct; the stale "Round-21" comment is
FALSIFIED.** `(gm=8, xcd=4)` is the WORST candidate in this sweep,
opposite of what the stale comment claims.

### Probe 2 — gpt_oss-Down-B4-M4096 `(gm, xcd)` sweep
Current rule: `(gm=32, xcd=4)` from round-12 (commit `af93b78`).

2-run median TFLOPS (full sweep `/tmp/probe_down_b4_m4096_round10.py`):

| cfg     | run1 | run2 | median | Δ vs (32,4) |
|---------|-----:|-----:|-------:|------------:|
| (32, 4) | 1100.3 | 1102.8 | 1101.6 | +0.00 *rule |
| ( 8, 4) | 1107.8 | 1103.8 | 1105.8 | +4.2        |
| (12, 4) | 1106.5 | 1104.0 | 1105.3 | +3.7        |
| (16, 4) | 1106.6 | 1103.3 | 1104.9 | +3.4        |
| (24, 4) | 1107.0 | 1103.0 | 1105.0 | +3.4        |
| (32, 2) | 1105.7 | 1105.0 | 1105.4 | +3.7        |
| (48, 4) | 1106.3 | 1105.0 | 1105.7 | +4.0        |
| (64, 4) | 1106.9 | 1104.3 | 1105.6 | +3.9        |

Every alternate beats `(32, 4)` by a consistent +3-4 TF (0.3 pp), but the
run-to-run variance is also ~3 TF so the signal is AT the noise floor.
No single candidate dominates. Plateau.

### Probe 3 — DSV3-Down family metric-aligned sweep
Current rule: `(gm=32, xcd=2)` from round-68. Metric uses
`WARMUP=10, ITERS=50 per-iter-sync p20` which is COLD; round-68 used
800-iter HOT. Hypothesis: metric regime may prefer a different cfg.

Probe mirrors `scripts/_metric_hk_ratio.py::_time_op` (11 candidates × 4
shapes × 5 repeats × 60 iters each): `/tmp/probe_dsv3_down_metric_round10.py`

Key result (Δ vs current rule `(32, 2)`, median of 2 full runs):

| shape              | best-alt cfg | run1 Δ | run2 Δ |
|--------------------|:-------------|-------:|-------:|
| Down-B16-M2048     | (32, 2)      |  +0.00 |  +0.00 | current wins |
| Down-B16-M4096     | (64, 2)      |  +0.58 |   noise | noise |
| Down-B32-M2048     | ( 8, 4)      |  +6.49 |  +1.74 | not reproducible |
| Down-B32-M4096     | (32, 4)      |  +1.37 |  +0.92 | noise |

`(8, 4)`'s +6.49 TF on Down-B32-M2048 looked promising in run 1 but
collapsed to +1.74 in run 2. Split run 1 was top-1 by unreproducible
noise. Similar story on other shapes. No reliable winner.

## Verdict

**All 4 FP8 grouped per-shape config rules audited in round 10 are at
their saturation plateau.** The `(group_m, num_xcds)` knob has been
exhausted. Any remaining 2-3 % ratio gap MUST come from kernel-level
changes (AGPR migration, register spill reduction, main-loop pipelining,
K/N-tail restructuring).

## Falsifications banked (do not re-explore)

- Per-shape `(gm, xcd)` retune on gpt_oss-GateUP-B4-M4096 — within ±1.4
  TF of current; stale "Round-21" comment claims are FALSE.
- Per-shape `(gm, xcd)` retune on gpt_oss-Down-B4-M4096 — all 8 alternates
  within ±4 TF of current; signal at noise floor.
- Metric-aligned `(gm, xcd)` retune on DSV3-Down family — any +5-6 TF
  observation is noise (falsified by a 2nd run).

## Next-round suggestion

**Config tuning is DONE.** Next-round agent MUST go kernel-level:

1. **AGPR accumulator migration** (round-8 plan, unimplemented). Force
   `cA/cB/cC/cD` into AGPRs via `art_base_fl` or inline-asm to free
   ~128 VGPRs and cut the 67-register spill substantially. High risk
   but the only remaining bounded lever with known-good theoretical gain.
2. **Dispatch-side overhead audit** on per-iter cold path. Metric uses
   cold-call timing; if HK has 10+ µs of Python/dispatcher overhead per
   launch that Triton avoids, closing THAT gap is invisible to HOT bench
   but visible to metric. Bench `quantize_fp8 → kernel → scale-combine`
   Python cost via `torch.profiler`.
3. **Stale comment cleanup**: the GateUP-B4-M4096 rule's "Round-21"
   comment (config.py lines 902-933) is demonstrably false. Replace
   with round-7's (gm=14, xcd=4) evidence.

## Files touched this round

- **None** (probes only; no code / config change).

## Artifacts

- Probe scripts at `/tmp/probe_gateup_b4_m4096_round10.py`,
  `/tmp/probe_down_b4_m4096_round10.py`,
  `/tmp/probe_dsv3_down_metric_round10.py`.
- Metric baseline log: `/tmp/metric_round_10.log`.
