# Round 21 — BF16 grouped (gm, xcd) re-tune against post-round-19/20 BUFFER kernel

## TL;DR

Round 19 (col-layout C-store FLAT→BUFFER, +85pp) and round 20 (K-tail /
N-tail scalar helpers FLAT→BUFFER, within-noise) reshaped the per-tile
completion latency profile of the HipKittens BF16 grouped RCR kernel.
The 4 BF16 gpt_oss `(group_m, num_xcds)` rules tuned in rounds 7-10
against the FLAT-store kernel are now stale — a metric-aligned per-iter-
sync re-sweep finds 4 better cells with +0.97 to +1.75pp wins per shape.

| shape (gpt_oss BF16)     | round-N rule | new rule  | per-iter-sync win | source     |
|--------------------------|-------------:|----------:|------------------:|-----------:|
| Down-B4-M4096            |       (8, 4) |  (32, 4)  | +8.93 TF  +0.76%  | round-10   |
| GateUP-B4-M4096          |       (2, 2) |  (12, 4)  | +20.68 TF +1.75%  | round-9    |
| GateUP-B32-M4096         |       (1, 2) |  (8,  4)  | +18.69 TF +1.49%  | round-70   |
| Down-B32-M2048 (split)   |       (1, 4) |  (16, 4)  | +11.51 TF +0.97%  | round default |

All 4 are pure scheduling-knob changes (group_m / num_xcds re-order the
persistent BF16 grouped RCR tile schedule; arithmetic + output bytes
unchanged). Bit-identical output verified for the Down-B4-M4096 update
(`/tmp/verify_bf16_down_b4_m4096_bit_eq.py`: max_abs_diff=0.0,
bit_eq=True; SNR vs fp32 ref = 49.62 dB before/after). Same property
holds for the 3 others by the same argument used in the round-7/9/10
commentary.

## Metric impact

Per-iter-sync mode is the acceptance metric used by
`scripts/_metric_grouped_only.py` (calls `_time_op` from
`_metric_hk_ratio.py`: 50-iter median of per-iter `cuda.synchronize()`-
fenced timings, 20th percentile). The kernel-side gains per shape
translate to:

| BF16 shape (gpt_oss only) | before ratio | after ratio | Δ      |
|---------------------------|-------------:|------------:|-------:|
| Down-B4-M4096             |        1.137 |       1.157 | +0.020 |
| GateUP-B4-M4096           |        1.158 |       1.171 | +0.013 |
| GateUP-B32-M4096          |        1.160 |       1.182 | +0.022 |
| Down-B32-M2048            |        1.167 |       1.183 | +0.016 |

(other 4 BF16 gpt_oss shapes' rules unchanged — no regression observed)

Combined effect on the 5-run metric mean:

| run               | grp_BF16 geomean | grp_FP8 geomean | score |
|-------------------|------------------:|----------------:|------:|
| round-20 final    |           1.1573 |          0.9619 |  879  |
| round-21 5-run    |           1.1649 |          0.9614 |  882  |
|                   |          +0.0076 |         -0.0005 |   +3  |

BF16 geomean +0.66pp; FP8 unchanged within noise. Score ticks +3 (from
3 separate BF16 ratio improvements that cap-cancel at 1.20 → contribute
to the geomean only via min(g/1.20, 1.0) before the cap).

## Why was the FLAT→BUFFER reroute moving the (gm, xcd) optimum?

Round 19 eliminated all `global_store_short` from the main loop (col-
layout C-store), which removes the per-tile epilog FLAT-instruction
serialization. The downstream effect on tile completion latency is
non-uniform: tiles that previously hit FLAT-class store stalls now
complete +5-8% faster, but the persistent grouped scheduler's
`(group_m, num_xcds)` choice is sensitive to the *spread* of completion
times across a wave (= XCD imbalance + WG-cohort stragglers).

The pre-round-19 rules picked small `group_m` (= small XCD-swizzle
cohort) to minimize variance — when each tile stalls on FLAT for ~10 µs,
a tighter cohort reduces tail wait. Post-round-19 the per-tile time is
smoother → larger `group_m` (= larger cohort, better L2 reuse on the
shared K-axis) wins on average without paying the variance penalty.

Same direction across all 4 shapes that retuned:

```
                        group_m           num_xcds
Down-B4-M4096            8 → 32  (+24)     4 → 4  (kept)
GateUP-B4-M4096          2 → 12  (+10)     2 → 4  (+2)
GateUP-B32-M4096         1 → 8   (+7)      2 → 4  (+2)
Down-B32-M2048           1 → 16  (+15)     4 → 4  (kept)
```

The 4 shapes whose rules did NOT change after this sweep are the ones
where (a) the round-9/10 cell happened to already sit in the new
optimum cluster (Down-B4-M2048 (gm=2, xcd=2), GateUP-B4-M2048 (gm=2,
xcd=2), GateUP-B32-M2048 (gm=8, xcd=4)) or (b) the new winner is the
old winner (Down-B32-M4096 (gm=1, xcd=4) is still the per-iter-sync
top at 1230.62 TF, with the next contender (gm=4, xcd=4) at -3.91 TF
-0.32%).

## What this round does NOT do

* **No FP8 config changes**: the FP8 (gm, xcd) sweep at
  /tmp/sweep_fp8_worst_round21.py + /tmp/verify_fp8_round21.py shows
  +0.5-2% kernel-side wins on B=4 shapes (Down-B4-M4096 1190 vs 1187,
  GateUP-B4-M4096 1481 vs 1452, GateUP-B4-M2048 1138 vs 1127) but those
  are *steady-state* (back-to-back) wins — the per-iter-sync metric
  shows them within Triton's ±2pp noise floor, and a full-metric
  measurement of the (gm=8, xcd=4) candidate for GateUP-B4-M4096
  (1230.7 TF post vs 1217.0 TF pre) was net-zero in score (Triton's
  per-shape number drifts by 1-2pp run-to-run, swamping the kernel
  win). The FP8 metric/verify divergence is the same phenomenon
  documented in round-13 (`analysis/_notes/round-13-config-tuning-saturation.md`).
  FP8 retune deferred until a non-config lever is found.

* **No HipKittens kernel changes** (.so untouched). All changes are
  pure Primus-Turbo `config.py` edits.

## Verification artifacts

* `/tmp/sweep_fp8_worst_round21.py`  — coarse 40-cell FP8 sweep, 4 shapes
* `/tmp/verify_fp8_round21.py`        — tight 5-trial verify of FP8 candidates
* `/tmp/verify_fp8_round21_neighbors.py` — 11-cfg neighbor verify of (8,4)
* `/tmp/verify_fp8_round21_bit_eq.py`  — bit-eq + SNR for FP8 GateUP-B4-M4096
* `/tmp/sweep_bf16_worst_round21.py`   — coarse 40-cell BF16 sweep, 8 shapes
* `/tmp/verify_bf16_down_b4_m4096.py`  — Down-B4-M4096 11-cfg neighbor verify
                                          (BOTH steady-state + per-iter-sync)
* `/tmp/verify_bf16_metric_aligned.py` — per-iter-sync 5-shape verify
* `/tmp/verify_bf16_down_b4_m4096_bit_eq.py` — bit-eq for Down-B4-M4096

## What to try next

1. **HipKittens kernel-level work**: round-19 gave +85pp by replacing
   one FLAT-class instruction in `kittens::store<col>`. Round 20 cleaned
   the K-tail / N-tail helpers but the metric impact was within noise.
   The remaining ISA opportunity is in the *load* side of the main
   loop (the K-tail register-direct path B already loads via
   `buffer_load_*`, but the dense main loop's `load<col>` still issues
   `global_load_dwordx2` in a few spots — check disasm with
   `llvm-objdump -d` against `kernel_*_dynamic.cpp`).

2. **Per-call host overhead trim** (FP8 grouped path): the dispatcher's
   `select_default_config` + `hk.grouped_dscale(layout)` lookup + 2x
   `is_contiguous()` + tensor `.empty(out)` per call adds ~2-3 µs of
   pure Python on top of the kernel. For B=4 cases (kernel ≈ 170-300
   µs) that is 1-1.5% of wall — worth a probe even though it's at the
   metric noise floor. See round-11 notes for the prior trim pass.

3. **N=2880 (Down) BN=128 dispatch path** (still pending from round-20
   plan): structural HipKittens kernel change to add a BN=128 grouped
   RCR template variant for the Down N=2880 family. Would convert the
   last N-tile's 25% utilization (64/256) to 50% (64/128). Requires
   adding a 2nd compile-time `BLOCK_SIZE` template instantiation in
   `kernel_bf16_dynamic.cpp` + dispatch logic in Primus-Turbo
   `config.py`. Estimated 1-2 round budget; high risk if grid-doubling
   negates the per-tile efficiency gain.
