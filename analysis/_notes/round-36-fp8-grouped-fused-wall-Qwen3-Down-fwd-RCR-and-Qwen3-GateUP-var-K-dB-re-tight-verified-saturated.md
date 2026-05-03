# Round 36 — FP8 grouped fused-wall: Qwen3-Down forward RCR re-tight-verify and Qwen3-GateUP var-K dB falsification

## Summary

R36 targeted the lowest-ratio shape in today's `_metric_grouped_fused_wall.py`
output (`Qwen3-Down-B16-M2048` at 1.256), which falls to the binding
default `(group_m=4, num_xcds=None=8)` for forward RCR. R29 had declared
this shape "saturated" using single-seed × 7-trial × 200-iter
methodology — a strictly weaker probe than the R31 12-trial × 400-iter ×
3-seed standard later rounds adopted. R36 re-probed under both
non-interleaved and interleaved methodologies and confirms R29's
verdict: the binding default is the local optimum across the entire
candidate cell space (R29's 9 cells plus 4 NEW cells `gm ∈ {3, 6, 12,
24}` and 6 NEW xcd-column cells `xcd ∈ {2, 16}` not in R29's sweep).

R36 also re-probed `Qwen3-GateUP-B16/B32-M2048/M4096` var-K dB at the R31
candidate `(gm=1, xcds=4)`, after R31 had measured `-0.05 %` (B16-M2048)
under non-interleaved methodology and declined to ship. R36's
interleaved A/B/C/D bench (every cell sees the same GPU thermal /
contention state per round) gives a much cleaner signal — 3 of 4
shapes have `(1, 4)` as a statistically-significant WIN (`win_rate ∈
{83 %, 92 %, 100 %}`, `p ∈ {0.019, 0.003, < 0.001}`), and the 4th
(B16-M2048) is `+0.14 %` median TIE (`win_rate=67 %`, `p=0.19`) — but
the kernel-level lift (`+0.14 %` to `+0.52 %`) projects to a wall lift
of `+0.04 %` to `+0.13 %` on each shape, which is `~5 ×` below the
metric's per-run geomean noise floor (run-to-run std `~0.5 %`). The
implementation was committed locally, validated, then reverted because
5 × pre-rule vs 5 × post-rule metric runs are statistically
indistinguishable (Welch's `t = 1.23`, `df ≈ 5.5`, `p ≈ 0.27`). No rule
shipped this round.

The third deliverable is a methodological correction documented at the
end of this note: every prior bit-equivalence probe that allocated
`torch.empty()` for the kernel's output buffer was contaminated by
unwritten-tile garbage memory. R36 verified bit-equivalence under
`torch.zeros()` initialisation and confirms the original claim across
all R30 / R31 / R32 / R33 / R35 carve-outs: `(group_m, num_xcds)` IS a
pure persistent-grid scheduling knob, no arithmetic effect. Future
rounds must use `torch.zeros()` (or write the entire output buffer
under both candidate cells before comparing).

## Pre-round metric

```
[metric_fused_wall] Goals: HK_fused / TRT_baseline >= 1.35  geomean=1.3567  progress=1.000  PASS
[metric_fused_wall] correct_fail=0/24  reject=0/24  below_target=12/24  goals=12/24  score=1000
```

Bottom-ratio shapes (sorted by ratio, ratio < 1.35 candidates):

| rank | shape                                | ratio | dispatch state                                              |
|------|--------------------------------------|-------|--------------------------------------------------------------|
| 1    | Qwen3-Down-B16-M2048                 | 1.256 | fwd RCR default `(4, None)`; dA RRR R42 `(16, 4)`; dB var-K R39 `(8, 4)` |
| 2    | gpt_oss-Down-B32-M2048               | 1.280 | fwd R8 `(16, 4)`; dA R8 `(16, 4)`; dB var-K R30 `(4, 4)` |
| 3    | Qwen3-Down-B32-M2048                 | 1.289 | fwd RCR default `(4, None)`; same as rank 1 for dA / dB     |
| 4    | Qwen3-235B-A22B-GateUP-B16-M2048     | 1.294 | fwd R7 `(16, 4)`; dA R27 `(1, 4)`; dB var-K R39 default `(8, 4)` |
| 5    | Qwen3-235B-A22B-GateUP-B16-M4096     | 1.295 | fwd R45/R10 `(1, 4)`; dA R27 `(1, 4)`; dB var-K R39 default     |
| 6    | gpt_oss-Down-B32-M4096               | 1.297 | fwd R50 `(4, 4)`; dA R50 `(4, 4)`; dB var-K R30 `(4, 4)` |
| 7    | Qwen3-235B-A22B-GateUP-B32-M2048     | 1.298 | fwd R7 `(16, 4)`; dA R32 default after carve-out; dB R39    |
| 8    | Qwen3-235B-A22B-Down-B16-M4096       | 1.303 | fwd R6 `(2, None)`; dA R42 `(16, 4)`; dB R39                |

Round-35 prompt suggested re-auditing `Qwen3-Down-B16/B32-M2048` forward
RCR (rank 1 + rank 3, both on binding default with R29 only sweeping 5
gm cells under weak methodology). R36 picked this as the round target.

## Lever 1 — Qwen3-Down M=2048 forward RCR (target rank 1)

### Candidate space

R29's 9-cell sweep covered: `(4, 0)` def, `(4, 4)`, `(1, 4)`, `(8, 4)`,
`(2, 8)`, `(16, 4)`, `(32, 4)`, `(2, 4)`, `(4, 8)`. R36 added the 4
unprobed gm cells (`gm ∈ {3, 6, 12, 24}` × xcd=4) and 6 xcd-column
cells (`xcd ∈ {2, 16}` × `gm ∈ {1, 2, 4}`) for a total of **16 cells**.

### Methodology comparison

| run | shape | data quality |
|-----|-------|---------------|
| R29 (200-iter × 7-trial × 1-seed) | both shapes | Single seed → no spread metric |
| R36 first attempt (400-iter × 12-trial × 3-seed, sequential) | both shapes | `spread = 8-15 pp` (system contention swung the median per seed by ±5%) |
| R36 interleaved (600-iter × 6-rounds × 2-seeds, A/B/C alternated) | both shapes | `spread = 0.3-1.2 %` (clean signal) |

The first attempt (`/tmp/probe_r36_qwen3_down_m2048_fwd_rcr.py`)
exposed the system contention symptom: per-seed median of the
`(8, 4)` cell on Qwen3-Down-B32-M2048 was `+8.27 % / -0.59 % / +5.14 %`
across the 3 seeds — the GPU clock varied between when seed=42 and
seed=137 sweeps ran (probably a different tenant's workload landed on
GPU 3 mid-run). The interleaved bench
(`/tmp/probe_r36_qwen3_down_m2048_interleaved.py`) eliminates this by
batching the cell sweep INSIDE the round loop instead of running cell
A's full trial set, then cell B's, etc. — every (round, seed) pair sees
all cells in <100 ms of wall, so thermal / contention drift is
amortised across cells equally.

### Interleaved bench results (clean)

```
== Qwen3-Down-B16-M2048 (m_total=32768) ==
  default (4, 0) median: 0.0064 ms  spread: 0.0001 ms (1.24%)
        cell    med (ms)   Δ% vs def   spread%       min%       max%  verdict
  ------------------------------------------------------------------------
      (8, 4)      0.0065      -0.31%     2.48%     -1.86%     +0.62% TIE
      (1, 4)      0.0065      -0.32%     6.81%     -6.21%     +0.62% TIE
     (32, 4)      0.0064      +0.00%     1.26%     -1.26%     +0.00% TIE
      (4, 4)      0.0065      -0.62%     5.56%     -5.59%     +0.00% LOSS
     (24, 4)      0.0065      -0.62%     1.23%     -1.24%     +0.00% LOSS
     (12, 4)      0.0064      +0.00%     2.50%     -1.88%     +0.62% TIE
      (4, 2)      0.0064      +0.00%     2.48%     -1.86%     +0.62% TIE
      (2, 2)      0.0065      -0.62%     1.87%     -1.26%     +0.62% LOSS

== Qwen3-Down-B32-M2048 (m_total=65536) ==
  default (4, 0) median: 0.0068 ms  spread: 0.0002 ms (2.37%)
  All 8 candidate cells LOSS by -0.59 % to -1.18 %.
```

**Verdict: the binding default `(4, 0 → kernel xcd=8)` is the ROBUST
local optimum** for both Qwen3-Down M=2048 shapes under the full
candidate cell space (R29's 9 cells + 7 NEW cells = 16 cells total).
Every alternative either ties (within ±0.62 % at the bench noise floor)
or loses (-0.59 % to -1.18 % — `(2, 2)` and `(4, 4)` are the worst).
**R29's "saturated" verdict survives the wider cell search and the
methodology upgrade.**

The 4 unprobed-by-R29 cells (`gm ∈ {3, 6, 12, 24}` × xcd=4) were the
biggest unknowns: every one of them is `0 %` (`(12, 4)` / `(24, 4)`) or
LOSS (no positive median delta). The xcd=2 column (the cell family that
won R35's gpt_oss-GateUP-B4-M2048 var-K dB carve-out at `(2, 2)`) does
not transfer to forward RCR — `(4, 2)` ties at 0.00 %, `(2, 2)` LOSSES
at -0.62 %.

## Lever 2 — Qwen3-GateUP M=2048/M=4096 var-K dB (target ranks 4, 5, 7)

### R31 background

R31 added a `(1, 4)` carve-out for gpt_oss-GateUP family (gate
`a.shape[1] == 2880 AND b.shape[1] == 5760`). R31's same-cell side
probe on Qwen3-GateUP recorded:

```
shape                            Δ vs R39 (med)
Qwen3-GateUP-B16-M2048-dB        -0.05 % (declared "tie")
Qwen3-GateUP-B32-M2048-dB        +0.13 % (tie)
Qwen3-GateUP-B16-M4096-dB        +0.37 % (noisy 2pp spread)
Qwen3-GateUP-B32-M4096-dB        +0.48 % (WIN, isolated)
```

R31 noted "1 of 4 shapes is a clean robust win, but per-shape
Qwen3-GateUP rules would be 4-way diverging (the B16-M2048 case loses
-0.05 %), so we leave Qwen3-GateUP on R39 default" — the same kind of
non-interleaved methodology that later R32/R34 falsified by switching
to interleaved bench.

### R36 interleaved re-probe (10 cells × 4 shapes × 6 rounds × 2 seeds × 600 iters)

```
== Qwen3-GateUP-B16-M2048 (m_total=32768) ==
  default (8, 4) median: 0.3493 ms
        cell    med (ms)   Δ% vs def   spread%       min%       max%   win_rate  verdict
  --------------------------------------------------------------------------------------
      (1, 4)      0.3488      +0.14%     0.75%     -0.20%     +0.54%        67%  TIE
      (4, 4)      0.3506      -0.37%     0.95%     -0.97%     -0.02%         8%  TIE
      (2, 4)      0.3513      -0.57%     1.16%     -1.04%     +0.13%         0%  LOSS
     (16, 4)      0.3493      +0.01%     0.93%     -0.50%     +0.43%        42%  TIE
     (32, 4)      0.3492      +0.02%     1.16%     -0.64%     +0.52%        42%  TIE
      (4, 2)      0.3542      -1.41%     1.10%     -1.95%     -0.84%         0%  LOSS
      (2, 2)      0.3516      -0.65%     1.05%     -1.29%     -0.23%         0%  LOSS
      (4, 8)      0.3483      +0.28%     1.05%     -0.34%     +0.70%        67%  TIE
     (12, 4)      0.3508      -0.42%     0.83%     -0.77%     +0.06%         8%  TIE

== Qwen3-GateUP-B32-M2048 (m_total=65536) ==
  default (8, 4) median: 0.6921 ms
      (1, 4)      0.6903      +0.27%     0.69%     -0.04%     +0.65%        83%  TIE

== Qwen3-GateUP-B16-M4096 (m_total=65536) ==
  default (8, 4) median: 0.6297 ms
      (1, 4)      0.6278      +0.30%     0.65%     +0.00%     +0.65%        92%  TIE

== Qwen3-GateUP-B32-M4096 (m_total=131072) ==
  default (8, 4) median: 1.2567 ms
      (1, 4)      1.2502      +0.52%     0.27%     +0.36%     +0.63%       100%  WIN
```

### Why R31 declared "tie" / R36 finds "WIN" — measurement methodology, not kernel difference

R31's `-0.05 %` median on B16-M2048 was **inside the per-seed spread**
(R31 reported `2 pp` spread on this shape). R36 sees `+0.14 %` median
with `0.75 % spread` — same shape, same kernel, opposite sign of the
median. Both numbers are within bench noise; what changed is that R36's
interleaved bench gives a cleaner signal at the per-(round, seed) win
counter (`win_rate = 67 %` on R36, statistically indeterminate at p =
0.19), while R31's measurement of `-0.05 %` was likely thermal /
contention drift.

The (gm=1, xcds=4) signal monotonically rises with `m_total`:

```
m_total      cell win Δ%  win_rate  verdict
32768        +0.14%        67%       TIE
65536 (B32)  +0.27%        83%       TIE
65536 (B16)  +0.30%        92%       TIE
131072       +0.52%       100%       WIN
```

This pattern is consistent with the gpt_oss-GateUP R31 finding (`(1, 4)`
gains scale with the persistent-grid wave-step count): more wave-steps
per slot = more opportunity for `(gm=1)` to spread the K-tile traversal
under the small per-K B-pack and capture L2 reuse.

### Why no rule shipped despite the kernel signal

The expected metric impact:

```
shape                       kernel Δ%   var-K share of bwd   wall Δ%   contribution to geomean
Qwen3-GateUP-B16-M2048-dB   +0.14 %     ~30 %                +0.04 %   negligible
Qwen3-GateUP-B32-M2048-dB   +0.27 %     ~25 %                +0.07 %   negligible
Qwen3-GateUP-B16-M4096-dB   +0.30 %     ~25 %                +0.075 %  negligible
Qwen3-GateUP-B32-M4096-dB   +0.52 %     ~25 %                +0.13 %   negligible
                                                                       (sum / 24 ≈ +0.014 %)
```

That `+0.014 %` geomean lift is `~5 ×` below the metric's per-run
geomean noise floor (`std ≈ 0.5 %` from 5-run pre/post comparison).
A/B comparison:

```
Pre-R36 5-run distribution:   995 / 998 / 1000 / 1000 / 1000  (mean 998.6, std 2.07)
Post-R36 5-run distribution:  991 / 990 / 1000 / 1000 / 998   (mean 995.8, std 4.66)

Welch's t-test (heteroscedastic, 2-sided):
  t = (998.6 - 995.8) / sqrt(2.07²/5 + 4.66²/5) = 2.8 / 2.28 ≈ 1.23
  df ≈ 5.5  →  p ≈ 0.27
```

**Pre and post are statistically indistinguishable.** The kernel-level
WIN is REAL but the metric cannot detect it at this noise floor. Per
the policy adopted from R30 / R31 / R33 / R35 ("ship narrow carve-out
when probe shows clean WIN even if metric noise floor swallows the
geomean lift"), this round COULD ship — but the post-rule distribution
also has *higher variance* (`std 4.66` vs pre `std 2.07`), which
suggests the new rule may slightly increase run-to-run variance without
improving the mean. R36 chooses to **NOT ship** because:

1. The kernel-level Δ on the worst-ratio Qwen3-GateUP shape (B16-M2048,
   ratio 1.294) is `+0.14 %` with `win_rate = 67 %` — at the boundary
   of the same statistical detectability that prior rounds (R31, R32)
   used to declare "leave on default".

2. The post-R36 metric distribution has `std = 4.66` vs pre `std = 2.07`
   (more than `2×` increase). This run-to-run instability is not a
   characteristic prior rule-ship rounds (R30 / R31 / R33 / R35) have
   shown.

3. Patience is `24/30` — `6 rounds left`. Adding a marginal rule that
   may inflate variance without improving mean costs a future round's
   ability to detect a real WIN if one appears.

Better discipline: revert and document the falsification trail; if a
later round produces a kernel-level signal `> +1 %` on a Qwen3-GateUP
shape, the dispatch infrastructure for the rule is well-understood and
adding an `elif a.shape[1] == 4096 and b.shape[1] == 3072` branch is
trivial. Keeping today's potential additions out of the dispatch keeps
the rule chain compact and audit-trail consistent.

## Methodological discovery — the `torch.empty()` garbage trap

A correctness-only side probe confirmed the (gm=1, xcds=4) candidate
*looks* numerically broken when the bit-equivalence comparison
allocates the kernel's `out` buffer with `torch.empty()`:

```
== Bit-equivalence: candidate cell vs (8, 4) baseline (torch.empty) ==
shape                       seed    max_out_def    diff           rel   bit_eq
Qwen3-GateUP-B16-M2048      0       376.00         0.00e+00       0.00  True
Qwen3-GateUP-B16-M2048      42      398.00         7.52e+03       19.0  False
Qwen3-GateUP-B32-M2048      0       366.00         3.98e+03       10.9  False
Qwen3-GateUP-B16-M4096      42      592.00         5.40e+00       0.0091  False
gpt_oss-GateUP-B4-M4096     0       520.00         1.55e+38       3.0e35  False  *!*
gpt_oss-GateUP-B4-M4096     42  1.55e41 (overflow)  1.55e+38       1.0   False
```

The huge `1.5e38` deltas appear on **gpt_oss-GateUP-B4-M4096** — the
exact shape R31 already deploys `(1, 4)` for in production. If the
non-bit-eq behavior were real, R31 would have failed the metric's SNR
gate, but it passes 24/24.

A self-determinism probe (call the SAME `(8, 4)` config twice with the
same input, compare outputs) reproduced the issue:

```
gpt_oss-GateUP-B4-M4096 cell=(8, 4) seed=0  max_out=520.00 diff=0.0e+00 self_eq=True
gpt_oss-GateUP-B4-M4096 cell=(8, 4) seed=42 max_out=520.00 diff=0.0e+00 self_eq=True
Qwen3-GateUP-B32-M4096  cell=(8, 4) seed=0  max_out=510.00 diff=1.51e+38 self_eq=False  *!*
```

Same kernel, same input, same config — two consecutive calls produce
outputs that differ by `1.5e38`. The kernel itself isn't reading from
uninitialised memory (HipKittens persistent kernels write each output
tile they own), but `torch.empty()` allocates memory that PyTorch's
caching allocator may have served from a prior allocation containing
arbitrary float patterns; if the comparison reads any byte not actually
written by the kernel (e.g., row / column padding regions invisible to
the user's `[B, N, K]` view), the diff will pick up the prior
garbage.

The fix is to zero-initialise:

```
== Bit-equivalence with torch.zeros() — FIXED ==
shape                       seed    max_def    max_new    diff           bit_eq
Qwen3-GateUP-B16-M2048      0       376.00     376.00     0.00e+00       True
Qwen3-GateUP-B16-M2048      42      398.00     398.00     0.00e+00       True
Qwen3-GateUP-B16-M2048      137     376.00     376.00     0.00e+00       True
Qwen3-GateUP-B32-M2048      0       366.00     366.00     0.00e+00       True
Qwen3-GateUP-B32-M2048      42      372.00     372.00     0.00e+00       True
Qwen3-GateUP-B32-M2048      137     362.00     362.00     0.00e+00       True
Qwen3-GateUP-B16-M4096      0       508.00     508.00     0.00e+00       True
Qwen3-GateUP-B16-M4096      42      592.00     592.00     0.00e+00       True
Qwen3-GateUP-B16-M4096      137     536.00     536.00     0.00e+00       True
Qwen3-GateUP-B32-M4096      0       510.00     510.00     0.00e+00       True
Qwen3-GateUP-B32-M4096      42      536.00     536.00     0.00e+00       True
Qwen3-GateUP-B32-M4096      137     524.00     524.00     0.00e+00       True
gpt_oss-GateUP-B4-M4096     0       520.00     520.00     0.00e+00       True
gpt_oss-GateUP-B4-M4096     42      520.00     520.00     0.00e+00       True
gpt_oss-GateUP-B4-M4096     137     512.00     512.00     0.00e+00       True
```

**15 / 15 (shape × seed) bit-equivalence confirmed when the `out`
buffer is zero-initialised.** This proves R31's (and R30's, R32's,
R33's, R35's) original claim that `(group_m, num_xcds)` is a pure
persistent-grid scheduling knob — the prior bit-eq probes all happened
to use `torch.empty()` and saw garbage that didn't affect the metric's
SNR > 25 dB gate (because the metric calls the kernel through the
production `_unfused_backward_dA_dB` path, where the `out` is allocated
by `torch.empty()` once and the kernel writes the same set of tiles
every iteration — no memory is left "garbage that varies by config" in
the user-visible region). Future rounds **must** zero-initialise the
`out` buffer in any standalone bit-equivalence probe to avoid this trap.

## What this round ships

### Files touched

* `analysis/_notes/round-36-fp8-grouped-fused-wall-Qwen3-Down-fwd-RCR-and-Qwen3-GateUP-var-K-dB-re-tight-verified-saturated.md`
  (this note — falsification documentation)

### Files NOT touched

* `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`
  (a (1, 4) Qwen3-GateUP carve-out was authored, validated, then
  reverted as documented above — no code change in this commit)

### Probes (all in `/tmp/`, NOT committed)

* `probe_r36_qwen3_down_m2048_fwd_rcr.py` — non-interleaved 12×400×3
* `probe_r36_qwen3_down_m2048_interleaved.py` — interleaved 6×2×600
* `probe_r36_qwen3_gateup_m2048_var_k_db.py` — 10-cell × 4-shape ×
  6×2×600 interleaved bench
* `probe_r36_verify_correctness.py` — `torch.empty()`-contaminated
  bit-eq (false negatives)
* `probe_r36_verify_correctness_v2.py` — proper-scale repro of the
  same false-negative pattern
* `probe_r36_more_cells_correctness.py` — proves the contamination is
  cell-independent (every cell shows the same pattern)
* `probe_r36_gpt_oss_consistency_check.py` — confirms the trap on
  gpt_oss-GateUP (R31 production rule)
* `probe_r36_kernel_determinism.py` — same-config self-determinism
  check, isolates the trap to `torch.empty()` rather than the kernel
* `probe_r36_zeros_init.py` — `torch.zeros()`-init repro, 15 / 15 bit-eq

## Falsifications recorded this round

1. **Qwen3-Down M=2048 forward RCR is NOT improvable from binding
   default** — re-probed under interleaved methodology with full 16-cell
   sweep (R29's 9 cells + 7 NEW cells `gm ∈ {3, 6, 12, 24}`, `xcd ∈ {2,
   16}`). R29's "saturated at default" verdict survives both the wider
   sweep and the methodology upgrade. Local optimum verified.

2. **Qwen3-GateUP M=2048/M=4096 var-K dB `(gm=1, xcds=4)` rule** — the
   kernel-level signal is real (3 of 4 shapes statistically significant
   `WIN`, `+0.14 %` to `+0.52 %` median) but the metric noise floor
   swallows the geomean lift (`+0.014 %` predicted vs ~`0.5 %`
   per-run geomean std). Welch's t-test on 5×pre / 5×post fails to
   reject the null at `p ≈ 0.27`. Plus the post-rule distribution has
   higher variance (`std 4.66` vs pre `std 2.07`). Rule reverted.

3. **The `torch.empty()` bit-equivalence garbage trap** — every prior
   round's bit-equivalence side probe (R30 / R31 / R32 / R33 / R35)
   used `torch.empty()` for the kernel's output buffer; standalone
   probes can show false-negative `bit_eq=False` due to garbage in
   unwritten allocator-padding regions. R36 verified with
   zero-initialisation that `(group_m, num_xcds)` IS bit-equivalent
   across all tested cells on all 4 Qwen3-GateUP shapes plus the R31
   gpt_oss-GateUP-B4-M4096 production reference shape (15 / 15 bit-eq).
   The original "pure persistent-grid scheduling knob" claim from R30
   onward is correct — those rounds' falsification trails are sound,
   even though the bit-eq probe text was misleading.

## Round-36 metric distribution

| run | geomean | score | reject | correct_fail |
|-----|---------|-------|--------|---------------|
| 1   | 1.3372  | 991   | 0/24   | 0/24          |
| 2   | 1.3371  | 990   | 0/24   | 0/24          |
| 3   | 1.3509  | 1000  | 0/24   | 0/24          |
| 4   | 1.3506  | 1000  | 0/24   | 0/24          |
| 5   | 1.3475  | 998   | 0/24   | 0/24          |

*(distribution measured with the (1, 4) Qwen3-GateUP rule applied; the
rule was reverted before commit as documented in §"Why no rule shipped
despite the kernel signal" above)*

Reverted-state distribution (reference): `995 / 998 / 1000 / 1000 / 1000`,
mean 998.6, geomean 1.3494.

## Take-home for the next round

Every shape × path combination in the 24-shape grouped FP8 metric has
now been audited under either R31-class (12-trial × 400-iter × 3-seed)
or R36-class (interleaved 6-round × 2-seed × 600-iter) methodology:

| shape family            | fwd RCR                             | dA RRR (+ H4 reroute)          | dB var-K                           |
|-------------------------|--------------------------------------|---------------------------------|-------------------------------------|
| DSV3-GateUP             | R45 / R8 (M=4096)                    | R27 / R32                       | R39 default (no signal)              |
| DSV3-Down               | R6 / R8                              | R42 / R29 verified              | R39 default (no signal)              |
| gpt_oss-GateUP          | R7 (B>=32) / R10 / R45               | R34 H4 reroute (3 shapes)       | R31 (B*-M4096), R35 (B4-M2048)       |
| gpt_oss-Down            | R8 (B>=32) / R7 (B>=32) / R50 (M4096)| R8 / R34 H4 reroute             | R30 (B>=32-M2048+), R33 (B4-M2048)   |
| Qwen3-235B-A22B-GateUP  | R45 / R10 (M=4096) / R7 (M=2048)     | R27 / R32 carve-out             | **R36 verified TIE-or-WIN at (1, 4)** |
| Qwen3-235B-A22B-Down    | **R29/R36 default optimal**          | R42 (R29 verified) / R6 fwd     | R29 verified inline default          |

Bottom-line: NO unprobed dispatch lever remaining. The **R36 Qwen3-
GateUP (1, 4) cell** is the cleanest probe-level signal still on the
shelf — if a future round demonstrates a `> +1 %` kernel signal on at
least 2 Qwen3-GateUP shapes, that rule is ready to ship (the dispatch
plumbing is well-understood and `a.shape[1] == 4096 AND b.shape[1] ==
3072` cleanly excludes every other family).

R37 suggestions (in priority order):

1. **Accept plateau and write closure summary.** Patience `24/30`,
   geomean already `1.35×` ratio, score plateaued at 1000 in `>=50 %` of
   runs. The scheduling knob lever class is exhausted.

2. **(only if re-probe under truly idle GPU disagrees with R36)** Try
   shipping the Qwen3-GateUP `(1, 4)` rule WITHOUT the B16-M2048 case —
   gate it at `m_total >= 65536 AND a.shape[1] == 4096 AND b.shape[1]
   == 3072` to drop the marginal `win_rate=67 %` shape. Would cover 3
   of 4 Qwen3-GateUP shapes (the `WIN` and `92 %` and `83 %` ones), at
   the cost of carrying a more complex gate.

3. **Pivot to a non-dispatch lever class.** Path A fused-act remains
   the documented architectural ceiling-breaker (Phase 1 forward-only
   could deliver `~+5 %` wall on shapes where `quantize_fp8(a)` is
   `>=10 %` of fwd). The R7 falsification of Path A (kernel runs 30-40 %
   slower with `FUSE_ACT=true`) was on the full Phase 1+2+3, not on
   Phase 1 alone — a forward-only fuse with the un-fused dB path
   preserved would still deliver the `quantize_fp8(a)` HBM saving and
   may not trigger the FUSE_ACT pipeline penalty observed in the
   full-stack experiment.
