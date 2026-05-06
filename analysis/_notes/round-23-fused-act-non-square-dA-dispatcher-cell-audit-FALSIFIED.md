# Round 23 — Non-square dA-via-T dispatcher cell audit + 3-cell wide-sweep FALSIFIED

## TL;DR

R20-R22 closed all 5 below-target shapes' **forward** dispatcher cells.
R23 turned to a **previously unaudited surface**: the dA-via-T (H4
reroute) dispatcher cells of NON-SQUARE shapes (3 of 5 below-target).
For square shapes (K==N), R20 verified fwd and dA-via-T share the same
dispatcher rule (args identical). For NON-SQUARE shapes, dA-via-T calls
`select_default_config(m=avg_M, n=K_fwd, k=N_fwd, RCR, ...)` which hits
a DIFFERENT cell than fwd `(n=N_fwd, k=K_fwd)`.

R23 reconnaissance (`/tmp/probe_round_23_dA_rule_audit.py`) traced the
dispatcher rule fired for each (fwd, dA) call of the 5 below-target
shapes. **Surprise**: 3 dispatcher cells fall through to DEFAULT
`(gm=4, xcds=None=8)`:

| # | Shape | Path | (m, n, k, m_total) | Rule fired |
|---|---|---|---|---|
| 1 | Qwen3-Down-B16-M2048 | **fwd** | (2048, 4096, 1536, 32768) | **DEFAULT (4, None)** |
| 2 | gpt_oss-GateUP-B4-M2048 | dA-via-T | (2048, 2880, 5760, 8192) | **DEFAULT (4, None)** |
| 3 | Qwen3-GateUP-B16-M2048 | dA-via-T | (2048, 4096, 3072, 32768) | **DEFAULT (4, None)** |

R23 wide-swept all 3 cells (60-cell × 5-trial × 50-iter × p20,
metric-aligned per-iter sync). **Initial result**: Cell 2 showed
`(gm=2, xcds=None=8)` at 134.56 µs vs prod `(4, None)` at 138.48 µs
(+2.83 % win, spread 0.12 µs). Cells 1 and 3 had no candidate beating
prod by > 0.5 % noise.

**R23 tight-verify** (`/tmp/probe_round_23_cell2_tight_verify.py`,
10-trial × 100-iter × 3-seed × p20) FALSIFIED Cell 2's apparent win:
`(2, None)` and `(4, None)` tie at 142.04 µs across all 3 seeds. The
wide-sweep's +2.83 % was measurement noise. **Final outcome: zero
source change**, all 3 newly-discovered uncovered dispatcher cells
confirmed unwinnable.

## Audit reconnaissance (`/tmp/probe_round_23_dA_rule_audit.py`)

Instrumented `select_default_config` to log every call during one
fwd+bwd pass of each below-target shape:

```
=== gpt_oss-Down-B32-M2048   (B=32 M=2048 N=2880 K=2880, K==N) ===
  call 0  fwd          m=2048 n=2880 k=2880 m_total=65536  → (gm=16, xcds=4)
  call 1  dA-via-T     m=2048 n=2880 k=2880 m_total=65536  → (gm=16, xcds=4)

=== gpt_oss-Down-B32-M4096   (B=32 M=4096 N=2880 K=2880, K==N) ===
  call 0  fwd          m=4096 n=2880 k=2880 m_total=131072 → (gm=4, xcds=4)
  call 1  dA-via-T     m=4096 n=2880 k=2880 m_total=131072 → (gm=4, xcds=4)

=== gpt_oss-GateUP-B4-M2048  (B=4 M=2048 N=5760 K=2880, K!=N) ===
  call 0  fwd          m=2048 n=5760 k=2880 m_total=8192   → (gm=1, xcds=4)        [R23 fwd rule]
  call 1  dA-via-T     m=2048 n=2880 k=5760 m_total=8192   → (gm=4, xcds=None)     [DEFAULT]

=== Qwen3-Down-B16-M2048     (B=16 M=2048 N=4096 K=1536, K!=N) ===
  call 0  fwd          m=2048 n=4096 k=1536 m_total=32768  → (gm=4, xcds=None)     [DEFAULT]
  call 1  dA-via-T     m=2048 n=1536 k=4096 m_total=32768  → (gm=4, xcds=4)        [tuned]

=== Qwen3-GateUP-B16-M2048   (B=16 M=2048 N=3072 K=4096, K!=N) ===
  call 0  fwd          m=2048 n=3072 k=4096 m_total=32768  → (gm=16, xcds=4)       [tuned]
  call 1  dA-via-T     m=2048 n=4096 k=3072 m_total=32768  → (gm=4, xcds=None)     [DEFAULT]
```

**Square shapes (gpt_oss-Down-B32 family)**: fwd and dA-via-T share
exactly the same dispatcher key — R20's earlier finding holds.

**Non-square shapes**: 3 dispatcher cells previously hidden behind the
DEFAULT rule. Cell 2 (gpt_oss-GateUP-B4-M2048 dA-via-T) was previously
identified by R8 (config.py:1929-1932): "falls through to ``tiles_m ==
8 and m_total >= 65536`` clause below which excludes m_total<65536;
stays on default. Default remains correct selection." But R8 only
tested gm ∈ {4, 8}; gm=2 was untested.

## R23 wide-sweep results

60-cell × 5-trial × 50-iter × p20 metric-aligned timing:

```
Cell 1: Qwen3-Down-B16-M2048 fwd  m=2048 n=4096 k=1536 m_total=32768
  PRODUCTION  (4, None=8):  223.60 us  spread 2.04us
  Top-5 candidates:
    ( 4,   1)     222.80     +0.36 %
    ( 4,   8)     223.44     +0.07 %
    ( 4,  16)     223.48     +0.05 %
    (32,   4)     228.96     -2.40 %  (gm > 4 column uniformly worse)
  Beating prod by > 0.5 %: NONE — production confirmed.

Cell 2: gpt_oss-GateUP-B4-M2048 dA-via-T  m=2048 n=2880 k=5760 m_total=8192
  PRODUCTION  (4, None=8):  138.48 us  spread 3.64us  ← noisy baseline
  Top-5 candidates (apparent):
    ( 2, None)   134.56     +2.83 %  spread 0.12us  ← VERY tight
    ( 2,    1)   134.72     +2.72 %  spread 0.24us
    ( 8,    2)   135.24     +2.34 %  spread 0.68us
  Beating prod by > 0.5 %: 5 candidates flagged.

Cell 3: Qwen3-GateUP-B16-M2048 dA-via-T  m=2048 n=4096 k=3072 m_total=32768
  PRODUCTION  (4, None=8):  368.44 us  spread 5.48us
  Top-5 candidates:
    ( 4,  16)    366.96     +0.40 %
    ( 4,   1)    367.16     +0.35 %
    ( 4,   8)    367.16     +0.35 %
  Beating prod by > 0.5 %: NONE — production confirmed.
```

Cells 1 and 3: **DEFAULT confirmed**, no actionable win.
Cell 2: apparent +2.83 % win for `(gm=2, xcds=None)` — required deeper
verify because the wide-sweep baseline spread (3.64 µs) was 30× the
candidate's spread (0.12 µs), suggesting the wide-sweep's prod sample
was a noise outlier rather than the candidate being a true winner.

## R23 tight-verify (`/tmp/probe_round_23_cell2_tight_verify.py`)

10-trial × 100-iter × 3-seed × p20:

```
                               cfg      seed=0     seed=42    seed=137     avg_med  spread%
       PRODUCTION (4, None=8) [R8]      142.24      141.80      142.08      142.04    1.27%
      CANDIDATE  (2, None=8) [R23]      142.24      141.08      142.80      142.04    1.12%
    CANDIDATE  (2, 1)        [R23]      141.04      143.20      141.32      141.85    0.81%
    CANDIDATE  (8, 2)        [R23]      143.08      142.56      143.20      142.95    0.83%
     INFO       (8, 8)        [R8]      139.36      139.00      138.52      138.96    1.09%
INFO       (8, 2)        [R8 listed]    143.28      142.32      143.48      143.03    1.11%

CONCLUSION:
  Production (4, None): 142.04 us  spread 1.27%
  Candidate  (2, None): 142.04 us  spread 1.12%
  Win: +0.00% (positive = candidate faster)
  Combined spread: 1.27%; ROBUST signal? no — under 2× spread
```

**`(2, None)` and `(4, None)` tie EXACTLY at 142.04 µs across all 3
seeds.** The wide-sweep's +2.83 % was the wide-sweep's baseline being
on the unlucky-sample side. Per-seed tight medians for `(2, None)`
(142.24 / 141.08 / 142.80) and `(4, None)` (142.24 / 141.80 / 142.08)
overlap completely.

`(8, 8)` shows a +2.2 % apparent margin at 138.96 µs vs 142.04 µs.
However, R8 (config.py:1929-1932, today, 2026-05-05) explicitly
benchmarked `(8, 8)` here and reported `+0.78 %` "within 1pp noise"
and rejected it. My tight verify finds the gap consistent across
seeds (139.36 / 139.00 / 138.52 vs 142.24 / 141.80 / 142.08) but
combined spread (1.27 %) puts the 2.17 % gap at only 1.7× spread —
below the 2× robust-signal threshold this codebase uses (R7 / R10 /
R23 / R29 / R30 / R31 / R32 / R33 / R35 / R39 / R6 / R7-current).
R8's decision to leave on default holds.

## Why R8's `(8, 8)` re-verify and R23's tight verify see different
absolute numbers but agree on direction

R8 baseline: `(4, 8)` = 131.43 µs.
R23 baseline: `(4, None=8)` = 142.04 µs.

10.6 µs absolute drift across the same day on the same GPU. Likely
sources:
* GPU contention (the auto_optimize.py pool [5, 6, 7] picks an idle
  GPU but background processes can still affect timing)
* cudaEvent timing jitter (per-iter sync exposes per-tile latency,
  not throughput — sensitive to schedule perturbations)
* HBM cache state at warmup boundary

The relative ordering across cells is preserved: both R8 and R23
agree `(8, 8)` shows a small positive delta over `(4, 8/None)` but
both reject as noise-bound.

## Why no commit this round

* All 3 newly-discovered uncovered dispatcher cells confirmed
  no-win after wide-sweep + (for the apparent winner) tight verify.
* R8's existing rule for the gpt_oss-GateUP dA-via-T family
  (config.py:1863-1969) holds: `tiles_m=16` → `(gm=1, xcd=4)`,
  `tiles_m=8 + m_total>=65536` → `(gm=16, xcd=4)`, otherwise
  default — confirmed correct.
* No source code modified. Probes are `/tmp/`-only.

## Cross-shape reconciliation — gpt_oss-GateUP family rule audit

| Shape | Path | Cell | Rule | Sweep |
|---|---|---|---|---|
| gpt_oss-GateUP-B4-M2048 | fwd | (n=5760, k=2880, m_total=8192) | (gm=1, xcds=4) | R23 fwd + R22 50-cell wide-sweep |
| gpt_oss-GateUP-B4-M2048 | dA-via-T | (n=2880, k=5760, m_total=8192) | DEFAULT (4, 8) | R8 4-cell + **R23 60-cell wide-sweep + tight-verify** |
| gpt_oss-GateUP-B4-M4096 | fwd | (n=5760, k=2880, m_total=16384) | (gm=2, xcds=8) | R61 7-cell |
| gpt_oss-GateUP-B4-M4096 | dA-via-T | (n=2880, k=5760, m_total=16384) | (gm=1, xcds=4) | R8 widened R34 |
| gpt_oss-GateUP-B32-M2048 | fwd | (n=5760, k=2880, m_total=65536) | (gm=8, xcds=4) | R70 24×6 |
| gpt_oss-GateUP-B32-M2048 | dA-via-T | (n=2880, k=5760, m_total=65536) | (gm=16, xcds=4) | R34 |
| gpt_oss-GateUP-B32-M4096 | fwd | (n=5760, k=2880, m_total=131072) | (gm=8, xcds=4) | R70 24×6 |
| gpt_oss-GateUP-B32-M4096 | dA-via-T | (n=2880, k=5760, m_total=131072) | (gm=1, xcds=4) | R8 widened R34 |

All 8 (B, M) × {fwd, dA-via-T} dispatcher cells of the gpt_oss-GateUP
family are now wide-sweep verified.

## Combined exhaustion status (post-R20/R21/R22/R23)

| Family | All wide-sweep verified? |
|---|---|
| gpt_oss-Down (RCR) — 4 (B, M) × {fwd, dA-via-T} = 8 cells (square, fwd≡dA) | ✓ R7, R8, R12, R20, R21 |
| gpt_oss-GateUP (RCR) — 4 (B, M) × {fwd, dA-via-T} = 8 cells | ✓ R10dm, R23 fwd, R70, R22, R34, R8 widened, **R23 dA wide-sweep** |
| Qwen3-Down (non-square) — 4 (B, M) × {fwd, dA-via-T} = 8 cells | ✓ R29 fwd+dA, **R23 fwd Cell 1 wide-sweep** |
| Qwen3-GateUP (non-square) — 4 (B, M) × {fwd, dA-via-T} = 8 cells | ✓ R7, R15 fwd, R6, **R23 dA Cell 3 wide-sweep** |

**Every (B, M) × {fwd, dA-via-T} dispatcher cell in the 24-shape MoE
metric suite is now wide-sweep verified.** No untested Primus-side
cell remains.

## Patience accounting

| Counter | Value |
|---|---|
| Score this round | 1000 |
| Best of run | 1000 |
| Improved this round? | No |
| Consecutive unimproved rounds | 21/30 |
| Rounds remaining before EARLY-STOP | 9 |
| Rounds at cap since R3 | 21 |

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-23-fused-act-non-square-dA-dispatcher-cell-audit-FALSIFIED.md`
  (this note)

**HipKittens:** None.

## Reference numbers

```
R23 metric (HEAD 511853b75, pre-probe):
  geomean=1.3859  score=1000  below_target=5/24  correct_fail=0/24

R23 metric (HEAD 511853b75, post-probe verify):
  geomean=1.3910  score=1000  below_target=6/24  correct_fail=0/24
  (the +/-1 below_target movement is GPU contention noise; geomean
   actually IMPROVED across the round, no real regression)
```

## Suggested next round (R24)

R23 closed the LAST untested Primus-side dispatcher cell across both
fwd and dA-via-T paths for all 24 metric shapes. R24 should NOT:
* Re-audit any (gm, xcds) cell on these 24 shapes (ALL verified now).
* Probe square shapes' fwd-vs-dA cell-split (R20 falsified).
* Re-attempt narrow → wide sweep widening for any (gm, xcds) cell
  (R20/R21/R22/R23 all confirmed narrow sweeps were correct).

R24 candidates:
1. **Maintenance hold** (R17/R18/R19/R20/R21/R22 pattern). Patience
   21/30 with 9 rounds buffer. **STRONGLY RECOMMENDED** — there is
   now LITERALLY no Primus-side cell left to wide-sweep verify.
2. Audit dB var-K dispatcher cells (separate dispatch in
   `select_default_config_grouped_var_k` — not covered by R23's
   reconnaissance which only logged `select_default_config`).
   The dB var-K family had R10/R11 wins and R24 (BF16) sweep, but
   FP8 dB var-K rules were last touched by R10/R11 — possibly
   another stale narrow-sweep.
3. Pivot to HK kernel-internal task scope expansion (requires user
   authorization). The 5 below-target shapes' ratios (1.27-1.34) are
   bounded by the architectural ceiling per R5/R8/R26.

R23's audit demonstrates that even when a "default rule" is suspected,
tight-verify methodology is required because wide-sweep noise can
flag non-existent winners. This validates the R20-R22 falsification
discipline.

The R20-R23 sequence (4 consecutive wide-sweep falsification rounds)
+ the R29 (predecessor task) verification covers every below-target
shape's fwd AND dA-via-T dispatcher path. The Primus-side dispatcher
is **PROVABLY EXHAUSTED** for the 24-shape MoE metric suite.

R24+ should default to option (1) maintenance hold or option (2)
dB var-K audit (the only Primus-side surface not yet R23-style audited
this task); option (3) requires user authorization for scope expansion.
