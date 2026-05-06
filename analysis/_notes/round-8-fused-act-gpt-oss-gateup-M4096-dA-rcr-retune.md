# Round 8 — fused-act FP8 grouped: gpt_oss-GateUP M=4096 dA RCR re-tune (R34 sibling, post-H4-reroute follow-up #3)

## Summary

- **Lever**: replace R34's `(gm=8, xcd=4)` cell with `(gm=1, xcd=4)` for the
  `tiles_n=11 + k=5760 + tiles_m=16` rule (gpt_oss-GateUP M=4096 dA after H4
  reroute, lines 1863-1873 of `primus_turbo/pytorch/kernels/hipkitten/config.py`).
- **Class**: H4-reroute follow-up audit completion. R6 covered Qwen3-GateUP M=4096
  dA RCR; R7 covered Qwen3-Down M=4096 dA RCR; R8 closes gpt_oss-GateUP M=4096 dA
  RCR. The discriminator: R34 (the predecessor configurator) tight-verified at
  12-trial × 200-iter × 3-seed but only swept `gm ∈ {4, 8, 16, 32}` — `gm == 1`
  was not in the candidate set. Round-8 widened the sweep and found a clean
  every-seed-positive winner.
- **Metric (single-run wall, noisy, score capped at 1000)**:
  - Before (1 run): geomean 1.3849, below_target 5/24, score 1000.
  - After (3 runs): geomean 1.3902 / 1.3882 / 1.3827 → mean 1.3870, score 1000
    on every run. Δ vs baseline ≈ +0.002 (sub-noise at the wall scale; clean
    +1.20% / +0.37% kernel-direct signal at the probe scale).

## Probe data (`/tmp/probe_round_8_gpt_oss_gateup_dA_rcr.py`)

200-iter × 7-trial × p20 × 3 seeds direct call to `grouped_rcr_dscale` with
post-H4 inputs: `a=[B*M, k=N_fwd=5760]` (grad_out), `b=[B, n=K_fwd=2880,
k=N_fwd=5760]` (b_T after H4). Tested 9 candidate cells per shape on the
4 gpt_oss-GateUP dA shapes:

```
shape                       cell      seed42  seed137  seed2024  med Δ vs cur  spread pp  verdict
gpt_oss-GateUP-B4-M4096-dA  (1, 4)    +1.16%  +1.24%   +1.20%    +1.20%        0.08       WIN  med/spread=15.0×
gpt_oss-GateUP-B4-M4096-dA  (8, 4)cur baseline                    +0.00%        0.28       (R34)
gpt_oss-GateUP-B4-M4096-dA  (16, 4)   +0.13%  +0.13%   +0.13%    +0.13%        0.30       TIE
gpt_oss-GateUP-B4-M4096-dA  (32, 4)   +0.53%  +0.49%   +0.57%    +0.53%        0.05       small WIN
gpt_oss-GateUP-B4-M4096-dA  (8, 8)    -1.81%  -1.78%   -1.80%    -1.80%        0.06       LOSS

gpt_oss-GateUP-B32-M4096-dA (1, 4)    +0.34%  +0.36%   +0.40%    +0.37%        0.06       WIN  med/spread=6.2×
gpt_oss-GateUP-B32-M4096-dA (8, 4)cur baseline                    +0.00%        0.20       (R34)
gpt_oss-GateUP-B32-M4096-dA (16, 4)   -0.01%  -0.01%   -0.01%    -0.01%        0.06       TIE
gpt_oss-GateUP-B32-M4096-dA (32, 4)   -0.11%  -0.11%   -0.11%    -0.11%        0.06       LOSS
gpt_oss-GateUP-B32-M4096-dA (8, 8)    -1.85%  -1.84%   -1.84%    -1.84%        0.06       LOSS
```

**`(gm=1, xcd=4)` is the unique every-seed-positive winner on both M=4096
shapes** with med/spread 6-15× (well above the standard "median > spread"
robust-signal threshold used by R7 / R10 / R23 / R29 / R30 / R31 / R32 / R33 /
R35 / R39 / R6-current / R7-current). The B4-M4096 win (+1.20%) is the largest
single-cell lift seen in the H4-reroute follow-up audit since R3.

## Sibling shapes (rule MUST keep tiles_m and m_total gating intact)

Re-probed alongside the M=4096 cells:

```
shape                       cell      seed42  seed137  seed2024  med Δ  verdict
gpt_oss-GateUP-B4-M2048-dA  default   baseline                    +0.00% (default-current; rule excludes via tiles_m==8 m_total<65536)
gpt_oss-GateUP-B4-M2048-dA  (1, 4)    -8.97%  -8.97%   -8.97%    -8.97% catastrophic LOSS — must NOT extend
gpt_oss-GateUP-B4-M2048-dA  (8, 8)    +0.78%  +0.78%   +0.78%    +0.78% small win (within noise; default holds)

gpt_oss-GateUP-B32-M2048-dA (16, 4)cur baseline                   +0.00% (R34)
gpt_oss-GateUP-B32-M2048-dA (1, 4)    -0.005% +0.83%   -0.02%    +0.27% INCONSISTENT (1/3 seed WIN; not robust); R34 holds
gpt_oss-GateUP-B32-M2048-dA (32, 4)   -0.04%  -0.04%   -0.04%    -0.05% TIE
```

- **B4-M2048 (m_total=8192, tiles_m=8)**: rule excludes via `tiles_m == 8 and
  m_total >= 65536` clause. Default (4, 8) confirmed best (probed (1, 4)
  catastrophic -8.97%; (8, 8) +0.78% sub-noise). **No rule change**.
- **B32-M2048 (m_total=65536, tiles_m=8)**: hits `tiles_m == 8 and m_total
  >= 65536` clause → R34 (gm=16, xcd=4). Re-probed (1, 4) shows +0.27% mean
  but per-seed inconsistent (1/3 seeds positive). **R34 holds**.

## Why (gm=1) wins for tiles_m=16 + k=5760 deep-K dA

Parallel to R23 (gm=1 for gpt_oss-GateUP-B4-M2048 *fwd* at k=2880 deep-K,
tiles_m=8). The persistent loop with gm=1 walks the entire 11-row N-axis
(= 11 K-tile reads of the per-K B-pack on RCR layout) under each individual
M-tile before advancing M, maximising B-pack L2 reuse. For the deep-K=5760
main loop (45 K-tiles vs k=2880's 23 K-tiles for fwd RCR), the per-iteration
B-pack L2 footprint is the binding cost; gm=1 lets each M-tile consume
11×45 = 495 K-tile-loads of the same B-pack column slab before evicting.

R34's gm=8 batches 8 M-tiles together, which is L2-efficient on the A-pack
axis (M-tile slabs reused across 8 N-tile rows = 88 K-tile-loads each) but
at deep-K the B-pack L2 pressure dominates because the K-tile is read 8×
per M-tile-batch step; gm=1 inverts that ratio. R34's coarse sweep simply
omitted gm=1 from the candidate set.

xcd=4 unchanged from R34 — same chiplet-balance reason documented for R8 /
R12 / R23 / R34 / R50 / R69 (4 XCDs split tiles_n=11 cleanly; xcds=8 over-
distributes for deep-k).

## Correctness (`/tmp/probe_round_8_correctness.py`)

```
gpt_oss-GateUP-B4-M4096-dA  (gm=8, xcd=4) vs (gm=1, xcd=4):
  seed=0    max_abs_diff=0.000000  bit_eq=True
  seed=42   max_abs_diff=0.000000  bit_eq=True
  seed=137  max_abs_diff=0.000000  bit_eq=True
gpt_oss-GateUP-B32-M4096-dA (gm=8, xcd=4) vs (gm=1, xcd=4):
  seed=0    max_abs_diff=0.000000  bit_eq=True
  seed=42   max_abs_diff=0.000000  bit_eq=True
  seed=137  max_abs_diff=0.000000  bit_eq=True
```

Bit-identical output across both shapes × 3 seeds. (gm, xcds) are pure
persistent-grid scheduling knobs (same property documented for every
(gm, xcds) RCR / RRR rule in `config.py`).

## H4-reroute audit completion (R6 + R7 + R8 closure)

After this round, every dA RCR family in the 24-shape MoE suite has been
re-verified post-R3 H4 reroute extension:

| Family                           | tiles_n | k    | tiles_m  | Cell              | Source     |
|----------------------------------|---------|------|----------|-------------------|------------|
| DSV3-Down dA                     | 8       | 7168 | {8, 16}  | (gm=4, xcd=4)     | R4-current |
| DSV3-GateUP dA                   | 28      | 4096 | {8, 16}  | (gm=32, xcd=2)    | R7-current audit (R20/R58/R67 RRR rule still optimal post-H4) |
| Qwen3-Down M=2048 dA             | 6       | 4096 | 8        | (gm=4, xcd=4)     | R4-current |
| Qwen3-Down M=4096 dA             | 6       | 4096 | 16       | (gm=8, xcd=4)     | R7-current |
| Qwen3-GateUP M=2048 dA           | 16      | 3072 | 8        | default (gm=4, xcd=8) | R6-current verified |
| Qwen3-GateUP M=4096 dA           | 16      | 3072 | 16       | (gm=2, xcd=None=8) | R6-current |
| gpt_oss-Down all dA              | 11      | 2880 | {8, 16}  | shares fwd rule (R7/R8/R12/R50, n==k same as fwd) | (no separate rule needed) |
| gpt_oss-GateUP B4-M2048 dA       | 11      | 5760 | 8        | default (gm=4, xcd=8) | R34 verified |
| gpt_oss-GateUP B32-M2048 dA      | 11      | 5760 | 8        | (gm=16, xcd=4)    | R34 (R8 re-verified) |
| **gpt_oss-GateUP M=4096 dA**     | **11**  | **5760** | **16** | **(gm=1, xcd=4)** | **R8-current (this round)** |

**The R6+R7+R8 audit chain is now complete** — no remaining un-audited dA RCR
family in the 24-shape MoE suite. Future drift would only come from new
HipKittens builds touching the RCR persistent kernel scheduling internals
(unlikely; no pending HK changes in this Primus run's dependency graph).

## Files touched

- `primus_turbo/pytorch/kernels/hipkitten/config.py`: 1 rule arm modified
  (gpt_oss-GateUP M=4096 dA, lines 1863-1873). Detailed in-code comment block
  documenting the R8 re-tune, sibling-shape audit, kernel-physical reasoning,
  and bit-equivalence verification (~90 lines).
- `analysis/_notes/round-8-fused-act-gpt-oss-gateup-M4096-dA-rcr-retune.md`:
  this note.

HipKittens: not modified this round.

## Suggestion for the next round

Per the H4 audit completion table above, all dA RCR rules are now tight-
verified at current methodology. The remaining lever space:

1. **dB var-K path re-verify**: R8 re-probed gpt_oss-Down-B32 (R30 (gm=4,
   xcd=4) holds — (8, 4) +0.52%/-0.22% mixed, not robust). Could extend
   audit to R39 / R31 / R33 / R35 cells, but expected returns small.
2. **HK kernel-side K-tail epilog work** for gpt_oss-Down (the persistent
   lowest-ratio shape at 1.27-1.28). Multi-round, score-capped, buffer-only.
3. **Maintenance phase**: docs round documenting Python-overhead floor +
   H4-reroute audit completion + remaining kernel-internal headroom.

R9 recommendation: dB var-K cells for the LCV-density families (gpt_oss-
GateUP B4 var-K is on R31's (gm=1, xcd=4) — re-probe to see if a wider
sweep finds a better cell). Same R34 / R8 methodology fits.
