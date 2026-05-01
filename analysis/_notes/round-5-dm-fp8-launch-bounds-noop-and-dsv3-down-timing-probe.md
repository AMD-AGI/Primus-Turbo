# Round 5 — FP8 grouped launch_bounds MIN=2 is no-op + DSV3-Down direct timing probe (mirror)

See `/workspace/code/HipKittens/analysis/_notes/round-5-dm-fp8-launch-bounds-noop-and-dsv3-down-timing-probe.md`
for the full probe table and `-Rpass-analysis=kernel-resource-usage`
resource deltas.

## Baseline (round-5 entry)

`score=821, geomean=0.9851, n=16` (single run; 5-run mean flat @ 816).

## Probes falsified / saturated this round

1. **`__launch_bounds__(_NUM_THREADS, MIN_BLOCKS_PER_CU=2)`** on
   `grouped_rcr_kernel` (was `1`). Resource remarks are bit-identical
   (VGPRs=256, spill 67/76/45/54, occ=2 waves/SIMD). 5-run metric mean
   816.0 (baseline 816.2). **Falsified as no-op.**

2. **Direct HK vs Triton per-call timing on 4 DSV3-Down FP8 shapes**
   (`_probe_dsv3_down_fp8_rocprof.py`, *not* committed, artifact-only):
   confirms a 40–73 µs/call gap on B=16 shapes that is
   **not** explained by group-offs binary search, K-tail, or N-tail
   (all inactive on DSV3-Down). Gap is **main-loop resident**.

## Why this was worth a round (even with no score delta)

Eliminates two candidate hypotheses for the 2–5% DSV3-Down gap:

- **Hypothesis: grouped kernel undershoots dense's register pressure
  budget due to launch_bounds MIN=1 hint.** → refuted; compiler achieves
  occ=2 regardless.
- **Hypothesis: group-offs binary-search or group-boundary handling
  eats per-call time on B=16 shapes.** → refuted; B=16 (few boundaries)
  loses 5%, B=32 (many boundaries) only loses 1–2% or wins.

Remaining candidates for round-6+: **register-tile orientation
(RBM/RBN swap, 1 round)** or **MFMA cell-shape (16x16x128, 2–3 rounds)**.

## Commits

- HipKittens: see note file above + `analysis/_notes/round-5-dm-fp8-launch-bounds-noop-and-dsv3-down-timing-probe.md`
- Primus-Turbo: this note.

## Round-6 plan

Prefer Option A first (register-tile orientation swap, 1-line probe,
immediate signal). Fall back to Option B (MFMA cell-shape 16x16x128
vs 32x32x64) if Option A is flat.
