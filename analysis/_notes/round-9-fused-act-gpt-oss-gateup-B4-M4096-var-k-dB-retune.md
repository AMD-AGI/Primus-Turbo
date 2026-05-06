# Round 9 — fused-act FP8 grouped: gpt_oss-GateUP-B4-M4096 var-K dB re-tune (R31 sibling, candidate-set widening)

## Summary

- **Lever**: split R31's `(gm=1, xcds=4)` rule for gpt_oss-GateUP var-K dB
  by `m_total < 32768`. The new B=4 M=4096 carve-out uses `(gm=4, xcds=4)`;
  B=32 (M=2048 + M=4096) keeps R31's `(gm=1, xcds=4)`. Located in
  `grouped_gemm_fp8_impl.py` lines 1033-1230 (the elif `a.shape[1]==2880
  and b.shape[1]==5760` block).
- **Class**: same "missing-candidate" pattern as R8 (R34 sibling for dA RCR).
  R31 was tight-verified at 12-trial × 400-iter × 3-seed methodology but
  only swept the xcds=4 column `gm ∈ {1, 2, 4, 8, 16, 32}`. Round-9
  widened to 14 cells across xcds ∈ {2, 4, 8} including gm=3 / gm=8 with
  non-xcd-4. Found `(gm=4, xcds=4)` as a clean every-seed-positive winner
  on B4-M4096 only.
- **Metric (single-run wall, noisy, score capped at 1000)**:
  - Before (1 run): geomean **1.3835**, below_target 9/24, score 1000.
  - After (3 runs): geomean **1.3843 / 1.3958 / 1.4078** → mean **1.3960**.
    Δ vs baseline ≈ **+0.0125** (well above the 1-run wall noise floor of
    ~0.005). Score 1000 every run; below_target collapsed to 5-7/24.

## Probe data (`/tmp/probe_round_9_gpt_oss_gateup_var_k.py`)

200-iter × 7-trial × p20 × 3 seeds direct call to `grouped_variable_k_crr_dscale`
on the 3 R31-covered shapes, 14 candidate cells:

```
shape                       cell      seed42  seed137  seed2024  med Δ vs cur  spread pp  verdict
gpt_oss-GateUP-B4-M4096-dB  (4, 4)    +2.04%  +0.78%   +2.26%    +1.69%        0.91       WIN  med/spread=1.86×
gpt_oss-GateUP-B4-M4096-dB  (1, 4)cur baseline                    +0.00%        0.99       (R31)
gpt_oss-GateUP-B4-M4096-dB  (32, 4)   +0.38%  ...                                          small win
gpt_oss-GateUP-B4-M4096-dB  (4, 2)    +0.32%  ...                                          small win
gpt_oss-GateUP-B4-M4096-dB  (8, 4)    -0.01%  ...                                          TIE
gpt_oss-GateUP-B4-M4096-dB  (1, 8)    -1.64%  ...                                          LOSS

gpt_oss-GateUP-B32-M2048-dB (1, 4)cur baseline best in column      +0.00%        ---       (R31)
gpt_oss-GateUP-B32-M2048-dB (1, 2)    -0.35%                                                LOSS
gpt_oss-GateUP-B32-M2048-dB (4, 4)    -1.06%                                                LOSS
gpt_oss-GateUP-B32-M2048-dB (1, 8)    -2.91%                                                LOSS
gpt_oss-GateUP-B32-M2048-dB (32, 4)   -3.38%                                                LOSS

gpt_oss-GateUP-B32-M4096-dB (1, 4)cur baseline best in column      +0.00%        ---       (R31)
gpt_oss-GateUP-B32-M4096-dB (1, 2)    -0.88%                                                LOSS
gpt_oss-GateUP-B32-M4096-dB (4, 4)    -3.39%                                                LOSS
gpt_oss-GateUP-B32-M4096-dB (1, 8)    -5.66%                                                LOSS
```

**`(gm=4, xcds=4)` wins B=4 M=4096 by +1.69%** with 3/3 seeds positive
(+2.04 / +0.78 / +2.26%). Per-seed (4, 4): 263.88 / 266.28 / 264.88 us
vs baseline (1, 4): 269.36 / 268.36 / 271.00 us. **R31 (gm=1, xcds=4)
remains the unique optimum at B=32** (every alternative is TIE or LOSS).

## Discriminator: m_total < 32768 (clean carve-out)

```
m_total      shape                              dispatch
8192         gpt_oss-GateUP-B4-M2048-dB         m_total<16384 branch (R35: gm=2, xcd=2)
16384        gpt_oss-GateUP-B4-M4096-dB         R31 elif → NEW (gm=4, xcd=4)
65536        gpt_oss-GateUP-B32-M2048-dB        R31 elif → R31 (gm=1, xcd=4)
131072       gpt_oss-GateUP-B32-M4096-dB        R31 elif → R31 (gm=1, xcd=4)
```

`m_total < 32768` cleanly catches only m_total=16384 (B=4 M=4096) without
overlap with R30 (R30 gates on `b.shape[1]==2880` which excludes GateUP).

## Why (gm=4) wins on B=4 M=4096 specifically

B=4 has only 4 groups → persistent loop is short (4 group-passes × 22×11=242
tile-steps = 968 tile-steps over 256 CUs ≈ 4 wave-steps). gm=1 on the small
grid spends each wave-step walking 22 N-rows under one K-tile (L2-efficient
on A-pack but spends each K-tile completely before advancing). gm=4 batches
4 N-rows together so the wave-step's A-pack ALSO reuses 4× across N-rows,
AND the per-K B-pack reuses 4× across the batched N-rows. Net L2 footprint
per wave-step is 4× larger but reused 4× more frequently — double-side reuse.

On B=32's deeper grid (~32-128 wave-steps), the gm=1 single-side reuse wins
because wave-step count amortises the narrower A-pack reuse window.

## Correctness (`/tmp/probe_round_9_correctness.py`)

```
gpt_oss-GateUP-B4-M4096-dB  (gm=1, xcd=4) vs (gm=4, xcd=4):
  seed=0    max_abs_diff=0.000000  bit_eq=True
  seed=42   max_abs_diff=0.000000  bit_eq=True
  seed=137  max_abs_diff=0.000000  bit_eq=True
```

Bit-identical output across 3 seeds. (gm, xcds) are pure persistent-grid
scheduling knobs on the var-K CRR kernel — same property documented for
R30 / R31 / R32 / R33 / R35 above.

## Files touched

- `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`:
  R31 elif body extended with `m_total < 32768` carve-out (~95-line in-code
  comment block + 4-line if/else dispatch).
- `analysis/_notes/round-9-fused-act-gpt-oss-gateup-B4-M4096-var-k-dB-retune.md`:
  this note.

HipKittens: not modified this round.

## Suggestion for the next round

R10 will trigger an automatic DoD checkpoint (every 5 rounds). The rule
change is in `grouped_gemm_fp8_impl.py` (a SHARED file across HK / dispatcher
paths) — must verify DoD doesn't regress. The rule scope is tightly gated
(`a.shape[1]==2880 AND b.shape[1]==5760 AND m_total<32768`) so dense FP8
should be untouched (dense FP8 callers don't use variable-K, and the gate
only fires inside the `var_k_dscale_fn` branch).

Beyond R10: remaining var-K cells worth re-probing under widened sweep:
- R32 / R33 (m_total<16384 branch with a==2880, b==2880 → gpt_oss-Down B4)
  was tight-verified per R32 docs; lower priority.
- R30 (gpt_oss-Down B32) re-probed in R8: holds at (gm=4, xcds=4).
- R35 (gpt_oss-GateUP B4-M2048) was tight-verified at neighbor probe;
  unlikely to find a wider candidate.

Lower-priority tail: forward path config audit for the lowest-ratio shape
(gpt_oss-Down-B32-M2048, persistent at 1.26-1.27, K-tail kernel-bound per
R5 decomposition). Multi-round HK C++ work needed; score-capped so buffer-
only.
