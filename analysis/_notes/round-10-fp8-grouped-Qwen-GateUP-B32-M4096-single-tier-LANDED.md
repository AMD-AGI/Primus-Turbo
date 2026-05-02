# Round 10 — FP8 grouped Qwen-GateUP-B32-M4096 single-tier rule (Lever F)

## TL;DR
- Re-tested R7's reverted Qwen-GateUP-M=4096 candidate `(gm=1, xcd=4)`.
- B32-M4096 sub-shape shows a clean +0.93 pp gain at 1.85× spread,
  with winner-min beating default-max in 7/7 trials.
- B16-M4096 sibling improved vs R7 (+0.23 pp → +0.47 pp) but still
  overlaps default distribution (gap = 0.6× spread).
- Added a B32-only rule guarded by `m_total >= 131072` so the clean
  B32 signal lands without over-extending to the noisier B16.
- Bit-equivalent.

## Per-shape result (single-trial metric snapshot, before vs after)
```
                                              before    after    Δ
grpFP8_Qwen3-235B-A22B-GateUP-B32-M4096       1.133  →  1.135   +0.2 pp (noisy)
grpFP8_Qwen3-235B-A22B-GateUP-B16-M4096       1.097  →  1.095   -0.2 pp (untouched, noise)
```
The single-trial metric snapshot under-shows the gain (single-shot
metric script has ±0.5 pp per-shape noise per R7 characterisation).
The clean signal sits in the tight verify below.

## Tight verify (200-iter × 7-trial p20)
At `/tmp/verify_qwen_gateup_b32_m4096_round10.py`:
```
Qwen-GateUP-B32-M4096:
  ( 1, 4)  2461.49 TF  +0.93 pp vs default *winner
                       (spread 0.51 %; min=2459.66, max=2472.15)
  ( 4, 4)  2452.03 TF  +0.54 pp                (spread 0.20 %)
  ( 4, 8)  2438.79 TF  baseline                (spread 0.50 %)
  ( 1, 8)  2374.17 TF  -2.65 pp                (spread 0.42 %)
```
Gap = 22.7 TF (+0.93 pp) at 1.85× spread; winner-min (2459.66 TF)
exceeds default-max (2446.60 TF) by 13.06 TF (+0.54 pp) → clean
separation in 7/7 trials.

Sibling B16-M4096 re-tested at the same probe:
```
Qwen-GateUP-B16-M4096:
  ( 1, 4)  2432.96 TF  +0.47 pp vs default *winner
                       (spread 0.29 %; min=2427.00, max=2434.07)
  ( 4, 4)  2423.18 TF  +0.07 pp
  ( 4, 8)  2421.51 TF  baseline                (spread 0.76 %)
```
B16 winner-min (2427.00 TF) is 0.5 TF BELOW default-max (2427.51 TF)
— distributions still overlap. Stays excluded via `m_total >= 131072`
guard. The +0.47 pp result is an improvement over R7's +0.23 pp but
still below the protocol threshold (gap ≥ 1× spread).

## Bit-equivalence
At `/tmp/verify_qwen_gateup_b32_m4096_correctness_round10.py`:
```
Qwen-GateUP-B32-M4096 FP8: max_abs_diff=0.0  bit_eq=True
```

## Aggregate score noise band
4 post-rule trials: 963 / 961 / 965 / 958 (mean 961.75, range 7).
Pre-rule baseline (this round): 959 (low). R9 ended at 962.
Net effect: aggregate within the characterised noise band (±2-3 from
GPU low-power state oscillation), but per-shape and tight-verify
signals are clean.

## Why land despite noisy aggregate
Same standard as R6/R7/R8/R9 lands: clean per-shape signal in tight
verify + bit-equivalent + structural narrowing of an existing rule
gap (R7 had reverted the M=4096 family rule because B16 was noisy;
this round lands JUST the B32 sub-tier where the signal is clean —
mirrors R8's `DSV3-GateUP-B32-M2048 single-tier` pattern).

## Why (gm=1, xcd=4) wins for tiles_n=12 + tiles_m=16
m_per_group=4096 ⇒ 16 M-tiles per group × 12 N-tiles per group = 192
tile-steps × 32 batches = 6144 tile-steps. NUM_CUS=256 persistent
slots ⇒ 24 wave-steps per slot. tiles_n=12 < tiles_m=16 → the
persistent loop benefits from stretching the N-axis traversal under
each M-tile, which `(gm=1)` does (one M-tile fully consumed before
advancing). xcds=4 picked over xcds=8 because the (1, 4) p20
(2461.49 TF) cleanly beats (1, 8) (2374.17 TF, -2.65 pp) — the
smaller XCD chiplet partition keeps more shaping work inside each
XCD's L2.

## Rule scope check (no collateral)
```
tiles_n == 12  ⇔ N == 3072         uniquely Qwen-GateUP in grouped FP8 metric
tiles_m == 16  ⇔ M_per_group == 4096   excludes M=2048 sibling (R7 rule below)
k       == 4096                    uniquely Qwen-GateUP in grouped FP8 metric
m_total >= 131072                  selects B=32 only (B16-M4096 m_total=65536)
```

## Files touched
- `primus_turbo/pytorch/kernels/hipkitten/config.py`: 1 new rule
  inserted at line 1314 (FP8 RCR branch, immediately above the R7
  Qwen-GateUP-M=2048 rule).
- `analysis/_notes/round-10-fp8-grouped-Qwen-GateUP-B32-M4096-single-tier-LANDED.md` (this file).

## Lever-F status after R10
| family               | precision | M=2048 | M=4096 |
|----------------------|-----------|--------|--------|
| Qwen-Down            | FP8       | default (R6 verified) | R6 ✓ |
| Qwen-Down            | BF16      | R9 ✓ | cube-small (verified) |
| Qwen-GateUP          | FP8       | R7 ✓ | **R10 (B32 only) ✓**, B16=default |
| Qwen-GateUP          | BF16      | default | default |
| DSV3-GateUP          | FP8       | B32 R8 ✓ | R8 ✓ |
| DSV3-GateUP          | BF16      | R10-orig | cube-small |
| DSV3-Down            | FP8       | (R20-67) | (R20-67) |
| DSV3-Down            | BF16      | R10-orig | R10-orig |
| gpt_oss-{Down,GateUP}| FP8/BF16  | R23/R7/R57/R61/R67/R68 (FROZEN ceiling) |

Lever F is now exhausted — every Qwen3 + DSV3 sub-shape has been
audited; gpt_oss is FROZEN. Two remaining at-noise tiers:
- Qwen-GateUP-B16-M4096 FP8 (gap 0.6× spread; would need a +0.2 pp
  more on the candidate or a tighter measurement to land).
- Qwen-GateUP-{B16,B32}-M{2048,4096} BF16 (no clean signal in any
  of the 4 sub-shapes).

## Suggested R11
**Lever E scout (ASM software pipeline)**: now Lever F is exhausted,
the only avenue left for the 7 gpt_oss K=2880 shapes (currently
1.026-1.087 ratio) is architectural. Insert the manual loop unrolls
+ explicit s_waitcnt scheduling around `rcr_8w_load_hoist`'s
`buffer_load_dwordx4 ... offen lds` async path and around the
mfma fences. Microbench gate FIRST (before metric run) to confirm
≥ 5 % per-K-iter throughput before touching production code paths.

If Lever E microbench fails (<3 % gain), accept the 962 ± 3 plateau
and shift to backward-only optimisations (`bench_grouped_gemm_turbo.py
--bwd`) which the metric does not exercise.
