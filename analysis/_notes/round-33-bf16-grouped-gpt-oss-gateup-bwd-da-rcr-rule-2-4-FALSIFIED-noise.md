# Round-33 — BF16 grouped GEMM: gpt_oss-GateUP post-H4 bwd dA (gm=2, xcds=4) FALSIFIED

## Status
**FALSIFIED** — paired 5-run delta = **-2.6 score** (treatment 881.0 vs
baseline 883.6, stdev ~2.2). Revert. 4 consecutive rounds of dispatch-
rule attempts (R29 5-lever aggregate, R30 R28-repro, R31 H4 gate, R32
(1,4), R33 (2,4)) all sub-noise negative — BF16 dispatch surface is
confirmed exhausted at this precision.

## Metric before / after
- HEAD `52820fd` single run: score = 879/1000 (lowest-progress
  `gpt_oss-GateUP-B32-M2048` @ ratio 1.051)
- R33 rule (gm=2, xcds=4) applied: single-run 885, paired 5-run mean 881.0
  (baseline 883.6, delta -2.6)
- Correctness: 9/9 PASS on corner shapes (bit-identical — group_m and
  num_xcds are scheduling knobs only)

## Background — R32 discovery re-visited
R32 established the post-H4 bwd dA scope for `gpt_oss-GateUP`:
- fwd RCR: a=x [M, K=2880], b=w [B, N=5760, K=2880] → kernel
  shape (m, n=5760 tiles_n=23, k=2880). Covered by R9 rule
  `tiles_n==22 ∧ k==2880 ∧ m_total>=8192 → (gm tier-based)`.
  Actually `tiles_n = ceil(5760/256) = 23`, not 22 — the R9 rule
  predicate is `tiles_n == 22` so GateUP fwd misses it in practice.
  (Separate issue; R34 can audit this rule scope.)
- bwd dA: a=grad_out [M, N=5760], b=w [B, N=5760, K=2880], trans_b=False
  → H4 gate triggers (K%256=64 != 0) → b transposed to
  [B, K=2880, N=5760], trans_b=True, layout="rcr".
  Kernel shape: (m, n=K=2880 tiles_n=12, k=N=5760).
- **No rule covers tiles_n==12 ∧ k==5760 in config.py**. Default
  binding heuristic (gm=4, xcds=8) runs this path.

## R33 attempt
Added rule:
```python
if (
    layout == "rcr"
    and tiles_n == 12
    and k == 5760
    and m_total is not None
):
    return HipKittenConfig(layout=layout, group_m=2, num_xcds=4, kernel=None)
```
Scope narrower than R32 (which used tiles_n==12 ∧ k==5760 ∧ (gm=1, xcds=4)).
Chose (gm=2, xcds=4) — the R24 4-rule aggregate winner for a similar
"mid-tiles_n, mid-k" geometry cell.

## Paired 5-run (stash-based self-control)
```
Run 1: baseline=879  treatment=883  delta= +4
Run 2: baseline=886  treatment=880  delta= -6
Run 3: baseline=884  treatment=883  delta= -1
Run 4: baseline=885  treatment=880  delta= -5
Run 5: baseline=884  treatment=879  delta= -5

baseline  mean=883.6  stdev=2.70
treatment mean=881.0  stdev=1.87
paired delta mean = -2.60
```
Run 1 (+4) was session-warmup bias; runs 2-5 all negative.

## Cross-round dispatch-rule falsification tally (R29-R33)
| Round | Attempt | Paired delta | Status |
|---|---|---|---|
| R24 (LAND) | 4-rule dB var-K aggregate | > +5 | ACCEPTED |
| R25 | Qwen3-GateUP single-family rule | ~0 | FALSIFIED |
| R26 | 3-rule aggregate (DSV3-Down + Qwen3-GateUP + gpt_oss) | ~0 | FALSIFIED |
| R27 | DSV3-GateUP dB var-K (gm=2, xcds=8) | +0.25% | sub-noise |
| R28 | bf16_transpose_3d (BK,BN)=(128,128) for K==N | +3.4 | sub-+5 |
| R29 | 5-lever aggregate | -0.2 | FALSIFIED |
| R30 | R28 repro + H4 wall profile | -1.0 | non-reproducible |
| R31 | H4 gate tighten (bypass RRR fast path) | **-79** | net-negative catastrophic |
| R32 | post-H4 bwd dA RCR (1, 4) | -1.4 | FALSIFIED |
| **R33** | **post-H4 bwd dA RCR (2, 4)** | **-2.6** | **FALSIFIED** |

## Conclusion
BF16 dispatch surface is **fully exhausted** at the current metric precision
(stdev ~2.5, sub-+5 noise floor eats all dispatch-only wins). 6 consecutive
rounds of rule attempts below the +5 gate. Continuing dispatch-rule
tuning will not advance the score beyond the 879-892 plateau.

**Next advance must be C++ kernel work.** The R31 data revealed that
native-RRR N-tail (the slow path H4 transposes around) runs at ~0.6× the
RCR fuse throughput for gpt_oss K=2880 — eliminating this slowness in
`kernel_bf16_dynamic.cpp` would let us drop the H4 workaround entirely
(saving ~6.9-7.1% wall on gpt_oss fwd per R30 profile) AND free up
dispatch rules scoped on post-H4 geometry.

## R34 suggestion — C++ kernel main line
Start with the **`grouped_ntail` launcher** in
`HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`:
1. Locate the RRR N-tail path (N_kernel % 256 != 0 handling) — per R31
   analysis, this is the slow branch responsible for the 0.6× throughput
   drop vs the RCR fast-path.
2. Identify whether the slowness is:
   - (a) Register pressure / VGPR spill from a wider epilogue,
   - (b) An extra masked store / divergent branch on the tail tile, or
   - (c) A sub-optimal tile-schedule (e.g. the persistent launcher
     issuing tail tiles last where they can't overlap).
3. Fix whichever of (a)/(b)/(c) is the dominant cost. If (c), may
   only need a launcher schedule change (group tail tiles with
   their full-tile siblings).
4. Smoke-test: rebuild `tk_bf16_layouts.so`, run the correctness
   probe (/tmp/probe_round29_correctness.py — 9 shapes, fwd+bwd),
   measure the paired 5-run delta.
5. If net ≥ +5, also consider tightening the H4 gate to re-route
   only the remaining slow cases (if any) and drop H4 for N % 256 ==
   0 ∧ K % 128 != 0 RRR cases — this will need a follow-up round of
   correctness validation on the K-tail phantom-read bug.

Expected headroom from eliminating H4 + ntail slowness:
- gpt_oss geomean: 1.086 → ~1.15 (+6% via H4-bypass), score +~30-40
- DSV3 / Qwen3 unchanged (H4 only fires on gpt_oss in the metric suite)

## Chat window + commit discipline
Chat session at ~75 min of 90 min window. R34+ likely cold-start.
Primus-Turbo HEAD moves to this docs-only commit. HipKittens
repo unchanged.
