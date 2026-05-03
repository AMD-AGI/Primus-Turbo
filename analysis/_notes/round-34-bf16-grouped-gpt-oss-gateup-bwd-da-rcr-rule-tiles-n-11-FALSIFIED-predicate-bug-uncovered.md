# Round-34 — BF16 grouped GEMM: gpt_oss-GateUP post-H4 bwd dA (tiles_n==11, k==5760, gm=1, xcds=4) FALSIFIED **+ R32/R33 predicate-bug uncovered**

## Status
**FALSIFIED** — paired 5-run delta = **-1.6 score** (treatment 881.0 vs
baseline 882.6, stdev ~4). BUT this round **uncovered a bug in R32/R33**:
both of those rounds specified `tiles_n == 12` which **never matched**
(actual `tiles_n = n // 256 = 2880 // 256 = 11`, floor-div, not ceil).
This means R32 (-1.4) and R33 (-2.6) were **pure noise runs** — the
supposed "rules" were vacuous predicates that never fired.

R34 finally fired the **correct** rule (`tiles_n==11 ∧ k==5760`) on
gpt_oss-GateUP post-H4 bwd dA, and it still lands sub-noise at -1.6.

## Metric before / after
- Baseline HEAD `a4a358c` single run: 886 (lowest
  `gpt_oss-GateUP-B32-M2048` @ ratio 1.053)
- With R34 rule (gm=1, xcds=4) single run: 885
- Paired 5-run: treatment **881.0** vs baseline **882.6** → **delta -1.6**
- Correctness: 9/9 PASS on corner shapes (bit-identical — group_m /
  num_xcds are scheduling knobs)

## R32/R33 predicate-bug discovery
Looking at `config.py` line 222: `tiles_n = n // 256` (floor division).

For gpt_oss-GateUP bwd dA post-H4:
- a = grad_out [M_total, N_fwd=5760]
- b = H4-transposed w [B, K_fwd=2880, N_fwd=5760] (trans_b=True)
- layout = "rcr"
- n = b.shape[-2] = K_fwd = 2880
- **tiles_n = 2880 // 256 = 11**, NOT 12

R32 and R33 both wrote `if ... tiles_n == 12 ...` which was a
**never-fires branch**. The rule body returning `HipKittenConfig(1, 4)`
or `HipKittenConfig(2, 4)` never executed. Their paired-5-run deltas
(-1.4, -2.6) were purely natural run-to-run noise with the dispatcher
still returning the binding default.

R34 corrected the predicate to `tiles_n == 11 ∧ k == 5760` (which **does**
match this scope: no other grouped metric shape has tiles_n==11 AND
k==5760). With (gm=1, xcds=4) — mirror of the existing Down rule's
top-tier winner — the paired 5-run was -1.6. So the rule **does fire**,
but **(gm=1, xcds=4) doesn't improve** this geometry.

## Paired 5-run (stash-based self-control)
```
Run 1: baseline=883  treatment=887  delta= +4
Run 2: baseline=879  treatment=884  delta= +5
Run 3: baseline=888  treatment=878  delta=-10
Run 4: baseline=879  treatment=878  delta= -1
Run 5: baseline=884  treatment=878  delta= -6

baseline  mean=882.6  stdev=3.78
treatment mean=881.0  stdev=4.24
paired delta mean = -1.60
```

Runs 1-2 suggested progress (+4, +5). Runs 3-5 all negative. Net sub-noise.

## Cross-round dispatch-rule falsification tally (R24-R34 corrected)
| Round | Attempt | Paired delta | Real? | Status |
|---|---|---|---|---|
| R24 (LAND) | 4-rule dB var-K aggregate | > +5 | yes | **ACCEPTED** |
| R25-R30 | various dispatch / transpose / H4 | ~0 to +3.4 | yes | sub-+5 |
| R31 | H4 gate tighten | **-79** | yes | catastrophic (N-tail is slow) |
| R32 | "(tiles_n==12, gm=1, xcds=4)" rule | -1.4 | **NO — vacuous** | noise |
| R33 | "(tiles_n==12, gm=2, xcds=4)" rule | -2.6 | **NO — vacuous** | noise |
| **R34** | **(tiles_n==11, k==5760, gm=1, xcds=4)** | **-1.6** | **YES** | **FALSIFIED** |

## Conclusion
- BF16 dispatch surface is **genuinely exhausted** at current metric
  precision — now with a **firing** rule (R34) over the last uncovered
  scope at (gm=1, xcds=4), still sub-noise.
- R32/R33 were vacuous-rule noise runs, NOT real dispatch-rule tests.
  They should be discounted when future rounds look at this cell.
- A different `(gm, xcds)` pair at the R34 scope might still squeeze a
  win (e.g. (gm=2, xcds=4), (gm=8, xcds=4), (gm=4, xcds=4)) — but the
  +5 gate is unforgiving with this ~4pt stdev, and per-call switching
  costs rule-lookup Python overhead.
- **Next advance MUST be C++ kernel work** (per R33 recommendation).

## R35+ main line (C++ kernel work)
Same as R33 recommendation, reinforced by R34 data:
1. Open `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`.
2. Locate `grouped_ntail` launcher (N_kernel % 256 != 0 path).
3. R31 profile showed this runs at ~0.6× RCR fuse throughput for
   gpt_oss K=2880 shapes, forcing the H4 transpose workaround.
4. Likely slow because (a) VGPR spill in wider epilogue, (b) masked
   store / divergent branch on tail tile, or (c) sub-optimal tile
   schedule (tail tiles issued last).
5. Fix → rebuild `tk_bf16_layouts.so` → run
   `/tmp/probe_round29_correctness.py` → paired 5-run.
6. Expected headroom: drop H4 entirely → save ~7% wall on gpt_oss
   fwd → score +30-40 (gpt_oss geomean 1.086 → ~1.15).

## Optional R35 small-surface attempts (if cold-start & C++ blocked)
- (gm, xcds) sweep at the R34 scope (tiles_n==11, k==5760):
  remaining cells (2,4), (4,4), (8,4), (16,4). One at a time;
  each is ~4pt stdev territory so require paired 5-run.
- `_select_block_shape` alternate BF16-specific heuristic (R28
  idea, R30 non-reproducible). Tiny potential, low risk.

## Chat window + commit discipline
Chat session at ~80 min of 90 min window. R35 will be **cold-start**
(new chat, no prior file reads visible). Must infer from git log.
Primus-Turbo HEAD moves to this docs-only commit. HipKittens
repo unchanged.
