# Round 79 — BF16 grouped: R28 BF16-transpose tile-shape lever RE-FALSIFIED (Δ+1 single-run, sub-+5 noise)

## Status
**FALSIFIED — kernel-level change reverted.** R28's archived BF16-specific
`(BK, BN)=(128, 128)` for K==N transpose tile-shape was re-applied in
R79. Bit-equality verified on all 4 gpt_oss H4 shapes. Single-run metric:
872 → 874 (+1 above prior-best 873). R28's predicted +3.4 mean is
re-confirmed as below the project's +5 single-run commit threshold.

## Round entry state
* HEAD: `1f73827` (post-R78 docs commit).
* Best score: 873 (R1 baseline).
* Patience counter: 1 (R78 was docs-only).
* GPU: HIP_VISIBLE_DEVICES=3 (auto-pinned).
* Lowest-progress shape: `gpt_oss_20B-GateUP-B32-M2048` (ratio 1.047,
  progress 0.838, weight 3×) — same as R78.

## Why this lever
R78 mapped out the structural path A (H4-elim via RRR N-tail MFMA, +20-22
score multi-round). R79 ran the metric and re-confirmed the score is
parked at 872-874. With Path A being multi-round (R3-R4+ for the
RRR-side kernel rewrite — see R78 note), R79 explicitly looked for a
SHIPPABLE single-round lever.

R28 (HEAD `0c038f2` baseline 879) had archived a clean kernel-level win
on `bf16_transpose_3d` for K==N shapes (gpt_oss-Down family) that was
**reverted only because the single-round metric move was +3.4 mean**
(below the +5 threshold). R28's note explicitly flagged:

> R28 archived a +3.4 real-but-sub-noise lever. ... R29 should bundle
> with R26/R27 OR re-land as "numerical-correctness improvement"
> (bit-equal, no-regression).

R29 attempted the bundle and FALSIFIED (Δ-0.2 net). The R28 lever was
left archived since.

R79 re-applied R28's `_select_block_shape_bf16(K, N) = (128, 128)` for
K==N (vs FP8's `(256, 128)`), because:
* The codebase has shifted (FP8 quantize-cache landed in r1@7919976,
  several falsification rounds — possibly small regressions). Maybe the
  relative impact is now larger.
* It's the only archived lever with bit-identical output and uniform-
  positive measurement on target shapes.

## Implementation
Single Python helper added in `primus_turbo/triton/utils/fp8_transpose.py`:
```python
def _select_block_shape_bf16(K: int, N: int) -> tuple[int, int]:
    if K > N:   return 128, 256   # GateUP (FP8 pick preserved, sub-noise)
    if K == N:  return 128, 128   # Down (R28 win; FP8 pick was 256, 128)
    return 128, 128
```
`bf16_transpose_3d` routed through this helper. FP8 path
(`fp8_transpose_3d` → `_select_block_shape`) untouched.

## Correctness
`/tmp/r79_transpose_correctness.py` — 4 H4 shapes, `torch.equal` on
uint8 view + `max_abs == 0.0`:
```
GateUP-B4   (4, 5760, 2880)   bit_eq=True  max_abs=0.0  PASS
GateUP-B32  (32, 5760, 2880)  bit_eq=True  max_abs=0.0  PASS
Down-B4    ( 4, 2880, 2880)  bit_eq=True  max_abs=0.0  PASS  ← (128,128) NEW
Down-B32   (32, 2880, 2880)  bit_eq=True  max_abs=0.0  PASS  ← (128,128) NEW
```
All 4 ALL-PASS. Bit-equality preserved (per R28 archived).

## Metric — single-run before / after

```
Baseline (entry):      score=872   gpt_oss=1.0748  DSV3=1.1184  Qwen3=1.1086
After R28 lever:       score=874   gpt_oss=1.0768  DSV3=1.1211  Qwen3=1.1096
                       Δ score=+2  Δ gpt_oss=+0.002  Δ DSV3=+0.003  Δ Qwen3=+0.001
```

vs prior best (873) → **Δ +1**. Below +5 threshold (need ≥ 878 single
run). Per-family geomean shifts are all within ±0.003 = noise floor.

Per-shape changes attributable to the transpose lever (gpt_oss-Down
H4-reroute family, K==N=2880):
| shape | before | after | Δ ratio | weighted Δ score |
|---|---:|---:|---:|---:|
| Down-B4-M2048   | 1.050 | 1.063 | +0.013 | +0.78 |
| Down-B4-M4096   | 1.097 | 1.101 | +0.004 | +0.30 |
| Down-B32-M2048  | 1.054 | 1.050 | -0.004 | -0.30 |
| Down-B32-M4096  | 1.088 | 1.091 | +0.003 | +0.225 |
| **net 4 shapes**| | | | **+1.0**  (matches R28 prediction shape-by-shape) |

R28 predicted +3.4 mean by bundling B=4 + B=32 + GateUP + Down. R79
single-run shows +1 net for the 4 Down shapes alone, with B=32 noise
canceling B=4 wins. Predictive model still accurate.

## Decision
Per project policy "Flat or down → revert + falsification round note":
**reverted**. `git checkout -- primus_turbo/triton/utils/fp8_transpose.py`
on the only changed file. No HipKittens code touched.

The archived R28 mechanism remains real (uniform-positive on B=4 Down,
allclose-safe). R79's run reaffirms it is not enough on its own to
clear +5 single-run; R29 already proved bundling it with the available
dispatch levers gets you to net -0.2.

## Why this matters for the round-counter
R79 spent its budget on a single-round shippable attempt (vs. R78's
multi-round Path A scaffold). The result: this archived lever
re-confirmed as sub-+5. After R79 we have:
* exhausted: dispatch (R32-34, R44-45, R66-77), MFMA-schedule (R51-55,
  R76), occupancy (R65, R73), LDS-swizzle (R49, R74), KI-spec (R52-53,
  R67), work-stealing (R61 LANDED, R75 falsified), persistent grid
  (R63), transpose tile-shape (R28 archived, R79 re-falsified),
  H4 gate tightening (R31), small focused changes generally.
* the only remaining structural lever per R78's analysis is **the
  multi-round Path A: native RRR + B-load N-mask → drop H4 transpose +
  drop ntail kernel** (+20-22 score predicted from R30/R31 wall
  decomposition).

## R80 direction
**Begin Path A R3 (Step 1)** — write a STANDALONE correctness probe
that calls `grouped_kernel<RRR, KI=0>` with `bpc = ceil_div(g.n, BLOCK_SIZE)`
on gpt_oss-GateUP B4-M2048 (smallest H4-eligible shape; min blast-radius
if it phantom-reads). If the existing G::load already SRD-clamps OOB N
columns to 0 for RRR (i.e., the per-group SRD bound + N-axis stride
fall in a regime where OOB col_global truly exceeds the bound — this
is what `kernel_bf16_dynamic.cpp:4283-4291` claimed it does NOT do, but
the comment was R5-era and not re-verified post-R56 LDS-staged ntail
landing), correctness PASSes for free and the lever opens up.

If correctness FAILs (the more likely outcome per the R5/R7 phantom-read
history), R3 needs to add an explicit per-lane B-load mask in the RRR
prologue / main_loop_iter via a `template <bool MASK_N_TAIL>` switch on
`device_gemm_tile_body`'s buffer_load_b128 — ~20-50 lines of C++,
multi-round (R3 = mask + correctness + standalone bench, R4 = primus
dispatch toggle + metric verify, R5 = optimize VGPR cost).

R80 picks the cheapest test:
1. Add `MASK_N_TAIL = (g.bpc * BLOCK_SIZE > g.n)` runtime check at
   `dispatch_grouped<RRR>` and also extend `g.bpc = ceil_div(g.n, BLOCK_SIZE)`
   for RRR.
2. Skip launching `grouped_ntail_kernel_lds_rrr<64>` when MASK_N_TAIL fires.
3. Run downsized correctness probe (`scripts/_metric_grouped_bf16_weighted_wall.py`'s
   gate → standalone python that calls grouped_gemm + Triton ref + bf16
   `check_allclose`) on 1 H4 shape.

**Critical**: The `g.fast_n` constants, the C-store mask, AND the
per-group SRD bound are all touched here — R80 must NOT skip the
DSV3-first warmup (HK BF16 K-tail cold-start sync-fault, see SKILL.md).
Run on H4-disabled fork of grouped_gemm_impl.py to bypass the fix-it-up
transpose path.

Committed-this-round: this falsification note (no code change).
