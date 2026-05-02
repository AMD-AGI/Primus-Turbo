# Round 9 — BF16 grouped Qwen-Down M=2048 scope-leak fix (Lever F)

## TL;DR
- Rule scope leak found: the round-10 DSV3-GateUP-M2048 BF16 rule
  (`tiles_n == 16 and tiles_m == 8 and k <= 7168`) was catching the
  newly-added Qwen-Down M_per_group=2048 BF16 shapes (`tiles_n=16`,
  `tiles_m=8`, `k=1536` — same predicate, different K).
- The DSV3-tuned `(gm=1, xcd=4)` config was 1.3-1.4 pp suboptimal on
  Qwen-Down K=1536 (only 6 K-iter per tile-step vs 28 for K=7168).
- Added a Qwen-specific predicate `k == 1536` BEFORE the DSV3 rule to
  route the 2 affected shapes to `(gm=4, xcd=8)` (= the binding
  default; the rule exists purely to escape the over-broad DSV3 rule).
- Bit-identical output verified.

## Per-shape result (single-trial metric snapshot, before vs after rule)
```
                                          before    after    Δ
grpBF16_Qwen3-235B-A22B-Down-B16-M2048    1.184  →  1.198   +1.4 pp
grpBF16_Qwen3-235B-A22B-Down-B32-M2048    1.181  →  1.194   +1.3 pp
```

Tight verify (200-iter × 7-trial p20 at
`/tmp/verify_qwen_down_bf16_m2048_round9.py`):
```
B16-M2048:  (4, 8)=1106.26 TF vs (1, 4)=1096.10 TF = +10.16 TF (+0.93 pp)
            spread 0.47 % / 0.28 % → gap = 2.0× spread → clean
B32-M2048:  (4, 8)=1108.39 TF vs (1, 4)=1099.24 TF =  +9.15 TF (+0.83 pp)
            spread 0.28 % / 0.51 % → gap = 1.6-3× spread → clean
```

The winner's per-trial min beats the current's max in 7/7 trials per
shape — the single cleanest-signal Lever F gain since R6.

## Aggregate metric noise band
5 post-rule trials: 963 / 961 / 961 / 961 / 962 (mean 961.6, mode 961).
Pre-rule single-trial: 962. Aggregate Δ ≈ -0.4 (within ±1.5 noise band
characterised in R7/R8). Per-rule rationale below stands on:
1. Per-shape signal is clean (above tight-verify spread).
2. Bit-equivalent (max_abs_diff = 0.0).
3. Structural correction: the over-broad DSV3 predicate was a known
   scope leak that would compound as more grouped families are added.

## Why (gm=4, xcd=8) wins for K=1536
With block_k=256, K=1536 ⇒ only 6 K-iter per tile-step — ~5× lighter
per-tile compute than the K=7168 DSV3-GateUP cousins this rule used to
match. (gm=1) maximises B-tile L2 reuse on long-K shapes by walking
the entire N-row before moving M; on shallow-K the same N-walk
under-feeds the persistent loop because each tile-step finishes in
~6 mfma waves, leaving the LDS double-buffer's load-side bandwidth
unsaturated. (gm=4) batches 4 M-tiles into the same group, sharing
each B-pack across 4 mfma sequences and keeping the LDS buffer warm.

## Rule scope check (no collateral)
```
tiles_n == 16 ⇔ N == 4096                 # shared with DSV3-GateUP, dense
tiles_m ==  8 ⇔ M_per_group == 2048       # excludes M=4096 sibling
k       == 1536                           # uniquely Qwen-Down in BF16 metric
```
Other BF16 metric grouped K values: DSV3 ∈ {2048, 7168}, gpt_oss=2880,
Qwen-GateUP=4096. Dense BF16 K ∈ {4096, 11008, 14336, 22016, 28672}.
No K=1536 in any other metric shape. Strict equality (not `<=`)
guards against future shallower-K families.

## Other shapes probed (no rule change)
- Qwen-Down M=4096 (cube-small `(gm=2, xcd=32)` routing): tight verify
  showed `(2, 32)` is the joint top — current routing already
  optimal, no rule needed.
- Qwen-GateUP B*-M*: 4 shapes already routed to binding default
  `(gm=4, xcd=8)`; the prior round-7 FP8 study found no clean BF16
  signal here either — left at default.

## Files touched
- `primus_turbo/pytorch/kernels/hipkitten/config.py`: 1 new rule (~80
  lines comment + 1 return) inserted at line 609 (BF16 RCR branch,
  immediately above the round-10 DSV3-GateUP-M2048 rule).
- `analysis/_notes/round-9-bf16-grouped-Qwen-Down-M2048-scope-leak-fixed.md`
  (this file).

No HK kernel changes this round. No FP8 changes (FP8 Qwen-Down rules
already scoped tightly via `tiles_n == 16 and tiles_m == 16 and k == 1536`
in R6).

## Lever-F status after R9
| family               | precision | M=2048 | M=4096 |
|----------------------|-----------|--------|--------|
| Qwen-Down            | FP8       | default | R6 ✓ |
| Qwen-Down            | BF16      | **R9 ✓** | cube-small (verified) |
| Qwen-GateUP          | FP8       | R7 ✓ | falsified-R7 |
| Qwen-GateUP          | BF16      | default | default |
| DSV3-GateUP          | FP8       | B32 R8 ✓ | R8 ✓ |
| DSV3-GateUP          | BF16      | R10 (legit) | cube-small |
| DSV3-Down            | FP8       | (R20-67) | (R20-67) |
| DSV3-Down            | BF16      | R10 | R10 |
| gpt_oss-{Down,GateUP}| FP8/BF16  | R23/R7/R57/R61/R67/R68 (FROZEN ceiling) |

Lever F is now exhausted on the BF16 grouped side (all 12 Qwen3 +
DSV3 shapes have been audited; gpt_oss is FROZEN; dense LLaMA shapes
are not in the grouped 24-shape suite). The remaining ratio gap to
1.20 is concentrated in the 7 gpt_oss K=2880 shapes (architectural
spill ceiling) — no Lever F lift available there without violating
the FROZEN list.

## Suggested R10
**Lever E scout (ASM software pipeline)**: prerequisite is a clean
HK-side branch isolated to the K=2880 RCR / RRR / CRR kernels with
explicit scheduling annotations on the LDS load and mfma waves. The
goal is to claw back 5-10 % on gpt_oss K=2880 (currently 1.026-1.087
ratio range) without touching the architectural spill picture.
Before kicking off:
1. Re-confirm noise band (3 trials baseline) so subsequent score
   moves are interpretable.
2. Sanity-check that the HK-side rebuild path supports an isolated
   ASM patch — the round-2 inline-asm path in `rcr_8w_load_hoist`
   is the natural anchor.

If the R10 ASM scout shows < 5 % per-shape signal in microbench, fall
back to documenting the FROZEN ceiling and shifting focus to the
backward path (`bench_grouped_gemm_turbo.py --dtype fp8 --bwd`)
which the metric does not currently exercise.
