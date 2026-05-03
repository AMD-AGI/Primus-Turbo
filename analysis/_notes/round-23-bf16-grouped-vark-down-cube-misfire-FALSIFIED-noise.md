# Round-23 (BF16 grouped GEMM) — gpt_oss-Down dB var-K cube-rule misfire identified, fix FALSIFIED (noise-bound)

**Commit (this round):** docs-only (probe + note kept; rule reverted).
**Status:** Real dispatcher misfire found and characterized; correct
fix is uniform-positive at the kernel level (+0.72 to +2.31%, avg
+1.52%, bit-equivalent on all 4 gpt_oss-Down dB var-K shapes), but
the wall-time delta is below the metric's ±5-7 score noise floor.
**Lever (this round):** A2 / B4 (var-K dB cfg dispatch — narrowing
of cube rule's accidental capture of grouped CRR var-K).

---

## Discovery: cube rule (config.py:691) was firing on grouped CRR var-K

Rebuilt the dispatch table for all 6 dB var-K paths in the BF16
metric (helper at top of `/tmp/round_23_dispatch_audit.log`):

| family            | (N_fwd, K_fwd) | tiles_m | tiles_n | k_arg | cfg @ (B=16, M=2048) | rule fired |
|-------------------|---------------:|--------:|--------:|------:|---------------------:|------------|
| gpt_oss-GateUP    | (5760, 2880)   |      22 |      11 |  2048 | (gm=4, xcds=4)       | **R1 line 1080** |
| **gpt_oss-Down**  | (2880, 2880)   |      11 |      11 |  2048 | **(gm=2, xcds=32)**  | **cube line 691** ← surprise |
| DSV3-GateUP       | (4096, 7168)   |      16 |      28 |  2048 | (gm=4, xcds=8)       | DEFAULT |
| DSV3-Down         | (7168, 2048)   |      28 |       8 |  2048 | (gm=4, xcds=8)       | DEFAULT |
| Qwen3-GateUP      | (3072, 4096)   |      12 |      16 |  2048 | (gm=4, xcds=8)       | DEFAULT |
| Qwen3-Down        | (4096, 1536)   |      16 |       6 |  2048 | (gm=4, xcds=8)       | DEFAULT |

The **cube rule** (line 691) is `tiles_m <= 16 and tiles_m == tiles_n
and k <= 12288`, layout-agnostic. For gpt_oss-Down dB var-K
(tiles_m=tiles_n=11, k=2048): all three predicates true → fires
BEFORE the gpt_oss CRR rule at line 1080. R1's commentary at line
1100 ("covers all 8 gpt_oss var-K calls") was thus partially wrong:
the 4 gpt_oss-GateUP shapes hit R1's rule, but the 4 gpt_oss-Down
shapes were silently captured by the cube rule, which was tuned at
R1+R5 for **dense LLaMA forward** 4096^3 / 4096^2 × 11008 — pure RCR
forward workloads, never validated for grouped var-K.

R22's probe missed this entirely because it compared against a
hand-picked `(gm=4, xcds=4)` baseline rather than the actual
production cfg (which differs per family).

## Probe (kernel-only, vs production cfg)

`scripts/_bf16_vark_db_gpt_oss_down_probe.py` — new this round, more
rigorous than R22's probe (compares against the actual production
`(gm=2, xcds=32)` cell, not R22's stand-in `(gm=4, xcds=4)`). 11
cells × 5 trials × 120 iters per cell, kernel-only timing, bit-eq
check at every cell.

Reuses R22's autograd-fwd+bwd warm-up workaround for the BF16 K-tail
cold-start sync-fault.

### Sweep results (vs production `(gm=2, xcds=32)`)

```
gpt_oss-Down B=4  M=2048  (m_total=8192,   tiles_m=11, tiles_n=11):
   gm= 1 xcd= 4   918.4 TF  +1.77 %  (top-1)
   gm=16 xcd= 4   918.1 TF  +1.74 %
   gm= 8 xcd= 4   916.8 TF  +1.59 %
   gm= 4 xcd= 4   914.4 TF  +1.32 %
   gm= 2 xcd=32   902.4 TF   *PROD*

gpt_oss-Down B=4  M=4096  (m_total=16384):
   gm= 1 xcd= 4  1012.1 TF  +2.31 %  (top-1)
   gm=16 xcd= 4  1011.1 TF  +2.20 %
   gm= 8 xcd= 4  1010.0 TF  +2.09 %
   gm= 4 xcd= 4  1007.6 TF  +1.84 %
   gm= 2 xcd=32   989.3 TF   *PROD*

gpt_oss-Down B=32 M=2048  (m_total=65536):
   gm= 1 xcd= 4  1066.3 TF  +1.29 %  (top-1)
   gm=16 xcd= 4  1065.2 TF  +1.19 %
   gm= 4 xcd= 4  1063.6 TF  +1.03 %
   gm= 8 xcd= 4  1062.1 TF  +0.90 %
   gm= 2 xcd=32  1052.7 TF   *PROD*

gpt_oss-Down B=32 M=4096  (m_total=131072):
   gm= 1 xcd= 4  1156.8 TF  +0.72 %  (top-1)
   gm= 8 xcd= 4  1155.3 TF  +0.59 %
   gm=16 xcd= 4  1155.0 TF  +0.57 %
   gm= 4 xcd= 4  1151.9 TF  +0.30 %
   gm= 2 xcd=32  1148.5 TF   *PROD*
```

**`(gm=1, xcds=4)` is uniform-positive top-1 on all 4 shapes:**
- avg +1.52 %, min +0.72 %, max +2.31 %
- bit-equivalent (max_abs=0, bit_eq=True at every cell)
- Top-1 across the entire 12-cell × 4-shape grid by avg.

This is **2.2× larger** than R22's gpt_oss-GateUP +0.69 % avg. The
cube rule was a substantial mismatch.

## Patch tested

```python
if (layout == "crr"
        and tiles_m == 11
        and tiles_n == 11
        and k <= 4096
        and m_total is not None):
    return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
# (then the existing cube rule at line 691)
```

Inserted BEFORE the cube rule. Scope check (verified live):
- gpt_oss-Down dB var-K (4 shapes): hits new rule → `(gm=1, xcds=4)`. ✓
- gpt_oss-GateUP dB var-K (control, tiles_m=22): unchanged → `(gm=4, xcds=4)`. ✓
- DSV3-Down dB var-K (control, tiles_m=28): unchanged → `(gm=4, xcds=8)`. ✓
- Dense 4096^3 RCR forward (control, m_total=None): unchanged → `(gm=2, xcds=32)`. ✓

Only gpt_oss-Down var-K paths flip; everything else is untouched.

## Result (R23 metric, with rule)

3 metric runs each, with rule and without (i.e. baseline + apply +
revert + apply x2 + revert):

| run | baseline (no rule) | with rule |
|---|---|---|
| 1 | 878 | 878 |
| 2 | 885 | 878 |
| 3 | 879 | 887 |
| **mean** | **880.7** | **881.0** |
| range | [878, 885] | [878, 887] |

**Net delta: +0.3 score — pure noise** (distributions completely
overlap; both span ~7-9 score points). Per-shape gpt_oss-Down ratios:

| shape | baseline | run1 | run2 | run3 | mean Δ |
|---|---|---|---|---|---|
| Down-B4-M2048   | 1.106 | 1.106 | 1.102 | 1.140 | +0.010 |
| Down-B4-M4096   | 1.103 | 1.105 | 1.100 | 1.121 | +0.006 |
| Down-B32-M2048  | 1.049 | 1.053 | 1.052 | 1.061 | +0.006 |
| Down-B32-M4096  | 1.084 | 1.084 | 1.083 | 1.094 | +0.003 |

Run 3's broad lift (gpt_oss-GateUP also up despite being unaffected
by the rule) confirms it's noise-driven, not the rule. Run 1+2 show
the genuine rule signal: B=32-M2048 +0.4pp consistent, others flat.

## Why kernel-only +1.5% doesn't move the metric

Decomposition of fwd+bwd wall on a B=32 gpt_oss-Down shape:
- forward grouped GEMM ≈ 33 % (fast 2880^3 / 5760×2880^2 fuse)
- backward dA RRR ≈ 33 % (H4 reroute → RCR fuse)
- backward dB var-K CRR ≈ 33 % (this rule)

A +1.52 % var-K kernel-only delta lands as +0.5 % wall delta →
+0.005 to ratio (1.10 → 1.105). Per-shape progress moves +0.004 →
weighted contribution per shape ≈ 4·3·0.004 / 40 = +0.0012 → score
+1.2. Single-sample metric ratio noise is ±0.005-0.010 → SNR ≈ 1:2.
Even averaged over 3 runs, the signal stays at +0.3 score.

R20's 3-rule aggregate succeeded because it stacked 3 dA RRR cells
covering 12 of 24 shapes (DSV3-GateUP/Qwen3-GateUP/DSV3-Down);
combined +0.6-1.5 pp per shape × 12 shapes crossed +5.

## Constraints honored

- ✅ No commits with score below noise threshold (R23 doc-only,
  rule reverted before commit)
- ✅ No host syncs / per-(M,N,K) hardcodes / can_handle tightening
- ✅ Bit-identical correctness (max_abs=0, bit_eq=True at every
  probed cell; metric correctness gate 0/24 fail at every run)
- ✅ FP8 metric untouched (the probed/proposed change is
  BF16-only, in `if dtype == "bf16"` branch)
- ✅ Probe script archived for the R24 candidate (multi-family
  dB var-K aggregate)

## Implications for next rounds

1. **gpt_oss var-K (CRR/dB) cfg surface is fully closed at metric
   level**: combined with R22's gpt_oss-GateUP probe, every gpt_oss
   dB var-K shape is now characterized at the kernel level. Best
   cells are uniform-positive but each individually below noise.
2. **R24 candidate — multi-family dB var-K aggregate** (R20-style):
   probe the 4 untuned-default families (DSV3-GateUP, DSV3-Down,
   Qwen3-GateUP, Qwen3-Down — all on `(gm=4, xcds=8)` per the audit
   above) PLUS this round's gpt_oss-Down rule. If 12+ shapes can be
   simultaneously lifted +1-2 pp at the kernel level, the combined
   wall delta may cross the +5 score threshold (mirror of R20).
3. **Dispatch rule layering**: this round documented that the
   layout-agnostic rules (cube line 691, skinny-tall line 703) can
   silently capture grouped var-K shapes they were never tuned for.
   Future audit-style rounds should re-run the dispatch table check
   whenever new layout-agnostic rules are added.

## Files

- `scripts/_bf16_vark_db_gpt_oss_down_probe.py` — new probe
  (kept for R24 multi-family aggregate or future re-test).
- `analysis/_notes/round-23-bf16-grouped-vark-down-cube-misfire-FALSIFIED-noise.md`
  — this falsification note.
- `primus_turbo/pytorch/kernels/hipkitten/config.py` — UNCHANGED
  (rule reverted before commit).
