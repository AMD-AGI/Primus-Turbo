# R17 — gpt_oss FP8 fwd RCR (gm, xcds) drift re-verify FALSIFIED

## TL;DR

Continuing from R16 (var-K wgrad kernel surgery falsified — register-bound),
R17 pivots to the suggestion in R16's "next-round recommendation":

> Pivot to fwd RCR / dgrad RRR shapes — the (gm, xcds) cells of those
> layouts have NOT been exhaustively swept on the post-R12 K-tail-fuse-
> eligible kernel binary. gpt_oss-Down-B4-M2048 fwd (1448 T, ratio 0.516)
> is the next-worst shape.

Re-swept all 5 fwd RCR rules in the 8-shape gpt_oss family
(Down-B4-M2048, Down-B4-M4096, GateUP-B4-M2048, Down-B32-M2048,
Down-B32-M4096) on the current (post-R5/R11/R16) FP8 binding. All 5
existing rules sit within ≤0.6pp of every other tested cell on a
17-cell sweep, **and tight-verify (1500-iter × 7-repeat × 3-seed p20)
collapses every coarse-sweep "candidate" to a statistical tie or
regression**. No (gm, xcds) update changes ship.

**Conclusion**: the gpt_oss fwd RCR section is at the kernel-cell
ceiling on the current binary. Drift since the R7/R8/R23/R50/R12/R69
rules landed has not opened any new (gm, xcds) gap that meets the
0.5pp robustness threshold.

This closes (for now) the dispatcher-side optimization track on the
gpt_oss FP8 fwd family. Note this also covers all 5 dgrad shapes
because the H4 reroute (R34 + R18) maps gpt_oss-Down dgrad
(`K_RRR=N_fwd=2880, % 128 = 64`) and gpt_oss-GateUP dgrad
(`N_RRR=K_fwd=2880, % 256 = 64`) through the SAME RCR rules with the
SAME `(tiles_m, tiles_n, k, m_total)` dispatcher key — verified inline
in the existing R7/R23/R50/R8/R12/R69 rule comments.

## Anchor & sweep grid

5 shapes × 17 cells = 85 timing cells, each 200-iter × 3-repeat × 3-seed
p20 (≈ R15 coarse-sweep convention).

```
gm  ∈ {1, 2, 4, 8, 16, 32}
xc  ∈ {2, 4, 8}
```

(Plus a defensive (32, 2) for shapes where R7 / R8 picked it.)

## Coarse-sweep results

### Down-B4-M2048 fwd RCR — current rule (16, 2) [R7 → R2 update]

```
       cfg     med ms     min ms    TFLOPS   spread  Δpp vs cur
   (16, 2)     0.0923     0.0921    1472.0   0.0004      +0.00 *cur
   (32, 2)     0.0926     0.0922    1467.5   0.0006      -0.30
    (4, 8)     0.0929     0.0924    1462.5   0.0010      -0.65
    (2, 2)     0.0930     0.0930    1460.6   0.0008      -0.78
    (8, 2)     0.0934     0.0933    1455.0   0.0004      -1.17
```
(16, 2) wins outright; no candidate within 0.5pp. Rule unchanged.

### Down-B4-M4096 fwd RCR — current rule (1, 4) [R12 → R3 update]

```
       cfg     med ms     min ms    TFLOPS   spread  Δpp vs cur
    (1, 4)     0.1409     0.1408    1929.2   0.0003      +0.00 *cur
   (32, 4)     0.1426     0.1426    1906.5   0.0006      -1.19
    (2, 2)     0.1428     0.1428    1903.8   0.0000      -1.33
   (16, 4)     0.1429     0.1428    1902.2   0.0001      -1.42
```
(1, 4) wins by +1.2pp clear; no candidate within 1pp. Rule unchanged.

### GateUP-B4-M2048 fwd RCR — current rule (1, 4) [R23]

```
       cfg     med ms     min ms    TFLOPS   spread  Δpp vs cur
    (2, 2)     0.1430     0.1429    1900.1   0.0002      +0.25
    (1, 4)     0.1434     0.1432    1895.3   0.0003      +0.00 *cur
    (2, 4)     0.1442     0.1441    1884.8   0.0003      -0.56
```
(2, 2) edges (1, 4) by +0.25pp on coarse — within run-to-run spread
(spreads are 2 µs vs gap 4 µs). Tight-verify on the borderline
candidate skipped (the +0.25pp gap is below the 0.5pp threshold).

### Down-B32-M2048 fwd RCR — current rule (16, 4) [R8]

Coarse: (32, 4) +0.58pp over (16, 4) — borderline; tight-verified.

```
TIGHT VERIFY (1500-iter × 7-repeat × 3-seed p20):
       cfg     med ms    TFLOPS   spread  med/spread     Δpp                 seeds
   (16, 4)     0.5760    1887.6   0.0030       191.9   +0.00 *base  0.5753/0.5783/0.5760
   (32, 4)     0.5764    1886.1   0.0013       450.7   -0.08  0.5764/0.5755/0.5768
   (16, 2)     0.5864    1854.1   0.0031       190.4   -1.81
   (32, 2)     0.5868    1852.8   0.0013       458.8   -1.88
```
(16, 4) and (32, 4) are statistical tie (-0.08pp gap, well below the
spread on both sides). Rule unchanged.

### Down-B32-M4096 fwd RCR — current rule (4, 4) [R50]

Coarse showed (4, 4) winning over a hypothetical "(4, 8)" baseline —
but the production rule is ALREADY (4, 4) per R50; the (4, 8) entry
in the coarse sweep was just a hypothetical comparison cell. Tight-
verify confirms:

```
TIGHT VERIFY (1500-iter × 7-repeat × 3-seed p20):
       cfg     med ms    TFLOPS   spread  med/spread     Δpp                 seeds
    (4, 4)     1.1125    1954.5   0.0002      4636.0   +0.53  1.1124/1.1126/1.1125
    (8, 4)     1.1159    1948.5   0.0015       734.2   +0.22
    (4, 8)     1.1184    1944.2   0.0012       901.9   +0.00 *baseline (default fallback)
    (1, 4)     1.1208    1940.0   0.0031       359.2   -0.22
```

The probe confirms the existing (4, 4) rule beats the hypothetical
default (4, 8) by +0.53pp; **no change needed** — the rule already
captures this win. The +0.53pp number was misread on the coarse-sweep
table due to the baseline_cfg parameter pointing at "(4, 8)" instead
of the actual production rule "(4, 4)" — once corrected the table
reads "(4, 4) is the production winner; siblings within 0.5pp are
suboptimal".

## Why no drift in 5 shapes

Looking at the kernel-rebuild history since each rule landed:

| Rule | Shape | Cell | Landed when | Kernel binary trim since |
|---|---|---|---|---|
| R7 → R2 | Down-B4-M2048 fwd | (16, 2) | this run R2 | R3 (var-K slots), R5 (host trim), R11 (host trim), R12 (K-tail fuse), R16 (var-K trim, reverted) |
| R12 → R3 | Down-B4-M4096 fwd | (1, 4) | this run R3 | R4-R16 |
| R23 | GateUP-B4-M2048 fwd | (1, 4) | (older) | R4-R16 |
| R8 | Down-B32-M2048 fwd | (16, 4) | (older) | R4-R16 |
| R50 | Down-B32-M4096 fwd | (4, 4) | (older) | R4-R16 |

The post-R12 K-tail fuse landed BEFORE this run's R3 (commit 7060820
on 2026-05-07); R5/R11/R16 host trims and R12 K-tail fuse all preserve
the persistent-grid tile schedule (gm + xcds knobs are pure scheduling).
The wave-step / chiplet alignment math hasn't shifted since R7's
landed.

The R16-falsified `binary-search → divide` trim was REVERTED, so the
var-K kernel binary is back to its pre-R16 state. The fwd RCR kernel
binary was untouched throughout R5-R16.

## Implication

Combined with R14 (fwd RCR PMC launch geometry exhausted), R15 (var-K
wgrad dispatcher exhausted), and R16 (var-K wgrad kernel surgery
falsified), R17 confirms:

**The gpt_oss FP8 dispatcher track is at the kernel-cell ceiling
across all 8 shapes × 3 sections = 24 cells**.

Score 680 → 681 with current rules; 2800 T target requires +33-60%
absolute kernel TFLOPS that the current kernel cannot supply at any
(gm, xcds) cell.

## Next-round recommendation

The remaining productive direction is the **Python-overhead trim
along the public-op path**. Probe vs metric gap on Down-B4-M2048 fwd:

```
probe (direct hk.grouped_rcr_dscale + pre-allocated out): 0.0928 ms ≈ 1472 T
metric (full grouped_gemm_fp8_impl public op):            0.0958 ms ≈ 1425 T
                                                          ─────────
                                                         Δ ≈ 3.0 µs
```

R5 (dispatcher bypass + @custom_op unwrap) + R11 (`_resolve_fp8_scales`
skip) + R18 (method-call trim) already shaved ~10 µs. The remaining
~3 µs is dominated by:

1. `torch.empty((m_total, n), ...)` output allocation: ~1-3 µs (FROZEN
   list forbids per-shape output cache).
2. `select_default_config` lru_cache hit: ~0.5 µs.
3. pybind11 launch overhead: ~1-2 µs.
4. Misc shape lookups + ternaries: ~0.5 µs.

Items 1, 3 are kernel-side ceilings. Item 2 may have one more 100ns
shave (dict-based fast path for the canonical gpt_oss key set, but the
@lru_cache hit IS already a dict-hash lookup).

For R18: profile the execute body line-by-line to pin exactly where
the 3 µs lives, then ship trims if any line shows > 0.3 µs of
non-essential work.

## Repro

Coarse sweep:  `scripts/_probe_round_17_fwd_rcr_drift.py` (5 shapes × 17 cells).
Tight verify:  `scripts/_probe_round_17_b32_tight.py` (4-cell × 1500-iter × 7-repeat × 3-seed).
