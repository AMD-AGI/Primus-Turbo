# Round 26 — BF16 grouped 3-rule aggregate (fwd RCR + dB var-K) FALSIFIED (noise-bound)

## Goal

Continue the R25 line: build the next R20/R24-style multi-rule
aggregate to clear the +5 metric noise floor. R25's single
Qwen3-GateUP fwd RCR rule (+1.68 % avg, but +0.5 expected score)
fell to noise. R26 audited 4 more families and bundled all
uniform-positive winners into one aggregate.

R26 baseline = **883** (single run; 5-run mean **882.6**).

## Probes

### `scripts/_bf16_rcr_fwd_multi_family_probe.py`

12-cell × 5-trial × 200-iter probe on 4 fwd RCR families:

| family | current rule | best uniform-positive cell | avg Δ |
|---|---|---|---|
| DSV3-GateUP M=2048 | R10 (gm=1, xcds=4) | NO uniform-positive | — |
| DSV3-GateUP M=4096 | cube (gm=2, xcds=32) | NO uniform-positive | — |
| **DSV3-Down all 4** | R10 (gm=16, xcds=2) | **(gm=16, xcds=4)** | **+0.47 %** (range +0.27..+0.74) |
| Qwen3-Down M=4096 | cube (gm=2, xcds=32) | NO uniform-positive | — |

Only DSV3-Down has headroom. Per-shape Δ vs prod (gm=16, xcds=2):
B16-M2k +0.57 %, B16-M4k +0.27 %, B32-M2k +0.74 %, B32-M4k +0.30 %.
Single-knob change (xcd 2→4); bit_eq=True vs (gm=4, xcds=4).
Mechanism: post-R19/R20 BUFFER-store kernel reduced cross-WG
memory traffic, making xcds=4's tighter chiplet rotation more
viable than R10's choice of xcds=2.

### `scripts/_bf16_vark_db_gpt_oss_gateup_probe.py`

12-cell × 5-trial × 120-iter probe on the 4 gpt_oss-GateUP
shapes (the only family on the R1 (gm=4, xcds=4) rule that R24
didn't re-tune):

| shape | prod (4,4) | best (1,4) | Δ |
|---|---|---|---|
| B=4-M2048    | 1192.5 | 1206.8 | +1.20 % |
| B=4-M4096    | 1252.7 | 1263.1 | +0.83 % |
| B=32-M2048   | 1117.5 | 1127.1 | +0.85 % |
| B=32-M4096   | 1205.8 | 1218.2 | +1.03 % |

avg +0.98 %, range [+0.83, +1.20] %, **uniform-positive**.
bit_eq=True. Same `(gm=1, xcds=4)` cell that wins for the 4
R24 dB var-K families AND for gpt_oss-Down (R23 finding,
R24 landed). The BF16 grouped var-K persistent kernel
converges on `(gm=1, xcds=4)` for moderate tiles_m + shallow
tiles_n geometries across families.

## Aggregate attempt — 3 rules

| rule | scope | change |
|---|---|---|
| R10 DSV3-Down fwd RCR | tiles_n==28 ∧ 8≤tiles_m≤16 ∧ k≤4096 | xcds 2 → 4 |
| R1 gpt_oss-GateUP dB var-K | tiles_n==11 ∧ 8≤tiles_m≤24 ∧ k≤4096 (gpt_oss-Down already split off via R24) | (gm,xcds) (4,4) → (1,4) |
| NEW Qwen3-GateUP fwd RCR | tiles_n==12 ∧ k==4096 ∧ m_total!=None | (gm=1, xcds=4) |

Dispatch verification confirmed: all target shapes routed to new
cells; dense LLaMA controls (m_total=None) and Qwen3-Down /
DSV3-GateUP controls unchanged.

## Verification — head-to-head 5-run means

| | run1 | run2 | run3 | run4 | run5 | mean |
|---|---|---|---|---|---|---|
| baseline (HEAD 0c038f2) | 878 | 888 | 880 | 880 | 887 | **882.6** |
| after 3-rule aggregate | 884 | 882 | 882 | 881 | 887 | **883.2** |

**Δ = +0.6** (well below +5 commit threshold; noise σ ≈ 4).

## Score arithmetic — why the aggregate didn't cross +5

| rule | shapes | weight | wall fraction | kernel Δ | per-shape progress Δ | weight·Δprogress contrib |
|---|---|---|---|---|---|---|
| Qwen3-GateUP fwd RCR | 4 | 1× | ~28 % | +1.68 % | ~0.0038 | 0.015 |
| DSV3-Down fwd RCR | 4 | 1× | ~31 % | +0.47 % | ~0.0012 | 0.005 |
| gpt_oss-GateUP dB var-K | 4 | **3×** | ~25 % | +0.98 % | ~0.002 | 0.024 |

Sum / 40 (total weight) = 0.0011 weighted_progress = **+1.1
expected score** — within metric noise envelope. Measured +0.6
matches expectation.

To cross +5 from this kind of stack we'd need either:
* ~10× more shapes covered (impractical — already covered most
  non-tuned families), or
* A single per-family kernel win in the 5-10 % range (none of
  the BF16 RCR / dB var-K probes this round or last has
  produced a >2 % uniform win since R20's dA RRR aggregate).

The R20/R24 wins worked because they covered 12-16 shapes at
+1-2 % AND ~50 % wall fraction (R20 dA RRR) or 12+ shapes with
3× weight (R24 dB var-K incl gpt_oss). R26's 12 shapes are
mostly 1× weight on smaller wall fractions — same expected
contribution as R25's lone 4-shape rule (~+0.5 score), just 3x
the rules.

## Workflow decision

**Reverted all 3 config changes** per the +5 threshold rule.
Both probe scripts archived for future use. The kernel-level
wins are real (every cell verified bit-eq, uniform-positive on
multiple seeds), they just don't aggregate enough to overcome
metric noise on a single-round test.

## Why this matters / what to try next

The big realization from R25 + R26: **the BF16 metric's noise
floor is ~±5 score, and 1× weight × 4-shape wins each contribute
only ~0.5 score after wall→ratio→progress dilution**. To clear
the noise threshold from BF16-only changes we now need EITHER:

1. **A single 5%+ kernel win on a hot path** — would require
   a structural kernel improvement (HK .cpp change), not a
   dispatch tweak. Candidates: K-tail forward kernel for
   gpt_oss B=32 (never structurally improved since R5);
   `bf16_transpose_3d` Triton kernel for H4 reroute (already
   at 5 TB/s effective per R4 — limited headroom).

2. **An aggregate of ~6+ uniform-positive rules** — exhausted
   the obvious surface. R20 + R24 + R26's probes have
   characterized every fwd RCR / dA RRR / dB var-K family in
   the BF16 metric. Remaining rules would need to come from
   cube / asymmetric / catch-all rule re-probes — same
   single-knob tweaks at the ~0.3-0.5 % kernel level (DSV3-Down
   was already that small).

3. **Cross-axis improvement** — e.g. inlining the H4 transpose
   into the kernel epilog so gpt_oss saves the ~5 % wall it
   currently spends in the standalone Triton transpose. This
   requires HK kernel work.

## Suggested R27 next step

* **R27 main line — K-tail forward kernel structural probe**:
  the gpt_oss B=32 K-tail kernel (K%128=64) has been stable
  since R5; never microbenchmarked against alternative
  scheduling at the kernel-source level. The bottom 4 metric
  shapes (gpt_oss B=32, ratio 1.05-1.09) have BWD dominating
  but FWD wall is also ~1.7-3.5 ms — even a 2-3 % FWD kernel
  improvement here would translate to +1-2 score given 3×
  weight. This requires reading
  `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  for the K-tail epilog and assessing whether the K=64 inner
  loop has obvious headroom (e.g. unrolling, prefetch
  scheduling).
* **R27 alt — selectively LAND just the gpt_oss-GateUP dB
  var-K rule**: of the 3 R26 changes, this is the highest
  per-shape weight (3×). Re-running with ONLY this rule (and
  many more metric samples to cut noise) might isolate its
  +0.7 expected contribution from the +0.6 noisy aggregate.
  But standing alone it's still ≪ +5 — not worth a commit.

## Files

* `scripts/_bf16_rcr_fwd_multi_family_probe.py` — 4-family fwd
  RCR probe. Reusable.
* `scripts/_bf16_vark_db_gpt_oss_gateup_probe.py` — gpt_oss-GateUP
  dB var-K probe. Reusable.
* `analysis/_notes/round-26-bf16-grouped-3-rule-aggregate-FALSIFIED-noise.md` — this note.
* No production change (config.py at HEAD 0c038f2).
