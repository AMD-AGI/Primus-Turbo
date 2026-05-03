# Round 25 — BF16 grouped fwd RCR Qwen3-GateUP (gm=1, xcds=4) FALSIFIED (noise-bound)

## Goal

After R24 landed the dB var-K 4-rule aggregate (+5.4 mean delta),
R25 audited which **fwd RCR** dispatches were still on the BF16
binding default `(gm=4, xcds=8)`. The audit (corrected for the
dB var-K dispatcher's reordered args) revealed a much larger
attack surface than expected — **6 metric shapes had their
forward leg on default**:

  - 4 Qwen3-GateUP shapes (B ∈ {16, 32}, M_per ∈ {2k, 4k},
    n=3072 ∧ k=4096; tiles_n=12 unique to this family in the
    BF16 metric).
  - 2 Qwen3-Down shapes M_per=2048 (the M_per=4096 shapes hit
    the cube rule (gm=2, xcds=32)).

Per R24's recommendation, this is a higher-leverage surface than
the dB var-K side — fwd is the largest single wall fraction
(~28-50 % of fwd+bwd; for B=32 GateUP measured at 28 %, for
B=16-M=2048 closer to 40 %). Even small per-family kernel wins
should register more strongly than dB var-K at the metric level.

## Probe — `scripts/_bf16_rcr_fwd_qwen3_probe.py`

For each family, 14 (gm, xcds) cells × 5 trials × 200 iters
on the BF16 grouped RCR kernel directly (no autograd warmup
needed — Qwen3 K_fwd ∈ {1536, 4096} both clean modulo 128/256,
no K-tail cold-start issue per R22).

### Qwen3-GateUP (n=3072, k=4096) — UNIFORM POSITIVE

| shape | prod (4,8) TF | best TF | Δ |
|---|---|---|---|
| B16-M2048 (mt=32768)  | 1330.4 | 1349.8 | +1.47 % |
| B16-M4096 (mt=65536)  | 1373.4 | 1394.5 | +1.54 % |
| B32-M2048 (mt=65536)  | 1352.6 | 1377.0 | +1.80 % |
| B32-M4096 (mt=131072) | 1377.7 | 1403.7 | +1.88 % |

Best cell: **`(gm=1, xcds=4)`** — uniform-positive avg +1.68 %,
range [+1.47 %, +1.88 %]. Bit-equivalent vs (gm=4, xcds=4)
(max_abs=0, bit_eq=True). Beats next-best cells (gm ∈ {8, 16, 32},
xcds=4) by +0.3-0.6 pp.

### Qwen3-Down M=2048 (n=4096, k=1536) — sub-noise

Best uniform-positive cell `(gm=4, xcds=0)` only +0.05 % avg
(range [+0.02 %, +0.08 %]) — sub-noise; not worth a rule (and
xcds=0 carries the same R24 Triton-allclose drift risk that
killed the DSV3-GateUP dB var-K cell).

## Attempt — single Qwen3-GateUP rule

Inserted before the cube rule (line 691):

```python
if (layout == "rcr"
        and tiles_n == 12
        and k == 4096
        and m_total is not None):
    return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
```

Predicate uniqueness verified: `tiles_n == 12 ∧ k == 4096`
matches only Qwen3-GateUP fwd in the BF16 metric (no other
family has n=3072; dense LLaMA RCR n ∈ {4096, 8192, 11008,
14336, 22016, 28672} all ≠ 3072 anyway). `m_total is not None`
excludes dense callers.

Dispatch table verification confirmed: all 4 Qwen3-GateUP fwd
shapes route to (gm=1, xcds=4); Qwen3-Down/DSV3/dense controls
unchanged.

## Verification (head-to-head 5-run means, same GPU, same session)

| | run1 | run2 | run3 | run4 | run5 | mean |
|---|---|---|---|---|---|---|
| baseline (HEAD 0569685) | 879 | 887 | 879 | 880 | 880 | **881.0** |
| after-rule              | 879 | 880 | 881 | 880 | 881 | **880.2** |

**Δ = -0.8** (sub-noise; well below the +5 commit threshold).

The kernel-only +1.68 % avg failed to register at the metric:

* fwd wall fraction for Qwen3-GateUP B=32-M=4096 ≈ 28 %, smaller
  for the B=16 brackets but B=16 has only B=16 weight units.
  Best-case wall delta: 1.68 % × 28 % ≈ 0.47 % per shape.
* 4 shapes × 1× weight = 4/40 weight units lifted by ~0.005
  ratio = ~0.004 progress = ~+0.4 score.
* Metric noise floor: ±5 score (per the 5-run baseline range
  879-887 here, σ ≈ 3.2). The expected +0.4 lift is ~8x below σ.

## Workflow decision

Per the round rules ("metric must improve by +5 to commit"),
**reverted the rule**. The probe script (`scripts/_bf16_rcr_fwd_qwen3_probe.py`)
and the falsification log are kept for future rounds (e.g. if
the same rule could be combined with 2-3 other forward-side
wins in an R20-style aggregate to clear the threshold).

## Why this matters / what to try next

* **The kernel signal is real and substantial** (+1.68 % avg
  with min +1.47 % is the *largest* per-family kernel-only win
  identified in any round so far for a fwd RCR family). It just
  doesn't aggregate to enough wall delta on a single 4-shape /
  4-weight family.
* The mechanism here is the same as R22/R23 (single-family wins
  drowning in metric noise), but R24's aggregate-of-5 trick
  needs MORE families to work on the fwd side because:
  - Each fwd-only family covers exactly 4 metric shapes (no 3x
    weight bonus like gpt_oss has on dB).
  - Fwd wall fraction (~28 %) is similar to dB var-K (~25 %),
    so per-family score lift is comparable.
  - Need ~4-5 fwd families (16+ shapes) to cross +5.

## Suggested R26 next step

**Find more fwd RCR families on default + bundle with Qwen3-GateUP**:

* Audit the current fwd RCR cells for **DSV3-GateUP M=2048**
  (B=16, B=32 — both currently `(gm=1, xcds=4)` — already at
  the same cell as Qwen3-GateUP's best, so probably no headroom),
  **DSV3-Down all 4** (currently `(gm=16, xcds=2)` from R5 —
  could probe for a better cell).
* The 2 Qwen3-Down M=2048 shapes still on default have only
  +0.05 % headroom — drop.
* If DSV3-Down fwd has a uniform-positive cell ≥ +0.5 %
  (8 weight units at 1× weight), bundling with Qwen3-GateUP's
  +1.68 % would produce a 12-weight aggregate ≈ +3-4 score —
  still below threshold. Need to find at least one MORE family
  with ≥ +0.5 % to bundle.
* **Alternative angle**: explore the H4 transpose path itself.
  R4's `bf16_transpose_3d` Triton kernel already at 5 TB/s, but
  for the gpt_oss-Down B=4 shapes the transpose is still 5 % of
  wall — revisit whether a fused transpose-into-load could
  shave that. (Out of scope for R26 unless aggregate fails.)
* **Stick with aggregates**: the R20/R24 pattern works when
  enough shapes are stacked. R26 should be focused on building
  the next 4-5 family fwd RCR aggregate.

## Files

* `scripts/_bf16_rcr_fwd_qwen3_probe.py` — archived for re-use.
* `analysis/_notes/round-25-bf16-grouped-fwd-rcr-qwen3-gateup-FALSIFIED-noise.md`
  — this note.
* No production change (config.py reverted to HEAD 0569685).
