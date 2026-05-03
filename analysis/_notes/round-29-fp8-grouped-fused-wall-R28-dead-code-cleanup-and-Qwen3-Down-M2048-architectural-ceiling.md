# Round 29 — FP8 grouped fused-wall: R28 dead-code rule cleanup + Qwen3-Down M=2048 architectural-ceiling re-confirmation

**Date**: 2026-05-02 (auto_optimize R29/100, plateau patience 17/30)
**Selected lever**: refactor (R28 dead-code rule removal) + Qwen3-Down M=2048 falsification trail
**Score**: pre-cleanup median 988.5 / mean 990.9 (8 runs at HEAD `507aff37`)
            post-cleanup median 992 / mean 991.6 (8 runs at `<this commit>`)
            — both samples within R23-quantified noise band `[981, 1000]`; the +3.5 median
            shift is well within 1-σ of the 8-sample mean (~3 points) and reflects sampling noise,
            not a real change (the cleanup removes dead code with zero runtime effect).
**Primus-Turbo HEAD before / after**: `507aff37` / `<this commit>`
**HipKittens HEAD**: `4caa6d9a` (unchanged — no kernel change this round)

## TL;DR

R29 audited the `Round-28` rule added to the FP8 grouped RCR forward
dispatch in `primus_turbo/pytorch/kernels/hipkitten/config.py` and
**confirmed it is dead code** (never matches any metric or DoD shape).
The rule guarded `tiles_n == 6 and k == 4096` — but no shape in the
24-shape grouped FP8 metric or the 8-shape dense FP8 metric has
`N_fwd == 1536` (which is what `tiles_n == 6` requires). The R28 commit's
narrative ("Qwen3-GateUP forward RCR shape `N_fwd=1536`") was based on
a misreading: Qwen3-235B-A22B-GateUP forward actually has `N_fwd =
2 * moe_intermediate_size = 2 * 1536 = 3072` (`tiles_n=12`), not 1536.

R29 also re-probed every Qwen3-Down M=2048 forward / dA / dB var-K
`(group_m, num_xcds)` cell on the actual metric bottom shapes and found
**no robust kernel-only lever** — every coarse 200-iter sweep signal
collapses to ±0.5 % under tight 400-iter × 12-trial × 3-seed verify.
The architectural ceiling on Qwen3-Down M=2048 (worst metric ratio,
1.243 on this round's baseline) holds: it is bounded by the symmetric
`quantize_fp8` HBM tax both backends pay equally — same finding as R8,
R26, R28.

This round ships:
1. Removal of the R28 dead rule from `select_default_config` + a
   replacement comment block documenting the falsification.
2. This round note.

No HipKittens commit. No backward-path change. Metric distribution
unchanged within noise.

## R29 baseline metric (HEAD `507aff37`, before any change)

8 metric runs at `_metric_grouped_fused_wall.py`:

```
981  986  988  988  989  997  998  1000   sorted
median 988.5    mean 990.9    range 19
```

**Bottom 8 shapes (consistent across runs)** — example from run 1
(score=981):

```
Qwen3-235B-A22B-Down-B16-M2048      ratio 1.243
Qwen3-235B-A22B-Down-B16-M4096      ratio 1.255
Qwen3-235B-A22B-GateUP-B16-M2048    ratio 1.262
gpt_oss_20B-Down-B32-M2048          ratio 1.262
Qwen3-235B-A22B-Down-B32-M2048      ratio 1.265
gpt_oss_20B-Down-B4-M2048           ratio 1.266
Qwen3-235B-A22B-GateUP-B16-M4096    ratio 1.286
Qwen3-235B-A22B-GateUP-B32-M2048    ratio 1.282
```

The actually-worst shape across the run distribution is
**`Qwen3-235B-A22B-Down-B16-M2048`** (ratio ≤ 1.249 in 6/8 runs). It
is one of only **2 metric shapes** that fall through to the FP8
binding default `(group_m=4, num_xcds=None=8)` — the other being its
B=32 sibling. Those are the shapes R28 *should* have targeted; R28
instead added a rule for a non-existent N_fwd=1536 shape.

## R29 audit — R28 rule is dead code

R28 commit `507aff37 perf(fp8-grouped-gemm): R28 — FP8 RCR fwd
Qwen3-GateUP N=1536 rule (4,4) — closes metric-shape gap left by R7`
added the following branch at line 1575 of
`primus_turbo/pytorch/kernels/hipkitten/config.py`:

```python
if (
    tiles_n == 6
    and k == 4096
    and m_total is not None
    and m_total >= 32768
):
    return HipKittenConfig(
        layout=layout, group_m=4, num_xcds=4, kernel=None,
    )
```

The branch is on the `if layout == "rcr":` block of `select_default_config`
(reachable only from forward RCR dispatch and dA RRR with trans_b=True
re-routing — the latter goes through `layout == "rrr"`, not "rcr").
For any metric shape to match, we'd need `N_fwd == 1536`. **No shape
in either the 24-shape grouped FP8 metric or the 8-shape dense FP8
metric has `N_fwd == 1536`**:

```
Grouped FP8 metric (24 shapes):
    DeepSeek-V3 N_fwd ∈ {4096 (GateUP), 7168 (Down)}     → tiles_n ∈ {16, 28}
    gpt_oss_20B N_fwd ∈ {5760 (GateUP), 2880 (Down)}     → tiles_n ∈ {22, 11}
    Qwen3-235B  N_fwd ∈ {3072 (GateUP), 4096 (Down)}     → tiles_n ∈ {12, 16}

Dense FP8 metric (8 shapes — Llama-2-7B / Llama-3.1-8B):
    N_fwd ∈ {4096, 6144, 12288, 22016, 28672}            → tiles_n ∈ {16, 24, 48, 86, 112}
```

R28 commit message claims:

> the metric uses the new shape N_fwd=1536 / K_fwd=4096 (tiles_n=6)

This is incorrect. **N_fwd=1536 corresponds to Qwen3-Down's K_fwd value,
not Qwen3-GateUP's N_fwd.** The R28 author confused
`moe_intermediate_size = 1536` with `N_fwd = 2 * moe_intermediate_size =
3072` (the GateUP layer's output dim per
`benchmark/ops/config.py::_generate_moe_test_cases`, line 275-277).

The Qwen3-GateUP M=2048 family (the shapes R28 attempted to optimize)
already has the **R7 rule** at line 1498 (`tiles_n == 12 and tiles_m == 8
and k == 4096` → `(gm=16, xcds=4)`), which was the same `(num_xcds=4)`
direction R28 picked. R28's `(gm=4, xcds=4)` rule never fired and
never overrode anything.

R28's reported `+2.1 mean / +3 median` improvement (12-run distribution
992.4 → 994.5) was random noise within the wide [981, 1000] band — the
same band re-quantified at this round's `507aff37` baseline (8 runs:
median 988.5 / mean 990.9). Random sampling from the same distribution
trivially produces ±5 mean shifts.

### Verification trace (this round)

```
Trace - which shapes match R28 rule (tiles_n==6, k==4096) ?
    Grouped FP8: 0 of 24 metric shapes
    Dense   FP8: 0 of  8 metric shapes
    DoD smoke shapes (test_dod_smoke.py): 0 hits
    HK FP8 grouped fwd+bwd DoD shapes: 0 hits

Conclusion: R28 rule never fires. Pure dead code.
```

After dead-rule removal, all 24 grouped FP8 metric shapes still
dispatch to bit-identical configs (verified by re-running the trace
script — see commit message). The default-fallthrough Qwen3-Down M=2048
shapes (B16, B32) still go to the FP8 binding default just as they
did pre-cleanup.

## R29 falsification trail — Qwen3-Down M=2048 has no robust kernel lever

R29 re-probed every plausible (gm, num_xcds) cell for the 3 GEMM
components used by Qwen3-Down M=2048 in the metric's fwd+bwd wall:

### Forward RCR (n=4096, k=1536, tiles_n=16, tiles_m=8)

9-cell sweep × 7 trials × 200 iters at `/tmp/probe_qwen3_down_m2048_fwd_r29.py`:

```
shape Qwen3-Down-B16-M2048 (B=16, M=2048):
    cfg          TFLOPS     Δ vs default (4, 0=8)
    (4, 0)def    1805.03    baseline
    (4, 4)       1775.88    -1.61 %
    (1, 4)       1781.71    -1.29 %
    (8, 4)       1788.80    -0.90 %
    (2, 8)       1724.97    -4.44 %
    (16, 4)      1761.13    -2.43 %
    (32, 4)      1784.86    -1.12 %
    (2, 4)       1776.96    -1.56 %
    (4, 8)       1819.54    +0.80 %     <- = default (xcd=0 falls back to 8)

shape Qwen3-Down-B32-M2048 (B=32, M=2048):
    same picture: default (4, 0) wins by 0.1-2.5 % over every alternative.
```

Every alternative cell regresses 1-4 % vs the binding default. The
+0.80 % on `(4, 8)` is the within-noise duplicate of the default —
both `(4, 0)` and `(4, 8)` produce identical kernel behavior because
the binding's `xcds=0` falls back to `BLOCK_SWIZZLE_NUM_XCDS=8`.

**Forward RCR for Qwen3-Down M=2048 is at the local optimum at the
binding default.** No rule needed.

### dA RRR (n=K_fwd=1536, k=N_fwd=4096, tiles_n=6)

Coarse probe at `/tmp/probe_qwen3_down_da_rrr_r29.py` flagged
`(gm=8, xcds=4)` as +1.38 % vs R42's `(gm=16, xcds=4)` on B16.

**Tight verify** (`/tmp/verify_qwen3_down_da_rrr_r29.py`, 400-iter ×
12-trial × 3-seed p17 median, mirror of R28 verify methodology):

```
shape                          (16, 4) baseline    (8, 4) proposed   Δ med
Qwen3-Down-B16-M2048-dA        2041.9 TF           2043.1 TF         +0.06 %
Qwen3-Down-B16-M4096-dA        2103.5 TF           2104.3 TF         +0.04 %
Qwen3-Down-B32-M2048-dA        2083.6 TF           2083.7 TF         +0.01 %
Qwen3-Down-B32-M4096-dA        2122.9 TF           2129.4 TF         +0.31 %
DSV3-Down-B16-M2048-dA         2370.3 TF           2361.1 TF         -0.39 %
DSV3-Down-B16-M4096-dA         2413.3 TF           2430.6 TF         +0.72 %
DSV3-Down-B32-M2048-dA         2383.4 TF           2381.6 TF         -0.08 %
DSV3-Down-B32-M4096-dA         2424.1 TF           2438.5 TF         +0.59 %
```

The coarse-probe +1.38 % on B16 collapses to +0.06 % under tight
verify — pure measurement noise. The R42 `(gm=16, xcds=4)` rule is
the robust optimum for the entire `tiles_n <= 8 + m_total >= 32768`
band (Qwen3-Down + DSV3-Down dA). **No rule change.**

### dB var-K CRR (n=K_fwd=1536, k=N_fwd=4096, tiles_n=6, tiles_m=16)

Coarse probe at `/tmp/probe_qwen3_down_var_k_r29.py` flagged
`(gm=2, xcds=4)` as +2.50 % vs the inline rule's `(gm=8, xcds=4)`
on Qwen3-Down B16-M2048-dB.

**Tight verify** (`/tmp/verify_qwen3_down_var_k_r29.py`, same
methodology):

```
shape                          (8, 4) baseline    (4, 4)         (2, 4)         (1, 4)         (16, 4)
Qwen3-Down-B16-M2048-dB        1925.4 TF          +0.35 %        +0.24 %        -0.82 %        +0.11 %
Qwen3-Down-B32-M2048-dB        1977.1 TF          -0.19 %        -0.18 %        -0.69 %        -0.72 %
Qwen3-Down-B16-M4096-dB        2302.3 TF          -0.21 %        -0.16 %        -0.47 %        -0.27 %
Qwen3-Down-B32-M4096-dB        2325.1 TF          -0.45 %        -0.20 %        -0.66 %        -0.60 %
DSV3-Down-B16-M2048-dB         1982.6 TF          -0.65 %        -0.45 %        +0.99 %        +0.49 %
DSV3-Down-B32-M2048-dB         2005.8 TF          -1.10 %        -1.00 %        +0.31 %        +0.35 %
```

The coarse +2.50 % on B16-M2048-dB collapses to +0.24 % under tight
verify. No cell wins all 4 Qwen3-Down shapes without regressing
DSV3-Down. The current inline rule `(gm=8, xcds=4)` for `m_total >=
16384` is the robust universal optimum. **No rule change.**

Identical conclusion to R8 probe 4 (`var-K (group_m, num_xcds)
refinement: NEAR-OPTIMAL`) and confirms the architectural ceiling —
Qwen3-Down M=2048's fwd+bwd wall ratio gap to the 1.35 target is
bounded by the **symmetric `quantize_fp8` HBM tax both backends pay
equally**, not a kernel-internal scheduling miss.

## What this round ships

### Files touched

* `primus_turbo/pytorch/kernels/hipkitten/config.py`
  - Removed the R28 if-branch at line 1575-1637 (66 lines including
    comment block).
  - Replaced with a 50-line documenting comment explaining the
    falsification, the R29 trace, and why Qwen3-Down M=2048 stays
    on the binding default.
* `analysis/_notes/round-29-fp8-grouped-fused-wall-R28-dead-code-cleanup-and-Qwen3-Down-M2048-architectural-ceiling.md`
  - This round note.

### Behavior preserved

* All 24 grouped FP8 metric shapes still dispatch to bit-identical
  `HipKittenConfig` returns (verified by re-running the dispatcher
  trace pre + post; all 22 `CUSTOM` rules + 2 `(default)` rules
  unchanged).
* No HipKittens kernel change; no backward-path change.
* No `quantize_fp8` / autograd / dispatcher change — DoD smoke not
  required.

### Metric distribution

```
pre  (HEAD 507aff37, 8 runs):  981 986 988 988 989 997 998 1000   med 988.5  mean 990.9
post (this commit, 8 runs):    981 986 989 992 992 997 998  998   med 992    mean 991.6
```

Distributions are statistically identical (Mann-Whitney would show
p > 0.5; the +3.5 median shift is within 1-σ of the 8-sample mean).
The cleanup is a refactor with zero runtime effect.

## Next-round recommendation

R30 should NOT spend another round on Qwen3-Down M=2048 / kernel-only
config tuning — that lever class is now formally exhausted (R8
analytical, R29 empirical, this note).

The remaining unspent levers are the same as R28-doc identified:

1. **HK kernel surgery (Path A — fused activation BF16→FP8 cvt
   inside the grouped kernel)** — closed at R8 / R7 with -26 % wall
   regression on forward DTR + LDS attempt. Task body's Phase 1
   plan exists; could be re-attempted with a different load
   primitive (e.g. DTL-compatible cvt path) but is multi-round and
   high-risk.

2. **C++ `quantize_fp8_tensorwise` HBM bandwidth lift** (R8 direction
   2). Currently 67 % of MI355X HBM peak; pushing to 80 % (NVIDIA TE
   level) would lift the metric geomean by ~+1.5 % on the symmetric
   tax — saves ~+2 score points across both backends. Out of HK
   scope; would need C++ extension work in
   `primus_turbo_cpp_extension.quantize_fp8_tensorwise`. Out of HK
   scope but available.

3. **Acknowledge plateau and let patience expire** (12 rounds left
   in current 30-round patience window). Doc-only / refactor rounds
   like R26 / R28-doc / R29 are the legitimate consequence of being
   at the architectural ceiling.

If R30 is a fresh-cold-start chat-window expiry, it should read this
note + R8 + R26 + R28-doc to understand the plateau before attempting
any new kernel-internal rule.

## Round meta

| Field | Value |
|---|---|
| HK SHA before / after | `4caa6d9a` / `4caa6d9a` (unchanged) |
| PT SHA before | `507aff37` |
| PT SHA after  | (this commit) |
| Forward+backward wall metric, pre  (8 runs) | med 988.5  mean 990.9 |
| Forward+backward wall metric, post (8 runs) | med 992    mean 991.6 |
| Patience increment | +1 (will be 18 after this round) |
| Best metric ever | 1000 (cap, hit twice in R27 run distribution; once at HEAD pre-revert) |
| R28 dead-rule status | REMOVED + falsification documented |
| Qwen3-Down M=2048 fwd RCR cell sweep | binding default optimal (R29 probe) |
| Qwen3-Down M=2048 dA RRR cell sweep   | R42 (gm=16, xcds=4) robust optimum (R29 tight verify) |
| Qwen3-Down M=2048 dB var-K cell sweep | inline (gm=8, xcds=4) robust optimum (R29 tight verify) |
| Architectural ceiling | symmetric `quantize_fp8` HBM tax bounds Qwen3-Down M=2048 wall ratio |

## Falsifications recorded this round

1. **R28 rule (`tiles_n == 6 and k == 4096`)** — never fires on any
   metric or DoD shape. R28 commit was based on a misreading
   (confused Qwen3 `moe_intermediate_size = 1536` with the GateUP
   layer's `N_fwd = 2 * moe_int = 3072`). Removed.

2. **Qwen3-Down M=2048 forward RCR (gm, xcd) tuning** — coarse 9-cell
   probe shows binding default beats every alternative by 1-4 %. No
   rule needed.

3. **Qwen3-Down M=2048 dA RRR (gm=8, xcds=4) candidate** — coarse
   +1.38 % on B16 collapses to +0.06 % under tight verify. R42
   `(gm=16, xcds=4)` retained.

4. **Qwen3-Down M=2048 dB var-K (gm=2, xcds=4) candidate** — coarse
   +2.50 % on B16-M2048 collapses to +0.24 % under tight verify.
   Inline rule `(gm=8, xcds=4)` retained.

5. **Qwen3-Down M=2048 architectural ceiling** — re-confirmed (R8 /
   R26 / R28). Wall ratio gap to 1.35 is bounded by the symmetric
   `quantize_fp8` HBM tax, NOT a kernel-internal scheduling miss
   that a Primus-side rule can close.

## DoD smoke status

Not run this round (no shared-code change — only `config.py` rule
removal; `select_default_config` returns are bit-identical for every
shape covered by DoD smoke and grouped FP8 metric).

Last DoD run was at SHA `7f0b5a8c` per the auto_optimize prompt
(score 608). That score is unchanged at this commit because no DoD
shape's dispatch changed.
