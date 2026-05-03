# Round 47 — BF16 grouped, per-tile arithmetic hoist in `grouped_var_k_kernel` — FALSIFIED

## Goal coming in

R46 falsification (conditional `s_barrier` is load-bearing; correctness
breaks if removed) recommended R47 attack the next R43-listed
structural opt:

> R47 surface ranked: (1) per-tile arithmetic hoisting (lowest
> correctness risk; cache m_start_g / M_g / ki_g / k_offset_tiles per
> group_idx via the same pattern grouped_kernel uses for b_srsrc_curr
> at line 3899)…

Mirroring the existing `b_srsrc_curr` hoist pattern from
`grouped_kernel` (line 3899-3906), the var-K kernel's per-`group_idx`
quantities — `m_start_g` (LDS read), `M_g` (LDS read + sub),
`ki_g` (`M_g >> 6`), `k_offset_tiles` (`m_start_g >> 6`) — could be
cached on `group_idx` and recomputed only when the persistent loop
crosses a group boundary.

## Hypothesis (R47)

When the persistent loop's `gt += NUM_CUS` step keeps the CU inside
the same group, the LDS read + 2 sub + 2 shr-by-6 collapses to a
single `s_cmp_eq` cache-hit branch. For the dB var-K geometries:

| Family | tiles_per_group | NUM_CUS / tiles_per_group |
|---|---:|---:|
| gpt_oss-Down (tiles_m=11, tiles_n=11) | 121 | 2.12 |
| gpt_oss-GateUP (tiles_m=22, tiles_n=11) | 242 | 1.06 |
| Qwen3-Down (tiles_m=16, tiles_n=6) | 96 | 2.67 |
| DSV3-Down (tiles_m=28, tiles_n=8) | 224 | 1.14 |
| DSV3-GateUP (tiles_m=16, tiles_n=28) | 448 | 0.57 → cache hits half the iters |

Pre-build expectation: the hoist helps DSV3-GateUP (cache hits ~half),
marginal on GateUP-style (0.94 cache-hit prob), and *negative* for
Down-style families (group_idx changes by ~2 per iter, so the cache
miss path always runs and the `if (group_idx != last_group_idx)`
branch becomes pure overhead).

## v1 attempt

Added 4 cached SGPRs (`last_group_idx`, `m_start_g_cached`,
`ki_g_cached`, `k_offset_tiles_cached`) above the loop, and gated the
LDS read + arithmetic on `group_idx != last_group_idx`. `k_offset_tiles`
moved up from below the dispatch swizzle to the cache block.

Build resource (clean, no spill):
```
                                    baseline   R47 v1
grouped_var_k_kernel<0>: VGPRs       256        256
                         TotalSGPRs   95         98   (+3 cached SGPRs)
                         SGPRs Spill  0          0
                         VGPRs Spill  0          0
                         Occupancy    2 waves    2 waves
```

Same occupancy. The +3 SGPRs is exactly the cached state (last_group_idx
+ 2 cached values; `M_g` was kept as a local `const int` since the
compiler could re-derive it from cached state).

## Evidence (correctness + perf)

```
metric run on R47 v1:
                              run-1    run-2    baseline
score                         879      879      880      (sub-noise)
gpt_oss_20B    geomean        1.0854   1.0864   1.0886   (-0.2..-0.3 pp)
DeepSeek-V3    geomean        1.1192   1.1233   1.1209   (drift ±0.2 pp)
Qwen3-235B-A22B geomean       1.1144   1.1128   1.1129   (~flat)
correct_fail                  0/24     0/24     0/24
```

Per-shape gpt_oss B=32 (the var-K-dominant subset) ratios across runs:

```
                          baseline → R47 v1 run-1 → R47 v1 run-2
GateUP-B32-M2048           1.056      1.047           1.052          (avg -0.06)
Down-B32-M2048             1.059      1.050           1.049          (avg -1.0pp)  *consistent regression*
GateUP-B32-M4096           1.087      1.091           1.087          (flat)
Down-B32-M4096             1.080      1.079           1.084          (flat)
```

`gpt_oss-Down-B32-M2048` HK TFLOPS: 1082.8 → 1079.4 → 1077.8 — both
runs land 3-5 TF below baseline, in the same direction. This is right
at the noise floor but is a directional regression on the second-worst
shape.

## Mechanism — why the hoist didn't help (and slightly hurt)

The pre-build math from the table above bears out:

* **gpt_oss-Down** (the regressing shape): tiles_per_group=121, so
  group crossings per iter ≈ 2.1 — the cache HIT rate is ~0%
  (group_idx changes by 2 most iters, so even if iter `i` was group X,
  iter `i+1` is group X+2, never X). The `if (group_idx !=
  last_group_idx)` branch always falls through to recompute, AND there's
  the extra branch overhead per iter.
* **gpt_oss-GateUP**: cache hit rate ~6 % (group crossings per iter
  ≈ 1.06). Sub-noise positive at most.
* **Other families**: not in the gpt_oss family weight, so even a
  small per-shape win (DSV3-GateUP could see ~50 % cache hits)
  contributes <0.1 score after weight dilution.

The compiler was already doing a great job: looking at the resource
usage, there's no spill on either baseline or v1, and SGPRs are
consistently 95-98 — the cached SGPRs are fungible, the compiler
already had room. The `s_offs[group_idx]` read goes through LDS which
is single-cycle on cache hit, so the "save" was illusory.

## Falsification consequence

R47 closes:

* **Per-tile arithmetic hoisting (R43-listed item 1)** for the var-K
  kernel. The hoist provides no measurable benefit because
  `tiles_per_group < NUM_CUS` for every gpt_oss dB var-K shape, so
  cache hits on `group_idx` are rare (≤6%) or zero. On Down-style
  shapes (tiles_per_group=121, group strides every iter) the added
  branch overhead causes a small but consistent regression
  (~0.3-0.5pp HK TF on `gpt_oss-Down-B32-M2048`).

R47 does NOT close:

* **Var-K KI specialisation (R43-listed item 4 / R47 backup #2)**.
  Currently `grouped_var_k_kernel<0>` is the only instantiation
  (KI=0 dynamic). Specialising for `KI=32` (gpt_oss + DSV3-Down +
  Qwen3-Down K_var=2048 → ki_g=32) and `KI=64` (M=4096 variants
  → ki_g=64) lets the compiler unroll the K-loop completely. This
  is the largest remaining lever.
* **Store batching 4 → 1 (R43-listed item 3 / R47 backup #3)**.
  Combine the 4 `store_c_tile_mn_masked_grouped` calls into one
  larger masked-store. Reduces per-tile bounds-check overhead.
  Highest correctness validation cost.
* **Other dispatch retries**: DSV3-GateUP dB var-K (R45 backup) — R24
  dropped this family due to xcds=0 allclose drift; could re-sweep
  with xcds ∈ {1, 2, 4, 8}.

## Action

* HipKittens: `kernel_bf16_dynamic.cpp` modified then reverted via
  StrReplace; final diff = 0 (rebuilt to confirm).
* Primus-Turbo: 1 commit (this falsification note).
* HipKittens: no commit (kernel reverted to baseline state).

## R48 next-action surface

Two candidates remain in the R43 var-K structural surface plus the
R45 dispatch backup:

1. **Var-K KI specialisation** (R43 item 4, R47 backup #2). For
   gpt_oss alone: `ki_g ∈ {32, 64}`. Need to specialise carefully to
   avoid the spill regressions R39-R42 saw on the forward kernel
   (16-30 VGPR spills with full-unroll on small KIs). Risk-mitigation
   plan: build with `KI=32` only first, check spill counts; if clean,
   add `KI=64`; if spill on any, fall back to `#pragma unroll 4`
   instead of full unroll.

2. **DSV3-GateUP dB var-K dispatch retry** (R45 backup). R24 dropped
   `xcds=0` due to allclose drift; re-sweep with `xcds ∈ {1, 2, 4, 8}`
   on (tiles_m=16, tiles_n=28) cells. Smaller upside (DSV3-GateUP is
   already 1.13-1.15) but cleanest dispatch surface remaining.

3. **Store batching 4 → 1** (R43 item 3, R47 backup #3). Highest
   correctness validation cost; defer until R49+ unless (1) and (2)
   both falsify.

Recommended: start R48 with (1). KI specialisation is the largest
upside remaining; the spill risk is real but bounded (R39-R42 spills
were on the FORWARD kernel which has a more complex schedule;
`grouped_var_k_kernel` currently has 0 spill at KI=0 so there's
headroom).
