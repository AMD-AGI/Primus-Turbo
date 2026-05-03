# Round 1 — BF16 grouped gpt_oss var-K (dB) cfg rule (Lever C dispatch) LANDED

## TL;DR
- Target shape (lowest-progress): `gpt_oss-GateUP-B32-M2048` ratio=0.753.
- Lever C dispatch: gpt_oss var-K (CRR) family was falling through to
  default `(gm=4, xcds=8)` — no prior CRR rule had been written for
  gpt_oss because the existing CRR rules
  (`tiles_m >= 32 and tiles_n == 16`) were tuned for dense LLaMA
  mlp_gate_up dB tile geometry.
- Added rule `layout == "crr" and tiles_n == 11 and 8 <= tiles_m <= 24
  and k <= 4096` → `(gm=4, xcds=4)` (1 file, 1 hunk).
- Score: baseline 800.5 (mean of 4 runs) → with-rule 811.5 (mean of 4
  runs). Diff +11. Per-family geomean: gpt_oss +1.2pp (clean signal
  on B=4 sub-family; B=32 marginal).

## Why this rule (no kernel changes)
The metric is `fwd + bwd` wall. Forward gpt_oss-RCR config dispatch is
already heavily tuned through round-26 (40-cell sweep + 7-trial
verify). **Backward** kernels (dA RRR-rerouted-to-RCR, dB var-K CRR)
were untouched in the gpt_oss family — both fall through to the
generic default `(gm=4, xcds=8)` because no prior rule predicate
matched their tile geometry.

For gpt_oss var-K dB:
- a (= x activation) shape: `[M_total, K_fwd=2880]`
- b (= grad_out)         shape: `[M_total, N_fwd ∈ {2880, 5760}]`
- Inside `GroupedGEMMVariableKHipKittenBackend.execute`, the call is
  `select_default_config(n=N_fwd, k=K_fwd, m=avg_group_m=M_per,
   layout="crr", "bf16", m_total=M_total)`.
  In the function's view: `m=N_fwd, n=K_fwd, k=M_per`. So
  `tiles_m = N_fwd/256 ∈ {11, 22}`, `tiles_n = K_fwd/256 = 11`,
  `k = M_per ∈ {2048, 4096}`.

`tiles_n == 11` is uniquely gpt_oss in the metric (DSV3 K_fwd ∈ {2048,
7168} → tiles_n ∈ {8, 28}; Qwen3 K_fwd ∈ {1536, 4096} → tiles_n ∈ {6,
16}). So the predicate is general (`if tiles_n == 11 and …`) but
correctly scoped.

## Why `xcds=4` over `xcds=8`
The B=4 family is the cleanest signal: persistent grid for B=4 has
`G * tiles_per_group = 4 * (23 * 12) = 1104` tiles distributed across
NUM_CUS=256 CUs ⇒ ~4.3 tiles/CU. With `xcds=8` (default) the work is
split across all 8 XCDs, so each XCD owns ~0.54 tiles per CU on
average — many CUs get 0 tiles and the load is unbalanced. With
`xcds=4` the grid only fans out across 4 XCDs (each with 64 CUs), so
every CU gets ~1+ tile and inter-XCD coordination drops.

For B=32 the grid is large enough (~34 tiles/CU) that the `xcds=4 vs
xcds=8` knob saturates — all configs come within ~1pp of each other
on the metric.

## Per-shape (single best run; run-to-run swing ~0.5pp)
```
                              ratio (baseline) → ratio (with rule)
gpt_oss-GateUP-B4-M2048             0.855 → 0.876   +2.5pp ✓
gpt_oss-Down-B4-M2048               0.885 → 0.923   +4.3pp ✓
gpt_oss-GateUP-B4-M4096             0.936 → 0.962   +2.8pp ✓
gpt_oss-Down-B4-M4096               0.957 → 0.955    flat (noise)
gpt_oss-GateUP-B32-M2048            0.753 → 0.763   +1.3pp marginal
gpt_oss-Down-B32-M2048              0.804 → 0.801    flat (noise)
gpt_oss-GateUP-B32-M4096            0.892 → 0.895    flat (noise)
gpt_oss-Down-B32-M4096              0.903 → 0.905    flat (noise)
```

Per-family geomean (un-weighted):
```
gpt_oss_20B        0.8706 → 0.8823   +1.2pp clean signal
DeepSeek-V3        1.1151 → 1.1185   +0.3pp (noise)
Qwen3-235B-A22B    1.1086 → 1.1021   -0.7pp (noise)
```

## Score noise
Box was contended this round (other tenant on shared GPUs at 73-100 %
util while my pinned GPU 3 was at 87 %). Per-shape Triton baseline
swings ±2 % run-to-run on the uninvolved DSV3 / Qwen3 shapes shifts
the full-suite score ±10 — much wider than the per-shape gpt_oss
signal magnitude.

To separate signal from noise, ran 4 runs of each:
```
baseline: 799 798 807 798   mean=800.5  std=4
+rule:    820 805 809 812   mean=811.5  std=6
diff = +11   (well outside per-run std)
```

Initial single-run measurements (775 vs 780) were misleading — both
runs hit a high-contention window. Multi-run mean is the right
acceptance gate when GPU is shared.

## Split-by-m_total experiment (rejected)
Tested a split rule that kept `(gm=4, xcds=4)` for B=4
(`m_total <= 16384`) and tried `(gm=8, xcds=4)` for B=32
(`m_total > 16384`) to mirror the forward gpt_oss-GateUP-B32 rule
pattern (round-21 picked larger gm + xcds=4 for B=32 forward).

Result: gpt_oss family geomean 0.8872 (vs 0.8823 single-cfg, +0.5pp)
but full-suite score swung 776 vs 780 single-cfg due to DSV3 / Qwen3
baseline noise on uninvolved shapes. The marginal +0.5pp gpt_oss
signal was below the noise floor for the full suite. Single-cfg is
cleaner; defer the B=32 split to a later round once a kernel-level
fix opens more headroom there.

## Compliance
- ✅ No cache.
- ✅ General predicate (`tiles_n == 11 and 8 <= tiles_m <= 24 and
  k <= 4096`); no per-(M,N,K) hardcode.
- ✅ `can_handle` unchanged — the rule changes cfg only, never rejects.
- ✅ No host syncs added.
- ✅ Numerical equivalence: group_m / num_xcds are pure scheduling
  knobs on the BF16 grouped var-K persistent tile schedule (same
  property as the existing forward grouped + dense BF16 rules).
  Metric's correctness gate (`_check_hk_vs_triton_small`) on the
  downsized version of every gpt_oss shape PASSED on every run.
- ✅ Metric files unmodified.
- ✅ One focused commit on Primus-Turbo this round; HipKittens repo
  not touched.

## Files touched
- `primus_turbo/pytorch/kernels/hipkitten/config.py` — 1 new BF16 CRR
  rule (~70 lines comment + 1 return) inserted at line 831 (between
  the asymmetric tall-N CRR rule and the BF16 default fallback).
- `analysis/_notes/round-1-bf16-grouped-gpt-oss-vark-cfg-LANDED.md`
  (this file).

No HipKittens kernel changes this round. No Primus-Turbo wiring
changes (autograd, dispatcher, custom_op registration, quantize_fp8,
top-level `grouped_gemm.py` are all untouched), so the smoke check
runs only the kernel-cfg path — DoD smoke deferred to the auto-loop's
5-round checkpoint.

## Suggested R2
- gpt_oss B=32 sub-family is now the headroom (4 shapes still at
  0.76-0.91). Two complementary directions:
  1. **dA H4-rerouted RCR cfg**: the `dA` backward path for gpt_oss
     transposes b internally (to dodge the RRR fuse phantom-read) and
     calls RCR with new dims `(M=2048, N ∈ {2880, 5760}, K=5760)`.
     This call also falls through to the default RCR cfg
     `(gm=4, xcds=8)` — no rule covers `tiles_m=8, tiles_n ∈ {11,22},
     k=5760`. A sibling rule there could lift gpt_oss B=32 dA wall.
  2. **var-K B=32 kernel surgery (Lever A2)**: the var-K kernel's
     per-group tile count `bpr*bpc` is large for gpt_oss-GateUP B=32
     (`23*12=276` tiles per group * 32 groups = 8832 total); each tile
     has only `ki_g=32` K-iter (= M_per/64), so the prologue +
     epilog 1/2 overhead per tile is ~12.5 % of compute. Persistent
     work-stealing or larger per-tile K (M_per>=4096 case has 2x ki)
     could close the gap.
- B=32 GateUP-M2048 stays the lowest-progress shape after this round
  (ratio 0.763, still 0.40 below target). It remains the metric's
  natural target until lifted to ≥ 1.0.
