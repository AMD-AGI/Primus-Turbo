# Round 66 — BF16 grouped, dB var-K cfg audit (gpt_oss-GateUP tiles_m=22) — CLOSED, no change

## Goal coming in

R65 (HK SHA `9a860d59`, PT `d4c2b47`) closed the per-XCD work-stealing +
extended gate lever as FALSIFIED (Δ_mean -1.4, neutral; +5/+20 VGPR
spill on hot KI=48/64/88 RCR / cold KI=88 CRR; per-XCD didn't tame R62-B
variance because the variance was not primarily atomic-contention-driven;
tiles=1472 working set still exceeds 4 MB per-XCD L2 partition).

After 6 consecutive falsified rounds (R60-R65) the metric plateau was
solidly stuck at 874-880, and R65's next-action surface ranked the **dA
backward audit** as the recommended R66 lever — pivot off the saturated
forward-kernel surface to a cfg-level audit of the dB var-K (CRR) family.

The R66 specific hypothesis under test:

> The R1 umbrella rule (`config.py:1203`, `tiles_n=11 + 8<=tiles_m<=24
> + k<=4096`) returns `(gm=4, xcds=4)`. R24's commit message claims it
> aggregates "5 family-specific rules" but the actual carve-outs cover
> tiles_m ∈ {11 (gpt_oss-Down), 16 (Qwen-Down), 12 (Qwen-GateUP), 28
> (DSV3-Down)} — leaving **tiles_m=22 (gpt_oss-GateUP dB var-K)** on
> the umbrella default. The R1 umbrella was tuned by sweeping the
> 8-shape gpt_oss SUITE; gpt_oss-Down (tiles_m=11) drove the choice.
> R24's data shows tiles_m=11 prefers `(gm=1, xcds=4)` (+1.52% avg).
> tiles_m=22 may benefit from the same split.

The dB var-K (CRR) layout is dispatched via `select_default_config(
n_fwd, k_fwd, _avg_group_m(M_total, B), "crr", "bf16", m_total=M_total)`.
For gpt_oss-GateUP: n_fwd=5760, k_fwd=2880, avg_m∈{2048, 4096}, M_total
∈ {8192, 16384, 65536, 131072}. Maps to `tiles_m=22, tiles_n=11,
k=avg_m∈{2048,4096}, m_total>0`.

## What was actually changed

**`primus_turbo/pytorch/kernels/hipkitten/config.py`** — temporarily
inserted a tiles_m=22 carve-out before the R1 umbrella, returning
`(gm, xcds)` from a candidate set, then reverted.

No HipKittens / kernel changes. No bench-relevant code touched.

## Per-cfg paired metric tests (HK / Triton score, MI355X GPU 3, idle)

### Phase 1 — Mirror Down split: candidate `(gm=1, xcds=4)`

5-sample paired alternating test, A = with split, B = revert (umbrella):

```
round 1: A=880  B=880
round 2: A=878  B=874
round 3: A=880  B=881
round 4: A=875  B=880
round 5: A=874  B=875
mean    : 877.4 vs 878.0  → Δ -0.6
range   :  6     7
all 24 shapes correctness PASS in every sample
```

Δ_mean = -0.6 (split LOSES by 0.6 within ±5 noise floor) → NEUTRAL.
Variance is matched (range 6 vs 7) — no tail-shift artifact either.

### Phase 2 — Sweep alternative cfgs against the umbrella baseline (878.0)

3-sample-per-cfg quick scan to detect any candidate that beats baseline
by more than the +5 commit threshold:

```
cfg(gm,xcds)   sample1  sample2  sample3  mean   Δ vs 878.0
( 2, 4)         874     874     877      875.0  -3.0
( 8, 4)         878     875     881      878.0   0.0  *tied
(16, 4)         873     875     880      876.0  -2.0
( 1, 8)         873     871     878      874.0  -4.0
( 2, 8)         872     877     873      874.0  -4.0
```

NO candidate beats the umbrella `(gm=4, xcds=4)` baseline. `(gm=8,
xcds=4)` ties at 878.0 mean, but the umbrella is structurally simpler
(no carve-out adds a code path) so the tie favors keeping umbrella.

Combined with Phase 1's `(1, 4)` neutral, the **6-cfg local search**
around `(4, 4)` finds no winner → tiles_m=22 IS at the umbrella's local
optimum.

## Why the R24-style split doesn't transfer

R24's gpt_oss-Down split argument was: *"tiles_m=11 has tiles/CU ~4.3
(B=4 × 11 × 11 / 256 ≈ 1.9 / CU floor), so xcds=8 was over-splitting
the work across all 8 XCDs and starving each XCD of useful tiles. xcds=4
keeps each XCD with at least 1 tile most of the time."* The umbrella's
(gm=4, xcds=4) was already at xcds=4 — R24's split changed gm, not xcds.

For tiles_m=22 (gpt_oss-GateUP), the per-launch tile count is **double**
the Down sibling: tiles_per_launch = tiles_m × tiles_n × B = 22 × 11 ×
B ∈ {968 (B=4), 7744 (B=32)}. The persistent grid sees 4-30 tiles/CU,
well above the xcds=4 saturation floor. At this density, gm becomes
purely a tile-batching knob with weak dependency on xcds — the
umbrella's `(gm=4, xcds=4)` is one of many near-optimal points on the
flat-top, and no nearby cfg beats it.

This is the dual of R24's argument: starvation only matters when
tiles/CU is low. tiles_m=22 has ~2× the tile density of tiles_m=11, so
the Down-style split's improvement vector vanishes.

## Outcome

R66 closes the "tiles_m=22 dB var-K split" lever. The R24 commit's
claim of "5 of 5 families covered" is now post-hoc sweep-validated for
the missing 5th family (gpt_oss-GateUP) — `(gm=4, xcds=4)` IS the right
choice. **No code change committed; no metric movement; no falsified
artifact in the kernel.**

The audit is itself the deliverable: the open question from R24 is
resolved. R65 ended with a per-XCD WS docs comment as the round artifact;
R66 ends with this round note (no kernel change).

## Compliance check

* No kernel code changed (HipKittens working tree clean).
* No Primus-Turbo code changed beyond temporary config.py edits during
  paired tests, all reverted via `git checkout`.
* No host-pad / uniform-judge / per-group launch / CPU sync introduced.
* No metric file modified.
* No `can_handle` tightening.
* All 24 shapes correctness PASS across every paired/cfg sample
  (`correct_fail=0` consistently).

## Recommendation for R67

After R60-R66 (7 consecutive non-improvements) the kernel-level metric
surface is solidly saturated:

* R61 work-stealing locked in (only PASS in this run).
* R55 sched_barrier locked in.
* R52/R53 KI=88/48 specs locked in.
* R10/R21/R26 forward cfg rules locked in.
* R24 dB var-K cfg rules locked in (now including R66's tiles_m=22
  closure).

The remaining headroom (score 880 → 943 historical best → 1000 target)
must come from a kernel-architecture lever, not config or scheduler
tuning. R67 candidates ranked by likely-yield × tractability:

1. **Block-K depth tuning for gpt_oss K=2880 (K_BLOCK 64→32 or 64→128).**
   The kernel is hardcoded `K_BLOCK=64`; gpt_oss K=2880 = 45 K-iter; a
   smaller K_BLOCK halves the per-tile MFMA latency at the cost of
   more LDS round-trips. A single-line constant flip + recompile + bench
   probe; cheap to falsify.
2. **Dispatcher-level: skip the `bf16_transpose_3d` round-trip on
   gpt_oss-Down forward (N=2880 misaligned).** The transpose costs
   ~200 µs on B=32-M4096 (10% of wall). The H4 reroute trades K-tail
   fuse correctness for transpose cost; if the *forward* RRR K-tail
   path can be reframed (e.g. by leveraging the FUSED_KTAIL infra now
   stable on RCR), the forward-only Down ratio could climb 3-5%.
3. **PMC-driven occupancy audit on the worst forward shape
   (`gpt_oss-GateUP-B32-M2048`, ratio 1.054).** The kernel may have
   sub-saturating wave occupancy; rocprofv3 `valuMfmaUtil` +
   `lds_bank_conflict` would show where the ceiling sits. Diagnostic
   only — doesn't directly move score, but informs which architecture
   lever is worth pursuing in R68+.

Option 1 is the cheapest first try (1-line + recompile + bench); option
2 is the highest-yield if it lands; option 3 is informational backstop
if 1 and 2 both fall flat. Avoid further cfg-level sweeps — the
diminishing returns are evident across R60-R66.
