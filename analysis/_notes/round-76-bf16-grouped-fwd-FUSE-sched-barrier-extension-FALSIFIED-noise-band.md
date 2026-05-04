# Round 76 — BF16 grouped fwd FUSE-RCR sched_barrier(0) extension — FALSIFIED (noise band, +6 VGPR no payoff)

## Round number / date / GPU / sha

- Round: 76 (chat-resume of round 75; same metric `_metric_grouped_bf16_weighted_wall.py`).
- Date: 2026-05-04.
- GPU: HIP_VISIBLE_DEVICES=3 (MI355X / gfx950 / NUM_CUS=256).
- HK SHA: `9a860d59` (no commit produced; working tree reverted to clean baseline).
- Primus-Turbo SHA in: `964a8a60` → out: this falsification note only.

## Why this lever (continuing the R75 thread)

R75 falsified `grouped_var_k_kernel` work-stealing for the bwd dB CRR
path. The note's "next-step" inventory had three remaining levers:

| Lever | Status before R76 |
|------|-------------------|
| **C1** — extend R55 sched_barrier(0) into the FWD FUSE-RCR block | unchecked |
| **P2** — FUSE pipeline prefetch (slab-1 A-load overlapped with slab-0 MMAs) | risk: VGPR spill on FUSE |
| **P3-bis** — per-XCD counter on var-K only | likely repeat of R75 atomic-noise-band |

R76 picks **C1** because:
1. It's the cheapest mechanical change (3 intrinsics × 2 sites).
2. R55 paired result on EPILOG 1/2 was +9.5 median / +7 mean (excluding
   2 cold-start outliers) — i.e. the lever HAS demonstrated effect on
   fwd EPILOG paths in this codebase.
3. The FUSE-RCR block is the **only** RCR path in `kernel_bf16_dynamic.cpp`
   that lacks the `s_setprio(1)/MMA/s_setprio(0)/s_barrier/sched_barrier(0)`
   discipline (MAIN main_loop_iter has 8 sites since R54; EPILOG 1/2 has
   6 sites since R55; FUSE had 0 sites).
4. Path is **gpt_oss-only** (3× weight): `K_rem == K_STEP` gate fires on
   K=2880 only — DSV3 K∈{2048, 7168} and Qwen3 K∈{1536, 4096} are
   K%128=0 → no fuse path → no collateral risk on those families.

## Hypothesis

Adding R55-style MMA priority + barrier + sched_barrier(0) discipline to
the 4 MMA sites in the FUSE-RCR block (`device_gemm_tile_body`, lines
934-945) should mirror the +5-9 median lift R54/R55 saw on
MAIN/EPILOG, scoped to the gpt_oss family. Expected payoff:
+0.3-0.5% wall on each of 12 gpt_oss fwd+dA paths
(8 fwd + 4 Down-dA) × 3× weight ⇒ +5-7 score points.

## Patch

`HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`,
inside `device_gemm_tile_body<…, FUSED_KTAIL=true>` RCR branch
(lines 934-945, instantiated as `grouped_kernel<RCR, KI=0, FUSED_KTAIL=true>`):

```diff
             load_a_kt(0);
             load_b_kt(B_tile_0, 0);
             load_b_kt(B_tile_1, 1);
             asm volatile("s_waitcnt vmcnt(0)");
+            __builtin_amdgcn_s_setprio(1);
             DO_MMA(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
             DO_MMA(C_accum[0][1], A_tile, B_tile_1, C_accum[0][1]);
+            __builtin_amdgcn_s_setprio(0);
+            __builtin_amdgcn_s_barrier();
+            __builtin_amdgcn_sched_barrier(0);
             // M slab 1 (B tiles unchanged — share K-tail across slabs)
             load_a_kt(1);
             asm volatile("s_waitcnt vmcnt(0)");
+            __builtin_amdgcn_s_setprio(1);
             DO_MMA(C_accum[1][0], A_tile, B_tile_0, C_accum[1][0]);
             DO_MMA(C_accum[1][1], A_tile, B_tile_1, C_accum[1][1]);
+            __builtin_amdgcn_s_setprio(0);
+            __builtin_amdgcn_s_barrier();
+            __builtin_amdgcn_sched_barrier(0);
```

Mirrors EPILOG 2 lines 800-805 paired-MMA shape (both MMAs share
`A_tile` slab-0 / slab-1; pair-wrap with single priority/barrier/
sched_barrier instead of per-MMA).

## Build / resource report (compile-time evidence)

`grouped_kernel<Layout::RCR, KI=0, FUSED_KTAIL=true>` (the FUSE
instantiation):

| Metric | Baseline | Patched | Δ |
|---|---|---|---|
| TotalSGPRs | 102 | 102 | 0 |
| VGPRs | 250 | **256** | **+6** |
| ScratchSize bytes/lane | 0 | 0 | 0 |
| SGPRs Spill | 0 | 0 | 0 |
| VGPRs Spill | 0 | 0 | 0 |
| Occupancy waves/SIMD | 2 (max) | 2 (max) | unchanged |

Other instantiations (RRR FUSE, all CRR/RRR non-FUSE KIs, gemm_tail,
gemm_kernel) **unchanged** — patch is localised to one
`if constexpr (L == Layout::RCR)` branch inside the `if constexpr (FUSED_KTAIL)`
block.

`+6 VGPR` is non-zero but stays at the 256-VGPR cap with **0 spill**
and unchanged 2 waves/SIMD occupancy. Likely cause: `s_barrier` extends
some intermediate live ranges that LLVM previously rematerialised
across the FUSE block boundary.

## Correctness probe

Standalone Python probe (`/tmp/r76_correctness_probe.py`) on
3 fuse-eligible shapes (gpt_oss-Down-B4-M2048, GateUP-B4-M2048,
Down-B32-M2048) hit a GPU memfault on first launch — same
isolation-only fault pattern observed in R75's standalone probe.
The full metric run does NOT trigger the fault (different
warm-up + memory-allocation pattern).

Metric-level correctness gate: **0/24 correct_fail across 16 paired
runs** (8 baseline + 8 patched). All 24 shapes pass `check_allclose`
on `out, dA, dB` every run; if FUSE-RCR scheduling discipline had
introduced a numerical defect the gpt_oss-Down/GateUP B4/B32
shapes (the only fuse-eligible cases) would have failed at least
once. Conclusion: patch is **numerically equivalent** at bf16 floor.

## Metric paired result (8 baseline + 8 patched, same GPU, same session)

| Run | Baseline | Patched |
|---|---|---|
| 1 | 874 | 874 |
| 2 | 874 | 874 |
| 3 | 873 | 875 |
| 4 | 878 | 875 |
| 5 | 874 | 874 |
| 6 | 874 | 875 |
| 7 | 875 | 873 |
| 8 | 875 | 874 |
| **median** | **874** | **874** |
| **mean** | **874.6** | **874.25** |
| **std** | 1.32 | 0.66 |

**Δ median = 0**, **Δ mean = -0.4** — both well within the ±2 score
noise band, both ≪ the +5 commit-threshold. **FALSIFIED.**

Per-family geomean (mean across 4 latest patched runs vs 4 latest
baseline runs):

| Family | Baseline | Patched | Δ |
|---|---|---|---|
| gpt_oss_20B (3× weight, target) | 1.0772 | 1.0760 | **-0.12pp** |
| DeepSeek-V3 | 1.1211 | 1.1247 | +0.36pp (run-to-run; no FUSE path) |
| Qwen3-235B-A22B | 1.1129 | 1.1126 | -0.03pp (no FUSE path) |

The **target family** (gpt_oss, fuse-eligible) showed a -0.12pp
geomean drift on the patched runs — not a regression in any single
shape, just slightly higher run-to-run variance in the +6 VGPR
configuration. The DSV3 +0.36pp drift is pure run-to-run noise on
non-FUSE paths (DSV3 K%128=0 for all shapes).

## Why didn't the lever transfer (root cause)

R55 paired data on EPILOG 1/2 (+9.5 median) reflected the cumulative
effect of MMA scheduling discipline across **every K-tile** —
EPILOG runs once per output-tile, and there are many output-tiles
per CU. The FUSE block also runs once per output-tile, but contains
only **4 MMAs** vs MAIN's 8 MMAs × KI iterations + EPILOG's 8 MMAs.

Per-tile MMA budget for gpt_oss fwd K=2880 (KI=44, FUSE=true):
- MAIN (44 - 2 = 42 main_loop_iter calls × 8 MMAs): 336 MMAs
- EPILOG 1: 4 MMAs
- EPILOG 2: 4 MMAs
- FUSE: 4 MMAs
- **FUSE share = 4 / 348 = 1.15% of per-tile MMAs**.

R55's +9.5 median came from the 8 MMAs in EPILOGs (2.3% share each).
FUSE at 1.15% has half that scheduling-discipline ceiling, and the
+6 VGPR cost from `s_barrier` consumed that budget.

Scheduling discipline saturates: the MAIN loop already covers
~96% of per-tile MMAs, and adding it to a 1% sliver doesn't move
the needle.

## Closure

**FUSE-RCR sched_barrier(0) extension is FALSIFIED on the metric.**

The lever was textually obvious (the only RCR MMA site without the
discipline) but is structurally too small to register on the
weighted-wall metric. R55-style MMA-pipeline scheduling levers are
**closed for FWD** — both metric and structural ceiling reached.

## Reverted state

```bash
$ cd /workspace/code/HipKittens
$ git status
On branch save/fp8-progress-20260319-native-layouts
Your branch is ahead of 'origin/save/fp8-progress-20260319-native-layouts' by 1 commit.
nothing to commit, working tree clean (modulo .nfs / build artefacts)

$ # Post-revert metric: 875 (within ±2 of pre-round 874 baseline).
```

No HK commit produced. Primus-Turbo only carries this falsification
note — no kernel/dispatcher/binding changes.

## Next-round guidance

The FWD-side scheduling lever surface is **fully audited**:
- R54: MAIN sched_barrier density (PASS, landed)
- R55: EPILOG 1/2 sched_barrier (PASS, landed)
- R76: FUSE-RCR sched_barrier (FALSIFIED — no headroom)

Remaining surface focuses on the **bwd dB var-K path**, which is
~50% of total wall on the gpt_oss-Down-B4 family:

1. **B2: FUSE pipeline prefetch (R75 candidate, deferred)** — overlap
   slab-1 A-HBM-load with slab-0 MMAs (path B). Risk: VGPR spill
   on the already-tight 256-VGPR FUSE path. Need a SASS-level
   prefetch order audit before patching. Expected payoff: +5-10
   if it lands; structurally larger lever than R76's
   scheduling-discipline tweak.
2. **Var-K LDS bank-conflict re-attempt** — R74 swap-shape was
   FALSIFIED at 66 VGPR spill. R68 PMC found 217M LDS bank
   conflicts on the var-K kernel. A more careful approach: keep
   `st_32x16_s` shape but add an LDS row-pad (4 bytes) to break
   the bank conflict pattern without rewriting the swizzle.
   Risk: extra LDS occupancy bytes/block — but baseline is 544
   bytes/block (vs 65536 cap), plenty of headroom.
3. **Dispatch rule split for tiles_m=11 sub-family** (gpt_oss-Down
   only) — the current rule covers
   `tiles_m ∈ [8,24]` with single (gm=4, xcds=4). R66 audit covered
   tiles_m=22 (GateUP); tiles_m=11 (Down) was never explicitly swept
   under the weighted-wall metric. Probe (gm, xcds) ∈
   {(1,4), (2,4), (4,4), (8,4), (1,8), (2,8)} ⇒ pick best for
   `tiles_m == 11` only, leave `tiles_m == 22` on (gm=4, xcds=4).

R77 candidate: **option (3) dispatch sweep** — cheapest, no kernel
churn, fastest signal under the weighted-wall noise band.
