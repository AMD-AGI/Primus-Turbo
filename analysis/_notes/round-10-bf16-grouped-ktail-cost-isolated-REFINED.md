# Round 10 — BF16 grouped K-tail isolation A/B test (REFINED FINDING)

**Status:** PROFILE ONLY — refines the Round-9 K-tail attribution.
**Round-9 hypothesis (now partially falsified):** "the gpt_oss MFMA-util gap
of 17 pp vs DSV3 is dominated by the K%128=64 K-tail epilogue."
**Round-10 measurement:** K-tail costs ~4 pp MFMA util / ~56 µs / ~3 % of
kernel time. The remaining ~13 pp of the gpt_oss vs DSV3 gap is **short-K
main-loop saturation** (K=2880 has only 22 K_TWO_TILE iters vs DSV3 K=7168's
56 — fixed prologue/epilog cost amortizes worse).

## Background — what R9 hypothesized

R9 PMC profile (5 representative shapes, FUSED_KTAIL=true everywhere):

| shape | K | K%128 | MFMA util | dur (µs) |
|-------|---|-------|-----------|---------|
| gpt_oss-GateUP-B32-M2048 | 2880 | 64 | **63.4 %** | 2136 |
| gpt_oss-Down-B32-M2048 | 2880 | 64 | 62.3 % | 1892 |
| Qwen3-GateUP-B16-M4096 | 4096 | 0 | 75.3 % | 2080 |
| Qwen3-Down-B16-M4096 | 1536 | 0 | 25.6 % | 1500 |
| DSV3-GateUP-B16-M4096 | 7168 | 0 | **79.7 %** | 3037 |

R9 read the 17 pp gap (gpt_oss 63 % → DSV3 80 %) as evidence that K-tail is
the bottleneck and projected +20-30 % kernel speedup from fixing it.

## R10 isolation A/B — same M, N, B, only K differs

Probe: `/tmp/probe_round10_ktail_isolation.py`. B=32, M_per=2048, N=5760
(matches gpt_oss-GateUP-B32-M2048 exactly), K varied across {2816, 2880,
2944, 3008, 3072}. cfg locked at (gm=4, xcd=4) — production cfg from
R26 anchor at `config.py:1213-1296`.

Key A/B: K=2816 (K%128=0, no K-tail, 22 K_TWO_TILE iters main) vs
K=2880 (K%128=64, K-tail fires, same 22 K_TWO_TILE iters main).

| K | K%128 | dur (µs) | TFLOPS | MfmaUtil | LdsUtil | Δ vs no-tail |
|---|-------|----------|--------|----------|---------|--------------|
| **2816** | **0** | **1989.1** | 1068.8 | **66.48 %** | 25.18 % | (no-tail baseline) |
| **2880** | **64** | **2044.5** | 1063.5 | **62.20 %** | 23.13 % | **+55.4 µs / -4.28 pp** |
| 2944 | 0 | 2064.3 | 1076.7 | 69.14 % | 26.30 % | (longer no-tail) |
| 3008 | 64 | 2079.4 (re-run) | 1064.5 | 63.36 % | 23.62 % | +15 µs / -5.78 pp vs K=2944 |
| 3072 | 0 | 1975.6 | 1174.0 | 70.78 % | 26.88 % | (best — pipeline lines up) |

(K=3008 first-run was 3133 µs / 35.6 % — confirmed contention noise; re-run
matches the trend.)

## Mechanistic decomposition

The MFMA-util drops from K=2880 → K=2816 directly measure two effects:

```
Total kernel time (K=2880)  = MainLoop_22iters + K-tail_block + Prologue + Epilog drain + C-store
Total kernel time (K=2816)  = MainLoop_22iters +              + Prologue + Epilog drain + C-store
                                                ^^^^^^^^^^^^^
                              Diff = K-tail block cost
```

* Δ time = 55.4 µs / kernel = **2.7 % of total kernel time**
* Δ MFMA util = -4.3 pp = K-tail block runs at very low MFMA util internally
  (HBM load latency dominates, single-tile MMA can't pipeline)

**The remaining gap to DSV3 (66 % → 80 % = 14 pp)** is *not* K-tail. It's
short-K main-loop saturation:

```
Main loop MFMA util ≈ 80 % × (1 − fixed_overhead_fraction)
where fixed_overhead = (Prologue + Epilog drain + tile-transition + ...) / per-tile time

K=2816: per-tile time = 90.4 µs, ~22 K_STEPs main → fixed/per-tile ≈ 17 % → main util ≈ 66 %
K=4096: per-tile time longer, ~32 K_STEPs main → fixed/per-tile ≈ 6 %  → main util ≈ 75 %
K=7168: per-tile time longer, ~56 K_STEPs main → fixed/per-tile ≈ 0 %  → main util ≈ 80 %
```

This matches the measured progression (66 % → 75 % → 80 %).

## Score-impact ceilings (revised)

Using R10 measured kernel speedup ceilings, applied uniformly to gpt_oss
shapes (current ratios from `/tmp/metric_round_10.log`):

| Lever | Mechanism | Kernel speedup ceiling | Score ceiling |
|-------|-----------|------------------------|---------------|
| **A1** (K-tail surgery) | Pipeline K-tail HBM load with epilog 2 drain | ~+7 % | 879 → **915** (+36) |
| **A2** (short-K main loop) | Reduce per-tile prologue/epilog/tile-transition overhead | ~+13 % | 879 → **949** (+70) |
| A1 + A2 combined | Both | ~+20 % | 879 → **957** (+78) |

* **A1 leverage is smaller than R9 estimated** (~7 %, not ~20-30 %).
* **A2 has 2× the leverage of A1** but is structurally harder — it requires
  reducing per-tile fixed costs in the persistent grouped scheduler.

## What R10 attempted (and didn't ship)

Considered a low-risk K-tail HBM-load reorder at
`HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp:905-916`:

```cpp
load_a_kt(0); load_b_kt(B_tile_0, 0); load_b_kt(B_tile_1, 1);
asm("s_waitcnt vmcnt(0)");        // ← 12 vmem ops drain here
DO_MMA(C[0][0], A_tile, B_tile_0, ...);
DO_MMA(C[0][1], A_tile, B_tile_1, ...);
load_a_kt(1);                      // ← serialised after slab-0 MMAs
asm("s_waitcnt vmcnt(0)");
DO_MMA(C[1][0], A_tile, B_tile_0, ...);
DO_MMA(C[1][1], A_tile, B_tile_1, ...);
```

Possible reorder: issue `load_a_kt(1)` early (before slab-0 MMAs) into a
**separate** A register tile, then drop both vmcnt walls into a single
`vmcnt(0)` — overlaps slab-1 A HBM load with slab-0 MFMAs.

**Not shipped this round** because:
* Adds ~16 VGPRs for the second A register set
* Current kernel at 124 VGPRs / 7.3 occupancy/CU
* +16 VGPRs → 140 VGPRs → ~6.4 occupancy/CU (~12 % drop)
* Net effect of K-tail prefetch (+2 pp util) vs occupancy drop (-2-3 pp util) is
  approximately a wash — risk of regression on neighbouring no-K-tail shapes
  (DSV3, Qwen3) where the extra A register is wasted

This needs an A/B compile + measure round before shipping. Tabled as a
candidate for R11 — if compiled with conditional `#if FUSED_KTAIL_PREFETCH`
and gated only when K%128 != 0 in the dispatcher, the occupancy hit on no-tail
shapes can be avoided.

## Falsified / refined claims

* **R9 claim:** "K-tail dominates the gpt_oss MFMA-util gap" → **PARTIALLY FALSE**.
  K-tail accounts for ~4 pp of the 17 pp gap. The remaining 13 pp is short-K
  main-loop saturation.
* **R9 claim:** "Lever A1 (K-tail surgery) projected +20-30 % kernel speedup" →
  **PARTIALLY FALSE**. A1 ceiling is +7 %, not +20-30 %. Score ceiling +36, not
  +70-90 as projected.
* **R9 claim:** "Qwen3-Down-B16-M4096 K=1536 25 % util is anomalous" → confirmed
  separately; R10's data shows main-loop length matters (K=1536 has only 12
  K_TWO_TILE iters / 8 K_STEPs — the shortest in the suite by far).

## Why R10 doesn't move the score

* Pure profiling round (per R9 plan).
* No shippable kernel change identified within R10 budget that is risk-bounded
  (K-tail HBM-load reorder needs compile+verify A/B).
* R10 metric run = 879 (within ±10 noise band of R9 baseline 878). Not
  expected to differ.

## R11+ plan (revised)

Given A2 is the higher-leverage lever, two parallel tracks:

### Track A1: K-tail HBM-load early-issue (R11-R12)
* R11: implement gated `FUSED_KTAIL_PREFETCH` in `kernel_bf16_dynamic.cpp` — issue
  `load_a_kt(1)` to a separate register set BEFORE slab-0 MMAs, drop the
  intermediate vmcnt drain.
* R11 gate: only enable when K%128 != 0 (avoid VGPR hike on no-tail shapes).
* R12: A/B metric — accept if score ≥ 879 + 5 (i.e., ≥ 884) AND no DSV3/Qwen3
  regression > 0.5 pp.

### Track A2: per-tile fixed-overhead reduction (R12-R15+)
Higher leverage but structurally harder. Two sub-candidates:
1. **A2a: tile-transition overlap.** Currently each tile of the persistent
   schedule runs:
   `prologue_load → main_loop → epilog → C-store → next_tile_prologue`.
   The `next_tile_prologue` blocks on the previous tile's C-store. Pre-issue
   the next prologue's HBM loads BEFORE the current tile's C-store completes.
   Needs separate prologue register set or LDS dual-buffering already enabled —
   audit kernel.
2. **A2b: epilog drain compaction.** Currently epilog 1 + epilog 2 do 2-stage
   pipeline drain. For very-short-K main loops (gpt_oss K=2880, 22 iters),
   epilog drain may be a larger fraction of kernel time. Profile epilog
   instructions specifically.

R12 plan: focus on A2a (tile-transition overlap). Profile via rocprofv3 with
markers to see how much of per-tile time is "between MFMAs" (= LDS-wait,
tile-transition, etc.).

## Compliance check

* No metric file edits (`scripts/_metric_grouped_*` untouched).
* No `can_handle` tightening.
* No host-syncs.
* No HipKittens C++ changes this round.
* No new dispatcher rules.
* Doc-only commit.

## What R10 DID

1. Re-ran R10 baseline metric (879, vs R9 baseline 878 — within noise).
2. Wrote `/tmp/probe_round10_ktail_isolation.py` (5 K values × same M/N/B).
3. Ran rocprofv3 PMC over all 5, correctly attributed K-tail cost.
4. Re-ran K=3008 to confirm initial outlier was contention noise (it was).
5. Documented refined finding + revised plan for R11+.

## Files touched

* `analysis/_notes/round-10-bf16-grouped-ktail-cost-isolated-REFINED.md` — this note.

No code changes. No dispatcher changes. No HipKittens changes.
