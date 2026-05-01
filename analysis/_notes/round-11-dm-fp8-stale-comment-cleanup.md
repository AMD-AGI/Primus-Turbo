# Round 11 (death-march) — FP8 grouped config.py stale-comment cleanup

**Scope:** `primus_turbo/pytorch/kernels/hipkitten/config.py`
gpt_oss-GateUP-B4-M4096 rule comment block (lines ~897-933 pre-round-11).

## TL;DR

Replaced the demonstrably-false "Round-21" comment with the round-10-dm
verification evidence. The rule itself (`group_m=14, num_xcds=4`) is
unchanged. **Score-neutral by construction (Python comment-only edit, no
import-time side effect, no kernel rebuild).** Score ≈ 810-816 across
runs (within noise band).

## Rationale

Round-10-dm (commit `6ec3733b`) audited 4 worst FP8 grouped per-shape
config rules and found all at the saturation plateau. In particular,
the "Round-21" comment in the gpt_oss-GateUP-B4-M4096 rule claimed
``(gm=8, xcd=4)`` dominates ``(gm=14, xcd=4)`` by +29.2 TF — a claim
left over from the pre-round-19 (FLAT-store) kernel.

Round-10-dm's 1500-iter × 7-repeat probe against the live (BUFFER-store)
kernel showed ``(gm=8, xcd=4)`` is the WORST candidate in the sweep,
opposite of the stale comment. The actual measured table:

```
cfg     | p14 TFLOPS | Δ vs (14,4)
(14, 4) | 1240.33    | +0.00 *rule
(14, 8) | 1241.28    | +0.95
(14, 2) | 1241.25    | +0.92
( 8, 8) | 1240.80    | +0.47
(16, 4) | 1240.18    | -0.16
(10, 4) | 1239.61    | -0.72
(12, 4) | 1239.31    | -1.02
( 8, 4) | 1238.93    | -1.40
```

Round-10 doc explicitly recommended this cleanup as the simplest
defensible round action ("Stale comment cleanup: ... Replace with
round-7's (gm=14, xcd=4) evidence"). Round-11 acts on that.

## What changed

- Removed: 30-line "Round-21" comment claiming `(8, 4)` wins.
- Added: 24-line round-10-dm comment block citing the live verification
  table + falsification of the stale claim.
- The `return HipKittenConfig(group_m=14, num_xcds=4, kernel=None)` is
  unchanged — bit-identical kernel dispatch.

## Why no perf-affecting change in round 11

Per round-25 saturated-knobs inventory + rounds 1-10-dm falsifications
(tracked in `analysis/_notes/round-{1..10}-dm-*.md`), every 1-round
single-knob lever currently exposed to the FP8 grouped RCR kernel is
exhausted:

- All `RCR_*_VMCNT` / `RCR_PREFETCH_LGKM` / `RCR_*_SCHED_BARRIER`
  exhausted (rounds 4, 14, 16, 22, 24, 25).
- `(group_m, num_xcds)` per-shape rules at saturation plateau
  (round-10-dm 1500-iter × 7-repeat verify).
- `chunk_size` (chiplet) dead-end (round 22).
- `RCR_TWO_TILE_MIN_KI` no-op for grouped at gpt_oss K (round 15).
- 2-tile main loop port catastrophic (round 6).
- LDS layout swap `ST_v2 → ST_v3` falsified (round 2-dm).
- `#pragma unroll` sweep flat across {1,2,4} (round 3-dm).
- launch_bounds MIN={2,4} no-op (rounds 5-dm, 7-dm).
- `WARPS_M / WARPS_N` flip (2/4 → 4/2) catastrophic (round 6-dm).
- `a_kt1` scoping cleanup score-neutral (round 7-dm).
- rcr_4w port plan invalidated by occupancy data (round 8-dm).
- Parallel LDS-init for `s_offs` perf-neutral (round 9-dm).
- All 4 worst-shape per-shape `(gm, xcd)` re-tunes saturated (round 10-dm).

## Remaining levers (multi-round, not in scope for round 11)

Per round-25 inventory and round-9-dm doc, only structural multi-round
projects remain:

1. **AGPR accumulator migration on `grouped_rcr_kernel`** — frees ~128
   VGPR by steering `cA/cB/cC/cD` into AGPRs. High risk, multi-round
   scope. Round-8-dm planning doc has the entry point; not started.

2. **K-tail epilog single-load merge** (round-25 doc §3) — fold the
   path-B `rcr_8w_load_hoist` for the K-tail block into the last main-
   loop K-tile's prefetch slot inside Epilog 2. Estimated 2 rounds;
   requires path B and Epilog 2 schedule co-design.

3. **FP8 MFMA cell-shape `16x16x128` → `32x32x64`** — re-derive `RBM/RBN`
   and the lane-cell mapping. Round-12 partially falsified at MFMA-count
   level. Estimated 2-3 rounds.

4. **BF16 grouped: BK=64 → BK=32 + num_stages=3** — out of scope per
   task body ("BF16 K_STEP=64→32 port ... now out of scope — BF16 is in
   `[watch]` only, score doesn't reward").

## Metric verification

GPU 3 (pinned by auto_optimize, 0% util, no KFD process VRAM) — clean
baseline confirmed via `rocm-smi --showpids` before each run.

```
[round-11 entry, pre-edit]   score = 810   geomean grp_FP8 = 0.9723
[round-11 variance check]    score = 816   geomean grp_FP8 = 0.9791
                            (Δ = +6, within ±10 noise band)
[round-11 post-edit]         score = TBD (post-commit verify)
```

Edit is comment-only; kernel dispatch path is bit-identical, so any
score wobble is metric variance not regression.

## Commits

- Primus-Turbo: this round.
- HipKittens: no change.

## Next-round suggestion

Pick one of the 3 multi-round structural projects above and start it
WIP-style (write the design doc + commit incremental scaffolding).
Recommend project (2) **K-tail epilog single-load merge** because:

- Smaller blast radius than AGPR migration (touches Epilog 2 +
  FUSED_KTAIL block only; main-loop unchanged).
- Concrete ~1 % wall savings on B=4 cases (round-25 estimate) that
  translate to ~+10-20 score on the worst-perf shapes.
- Failure mode is graceful: WIP commits each round are reviewable.

Round-12 step 1 should be: read the current Epilog 2 + FUSED_KTAIL
schedule (lines 2234-2440 of `kernel_fp8_layouts.cpp`), enumerate the
per-tile load coordinates, and design the single-load merge schedule
on paper before any kernel edit.
