# Round 4-dm — FP8 K-tail vmcnt(8)-overlap + load-reorder falsified

**Date**: 2026-05-01 (Round 4 of 100)
**Primus-Turbo HEAD (entry)**: `9195a5f` (round-3-dm unroll docs)
**HK HEAD (entry)**: `6b47f420` (round-3-dm unroll saturated)

## Round-4 summary

- **Baseline metric**: 817 (geomean 0.9803). Lowest `grpFP8_gpt_oss_20B-Down-B4-M4096 @ 0.928` this run.
- **Pivot from round-3 suggestion**: round-3 recommended round-4 be a
  read-only audit of K-tail, no kernel edit. Reviewed the code + round-27
  falsification note; identified two lower-risk K-tail schedule probes
  that DON'T extend `a_kt1` live-range (the round-27 killer). Tried
  both.
- **Probe 1** — add `s_setprio(1/0)` wrap to the 4 K-tail mfmas (the
  only mfma site missing setprio; main loop + Epilog 1+2 all wrap).
  5-run mean 815.2 vs baseline 816.2 = **-1 score (noise)**.
- **Probe 2** — reorder loads so `a_kt1` issues LAST, split vmcnt(0)
  into vmcnt(8) + vmcnt(0) so cA+cB mfmas overlap a_kt1's HBM latency.
  Build OK with spill +2 to +6 on `FUSED_KTAIL=true` variants.
  Single-run score **797 = -20 vs baseline 817**. All 8 gpt_oss FP8
  shapes regressed 2.4-4.4 pp.
- Both probes reverted; final verify 823 (noise-band restored).

## Key mechanism insight

Round-27 failure mode ("extending `a_kt1` live range past Epilog 1+2
triggers spill cascade") generalises: **ANY K-tail schedule change
that extends a register's live range past the current allocator's
bin-packing envelope triggers the same spill cascade**.

Probe 2 extended `b0, b1, a` lifetime by 2 mfmas between vmcnt(8) and
vmcnt(0); spill went 45 → 51 (+6) on the main `FUSED_KTAIL=true`
variant; each spill slot costs ~20-50 cyc per tile; net -3 to -4 pp
per K-misaligned shape. 

VGPR budget in the FP8 grouped K-tail specialisations is at the 256
ceiling with 45-54 B/lane scratch. **There is no schedule-only lever
left on K-tail** — further improvement requires a prior register-tile
re-derivation to free VGPR headroom first.

## Implication for the round-8 §1 amortize plan

The round-8 §1 "multi-tile-M amortize" design requires:
- sharing `b0, b1` K-tail registers across multiple output tiles
- doubling `cA/cB/cC/cD` accumulator lifetime across two tiles

Both are large live-range extensions. Given the round-4-dm + round-27
evidence, amortize will almost certainly hit the same spill cliff.
**The round-8 §1 design is NOT directly implementable** without a
pre-requisite register-tile compaction pass.

## Updated cumulative saturated/falsified inventory

Every single-knob / schedule lever on FP8 grouped K-tail is now
formally falsified:

- `a_kt1` hoist before Epilog 1 (round-27): -4.9 pp
- `load_a_kt(a_kt1)` single-load merge (round-25 planned, subsumed
  by round-27)
- K-tail mfma setprio wrap (round-4-dm): neutral
- K-tail load reorder + vmcnt(8) overlap (round-4-dm): -20 score
- K-tail `RCR_EPILOGUE_VMCNT` (round-25): saturated

Plus all main-loop + epilog + config knobs (rounds 4-8, 14-16, 22-25,
1-dm, 2-dm, 3-dm).

## What's left (revised from round-3-dm)

The round-3-dm note recommended "structural project (1) K-tail
amortize". Round-4-dm evidence shows that project's design is
implausible at current VGPR layout. Revised ordering:

1. **~~K-tail amortize~~ — NOT feasible at 256 VGPR ceiling** (removed)
2. **DSV3-Down 1-round main-loop probe** — 4 shapes, ratios 0.95-0.97,
   NO K-tail/N-tail (simpler failure modes). Task-body diagnostic:
   "main loop throughput". Quick data-gather worth 1 round.
3. **Task-body lever E** (direct HBM→reg main loop) — 4-8 rounds,
   different failure mode (HBM traffic mult by WARPS_N=4). Untried.
4. **FP8 MFMA cell-shape 16x16x128 → 32x32x64** — 2-3 rounds, round-12
   partially falsified at MFMA-count level but the register-tile
   re-derivation could free VGPR headroom as a side-effect.

## Round-5 suggestion

**Option A** (recommended): 1-round DSV3-Down main-loop probe.
Examine whether Triton's per-K-iter schedule for K=2048 (ki=16) has
any structural difference from HK's K-iter (calibrated for ki=22
gpt_oss). If the DSV3 ratios pop up, it's a free win; if not, the
probe clearly delineates that the FP8 main-loop schedule is
structurally optimal for both K-regimes and future rounds should
pivot to lever E (option 3).

**Option B**: Start lever E (option 3) — the only remaining untried
path with documented +5-7pp yield. 4-8 rounds to land.

## Files touched

- HipKittens: `kernel_fp8_layouts.cpp:2414-2427` (edit + revert).
  `analysis/_notes/round-4-dm-fp8-ktail-vmcnt-overlap-falsified.md`
  (new).
- Primus-Turbo: `analysis/_notes/round-4-dm-fp8-ktail-overlap-falsified.md`
  (this file).

## Commits

- HipKittens: `docs(round-4-dm): FP8 K-tail vmcnt(8)-overlap + load-reorder falsified (-20 score, VGPR live-range cascade mirrors round-27)`
- Primus-Turbo: `docs(round-4-dm): FP8 K-tail schedule probes falsified`
