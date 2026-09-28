# Round 11 -- opt (fast round, GPU 2 / fa-g2)

All builds and timings ran in container fa-g2 (physical GPU 2, logical 0). Bulk output is in
`/tmp/op-evolve-gfx1250-flydsl-attn-fwd-b0-20260927-r011`. Parsed output is in `raw/`. Every ranking process
had no beat, a champ2 A/A copy, PYTHONHASHSEED=0, the blocked ruler (h40) and 3 rotated arm orders (h31/h35).
rocm-smi --showpids/--showuse were captured before and after each run in `raw/*/smi_*.txt`. dmesg was monitored
for the whole round and showed no new amdgpu faults.

## What I read

- findings: facts.md, pool.md, dead_ends.md, route.md (including the operator tables).
- rounds/010/1-profiling: profiling_summary.md and profiling.yaml. rounds/010/4-reflect/reflect.md.
  rounds/010/3-act: measure/run.sh (protocol template) and tools/{compile_isa.py, isa_all.sh, occ2_check.py}.
- rounds/010/op/impl.py, which carries the r10 occ2 m32x4 kernel, prod-gated and not shipped.
- op/ut/common.py (SHAPES), gates.py (check_correctness), validation.py (read_spec: gate 49 dB).
- corpus: listed under explored.consulted below.

## Survey (blocked ruler, TFLOPS, `raw/survey/`)

| shape | current | beat (own process) | current/beat |
| --- | --- | --- | --- |
| fast | ~143.8 (3 sessions) | 155.08 | 0.927 |
| proxy | 1604.33 | 1599.09 | 1.003 |
| prod | 1503.03 (A/A champ2 1503.01) | 1559.85 | 0.964 |

Fast is the biggest relative gap. Fast is also the only shape that is not power- or clock-bound: it runs at
~2.3 GHz with sclk flat over the run, and it is underfill-limited (h37: BLOCK_M=128 gives only 64 WGs on
256 CUs). Prod is clock-bound: r10 g24 cut cycles 6.6% and wall moved x0.9929. So this round goes after fast's
grid fill (corpus routes/1-metrics-to-techniques.md row A6: below the fill crossover, only tile size or dispatch
helps).

## r11.i3 (instrument, not a pool entry) -- s_set_vgpr_msb census (`raw/loops.py`, `raw/isa_census.txt`)

- Our clean prod loop (m32x8): 163-169 `s_set_vgpr_msb` per ~952 instructions (~17%). It is the most frequent
  instruction in the loop, ahead of v_exp (130) and WMMA (96). The masked loop has 35-39.
- They come from the exp stage: `v_pk_fma_f32` reads operands in the low VGPR bank and `v_exp` writes P into the
  high bank (>255), and the two are interleaved, so the bank switches on nearly every instruction.
- Beat (`fmha_bf16_pertokenBf16_hd128_128x256_mask.co`, disassembled with the rocm-sdk llvm-objdump): 1602 msb for
  896 WMMA = 1.79 msb/WMMA. Ours is 163/96 = 1.70. Beat pays the same tax, so msb is not what separates us
  from beat.
- The msb instructions come from the LLVM pass amdgpu-lower-vgpr-encoding, which has no knob. Not a candidate.
  I recorded it so it is not re-derived.

## r11.i1.g45 -- occ2 m32x4 (BLOCK_M=128, 4 waves x 32 rows) for under-filled grids

`arms/occ2fast`: the r10 op with the impl.py gate extended so that `d==128 and grid_m32 < #CU` also takes
`_kern_occ2`. This is r10's unshipped occ2-at-fast arm (g44 rider). Same grid as m16x8 (64 WGs at fast).

| session | champ | champ2 | occ2 | occ2/champ |
| --- | --- | --- | --- | --- |
| survey s1-3 | | | | x1.016 / 1.019 / 1.021 |
| fast4 s1 | 142.71 | 142.90 | 146.41 | 1.026 |
| fast4 s2 | 142.90 | 142.34 | 144.64 | 1.012 |
| fast4 s3 | 142.71 | 143.86 | 144.64 | 1.014 |

+1..2.6%, with A/A within 0.8%. At the same grid, the gain confounds 32-row waves with g18 speculation (which
m16x8 lacks). Dominated by g20 below.

## r9.i1.g20 -- BLOCK_M=64 fast gate, realised as m32x2 (2 waves x 32 rows)

r9 wrote g20 as 4 waves x 16 rows and blocked it on L12 (the 4-wave body). r10 delivered a 4-wave x 32-row body
(m32x4, occ2). Halving its wave count keeps 32-row waves, speculation and the whole kernel body, and gives
BLOCK_M=64: fast goes 64 -> 128 WGs.

- `arms/m32x2probe/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x2.py` is m32x4 with `NUM_WAVES = 2`.
- **Build failure 1**: compile raised `NotImplementedError("V2 TDM loader assumes 8 waves")`.
  `fmha_b16_buffer_managers.py` whitelists num_waves in 5 places. I widened them to `(2, 4, 8)` in the probe
  copy only. No other change was needed.
- Compile: VGPR 454, 0 spill.
- Gate in impl.py: `if d == 128 and grid_m32 < _NUM_CU: kern = _kern_m32x2`. proxy/prod have grid_m32 >= #CU,
  so their dispatch is unchanged.
- Correctness (`raw/probe_check.py`, `raw/check_m32x2/`): uses gates.check_correctness with gate_db from
  validation.read_spec(), over every SHAPES entry except proxy/prod, causal and non-causal. 14/14 cases pass
  the 49 dB gate and are bitwise equal (o and lse) to the m32x4 arm. ALL_OK.

| session | champ | champ2 | m32x2 | m32x2/champ |
| --- | --- | --- | --- | --- |
| fast4 s1 | 142.71 | 142.90 | 155.53 | 1.090 |
| fast4 s2 | 142.90 | 142.34 | 154.19 | 1.079 |
| fast4 s3 | 142.71 | 143.86 | 154.19 | 1.080 |
| m32x1 run s1 | 142.33 | 143.29 | 154.65 | 1.087 |
| m32x1 run s2 | 144.64 | 143.47 | 156.89 | 1.085 |
| m32x1 run s3 | 143.86 | 144.25 | 155.98 | 1.084 |

x1.08-1.09 in 6/6 sessions. A/A is at most 0.8% (fast4 s3), so the gain is 10x the noise. 154-157 TFLOPS is
beat's level (155.08 in its own process).

## r11.i2.g46 -- m32x1 (BLOCK_M=32, 1 wave x 32 rows): full CU fill at fast (256 WGs)

- `arms/m32x1probe`: m32x4 with `NUM_WAVES = 1`, managers allow (1, 2, 4, 8), and the same gate.
- Compile: VGPR 460, 0 spill, LDS 163 840 B, WG 32 threads.
- Correctness (`raw/check_m32x1/`): 14/14 pass the gate, bitwise equal to m32x4. ALL_OK.

| session | champ | champ2 | m32x2 | m32x1 | m32x1/champ | m32x1/m32x2 |
| --- | --- | --- | --- | --- | --- | --- |
| s1 | 142.33 | 143.29 | 154.65 | 148.64 | 1.044 | 0.961 |
| s2 | 144.64 | 143.47 | 156.89 | 151.57 | 1.048 | 0.966 |
| s3 | 143.86 | 144.25 | 155.98 | 150.31 | 1.045 | 0.964 |

m32x1 beats champ but loses to m32x2 by ~4%. Filling all 256 CUs does not pay once each WG is a single wave:
per CU, the K/V tile is fetched into LDS for 32 q rows only (twice the traffic per row of m32x2), and a single
wave has no SIMD-mate to hide its TDM waits and barriers. The fill optimum at fast is at 128 WGs.

## Decisions

- **r9.i1.g20 is the candidate.** Its L12 blocker is lifted by r10's m32x4 body. Its realisation changes from
  4x16 to 2x32, which keeps speculation. Expected: fast x1.08-1.09 (to about beat). proxy/prod unchanged (the
  gate is not taken there). The act step must port it into `rounds/011/op`:
  1. the m32x4 file from r10 (occ2, still prod-gated off unless r10 is re-argued);
  2. the m32x2 file;
  3. the managers whitelist;
  4. the gate.

  Then:
  - run the full correctness gate, including the verify_r6 256-case adversarial suite, because speculation
    now runs at fast (h34/h16);
  - check that proxy/prod ISA is byte-identical to current;
  - check RSRC3 against the 32 640 B inst-prefetch cap;
  - rank with the h31 3-rotation A/A.
- **r11.i2.g46 (m32x1) is the second arm**, built separately from the same base. Measured here as dominated
  by g20 (x0.96). It stays in the route only as the alternative for the same gate line. g20 and g46 compete
  for the same gate line, so there is no merge: ship the best.
- **r11.i1.g45 (occ2 at fast)** is measured and dominated by g20. Recorded in the pool as a closed probe.
- **r8.i1.g19** (speculation in m16x8) is superseded: with g20, m16x8 no longer runs at any scored shape.
- **r5.i3.g15 split-KV** (deep-born, open) is not taken. g20 recovers fast's underfill to about beat's level
  without a second reduction kernel. split-KV still needs the h16 ruling, and it has to beat x1.08 at 128 WGs,
  not x1.0 at 64.
- ISA dumps (isa_tmp) moved to container fa-g2 scratch `/tmp/op-evolve-gfx1250-flydsl-attn-fwd-b0-20260927-r011/`.
- msb census (r11.i3) is an instrument, not a pool entry. added_to_pool is capped at two.

## explored.consulted

- knowledge/backends/flydsl/attention/README.md
- knowledge/backends/flydsl/attention/recipes/hd128.md (gfx950; poor match, no underfill guidance)
- knowledge/backends/hipkittens/attention/recipes/gqa_d128.md §6 (gfx950 bwd, hand-assigned registers, v_accvgpr
  traffic; read for the msb question)
- knowledge/backends/*/attention/ (directory listing for cross-backend recipes)
- knowledge/optimization/routes/1-metrics-to-techniques.md (row A6 grid fill, table B occupancy/drain rows)
- grep `vgpr_msb` over knowledge/ (no hits)

# Act (step 5-6) -- route rows in order

Route rows 1-3 (h37, h34, h28, all musts): discharged, no build. h28 binds the protocol: the ranking processes
contain no beat, and beat is timed in its own process.

## Row 4 -- r9.i1.g20 build (rounds/011/op)

- Compile cache FLYDSL_RUNTIME_CACHE_DIR=/tmp/flycache in fa-g2: cleared at 2026-09-28T04:04:30Z (3.5M). `__pycache__`
  was removed from rounds/011/op before the first build.
- Diff vs op/current:
  1. new `flydsl_fwd/fmha_fwd_prefill_a16w16_m32x2.py`: r10's m32x4 with `NUM_WAVES = 2`;
  2. `fmha_b16_buffer_managers.py`: the r10 num_warps plumbing plus the `num_waves in (2, 4, 8)` whitelist
     (5 sites);
  3. `impl.py`: `kern = _kern_m32x2 if (d == 128 and grid_m32 < _NUM_CU) else _kern`. The m16x8 import became
     unused and was removed. The file stays on disk.
- Build failures: none in act. The one "V2 TDM loader assumes 8 waves" failure was met and fixed in the probe
  (see above).
- ISA gates (`raw/isa_act/summary.txt`):
  - m32x8 final ISA is byte-identical current vs op at prod/nc_g4/c_g1/nc_g1/c_g2/nc_g2, so the proxy/prod
    kernel is unchanged;
  - m32x2: VGPR 454/448, 0 spill, 0 scratch, LDS 163840;
  - code 28.8-30.0 KB, under the 32 640 B RSRC3 cap (current m32x8 prod reads 30 780 B with the same method).
- Correctness (`raw/correct/`, detached in fa-g2, default user, PYTHONHASHSEED=0):
  - op/validation.py: all 16 correctness rows PASS at the spec 49.0 dB. o min 49.82 dB (short_q full, same as
    baseline); fast 51.23, proxy 50.89, prod 50.83. Determinism 200/200 at fast. rc=2 on the speed bar only
    (geomean vs beat 0.9602, beat in the same process: fast 0.9886, proxy 0.9270, prod 0.9659).
  - ut/test_correctness.py --determinism 200: RESULT PASS (rc 0).
  - tools/g20_check.py, bitwise vs op/current forced into m32x8 (the body that passed verify_r6) (rc 0):
    - A: 16/16 ut cases bitwise (14 take m32x2, and proxy/prod take m32x8);
    - C: verify_r6 adversarial 32 kinds x {toy, short_q, gqa4} x {causal, non-causal} = 192/192 bitwise, 0
      non-finite.

## Row 4 -- measurement (`raw/measure/`, launched 2026-09-28T04:07:54Z)

Protocol: fa-g2, default container user, PYTHONHASHSEED=0, blocked ruler (h40). Each session is one ranking
process with 5 arms (cand = rounds/011/op, champ = op/current, champ2 = byte copy of op/current for A/A,
r010 = rounds/010/op, g46 = row 5's arm) in rotated order, followed by beat alone in its own process (h31). 3
sessions. Device note: rocm-smi listed a KFD pid from fa-g0 (physical GPU 0, a bwd ruler run). That is not our
card, and fa-g2 GPU use was 0% (`raw/measure/device_note.txt`). dmesg: nothing new over the run.

Reference arms: op/current and rounds/010/op are the incumbent and the only rounds on the current structure.
rounds/010/op is not byte-equal to op/current: it still carries r10's unshipped occ2 prod gate. Both were
re-run. Rounds before 10 were not re-measured.

| shape | cand | champ (op/current) | champ2 A/A | r010 | g46 | beat (own process) |
| --- | --- | --- | --- | --- | --- | --- |
| fast | 155.54 | 143.47 | 143.47 | 142.53 | 150.30 | 155.98 |
| proxy | 1597.98 | 1600.58 | 1600.58 | 1600.96 | 1603.58 | 1566.96 |
| prod | 1500.65 | 1501.88 | 1499.22 | 1494.32 | 1500.55 | 1557.09 |

(medians of 3 sessions, TFLOPS. Per-session values are in `raw/measure/summary.txt`.)

Paired ratios in the same process (median of 3, per session):

| shape | cand/champ | A/A champ2/champ | cand/r010 | g46/cand |
| --- | --- | --- | --- | --- |
| fast | 1.0905 (1.100/1.091/1.081) | 1.0040 | 1.0899 | 0.9611 (0.966/0.961/0.961) |
| proxy | 0.9988 (0.996/1.011/0.999) | 0.9977 | 0.9961 | 1.0026 |
| prod | 0.9992 (1.003/0.997/0.999) | 0.9976 | 1.0060 | 0.9999 |

- fast: +9.05%, 20x the A/A spread, in 3/3 sessions. cand/beat 0.997 (was 0.920).
- proxy/prod: inside A/A. That is expected, since their m32x8 ISA is byte-identical to current (gate not taken).
- vs target (1.0 x beat, beat measured this session in its own process): fast 0.9972, proxy 1.0198, prod 0.9638.
  score (uncapped mean) 0.9936, capped 0.9870. Clocks: beat's proxy process ran at sclk 1102-1177, while the
  ranking processes ran proxy at 1308-1538. A cross-process proxy ratio carries that clock difference.
- Best ever on the current structure, re-measured here: fast 143.47 (champ), proxy 1600.96 (r010), prod 1501.88
  (champ). cand vs best: fast 1.0841, proxy 0.9981, prod 0.9992 (medians of medians). Every shape is >= 95%
  of its best. proxy, the only shape at target, stays at target (1.0198).

Expected (written before building): "fast +8-9% (to ~beat), proxy and prod unchanged (gate not taken)."
Measured: fast +9.05% (0.997 of beat), proxy/prod unchanged. As predicted.

## Row 5 -- r11.i2.g46 (separate arm, `arms/g46`)

Built from the same base as row 4, with m32x1 (NUM_WAVES=1) in the g20 slot and the managers allowing
(1, 2, 4, 8). Correctness (`raw/measure/g46_check.log`): all small shapes pass the spec gate and are bitwise
equal to row 4. Timing: fast 150.30 = x1.048 vs champ and x0.961 vs row 4 in 3/3 sessions. proxy/prod equal.
Not shipped. The working copy holds row 4 (the winner). The m32x1 probe result is confirmed: the fill optimum at
fast is 128 WGs x 2 waves, not 256 WGs x 1 wave.

## Verdict

Row 4 is delivered and left in rounds/011/op. Row 5 is delivered as a sibling arm, lost, and not shipped. Rows
6-14 are next round's. Bound for this round's lever: latency (fast underfill; sclk flat at ~2.3 GHz). The
job's remaining gap is prod, which is power/clock-bound (sclk 993-1257 during prod).
