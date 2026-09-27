# Round 8 (fast) -- opt: reading, instruments, choice, route

Written as the round went. No card run was made in this step; every measurement below is compile-only
(`COMPILE_ONLY=1`, prod shape, container `fa-repro`, flydsl 0.3.4.1) plus static ISA census.
GPU0 was checked before starting: `rocm-smi --showpids` "No KFD PIDs", use 0%, no python in the container.
Parsed output is in `raw/` (`census_*.txt`, `loops_census.py`, `msb_census.py`); the scratch builds and the
kept `22_final_isa.s` of every arm are in `raw/cc/` (the container cannot see /tmp, as round 7 recorded).

## 1. Reading

- findings: `facts.md`, `dead_ends.md`, `pool.md`, `route.md` (operator tables h2-h27),
  `rounds/007/1-opt/opt.md`, `rounds/007/2-reflect/reflect.md`.
- Round-5 profile: not re-opened beyond what facts and r7's opt quote from it (verdict latency/low, stall
  attribution unavailable, 0.981 busy, HBM excluded). Nothing I planned needs a counter it lacks. The questions
  this round are about **instruction counts on the serial per-tile path**, which a compile census answers.
- State: the pool is effectively empty (g03, g04 and g16 closed awaiting retirement; g15 predicted null;
  g19 blocked on a depth-3 ring). The last three rounds were not accepted. So sources (c) and (d) are mandatory.

## 2. Look

### 2a. Survey

`rocprofv3 --stats` was not re-run. Rounds 5, 6 and 7 recorded that it writes no files on this stack
(3 attempts, rc=0). `op/current` is byte-identical to the round-5 tree, so the kernel list stands:
one kernel per call, 94.4% of prod time.

### 2b. r8.i3.g22 -- instrument: clean-loop census across nine builds that already have a prod number

Prediction before running: if the per-tile instruction count sets the time, every measured loser should
have a longer clean loop than the champion. What would change my mind: g14 (which facts says "removed
instructions" and lost 1.8%) really having a shorter loop.

`raw/census_msb.txt` (whole kernel and the 64-WMMA clean loop, LO path without the rescale block):

| build | clean loop instr | s_set_vgpr_msb | SALU | v_nop | measured prod vs r4 |
| --- | --- | --- | --- | --- | --- |
| r4 champion | 513 | 40 | 116 | 15 | 1.000 |
| r7 P1 nobar (racy) | 511 | 40 | 114 | 15 | 1.045-1.052 |
| r6 g14 (lane row-sum) | **536** | 56 | 129 | 27 | 0.981-0.985 |
| r7 P2 nodefer | 513 (+32 pk_mul unconditionally) | 56 | 88 | 20 | 0.914-0.919 |
| r7 p22n | 566 | 40 | 155 | 24 | 0.934-0.941 |
| r7 p22s | 604 | 39 | 164 | 21 | 0.963-0.970 |
| r5 g13 | 620 | 90 | 178 | 24 | 0.8485 |
| r7 coexec (off card) | 526 | 53 | 134 | 8 | -- |

Readings:
- **Correction to facts.** `r5.i2.g14` did not remove instructions from the serial path. It removed 6
  permlanes but the clean loop **grew 513 -> 536 (+4.5%)**: msb 40 -> 56, v_nop 15 -> 27, SALU 116 -> 129. Round
  7 cited g14 as the counter-example to "the instruction count sets the time". It is not a counter-example.
  Every build that lengthened the clean loop lost.
- The count is not the whole price. nodefer has the same count but runs 32 unconditional `v_pk_mul` (VALU work
  instead of SALU) and lost 8%. p22s is longer than p22n and faster, because its split gap is worth something.
  So count is a direction test, not a cost model (h13 stands). But no build with a longer loop has ever won.
- `s_set_vgpr_msb` is 40 of 513 (7.8%). Every loser except p22n/p22s raised it. It moves with register
  allocation, and no source-level knob I found lowers it (see 2c: max-ilp raises it to 51, max-memory-clause
  lowers it only to 38). This is the gfx1250 analogue of HipKittens' `v_accvgpr` traffic
  (`hipkittens/.../gqa_d128.md` s6.1, 1.72x from hand-placed registers). FlyDSL has no register-placement API,
  so it is recorded here and not pooled.

### 2c. r8.i4.g23 -- instrument: compile census of LLVM codegen flags on op/current

FlyDSL passes `compile_hints["llvm_options"]` to LLVM as cl::opts, so any backend option is a one-line arm.
Option names were checked against the strings in `libFlyPythonCAPI.so`. Prediction: most are inert. At most
one or two move the clean loop by more than 1%. `raw/census_flags.txt`:

| arm | option | md5 | VGPR/SGPR/spill | clean loop | msb | s_delay_alu | v_nop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ctl | (champion: expert-scheduling-mode=True) | 7b078261 | 456/100/0 | 513 | 40 | 14 | 15 |
| **nodelay** | `amdgpu-enable-delay-alu=False` | 07ae061a | 456/100/0 | **499 (-14, -2.7%)** | 40 | **0** | 15 |
| noexpert | `ENABLE_SCHED_MODE2=False` | 51a6bf86 | 456/100/0 | 510 (-3) | 40 | 14 | 15 |
| mmc | `amdgpu-sched-strategy=max-memory-clause` (h12 L24) | 45686581 | 456/100/0 | 518 (+5) | 38 | 18 | 16 |
| ilp | `amdgpu-sched-strategy=max-ilp` | 34d13267 | 462/100/0 | 529 (+16) | 51 | 15 | 19 |
| mmcp | mmc + `misched-postra=True` | = mmc | | | | | |
| postra | `misched-postra=True` | = ctl (7b078261) | | | | | |
| relax | `amdgpu-schedule-relaxed-occupancy=True` | = ctl (7b078261) | | | | | |

- `amdgpu-insert-delay-alu` is rejected ("Unknown LLVM option"). The first nodelay compile failed with rc=1 for
  that reason. It was renamed to `amdgpu-enable-delay-alu` and recompiled clean. That is the only build failure
  this round.
- postra, relax and mmcp's post-RA half are **inert**, with byte-identical ISA. h12's "post-misched" half of L24
  is closed on this kernel without a card run.
- nodelay removes all 14 `s_delay_alu` per tile. On gfx11+/gfx12 encodings `s_delay_alu` is an issue-scheduling
  hint; the hazard nops are a separate pass, and v_nop is unchanged at 15. So the arm should be correctness-neutral.
  That is inferred, and the 49 dB x16 and 200-run determinism gates must confirm it. Its speed sign is genuinely
  open: the hint may be what lets the SQ switch to the partner wave instead of stalling.

### 2d. r8.i5.g24 -- instrument: census of the unbuilt proto arms on the r4 tree

Round 7's reflect named the mistake: it put h24 on the card without a per-loop census. So I ran that census first
for every proto arm still open in the route. `raw/census_p12.txt`, `raw/census_flags.txt`; the protos applied
clean with `patch -p1` (offset 19).

| arm | what | VGPR/SGPR/spill | clean instr per tile | barriers per tile | vs ctl 513 |
| --- | --- | --- | --- | --- | --- |
| u2pp2 (h23 L14) | two tiles per iteration, depth 2 | 456/96/0 | 524.5 | 1 | **+2.2%** |
| u1pp3 (g15 / h23 L13) | depth-3 ring, `tensorcnt 0x2` | 456/105/0 | 530 | 1 | **+3.3%** |
| **p12s** (h24 at G=1) | split signal/wait, barrier per tile, **no `tile%G` branch** | 450/102/0 | about 516 (553/556 spans incl. rescale) | 1 | **about +0.6%** |
| p12n (h24 at G=1) | proto ring, no split | 450/100/0 | about 520 | 1 | about +1.3% |
| r7 p22s, for reference | G=2, split | 448/99/0 | 604 | 0.5 | +18% |

Readings:
- **p12s carries the split gap at almost no instruction cost.** In the ISA, `s_barrier_signal` sits at the end of
  the tile (after the PV `s_wait_dscnt`) and `s_barrier_wait` is 89 instructions into the next tile. Round 7
  measured the split at about +3 points (p22s 0.966 vs p22n 0.939, 5/5 sessions). But that gain sat on a loop
  paying +10-18% for the runtime `tile % 2` bookkeeping, which p12s does not have.
- **U2 as the carrier for "barrier per pair" costs +2.2% before it saves anything.** Halving the barriers is worth
  at most half of the 4.7% ceiling. So "L8 on a trace-time x2 body", the route round 7 named, is roughly net zero
  on this census. I am not pooling it. g19 would still need the depth-3 ring.
- **g15 (u1pp3) costs +3.3% instructions per tile** for a predicted gain below 0.5%. It is now priced as an
  expected loss alone. It stays only as g19's ring prerequisite.

## 3. Sources

- (a) pool/route: nothing buildable except g19, which is blocked on the ring, and g15, now priced as a loss.
- (b) the instruments above: the serial per-tile count is the best predictor this job has, and two changes shorten
  it or leave it flat while buying something: nodelay/noexpert (-14/-3) and p12s (+3 for the split gap).
- (c) corpus: `backends/flydsl/attention/techniques.md`. "Which cluster the work lands in" gives the
  compiler-strategy lever at +0.6-0.7% (the forward case, with bit-identical output) and warns that inert strategies
  are common; that matched 3 of my 7 flag arms. It also says barrier count is not the cost, and that per-barrier
  price rises as barriers are removed. That is consistent with p22's loss and with preferring the split to the
  count. `optimization/routes/1-metrics-to-techniques.md` (instruction-mix and wave-cycle rows: fewer issue slots
  or better placement; count is a direction test).
- (d) cross-backend: `backends/hipkittens/attention/recipes/gqa_d128.md` s6.1 (register placement, 1.72x at an
  equal instruction count, via operand-copy traffic; our analogue is msb, 7.8% of the loop, with no FlyDSL knob),
  s6.3 (lazy rescale, already here as defer, and L20 is dead twice), s8 Forward (the stagger idiom, which is g19).

## 4. Choice

Two new pool entries (details in `findings/pool.md`):
- **r8.i1.g20: codegen-flag arms.** nodelay, noexpert, and mmc (which discharges h12's L24).
  Each arm is a one-line change and a throwaway card arm versus ctl at prod.
- **r8.i2.g21: h24's split signal/wait at G=1 (p12s),** with p12n as the proto-overhead control.

They touch different lines: g20 changes the compile hints, g21 changes the barrier code. So they are measured apart
from op/current, and the merge is built only if neither lost.

Expected, and I can be wrong:
- g20: nodelay +1% at prod (range -2 to +2.5; sign uncertain), noexpert null (within 0.5%), mmc null to -0.5%.
  About a 40% chance that any g20 arm clears the 0.47% floor.
- g21: p12s +1.5% at prod (range 0 to +3), p12n -0.5 to 0. The p12s-minus-p12n split gap should be about +1-2 points.
  That is smaller than r7's 3 points, because r7's p22 loop had more work in the gap.
- If both win, the merge should be sub-additive: about +2%.

Deep-born entries passed over: `r5.i3.g15` (born in deep round 5). Its census here says +3.3% instructions per
tile for a predicted gain below 0.5%, so it is an expected loss alone. It is kept only as g19's ring prerequisite.

## 5. Route and pool

- `findings/route.md` `## Route` has been rewritten: h22 (record only, closed r7) > g20 > g21 > g19. The operator
  advise items follow, each with its condition.
- `findings/pool.md`: r8.i1.g20 and r8.i2.g21 were added. Evidence lines were added to g15 (census) and g19
  (U2 carrier cost).
- Instruments r8.i3.g22, r8.i4.g23 and r8.i5.g24 are recorded here only, as round 7 did with g17/g18.
- For the reflect: facts.md's round-7 sentence "`r5.i2.g14` *removed* instructions and also lost" is wrong at the
  clean-loop level (+23 per tile, s2b).

## explored.consulted

- knowledge/backends/flydsl/attention/README.md
- knowledge/backends/flydsl/attention/techniques.md (s "The constraint is softmax issue", "Two ways to overlap",
  "Which cluster the work lands in")
- knowledge/backends/flydsl/attention/recipes/hd128.md (s5 forward table, s6 b5)
- knowledge/backends/hipkittens/attention/recipes/gqa_d128.md (s6.1-6.7, s8 Forward)
- knowledge/optimization/routes/1-metrics-to-techniques.md (instruction-mix, wave-cycle rows)
- Primus-Turbo/output/0927__flydsl/proto/unroll (proto.diff, isa/ census) and proto/barriers (proto.diff)

## Build failures

- nodelay, first compile: rc=1. `RuntimeError: Unknown LLVM option: amdgpu-insert-delay-alu`. The fix was
  `amdgpu-enable-delay-alu`, which compiled with rc=0.
- The first compile pass produced no ISA, because `FLYDSL_DUMP_IR=1` is required as well as `FLYDSL_DUMP_DIR`.
  The whole set was re-run with it; all rc=0.
- The working copy `rounds/008/op` is untouched, identical to `op/current`.

## 6. Build (step 5)

The arms are built as separate trees in `raw/arms/<arm>`, each a copy of `op/current` plus one change:
- nodelay, noexpert, mmc: the one-line flag edits from s2c.
- p12s, p12n: the proto diff with `PROTO_ENV`/`PROTO_FENCE` env reads deleted and constants hard-coded
  (G=1, NG=2, SPLIT True/False).

The compile cache (`/root/.flydsl/cache` in `fa-repro`) is cleared at the head of the first card script.
The idle check before it: "No KFD PIDs", use 0%.

Screening: prod only, 6 arms (inc = `rounds/004/op` plus 5), 3 slot-rotated sessions in one container
process chain (`raw/screen/run.sh`). A dmesg fault monitor is armed alongside.

## 7. Measure (step 6)

### Screening: prod, 3 slot-rotated sessions, 6 arms, one process each (`raw/screen/`, all rc 0, sclk 1038-1041)

| arm | s1 | s2 | s3 | vs inc (s1/s2/s3) | verdict |
| --- | --- | --- | --- | --- | --- |
| inc (rounds/004/op) | 1087.26 | 1079.01 | 1087.30 | 1 | -- |
| **nodelay (g20)** | 1095.26 | 1086.44 | 1095.17 | **1.0074 / 1.0069 / 1.0072** | **won** (above the 0.47% floor 3/3) |
| noexpert (g20) | 1054.21 | 1061.44 | 1063.42 | 0.970 / 0.984 / 0.978 | lost |
| mmc (g20, h12 L24) | 1066.43 | 1073.79 | 1075.04 | 0.981 / 0.995 / 0.989 | lost |
| p12s (g21) | 1048.91 | 1048.55 | 1042.06 | 0.965 / 0.972 / 0.958 | lost |
| p12n (g21 control) | 1068.13 | 1068.80 | 1059.03 | 0.982 / 0.991 / 0.974 | lost |

Readings:
- **nodelay is the only winner: +0.72%, with a spread of 0.05% across sessions.** It sits right at the low edge of the
  predicted +1%, and the sign came out positive. So `s_delay_alu` is not load-bearing for wave interleave here.
  Removing 14 of 513 instructions per tile (2.7%) bought 0.7%. That is roughly a quarter of proportional, consistent
  with the partner wave hiding part of the issue stream.
- **g21 is falsified in both directions.** p12s lost 3.5%, and **split was worse than no-split by about 1.6 points**
  (p12s - p12n = -1.7/-1.9/-1.6), the opposite of round 7's +3 at G=2. The census (+0.6% instructions) did not
  predict the loss. The instruction count was not what moved: moving the wait 89 instructions into the next tile,
  when a single barrier per tile is left, exposes the partner's lag where the champion's barrier absorbed it. The
  fences (`BARRIER_FENCE`) and the proto ring's 1 KB slot layout are the other difference from ctl. This is inferred,
  since no counter was taken. p12n's -2% is the proto ring's own price at G=1.
- noexpert lost 2.3% despite -3 instructions. mmc lost 1.2% (+5). So the count is a direction test and not a cost
  model, as h13 already said: noexpert is the counter-case, being shorter but slower, because the schedule order
  matters.
- **No merge:** g21 lost, so per the route the merge is not built. **nodelay ships** and is in `rounds/008/op`.
  `files_changed` is one file, with the flag added in both `llvm_options` dicts.

### Correctness of the shipped arm (`raw/final/ut.out`, `val.out`, working copy `rounds/008/op`)

- `validation.py`: 16/16 PASS at 49 dB, worst o 49.82 dB (short_q full). Determinism 200/200 with
  hash o=26a89a2db0cd lse=d6ac8da1e101, **the same hash as the round-4 champion** (`rounds/007/1-opt/raw/ab/ut_inc.out`).
  Every dB figure matches to the second decimal, so the output is bitwise identical. rc=2 comes only from the speed
  bar (geomean vs beat 0.6273 < 1.0).
- `ut/test_correctness.py`: rc=2, from the same 7 lines between 49.82 and 49.99 dB that fail its hard-coded 50 dB
  in exactly the same way for the champion (see round 7). No new failure.

### Same-session measurement (`raw/final/a1-a4`, all shapes, beat in the process, 4 sessions)

Slot orders: a1 inc,r8,beat · a2 beat,r8,inc · a3 r8,inc,beat · a4 beat,inc,r8. Each of r8 and inc is behind beat
in 2 of the 4.

| shape | r8 (mean of 4) | inc rounds/004 (mean) | beat (mean) | r8/inc per session | r8/inc | r8/beat |
| --- | --- | --- | --- | --- | --- | --- |
| fast | 35.84 | 37.28 | 69.90 | 0.962 / 0.947 / 0.999 / 0.933 | 0.961 | 0.5128 |
| proxy | 766.90 | 802.06 | 1029.62 | 1.006 / 0.913 / 0.993 / 0.911 | 0.956 | 0.7448 |
| prod | 1093.10 | 1086.36 | 1405.76 | 1.008 / 1.006 / 1.008 / 1.003 | **1.0062** | 0.7776 |

score = 0.6784 (mean r8/beat).

**Instrument (`raw/final/n1-n4`, fast and proxy without beat in the process, r8 and inc alternating):**
- fast r8/inc: 1.003 / 1.003 / 1.003 / 0.998 (53.93 vs 53.84).
- proxy r8/inc: 1.002 / 1.005 / 1.005 / 1.003 (929.9 vs 926.6).
- Both arms run 35-45% faster on these two shapes when beat is not loaded.

The fast/proxy loss above therefore exists **only when beat has run earlier in the same process**, and there it is
lopsided. When r8 follows beat, proxy drops to 0.91 of inc; inc following beat drops only about 2%. nodelay is +0.2-0.5%
on fast/proxy when measured alone. I do not have the mechanism. Candidates are the foreign-kernel I-cache/state
effect already seen in r6 (g16) or clock state (sclk 1040-1059 in the beat-first sessions vs 1021-1050). What is
reported to the acceptance rule is the with-beat mean, as the protocol requires, and that is what will decide it.
With fast 0.961 and proxy 0.956 of inc, both are above 95% but the throughput mean is not above the champion. I
expect the round **not** to be accepted, despite a clean prod win.

### Against the expectation written before building

- nodelay was predicted at +1% (range -2 to +2.5); measured +0.62-0.72% at prod.
- noexpert and mmc were predicted null; they measured -2.3% and -1.2%. That is wrong in sign: the order the
  scheduler picks matters more than the count.
- p12s was predicted at +1.5%; it measured -3.5%. The prediction was wrong, and the split-gap mechanism does not
  transfer to G=1.

### Files left in the working copy

`rounds/008/op/flydsl_fwd/fmha_fwd_prefill_a16w16_m32x8.py`: `"amdgpu-enable-delay-alu": False` added in both
`llvm_options` dicts (the d128 path and the second launch builder). Nothing reverted.
