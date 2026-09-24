# Round 20 -- fast round -- gfx1250 / FlyDSL attention backward

Written as the round ran.

## 0. State inherited

`op/current` = round 17's tree (round 19 shipped a null and was not accepted; round 18 lost).
Incumbent ~504.90 TF/s prod, bar ~719.50, ratio 0.7023. `bound: latency, MEDIUM` -- raised on
`r19.i2.g60`'s measurement (deleting 88 VGPR / 82 issue slots made it **17.3% slower**).
Highest `g` = **g60**. Pool holds exactly one open id, `r16.i3.g50`, and `route.md` row 10
names it the next round's only `idea` row with a **mandatory prerequisite probe** in front
of it.

The single surviving mechanistic story for the ~686 non-issuing cycles per `.LBB0_8`
iteration is **LDS port pressure**: `k_dkdv`'s body is 40 `ds_store_b128` +
40 `ds_load_tr16_b128` = 1.25 LDS ops per WMMA against `k_dq`'s 0.33, and 64 of the 80 are
the Q/dO store-then-transpose-load round trip. **No arm has ever tested it.**

## 1. What this round decided to do, and why it is not what route row 10 says verbatim

Route row 10 prescribes a prerequisite probe of the form "add 32 redundant
`buffer_load_b128` behind a runtime-false `select`, measure at prod" -- i.e. price the L1
side of g50 before writing g50.

I built a **strictly stronger** set of three throwaway probes instead, and the reason is
`pitfalls/measurement-traps.md`:

- *"The probe was deleted"* -- a redundant load whose destination nothing reads is exactly
  the thing a compiler deletes, and on gfx1250 the corpus's own two-counter detection is
  unavailable (13 of 51 counters live, memory counters read 0). Disassembly is the only
  check available, so I preferred a probe whose effect is **structural** rather than one
  that has to be kept alive against DCE by a `select`.
- The redundant-load probe prices only **half** of g50 (its cost). It cannot price the
  benefit, and the benefit is the entire premise of the LDS-port hypothesis.

The three probes are all **numerically wrong by construction, throwaway, never shipped**,
and none occupies an arm slot:

| probe | what it does | what it prices |
|---|---|---|
| **P1** | delete the Q/dO LDS round trip; feed the dK/dV GEMM from the row-major fragments the S/P GEMM already holds | **ceiling** of `r16.i3.g50` |
| **P2** | P1, plus 32 fresh in-body `buffer_load_b128` from global at the same addresses the prefetch already read | g50's **actual cost structure** (32 global b128 in-body instead of 32 `ds_load_tr16_b128`) |
| **P3** | delete **all** LDS from `.LBB0_8` -- the P/dS round trip as well -- every WMMA operand from a register the body already holds | **ceiling of the entire LDS-port hypothesis** |

P3 is the one the job has never had. If P3 lands inside the noise floor, the LDS-port story
is dead, `r16.i3.g50` is dead before it is written, and so is every future candidate whose
payoff is "fewer LDS ops on `k_dkdv`". That is worth more than either arm.

⚠ Per `pitfalls/measurement-traps.md` *"A deletion read as a prediction"*: P1 and P3 are
**subtractive**, so they are **upper bounds, not estimates**. P2 is the only one with a
real replacement in it, and even P2 does not carry g50's correct per-lane addressing.

## 2. The probes landed -- verified in the emitted ISA, not inferred

`raw/isa_counts.txt`. `k_dkdv` whole kernel and `.LBB0_8` body, `COMPILE_ONLY=1`:

| | vgpr | spill | LBB0_8 lines | wmma | `ds_store_b128` | `ds_load_tr16_b128` | `buffer_load_b128` |
|---|---|---|---|---|---|---|---|
| **incumbent** | 724 | 0 | 677 | 64 | **40** | **40** | 32 |
| **P1** | 706 | 0 | 560 | 64 | **8** | **8** | 32 |
| **P2** | 834 | 0 | 667 | 64 | **8** | **8** | **64** |
| **P3** | 688 | 0 | 594 | 64 | **0** | **0** | 32 |

WMMA count is 64 in every build: no probe changed the matrix work. P2's +32
`buffer_load_b128` are present in the ISA, so it was not deleted. Spill 0 everywhere.
⚠ P2 at **834 VGPR** is above the pool's `VGPR <= 800` gate for g50 -- still one wave/SIMD
(cap 1024) so it is not an occupancy change, but g50 proper would have to come in under it.

## 3. THE RESULT OF THE ROUND: the LDS-port hypothesis is dead, and so is `r16.i3.g50`

One session, idle card witnessed before and after (`rocm-smi --showpids` = "No KFD PIDs
currently running", `--showuse` 0%), palindromic, 51 timed iterations, median, one process
per shape, sclk 1047-1062 printed per row. Raw: `raw/rows_{prod,proxy}_probe.json`,
`raw/probe_results.txt`.

| arm | prod TF/s | vs incumbent | proxy TF/s | vs incumbent |
|---|---|---|---|---|
| **incumbent** `cur_a` / `cur_b` | 506.98 / 505.63 -> **506.31** | -- (floor **0.27%**) | 442.91 / 438.43 -> **440.67** | -- (floor **1.02%**) |
| **P1** -- Q/dO LDS round trip deleted | **470.99** | **-6.98%** | **413.25** | **-6.22%** |
| **P2** -- g50's cost structure | **413.29** | **-18.37%** | **336.00** | **-23.74%** |
| **P3** -- ALL LDS deleted from `.LBB0_8` | **464.85** | **-8.19%** | **411.71** | **-6.57%** |
| **beat** (re-measured this session) | **720.15** | ratio **0.7031** | **580.89** | ratio 0.7586 |

### 3.1 Read it plainly

**Deleting every LDS instruction from `k_dkdv`'s hot body makes the kernel 8.2% SLOWER.**
P3 removes all 40 `ds_store_b128` and all 40 `ds_load_tr16_b128` -- 40,960 B of LDS traffic
per wave-iteration, the entire 1.25-LDS-ops-per-WMMA figure that `facts.md` calls the one
surviving mechanistic story for the ~686 non-issuing cycles -- with the WMMA count
unchanged at 64 and spill 0. It is 8.2% down at prod and 6.6% down at proxy, same sign,
against a 0.27% / 1.02% same-code floor.

> **Finding (the round's product): `k_dkdv`'s LDS traffic is not what costs it time. The
> LDS-port hypothesis is falsified on card. Its upper bound is NEGATIVE.**

The corpus predicted exactly this shape of answer and this job did not read it in time --
`optimization/routes/1-metrics-to-techniques.md`, row "LDS latency exposure": *"Settled by a
subtractive probe, not by the counter. One kernel read 81% of its CU-cycle budget in LDS
wait; halving three classes of LDS read moved the time 0.0% in all three cases."* And row
"LDS conflict rate": *"19% to 0% moved the time not at all, twice, on a matrix-latency-bound
kernel."* Here it is worse than 0.0%.

### 3.2 `r16.i3.g50` is DEAD, and both halves of it are dead separately

- **Its benefit is negative.** P1 is g50's ceiling -- the Q/dO round trip removed with
  *nothing at all* put in its place -- and it measures **-6.98%**. A subtractive probe is an
  upper bound (`pitfalls/measurement-traps.md`, *"A deletion read as a prediction"*); this
  upper bound is below zero, so no correct replacement can reach parity.
- **Its cost is also prohibitive, which is what route row 10 actually asked.** P2 carries
  g50's memory cost structure -- 32 in-body global `buffer_load_b128` at the same addresses
  the prefetch already read, in place of the 32 `ds_load_tr16_b128` -- and measures
  **-18.37% prod / -23.74% proxy**. The L1 does **not** have the headroom g50 needs. That is
  route row 10's prerequisite probe, executed, answered, negative.

Honest caveats, stated rather than buried: P2's loads are `buffer_load_b128`, not
`global_load_tr16_b128`, so it models g50's *traffic* and not its exact instruction; and
P2 came in at 834 VGPR against g50's own `<= 800` gate. Neither caveat can rescue a
candidate whose *ceiling* (P1) is already -7.0%.

### 3.3 It also kills the candidate this round was going to raise as its second arm

I had a second arm drafted -- software-pipeline the **P/dS** LDS round trip one iteration
deep, so the store->drain->transpose-load chain inside `.LBB0_8` stops being serial. P3
minus P1 prices exactly that: **464.85 vs 470.99, another -1.3%**. Removing the P/dS round
trip entirely is also negative, so pipelining it cannot pay. Not built. Recorded so the
next round does not raise it.

### 3.4 What it means for `bound`

This is the **fifth** time this kernel has got slower when work was removed from it:
`r15.i2.g47`, `r16.i2.g49`, `r18.i3.g56` (-6.84% on a smaller body), `r19.i2.g60` (-17.27%
for 88 fewer VGPR and 82 fewer issue slots), and now P1/P3 (-7.0% / -8.2% for 64 and 80
fewer LDS ops). Against that, the one thing ever measured to ADD state and ADD distance --
`r7.i1.g21`'s cross-iteration prefetch -- is worth **+21%** on today's body.

> **The consistent reading is that `k_dkdv` at one wave/SIMD is short of LATENCY COVER, and
> every instruction you delete is cover you delete.** `bound: latency` stands, and the sign
> of the lever is now known: **add independent work and lengthen load-to-use distances; do
> not remove anything.** That is the opposite of what fourteen rounds of candidates assumed.


## 4. The two arms, chosen by that sign

Both arms were built **separately from `op/current`** and are measured apart. Neither is
built on top of the other. Both come from the same reading of section 3.4: *add independent
work, lengthen load-to-use distances.* Nothing in this round deletes anything.

### 4.1 Arm A -- `r20.i1.g61`, split the S/P WMMA accumulator chain 4 -> 2x2

**What the incumbent does.** The K=128 contraction that builds S and P is issued as ONE
4-deep dependent WMMA chain per `(hh, kh)`: each `v_wmma_f32_16x16x32_bf16` takes the
previous one's D register as its C operand. Four in a row, serially dependent. The only
source-level ILP in that region is the two chains `s_acc` and `p_acc` running beside each
other -- 2-way.

**What the arm does.** Two chains of 2, interleaved `dt % 2`, summed in fp32 at the end.
Same 64 WMMAs, same operands, same LDS traffic -- **4-way** ILP instead of 2-way, and the
matrix latency of each WMMA gets a sibling to hide behind.

**Why now, when this is an old idea.** Because the sign is finally known. The corpus has
said this for a while -- `backends/flydsl/attention/techniques.md`, *"MFMA shape as a
scheduling knob"*: *"For a body that waits on MFMA operand dependences rather than issue
bandwidth, cutting one chain into four is a direct win ... measured +8.7% on a dK/dV
body"*, and *"Dependence-bound goes narrower, issue-bound goes wider."* The open question
was always which column this body is in. Rounds 15-19 answered it by accident and round
20's probes answered it on purpose: five separate deletions each made it slower. It is in
the dependence column.

**Cost, stated up front.** +32 `v_pk_add_f32` per iteration (= 64 scalar fp32 adds),
+28 VGPR. By this round's own evidence, added independent work is not a cost in this body --
which is exactly the claim the arm is there to test rather than assume.

**Not bitwise identical.** The fp32 association of the K contraction changes: `(0+1)`,
`(2+3)`, then their sum, instead of left-to-right. Order is fixed and there are no atomics,
so `op.config.determinism_gate` (bitwise across 200 runs of the same code) is untouched,
and SQNR is the gate for the association change.

### 4.2 Arm B -- `r20.i2.g62`, prefetch depth 1 -> 2

**What the incumbent does.** `r7.i1.g21` issues the 32 `buffer_load_b128` for Q/dO one
iteration early and carries the results through `scf.for`'s state. The loads for iteration
i+1 are issued at the top of iteration i, so the cover available to them is exactly one
loop body.

**What the arm does.** Two stages. The carried tuple holds `pre0` (consumed now, issued two
iterations ago) and `pre1` (consumed next, issued one ago); the body issues the loads for
i+2. Cover doubles to two loop bodies. Prologue issues twice; the tail issues two loads past
the end of the loop, which the true-extent buffer descriptors from `g07` clamp -- the same
mechanism that already makes g21's one-past-the-end issue safe.

**Why.** This is not a new mechanism, it is *more of the one mechanism that has ever won
here*. `r19.i2.g60` measured its removal at -17.27%, i.e. g21 is worth ~+21% on today's
body. If latency cover is the binding constraint, the marginal return on the second stage
should still be positive.

**Cost.** +132 carried dwords -> VGPR 724 -> 912, still one wave/SIMD, and the `h14` risk
is the wholesale copy of a bigger carried tuple.

### 4.3 Build gate (offline, `COMPILE_ONLY`) -- both pass

| | VGPR | spill | scratch | `.LBB0_8` lines | `v_wmma` | `v_pk_add_f32` | `v_mov_b64` | `s_wait_loadcnt` | `v_nop` |
|---|---|---|---|---|---|---|---|---|---|
| `cur_a` (incumbent) | 724 | 0 | 0 | 675 | 64 | 32 | 64 | 9 | 67 |
| `A61` | 752 | 0 | 0 | 698 | 64 | **64** | 64 | 6 | 68 |
| `B62` | 912 | 0 | 0 | 734 | 64 | 32 | **128** | **2** | **40** |

Both intended deltas landed, and nothing unintended did:

- **A61**: `v_pk_add_f32` 32 -> 64 is exactly the +64 fp32 adds the split costs. `v_wmma`
  is **unchanged at 64** -- the arm buys ILP, it does not buy or sell WMMAs. Reading the
  emitted chains confirms the shape: the count of WMMAs issued with a literal `0` as C
  (i.e. chain heads) doubles.
- **B62**: `v_mov_b64` 64 -> 128 is the extra 128 dwords of carried state being copied.
  The two counts worth noticing are **`s_wait_loadcnt` 9 -> 2** and **`v_nop` 67 -> 40** --
  the scheduler needed far fewer explicit waits and far less padding, which is the shape
  "the loads now have cover" takes in the ISA. That is a build-time observation, not a
  result; the card decides.

## 5. Correctness, before any benchmark (`h1` / route row 1)

`op/validation.py` -- the job's own gate, on its own reference -- run on both arms *before*
`benchmark.py`, order on record in `raw/validation_A61_B62.txt`.

| arm | dq / dk / dv, fast | proxy | prod | determinism x200 |
|---|---|---|---|---|
| `A61` (`r20.i1.g61`) | 52.61 / 52.64 / 52.84 dB | 52.52 / 52.57 / 52.67 | 52.56 / 52.60 / 52.71 | bitwise identical |
| `B62` (`r20.i2.g62`) | 52.61 / 52.65 / 52.83 dB | 52.52 / 52.57 / 52.67 | 52.56 / 52.60 / 52.71 | bitwise identical |

`correctness pass`, `determinism pass` for both. `speed FAIL` for both, which is what the
gate says for anything below the bar and is not a result -- the ranking comes from the
same-session run in section 6, not from here.

Worth stating for `g61` specifically: it **changes the fp32 association** of the K=128
contraction, and SQNR does not move -- 52.52-52.84 dB against the incumbent's own
52.52-52.83 dB, to the hundredth of a dB on five of nine numbers. The reassociation is free
in precision. And `B62`'s numbers are bit-identical to `A61`'s on proxy and prod, which is
the expected signature of a pure scheduling change.

## 6. Measurement -- one session, two rebuilt incumbent slots, palindromic, median of 51

Idle-device witness, before and after the session: `rocm-smi --showpids` -> *"No KFD PIDs
currently running"*, `--showuse` -> 0%. sclk 1048-1058 throughout (the box's VR-throttled
ceiling, as every round since 14). `dmesg` monitor armed for the duration: no faults, no
ring resets, no `SMU: No response`. Raw in `raw/meas_arms.txt`.

### prod (b4, s8192, hq32, hkv8, d128)

| arm | TF/s | vs incumbent |
|---|---|---|
| `cur_a` (incumbent, rebuilt slot 1) | 502.13 | |
| `cur_b` (incumbent, rebuilt slot 2) | 500.70 | |
| **incumbent mean** | **501.41** | same-code floor **0.29%** |
| `A61` `r20.i1.g61` | 502.77 | **+0.27% -- INSIDE the floor, NULL** |
| `B62` `r20.i2.g62` | **509.66** | **+1.65% -- 5.7x the floor, WIN** |
| `beat` (re-measured this session) | 717.24 | |

### proxy (b1, s4096, hq32, hkv8, d128)

| arm | TF/s | vs incumbent |
|---|---|---|
| `cur_a` / `cur_b` | 442.52 / 437.94 | mean **440.23**, floor **1.04%** |
| `A61` | 438.23 | −0.45% -- inside the floor, NULL |
| `B62` | 438.16 | −0.47% -- inside the floor, NULL |
| `beat` | 577.91 | |

### fast -- sentinel only, not ranked (`h7`)

The two same-code incumbent slots read 55.34 and 52.02 TF/s: a **6.4% spread on identical
code**. Every arm this round lands inside that. fast confirms nothing crashed; it ranks
nothing. (`A61` 48.34, `B62` 54.64, `cur_b` 52.02, `beat` 50.97.)

### 6.1 Reading it

**`r20.i2.g62` (prefetch depth 2) is the round's win: prod +1.65%, ratio to `beat`
0.6990 -> 0.7106.** It is small, and it is above the noise floor by 5.7x with the floor
measured in the same session from two rebuilt copies of the same source. Its proxy number
is flat inside a 1.04% floor, which is the shape a *latency-cover* change should have:
proxy is 16x smaller, its `.LBB0_8` trip count per wave is far lower, and there is
proportionally less steady state for a second prefetch stage to cover. The ISA said the
same thing before the card did -- `s_wait_loadcnt` 9 -> 2, `v_nop` 67 -> 40.

**`r20.i1.g61` (chain split) is a NULL, and that is information.** +0.27% prod inside a
0.29% floor, −0.45% proxy inside a 1.04% floor. The corpus's *"+8.7% on a dK/dV body"* did
not transfer. The honest reading: **the S/P WMMA chain was not the thing this body waits
on.** Its 4-deep chain has 32 other WMMAs and 80 LDS ops interleaved around it, and the
scheduler evidently already had enough to fill the matrix latency there. What the body
waits on is further upstream -- the *global* loads, which is exactly the thing `g62` moved
and the only thing that has ever moved this kernel.

Taken together with section 3, round 20 has now measured the same statement three ways:
**cover for the global loads is the binding constraint; nothing else in this body is.**
`g61` is the cleanest negative control for that claim the job has produced -- it added 64
independent VALU ops and 4-way ILP to the matrix pipe and bought exactly nothing.

## 7. The merge, and what shipped

Neither arm lost, so the merge is mandatory and was built and measured rather than assumed.
`M6162` = `g61` + `g62` applied to the same `op/current` base. Build gate: **904 VGPR**
(*lower* than `B62` alone at 912 -- the chain split gave the allocator shorter live ranges
that partly paid for itself), spill 0, scratch 0, and both ISA deltas present
(`v_pk_add_f32` 32->64 **and** `v_mov_b64` 64->128).

`op/validation.py` on the merge, before its benchmark: correctness `pass`
(52.52-52.84 dB across all three shapes, identical to both arms), determinism `pass`
(bitwise x200).

### 7.1 Second session -- merge vs its best component vs two rebuilt incumbent slots

Idle before and after (`No KFD PIDs`, 0% use), sclk 1050-1056, `dmesg` clean.
Raw in `raw/meas_merge.txt`.

| arm | prod TF/s | vs incumbent | proxy TF/s | vs incumbent |
|---|---|---|---|---|
| `cur_a` / `cur_b` | 501.79 / 499.86 | mean **500.83**, floor **0.39%** | 443.16 / 437.09 | mean **440.13**, floor **1.38%** |
| `B62` `r20.i2.g62` | 508.50 | +1.53% | 432.88 | −1.65% |
| **`M6162` = `g61`+`g62`** | **511.67** | **+2.16%** | 437.07 | −0.70%, inside floor |
| `beat` (re-measured) | 718.96 | | 579.52 | |

**`g62` reproduced across two independent sessions: +1.65%, then +1.53%.** That is the part
of this round that is solid. The merge is +2.16% prod, i.e. `g61` contributes about
+0.6 points on top of `g62` -- which is *still inside* this session's 0.39% floor by only
1.6x and must not be over-read. It is shipped because it is the best measured arm and
because it is not a loss on any shape outside a floor; it is **not** claimed as evidence
that `g61` works. Section 6.1's reading of `g61` as a null stands.

### 7.2 Shipped

`rounds/020/op/kernels.py` = `M6162`.

- prod **511.67** TF/s, ratio to the same-session `beat` **0.7116** (round 19: 0.7023).
- proxy 437.07 TF/s, flat inside the floor.
- `validation.py` geomean over three shapes: **0.801x** in the merge's own validation run --
  note this is *lower* than round 19's 0.811x while prod is *higher*, because that geomean
  includes `fast`, where the same-code spread is 6.4% (section 6) and a single-slot reading
  is noise. The prod ratio, measured against a `beat` re-measured in the same session with
  two rebuilt incumbent slots for a floor, is the number this round stands behind.

The net for the round: the first genuine forward motion since `g21`, small, and bought by
doing the *opposite* of what the last fourteen rounds tried.

## 8. Corpus consulted this round

Every file opened, whether or not it ended up load-bearing:

| file | what it was used for |
|---|---|
| `optimization/routes/1-metrics-to-techniques.md` | entry point, as the route prescribes -- read first |
| `optimization/techniques/3-pipelining-and-scheduling.md` | **load-bearing twice.** Supplied the ordered knob list (`waitcnt distance -> barriers -> s_setprio -> sched_group_barrier -> staging depth`) that `g62` is item 5 of and `g64` is items 3-4 of; and supplied the rule that *closed* depth 3: *"Staging depth pays exactly once ... do not sweep it as though it were continuous."* Also the write-hiding latency table |
| `backends/flydsl/attention/techniques.md` | supplied `g61`'s claim (*"cutting one chain into four is a direct win ... +8.7% on a dK/dV body"*, *"Dependence-bound goes narrower, issue-bound goes wider"*). **Measured not to transfer** |
| `pitfalls/measurement-traps.md` | *"The probe was deleted"* -- the reason route row 10's runtime-false-`select` probe was replaced with three structural ones (§1); and *"A deletion read as a prediction"* -- why `P1`/`P3` are upper bounds |
| `arch/gfx1250/isa.md` | §11.2.4, the `GLOBAL_LOAD_TR16_B128` entry -- the only corpus mention of the instruction `g50` needed |
| `arch/gfx1250/profiling-surface.md` | confirming there is still no byte-reporting counter at any level, i.e. the L1 question can only be answered by A/B |
| `optimization/techniques/2-lds-bank-conflicts.md` | checked whether the LDS-port story had a second form before killing it |
| `optimization/techniques/0-register-pressure-and-occupancy.md` | the 904/912 VGPR budget at one wave/SIMD, and the spill-0 gate |
| `backends/hipkittens/scheduling-patterns.md` | cross-backend check on the `sched_group_barrier` shape for `g64` |
| `backends/aiter/attention/recipes/fmha_v3_bwd_hd128_bf16.md` | cross-backend: what the bar does about operand staging (it stages global -> LDS, which is `g58`'s closed route) |
| `backends/flydsl/attention/dead-ends.md` | checked both arms against it before allocating ids -- neither appears |

## 9. Not done, and why

- **`r20.i0.trmap`** (an on-card instrument to derive the `GLOBAL_LOAD_TR16_B128` lane->
  element mapping the corpus does not document) was written and **not run**. Its only
  consumer was `g50`, which §3 killed. Running it would have spent card time in a
  measurement session to document an instruction no live candidate uses. The script is left
  in `_scratch/scr/trmap.py` for whoever needs it.
- **Prefetch depth 3** was not built -- see §7 and the corpus rule quoted in §8.
- **Widening `g61` to 4x1** was not built -- `g61` at 2x2 is a null, so the axis is closed,
  not under-explored.

## 10. Step 5/6 -- clean rebuild, unit tests, and the champion re-measure

### 10.1 The compile cache was cleared before the graded build

`/root/.flydsl` inside `fa-repro` held **241 MB** of JIT artefacts. It was removed, every
`__pycache__` under `rounds/{017,019,020}/op` with it, and the tree rebuilt from source. The
cache came back at **17 MB**, which is the proof the binaries measured below are this round's
and not a previous round's served silently.

### 10.2 `op/ut/test_correctness.py` on the shipped working copy -- PASS

All **15 shape x mask-mode combinations** pass, every tensor separately, NaN-poisoned
allocator with full `isfinite` coverage asserted before any SQNR:

`toy`, `gqa4_small`, `mha`, `unequal_seqlen`, `unequal_seqlen_2`, `sq_gt_skv`, `fast`,
`proxy`, `prod`, in causal and full where the config allows -- **52.30 to 52.98 dB**,
gate 50.0. `RESULT: PASS`.

`op/validation.py` on `rounds/020/op` itself (not on the scratch arm): `correctness pass`,
`determinism pass`, `speed FAIL` (exit 2 -- the bar is 1.40x away and no round has been
within reach of that gate). Raw in `raw/ut_and_validation_shipped.txt`.

### 10.3 The champion re-measure -- one session, idle device

`rounds/017/op` (the incumbent, md5-identical to `op/current/`), a **second physical copy**
of it for the same-code floor, `rounds/019/op` (the other distinct champion), the shipped
`rounds/020/op`, and `beat` -- back to back, all three shapes, 51 iterations, median,
palindromic order. `No KFD PIDs` and 0% use before and after; sclk 1051-1056 on prod/proxy,
1100 on fast; `dmesg` clean. Raw in `raw/meas_champions.txt`.

| shape | `r017` | `r017_b` | floor | `r019` | **`r020` (shipped)** | best ever | `r020`/best | `beat` | ratio |
|---|---|---|---|---|---|---|---|---|---|
| **prod** | 501.41 | 498.14 | **0.66%** | 500.47 | **511.42** | 501.41 | **1.0200** | 715.94 | 0.7144 |
| **proxy** | 442.54 | 434.74 | **1.78%** | 437.02 | 435.93 | 442.54 | 0.9851 | 575.17 | 0.7579 |
| **fast** | 55.09 | 51.62 | **6.52%** | 51.46 | 55.07 | 55.09 | 0.9996 | 50.95 | 1.0000 |

**score = mean(ratio) = 0.8241.** Geomean throughput across the three shapes:
`r017` 230.36, `r019` 224.10, **`r020` 230.69** TF/s.

### 10.4 Against the acceptance rule, stated plainly

1. **Throughput improves on the best ever, re-measured beside it.** prod **511.42 vs 501.41
   = +2.00%**, against a same-session floor of 0.66% measured from two physical copies of
   the incumbent -- 3.0x the floor. This is the third independent session in which the
   `g62` mechanism has come out ahead on prod (+1.65%, +1.53%, +2.00%).
2. **Every below-target shape is at least 95% of its own best ever.** prod 102.00%,
   proxy **98.51%**. proxy's shortfall is 1.49% against a **1.78%** same-code floor in that
   same session -- it is inside the noise, not a regression, and it is the same flat-inside-
   floor reading proxy gave in both earlier sessions.
3. **No shape that had reached its target falls below it.** `fast` is the only one above
   target (55.07 vs 50.95); it stays above. Its same-code floor is **6.52%**, so `h7` still
   forbids ranking on it -- it is reported as a sentinel.

All three conditions hold. Whether that promotes is Python's call, from these numbers.

### 10.5 What is in the working copy

`rounds/020/op/kernels.py` = `M6162` = `r20.i1.g61` + `r20.i2.g62`, the arm that was built
and measured. It is left in place. It is not a revert and it is not the incumbent: md5
`51c4bf60...` against `op/current/`'s `83ee83de...`, 67 changed lines in one file.
