# Facts -- gfx1250-flydsl-attn-bwd

This file is the compact inherited state of the run. Candidate entries begin with their id;
standing facts do not. Every result names the round that established it.

## Current bottleneck

⚠ **ROUND 22 SUPERSEDES ROUNDS 20 AND 21. Read this block first; the two below are history.**

**bound: latency. confidence: MEDIUM.** The word is unchanged from round 21. What changed is
that round 21's *explanation* for it is now falsified, and the replacement has made a correct
prediction.

**Round 21's "per-kernel signed lever" is WITHDRAWN.** It said `k_dkdv` and `k_dq` have
opposite signs on the prefetch-spread lever, discriminated by body shape (`k_dq` 96 `v_wmma` /
16+16 LDS ops, `k_dkdv` 64 / 40+40). Round 22 built exactly the arm that model says should
*gain* -- `r22.i2.g68`, spread `k_dkdv`'s prefetch -- and it lost **-30.0% prod / -26.1%
proxy** against same-session floors of 0.40% / 2.08%. **Spreading the load clump loses on both
kernels. There is no sign flip.** Do not reason from "this kernel is compute-dense /
memory-heavy" again.

**What replaces it: cover distance.** The binding quantity is the **worst-case load-to-use
distance between a prefetch load's issue and the next *full* `s_wait_loadcnt 0x0`**. Incumbent
`k_dkdv`: last load idx 188, drain at idx 13 of the next iteration, **~603 instructions**.
`g68`: last load idx 789, drain at idx 25, **~41**. One variable explains `g28` (-7.8%, x3
sessions), `g62` (+1.65%), `g63` (-19.33%), `g66` (-21.93%), `g68` (-30.0%) and the probe
`P70` (-10.86%), and it **predicted `g68`'s sign and magnitude before any card time**.

**The filter the next round must apply first:** *any edit that moves the last prefetch load
later in the body loses* -- five for five. Before building on the prefetch axis, answer: **is
this a re-spread in disguise?** The prefetch depth/placement axis is otherwise exhausted:
both directions, both kernels, plus the subtractive ceiling (`P70`: deleting `k_dq`'s prefetch
costs 10.86%, so "shorten it" has a negative ceiling too).

**Why confidence is MEDIUM and not HIGH.** Three independent reorder-only results this round
are all negative and none can happen on a bandwidth-bound body: `g68` changed only issue
position (identical bytes, WMMA count, LDS op count, VGPR, spill 0) and cost 30%; `P70`
changed no HBM byte at all and cost 10.86%. That is a strong case for *latency*. But gfx1250
exposes **no byte-reporting and no stall-reason counter at any level**, so every positive
statement here is inferred from controlled A/B deltas and disassembly, never read off an
instrument. Raising this to HIGH needs an instrument this part does not have.

**Power is eliminated, not assumed away.** `Current Socket Graphics Package Power (W)` reads
855.0 idle and a median of **853 across a full prod run** (60 samples) -- a static register,
not a measurement, so the power-cap question is **untestable on this box**. And sclk is flat
> ⚠ **RETRACTED 2026-09-24 (end of day). THIS PARAGRAPH IS FALSE.**
> The 60-sample trace it rests on was taken while **nothing was running**.
> `rounds/022/_scratch/scr/pw.sh:12` passes `--arms cur_a`, which `benchmark.py:160-162`
> rejects via `ap.error` -> `SystemExit(2)`; stdout and stderr go to `/dev/null` and the
> script ends in `wait`, so it reported rc=0 and produced 60 plausible-looking idle rows.
> The real 264-sample 1 Hz trace across the actual scored block
> (`rounds/022/_scratch/run/meas/power_trace.txt`) shows **1100 MHz on only 67 samples,
> 1040-1055 on 12, and <= 1030 on 185** — 1100 is the IDLE clock and the prod window runs
> 998-1029 MHz, about a 9% droop. `benchmark.py:11-12`'s own comment already said the VR
> limits to 1100 and drifts to 967 inside a timing window.
> **"The card was throttling" is NOT ruled out.** What survives: same-session palindromic
> A/B exposes every arm to the same droop, so A/B *deltas* remain valid.
> Package power was never measured under load at all — 851/853/855 W are all idle, and
> `meas.sh:10` greps `Average Graphics Package Power` while this box prints
> `Current Socket Graphics Package Power`, so the loaded trace captured no power column.

at **1100 MHz for 60/60 samples** under sustained load. *"The card was throttling"* is no
longer an available explanation for any measurement in this job.

⚠ **The number the next round should actually be looking at.** Round 22's same-session
champion re-measure puts **`r017` 505.85, `r019` 507.13 and `r020` 516.87 within 2.2% of each
other at prod**, and the measured null-control floor on this part is **1.57% (n=5)** -- which
is where `g62`'s banked "+1.65%" sits. **This operator has not moved outside the noise since
round 17.** The bar is "beat the champion by more than the floor", not "be positive".

Round 22 was **not accepted**: prod 361.7 is 70.0% of its best ever 516.9. `op/current/` holds
round 20. Per-shape best-ever, re-measured this round: prod **r020 516.87**, proxy **r019
443.73**, fast **r019 55.32** -- they are not the same round.

---

⚠ **ROUND 21 (superseded by the block above; its model is withdrawn). Round 21 superseded round 20.**

**bound: latency. confidence: MEDIUM -- but the lever is now known to be PER-KERNEL SIGNED.**

Round 21 took round 20's one positive mechanism -- deepen the cross-iteration prefetch,
`+1.65% prod` on `k_dkdv` -- and applied it unchanged to `k_dq`. It lost **-19.33% prod**
(`r21.i1.g63`, dead_ends). A second arm built purely to isolate the cause, `r21.i4.g66`
(pool), bought the same load spread through `sched_group_barrier` at **VGPR 960 -> 960,
spill 0** and lost **-21.93%**. So the loss is not spill, not register pressure, not code
size. **It is the spread itself.**

**The correction the next round must carry:** "add independent work and lengthen load-to-use
distance" is a **`k_dkdv`** result. On `k_dq` the sign is reversed -- its tight ~50-instruction
prefetch clump is load-bearing, and widening it is worth about -20%. The two kernels differ in
exactly the way that predicts this: `k_dq` is 96 `v_wmma` / 16+16 LDS ops (compute-dense, loads
already covered), `k_dkdv` is 64 / 40+40 (memory-heavy, loads need cover). `r22.i1.g67` and
`r22.i2.g68` in the pool are the two halves of testing that sentence.

**The model did NOT survive the measurement.** `g63`'s named metric moved exactly as predicted
-- `s_wait_loadcnt` 4 -> 2, the full `0x0` drain at 93% of the body gone -- and the kernel got
19% slower anyway. Confidence stays MEDIUM rather than rising: "latency" is still the right
word for what round 20 measured, but round 21 shows that *which* latency and *which direction
helps* is not something the wait-counter census can tell you. `h18` restated: an ISA delta is
necessary, never sufficient.

**Retired as a reason to spend card time:** *"VGPR has headroom and spill is 0."* That was the
entire pre-build case for `g63`. Spill 0 remains a hard gate (`h3` -- spill is a kill, not a
cost); it is no longer evidence that an arm will win.

Round 21 was **not accepted**: prod 412.1 is 80.9% of its best ever 509.1. `op/current/` holds
round 20. Per-shape best-ever after this round's re-measure: prod **r020 509.15**, proxy
**r019 438.99**, fast **r019 54.73** -- they are not the same round.

---

⚠ **ROUND 20 SUPERSEDES ROUND 19'S BLOCK BELOW. Read this first.**

**bound: latency. confidence: MEDIUM** -- unchanged in name, but for the first time supported
by a POSITIVE result rather than only by falsifications.

What changed: `r20.i2.g62` deepened the cross-iteration Q/dO prefetch one stage and was worth
**+2.00% prod** against a re-measured champion on a 0.66% floor, with `s_wait_loadcnt` **9 -> 2**
and `v_nop` **67 -> 40** in the ISA. Adding latency cover pays. Meanwhile `r20.i1.g61` added
4-way matrix ILP and 64 independent fp32 ops for **+0.27% inside a 0.29% floor** -- the matrix
pipe is not what this body waits on.

**Round 19's LDS-port hypothesis is FALSIFIED.** Three on-card probes: deleting the Q/dO LDS
round trip is −6.98%, deleting **all 80** LDS ops with the WMMA count unchanged is −8.19%.
Those LDS ops are cover, not cost. See `dead_ends.md`, *"The LDS-port story ... (falsified in
round 20)"*. With it go `r16.i3.g50` and every candidate arguing "fewer LDS ops".

**What the next round should take from this:** the sign of the lever is now measured, and it
is the opposite of what fourteen rounds of candidates assumed. Six separate deletions have
each made this kernel slower (`g47`, `g49`, `g56`, `g60`, P1, P3); the one mechanism that has
ever moved it -- `r7.i1.g21`'s cross-iteration prefetch -- moved it again when deepened.
**Add independent work and lengthen load-to-use distance. Do not remove anything.** Register
headroom is what buys cover, so spill 0 is the gate, not occupancy (`h12`, `h14`, `g60`).

**Why still MEDIUM and not HIGH:** gfx1250 exposes no counter that attributes stall cycles by
reason, so "latency" remains an inference from A/B alone -- a strong one now (one positive,
eight negatives, all measured), but still one agent's reading with no counter behind it. The
specific number that is now UNSUPPORTED is round 19's attribution of the ~686 non-issuing
cycles: the calibration below still stands, the mechanism assigned to it does not. A deep round
should treat the calibration as a budget to explain, not as a solved one.

---
⚠ **ROUND 19 SUPERSEDES ROUND 18'S BLOCK BELOW. Read this first.**

**bound: latency. confidence: MEDIUM** -- raised from LOW, and for the first time on a
measurement rather than on a model.

What raised it: `r19.i2.g60` removed 88 VGPR, 57 `s_set_vgpr_msb` and 58 `v_nop` (-71%) from
`k_dkdv` with spill 0 and bit-identical arithmetic, and ran **17.27% slower at prod**
(417.73 vs 504.90 TFLOP/s, same session, same device). A kernel that gets slower when you
delete 82 issue slots is **not issue-bound**, so `h16`'s "issue roof" premise is dead along
with the last reason to trade `BLOCK_KV` for registers. Compute was already out (2048
FLOP/SIMD-cycle, confirmed twice; 30.4% matrix busy in the hot body). Memory is out for the
ninth time, now by arithmetic alone: `h8`'s own 107 GB at the 4.39 TB/s HBM roof is **24.4 ms**
against a **measured 10.909 ms** total -- HBM cannot physically be supplying that traffic, so
`facts.md`'s 86x byte figure is not measuring HBM. What is left is latency.

Calibration, which is new and is the number to argue with: prod `k_dkdv` is 62.93% of
10.909 ms over 4,177,920 loop iterations across 1024 SIMDs = **~1809 real cycles per `.LBB0_8`
iteration** against an issue-only floor of 1123. **~686 cycles (38%) are not issuing.**

The one surviving mechanistic story for those 686 cycles is **LDS port pressure, not
bandwidth**: `k_dkdv`'s `.LBB0_8` holds 40 `ds_store_b128` + 40 `ds_load_tr16_b128` =
**1.25 LDS ops per WMMA** against `k_dq`'s **0.33** (3.8x), and 64 of the 80 are the Q/dO
store/transpose-load round trip (40,960 B per wave-iteration, ~91 B/cycle/CU). `k_dq`, whose
body has zero modelled stall and 54.3% matrix busy, is the control. It is a hypothesis, not a
result -- **no arm has tested it**, `r16.i3.g50` is the arm that would, and gfx1250 exposes no
counter that reports bytes at any level, so it can only ever be settled by an on-card A/B.

**Why MEDIUM and not high:** this reading rests on one falsification (`g60`) plus a census,
by one agent, with no counter behind it. A deep round should trust the *negative* results here
(not compute, not HBM, not issue slots -- each has a measured arm behind it) more than the
positive LDS-port story.

**What the next round must not do:** rank candidates by static issue slots, instruction count,
wait-site counts or `attrib*.py` output. That tool is retired to a build gate at round 19
(see `dead_ends.md`) after a fourth misprediction, and body size has now reversed sign twice
(`r17.i2.g53`, `r19.i2.g60`). Prefer semantics-preserving throwaway probes on card.

Same-session figures, round 19 (re-measure every round, never carry forward): incumbent
**504.90 / 440.29 / 54.08** TFLOP/s (prod / proxy / fast, two rebuilt slots each, floors
0.61% / 1.42% / 5.74%), bar **719.50 / 584.97 / 52.06**, prod ratio **0.7023**,
`validation.py` geomean 0.811x. Round 19's shipped arm `r19.i1.g59` is a null (+0.08% prod)
and the round was not accepted; `op/current/` still holds round 17.


⚠ **ROUND 18 SUPERSEDES THE PARAGRAPH BELOW. Read this first.**

**bound: latency. confidence: LOW, and the word "low" is load-bearing -- round 18 falsified its
own model by measurement.** Read this before ranking anything by `.LBB0_8`'s stall budget.

Round 18 built the candidate that the stall budget below points at: it removed the source
instruction of the site charged with **470 of `.LBB0_8`'s 562 stall cycles (83.6%)**.
`buffer_load_b32` 4 -> 0, `s_wait_loadcnt` 7 -> 3, body 678 -> 667 instructions, VGPR 724 -> 718,
`v_wmma` unchanged. It was **correct** (dk/dv 52.60 / 52.71 dB at prod, determinism x200) and it
measured **prod 469.56 vs 504.03 = -6.84%**, proxy -8.40%, fast -5.53%, same sign on all three
against a 0.36% same-code floor. See `dead_ends.md` `r18.i3.g56`.

> **The numbers below stand as measurements. The model that turns them into a ranking does not.**
> `.LBB0_8`'s decomposition (1685 cyc = 512 WMMA + 611 issue + 562 stall) is reproducible and the
> 1.9x `k_dkdv`/`k_dq` per-element spread is real. What round 18 disproved is the inference from
> them: that removing the largest stall site buys time. It does not, and the job has now seen
> three cases (`r15.i2.g47`, `r16.i2.g49`, `r18.i3.g56`) where every static indicator improved
> and the kernel got materially slower. The first two constrained the scheduler; the third did
> not, so that explanation is exhausted.

**What the next round should NOT do:** rank candidates by static issue slots, wait-site counts, or
the stall budget alone. **What is missing:** an instrument that prices latency rather than locating
it -- concretely, an ASYNCcnt queue in the attribution tool (`route.md` row 5), without which
`r18.i5.g58` and `r16.i3.g50` cannot be priced at all.


Round 18 re-measured both sides in one session: the incumbent is **504.84 TFLOP/s prod**
(three rebuilt slots, 0.21% spread) and the bar is **720.98 TFLOP/s** (re-measured, up from
709.89), so the ratio is **0.7002**. Round 17's `bound: compute` and every ranking built on
65.6% / 86.4% / 58.9% are **void**: the gfx1250 bf16 WMMA rate is **2048 FLOP/SIMD-cycle**, now
confirmed twice by independent routes (throughput back-solve 29.5%, and `attrib_ss.py` at
WMMA_cost=8 giving `k_dkdv` `.LBB0_8` 30.4% matrix busy), so those utilisations all halve.

The prod gap is **not** all the 7/5 structural factor. Per S-element non-WMMA issue is 0.315 for
`k_dq` against **0.597** for `k_dkdv` -- 1.9x. `k_dq`'s hot body has **zero stall** at
WMMA_cost=8 (pure issue-bound, fully latency-covered). If `k_dkdv` matched `k_dq`'s per-element
efficiency, prod would be ~7.29 ms ~= **754 TF/s**, above the bar, without touching the fusion
question. `k_dkdv` `.LBB0_8`'s budget is 1685 cycles = 512 WMMA (30.4%) + 611 other issue (36.3%)
+ 562 stall (33.4%), and **470 of the 562 stall cycles are one site** (`@228 loadcnt N=35` popping
a `buffer_load_b32` stuck behind the 32-deep b128 prefetch in the in-order LOADcnt FIFO).
Evidence: `rounds/018/1-opt/opt.md` sections 2, 3 and 11.

Round 18 also established two negative constraints: the tile-widening axis of `k_dkdv` is closed
by measurement on **both** axes (`r3.i5.g14` KV, and the q axis via `r10.i2.g28`'s mechanism --
the 470 cycles scale with the batch, they do not amortise), and **the offline screen (h3) cannot
see correctness for an async-to-LDS change** -- `r18.i3.g56` passed all four screens with
`spill: 0` while producing dk/dv at -56/-73 dB.

---

Round 17 leaves **prod compute/work-count bound, medium confidence**. The accepted implementation
runs at **507.790 TFLOP/s, 0.70611x beat** at prod; the apparent +0.61% over the rebuilt round-16
champion is noise because the prod kernel ISA is byte-identical. The round-17 profile measured
issued/algorithmic WMMA FLOP at **1.4080 = 7/5**, whole-op issued matrix utilisation at **65.6%**,
`k_dq` at **86.4%**, and `k_dkdv` at **58.9%**. The only assumption capable of flipping the word
"compute" is the undocumented gfx1250 bf16 rate: 1024 FLOP/SIMD-cycle is inferred; 2048 would
halve those percentages. Evidence: round 17 `1-profiling/6-bound-analysis/analysis.md`,
`1-profiling/profiling.yaml`, and `3-act/act.yaml`.

The separate fast concurrency bottleneck is now closed. Round 17 split `k_dq`'s causal-effective
KV range with `nsp_q=8`, raised `SQ_WAVES` **128 -> 1024**, and measured **56.451 TFLOP/s** versus
the rebuilt fast champion's **38.896** and beat's **50.915**. Fast is score-capped at 1.0;
subsequent score progress must come from proxy (**0.77307**) or prod (**0.70611**). The measured
cap curve was `{4: 52.27, 8: 53.69, 16: 46.40}` TFLOP/s: the gain is the net of better fill,
duplicated prologues, and reduction cost, not eightfold fill alone. Evidence: round 17
`3-act/act.yaml` and `3-act/act.md`.

Round 17 did **not** produce a stall attribution, memory-byte measurement, or thread trace. PC
sampling faulted, ATT captured no runtime-loaded FlyDSL kernel, and kernel metrics were skipped.
Static ISA properties remain build gates only, never time predictors. Evidence: round 17
`1-profiling/profiling.yaml` coverage and campaign correction 3.

## Confirmed facts

## r1.i1.g01 -- causal tile skipping in both main kernels

Round 1 removed fully masked causal tiles and measured **1.326x geomean**, including **1.680x at
prod**, with bitwise identity where compared and the full correctness/determinism gates passing.
The gain shrinks at shallow grids because removing work can remove idle time rather than busy time.

## r1.i2.g02 -- full 32-query contraction in `k_dkdv`

Rounds 1-2 replaced two half-empty K=32 WMMAs with a full contraction: **48 -> 32 WMMA per
32-query step**. After the round-2 address clamp it shipped in the accepted merged path, measured
at **1.893x end to end** against the then-incumbent.

## r1.i3.g03 -- dispatch expensive causal workgroups first

Round 6's grid-axis permutation measured **+2.60% / +2.68% prod**, bitwise identical, and shipped.
The unimplemented AITER-style `j`/`n-1-j` pairing is no longer live: round 13's census put prod
`k_dkdv` at **100.0% dispatch efficiency**, and round 17 measured fast `k_dkdv_sp` at exactly
**1024 waves**, so pairing would reduce a grid that is already shallow.

## r1.i5.g05 -- fast-shape grid starvation, resolved per kernel

Round 14 tuned `k_dkdv_sp` to **1024 waves** with `nsp=16`; its `nsp=32` control measured **0.911x**.
Round 17 closed the remaining `k_dq` half by raising it **128 -> 1024 waves** with `nsp_q=8`.
Fast then reached **56.451 TFLOP/s**, above its **50.915** target.

## r1.i6.g06 -- widen `k_dkdv` KV coverage from 16 to 32 on one wave

Round 2 measured **1.677x** for the 16->32 rung, with fewer global and transpose loads per unit
work. The next 32->64 rung is the separate dead end `r3.i5.g14`.

## r1.i7.g07 -- true descriptor extents suppress the nondeterministic over-read fault

Round 2 replaced flat 1 GiB `num_records` on `k_dkdv` buffers with true extents. It did not locate
the original over-read, but formerly faulting 32-deep arms subsequently passed repeated full gates.
The clamp is a safety constraint and costs no measurable time.

## r2.i2.g09 -- remove duplicate Q/dO global loads in `k_dkdv`

Round 4 reused the normal global fragments for both the S/P WMMAs and LDS stores, removing the
separate staging reads. It measured **+11.0% geomean alone** and shipped. The 32 remaining
`buffer_load_b128` are global-to-register and dual-use; there is no global-to-LDS staging burst
left for TDM to replace.

## r3.i1.g10 -- widen `k_dq` query tile from 16 to 32

Round 3 measured **1.193x end to end** (`1.083/1.201/1.305` fast/proxy/prod), with zero spill and
bitwise identity. More work per wave paid despite lower residency.

## r3.i2.g11 -- store dQ/dK/dV as bf16 in the kernels

Round 3 removed three conversion launches and fp32 output temporaries. The conversion was RNE and
bitwise identical; measured speed was **1.017/1.010/1.000x**, so this is mainly a simpler output
path rather than a prod performance lever.

## r3.i3.g12 -- remove the duplicate K load in `k_dq`

Round 5 reduced `k_dq` loop-body `buffer_load_b128` **48 -> 32**, VGPR **826 -> 800**, with no
spill and bitwise identity. It measured **+0.9% prod, 0.0% proxy, -1.9% fast** and shipped as a
free structural cleanup. It is a standing example that byte removal does not predict time.

## r3.i4.g13 -- widen `k_dq` query tile from 32 to 64

Round 3 measured **+6.29% geomean** (`+1.0/+6.1/+12.1%` fast/proxy/prod); `k_dq` itself fell
**33.4%** at prod. The build used 826 VGPR, zero spill. Round 17's split preserves this tile and
adds parallelism along the KV loop instead of undoing its density gain.

## r4.i1.g16 -- avoid the 64-way LDS row-stride collision

Round 4 padded the 256-byte Q/dO row and 64-byte P/dS row to strides that walk the 64 banks.
It measured **+10.7% geomean alone** and **+50.2%** when merged with `r2.i2.g09`.

## r5.i1.g17 -- cache FlyDSL compiled launchers on the host path

Round 5 replaced repeated JIT cache-key resolution with `flyc.compile`'s compiled-function path.
Host issue fell **0.2688 -> 0.0313 ms**, fast improved **36.2%**, and geomean improved **11.6%**
with byte-identical GPU code. Re-measure host share whenever short-shape GPU work changes greatly.

## r6.i1.g19 -- split `k_dq` into mask-free and masked KV loops

Round 6 removed causal predicates from the full region and measured **+3.03% / +2.95% prod** in
two sessions, bitwise identical across all three shapes.

## r6.i2.g20 -- transpose the GQA loop and split `k_dkdv`'s masked body

Round 6 made the mask-free split expressible in `k_dkdv`; it shipped at only **+0.66% prod**.
The fixed loop order is deterministic but intentionally not bitwise identical to the previous
accumulation order.

## r7.i1.g21 -- prefetch `k_dkdv` Q/dO one iteration ahead

Round 7 carried the next iteration's 32 b128 fragments through the loop state and measured
**+8.3% prod** on that baseline. Later rounds established that preserving their long
load-to-use distance is load-bearing.

## r8.i1.g23 -- delete the conservative single-wave barriers in the main kernels

Round 8 removed barriers whose workgroups are one wave and measured **+6.27% prod**, bitwise
identical. The paired `r8.i2.g24` change did the same for the sibling single-wave kernel; both
must be restored before any multi-wave rewrite.

## r9.i1.g25 -- keep the rare masked `k_dkdv` loop from carrying the full prefetch tuple

Round 9 changed the allocator shape and reduced `k_dkdv` **956 -> 740 VGPR** with zero spill.
Time was null, but the register result is shipped and is the current allocation.

## r9.i2.g26 -- apply cross-iteration K/V prefetch to `k_dq`

Round 9 measured **+11.40% / +11.60% prod** in two sessions; `k_dq` itself fell **23.0%**.
VGPR rose **812 -> 960**, still zero spill and one wave/SIMD. The last-iteration load must remain
clamped to its own tile because `k_dq` descriptors retain flat extents.

## r10.i0.g29 -- shipped-ISA census instrument

Round 10 established a reproducible census of loop bodies, waits, register-window prefixes, and
instruction classes. Rounds 10-17 established its boundary: it proves what a build contains but
does not rank performance.

## r12.i0.g38 -- occupancy steps and corrected allocation interpretation

Round 12 measured `regs_per_multiprocessor = 131072`, giving **1024 VGPR per wave at one
wave/SIMD** and a two-wave threshold of **512**. Campaign correction 1, applied in round 17,
established that rocprofv3 reports half the ISA allocation: current `k_dkdv` is **740 VGPR** and
`k_dq` **960**, so both are one wave/SIMD. `k_dkdv` is VGPR-capped; its 70,656-byte LDS block
costs no occupancy.

## r12.i2.g40 -- dispatch longest `k_dq` query tiles first

Round 12 changed only grid ordering and measured **+4.91% / +4.98% prod**, bitwise identical.
It is the second confirmation that causal tail ordering can matter even when per-wave issue
efficiency looks high.

## r12.i3.g41 -- grid census distinguishes ordering problems from underfilled one-wave grids

Round 13's census found both prod main kernels at **100% dispatch efficiency**, proxy `k_dkdv`
at **50.4%** with exactly one dispatch wave, and shallow fast grids. Ordering cannot improve a
single wave launched all at once; splitting is required. This produced `r13.i1.g42` and round
17's `r17.i1.g52`.

## r13.i1.g42 -- deterministic q-loop split-K for `k_dkdv`

Round 13 measured **1.7729x fast, 1.3343x proxy, 1.0040x prod** and shipped. It writes fp32
partials and folds them deterministically; prod uses the untouched `nsp=1` path.

## r14.i1.g44 -- derive and measure the `k_dkdv` split cap

Round 14 measured fast `nsp=16` as the interior optimum; an `nsp=32` control measured **0.911x**.
The constants belong to that body and were not assumed for round 17's `k_dq` split.

## r14.i2.g45 -- one fused deterministic dk/dv reduction kernel

Round 14 replaced four torch reduction/conversion launches with one flat kernel. It measured
roughly **+21-25% fast** and **+4% proxy**, with prod on the untouched path. Fixed ascending fp32
folding is deterministic but may differ by a few bf16 elements from torch's pairwise reduction.

## r16.i4.g51 -- pin LSE/delta loads ahead of the Q/dO prefetch loads

Round 16 combined the four b32 cross-iteration loads with a hard scheduling fence placed between
them and the 32 b128 loads. It measured **+0.66% prod / +1.03% proxy / -0.20% fast**, below the
card-level 1.57% floor but directionally positive, and passed correctness/determinism. The same
fence after the b128 group is the `r16.i2.g49` dead end at -16.25% prod.

## r17.i1.g52 -- split `k_dq` only where its own grid underfills the machine

Round 17 added `k_dq_sp` plus a deterministic one-tensor reducer. The cap sweep measured
`nsp_q={4,8,16}` at **52.27/53.69/46.40 TFLOP/s**; 8 is the interior optimum. At the scoring
measurement, `SQ_WAVES` rose **128 -> 1024**, fast measured **56.451 TFLOP/s** versus the rebuilt
champion's **38.896**, and exceeded beat **50.915**. Correctness passed 15/15 at **52.30 dB**
minimum and determinism passed 200 runs. Proxy and prod use a separate `nsp_q=1` kernel whose ISA
is byte-identical to round 16; their apparent +1.94%/+0.61% are noise, not gains. Round 17 also
showed that a single same-code prod spread of 0.32% is not a confidence interval: the predeclared
+/-0.3% band was exceeded by byte-identical code.

## Constraints

### Profiling surface

Round 17 narrowed the old blanket rocprofv3 ban: SQ/GRBM/I-cache counter passes work when no
faulting reference GEMM is dispatched. PC sampling remains categorically banned after a fourth GPU
fault. ATT is safe but captures no `hipModuleLoad` JIT FlyDSL kernels. gfx1250 exposes no measured
memory bytes, WMMA FLOP counter, useful occupancy counter, or stall-reason counter. Never invent a
roofline or a stall attribution from those absences.

### Static metrics and occupancy

Round 17 and campaign correction 3 make the rule absolute: ISA counts, waits, VGPRs, and schedules
are gates for spills, full drains, and build identity; they do not order candidates. Correct every
rocprofv3 `VGPR_Count` by two before occupancy arithmetic. Shrinking `k_dkdv` LDS cannot raise
occupancy because 740 VGPR already forces one wave/SIMD.

### Correctness and output ownership

Rounds 1-17 require NaN-poison coverage checks, the full shape suite, and 200-run bitwise
determinism. Atomics are outside the accepted design because each output must have a deterministic
single writer or fixed-order reduction. A split may change association and remain deterministic;
record that separately from bitwise identity to the prior implementation.

### Measurement discipline

Round 17 confirmed that fast is highly arm-order sensitive and that absolute results from different
arm tables cannot be subtracted. Use same-session rebuilt champions, multiple replicas for fast,
one process per shape, and an idle-device witness. Never ship a prod regression below the round's
own measured noise floor. The prod noise floor still needs a real repeated null measurement; the
round-17 0.32% single comparison was not a confidence interval.

### Shape-boundary facility fault

Rounds 14-17 observed intermittent `fast -> proxy` HSA memory faults with changing addresses and no
GPU reset. Round 17 reproduced one fault after correctness, then reran unchanged successfully.
Treat it as the documented delayed `hipModuleUnload`/GC facility hazard, not candidate evidence;
continue using one benchmark process per shape.

### TDM and global-to-LDS staging

Campaign correction 6, applied in round 17: current `k_dkdv` has no global-to-LDS staging burst.
Its 32 b128 loads land in registers and feed WMMAs directly, with the same registers later stored
for the transpose path. TDM or cluster-global-to-LDS proposals undo `r2.i2.g09` and abandon
`r7.i1.g21`; they are not free substitutions.

### FlyDSL lowering traps

Round 17 found two reusable compiler facts. A one-element `flyc.jit` carried state is unpacked to
the value itself at the loop return boundary, unlike a multi-element state. Also, source evaluation
order changes final scheduling and register allocation: hoisting an epilogue address CSE changed
`k_dq` from 3750 to 3741 ISA lines and produced 2111 diff lines until the original left-to-right
evaluation order was restored.

## r20.i2.g62 -- deepen `k_dkdv`'s cross-iteration Q/dO prefetch from one stage to two

Round 20, accepted, score 0.85104. `r7.i1.g21`'s prefetch carried one stage; `g62` carries
`pre0`+`pre1`, issues the loads for iteration `i+2`, issues twice in both prologues, and lets
`r1.i7.g07`'s true buffer extents clamp the two-past-the-end tail issue (no explicit clamp in
`k_dkdv` -- `k_dq` does NOT get this for free, see `r21.i1.g63`).

- ISA: VGPR 724 -> 912, spill 0, scratch 0. `s_wait_loadcnt` **9 -> 2**, `v_nop` **67 -> 40**.
- prod **+1.65%**, reproduced **+1.53%** in a second independent session, **+2.00%** against a
  re-measured `rounds/017/op` in the champion session (511.42 vs 501.41 TFLOP/s) against a
  0.66% same-code floor from two physical copies of r017.
- Merged with `r20.i1.g61` (mandatory, neither arm lost): **904 VGPR -- lower than g62 alone**,
  prod 511.42, proxy 435.93, fast 55.07. That merge is what `op/current/` now is.
- `r11.i5.g37` is the same idea attempted at round 11 and dead; it failed on register pressure
  before `g52`/`g25` had trimmed the carried state. The idea was not wrong, the round was early.

⚠ **Depth 3 is not a candidate.** `s_wait_loadcnt` is already at 2; there is no third wait to
move, and `optimization/techniques/3-pipelining-and-scheduling.md` says staging depth pays
exactly once.
