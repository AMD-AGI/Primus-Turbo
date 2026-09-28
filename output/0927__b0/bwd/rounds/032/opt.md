# Round 32 -- fast round, agent-directed

Incumbent `op/current` = `rounds/029/op` (r19h + h70/u2n + `r27.i3.g86`).
`rounds/032/op` started byte-identical to it (`diff -rq` clean).
Everything below ran through the runner (`docker exec fa-g3`), physical GPU 3 (h65).
Bulk under `/tmp/op-evolve-...-r032`; what was parsed is in `raw/`.

## Why this round did not open with a candidate

`pool.md` had exactly one open entry (`r27.i2.g85`) and it had been passed over five
consecutive rounds. Round 31 was not accepted. Both of the prompt's triggers for
corpus reading were live, so the round opened by reading rather than by editing.

`explored.consulted`:
- `job_context/findings/facts.md` (747 lines, whole)
- `job_context/findings/dead_ends.md` (whole)
- `job_context/findings/pool.md` (whole)
- `job_context/findings/route.md` (whole)
- `job_context/op/current/kernels.py` (whole, 1598 lines)
- `job_context/op/current/{benchmark.py,validation.py}`
- `knowledge/hardware/gfx1250_isa_notes.md` (wave32, DPP absent, `DS_LOAD_TR16_B128`,
  `DS_PERMUTE_B32` index bits [6:2], `s_set_vgpr_msb`, `S_WAIT_DSCNT`/`S_WAIT_LOADCNT`)
- `knowledge/profiling/` -- consulted as a reference for counter definitions only
  (the chapters were NOT run; the gfx1250 counter surface is 8 counters and none of
  them is an instruction, LDS or byte counter, so no chapter is executable here)
- `knowledge/corpus/` cross-lane transpose note (the source of `r27.i2.g85`)
- `rounds/029/op/kernels.py`, `rounds/030/3-act/raw/{dump_isa.py,sample.py}`,
  `rounds/031/1-opt/raw/{bench2.sh,isa.sh}`, `rounds/031/.../power_*_prod.txt`

## r27.i2.g85 is killed offline, on its own terms

The pool entry's own blocking condition was "re-derive the corpus's wave64 lane
arithmetic as gfx1250 wave32 + index bits [6:2]". Doing that derivation kills it, and
it is free, so it is settled here rather than deferred a sixth time.

The corpus's +1.55x was measured on gfx950/MI355X, which is **wave64**, and against a
`ds_read_b64_tr_b16` baseline (8 B/lane). On gfx1250 both terms move against it:

1. `DS_PERMUTE_B32`/`DS_BPERMUTE_B32` take index bits [6:2] -- 32 lanes, byte
   addressed. Wave32 halves the permute's reach; a wave64 shuffle pattern does not
   port, it has to be rebuilt as two halves plus a merge.
2. `kernels.py` is **already on `ds_load_tr16_b128`** (helper `tr()`, one instruction,
   16 B/lane, transposed on the way out). The replacement instruction
   `ds_bpermute_b32` moves 4 B/lane and does not transpose. That is 4:1 against the
   permute on data moved per instruction *before* any index arithmetic or re-packing.

Two independent confirmations already on file: `g35` (the wave64-era cross-lane VALU
transpose) was killed offline in round 12 for the same reason, and round 20 measured
that **deleting all 80 LDS ops in `k_dkdv` costs 8.19%** -- the Q/dO LDS round trip is
cover, not cost, so removing it cannot be the win the corpus advertises.

Moved to `dead_ends.md`. The five-round deferral is resolved.

## C1 -- the prod clock limiter is not thermal

facts.md's open question was "prod sustains ~1753-1790 MHz vs 2362 idle, it is not the
PPT limit, the actual limiter is unknown; thermal and VR are untested". Answered from
round 31's own stored power/clock traces (`rounds/031/.../power_*_prod.txt`), no new
GPU time:

- junction rises 51.7 -> 76.7 C in about 4 s and then is **dead flat**, against
  `temp2_crit` 105 C and `temp2_emergency` 120 C. 28 C of headroom, no decay.
- sclk drops 2364 -> ~1800 MHz **inside one 200 ms sample**, with no ramp and no
  recovery, and it settles at a *different* value per arm (1760 / 1790 / 1818 MHz),
  reproducibly.

A thermal limit ramps and decays. This is an instantaneous activity/current setpoint.
Thermal is eliminated; VR/current is the remaining candidate and is consistent with
the memory note that this box is VR-throttled. `amd-smi` reports
`THROTTLE_STATUS: N/A` but does expose per-XCD `GFX_0..GFX_7` clocks, which is an
instrument nobody has used and the only remaining way to read this directly.

Consequence for this round: the clock is arm-dependent, so any candidate's measured
delta is partly a clock response, and per-arm clock has to be read beside the time.

## C2 -- transcendental rate (instrument, 1 wave/SIMD, `raw/trate_1wave.txt`)

| op | latency (ch1) | rel | throughput (ch8) | rel |
|---|---|---|---|---|
| `v_mul_f32` | 2.173 ns | 1.00 | 0.456 ns | 1.00 |
| `v_fma_f32` | 2.173 ns | 1.00 | 0.469 ns | 1.03 |
| `v_cvt_pk_bf16_f32` | 2.178 ns | 1.00 | 0.485 ns | 1.06 |
| `v_exp_f32` | 3.461 ns | 1.59 | 0.886 ns | **1.94** |
| `v_rcp_f32` | 3.468 ns | 1.59 | 0.897 ns | 1.97 |
| `v_exp_f16` | 3.469 ns | 1.59 | 2.175 ns | **4.77** |

Two results:

- `v_exp_f32` is **half rate** on gfx1250, as on gfx950. Pricing it into the hot loop:
  32 exps/iteration * ~1 extra cycle = ~32 cycles of 2282 = **1.4% of `k_dkdv`
  ~= 0.86% of prod**. That is the whole ceiling of any exp-count arm.
- `v_exp_f16` is **2.4x worse than f32**, not better. An f16-softmax arm is dead
  before it is built. This is the one place where the obvious port is backwards.

This *contradicts* the standing hypothesis. Rounds 29/31 shaved exponent work and lost
(`g91` -0.600%, `g92` -3.309%); the transcendental rate was the last available
explanation for why shaving did not pay, and at 0.86% of prod it is far too small to
be it. Something else absorbs removed work -- see C3.

## C3 -- one wave cannot overlap VALU with the matrix pipe (`raw/ovl_1wave.txt`)

Mixed (WMMA, VALU) points against the strict sum of the pure rows:

| point | measured | sum of parts | delta |
|---|---|---|---|
| 4W 16V | 25.56 | 25.55 | +0.0% |
| 4W 32V | 32.38 | 32.54 | -0.5% |
| 4W 64V | 47.00 | 47.17 | -0.4% |
| 8W 16V | 41.03 | 42.07 | -2.5% |
| 8W 32V | 48.07 | 49.06 | -2.0% |
| 8W 64V | 62.71 | 63.69 | -1.5% |

Every mixed point is within ~2.5% of the strict sum. **A single wave gets essentially
no VALU/matrix overlap on gfx1250.** Derived constants: single-wave VALU throughput
~1 instr/cycle at >=5-way ILP; VALU dependent latency ~4.8 cycles; WMMA throughput
~4.27 ns (~10 cycles), WMMA dependent latency ~6.5 ns (~15 cycles).

`k_dkdv` runs at **1 wave/SIMD** (facts.md), so this is its regime exactly.

The three LDS rows (`0L`/`8L`) of that table are **confounded and must not be quoted**:
I put `s_wait_dscnt 0x0` after every `ds_load_b32`, which serialises all eight. They
are recorded in `raw/` for honesty and are not used below.

## The budget this buys -- and why count-shaving is provably finished

`k_dkdv`'s hot loop `.LBB0_11` is 660 instructions, iteration ~2282 cycles
(`raw/k_dkdv_hotloop_mix.txt`, `raw/k_dkdv_hotloop_LBB0_11.s`):

- 64 WMMA at ~10 cycles issue = **~646 cycles (28%)**
- ~596 VALU/SALU/LDS/load at ~1 cycle = **~628 cycles (28%)**
- **~1008 cycles that are neither = 44%, pure wait**

This is the first *measurement* behind facts.md's unmeasured guess that removing VALU
"removes scheduler slack rather than work". It says more than that. C3 says the pipes
are additive, so deleting N instructions from the body should return N cycles --
~0.044% of prod each. `g92` deleted 29 instructions (660 -> 631) and should have
returned +1.3%; it measured **-3.31%**. The 1008-cycle wait term is **elastic**: it
grows to absorb whatever issue slots you free.

An elastic wait that refills is a **latency floor** -- iteration time is set by a
dependence chain whose length does not change when independent work is removed. That
closes the entire instruction-count axis (rounds 24, 29, 31, arms g30/g73/g36/g74/
g59/g61/g31/g83/g91/g92) not as a run of bad luck but on mechanism, and it says the
only lever left is **the chain itself**.

## Where the chain is -- the wait census

Waits in the body, in order (line numbers within `.LBB0_11`):

```
 49: s_wait_loadcnt 0xc     228: s_wait_loadcnt 0x23    507: s_wait_loadcnt 0x7
 52: s_wait_loadcnt 0x4     316: s_wait_loadcnt 0x22    514: s_wait_loadcnt 0x6
485: s_wait_dscnt  0x6      488: s_wait_loadcnt 0x20    557: s_wait_loadcnt 0x4
491: s_wait_dscnt  0x2      494: s_wait_dscnt  0x0      600: s_wait_loadcnt 0x0
```

Line 494 is a **full LDS drain** at instruction 494 of 660, immediately before the 32
dV/dK WMMAs. The structure it sits in is:

  [softmax] -> `ds_store_b128` P/dS -> `ds_load_tr16_b128` P/dS -> **wait dscnt 0** -> dV/dK WMMAs

The P/dS round trip is stored and re-read **inside one iteration**, with nothing
between the store and the load to cover it, and -- at 1 wave/SIMD -- no second
resident wave to hide it either. By contrast the Q/dO round trip is already covered:
its stores are at the top of the `hh` loop (`kernels.py:398-404`) and its `tr()` loads
at `:480-483`, with the whole S/P WMMA run and the softmax in between. Q/dO has cover.
P/dS has none. That asymmetry is the candidate.

## 66 `v_nop` -- 10% of the body, never counted in this campaign

Never appeared in any prior round's instruction accounting. Positions within the hot
loop: 130-133, 204-207, 237-239, 321-324 (18, attached to WMMAs and one after a
`v_exp_f32`) and **508-653 (48, inside the prefetch-tuple rotation block)**.

The mechanism, read off the disassembly:

```
v_wmma_f32_16x16x32_bf16 v[58:65], v[182:189], v[146:153] /*srcB = v[402:409]*/
s_wait_loadcnt 0x7
v_nop x4
v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[110:111]   <- clobbers that srcB
v_mov_b64_e32 v[148:149] ...
```

The register allocator has **coalesced the carried prefetch tuple's rotation
destination onto the registers holding this iteration's transposed WMMA B operands**.
Each rotation copy that overwrites a just-read WMMA source pays the gfx12 WAR hazard
in dead issue slots. The pattern repeats ~15 times.

Cost: 66 cycles of 2282 = 2.9% of `k_dkdv` ~= **1.76% of prod**, and it is joined by
109 `s_set_vgpr_msb` in the same body (175 instructions, 26% of the loop, doing no
arithmetic -- ~4.7% of prod). This is the largest identified block of pure waste in
the kernel. It is also *not* what `ku2`/`g72` attacked: those deleted the rotation by
unrolling (+6.5% and -4.52%); the nops are a placement problem, not a count problem.

Recorded as an observation. It is not one of this round's two arms, because the only
levers on it from FlyDSL source are indirect (the scheduling-intrinsic family is dead,
0-for-5) and `g22` established that a pure source reorder of LDS ops came back
byte-identical. It is filed to the pool with the instrument that would settle it.

## Also checked and dismissed on paper, no arm spent

- **Bandwidth.** `P77` pinned the whole Q/dO locality effect at <=1.83% and showed it
  is clock-mediated; the compulsory traffic floor is ~0.161 TB/s. Not the limiter.
- **The 1.4076 WMMA FLOP redundancy** (7 GEMMs where 5 suffice). Every route out is
  already measured and dead: `FUSED5` 0.39x, 4-wave 0.872x, `g76` HBM materialisation.
- **f16 softmax.** Killed by C2 before construction (`v_exp_f16` 4.77x).
- **Locked-clock instruments.** Permanently barred (`/sys` read-only; DPM offers only
  500/2364/2400 MHz).

## Arms

Both arms attack the **wait term** via the P/dS round trip -- the one structure the
census shows is uncovered. They touch the same lines, which the round's rules permit
when declared: they are two doses of one mechanism at very different blast radius, and
the small one is insurance that the mechanism gets measured even if the large one
cannot be built inside the round. Each was built **alone from `op/current`**; B is
from the same base as A, never on top of it.

### r32.i1.g93 -- P/dS double buffer, one-iteration software pipeline (`trees/A`)

Iteration i stores P/dS into ring[i&1]; the 64 dV/dK WMMAs read ring[~i&1], written by
iteration i-1. Q/dO doubled with it (the WMMAs consuming ring[~i&1]'s P/dS must pair it
with the same iteration's Q/dO). LDS 70656 -> 75776 B, which still gives
floor(327680/75776) = 4 workgroups/CU, so occupancy is unchanged.

The parity has to be a **Python int, not an `fx.Int32`**. With a runtime parity LLVM
cannot prove the read ring and the write ring are disjoint and keeps the conservative
wait regardless, which makes the whole arm a no-op. A compile-time parity only exists
if the loop is unrolled by two -- so this arm is necessarily built on `KV_U2`, the
existing x2 unroll, and `r32.i2.g94` below is its control.

Correctness: the first piped body reads ring 1 before anything has written it, so ring 1
is zeroed at workgroup entry (44 `ds_store_b128`, once, against ~508 qloop iterations).
`a_p = 0` zeroes the WMMA contribution arithmetically, but `0 * Inf = NaN`, so the ring
has to be really zeroed rather than merely unread. The last piped body (par=1) leaves
P/dS in ring 1 with no consumer, so `_qloop_full_any` runs one extra dV/dK phase after
the loop -- once per workgroup, not per iteration. When `n2 == 0` the loop never ran and
ring 1 is still the zeroed image, so the drain adds exactly nothing.

**Build failure 1** (`scratch/isa/A/log`, first gate attempt, `RC A 1`):
`NameError: name 'pipe' is not defined` at `kernels.py:299`, in `_ldqd`. The patch
anchored the parity block on `q0 = qt * fx.Int32(32)`, which occurs in `_ldqd` at line
273 **before** `_body` at 355, so the block landed in the wrong function. Re-anchored on
the `_body` signature. Second gate attempt `RC A 0`.

### r32.i2.g94 -- the x2 unroll alone (`trees/B`), the control

One line, `KV_U2 = False -> True`. It exists to separate the pipeline's contribution
from the unroll's, because A cannot be built without the unroll. Both are built alone
from `op/current`; neither is built on top of the other.

### Free static gate (h3 / h18) -- and the arm is falsified before it reaches the card

| tree | ISA VGPR | spill | AGPR | hot loop | WMMA | instrs/WMMA | `v_nop` | `s_set_vgpr_msb` |
|---|---|---|---|---|---|---|---|---|
| `op/current` | 729 | 0 | 0 | 660 | 64 | 10.31 | 66 | 109 |
| B (unroll) | 644 | 0 | 0 | 1129 | 128 | **8.82** | 53 | 211 |
| A (pipeline) | 679 | 0 | 0 | 1129 | 128 | **8.82** | 53 | 211 |

Spill 0 everywhere, so h3 passes and both arms are allowed to the card.

**A and B have the same hot loop.** 1129 instructions, 128 WMMA, 53 `v_nop`, 211
`s_set_vgpr_msb`, and an *identical* wait list, including the `s_wait_dscnt 0x0` the arm
was built to remove. Diffing the two blocks line by line gives 461 differences and every
one of them is SGPR renumbering (`s_abs_i32 s20, s13` vs `s_abs_i32 s22, s15`).

The parity is not the problem -- it reached the ISA. A's hot loop carries 39 distinct
LDS offsets including the whole ring-1 family (17408, 17440, 17472, ... = QSEG + k);
B's carries 19 and none above 8800. The two rings are genuinely disjoint at compile
time, by constant offsets, and the whole-file `ds_store_b128` count is 236 for A against
192 for B -- the +44 are the ring-zeroing stores, so the drain and the zeroing are both
present and correct.

So: **given provably disjoint, compile-time-constant LDS rings, the LLVM scheduler still
does not hoist the transposing loads across the stores.** It keeps the original order and
the original full drain. The schedule is pinned by something other than the memory
dependence, which is the third independent observation of the same thing -- `g22` (source
reorder of LDS transposes came back byte-identical) and the scheduling-intrinsic family
(0 for 5) being the other two.

This is a falsification of the arm's premise, obtained for free, before any GPU time. It
is reported rather than buried: the mechanism (an uncovered P/dS round trip at 1 wave/
SIMD) is still real and still the largest single item in the wait budget, but **it is not
reachable through the LLVM scheduler from FlyDSL source.** Reaching it needs a lever this
job does not have -- see the pool entry.

Both arms were measured anyway. An arm that is falsified statically is still cheap to run
beside the champion, and B's number is wanted independently: it is the first measurement
of the x2 unroll in the light of C3.

## Measurements

Two sessions on the idle physical GPU 3 via `fa-g3`, blocked ruler (lead 4 + block 9,
palindromic, h66), one process per shape, all arms interleaved inside it, two passes,
fresh `FLYDSL_RUNTIME_CACHE_DIR` per session (h72). `rocm-smi --showpids`/`--showuse`
before and after both: **GPU use 0%, no KFD PIDs on this card**.
`raw/bench_champ.sh`, `raw/bench_arms.sh`; JSON under `scratch/bench_champ/`,
`scratch/bench_arms/`.

### Session floor (same code, two passes)

| shape | floor this session |
|---|---|
| prod | **0.405%** |
| proxy | 0.046% |
| fast | 1.514% |

"Lost" below means below the incumbent by more than these, and these are what the arms
are graded against -- not a carried constant.

### The bar and the incumbent, re-measured in session

| shape | `beat` ms | `r029` (= `op/current`) ms | ratio |
|---|---|---|---|
| prod | 6.7197 / 6.7305 | 8.3467 / 8.3806 | **0.805x** |
| proxy | 0.45698 / 0.45855 | 0.56923 / 0.56897 | 0.804x |
| fast | 0.09376 / 0.09262 | 0.05508 / 0.05426 | 1.70x (we lead) |

Unchanged from round 30/31's reading: prod ~0.80x of the bar, and `fast` is ours.

### The arms

Positive = faster than the incumbent.

| arm | prod | proxy | fast | verdict |
|---|---|---|---|---|
| `r32.i1.g93` (pipeline, A) | **-6.573% / -6.180%** | -6.820% / -6.322% | -2.239% / -5.313% | ❌ LOST |
| `r32.i2.g94` (unroll alone, B) | **-6.510% / -6.156%** | -6.859% / -7.237% | -4.380% / -4.105% | ❌ LOST |

**Both arms lost, and they lost by the same amount.** A and B differ by 0.063 and 0.024
percentage points at prod, against a 0.405% floor -- they are one measurement, twice.
That is the card confirming what the static gate already said: the double buffer, the
drain and the ring zeroing changed the ISA's data flow and changed **nothing** about the
schedule, so they changed nothing about the time.

The `fast` column is claimed as **nothing** in either direction: the floor there is
1.514% and A's two passes disagree by 3.07 points.

## What the losers are worth

The round's most valuable number is `r32.i2.g94`'s, and it is worth more than a win
would have been.

`B`'s hot loop does **the same work in 14.5% fewer instructions per unit of work**
(8.82 instructions per WMMA against the incumbent's 10.31; 53 `v_nop` per 128 WMMA
against 66 per 64; VGPR 729 -> 644, 0 spill). Every static metric this campaign has ever
used to argue for an arm moves the right way, several of them by a lot. It measured
**-6.3% at prod**, reproducibly, on both passes, in both shapes that rank.

That independently reproduces the operator lab's `ku2` figure (h71: +6.5% slower) on
this session's clock and on the post-u2n champion, which had never been done here.

And it is the sharpest confirmation available of C3's model:

> the ~1008 cycles/iteration of wait are **elastic**. Free up issue slots and the wait
> grows to absorb them. Iteration time is set by a dependence chain, and neither
> deleting independent instructions nor deleting carried state shortens that chain.

Three arms now agree, across two kernels and two mechanisms: `g92` (-29 instructions,
-3.31%), `g93`/`g94` (-1.49 instructions per WMMA, -6.3%). **The instruction-count axis
is closed on `k_dkdv` by measurement, not by exhaustion.**

The chain itself is still the right target -- the P/dS round trip is still uncovered and
is still the largest single item in the wait budget. What this round establishes is that
**it cannot be reached from FlyDSL source through the LLVM scheduler**, by a dependence
change (this round), by a source reorder (`g22`), or by a scheduling intrinsic (0 for 5).
A remedy needs a lever on instruction placement that this job does not currently have.
That is what goes to the pool, and it is a different kind of item from anything in it.

## Correctness

`op/validation.py` through the runner, on the shipped tree and on A, using the gate's own
precision check -- nothing written for the occasion.

| tree | `validation.py` exit code **as measured** | correctness | determinism | speed |
|---|---|---|---|---|
| `rounds/032/op` (shipped = `g94`) | **2** | pass | pass | FAIL |
| `trees/A` (`g93`) | **2** | pass | pass | FAIL |

Exit 2 is the job's standing state and is unchanged from round 31: the target (h69,
proxy AND prod each >= `beat`) is not met at 0.744x / 0.751x. **Both arms are correct**;
they are slow. `g93`'s pass matters independently -- it restructures LDS into two rings,
zeroes one at entry and adds a drain phase after the loop, so a correctness failure was
the live risk, and the arm's -6.3% is a real timing of correct code rather than an
artefact of a broken one.

Round 32 ships **`r32.i2.g94`** into `rounds/032/op` as the round's best arm, per the
rule that a losing round still ships its best arm and reports every arm. It is
`outcome: delivered`, **not accepted on speed** -- it is 6.3% slower than the incumbent
and `job_context/op/current/` is untouched and stays `rounds/029/op`. A ranks marginally
worse than B on both prod passes (-6.573/-6.180 against -6.510/-6.156), inside the floor,
and B is one line against A's restructure, so B ships.

## What this round did not do, stated so the absence reads as a decision

- **No `rocprofv3 --stats` survey.** h4 says rocprofv3 is a dead end on this op and the
  gfx1250 counter surface is 8 counters, none of them an instruction, LDS or byte counter.
  Round 31 already took the survey this would repeat. The three instruments bought instead
  (C1/C2/C3) each answered a question `facts.md` records as open, and C2 and C3 are
  measurements the counter surface **cannot** produce at all.
- **No locked-clock instrument.** Permanently barred: `/sys` is read-only and DPM offers
  only 500/2364/2400 MHz.
- **No arm on the `v_nop` hazard**, despite it being the largest pure-waste term found.
  The lever does not exist yet; filed as `g96` with three free first cuts instead of
  spending an arm on a lottery.

---

# Step 5-6 -- building and measuring what the route says

The `## Route` table's two `idea` rows are row 17 (`r32.i3.g95`) and row 18
(`r32.i4.g96`). Taken from the top. Row 18 first, because its condition schedules it
as `grep` rather than as card time and it costs nothing to settle before a build.

Compile cache cleared before the first build (`/tmp/flycache_r32_*`, `~/.cache/flydsl`):
a backend serving a stale kernel would have this round measure round 31's binary and
report it as this round's, with no error anywhere.

## Row 18 -- `r32.i4.g96`, the three free first cuts

Full census in `raw/g96_cut1_kdqg_census.txt`. Cut (1) is decisive and it is the first
concrete mechanism this campaign has for why the two kernels convert instructions into
time at different rates:

| | `k_dqg` `.LBB0_4` | `k_dkdv` `.LBB0_11` |
|---|---|---|
| hot loop lines | 1312 | 661 |
| `v_nop` | 31 (2.4%) | 66 (10.0%) |
| `v_mov_b64` (rotation copies) | **1** | **64** |
| WMMA | 192 | 64 |
| **`v_nop` per WMMA** | **0.161** | **1.031** |

And what *precedes* each `v_nop` run: in `k_dkdv`, **28 runs are preceded by
`v_mov_b64_e32`** -- the rotation copies -- against 17 by a WMMA. In `k_dqg`, **zero**
runs are preceded by a `v_mov_b64`; every one follows a WMMA or an `s_set_vgpr_msb`.

So `k_dqg` issues **three times** the WMMA of `k_dkdv` inside the same loop, carries
**one** rotation copy, and pays **6.4x fewer** `v_nop` per WMMA -- none of them
attributable to a rotation. The hazard is **rotation-specific, confirmed**. The
residual WMMA-preceded `v_nop` is the ordinary gfx12 back-to-back WMMA hazard and is
present in both kernels; it is not the defect.

Correction to the pool entry as filed: it predicted `k_dqg` carries **zero**
`v_mov_b64`. It carries **one** in the hot loop (126 whole-file, all outside
`.LBB0_4`). One against 64 -- the prediction holds in substance.

Cut (2), also free: of the three trees already on disk, `B`/`g94` is the one that
avoids the coalescing, taking `v_nop`/WMMA from 1.031 to 0.414 -- and it **measured
-6.3%**. So a tree that avoids the hazard exists, is already built, and is slower.

Cut (3) **NOT TAKEN**, and stated as a decision. Cuts (1) and (2) together price the
remedy, and h18 forbids filing an arm justified by "fewer `v_nop`" -- that ledger is
5 for 5. **No arm filed.** Row 18's outcome is the mechanism, which is what it was
scheduled to buy.

## Row 17 -- `r32.i3.g95`, the register-carried dV/dK pipeline

**Built from `op/current` with `KV_U2 = False`**, exactly as the row's condition
requires -- not on top of `g94`, which the same row records at -6.3%.

### What it does

The four transposed A operands (P and dS, one pair per `kh`) are produced at the
**tail** of iteration `i` and consumed at the **top** of iteration `i+1`, carried in
the `scf.for` state. The B operands had to move with them: `_p3` reads `lds_do`/`lds_q`
at the top of the body, **before** the staging loop overwrites that image with tile
`i`'s Q/dO, so A and B both come from iteration `i-1`. The S/P GEMM was never a
problem here -- it feeds from the prefetch **registers** (`qfr`/`dfr`), not from LDS.

Peel and drain, both required and both cheap: `_qloop_full_any` primes the carried
operands with **zeros** and drains one `_p3` after the loop, where `lds_do`/`lds_q`
still hold the final tile so the pairing holds. The Q/dO LDS image is **zeroed once at
kernel entry** (34 blocks, outside every loop) because the primed zero A operand
multiplies whatever is in that image on iteration 0, and `0 * NaN = NaN` -- LDS is not
zero at wave start. `qloop_mask` keeps the old in-iteration phase unchanged (`ap=None`).

### The free kill gate -- PASSED, and it is the interesting part

Full table in `raw/g95_static_gate.txt`. The gate asked whether `s_wait_dscnt 0x0`
still heads the dV/dK run. It does not -- **the dV/dK run does not exist any more.**

| | `op/current` | `g95` |
|---|---|---|
| ISA VGPR | 729 | 889 |
| spill | 0 | **0** |
| hot loop lines | 661 | **634** |
| `v_nop` | 66 | **26 (-61%)** |
| WMMA | 64 | 64 |
| head of dV/dK run | `s_wait_dscnt 0x2` -> WMMA -> `s_wait_dscnt 0x0` -> WMMA | no wait gates any dV/dK WMMA |

The 64 dV/dK WMMAs are now **interleaved into the softmax VALU stream**, each sitting
between `v_pk_add_f32` of the *next* iteration's dS computation. Three rounds of asking
the scheduler for this politely -- `g22` (byte-identical output), the five scheduling
intrinsics (0 for 5), `g93` (ignored a provably disjoint LDS ring) -- produced nothing.
**A carried SSA value produced it on the first attempt.** That is a real result about
this toolchain regardless of the timing, and it is the answer to the question `g93`
was built to ask.

The +160 VGPR against a predicted +32 is the interleaving paying for itself: holding
both phases live at once extends the B operands' and the softmax temporaries' live
ranges across each other. Spill stayed 0, 889 < 1024, and `k_dkdv` was already at
1 wave/SIMD at 729, so **no occupancy rung is crossed** -- h12/row 4 says the number is
a spill screen and nothing else, so this went to the card on that basis.

### Measured -- five arms interleaved, two passes, one process per shape

Device idle before and after (`No KFD PIDs`, GPU use 0%), fresh cache dir, blocked
ruler, `beat` and all champions re-measured in the same session. `sclk_start` 1762 /
`sclk_end` 1796 on every arm -- identical clock excursion, so no arm is being scored
against a different clock.

| arm | prod p1 / p2 (ms) | vs `r029` | proxy vs `r029` | fast vs `r029` |
|---|---|---|---|---|
| `beat` | 6.72572 / 6.71212 | +24.33% / +24.87% | +24.04% / +23.42% | -40.32% / -41.23% |
| **`r029` (= `op/current`)** | 8.36219 / 8.38134 | -- | -- | -- |
| `r030` | 8.35900 / 8.39468 | +0.04% / -0.16% | +0.84% / +0.09% | +0.47% / -1.02% |
| `r031` | 8.41585 / 8.42785 | -0.64% / -0.55% | -2.33% / -1.05% | +0.18% / +0.71% |
| **`r32C` (`g95`)** | 8.63451 / 8.64822 | **-3.15% / -3.09%** | -0.25% / -1.09% | +1.73% / -0.66% |

Session floor (same code, two passes): prod **0.23%**, proxy **0.81%**, fast **2.18%**.

**`g95` LOST.** -3.15% / -3.09% at prod is seven times the prod floor and repeats in
sign and size across passes -- it is not noise. Proxy is inside its own floor and fast
is far inside its own; **prod is the only shape that ranks (h7)** and prod is decisive.

`r029` is re-confirmed champion: `r030` ties it inside the floor, `r031` is slightly
behind. `op/current` does not move.

### What this measurement means -- the fourth confirmation of C3, at kernel scale

`g95` is the first arm in this campaign that **attacked the wait term and reached it**.
It removed the blocking `s_wait_dscnt 0x0`, it got the matrix work interleaved into the
VALU stream, it cut `v_nop` by 61%, and it made the loop body *shorter* -- and it still
lost 3.1%.

`r32.i0.C3` predicted exactly this and the arm is its strongest confirmation. At
1 wave/SIMD gfx1250 gives **no VALU/matrix overlap** -- every mixed point in the
microbenchmark was within 2.5% of the strict sum of its parts. Interleaving WMMA into a
VALU stream is therefore worth **nothing** on this hardware at this occupancy; the two
streams do not co-issue, so a WMMA placed between two `v_pk_add_f32` still costs its
full issue slot, and the added register pressure costs on top. The instrument measured
that on two kernels of microbenchmark; the arm has now measured it on the real kernel.

**So the mechanism is settled, and it is not "shorten the dependence chain" either.**
h14/row 5 said a loss here must be read as a failure of the u2n re-attribution rather
than as a register-pressure result, and that is how it is recorded. The remaining
mechanism that attacks the ~1008-cycle wait term is **a second resident wave** -- and
h59/row 15 already states why that is blocked by arithmetic, not by preference.

`g95` is nonetheless the round's **best arm**, and by a factor of two: `g93` -6.4%,
`g94` -6.3%, `g95` -3.1%. It is what `rounds/032/op` ships.

## Correctness of the shipped tree (`rounds/032/op` = `g95`)

Both gates, through the runner, on the tree that was benchmarked (`md5=a22bd125`):

- **`op/ut/test_correctness.py --impl rounds/032/op`** -- 15 shape x mode cases
  (`toy`, `gqa4_small`, `mha`, `unequal_seqlen`, `unequal_seqlen_2`, `sq_gt_skv`,
  `fast`, `proxy`, `prod`, causal and full where the config allows), every tensor
  52.43-52.98 dB against a 50.0 dB gate, every element finite under a NaN-poisoned
  allocator. **RESULT: PASS, exit 0.**
- **`op/validation.py`** -- correctness **pass** (dq/dk/dv 52.52-52.83 dB on the three
  scored shapes), determinism **pass** (dk/dv bitwise over 200 runs, dq bitwise against
  a 70 dB floor), speed **FAIL** (fast 1.696x, proxy 0.796x, prod 0.778x; the gate
  needs proxy AND prod each >= 1.00x). **Exit 2.**

This matters more here than usual: `g95` adds a peel, a drain, a one-time LDS zeroing
and a lag between the A and B operands of 64 WMMAs. A pairing error would have shown up
as a silent numerical error on exactly one tile per workgroup, and `sq_gt_skv` and the
two `unequal_seqlen` shapes are the cases that would catch it. They pass. The precision
check was the gate's own -- not one written for the occasion -- and it was not weakened.

## What is in the working copy, and what is not

`rounds/032/op` = `g95`, the arm that was measured, left in place after losing.
`outcome: delivered`. `job_context/op/current/` was never edited and is still
byte-identical to `rounds/029/op`.

## What this round did not do

- **No merge arm.** The rule is to merge only if neither arm lost. All three lost.
- **No fourth idea.** The route's ceiling is four `idea` rows; it carried two, and row
  18 was scheduled as `grep` precisely so it could not consume card time.
- **No re-measurement of rounds before 28.** Different architecture, replaced by the
  round-28 refactor; several are faster than anything on the current structure and none
  of them is a standard for this round.
- **No weakening of any gate**, and no mechanism switch mid-round: `g95` is the
  hypothesis row 17 named, built where row 17 said to build it (`KV_U2 = False`).
