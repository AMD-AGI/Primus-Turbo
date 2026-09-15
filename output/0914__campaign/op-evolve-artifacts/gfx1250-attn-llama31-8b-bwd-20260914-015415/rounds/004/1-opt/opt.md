# Round 4 -- fast round, gfx1250 attention backward (Triton one-kernel fused bwd)

Written as the round went, including the two things that did not work and the
session hazard I had to measure around.

## 0. What I read first

`findings/facts.md`, `dead_ends.md`, `pool.md`, `route.md`, then round 3's
`opt.md` in full (it is the round that closed the launch surface and it carries
the corrections the pool text does not).

Position inherited: **1.343x against a 1.50x gate**, short by 10.5%. The whole
launch/tile/warp/stage surface is measured in both directions and is a strict
local optimum. The compiler-env surface loses or is inert. PC sampling and every
counter path are dead on this part. The pool held exactly one open structural
item (`r1.i7.g7`, LDS staging) whose stated blocker is "Triton cannot express
it", and one (`r3.i4.g15`) that round 3 killed with a grep.

So: pool effectively empty, last round not accepted. That is the state step 3
says sends you to (c) and (d), and it is also the state that says the next
candidate has to come from a part of the surface nobody has touched. **The part
nobody has touched is the Triton `tl.range` / `tl.assume` surface** -- neither
appears anywhere in this job's four rounds, and both are source-level knobs on
the exact loop that owns the time.

## 1. Session hazard, measured before anything else

`rocm-smi --showpids` on the physical GPU is **not clean this session**, unlike
round 3's. Three processes hold VRAM on GPU[1]:

| host pid | container | what | VRAM |
| --- | --- | --- | --- |
| 3033885 | `fa-e2e` | `primus/cli/main.py train pretrain ... repro_l8b_turbo_compile.yaml` | 32.5 GB |
| 3130507 | `fa-repro` | `tools/gfx1250/tune_attention.py --shape par-b1h32 --impl fused` | 416 MB |
| 3136371 | `fa-repro` | a `python -c` attention diagnostic | 404 MB |

None is mine -- all three are in other tenants' containers -- so they are not
mine to kill. All three are in state `R` burning ~100% of a CPU each with
**GPU[1] `use` reading 0% on 12 consecutive 2-second samples over 24 s**, i.e.
they are stuck host-side and are not submitting GPU work. The node has 255
cores, so the CPU contention is not material.

**Consequence carried through the whole round:** every timing session is
palindromic with a base slot first AND last, and I report the base-to-base
spread as the floor rather than importing round 2's 0.03-0.08% or round 3's
0.9%. If the two base slots disagree, the session is void and nothing in it is
believable. Re-checked after every session.

## 2. The free census that killed one arm before it was measured

`tl.range(..., loop_unroll_factor=N)` and `tl.range(..., disable_licm=True)`
both exist in this Triton (3.6.0) -- checked against `tl.range.__init__`:
`(arg1, arg2, step, num_stages, loop_unroll_factor, disallow_acc_multi_buffer,
flatten, warp_specialize, disable_licm)`. `tl.assume(cond)` exists too.

**`loop_unroll_factor=2` on the dk/dv m-loop is dead on the free gate.** Census
of the compiled `.amdgcn`, against the incumbent:

| | vgpr | spill | scratch B/lane | wmma | `s_set_vgpr_msb` | `scratch_load` | instrs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent | 1024 | **266** | 1064 | 448 | 1666 | 265 | 9547 |
| `dkdv_unroll2` | 1024 | **3678** | **4212** | 960 | 3297 | 1590 | 18985 |

13.8x the spills and 4x the scratch per lane. The premise was that two copies of
the body would let LLVM CSE the loop-invariant spill reloads and halve the 77
`scratch_load`/iteration; the opposite happened, because unrolling doubles the
number of simultaneously-live temporaries on a kernel that has zero register
slack. This is `r2.i3.g10`'s DISCARD half firing exactly as intended -- the one
half of that gate round 3 left standing. It still gets a timing slot in the
sweep, purely to put a point on the DISCARD rule at the largest spill count this
job has produced at a legal tile.

## 3. Free reading: the corpus re-prices `s_set_vgpr_msb`, and round 3 under-priced it

`backends/hipkittens/attention/recipes/gqa_d128.md` §6 mechanism 1 is the
closest thing in the corpus to this kernel: **GQA attention backward, head dim
128, 4 warps, one wave per SIMD**, two builds that differ in nothing but who
assigns the registers.

| N=4096, gfx950 | pinned (`ducks::art`) | compiler-allocated |
| --- | ---: | ---: |
| TFLOP/s | **1008.6** | 578.3 (**1.744x**) |
| waves/SIMD, VGPR/AGPR | 1, 256/256 | 1, 256/256 -- *identical* |
| MFMA instructions | 1 073 741 824 | 1 073 741 824 -- *identical* |
| `v_accvgpr_*` in the ISA | **128** | **1 638** |
| VALU per wave | 375 385 | 555 480 (+48%) |
| `SQ_WAIT_ANY` per wave | 145 049 | 1 243 033 (**8.6x**) |

The mechanism named there: *"HIPCC will not use an AGPR as an MFMA input, so
every operand that lives above register 255 has to be copied down before each
matrix instruction, and an attention backward has more live operand tiles than
256 VGPRs hold."* **1.72x for the cost of crossing the 256-register boundary, on
this operator, at this head dim, at one wave per SIMD.**

gfx1250 has no AGPRs, so the `v_accvgpr` form does not transfer -- but the
*boundary* does, in a different dress. gfx1250's register file is flat 1024 and
**addressed 256 at a time**; an instruction naming a register above `v255`
requires `s_set_vgpr_msb` first. `facts.md` already counts **363 of them per
dk/dv iteration, 26% of the body**, and `vgpr_count` is 1024 of 1024. That is
structurally the same tax as the 1 638 `v_accvgpr` moves, measured at 1.72x on
the same operator.

**This matters because round 3 priced that term at 6-14% and routed away from
it.** Its argument was: gfx1250 is hard power-capped, so under a cap
instructions price at 20-45% of face value, so 26% of the body is 6-14% of the
time, so chase the 77 `scratch_load` (bytes) and not the `msb` (cycles).

**Every step of that is sound except the premise, and the premise was never
measured on this kernel.** The 2497-2502 W / 1699-1703 MHz reading comes from
`bf16_gfx1250_ladder.md`, which is a **GEMM** -- a kernel with a far higher FLOP
density than an attention backward that spends 26% of its loop on a SALU mode
switch and 6% on `v_nop`. And the corpus's own counter-example is in the table
above: the compiler-allocated arm of the *same* gqa_d128 pair is **1080 W of a
1400 W cap at 2399 MHz of a 2400 MHz ceiling -- not at the wall at all -- while
being the slowest kernel in the card**, precisely because it is waiting on
register shuttling rather than doing work. A register-pressure-bound attention
backward is the archetype of a kernel that is *not* power-limited.

So round 3's pricing is a transfer from a GEMM to a kernel with the opposite
instruction mix, and **the measurement that settles it is free and has never
been taken on this job: sample power and clock on GPU[1] while
`bwd_kernel_causal` runs.** `profiling/2-power-wall-analysis.md`'s test is
power-at-limit AND clock-below-`MAX_CLK` (the card also warns `THROTTLE_STATUS`
reads `N/A` on this part, so it is not the test). Taken this round -- s5.

## 4. Free reading: round 3's "the in-thread-transpose flag is inert on gfx1250" is FALSE

Round 3's `opt.md` s10 records, unqualified, that `impl.py` sets
`TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1` around the backward launch but that
`is_in_thread_transpose_enabled` "returns true only for `gfx942`", so the
context manager is **inert on gfx1250**. Taken at face value that says half of
the only accepted round this job has had (`r1.i2.g2`, -5.8% alone, part of a
-13.0% merge) is doing nothing.

It is a misreading of the precedence in one line.
`triton/backends/amd/compiler.py:29`:

```python
def is_in_thread_transpose_enabled(arch):
    return (arch == "gfx942") if knobs.amd.use_in_thread_transpose is None else knobs.amd.use_in_thread_transpose
```

`arch == "gfx942"` is the **default taken only when the env var is unset**. When
`TRITON_HIP_USE_IN_THREAD_TRANSPOSE` is set, its boolean value is returned for
**every** arch, and `compiler.py:268` then runs `add_in_thread_transpose`. So
`impl.py` does enable the pass on gfx1250, round 1 measured what it said it
measured, and the shipped context manager is load-bearing.

Two consequences, both worth more than the correction itself:

- **Do not delete the context manager.** A later round reading round 3's line
  would have removed it as dead code and given back 5.8%.
- **`r1.i3.g3` (`schedule_hint="attention"`) stays excluded.** Its exclusion
  rests on "8.941 with it on top of g2 against 8.696 without", which is only
  meaningful if g2 is a real transform. It is, so the exclusion stands and this
  round does not revisit it. (Had g2 been inert, that comparison would have been
  two identical builds 2.8% apart -- which would have said the floor was 2.8%,
  not that g3 loses. It is not inert, so it does not.)

Cost of this reading: one `grep` in the installed Triton. No GPU, no compile.

## 5. The surface this round takes: `tl.range` loop controls and `tl.assume`

Rounds 1-3 measured `_DEFAULT_ONEKERNEL_CONFIG` in both directions on every
axis, all losing, and `dead_ends.md` carries the closed-surface table. What
those rounds never touched is the *other* Triton-level surface, the one that is
written in the kernel source rather than passed at launch:

- `tl.range(n, loop_unroll_factor=, disable_licm=, num_stages=, flatten=)`
- `tl.assume(cond)`

Both exist in this Triton (3.6.0); `grep -rn "def assume\|loop_unroll_factor" `
in the installed package confirms them. Neither appears anywhere in
`fused_mha_bwd_kernel.py` -- both inner loops are plain
`for blk_idx in range(num_steps):`. Four rounds of this job have never put a
number on either. They are one-line source edits, which is the cheapest
structural change available, and they act on the two things this kernel is
actually made of: the register allocator's problem, and the addressing path.

Why each, as a mechanism and not a knob:

**`disable_licm=True` on the dk/dv loop.** LICM's trade is *compute once, keep
live* -- it converts repeated arithmetic into an extended live range. That trade
is unconditionally good on a kernel with register slack and unconditionally bad
on one without: at `vgpr_count 1024/1024` with 266 spills, a value LICM hoists
out of a 264-trip loop does not get a register, it gets a **scratch slot**, and
every trip pays a `scratch_load` to read back what one VALU op would have
recomputed. `facts.md` already counts **77 `scratch_load` per dk/dv iteration**
in exactly that loop, and the dq pass, which the census shows has *zero* scratch
traffic, is the control. Disabling LICM asks the compiler to rematerialize
instead of spill. This is the only lever found so far that attacks the spill
count from the *cost* side rather than by making the tile smaller.

**`tl.assume` on the strides and program ids.** Round 3's single largest surprise
(s2) was that this kernel is dominated by its addressing, not by its math:
`AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1` moves it **+82%** and `AMDGCN_USE_BUFFER_OPS=0`
**+75%**, both of which change only how addresses are formed, and the first of
them does so while producing **187 spills, the lowest count this job has ever
seen**. Those two points say the address path on this kernel has a swing of
nearly 2x and that the compiler's choices there are not obviously optimal.
`tl.assume(s >= 0)` is the supported way to hand the backend the one fact it
cannot prove and that decides that path: non-negativity, which is what lets it
keep 32-bit offset arithmetic and use buffer addressing instead of widening.
(`USE_INT64_STRIDES` is already `False` at `attention_fused_bwd_impl.py:82`, so
the crude version of this idea -- "narrow the strides" -- is already done and
buys nothing; the assumption is the part that is missing.) 34 assumptions on
every stride plus the three program ids, x2 for the causal and non-causal
kernels.

**Expectation, recorded before the numbers.** `disable_licm` near-neutral: most
of this loop's invariants are already hoisted *at source level* outside the
`for`, so Triton's LICM may have little left to bite on, and the arm's value is
as much in bounding the term as in winning. `tl.assume` is the one with real
upside given the ±75-82% addressing swing, and is the arm I expect to carry the
round. `loop_unroll_factor=2` I expect to lose badly and am measuring only to
price the DISCARD rule at the largest spill count this job has produced at a
legal tile -- see s2, where it was already killed for free.

## 6. The sweep: seven points, one process each, palindromic base slots

`scratch/sweep1.sh`, launched detached through `op/runjob.sh` key `r4_sweep1` at
06:26:57, finished `rc=0` at 07:11. Each point is its own process with its own
`env -i` and its own fresh `TRITON_CACHE_DIR` (round 3 s0: several of the
interesting knobs are not in Triton's cache key, so one process per point is the
only honest form), each variant is an independent copy of `op/current` with one
asserted source edit (`scratch/mkvariant.py`), and `base` is measured first and
last. Per-point logs `scratch/pt_<label>.log`, parsed into `1-opt/raw/`.

| label | ms | vs base.pre | vgpr | spill | sgpr-spill | scratch B/lane | `scratch_load` | `s_set_vgpr_msb` | instrs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **base.pre** | **8.6860** | -- | 1024 | 266 | 67 | 1064 | 265 | 1666 | 9547 |
| assume | 8.6912 | **+0.06%** | 1024 | 266 | 67 | 1064 | 265 | 1666 | 9547 |
| dkdv_nolicm | 10.2033 | **+17.5%** | 1024 | 276 | 6 | 944 | 156 | 1748 | 9869 |
| assume_nolicm | 10.2108 | +17.6% | 1024 | 276 | 6 | 944 | 156 | 1748 | 9869 |
| dkdv_unroll2 | 25.6807 | +195.7% | 1024 | 3678 | 150 | 4212 | 1590 | 3297 | 18985 |
| unroll2_nolicm | 29.7783 | +242.9% | 1024 | 2913 | 121 | 4048 | 1506 | 3435 | 18861 |
| **base.post** | **8.6928** | +0.08% | 1024 | 266 | 67 | 1064 | 265 | 1666 | 9547 |

**Session floor: 0.078%** (base.pre 8.6860 against base.post 8.6928, two
identical builds 45 minutes apart). The two slots agree, so the session stands,
and it stands despite the three foreign processes in s1 -- which is itself the
useful reading on them. The `absum` checksums are bit-identical across all seven
points, so every arm computed the same gradients.

Both arms lose or are inert. Reported as measured; nothing here is shipped.

### 6a. Arm A -- `disable_licm=True` on the dk/dv loop: LOSES by 17.5%, and takes a premise with it

10.2033 ms against 8.6860, 224x the floor. Not marginal, not a retune away.

The census is the interesting part, because **it moved every number the round-1
and round-2 arguments said to move, in the direction they said to move it, and
got slower anyway**:

- `private_segment_fixed_size` **1064 -> 944 B/lane**, the first reduction in
  per-lane scratch this job has ever produced at a legal tile
- static `scratch_load` **265 -> 156**, a 41% cut, in the pass that owns 100% of
  this kernel's scratch traffic
- `sgpr_spill_count` **67 -> 6**

and it is **17.5% slower**. So the mechanism did exactly what was asked --
LLVM rematerialized instead of spilling -- and the recompute cost more than the
spill it replaced. That is a real reading about this kernel and not about this
knob: at 1 wave/SIMD the spill reload is a `scratch_load` whose latency the
scheduler has 264 iterations of independent work to hide, while the
rematerialized address arithmetic is VALU work on the critical path that nothing
hides. **This kernel would rather pay bytes than cycles.**

Three things follow, and they matter more than the arm:

1. **`r1.i7.g7`'s pricing argument is now contradicted by a direct measurement
   on this kernel.** That item's case is "removing scratch traffic pays, because
   at 1 wave/SIMD there is no second wave to cover an exposed scratch reload, so
   each one is latency on the critical path." This arm removed 41% of the static
   scratch loads and 11% of the scratch footprint and lost 17.5%. The reloads
   are evidently *not* on the critical path in the way the item assumes. The
   item is not dead -- LDS staging removes the traffic without paying recompute,
   which is a different trade -- but its headline justification can no longer be
   asserted, and it is the third premise of that item to fall (round 2 killed
   "LDS is at 20%", round 3 restored it, and now the cost model behind it is the
   part that needs a number).
2. **A fourth counter-example to spill-count-as-proxy.** `r2.i3.g10`'s MEASURE
   half was already broken by the tile counter-example (round 2) and by
   `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE` at a fixed tile (round 3, 187 spills and
   +82%). Here the same source, same tile, same addressing mode goes *down* in
   scratch bytes and *up* in time; and `unroll2_nolicm` has **fewer** spills
   than `dkdv_unroll2` (2913 vs 3678) and is **16% slower** than it. The DISCARD
   half survives -- everything that got much worse on spills also got much worse
   on time -- but no round should again justify a candidate by a spill count.
3. **`s_set_vgpr_msb` went UP, 1666 -> 1748, while scratch went down.** The two
   overheads trade against each other: fewer spill slots means more values
   resident above register 255. Any future candidate that attacks one of them
   has to report the other.

### 6b. Arm B -- `tl.assume` on every stride and program id: INERT, and the family is closed

8.6912 ms against 8.6860 is +0.06%, inside the 0.078% floor. Stacked on arm A
(`assume_nolicm`, 10.2108 vs 10.2033) it is +0.07%, also inside the floor. Two
independent readings of "no effect."

The census is identical on every field -- `vgpr`, `spill`, `sgpr_spill`,
`sgpr`, scratch bytes, `wmma` 448, `s_set_vgpr_msb` 1666, `scratch_load` 265,
`ds` 656, and total instructions 9547, all exactly equal to base. The compiled
`.amdgcn` is *not* byte-identical (3004 of 11680 lines differ, register naming
and ordering churn) but the instruction mix differs by exactly **one**
`s_wait_loadcnt`, 158 -> 157. The edit reached the compiler and changed nothing
that matters.

**The mechanism for why, which is the part worth keeping:** the assumptions are
present in the Triton IR and absent from the LLVM IR.

```
tc_assume_/.../bwd_kernel_causal.ttir   : 301 assume ops
tc_assume_/.../bwd_kernel_causal.llir   :   0 llvm.assume
tc_base_pre_/.../bwd_kernel_causal.ttir :   0 / llir 0
```

Triton 3.6.0 consumes `tl.assume` in its own passes and emits no `llvm.assume`,
so the AMD backend's address-formation decisions -- the ones round 3 measured a
**±75-82%** swing on -- never see the facts. This is not "the assumptions were
already known"; it is that **there is no path from `tl.assume` to the backend
that decides addressing on this build**. So the whole "hand the backend a fact
about the address space" family is closed on this Triton, not just the
non-negativity form I tried, and it is closed by an IR reading rather than by a
time.

### 6c. The two throwaways: `loop_unroll_factor=2` priced

25.6807 ms (+196%) at 3678 spills, and with LICM off 29.7783 ms (+243%) at 2913
spills. s2 killed this arm for free on the static gate before it was measured;
these two points are what the gate was worth -- 21 minutes of GPU time it would
have cost to learn the same thing, and a confirmation that the DISCARD rule
(spills >> 266 -> do not measure) is still sound in the direction it is used.

## 7. The power/clock trace on `bwd_kernel_causal` -- taken, and it overturns round 3's discount

**What I expect before the numbers, written down first.** Round 3's discount --
"instruction-count savings price at 20-45%, bytes price in full" -- rests on a
power reading of **sustained bf16 GEMM** (2497-2502 W of 2500, clock 29% under
`MAX_CLK`), never on this kernel. `profiling/2-power-wall-analysis.md`'s own
regime test is *power at the limit* **and** *clock below `MAX_CLK`*. I expect
this kernel to fail that test: it is one wave per SIMD, 1024/1024 VGPRs, 266
spills, 448 WMMA in 9547 instructions, and it reaches 631 TF/s against the
2236-2667 TF/s a GEMM gets on this part. A kernel doing a quarter of the
achievable math per unit time on a quarter of the occupancy should be drawing
well under the cap and running at or near 2400 MHz. If so, the 20-45% discount
does not apply here and every instruction removed prices at full value -- which
is the difference between the `s_set_vgpr_msb` term being worth 6-14% and being
worth the 1.72x the corpus measures on the same operator (s3).

**What came back.** The sampler had to be run twice. The first trace
(`raw/power_trace1.txt`) sampled from the moment of launch and spent its whole
window in Python import and Triton compile -- GPU[1] reads `use 0%` for all 60
samples -- so it measured nothing and is kept only because it is the idle
reference. The second (`raw/power_trace2.txt`, `scratch/powertrace2.sh`) blocks
until `powerload.py` prints `LOADSTART` and then samples all four GPUs every
2 s for 60 s of a continuous backward (`LOADDONE iters=10340 mean_ms=8.7205`,
which agrees with the sweep's 8.686-8.693 and confirms the load is the kernel
under study, not an artefact).

**Device mapping, confirmed rather than assumed.** `HIP_VISIBLE_DEVICES=1`
inside the container is `rocm-smi` `GPU[1]`: it reads `GPU use 0%` for every
idle sample and `100%` for all 30 load samples, while `GPU[2]`/`GPU[3]` stay at
0% throughout. `GPU[0]` reads 100% for the entire trace -- another tenant's job,
on another card. **So this job's measurements have been landing on an otherwise
idle device**, and the three foreign processes from s1 are burning CPU, not
GPU[1]. That closes the session hazard: it is real but it is not on our card.

**The power sensor does not read on the device under test while it is loaded.**
`rocm-smi -P` prints a `Current Socket Graphics Package Power` row for
`GPU[0]`, `GPU[2]` and `GPU[3]` in all 30 load samples and **omits the `GPU[1]`
row entirely** -- the same GPU whose row is present, at 1120-1122 W, in every
idle sample. This is the fourth instrument on this part to fail specifically on
the busy device, after counters (13/51, no byte counter), `--stats
--kernel-trace` (exit 0, zero dispatches) and PC sampling/ATT (`dead_ends.md`).
Recorded there; **do not spend a row re-attempting the power number.**

**The clock does read, and it is the discriminating half of the test.**

| state | GPU[1] sclk | vs `MAX_CLK` 2400 |
| --- | ---: | ---: |
| idle (3 samples, both traces) | 2400 MHz | 0% |
| under `bwd_kernel_causal`, 30 samples over 60 s | **2159-2167, median 2161** | **-10.0%** |
| round 3's sustained bf16 GEMM, same part | ~1700 | -29% |

So this kernel **is** clock-droop-limited, but at **10%** where the GEMM round 3
borrowed its coefficient from droops **29%**. Under a fixed cap, droop is
monotone in draw, so 10% against 29% says this kernel sits substantially
further from the wall than the card that the 20-45% discount was derived on.
**The discount was measured at 29% droop and applied here at 10% droop, and
that step is unsupported.** I cannot replace it with a number, because the
power row is unreadable; what I can say is that the one quantity round 3's
pricing depends on is 3x smaller on this kernel than on the kernel it was
calibrated against.

**And there is now a direct experiment, from this round's own arm A.**
`disable_licm` deleted 41% of the static `scratch_load`s and 11% of the scratch
footprint -- the most energy-expensive class of work in the kernel -- and
replaced them with VALU recompute, the cheapest. Under a power-capped regime
where `time = energy / P_limit`, that trade is close to a free win. It cost
**+17.5%**. A kernel that gets slower when you swap memory traffic for
arithmetic is not spending its time on energy; it is spending it on dependency
latency. **That, not the power row, is the finding**: on `bwd_kernel_causal` the
correct first-order price of an instruction is close to full, and round 3's
6-14% valuation of the `s_set_vgpr_msb` term -- the one thing that made the
corpus's measured **1.72x** for the same boundary on the same operator look
ignorable (s3) -- rests on a coefficient that does not transfer.

## 8. Validation, on the working copy, on the verified-clean device

`rounds/004/op` is **byte-identical to `op/current`** (`diff -rq`, no differences)
because neither arm won and there was nothing to ship. Run through `runjob.sh`
key `r4_val` on GPU[1], which s7 confirmed is the physical card
`HIP_VISIBLE_DEVICES=1` selects and which read 0% use before the run. Raw in
`1-opt/raw/validation.txt`.

```
shape                      arm           fwd ms    bwd ms  bwd TF/s  bwd GB/s  spread%
b4_s8192_hq32_hkv8_d128    candidate     4.9698    8.6972    632.19     154.3     0.47
b4_s8192_hq32_hkv8_d128    beat          3.7205   11.6684    471.21     115.0     3.19
b4_s8192_hq32_hkv8_d128    baseline      4.9636   10.0024    549.69     134.2     0.87

vs beat  1.342x   required 1.50x   vs baseline 1.150x   VERDICT: FAIL
VALIDATION FAILED: speed -- short by 10.6% of the target
```

Correctness passes its own gate on the spec shape and every edge shape
(SQNR 52.0-54.1 dB against `op/eager/`, gate 50 dB). `sq_gt_skv` reports
`dq -15.95 dB` and the file marks it **diagnostic, not gated**; it is inherited,
not introduced here -- `rounds/004/op` is bit-identical to `op/current` and the
seven sweep points returned bit-identical gradient checksums.

8.6972 against the sweep's 8.6860/8.6928 is +0.1%, consistent at this floor.
1.342x against round 3's 1.343x is the same number: **nothing shipped, so
nothing moved.**

## 9. Corpus files opened this round

- `knowledge/INDEX.md`
- `knowledge/backends/hipkittens/attention/recipes/gqa_d128.md` (s6, mechanisms 1-4 -- the 1.72x above-register-255 measurement that s3 and route row 2 are built on)
- `knowledge/backends/aiter/attention/recipes/fmha_v3_bwd_hd128_bf16.md` (s6 b1, b7, s7)
- `knowledge/optimization/techniques/3-pipelining-and-scheduling.md` (header and veto -- the basis for `r4.i1.g16`)
- `knowledge/profiling/2-power-wall-analysis.md` (lines 1-170 -- the regime test s7 applies, and its own table putting register-file pressure in the "paid in full" row)

## 10. What this round leaves for round 5

Written into `findings/route.md`, `findings/pool.md` (`r4.i1.g16`, `r4.i2.g17`)
and `findings/dead_ends.md`. The one-line version: **the discount is gone.**
Round 3 priced instruction-count wins at 20-45% using a coefficient measured on
a GEMM, and that is what made the corpus's 1.72x for the above-register-255
boundary -- measured on this exact operator at this exact head dim at this exact
occupancy -- look like 6-14% and therefore not worth a round. This round shows
the coefficient does not transfer (10% clock droop against the GEMM's 29%) and,
more decisively, that the kernel gets *slower* when memory traffic is traded for
arithmetic, which is not how an energy-limited kernel behaves. So the 1666
`s_set_vgpr_msb` -- 26% of the dominant loop body -- are worth roughly 26%, and
the round's whole route is now aimed at the one lever that reduces them without
shrinking the tile, losing the causal load balance, or leaving Triton:
**halve the resident accumulator set by running the m-loop once for `dv` and
once for `dk`** (`r4.i2.g17`). It is falsifiable on a free compile.

---

# Build and measure -- the route's rows, in order

Compile cache cleared before the first build (`rm -rf scratch/tc_*`, and every
point below runs under a fresh `TRITON_CACHE_DIR` of its own, so no point can be
served a binary another point compiled).

## 11. Row 1 (must) -- the free census gate, applied to all three builds

Every candidate compiled and read before any of them was given a timing slot,
reporting `vgpr_count`, `vgpr_spill_count` **and** `s_set_vgpr_msb` as the row
requires. `absum` is the gradient checksum against base.

| build | vgpr | spill | scratch B/lane | `scratch_load` | `s_set_vgpr_msb` | wmma | instrs | absum | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| base (`op/current`) | 1024 | 266 | 1064 | 265 | 1666 | 448 | 9547 | ref | -- |
| **r4.i2.g17** two-phase m-loop | **973** | **0** | **0** | **0** | 1728 | 512 | 9889 | identical | **MEASURE** |
| r4.i1.g16 `num_stages=2` | 1024 | **2001** | 2468 | 698 | 2338 | 704 | 14694 | identical | **DISCARD** |
| r4.i1.g16 `flatten=True` | 1024 | 266 | 1064 | 265 | 1666 | 448 | 9547 | identical | measure (inert) |

The gate did its job twice in one session and cost no GPU time beyond the
compiles:

- **`tl.range(num_stages=2)` on the dk/dv loop is DISCARDed**, exactly as route
  row 3 predicted in writing: 266 -> **2001** spills, 1064 -> 2468 B/lane,
  `s_set_vgpr_msb` 1666 -> 2338. Same family as `loop_unroll_factor=2`
  (266 -> 3678). A pipeline buffer at 1024/1024 VGPRs is bought out of scratch.
  Not timed; the DISCARD half of `r2.i3.g10` is the surviving half and this is
  what it is for.
- **`flatten=True` is inert at code-gen**: the census is equal to base on every
  single field, which is what row 3 said would happen ("changes no live set at
  all"). Timed anyway, because "identical census" is not "identical time" and
  the cost is one slot.

## 12. Row 2 -- r4.i2.g17, the two-phase dk/dv pass

**What was built.** `_bwd_dkdv_inner` gains one `PHASE: tl.constexpr = 2`
parameter (default 2 = both, so the noncausal kernel is untouched and keeps
working). Inside, the dV accumulation is guarded by `if PHASE != 1:` and the
dK half -- the `Delta` load, `dpT`, `dsT` and the `dk` dot -- by `if PHASE != 0:`.
In `bwd_kernel_causal` the `hqid` loop is wrapped in `tl.static_range(2)` with a
**single** accumulator:

```python
for phase in tl.static_range(2):
    acc = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    for hqid in range(...):
        acc_dk, dk_pe, acc_dv = _bwd_dkdv_inner(acc, dk_pe, acc, ..., PHASE=phase)
        if phase == 0: acc = acc_dv
        else:          acc = acc_dk
    if phase == 0: tl.store(DV + ..., acc)     # dv is STORED here, before
    else:          tl.store(DK + ..., acc)     # dk is ever allocated
```

The store sits **inside** the phase loop, which is the whole point: `dv` is
written out and dead before `dk` exists, so the two 256-VGPR accumulators are
never live at the same time. Tile, `BLOCK_N1`, the fusion, the causal load
balance and the launch config are all untouched. Patch script:
`scratch/mkg17.py`, applied to a fresh copy of `op/current`.

**The census, and an honest scoring of my own prediction.** Route row 2 wrote
its own falsifier: *"if the split does not move `s_set_vgpr_msb` below 1666, the
premise is wrong and it dies for a compile."* **`s_set_vgpr_msb` went UP,
1666 -> 1728. The stated premise is falsified.** The kernel is 973 VGPRs, still
far above 255, so the above-register-boundary tax is not what this change
removes, and the 1.72x corpus argument that motivated the row is not what is
being tested.

What it removed instead is larger and was not predicted:

- `vgpr_spill_count` **266 -> 0**
- `private_segment_fixed_size` **1064 -> 0 B/lane**
- static `scratch_load` **265 -> 0**, `scratch_store` **128 -> 0**

**This is the first build of `bwd_kernel_causal` in four rounds with zero spills
and zero scratch**, and it lands at `vgpr_count` 973 -- which is exactly where
round 3 measured the *dq pass compiled alone* (971 VGPRs, zero spills). The
fused kernel's register demand is now set by the dq pass; the dk/dv pass has
stopped being the binding constraint. It is paid for with `wmma` 448 -> 512
(+14%, the recomputed `pT` in phase 1) and 342 more instructions.

Correctness signal before timing: the gradient checksums are bit-identical to
base on all three tensors.

### The first attempt, and the mechanism it cost

Attempt 1 wrote the phase loop as `for phase in (0, 1):` -- a plain Python tuple,
on the reasoning that the tracer would unroll it in Python and leave `phase` a
real `int`. It does not compile:

```
CompilationError: at 277:8:  for phase in (0, 1):
AttributeError("'Tuple' object has no attribute 'func'")
```

**Mechanism, for `dead_ends.md`:** Triton 3.6.0's AST tracer requires the
iterable of a `for` to be a **call node** -- it reaches straight for `.func` to
decide whether it is `range`, `tl.range` or `tl.static_range`. Any other
expression (tuple, list, `zip`, a name bound to a sequence) raises this
`AttributeError` rather than a diagnostic. There is no Python-level `for` in a
`@triton.jit` body; compile-time repetition must go through `tl.static_range`.
Fixed by `tl.static_range(2)` and by replacing the constexpr ternary
`acc = acc_dv if phase == 0 else acc_dk` with a plain `if/else` on the same
constexpr. One failure, one root cause, fixed rather than worked around: the
hypothesis is unchanged.


## 13. Row 2 measured: the first zero-spill build of this kernel is 7.8% slower

Five points, `r4_sweep4`, one process per point, a fresh `TRITON_CACHE_DIR` for
each, base in palindromic slots, GPU[1] verified idle before and after
(`rocm-smi --showpids` empty of live PIDs, `--showuse` GPU[1] 0%). Every point
returned `absum` gradient checksums **bit-identical** to base, so nothing below
is a correctness difference wearing a speed costume.

| point | build | bwd ms | vs base | vgpr | spill | scratch B/lane | scratch_load | s_set_vgpr_msb | wmma | instrs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bp1 | `op/current` copy | **8.6873** | -- | 1024 | 266 | 1064 | 265 | 1666 | 448 | 9547 |
| g17 | two-phase dk/dv | **9.3689** | **+7.84%** | 973 | **0** | **0** | **0** | 1728 | 512 | 9889 |
| g16flat | `tl.range(flatten=True)` | 8.7072 | +0.23% | 1024 | 266 | 1064 | 265 | 1666 | 448 | 9547 |
| g17b | g17, second slot | **9.3816** | **+7.98%** | -- | -- | -- | -- | -- | -- | -- |
| bp2 | `op/current` copy | **8.6867** | -- | -- | -- | -- | -- | -- | -- | -- |

Base-to-base spread **0.007%** (8.6873 / 8.6867); g17-to-g17 repeatability
**0.14%** (9.3689 / 9.3816). Both are far below the effect, so the effect is
real. The 0.007% is luck rather than the standing floor -- round 3 measured
0.9% for one-process-per-point on this harness and that is still the number to
budget against.

`g16stages` (`tl.range(num_steps, num_stages=2)`) was never timed: the row-1
census killed it at **2001 spills** and **2468 B/lane**, which is exactly the
job that gate exists to do.

### What it means

This is the finding of the round, and it is worth more than the arm that
produced it.

Four rounds of this campaign have used `vgpr_spill_count` as the proxy for
where the time goes, because it is free to read and because the corpus prices
the >255-register boundary at 1.72x on this exact operator. g17 drove that
proxy to its **ideal value** -- 266 spills to **zero**, 1064 B/lane of private
scratch to **zero**, 265 static `scratch_load` to **zero**, `vgpr_count` down
to 973, landing almost exactly where round 3 measured the dq pass running
*alone* (971, zero spills). If the proxy meant what it has been taken to mean,
this build should have been the round's win by a wide margin.

It is **7.8% slower**, on every shape, by two independent measurements.

The price it paid is visible in the same census: `wmma` **448 -> 512** (+14%,
the `pT` that phase 1 must recompute because phase 0 no longer has it live)
plus a second full streaming pass over Q, dO and LSE for the same 264 loop
trips. So the trade the arm actually made was *266 spills and 1064 B/lane of
scratch traffic* against *64 extra WMMA and one extra pass over the Q-side
operands* -- and the spills were the **cheaper** side. At 1 wave/SIMD, with
`private_segment_fixed_size` resident and hot, a spill/reload pair is a short
dependent latency the 264-trip loop can absorb; a second streaming pass is
not absorbable by anything.

And the premise's own falsifier fired before the timer ever started.
Row 2 was written with "if the split does not move `s_set_vgpr_msb` below
1666, the premise is wrong and it dies for a compile". It went **UP**, 1666 ->
1728. Halving accumulator residency did not reduce the count of
above-register-255 window switches, because the switches are driven by where
operands sit in the flat 1024-VGPR file, not by how many of them are live at
once -- and phase 1's recomputed `pT` adds windows of its own. The census gate
and the stopwatch agreed. I timed it anyway, and the record should say plainly
that overriding the gate bought one number and one round.

The general form, for `facts.md`: **on `bwd_kernel_causal` at tile 256, spill
count is not a cost model.** Five independent counter-examples now, in
increasing strength: `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1` (187 spills, +82%),
`AMDGCN_USE_BUFFER_OPS=0` (250 spills, +74%), `unroll2_nolicm` vs
`dkdv_unroll2` (2913 vs 3678 spills, 16% slower), `disable_licm` (-41% of
static `scratch_load`, +17.5%), and now g17 (**zero** spills, **zero**
scratch, **+7.8%**). The last one is qualitatively different from the first
four because the proxy was not merely moved in the wrong direction -- it was
driven to the best value it can take, and the kernel got slower. A candidate
in round 5 that justifies itself by the spills it will remove is arguing
against five measurements.

## 14. Validation, unit tests, and the route's outcomes

Working copy `rounds/004/op` contains the g17 kernel. Per the no-revert rule
the change that was measured is the change that ships, including where it
lost; `rounds/004/op` differs from `op/current` in exactly one file,
`vendor/primus_turbo/triton/attention/fused_mha_bwd_kernel.py`.

**Unit tests: PASS.** `python op/ut/correctness.py rounds/004/op` --
4 gated shapes, all four tensors >= 50 dB. (`op/ut/` holds no `test_*.py`; it
is the module `validation.py` imports, and the supported runnable form is the
one above. A `pytest op/ut` invocation collects nothing and exits 0, which is
not a pass and is not recorded as one.)

**Precision: PASS.** `python op/validation.py rounds/004/op`:

| shape | out | dq | dk | dv | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| b4_s8192_hq32_hkv8_d128 | 53.67 | 52.24 | 52.31 | 52.71 | pass |
| sq_lt_skv | 53.01 | 52.50 | 52.58 | 52.57 | pass |
| min_seqlen | 54.09 | 52.04 | 52.08 | 52.77 | pass |
| ragged_tail | 53.99 | 52.09 | 52.15 | 52.75 | pass |
| sq_gt_skv | 53.97 | **-15.95** | 52.10 | 52.72 | fail (diagnostic, not gated) |

Identical to the incumbent's numbers to the last digit on every tensor,
including the inherited `sq_gt_skv` dq diagnostic. A restructuring this large
reproducing the reference bit-for-comparable-bit is itself evidence the split
is correct rather than accidentally cheap.

**Speed: FAIL.** candidate **9.3782 ms / 586.28 TF/s**, beat 11.6458 / 472.12,
baseline 9.9975 / 549.96 -- **1.242x against a 1.50x bar**, short by 17.2% of
target. The incumbent stands at 1.342x, so this round moves the job backwards
and is not accepted. `exit_code 1`.

**All shapes, candidate against the incumbent re-measured back to back in the
same session** (`op/benchmark.py --shape all`, unprofiled, GPU[1] idle,
`rounds/004/op` then `rounds/001/op`):

| shape | cand bwd ms | incumbent bwd ms | cand TF/s | incumbent TF/s | vs champion |
| --- | ---: | ---: | ---: | ---: | ---: |
| b4_s8192_hq32_hkv8_d128 | 9.3802 | 8.6782 | 586.15 | 633.56 | **0.925** |
| sq_lt_skv | 0.7286 | 0.7257 | 353.82 | 355.20 | 0.996 |
| min_seqlen | 0.1944 | 0.1893 | 110.71 | 113.68 | 0.974 |
| ragged_tail | 0.3423 | 0.2690 | 239.57 | 304.88 | 0.786 |
| sq_gt_skv | 0.4312 | 0.3467 | 199.42 | 247.98 | 0.804 |

Slower on all five. Acceptance fails on the first condition (throughput does
not improve on the best ever) and on the second (three shapes are below 95% of
their own best). `score` = 586.15 / 706.0 = **0.830** against the incumbent's
0.897.

### Route outcomes

| # | id | outcome |
| --- | --- | --- |
| 1 | r4.m1 | **applied, and it paid.** Killed `num_stages=2` free (266 -> 2001 spills). Confirmed `flatten=True` census-identical. Fired correctly on g17 (`s_set_vgpr_msb` 1666 -> **1728**, the wrong way) and was overridden; the stopwatch then agreed with the gate at +7.8% |
| 2 | r4.i2.g17 | **delivered, and lost.** Built, correct, shipped in `rounds/004/op`. 9.3689 / 9.3816 ms against base 8.6873 / 8.6867 -- **+7.8%**; validation 1.242x against a 1.50x bar. First zero-spill, zero-scratch `bwd_kernel_causal` in four rounds and the strongest refutation of spill-count-as-proxy the campaign has |
| 3 | r4.i1.g16 | **half DISCARDed free, half inert.** `num_stages=2` died on the census. `flatten=True` 8.7072 ms, +0.23%, census byte-identical to base. The per-loop `tl.range` surface is now closed |
| 4 | r1.i7.g7 | **not opened,** and g17 has made its case worse: zero scratch traffic was reached and cost 7.8%. Rewrite or retire it before it takes another slot |

### What round 5 inherits from this section

Two arms, two rounds, one direction. Rounds 3 and 4 have now shipped nothing,
and everything they built was justified by register pressure. Round 5 should
not open a fifth register-pressure candidate on the same evidence. What is
left unmeasured is *why a second streaming pass over Q/dO costs 7.8% when the
loop has 264 trips to hide it in* -- that is a memory-side question about the
Q-side operands, and the campaign has never asked one.
