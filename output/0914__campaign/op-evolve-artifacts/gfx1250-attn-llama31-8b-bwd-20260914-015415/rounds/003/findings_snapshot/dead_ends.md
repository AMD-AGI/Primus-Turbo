# Dead ends

Mechanisms, not symptoms. An entry here should kill a candidate before it is built.

## Counter-based profiling on gfx1250 (found in round 1)

`rocprofv3 --stats --kernel-trace` runs to completion on this job's benchmark and
records **zero kernel dispatches**. Round 1 did not find out why and worked around it
with `torch.profiler`; that part is a symptom and is recorded as one.

The mechanism that IS established, and that kills a whole class of plans: gfx1250 has
**51 counters defined against gfx950's 630, of which only 13 read non-zero**; the
**entire WMMA FLOP counter family reads exactly 0** (marked emulated); and there is
**no byte counter at any level**. So there is no counter-derived roofline, no
memory-traffic analysis and no achieved-FLOP figure on this part, and any plan whose
first step is "read the counters" is dead before it starts. `rocprofv3` also **exits 0
and writes no CSV when given an unknown counter** -- a silent failure, which is
consistent with what round 1 hit. What is reported to work is stochastic PC sampling
and ATT; that is `r1.i6.g6`, still open.

## Sweeping `BLOCK_N1` / `BLOCK_M2` through the config (measured in round 1)

They are read from the config and then **overwritten** by `fused_backward_tile(seqlen_k)`
inside `dense_fused_backward` before the launch. A sweep over them therefore measures
one point repeatedly, reports no error, and looks like a flat surface. Round 1 lost a
sweep to this. `BLK_SLICE_FACTOR` is on the same config dict and is *not* overwritten,
which is the only reason it is still a live candidate.

## Adding waves to hide the spill (measured in round 1)

`num_warps=8` measures **22.5-23.6 ms against 9.996**, 2.36x slower. Mechanism: the
gfx1250 per-lane VGPR budget is `131072 / NUM_THREADS`, so 4 warps of wave32 gives
exactly the 1024 the static census reports and 8 warps gives 512 -- to a kernel already
spilling 326. `waves_per_eu: 2` is the same story at 50.6 ms. The general form is the
occupancy precondition `current_vgpr x new_waves <= 1024`: at 1024 current, no wave
count above 1 is legal. Latency hiding on this kernel has to come from inside the wave,
which is why the live direction is removing the spill rather than covering it.

## Deleting instructions from the hot loop, when the census is not checked (round 2)

The dominant argument of rounds 1-2 -- "this instruction is redundant, delete it and
the loop gets shorter" -- **is refuted on this kernel**. Round 2 spent both arms on it
and both lost by ~7% against a 0.03-0.06% session floor:

| arm | what it deleted | did it work at source level? | `vgpr_spill_count` | bwd ms |
| --- | --- | --- | --- | --- |
| `r2.i1.g8` fold `sm_scale` into the `exp2` constant | 32 `v_pk_mul_f32`/iter | **yes**, 448 -> 416 exactly | 266 -> **307** | 8.691 -> 9.347 |
| `r2.i2.g9` invariant offsets + scalar cursor | 41 `v_add`/iter | **no**, 115 -> **124** | 266 -> **325** | 8.691 -> 9.302 |

Mechanism: at `vgpr_count 1024` of a 1024-VGPR file there is **no slack**. Folding a
short-lived value into a longer expression (arm A) extends the live range of the value
it was consuming; replacing a coalesced loop-carried pointer tensor with
`ptr + scalar * stride` at each use (arm B) makes the allocator re-materialise the
full address tensor per load. Either way the allocator pays for it in spills, and on
this kernel spills ARE the time (`r2.i3.g10`).

A null arm -- a byte-identical copy of the incumbent's directory -- was measured in the
same session shape and read 8.686 ms against the incumbent's 8.691, so the losses are
the edits and not the harness.

**Kill rule:** a candidate of the form "remove instruction X from the loop body" must
show `vgpr_spill_count < 266` in the compiled `.amdgcn` before it is measured. That
check is free (`r2.i3.g10`). Without it the candidate is not worth GPU time.

## `BLK_SLICE_FACTOR > 1` (`r1.i5.g5`, measured round 2)

**8.8726 ms against 8.6720 ms default**, +2.3%, in one process against a 0.07% floor.
Mechanism: it splits only the MASKED diagonal block, which is 1/32 of the work at
`BLOCK_N1=256`/`seqlen 8192`, and pays for the split with an extra loop trip and its
prologue in the pass that is already spilling. The saving is bounded by the diagonal;
the cost is not.

## Decoupling `BLOCK_M1` from `BLOCK_N2` (measured round 2)

`_check_block_invariant` requires `BLOCK_M1 == BLOCK_N2`, and that rule is
**over-strict** -- `BLOCK_M1` steps the dk/dv pass and `BLOCK_N2` the dq pass, they are
used in independent passes and nothing couples them. So this is a real, previously
unavailable degree of freedom. **Every point in it loses**, one process, same floor:

| | bwd ms |
| --- | --- |
| default `M1=32, N2=32` | 8.6720 |
| `N2=16` | 9.4940 |
| `M1=16` | 11.2990 |
| `M1=16, N2=16` | 12.0620 |
| `M1=64` | 15.9580 |

Mechanism, and it is why the direction is exhausted rather than under-explored: the
spilled state is `dk`+`dv`+`k`+`v`, which is sized by `BLOCK_N1` and `HEAD_DIM` and is
**loop-invariant**. `BLOCK_M1` is the loop STEP. Shrinking it does not shrink the
residency at all, it just runs the same reload more times.

Also: **`BLOCK_M < 16` is illegal on this arch.** `BLOCK_M1=16, BLK_SLICE_FACTOR=2`
gives `MASK_BLOCK_M1=8` and the compiler emits `no matching matrix core intrinsic for
wmma version 3 with instruction shape [0, 0, 128]` at `tl.dot`, then produces a kernel
that runs at 39.45 ms and 81.9 dB. The wmma is 16x16x32 and there is no fragment below
it. Do not read the 81.9 dB as "correct but slow".

## Three vendored `primus_turbo` trees in one process (round 2 harness trap)

Segfault (rc 139) at the first `attention` call, with all three trees imported and no
Python traceback. `impl.py` makes the *opaque-type* registration idempotent and round 1
moved the custom op to `primus_turbo_opevolve::`, but that namespace is shared by every
copy, so the third `torch.library` fragment on the same `OperatorEntry` takes the
process down. `validation.py` never hits it because only ONE of its three arms carries
a vendored tree (`beat` is `flex_attention`).

**A multi-arm harness must run one arm per process.** Arms then share a *session*
rather than a process: run them back to back, palindromically, inside a single
`docker exec`. Round 2's base slots came back 0.03-0.06% apart that way, which is
tighter than round 1 achieved in-process.

## r2.i4.g11 -- splitting `HEAD_DIM` in the dk/dv pass (killed round 2, before building)

The bet was: run the K/V block over `HEAD_DIM` in two halves of 64, so the dk/dv
pass's 768 VGPRs of loop-invariant state (`dk` 256 + `dv` 256 + `k` 128 + `v` 128)
becomes 384, for about +25% wmma.

**The arithmetic is wrong, and reading `_bwd_dkdv_inner` is enough to see it.**
`HEAD_DIM` is the free axis of `dk` and `dv`, but it is the **contraction** axis of
`qkT = tl.dot(k, qT)` and `dpT = tl.dot(v, tl.trans(do))`. A half-`HEAD_DIM` pass
still needs all of `k` and all of `v` resident to form `qkT` and `dpT`, so the real
figure is **768 -> 512, not 384** -- and it costs **+50% wmma**, not +25%, because
both dots run twice, plus a second full pass over Q and dO.

**And that last cost is the one round 2 measured as fatal.** `BLOCK_N1=128` is a
strictly better version of the same trade -- it reaches 768 -> 384, with no recompute,
for the same doubling of Q/dO traffic -- and it loses by 20-33% (11.8-13.3 ms vs
8.691), *while compiling to zero spills*. Any candidate that pays 2x Q/dO traffic to
buy residency is dominated by a tile change that has already been timed and lost.

Abandoned at `look`, on that evidence, without a build. The general form: **on this
kernel, residency bought with Q/dO re-reads is not worth buying.** What is still open
is lowering residency *at constant traffic*, inside a single pass over the m-loop.

## r2.i1.g8 -- folding `sm_scale` into the `exp2` constant (measured round 2, -7.5%)

Delete the 32 `v_pk_mul_f32` that scale `qkT` each iteration by pre-multiplying
`sm_scale` into `RCP_LN2` at the `exp2`. Source-level it worked exactly as designed:
the census shows `v_pk_mul_f32` 448 -> 416, the precise 32 targeted, nothing else in
the histogram moved by more than noise.

**It measured 9.347 ms against 8.691, +7.5%, on a 0.03-0.06% floor.** Mechanism:
extending the live range of the scaled constant across the dot cost **41 more spills**
(266 -> 307, scratch 1064 -> 1228 B/lane). At 1024 VGPRs and one wave per SIMD the
allocator has no slack, so a deletion that lengthens any live range is paid for at a
worse exchange rate than the instructions it removes -- 32 VALU out, 41 spill slots in,
each reloaded every iteration.

## r2.i2.g9 -- loop-invariant offsets plus a scalar row cursor (measured round 2, -7.0%)

Replace the loop-carried pointer `iter_args` (`tensor<128x32xi32>` and
`tensor<32x128xi32>`, confirmed as the `scf.for` `iter_args` in the TTGIR) with
offset tensors built once from `tl.arange` outside the loop, plus a scalar `curr_m`
added at the load: `tl.load(qT_ptrs + curr_m * stride_qm, ...)`. Mirrored in
`_bwd_dq_inner`. The point was to carry two scalars instead of two full-width integer
tensors across the loop boundary.

**It measured 9.302 ms against 8.691, +7.0%.** Mechanism: the vectorised adds were not
removed, they were *moved* -- the census shows `v_add*` going **up**, 115 -> 124, not
down -- and the now loop-invariant offset tensors became long-lived values that the
allocator spilled, 266 -> 325 (scratch 1064 -> 1288 B/lane). Hoisting a value out of a
loop is only a win if there is a register to keep it in; here it converts a recomputed
tensor into a spilled one, which is strictly worse.

This is the shipped working copy of round 2, and it is what was rejected.

**Both arms together are the round's lesson**: on a kernel at 1024/1024 VGPRs, a
source-level instruction deletion is a *request* to the register allocator, and the
allocator's answer is the thing that gets timed. Run `census_gate.py` (`r2.i3.g10`)
first, every time.

## The launch/tile surface is CLOSED (completed round 3)

Rounds 1-3 have now measured every axis of `_DEFAULT_ONEKERNEL_CONFIG` in both
directions. `BLOCK_N1=BLOCK_M2=256`, `BLOCK_M1=BLOCK_N2=32`, `num_warps=4`,
`waves_per_eu=0`, `num_stages=1`, `BLK_SLICE_FACTOR=1` is a **strict local optimum**:
every neighbour on every axis is slower, several by more than 2x.

| axis | points measured | best | worst measured |
| --- | --- | --- | --- |
| `BLOCK_N1 = BLOCK_M2` | 128, **256**, 512 | 256 (8.68) | 512 = 59.10 ms, 5704 spills |
| `BLOCK_M1` | 16, **32**, 64 | 32 | 64 = 15.96 ms |
| `BLOCK_N2` | 16, **32**, 64, 128 | 32 | 128 = 24.35 ms, 2822 spills |
| `num_warps` | **4**, 8 | 4 | 8 = 22.5-23.6 ms |
| `waves_per_eu` | **0**, 1, 2 | 0 | 2 = 50.6 ms |
| `num_stages` | **1**, 2 | 1 | 2 = 12.2-12.9 ms |
| `BLK_SLICE_FACTOR` | **1**, 2 | 1 | 2 = 8.87 ms |
| `matrix_instr_nonkdim` | **16**, 32 | inert | byte-identical `.amdgcn` |
| `kpack` | **1**, 2 | inert | byte-identical `.amdgcn` |

**`BLOCK_M < 16` and non-power-of-two tiles are illegal**, so the axes are not even
continuous: `BLOCK_N1` 192 and 384 fail to compile with `ValueError: Shape element 0
must be a power of 2` out of `tl.zeros([BLOCK_N1, HEAD_DIM])`. There is no point between
256 and 512, and 512 is 6.8x slower.

Any future candidate of the form "retune a block size" is dead on arrival. What is left
is the kernel body.

## r3.i1.g12 -- `BLOCK_N2` upward, the dq pass's loop step (measured round 3, +106% / +180%)

Round 2 swept `BLOCK_N2` only down. The premise for going up came from round 3's
half-kernel probe: the dq pass compiled **alone** carries `vgpr_count 971` with **zero**
spills and zero scratch, so it looked like the one pass with register slack, and
`BLOCK_N2` is its loop STEP, not its residency (`dq` is sized by `BLOCK_M2`).

| | spill | scratch B/lane | wmma | instrs | bwd ms |
| --- | --- | --- | --- | --- | --- |
| `BLOCK_N2=32` (default) | 266 | 1064 | 448 | 9547 | 8.679-8.685 |
| `BLOCK_N2=64` | **554** | 1796 | 640 | 12935 | **17.865** |
| `BLOCK_N2=128` | **2822** | 4984 | 1024 | 20482 | **24.345** |
| `BLOCK_N2=64, waves_per_eu=1` | 609 | 1900 | 640 | 12979 | 16.736 |

**Mechanism, and it generalises past this axis: slack a pass has ALONE is not slack it
has INSIDE the fused kernel.** `bwd_kernel_causal` is one function and LLVM allocates it
from one interference graph at 1024 of 1024 VGPRs. Widening the dq step widens the
`kT`/`vT`/`qk`/`ds` tiles that have to coexist with the dk/dv pass's 768 loop-invariant
registers, and the allocator pays for it in the dk/dv loop: `wmma` rises 43% while spills
rise 108%. Never size one pass from a census taken with the other pass deleted.

## r3.i2.g13 -- `BLOCK_N1 = BLOCK_M2` above 256 (measured round 3, +581%)

The last untested direction on the tile axis, and the only one the trend pointed at:
tile 128 is 11.8-13.3 ms with ZERO spills, tile 256 is 8.68 ms with 266 spills, so
bigger-and-spillier had been winning. It stops at 256. **Tile 512 = 59.099 ms, 5704
spills, 6292 B/lane scratch.** `dk` + `dv` alone is 512 + 512 = 1024 accumulator VGPRs
at that tile -- the entire register file, before `k`, `v` or one temporary.

Reached by patching `fused_backward_tile()`, since `BLOCK_N1`/`BLOCK_M2` in the config
are overwritten from it (see the round-1 entry above).

## Unfusing the dq and dk/dv passes into two launches (priced round 3, starts 12.5% down)

Not a candidate until it pays for this first. `bwd_kernel_causal` gives ONE program id
both jobs -- `start_n = pid * BLOCK_N1` for dk/dv and `start_m = pid * BLOCK_M2` for dq,
with `BLOCK_N1 == BLOCK_M2` -- so at `seqlen 8192`, `BLOCK_N1 256`, `BLOCK_M1 32`,
causal, **every program does exactly 264 loop trips**:

| | dk/dv m-trips | dq n-trips | total |
| --- | --- | --- | --- |
| pid 0 | 256 | 8 | 264 |
| pid 15 | 128 | 136 | 264 |
| pid 31 | 8 | 256 | 264 |

The fusion IS the causal load balancer. It is the same mechanism AITER's
`fmha_v3_bwd_hd128_bf16` s6 b6 calls "worth copying" (pair K-tiles from both ends), and
this kernel already has it for free. So the `BLOCK_N1 == BLOCK_M2` clause of
`_check_block_invariant` is load-bearing twice: the grid, and the balance.

Measured consequence, from the `SKIP_DKDV`/`SKIP_DQ` constexpr probe: dk/dv alone 5.906
ms, dq alone 3.858 ms, **sum 9.764 against 8.678 fused, +12.5%** -- four dispatch rounds
of wildly unequal cost on 256 CUs once the pairing is gone.

And the usual reason for splitting does not apply either: **the dq pass compiled alone is
`vgpr_count 971`**, not <= 512, so `vgpr x waves <= 1024` still pins it at 1 wave/SIMD.
A split buys no occupancy and loses the balance.

## r3.i3.g14 -- stochastic PC sampling / ATT on this kernel (attempted round 3, does not run)

The corpus's stated replacement for the 38 dead gfx1250 counters, and round 3's `must`
row. **It does not run on `bwd_kernel_causal`, and it fails in two different ways
depending on the sampling interval** -- which is what makes this a mechanism and not a
flake.

| interval | result |
| --- | --- |
| 2048 cycles | GPU page fault inside the sampled process: `Memory access fault by GPU node-3 ... Reason: Page not present or supervisor privilege`, then `GPU core dump skipped because PC Sampling active`, `rocprofv3 caught signal 6`, then a hang in finalisation -- `Timeout while waiting for queue sync: 1 kernels still active` |
| 1048576 cycles (the coarsest the part accepts) | No fault **and no samples**. Process alive 13 minutes, GPU[1] at 0% utilisation, no CSV written, output file unmodified since tool initialisation. Killed |

Command, verbatim from `arch/gfx1250/profiling-surface.md`:
`rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit
cycles --pc-sampling-interval N --output-format csv ...`

**Mechanism.** The two failures are the two halves of one problem: sample often and the
interrupt faults on the wave save area, sample rarely and the queue never drains. The
kernel being sampled is resident at **1024 of 1024 VGPRs with 1064 B/lane of private
scratch on a single wave per SIMD** -- the worst case for a sampling path that has to save
and restore wave state to take a sample, and there is no second wave on the SIMD to make
progress while it does. This is the same family as the round-1 finding that `rocprofv3
--stats --kernel-trace` exits 0 having recorded **zero dispatches** on this benchmark: the
tool is not merely under-featured on gfx1250, its wave-interrupt path does not survive this
kernel. The device was verified healthy after both attempts (GPU[1] 0% use, no KFD PIDs, a
4096^3 bf16 matmul returns correctly), so this is not machine state.

**What it kills, and it is more than one candidate.** There is **no instruction-level
dynamic instrument on gfx1250 for this kernel**. Counters: 13 of 51 non-zero, no byte
counter at any level, WMMA FLOP counters exactly 0. Kernel trace: zero dispatches. PC
sampling and ATT: this entry. Any plan whose first step is "measure where the stalls are"
is dead before it starts, and any route row gated on such a measurement will return no
reading rather than a reading -- which is how round 3 ended with two idea rows it could not
open. **Route around it**: what is left and does work is the static `.amdgcn` census,
`torch.profiler` at dispatch granularity, and deletion probes (`opt.md` s4 -- constexpr
flags that delete a pass, priced by the difference).

One honest limit on this entry: it says the instrument does not run, not that it could
never be made to. It was attempted twice, at the two ends of the legal interval range, in a
fast round. If a later round has a reason to need it badly enough, the untried directions
are a smaller input shape, a different ROCm, or ATT rather than stochastic sampling.
