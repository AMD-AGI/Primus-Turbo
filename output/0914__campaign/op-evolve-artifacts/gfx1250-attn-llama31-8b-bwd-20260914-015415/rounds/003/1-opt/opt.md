# Round 3 -- fast round, gfx1250 attention backward (Triton one-kernel fused bwd)

Incumbent: round 1's ship, 8.69 ms bwd / 633 TF/s. Gate needs 1.50x over `op/beat`
(flex_attention, ~11.68 ms) = ~7.79 ms, i.e. ~10.3% still to find. Round 2 was
rejected (both arms ~7% slower).

Written as the round went, not afterwards.

## 0. What I read first

`findings/facts.md`, `dead_ends.md`, `pool.md`, `route.md`. State inherited:

* Bottleneck line (round 2, MEDIUM): instruction issue in the dk/dv loop body,
  driven by register pressure, on a power-capped part.
* Pool: `r1.i3.g3` (`schedule_hint`, anti-synergistic with the shipped ITT knob),
  `r1.i7.g7` (LDS staging, premise unrestated), `r1.i4.g4` refuted.
* Route row 1 of round 2 was a **`must`**: `r2.i3.g10`, "gate every candidate on
  `vgpr_spill_count` before spending GPU time on it". It was delivered and it is
  what round 3 inherited as the job's only quantitative predictor.

Highest `g` allocated anywhere: 11.

## 1. The static surface I censused before touching anything (free, no GPU cost)

Read `triton/backends/amd/compiler.py` in the pinned image. Findings that matter
and are not in `facts.md`:

* `is_async_copy_enabled()` already returns **True for gfx1250 by default**. So
  round 1's "`TRITON_HIP_USE_ASYNC_COPY` is noise" is not "the knob does nothing",
  it is "the knob was already on". Confirmed below: forcing it off produces a
  **byte-identical** `.amdgcn` at `num_stages=1`.
* `is_pingpong_schedule_enabled()` is False for gfx1250, and the pass is gated a
  second time by `if use_block_pingpong and options.num_stages > 1`. `num_stages=2`
  is already measured at 12.2-12.9 ms, so pingpong is unreachable without paying a
  known 40% loss first. Confirmed below: identical binary at `num_stages=1`.
* `amdgpu-waves-per-eu` is emitted as `"<N>, <N>"`, so `waves_per_eu=0` is `"0,0"`
  = no limit. Consistent with `r1.i1.g1`.
* **gfx1250 is specifically excluded from `set_all_fn_arg_inreg()`** -- kernel
  arguments are not placed in user SGPRs on this arch, they are `s_load`ed. That
  is upstream's choice, not a tuning knob, and it is a plausible part of why this
  kernel carries `sgpr_spill_count 67`.
* `USE_INT64_STRIDES` is already `False` at this launch. Nothing to win there.

### The cache-key trap I had to avoid

`get_cache_invalidating_env_vars()` returns exactly four names:
`AMDGCN_USE_BUFFER_OPS`, `TRITON_HIP_USE_ASYNC_COPY`, `TRITON_HIP_USE_BLOCK_PINGPONG`,
`TRITON_HIP_USE_IN_THREAD_TRANSPOSE`. **`AMDGCN_SCALARIZE_PACKED_FOPS` and
`AMDGCN_ANALYZE_SMALL_TENSOR_RANGE` are NOT in the key.** Sweeping either inside
one process serves a stale binary and reads as "this knob does nothing" -- the
exact silent-failure signature `op.reference.api` warns about and that round 1 lost
a `BLOCK_N1` sweep to. So every point below ran in **its own process with its own
`TRITON_CACHE_DIR`**, and every point prints the census of the binary it actually ran.

## 2. The knob sweep (8 points, one process and one fresh cache each)

`rounds/003/1-opt/raw/knobsweep.txt`. GPU[1] verified idle before
(`rocm-smi --showpids`: no KFD PIDs). Base slots first and last.

| point | `vgpr_spill_count` | scratch B/lane | total instrs | bwd ms |
| --- | --- | --- | --- | --- |
| base.pre | 266 | 1064 | 9547 | 8.679 |
| `AMDGCN_SCALARIZE_PACKED_FOPS=1` | 285 | 1140 | 10419 | 10.007 |
| `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1` | **187** | 716 | 10695 | **15.839** |
| `AMDGCN_USE_BUFFER_OPS=0` | 250 | 820 | 10757 | 15.147 |
| `TRITON_HIP_USE_BLOCK_PINGPONG=1` | 266 (byte-identical binary) | 1064 | 9547 | 8.714 |
| `TRITON_HIP_USE_ASYNC_COPY=0` | 266 (byte-identical binary) | 1064 | 9547 | 8.686 |
| `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=0` | **INVALID, see below** | | 9547 | 8.685 |
| base.post | 266 | 1064 | 9547 | 8.599 |

Two of these points are worth more than the sweep.

**The `itt_off` row measured nothing and I am recording it as an error, not a
result.** `impl.py` sets `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1` inside a context
manager around the launch, so my process-level `=0` was overwritten before the
compile. The binary is byte-identical to base, which is how I caught it. Anyone
re-testing that knob has to edit `impl.py`, not the environment.

**Measurement floor for THIS harness is ~0.9%, not round 2's 0.03-0.08%.**
base.pre 8.679 vs base.post 8.599 is 0.93% apart. Round 2's tight floor was
in-process, back-to-back; mine is one process per point across a ~10 minute
session and drifts more. Anything under ~1% here is unresolved.

### FINDING: `r2.i3.g10`'s law is refuted AT A FIXED TILE, not only across tiles

Round 2 already knew "time tracks `vgpr_spill_count`" fails across tiles, and
narrowed it to "within one tile, fewer spills is faster, monotonically". That
narrowed form is now dead too.

`AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1` is the counter-example. **Same tile
(`BLOCK_N1=BLOCK_M2=256`), same source file, same launch config** -- only the
LLVM-side buffer-offset range analysis changes. It compiles to `vgpr_spill_count
187`, the LOWEST spill number this job has ever produced at this tile (30% below
the incumbent's 266, scratch 1064 -> 716 B/lane), and it runs at **15.839 ms
against 8.679, +82%.** `AMDGCN_USE_BUFFER_OPS=0` is a second such point: 250
spills, +74%.

So the census gate is not an objective, and it is not even a safe screen in the
"fewer spills is at least not worse" direction. Its **discard** rule
(`spills >= 266 -> do not measure`) is still defensible as a heuristic -- every
point this job has measured with MORE spills than 266 was indeed slower -- but
its **measure** rule now has a counter-example at 1.82x. Route row 1 of round 2
carried it as a `must`; it goes into the route this round demoted to "necessary,
not sufficient" with that caveat attached.

What the two slow points have in common is that they both change **addressing**,
not register pressure: buffer ops off, or buffer offsets re-analysed. Both cost
~12% more instructions and ~75-82% more time, which no instruction-count or
spill-count model explains. On this kernel the memory *addressing path* is worth
far more than 266 spill slots are -- a statement `facts.md` does not contain and
that sits uneasily with "the kernel is not waiting on memory (154 GB/s)".

### Knobs that are simply unreachable on this part, established statically + confirmed

* block pingpong: gated off for gfx1250 AND gated on `num_stages > 1`, which is
  already measured at 12.2-12.9 ms. Identical binary at `num_stages=1`. Dead.
* async copy: already on by default for gfx1250, and at `num_stages=1` it makes
  no difference to the emitted code at all (identical binary either way).
  Round 1's "noise" reading was right for the wrong reason.

## 3. FINDING: there is no cheap slice outside `bwd_kernel_causal`

Nobody in this job had taken the dispatch-level breakdown of the 8.69 ms;
`facts.md` talks only about the main kernel. `torch.profiler` (rocprofv3 is dead
on this part), 20 iterations, `rounds/003/1-opt/raw/breakdown.txt`:

| | ms/iter | share |
| --- | --- | --- |
| `bwd_kernel_causal` | 8.5832 | **99.2%** |
| `_bwd_preprocess` | 0.0445 | 0.5% |
| everything else (LSE gather `aten::index`, `floor_divide`, `remainder`, `arange`, `add`, `mul`, `zeros_like`) | 0.033 total | 0.4% |
| wall-clock median for the same build | 8.639 | |

The host-side packed-LSE gather and the three `empty_like` allocations that the
adapter does inside the timed backward cost **0.4% combined**. `PRE_BLOCK` and the
preprocess pass are worth 0.5% in total and cannot pay for their own round.
**Every remaining percent has to come out of the one kernel.** Closing this is
worth recording precisely because it is the kind of question each round assumes
the next one will answer.


## 4. `r3.i0` (free, no shipped code) -- price each pass by deleting it

The corpus's own no-counter fallback is "price each phase by deleting it"
(`optimization/routes/1-metrics-to-techniques.md`). Built as a throwaway
(`scratch/probe_halves/`): two `tl.constexpr` flags `SKIP_DKDV` / `SKIP_DQ` threaded
through both kernels and guarding the two sections, defaulted to 0 and added to
`_ZERO_MEANS_UNSET`. Constexpr, so each setting is its own binary and its own
register allocation. Four points, base first and last.

| point | vgpr | spill | sspill | scratch B/lane | wmma | msb | instrs | ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| probe.full | 1024 | 266 | 67 | 1064 | 448 | 1666 | 9547 | 8.678 |
| probe.dkdv_only (`SKIP_DQ=1`) | 1024 | 263 | 31 | 1064 | 256 | 1026 | 5793 | **5.906** |
| probe.dq_only (`SKIP_DKDV=1`) | **971** | **0** | **0** | **0** | 192 | 564 | 3611 | **3.858** |
| probe.full2 | 1024 | 266 | 67 | 1064 | 448 | 1666 | 9547 | 8.679 |

Base slots reproduce the incumbent to the byte, so the constexpr addition is inert at 0/0.

Three results, all new to the job:

1. **All 266 spills are the dk/dv pass's own.** dk/dv alone spills 263 of the fused
   kernel's 266; dq alone spills **zero** at `vgpr_count 971`, `scratch 0 B/lane`.
   Fusion costs the register allocator almost nothing -- the two passes are sequential
   and do not interfere. So "unfuse to relieve pressure" has no pressure to relieve.
2. **A split would not buy the dq half any occupancy.** `vgpr_count 971`, not <= 512.
   The precondition `vgpr x waves <= 1024` still pins the dq kernel at 1 wave/SIMD on
   its own. And the 971 carries zero spills, so it is genuine live state, not allocator
   pressure leaking from the dk/dv half. This kills the "split the one-kernel so dq runs
   at 2 waves/SIMD" candidate on its own evidence.
3. **The halves sum to 9.764 ms against 8.678 fused, +12.5%.** The cause is visible in
   the source and it is the important finding of this section -- see s5.

## 5. Why the fusion is load-bearing, and why corpus item B9 is already done

`bwd_kernel_causal` gives ONE program id both jobs: `start_n = pid * BLOCK_N1` for
dk/dv and `start_m = pid * BLOCK_M2` for dq, with `BLOCK_N1 == BLOCK_M2` enforced.
At `seqlen 8192`, `BLOCK_N1 = 256`, `BLOCK_M1 = BLOCK_N2 = 32`, causal:

| | dk/dv m-trips | dq n-trips | total |
| --- | --- | --- | --- |
| pid 0 | (8192-0)/32 = 256 | (0+256)/32 = 8 | 264 |
| pid 15 | 128 | 136 | 264 |
| pid 31 | 8 | 256 | 264 |

**Every program does exactly 264 trips.** The causal triangle is balanced by pairing a
K-tile with the Q-tile at the same index, whose work is its complement.

AITER's `fmha_v3_bwd_hd128_bf16` §6 b6 -- "pair K-tiles from both ends ... a two-line
host change ... makes causal attention perfectly balanced ... worth copying" -- is the
same mechanism, and **this kernel already has it, for free, as a consequence of fusing.**
Corpus candidate B9 is therefore not available: it is shipped.

Two things follow.

- The `BLOCK_N1 == BLOCK_M2` clause of `_check_block_invariant` is **not** merely a grid
  artefact. It is what makes the pairing exact. Round 2 established the *other* clause
  (`BLOCK_M1 == BLOCK_N2`) over-strict; this one is load-bearing twice over.
- The +12.5% the split costs in s4 is explained: split apart, each kernel's work runs
  from 256 trips down to 8 across 32 pids, over 1024 workgroups on 256 CUs -- four
  dispatch rounds of wildly unequal cost, i.e. a tail. **Any candidate that unfuses the
  two passes has to re-earn this balance before it earns anything else**, and it starts
  12.5% down. Recorded as the reason the split is not an arm this round.

## 6. `r3.i1.g12` -- `BLOCK_N2` upward (the dq loop step): REFUTED

Round 2 swept `BLOCK_N2` only downward (16, and `BLOCK_M1` in both directions), all
losing, and concluded the axis was exhausted. The premise for going *up* was s4's new
number: the dq pass carries zero spills, so it is the one pass with register slack, and
`BLOCK_N2` is its loop STEP -- raising it halves the trip count and its per-trip overhead
without growing the `dq` accumulator, which is sized by `BLOCK_M2`. Built as a throwaway
(`scratch/probe_n2/`, the `BLOCK_M1 == BLOCK_N2` clause deleted).

| point | vgpr | spill | scratch B/lane | wmma | msb | instrs | ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n2.base | 1024 | 266 | 1064 | 448 | 1666 | 9547 | 8.685 |
| `BLOCK_N2=64` | 1024 | **554** | 1796 | 640 | 2538 | 12935 | **17.865** |
| `BLOCK_N2=128` | 1024 | **2822** | 4984 | 1024 | 4435 | 20482 | **24.345** |
| `BLOCK_N2=64, waves_per_eu=1` | 1024 | 609 | 1900 | 640 | 2632 | 12979 | 16.736 |
| n2.base2 | 1024 | 266 | 1064 | 448 | 1666 | 9547 | 8.679 |

**+106% and +180%.** The premise was wrong in a specific way worth recording: the slack
the dq pass has *alone* is not slack it has *inside the fused kernel*. The fused function
is allocated from one interference graph at 1024 of 1024 VGPRs, so widening the dq step
widens `kT`/`vT`/`qk`/`ds` tiles that must coexist with the dk/dv pass's 768 loop-invariant
registers, and the allocator pays in the dk/dv loop -- note `wmma` rises only 43% while
spills rise 108% and the whole `msb` count rises with it.

**The tile/step surface is now closed in every direction**: `BLOCK_N1` 128 (round 1,
11.8-13.3 ms) and 256 (incumbent); `BLOCK_M1` 16 / 64 and `BLOCK_N2` 16 (round 2, all
losing); `BLOCK_N2` 64 / 128 (here); `num_warps` 8; `waves_per_eu` 1 / 2;
`BLK_SLICE_FACTOR` 2; `num_stages` 2.

## 7. `kpack` and `matrix_instr_nonkdim` are inert on this path (free)

Run in the same session as s6. `kpack=2` -> 8.680 ms, `matrix_instr_nonkdim=32` ->
8.686 ms, both against 8.679-8.685 base. Not "within noise": the emitted `.amdgcn` is
**byte-identical to base** on every census field (1024/266/67/1064/448/1666/9547).

This matters because corpus item B2 -- "pick the WMMA shape per product from the
contracted dimension, not one shape for the kernel", the only corpus mechanism with a
measured number on a dK/dV body (+8.7%) -- routes through `matrix_instr_nonkdim` in
Triton, and that knob does not reach the gfx1250 lowering. gfx1250 wmma v3 bf16 is
`v_wmma_f32_16x16x32_bf16` and Triton emits it regardless. **B2 is unreachable from
Triton on this part**, and per-product shape selection would need a different backend.

## 8. `r3.i2.g13` -- the tile ABOVE 256: REFUTED, and the axis is now closed

The only untested direction left on the tile axis. Its premise was the one monotone
trend in the job: `BLOCK_N1 = BLOCK_M2` at 128 measures 11.8-13.3 ms with **zero**
spills, at 256 measures 8.68 ms with 266 spills -- bigger tile, more spills, faster --
because Q/dO traffic per output halves and round 2 established Q/dO traffic, not spill
count, as the term that dominates across tiles. Nobody had ever gone up. Round 1's
config sweep could not reach it (`BLOCK_N1`/`BLOCK_M2` are overwritten by
`fused_backward_tile`), so it was swept from there in a throwaway (`scratch/probe_tile/`).

| point | vgpr | spill | scratch B/lane | wmma | instrs | ms |
| --- | --- | --- | --- | --- | --- | --- |
| tile 256 (pre) | 1024 | 266 | 1064 | 448 | 9547 | 8.673 |
| tile 384 | -- | -- | -- | -- | -- | **compile error** |
| tile 512 | 1024 | **5704** | 6292 | 896 | 23231 | **59.099** |
| tile 192 | -- | -- | -- | -- | -- | **compile error** |
| tile 256 (post) | 1024 | 266 | 1064 | 448 | 9547 | 8.678 |

Build failures, recorded as they happened: `PROBE_TILE=384` and `PROBE_TILE=192` both
fail to compile with `ValueError: Shape element 0 must be a power of 2`, raised out of
`tl.zeros([BLOCK_N1, HEAD_DIM])` and reaching `tl.dot`. **Triton block shapes must be
powers of two**, so the tile axis is not continuous: the only point above 256 is 512.

512 costs **21x the spills and 6.8x the time**. `dk`+`dv` alone is 512+512 = 1024
accumulator VGPRs at that tile, the entire file, before `k`, `v` or any temporary.

**Conclusion for the round, and the reason it ships nothing.** With s6 and this section,
every knob on this kernel's launch/tile surface has now been measured in both directions
and 256/32/32/4-warps is a strict local optimum:

| axis | points measured | best |
| --- | --- | --- |
| `BLOCK_N1 = BLOCK_M2` | 128, **256**, 512 (384/192 illegal) | 256 |
| `BLOCK_M1` | 16, **32**, 64 | 32 |
| `BLOCK_N2` | 16, **32**, 64, 128 | 32 |
| `num_warps` | **4**, 8 | 4 |
| `waves_per_eu` | **0**, 1, 2 | 0 |
| `num_stages` | **1**, 2 | 1 |
| `BLK_SLICE_FACTOR` | **1**, 2 | 1 |
| `matrix_instr_nonkdim` | **16**, 32 (byte-identical) | inert |
| `kpack` | **1**, 2 (byte-identical) | inert |
| compiler env knobs | 8 points, s2 | all lose or inert |

Nothing on this surface is left to try. What remains is a rewrite of the kernel body,
and s4/s5 say which rewrites are already dead.

## 9. Corpus and cross-backend reading (step 3c/3d)

Files opened are listed in `explored.consulted` in the reply. Four things changed a
decision; the rest either confirmed `facts.md` or is out of scope for this part.

**9a. The power regime re-prices the whole candidate list, and it is measured on this
part.** `backends/hipkittens/gemm/recipes/bf16_gfx1250_ladder.md` is the corpus's only
gfx1250 artefact (same 256-CU, 2500 W part): under sustained bf16 GEMM, socket power
sits at **2497-2502 W of a 2500 W cap** with the GFX clock held at **1699-1703 MHz
against `MAX_CLK` 2400** -- 29% below maximum. With `profiling/2-power-wall-analysis.md`:
**price "fewer bytes" at full value and "fewer cycles / fewer instructions" at 20-45%.**

Applied to `facts.md`'s current-bottleneck line: the dk/dv loop's 363 `s_set_vgpr_msb`
(26% of the body) and 84 `v_nop` (6%) are an *instruction-count* overhead, and
`s_set_vgpr_msb` is a SALU bank switch that switches almost nothing. **That 32% is not
a 32% time opportunity; at the corpus's coefficient it is 6-14%, and the energy-valued
part of the 37% figure is the 77 `scratch_load` per iteration.** This is why the round
routes the next one at scratch bytes and not at the `msb` count, which is the larger
number.

Two hygiene warnings from the same card, both adopted: `THROTTLE_STATUS` reads `N/A`
on this part even while pinned at the cap, so it is not the power-wall test (use
power-at-limit plus clock-below-`MAX_CLK`); and a baseline taken on a long-uptime host
measured **19% low on exactly the data-movement rungs** after a GPU hang, so a baseline
is only comparable inside its own session. This round took base slots first and last in
every sweep for that reason and they agree to 0.07%.

**9b. Corpus B9 (AITER causal K-tile pairing, "worth copying") is already shipped here.**
See s5. Removed from the candidate list.

**9c. Corpus B2 (per-product WMMA shape, +8.7% measured on a dK/dV body) is unreachable
from Triton on this part.** See s7: `matrix_instr_nonkdim` emits a byte-identical binary.

**9d. Corpus B1 / B5 / B6 all say the same thing and all say it about bytes:**
- B1, `backends/flydsl/attention/recipes/hd128.md` §6 b1: at D=128 the shipping config
  pays for its accumulators by taking K/V from **LDS** rather than registers
  (`k_reg=False`, `kv_halves=2`), and partial traffic per FLOP depends on nothing but
  `BLOCK_KV` -- i.e. **this axis lowers residency without touching Q/dO traffic**, which
  is the exact open form `dead_ends.md` left after killing `r2.i4.g11`. `hd64.md` keeps V
  in registers at D=64, so the corpus's own position is that this is a D=128 decision.
  It targets the `k` 128 + `v` 128 of our 768, not the 512 of accumulator.
- B5, ladder rung 02: `global_load_async_to_lds_b128` measured **+44.78%** here,
  mechanism "lands operands in LDS without staging them through VGPRs, so the registers
  and issue slots the register-mediated copy spent go back to the K loop".
- B6, ladder rung 10: LDS-staged epilogue measured **+12.09%** on this part (the
  publisher measured +7.32% on theirs), `buffer_store_b64` -> `global_store_b128`, and
  **it costs no LDS** because the staging tile aliases the operand ring the loop has
  finished with. Our dk/dv pass ends by storing 512 accumulator VGPRs and the census
  shows 128 static `scratch_store`.

This is one mechanism family, it is the only family with gfx1250-measured numbers behind
it, and it lands on `r1.i7.g7`, which keeps its id and is restated in `pool.md` around
B1/B5/B6 instead of around the "LDS is at 20%" premise round 2 corrected away. The
blocker is unchanged and is now named precisely: **`group_segment_fixed_size: 0` means
Triton allocates this kernel no LDS at all**, and Triton has no surface to ask for an
explicit shared tile. `TRITON_HIP_USE_ASYNC_COPY` is on by default for gfx1250 (s2) and
changes nothing at `num_stages=1`, so the flag is not the mechanism.

**9e. Two scope corrections to carry.** `optimization/techniques/0-register-pressure-and-occupancy.md`
and `optimization/routes/1-metrics-to-techniques.md` are `applies_to: gfx942, gfx950`
and their metric tables are dead here; what survives is the no-counter fallback **"price
each phase by deleting it"**, which is s4. And `knowledge/backends/triton/` does not
exist -- the corpus names zero Triton env knobs, so step 3(d) returns nothing at the
Triton level; the knobs it names are `-mllvm`-level, and `gqa_d64.md` records four of
them (`-fno-offload-uniform-block`, `--lsr-drop-solution=1`, `-enable-post-misched=0`,
`-amdgpu-early-inline-all=true`) that **silently miscompile**, same VGPR/spill report, no
warning, SQNR 52 dB -> 0.6 dB with ~18% NaN rows. Triton 3.6.0's `make_amdgcn` passes
`flags = []` and has no LLVM flag passthrough anyway (s2), so this is closed both ways.
Also not transferable: HipKittens' `art` pinned-register mechanism (1.72x) is an **AGPR**
mechanism and gfx1250 has no AGPRs.

## 10. Outcome

**Both arms lost, so nothing enters the working copy.** `rounds/003/op` is byte-identical
to `job_context/op/current`. Arms were built as throwaways rather than in the working
copy because both were config-reachable, which made them free and reversible; the losses
are 2.1x and 6.8x, far outside any merge consideration, so no merge was built.

Noted, not acted on: `impl.py` sets `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1` around the
backward launch, but `is_in_thread_transpose_enabled` in this Triton returns true only
for `gfx942`, so the context manager is **inert on gfx1250**. It is pre-existing shipped
code and removing it is not a speedup, so it is left alone and recorded here. (It also
means a sweep point that sets that variable to 0 in the process environment measures
nothing -- the context manager overwrites it. This round lost one sweep point to that
and caught it because the emitted `.amdgcn` was byte-identical to base; see s2.)

## 11. Validation, as measured

`op/validation.py rounds/003/op`, through the runner, GPU[1] verified idle by
`rocm-smi --showpids` / `--showuse` before and after. Raw in `raw/validate_ship.txt`.
The working copy is byte-identical to `op/current`, so this is a re-measurement of the
incumbent in this session, which is also the round's honest ship report.

| | fwd ms | bwd ms | bwd TF/s | bwd GB/s | spread% |
| --- | --- | --- | --- | --- | --- |
| candidate | 4.9526 | **8.7011** | 631.90 | 154.3 | 0.53 |
| beat (flex_attention) | 3.7258 | 11.6821 | 470.65 | 114.9 | 2.78 |
| baseline | 4.9832 | 10.0082 | 549.37 | 134.1 | 0.58 |

**1.343x vs beat, needs 1.50x -- FAIL, short by 10.5% of the target.** 1.150x vs
baseline. Round 1 shipped 8.683 ms / 1.345x; this session reads 8.7011 / 1.343x, a 0.2%
difference against a 0.53% in-run spread, so the incumbent has not moved and nothing in
this round moved it.

Correctness, `validation.py`'s own SQNR gate (>= 50 dB on out/dq/dk/dv, fp64, against
`op/eager/`), unchanged because the code is unchanged: **pass on the spec shape
(53.67 / 52.24 / 52.31 / 52.71) and on all four edge shapes that are gated.** The
`sq_gt_skv` diagnostic row reads `dq -15.95 dB`; it is reported as not gated, it is
inherited rather than introduced here, and this round wrote no kernel code at all.

## 12. What the round leaves

- The config surface is closed and documented as closed, so no future round spends GPU
  time retuning a block size.
- The fusion is understood as the causal load balancer, which prices every unfusing
  candidate at -12.5% before it starts and removes corpus item B9 from the list.
- The two remaining families are separated and priced against the measured power cap,
  and `r3.i3.g14` is the one cheap measurement that decides between them.
- `r1.i7.g7` keeps its id and is restated on gfx1250-measured corpus mechanisms
  (+44.78% rung 02, +12.09% rung 10) with its blocker named exactly: Triton has no
  surface for an explicit shared tile and allocates this kernel zero LDS.
- `r3.i4.g15` is new, and its gate -- reading the `DK`/`DV` store widths out of the
  shipped `.amdgcn` -- is free and untaken.

## 13. Route row 3 (`r3.i4.g15`): DEAD on its own gate, for one grep

The row said: read the `DK`/`DV` store widths out of the shipped `.amdgcn` first, and if
they are already `b128` the row dies for one grep. They are.

| store opcode | count |
| --- | --- |
| `ds_store_b128` | 160 |
| `buffer_store_b128` | **96** |
| `scratch_store_b32` | 82 |
| `ds_store_b32` | 68 |
| `scratch_store_b128` | 46 |
| `ds_store_2addr_b64` | 8 |
| `buffer_store_b64` | **0** |

**Every global store in this kernel is already `b128`, the widest form.** Corpus rung 10's
`buffer_store_b64 -> global_store_b128` coalescing win (+12.09% measured on this part)
has nothing here to convert. `r3.i4.g15` is closed at `look`, on the evidence its own
condition asked for, at a cost of one grep and no GPU time. That is what the row was for.

## 14. An unplanned reading that FALSIFIES a premise round 2 wrote into the pool

While reading store widths I read the Triton metadata beside the `.amdgcn`:

```
shared = 65536        # bwd_kernel_causal.json
global_scratch_size = 0
```

against the ELF's `.group_segment_fixed_size: 0`.

**Round 2's correction was itself wrong, and round 1 was right.** `group_segment_fixed_size`
is 0 because Triton passes this kernel's LDS as **dynamic** shared memory at launch, not
because the kernel uses none. It uses **65,536 B of 327,680 (20%)**, exactly the figure
round 1's `r1.i7.g7` was built on and that round 2 retracted as "never a reading anyone has
taken on THIS build". The 656 `ds_*` ops in the census -- 160 `ds_store_b128`, 68
`ds_store_b32`, 8 `ds_store_2addr_b64` and the `ds_load`s -- are real traffic against a
real allocation, which is also the `facts.md` anomaly noted in s1 (108 `ds_*` per dk/dv
iteration against an apparently zero LDS segment) resolved.

Two consequences, and the second one matters more than this round's arms did:

1. `r1.i7.g7`'s headroom argument is **restored**: 65,536 of 327,680 used, ~262 KB free.
   The pool entry was rewritten at step 4 around the corpus mechanism but still carried
   round 2's "Triton allocates this kernel no LDS at all" as the blocker. **That blocker is
   false.** Triton does allocate and use LDS here.
2. What remains true is narrower and is the real question for the next round: Triton
   allocates LDS where *its own lowering* wants it, and offers no surface to say "hold `k`
   and `v` there across the m-loop". So the item is still not a knob -- but it is no longer
   "the backend cannot use LDS", it is "the backend will not let me choose what goes in
   it", which is a different and much smaller gap, and one that `num_stages`/layout hints
   sit closer to than a rewrite does.

This was not on the route. It cost one file read, it came out of executing row 3's gate,
and it is the most useful thing the round produced.

## 15. Route row 1 (`r3.i3.g14`): ABANDONED -- PC sampling does not run on this kernel

The must-row, the gate on rows 2-3, and the corpus's stated replacement for the 38 dead
gfx1250 counters. Two attempts, two different failures, and the pair is what identifies
the mechanism.

| attempt | interval | result |
| --- | --- | --- |
| 1 | 2048 cycles | **GPU page fault.** `Memory access fault by GPU node-3 ... Reason: Page not present or supervisor privilege`, then `GPU core dump skipped because PC Sampling active`, `rocprofv3 caught signal 6`, and a hang in finalisation: `Timeout while waiting for queue sync: 1 kernels still active` |
| 2 | 1048576 cycles (the coarsest the part accepts) | **No fault, and no samples.** Process alive 13 minutes, GPU[1] at 0% utilisation, no CSV written, output file unmodified since tool initialisation. Killed |

Device verified healthy after both: GPU[1] 0% use, no KFD PIDs, and a 4096^3 bf16 matmul
returns correctly. Neither attempt left anything running. Raw in
`raw/pcsample_attempt1.txt` and `raw/pcsample_attempt2.txt`.

**Mechanism.** The two failures are not one flake at two rates -- they are the two halves
of one problem. Sample often and the interrupt faults on the save area; sample rarely and
the queue never drains. The kernel being sampled is resident at **1024 of 1024 VGPRs with
1064 B/lane of private scratch on a single wave per SIMD**, which is the worst case for a
sampling path that has to spill and restore wave state to take a sample. This is the same
family as the already-recorded finding that `rocprofv3 --stats --kernel-trace` exits 0
having recorded **zero dispatches** on this benchmark: the tool is not merely
under-featured on gfx1250, its wave-interrupt path does not survive this kernel.

**It closes the last instrument the corpus offered for this part.** Counters: 13/51
non-zero, no byte counter, WMMA FLOP counters exactly 0. Kernel trace: zero dispatches.
PC sampling and ATT: this section. What is left is what rounds 2 and 3 actually used --
the static `.amdgcn` census, `torch.profiler` at dispatch granularity, and deletion
probes -- and the next round should be routed on the understanding that **no
instruction-level dynamic instrument exists on this part**, rather than spending another
row discovering it.

**Consequence for the rest of the table, and why nothing was built.** Row 2 (`r1.i7.g7`)
was explicitly gated: "pass row 1 as byte-bound first ... it should not be opened until
row 1 says the bytes are worth it." Row 1 returned no reading, so opening row 2 would mean
building the round's largest structural item on an inference the table forbade -- and s14
has just changed its premise anyway. Row 3 was already dead on its own free gate (s13).
So both rows the table could reach were unreachable for reasons the table itself
specified, and **no code was written.** `rounds/003/op` is byte-identical to `op/current`:
there is no change to leave in the working copy and none was reverted.

## 16. Final measurement, in one session

Four processes back to back on an idle GPU[1], palindromic, `op/benchmark.py --shape all`.
Raw in `raw/final_bench.txt`.

| order | arm | spec-shape bwd ms | bwd TF/s |
| --- | --- | --- | --- |
| 1 | rounds/003/op | 8.6742 | 633.86 |
| 2 | rounds/001/op (incumbent) | 8.6869 | 632.93 |
| 3 | rounds/001/op (incumbent) | 8.6788 | 633.52 |
| 4 | rounds/003/op | 8.6800 | 633.43 |

**633.6 vs 633.2 TF/s, `vs_champion` 1.0007** -- 1.0 by construction, since the trees are
byte-identical, and measured rather than assumed. The 0.07% spread across four processes is
this harness's floor and is tighter than the 0.93% s2 measured earlier in the round.

`score` 0.8975 against target 706 TF/s; round 1 recorded 0.8968. `validation.py` exit 1,
1.343x vs beat against the 1.50x gate.
