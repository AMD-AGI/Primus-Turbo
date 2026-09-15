# Pool

Ideas worth keeping, with the evidence behind each. `g` is monotonic across the job
and never reused; the highest `g` allocated by this job is 11.

`r1.i1.g1` and `r1.i2.g2` were executed in round 1 and have moved to `facts.md`.
`r1.i5.g5`, `r2.i1.g8` and `r2.i2.g9` were executed in round 2, all three lost, and
have moved to `dead_ends.md`. `r1.i6.g6` was answered in round 2 and has moved to
`facts.md`. `r1.i4.g4` is **refuted** -- see below. `r3.i1.g12` and `r3.i2.g13` were
executed in round 3, both lost by 2.1x and 6.8x, and have moved to `dead_ends.md`.
The highest `g` allocated is now **15**. `r3.i3.g14` was executed in round 3,
would not run, and has moved to `dead_ends.md`.

## r1.i3.g3 -- `schedule_hint="attention"` (open; anti-synergistic with g2)

Not executed. Alone it measures 9.633 ms (-3.6%) and with g1 9.526 ms, both real.
But **on top of g2 it loses**: 8.941 ms against 8.696 ms without it. Two schedulers
arguing about the same instruction stream. Keep it for the case where g2 is
withdrawn or the kernel is restructured; do not stack it on g2 as it stands.
`"memory-bound-attention"` (9.598 ms) and `"attention,memory-bound-attention"`
(10.032 ms) are the same family and subsumed.

## r1.i4.g4 -- shorten live ranges in `bwd_kernel_causal` (REFUTED AS STATED, round 2)

**Do not take this item as written.** Round 2 executed it twice, as the only two arms
of the round, and both arms LOST by ~7%: `r2.i1.g8` and `r2.i2.g9` each shortened a
live range at the source level and each made `vgpr_spill_count` WORSE (266 -> 307 and
266 -> 325). Source-level live-range reasoning does not predict what LLVM's allocator
does at 1024/1024 VGPRs; see `r2.i3.g10` for what does. The item survives only in the
narrow form "reduce `vgpr_spill_count`", which is a measurable target rather than an
argument, and any candidate claiming it must show the census before it is measured.

The original text is kept below because its negative results are still valid.

Not executed -- this is the structural item, and it is the only one with the size to
reach the gate. Static census: `vgpr_count 1024`, `vgpr_spill_count 326`,
`sgpr_spill_count 65`, `private_segment_fixed_size 1308` B/lane, 1 wave/SIMD. The
kernel achieves 549 TFLOP/s where bf16 GEMM on this chip reaches 2236-2667.

What is NOT the answer, measured this round: more warps (`num_warps=8` is 2.36x
SLOWER, not faster), smaller or larger tiles, `num_stages=2`. The tile/warp surface
is a sharp local optimum and every neighbour is worse. So the fix has to be inside
the kernel body -- recompute instead of keeping a tile live, or split the dq phase
from the dk/dv phase's register set -- not in the launch config.

## r1.i5.g5 -- `BLK_SLICE_FACTOR > 1` -- DEAD, measured round 2 (+2.3%)

Measured: 8.8726 ms against 8.6720 ms default, in one process, against a floor of
0.07%. Moved to `dead_ends.md`. Original text kept for the reasoning it records.



Not executed and not measured this round. `_DEFAULT_ONEKERNEL_CONFIG` pins
`BLK_SLICE_FACTOR: 1`; it slices the inner block and is the one config field that
plausibly shortens live ranges without changing the tile shape the rest of the
surface likes. Unlike `BLOCK_N1`/`BLOCK_M2` it is not overwritten by
`dense_fused_backward`, so the env tune knob actually reaches it. One sweep.

## r1.i6.g6 -- an instrument saying WHERE the time goes -- ANSWERED, round 2

Answered statically, with no profiler and no GPU, by mapping loop regions in the
compiled `.amdgcn` via backward-branch analysis and counting instructions per region.
Result in `facts.md`: all scratch traffic is in the dk/dv pass, the dq pass has none,
and 37% of the dominant loop body is register-pressure overhead. Moved to `facts.md`.
`rocprofv3` remains dead on this part and that is now irrelevant.

Original text kept below.



Not executed. `rocprofv3 --stats --kernel-trace` records **zero dispatches** on this
job's benchmark (see `rounds/001/1-opt/opt.md`), so every candidate above g4 is
currently being chosen from a static census and end-to-end times. gfx1250 has only
13/51 counters non-zero and no byte counters at any level, but PC sampling
(stochastic) and ATT are reported to work. Until one of them runs, g4 is a guess
about which live range to attack.

## r1.i7.g7 -- stage the spilled state in LDS (open; restated round 3 around the corpus mechanism)

Still not executed, and after round 3 it is **the only surviving structural candidate**.
Round 3 closed every other direction: the whole tile/step/warp/stage surface is measured
in both directions and 256/32/32/4 is a strict local optimum (`opt.md` s8), the compiler
env surface loses or is inert (s2, s7), and unfusing the two passes starts 12.5% down
because the fusion is what balances the causal triangle (s5).

**Round 2's correction stands and the premise is now rebuilt on top of it.** The shipped
build reports `group_segment_fixed_size: 0` -- Triton allocates this kernel **no LDS at
all** -- so "LDS is at 20%" was never a reading of this build and the headroom argument
below is void as written. What replaces it is three corpus mechanisms that all say the
same thing and all have gfx1250-measured numbers, unlike the original argument:

| | source | what it moves out of registers | measured |
| --- | --- | --- | --- |
| B1 | `backends/flydsl/attention/recipes/hd128.md` s6 b1 | K/V B-operands (`k_reg=False`, `kv_halves=2`) -- **at constant Q/dO traffic** | the shipping hd128 config; `hd64.md` keeps V in registers, so this is a D=128 decision |
| B5 | ladder rung 02 | the VGPR staging of a global load (`global_load_async_to_lds_b128`) | **+44.78%** on this part |
| B6 | ladder rung 10 | the epilogue store, `buffer_store_b64` -> `global_store_b128`, **costing no LDS** (aliases the finished operand ring) | **+12.09%** on this part |

B1 is the one that matters most here because it is aimed at exactly the open form
`dead_ends.md` left after killing `r2.i4.g11`: **lower residency without buying it with
Q/dO re-reads.** Our 768 loop-invariant VGPRs are `dk` 256 + `dv` 256 + **`k` 128 + `v`
128**, and B1 targets the 256 in K/V, which is the half that is *not* an accumulator and
*not* sized by anything Q/dO touches.

**The blocker is now named exactly, and it is not a knob.** `TRITON_HIP_USE_ASYNC_COPY`
is on by default for gfx1250 and produces a byte-identical binary at `num_stages=1`
(`opt.md` s2), so round 1's "noise" reading was right for the wrong reason -- the flag is
not the mechanism. `num_stages=2` is 12.2-12.9 ms. Triton has no surface for an explicit
shared tile, so B1/B5/B6 are **not expressible in this backend**; reaching them means
writing the dk/dv pass somewhere that can name LDS. That is the size of the item, and it
should be taken only after `r3.i3.g14` says the dk/dv loop is byte-bound rather than
issue-bound, because under the measured 2500 W cap a cycles-only win prices at 20-45%
while a bytes win prices at full.

Original text kept below.

`bwd_kernel_causal` uses **LDS 65,536 B of 327,680 (20%)** while spilling **326 VGPRs
/ 1308 B per lane to scratch**. Every backward attention in the corpus runs its LDS
near the limit and spills nothing:

| kernel | LDS used / limit | VGPR | spill |
| --- | --- | --- | --- |
| this kernel | 65,536 / 327,680 (20%) | 1024 | 326 (1308 B/lane scratch) |
| AITER `fmha_v3_bwd_hd128_bf16` (gfx950, asm) | 163,840 / 163,840 (100%) | 256+256 | 0 |
| FlyDSL `dkdv` hd128 a16 | 116,736 | 512 | 10 dwords / 20 B |
| HipKittens `gqa_d128` bwd | 140,288 / 160,000 (88%) | 256 | 0 |

So the single largest structural difference between this kernel and all three
references is that **it holds in registers what they hold in LDS**. AITER's form is
the extreme: all 65 vector loads carry the `lds` modifier, so Q/dO/LSE never land in
a register at all, and there are 512 static `ds_read_b64_tr_b16` with zero `ds_write`
of transposed data and exactly 0 bank conflicts.

Why this is the right *kind* of change on this part: gfx1250 is hard power-capped
(2497-2502 W of 2500 W, GFX clock held 29% below `MAX_CLK` under sustained bf16
GEMM), and under a cap **removing work pays while merely overlapping work does not**.
Moving a spill reload to LDS deletes scratch traffic rather than hiding it. At 1
wave/SIMD -- which `num_warps=4` at 1024 VGPR/lane forces, and which
`vgpr x new_waves <= 1024` makes unescapable -- there is no second wave to cover an
exposed scratch reload, so each one is latency on the critical path. The 549 vs
2667 TFLOP/s gap is consistent with that being the dominant term.

The Triton-level handles, cheapest first: `TRITON_HIP_USE_ASYNC_COPY` (global->LDS
without a register landing, the direct analogue of AITER's `lds` modifier);
`num_stages > 1` **only if** it produces an LDS-resident double buffer rather than a
register-resident one -- the corpus's repeated failure mode at 1 wave/SIMD is a
register double-buffer that blows the budget; and, structurally, rewriting the K/V
or dO staging in the kernel to go through `tl.load` into an explicitly shared tile.

Two traps recorded against this, both gfx1250-specific: the published LDS swizzle
phase tables are measured on **wave64** and this part is **wave32**, so a swizzle
that looks portable must be re-derived; and one swizzle cannot serve two access
widths (HipKittens carries two buffers in its backward for exactly this reason).
Also, LDS bank-conflict fixes are not reliably worth anything on their own -- the
corpus has 19%->0% moving the clock not at all, twice, and a 0% single-pool layout
losing to a 14% two-pool one. The win being claimed here is **removing the scratch
traffic**, not removing conflicts.

## r3.i4.g15 -- widen the dk/dv epilogue store (open; one free reading gates it)

Not executed. The cheap, separable half of `r1.i7.g7`'s family, and the only candidate
left that survives whichever way `r3.i3.g14` reads, because it removes transactions
rather than cycles.

`backends/hipkittens/gemm/recipes/bf16_gfx1250_ladder.md` rung 10 measures **+12.09%**
[+11.81, +12.37] **on this part** for staging an epilogue through LDS so scattered
`buffer_store_b64` become `global_store_b128` -- against the publisher's own carefully
measured +7.32% [+7.20, +7.43] on theirs, two non-overlapping intervals, i.e. a real
difference between the parts. And s6 a6 of the same card: **it costs no LDS at all**,
because the staging tile aliases the operand ring the K loop has finished with.

The position is analogous: the dk/dv pass ends by writing out 512 accumulator VGPRs
(`dk` 256 + `dv` 256) through `tl.store(DK + ..., dk, mask=mask_kv)` and the matching
`DV` store, and the shipped census reports 128 static `scratch_store`.

**First step is free and has never been taken: grep the shipped `.amdgcn` for the
`global_store_*` widths on the `DK`/`DV` stores.** If Triton already emits `b128` this
item dies for one grep. If it emits `b64` or narrower, the size of the item is whatever
it takes to make the store coalesced -- which in Triton is a layout question, not an LDS
question, so unlike `r1.i7.g7` it may be reachable without leaving the backend.

Ranked below `r1.i7.g7` only because that row targets the larger term; ranked above
everything else because its gate costs nothing.
