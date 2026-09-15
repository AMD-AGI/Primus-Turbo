# Pool

Ideas worth keeping, with the evidence behind each. `g` is monotonic across the job
and never reused; the highest `g` allocated by this job is 7.

`r1.i1.g1` and `r1.i2.g2` were executed in round 1 and have moved to `facts.md`.

## r1.i3.g3 -- `schedule_hint="attention"` (open; anti-synergistic with g2)

Not executed. Alone it measures 9.633 ms (-3.6%) and with g1 9.526 ms, both real.
But **on top of g2 it loses**: 8.941 ms against 8.696 ms without it. Two schedulers
arguing about the same instruction stream. Keep it for the case where g2 is
withdrawn or the kernel is restructured; do not stack it on g2 as it stands.
`"memory-bound-attention"` (9.598 ms) and `"attention,memory-bound-attention"`
(10.032 ms) are the same family and subsumed.

## r1.i4.g4 -- shorten live ranges in `bwd_kernel_causal` to escape the 326-VGPR spill

Not executed -- this is the structural item, and it is the only one with the size to
reach the gate. Static census: `vgpr_count 1024`, `vgpr_spill_count 326`,
`sgpr_spill_count 65`, `private_segment_fixed_size 1308` B/lane, 1 wave/SIMD. The
kernel achieves 549 TFLOP/s where bf16 GEMM on this chip reaches 2236-2667.

What is NOT the answer, measured this round: more warps (`num_warps=8` is 2.36x
SLOWER, not faster), smaller or larger tiles, `num_stages=2`. The tile/warp surface
is a sharp local optimum and every neighbour is worse. So the fix has to be inside
the kernel body -- recompute instead of keeping a tile live, or split the dq phase
from the dk/dv phase's register set -- not in the launch config.

## r1.i5.g5 -- `BLK_SLICE_FACTOR > 1` (untested knob, cheap)

Not executed and not measured this round. `_DEFAULT_ONEKERNEL_CONFIG` pins
`BLK_SLICE_FACTOR: 1`; it slices the inner block and is the one config field that
plausibly shortens live ranges without changing the tile shape the rest of the
surface likes. Unlike `BLOCK_N1`/`BLOCK_M2` it is not overwritten by
`dense_fused_backward`, so the env tune knob actually reaches it. One sweep.

## r1.i6.g6 -- get an instrument that says WHERE inside the kernel the time goes

Not executed. `rocprofv3 --stats --kernel-trace` records **zero dispatches** on this
job's benchmark (see `rounds/001/1-opt/opt.md`), so every candidate above g4 is
currently being chosen from a static census and end-to-end times. gfx1250 has only
13/51 counters non-zero and no byte counters at any level, but PC sampling
(stochastic) and ATT are reported to work. Until one of them runs, g4 is a guess
about which live range to attack.

## r1.i7.g7 -- stage the spilled state in LDS: 262 KB of LDS is sitting unused

Not executed; the structural item, and the one the corpus points at hardest.

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
