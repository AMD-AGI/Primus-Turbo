<!-- operator-tables: written by the framework, not by the round -->

## Standing constraints -- read every round; never executed, never retired

_None._

## Refactors -- run by the loop at a round's head, one at a time

_None._

## Operator items awaiting a place in the route

Put each of these in the Route table below. **Every `must` one must sort above every `idea` row**, and that is checked after you plan and before you build anything.

_None._

<!-- /operator-tables -->

## Route -- designed by round 3, after it closed the config surface and both of its arms lost

| # | type | id | what it is | condition | outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | must | r3.i3.g14 | stochastic PC sampling / ATT on `bwd_kernel_causal`, read only to decide whether the dk/dv loop is **issue-bound or byte-bound** | gates rows 2-3 and costs no code change. It is required because the two remaining families are priced differently and the cheap proxy is gone: gfx1250 is hard power-capped (2497-2502 W of 2500, clock held 29% under `MAX_CLK`), and under a cap bytes price at full value while cycles/instructions price at **20-45%**. `facts.md`'s "37% of the dk/dv body is overhead" is an instruction-count figure dominated by 363 `s_set_vgpr_msb` and 84 `v_nop`, which switch almost nothing -- so that 37% is 6-14% of time, not 37%, and the energy-valued part is the 77 `scratch_load`/iter. Counters are dead here (13/51 non-zero, no byte counter, WMMA FLOP counters exactly 0); `arch/gfx1250/profiling-surface.md` gives the verbatim commands and the discriminating signature (WMMA-bound: 76.5% `ARBITER_NOT_WIN`, 82.8% issue, mean `Wave_Count` 1.4 -- streaming: 49.7% `WAITCNT`, 97.4% `NO_INST`, 2.2% issue). **If it reads issue-bound, row 2 is not worth its size and row 3 is the round.** |**ABANDONED at `look`, and it closes the last instrument on this part.** Two attempts, two different failures, and the pair is the mechanism. Interval 2048 cycles: GPU page fault inside the sampled process -- `Memory access fault by GPU node-3 ... Reason: Page not present or supervisor privilege`, then `GPU core dump skipped because PC Sampling active`, `rocprofv3 caught signal 6`, and a hang in finalisation (`Timeout while waiting for queue sync: 1 kernels still active`). Interval 1048576, the coarsest the part accepts: no fault and NO samples -- alive 13 minutes, GPU[1] at 0% use, no CSV, output unmodified since tool init. Sample often and the interrupt faults on the save area; sample rarely and the queue never drains. The kernel is resident at 1024 of 1024 VGPRs with 1064 B/lane of scratch on one wave per SIMD, which is the worst case for a path that must save wave state to take a sample -- same family as `--stats --kernel-trace` exiting 0 with zero dispatches. Device verified healthy after both. **There is no instruction-level dynamic instrument on gfx1250**: counters 13/51 with no byte counter, kernel trace dead, PC sampling and ATT dead. Do not spend another row finding this out. Raw in `rounds/003/1-opt/raw/pcsample_attempt{1,2}.txt`. |
| 2 | idea | r1.i7.g7 | get the dk/dv pass's `k` 128 + `v` 128 loop-invariant VGPRs out of the register file and into LDS, **at constant Q/dO traffic** | pass row 1 as byte-bound first. This is the only surviving structural item and the premise is now the corpus mechanism rather than round 1's void "LDS is at 20%" reading: FlyDSL `hd128.md` s6 b1 ships `k_reg=False`/`kv_halves=2` at exactly D=128 (and keeps V in registers at D=64, so it is a D=128 decision); ladder rung 02 measures **+44.78% on this part** for landing operands in LDS without a VGPR staging; rung 10 measures **+12.09% on this part** for an LDS-staged epilogue that costs no LDS. It targets the half of our 768 that is NOT an accumulator, which is the exact open form `dead_ends.md` left after `r2.i4.g11`. **The blocker is that Triton cannot express it** -- the shipped build reports `group_segment_fixed_size: 0`, `TRITON_HIP_USE_ASYNC_COPY` is on by default for gfx1250 and emits a byte-identical binary at `num_stages=1`, and `num_stages=2` is 12.2-12.9 ms. So this row is a backend decision, not an edit, and it should not be opened until row 1 says the bytes are worth it |NOT OPENED, correctly -- row 1 returned no reading and this row was explicitly gated on it reading byte-bound, so building the round's largest structural item would have meant acting on an inference the row forbade. **But its stated blocker is now FALSE, found while executing row 3's gate.** The Triton metadata beside the `.amdgcn` reports `shared = 65536`: `group_segment_fixed_size: 0` is 0 because Triton passes this kernel's LDS **dynamically at launch**, not because it allocates none. The kernel uses 65,536 of 327,680 B (20%) -- exactly round 1's original figure, which round 2 retracted on a misreading of the static ELF field, and which also explains the 656 `ds_*` ops the census has always shown. So `r1.i7.g7`'s headroom argument is RESTORED with ~262 KB free, and the real gap is smaller and different: Triton does use LDS here, it just offers no surface to say *what* lives there across the m-loop. Restate the item as a layout/staging question before sizing it as a rewrite. |
| 3 | idea | r3.i4.g15 | widen the dk/dv epilogue: read the actual global store widths for `DK`/`DV` out of the shipped `.amdgcn` first, and only then decide whether there is a `b64` -> `b128` coalescing win there | the cheap half of row 2's family, separable from it, and it survives row 1 reading EITHER way because it removes transactions rather than cycles. Ladder rung 10 measured **+12.09%** here against the publisher's own +7.32% on their part -- two non-overlapping intervals, so it is a real gfx1250 difference -- for exactly this change in exactly this position, a pass ending with 512 accumulator VGPRs to write out. **Reading the store widths is free and has never been done**; if they are already `b128` the row costs one grep and dies, which is why it is ranked below row 2 rather than above it |**DEAD at `look`, for one grep, exactly as the condition intended.** The shipped `bwd_kernel_causal.amdgcn` emits **96 `buffer_store_b128` and ZERO `buffer_store_b64`** (also 160 `ds_store_b128`, 46 `scratch_store_b128`, 82 `scratch_store_b32`). Every global store is already at maximum width, so corpus rung 10's `b64 -> b128` coalescing mechanism (+12.09% measured on this part) has nothing here to convert. No GPU time spent. |

Rows 1 and 2 are the round's two arms; row 1 is an instrument and a gate rather than a
competitor, so the GPU budget goes to rows 2 and 3.

**Round 3 shipped nothing, and the route is rewritten around what it closed.** Its two
arms were the last two untested directions on the launch/tile surface, and both lost by
more than 2x: `r3.i1.g12` (`BLOCK_N2` upward, 64 -> 17.865 ms, 128 -> 24.345 ms) and
`r3.i2.g13` (tile above 256, 512 -> 59.099 ms, and 192/384 are illegal because Triton
block shapes must be powers of two). With rounds 1-2 that makes **every axis of
`_DEFAULT_ONEKERNEL_CONFIG` measured in both directions, all losing** -- see the closed-
surface table in `dead_ends.md`. `matrix_instr_nonkdim` and `kpack` emit byte-identical
binaries. So no row of this table is a retune, and none can be.

Three things round 3 established that this table is built on, all in `dead_ends.md`:

- **The fusion is the causal load balancer.** One pid does dk/dv for K-tile `pid` and dq
  for Q-tile `pid`, and the two are exact complements: every program does 264 trips. That
  is AITER's "pair K-tiles from both ends", already shipped. Measured cost of losing it:
  the halves sum to 9.764 ms against 8.678 fused. **Any candidate that unfuses starts
  12.5% down**, and it buys no occupancy either, because the dq pass compiled alone is
  `vgpr_count 971`, not <= 512.
- **Slack a pass has alone is not slack it has in the fused kernel.** The dq pass alone
  is 971 VGPRs with ZERO spills; widening its loop step inside the fused kernel cost 108%
  more spills, paid in the dk/dv loop. One function, one interference graph, 1024 of 1024.
- **`vgpr_spill_count` is not a proxy for time across anything but a fixed tile and a
  fixed addressing mode.** Round 2 left `r2.i3.g10`'s MEASURE rule standing after the
  tile counter-example; round 3 broke it at a FIXED tile too --
  `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1` compiles the same source, same tile, same launch
  to **187 spills, the lowest this job has ever produced**, and runs at 15.839 ms against
  8.679, **+82%**. `AMDGCN_USE_BUFFER_OPS=0` is a second such point. Both change
  addressing, not pressure. The DISCARD half of the gate survives as a heuristic; the
  MEASURE half does not, and no row here is justified by a spill count alone.

Items deliberately not in the table. `r1.i3.g3` (`schedule_hint`) -- still excluded, still
anti-synergistic with the shipped `g2` (8.941 vs 8.696 on top of it), and round 3's power
finding demotes it further: it is a pure scheduling change, so it prices at 20-45% even
if it stopped losing. `r1.i4.g4` -- refuted in round 2, and round 3 adds nothing that
rehabilitates it. The AITER causal-pairing item (corpus B9) -- **not takeable, already
shipped**, see above. Per-product WMMA shape selection (corpus B2, the only corpus
mechanism with a measured number on a dK/dV body, +8.7%) -- **unreachable from Triton on
this part**: `matrix_instr_nonkdim=32` emits a byte-identical `.amdgcn`, gfx1250 wmma v3
bf16 is `v_wmma_f32_16x16x32_bf16` and Triton emits it regardless. `-mllvm` pass-through
-- closed twice: Triton 3.6.0's `make_amdgcn` passes `flags = []`, and `gqa_d64.md`
records four such flags that **silently miscompile** (same VGPR/spill report, no warning,
SQNR 52 dB -> 0.6 dB with ~18% NaN rows).
