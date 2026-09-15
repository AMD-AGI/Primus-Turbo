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
