# Round 2 -- reflect

**Not accepted.** Shipped `r2.i2.g9` at 591.6 TF/s on `b4_s8192_hq32_hkv8_d128`
against a best-ever 633.1 -- 93.4%, under the 95% floor. `op/current/` keeps round 1.

## What happened

Two arms, built from the same base, measured apart, one process each.

| arm | what it deleted | spills | bwd ms | vs base |
| --- | --- | --- | --- | --- |
| base (round 1) | -- | **266** | 8.691 | -- |
| `r2.i1.g8` fold `sm_scale` into `exp2` | 32 `v_pk_mul_f32`, exactly as designed | 307 | 9.347 | **+7.5%** |
| `r2.i2.g9` invariant offsets + scalar cursor | pointer `iter_args` | 325 | 9.302 | **+7.0%** |
| null control (byte-identical copy) | -- | 266 | 8.691 | +0.0% |

No merge -- the rule is merge only if neither lost, and both lost. Shipped the better
of the two, per "ship the best arm even when all of them lost". Correctness passes
(52.04 dB min, `op/validation.py`); the round is still 16.4% short of the `beat` gate.

Route row 2 (`r2.i4.g11`) was killed at `look`, before building, by the instrument the
same round delivered.

## What did not work, and why

Both arms did exactly what their source-level claim said and both got slower, by
almost the same amount. That is not two coincidences, it is one mechanism:
**at 1024/1024 VGPRs and one wave per SIMD, a source-level deletion is a request to
the register allocator, and what gets timed is the allocator's answer.** g8 traded 32
VALU for 41 spill slots reloaded every iteration; g9's hoisted offset tensors became
long-lived spilled values and its `v_add` count went *up*. Mechanism and numbers are
in `dead_ends.md`.

## What I got wrong

1. **The prediction, in sign and size.** I wrote "roughly 2-3% each, and the merge, if
   both survive, might reach 4-6%". Measured -7.5% and -7.0% against a 0.03-0.06%
   floor. I reasoned about the instructions I was deleting and never about the
   allocator's response to deleting them -- on this kernel that is the only term that
   matters, and I had round 1's `num_warps=8` result (2.36x slower) telling me so.

2. **I over-generalised my own law the same round I wrote it.** Step 4's
   "time tracks `vgpr_spill_count`, and nothing else does" rests on four points that
   are *all* at `BLOCK_N1=256`. Running the new gate at `BLOCK_N1=128` returned
   `vgpr_spill_count=0`, `vgpr_count=771`, zero scratch -- a perfect score -- for a
   tile round 1 had already timed at 11.8-13.3 ms. The law is real **within a tile**
   and empty across tiles. Corrected in `facts.md` under `r2.i3.g10`; use the gate as
   a screen, never as an objective.

3. **I proposed g11 without reading the inner loop carefully enough.** I claimed
   768 -> 384 for +25% wmma. `HEAD_DIM` is the *contraction* axis of `qkT` and `dpT`,
   so `k` and `v` stay fully resident: 768 -> 512 for +50% wmma and 2x Q/dO traffic.
   `BLOCK_N1=128` is strictly better at the same trade and already loses by 20-33%.
   I put a dominated idea in the pool at step 4 and only caught it at step 5.

The two corrections both came from free instruments, not from GPU time. That is the
one thing the round got right: three GPU measurements, two refutations for nothing.

## What this leaves round 3

The traffic-for-residency family is closed by measurement -- every way of lowering
dk/dv residency by re-reading Q/dO loses, whether it spills or not. What is open is
**lowering residency at constant Q/dO traffic, inside one pass over the m-loop.**
Nothing currently in the pool does that. Gate it with `census_gate.py` first.
