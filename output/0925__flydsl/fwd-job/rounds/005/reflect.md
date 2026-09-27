# Round 5 reflection

Round 5 was delivered but not accepted. The selected arm, `r5.i1.g13`, was correct and compiled exactly as
planned, yet it was substantially slower on every scored shape. The incumbent therefore remains the round-4
champion.

## My objections, checked first

My numerical forecast for g13 was wrong. I predicted at least +5% at prod; the planner predicted at least +3%.
The measured result was 0.8485x champion at prod—912.99 versus 1075.98 TFLOP/s—and the clock probe moved from
1.9601 to 2.3585 Mcycles/call, +20.3% (`rounds/005/3-act/act.yaml`). Neither forecast survived.

My warning that static ISA properties cannot price time did survive. The implementation met every static gate:
512 VGPR, zero spill, zero scratch, and 103 of 330 `v_exp` instructions within eight instructions of a WMMA
(`rounds/005/2-plan/raw/vgpr_probe/summary.txt`). Those facts proved that the intended code existed and was safe
to run; they did not predict the 15.15% production regression. The round is another direct example of campaign
correction 3.

My original preference for running the smaller mandatory row-sum arm before the pipeline also identified the
operational risk that materialised. The final dialogue accepted g13 first after its compile risk was discharged,
but act permits only one candidate. Because g13 consumed that slot and lost, `r5.i2.g14` was not built or measured
and h22 remains outstanding (`rounds/005/3-act/act.yaml`). This does not establish that g14 would have won; it
establishes that the plan's promise to measure two separate arms could not be fulfilled by act's one-candidate
contract.

The planner accepted two other review objections before execution. The g14 forecast was internally inconsistent
(a 1-2% cycle range paired with a >=2% throughput floor), so the final plan preserved separate planner and reviewer
predictions. The g15 exposed-or-hidden conditional was two predictions, so the final plan adopted the reviewer's
single null forecast. Neither arm ran, so neither performance prediction was tested.

The objection to retiring `r1.i4.g04` also held. Round 5's matched fast timing found current/beat warm ratios of
1.240 and 1.193, between the agreed <=1.15 stale and >=1.3 keep thresholds
(`rounds/005/2-plan/raw/warm_flush/summary.md`). The evidence is inconclusive for split-KV, so it remains open and
deferred.

## The selected candidate and the prediction

g13 ported the h21 QK(i+1)/softmax(i) software pipeline onto the round-4 champion. It double-buffered score
registers, re-phased the LDS ring to `[K(i+1) | V(i)]`, placed softmax and the next QK in one region, and peeled the
last tile. Both causal GQA2 and GQA4 compiled at 512 VGPR, 97 SGPR, zero spill and zero scratch. The output was
bitwise identical to champion, including the 200/200 determinism check and hashes
`o=26a89a2db0cd`, `lse=d6ac8da1e101`. Validation passed all 16 cases at the specification's 49 dB threshold; the
unit test's seven 49.82-49.99 dB failures were identical to the champion's pre-existing failures against its
hard-coded 50 dB threshold (`rounds/005/3-act/act.md`).

Performance failed cleanly and reproducibly:

| shape | g13 TFLOP/s | champion TFLOP/s | g13/champion |
|---|---:|---:|---:|
| fast | 30.20 | 34.20 | 0.883 |
| proxy | 674.46 | 871.42 | 0.774 |
| prod | 912.99 | 1075.98 | 0.8485 |

The score was 0.59507 against the same-session incumbent's 0.71966. The independent clock probe was stronger
than the benchmark direction: control ran at 1.9601 Mcycles/call, 974 MHz and about 1851 W; g13 ran at
2.3585 Mcycles/call, 990 MHz and about 1740 W. The candidate therefore spent 20.3% more cycles while receiving a
1.7% higher clock and drawing roughly 110 W less. This is not a clock or socket-power loss
(`rounds/005/3-act/act.yaml`). The result is the `slower_failed` cell: both the speed and claimed mechanism failed.

## What the failure means

The measured conclusion is narrow and strong: this implementation of L6 at `n_block=64` is dead. Double-buffering
S in one basic block while re-phasing the K/V LDS ring did not hide softmax; it added a large cycle cost. The ISA
contains more `s_wait_dscnt` sites (60 -> 112) and barriers (12 -> 14), so the likely explanation is that the
compiler serialized softmax behind the new LDS dependencies and had no scheduling room at the 512-VGPR cap
(`rounds/005/3-act/act.md`). That causal explanation is not measured. This card rejects stall counters, PC sampling
is forbidden, and ATT does not capture the FlyDSL kernel, so no dynamic stall attribution exists.

The result does not overturn the established softmax evidence. Round 3's exp-removal probe and round 4's packed
softmax wins still show that softmax work is on the critical path. Round 5 only shows that this cross-tile
double-buffering route fails to overlap it. A future overlap proposal must avoid the same LDS dependency structure
or first free register budget, and it must receive a new candidate id.

## The fast-path measurement

Round 5 also took the matched warm/flushed measurement requested in plan. It refuted the profiler's cold-L2
reading: current was 74.71 us with warm L2 and 75.47 us after an L2 flush; beat was 34.49 and 34.77 us. Host launch
was also equal at 16.43 versus 16.60 us. What moved current was idle interval: 37-38 us back-to-back, about 42 us
after a short sleep, and 71-75 us after about 1.8 ms idle. Beat stayed at 32-35 us
(`rounds/005/2-plan/raw/warm_flush/summary.md`).

Thus roughly 36 us of the scored fast gap is a current-only, idle-sensitive device-side dispatch penalty, while
the remaining warm kernel gap is about 6-7 us. The mechanism remains unknown. Instruction-cache loss or idle-CU
power gating is plausible but unverified; explicit instruction prefetch is already dead because the descriptor
prefetches the whole kernel. `r1.i3.g03` remains the diagnostic priority for fast, and `r1.i4.g04` remains open but
behind it.

## Disagreements and carry-forward

The g13 disagreement is closed with verdict `neither`: the planner's >=3% and reviewer's >=5% predictions both
failed. The implementation goes to `dead_ends.md`.

The g14 disagreement remains because act did not build it. It is the highest-priority carry: measure it alone from
the champion, with the planner's +1-2% and reviewer's >=2% predictions kept separate. Below the session's prod
noise floor closes the per-tile cross-lane-reduction mechanism.

g15 remains an optional closure arm with a null prediction and its wait/spill/clock gates. g04 remains conditional:
the agreed warm-time thresholds were not crossed, so its eventual decision requires controlling g03's penalty or a
direct deterministic split-KV measurement.

## Memory rewrite

- Moved `r5.i1.g13` from pool to dead ends on round-5 measurements.
- Moved stale `r1.i2.g02`, `r2.i2.g06` and already-closed `r3.i2.g08` from pool to dead ends, preserving their
  original measured or analytical reasons and recording the round-5 consolidation.
- Moved shipped `r4.i2.g10` from pool to facts; it has been part of the champion since round 4.
- Kept `r5.i2.g14`, `r5.i3.g15`, `r1.i3.g03` and `r1.i4.g04` in the pool.
- Replaced the bottleneck section with the round-5 regime and removed superseded round-by-round bottleneck prose.
- Did not edit `route.md`.
