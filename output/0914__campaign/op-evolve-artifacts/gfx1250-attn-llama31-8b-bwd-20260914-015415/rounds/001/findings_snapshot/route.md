<!-- operator-tables: written by the framework, not by the round -->

## Standing constraints -- read every round; never executed, never retired

_None._

## Refactors -- run by the loop at a round's head, one at a time

_None._

## Operator items awaiting a place in the route

Put each of these in the Route table below. **Every `must` one must sort above every `idea` row**, and that is checked after you plan and before you build anything.

_None._

<!-- /operator-tables -->

## Route -- designed by round 1, from this round's reading (nothing was inherited)

| # | type | id | what it is | condition | outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | idea | r1.i2.g2 | in-thread transpose, the largest single win found | none -- must be scoped to the launch, not set process-wide, or it recompiles `beat` too | **DONE, shipped as half of the merge.** Alone 9.4248 ms / 1.238x `beat` (4 runs, 9.4128-9.4248). The scoping held: `beat` read 11.6543-11.6887 ms across 6 runs with and without the knob live. |
| 2 | idea | r1.i1.g1 | `waves_per_eu` hint removed | independent of row 1 and superadditive with it (8.696 ms together vs 9.418 / 9.571 apart), so both are taken and the merge is measured | **DONE, shipped as the other half.** Alone 9.5873 ms / 1.216x. Merge confirmed superadditive: 8.683 ms, 633.2 TF/s, **1.1529x the incumbent re-measured beside it**, 1.340x `beat`. Gate still FAILS (needs 1.50x). |
| 3 | idea | r1.i5.g5 | `BLK_SLICE_FACTOR > 1` | rider only, never the round's candidate -- one sweep, and unlike `BLOCK_N1`/`BLOCK_M2` it is not overwritten by `dense_fused_backward` | NOT REACHED. The round's budget went to the dispatcher use-after-free (see `opt.md` s3 bug 3), which cost four validation runs. Still open, still cheap, still a rider. |
| 4 | idea | r1.i7.g7 | stage the spilled state in LDS -- LDS is at 65,536 of 327,680 B (20%) while 1308 B/lane goes to scratch | take `TRITON_HIP_USE_ASYNC_COPY` and `num_stages` first, and read the resource report each time: at 1 wave/SIMD the way this fails is a REGISTER-resident double buffer, which makes the spill worse, not better | NOT REACHED -- this is round 2's work. Round 1 closed the launch-configuration surface (13.2%); the remaining 10.7% to the gate is here. |

Rows 1 and 2 are the round's two arms. Row 3 is the cheap untested knob left over
from the config table. Row 4 is the only item with the size to reach the 1.50x gate
from 1.24x, and it replaces the earlier `r1.i4.g4` ("shorten live ranges"), which
named the symptom but no mechanism and was blocked on an instrument this round never
got working (`r1.i6.g6`): `rocprofv3` records zero dispatches here. `g7` needs no
instrument, because the static resource report already carries the whole argument --
every backward attention in the corpus runs LDS at 85-100% and spills nothing, this
one runs LDS at 20% and spills 326 VGPRs, and on a hard power-capped part deleting
that traffic pays where overlapping it does not. `r1.i3.g3` (`schedule_hint="attention"`)
is excluded from the table on purpose: it wins alone and loses on top of row 1.
