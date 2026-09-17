<!-- operator-tables: written by the framework, not by the round -->

## Standing constraints -- read every round; never executed, never retired

_None._

## Refactors -- run by the loop at a round's head, one at a time

_None._

## Operator items awaiting a place in the route

Put each of these in the Route table below. **Every `must` one must sort above every `idea` row**, and that is checked after you plan and before you build anything.

_None._

<!-- /operator-tables -->


## Route -- designed by round 1, from the round-1 survey and the corpus (no prior round to inherit from)

There were no operator items, no standing constraints and no refactors outstanding when
this round planned, and no `pool.md` existed -- so every id below was born this round.
There is no deep-born pool entry to pass over.

| # | type | id | what it is | condition | outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | idea | r1.i2.g02 | full 32-deep contraction in `k_dkdv`'s dK/dV GEMM | the WMMA model prices it at 0.667x on 67-83% of runtime; **arm B this round** | |
| 2 | idea | r1.i1.g01 | causal tile skipping in `k_dkdv` and `k_dq` | removes ~half the work, bit-identical; **arm A this round**. Measured well below the 2x its work-removal implies -- read the result together with g03 before concluding anything about causal skipping itself | |
| 3 | idea | r1.i3.g03 | AITER K-tile pairing to load-balance the skipped grid | **only meaningful if g01 ships**; it is the diagnosis of why g01 underdelivered, not an independent idea. Do not build it against an unskipped kernel | |
| 4 | idea | r1.i4.g04 | multi-wave workgroup and a tile bigger than 16x16 | **not a fast-round build** -- both main kernels, their LDS staging and their fragment indexing change together. The ladder ranks it the single largest term on gfx1250 (+81% for the tile alone). Schedule as a large/deep round; do not nibble at it | |

`r1.i5.g05` (the `fast` shape launching 128 workgroups onto 256 CUs) is deliberately NOT a
route row: it is a standing fact about how to READ the `fast` column, not a candidate. A
round that treats a flat `fast` number as evidence against its candidate will draw the
wrong conclusion. It is written up in `pool.md`.
