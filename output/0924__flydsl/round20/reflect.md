# Round 20 -- reflect

Accepted. Score 0.85104. `op/current/` is now `r20.i1.g61 + r20.i2.g62`.

## What happened

Two arms, built separately from `op/current`, measured apart, merged because neither lost.

- **`r20.i2.g62`** -- prefetch depth 1 -> 2 in `k_dkdv`. prod **+1.65%**, reproduced **+1.53%**,
  **+2.00%** against a re-measured r017 in the champion session. `s_wait_loadcnt` 9 -> 2,
  `v_nop` 67 -> 40, VGPR 724 -> 912, spill 0. This is the round.
- **`r20.i1.g61`** -- S/P WMMA chain split 4 -> 2x2. **NULL** (+0.27% prod inside a 0.29% floor).
- **Merge** -- 904 VGPR, *lower* than g62 alone. prod 511.42 / proxy 435.93 / fast 55.07.
- Six builds, zero build failures, zero spills, `op/ut/` 15/15 at 52.30-52.98 dB.

Before any of that, three throwaway probes (no `g`) killed the hypothesis the round was
supposed to test. Details in `../1-opt/opt.md` §3.

## What did not work

`r20.i1.g61` bought nothing. The corpus said 4-way chain splitting is *"a direct win ... +8.7%
on a dK/dV body"*; it did not transfer. I shipped it anyway only because the two-arm merge rule
required it and the merge's register allocation came out better. **It is not evidence of
anything except that the matrix pipe is idle-waiting, not dependency-stalled.** Twenty rounds
from now, the useful sentence is: the ILP axis on this kernel is closed, tested, null.

## What I got wrong

**I inherited round 19's LDS-port story and set out to test it as though it were the leading
explanation. It was wrong, and three probes were enough to show it.** Deleting *all 80* LDS ops
from the hot body, WMMA count unchanged, is **8.19% slower**. So the 1.25-LDS-ops-per-WMMA
census that made the story attractive was measuring cover, not cost -- an artifact of reading a
static instruction mix as if it were a cost model, which is precisely what round 19's own
facts.md warned the next round not to do. I did it anyway, in a different coordinate system.

The second misjudgement is smaller and structural: my first `## Route` table packed seven
operator items into four rows. The operator rejected it. Packing read as condensing and was
diluting -- each of those ids was a separate must-do, and folding them hid which ones the round
was actually answering.

The thing that saved the round was not insight, it was that **probes are cheap and hypotheses
are not**. The probes cost ~20 minutes of card time and moved the round from "test a story I
believed" to "the story is false and here is the sign of the real lever".

## What the next round inherits

The sign of the lever: **add independent work, lengthen load-to-use distance, do not remove
anything.** Six deletions in a row have lost. `r21.i1.g63` (same prefetch edit on `k_dq`, which
is ~37% of prod time) is the direct continuation, with an explicit-clamp correctness gate and a
960-VGPR spill gate that may well kill it. `r21.i2.g64` is items 3 and 4 of the pipelining
list, never tried in twenty rounds, budgeted as a coin flip.

Not a candidate: depth 3. `s_wait_loadcnt` is at 2; there is no third wait to move.
