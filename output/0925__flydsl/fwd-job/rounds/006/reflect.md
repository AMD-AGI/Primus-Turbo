# Round 6 reflect (fast)

**Not accepted.** `op/current` stays at round 4. The framework's run measured geomean 0.6877 (fast 0.5555,
proxy 0.7746, prod 0.7560).

## What happened

- **Look:** spent most of the budget on the "idle" fast penalty (g03).
  - It is not idle. It appears only after beat has run in the process, and it comes with I-cache misses
    0 -> 808.
  - It also costs proxy +31% and prod +1.7%.
  - That made g04 stale and filed g16.
- **Row 1, h22 -> r5.i2.g14:**
  - Built cleanly on the first compile: permlanex16 16 -> 10, 0 spill, same dB as champion.
  - Lost at prod: 0.981 / 0.985 / 0.982 in 3 arm orders.
- **Row 2, r6.i1.g16:**
  - The discriminator closed it without an arm. A 1.39x larger build (r5 g13) paid only +2.9 us more penalty.

## What I got wrong

- **g14's prediction:** I carried +1% (planner +1-2%) and it measured about -1.8%.
  - Removing a permlane from a chain is only worth something if that chain is critical. It was not.
  - I also did not account for the register-allocation shuffle (`s_set_vgpr_msb` 240 -> 322), which is the one
    visible static cost.
  - Round 5's warning, "static counts are gates, not prices", applied here and I leaned on the permlane count
    anyway.
- **g16's mechanism:** I wrote a code-size refetch story with a matching arithmetic (216 lines x 140 ns = 30 us)
  and predicted proxy +5-8% from shrinking the image.
  - The arithmetic fit was a coincidence; the scaling test says no.
  - The miss counter moves, but what it measures is not proportional to image size. Plausible alternatives:
    1-of-8 XCD sampling, first-touch of a fixed set of lines, or state beat leaves behind.
  - Filing g16 before measuring the cheap discriminator was premature. The discriminator took under 3 minutes
    and could have run in the look step.
- **Formal score vs my own run:** fast 0.592 in my run vs 0.5555 in the framework's. That is the bimodal fast
  median at work, the thing I had just documented.

## Memory changes

- **Moved:** r5.i2.g14, pool -> dead_ends. The mechanism is half measured: off the critical path is measured; the
  loss's cause is a symptom.
- **Also recorded in dead_ends:** the g16 discriminator result, under a non-id heading. The g16 pool entry was
  not moved, because only one removal is allowed. The next reflect should retire it.
- **facts:** the current-bottleneck section got a round-6 paragraph correcting the "idle-sensitive" reading.
