# Round 3 reflect

**Not accepted. Nothing was built, so there are no figures.** The route's must-row was an
instrument, it would not run, and the two idea rows were closed by their own conditions --
one before the round started spending GPU time, one by the gate it had been given. That is
the route working as designed and it is still a round that produced no code.

## What I got wrong

**I put an instrument in the `must` slot and left the round with no fallback.** The table
made rows 2 and 3 conditional on row 1's reading. When row 1 returned no reading at all --
a case I did not plan for, having only planned for "issue-bound" and "byte-bound" -- both
idea rows were unreachable by their own conditions and the round had nothing to build. A
gate that can return *nothing* needs a branch for that, and mine did not have one. If I had
ordered row 3's free grep first (it cost one command and killed its own row) I would at
least have known before committing the round's shape.

**Both arms I did build in the deciding step were wrong by more than 2x, not marginally.**
I predicted `BLOCK_N2=64` would help because the dq pass measured 971 VGPRs with *zero*
spills when compiled alone; it measured 17.9 ms against 8.68. The error is worth keeping
because it is general: **slack a pass has alone is not slack it has inside the fused
kernel.** One function, one interference graph, 1024 of 1024 VGPRs -- widening the dq step
was paid for in the dk/dv loop, 108% more spills for 43% more wmma. I had derived the
premise from my own half-kernel probe and did not notice that the probe deletes exactly the
thing that makes the premise false.

**I trusted round 2's retraction instead of reading the field myself.** Round 2 wrote into
the pool that this kernel gets no LDS, on the strength of `group_segment_fixed_size: 0` in
the ELF. I repeated it in a rewritten pool entry and in a route condition. It is wrong:
`shared = 65536` in the Triton metadata -- Triton passes the LDS *dynamically*, so the
static field is 0 by construction. Round 1's original 20%-of-320-KB figure was right all
along and I helped bury it for a round. I found this by accident, executing a different
row's gate. Cost of checking: one file read, which neither round 2 nor I did before writing
it down twice.

## What did not work, and is now closed

Stochastic PC sampling does not run on this kernel, and the way it fails identifies the
mechanism rather than a flake -- fine interval, GPU page fault; coarsest interval, no fault
and no samples at all. With counters at 13/51 and kernel-trace already recording zero
dispatches, **gfx1250 offers no instruction-level dynamic instrument for this kernel**. The
next round should route on the static census, dispatch-level `torch.profiler` and deletion
probes, and should not spend a row rediscovering this.

## What the round is actually worth

Three negative results and one correction, none of them a speed-up: the launch/tile surface
is now closed in every direction and no future round should retune a block size; the fusion
of the dq and dk/dv passes *is* the causal load balancer, which prices every unfusing
candidate at -12.5% before it starts; the epilogue is already at maximum store width; and
the LDS premise above is repaired. The last of those is the one that changes what the next
round can attempt.

**One thing I may not fix here.** `r3.i4.g15` (widen the dk/dv epilogue) is in `pool.md`
and is refuted -- its own gate found 96 `buffer_store_b128` and zero `buffer_store_b64`.
I am allowed exactly one removal and it belongs to the executed candidate, so g15 stays in
the pool this round. **The next round should retire it to `dead_ends.md` unread**; the
evidence is in `rounds/003/1-opt/opt.md` s13 and in that entry's own text.
