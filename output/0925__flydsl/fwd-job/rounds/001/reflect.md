# Round 1 reflect (fast)

**Rejected** -- on the 7 edge-case precision misses (49.82-49.99 dB < 50) that op/current fails
identically. The speed change was never the question.

What happened: r1.i1.g01 (longest-first dispatch remap) is a real, large win -- proxy 1.48-1.60x,
prod 1.09-1.11x vs round 0 in the same session, fast neutral in a controlled probe. r1.i2.g02
(O_VARIANT v1) lost at proxy in both sessions (-3/-4%), prod +0.9/+1.6% inside spread; the
merge was worse than g01 alone at proxy (-7.9%). Shipped g01 alone.

What I got wrong:
- **I spent the round on speed knowing the gate could not pass.** op_setup_v000.md said, before
  round 1 started, that the baseline misses 50 dB on 7 cases and that clearing it "needs an
  operator decision". I read that mid-build as "pre-existing, not mine" and carried on. True, but
  it made acceptance impossible from the start; the round should have raised it as the blocker
  at planning, not in the final report.
- **Under-predicted g01**: said proxy +20-40%, prod +3-6%; measured +48-60% and +8.6-10.7%. I
  discounted the model's upper bound for non-FIFO dispatch; at proxy it held exactly, at prod it
  was EXCEEDED -- so the model is also missing a term (non-linear tile cost or L2 reuse).
- **Fast**: benchmark.py's fast median reads g01 at 0.92-0.93 of ctrl, which would also have
  tripped the 95%-of-best rule. The controlled probe says the kernel is neutral; the median is
  the post-ASM penalty landing by arm position. The harness, not the kernel, sets fast's number.
- g02's prior +1.90% (flydsl 0.3.2, different harness) did not reproduce as a net win.
