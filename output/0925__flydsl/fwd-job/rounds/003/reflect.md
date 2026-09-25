# Round 3 reflect (fast) -- not accepted

**What happened.** Built A = r3.i1.g07 (8x `s_prefetch_inst_pc_rel` in the prologue) and B = r3.i2.g08
(LO-wave `s_setprio` over QK), each from op/current. Both compiled first try, identical resources,
output bitwise = incumbent. Neither won: A 0.980 / 1.022 / 0.996 (fast/proxy/prod) of round 2,
score 0.67596 vs 0.67525 (< 0.70% bar); B 0.943 / 0.973 / 1.000. No merge. A left in the working copy.

**What did not work.** g07 is dead on mechanism: the kernel descriptor's INST_PREF_SIZE (216 x 128 B)
already prefetches all 27.6 KB of code at launch, so the explicit prefetch duplicated it. The
post-beat penalty did not shrink (proxy 29 -> 39 us, noise ~10). g08: no stagger gain at prod.

**What I misjudged.**
- The premise of g07. I read "ASM prefetches, FlyDSL does not" off the ASM disassembly and never
  checked FlyDSL's own kernel descriptor before building; the field was visible in the dumped `.s`
  all along. One grep before step 4 would have killed the arm. My expected +35-50% fast in validation
  order was built on that premise; measured ~0.
- The round-2 "per-CU I$ eviction" story, which I took as established. The penalty survives full
  code prefetch, so it is not a cold fetch; its cause is now open.
- Measurement: session 1 showed A +15% at proxy -- pure palindrome-slot bias. Caught only because
  I rotated arm order over 3 sessions; a single session would have shipped a false win. My first
  g03 probe also had a cold "alone" baseline (another arm's binary ran last).

**What the round did produce.** The VALU floor rider: removing softmax exp2 cuts prod 13.4% with
WMMA unchanged -- the first direct evidence that softmax VALU is on the body's critical path.
