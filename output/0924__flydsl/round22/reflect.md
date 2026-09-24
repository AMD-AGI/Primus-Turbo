# Round 22 -- reflect

**Not accepted.** prod 361.7 = 70.0% of its best ever 516.9. `op/current/` still holds round 20.

## What happened

Round 21 left two candidates that were the two halves of one sentence it had written:
*"`k_dkdv` and `k_dq` have opposite signs on the prefetch-spread lever, and the discriminant
is body shape."* Round 22 executed both. **Neither survived, and neither did the sentence.**

- `r22.i2.g68` -- spread `k_dkdv`'s prefetch (the half the model said should *gain*).
  **-30.0% prod / -26.1% proxy**, against same-session floors of 0.40% / 2.08%.
- `r22.i1.g67` -- form 1 died on the offline screen at zero card time; form 2 (the subtractive
  probe, deleting `k_dq`'s prefetch outright) measured a **negative ceiling, -10.86%**.

Spreading the load clump loses on **both** kernels. There is no sign flip.

## What I got wrong

**I inherited a model and shipped the arm that killed it — but I only stopped believing the
model after reading the disassembly, not before proposing the arm.** The "opposite signs"
sentence was written into `facts.md` as the round's headline finding, and `g68` was pooled on
its strength. Reading the two ISA dumps side by side at the start of this round gave a simpler
variable in about ten minutes, and that variable predicted `g68`'s sign and rough magnitude
before any card time. **That reading was available to round 21 for free and nobody took it.**
The cost of not taking it is two rounds and four arms.

**Second misjudgement, smaller and more embarrassing:** I spent the round's one genuinely new
instrument on a hypothesis that could have been killed in thirty seconds. I traced power
because the corpus said gfx1250 sits pinned at a 2500 W cap and `benchmark.py`'s sclk witness
read 1100 at fast but ~1052 at prod, which looked like throttling. The sensor on this box
**does not respond to load at all** — 855.0 W idle, 853 W median across a full prod run. One
idle-vs-busy reading would have shown that before I wrote the sampler.

**What I did right, and would do again:** I wrote the prediction down before the card, and I
declined to build a sixth `sched_group_barrier` arm (family 0-5) and a re-run of `g28`'s
unroll-2 form (three recorded sessions at -7.8%). Reading the ledger is cheaper than
re-measuring it.

## The thing nobody has said out loud yet

The same-session champion re-measure puts `r017`, `r019` and `r020` within **2.2% of each
other at prod** (505.85 / 507.13 / 516.87). The measured null-control floor on this part is
**1.57%**, and `g62`'s banked "+1.65%" sits on top of it. **This operator has not moved
outside the noise since round 17.** Five accepted rounds separate those three numbers. The
next round's bar should be "beat the champion by more than the floor", not "be positive".
