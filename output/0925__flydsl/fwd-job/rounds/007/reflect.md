# Round 7 reflect

**Not accepted.** `op/current` still holds round 4. The framework's validation scored geomean vs beat at 0.6724
(fast 0.5427, proxy 0.7471, prod 0.7498).

## What happened

- **Look phase:**
  - Probe r7.i1.g17: removing the per-tile barrier, a timing-only probe, bought +4.5 to +5.2% at prod.
  - Probe r7.i2.g18 (L20 branch-free rescale) lost 8.3%, which closed h22.
- **Built:** h24, the barriers proto. p22n and p22s were each applied onto the champion.
  - Correctness was clean and bitwise equal to the champion.
- **Both arms lost at prod** in 5/5 same-session runs:
  - p22n -6%.
  - p22s -3.5%. p22s is left in the working copy.

## What did not work, and what I misjudged

- I predicted p22n +1.5% and the best arm about +2%. **I was wrong in sign.**
- I read the proto's REPORT ("ready-for-card") and its compile-only VGPR/spill numbers as enough. I never ran a
  per-loop instruction census on the proto before spending the card on it.
  - That census was available for free, and I had run it on the champion in the same round.
  - It would have shown the runtime `tile % G` branch and the +10% instructions per tile (553 -> 606).
  - Given the 4.7% ceiling I had just measured, it would have predicted the loss.
- I treated "barrier count halved" as the whole effect of h24. The cost side, which is ring bookkeeping on the
  serial path, was not in my model.
- One surprise the other way: split signal/wait was worth about 3 points over no-split. I had expected +0 to +1%.
- `ut` still fails on its hard-coded 50 dB for the champion too. This is an open gap between `ut` and the spec's
  49 dB, now seen in 3 rounds.

## Memory

- The executed candidate h24 was an operator hint, not a pool entry, so no pool id moves.
  - Its mechanism is appended to `dead_ends.md` under a heading with no id.
- `r7.i3.g19` stays in the pool. It was not built, and its premise (phase offset) is not refuted.
  - Its prerequisite of a depth-3 ring will not come from h24 now.
  - Its prediction is anchored "over whichever h24 arm wins", and no arm won. Read it against the champion.
- `facts.md` has a round-7 bottleneck paragraph appended. Confidence is low.
