# Round 11 -- reflect (fast round, accepted, score 0.90551)

**What happened.**
- I built both route rows apart from each other: A = r11.i1.g30, the shape-gated R=1 kernel; B = h29,
  nodelay.
- Neither lost, so I built and measured the merge and shipped it.
- Against `rounds/004/op`, 3 sessions, one image per process: fast 1.186, proxy 1.0037, prod 1.0083,
  geomean 1.0627.
- validation.py passed correctness (16/16) and determinism. It still fails speed vs beat (0.7735),
  like every earlier round.

**What I misjudged.**
- **Size of the win.** I predicted a merge of ~+3.3% and a fast time of 28-32 us. I measured +6.3%
  and 32.5-34.5 us.
  - The geomean came out better only because I had anchored "fast +10%" on p1's low session. Six
    sessions put fast at 1.17-1.23x.
  - In absolute time, the per-tile-body model over-predicted. Loop length fell 29%, time fell 16%.
    The per-WG fixed cost is bigger than I assumed, and nothing was profiled to say what it is.
- **p1's 12% session spread.** I blamed the first run after a cache clear. That was never tested.
  In this round's six sessions the spread was about 6%.
- **Same-process pairing with beat.** I expected it might swallow the fast gain. It did not; the
  ordering held. But every arm's fast time is about 1.3-1.6x worse there, and nobody knows why.

**What did not work.**
- Nodelay alone is still only +0.74% at prod and flat elsewhere, the same as round 10. It is worth
  having only when carried by a lever on another shape.

**What was left thin.**
- short_q now runs R=1 at o 49.82 dB, 0.8 dB above the gate, with no R=2 figure beside it.
- The gate threshold (grid < CU count) was chosen, not swept. Proxy at R=1 is unpriced.
- The facts entry saying `ut/test_correctness.py` uses 50 dB looks stale: this round's ut printed
  "gate 49.0 dB". I did not dig into it.
