# Round 12 — bwd QKV A2AGemm: the last `wgrad_late` wgrad tiles after every dgrad tile (ROLLED BACK)

1. Hypothesis (from R10 / R11 traces): the 160 wgrad tiles claimed first hold their CUs until the
   last push at ~45 % busy, leaving ~80 CUs for dgrad, which needs ~95 to keep up with arrivals;
   the backlog starts at the last push and adds a whole tile (~77 us). Claiming some wgrad tiles
   after the dgrad tiles gives dgrad more CUs during the exchange.
2. Change (snapshot in `kernel_snapshot/`): `wgrad_late`: claim order wgrad[0, 160 - late) | dgrad |
   wgrad[160 - late, 160).
3. Results (bwd QKV, `e9_wgrad.py`, us): late 0 / 20 / 40 / 60 / 80 = 16K 303 / 352 / 357 / 354 / 356,
   32K 513 / 667 / 661 / 669 / 688. Correct (5e-5).
   Trace 16K late 40: push ends 203-212; dgrad ends 284 (R10 ~300: no backlog, last tile = push end
   + ~80 us); the late wgrad tiles start at 203-215 (after the last dgrad claim) and run 115-134 us
   unpaced -> kernel end 337-344.
4. Decision: ROLLED BACK.

## Rollback Analysis
- Change made: move part of the wgrad tiles behind the dgrad tiles in claim order.
- Expected effect: no dgrad backlog, end ~ push end + one dgrad tile.
- Actual effect: dgrad as expected (284 vs ~300), but a wgrad tile claimed at the end runs 115-134
  us under contention: a new, longer tail.
- Root cause: every wgrad tile needs all of M; claimed late it is a full-length tile after the last
  push. Claimed early it is paced by arrival (finishes ~at the last push) but wastes its CU.
- Lesson learned: in step the floor is push end + one 128 x 256 x 5120 dgrad tile (~77 us):
  16K ~280 us (82 % hidden), 32K ~480 us (~97 %). Going below needs a K walk by arrival (staggered
  sources) for the last m-tiles and a CU budget where the paced wgrad tiles leave ~95 CUs to dgrad
  through the last ~80 us; split-K through memory (R11) and late wgrad (R12) both fail.

## Real-training transfer check
- Bucket: K1 (claim order); no cache. Benchmark gain: negative. Decision: REJECT (rolled back).
