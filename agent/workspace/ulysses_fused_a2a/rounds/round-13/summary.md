# Round 13 — bwd QKV A2AGemm: block-local staggered sources + K walk by arrival (ROLLED BACK)

1. Hypothesis (R11 / R12): the in-step tail is one whole dgrad tile (~77 us) after the last push.
   Staggering the sources inside small blocks of `n * D` m-tiles (rank `r` rotates its block walk by
   `r * D`) lets every dgrad tile walk K by arrival (`_arrival_k_seg`), so only the last quarter of K
   is left after the last push, while each source's rows stay close to row order for the wgrad M walk.
2. Change (snapshot in `kernel_snapshot/`): `stagger` (D) and `stagger_block` consts; push walk,
   `_arrival_k_seg.first` and `_claimed_tile` block-local; wrapper `stagger=None` (auto = previous).
3. Results (bwd QKV us, `e9_wgrad.py`; cfg `w,at,BMw,BM,push,D,push_split`; logs `sweep_*.log`):
   - 16K baseline (BM 128 in step) 301-303.
   - BM 256: in step 327, D=2 356, D=1 380 (push_split 8); a 256-row dgrad tile takes ~155 us
     under load: twice the 128-row tile, no gain in CU-us.
   - BM 128 push_split 8: D=1/2/4 371-379: 16-row push units with a flag each stretch the push
     from ~205 to ~286 us.
   - BM 128 push_split 1/2: D=1 304, D=2 307-310, D=4 308.
   - wgrad_at with D=1: 32-128 302-306 (all still claimed at t=0), 160 306, 200 329, 240 351.
   - 32K: baseline 515; D=1 518, D=2 520-530, D=4 531.
   Correct throughout (4e-5 / 5e-5).
4. Trace 16K D=1 push_split 1: push ends 199-211; after its last segment a dgrad tile takes only
   17-28 us (was ~77), but ~10 % of dgrad tiles are claimed at 190-220 us, when a CU frees, and run
   their whole K after the last push -> end 287-306. wgrad tiles end 200-250.
   wgrad_at 160: the wgrad tiles claimed at 116-152 still run 135-160 us (contended) -> end 270-300.
5. Decision: ROLLED BACK (no gain at 16K or 32K).

## Rollback Analysis
- Change made: block-local stagger so dgrad tiles walk K by arrival.
- Expected effect: tail = last K segment (~20 us) -> 16K ~240-270 us.
- Actual effect: the per-tile tail drops to ~20 us as planned, but the kernel end is unchanged: the
  dgrad tiles that only get a CU at the last push still run a whole tile.
- Root cause: CU budget, not per-tile latency. 160 wgrad tiles span all of M and are pinned for the
  whole exchange at ~45 % busy; 16 pushers; dgrad needs ~96 CUs (256 x ~77 us over ~205 us) but gets
  ~80. A wgrad tile claimed later runs 120-160 us under load (R12, R13 wgrad_at), so it cannot be
  moved off the exchange window either.
- Lesson learned: in this kernel the bwd QKV tail is set by CU-us, not by the K-walk order. It only
  shrinks if the dgrad tile gets cheaper under load (77 us = 4.4 TFLOP/s per CU, ~45 % of MFMA peak)
  or the wgrad tiles stop holding CUs idle (needs more than one block per CU or a smaller wgrad
  accumulator footprint; LDS 144 KB and 128 acc VGPRs per lane currently allow neither).

## Real-training transfer check
- Bucket: K1 (push order / claim order); no cache. Benchmark gain: none. Decision: REJECT.
