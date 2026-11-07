# Round 11 — split-K of the last in-step dgrad tiles in the bwd QKV A2AGemm (ROLLED BACK)

1. Hypothesis: after R10 the bwd QKV kernel ends ~90 us after the last push (16K: push 205, end
   ~300); the last m-tiles' dgrad tiles (in step: they start after their last source lands) take
   ~77 us on one CU each. Splitting the K of the last `split_m` m-tiles' tiles into one part per
   source (K_r = 1280 each), fp32 partials, and a fixed-order reduction by the last part should cut
   that to ~20 us + a short reduce.
2. Change (snapshot in `kernel_snapshot/`):
   - mega bf16 tile: `k_iters` (walk only that many K chunks, K stays the row stride) and
     `store_c_cls` on the NN path.
   - A2AGemm: `split_m`; the last `split_m * n_tiles` GEMM claims become `n` part claims each;
     a part waits for its source's flags (`_source_k_seg`), writes its fp32 tile to a scratch
     buffer, bumps a never-reset per-tile counter in `state[4:]` (last part = count multiple of n),
     and the last part reduces `((P0 + P1) + P2) + P3` into `Y` (bitwise deterministic).
3. Results (bwd QKV only, `e9_wgrad.py`, us 16K / 32K; R10 = split 0):

   | split_m | 0 | 4 | 8 | 12 | 16 |
   |---|---|---|---|---|---|
   | 16K | 303 | 352 | 370 | 393 | 411 |
   | 32K | 514 | 570 | 595 | 614 | 637 |

   Correct (rel err 5e-5 vs serial). Trace, 16K split_m 4, per part: MMA + partial store 32-44 us
   (1/4 of the K: ~19 us of MMA), reduce 36-47 us; split tile 72-91 us vs 77 us whole.
   Variants (fresh FlyDSL cache each): partial stores / reduce loads SYS_NT vs cached, agent
   acq_rel + acquire fence vs relaxed count, batched reduce loads: reduce stays 32-48 us.
   Reduce with no loads (stores only): 1-3 us; with one partial (128 KB): 11-19 us.
4. Decision: ROLLED BACK (every split_m slower).

## Rollback Analysis
- Change made: split-K by source of the last in-step dgrad tiles, fp32 partials through memory.
- Expected effect: tail after the last push 77 -> ~30 us.
- Actual effect: each part pays ~15 us to write 128 KB and the reducer ~40 us to read 512 KB.
- Root cause: under this kernel's load (every CU streaming, XGMI traffic landing) one CU moves
  only ~10-15 GB/s of uncached partials; a tile's own operands come at ~50 GB/s mostly from L2.
  The exchange costs as much as the K it saves.
- Lesson learned: no cross-CU fp32 exchange inside these kernels; cut the tail with work that
  stays in registers (a K walk by arrival: staggered sources), and fix the CU budget first: the
  160 wgrad tiles hold their CUs until the last push at ~45 % busy, so the dgrad tiles back up.

## Real-training transfer check
- Bucket: K3 (single-launch fusion); no cache. Benchmark gain: negative. Decision: REJECT (rolled back).
