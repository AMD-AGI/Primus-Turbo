## Hypothesis

A divergent `if thread_index == 0: <side effect>` placed after `push_rows`' index loads sped the L2 forward
gemm_combine up by ~60 us (1440 -> 1380 us), with no change to the work done. A forward-only ISA dump shows why:
without it the compiler cannot prove one pool row index wave-uniform, keeps that row's buffer descriptor in
VGPRs and wraps each of its loads in a readfirstlane waterfall with `s_wait_loadcnt 0x0` inside (6 per pass,
4 passes per fold row). The fold sum then issues 7, 1, 1, 1, 1, 1, 1, 20, 4, 19 loads between waits instead of
one burst of ~53. A fold row belongs to one wave, so every index loaded for it is wave-uniform; pinning the pool
rows to SGPRs should give the burst without the branch.

## Single change

Kernel: `combine_bf16_tile.push_rows` (shared `bf16/ep_intranode.py`) passes the loaded pool rows through
`_readfirstlane_i32` on gfx1250 (gfx950 codegen unchanged).

Prerequisite, no codegen change: `SymBuffer.__cache_signature__` (`bf16/symm_buffer.py`) also hashes
`ep_intranode.py`. FlyDSL's disk-cache key only follows helper functions in the launcher's own directory, so
the `bf16_gfx1250` launchers kept replaying binaries built from the old `ep_intranode.py`: the first round-8
bench (`artifacts_stale_cache/`, step 1239.321 TF) ran the round-7 binary. Verified with the cache on:
pin 1387 us, pin disabled 1450 us, pin restored 1385 us (`artifacts/pck_key.log`).

## Results

### Validation
- Time: 2026-10-23 06:12 (UTC)
- Validation level: full (EP2 pytest 5 passed, EP1 pytest 19 passed, quick bench all PASS), bench repeated
  3x plus a same-session A/B with the pin disabled
- Raw artifacts: `artifacts/test_ep.log`, `artifacts/test_ep1.log`, `artifacts/quick_validation.log`,
  `artifacts/bench.csv`, `artifacts/quick_rep{2,3,4}.log`, `artifacts/quick_ab_nopin.log`,
  `artifacts/isa_fold_sum_load_runs.txt`

| Shape | R7 step TF (run 1) | No pin, same session | R8 step TF (run 1; reps 2-4) | R8 fwd ms (run 1; reps) | No-pin fwd ms |
|---|---|---|---|---|---|
| dsv3_t8192_random | 1509.5 | 1511.2 | 1532.3; 1525.6, 1521.2, 1523.1 | 3.946; 3.965, 4.000, 4.035 | 4.034 |
| dsv3_t8192_skew1 | 979.3 | 981.6 | 988.7; 985.6, 987.9, 989.2 | 6.164; 6.215, 6.171, 6.200 | 6.285 |
| dsv3_t4096_random | 1289.8 | 1294.4 | 1313.7; 1296.0, 1300.8, 1301.6 | 2.153; 2.176, 2.156, 2.163 | 2.188 |
| Aggregate (geomean) | 1239.985 | 1242.921 | 1257.834; 1249.051, 1250.349, 1251.680 | | |

Run 1 is +1.439% over round-7; the mean of the four runs (1252.228) is +0.987% over round-7 and +0.749% over
the same-session no-pin run. Min SNR 48.5 dB, unchanged.

Isolated L2 forward gemm_combine (`probe_combine_knobs.py 8192 7168 32 8 2048`, rank 0, cache off):
1442.2 -> 1383.2 us. ISA: waterfall compares 168 -> 112 (the rest guard the remote publish stores, see below).

## Decision

`ACCEPTED` — every run beats both round-7 and the same-session no-pin run on every shape; the gain is in the
forward (aggregate fwd 1201.3 -> 1211.4..1224.6 TF), the backward moves within noise.

## Attribution

The L2 forward combine sum (8 top-k rows, 4 passes per fold row) was serialized by readfirstlane waterfalls
on one pool row's descriptor. The backward's L1 dgrad combine (grad_gate path) did not move with the branch
experiment (2448 vs 2454 us), consistent with the backward's flat bench numbers.

## Real-training transfer check
- Bucket: K1 (codegen: uniformity hint on existing loads)
- Production path: stage2_forward (gemm_combine L2) — every training step's forward
- Skewed routing: dsv3_t8192_skew1 +0.7% over the no-pin run (mean of 4)
- Decision: ACCEPTED

## Next direction

The remaining 56 waterfalls wrap the fold sum's remote `buffer_store_b128` stores: `combine_token_res` is built
from `dst_rank`, loaded per segment into VGPRs. Pin `dst_rank` (wave-uniform per workgroup) to SGPRs.
