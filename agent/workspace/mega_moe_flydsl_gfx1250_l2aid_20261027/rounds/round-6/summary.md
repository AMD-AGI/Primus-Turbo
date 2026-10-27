## Hypothesis

The gemm_combine clock-stamp timeline (`tools/gc_trace.py`, EP4 t8192 random, round-4 tree) shows the 16 lead
combine workgroups, placed after 25% of the GEMM tiles, push only 72-121 MB of the ~320 MB remote before the GEMM
ends, while slowing the GEMM by ~89 µs (L2 fwd) and ~144 µs (L1 dgrad) on average: they hold 16 CUs, spin on
pool-block flags, and their remote stores share the memory path with the GEMM's loads. With one lead at 100% (all
combine workgroups after the GEMM), both launches end ~100 µs earlier on the slowest rank. Removing the lead
workgroups should shorten both combine stages.

## Single change

`gemm_combine_kernel.py`: `num_lead_combine` default 16 -> 0 (the assert allows 0), so the grid is the GEMM tiles
followed by one combine workgroup per CU. `lead_gemm_percent` no longer matters by default; both knobs stay.
The A/B tested env-selected variants (each in its own FlyDSL cache): `l0` = 1 lead at 100% of the tiles (the same
workgroup-to-role mapping as 0 leads: combine index `bid - num_gemm_tiles` for every combine workgroup), `l4`,
`l8` (at 25%) against acc = 16 at 25%.

GPU note: from 15:30 GPUs 0,1 belong to another user. This round and the following ones run EP2 on GPUs 2,3;
EP4 tests are skipped and the step numbers are not comparable with rounds 1-5 (EP4).

## Results

### Validation
- Time: 2026-10-27 15:48
- Validation level: full (`run_round.sh 6 full`, GPUS=2,3): EP4 skipped; EP2 5 passed; EP1 19 passed; grouped
  GEMM gfx1250 31 passed. Bench 3/3 PASS (SNR gate).
- Raw artifacts: `artifacts/ab_r6.txt`, `artifacts/ab_r6_summary.txt`, `artifacts/run_round6.txt`,
  `artifacts/test_*.log`, `artifacts/bench_*.csv`

### Same-session A/B (`tools/ab.sh r6 2`, EP2, fwd+bwd ms, mean of 2 reps)

| Variant | t4096 random | t8192 random | t8192 skew1 | Step geomean TF | vs acc |
|---|---|---|---|---|---|
| acc (3ef91841, 16 @ 25%) | 6.460 | 11.157 | 17.413 | 1274.214 | — |
| l0 (1 @ 100%) | 6.423 (+0.58%) | 11.075 (+0.74%) | 17.285 (+0.74%) | 1282.964 | +0.687% |
| l4 (4 @ 25%) | 6.431 (+0.46%) | 11.079 (+0.70%) | 17.380 (+0.19%) | 1279.949 | +0.450% |
| l8 (8 @ 25%) | 6.431 (+0.45%) | 11.183 (-0.24%) | 17.426 (-0.07%) | 1274.781 | +0.045% |

`l0` is faster than acc on every shape in both reps (6 of 6 pairs). Full-validation bench of the committed tree
(0 leads): 11.027 / 17.229 / 6.430 ms, step geomean 1285.765 TF (fwd 1241.340, bwd 1309.268).

### vs the EP2 baseline (`tools/ab.sh ep2base 2`, same session, fwd+bwd ms, mean of 2 reps)

| Tree | t4096 random | t8192 random | t8192 skew1 | Step geomean TF | vs base |
|---|---|---|---|---|---|
| base (8bbd8c89, round 1) | 6.559 | 11.264 | 17.564 | 1260.143 | — |
| round 4 (3ef91841) | 6.463 (+1.48%) | 11.171 (+0.82%) | 17.462 (+0.58%) | 1272.241 | +0.960% |
| round 6 (d97bd80c) | 6.428 (+2.04%) | 11.090 (+1.57%) | 17.262 (+1.75%) | 1282.641 | +1.785% |

Artifacts: `artifacts/ab_ep2base.txt`, `artifacts/ab_ep2base_summary.txt`. Later EP2 rounds report vs this base.

## Decision

`ACCEPT`. Default 0 lead combine workgroups; committed locally, accepted tree synced.

## Attribution

On MI455X a lead combine workgroup costs the GEMM more than its early push saves. The tail push after the GEMM is
latency-bound (34-40 µs per 16-row chunk at 0.6-0.96 TB/s over 240 workgroups), so pushing a quarter of the rows
early through 16 workgroups does not shorten it much, while 16 spinning CUs plus their remote stores stretch the
GEMM in every combine stage (L2 fwd, L2 dgrad, L1 dgrad). Fewer leads is monotonically better (8 -> 4 -> 0).

## Next direction

With no lead, the combine stage is GEMM (fused 1075-1137 µs vs 950 µs standalone at EP4 L2 fwd), then the tail
push (~360 µs), then the reduce (~100 µs). Next: why the fused GEMM is 13-20% slower than the standalone grouped
GEMM, and the tail push's per-chunk latency. Run the EP2 stage bench of this tree first to re-rank the stages.

## Real-training transfer check
- Bucket: K1 (kernel-internal workgroup role assignment)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Workload distribution robustness: holds under skew (t8192 skew1 +0.74%, same as random)
- Benchmark gain on this round: +0.69% combined-step (A/B, EP2)
- Estimated real-training gain: +0.69% combined-step (scheduling only, every MoE layer's three combine stages)
- Decision: ACCEPT-as-real
