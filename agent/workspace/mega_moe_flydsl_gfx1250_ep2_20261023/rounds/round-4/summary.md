## Hypothesis

`dispatch_bf16_tile` waits for every remote store to retire, barriers the workgroup and publishes a flag after
each of the 32 local experts, with only ~2 rows per copy warp per expert at EP2 T=8192: a serialized remote
round trip per expert. The dispatch-only launch (the x re-dispatch for dW1, fully exposed in the backward) has no
GEMM tile waiting on those flags, so it can publish once at the end and drop 31 round trips per comm block.

## Single change

Kernel: `dispatch_bf16_tile` (shared `bf16/ep_intranode.py`) takes `experts_per_publish` (default 1, the old
per-expert publish for every existing caller, including gfx950): that many experts share one store retire +
barrier, then their counters and flags go out in ascending expert order (the single-flag wait still holds). The
gfx1250 launcher passes it as a constexpr: all local experts for dispatch-only, 1 for dispatch+GEMM.

Probe (`artifacts/probe_publish*.log`, EP2 T=8192 H=7168 32 experts/rank topk 8, 64 comm CUs):

| experts per publish | dispatch_only us | dispatch+GEMM (N=4096) us |
|---|---|---|
| 1 | 490.1 | 2198 / 2201 / 2181 |
| 2 | 438.7 | 2173 / 2427 / 2170 |
| 4 | 404.7 | 2192 / 2181 / 2187 |
| 8 | 401.1 | 2637 |
| 32 (all) | 391.9 / 393.5 | 2321 |

Dispatch-only: -97 us (-20%). Fused dispatch+GEMM: groups of 2-4 are noise, 8+ is worse (the first tiles wait
longer for their flag), so the fused path stays per-expert.

## Results

### Validation
- Time: 2026-10-23 05:40 (UTC)
- Validation level: full (EP2 pytest 5 passed, EP1 pytest 19 passed, quick bench all PASS)
- Raw artifacts: `artifacts/test_ep.log`, `artifacts/test_ep1.log`, `artifacts/quick_validation.log`, `artifacts/bench.csv`

| Shape | R2 fwd / fwd+bwd ms | R4 fwd / fwd+bwd ms | R2 step TF | R4 step TF | Delta |
|---|---|---|---|---|---|
| dsv3_t8192_random | 4.435 / 12.315 | 4.364 / 12.106 | 1406.2 | 1430.4 | +1.72% |
| dsv3_t8192_skew1 | 6.397 / 18.292 | 6.393 / 18.001 | 946.7 | 962.0 | +1.62% |
| dsv3_t4096_random | 2.306 / 7.036 | 2.317 / 6.994 | 1230.5 | 1238.1 | +0.62% |
| Aggregate (geomean) | | | 1178.823 | 1194.345 | +1.317% |

Min SNR 48.5 dB (gate 40), unchanged from round-2.

## Decision

`ACCEPTED` — aggregate +1.317%, every shape improves, no regression; the gain is in the backward
(fwd+bwd -0.21 / -0.29 / -0.04 ms), where the dispatch-only launch runs.

## Attribution

The x re-dispatch for dW1 was paying ~100 us of per-expert publish latency on top of ~390 us of data movement.
Grouping the fused dispatch+GEMM's publishes does not help: its dispatch overlaps the GEMM, and later flags delay
the first tiles.

## Real-training transfer check
- Bucket: K2 (kernel-side synchronization change)
- Production path: `fused_mega_moe_gfx1250_impl.stage1_backward` -> `dispatch_only_gfx1250` runs once per MoE layer
  backward in training; the fused dispatch+GEMM path is unchanged.
- Skewed routing: dsv3_t8192_skew1 +1.62%.
- Decision: ACCEPTED

## Next direction

The remaining dispatch-only ~390 us is the data movement itself (117 MB remote at ~300 GB/s from 64 comm CUs):
give the dispatch-only launch more comm CUs (no GEMM shares the launch), or hide it behind dW2.
