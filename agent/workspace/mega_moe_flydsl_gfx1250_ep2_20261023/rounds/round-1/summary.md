## Hypothesis

Baseline round. No optimization change yet; record the starting correctness/performance and choose representative shapes.

## Single change

No code change. Freeze the starting kernel and environment state (Turbo `ca0a9c2f`, branch `feat/mega_moe/gfx1250-bf16`).

## Results

### Validation
- Time: 2026-10-23 05:02 (UTC)
- Validation level: full
- Test command: `pytest test_mega_bf16_gfx1250_fused_moe_ep.py` (GPUs 0,1: 5 passed) + `test_mega_bf16_gfx1250_{gemm_combine,dispatch_gemm,fused_moe}.py` (GPU 0: 19 passed)
- Benchmark command: `torchrun --nproc-per-node 2 quick_test_bench.py` (GPUs 0,1)
- Raw artifacts: `artifacts/test_ep.log`, `artifacts/test_ep1.log`, `artifacts/quick_validation.log`, `artifacts/bench.csv`

| Shape | T/rank | H | I | experts/rank | topk | Check (min SNR dB) | Fwd ms | Fwd+bwd ms | Fwd TF | Bwd TF | Step TF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dsv3_t8192_random | 8192 | 7168 | 2048 | 32 | 8 | PASS (48.5) | 4.609 | 12.920 | 1253 | 1389 | 1340.4 |
| dsv3_t8192_skew1 | 8192 | 7168 | 2048 | 32 | 8 | PASS (48.8) | 6.698 | 18.991 | 862 | 939 | 911.9 |
| dsv3_t4096_random | 4096 | 7168 | 2048 | 32 | 8 | PASS (48.5) | 2.406 | 7.190 | 1200 | 1207 | 1204.3 |

### Aggregate
- Combined Step TFLOPS (geomean): 1137.514
- Fwd aggregate: 1089.939, Bwd aggregate: 1163.258

## Decision

`BASELINE`

## Attribution

Per-kernel trace of one EP2 fwd+bwd (T=8192 random, rank 0, `/home/xiaompen/mega_scratch/prof_ep2`): 13.46 ms span, GPU busy 98.7%.
GEMM-only reference for the six GEMMs is 9.14 ms. Overheads over it: L1 dgrad + combine +1.11 ms,
L2 + combine +0.95 ms, x re-dispatch for dW1 0.48 ms (exposed, no GEMM to hide under), two torch pool
gathers 0.41 ms, prologue + D2H 0.36 ms, SwiGLU fwd+bwd 0.30 ms, fused dispatch 0.18 ms.
GEMMs themselves run at 1.8-2.0 PF (FlyDSL == hipBLASLt). The combine overhead exists at EP1 too (+0.40 / +0.55 ms)
and does not depend on routing (clustered routing: same overhead), i.e. it is the combine/reduce data movement rate.

## Next direction

Highest-value items, one per round: (1) hide the x re-dispatch for dW1, (2) drop the torch pool gathers,
(3) raise combine/reduce throughput.

## Real-training transfer check
- Bucket: baseline (no change)
- Decision: BASELINE
