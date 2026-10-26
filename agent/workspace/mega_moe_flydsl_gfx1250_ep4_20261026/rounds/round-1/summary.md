## Hypothesis

Baseline round. No optimization change yet; record the EP4 starting correctness/performance of the EP2
campaign's last accepted round (1149c02b) on GPUs 0-3.

## Single change

No code change. Freeze the starting kernel and environment state (`kernel_snapshot/git_state.txt`: 1149c02b,
clean tracked tree).

## Results

### Validation
- Time: 2026-10-26 02:26
- Validation level: full
- Test command: `run_round.sh 1 full` (EP4 one pytest process per test, EP2 on GPUs 0,1, EP1 on GPU 0)
- Benchmark command: per shape `HIP_VISIBLE_DEVICES=0,1,2,3 ONLY_SHAPE=<label> torchrun --nproc-per-node 4 quick_test_bench.py`
- Raw artifacts: `artifacts/test_ep4_*.log`, `test_ep2.log`, `test_ep1.log`, `quick_<shape>.log`, `bench.csv`,
  `ep2_quick.log`, `ep2_bench.csv`

Tests: EP4 test_deepseek_v3 2/2, test_ragged_skewed 1/1, test_one_token 1/1, test_staged_bias_gpt_oss 1/1;
EP2 5/5; EP1 19/19.

| Shape | Check | SNR min | fwd ms | fwd+bwd ms | Fwd TF | Bwd TF | Step TF |
|---|---|---|---|---|---|---|---|
| dsv3_t8192_random | PASS | > 48 dB | 4.119 | 11.521 | 1402 | 1560 | 1503.1 |
| dsv3_t8192_skew1 | PASS | > 48 dB | 9.509 | 27.406 | 607 | 645 | 631.9 |
| dsv3_t4096_random | PASS | > 48 dB | 2.311 | 6.770 | 1249 | 1295 | 1279.0 |

### Aggregate
- Combined Step TFLOPS (geomean): 1066.987
- Forward aggregate: 1020.505; Backward aggregate: 1092.019
- EP2 (info): 1279.377 step (11.116 / 17.130 / 6.512 ms)

## Decision

`BASELINE`

## Attribution

EP4 vs EP2 at the same per-rank work: T=8192 random +0.41 ms, T=4096 random +0.26 ms, skew1 +10.3 ms. The
EP2 campaign's EP4 replay (`../mega_moe_flydsl_gfx1250_ep2_20261023/ep4_validation/summary.md`) puts this
commit at 11.53-12.08 ms on T=8192 random (bimodal) and 27.46 ms on skew1.

## Next direction

1. Fused dispatch comm workgroups: `dispatch_gemm_nt_gfx1250` defaults to 64, i.e. 16 per destination at EP4
   (32 at EP2), for the L1 forward (x) and L2 dgrad (dy) launches.
2. Lead combine knobs (16 WGs, 25%) were swept only at EP2.
3. skew1 at EP4 is 2.4x random (1.5x at EP2): find which kernel carries it.
