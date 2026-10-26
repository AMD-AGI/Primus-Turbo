## Hypothesis

The dispatch prologue ends with a host `.item()` on the pool-row count, so the host only learns the rows after the
whole kernel (phases D scatter, grid_sync D, xgmi E: ~110-130 us at EP4 T=8192) and then spends ~165 us on the
handle's four clones before it launches `dispatch_gemm`. The GPU sits idle ~100-130 us per step between the
prologue and `dispatch_gemm` (round-2 trace). Publishing the rows to host memory at the end of phase C lets the
host enqueue the clones and `dispatch_gemm` while phases D-E still run.

## Single change

`bf16/dispatch_prologue_kernel.py` (gfx1250 only; gfx950 keeps the D2H):
- block 0 stores `total_rows` at system scope into a coherent pinned host int32 (`hipHostMalloc` mapped |
  coherent, device address passed in a 1-element int64 tensor) right after it writes
  `num_tokens_per_expert_prefix[-1]`;
- the host arms the cell to -1 before the launch and polls it; if the kernel is done and the cell is still -1 it
  falls back to the device read.
`pytorch/core/pyhip_runtime_wrapper.py`: `hipHostMalloc` / `hipHostGetDevicePointer`.
Patch: `artifacts/round3_host_rows_notify.patch`.

## Results

### Validation
- Time: 2026-10-26 03:25
- Validation level: full
- Test command: `run_round.sh 3 full`
- Raw artifacts: `artifacts/test_ep4_*.log`, `test_ep2.log`, `test_ep1.log`, `quick_<shape>.log`, `bench.csv`,
  `ab_r3.txt`, `probe_notify_ep4.log`, `gaps_ss_*.txt`, `cmp_ss_base_vs_r3.txt`

Tests: EP4 test_deepseek_v3 2/2, test_ragged_skewed 1/1, test_one_token 1/1, test_staged_bias_gpt_oss 1/1;
EP2 5/5; EP1 19/19.

Mechanism (`probe_notify_ep4.log`, 25 calls per rank): rows seen before the kernel finished 25/25, fallback 0/25,
rows equal to the device value 25/25. Steady-state trace (`gaps_ss_*.txt`, 8 back-to-back fwd+bwd, T=8192 random):
prologue end -> `dispatch_gemm` start 104-129 -> 18-20 us; GPU idle per iteration 101-124 -> 28-31 us.

Full-set run:

| Shape | Check | fwd ms | fwd+bwd ms | Fwd TF | Bwd TF | Step TF |
|---|---|---|---|---|---|---|
| dsv3_t8192_random | PASS | 4.107 | 11.468 | 1405 | 1568 | 1510.1 |
| dsv3_t8192_skew1 | PASS | 9.444 | 27.276 | 611 | 647 | 634.9 |
| dsv3_t4096_random | PASS | 2.275 | 6.752 | 1268 | 1290 | 1282.5 |

Same-session A/B, baseline tree (1149c02b) vs this change, alternating (`ab_r3.txt`), fwd+bwd ms:

| Shape | baseline (3 reps) | change (3 reps) | median |
|---|---|---|---|
| dsv3_t8192_random | 11.518 / 11.513 / 11.530 | 11.632 / 11.534 / 11.495 | +0.14% |
| dsv3_t4096_random | 6.707 / 6.759 / 6.762 | 6.731 / 6.753 / 6.756 | -0.09% |

### Aggregate
- Full-set run: Combined Step TFLOPS 1071.307 (+0.405% vs 1066.987); fwd 1029.067, bwd 1094.030
- Not confirmed: the alternating A/B shows no gain beyond run-to-run noise

## Decision

`ROLLED BACK` — the idle is gone but the step time is not shorter; the full-set +0.4% is inside the A/B spread.
Code restored to 1149c02b (`git checkout` of both files).

## Attribution

The ~80 us of per-rank GPU idle removed does not come off the step. In the trace the iteration shrinks only
11-61 us (7-iteration windows), while the kernels with cross-rank waits grow: gemm_combine +35..+62 us, the
prologue +10..+48 us (xgmi barriers) on ranks 1-3 (`cmp_ss_base_vs_r3.txt`). The freed time turns into waiting
on peers; a second, unverified possibility is power-limited clocks absorbing a 1% higher duty cycle.

## Rollback Analysis
- Change made: host-mapped rows notify at the end of prologue phase C (gfx1250)
- Expected effect: about -100 us per step (-0.9% at T=8192, -1.5% at T=4096)
- Actual effect: +0.14% / -0.09% median in the A/B (noise); GPU idle per iteration -80 us
- Root cause: the post-prologue idle is not on the EP4 step's critical path; removing it lengthens cross-rank waits
- Lesson learned: confirm a GPU-idle removal with a same-session A/B of the full step before counting the idle as
  the gain; at EP4 the step is bound by cross-rank coupling, not by per-rank idle gaps of ~1%

## Next direction

1. L2-cluster / AID-aware workgroup placement once the reference doc is readable (GEMM tile locality).
2. Shorten the post-GEMM combine backlog (round 2): order the pool so fold rows complete earlier, or push the
   non-fold partials early.
3. Clock / power telemetry during the step (amd-smi) to test whether busy-time reductions are absorbed by clocks.

## Real-training transfer check
- Bucket: K2 (host scheduling)
- Decision: ROLLED BACK (no step gain)
