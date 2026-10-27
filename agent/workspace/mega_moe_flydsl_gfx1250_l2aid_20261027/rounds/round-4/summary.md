## Hypothesis

dW1 needs x at every rank again, and the shipped `stage1_backward` re-dispatches it with `dispatch_only` after
the L1 dgrad `gemm_combine`: 0.28-0.30 ms at T=8192 random with nothing to overlap (stage bench: the one comm leg
hidden 0%). That transfer needs no routing — every rank can hold all of x at rows `src_rank * T_max + token`, the
`recv_token` index dW1 already reads through `pool_row_to_recv_token` — so the copy engines can move it during the
2.3 ms L1 dgrad + combine without taking a CU.

## Single change

- New `primus_turbo/flydsl/mega/bf16_gfx1250/x_redispatch_kernel.py`: per SymmBuffer key a second symmetric receive
  buffer `[num_ranks * T_max, H]` bf16 + an i64 flags region, a side stream, and two 1-WG FlyDSL kernels:
  `x_copy_signal` (bump this rank's epoch, store it into `flags[RANK]` of every rank, system scope) and
  `x_copy_wait` (spin until `flags[s] >= epoch` for every source s).
- `fused_mega_moe_gfx1250_impl.stage1_backward`: before `gemm_combine`, the side stream (behind the current stream)
  copies x into rows `rank * T_max` of every rank's buffer (own rank first, then rank+1, +2, +3; hipMemcpy D2D on
  IPC peer pointers = SDMA), then signals; after `gemm_combine` the compute stream waits on the side stream and runs
  `x_copy_wait`; dW1 reads the second buffer. `dispatch_only` is no longer called in the step.
- WAR on the second buffer: this rank's copy starts after its stage-2 backward dispatch of dy, which waits on every
  peer's dy, so every peer has finished the dW1 that last read its buffer (holds per MoE layer too).
- Cost: +470 MB symmetric memory per rank at T_max=8192 (4 x 8192 x 7168 x 2 B).

## Results

### Validation
- Time: 2026-10-27 15:04
- Validation level: full (`run_round.sh 4 full`): EP4 deepseek_v3 2, ragged_skewed 1, one_token 1, staged_bias 1;
  EP2 5; EP1 19; grouped GEMM gfx1250 31 — all passed. Bench 3/3 PASS (SNR gate).
- Stress (`tools/stress_xredisp.py`, new x / dy / routing every iteration, bitwise vs dispatch_only + variable_k):
  T=256 x 60 and T=8192 x 50 iterations, 0 buffer and 0 dW1 mismatches over 4 ranks. No stale GL2 line of the
  second buffer after the remote SDMA writes.
- Raw artifacts: `artifacts/ab_r4.txt`, `artifacts/ab_r4_summary.txt`, `artifacts/kt_r4_compare.txt`,
  `artifacts/stress_t{256,8192}.log`, `artifacts/test_*.log`, `artifacts/bench_*.csv`

### Same-session A/B (`tools/ab.sh r4 2`, fwd+bwd ms, mean of 2 reps)

| Variant | t4096 random | t8192 random | t8192 skew1 | Step geomean TF | vs acc |
|---|---|---|---|---|---|
| acc (8bbd8c89) | 6.870 | 12.000 | 28.099 | 1038.774 | — |
| sdma | 6.744 (+1.88%) | 11.861 (+1.17%) | 27.976 (+0.44%) | 1050.843 | +1.162% |

Both reps faster on every shape. Full-validation bench (repo, default on): 11.868 / 27.931 / 6.745 ms, step
geomean 1051.126 TF (fwd 995.977, bwd 1081.099), +1.10% vs the round-1 baseline 1039.651.

### Kernel trace (`tools/kt.sh r4 1 8192`, mean over ranks, last 8 iterations)

| Kernel | acc µs | sdma µs |
|---|---|---|
| `dispatch_gemm_nt_1 #3` (dispatch_only x) | 303.8 | — |
| L1 dgrad `gemm_combine_nt_1 #2` | 2371.2 | 2436.5 (+2.75%) |
| `x_copy_wait` (on the critical path) | — | 1.8 |
| local slice copy (`copyBuffer`, blit) | — | 11.3 |

The peer copies do not show as kernels (SDMA). The L1 dgrad gemm_combine pays ~65 µs: the copies share the XGMI
links with its combine push. `x_copy_wait` at 1.8 µs means the copies have always landed when dW1 starts.

## Decision

`ACCEPT`. Default on, no switch; committed locally, accepted tree synced.

## Attribution

-304 µs exposed dispatch, +65 µs combine contention on the links, +~15 µs elsewhere (wait, local copy, noise):
~-0.14 ms per step at t8192 random, matching the A/B (-0.139 ms).

## Next direction

The two combine stages carry the largest fused - gemm_only gap (L1 dgrad 0.54 ms, L2 fwd 0.46 ms): the push of a
fold row waits for its token's highest local expert, so it piles up at the end of the GEMM. The copy-vs-combine
link contention (~65 µs) is a secondary target (issue order / timing of the side-stream copies).

## Real-training transfer check
- Bucket: a (removes a fully exposed comm leg of every MoE layer's backward; the step-level win is shape-general)
- Decision: ACCEPT
- Skew: +0.44% at skew1 (smaller: the step is dominated by the hot rank's GEMMs, the removed leg is the same size)
- Memory: +470 MB symmetric memory per rank at T_max=8192
