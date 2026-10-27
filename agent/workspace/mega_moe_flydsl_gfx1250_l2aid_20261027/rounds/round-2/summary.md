## Hypothesis

`gemm_combine` (L2 fwd NT, L1 dgrad NN) decodes its tiles row major over 28 N blocks; 28 is 4 mod 8, so both
AIDs read every weight block and both GL2s fill with all of it. Decoding `AID_GROUP_M` pool blocks at a time, m
fastest, with each AID taking one N half of the group (`aid_half_index`), halves the weight each GL2 must hold
at every routing (`tools/aid_model.py`: B 1792 -> 901 blocks at T=8192 random). Expected: a fraction of the
reference's +0.8%.

## Single change

`gemm_combine_kernel.py`: constexpr `AID_GROUP_M` (env `MEGA_AID_SPLIT_GC`, 0 = shipped row-major decode);
`gemm_nt_tile.aid_half_index` (bijective for any `bid` alignment). Variants 2 and 4 measured.

## Results

### Validation
- Time: 2026-10-27 06:00
- Validation level: quick (EP1 pytest with each variant + the A/B's own SNR check on the 3 shapes)
- Raw artifacts: `artifacts/test_ep1_gc2.log`, `artifacts/test_ep1_gc4.log` (15 passed each), `artifacts/ab_r2.txt`,
  `artifacts/ab_r2_summary.txt`, `artifacts/kt_r2.txt`, `artifacts/kt_r2_compare.txt`

### Same-session A/B (`tools/ab.sh r2 2`, fwd+bwd ms, mean of 2 reps, all PASS)

| Variant | t4096 random | t8192 random | t8192 skew1 | Step geomean TF | vs acc |
|---|---|---|---|---|---|
| acc (8bbd8c89) | 6.854 | 11.986 | 28.130 | 1039.681 | — |
| gc2 | 6.829 (+0.35%) | 11.912 (+0.61%) | 28.398 (-0.94%) | 1039.734 | +0.005% |
| gc4 | 6.857 (-0.06%) | 11.976 (+0.08%) | 28.061 (+0.25%) | 1040.625 | +0.09% |

### Kernel trace (`tools/kt.sh r2 2 8192`, all ranks, 2 reps alternating, mean us)

| Kernel | acc | gc2 | gc4 |
|---|---:|---:|---:|
| L2 fwd gemm_combine (NT, K=2048) | 1517.9 | 1709.6 (+12.6%) | 1524.4 (+0.4%) |
| L1 dgrad gemm_combine (NN, K=4096) | 2387.3 | 2294.8 (-3.9%) | 2379.7 (-0.3%) |
| GPU busy / iteration | 11094.7 | 11235.1 (+1.3%) | 11085.1 (-0.1%) |

Unchanged kernels move by up to 5% between variants (e.g. L1 fwd dispatch_gemm +5.2% under gc2, identical
code): the fused kernels wait on remote ranks, so per-kernel trace deltas carry cross-rank skew.

### Stage bench after this round (`tools/stage_bench.sh`, accepted tree = 8bbd8c89)

`tools/bench_mega_gfx1250.py` is benchmark/ops/training/bench_mega_moe.py's gemm_only / comm_only / fused view
on the gfx1250 kernels (the original bench drives the gfx950 kernels: `buffer.load.lds` does not lower on
gfx1250, LLVM ERROR). Full table: `artifacts/stage_bench_acc/summary.txt`. t8192 random, ms:

| Stage | gemm_only (TF, % dense) | comm_only | fused | hidden / comm | fused / gemm_only |
|---|---|---|---|---|---|
| L1 fwd dispatch_gemm | 1.708 (2408, 97%) | 0.617 (64 CU) | 2.035 | 47% | 119% |
| L2 fwd gemm_combine | 0.931 (2207, 95%) | 1.104 | 1.386 | 59% | 149% |
| L2 dgrad dispatch_gemm | 0.864 (2380, 97%) | 0.617 | 1.153 | 53% | 134% |
| dW2 variable-K | 1.046 (1966, 84%) | — | 1.046 | — | 100% |
| L1 dgrad gemm_combine | 1.742 (2360, 94%) | 1.108 | 2.286 | 51% | 131% |
| dW1 x re-dispatch + variable-K | 1.981 (2076, 86%) | 0.284 (all CU) | 2.296 | 0% (serial) | 116% |
| Sum | 8.272 | 3.730 | 10.201 | 48% | 123% |

fused_mega_moe step 11.897 ms; 1.696 ms outside the six GEMM stages. bench_dsv3 (256 experts, balanced):
hidden 28% of comm, fused / gemm_only 122%; skew1: 68%, 113%.

## Decision

`ROLLBACK`. Neither variant clears the noise on the step (+0.005%, +0.09%); gc2 slows the L2 fwd gemm_combine by
12.6% while speeding the L1 dgrad one by 3.9%. `gemm_combine_kernel.py` restored to 8bbd8c89;
`aid_half_index` stays in `gemm_nt_tile.py` for round 3.

## Attribution

The standalone grouped GEMMs already run at 94-97% of the dense single-weight roofline: the GL2 traffic the split
removes was not limiting them. Inside the fused kernels the loss is the overlap (fused 1.19-1.49x gemm_only, 48%
of the comm hidden), which this change does not touch. The NN gain alone (L1 dgrad -3.9%, ~0.8% of the step)
is a candidate for a later NN-only round.

## Next direction

1. Round 3 as planned: dispatch_gemm AID split.
2. The stage bench puts most of the step's slack in the fusion overlap, not in GL2 placement: combine stages
   1.31-1.49x gemm_only, dispatch stages 1.19-1.34x, the dW1 x re-dispatch fully exposed (0.28 ms).

## Real-training transfer check
- Bucket: c (no change kept)
- Decision: ROLLBACK
- Skew: gc2 -0.94% at skew1, gc4 +0.25%
