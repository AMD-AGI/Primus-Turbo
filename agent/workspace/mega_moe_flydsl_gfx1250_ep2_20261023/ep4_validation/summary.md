# EP4 validation of rounds 2-9 (GPUs 0-3, 2026-10-26 UTC)

Each commit's tree is overlaid on a copy of `primus_turbo` (`/home/xiaompen/mega_scratch/ep4val/src_<sha>`,
own FlyDSL cache dir), benched with `quick_test_bench.py` (`bench_ep.py`: tree on `sys.path`, one shape per
torchrun so the symmetric buffer is created once per process). `run_ep4.sh COMMIT TAG [SHAPES]`.

## fwd+bwd ms (delta vs previous row)

| Round | commit | T=8192 random | T=8192 skew1 | T=4096 random |
|-------|--------|---------------|--------------|---------------|
| base | ca0a9c2f | 14.903 / 14.903 (2 runs) | 31.055 / 30.934 | 8.178 / 8.189 |
| 2 | 5aa03e07 | 14.287 (-0.616) | 30.285 (-0.710) | 8.045 (-0.139) |
| 4 | bbca1687 | 14.203 (-0.084) | 29.734 (-0.551) | 7.979 (-0.066) |
| 5 | 172e0448 | 13.799 (-0.404) | 29.439 (-0.295) | 7.854 (-0.125) |
| 6 | a9b233b9 | 12.441 (-1.358) | 28.698 (-0.741) | 7.201 (-0.653) |
| 7 | 090870f5 | 12.119 (-0.322) | 28.825 (+0.127) | 7.092 (-0.109) |
| cache key | 9bf4c612 | 12.129 (+0.010) | 28.492 (-0.333) | 7.091 (-0.001) |
| 8 | 1dbd82fc | 11.812 (-0.317; mean of 4: 11.805-11.826) | 28.135 (-0.357) | 6.933 (-0.158) |
| 9 | 1149c02b | 11.553 (-0.259; median of 5: 11.525-12.079) | 27.456 (-0.679; 2 runs) | 6.784 (-0.149; 2 runs) |

Base -> round 9: -3.35 / -3.54 / -1.40 ms (-22.5% / -11.4% / -17.1%); step TF 1162 -> 1499 (+29.0%),
559 -> 631 (+12.9%), 1058 -> 1276 (+20.6%); geomean step TF 882.3 -> 1064.7 (+20.7%; EP2 bench.csv +12.7%).

Single runs except where noted. Forward-only timing is noisy at EP4 (round 8 reps: 4.26-4.55 ms, round 9 reps:
4.32-5.37 ms); fwd+bwd is the stable metric. Round 9 at T=8192 random is bimodal (3 of 5 runs ~11.53 ms,
others 11.79 / 12.08); round 8 is not (4 runs within 0.02 ms).

## Correctness

- Bench: all 39 CSVs PASS, min SNR 48.64 dB.
- HEAD pytest `test_mega_bf16_gfx1250_fused_moe_ep.py` at EP4: 4 passed together; `test_staged_bias_gpt_oss`
  hangs when run after the others (`test_ep4_head.log`: ALLGATHER seq 115 timeout, rank 3 missing from the
  symmetric-buffer rendezvous after the buffer key changes). Baseline ca0a9c2f hangs the same way
  (`test_ep4_base*.log`). Run alone at HEAD it passes (`test_ep4_head_gpt_oss.log`, 22.8 s).
- 37 of 39 EP4 bench torchruns crash at process exit after the results print (`ChildFailedError`; mostly
  SIGSEGV on one or more ranks, the rest SIGTERM'd by torchrun, a few SIGABRT), baseline included; not
  investigated.
