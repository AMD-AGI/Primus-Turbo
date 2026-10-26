| Round | Status | Description | Fwd Avg TFLOPS | Fwd Peak TFLOPS | Bwd Avg TFLOPS | Bwd Peak TFLOPS | Combined Step TFLOPS | vs Baseline | Key Finding |
|-------|--------|-------------|----------------|-----------------|----------------|-----------------|----------------------|-------------|-------------|
| 1 | BASELINE | 1149c02b (EP2 campaign round-9), EP4 GPUs 0-3 | 1020.505 | 1402.000 | 1092.019 | 1560.000 | 1066.987 | — | skew1 at EP4 is 2.4x random (EP2 1.5x); EP2 info 1279.377 |
| 2 | REJECTED | lead combine knobs at EP4 (8-48 WGs, 10-40%, fwd / dgrad) | — | — | — | — | — | screen T=8192 random best -0.03% (noise) | post-GEMM push backlog is set by when fold rows become pushable, not by pusher count |
| 3 | ROLLBACK | prologue publishes rows to host memory after phase C (gfx1250) | 1029.067 | 1405.000 | 1094.030 | 1568.000 | 1071.307 | step +0.405% full run; A/B +0.14% / -0.09% (noise) | GPU idle -80 us per step, but the freed time turns into cross-rank waits |
