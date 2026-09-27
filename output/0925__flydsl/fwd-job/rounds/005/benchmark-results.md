> Unprofiled. These are the round's comparison numbers. Every figure in the analyses below
> comes from a profiled run, where dispatches are serialised and the clock rises -- 1.97 GHz
> under the profiler against 1.74 GHz sustained on the same kernel. **The round's score is
> this file and only this file.**

# Round 5 benchmark -- `op/current` vs `beat` (aiter gfx1250 ASM forward)

Run 2026-09-27 09:21-09:23 UTC in container `fa-repro`, GPU 0 (single card, PCI 0001:01:00.0, gfx1250).
`benchmark.py --arms current,beat --shapes <s>`, one shape per process, median of 101 CUDA-event
iterations, 8 s continuous warmup per arm, palindromic arm order, L2 flushed, causal (bottom-right), bf16.
FlyDSL compile cache cleared before the run. Target = `beat` in the same run, margin 0% (op.target.beat_margin_pct).

| shape | (B, Sq, Skv, Hq, Hkv, D) | current ms | current TFLOP/s | beat ms | beat TFLOP/s (target) | current/beat | sclk start→end MHz |
|---|---|---:|---:|---:|---:|---:|---|
| fast  | (1, 1024, 1024, 8, 2, 128)  | 0.05616 | 38.27   | 0.03073 | 69.96   | 0.547 | 1100→1100 |
| proxy | (1, 4096, 4096, 32, 8, 128) | 0.19549 | 703.21  | 0.13636 | 1008.13 | 0.698 | 1027→1062 |
| prod  | (4, 8192, 8192, 32, 8, 128) | 2.03012 | 1083.33 | 1.55882 | 1410.87 | 0.768 | 1019→1031 |

Cross-check: round 4's gate measured the same code at prod 1082.6 TF/s vs beat 1401.6 (0.7724); this run
reads 1083.33 vs 1410.87 (0.768). `current` agrees within 0.07%.

Raw RESULT lines and JSON rows: `0-preflight/raw/bench_{fast,proxy,prod}/`.
