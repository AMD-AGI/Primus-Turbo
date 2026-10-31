# 2026-10-30 e15-02 GPU2: is the shipped HK MLA forward the best one? (Phase 0 of the second fwd campaign)

Question: on Primus-Turbo `dev/lhz/dsv3-attn-hk-merge` (a95dd334, PR #588) the HIP forward `k_mla_fwd` (a08-1 E2,
hsaco 1d95fa84, rung `f0` of `mla/ladder_fwd2.json`) is bitwise equal to the FlyDSL forward 917e240b. Is it faster?
Answer: **no -- it is 0.8-1.8% slower than FlyDSL 917e240b at the production shape (cold, hot and burst timing), with
the same cycle count (1.138 vs 1.133 Mcycles); with zero inputs (no power cap) the two are equal.** The forward is
power-capped: zeros run 1.385x faster at the same cycle count.

All on e15-02 GPU2 (container lhz-hkl-g2), hkl.ladder, 10 rounds x 20 calls, null control on the top rung,
`HKL_SCALE=0.1352337788608801` (DeepSeek-V3's softmax scale), FlyDSL from `HKL_TURBO_DIR=Primus-Turbo-hk-fwd`.
Timing modes: cold = L2 flushed before every call; hot = back-to-back calls in one event pair (the README method);
burst = every call after 20 ms of idle GPU (the clock has not been pulled down: closest to a kernel inside a step).

| shape | mode | FlyDSL 917e240b ms | HK E2 (f0) ms | HK / FlyDSL | Mcycles FlyDSL / HK |
|---|---|---:|---:|---:|---|
| prod (b2 s4096 h128) | cold | 0.7909 | 0.7972 | 1.008 | 1.133 / 1.138 |
| prod | hot | 0.7876 | 0.7950 | 1.009 | 1.128 / 1.133 |
| prod | burst | 0.7045 | 0.7174 | 1.018 | 1.558 / 1.586 |
| prod, zeros | cold | 0.5756 | 0.5754 | 1.000 | 1.111 / 1.109 |
| prod, zeros | hot | 0.5724 | 0.5725 | 1.000 | 1.105 / 1.104 |
| b4 s4096 h128 | cold | 1.5795 | 1.6028 | 1.015 | 2.213 / 2.239 |
| b1 s4096 h16 | cold | 0.0748 | 0.0727 | 0.972 | 0.147 / 0.142 |
| s640 (b1 h16) | cold | 0.0151 | 0.0156 | 1.033 | 0.033 / 0.034 |

Files: `<shape>_<mode>.md / .json` (hkl.ladder output: per-round times, sclk, Mcycles = ms x median sclk).
The campaign that follows is in `../20261030_e15-02_gpu2_fwd1/README.md`.
