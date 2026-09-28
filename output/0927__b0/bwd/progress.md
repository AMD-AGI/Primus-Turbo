# B0 bwd job progress (gfx1250-flydsl-attn-bwd-20260917-115934, physical GPU 1 / fa-g1)

prod FLOP 5.498229e12; ms = FLOP / TF/s. All figures are the round's own same-session measurement on B0.
Champion before B0: round 20 (A0 511 TF/s; B0 re-measure ~633-650 TF/s, 0.74-0.76x ASM).

| round | mode | UTC | accepted | prod TF/s (ms) | same-session champion prod | gain (geomean) | % ASM (prod) | proxy / fast TF/s | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| 24 | fast | 12:34-13:32 | no | 638.8 (8.607) | 650.2 (8.456) | 0.9665 | 75.0% (champ 76.3%) | 578.1 / 93.5 | h60 unroll -4.5% DEAD; X1 0.804x (4-wave cost = barrier re-alignment); X2 0.872x DEAD; h33 defects a/b/c shipped in round tree only |
| 25 | deep | 09-27 13:42 -> 09-28 02:39 (act re-run on GPU3 after day-1 stop) | no | 615.6 (8.931) | 637.6 (8.624) | prod 0.9656 / geomean 0.9878 | 74.4% (champ 77.0%) | 515.7 / 97.5 | g77 L2 phase-align via reversed scan: -3.4% prod; confounded by register allocation (I-fetch +15.2%, bit-identical on GPU1 and GPU3); family not decided |
| 26 | fast | 09-28 02:52-03:54 | no | 640.0 (8.591) | 638.6 (8.610) | +0.21% prod / 1.0036 (< 1.007) | 76.5% (champ 76.4%) | 596.2 / 97.3 | armA (g62+h33 clamp form) null; g82/g83 abandoned at free gates; h33 clamps are a correctness fix with a VGPR cost |
| ruler | audit | 09-28 03:02-04:19 | -- | -- | -- | blocked harness installed; FlyDSL-vs-ASM interleave bias ~3.4% (x-beat ~3% too low), FlyDSL-vs-FlyDSL <0.4%; **r19 1.7% faster than r20 at prod** (h67); r25 loss confirmed | -- | ~78.8% (blocked beat/current 0.7878) | -- | -- |
| 27 | fast | 09-28 ~04:00-05:10 | no (job ENDED: false target_met) | 641.3 (8.573) | 639.0 | 0.9929 | 78.4% (blocked beat 817.8) | 585.7 / 98.4 | gate geomean 1.007x beat (fast 1.66x hides prod/proxy ~0.78x) -> job stopped as target_met; validation fix needs user approval |
| 28 | refactor (h68) | 09-28 ~06:50 | **promoted** (correct vs eager, 1 turn) | -- | -- | adopt r19h (r19 + h33 clamps); validation target now proxy AND prod >= beat (h69); refcache provenance refreshed | -- | -- | -- | user-approved operator changes |
| 28 opt | fast | 09-28 06:50-~07:50 | no | 637.2 (8.629) | r19h 640.6 (8.583) | 0.9967 | 78.0% (beat 816.8) | 588.2 / 98.3 | round's own arms on top of the new r19h champion null; h70 (u2n) applies from r29 |
| **29** | fast | 09-28 07:40-08:40 | **yes** | **656.5 (8.375)** | r19h 639.0 (8.604) | prod +2.7% (opt: +2.48%), proxy +2.8%; geomean 1.018 | **80.4%** (beat 817.0) | 603.8 / 98.6 | h70 u2n: k_dq kv-loop unroll x2 -- lab result confirmed on the job's card |
| 30 | deep | 09-28 08:40-10:30 | no | 656.4 (8.377) | r29 657.7 | prod -0.2%, proxy +0.2%, fast +1.0%; geomean 1.0033 | 80.0% (beat 820.7) | 605.9 / 100.4 | g89 null (see rounds/030/act.md) |
