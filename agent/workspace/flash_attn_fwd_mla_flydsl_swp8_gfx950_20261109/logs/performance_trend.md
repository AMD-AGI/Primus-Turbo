# Performance trend (GPU 0, Forward TFLOPS geomean; ratio = swp8 / aiter same GPU)

Forward-only campaign: Bwd columns omitted, combined-step metric = Fwd geomean.

| round | status | description | full geomean | aiter geomean | ratio | b1_s4k | b1_s8k | b1_s16k | b1_s32k | b1_s64k | b2_s4k | b2_s16k | vs baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | BASELINE | swp8 v2 (aiter 8-stage port, stagger + hints); round 7 = 1057.20 | 965.66 | 1140.55 | 0.8467 | 846.6 | 963.7 | 1013.4 | 1047.8 | 1062.2 | 846.5 | 1005.2 | — |
| 1 | ROLLBACK | QK zero seed + shift in s5 | 964.14 | 1139.58 | 0.8461 | 858.1 | 963.2 | 1010.9 | 1039.5 | 1043.7 | 851.8 | 1002.9 | -0.157 % |
| 2 | ACCEPT (trade-off) | i32 while-loop counter, SALU loop exit + K clamp | 978.41 | 1140.88 | 0.8576 | 871.3 | 980.7 | 1028.1 | 1053.5 | 1053.0 | 863.1 | 1020.5 | +1.321 % |
| 3 | ROLLBACK | zero seed + packed v_pk_add_f32 shift | 949.89 | 1139.39 | 0.8337 | 854.0 | 950.3 | 988.5 | 1020.8 | 1035.8 | 841.0 | 978.1 | -1.633 % |
| 4 | ACCEPT | K(t+2) DMA 1 in s4 + 2 in s6 | 987.84 | 1139.57 | 0.8669 | 875.3 | 985.5 | 1034.8 | 1068.1 | 1083.1 | 865.7 | 1026.7 | +2.297 % |
| 5 | ROLLBACK | rowsum moved s3 -> s5 (exp/cast sank into s4) | 929.23 | 1139.22 | 0.8157 | 817.8 | 926.0 | 985.3 | 1006.1 | 1007.0 | 821.4 | 963.4 | -3.773 % |
| 6 | ACCEPT | rowsum s3 -> s5 + P packs pinned in s3 | 992.33 | 1136.42 | 0.8732 | 877.0 | 982.6 | 1040.0 | 1077.5 | 1091.5 | 870.1 | 1033.1 | +2.762 % |
