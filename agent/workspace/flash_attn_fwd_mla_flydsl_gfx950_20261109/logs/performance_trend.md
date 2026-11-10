# Performance trend (GPU 0, Forward TFLOPS geomean; ratio = proto / aiter same GPU)

| round | status | description | full geomean | aiter geomean | ratio | b1_s4k | b1_s8k | b1_s16k | b1_s32k | b1_s64k | b2_s4k | b2_s16k | quick geomean | quick ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | BASELINE | bm256 kh4 loop_masks | 919.04 | 1140.64 | 0.8057 | 800.8 | 911.4 | 987.3 | 1002.6 | 1020.7 | 780.9 | 961.5 | 891.07 | 0.8044 |
| 2 | ROLLBACK | QK sched hints pace tail K reads (K_STEPS_QK > 8) | 913.58 | 1138.82 | 0.8022 | 794.1 | 901.1 | 978.9 | 990.6 | 1010.0 | 786.4 | 963.8 | - | - |
| 3 | ACCEPT | k_head 4 -> 2 (spill 105 -> 90) | 947.53 | 1140.09 | 0.8311 | 839.4 | 949.0 | 1002.0 | 1020.8 | 1044.4 | 822.8 | 979.4 | 922.04 | 0.8325 |
| 4 | ACCEPT | k_head 2 -> 1 (marginal) | 951.43 | 1139.05 | 0.8353 | 841.1 | 952.4 | 1010.3 | 1025.3 | 1043.6 | 826.3 | 986.3 | 925.21 | 0.8350 |
| 5 | ACCEPT | fold mem/PV barriers for 8-wave WGs | 996.81 | 1139.84 | 0.8745 | 882.5 | 986.9 | 1050.1 | 1082.5 | 1100.7 | 861.3 | 1041.9 | 968.15 | 0.8738 |
| 6 | ROLLBACK | reload Q from global for epilogue QKs | 913.70 | 1139.10 | 0.8021 | 726.9 | 891.4 | 994.5 | 1056.2 | 1088.2 | 731.6 | 981.2 | 870.99 | 0.7859 |
| 7 | ACCEPT | per-pack Q fptrunc (spill 90 -> 0) | 1058.67 | 1139.49 | 0.9291 | 1004.8 | 1056.4 | 1088.5 | 1095.2 | 1110.1 | 987.9 | 1074.1 | 1046.52 | 0.9440 |
| 8 | ROLLBACK | k_head 1 -> 2 (spill-free) | 1054.93 | 1140.19 | 0.9252 | 1003.4 | 1054.5 | 1084.9 | 1094.0 | 1100.3 | 983.4 | 1070.0 | 1041.78 | 0.9401 |
| 9 | ROLLBACK | no fence between row-sum and QK (loop C1/C5) | 1058.40 | 1140.32 | 0.9282 | 1005.2 | 1056.3 | 1094.8 | 1096.7 | 1102.8 | 985.1 | 1074.2 | 1045.66 | 0.9437 |
| 10 | ROLLBACK | P*V setprio on later wave half only | 1034.43 | 1140.02 | 0.9074 | 984.4 | 1034.3 | 1059.2 | 1067.2 | 1080.6 | 973.4 | 1047.0 | 1023.36 | 0.9232 |
| 11 | ROLLBACK | P*V setprio on first wave half only | 1044.56 | 1140.90 | 0.9156 | 993.8 | 1043.7 | 1071.6 | 1082.2 | 1092.6 | 976.8 | 1057.0 | 1032.52 | 0.9315 |
| 12 | ACCEPT (trade-off) | XCD-aware CTA order: 8 q-blocks of one head adjacent per XCD (L2 hit 0.37 -> 0.80) | 1104.55 | 1141.17 | 0.9679 | 965.3 | 1088.1 | 1162.5 | 1195.2 | 1216.5 | 969.7 | 1165.1 | 1072.63 | 0.9664 |
| 13 | ACCEPT | XCD order: last 8 q ranks back to longest-first (rank-major) order | 1121.20 | 1141.21 | 0.9825 | 998.8 | 1112.8 | 1170.8 | 1200.7 | 1216.6 | 998.8 | 1173.1 | 1091.82 | 0.9843 |
| 7 (n02 rebase) | REBASE | round 7 re-measured on smci355-ccs-aus-n02-25 (machine change; rows above are n04-25) | 1088.92 | 1204.47 | 0.9041 | 1066.0 | 1109.1 | 1128.9 | 1083.1 | 1113.3 | 1037.8 | 1086.8 | 1085.90 | 0.9295 |
| 13 (n02 rebase) | REBASE | round 13 re-measured on n02-25, same session | 1179.59 | 1207.66 | 0.9768 | 1079.8 | 1187.5 | 1207.3 | 1247.5 | 1281.7 | 1064.8 | 1205.8 | 1154.91 | 0.9837 |
| 14 | ACCEPT (trade-off) | sweep: XCD_Q_GROUP 8 -> 16 (tail 8); n02-25 | 1203.93 | 1204.14 | 0.9998 | 1063.8 | 1191.8 | 1285.2 | 1310.2 | 1326.0 | 1035.5 | 1250.7 | 1170.51 | 1.0027 |
| 15 | ACCEPT | XCD group per launch: 16, or 8 when < 16 ranks precede the tail (S4K); n02-25 | 1212.71 | 1208.92 | 1.0031 | 1084.6 | 1191.1 | 1284.2 | 1312.9 | 1329.0 | 1066.7 | 1249.3 | 1185.40 | 1.0089 |
| 15 (same session as 16) | REBASE | round 15 re-measured alternating with round 16 (2-run mean; quick: 3 rotations, no aiter) | 1208.81 | 1206.91 | 1.0016 | 1080.8 | 1188.1 | 1278.4 | 1307.9 | 1326.0 | 1063.6 | 1245.5 | 1183.7 | - |
| 16 | ACCEPT | QK on 16x16x32 MFMA: permlane16 swap at consumers, paired d chunks, middle-block K DMA swizzle (2-run mean; quick: 3 rotations, no aiter) | 1217.15 | 1201.92 | 1.0127 | 1079.3 | 1187.8 | 1284.3 | 1320.3 | 1353.7 | 1063.3 | 1264.7 | 1192.0 | - |
| 16 (same session as 17) | REBASE | round 16 re-measured alternating with round 17 (2-run mean; quick: 3 rotations, no aiter) | 1222.74 | 1208.66 | 1.0116 | 1085.9 | 1193.0 | 1290.0 | 1324.8 | 1358.9 | 1069.8 | 1269.7 | 1196.5 | - |
| 17 | ACCEPT | PV on 16x16x32 MFMA off the raw QK layout (no score-layout swaps), V tr reads with odd-row chunk ^ 2 swizzle (2-run mean; quick: 3 rotations, no aiter) | 1237.68 | 1206.27 | 1.0260 | 1090.8 | 1202.4 | 1311.3 | 1345.3 | 1376.7 | 1081.3 | 1291.6 | 1212.4 | - |
| 17 (same session as 18) | REBASE | round 17 re-measured alternating with round 18 (2-run mean; quick: 3 rotations, no aiter) | 1239.81 | 1207.03 | 1.0272 | 1094.2 | 1204.2 | 1317.4 | 1350.7 | 1377.4 | 1080.3 | 1290.6 | 1212.5 | - |
| 18 | ACCEPT (trade-off) | XCD q-group levels (32, 16, 8) over an 8-rank tail; sweep vs (16, 8) (1245.26); b1_s4k -1.06 %, b2_s4k -0.53 % (code generation, same CTA order) (2-run mean; quick: 3 rotations, no aiter) | 1247.73 | 1202.80 | 1.0374 | 1082.6 | 1223.8 | 1323.7 | 1366.6 | 1400.5 | 1074.6 | 1305.3 | 1216.4 | - |
| 18 (same session as 19) | REBASE | round 18 re-measured alternating with round 19 (2-run mean; quick: 3 rotations, no aiter) | 1250.26 | 1206.38 | 1.0364 | 1086.8 | 1235.2 | 1322.7 | 1368.1 | 1404.3 | 1074.7 | 1302.5 | 1216.0 | - |
| 19 | ROLLBACK | rewrite (32,16,8) XCD map as independent section maps + selects; CTA order unchanged; main-loop ISA identical to r18 (2-run mean; quick: 3 rotations, no aiter) | 1247.67 | 1205.16 | 1.0353 | 1082.2 | 1231.5 | 1323.1 | 1366.3 | 1401.5 | 1072.5 | 1299.8 | 1214.6 | - |
| 18 (same session as 20) | REBASE | round 18 re-measured alternating with round 20 (2-run mean; quick: 3 rotations, no aiter) | 1248.61 | 1204.99 | 1.0362 | 1081.1 | 1232.7 | 1316.3 | 1366.6 | 1402.7 | 1077.3 | 1306.2 | 1217.6 | - |
| 20 | ACCEPT | dense causal drain skips every dead-able diagonal half (QK max_m3 / max_m2 LO, P*V max_m4 HI / max_m3 / max_m2 LO); bitwise equal (2-run mean; quick: 3 rotations, no aiter) | 1263.08 | 1208.66 | 1.0450 | 1111.1 | 1248.2 | 1333.7 | 1370.0 | 1406.0 | 1097.1 | 1312.0 | 1235.2 | - |
