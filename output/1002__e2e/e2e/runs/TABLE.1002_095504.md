## 1. 每个进程、每个 arm 的稳态数字

| 进程 | arm | n | 单步 ms 中位数（IQR） | tps | 对 asm 单步比：相邻配对 / 周期 | 对 flyr29 单步比：相邻配对 / 周期 | attn fwd ms/步 | attn bwd ms/步 | FA ms/步 | FA 占单步 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| a0e2e_3arm_p1_1002_095504 | asm | 21 | 1350.6（1348.3–1353.6） | 24,262 | 1 | — | 34.75 | 166.54 | 201.40 | 14.89% |
| a0e2e_3arm_p1_1002_095504 | flyr29 | 19 | 1389.6（1383.8–1392.1） | 23,581 | 1.0269（n=16）/ 1.0279（n=4） | 1 | 43.86 | 214.76 | 258.57 | 18.66% |
| a0e2e_3arm_p1_1002_095504 | fly | 21 | 1349.4（1341.4–1353.6） | 24,283 | 0.9990（n=19）/ 0.9994（n=4） | 0.9720（n=17）/ 0.9728（n=4） | 43.88 | 162.15 | 206.03 | 15.27% |
| a0e2e_3arm_p2_1002_095504 | asm | 21 | 1350.6（1346.1–1351.9） | 24,262 | 1 | — | 34.63 | 166.41 | 201.01 | 14.87% |
| a0e2e_3arm_p2_1002_095504 | flyr29 | 21 | 1388.5（1379.0–1390.6） | 23,599 | 1.0273（n=19）/ 1.0280（n=4） | 1 | 43.82 | 215.41 | 259.67 | 18.69% |
| a0e2e_3arm_p2_1002_095504 | fly | 19 | 1349.9（1342.3–1353.6） | 24,275 | 0.9985（n=16）/ 1.0015（n=4） | 0.9732（n=17）/ 0.9734（n=4） | 43.91 | 162.62 | 206.17 | 15.27% |

B0 对照：fly/asm 单步比 1.0323 / 1.0328（B0 09-28，fwd r16 + bwd r29）；B0 fly fwd 48.0–49.3 / bwd 285.4–290.0，asm fwd 38.8–39.3 / bwd 233.1–236.7 ms/步（trace）。

## 2. 两两对比（相邻两步、arm 不同、都在稳态窗口内；差值 = 前者 − 后者）

| 进程 | 对比 | 相邻 n | 单步比（相邻 / 周期） | Δ单步 ms | ΔFA ms（events） | 其中 Δfwd | 其中 Δbwd | Δ单步 − ΔFA |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| a0e2e_3arm_p1_1002_095504 | flyr29 vs asm | 16 | 1.0269 / 1.0279 | +36.3 | +57.36 | +9.07 | +48.36 | -21.0 |
| a0e2e_3arm_p1_1002_095504 | fly vs asm | 19 | 0.9990 / 0.9994 | -1.4 | +4.89 | +9.18 | -4.50 | -6.3 |
| a0e2e_3arm_p1_1002_095504 | fly vs flyr29 | 17 | 0.9720 / 0.9728 | -39.1 | -53.04 | +0.05 | -53.08 | +14.0 |
| a0e2e_3arm_p2_1002_095504 | flyr29 vs asm | 19 | 1.0273 / 1.0280 | +36.9 | +58.98 | +9.16 | +49.79 | -22.1 |
| a0e2e_3arm_p2_1002_095504 | fly vs asm | 16 | 0.9985 / 1.0015 | -2.0 | +4.27 | +9.24 | -4.56 | -6.3 |
| a0e2e_3arm_p2_1002_095504 | fly vs flyr29 | 17 | 0.9732 / 0.9734 | -37.0 | -52.99 | +0.01 | -52.89 | +16.0 |

## 2b. 两个进程的一致性（计划 §4 判定 3：同一对比的单步比，两进程相差 ≤0.3%；B0 为 0.05%）

| 对比 | a0e2e_3arm_p1_1002_095504 相邻中位数（n） | a0e2e_3arm_p2_1002_095504 相邻中位数（n） | 最大相对差 | 判定 |
|---|--:|--:|--:|---|
| flyr29 / asm | 1.0269（16） | 1.0273（19） | 0.04% | 正常 |
| fly / asm | 0.9990（19） | 0.9985（16） | 0.05% | 正常 |
| fly / flyr29 | 0.9720（17） | 0.9732（17） | 0.13% | 正常 |

## 4. trace（kineto；A0 09-28 曾只记录 ~4 个 kernel/步，INVALID 的不用）

| 进程 | trace | 有效 | arm | GPU kernels | FA sum / wall ms | attn_fwd ms | attn_bwd sum / wall / span ms | gemm | gemm_aux | elementwise | idle |
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| a0e2e_3arm_p1_1002_095504 | iteration_11 | 否：9 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 9 | 0.0 / 0.0 | — | — / — / — | — | 1080.4 | 663.7 | 205.1 |
| a0e2e_3arm_p1_1002_095504 | iteration_22 | 否：9 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 9 | 0.0 / 0.0 | — | — / — / — | — | 775.0 | — | 188.0 |
| a0e2e_3arm_p1_1002_095504 | iteration_33 | 否：8 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 8 | 0.0 / 0.0 | — | — / — / — | — | 510.0 | 69.8 | 190.8 |
| a0e2e_3arm_p1_1002_095504 | iteration_44 | 否：15 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 15 | 0.0 / 0.0 | — | — / — / — | — | 1665.7 | 113.6 | 87.8 |
| a0e2e_3arm_p1_1002_095504 | iteration_55 | 否：8 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 8 | 0.0 / 0.0 | — | — / — / — | — | 590.2 | 99.4 | 632.4 |
| a0e2e_3arm_p1_1002_095504 | iteration_66 | 否：8 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 8 | 0.0 / 0.0 | — | — / — / — | — | 1312.6 | — | 361.5 |
| a0e2e_3arm_p1_1002_095504 | iteration_77 | 否：8 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 8 | 0.0 / 0.0 | — | — / — / — | — | 1403.2 | 415.4 | 126.0 |
| a0e2e_3arm_p1_1002_095504 | iteration_88 | 否：10 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 10 | 0.0 / 0.0 | — | — / — / — | — | 1416.2 | 440.5 | 83.3 |
| a0e2e_3arm_p2_1002_095504 | iteration_11 | 否：9 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 9 | 0.0 / 0.0 | — | — / — / — | — | 951.9 | 429.0 | 210.4 |
| a0e2e_3arm_p2_1002_095504 | iteration_22 | 否：5 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 5 | 0.0 / 0.0 | — | — / — / — | — | 709.2 | 100.8 | 508.6 |
| a0e2e_3arm_p2_1002_095504 | iteration_33 | 否：8 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 8 | 0.0 / 0.0 | — | — / — / — | — | 734.3 | 134.5 | 158.1 |
| a0e2e_3arm_p2_1002_095504 | iteration_44 | 否：12 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 12 | 0.0 / 0.0 | — | — / — / — | — | 942.2 | 681.4 | 92.3 |
| a0e2e_3arm_p2_1002_095504 | iteration_55 | 否：8 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 8 | 0.0 / 0.0 | — | — / — / — | — | 720.2 | 169.9 | 482.2 |
| a0e2e_3arm_p2_1002_095504 | iteration_66 | 否：10 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 10 | 0.0 / 0.0 | — | — / — / — | — | 389.7 | 126.0 | 838.5 |
| a0e2e_3arm_p2_1002_095504 | iteration_77 | 否：5 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 5 | 0.0 / 0.0 | — | — / — / — | — | 522.2 | 52.6 | 244.8 |
| a0e2e_3arm_p2_1002_095504 | iteration_88 | 否：4 GPU kernels < 1500; no kernel attributed to an e2e::attn_* range |  | 4 | 0.0 / 0.0 | — | — / — / — | — | 218.9 | 14.1 | 501.0 |

## 5. 运行健康

| 进程 | 有效 | rc | 步数 | 非有限步 | nkfix 事件 | BLAS 改写 | watchdog | memguard | 外来 KFD | 代码树 | dmesg 故障 / INFO | 新 CPU MCE | sclk 负载中位数 MHz | 功耗中位数 W | 峰值显存（进程级） | 墙钟 s | loss 首 / 末 |
|---|---|--:|--:|--:|---|--:|---|---|---|---|--:|--:|--:|--:|--:|--:|---|
| a0e2e_3arm_p1_1002_095504 | yes | 0 | 92 | 0 | [] | 0 | — | — | — | 基准（31 个文件） | 0 / 0 | 0 | 1510.5 | 1132.118 | 384.44 GiB（88.99%） | 160 | 12.25951 / 3.50734 |
| a0e2e_3arm_p2_1002_095504 | yes | 0 | 92 | 0 | [] | 0 | — | — | — | 同 a0e2e_3arm_p1_1002_095504（31 个文件） | 0 / 3 | 1 | 1500.5 | 1137.051 | 384.44 GiB（88.99%） | 158 | 12.25951 / 3.50642 |
