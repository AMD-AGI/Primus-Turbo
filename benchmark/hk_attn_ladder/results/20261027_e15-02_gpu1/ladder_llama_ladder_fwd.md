Llama-3.1-8B attention forward (bf16, d128, GQA 32/8, causal, BSHD) on gfx1250, b4 sq8192 skv8192 hq32 hkv8 d128/128 causal bshd, c822521170f1 GPU None, 2026-10-27 08:04:04

| Rung | ms | TFLOP/s | Step | Step (cycles) | sclk MHz | Mcycles | Feature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 1.3386 | 1643 |  |  | 1475 | 1.972 |  |
| f00 | 2.5452 | 864 |  |  | 1644 | 4.187 | naive: 64 queries x 1 head per WG, register-mediated single-buffer K/V fill, 2 barriers per block, eager rescale |
| f01 | 2.9391 | 748 | 0.866x | 0.815x | 1750 | 5.138 | double-buffered LDS: block j+1's global loads in flight during block j's compute, 1 barrier per block |
| f02 | 2.2854 | 962 | 1.285x | 1.389x | 1616 | 3.700 | TDM fill: one 32-row tensor_load_to_lds per wave per block instead of the register path |
| f03 | 2.4314 | 905 | 0.938x | 0.758x | 2000 | 4.868 | GQA packing: the 4 q heads of one kv head per WG share every K / V tile and LDS operand |
| f04 | 2.1552 | 1020 | 1.129x | 1.168x | 1937 | 4.173 | longest-first workgroup order (causal load balance) |
| f05 | 2.1933 | 1003 | 0.983x | 0.974x | 1955 | 4.287 | 4-slot TDM ring with prefetch distance 2 (two blocks of compute cover each block's TDM latency) |
| f06 | 2.2406 | 982 | 0.979x | 0.970x | 1973 | 4.421 | split barrier: signal after S^T(j), wait at the top of block j+1 (softmax + PV absorb wave skew) |
| f07 | 2.2412 | 981 | 1.000x | 0.999x | 1976 | 4.429 | lock_simd: no WMMA arbitration pause at one wave per SIMD |
| f08 | 2.3802 | 924 | 0.941x | 0.933x | 1992 | 4.743 | lazy rescale: O / l rescaled only when a query's max grew by > 8 nat (uniform branch) |
| v0 | 1.2561 | 1751 | 1.899x | 2.507x | 1508 | 1.897 | first HK forward (Primus-Turbo fd4a1bd0): f08 + software pipelining (QK(c+1) || softmax(c) on 32-row chunks, double-buffered S^T), mask on the last stage only, Q through TDM, O through LDS, 320 KB LDS |
| v0_resc | 1.2165 | 1808 | 1.032x | 1.049x | 1484 | 1.806 | lazy-rescale test as one max + one VCC branch |
| v0_resc_desc | 1.1844 | 1857 | 1.023x | 1.041x | 1458 | 1.729 | TDM descriptor math inside the QK WMMA run, burst LDS bases one stage ahead |
| p1 | 1.1447 | 1921 | 1.037x | 1.066x | 1416 | 1.620 | K(c+2) and V(c+1) loads as one burst per chunk (Primus-Turbo cade7db6 pin) |
| v1_v0body | 1.2194 | 1804 | 0.937x | 0.902x | 1472 | 1.796 | v1 source with the v0 body (v0 grid) |
| v1_v0body_pairs | 1.2035 | 1827 | 1.014x | 1.026x | 1454 | 1.751 | persistent grid: zig-zag causal pairs, XCD-major |
| v1_lsum0 | 1.1355 | 1937 | 1.060x | 1.101x | 1401 | 1.591 | lean SM1 / FSPLIT, F1 interleave, MDS2 (v1 as validated) |
| v11_best | 1.1263 | 1953 | 1.008x | 1.009x | 1400 | 1.579 | v1.1: RESC + one burst per stage (MDS3) + 2-stage unroll (never run on a GPU) |
| x_desc | 1.2303 | 1788 |  |  | 1480 | 1.820 | DESC alone on v0 |
| x_merge | 1.2269 | 1793 |  |  | 1480 | 1.817 | MERGE alone on v0 |
| x_p1_tail | 1.1688 | 1882 |  |  | 1426 | 1.667 | negative: row sums moved into the PV run (TAIL) |
| x_v1_lsum1 | 1.1570 | 1901 |  |  | 1440 | 1.664 | negative: row sums by a WMMA with an all-ones operand (LSUM=1) |
| x_pp | 1.3392 | 1642 |  |  | 1476 | 1.978 | negative: 8-wave ping-pong (HK's CDNA4 attention schedule) |
| x_pp#null | 1.3383 | 1643 |  |  | 1473 | 1.969 | negative: 8-wave ping-pong (HK's CDNA4 attention schedule) |
