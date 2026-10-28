DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-28 02:38:12

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r8a | 2.3467 | 1523 |  |  |  | 1297 | 3.053 | - | - | dq: when (Skv - Sq) % 64 == 32 the last steady unit runs as a generic unit, whose burst handles the first masked row's partly stored pairs (fixes the r2d..r7 rect32 dQ read; production shapes unchanged) |
| r9w | 2.9477 | 1213 | r8a | 0.799x | 0.647x | 1604 | 4.723 | - | - | dK/dV wave-specialized: 4-wave workgroups, producer waves (S/dP, softmax, dS export) and consumer waves (dV/dK, TDM) on different SIMDs, P/dS through a double-buffered LDS exchange, 4-stage ring |
| r9w#null | 2.9505 | 1211 |  |  |  | 1603 | 4.725 | - | - | dK/dV wave-specialized: 4-wave workgroups, producer waves (S/dP, softmax, dS export) and consumer waves (dV/dK, TDM) on different SIMDs, P/dS through a double-buffered LDS exchange, 4-stage ring |
