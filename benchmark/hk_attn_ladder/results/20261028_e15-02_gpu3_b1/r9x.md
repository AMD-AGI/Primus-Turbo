DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-28 02:48:10

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r8a | 2.3521 | 1520 |  |  |  | 1298 | 3.050 | - | - | dq: when (Skv - Sq) % 64 == 32 the last steady unit runs as a generic unit, whose burst handles the first masked row's partly stored pairs (fixes the r2d..r7 rect32 dQ read; production shapes unchanged) |
| r9w | 2.9559 | 1209 | r8a | 0.795x | 0.646x | 1601 | 4.728 | - | - | dK/dV wave-specialized: 4-wave workgroups, producer waves (S/dP, softmax, dS export) and consumer waves (dV/dK, TDM) on different SIMDs, P/dS through a double-buffered LDS exchange, 4-stage ring |
| r9x | 2.9326 | 1219 | r9w | 1.007x | 1.010x | 1594 | 4.672 | - | - | wave-specialized dK/dV v2: producers pipeline B(k-1) (softmax, exchange stores) with A(k) (S/dP WMMAs, causal mask as the S C-init), consumers lag 2 steps and export dS^T, split barriers, 5-stage ring |
| r9x#null | 2.9341 | 1218 |  |  |  | 1588 | 4.658 | - | - | wave-specialized dK/dV v2: producers pipeline B(k-1) (softmax, exchange stores) with A(k) (S/dP WMMAs, causal mask as the S C-init), consumers lag 2 steps and export dS^T, split barriers, 5-stage ring |
