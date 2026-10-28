DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-28 02:54:26

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r8a | 2.3615 | 1514 |  |  |  | 1302 | 3.074 | - | - | dq: when (Skv - Sq) % 64 == 32 the last steady unit runs as a generic unit, whose burst handles the first masked row's partly stored pairs (fixes the r2d..r7 rect32 dQ read; production shapes unchanged) |
| r9x | 2.9509 | 1211 |  |  |  | 1588 | 4.695 | - | - | wave-specialized dK/dV v2: producers pipeline B(k-1) (softmax, exchange stores) with A(k) (S/dP WMMAs, causal mask as the S C-init), consumers lag 2 steps and export dS^T, split barriers, 5-stage ring |
| r9y | 2.9411 | 1215 | r9x | 1.003x | 1.003x | 1590 | 4.685 | - | - | wave-specialized v2 with TDM prefetch distance 3 (6-stage ring) |
| ws_nop | 2.2865 | 1563 | r9x | 1.286x | 1.160x | 1765 | 4.044 | - | - | ablation of r9x: producers skip S/dP + softmax |
| ws_nop3 | 2.2539 | 1586 | r9x | 1.310x | 1.171x | 1777 | 4.005 | - | - | ablation of r9y: producers skip S/dP + softmax |
