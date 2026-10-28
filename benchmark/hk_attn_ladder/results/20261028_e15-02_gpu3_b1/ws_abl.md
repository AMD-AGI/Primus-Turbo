DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-28 02:52:15

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r8a | 2.3568 | 1517 |  |  |  | 1300 | 3.065 | - | - | dq: when (Skv - Sq) % 64 == 32 the last steady unit runs as a generic unit, whose burst handles the first masked row's partly stored pairs (fixes the r2d..r7 rect32 dQ read; production shapes unchanged) |
| r9x | 2.9589 | 1208 |  |  |  | 1592 | 4.720 | - | - | wave-specialized dK/dV v2: producers pipeline B(k-1) (softmax, exchange stores) with A(k) (S/dP WMMAs, causal mask as the S C-init), consumers lag 2 steps and export dS^T, split barriers, 5-stage ring |
| ws_nop | 2.2943 | 1558 | r9x | 1.290x | 1.166x | 1766 | 4.045 | - | - | ablation of r9x: producers skip S/dP + softmax |
| ws_noc | 2.6801 | 1334 | r9x | 1.101x | 1.011x | 1740 | 4.663 | - | - | ablation of r9x: consumers skip dK + export |
