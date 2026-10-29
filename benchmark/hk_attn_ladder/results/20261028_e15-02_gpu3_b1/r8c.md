DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-29 02:45:06

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r8a | 2.4178 | 1478 |  |  |  | 1266 | 3.064 | - | - | dq: when (Skv - Sq) % 64 == 32 the last steady unit runs as a generic unit, whose burst handles the first masked row's partly stored pairs (fixes the r2d..r7 rect32 dQ read; production shapes unchanged) |
| r8b | 2.4197 | 1477 | r8a | 1.000x | 1.000x | 1266 | 3.064 | - | - | dK/dV asm patch v2: the Q^T tr16 units held up by in-flight export stores reading their registers re-targeted too (14 tuples, was 5), vm_vsrc waits recomputed (the store-data waits in the dV run go); bits unchanged |
| r8c | 2.4323 | 1470 | r8a | 0.994x | 0.998x | 1262 | 3.070 | - | - | dK/dV: S = Q K^T as two half-chains (even / odd dt) summed before the softmax, so no S/dP WMMA accumulates into the result of the WMMA right before it (15.9 vs 10.1 cycles per WMMA, micro/wmma_chain); asm patch v2 (msrc) |
| r8c#null | 2.4273 | 1473 |  |  |  | 1260 | 3.058 | - | - | dK/dV: S = Q K^T as two half-chains (even / odd dt) summed before the softmax, so no S/dP WMMA accumulates into the result of the WMMA right before it (15.9 vs 10.1 cycles per WMMA, micro/wmma_chain); asm patch v2 (msrc) |
