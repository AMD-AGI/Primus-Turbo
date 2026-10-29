DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-29 02:38:57

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r7 | 2.4255 | 1474 |  |  |  | 1265 | 3.069 | - | - | redundant causal test dropped from the masked dK/dV loop |
| r8a | 2.4245 | 1474 | r7 | 1.000x | 0.998x | 1268 | 3.072 | - | - | dq: when (Skv - Sq) % 64 == 32 the last steady unit runs as a generic unit, whose burst handles the first masked row's partly stored pairs (fixes the r2d..r7 rect32 dQ read; production shapes unchanged) |
| r8b | 2.4169 | 1479 | r8a | 1.003x | 1.005x | 1266 | 3.061 | - | - | dK/dV asm patch v2: the Q^T tr16 units held up by in-flight export stores reading their registers re-targeted too (14 tuples, was 5), vm_vsrc waits recomputed (the store-data waits in the dV run go); bits unchanged |
| r8b#null | 2.4196 | 1477 |  |  |  | 1264 | 3.062 | - | - | dK/dV asm patch v2: the Q^T tr16 units held up by in-flight export stores reading their registers re-targeted too (14 tuples, was 5), vm_vsrc waits recomputed (the store-data waits in the dV run go); bits unchanged |
