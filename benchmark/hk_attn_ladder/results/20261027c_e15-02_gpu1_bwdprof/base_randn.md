DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, c822521170f1 GPU None, 2026-10-27 15:05:30

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 3.5585 | 1004 |  |  |  | 1515 | 5.391 | - | - |  |
| hk_v0 | 2.8046 | 1274 |  |  |  | 1428 | 4.005 | - | - | first HK backward (d3590be4): 48 kv rows per wave on 2-wave workgroups (each Q / dO operand feeds 3 WMMAs), a 3-stage Q / dO ring with per-q-tile readback, dS^T exported in whole 128-B lines (DPP regroup) to a causal-packed layout (4.33 GB), 384-query dQ workgroups on a K ring, lock_simd |
| r7 | 2.4721 | 1446 | hk_v0 | 1.135x | 1.295x | 1251 | 3.093 | - | - | redundant causal test dropped from the masked dK/dV loop |
| r7#null | 2.4754 | 1444 |  |  |  | 1252 | 3.098 | - | - | redundant causal test dropped from the masked dK/dV loop |
