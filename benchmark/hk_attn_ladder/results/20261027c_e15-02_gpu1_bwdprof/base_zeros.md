DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, zeros inputs, c822521170f1 GPU None, 2026-10-27 15:06:18

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 2.6780 | 1335 |  |  |  | 2022 | 5.419 | - | - |  |
| hk_v0 | 2.0767 | 1721 |  |  |  | 1974 | 4.097 | - | - | first HK backward (d3590be4): 48 kv rows per wave on 2-wave workgroups (each Q / dO operand feeds 3 WMMAs), a 3-stage Q / dO ring with per-q-tile readback, dS^T exported in whole 128-B lines (DPP regroup) to a causal-packed layout (4.33 GB), 384-query dQ workgroups on a K ring, lock_simd |
| r7 | 1.7360 | 2059 | hk_v0 | 1.194x | 1.289x | 1831 | 3.176 | - | - | redundant causal test dropped from the masked dK/dV loop |
| r7#null | 1.7422 | 2052 |  |  |  | 1826 | 3.183 | - | - | redundant causal test dropped from the masked dK/dV loop |
