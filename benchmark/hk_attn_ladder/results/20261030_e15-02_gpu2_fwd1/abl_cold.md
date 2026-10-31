DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:43:49

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f3a | 0.7794 | 1764 |  |  |  | 1428 | 1.113 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| a_nosoft | 0.6593 | 2085 |  |  |  | 1424 | 0.936 | - | - | ablation: softmax = P = bf16(S) only |
| a_nolds | 0.6547 | 2100 |  |  |  | 1614 | 1.058 | - | - | ablation: no LDS reads in the tile regions |
| a_notdm | 0.6698 | 2052 |  |  |  | 1582 | 1.060 | - | - | ablation: no K/V tile TDM copies |
| a_none | 0.4469 | 3076 |  |  |  | 1922 | 0.860 | - | - | ablation: all three |
| a_none#null | 0.4663 | 2948 |  |  |  | 1850 | 0.861 | - | - | ablation: all three |
