DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 13:44:39

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f3a | 0.7051 | 1950 |  |  |  | 2212 | 1.558 | 1174 | 827.71 | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| a_nosoft | 0.6018 | 2284 |  |  |  | 2215 | 1.333 | 1165 | 701.04 | ablation: softmax = P = bf16(S) only |
| a_nolds | 0.6468 | 2125 |  |  |  | 2218 | 1.434 | 1158 | 749.37 | ablation: no LDS reads in the tile regions |
| a_notdm | 0.6558 | 2096 |  |  |  | 2217 | 1.454 | 1160 | 761.16 | ablation: no K/V tile TDM copies |
| a_none | 0.4977 | 2762 |  |  |  | 2224 | 1.107 | 1141 | 568.35 | ablation: all three |
| a_none#null | 0.5083 | 2705 |  |  |  | 2223 | 1.130 | 1143 | 582.62 | ablation: all three |
