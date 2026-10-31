DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, zeros inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:45:32

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f3a | 0.5589 | 2460 |  |  |  | 1940 | 1.083 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| a_nosoft | 0.4688 | 2933 |  |  |  | 1977 | 0.927 | - | - | ablation: softmax = P = bf16(S) only |
| a_nolds | 0.5466 | 2515 |  |  |  | 1930 | 1.055 | - | - | ablation: no LDS reads in the tile regions |
| a_notdm | 0.5405 | 2544 |  |  |  | 1938 | 1.048 | - | - | ablation: no K/V tile TDM copies |
| a_none | 0.4323 | 3180 |  |  |  | 1998 | 0.866 | 2139 | 912.68 | ablation: all three |
| a_none#null | 0.4302 | 3196 |  |  |  | 2008 | 0.865 | 2144 | 915.52 | ablation: all three |
