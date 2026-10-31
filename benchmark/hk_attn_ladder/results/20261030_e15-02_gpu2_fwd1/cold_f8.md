DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:57:59

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7970 | 1725 |  |  |  | 1429 | 1.140 | - | - |  |
| f3a | 0.7797 | 1763 |  |  |  | 1426 | 1.112 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f8a | 0.7865 | 1748 | f3a | 0.991x | 0.984x | 1434 | 1.128 | - | - | fwdgen2: one generated LOOP region per block -- TDM issue, tile barriers and the tile counter in generated code, no shell code between tiles |
| f8a#null | 0.7882 | 1744 |  |  |  | 1432 | 1.129 | - | - | fwdgen2: one generated LOOP region per block -- TDM issue, tile barriers and the tile counter in generated code, no shell code between tiles |
