DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 13:58:41

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7139 | 1926 |  |  |  | 2211 | 1.579 | 1177 | 839.40 |  |
| f3a | 0.7056 | 1948 |  |  |  | 2212 | 1.561 | 1176 | 828.48 | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f8a | 0.7104 | 1935 | f3a | 0.990x | 0.991x | 2212 | 1.571 | 1175 | 834.37 | fwdgen2: one generated LOOP region per block -- TDM issue, tile barriers and the tile counter in generated code, no shell code between tiles |
| f8a#null | 0.7106 | 1935 |  |  |  | 2212 | 1.571 | 1176 | 835.73 | fwdgen2: one generated LOOP region per block -- TDM issue, tile barriers and the tile counter in generated code, no shell code between tiles |
