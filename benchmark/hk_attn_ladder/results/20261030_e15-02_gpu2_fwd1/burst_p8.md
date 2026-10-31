DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 14:02:44

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f3a | 0.7075 | 1943 |  |  |  | 2212 | 1.564 | 1174 | 829.84 | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f8x | 0.7049 | 1950 |  |  |  | 2212 | 1.559 | 1174 | 827.76 | f3a's flags on the patch-08 tree: the scheduler's LDS-in-flight model now retires loads at completion (new schedule) |
| f8a | 0.7122 | 1930 | f3a | 0.993x | 0.993x | 2212 | 1.575 | 1174 | 836.09 | fwdgen2: one generated LOOP region per block -- TDM issue, tile barriers and the tile counter in generated code, no shell code between tiles |
| f8b | 0.7581 | 1813 | f8a | 0.938x | 0.938x | 2212 | 1.677 | 1176 | 889.87 | + K / Q(0) of the next tile read right after the barrier wait (cross-tile prefetch) |
| f8b#null | 0.7587 | 1812 |  |  |  | 2212 | 1.678 | 1176 | 891.81 | + K / Q(0) of the next tile read right after the barrier wait (cross-tile prefetch) |
