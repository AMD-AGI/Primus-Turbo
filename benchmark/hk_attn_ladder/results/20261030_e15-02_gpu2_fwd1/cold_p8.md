DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 14:01:51

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f3a | 0.7808 | 1761 |  |  |  | 1429 | 1.116 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f8x | 0.7769 | 1770 |  |  |  | 1424 | 1.108 | - | - | f3a's flags on the patch-08 tree: the scheduler's LDS-in-flight model now retires loads at completion (new schedule) |
| f8a | 0.7865 | 1748 | f3a | 0.993x | 0.988x | 1436 | 1.129 | - | - | fwdgen2: one generated LOOP region per block -- TDM issue, tile barriers and the tile counter in generated code, no shell code between tiles |
| f8b | 0.8192 | 1678 | f8a | 0.958x | 0.929x | 1484 | 1.215 | - | - | + K / Q(0) of the next tile read right after the barrier wait (cross-tile prefetch) |
| f8b#null | 0.8227 | 1671 |  |  |  | 1486 | 1.221 | - | - | + K / Q(0) of the next tile read right after the barrier wait (cross-tile prefetch) |
