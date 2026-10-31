DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:31:23

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7961 | 1727 |  |  |  | 1429 | 1.138 | - | - |  |
| f3a | 0.7815 | 1759 |  |  |  | 1426 | 1.115 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f4a | 0.7797 | 1763 | f3a | 1.002x | 1.010x | 1414 | 1.105 | - | - | persistent workgroups on ticket counters (FlyDSL 917e240b's per-XCD groups, stealing, shortest blocks last, self-reset) |
| f4a#null | 0.7808 | 1761 |  |  |  | 1415 | 1.107 | - | - | persistent workgroups on ticket counters (FlyDSL 917e240b's per-XCD groups, stealing, shortest blocks last, self-reset) |
