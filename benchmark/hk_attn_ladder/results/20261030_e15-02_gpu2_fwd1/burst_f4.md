DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 13:32:04

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7065 | 1946 |  |  |  | 2211 | 1.560 | 1177 | 835.82 |  |
| f3a | 0.6997 | 1965 |  |  |  | 2211 | 1.547 | 1175 | 821.79 | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f4a | 0.6967 | 1973 | f3a | 0.999x | 0.999x | 2211 | 1.541 | 1176 | 823.33 | persistent workgroups on ticket counters (FlyDSL 917e240b's per-XCD groups, stealing, shortest blocks last, self-reset) |
| f4a#null | 0.6948 | 1978 |  |  |  | 2211 | 1.536 | 1177 | 818.91 | persistent workgroups on ticket counters (FlyDSL 917e240b's per-XCD groups, stealing, shortest blocks last, self-reset) |
