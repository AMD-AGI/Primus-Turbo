DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b1 sq640 skv640 hq16 hkv16 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:11:44

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.0151 | 139 |  |  |  | 2153 | 0.033 | 1996 | 30.11 |  |
| f0 | 0.0156 | 135 |  |  |  | 2152 | 0.034 | 1990 | 30.93 | the shipped HK forward of dev/lhz/dsv3-attn-hk-merge (a08-1 E2: persistent static schedule, 4-stage ring, generated tile regions) |
| f0#null | 0.0159 | 132 |  |  |  | 2153 | 0.034 | 1984 | 31.52 | the shipped HK forward of dev/lhz/dsv3-attn-hk-merge (a08-1 E2: persistent static schedule, 4-stage ring, generated tile regions) |
