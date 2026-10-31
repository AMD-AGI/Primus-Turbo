DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b1 sq4096 skv4096 hq16 hkv16 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:12:26

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.0748 | 1149 |  |  |  | 1968 | 0.147 | - | - |  |
| f0 | 0.0727 | 1182 |  |  |  | 1958 | 0.142 | - | - | the shipped HK forward of dev/lhz/dsv3-attn-hk-merge (a08-1 E2: persistent static schedule, 4-stage ring, generated tile regions) |
| f0#null | 0.0722 | 1190 |  |  |  | 1958 | 0.141 | - | - | the shipped HK forward of dev/lhz/dsv3-attn-hk-merge (a08-1 E2: persistent static schedule, 4-stage ring, generated tile regions) |
