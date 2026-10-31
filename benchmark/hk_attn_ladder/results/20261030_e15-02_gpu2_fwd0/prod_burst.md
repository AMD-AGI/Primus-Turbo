DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 13:09:42

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7045 | 1951 |  |  |  | 2211 | 1.558 | 1177 | 829.23 |  |
| f0 | 0.7174 | 1916 |  |  |  | 2211 | 1.586 | 1177 | 844.66 | the shipped HK forward of dev/lhz/dsv3-attn-hk-merge (a08-1 E2: persistent static schedule, 4-stage ring, generated tile regions) |
| f0#null | 0.7160 | 1920 |  |  |  | 2211 | 1.583 | 1178 | 844.30 | the shipped HK forward of dev/lhz/dsv3-attn-hk-merge (a08-1 E2: persistent static schedule, 4-stage ring, generated tile regions) |
