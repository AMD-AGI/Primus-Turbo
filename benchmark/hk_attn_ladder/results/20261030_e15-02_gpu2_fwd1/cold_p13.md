DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:01:45

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7956 | 1728 |  |  |  | 1427 | 1.137 | - | - |  |
| f14a | 0.7597 | 1810 |  |  |  | 1397 | 1.061 | - | - | XBLK: ring slots n..n+2 load the next block's tiles 0..2 (no 1-row copies); the epilogue issues only the next Q |
| f15a | 0.8356 | 1645 | f14a | 0.911x | 0.840x | 1515 | 1.266 | - | - | --issue hw: the ATT-measured issue model (1 issue/cycle, LDS op holds others 4 cycles, WMMA 8 apart, RESC 30); DS_MAX on consumer-retired loads |
| f15b | 0.8331 | 1650 |  |  |  | 1514 | 1.261 | - | - | f15a with the DS_MAX bound on modelled completion (+ s_wait_dscnt where the consumer count needs it) |
| f15c | 0.8414 | 1634 |  |  |  | 1518 | 1.277 | - | - | f15b with LDS loads completing 8 cycles apart at the most (shared LDS bandwidth) |
| f15c#null | 0.8415 | 1634 |  |  |  | 1518 | 1.276 | - | - | f15b with LDS loads completing 8 cycles apart at the most (shared LDS bandwidth) |
