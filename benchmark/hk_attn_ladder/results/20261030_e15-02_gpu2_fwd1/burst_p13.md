DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 15:02:45

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7008 | 1962 |  |  |  | 2211 | 1.549 | 1178 | 825.74 |  |
| f14a | 0.6724 | 2044 |  |  |  | 2212 | 1.487 | 1177 | 790.78 | XBLK: ring slots n..n+2 load the next block's tiles 0..2 (no 1-row copies); the epilogue issues only the next Q |
| f15a | 0.7706 | 1784 | f14a | 0.874x | 0.874x | 2212 | 1.704 | 1176 | 906.15 | --issue hw: the ATT-measured issue model (1 issue/cycle, LDS op holds others 4 cycles, WMMA 8 apart, RESC 30); DS_MAX on consumer-retired loads |
| f15b | 0.7722 | 1780 |  |  |  | 2212 | 1.708 | 1175 | 909.66 | f15a with the DS_MAX bound on modelled completion (+ s_wait_dscnt where the consumer count needs it) |
| f15c | 0.7755 | 1773 |  |  |  | 2212 | 1.715 | 1177 | 913.17 | f15b with LDS loads completing 8 cycles apart at the most (shared LDS bandwidth) |
| f15c#null | 0.7761 | 1771 |  |  |  | 2212 | 1.717 | 1176 | 914.17 | f15b with LDS loads completing 8 cycles apart at the most (shared LDS bandwidth) |
