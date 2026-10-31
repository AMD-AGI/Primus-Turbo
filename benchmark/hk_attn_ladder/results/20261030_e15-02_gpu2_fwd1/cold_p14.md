DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:16:42

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7960 | 1727 |  |  |  | 1427 | 1.136 | - | - |  |
| f14a | 0.7653 | 1796 |  |  |  | 1396 | 1.067 | - | - | XBLK: ring slots n..n+2 load the next block's tiles 0..2 (no 1-row copies); the epilogue issues only the next Q |
| f16a | 0.7769 | 1770 | f14a | 0.986x | 0.962x | 1430 | 1.111 | - | - | --issue m1ds: model1 + an LDS op holds the next non-LDS issue 4 cycles (no LDS op that pushes a ready WMMA) |
| f16b | 0.7738 | 1777 |  |  |  | 1424 | 1.102 | - | - | f16a with DS_RATE 4 |
| f16c | 0.7796 | 1763 |  |  |  | 1430 | 1.117 | - | - | f16a with DS_RATE 8 |
| f16d | 0.7753 | 1773 |  |  |  | 1426 | 1.106 | - | - | f16a with an LDS block of 3 cycles |
| f16d#null | 0.7767 | 1770 |  |  |  | 1428 | 1.109 | - | - | f16a with an LDS block of 3 cycles |
