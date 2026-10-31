DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:30:05

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7966 | 1726 |  |  |  | 1430 | 1.140 | - | - |  |
| f17a | 0.7490 | 1836 |  |  |  | 1370 | 1.025 | - | - | --sgpr-lat 20 (VALU-written SGPR to SALU: 20 cycles) + --resc branch2 (the RESC s_cmp scheduled 9 cycles ahead) |
| f18a | 0.7503 | 1832 |  |  |  | 1370 | 1.028 | - | - | f17a + --wait-merge 2 (pairs of s_wait_dscnt merged) |
| f18b | 0.7491 | 1835 | f17a | 1.002x | 1.001x | 1370 | 1.026 | - | - | f17a + --wait-merge 4 |
| f18c | 0.7505 | 1832 |  |  |  | 1374 | 1.031 | - | - | f17a + --wait-merge 8 |
| f18d | 0.7477 | 1839 |  |  |  | 1370 | 1.023 | - | - | f17a + DS_MAX 63 |
| f18e | 0.7481 | 1838 |  |  |  | 1371 | 1.025 | - | - | f17a + --wait-merge 4 + DS_MAX 63 |
| f18e#null | 0.7477 | 1839 |  |  |  | 1370 | 1.025 | - | - | f17a + --wait-merge 4 + DS_MAX 63 |
