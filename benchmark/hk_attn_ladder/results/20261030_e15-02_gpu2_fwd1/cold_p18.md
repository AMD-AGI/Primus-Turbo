DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:41:58

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7984 | 1722 |  |  |  | 1428 | 1.139 | - | - |  |
| f18d | 0.7488 | 1836 |  |  |  | 1368 | 1.024 | - | - | f17a + DS_MAX 63 |
| f19f | 0.7457 | 1844 |  |  |  | 1364 | 1.018 | - | - | f18d + WMMA 32 + VALU 8 |
| f20a | 0.7459 | 1843 |  |  |  | 1366 | 1.018 | - | - | f18d + incremental TDM descriptors in the TILE_C loop |
| f21a | 0.7439 | 1848 |  |  |  | 1366 | 1.017 | - | - | TILE_F: tile 0 of a block with ncc > 0 = INIT's set-up + the body, no no-op prefix; the TILE_C loop from tile 1 |
| f21b | 0.7450 | 1845 |  |  |  | 1364 | 1.016 | - | - | f21a + VALU result latency 8 |
| f21c | 0.7427 | 1851 |  |  |  | 1362 | 1.012 | - | - | f21a + WMMA 32 + VALU 8 |
| f21c#null | 0.7997 | 1719 |  |  |  | 1450 | 1.159 | - | - | f21a + WMMA 32 + VALU 8 |
