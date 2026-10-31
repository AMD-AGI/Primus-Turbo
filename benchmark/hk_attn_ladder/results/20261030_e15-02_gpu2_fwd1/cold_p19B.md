DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:50:20

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7978 | 1723 |  |  |  | 1426 | 1.137 | - | - |  |
| f18d | 0.7477 | 1839 |  |  |  | 1365 | 1.020 | - | - | f17a + DS_MAX 63 |
| f21c | 0.8002 | 1718 |  |  |  | 1452 | 1.162 | - | - | f21a + WMMA 32 + VALU 8 |
| f22b | 0.7970 | 1725 |  |  |  | 1440 | 1.147 | - | - | f22a + TILE_F |
| f19f | 0.7481 | 1838 |  |  |  | 1364 | 1.019 | - | - | f18d + WMMA 32 + VALU 8 |
| f22a | 0.7465 | 1841 |  |  |  | 1360 | 1.016 | - | - | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
| f22a#null | 0.7463 | 1842 |  |  |  | 1362 | 1.016 | - | - | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
