DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:48:50

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7963 | 1726 |  |  |  | 1426 | 1.137 | - | - |  |
| f18d | 0.7476 | 1839 |  |  |  | 1368 | 1.023 | - | - | f17a + DS_MAX 63 |
| f19f | 0.7452 | 1845 |  |  |  | 1363 | 1.016 | - | - | f18d + WMMA 32 + VALU 8 |
| f22a | 0.7471 | 1840 |  |  |  | 1366 | 1.021 | - | - | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
| f21c | 0.7403 | 1857 |  |  |  | 1363 | 1.008 | - | - | f21a + WMMA 32 + VALU 8 |
| f22b | 0.7397 | 1858 |  |  |  | 1360 | 1.007 | - | - | f22a + TILE_F |
| f22b#null | 0.7406 | 1856 |  |  |  | 1362 | 1.011 | - | - | f22a + TILE_F |
