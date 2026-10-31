DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 16:01:20

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7983 | 1722 |  |  |  | 1431 | 1.142 | - | - |  |
| f22a | 0.7466 | 1841 |  |  |  | 1368 | 1.021 | - | - | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
| f24a | 0.7406 | 1856 | f22a | 1.007x | 1.012x | 1360 | 1.008 | - | - | FINAL_A: one scheduled region -- the last prefix, then out / lse of all 4 q tiles stored to the staging by the generated code |
| f24b | 0.7375 | 1864 |  |  |  | 1358 | 1.001 | - | - | f24a + TILE_F |
| f24b#null | 0.7360 | 1868 |  |  |  | 1355 | 1.004 | - | - | f24a + TILE_F |
