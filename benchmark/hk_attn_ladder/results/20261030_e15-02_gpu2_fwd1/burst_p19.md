DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 15:51:33

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7176 | 1916 |  |  |  | 2212 | 1.587 | 1177 | 844.93 |  |
| f18d | 0.6713 | 2048 |  |  |  | 2212 | 1.485 | 1174 | 789.26 | f17a + DS_MAX 63 |
| f19f | 0.6673 | 2060 |  |  |  | 2212 | 1.476 | 1175 | 784.21 | f18d + WMMA 32 + VALU 8 |
| f22a | 0.6673 | 2060 |  |  |  | 2212 | 1.476 | 1175 | 783.63 | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
| f21c | 0.6630 | 2073 |  |  |  | 2212 | 1.466 | 1175 | 778.96 | f21a + WMMA 32 + VALU 8 |
| f22b | 0.6633 | 2072 |  |  |  | 2212 | 1.467 | 1175 | 779.84 | f22a + TILE_F |
| f22b#null | 0.6621 | 2076 |  |  |  | 2212 | 1.465 | 1174 | 777.93 | f22a + TILE_F |
