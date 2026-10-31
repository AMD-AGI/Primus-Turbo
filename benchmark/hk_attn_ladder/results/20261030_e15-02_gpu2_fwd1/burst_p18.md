DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 15:43:16

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7100 | 1936 |  |  |  | 2212 | 1.570 | 1175 | 835.00 |  |
| f18d | 0.6632 | 2073 |  |  |  | 2212 | 1.467 | 1173 | 778.37 | f17a + DS_MAX 63 |
| f19f | 0.6608 | 2081 |  |  |  | 2212 | 1.461 | 1174 | 775.81 | f18d + WMMA 32 + VALU 8 |
| f20a | 0.6595 | 2084 |  |  |  | 2212 | 1.459 | 1172 | 772.82 | f18d + incremental TDM descriptors in the TILE_C loop |
| f21a | 0.6584 | 2088 |  |  |  | 2212 | 1.456 | 1173 | 773.43 | TILE_F: tile 0 of a block with ncc > 0 = INIT's set-up + the body, no no-op prefix; the TILE_C loop from tile 1 |
| f21b | 0.6548 | 2099 |  |  |  | 2212 | 1.448 | 1174 | 768.78 | f21a + VALU result latency 8 |
| f21c | 0.6564 | 2094 |  |  |  | 2212 | 1.452 | 1174 | 770.39 | f21a + WMMA 32 + VALU 8 |
| f21c#null | 0.6552 | 2098 |  |  |  | 2212 | 1.449 | 1174 | 769.96 | f21a + WMMA 32 + VALU 8 |
