DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 15:38:39

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7068 | 1945 |  |  |  | 2211 | 1.563 | 1175 | 832.80 |  |
| f18d | 0.6618 | 2077 |  |  |  | 2212 | 1.464 | 1174 | 775.99 | f17a + DS_MAX 63 |
| f19a | 0.6548 | 2099 |  |  |  | 2212 | 1.449 | 1172 | 770.33 | f18d + WMMA result latency 32 in the scheduler |
| f19b | 0.6664 | 2063 |  |  |  | 2212 | 1.474 | 1172 | 781.07 | f18d + LDS load latency 400 |
| f19c | 0.6643 | 2070 |  |  |  | 2212 | 1.469 | 1174 | 781.17 | f18d + LDS load latency 220 |
| f19d | 0.6621 | 2076 |  |  |  | 2212 | 1.464 | 1173 | 778.58 | f18d + WMMA 32 + LDS 400 |
| f19e | 0.6546 | 2100 |  |  |  | 2212 | 1.448 | 1173 | 772.19 | f18d + VALU result latency 8 |
| f19e#null | 0.6576 | 2091 |  |  |  | 2212 | 1.454 | 1173 | 772.97 | f18d + VALU result latency 8 |
