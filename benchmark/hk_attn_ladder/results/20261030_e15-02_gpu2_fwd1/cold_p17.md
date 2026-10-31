DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:37:21

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7981 | 1722 |  |  |  | 1430 | 1.141 | - | - |  |
| f18d | 0.7480 | 1838 |  |  |  | 1370 | 1.025 | - | - | f17a + DS_MAX 63 |
| f19a | 0.7490 | 1835 |  |  |  | 1370 | 1.026 | - | - | f18d + WMMA result latency 32 in the scheduler |
| f19b | 0.7510 | 1831 |  |  |  | 1372 | 1.030 | - | - | f18d + LDS load latency 400 |
| f19c | 0.7513 | 1830 |  |  |  | 1372 | 1.029 | - | - | f18d + LDS load latency 220 |
| f19d | 0.7503 | 1832 |  |  |  | 1370 | 1.027 | - | - | f18d + WMMA 32 + LDS 400 |
| f19e | 0.7469 | 1841 |  |  |  | 1366 | 1.021 | - | - | f18d + VALU result latency 8 |
| f19e#null | 0.7463 | 1842 |  |  |  | 1365 | 1.017 | - | - | f18d + VALU result latency 8 |
