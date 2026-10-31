DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 15:31:23

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7162 | 1919 |  |  |  | 2211 | 1.584 | 1174 | 839.68 |  |
| f17a | 0.6688 | 2056 |  |  |  | 2212 | 1.479 | 1173 | 783.78 | --sgpr-lat 20 (VALU-written SGPR to SALU: 20 cycles) + --resc branch2 (the RESC s_cmp scheduled 9 cycles ahead) |
| f18a | 0.6699 | 2052 |  |  |  | 2212 | 1.482 | 1172 | 785.53 | f17a + --wait-merge 2 (pairs of s_wait_dscnt merged) |
| f18b | 0.6705 | 2050 | f17a | 1.001x | 1.001x | 2212 | 1.483 | 1172 | 786.52 | f17a + --wait-merge 4 |
| f18c | 0.6702 | 2051 |  |  |  | 2212 | 1.483 | 1172 | 785.25 | f17a + --wait-merge 8 |
| f18d | 0.6684 | 2057 |  |  |  | 2212 | 1.478 | 1172 | 782.38 | f17a + DS_MAX 63 |
| f18e | 0.6694 | 2054 |  |  |  | 2212 | 1.481 | 1172 | 784.36 | f17a + --wait-merge 4 + DS_MAX 63 |
| f18e#null | 0.6685 | 2057 |  |  |  | 2212 | 1.479 | 1172 | 783.11 | f17a + --wait-merge 4 + DS_MAX 63 |
