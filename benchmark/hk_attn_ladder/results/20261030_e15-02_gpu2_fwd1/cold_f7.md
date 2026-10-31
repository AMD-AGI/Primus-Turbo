DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:49:01

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f3a | 0.7812 | 1760 |  |  |  | 1424 | 1.113 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f7w1 | 0.8310 | 1654 |  |  |  | 1494 | 1.246 | - | - | f3a with the WMMA stream order PV0p,PV1p,QK0,PV2p,QK1,PV3p,QK2,QK3 |
| f7w3 | 0.8386 | 1639 |  |  |  | 1506 | 1.263 | - | - | f3a with the WMMA stream order PV0p,PV1p,QK0,QK1,PV2p,PV3p,QK2,QK3 |
| f7w4 | 0.7799 | 1763 |  |  |  | 1422 | 1.110 | - | - | f3a with the WMMA stream order PV0p,PV1p,PV2p,QK0,PV3p,QK1,QK2,QK3 |
| f7w4#null | 0.7766 | 1770 |  |  |  | 1422 | 1.105 | - | - | f3a with the WMMA stream order PV0p,PV1p,PV2p,QK0,PV3p,QK1,QK2,QK3 |
