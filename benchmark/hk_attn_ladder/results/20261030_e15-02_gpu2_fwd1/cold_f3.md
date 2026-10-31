DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:22:59

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7966 | 1726 |  |  |  | 1430 | 1.138 | - | - |  |
| f1a | 0.7894 | 1742 |  |  |  | 1405 | 1.110 | - | - | va_vdst rule llvm_war_xdl (the backward's): a WAR against an in-flight WMMA waits for its own class count, not va_vdst(0) |
| f3a | 0.7803 | 1762 | f1a | 1.012x | 0.998x | 1425 | 1.112 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f3a#null | 0.7796 | 1763 |  |  |  | 1426 | 1.113 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
