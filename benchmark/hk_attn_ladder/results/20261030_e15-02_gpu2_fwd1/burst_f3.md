DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 13:23:40

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7157 | 1921 |  |  |  | 2212 | 1.583 | 1176 | 840.42 |  |
| f1a | 0.7145 | 1924 |  |  |  | 2211 | 1.580 | 1175 | 839.44 | va_vdst rule llvm_war_xdl (the backward's): a WAR against an in-flight WMMA waits for its own class count, not va_vdst(0) |
| f3a | 0.7094 | 1938 | f1a | 1.007x | 1.007x | 2211 | 1.569 | 1175 | 834.08 | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f3a#null | 0.7070 | 1945 |  |  |  | 2212 | 1.564 | 1175 | 829.91 | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
