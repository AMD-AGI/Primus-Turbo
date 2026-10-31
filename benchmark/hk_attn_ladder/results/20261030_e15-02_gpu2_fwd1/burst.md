DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 13:19:30

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7150 | 1923 |  |  |  | 2211 | 1.581 | 1175 | 841.90 |  |
| f0 | 0.7242 | 1898 |  |  |  | 2211 | 1.601 | 1176 | 852.06 | the shipped HK forward of dev/lhz/dsv3-attn-hk-merge (a08-1 E2: persistent static schedule, 4-stage ring, generated tile regions) |
| f1a | 0.7130 | 1928 | f0 | 1.015x | 1.015x | 2211 | 1.576 | 1176 | 839.40 | va_vdst rule llvm_war_xdl (the backward's): a WAR against an in-flight WMMA waits for its own class count, not va_vdst(0) |
| f1b | 0.7475 | 1839 | f1a | 0.953x | 0.953x | 2211 | 1.652 | 1176 | 878.91 | + no order edges grouping the V / Q(3) reads behind the last PV / QK(0) WMMA |
| f1b#null | 0.7471 | 1840 |  |  |  | 2211 | 1.652 | 1177 | 880.02 | + no order edges grouping the V / Q(3) reads behind the last PV / QK(0) WMMA |
