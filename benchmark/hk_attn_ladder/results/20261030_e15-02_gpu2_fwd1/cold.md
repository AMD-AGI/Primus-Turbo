DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:18:40

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7946 | 1730 |  |  |  | 1428 | 1.135 | - | - |  |
| f0 | 0.8019 | 1714 |  |  |  | 1423 | 1.143 | - | - | the shipped HK forward of dev/lhz/dsv3-attn-hk-merge (a08-1 E2: persistent static schedule, 4-stage ring, generated tile regions) |
| f1a | 0.7898 | 1741 | f0 | 1.015x | 1.028x | 1408 | 1.114 | - | - | va_vdst rule llvm_war_xdl (the backward's): a WAR against an in-flight WMMA waits for its own class count, not va_vdst(0) |
| f1b | 0.8275 | 1661 | f1a | 0.957x | 0.924x | 1458 | 1.206 | - | - | + no order edges grouping the V / Q(3) reads behind the last PV / QK(0) WMMA |
| f1b#null | 0.8251 | 1666 |  |  |  | 1457 | 1.202 | - | - | + no order edges grouping the V / Q(3) reads behind the last PV / QK(0) WMMA |
