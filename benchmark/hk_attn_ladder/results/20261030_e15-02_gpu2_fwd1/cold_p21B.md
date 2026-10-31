DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:57:34

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7982 | 1722 |  |  |  | 1429 | 1.141 | - | - |  |
| f23s2 | 0.7465 | 1842 |  |  |  | 1367 | 1.020 | - | - | diagnostic: f22a with 2.4 KB of dead s_nop at the kernel start |
| f23e | 0.7454 | 1844 |  |  |  | 1365 | 1.017 | 1975 | 1276.43 | diagnostic: f22a with 7.2 KB of dead s_nop at the kernel end |
| f23s | 0.7461 | 1843 |  |  |  | 1366 | 1.020 | - | - | diagnostic: f22a with 7.2 KB of dead s_nop at the kernel start (code shifted) |
| f22a | 0.7471 | 1840 |  |  |  | 1368 | 1.021 | - | - | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
| f22a#null | 0.7475 | 1839 |  |  |  | 1366 | 1.023 | - | - | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
