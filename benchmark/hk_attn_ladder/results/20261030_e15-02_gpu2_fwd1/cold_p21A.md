DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:56:15

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7958 | 1727 |  |  |  | 1429 | 1.137 | - | - |  |
| f22a | 0.7464 | 1842 |  |  |  | 1366 | 1.020 | - | - | f19f + per-q-tile O resources with 32-bit bounds (no 64-bit clamp chains in the epilogue) |
| f23s | 0.7452 | 1845 |  |  |  | 1366 | 1.018 | - | - | diagnostic: f22a with 7.2 KB of dead s_nop at the kernel start (code shifted) |
| f23e | 0.7464 | 1842 |  |  |  | 1366 | 1.021 | - | - | diagnostic: f22a with 7.2 KB of dead s_nop at the kernel end |
| f23s2 | 0.7461 | 1843 |  |  |  | 1366 | 1.019 | - | - | diagnostic: f22a with 2.4 KB of dead s_nop at the kernel start |
| f23s2#null | 0.7462 | 1842 |  |  |  | 1366 | 1.021 | 1998 | 1281.62 | diagnostic: f22a with 2.4 KB of dead s_nop at the kernel start |
