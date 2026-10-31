DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 14:04:40

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f8x | 0.7783 | 1766 |  |  |  | 1423 | 1.113 | - | - | f3a's flags on the patch-08 tree: the scheduler's LDS-in-flight model now retires loads at completion (new schedule) |
| f8a | 0.7883 | 1744 |  |  |  | 1432 | 1.128 | - | - | fwdgen2: one generated LOOP region per block -- TDM issue, tile barriers and the tile counter in generated code, no shell code between tiles |
| f9a | 0.7852 | 1751 | f8a | 1.003x | 1.002x | 1434 | 1.125 | - | - | + one drain at the LOOP prologue's end: a known state at the loop entry, so the tile body's join waits only for its back edge |
| f9a#null | 0.7844 | 1753 |  |  |  | 1433 | 1.124 | - | - | + one drain at the LOOP prologue's end: a known state at the loop entry, so the tile body's join waits only for its back edge |
