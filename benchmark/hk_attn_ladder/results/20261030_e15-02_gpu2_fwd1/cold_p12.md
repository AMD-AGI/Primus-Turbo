DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 14:45:20

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7952 | 1729 |  |  |  | 1430 | 1.137 | - | - |  |
| f12a | 0.7763 | 1771 |  |  |  | 1420 | 1.104 | - | - | VGPR plan 2: S, PR, softmax scratch, m / d / corr / m' in one block (block 2); V split over blocks 3 and 0 |
| f14a | 0.7641 | 1799 | f12a | 1.018x | 1.035x | 1397 | 1.067 | - | - | XBLK: ring slots n..n+2 load the next block's tiles 0..2 (no 1-row copies); the epilogue issues only the next Q |
| f14a#null | 0.7611 | 1806 |  |  |  | 1396 | 1.062 | - | - | XBLK: ring slots n..n+2 load the next block's tiles 0..2 (no 1-row copies); the epilogue issues only the next Q |
