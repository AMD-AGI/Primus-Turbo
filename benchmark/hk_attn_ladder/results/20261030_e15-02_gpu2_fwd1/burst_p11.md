DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 14:12:58

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7153 | 1922 |  |  |  | 2211 | 1.581 | 1177 | 842.64 |  |
| f8x | 0.7050 | 1950 |  |  |  | 2212 | 1.559 | 1176 | 829.18 | f3a's flags on the patch-08 tree: the scheduler's LDS-in-flight model now retires loads at completion (new schedule) |
| f12a | 0.7031 | 1955 |  |  |  | 2212 | 1.555 | 1176 | 826.72 | VGPR plan 2: S, PR, softmax scratch, m / d / corr / m' in one block (block 2); V split over blocks 3 and 0 |
| f12a#null | 0.7029 | 1956 |  |  |  | 2212 | 1.555 | 1175 | 826.16 | VGPR plan 2: S, PR, softmax scratch, m / d / corr / m' in one block (block 2); V split over blocks 3 and 0 |
