DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 14:46:14

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7128 | 1929 |  |  |  | 2211 | 1.576 | 1175 | 837.61 |  |
| f12a | 0.6999 | 1964 |  |  |  | 2212 | 1.547 | 1175 | 822.11 | VGPR plan 2: S, PR, softmax scratch, m / d / corr / m' in one block (block 2); V split over blocks 3 and 0 |
| f14a | 0.6842 | 2009 | f12a | 1.024x | 1.023x | 2212 | 1.513 | 1174 | 801.33 | XBLK: ring slots n..n+2 load the next block's tiles 0..2 (no 1-row copies); the epilogue issues only the next Q |
| f14a#null | 0.6839 | 2010 |  |  |  | 2212 | 1.513 | 1174 | 803.28 | XBLK: ring slots n..n+2 load the next block's tiles 0..2 (no 1-row copies); the epilogue issues only the next Q |
