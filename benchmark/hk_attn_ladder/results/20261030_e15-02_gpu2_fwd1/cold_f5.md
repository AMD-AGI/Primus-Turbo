DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 13:38:49

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7950 | 1729 |  |  |  | 1429 | 1.135 | - | - |  |
| f3a | 0.7817 | 1759 |  |  |  | 1425 | 1.115 | - | - | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f5a | 0.7900 | 1740 | f3a | 0.990x | 0.973x | 1448 | 1.145 | - | - | rescale decision from the lane's half max (no permlane), corr = 1 on the common path; full max, m', corr, O *= corr out of line behind a not-taken branch (bitwise) |
| f5b | 0.7673 | 1792 | f5a | 1.031x | 1.051x | 1422 | 1.091 | - | - | + rescale threshold 8 (log2 units, FA4's): the running max may lag by a factor 256 (not bitwise; SNR gate) |
| f5b#null | 0.7649 | 1797 |  |  |  | 1420 | 1.088 | - | - | + rescale threshold 8 (log2 units, FA4's): the running max may lag by a factor 256 (not bitwise; SNR gate) |
