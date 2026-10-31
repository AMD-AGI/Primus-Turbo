DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 13:39:39

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7139 | 1926 |  |  |  | 2211 | 1.578 | 1176 | 840.15 |  |
| f3a | 0.7063 | 1946 |  |  |  | 2212 | 1.562 | 1176 | 830.47 | lazy rescale: O(i) *= corr(i) under a per-q-tile wave-uniform branch (FlyDSL's _rescale_if; bitwise) |
| f5a | 0.7174 | 1916 | f3a | 0.984x | 0.984x | 2212 | 1.587 | 1174 | 845.32 | rescale decision from the lane's half max (no permlane), corr = 1 on the common path; full max, m', corr, O *= corr out of line behind a not-taken branch (bitwise) |
| f5b | 0.6930 | 1984 | f5a | 1.035x | 1.035x | 2212 | 1.533 | 1175 | 814.61 | + rescale threshold 8 (log2 units, FA4's): the running max may lag by a factor 256 (not bitwise; SNR gate) |
| f5b#null | 0.6952 | 1977 |  |  |  | 2212 | 1.538 | 1174 | 816.38 | + rescale threshold 8 (log2 units, FA4's): the running max may lag by a factor 256 (not bitwise; SNR gate) |
