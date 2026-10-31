DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, burst timing, 534baaaa14ae GPU None, 2026-10-30 14:08:40

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f8x | 0.7009 | 1961 |  |  |  | 2212 | 1.550 | 1174 | 823.79 | f3a's flags on the patch-08 tree: the scheduler's LDS-in-flight model now retires loads at completion (new schedule) |
| pad_msb | 0.7529 | 1826 |  |  |  | 2212 | 1.666 | 1176 | 885.77 | timing probe: TILE_C + 1 redundant msb per s_set_vgpr_msb |
| pad_salu | 0.7053 | 1949 |  |  |  | 2212 | 1.560 | 1176 | 829.18 | timing probe: TILE_C + 64 redundant salu |
| pad_valu | 0.7256 | 1895 |  |  |  | 2212 | 1.605 | 1176 | 854.87 | timing probe: TILE_C + 64 redundant valu |
| pad_nop | 0.7019 | 1959 |  |  |  | 2212 | 1.552 | 1176 | 824.23 | timing probe: TILE_C + 64 redundant nop |
| pad_nop#null | 0.7007 | 1962 |  |  |  | 2212 | 1.550 | 1176 | 823.55 | timing probe: TILE_C + 64 redundant nop |
