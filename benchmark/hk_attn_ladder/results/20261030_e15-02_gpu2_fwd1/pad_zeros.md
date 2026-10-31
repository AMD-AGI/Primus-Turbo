DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, zeros inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 14:09:35

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f8x | 0.5545 | 2479 |  |  |  | 1932 | 1.071 | - | - | f3a's flags on the patch-08 tree: the scheduler's LDS-in-flight model now retires loads at completion (new schedule) |
| pad_msb | 0.6109 | 2250 |  |  |  | 1960 | 1.197 | - | - | timing probe: TILE_C + 1 redundant msb per s_set_vgpr_msb |
| pad_salu | 0.5619 | 2447 |  |  |  | 1936 | 1.087 | - | - | timing probe: TILE_C + 64 redundant salu |
| pad_valu | 0.5839 | 2354 |  |  |  | 1940 | 1.132 | - | - | timing probe: TILE_C + 64 redundant valu |
| pad_nop | 0.5552 | 2476 |  |  |  | 1933 | 1.073 | - | - | timing probe: TILE_C + 64 redundant nop |
| pad_nop#null | 0.5559 | 2473 |  |  |  | 1934 | 1.075 | - | - | timing probe: TILE_C + 64 redundant nop |
