DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 14:07:49

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f8x | 0.7784 | 1766 |  |  |  | 1422 | 1.106 | - | - | f3a's flags on the patch-08 tree: the scheduler's LDS-in-flight model now retires loads at completion (new schedule) |
| pad_msb | 0.8250 | 1666 |  |  |  | 1477 | 1.220 | - | - | timing probe: TILE_C + 1 redundant msb per s_set_vgpr_msb |
| pad_salu | 0.7853 | 1751 |  |  |  | 1430 | 1.122 | - | - | timing probe: TILE_C + 64 redundant salu |
| pad_valu | 0.8050 | 1708 |  |  |  | 1456 | 1.171 | - | - | timing probe: TILE_C + 64 redundant valu |
| pad_nop | 0.7785 | 1766 |  |  |  | 1425 | 1.110 | - | - | timing probe: TILE_C + 64 redundant nop |
| pad_nop#null | 0.7798 | 1763 |  |  |  | 1424 | 1.110 | - | - | timing probe: TILE_C + 64 redundant nop |
