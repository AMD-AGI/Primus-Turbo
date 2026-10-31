DeepSeek-V3 MLA attention forward, second campaign (2026-10-30, GPU2): from Primus-Turbo dev/lhz/dsv3-attn-hk-merge a95dd334, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, cold timing, 534baaaa14ae GPU None, 2026-10-30 15:24:26

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.7969 | 1725 |  |  |  | 1428 | 1.137 | - | - |  |
| f14a | 0.7654 | 1796 |  |  |  | 1394 | 1.068 | - | - | XBLK: ring slots n..n+2 load the next block's tiles 0..2 (no 1-row copies); the epilogue issues only the next Q |
| f17a | 0.7480 | 1838 | f14a | 1.021x | 1.041x | 1370 | 1.027 | - | - | --sgpr-lat 20 (VALU-written SGPR to SALU: 20 cycles) + --resc branch2 (the RESC s_cmp scheduled 9 cycles ahead) |
| f17b | 0.7544 | 1822 |  |  |  | 1382 | 1.042 | - | - | f14a + --sgpr-lat 20 only |
| f17c | 0.7573 | 1815 |  |  |  | 1384 | 1.047 | - | - | f14a + --resc branch2 only |
| f17d | 0.7489 | 1836 |  |  |  | 1368 | 1.024 | - | - | f17a with RESC_COST 25 |
| f17d#null | 0.7512 | 1830 |  |  |  | 1369 | 1.032 | - | - | f17a with RESC_COST 25 |
