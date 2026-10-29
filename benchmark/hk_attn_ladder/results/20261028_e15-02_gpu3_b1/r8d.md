DeepSeek-V3 MLA attention backward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, randn inputs, 7b22589a8076 GPU None, 2026-10-29 02:51:14

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | W | mJ/call | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r8a | 2.4180 | 1478 |  |  |  | 1267 | 3.062 | - | - | dq: when (Skv - Sq) % 64 == 32 the last steady unit runs as a generic unit, whose burst handles the first masked row's partly stored pairs (fixes the r2d..r7 rect32 dQ read; production shapes unchanged) |
| r8d | 2.4556 | 1456 | r8a | 0.984x | 0.964x | 1293 | 3.177 | - | - | dK/dV: every ring barrier also retires the newest tile (tensorcnt 0), so the full loop reads back hh0 of the next stage before its barrier, under dK kh0, instead of after it (its latency was exposed at the top of every iteration) |
| r8d#null | 2.4509 | 1458 |  |  |  | 1292 | 3.165 | - | - | dK/dV: every ring barrier also retires the newest tile (tensorcnt 0), so the full loop reads back hh0 of the next stage before its barrier, under dK kh0, instead of after it (its latency was exposed at the top of every iteration) |
