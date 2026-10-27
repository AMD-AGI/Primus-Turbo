Llama-3.1-8B attention forward (bf16, d128, GQA 32/8, causal, BSHD) on gfx1250, b4 sq8192 skv8192 hq32 hkv8 d128/128 causal bshd, c822521170f1 GPU None, 2026-10-27 10:34:15

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| flydsl | 1.3393 | 1642 |  |  |  | 1474 | 1.973 |  |
| f00 | 2.5455 | 864 |  |  |  | 1637 | 4.171 | naive: 64 queries x 1 head per WG, register-mediated single-buffer K/V fill, 2 barriers per block, eager rescale |
| f01t | 2.4029 | 915 | f00 | 1.059x | 1.045x | 1660 | 3.986 | TDM fill (one buffer, 2 barriers per block): no VGPRs, VALU or buffer-load waits for the fill |
| f02 | 2.2876 | 961 | f01t | 1.050x | 1.078x | 1616 | 3.696 | double-buffered TDM fill: block j+1's tensor_load_to_lds ops in flight during block j's compute, 1 barrier per block |
| f03b | 1.9304 | 1139 | f02 | 1.188x | 1.324x | 1449 | 2.797 | operand bursts: a phase's LDS operands loaded at once (V before the softmax), not one load before each WMMA |
| f04r | 1.8837 | 1168 | f03b | 1.024x | 1.049x | 1415 | 2.667 | 3-slot TDM ring, prefetch distance 2 (enabler of the pipelined rung) |
| f05p | 1.8767 | 1172 | f04r | 1.003x | 1.023x | 1388 | 2.608 | software pipelining: S^T(j+1) WMMAs under the softmax of block j (one branch-free region) |
| f06g | 1.5886 | 1384 | f05p | 1.183x | 1.004x | 1630 | 2.597 | GQA packing on the pipelined kernel: each K / V operand feeds the 4 q heads of its kv head |
| f07l | 1.4807 | 1485 | f06g | 1.074x | 1.139x | 1540 | 2.280 | longest-first workgroup order (causal load balance) |
| f08s | 1.4508 | 1516 | f07l | 1.021x | 1.029x | 1528 | 2.217 | lock_simd: no WMMA arbitration pause at one wave per SIMD |
| f09m | 1.4157 | 1554 | f08s | 1.022x | 1.019x | 1533 | 2.169 | causal mask on the diagonal block only (peeled loop, no branch in the hot loop) |
| f10z | 1.2383 | 1776 | f09m | 1.144x | 1.184x | 1481 | 1.834 | lazy rescale behind one unlikely branch per chunk (v0's RESC form), max taken a phase ahead |
| v0 | 1.2522 | 1756 | f10z | 0.991x | 0.974x | 1508 | 1.890 | first HK forward (Primus-Turbo fd4a1bd0), the shipped form of f10z: 4-slot ring with split barriers, Q by TDM, O through LDS, a finer QK || softmax interleave (same speed as f10z) |
| v0_resc | 1.2176 | 1806 | v0 | 1.026x | 1.046x | 1482 | 1.806 | lazy-rescale test as one max + one VCC branch |
| v0_resc_desc | 1.1849 | 1856 | v0_resc | 1.029x | 1.045x | 1457 | 1.730 | TDM descriptor math inside the QK WMMA run, burst LDS bases one stage ahead |
| p1 | 1.1442 | 1922 | v0_resc_desc | 1.035x | 1.064x | 1417 | 1.622 | K(c+2) and V(c+1) loads as one burst per chunk (Primus-Turbo cade7db6 pin) |
| v1_v0body | 1.2176 | 1806 | v0 | 1.029x | 1.054x | 1473 | 1.792 | v1 source with the v0 body (v0 grid) |
| v1_v0body_pairs | 1.2046 | 1826 | v1_v0body | 1.011x | 1.022x | 1454 | 1.754 | persistent grid: zig-zag causal pairs, XCD-major |
| v1_lsum0 | 1.1382 | 1932 | v1_v0body_pairs | 1.059x | 1.098x | 1404 | 1.599 | lean SM1 / FSPLIT, F1 interleave, MDS2 (v1 as validated) |
| v11_best | 1.1297 | 1947 | v1_lsum0 | 1.007x | 1.008x | 1403 | 1.584 | v1.1: RESC + one burst per stage (MDS3) + 2-stage unroll (never run on a GPU) |
| f01 | 2.9468 | 746 | f00 | 0.864x | 0.808x | 1751 | 5.160 | negative: register-path double buffer -- 64 more VGPRs hold block j+1's loads, occupancy 4 -> 3 waves/SIMD (ATT); the TDM path (f01t, f02) double-buffers without VGPRs |
| f03 | 2.4275 | 906 | f02 | 0.941x | 0.757x | 2008 | 4.873 | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): GQA packing on f02 -- 4x the work per WG at 1 wave/SIMD, nothing to hide the softmax under |
| f04 | 2.1617 | 1017 | f03 | 1.124x | 1.164x | 1938 | 4.192 | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): longest-first workgroup order (causal load balance) |
| f05 | 2.1946 | 1002 | f04 | 0.985x | 0.982x | 1943 | 4.264 | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): 4-slot TDM ring with prefetch distance 2 (two blocks of compute cover each block's TDM latency) |
| f06 | 2.2430 | 980 | f05 | 0.979x | 0.969x | 1962 | 4.400 | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): split barrier: signal after S^T(j), wait at the top of block j+1 (softmax + PV absorb wave skew) |
| f07 | 2.2425 | 981 | f06 | 1.000x | 1.001x | 1962 | 4.400 | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): lock_simd: no WMMA arbitration pause at one wave per SIMD |
| f08 | 2.3735 | 927 | f07 | 0.945x | 0.932x | 1989 | 4.722 | side chain (GQA before pipelining; one S^T -> softmax -> PV dependency chain per block): lazy rescale: O / l rescaled only when a query's max grew by > 8 nat (uniform branch) |
| f09z | 1.5373 | 1431 | f08s | 0.943x | 0.862x | 1672 | 2.568 | negative: lazy rescale with a per-head branch at every chunk end (the branches cut the interleaved schedule) |
| x_desc | 1.2312 | 1786 | v0 | 1.016x | 1.037x | 1478 | 1.816 | DESC alone on v0 |
| x_merge | 1.2293 | 1789 | v0 | 1.019x | 1.040x | 1481 | 1.818 | MERGE alone on v0 |
| x_p1_tail | 1.1680 | 1883 | p1 | 0.980x | 0.975x | 1426 | 1.664 | negative: row sums moved into the PV run (TAIL) |
| x_v1_lsum1 | 1.1586 | 1898 | v1_lsum0 | 0.984x | 0.959x | 1439 | 1.667 | negative: row sums by a WMMA with an all-ones operand (LSUM=1) |
| x_pp | 1.3420 | 1639 | v0 | 0.932x | 0.954x | 1474 | 1.977 | negative: 8-wave ping-pong (HK's CDNA4 attention schedule) |
| x_pp#null | 1.3394 | 1642 |  |  |  | 1472 | 1.974 | negative: 8-wave ping-pong (HK's CDNA4 attention schedule) |
