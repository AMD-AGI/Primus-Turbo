Llama-3.1-8B attention backward (bf16, d128, GQA 32/8, causal, BSHD) on gfx1250, b4 sq8192 skv8192 hq32 hkv8 d128/128 causal bshd, c822521170f1 GPU None, 2026-10-27 08:10:47

| Rung | ms | TFLOP/s | Step | Step (cycles) | sclk MHz | Mcycles | Feature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flydsl | 4.3639 | 1260 |  |  | 1517 | 6.623 |  |
| b00 | 9.2485 | 595 |  |  | 1805 | 16.702 | naive FA2 backward: prep + dK/dV kernel + separate dQ kernel (S, dP recomputed), register-mediated single-buffer fills |
| b01 | 9.5017 | 579 | 0.974x | 0.973x | 1807 | 17.167 | double-buffered fills: the next tiles' global loads in flight during this iteration, 1 barrier per iteration |
| b02 | 7.3086 | 752 | 1.299x | 1.422x | 1651 | 12.072 | TDM fills: one 32-row tensor_load_to_lds per wave per tile pair instead of the register path |
| b03 | 8.1759 | 672 | 0.895x | 0.811x | 1821 | 14.890 | dQ kernel takes the 4 q heads of one kv head per WG (K / V tiles shared) |
| b04 | 12.6290 | 435 | 0.647x | 0.645x | 1826 | 23.069 | dQ fused into the dK/dV kernel: dS^T through LDS, dQ = dS K, fp32 atomics (5 GEMMs per pair instead of 7) |
| b05 | 10.5460 | 521 | 1.198x | 1.094x | 2004 | 21.090 | 32 kv rows per wave (128 per WG): every Q / dO LDS operand feeds 2 WMMAs, half the Q / dO traffic per kv row |
| b06 | 14.8205 | 371 | 0.714x | 0.690x | 2071 | 30.686 | dQ one iteration late from a double-buffered dS^T: 1 barrier per iteration instead of 2 |
| b07 | 14.7442 | 373 | 1.007x | 1.007x | 2069 | 30.533 | lock_simd in the dK/dV kernel (one wave per SIMD) |
| v0 | 3.8555 | 1426 | 3.827x | 5.629x | 1410 | 5.431 | first HK backward (Primus-Turbo fd4a1bd0): b07 + 32-query steps (q pairs, heads inner, unmasked steps first) on a 4-deep TDM ring of Q / dO / LD records with one split-barrier phase per step, dQ_acc in C-fragment order (one 128-B line per atomic), and the 9-phase software pipeline (softmax interleaved into the WMMA runs, dQ(s-1) under step s) |
| v0_regs | 3.7960 | 1448 | 1.015x | 1.035x | 1381 | 5.248 | holds on the S / dP WMMA source registers (no WAR v_nops) |
| v0_regs_masked | 3.7580 | 1463 | 1.010x | 1.024x | 1364 | 5.124 | diagonal steps use the full step's operand layout |
| v0_regs_masked_atom4 | 3.7310 | 1474 | 1.009x | 1.013x | 1357 | 5.060 | the 32 dQ atomics of a step as 4 sub-bursts of 8 |
| v0_rma_tdm3 | 3.6815 | 1493 | 1.013x | 1.031x | 1333 | 4.905 | wave 3 issues a real 16-byte dummy TDM op instead of a role branch |
| p1 | 3.7046 | 1484 | 0.992x | 0.992x | 1335 | 4.944 | loop-entry DS wait + early LDS base math (Primus-Turbo cade7db6 pin) |
| v1_sched0 | 3.6415 | 1510 | 1.018x | 1.049x | 1296 | 4.713 | v1 protocol under expert scheduling, memory ops still in bursts |
| v1 | 3.6156 | 1521 | 1.007x | 1.028x | 1268 | 4.584 | every memory op interleaved under the WMMA runs |
| v1_atom2 | 3.5744 | 1538 | 1.011x | 1.032x | 1242 | 4.443 | one dQ atomic per WMMA gap (v1 as validated) |
| v11_regs | 3.5673 | 1541 | 1.002x | 1.018x | 1224 | 4.368 | v1.1: register holds |
| v11_regs_tdm3 | 3.5497 | 1549 | 1.005x | 1.009x | 1220 | 4.330 | v1.1: dummy TDM op instead of a role branch |
| v11_best | 3.5418 | 1552 | 1.002x | 1.007x | 1214 | 4.299 | v1.1: diagonal steps on the full layout (MASKED=2) |
| x_masked | 3.8166 | 1441 |  |  | 1392 | 5.321 | MASKED alone on v0 |
| x_atom4 | 3.8341 | 1434 |  |  | 1396 | 5.358 | ATOM=4 alone on v0 |
| x_tdm3 | 3.8515 | 1428 |  |  | 1396 | 5.374 | TDM3 alone on v0 |
| x_addr | 3.8489 | 1429 |  |  | 1397 | 5.374 | ADDR alone on v0 |
| x_entry | 3.8604 | 1424 |  |  | 1396 | 5.387 | ENTRY alone on v0 |
| x_onephase | 4.6270 | 1188 |  |  | 1592 | 7.371 | negative: one memory phase per step (zero-window barrier) |
| x_v1_noexp | 5.6520 | 973 |  |  | 1791 | 10.126 | negative: v1's interleaved source in the default scheduling mode |
| x_v1_atom0 | 3.7678 | 1459 |  |  | 1332 | 5.017 | negative: all 32 atomics as one burst |
| x_v1_atom0#null | 3.7630 | 1461 |  |  | 1333 | 5.016 | negative: all 32 atomics as one burst |
