DeepSeek-V3 MLA attention forward (bf16, qk 192 / v 128, MHA 128 heads, causal, SBHD) on gfx1250, b2 sq4096 skv4096 hq128 hkv128 d192/128 causal sbhd, c822521170f1 GPU None, 2026-10-27 11:49:06

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| flydsl | 0.8591 | 1600 |  |  |  | 1454 | 1.248 |  |
| flydsl_b12f5953 | 0.8205 | 1675 |  |  |  | 1385 | 1.137 |  |
| x6_m1 | 1.2447 | 1104 |  |  |  | 1754 | 2.184 | first HK MLA forward (X6 M1, pure HIP): 256-query WG (4 waves x 4 q tiles), 64-kv tiles on a 4-stage TDM ring, each tile serial: LDS burst -> 96 QK WMMA -> softmax -> 64 PV WMMA |
| x6_m2f | 1.0172 | 1352 | x6_m1 | 1.223x | 1.356x | 1582 | 1.610 | M2: 32-kv chunks software-pipelined (QK(c+1) || exp(c), PV(c) || P / row sums / next mask), split barrier, branch-free TDM (NULL descriptors), K double-buffered in registers, a chunk's LDS reads (V^T(c), K(c+2)) as one burst at the head of phase 1 (X6's best) |
| s1a | 0.9893 | 1390 | x6_m2f | 1.030x | 1.084x | 1504 | 1.486 | marker shell (LLVM: prologue, TDM ring, barriers, epilogue) + generated asm tile regions (asmkit: MSB, expert waits, hazards; list scheduler), no cross-tile pipelining |
| s1a_diag | 0.9556 | 1439 | s1a | 1.032x | 1.031x | 1504 | 1.437 | diagonal tiles specialized (TILE_D0..3 skip the q tiles past the diagonal) |
| s1a2_diag | 0.9389 | 1464 | s1a_diag | 1.018x | 1.032x | 1483 | 1.392 | softmax scale c from an SGPR pair, Q operands regrouped late, QK / PV WMMA order interleaved (QK0,QK1,PV0,QK2,QK3,PV1,PV2,PV3) |
| s1b_A | 0.9069 | 1516 | s1a2_diag | 1.035x | 1.059x | 1450 | 1.314 | cross-tile software pipelining (region boundary moved into the middle of the tile) with the slot scheduler (DS latency 290, DS first) |
| e1 | 0.8333 | 1650 | s1b_A | 1.088x | 1.142x | 1382 | 1.150 | epilogue: O / LSE staged through the wave's own Q rows and written as whole rows (straight line, rolling XCNT window) instead of 4 x 8 stores each followed by s_wait_xcnt 0 |
| e2 | 0.8301 | 1656 | e1 | 1.004x | 1.008x | 1376 | 1.143 | persistent: 256 WGs on a static schedule, the K / V ring continuous across blocks, the next block's Q and tiles 0..2 prefetched after its barrier (12-argument kernel) |
| x6_m2e | 1.4096 | 975 | x6_m1 | 0.882x | 0.843x | 1835 | 2.587 | not a measured version: the final M2 source built with X6_BURST=0 (same resources / kernarg notes as a08-1's m2e, no recorded sha), 1.41 ms on e15-02 against a08-1 m2e's 0.995; M2's chunk pipelining itself (m2d / m2e) was not preserved as source |
| x6_m2fx | 1.0247 | 1342 | x6_m2f | 0.995x | 0.995x | 1584 | 1.621 | negative: m2f under expert scheduling (s_wait_alu va_vdst(0) drains, 35-48 cyc each) |
| ref | 1.0888 | 1263 | s1a | 0.905x | 0.851x | 1604 | 1.746 | REF: s1a's marker shell with the tile code in HIP (LLVM-compiled; the generator's correctness reference) |
| ref_diag | 1.0653 | 1290 | s1a_diag | 0.896x | 0.840x | 1605 | 1.710 | REF with the diagonal specialization (HKGEN_DIAG=1) |
| s1a_prog | 1.1788 | 1166 | s1a | 0.838x | 0.726x | 1735 | 2.048 | negative: generated regions in program order (no list scheduling) |
| s1a2 | 0.9620 | 1429 | s1a | 1.027x | 1.044x | 1480 | 1.423 | s1a2's tile code without the diagonal specialization |
| s1a3_n200 | 0.9315 | 1476 | s1a2_diag | 1.004x | 1.006x | 1482 | 1.381 | flat: s1a2 scheduled with a 200-cycle DS latency model |
| s1a3_d290 | 0.9547 | 1440 | s1a2_diag | 0.985x | 0.973x | 1502 | 1.433 | flat: LDS reads deferred into the previous tile (defer3; model 0.79-0.85, measured no gain) |
| s1b_B | 0.9201 | 1494 | s1a2_diag | 1.020x | 1.028x | 1472 | 1.354 | s1b's pipelining with the old greedy list scheduler |
| fdev2_Ap | 0.9136 | 1505 | s1b_A | 0.995x | 0.989x | 1456 | 1.329 | flat: barrier arrive moved from WMMA 132 to 99 (A'; barrier waits are not the bottleneck) |
| fdev2_A2 | 0.9053 | 1519 | s1b_A | 1.003x | 1.003x | 1450 | 1.312 | flat: rescheduled with the refitted r4 model (DS latency 150, rate 4; model -3..-4%, measured -1%) |
| e1d | 0.9169 | 1499 | e1 | 0.906x | 0.863x | 1451 | 1.330 | diagnostic: E1 with the old 16-row scattered write-back |
| ref_e2 | 0.9907 | 1388 | e2 | 0.835x | 0.730x | 1575 | 1.559 | E2's persistent shell with the tile code in HIP (REF) |
| e2_np | 0.8398 | 1637 | e2 | 0.987x | 0.973x | 1397 | 1.175 | E2's code object launched non-persistent (npw = 0: grid (nqb, H, B), one block per WG) |
| e2_np#null | 0.8426 | 1632 |  |  |  | 1395 | 1.174 | E2's code object launched non-persistent (npw = 0: grid (nqb, H, B), one block per WG) |
