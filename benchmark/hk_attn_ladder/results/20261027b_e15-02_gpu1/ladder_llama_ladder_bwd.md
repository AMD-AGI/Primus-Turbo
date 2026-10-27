Llama-3.1-8B attention backward (bf16, d128, GQA 32/8, causal, BSHD) on gfx1250, b4 sq8192 skv8192 hq32 hkv8 d128/128 causal bshd, c822521170f1 GPU None, 2026-10-27 10:44:55

| Rung | ms | TFLOP/s | Over | Step | Step (cycles) | sclk MHz | Mcycles | Feature |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| flydsl | 4.3619 | 1261 |  |  |  | 1516 | 6.612 |  |
| b00 | 9.2957 | 591 |  |  |  | 1802 | 16.746 | naive FA2 backward: prep + dK/dV kernel + separate dQ kernel (S, dP recomputed), register-mediated single-buffer fills |
| b01t | 8.1605 | 674 | b00 | 1.140x | 1.164x | 1763 | 14.393 | TDM fills (one buffer, 2 barriers per iteration): no VGPRs, VALU or buffer-load waits for the fill |
| b02 | 7.3230 | 751 | b01t | 1.114x | 1.193x | 1648 | 12.075 | double-buffered TDM fills: the next iteration's tiles in flight during this one's compute, 1 barrier per iteration |
| b03b | 6.9113 | 796 | b02 | 1.060x | 1.071x | 1632 | 11.269 | operand bursts (next group's burst during this group's WMMAs, tr16 operands during the softmax), NL / delta requested a run ahead |
| b04f | 6.1450 | 895 | b03b | 1.125x | 1.113x | 1646 | 10.122 | dQ fused in v0's split: 128 kv / WG, 32-query steps, dS^T exchanged through LDS, each wave's K slice of dQ resident, 128-B-line fp32 atomics (5 GEMMs instead of 7) |
| b05g | 5.3190 | 1034 | b04f | 1.154x | 1.185x | 1606 | 8.541 | longest-first workgroup order: kv heads fastest, kv blocks ascending (v0's grid order) |
| b06k | 4.9549 | 1110 | b05g | 1.074x | 1.033x | 1670 | 8.271 | wave kinds per step (v0's FULL / DIAG / DEAD): the mask only on diagonal squares, fully masked squares skipped |
| b07l | 4.9789 | 1104 | b06k | 0.996x | 0.985x | 1686 | 8.390 | NL / delta rows by TDM with the step's Q / dO tiles, read from LDS (v0's LD records): no global loads in the loop (flat alone; the enabler of b08d) |
| b08d | 4.8105 | 1143 | b07l | 1.036x | 1.065x | 1638 | 7.882 | deferred dQ: dS operands requested right after the barrier, dQ(s) WMMAs + atomics in step s+1 after its S / dP run |
| b09i | 4.6849 | 1174 | b08d | 1.026x | 1.037x | 1622 | 7.598 | dQ(s-1) WMMAs and step s's softmax in one scheduling region (no branch, no sched_barrier), atomics after the dS^T stores |
| b10p | 4.5285 | 1214 | b09i | 1.034x | 1.053x | 1595 | 7.229 | two pipelined regions per step (co-execution scheduler): S / dP(q tile 1) || softmax(q tile 0), dQ(s-1) || softmax(q tile 1) |
| b11s | 4.3938 | 1251 | b10p | 1.031x | 1.046x | 1573 | 6.906 | v0's split step barrier: tr16 of dV / dK in flight past the signal, dV / dK WMMAs in the window, 3-slot ring with ops after the wait |
| b12l | 4.3390 | 1267 | b11s | 1.014x | 1.025x | 1554 | 6.739 | lock_simd: no WMMA arbitration pause at one wave per SIMD |
| v0 | 3.8558 | 1426 | b12l | 1.125x | 1.242x | 1410 | 5.433 | first HK backward (Primus-Turbo fd4a1bd0), the shipped form of b12l: next step's operand sets double-buffered in registers (x2-unrolled parity steps, readback a phase ahead), a 4-slot ring with TDM roles (wave 3 moves K / V), LD records per 4 queries, S / dP by kv tile with the softmax interleaved by sched_group_barrier |
| v0_regs | 3.7794 | 1455 | v0 | 1.020x | 1.048x | 1372 | 5.186 | holds on the S / dP WMMA source registers (no WAR v_nops) |
| v0_regs_masked | 3.7582 | 1463 | v0_regs | 1.006x | 1.023x | 1353 | 5.086 | diagonal steps use the full step's operand layout |
| v0_regs_masked_atom4 | 3.7240 | 1476 | v0_regs_masked | 1.009x | 1.016x | 1344 | 5.005 | the 32 dQ atomics of a step as 4 sub-bursts of 8 |
| v0_rma_tdm3 | 3.6768 | 1495 | v0_regs_masked_atom4 | 1.012x | 1.027x | 1326 | 4.876 | wave 3 issues a real 16-byte dummy TDM op instead of a role branch |
| p1 | 3.7104 | 1482 | v0_rma_tdm3 | 0.991x | 0.989x | 1328 | 4.923 | loop-entry DS wait + early LDS base math (Primus-Turbo cade7db6 pin) |
| v1_sched0 | 3.6492 | 1507 | v0 | 1.057x | 1.150x | 1296 | 4.724 | v1 protocol under expert scheduling, memory ops still in bursts |
| v1 | 3.6170 | 1520 | v1_sched0 | 1.009x | 1.020x | 1282 | 4.638 | every memory op interleaved under the WMMA runs |
| v1_atom2 | 3.5795 | 1536 | v1 | 1.011x | 1.027x | 1263 | 4.516 | one dQ atomic per WMMA gap (v1 as validated) |
| v11_regs | 3.5646 | 1542 | v1_atom2 | 1.004x | 1.023x | 1238 | 4.411 | v1.1: register holds |
| v11_regs_tdm3 | 3.5437 | 1552 | v11_regs | 1.006x | 1.016x | 1224 | 4.339 | v1.1: dummy TDM op instead of a role branch |
| v11_best | 3.5443 | 1551 | v11_regs_tdm3 | 1.000x | 1.006x | 1218 | 4.319 | v1.1: diagonal steps on the full layout (MASKED=2) |
| b01 | 9.5207 | 578 | b00 | 0.977x | 0.975x | 1806 | 17.189 | negative: register-path double buffer -- 64 more VGPRs for the in-flight loads cost a wave per SIMD (ATT); the TDM path (b01t, b02) double-buffers without VGPRs |
| b05r | 6.4807 | 848 | b04f | 0.949x | 0.938x | 1664 | 10.796 | negative: 3-slot ring + split step barrier on b04f -- the signal's DS drain right after the dS^T stores costs more than the skew it absorbs |
| b05s | 6.3758 | 862 | b04f | 0.963x | 0.957x | 1657 | 10.568 | negative: split step barrier with two Q / dO buffers -- the early signal waits for step s+1's TDM op |
| b05n | 6.4805 | 848 | b04f | 0.949x | 0.946x | 1651 | 10.701 | negative: the same prefetch on b04f (before the work order and the wave kinds) |
| b07n | 5.1421 | 1069 | b06k | 0.963x | 0.937x | 1716 | 8.827 | negative: next step's operands a step ahead in registers (NL / delta, Q / dO readback) -- 990 VGPRs |
| x_b10p_nocoexec | 4.6392 | 1185 | b10p | 0.976x | 0.962x | 1617 | 7.501 | b10p in the default scheduler (no co-execution strategy) |
| b03 | 8.1845 | 672 | b02 | 0.895x | 0.811x | 1818 | 14.888 | side chain (first draft: GQA / fusion before the v0 decomposition): dQ kernel takes the 4 q heads of one kv head per WG (K / V tiles shared) |
| b04 | 12.6665 | 434 | b03 | 0.646x | 0.645x | 1824 | 23.089 | side chain (first draft: GQA / fusion before the v0 decomposition): dQ fused into the dK/dV kernel: dS^T through LDS, dQ = dS K, fp32 atomics (5 GEMMs per pair instead of 7) |
| b05 | 10.5619 | 521 | b04 | 1.199x | 1.090x | 2006 | 21.181 | side chain (first draft: GQA / fusion before the v0 decomposition): 32 kv rows per wave (128 per WG): every Q / dO LDS operand feeds 2 WMMAs, half the Q / dO traffic per kv row |
| b06 | 14.8766 | 370 | b05 | 0.709x | 0.692x | 2056 | 30.579 | side chain (first draft: GQA / fusion before the v0 decomposition): dQ one iteration late from a double-buffered dS^T: 1 barrier per iteration instead of 2 |
| b07 | 14.8173 | 371 | b06 | 1.008x | 1.005x | 2053 | 30.420 | side chain (first draft: GQA / fusion before the v0 decomposition): lock_simd in the dK/dV kernel (one wave per SIMD) |
| x_masked | 3.8033 | 1446 | v0 | 1.013x | 1.026x | 1391 | 5.297 | MASKED alone on v0 |
| x_atom4 | 3.8191 | 1440 | v0 | 1.010x | 1.023x | 1391 | 5.307 | ATOM=4 alone on v0 |
| x_tdm3 | 3.8260 | 1437 | v0 | 1.008x | 1.022x | 1390 | 5.318 | TDM3 alone on v0 |
| x_addr | 3.8239 | 1438 | v0 | 1.008x | 1.023x | 1388 | 5.312 | ADDR alone on v0 |
| x_entry | 3.8487 | 1429 | v0 | 1.002x | 1.011x | 1398 | 5.380 | ENTRY alone on v0 |
| x_onephase | 4.6235 | 1189 | v0 | 0.834x | 0.737x | 1594 | 7.368 | negative: one memory phase per step (zero-window barrier) |
| x_v1_noexp | 5.6420 | 975 | v1 | 0.641x | 0.459x | 1794 | 10.114 | negative: v1's interleaved source in the default scheduling mode |
| x_v1_atom0 | 3.7781 | 1455 | v1 | 0.957x | 0.908x | 1352 | 5.106 | negative: all 32 atomics as one burst |
| x_v0_noint | 3.7997 | 1447 | v0 | 1.015x | 1.028x | 1392 | 5.291 | v0 without the sched_group_barrier interleave of the softmax into the S / dP and dQ WMMA runs |
| x_v0_nodead | 3.8390 | 1432 | v0 | 1.004x | 1.017x | 1392 | 5.340 | v0 with DEAD waves running the masked diagonal path |
| x_v0_nodead#null | 3.8427 | 1431 |  |  |  | 1391 | 5.342 | v0 with DEAD waves running the masked diagonal path |
