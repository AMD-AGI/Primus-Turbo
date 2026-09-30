# k_dqg (r29) ATT deep read: prod shape, dispatch 11, SIMD 3 of CU 1

I only read files and parsed csv/json with stdlib python3. Nothing touched the GPU. My helper scripts are in `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/notes/`: `dqg_loop.py`, `dqg_wall.py`, `dqg_wavecmp.py`, `dqg_budget.py`, `dqg_waits.py`, `dqg_runs.py`, plus the phase listing `dqg_runs_fast.txt`.

Abbreviations: CSV = `probe/p2_r29/stats_ui_output_agent_27913_dispatch_11.csv`, UI = `probe/p2_r29/ui_output_agent_27913_dispatch_11/`, K = `arms/r29/kernels.py`. "Wall" means the time from one instruction's issue to the next one's issue, taken from `UI/se0_sm3_sl0_wv*.json`. This is the real time. CSV "Latency" double-counts WMMA's 8-cycle occupancy under co-issued VALU, so I do not use it for the budget.

## 0. What was traced

- **Wave sequence.** 16 waves ran one after another on one SIMD (simd 3, slot 0). Their wave times sum to 3,946,005 cycles (dqg_wall.py over UI). This matches PMC k_dqg 4.00e6 cycles per SIMD (PLAN/roofline).
- **Registers.** VGPR 991, LDS 8704 B (`arms/r29/.dump/dqg/k_dqg_0/21_final_isa.s:4414, 4399`). That is 1 wave per SIMD and 1 wave per workgroup: DQ_NW=1, DQ_BQW=64, DQ_PF=True, DQ_U2=True (K:1171-1175).
- **Hot loop.** Vaddr 10808..19476, 1018 static instructions, each hit 1018 times, so 1018 loop trips over the 16 waves (CSV). This is `kvloop_full`, unrolled by 2 (K:1360-1386).
  - One trip is two `_body` calls, 64 kv rows × 64 q rows.
  - Work per trip: 192 WMMA (S 64 + dP 64 + dQ 64), 64 buffer_load_b128, 32 ds_store_b128, 32 ds_load_tr16_b128.
  - Source for these counts: CSV, counted with dqg_loop.py.
- **Where the wave time goes** (dqg_wall.py / budget):

| Part | Share of wave time | Cycles |
|---|--:|--:|
| Loop | 92.2% | 3,639,990 |
| Epilogue (256 × `buffer_store_b16` + cvt) | 4.3% | 167,973 |
| Mask loop (2 blocks per wave, not prefetched; hitcount 32, Vaddr 20420-26216) | 1.8% | 69,283 |
| Prologue | 1.3% | 51,340 |
| Launch and tail | 0.2% | — |

- **Time per trip.** 3576 cycles on average, or **18.6 cycles per WMMA**. At 8 cycles per WMMA the floor is 1536, so WMMA-pipe use is **43%**.
- **Two populations of waves.**
  - Fast waves (wv1-3, 5-7, 9-11, 13-15; 595 trips): 3321 cycles per trip, median 3260.
  - Slow waves (wv0, 4, 8, 12; 423 trips): 3934 cycles per trip.
  - All of the extra time in the slow waves is in the memory path. Of the +613 cycles: loadcnt +220, VMEM issue +195, ds_load +96, xcnt +45, dscnt +29, ds_store +27 (dqg_budget.py).
  - I have not explained why every 4th wave on the SIMD is slow. They also have the most trips (127/98/97/101), but wv5 (93 trips) is fast. It looks like L2/TA contention that the short prefetch lead exposes (see section 2).

## 1. Cycle budget per loop trip, by class

Wall cycles, trip-weighted over all 16 waves (dqg_budget.py). "% kernel" = trips × cycles ÷ 3,946,005.

| class | all | fast | slow | % of loop | % of k_dqg |
|---|--:|--:|--:|--:|--:|
| WMMA issue (192; gaps after a WMMA, not into DS) | 1109 | 1109 | 1109 | 31.0 | 28.6 |
| of which gaps ≥12 cyc (dependent/operand) | 158 | 158 | 158 | 4.4 | 4.1 |
| VALU total | 1152 | 1152 | 1152 | 32.2 | 29.7 |
| · v_pk (256 mul + 128 add) | 543 | | | 15.2 | 14.0 |
| · v_exp (128) | 287 | | | 8.0 | 7.4 |
| · v_cvt_pk_bf16 (64) | 94 | | | 2.6 | 2.4 |
| · other VALU (30 address v_or, ...) | 144 | | | 4.0 | 3.7 |
| · v_nop (31) | 84 | | | 2.3 | 2.2 |
| s_wait_loadcnt | 302 | 210 | 431 | 8.4 | 7.8 |
| VMEM issue (64 loads, back-pressure) | 214 | 133 | 328 | 6.0 | 5.5 |
| WMMA→DS switch (WMMA followed by ds_*) | 181 | 181 | 181 | 5.1 | 4.7 |
| SALU (s_clause etc.; ~60 is VMEM back-pressure seen at s_clause) | 167 | 167 | 167 | 4.7 | 4.3 |
| ds_load_tr issue (32) | 151 | 111 | 207 | 4.2 | 3.9 |
| s_wait_xcnt | 124 | 105 | 150 | 3.5 | 3.2 |
| ds_store issue (32) | 122 | 111 | 138 | 3.4 | 3.1 |
| s_wait_dscnt | 55 | 43 | 72 | 1.5 | 1.4 |
| **total** | **3576** | **3321** | **3934** | 100 | 92.2 |

How to read it:
- WMMA plus VALU is 2261 cycles against a 1536 floor, so **about 725 cycles per trip (20% of the loop) is VALU that is not hidden**.
- The memory path costs 302 + 214 + 124 + ~60 = **about 700 cycles per trip (about 18% of the kernel)**: loadcnt + VMEM issue + xcnt + back-pressure at s_clause.
- DS plus switch costs 151 + 122 + 181 + 55 = **509 cycles (13%)**.

## 2. What each wait is waiting on

This is an in-order loadcnt/xcnt model of the loop, cyclic across the back-edge (dqg_waits.py). "lead" is the CSV latency-sum distance from the load's issue to the wait.

| Vaddr (CSV line) | wait | stall/trip | waits for | lead |
|---|---|--:|---|--:|
| 11344 (L670) | `s_wait_loadcnt 0x20` | **203.9** (CSV), 302 wall | Block ii's K/V, loaded by the **previous trip's tail clump** (18676-19440, CSV L1813-1913); the last of these is `19440 buffer_load v[2:5]` | **214-490 cyc** |
| 11128 (L647) | `s_wait_xcnt 0x2` | **109.0** | Address VGPRs v90/v91/v92 of loads 11088/11104/11116 must leave the TA before `v_or_b32 v90, 0xc0, v69` (11132/11144/11156) rewrites them for loads 11168-11192 | 3-122 |
| 11140 / 11152 | `s_wait_xcnt 0x1 / 0x0` | 7.6 / 4.3 | Same mechanism | |
| 13464-13744 | `s_wait_loadcnt 0x1f..0xb` (10 waits) | ≤7 each | Head clump (10828-11328) loads for block ii+1: K rows before the body-2 ds_store of K | ~1400-1540 |
| 15668 / 16272 / 16500 | `s_wait_loadcnt 0x7 / 0x3 / 0x0` | 0-2 | Last V/K fragments of the head clump, used by body-2 S/dP | 2386-2875 |

Mapping back to source:
- **Head clump** (32 loads, cum 0-267 in the fast-wave phase listing) is body #1's `nxt = _ldkv(kv0_n)` (K:1279 → `_ldkv` K:1268-1275, called from K:1379-1381). It prefetches block ii+1 and has a 1400-2900 cycle lead. **It never stalls.**
- **Tail clump** (32 loads, cum ~2890-3314) is body #2's `_ldkv(jj*KV_STEP)` (K:1382-1385). As designed in the U2 comment (K:1349-1356), the compiler places it **after** the carried set is dead, so it writes onto the carried registers. That puts it at the very end of the trip, only about 200-500 cycles before `11344` consumes it. **This one short-lead prefetch is the entire loadcnt stall**: 302 cycles per trip, 7.8% of the kernel. Waves where memory is slower (slow group) see 431.
- The **xcnt stalls** exist only because the compiler rebuilds 3 address VGPRs with `v_or_b32 vN, <const>, v69/v70`. The constants are 0xc0 and 0xe0, column offsets from `half + dt*4` (K:1226-1227). They could be the buffer instruction's immediate `offset:` field, as it already does in the prologue at `8392 ... offen offset:96`. **Folding them removes the three xcnt waits: 124 cycles per trip, 3.2%.**
- **VMEM issue back-pressure.** 64 b128 wave32 loads (512 B each) arrive in two bursts. One issue stall alone reaches 269 cycles per trip in wv0 (Vaddr 11316, dqg_wavecmp.py wv0 vs wv1: 269.6 vs 1.0).

Constraint from history: h37 (`0923__flydsl/hint.md:2859`) says spreading k_dq's loads over the WMMAs with `sched_group_barrier(VMEM,1)/(MFMA,3)` cost +22% (on B0, r20-era k_dq). So **keep the tail loads as a clump but move it earlier**. The clump could start as soon as body 2's last S/dP use of the carried set is done: the waits at 16500 fall around cum 2233, versus the clump's current ~2890-3314. That gives about 700-1000 more cycles of lead with the same register lifetime. This is a different lever from g63 (prefetch depth 2), which h37 tested.

## 3. Phase layout of one trip

Fast waves, wall cumulative cycles (`notes/dqg_runs_fast.txt`, trip ≈ 3350):

| cum | phase |
|---|---|
| 0-267 | Head clump: 32 × buffer_load (block ii+1), with xcnt 106 + 8 + 5 inside it |
| 267-389 | `s_wait_loadcnt 0x20` (122): block ii's data from the previous tail clump |
| 389-509 | 16 × ds_store K(ii) (K:1296-1299), 58 cycles; 2 × ds_load_tr |
| 509-1106 | Body 1 S/dP: 28 WMMAs back to back at 7.75 cyc (217), then kt0 softmax interleaved 1 WMMA : 5-7 VALU (≈12-13 cyc per WMMA) plus 7-wide v_exp bursts (17 cyc) |
| 1106-1330 | 14 × ds_load_tr: all of body 1's K^T into registers (K:1340-1344); then body 2's 16 ds_store of K(ii+1) gated by loadcnt 0x1f..0xb (single LDS buffer, 8704 B) |
| 1330-1930 | kt1 softmax + body 1 dQ + body 2 S/dP; contains the one fully dense run, 42 WMMAs in 335 cyc = 8.0 (Vaddr 15196-15660) |
| 1930-2233 | Remaining body 2 S/dP; last head-clump waits (0x7, 0x3, 0x0) |
| 2233-2575 | Body 2 softmax, largely **not overlapped**: `X ×30` 59 cyc, `v ×22` 40, `v ×74` 74 (Vaddr 17428-18348) |
| 2575-3314 | Body 2 dQ: `W ×4` groups at ~54 cyc (13.5 per WMMA), six ~30-cycle WMMA→ds_load_tr switches (13296, 18364, 18664, 18796, 19180, ...), with the tail clump (block ii+2) interleaved |

## 4. VALU composition

- **Per WMMA** (per trip ÷ 192; CSV opcode counts): 3.36 VALU per WMMA.
  - v_pk_mul_f32 1.33, v_pk_add_f32 0.67, v_exp 0.67, v_cvt_pk_bf16 0.33, v_nop 0.16, v_or (address) 0.16, other ~0.04.
- **Per softmax element** (128 per lane per trip): 2 pk_mul + 1 pk_add, i.e. 6 fp32 ops, plus 1 exp and 0.5 cvt. This is the VF_Q=False path (K:1323-1333): ×scale, −lse, ×LOG2E, −delta, ×pf, ×scale.
- **Under the roofline cost model** (`0930__roofline/REPORT.md`: ~4 plain VALU and ~2 exp hide per WMMA; packed ≈2×), the load in plain-VALU equivalents is about 6.1 per WMMA against 4 of capacity. That is at least **~400 cycles per trip of irreducible exposed VALU even with perfect interleave**. The measured 725 exposed cycles also include the non-overlapped runs listed in §3.
- **The VF_Q axis is closed.** fma softmax on top of U2 lost: u2f +2.0% op, u2b +6% (`0927__b0/lab-kdq/REPORT.md:42-43, 87-88`, h70).

## 5. DS usage

- **LDS traffic** per trip: 32 × ds_store_b128 (K rows, 16 KB) + 32 × ds_load_tr16_b128 (K^T, 16 KB) = 32 KB in 3576 cycles, about **9 B/clk**. That is 3.5% of the 256 B/clk segment. **Bandwidth is irrelevant here.**
- **The cost is issue time and switches:** ds 273 + switch 181 + dscnt 55 = 509 cycles per trip, 13% of the kernel. The switches are about 30 cycles each, matching the roofline's ~29. With 1 wave per SIMD nothing hides them.
- **LDS holds only K**, because dQ = dS·K needs K^T. Q, dO, V and the S/dP operands all stay in VGPRs, which is why VGPR is 991.

## 6. k_dqg vs k_dkdv (same trace, dispatch 10, loop 15628-19708, 3780 trips)

| | k_dqg | k_dkdv |
|---|--:|--:|
| VGPR / LDS | 991 / 8704 B | 729 / 70656 B (`.dump/dkdv/k_dkdv_0/21_final_isa.s:2526, 2511`) |
| WMMA per trip | 192 | 64 |
| wall per trip | 3576 | 1715 (median 1642) |
| **cycles per WMMA / WMMA-pipe use** | **18.6 / 43%** | **26.8 / 30%** |
| VALU per WMMA | 3.36 (no v_mov) | ~5.4: v_nop 66, v_mov_b64 64 (rotation), pk_mul 64, pk_add 32, exp 32, cvt 32 per 64 WMMA |
| DS ops per WMMA / switches | 0.33 / ~6 per 192 | 1.25 / 7 per 64 |
| top wait | loadcnt 0x20 302 (tail-clump K/V, lead ~214-490) | dscnt 200 (11.7%); loadcnt 0x23 165: waits on `buffer_load_b32` LSE/DEL at 15792, same trip, lead ~452 (dispatch_10 csv L1600) |
| xcnt | 124 (address VGPR recycling) | none in loop |
| share of bwd cycles (PMC) | 4.00e6 / 11.44e6 = 35% | 7.26e6 = 63% |

Structural difference:
- k_dqg is the denser kernel. One K/V fragment feeds 4 q tiles (BQW=64), and S/dP read Q/dO straight from registers.
- Its losses are memory-lead (loadcnt/xcnt/VMEM issue, ~18%) plus exposed softmax VALU (~20%).
- k_dkdv loses mainly to DS round-trips and switches, plus the unprefetched LSE/DEL b32 loads.
- k_dqg recomputes S and dP. That is 2 of its 3 GEMMs: **128 × 1018 of the WMMAs, or 28.6% of the fly bwd WMMA total** (7 GEMMs vs the ASM's 5). The ASM avoids this by doing dQ in the main kernel with `buffer_atomic_add_f32`.

## 7. Levers from this trace, ranked by exposed k_dqg cycles

1. **Move body 2's prefetch (tail clump, K:1382-1385) earlier, kept as a clump.** Target: 7.8% loadcnt + part of the slow-wave excess (+613 cycles on 42% of trips ≈ 6.6% of the kernel; this overlaps with the loadcnt share). Place it right after the last head-clump consumer (~Vaddr 16500) instead of at 18676+. It must not be spread: h37 measured +22% for that. **Test:** ISA check that loadcnt 0x20's lead exceeds ~1000 cycles, then ATT.
2. **Fold the column offsets into the buffer `offset:` immediate.** Targets xcnt 3.2% + 30 v_or + part of VMEM issue. Write the address as base + constant bytes, not an OR with `half`, in `gfrag2` K:1230-1233; for example precompute `t0` and add `dt*4`, `+2` as constants. **Test:** compile-only; the xcnt waits should disappear from the loop.
3. **Bigger redesign (not tested): load K straight into LDS asynchronously**, the way ASM uses TDM. This removes 16 ds_store + 64 VGPR per block and would allow a real double-buffer, at the cost of ds_loads for the S operand, which adds switches.
4. **Closed / low value:**
   - Epilogue through LDS: 4.3% of wave time here, but measured null (lab-kdq c_epi, h70).
   - VF_Q: loss (h70).
   - 2 waves per SIMD (o1, BQW=32): +14.7% loss (lab-kdq round 1, line 29).
   - sched_barrier between the bodies: +16-21%.

The mask loop (1.8%) and prologue (1.3%) are small.