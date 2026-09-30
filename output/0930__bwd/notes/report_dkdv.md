# k_dkdv (r29) ATT deep-read: per-iteration cycle budget and stall sources

All numbers are per loop iteration of one wave unless stated otherwise. Scratch outputs I generated:
- `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/notes/dkdv_loop_wall.txt`: one row per loop instruction, giving wall cycles per iteration measured from the wave timelines (`w`), plus stall, latency and a running total.
- `.../notes/dkdv_loop_decoded.txt`: the loop ISA with `s_set_vgpr_msb` applied, so registers show their real numbers (V0-V1023). I checked the decoding against the `/*vNNN*/` comments in `21_final_isa.s`.
- `.../notes/dkdv_loop.txt`: the CSV-based version of the same table.
- `.../notes/dkdv_wall_agg.json`: raw per-instruction totals.

Short names used below:
- **CSV** = `probe/p2_r29/stats_ui_output_agent_27913_dispatch_10.csv`
- **WALL** = `notes/dkdv_loop_wall.txt`
- **ISA** = `arms/r29/.dump/dkdv/k_dkdv_0/21_final_isa.s`
- **SRC** = `arms/r29/kernels.py`

## 0. Method and a warning about the CSV "Latency" column

- **Hot loop.** It covers Vaddr 15628 to 19708: CSV:1374 (`s_add_co_i32 s14,s4,1`) through CSV:2032 (`s_cbranch_scc1 64515`). Every instruction there has Hitcount 3780. In the ISA it is `.LBB0_11` (ISA:1384-2043), and it is `qloop_full` (SRC:527-541) running `_body` (SRC:355).
- **Instruction counts per iteration:**
  - 550 instructions, 64 of them WMMA
  - 40 `ds_store_b128` and 40 `ds_load_tr16_b128`
  - 32 `buffer_load_b128` and 4 `buffer_load_b32`
  - 64 `v_mov_b64` and 66 `v_nop`
  - 64 `v_pk_mul`, 32 `v_pk_add`, 32 `v_exp`, 32 `v_cvt_pk_bf16`
- **What the trace covers.** 7 waves were traced, all on the same SIMD (`se0_sm3_sl0_wv0..6`) and running one after another; the prologue has Hitcount 7. They ran 3780 full iterations in total: 788, 88, 828, 124, 896, 40 and 1016 per wave. The 28-hit block is the masked loop (4 iterations per wave).
- **Wall time per iteration** (timestamp difference between successive loop heads, from the `ui_output_agent_27913_dispatch_10/se*_wv*.json` files):
  - mean 1715 cycles, median 1641, minimum 1474
  - p10 1518, p90 2029
  - This agrees with the PMC figure of 1766 (the PMC count includes the prologue and epilogue).
- **The CSV Latency column overstates WMMA.** Its per-iteration sum is 1937 cycles, 13% more than wall time, because every WMMA is charged 8 cycles even when the next instruction issues 1 cycle later. `attsum`'s "wmma 33.9%" is therefore wrong. I use wall attribution throughout: `w(i)` = issue time of the next instruction minus issue time of instruction i. These values add up exactly to 1714.9 (WALL, last line).
- **"stall = 7" on WMMAs is not a dependency.** 24 WMMAs each show stall 26460 = 7 × 3780 (CSV, e.g. at Vaddr 18940-18996 and 16924-16984). That is simply back-to-back issue at one WMMA every 8 cycles: 7 cycles waiting plus 1 cycle issuing.

## 1. Per-iteration budget

### 1a. By phase, in program order (from the timelines)

| Phase | Vaddr | Wall | WMMA | Excess over 8 cyc/WMMA | Share of 1715 |
|---|---|--:|--:|--:|--:|
| A. Loop index math in SALU (`qi=ii//G`, `qj=jj//G`, SRC:531,539) | 15628-15748 | 46.0 | 0 | 46.0 | 2.7% |
| B. VALU address math + 4 `buffer_load_b32` for next iteration's lse/delta (SRC:293) | 15756-15852 | 53.2 | 0 | 53.2 | 3.1% |
| C. `s_wait_loadcnt 0xc` + 7 ds_store (Q/dO, hh=0, SRC:393-395) + DS→WMMA switch | 15864-15920 | 51.7 | 0 | 51.7 | 3.0% |
| D. 2 WMMA + 14 `v_or` addresses + 32 `buffer_load_b128` prefetch (SRC:305) + VMEM→WMMA | 15932-16472 | 116.5 | 2 | 100.5 | 5.9% |
| E. 1 WMMA → WMMA→DS switch → 4 ds_store → DS→WMMA switch | 16488-16532 | 57.7 | 1 | 49.7 | 2.9% |
| F. S/dP WMMAs + softmax + 6 ds_store (incl. P) + two switches | 16536-16900 | 131.5 | 7 | 75.5 | 4.4% |
| G. S/dP WMMAs + softmax | 16912-17136 | 87.0 | 10 | 7.0 | 0.4% |
| **H. `s_wait_loadcnt 0x23`** | 17140 | **133.9** | 0 | 133.9 | **7.8%** |
| I. S/dP WMMAs + exp + dS | 17144-17776 | 151.6 | 12 | 55.6 | 3.2% |
| J. dS / scale / cvt VALU run with no WMMA (SRC:445) | 17784-18348 | 91.0 | 0 | 91.0 | 5.3% |
| K. 23 ds_store + 40 `ds_load_tr16` (SRC:449-454, 466-486) | 18360-18860 | 123.5 | 0 | 123.5 | 7.2% |
| **L. `s_wait_dscnt` 0x6 / 0x2 / 0x0** | 18872-18924 | **214.7** | 2 | 198.7 | **11.6%** |
| **M. 30 dK/dV WMMAs interleaved with 64 `v_mov_b64` + 51 `v_nop`** | 18928-19700 | 432.5 | 30 | **192.5** | **11.2%** |
| N. Back-edge `s_cbranch` | 19708 | 24.1 | 0 | 24.1 | 1.4% |
| **Total** | | **1714.9** | 64 | **1202.9** | 70.1% |

The matrix-pipe floor is 64 × 8 = 512 cycles, which is 29.9% of the wall. The other 1203 cycles are exposed non-WMMA time.

### 1b. By category (the same 1715 cycles, split so nothing is counted twice)

| Category | Cycles | % | Where |
|---|--:|--:|---|
| WMMA issue floor | 512 | 29.9 | 64 × 8 |
| `s_wait_dscnt` stalls | 199 | 11.6 | 0x6: 188.0 (WALL:394), 0x2: 10.7 (WALL:398), 0x0: 1.9 |
| `s_wait_loadcnt` stalls | 134 | 7.8 | almost all at 0x23 (WALL:183, 133.9). The other 8 loadcnt waits total 31.4 cycles, only 1-3 cycles each once their clause gaps are removed (WALL:46,49,245,396,411,418,459,501) |
| Rotation `v_mov_b64` + WAR `v_nop` in the dK/dV tail | 192.5 | 11.2 | phase M; 16 WMMA gaps of 20-22 cycles instead of 8 (WALL:410-548) |
| DS issue, final phase (23 st + 40 ld) | 123.5 | 7.2 | phase K; stores slow to 3-4.6 cycles each from 18440 onward (back-pressure, WALL:341-353) |
| Mid-body Q/dO stores + DS↔WMMA switches | ~161 | 9.4 | C + E + the store/switch part of F. Switch gaps: 25.8 (WALL:54), 21+24.7 (WALL:112,116), 27+25.5 (WALL:151,157) |
| VMEM prefetch issue | 100.5 | 5.9 | phase D; 32 × ~1.2 cycles + 4 `s_clause` (~3 each) + 14 address ORs + VMEM→WMMA 12 (WALL:111) |
| VALU not hidden behind WMMA | ~169 | 9.9 | J 91; I 56; F ~16; G 7 |
| Loop overhead (SALU division, address setup, branch) | 123.3 | 7.2 | A 46 + B 53 + N 24; includes VALU→VMEM address dependencies of 16 and 17 cycles (WALL:38,76) |

### 1c. VALU split by instruction type

Cycles here are wall-attributed issue cycles, including those that overlap a WMMA. They do not add up to the "not hidden" figure above.

| VALU type | Instructions | Cycles | Source |
|---|--:|--:|---|
| Softmax recompute: S·scale (`v_pk_mul` with V642), −lse (`v_pk_add` with V650/V648), ·log2e (`s[6:7]`) | 40 | 73.6 | SRC:442-443 |
| exp (`v_exp_f32`) | 32 | 75.0 | SRC:443 (`_exp2`) |
| dS: dP−delta (V646/V644), P·(dP−D), ·scale | 56 | 74.0 | SRC:445 |
| cvt/pack to bf16 | 32 | 33.0 | SRC:444-445 (`.to(BFloat16)`) |
| Address/index math (`v_or`, `v_mad`, `v_add_lshl`) | 24 | 51.0 | SRC:278-305 (`_ldqd`) |
| Rotation copies (`v_mov_b64` ×64, `v_mov_b32`/dual ×3) | 67 | 129.0 | loop-carried `nxt` (SRC:368, 541) |
| `v_nop` (4 after-WMMA-result groups + 16 WAR groups) | 66 | 98.0 | compiler-inserted hazard padding |
| SALU (`s_*`) + `s_clause` | 41 | 64.0 | SRC:531-539 |

## 2. What each big wait is waiting for

### `s_wait_loadcnt 0x23` at Vaddr 17140 (ISA:1611, CSV:1600): 133.9 cycles per iteration (7.8%)

- **What it counts.** Outstanding VMEM loads at this point are all from the current iteration, because the previous iteration's are drained by `s_wait_loadcnt 0x0` at 19448. They are:
  - 4 `buffer_load_b32` at 15792-15832 (ISA:1424-1428)
  - 32 `buffer_load_b128` at 16088-16472

  That makes 36. Waiting until at most 35 remain means the oldest one must finish: `buffer_load_b32 v181 /*v693*/` at 15792. This is the **lse load for hh=0 of the next iteration** (`_ldqd`, SRC:293, called through `nxt = _ldqd(qt_n, gh_n)` at SRC:368).
- **Why it is waited on here.** The next instruction is `v_mov_b32 v138/*v650*/, v181/*v693*/` (ISA:1612, CSV:1601). This is the loop-carried copy into `pre[0]`, the register holding lse_q for hh=0 (SRC:404). The scheduler placed it immediately after V650's last use in this iteration, the `v_pk_add` at 17104. The copy forces a wait for a prefetch that is only needed in the *next* iteration. The load was issued about 464 cycles earlier.
- **Bimodal behaviour.** Measured per iteration from the timelines:
  - median wait 3 cycles
  - 29.4% of iterations (1111 of 3780) wait more than 100 cycles, averaging 434
  - This makes the global-load latency sometimes about 900 cycles
  - The slow-iteration spikes show no pattern by iteration index mod 8
- **Why it matters.** It accounts for **all** of the spread between iterations. Phase H is 3 cycles in the fastest 10% and the middle 80% of iterations, and 604 in the slowest 10%; every other phase is identical across those groups except L.
- **The other lse/delta copies do not stall:**
  - `s_wait_loadcnt 0x22` at 17624 → `v_mov v136/*V648*/ ← v214/*V726*/` (lse, hh=1): 2 cycles
  - `s_wait_loadcnt 0x20` at 18888 → `v_dual_mov` of delta (V727, V728): 3 cycles
  - They are placed later, so their loads have already arrived.
- The 32 b128 prefetches wait at `s_wait_loadcnt 0x0` at 19448 (CSV:1972). Stall there is 0.02 cycles per iteration: they land within about 1310 cycles of issue.

### `s_wait_dscnt 0x6` at Vaddr 18872 (ISA:1868, CSV:1857): 188.0 cycles (11.0%); plus 0x2 at 18908: 10.7, and 0x0 at 18924: 1.9

- **What it counts.** DS operations issued in order before the wait (phase K):
  - 23 `ds_store_b128`:
    - 3 P/dS stores to `v159`
    - 16 Q/dO hh=1 stores to `v145` (SRC:393-395; the compiler delayed these to the very end)
    - 4 P/dS stores to `v160` (SRC:449-454)
  - then 40 `ds_load_tr16`:
    - 32 b_do/b_q loads at 18544-18796 (SRC:480-481)
    - then 8 a_p/a_ds loads at 18804-18860 (SRC:485-486)
- **What must complete.** `dscnt ≤ 6` means everything except the last 6 loads (18820-18860) has to finish: 23 stores + 34 loads, about 29 KB.
- **Why.** The first dK/dV WMMA (18876) reads A = V[694:701]. That is a_p for kh=0, loaded by the 33rd and 34th loads (18804 and 18812). The compiler issues a_p/a_ds **last**, and DS operations complete in order, so the first dV WMMA waits for all of b_do/b_q and all 16 delayed Q/dO stores.
- The 0x2 wait (18908) is for a_ds for kh=0 (V[710:717], loads 18836/18844), used by WMMA 18912. The 0x0 wait covers the last pair, V[718:725].
- **Timing.**
  - Median wait 160 cycles, p10 148, p90 251, minimum 113.
  - Stores go through at about 1 cycle each until the 12th; after that the LDS queue pushes back and they take 3-4.6 cycles each (WALL:341-353). The first 10 tr loads take about 3 cycles each.
  - From the first store to the first WMMA is 294 cycles (median) for about 32 KB, i.e. about 110-124 B/clk for this one wave.
- **Is it contention with other SIMDs?** The `other_simd_*.json` files record only VMEM and LDS events. Their LDS issue rate during our DS window is 0.018-0.021 per cycle, below the 0.047 per cycle they average overall. So this is this wave's own in-order drain, not other SIMDs competing for the LDS.

### Other waits

`s_wait_loadcnt` 0xc (15864), 0x4 (15884), 0x7 (19016), 0x6 (19044) and 0x0 (19448) each stall 0-2 cycles; they are covered.

## 3. WMMA stalls and their causes

1. **True accumulator RAW stalls: none.** In the S/dP section, `s_acc` and `p_acc` alternate (SRC:410-415), giving 2 independent chains. That exactly covers the 16-cycle dependent latency, and the WMMA gaps are 8 cycles (for example the 24 WMMAs with stall 7 and w 8).
2. **WMMA→DS switch:**
   - 16488: w = 29, i.e. 21 extra cycles before `ds_store` 16500 (WALL:112)
   - 16836→16848: the cvt is charged 28, then `ds_store` (WALL:151)
   - Total about 48 cycles.
3. **DS→WMMA switch.** The last store before a WMMA is charged 25.8 / 25.7 / 26.5 cycles (15920, 16524, 16900; WALL:54,116,157). Total about 75 cycles. The phase K→L transition is hidden inside the dscnt wait.
4. **VMEM→WMMA and VALU→DS transitions:**
   - 16472 `buffer_load` → WMMA: 13 cycles (WALL:111)
   - 18348 cvt → `ds_store`: 18 cycles (WALL:330)
5. **WMMA result → VALU (RAW) padding.** Groups of 4 `v_nop` at 16560, 16992 and 17652 (after the S/dP accumulators, before `v_pk_mul`), plus 19020: about 7-8 cycles per group, about 30 cycles in total.
6. **WAR on WMMA sources in the tail.** Each `v_mov_b64` overwrites a register that the WMMA just issued is still reading as its B operand. For example, WMMA 19008 reads B = V[402:409], and 19036 writes `v_mov_b64 V[402:403] ← V[622:623]`, the prefetched Q from `buffer_load` at 16388. The compiler inserts 3 `v_nop` before each group of 4 `v_mov_b64`. Result: the 16 tail WMMAs run 20-22 cycles apart instead of 8, about 192 cycles in total.
   - **Cause.** The loop-carried prefetch (`nxt`, SRC:368 → `st[NST:]`, SRC:541) is allocated to the same registers that hold the tr16 B operands during the dK/dV GEMM. The prefetch therefore lands in staging registers V[514:641] and is copied back on the back-edge.
7. **Small leftovers:**
   - 18996: w = 15, i.e. 7 cycles extra before 19008, the change of A operand from V710 to V694 (WALL:409)
   - 19240: w = 9 (+1)
   - 18912: +7 after the dscnt 0x2 wait

## 4. How much time each stall class would recover if hidden

These are upper bounds as a fraction of k_dkdv loop time (1715 = 100%), measured on the r29 traces.

| Class | Cycles | Upper bound | Realistic fix and expected gain |
|---|--:|--:|---|
| dscnt drain at the dK/dV start (L) | 199 | 11.6% | Issue a_p/a_ds before b_do/b_q, and move the 16 hh=1 Q/dO stores out of the final DS group. The first WMMA then waits for about 12 DS operations instead of 57 (at ~4.9 cycles each ≈ 60 cycles). Expect about 130 cycles, **~7.5%** |
| Rotation `v_mov_b64` + WAR `v_nop` (M) | 192.5 | 11.2% | Ping-pong the carried registers (unroll by 2) or prefetch straight into registers that are not B operands. **But:** the ku2 arm (unrolled by 2: 729→644 VGPR, 0 `v_mov_b64`) measured **+7.3% slower** on B0 (`0927__b0/lab-kdq/REPORT.md:30`). That run was pre-reflash and has no ATT. Take an ATT of ku2 before trusting this row |
| `s_wait_loadcnt 0x23`, the lse copy placed too early (H) | 134 | 7.8% | Place the V650←V693 copy after the loop's `loadcnt 0` (19448), or prefetch lse/delta 2 iterations ahead. Neither changes the computed values (bit-identical output). Removes the 29% of iterations that stall about 434 cycles. **~7.5%**, and removes all the spread between iterations |
| Mid-body Q/dO stores + 5 switches (C/E/F) | ~161 | 9.4% | Merge the 17 early Q/dO stores into one group: 3 fewer switches ≈ 75-100 cycles, but adds about 17 operations to the final drain. Net about 3-5%. Better: TDM (tensor load to LDS, as ASM does) for Q/dO, which also removes D and most of M |
| VMEM prefetch issue (D) | 100.5 | 5.9% | Spread the 32 `buffer_load`s across the WMMA-dense tail, or use TDM (1-2 instructions). ~4-5% |
| Loop overhead: SALU division + address + branch (A+B+N) | 123 | 7.2% | Increment (qi, gh) counters instead of the runtime division `ii//G` twice (~40 cycles) and precompute address offsets. **~3%** |
| VALU not hidden (J + I/F/G) | ~169 | 9.9% | J (91) is the dS/cvt run with no WMMA underneath. It could overlap if kh=0's dK/dV started before kh=1's dS, which needs a split LDS round trip. VF_KV (fma) measured null on B0 (`lab-kdq/REPORT.md:28`). ~3-5% at most |
| DS issue in the final phase (K) | 123.5 | 7.2% | Only removable by fewer or wider DS operations (the Q/dO LDS staging is the big part: 32 st + 32 tr loads per iteration). TDM path or 2 waves per SIMD |
| WMMA floor | 512 | — | Reached only if everything above is hidden: 1715→512 is 3.35×. With 2 waves per SIMD, L + K + switches (~480 cycles, 28%) become hideable in principle, but that needs ≤512 VGPR from today's 729 |

**Combined, low-risk and bit-identical:** H (7.5%) + reordering the final DS phase (~7.5%) + removing the SALU division (~3%) ≈ **16-18% of k_dkdv loop cycles**. k_dkdv is about 63% of fly bwd cycles (7.26e6 of 11.44e6), so that is roughly 10-11% of the op, about 6.58 → ~5.9 ms against ASM's 5.503. Closing the rest needs TDM staging of Q/dO (the D + C/E/F + M + part of K classes, ~500 cycles, 30%) or 2 waves per SIMD.

**Correction to the roofline model** (`0930__roofline/REPORT.md:129-141`). It assigned about 600 cycles to "memory wait (loadcnt)". The trace shows only 134 cycles of global-load wait (all from the lse copy), 199 cycles of DS drain wait, 192 of rotation copies and nops, and about 161 of switches and mid-body stores.