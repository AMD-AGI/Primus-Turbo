# arm s3_trim: ATT-guided trim of exposed non-WMMA work in s3's k_dkdv hot loop

Base: `arms/s3` (k_dkdv = dkdv_tdm3, k_dqg = dqg_ts, impl.py side stream). Only `kernels.py`
(`_dkdv_impl` + switch constants) differs; `impl.py`, `_env.py`, `__init__.py` are s3's.
k_dqg and k_delta compile to byte-identical ISA vs s3 (`_dqg_trim/` vs `_dqg_ref/`, `cmp` of
21_final_isa.s). The TDM ring (issue points, stages, tensor waits) is untouched. The
division/modulo SALU (s3_df's job) is left alone.

Status: compile-only plus CPU proof. **Not run on a GPU.**

## 1. Where s3's k_dkdv iteration goes (ATT wall attribution)

`att/wall.py` computes w(i) = issue(next) − issue(i) from the 8 per-wave timelines in
`probe/att_s3/ui_output_agent_62836_dispatch_10`, following the report_dkdv.md method. The
output is `att/dkdv_wall_s3.txt`: 4068 full iterations, **1439 cycles/iter** (median 1433),
Vaddr 15260-18808.

| phase (s3) | wall/iter | what is exposed |
|---|--:|---|
| loop head: SALU (div/mod, TDM descriptors) plus LSE/delta address | 207 | `v_or` 8 + `v_add_lshl` 7+1 + `s_clause` 16+3 (VALU→VMEM) = **35** for the LSE/delta address |
| S/dP: 32 WMMA + softmax (to the first ds_store) | 506 | floor 256; 37-VALU tail after the last WMMA, 23 v_nop |
| **DS phase: 8 st + 40 tr16 + 32 readback** | **454** | **17 address VALU (`v_add_nc_u32`/`v_dual_add`) = 256.6**, each followed by a 16-18 cycle VALU→DS stall. The ds op before each VALU takes 7 instead of 1: **98.1** of DS→VALU switch. So 355 of the 454 cycles are address math |
| dK/dV: 32 WMMA | 272 | floor 256 |

Cause of the DS-phase VALU: `lds_do + lane_r*272 + (lane_c + dtile*16)*2`. LLVM LICM hoisted
`lane_r*272 + lane_c*2 + dtile*32` into 8 invariant VGPRs (v14, v34-v40), one per dtile, so the
constant could no longer fold into the DS `offset:` field. Every tr16 pair then needed a fresh
`v_add` with the loop-variant stage base. The readback had the same problem: 4 adds, including a
`-0x2200` that can never be an immediate.

## 2. Changes (kernels.py; each one has its own switch, and False gives s3's code)

| switch (line) | change |
|---|---|
| `TRIM_DSADDR` (:72) | :366-367 lane-only invariants `lb_tr = lane_r*272 + lane_c*2` and `lb_rd = row*272 + half*16`. :407 `_rdqd(stage_off, rbase)`: 32 readback loads at `rbase + const`. :572 / :476 one base each (`lds0+cur+lb_tr`, `lds0+rb_off+lb_rd`) computed **before** the DS fence. :607-640 B tr16 at `tr_base + dtile*32 (+8704 for Q)`. Every per-op difference is now a constant, and ISel folds it into `offset:` (0..13280) |
| `TRIM_DSORDER` (:70) | :607-626 (full loop only) DS burst in dK/dV consumption order, each group fenced: stores, a_p0, b_do×8, a_ds0, b_q×8, a_p1, a_ds1, then readback. Waits step `0x3e` (4 WMMAs), 0x3b, 0x3a, 0x37, 0x36 … instead of the 8th WMMA waiting at 0x24 |
| `TRIM_LSEOFF` (:60) | :291-315 LSE/delta prefetch = `raw_ptr_buffer_load(rsrc, voffset=4*row` (invariant VGPR)`, soffset=readfirstlane(4*(base_l+q0)) + 64*hh)`. No VALU, no VALU→VMEM stall. The masked body keeps s3's loads (`trim=False`, :489), because the new form perturbed its schedule (3→9 W↔D switches) |
| `TRIM_SDP="mix"` (:62) | :468-480 full loop does TDM first, then `_ldl`. The SGPR soffsets are no longer recycled by the TDM SALU (with `_ldl` first, LLVM added `s_wait_xcnt 0x0`). The 4 loads then sit in the S/dP sched region. "iso" (own region) is kept as an option |
| `TRIM_LATEROT` (:68) | :680 `sched_barrier(0)` before the body's return. The LSE/delta back-edge copies and their `s_wait_loadcnt 0x1/0x0` now sit after the 32 dK/dV WMMAs instead of after the first one. This keeps about one full iteration of latency cover, which the shorter DS phase would otherwise have eaten |
| `TRIM_SGB` (:66) | None. This is the experiment described in §5; it is off |

## 3. Compile (`tools/compile.sh arms/s3_trim dkdv dkdv_sp`, RC=0)

| kernel | VGPR | spill v/s | scratch | LDS | wmma | ds |
|---|--:|--:|--:|--:|--:|--:|
| k_dkdv s3 → s3_trim | 721 → **713** | 0/0 | 0 | 70656 | 128 | 256 |
| k_dkdv_sp s3 → s3_trim | 717 → **707** | 0/0 | 0 | 70656 | 128 | 192 |

## 4. ISA evidence: hot loop (`isa_loop.py`, `att/isa_loop_census.txt`; loops in `att/loop_*.s`)

| full loop per iteration | s3 k_dkdv (.LBB0_11) | s3_trim k_dkdv (.LBB0_8) | s3 _sp | s3_trim _sp |
|---|--:|--:|--:|--:|
| instructions | 479 | **458** | 482 | **457** |
| VALU (non-WMMA) | 205 | **173** | 205 | **173** |
| address VALU (`v_add_nc/v_dual_add/v_mad/v_or/v_add_lshl`) | 20 | **2** (`v_mad_u32` ×2, before the fence) | 20 | **2** |
| **VALU inside the DS phase** | **17** | **0** | 17 | **0** |
| v_nop | 23 | **9** | 23 | **9** |
| SALU | 104 | 104 | 106 | 104 |
| WMMA / DS / W↔D switches | 64/80/2 | 64/80/2 | 64/80/2 | 64/80/2 |
| s_wait_xcnt | 0 | 0 | 0 | 0 |

- The masked loop (.LBB0_4) is unchanged in shape: 512-513 instructions, 3 W↔D switches.
- DS phase in the ISA: 8 `ds_store` are followed by 72 back-to-back ds loads. The 32 B-operand tr16 all use one VGPR (v418), with immediates 0..13280. The 32 readbacks all use one VGPR (v516), with immediates 0..13280.
- S/dP phase: the scheduler now interleaves 4 S chains (15 WMMAs issue back to back first), so v_nop drops from 23 to 9. The softmax tail after the last WMMA grows from 37 to 58 VALU.

## 5. Expected gain per k_dkdv iteration (s3 = 1439 cycles ATT)

| item | evidence | saving |
|---|---|--:|
| DS-phase address VALU and switches | ATT: 256.6 + 98.1 = 355 cycles; now 0 VALU. Part of it returns as LDS-delivery dscnt waits: 25 KB before WMMA 8 at 256 B/clk ≈ 64 cycles; worse under 4-wave contention. DSORDER cuts that to about 13 KB | **−190 … −290** |
| LSE/delta address in the loop head | ATT 35 → about 4 (`s_add` + clause) | **−30** |
| S/dP schedule change | dependency model `dep_model.py` (calibrated at 482 vs ATT 506 on s3): 482 → 450 | −30 … +10 |
| **total** | cyclic model `cyc_model.py` (LDS queue plus RAW latencies; it does not model s3's ds→VALU switch cost, so it under-counts): −185 / −156 / −136 at 256 / 128 / 85 B/clk | **≈ −150 … −300 cycles/iter; central estimate −220 (≈ −15%)** |

k_dkdv PMC 6.06e6 cycles would drop to about 5.1-5.3e6. The op-level effect depends on k_dqg
(3.29e6 on the side stream) staying hidden. Main risk: LDS contention in seg0 (TDM writes +
tr16 + readback ≈ 175 B/clk average at 4 waves/CU against 256), which would surface as
`s_wait_dscnt` stalls before the first dK/dV WMMAs.

Variants tried at compile time (dirs kept):
- `_var_iso` (TRIM_SDP="iso", before LATEROT/DSORDER): model S/dP 501, worse than "mix" 450.
- `_var_sgb4` / `_var_sgb5` (`sched_group_barrier` {1 WMMA, 4 or 5 VALU}×32): model S/dP 689, 50 v_nop, and a 103-VALU tail. Rejected, which matches levers-tried #8.

## 6. Bitwise vs s3: expected identical

- **DS addresses:** every address has the same value (proof A1/A2), so the same LDS bytes are loaded.
- **LSE/delta:** same byte address (proof A3), same descriptor; all accesses are in bounds, so the soffset range-check rules do not matter.
- **WMMA chains:** every accumulator chain keeps its operand order, because only issue order across independent chains changed. `reuse_check.py` shows all 56 `matrix_a_reuse` follow a WMMA with the same A VGPRs.
- **FP instruction multiset:** normalised by register, it matches s3 except one `v_pk_add_f32`, which computes `(-lse)+s` instead of `s+(-lse)`. That is a commuted IEEE add with an exact negation, so it gives the same bits.
- **LDS ordering:** P/dS stores still precede their tr16 reads, and in-order LDS gives RAW safety.

## 7. Proof: `python3 bounds_proof.py` gives RESULT: ALL PASS (`att/bounds_proof.log`, about 60 s)

The TDM ring model is dkdv_tdm3's, unchanged, plus the following:
- **Shapes:** prod (k_dkdv), fast, toy, gqa4_small (b2 s128 hq8 hkv2) and unequal_seqlen_2 (b2 sq1024 skv2048 hq4 hkv1) (these run k_dkdv_sp, nsp=16), each with causal 0 and 1.
- **A1/A2:** all 6144 DS addresses (3 stages × 32 lanes × every operand) are equal to s3's, have immediates in [0, 65536), and stay inside their stage.
- **A3:** every `_ldl` call is checked: masked, prologue (clamped), and full-loop prefetch `min(i+1, n-1)`. For each one: element range inside [0, B·Hq·Sq), inside its own (b, qh) row, and soffset < 2^31. The per-lane identity is checked on every lane for the small shapes.
- **ISA gates on both kernels:**
  - T1: 0 VALU in the DS phase.
  - T2/T3: one base VGPR each, with exactly A1's and A2's immediate sets.
  - T4: an SGPR soffset, an invariant voffset that is never written in the loop, and 0 xcnt.
  - T5: every reuse hint is legal.
  - dkdv_tdm3's gates also pass: dscnt 0 before the first WMMA, readback after `s_wait_tensorcnt 0x2`, and tr16 retired within the iteration.

## Files
- `kernels.py` (the arm) and `bounds_proof.py`.
- `att/wall.py`, `att/dkdv_wall_s3.txt`: ATT wall attribution.
- `isa_loop.py`, `seq.py`, `dep_model.py`, `sdp_model.py`, `cyc_model.py`, `reuse_check.py`: ISA census and models.
- `att/models.txt`, `att/isa_loop_census.txt`, `att/loop_{s3,trim}_{dkdv,dkdv_sp}.s`: evidence.
- `_ref_s3/`: s3 recompiled as the reference.
- `_dqg_trim/`, `_dqg_ref/`: k_dqg and k_delta identity check.
- `core`: a 3 GB root-owned core dump from a failed compile of an intermediate `v1f32` buffer-load attempt (LLVM "Do not know how to scalarize"). It cannot be deleted without root. Remove it with `sudo rm`.
