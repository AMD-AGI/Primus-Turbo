# dkdv_trorder (variant a) -- fenced 4-region emission of k_dkdv's final DS phase

Base: arms/r29. Switch: `TRORDER = True` (kernels.py:56; False = r29 code path, kept verbatim
in the `else:` branch). `QDO_BURST = False` (kernels.py:57; variant b lives in ../dkdv_trorder_b).

## Change (kernels.py line numbers in this arm)
- L56-57: new module constants TRORDER, QDO_BURST.
- L383-415: the hh staging loop is refactored into `_stage(hh, qp, dp)` / `_qd(hh)` helpers
  (same store expressions, same order when QDO_BURST=False); an optional fenced burst for
  variant b (inactive here).
- L495-541: new `if const_expr(TRORDER)` final DS phase, each region closed by
  `rocdl.sched_barrier(0)`:
  - R1 a_p(kh0) + b_do[0..7] tr loads
  - R2 a_ds(kh0) + b_q[0..7] tr loads | fence | 8 dV kh0 WMMA
  - R3 a_p(kh1) + a_ds(kh1) tr loads, then 8 dK kh0 WMMA
  - R4 8 dV kh1 + 8 dK kh1 WMMA
- L543-566: the old r29 code, re-indented under `else:`.
Address expressions are the r29 formulas (moved into the `_a`/`_b` helpers); each `new[k]` is still
one WMMA on `acc[k]` with the same A/B operands and reuseA flags.

## ISA evidence (compile.sh; `.dump/dkdv/k_dkdv_0/21_final_isa.s`, loop .LBB0_11 = L1381-2060)
- Resources: 729 VGPR (r29 729), 0 VGPR/SGPR spill, private_segment 0, LDS 70656. The loop has
  64 WMMA, 40 ds_store and 40 ds_load_tr16. There are 679 instructions against r29's 659.
- The tr16 issue order in the ISA matches the source: a_p0, b_do x8, a_ds0, b_q x8 (36 loads at L1841), then 4 kh1 A loads at L1897.
- The first dK/dV WMMA (L1880) is preceded by **`s_wait_dscnt 0x20`**. r29 has 0x6 (ISA:1868 / CSV:1857). The
  following waits step down 0x1e..0x2, one per WMMA. The first `s_wait_dscnt 0x0` is at L2005,
  before the **25th** tail WMMA. GATE PASS (>= 0x10; no 0x0 before the 17th; VGPR <= 760; 0 spill).
- The final DS group now holds **6 ds_store** (r29: 23). 17 stores were scheduled up to L1687
  among the S/dP WMMAs (10 WMMAs follow them before the final DS group).
- Side effect: r29's early LSE back-edge copy (`s_wait_loadcnt 0x23` + `v_mov v138<-v181`,
  CSV:1600) is gone. The loop's loadcnt waits are now 0xc, 0x4, 0x12 and 0x0; the 4 lse/delta copies are
  `v_dual_mov` after the tail loadcnt 0x0. This overlaps the dkdv_lse_late arm.
- Caveat: the tail `s_wait_loadcnt 0x0` for the 32 b128 prefetch moved earlier. It now comes after 17 tail WMMAs
  (r29: after 27, stall 0.02 cyc). It should still be covered (~1300 cyc lead), but ATT should confirm.
- Phase M, the v_mov_b64 rotation with WAR v_nop, is still present: 60 v_mov_b64 / 51 v_nop in the loop, against 64 / 66 in r29.

## Bitwise vs r29: expected YES
Only the issue order and scheduling fences change. The per-slot accumulator order is the same, and so are the values and addresses.

## Bounds proof
`bounds_proof.py` runs on the host and prints OK. For prod, fast and toy, it enumerates every lane's LDS byte range for the staging stores,
the P/dS stores and the tr16 loads. The new multiset equals r29's, and every access falls inside its own ring and the 70656 B
allocation. No global index, predicate or loop bound changed.

`isa_seq.py` is a CPU helper that compresses the loop's DS/WMMA/wait sequence.
