# arm dkdv_lse_late (variant a: dkdv_lsefence)

## What changed (kernels.py vs arms/r29/kernels.py)
- L56-57: new module constant `LSE_FENCE = 1` (0 gives r29, 1 is variant a and the DEFAULT, 2 is variant a plus a tail fence).
- L484-488: `rocdl.sched_barrier(0)` after the b_do/b_q `ds_load_tr16` construction loop, before `new = [None]*...` (k_dkdv `_body`; applies to both the masked and the full q-loop).
- L510-514: only when `LSE_FENCE == 2`, a second `sched_barrier(0)` after the dK/dV WMMA loop. Not the default.
- No index, address or predicate change. Variant b (the exp2-anchored index dependency) was NOT built because variant a passes the ISA gate.

## ISA evidence (compile-only, `.dump/dkdv/k_dkdv_0/21_final_isa.s`, hot loop .LBB0_11)
Resources: vgpr 729 (same as r29, limit 737), 0 spill, private_segment 0, lds 70656, wmma 128, ds 224.

Loadcnt/dscnt waits in the hot loop. `@wN` means the wait comes after N WMMAs of the loop body (1-32 are S/dP plus hh, 33-64 are dK/dV):

| r29 | this arm (LSE_FENCE=1) |
|---|---|
| loadcnt 0x23 @w20 -> `v_mov v138/*v650*/, v181/*v693*/` (ISA:1611-1612) | **gone** |
| loadcnt 0x22 @w28 -> lse hh=1 copy | gone |
| dscnt 0x6 @w32, loadcnt 0x20 @w33, dscnt 0x2, 0x0 | loadcnt **0xc** @w32 (ISA:1866), then dscnt 0x2 @w32, 0x0 @w40, 0x0 @w48 |
| loadcnt 0x7 @w44, 0x6 @w44, 0x4 @w53, 0x0 @w59 | loadcnt 0x8 @w49, 0x0 @w50 |

- All 4 lse/delta carried copies (`v_dual_mov v578<-v728, v582<-v726, v584<-v727, v586<-v629`, ISA:1876-1877) now sit in one group after all 32 S/dP WMMAs, just before the first dK/dV WMMA. They are gated by the same `s_wait_loadcnt 0xc`. That wait also sits in front of 8 b128 prefetch copies (v_mov_b64), right before phase K's `s_wait_dscnt 0x2`.
- Gate: the 0x23 + v_mov v138 pair is gone. No loadcnt wait with a threshold above 0x10 remains between the S/dP WMMAs and phase K. VGPR 729 <= 737, 0 spill. **PASS.**
- Caveat: 0xc is a DEEPER wait than 0x23. It needs the 4 b32 plus the first 20 b128 prefetches. It sits at phase-K entry, where r29 already stalls about 187 cyc/iter on `dscnt 0x6`, so the load latency should overlap that DS drain. In iterations where the load takes about 900 cycles, it could still show up as a loadcnt stall at phase-K entry instead of at 17140. The b128 copy waits also moved earlier: the final 0x0 now comes after WMMA 50 instead of 59. Only the ATT can settle this. Watch the loadcnt 0xc stall at the phase-K entry and the old phase H.
- The masked loop (.LBB0_4) was also rescheduled: its loadcnt thresholds went from 0x1e/0x19/.../0x2 to 0x1e/0x15/.../0x2, and 0x0 is still at the end.

## LSE_FENCE=2 (tried, compile-only; ISA kept in isa_lsefence2.s)
vgpr 721, 0 spill. Hot loop: no loadcnt wait except one `s_wait_loadcnt 0x0` after WMMA 64. All carried copies (4 b32 + 32 b128, 63 movs) are in the tail. The cost is that phase K now waits `dscnt 0x0` before each group of 8 WMMAs (@w32/40/48/56), instead of 0x6/0x2 at entry. Use it as the follow-up if variant a shows its stall moved to the 0xc at phase-K entry.

## Bitwise
Expected to be bitwise identical to r29. `20_llvm_ir.ll` differs from r29's only in the `llvm.amdgcn.sched.barrier` calls (4 -> 6; diff empty after filtering them out). Only the schedule changes. The FP ops and their operands do not.

## Bounds proof
Not required, since no index, address or predicate expression changed. No bounds_proof.py was written.
