# abl_a4 -- ablation: k_dqg U2 kvloop_full without in-loop K/V global loads

**Output is WRONG by design.** Only timing is meaningful: the upper bound of the k_dqg memory path.

## Change (kernels.py; only k_dqg; k_dkdv untouched)
- L1175-1177: new switch `DQ_ABL_NOKVLOAD = True` (False = r29 exactly).
- L1282-1293 (`_dqg_impl._body`, carry=True branch, used only by kvloop_full):
  instead of `nxt = _ldkv(kv0_n)`, `pre` is passed through an empty tied inline asm
  (`""`, `"=v,0"`, no side effects) and `nxt = pre` is carried on unchanged.
  - Why the asm: the first attempt with a plain `nxt = list(pre)` let LLVM CSE the second U2
    body's S/dP WMMAs and exp against the first body's (loop had 128 WMMA / 64 exp instead of
    192 / 128) -- that would have ablated compute too. The opaque zero-instruction copy blocks it.
  - Putting the asm on `nxt` at body end instead produced 56 VGPR spills (input still live);
    applying it to `pre` at body entry costs nothing (vgpr 806, no spill).
- The K ds_store_b128 + ds_load_tr16 path, kvloop_mask (masked tail, still loads K/V), the
  prologue `_ldkv(0)` and all index/address/predicate expressions are unchanged.

## ISA evidence (.dump/dqg/k_dqg_0/21_final_isa.s; loop .LBB0_2, L584-2088 vs r29 .LBB0_4 L615-1925)
| per loop trip (2 bodies) | r29 | abl_a4 |
|---|---|---|
| buffer_load_b128 | 64 | **0** |
| s_wait_xcnt | 3 | **0** |
| s_wait_loadcnt | 14 | 1 (at loop top: `0x0` for the prologue prefetch; free after trip 1) |
| s_wait_dscnt | 13 | 16 |
| v_wmma_f32_16x16x32_bf16 | 192 | 192 |
| v_exp_f32 | 128 | 128 |
| ds_store_b128 / ds_load_tr16_b128 | 32 / 32 | 32 / 32 |
| v_nop | 31 | 85 (WMMA hazard fill no longer covered by address VALU) |
| address VALU (v_or/v_mul_lo/v_add_lshl) | 34 | 0 |
Kernel: vgpr 806 (r29 991), spill 0, scratch 0, LDS 8704, total wmma 288 (= r29).
Mask loop (.LBB0_6) unchanged: 32 buffer_load_b128, 96 wmma.

## Bitwise vs r29
Not identical -- dQ is wrong (every full-loop block reuses K/V block 0). Timing only.

## Bounds
No index/address/predicate expression changed; the only change removes global loads (the
remaining accesses are a subset of r29's). No bounds_proof.py needed.

## Note for interpretation
VGPR drops 991 -> 806 (one K/V set instead of two live); occupancy is fixed by DQ_NW=1/LDS
anyway, but the register pressure change can also shift scheduling. The 54 extra v_nop are a
real cost the ablation adds (the address math used to fill WMMA hazards).
