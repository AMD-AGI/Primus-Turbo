# abl_a1 -- ablation: k_dkdv without the P/dS LDS round trip (OUTPUT WRONG)

Timing only. Upper bound for dkdv_flip (P/dS kept in registers, no LDS transpose), and partly for trorder.

## Change (kernels.py vs arms/r29/kernels.py; k_dkdv `_body` only, k_dqg untouched)
- L39-40: import `scf` dialect and `ir` (used only by the optional guard).
- L58-63: `ABL_A1 = True` (False = exact r29 path), `ABL_A1_GUARD = False`.
- L457-477 (P/dS store site): under ABL_A1 the two `llvm_dialect.store` of P/dS go away. A zero-instruction
  `llvm.inline_asm "; abl_a1 keep P/dS"` with side effects USES the packed P/dS v8bf16 vectors, so the
  softmax/dS VALU (mask, exp, mul, cvt_pk_bf16) is still computed. With `ABL_A1_GUARD=True` the stores
  are also emitted under a runtime-false, wave-uniform `scf.if (qt < 0)`.
- L509-515: `a_p = b_q[0]`, `a_ds = b_do[0]` (same v16bf16 shape, same register fragments that feed
  the B operands) instead of the 2x2 `tr()` loads from lds_p/lds_ds -> 8 ds_load_tr16_b128 fewer per iteration.
- No index, address or predicate expression changed. Only accesses were removed.

## Why the default is not the literal spec (the scf.if guard)
First compile with only the `qt < 0` scf.if guard: LLVM sank the entire softmax chain (v_cmp/v_cndmask
mask, v_exp x8, v_pk_mul, v_cvt_pk_bf16) INTO the never-taken block, so the P/dS VALU would have been
skipped at runtime too. It also split every iteration into 5 basic blocks and
VGPR went 729 -> 942 (full-loop body 659 -> 879 instructions). Adding the volatile asm use pulled the VALU
back out (all 8 v_exp before the asm marker, the block holds only the 2 ds_store), but the CFG split and
VGPR 937 remained. So the default drops the stores and keeps only the asm use. The CFG is then the same as r29
(same labels/branches) and VGPR is 735. `ABL_A1_GUARD=True` gives the literal-spec variant.

## ISA evidence (compile.sh, .dump/dkdv/k_dkdv_0/21_final_isa.s)
Resources: dkdv vgpr 735 (r29 729), spill 0, sgpr_spill 0, private_segment 0, LDS 70656 (unchanged).
dqg ISA is identical to r29's (vgpr 991).

| loop body | instr | wmma | tr16 | ds_store | exp | cvt_pk_bf16 | cndmask |
|---|---|---|---|---|---|---|---|
| r29 qloop_full (.LBB0_11)    | 659 | 64 | 40 | 40 | 32 | 32 | 0 |
| abl_a1 qloop_full (.LBB0_11) | 670 | 64 | 32 | 32 | 32 | 32 | 0 |
| r29 qloop_mask (.LBB0_4)     | 650 | 64 | 40 | 40 | 32 | 32 | 32 |
| abl_a1 qloop_mask (.LBB0_4)  | 680 | 64 | 32 | 32 | 32 | 32 | 32 |

- 0 P/dS ds_store in either loop: the remaining 32 are the Q/dO staging stores, which r29 also has.
- 32 tr16 per loop, down from 40. WMMA stays at 64. All softmax VALU is still present and executed.
- Opcode diff in the full loop (r29 -> abl): tr16 40->32, ds_store 40->32, v_nop 66->53, s_wait_loadcnt 9->8,
  but s_set_vgpr_msb 109->150 (+41). The body gets 11 instructions LONGER, even though 16 memory ops and 13 nops are gone.
  The +41 MSB switches come from the new register assignment (A operand now aliases a B fragment
  in a different VGPR bank). They are cheap SALU, but they count against issue, so the measured delta is a
  slightly pessimistic upper bound.
- Waits in the full loop: first dK/dV WMMA at body slot 494 (r29: 485), preceded by `s_wait_dscnt 0x2`/`0x0`
  on the Q/dO tr16 burst (r29 had the same pair after the P/dS tr16). The only new LDS dependence left is
  Q/dO store -> tr16. The side-effecting asm is a memory-ordering chain point, so loads and stores cannot move across it.
  That constrains scheduling a little.

## Correctness
Not bitwise identical to r29: dK/dV are wrong by design (A operand is Q/dO instead of P/dS). dQ (k_dqg) is unchanged.

## Bounds proof
Not required and not written: no index, address or predicate expression was changed. The arm performs a strict
subset of r29's LDS/global accesses (the 8 P/dS stores and 8 P/dS tr16 loads per iteration are removed), and
every remaining access uses the r29 expression unchanged. With ABL_A1_GUARD=True the stores keep r29's
addresses and never execute.
