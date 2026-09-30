# abl_a2 -- ablation: Q/dO global path removed from k_dkdv qloop_full (OUTPUT WRONG)

## Change (kernels.py vs r29/kernels.py)
- L56: `ABL_A2 = True` module switch (False = exact r29 behaviour).
- L275: `_ldqd(qt, gh, qdo=True)`; L303-304: `if not qdo: return out` right after the
  4 LSE/delta buffer_load_b32 and their sched_barrier(0) -- the 32 Q/dO b128 are skipped.
- L371-373 (`_body`, carry path): `nxt = _ldqd(qt_n, gh_n, qdo=False) + list(pre[4:])`
  -- LSE/delta prefetch unchanged, Q/dO carried forward unchanged (loop-invariant phi,
  so every iteration reuses the prologue's Q/dO). qloop_mask (carry=False), the prologue
  _ldqd, LDS staging stores, ds_load_tr16, k_dqg: untouched. No index/address/predicate
  expression changed (only loads removed).

## Compile (tools/compile.sh abl_a2 dkdv)
RC=0; vgpr 712 (r29 729), vgpr/sgpr spill 0, private_segment 0, LDS 70656, wmma 128, ds 224.

## ISA evidence (qloop_full loop = .LBB0_11, lines 1401-1838 of 21_final_isa.s)
| metric            | r29 | abl_a2 |
|-------------------|-----|--------|
| instructions      | 659 | 437    |
| buffer_load_b128  | 32  | 0      |
| buffer_load_b32   | 4   | 4      |
| v_mov_b64         | 64  | 0      |
| v_nop             | 66  | 8      |
| VALU / SALU       | 380/163 | 237/116 |
| wmma / ds / tr16 / ds_store | 64/80/40/40 | 64/80/40/40 |
Whole kernel buffer_load: 140 -> 108 (exactly -32). Mask loop .LBB0_4 unchanged (32 b128 + 4 b32).
Waits: the loop's s_wait_loadcnt 0x22..0xa at the head only bind on iteration 1 (prologue
b128 still in flight); in steady state <=4 loads are outstanding so they are free. The b32
rotation waits are 0x2 at slot 222 and 0x0 at slot 377 (b32 issued at slot 43, ~180/334
slots before) -- comparable to r29 (b32 rotation waits 0x23/0x22/0x20 at ~188-445 slots).
The 0x0 is a full drain but only of this iteration's 4 b32, ~330 slots after issue.

## Result expectation
NOT bitwise identical to r29 (dK/dV wrong: every q iteration uses the first pair's Q/dO).
Timing = light-speed of k_dkdv with zero Q/dO global traffic in the steady loop.

## Bounds proof
Not required/not written: no index/address/predicate expression changed; loads were only
removed (the remaining LSE/delta and prologue loads use r29's unchanged, clamped addresses).
