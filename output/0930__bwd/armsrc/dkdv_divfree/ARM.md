# dkdv_divfree

Removes the runtime signed division `ii // G` (G is a kernel argument) from k_dkdv's q loops.
(qi, gh) are carried as two uniform i32 scf.for iter_args (they land in SGPRs) and advanced
by a wrap: gh+1 == G -> (qi+1, 0). The prefetch index jj = min(ii+1, n-1) reuses the same
wrap (jj == ii+1 while ii+1 < n, else jj == ii). Switch: `DIVFREE` (kernels.py:56, default True;
False = the r29 code path, kept verbatim in the else branches).

## Diff vs r29/kernels.py
- :56-57   `DIVFREE = True`
- :517-530 comment + `_wrap(qc, gc)` helper (s_add / s_cmp / 2x s_cselect)
- :532-548 `qloop_mask`: counters at st[NST], st[NST+1]; yields body + [qn, gn]
- :550-579 `qloop_full`: counters at st[-2:], prefetch = st[NST:-2]; jj clamp = select between
  wrap(qc,gc) and (qc,gc) on `ii+1 < n`
- :589-620 `qloop_full2` (KV_U2 path, default off): two wraps per trip, same clamp on 2*n2
- `qloop_tail` unchanged (runs <= 1 trip, only under KV_U2)
- :650-652 `_z2` = two i32 zeros (start state (0,0)); :669,678,674/683 callers pass `init + _z2`
  to qloop_mask and `list(out)[:NST] + prefetch + _z2` to the full loop (both PARTIAL and non-PARTIAL).
- Each loop now has a single `final = yield res` (two yields under `if const_expr` break the
  flydsl AST rewriter: "block already has a terminator").

## ISA evidence (k_dkdv, prod shape, .dump/dkdv/k_dkdv_0/21_final_isa.s)
| | r29 | divfree |
|---|---|---|
| main loop .LBB0_11 lines | 1384-2043 (659) | 1346-1990 (644) |
| loop SALU (excl. set_vgpr_msb/wait/clause) | 36 | 21 |
| s_abs / s_mul_hi in whole kernel | 8 | 0 |
| mask loop .LBB0_4 length / SALU | 650 / 67 | 633 / 50 |
| vgpr / sgpr / spill / scratch / LDS | 729 / 78 / 0 / 0 / 70656 | 729 / 77 / 0 / 0 / 70656 |
| wmma / ds | 128 / 224 | 128 / 224 |
Main loop index math is now: s_add, s_cmp_ge, s_cselect x2, s_add_co_ci, s_cmp_lt, s_cselect x2 (+2
s_mov back-edge copies). No s_mul_hi/s_abs/v_s_rcp remains anywhere in the kernel (the preheader
reciprocal setup is gone too). s_wait_loadcnt pattern (0xc, 0x4, 0x23, ...) unchanged; VALU count
in loop unchanged (380).
KV_U2=True variant was also compile-checked (in _u2check/, root-owned dump left behind): RC=0,
vgpr 644, no spill; its 4 remaining s_abs/s_mul_hi come from qloop_tail (outside the pair loop).
k_dkdv_sp (PARTIAL, used at the fast shape) is NOT compiled by tools/compile.sh; it uses the same
loop functions and the same caller change.

## Bitwise vs r29
Expected bitwise identical: every body receives the same (qt, gh, qt_n, gh_n) integers in the same
order (proved on CPU); only the SALU computing them changed.

## Bounds proof
bounds_proof.py (host python3, no torch): PASS. Replays r29 vs divfree index sequences for every
workgroup/split, causal on/off, at prod (b4 s8192 hq32 hkv8, nsp1), fast (b1 s1024 hq8 hkv2, nsp16)
and toy (b1 s128 hq2 hkv1, nsp16), for qloop_mask, qloop_full and qloop_full2+tail; asserts tuple
identity and that every Q/dO/LSE/delta address of every lane (incl. the clamped prefetch at n-1 and
the prologue) is inside the true extent. Also wrap == divmod for G in 1..16.
