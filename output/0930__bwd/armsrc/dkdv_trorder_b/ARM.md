# dkdv_trorder_b (variant b = trorder + qdoburst)

It is dkdv_trorder (variant a) plus `QDO_BURST = True` (kernels.py:57). All 32 Q/dO staging
ds_stores are issued as one burst, fenced with sched_barrier(0) on both sides, at the top of `_body`, right after
`nxt = _ldqd(...)` (kernels.py:395-415). The hh loop then only builds qfr/dfr and runs S/dP. The final DS
phase is the same as in variant a (kernels.py:495-541).

## ISA evidence (.dump/dkdv/k_dkdv_0/21_final_isa.s, loop .LBB0_11 = L1397-2039)
- Resources: 729 VGPR, 0 spill, private_segment 0, LDS 70656. The loop has 64 WMMA, 40 ds_store and 40 tr16. There are 642 instructions against r29's 659.
- Loop order: 4 b32 and 32 b128 prefetch loads (L1437/1458). Then `s_wait_loadcnt 0x2c`/`0x24`, which is free in steady state
  because the previous iteration ended at loadcnt 0. Then **32 ds_store** in one run (L1500-1533), then 17 WMMA, 2 P/dS stores,
  15 WMMA, and the final DS group: **6 ds_store** + 36 tr16.
- The first dK/dV WMMA (L1849) is preceded by **`s_wait_dscnt 0x20`**. The first dscnt 0x0 is at L1973, before the 25th
  tail WMMA. GATE PASS.
- The tail `s_wait_loadcnt 0x0` is at L1956, after 24 tail WMMAs. That is later than in variant a and close to r29.
- The early LSE copy (loadcnt 0x23) is gone here too. The loadcnt waits are 0x2c, 0x24, 0x12, 0x4 and 0x0.
- Risk: the 32-store burst sits back-to-back. The report measured LDS back-pressure after about 12 stores at 3-4.6 cyc each. It is
  followed by 17 LDS-independent WMMAs, but ATT has to show whether it stalls the issue.

## Bitwise vs r29: expected YES (same stores, addresses and values; only the issue position changes).
## Bounds proof: `bounds_proof.py` (same file as variant a, which covers the store burst), host run OK.
