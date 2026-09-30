# dqg_tailpf (k_dqg, U2 kept)

## Change (kernels.py; k_dkdv untouched -- its ISA diffs empty vs r29, 729 VGPR)
- L1175-1187: `DQ_TAILPF = "b"` (None = r29 path, "a" = one 32-load clump) and
  `DQ_TAILPF_MASK = 0x406`.
- L1281-1293: `_ldkv` split into `_ldkv_kt(kv0, kt)` (same gfrag2 calls, same order).
- L1295-1302: `_body(..., pf2=None, nxt_in=None)`: `pf2` = kv0 of an extra prefetch issued
  inside this body; `nxt_in` = prefetch already issued, returned instead of loading.
- L1332-1337 (variant b): right after kt's S/dP WMMA loop (the kt half of `pre` is dead),
  `sched_barrier(MASK); nxt2 += _ldkv_kt(pf2, kt); sched_barrier(MASK)`.
- L1365-1368 (variant a): the same fence, with one 32-load `_ldkv(pf2)` after the kt loop.
- L1386: the carry return appends nxt2.
- L1389-1412: new `kvloop_full` branch for `DQ_PF and DQ_U2 and DQ_TAILPF`. It uses the same
  `jj` and the same clamp. Body 1 gets `pf2=jj*KV_STEP`, and body 2 gets `nxt_in` = body 1's
  pf2 set. The r29 U2 branch stays as the `elif` below. (A `const_expr` if inside the jit
  loop with a `yield` in both arms did not compile, because both yields were emitted, so the
  switch sits outside the jit function.)

## Fence mask (deviation from spec; compile-tested)
| mask | result |
|---|---|
| 0x78F (spec, only VMEM blocked) | **no effect**: the ISA matches r29 in load placement (tail loads at WMMA 171-192, lead 2 WMMA). The IR has the 4 barriers in the right places, but the fence itself floats past the WMMAs, so the loads still sink |
| 0x786 (VMEM+WMMA blocked) | pf2 loads land at WMMA 32/64 and VGPR is 1022. But body 2's K ds_stores rise to WMMA ~8, so head-clump waits `0x11..0x3` sit at WMMA 7-8 (a new stall) |
| **0x406** (only VALU/SALU/TRANS cross) | chosen, see below |

## ISA evidence (loop .LBB0_4, `isa_gate.py .dump/dqg/k_dqg_0/21_final_isa.s`)
| | r29 | tailpf b (0x406) | tailpf a (0x406) |
|---|---|---|---|
| VGPR / spill / scratch | 991/0/0 | **1011/0/0** | 879/0/0 |
| tail (ii+2) loads position | WMMA 174-192 | 16 @ WMMA 32, 16 @ WMMA 64 | 32 @ WMMA 64 |
| lead of last tail load to loop-head wait | **2 WMMA** (67 instrs) | **128 WMMA** (800 instrs) | 128 WMMA |
| loop-head wait | 0x20 @ instr 61 | 0x19 @ instr 51 (after 25 head loads = all 32 tail done) | 0x0 @ instr 1 |
| head-clump (ii+1) waits | WMMA 55-148 | WMMA 64-144 (0x3d..0x20) | 0x9/0x3 @ WMMA 92 (head clump sank to WMMA 64) |
| s_wait_xcnt in loop | 3 | **0** | 10 |
| v_mov_b64 / loop instrs | 1 / 1310 | 1 / 1326 | 1 / 1643 |
Variant b passes the gate (>=100 WMMA lead, <=1024 VGPR, 0 spill, 0 scratch). Variant a is
worse: its head clump sinks to the fence, and the loop head waits 0x0.

## Bitwise
Expected to be bitwise-identical to r29. The loaded values (same `_ldkv(jj*KV_STEP)` and the
same clamp), the WMMA operands and the accumulation order are all unchanged. Only when the
loads issue has changed.

## Bounds
No index, address or predicate expression changed. `bounds_proof.py` still enumerates every
K/V block that the U2 loop loads, for prod, fast and toy shapes with causal 0/1:
ALL IN BOUNDS (max kv row 8191/1023/127).
