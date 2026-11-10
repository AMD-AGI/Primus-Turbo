# Round 17 — ROLLED BACK — 2026-11-10 10:20 — deferred GEMM3 inside the Q/dO rendezvous (g3_rdv)

## Hypothesis
ATT of the r15 code (profiles/att_r15_windows.txt, B1 S8K CU 1): cycles 40-570 of the iteration,
from the head-step's WAR barrier to the first transpose-read after the publish barrier, hold no
MFMA (WAR barrier 101, Q/dO commit + next prefetch 344, lgkmcnt(0) 50, publish barrier 31). The
deferred GEMM3 reads only the band-resident K and the previous trip's dS, and dS is published by
the WAR barrier (gpu.barrier lowers to fence release + s_barrier + fence acquire; 21_llvm_ir.ll),
so GEMM3 can run between the two barriers and cover the commit and the publish drain. The
builder's G3_SHADOW is the same move on the LDS-DMA route, disabled there because a pending
buffer_load ... lds forces vmcnt(0) at GEMM3's first read; the Q_PREF route has no LDS DMA.

## Single change
`flash_attn_bwd_mla.py`: builder knob `g3_rdv` (0 = off; 1 = GEMM3 right after the WAR barrier,
ahead of the Q/dO commit; 2 = after the commit, the next prefetch issue and the (-delta, lse)
commit, ahead of the publish drain). Diff: artifacts/g3_rdv.diff; tested source:
artifacts/tested_source/. Off compiles to the r16 ISA (md5 c587bf720b).

## Screening (artifacts/isa_r17.txt, hot_runs_*.txt)
Both modes keep 94 spills / 380 B scratch and the hot loop's 3 scratch loads / 2 vmcnt(0).
- Mode 1: the run before the first MFMA shrinks 77 -> 27 instructions (WAR barrier + GEMM3 reads),
  the Q/dO ds_writes land between GEMM3's last MFMAs, and the 8 prefetch loads, the VALU address
  work and the (-delta, lse) commit stay as a 68-instruction block in front of the publish barrier.
- Mode 2: the leading run stays 76 (commit, prefetch, GEMM3 reads; the ds_writes come first in
  program order and the reads cannot be hoisted over them); 26 of GEMM3's MFMAs land between the
  barriers.

## Results (dev/sweep_proc.py, 4 rotated rounds, one process per config; artifacts/sweep_r17.raw)

| config | B2 S4K ms | B1 S8K ms | B1 S16K ms | gm |
|---|---|---|---|---|
| r16 | 4.969 | 8.388 | 30.884 | 0 |
| g3_rdv=1 | 5.213 (-4.67 %) | 9.158 (-8.40 %) | 34.971 (-11.69 %) | -8.30 % |
| g3_rdv=2 | 5.002 (-0.65 %) | 8.585 (-2.29 %) | 31.623 (-2.34 %) | -1.76 % |

SNR identical to r16 (47.00 / 49.74 / 51.28 dB). Full set not run.

## Why it loses (artifacts/publish_window.txt)
GEMM1a's first two K-pack reloads (scratch_load_dwordx4 a[88:91], a[92:95]) are issued right
after the publish barrier in every variant: they are memory ops and cannot move above its acquire
fence. In r16 the whole GEMM3 (about 30 MFMAs) then runs between their issue and their use, and
the first vmcnt wait after them in the region is the hot loop's existing one in GEMM1a. With
GEMM3 moved in front of the barrier:
- mode 1 waits `vmcnt(1)` / `vmcnt(0)` on them about 25 instructions later, and the 8 Q/dO prefetch
  loads were issued just before the barrier; gfx950's vmcnt is in-order, so the wait also covers
  the prefetch's HBM latency;
- mode 2 waits on them about 40 instructions later, with about 4 GEMM3 MFMAs left as cover.
So GEMM3 after the publish barrier is what hides the reload (and, through the in-order counter,
part of the prefetch). Hypothesis level: not ATT-measured for these variants.

## Decision
ROLLED BACK: -1.76 % / -8.30 % gm. Source restored to the r16 snapshot (byte-identical;
kernel_snapshot/ = r16). Rollback counter 1.

## Finding: dS slot 0 has no WAR fence on the MLA path (r16 code, unchanged by this round)
With one q head per kv head (GQA_GROUP_SIZE 1), the deferred GEMM3 at the top of trip t reads dS
slot `((0 - 1) % 1) % 2 = 0`, and trip t writes its own dS to slot `0 % 2 = 0`; the second slot is
allocated but unused. Between GEMM3(t-1)'s reads and dS(t)'s writes there is no barrier (the hot
loop has two, both ahead of GEMM3). The r15 ATT iteration (4508 cycles) has the last GEMM3 dS
read at about cycle 1030 and the first dS ds_write_b128 at about cycle 3235, against a measured
barrier skew of about 100 cycles, so a wave would have to fall about 2200 cycles behind another
after the same barrier for a read to see the next trip's dS. Not observed (dQ SNR stable across
every run of the campaign); formally a race. Fix options: alternate the dS slot by trip parity
(carried flag; runtime slot offset on the dS write and GEMM3 read bases), or a barrier between
GEMM3 and the first dS write (one more rendezvous per trip). The g3_rdv placement closes it too
but costs 1.8-8.3 %.

## Real-training transfer check
- Bucket: K1 (codegen: instruction placement inside the q loop)
- Cache key (if any): compile-time knob
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: -1.76 % (best mode), rolled back
- Estimated real-training gain: none (rolled back)
- Decision: REJECT (perf)

## Next direction
Decide the dS slot hazard first (see Finding). Then: the reload/prefetch coupling suggests the
measured vmcnt(0) waits in GEMM1a (133 + 161 cycles in the r15 ATT) may be the Q/dO prefetch's
HBM latency seen through the in-order counter rather than scratch latency; an ATT of r16 with the
per-load issue/return cycles would settle it.
