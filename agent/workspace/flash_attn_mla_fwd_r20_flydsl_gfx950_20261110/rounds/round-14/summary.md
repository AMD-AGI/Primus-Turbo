# Round 14 — ACCEPTED (trade-off) — 2026-11-11 10:05 (UTC+8)

## Hypothesis
Since round 11 the persistent prologue's K1 is prefetched by the previous job, but the pre-QK0
`_waitcnt_vm_n(VM_DRAIN_V)` (there for K1) still waits for the first half of V0, issued just
before it. V0 is first read after the next cluster's drain (`_qk_cluster_sync`'s `vmcnt(0)` in the
loop's C1, or `VM_DRAIN_KV` in drain C0 when there is no loop iteration), so dropping that wait leaves
the whole V0 tile in flight under QK0: one partial memory round trip off every job's prologue,
about +0.2-0.5 % at 4K and ~0 at long S.

## Single change
`CODEGEN_REV` 1051, delta `artifacts/r14_delta_vs_r13.patch`: `_main_body`'s prologue skips
`_waitcnt_vm_n(ctx.VM_DRAIN_V)` when `next_job is not None` (persistent); `lgkmcnt(0)` stays.
ISA at H=128 (`artifacts/r14_persistent_h128.s`): the four V0 DMAs are followed directly by the
pre-QK0 barrier; VGPRs 497, 0 spills; the KV loop blocks are identical to round 13's.

## Results (full set, 2 rotations, base = round-13 snapshot; two separate processes)

| shape | cand | base (R13) | cand / base (run 1) | cand / base (run 2) |
|---|---|---|---|---|
| dsv3 (B2 S4K H128) | 1110.5 | 1107.7 | 1.0025 | 1.0027 |
| dsv2lite (B8 S4K H16) | 1075.0 | 1073.4 | 1.0015 | 1.0014 |
| kimik2 (B4 S4K H64) | 1102.5 | 1099.6 | 1.0026 | 1.0028 |
| kimik3 (B2 S7K H96) | 1222.3 | 1220.8 | 1.0012 | 1.0013 |
| B1 S4K | 1098.7 | 1096.3 | 1.0022 | 1.0017 |
| B1 S8K | 1246.5 | 1246.0 | 1.0004 | 1.0007 |
| B1 S16K | 1291.9 | 1293.9 | 0.9985 | 0.9977 |
| B1 S32K | 1327.4 | 1329.2 | 0.9986 | 1.0001 |
| B1 S64K | 1331.1 | 1336.4 | 0.9960 | 1.0118 |
| B2 S4K | 1107.7 | 1104.8 | 1.0026 | 1.0025 |
| B2 S16K | 1303.5 | 1303.3 | 1.0002 | 0.9994 |

Geomean 1197.50 vs 1196.81 (+0.06 %, `artifacts/r14.csv`); repeat 1200.06 vs 1197.65 (+0.20 %,
`artifacts/r14_rep2.csv`). 11/11 PASS, bitwise identical to round 13 in both runs. Edge smoke
(`artifacts/r14_smoke.log`) 8/8 bitwise identical.

Kept one-job-per-CTA path (`persistent=False`, `wip/np14` vs `wip/np13`): the change is gated on
`next_job`, and the H=8 ISA is identical instruction for instruction (7727 instructions), so no
benchmark needed.

## Decision
ACCEPTED (trade-off) per Rule 3: geomean up in both runs (+0.06 %, +0.20 %), no shape regressed by
5 % or more; B1 S16K -0.15 / -0.23 % in both runs. 1.191x aiter (1005.24).

## Attribution
- The prologue part reproduces: every shape with short jobs (4K, kimik3) +0.12-0.28 % in both runs,
  within 0.05 % between runs. That is the size estimated for half a V tile's latency per job.
- Long S has few, long jobs, so the prologue share is ~0. B1 S64K moved -0.4 % then +1.2 % with
  identical binaries, so that shape is bistable between processes (also seen in rounds 11-13) and
  says nothing about this change. B1 S16K is -0.2 % in both runs; with an unchanged loop, the likely
  cause is the V0 DMA now landing in LDS while QK0 reads K0 (more LDS traffic under QK0). Not
  measured separately.

## Real-training transfer check
- Bucket: K1 (fixed-cost / per-job overlap).
- Cache key: no cache.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +0.06 % / +0.20 % combined-step (forward-only: combined step =
  forward).
- Estimated real-training gain: +0.1-0.2 % at these shapes. One wait removed per job, bitwise-neutral,
  independent of tensor identity and data.
- Decision: ACCEPT-as-real.

## Next direction
1. The job-top `s_waitcnt vmcnt(0)` waits for the previous job's 16 O stores. Issue order at the end
   of a job is the K prefetch (12), the LSE store (conditional), then the O stores, so `vmcnt(16)`
   retires the prefetch and leaves the stores in flight while the Q loads issue. vmcnt retires in
   order, so the first Q use still waits for the stores; the gain is the Q load latency overlapping
   the store drain instead of following it (about one memory round trip per job).
2. Then the next job's Q load in the drain (AGPR destinations), to take the Q round trip itself off
   the prologue.
3. With 15 VGPRs free, revisit what round 8 lost to register pressure (carrying the C1 K head over
   the back edge).
