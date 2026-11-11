# Round 13 — ACCEPTED — 2026-11-11 09:55 (UTC+8)

## Hypothesis
Round 12's loop moved with register allocation alone. The loop-invariant values parked in AGPRs
a236-a255 and read back inside the KV loop (16-20 `v_accvgpr_read` per iteration) are not DMA
voffsets, as round 12 assumed: the K / V voffsets sit in VGPRs (v228-v242). They are the addresses of
the V buffer-1 `ds_read_b64_tr_b16` reads. Only 96 of 192 transposed V reads carry an `offset:`:
V buffer 1 starts at byte 65536, the limit of the 16-bit DS offset field, so its constant base never
folds and every buffer-1 read gets its own address register, which LICM hoists out of the KV loop.
Giving buffer 1 an opaque base (one register per read base) leaves only each read's small offset as the
immediate; that should drop ~16 loop-invariant registers and the AGPR read-backs from the loop, with
a gain that does not depend on the allocator's luck.

## Single change
`CODEGEN_REV` 1050, delta `artifacts/r13_delta_vs_r12.patch` (helper only): `_bind_lane_layout`
precomputes the byte offsets of the two PV16 read bases per V buffer (`v16_read_off`) and passes
buffer 1's through `_pin_regs`; `_lds_load_v16` builds its lane pointers from them.
ISA at H=128 (`artifacts/r13_persistent_h128.s`): 186 of 192 transposed reads with `offset:`;
VGPRs 512 -> 497, spill slots 3 -> 0, scratch 16 -> 0 B; AGPR reads in the KV loop 16 -> 0;
instructions per loop iteration 878 -> 865.

## Results (full set, 2 rotations, `artifacts/r13.csv`, base = round-12 snapshot)

| shape | cand | base (R12) | cand / base |
|---|---|---|---|
| dsv3 (B2 S4K H128) | 1107.0 | 1091.3 | 1.0144 |
| dsv2lite (B8 S4K H16) | 1074.2 | 1060.2 | 1.0132 |
| kimik2 (B4 S4K H64) | 1103.8 | 1068.7 | 1.0328 |
| kimik3 (B2 S7K H96) | 1219.6 | 1184.6 | 1.0295 |
| B1 S4K | 1095.2 | 1079.9 | 1.0142 |
| B1 S8K | 1245.2 | 1232.0 | 1.0107 |
| B1 S16K | 1293.6 | 1271.8 | 1.0171 |
| B1 S32K | 1328.6 | 1305.1 | 1.0180 |
| B1 S64K | 1336.1 | 1309.5 | 1.0203 |
| B2 S4K | 1105.1 | 1089.8 | 1.0140 |
| B2 S16K | 1302.9 | 1285.3 | 1.0137 |

Geomean 1196.85 vs 1175.70 (+1.80 %), 11/11 PASS, every shape up (+1.1 % to +3.3 %), bitwise
identical to round 12. Edge smoke (`artifacts/r13_smoke.log`) 8/8 bitwise identical.

The one-job-per-CTA path shares `_bind_lane_layout`, so it changes too (`wip/np13` vs `wip/np12`,
`artifacts/r13_nonpersistent.*`): geomean 1157.76 vs 1139.89 (+1.57 %), 11/11 bitwise, every shape up.

## Decision
ACCEPTED (geomean +1.80 %, no regression). 1.191x aiter (1005.24); 1.40x needs another +17.6 %.

## Attribution
- The gain is in the KV loop: long S (pure loop) +1.7-2.0 % in the H=128 binary, and the H64 / H96
  binaries, whose loops had lost the most in round 12, +3.0-3.3 %. 16 fewer VALU-slot instructions
  per iteration in MFMA shadows that are at the issue limit, and 15 free VGPRs, so nothing parks.
- It also removes the cause of round 11 / 12's per-binary swings (where the parked reads landed).
  Round 12's B1 S64K regression (-2.4 %) is recovered: S64K is now +2.0 % above round 12 and -0.5 %
  vs round 11's lucky allocation, with 4K keeping round 12's prologue gain.

## Real-training transfer check
- Bucket: K1 (kernel-internal: LDS addressing / register allocation).
- Cache key: no cache.
- id(...) audit: no key; nothing depends on tensor identity.
- 4-step pen-and-paper trace hit rate: n/a (no cache).
- Benchmark gain on this round: +1.80 % combined-step (forward-only: combined step = forward).
- Estimated real-training gain: +1.80 % at these shapes. Pure instruction / register change inside
  the kernel, bitwise-neutral, independent of tensor identity and data.
- Decision: ACCEPT-as-real.

## Next direction
1. The prologue's pre-QK0 `vmcnt(VM_DRAIN_V)` waits for the first half of V0 (K1 is resident since
   round 11); V0 is first read in C1 after a `vmcnt(0)`, so leave the whole V0 tile in flight there.
2. Then wait only for the K prefetch in the prologue (`vmcnt(<O stores>)`), not the O stores.
3. With 15 VGPRs free, revisit what round 8 lost to register pressure (carrying the C1 K head over the
   back edge) and the Q prefetch for the next job.
