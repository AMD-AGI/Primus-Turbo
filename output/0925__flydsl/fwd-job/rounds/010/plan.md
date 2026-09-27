# Round 10 plan

**Selected: `r8.i1.g20` (h29, nodelay re-land), shipped alone.**
Prediction: mean prod_vs_champion >= 1.007 over >=3 rotated no-beat sessions, every session > 1.003.
fast/proxy 0.99-1.01. Loop 513 -> 499, VGPR 456, 0 spill, hash o=26a89a2db0cd. Details: `plan.yaml`.

## Why

It is the only `must` and the only win measured in this job: prod 1.0062-1.0110 in rounds 8-9 (`facts.md` r8.i1.g20).
Round 9 lost it by shipping it merged with lock_simd. This round it ships alone and becomes the base for every later arm.

## Instrument taken this round (`raw/postflush/summary.md`)

The planning exchange put first a cheap discriminator for `r10.i4.g29`. That candidate claimed the 256 MiB flush's
write-back makes current pay 12-16 us at fast. I ran it in the plan stage rather than filing it:

- **One implementation per process.** After the scored z256 pattern, current reads 38.50-38.70 us,
  the same as GPU-sleep, a 256 MiB read, or a warm back-to-back call (±0.4 us). There is no flush term.
  - The only elevated rows are `none`/`z8`: host launch (~15.5 us) lands inside the timing window.
  - beat shows the same rows elevated. It is not a kernel property.
- **Several FlyDSL images interleaved in one process.** This reproduces round 9's 50-55 us median (4 images: inc 50.32).
  - 2 images: 38.9-41.7 us.
  - current+beat: 66.0 us.
  - The fast "excess" was the harness interleaving images: the r9.i3.g26/r6.i1.g16 cold start.
  - **Consequence for act:** never more than candidate + champion in a process.

g29 is therefore falsified and its depth-3 arm is not built.

## What was argued ([dialogue.md](dialogue.md))

- **r10.i1.g27** (reviewer, fixed zero reference). Both sides concluded it is impossible, not merely unmeasured.
  - h16's ±1e4 domain overflows exp2 and gives l=0 on an all-negative row.
  - The op contract has no score bound (`impl.py:81`).
  - A guard would bring back the row max the change removes.
  - Filed for retirement.
- **r10.i4.g29.** The reviewer changed my A/B/C test into a write-volume / read / no-write / delay control set. I accepted it.
  The result is above.
- **r8.i1.g20.** The prediction became an aggregate: mean >= 1.007, every session > 1.003.
- **r10.i3.g28 pairing.** The premise is now 1 WG per LDS-owning unit, not per CU. It is judged against nodelay.
  The static loop-length gate is dropped (correction 3). Its value is unpriced.
- **r7.i3.g19.** Kept in the pool. It needs a priced depth-3 carrier on nodelay first.

The exchange closed converged. No disagreement survives.

## Open

- **Pairing (`r10.i3.g28`)** is the next idea after nodelay.
- **Body lever for g19.** No other body lever is priced. g19 still needs a ring that costs +3.3% per tile.
- **Fast gap after h28 cleanup.** Warm and single-image, current is 38.7 us vs beat 33.1 us (1.17x) at 32 WGs.
  That is g04's split-KV territory, and g04 was ruled stale at 1.13.
