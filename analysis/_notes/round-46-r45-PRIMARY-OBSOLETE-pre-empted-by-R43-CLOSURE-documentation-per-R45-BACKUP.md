# Round 46 — R45 PRIMARY OBSOLETE (already shipped at R43); CLOSURE per R45 BACKUP

## TL;DR

R45's forward pointer for R46 was a **methodology-corrected GateUP-B4-M2048
dgrad-via-H4 (gm, xcds) drift audit** holding production slots=200 + cs=24
FIXED. Hypothesis: production was `(gm=1, xcds=None=8, slots=200, cs=24)`;
sweep `{(1,8)*current, (1,4), (1,2), (1,1), (2,8), (4,8), (8,8)}` to test
if xcds drift moves the optimum off xcds=8.

**That audit was already executed in commit 743599f (daemon round-43)**,
which the commit message explicitly labels "round-43 R46". The R45 author
worked from stale rule state (assumed production xcds=None=8) but R43 had
already shipped `(gm=1, xcds=4, slots=200, cs=24)` with two-run
wmin_beats_lmax evidence (+1.01% / +1.95%). Re-executing R45 PRIMARY in
R46 would re-confirm an already-shipped change — a no-op against the
metric.

Per R45's BACKUP option: **closure documentation**. R44 (Down-B4-M2048
fwd RCR drift) FALSIFIED + R45 (GateUP-B4-M2048 fwd RCR drift) FALSIFIED
+ R43 ship of GateUP-B4-M2048 dgrad-via-H4 xcds None→4 = three
substantive empirical conclusions on the B=4 RCR (gm, xcds) lever class.
The dispatcher (gm, xcds) lever class on the gpt_oss FP8 B=4 RCR cells
is feature-complete on the current binding. Structural ~700 plateau
holds for the lever class.

## Evidence trail

### R43 commit (743599f) — verbatim head

```
perf(config): round-43 R46 — GateUP-B4-M2048 dgrad-via-H4 num_xcds None→4 (+1.0–2.0% wmin_beats_lmax)

R45 PRIMARY forward-pointer executed. With production R10 (slots=200) +
R15 (chunk_size=24) + R16 (gm=1) levers FIXED, the (gm, xcds) sweep on
the GateUP-B4-M2048 dgrad-via-H4 RCR rule (config.py:3022-3029) shows
(gm=1, xcds=4) wins +1.01% then +1.95% over production (gm=1,
num_xcds=None=8) in two independent 5-seed × 2000-iter p20 runs.
wmin_beats_lmax holds in BOTH runs — every seed of (1, 4) beats every
seed of baseline.
```

R43 reads "R45 PRIMARY forward-pointer executed" — but R45 had not yet
run when commit 743599f landed. The cause is timing aliasing: the
internal "R-numbering" inside config.py comment blocks reflects the
audit sequence of a **rule** (the RCR cell at config.py:2645+), which
is independent of the daemon's round counter. R43 (daemon) shipped
what the rule-history-numbering called "Round-46 of this rule" (the
46th retune of this cell-class across all runs).

The R45 author then literally interpreted "R46" as the next daemon
round and forward-pointed R46 to re-do the audit. Two timing aliases
collided.

### R45 forward pointer prescription (literal quote)

> "PRIMARY — GateUP-B4-M2048 dgrad-via-H4 RCR rule (gm, xcds) drift
> audit, METHODOLOGY-CORRECTED. With the R10/R15/R16 levers held FIXED
> at production (slots=200, cs=24), the (gm, xcd) optimum may have
> shifted. R45 baseline data point: (1, 4) at slots=0/cs=0 = 1846.4 T;
> production (1, xcds=None=8, slots=200, cs=24) per R16 estimate ≈
> 2082 T. Probe should clone _probe_round_45 → swap to time_dgrad-only,
> FIX slots=200 + cs=24, sweep cells {(1, 8)*current, (1, 4), (1, 2),
> (1, 1), (2, 8), (4, 8), (8, 8)} where (1,8) is the current
> production."

### Why R45 PRIMARY is OBSOLETE

The current production for GateUP-B4-M2048 dgrad-via-H4 is at
config.py:3108-3115:

```
return HipKittenConfig(
    layout=layout,
    group_m=1,
    num_xcds=4,         # ← already 4, not None=8
    kernel=None,
    num_slots=200,
    chunk_size=24,
)
```

R45's "(1,8)*current" baseline cell was wrong: production has been
(gm=1, xcds=4) since 743599f (3 rounds ago). The proposed sweep would
A/B (1,4)*new-baseline against {(1,8), (1,2), (1,1), (2,4), (4,4),
(8,4)}. The (1,4) → (1,8) and (1,4) → (1,2) directions are exactly
the in-tab data points already shipped:

```
R43 commit data (run-1 vs run-2):
  (1, 8)   0.1312 / +0.00 base   0.1296 / +0.00 base
  (1, 4)   0.1299 / +1.01 ship   0.1270 / +1.95 ship
  (1, 2)   0.1320 / -0.61        0.1297 / -0.12
  (1, 1)   0.1364 / -3.93        0.1342 / -3.55
  (2, 8)   0.1345 / -2.47        0.1329 / -2.56
  (4, 8)   0.1346 / -2.53        0.1327 / -2.41
  (8, 8)   0.1352 / -2.99        0.1334 / -2.93
```

Re-running with (1,4) as the new baseline would reproduce the same
ranking — every other cell loses to (1,4), and (1,8) is now -1.01%
to -1.95% relative to current production. No new information, no
metric movement.

## R44+R45 falsification chain — closure status

Three consecutive substantive findings on the B=4 RCR (gm, xcds) lever
class within this 5-round window:

| Round | Cell | Verdict | Action |
|---|---|---|---|
| R43 | GateUP-B4-M2048 dgrad-via-H4 | xcds None→4 SHIP | perf commit 743599f |
| R44 | Down-B4-M2048 fwd RCR | drift FALSIFIED ((16,2) at local opt) | docs commit d6ca891 |
| R45 | GateUP-B4-M2048 fwd RCR | drift FALSIFIED ((1,4) at local opt) | docs commit d5cbbfa |

The two FALSIFIED cells span both (B=4 fwd) RCR rules in the gpt_oss
FP8 metric. The shipped cell exhausts the only remaining (gm, xcds)
slot in the post-H4 dgrad slot class on B=4 cells. The B=4 fwd/dgrad
RCR (gm, xcds) lever surface is now empirically saturated.

## Per-task scoring math (closure)

Score breakdown (from task file phase-0 baseline, 696 median):
- fwd avg ≈ 1898 T → progress 0.678
- dgrad avg ≈ 2097 T → progress 0.749
- wgrad avg ≈ 1807 T → progress 0.645

R43's ship adds +0.0007 → +0.7 score median (sub-noise). R44 + R45
were FALSIFIED so contribute 0. The 5-round window ending here
contributed +0.7 expected score, fully consistent with the observed
[691, 696] band on the 5 daemon metrics (R41-R45). No fraction of the
gap to 900 has been closed by the (gm, xcds) lever class.

## R47 forward pointer (substantive direction, not closure)

The B=4 RCR (gm, xcds) audit is closed, but the suite has 6 more
slot-class × cell combinations un-audited under the
methodology-corrected (production levers FIXED) regime:

### PRIMARY — GateUP-B4-M4096 dgrad-via-H4 RCR rule

Currently config.py:2603-2617 (`tiles_m == 16` branch):

```
return HipKittenConfig(
    layout=layout,
    group_m=1,
    num_xcds=4,
    kernel=None,
    fuse_ktail_off=0,
)
```

No slots/cs override (the R15 audit 4 at
`scripts/_probe_round_15_gateup_b4_m4096_dg.py` found "NO chunk_size
win there", documented at config.py:2913-2918). So this rule sits at
production `(gm=1, xcds=4, slots=NUM_CUS=256, cs=64)` — the
methodology-corrected sweep is the same `(gm, xcds)` grid but with
slots=256, cs=64 FIXED.

This is the LARGEST B=4 cell (m_total=16384, 1.6× the M2048 sibling)
and has NEVER been swept under fixed production levers — only the
chunk_size sweep (R15-4) was done. Probe template:
- Clone `scripts/_probe_round_45_gateup_b4_m2048_drift.py`
- Swap shape to (B=4, M=4096, N=5760, K=2880)
- Set SLOTS=256, CHUNK=64 (production for this rule)
- Sweep cells `{(1,4)*current, (1,8), (1,2), (2,4), (4,4), (8,4),
  (1,1), (16,4)}`
- 5 seeds × 2000-iter p20 each
- SHIP gate: best ≥ 1.0% lift AND wmin_beats_lmax

Estimated probe wall: ~12s remote.

### BACKUP-1 — Down-B4-M4096 dgrad-via-H4 RCR rule audit

Same logic as PRIMARY but for the Down-B4-M4096 cell. Find the rule,
identify production levers, mirror the (gm, xcds) sweep methodology.

### BACKUP-2 — Switch tasks

If R47 PRIMARY also returns NEUTRAL/FALSIFIED, the 4-round chain
(R44-R47) on B=4 RCR (gm, xcds) constitutes definitive closure. The
case for daemon transition to a new optimization task (kernel-template
or NEW DIRECTION A/D from SKILL.md) becomes overwhelming. SKILL.md
NEW DIRECTIONS recommends D (SALU coord-decode for var-K wgrad,
85% SALU bound) as the highest-EV first multi-round project.

## Files touched this round

- `Primus-Turbo/analysis/_notes/round-46-r45-PRIMARY-OBSOLETE-pre-empted-by-R43-CLOSURE-documentation-per-R45-BACKUP.md`:
  this note.

No edits to `config.py` (no rule change — R45 PRIMARY's target was
already shipped at R43). No HK kernel edits. No new probe (the
proposed probe would re-confirm R43's already-committed sweep).

## Decision

**NEUTRAL — no-op-by-design**. R45 forward-pointer to R46 was based
on stale rule state and pointed at an already-shipped change. Closure
documentation per R45 BACKUP. Daemon will re-run canonical metric and
land within the [691, 696] noise band.

## Streak status

Pre-R46 streak: 4 rounds since last improved (R41 → R42-R45 not improved).
Patience cap: 40. R46 expected NEUTRAL → streak 5. Plenty of budget
for the R47 GateUP-B4-M4096 audit to be the next genuine attempt.
