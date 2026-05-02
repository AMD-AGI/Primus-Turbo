# Round 62 — FP8 grouped: Lever A audit COMPLETE, already applied; roadmap revised

**Date**: 2026-05-02 (R62 of 100)
**HEAD before**: 68ae41be491466291cbc6cf04a36938ed936dacf
**Score**: baseline 981 → final 981 (no code change, no regression)
**Goal**: per R61 closure plan, audit production grouped FP8 K-loop for sync vs async load patterns; identify VGPR-staging hot spots; prototype Lever A modification.

## TL;DR

**Lever A is ALREADY FULLY APPLIED in production.** Audit of
`grouped_rcr_kernel` confirms all global→shared loads in the K-loop
go through `rcr_8w_load_hoist`, which uses
`buffer_load_dwordx4 ... offen lds` ASM intrinsic = direct global→LDS
DMA (no register staging). There is **no `g2s_pass<i>` synchronous
path remaining in the grouped kernel** (those exist only in the dense
`do_cluster` helper at line 1267, not used by grouped).

This **invalidates the R61 closure note's R62-R65 roadmap**. There is no
"async load porting" work left to do for grouped FP8.

## Production grouped FP8 resource reports (current state)

Build with `-Rpass-analysis=kernel-resource-usage`:

| Template spec | TotalSGPRs | VGPRs | AGPRs | Scratch [B/lane] | Occ | VGPR Spill |
|---|---|---|---|---|---|---|
| `<0, false, false>` | 64 | 256 | 0 | 220 | 2 | 54 |
| `<0, true,  false>` | 66 | 256 | 0 | 156 | 2 | 38 |
| `<0, false, true>`  | 77 | 256 | 0 | 140 | 2 | **34** |
| `<0, true,  true>`  | 79 | 256 | 0 | 152 | 2 | 37 |

(KI_HINT=0, N_MASKED_STORE=second bool, FUSED_KTAIL=third bool.)

**Counter-intuitive finding (preserved from R34 docstring)**: adding
features (FUSED_KTAIL=true, N_MASKED_STORE=true) gives LLVM a richer
liveness graph and produces LOWER VGPR spill. The "minimal" spec
`<false, false>` has the WORST spill at 54.

## Spec routing analysis — are we using the best spec for each shape?

Audit of `dispatch_grouped_rcr` (line 6132 onwards):

```cpp
fuse_ktail_eligible =
    (g.bpc > 0) && (g.ki > 0) &&
    ((K_REM == 64) || (K_REM == 0)) &&  // Round-34: extended to K%128==0
    (g.m_per_group >= 16) && (g.m_per_group % 16 == 0);
```

For the 24-shape suite:

- **DSV3 GateUP/Down (K=7168/2048, K%128==0)**: K_REM=0 → eligible=true,
  n_aligned=true → `<false, true>` (spill **34**, BEST)
- **gpt_oss GateUP/Down (K=2880, K_REM=64)**: eligible=true, n_aligned
  depends on N (5760/2880, both NOT 256-aligned → masked) →
  `<true, true>` (spill **37**)
- **Qwen3 GateUP/Down (K=4096/1536, K%128==0)**: K_REM=0 → eligible=true,
  n_aligned=true (3072/4096 both 256-aligned) → `<false, true>` (spill **34**)

**ALL 24 metric shapes route to spill ∈ {34, 37}**. The spill=54 spec is
**NEVER USED** — it's only present as a fallback for shapes that fail
the `lds_k_tail_safe_for_fuse` check (m_per_group < 16 or % 16 != 0),
which no metric shape hits.

So **production grouped FP8 is at optimal spec routing**. R34's
"FUSED_KTAIL=true gives better LLVM allocation" insight has been
extended to cover EVERY metric shape.

## What's left? Plateau analysis

With Lever A applied + spec routing optimal + all R49+ micro-knobs
(vmcnt, sched_barrier, setprio, swizzle, etc.) frozen, the remaining
levers from the task body are:

| Lever | Status | Comment |
|---|---|---|
| A. Async global→LDS | **APPLIED** (rcr_8w_load_hoist) | nothing to change |
| B. Dual LDS ping-pong | **APPLIED** (As[2][2], Bs[2][2]) | already at LDS budget cap (140 KB / 64 KB occ-2 ≈ tight) |
| C-1. Reduce VGPR pressure | various tried, plateau | spill at 34-37, marginal room |
| C-2. Force AGPR allocation | **CLOSED R61** (LLVM bug) | won't unblock until LLVM upgrade or different cell shape |
| D. 32x32x64 mfma cell shape | **FALSIFIED** R37+ R56-64 | scaffold present but switch loses 5+pp |
| E. Manual ASM main-loop scheduling | not tried, high risk | last resort if A+B+C+D all fail |
| F. Qwen-Down K=1536 short-K variant | not tried, low impact | only 4/24 case, +0.8pp geomean ceiling |

The **per-shape ratio table** from R62 baseline (sorted ascending):

```
worst 5 FP8 cases (need ≥ 1.20 to score):
  1. gpt_oss GateUP B32 M4096      1.077  (need +11.4%)
  2. gpt_oss Down   B32 M2048      1.104  (need +8.7%)
  3. gpt_oss GateUP B32 M2048      1.107
  4. gpt_oss GateUP B4  M4096      1.108
  5. gpt_oss Down   B32 M4096      1.111
worst Qwen cases:
  Qwen Down   B16 M4096            1.136
  Qwen GateUP B32 M2048            1.167
  Qwen GateUP B16 M4096            1.168
```

The **gpt_oss** subset dominates the worst-5 list. They all use the
`<true, true>` spec (spill 37, includes K-tail + N-mask). The
arithmetic-intensity ceiling for K=2880 with ~4% K-tail + ~4% N-mask
overhead suggests we'd need a 11% main-loop speedup to lift case 1 from
1.077 → 1.20 — that's beyond what plausible micro-opts can deliver.

The **Qwen3** subset (K=4096/1536, all 128-aligned, no K-tail/N-mask
overhead) ranges 1.136-1.224. Lever F (K=1536 specialization) targets
the bottom 2 Qwen Down cases.

## Decision and revised roadmap

R62 was an **audit-only round**. The finding (Lever A already applied)
forces a re-evaluation of the pivot strategy.

**R63-R64 plan** (revised, conservative):
- **R63**: try Lever F-style specialization for Qwen Down K=1536 — add
  a new template instantiation `grouped_rcr_kernel<KI_HINT=12, ...>`
  with compile-time-known K iteration count. Dispatcher routes Qwen
  Down (K=1536, ki=12) to it. Goal: reduce loop epilog overhead via
  better unroll / liveness for the KI_HINT=12 spec; target +2-3pp on
  4 cases ≈ +0.5pp geomean.
- **R64**: if R63 wins, also try KI_HINT=32 spec for Qwen GateUP
  (K=4096, ki=32). Same pattern, different KI_HINT.
- **R65**: if both win, score climbs ~+5-8pp = score 985-988. If
  neither wins, declare plateau and accept score 977-983.

**Lever E (manual ASM)** stays as the last resort. Estimated effort: 5+
rounds, high risk. Only attempt if R63-R64 falsify and patience > 15
rounds remaining.

## Files touched (R62)

- `Primus-Turbo/analysis/_notes/round-62-dm-...`: this note (audit only)
- HK kernel source: NO CHANGES (audit was read-only)

## Production kernel impact

NONE. R62 is a pure documentation round. Metric: 981 → 981 (no code).

## Why NOT continue debugging C-2?

The R61 closure note documents the deterministic LLVM AGPR bug. R62
audit confirms there's no "easy escape" via Lever A. Re-opening C-2
would require either (a) writing an entirely different kernel skeleton
that doesn't trigger the bug — high risk, multi-round effort, OR
(b) waiting for an LLVM upgrade — outside this run's control.

Lever F is lower risk (template specialization, no architectural change)
and gives a clear +/- signal in 1 round. That's the right next move.
