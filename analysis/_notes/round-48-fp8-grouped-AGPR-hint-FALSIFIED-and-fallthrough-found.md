round-48-fp8-grouped-AGPR-hint-FALSIFIED-and-fallthrough-found.md
=================================================================

Round: 48 / 100
Date: 2026-05-02
SHA: b2a0afbc (pre) → TBD (post)
Task: grouped FP8 tensorwise (fwd + var-K bwd + dA bwd) on the 24-shape suite

## TL;DR

R47 (last round) ladder step 3 (inline-asm AGPR migration via ``+a``
constraint hint) was implemented and **falsified**. Standalone unit
test ``/tmp/test_agpr2.hip`` confirmed clang accepts ``+a`` and uses
AGPR for the result, but applying the same hint inside ``rrr_mma``,
``rcr_mma``, ``crr_mma`` of the grouped FP8 kernel **regressed all 4
``grouped_rcr_kernel<*,*>`` template specs by 4-6× spill** + halved
the VGPR budget:

```
                              before (R47/R48 baseline)   after AGPR hint
Spec <F,F>                    VGPR 256 / AGPR  0 / 54sp  → V 128 / A 128 / 129sp
Spec <T,F>                    VGPR 256 / AGPR  0 / 38sp  → V 128 / A 128 / 129sp
Spec <F,T>                    VGPR 256 / AGPR  0 / 34sp  → V 128 / A 128 / 188sp
Spec <T,T> (gpt_oss target)   VGPR 256 / AGPR  0 / 37sp  → V 128 / A 128 / 197sp  *5.3× worse*
grouped_rrr_kernel            VGPR 256 / AGPR  0 / 65sp  → V 128 / A 128 / 129sp
grouped_var_k_kernel_fp8      VGPR 256 / AGPR  0 / 37sp  → V 128 / A 128 / 143sp
```

**Root cause** (post-hoc): on gfx950 the per-lane register file is
**shared** between V and A — using 128 AGPR forces VGPR limit down to
128 (occupancy 2 wave/SIMD constraint). The non-acc working set
(a/b/offsets/SRDs/loop control = ~120 VGPR) no longer fits in 128
VGPR → cascading spill of MORE values to scratch (197 vs 37 = +160).
The hint forces AGPR for accumulators (intended) but at the cost of
forcing other live values to scratch (unintended). LLVM already had
optimal allocation in baseline (256 V, 0 A, 37 spill); the hint
overturned that without a compensating mechanism.

The earlier R47 hypothesis "match rcr_4w's 256 AGPR pattern via
hints" was incorrect: rcr_4w's AGPR allocation works because its
**accumulator alone** saturates the VGPR budget (256 VGPR), forcing
the compiler to use AGPR for acc as the only viable path. In our
grouped kernel the acc is only 128 VGPR, so the V+A allocation can't
both fit cleanly without spilling SOMETHING.

The hint was reverted; current state matches R47 baseline exactly:

```
$ git diff HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp
# (empty)
```

## Metric this round (revert verified)

```
Run 1 (R47 baseline, just after auto_optimize.py recorded 998):
  geomean grp_BF16 = 1.2145, grp_FP8 = 1.1923, score = 997
Run 2 (after R48 revert):
  geomean grp_BF16 = 1.2395, grp_FP8 = 1.1806, score = 992
```

Run-to-run noise = 5 score points / 1.1pp on grp_FP8 geomean. Both
runs same kernel (b2a0afbc HEAD = R47 doc-only commit). Effective
plateau at **score 992-998 (centred ~995)**.

## Worst FP8 shapes (R48 metric, run 1)

```
ratio  shape                                hk_TFLOPS  triton_TFLOPS  GAP
1.070  gpt_oss_20B-GateUP-B4-M4096            1921.0       1795.9    -13.0pp
1.090  gpt_oss_20B-Down-B32-M2048             1821.5       1671.2     -11.0pp
1.102  gpt_oss_20B-Down-B32-M4096    *no rule  1935.3       1756.2     -9.8pp
1.119  gpt_oss_20B-GateUP-B32-M4096           2097.5       1874.1     -8.1pp
1.132  gpt_oss_20B-Down-B4-M4096              1778.7       1571.4     -6.8pp
1.132  Qwen3-235B-A22B-Down-B16-M4096         1838.0       1624.0     -6.8pp
1.157  Qwen3-235B-A22B-Down-B32-M4096         1849.8       1599.3     -4.3pp
1.161  DeepSeek-V3-GateUP-B16-M4096           2735.5       2355.7     -3.9pp
1.167  Qwen3-235B-A22B-GateUP-B32-M2048       2451.8       2101.5     -3.3pp
1.172  Qwen3-235B-A22B-Down-B32-M2048         1808.4       1543.2     -2.8pp
```

The bottom 5 are all gpt_oss K=2880 — same plateau identified in
R44-R46. Top of the worst list: **gpt_oss-GateUP-B4-M4096 ratio 1.070**.

## NEW: gpt_oss-Down-B32-M4096 dispatch fall-through detected

Audit of ``primus_turbo/pytorch/kernels/hipkitten/config.py`` shows
that **gpt_oss-Down-B32-M4096** (tiles_n=11, tiles_m=16, k=2880,
m_total=131072) has **NO matching rule** and falls through to the
binding default ``(group_m=4, num_xcds=None=8)``. Sibling cases:

  - tiles_n=11, tiles_m=8, m_total=8192 (B=4 M=2048):  rule (gm=2, xcd=2) — round-7
  - tiles_n=11, tiles_m=16, m_total=16384 (B=4 M=4096): rule (gm=32, xcd=4) — round-12
  - tiles_n=11, tiles_m=8, m_total=65536 (B=32 M=2048): rule (gm=16, xcd=4) — round-8
  - **tiles_n=11, tiles_m=16, m_total=131072 (B=32 M=4096): NO RULE → default (gm=4, xcd=8)**

Round-12 commentary (line 1115 of config.py) explicitly says:
> No metric shape regression expected.

It was correct that no shape REGRESSED, but it ALSO MISSED that the
B=32 M=4096 sibling could benefit from the same family of rules. R49
should sweep this shape and add a rule if a winning (gm, xcd) is
found.

The expected gain is small per shape (+1-3pp), but combined with
R47's AGPR finding (which needs deeper restructure to actually pay
off), even small dispatch closures help bridge the geomean 1.1923 →
1.20 gap.

## R47 ladder status update

| Lever                               | R47 status     | R48 result                   |
|-------------------------------------|----------------|------------------------------|
| C-4 ``-mllvm -amdgpu-mfma-vgpr-form=0`` | NEXT R48       | **FALSIFIED** (no-op, default) |
| C-3 ``+a`` inline-asm AGPR hint     | R49+ if C-4 fail | **FALSIFIED** (5× spill regression) |
| C-2 warp-tile restructure to 4w     | R50+ fallback  | unchanged                    |

The simple hint-based AGPR migration path is dead. Remaining options:

  - **C-2** (warp-tile to 4w): structural, 2-3 rounds, retains all
    the trade-offs documented in R8-dm. Per R47 commentary the rcr_4w
    has occ=1 wave/SIMD vs grouped's occ=2 — porting may regress
    occupancy as well.
  - **R49 dispatch fall-through closures** (cheap, kernel-untouched):
    sweep gpt_oss-Down-B32-M4096 and add a rule. ~3pp expected on
    one shape ≈ +0.05pp geomean. Helps bridge but won't solo close
    the gap. Multi-round investment in more sweeps could compound.
  - **C-1' (per-spec specialization)**: see if the gpt_oss-specific
    ``<T,T>`` template spec can have its 37 spill reduced via per-
    spec K_REM=64 constexpr branch (eliminates the 2 cmp ops + 2
    cndmask ops for SENTINEL voffset on every K-tail lane = 5-10
    VGPRs freed = potentially close the 37 spill to 25-30). Same
    template-specialization technique as ``FUSED_KTAIL`` itself; just
    extending one more axis. **Very specific to gpt_oss**. 1-2 rounds.

## What this round actually changed

- **No kernel changes** (AGPR hint reverted; HK working tree clean).
- **No Primus-Turbo Python changes**.
- **Only this round note** ⇒ Primus-Turbo doc-only commit.

HipKittens HEAD unchanged at ``92407889``. Working tree clean after
revert.

## Validation paper-trail

```
/tmp/metric_round_48.log              (R47 baseline state, score 997)
/tmp/build_round_48_agpr.log          (with AGPR hint: 197 spill on <T,T>)
/tmp/build_round_48_revert.log        (after revert: 37 spill on <T,T>)
/tmp/metric_round_48_revert.log       (revert verified, score 992 — within noise)
/tmp/test_agpr.hip + /tmp/test_agpr2.hip (standalone clang +a constraint test)
```

## Next round (R49) action ladder

1. **First (5 min)**: ``-mllvm -amdgpu-mfma-vgpr-form=0`` was already
   tested R48 (no-op, default = compiler heuristic). **Skip.**

2. **Best ROI (1-2 rounds)**: C-1' per-spec K_REM=64 constexpr
   specialization for FUSED_KTAIL=true gpt_oss path. Add a 4th
   template parameter ``int KREM_HINT = 0`` to grouped_rcr_kernel
   and grouped_var_k_kernel_fp8. When ``KREM_HINT == 64``, replace
   the runtime ``b128_lo_valid = (k_lane_byte + 16) <= K_REM`` with
   constexpr conditional based on lane id (``laneid < 32`` for
   K_REM=64). Saves ~5-10 VGPRs and a few cy/lane in the K-tail
   block. Update dispatch (``dispatch_grouped_rcr``,
   ``dispatch_grouped_var_k_fp8``) to pass ``KREM_HINT=64`` for
   gpt_oss K=2880 shapes.

3. **Cheap closure (1 round)**: sweep gpt_oss-Down-B32-M4096
   (tiles_n=11, tiles_m=16, m_total=131072) at default (gm=4, xcd=8)
   vs candidates from sibling rules. Expected 1-3pp gain on this
   single shape ≈ +0.05pp geomean.

4. **Multi-round structural** (R50+): C-2 warp-tile restructure to
   rcr_4w-style (WARPS_M=2, WARPS_N=2, RBN=64 → 256-VGPR per-warp
   accumulator → forces AGPR allocation). Retains all the trade-offs
   from R8-dm.

The patience streak is now 1/30 (R47's 998 was the last improvement;
R48 was -1 in noise). Plenty of runway to run multi-round structural
work.

## Falsification register

| Lever                                        | Status         | Round   |
|----------------------------------------------|----------------|---------|
| Lever A (async g→LDS) — base shipped         | SHIPPED        | R54-dm  |
| Lever B (dual LDS) — base shipped            | SHIPPED        | early   |
| Lever C-1 (restrict / lifetime hints)        | SATURATED      | R12,R54 |
| **Lever C-1' (per-spec K_REM constexpr)**    | **NEXT R49**   | —       |
| Lever C-3 (art_base AGPR migration)          | not impl       | —       |
| **Lever C-3' (``+a`` inline-asm hint)**      | **FALSIFIED**  | **R48** |
| **Lever C-4 (mfma-vgpr-form mllvm flag)**    | **FALSIFIED**  | **R48** |
| Lever C-2 (warp-tile restructure to 4w)      | NOT STARTED    | —       |
| Lever D K-tail-only port                     | FALSIFIED      | R62-dm  |
| Lever D full main-loop port (R-B 5+)         | NOT STARTED    | —       |
| Lever E (ASM main-loop)                      | NOT STARTED    | —       |
| Lever F (Qwen3 K=1536 short-K variant)       | FALSIFIED      | R35-grp |
| ``amdgpu_waves_per_eu(2,2)`` attribute       | FALSIFIED      | R47     |
| Drop ``__launch_bounds__(_, 1)`` entirely    | FALSIFIED      | R47     |
| sched_barrier / LICM / anti-CSE class        | FALSIFIED      | R31-32  |
| K-tail micro-knobs (vmcnt / reorder)         | SATURATED      | R3-R55  |

## Attribution

- HipKittens HEAD: ``92407889`` — UNCHANGED this round (AGPR hint experiment reverted)
- Primus-Turbo: only this doc note
- No ``config.py`` / ``dispatch.py`` / kernel changes
- No metric / test edits
