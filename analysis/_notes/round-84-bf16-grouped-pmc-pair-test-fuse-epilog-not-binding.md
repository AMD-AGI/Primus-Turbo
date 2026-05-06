# Round 84 — bf16 grouped GEMM weighted wall

> **Context:** auto_optimize round 7 / 100, MI355X. Continuation of
> R83's PMC pair-test plan. R83 closed with two candidate levers — (1)
> FUSE epilog live-state trim, (2) layout-level / KI specialization —
> and a planned PMC pair test to disambiguate which is binding.

**Status:** PMC pair test **DONE** + **R84 KI=44 FUSE specialization
RE-FALSIFIED** (28-VGPR spill, -40 score; same root cause as R83's
KI=88 attempt but with the corrected `g.ki` value). **Diagnostic
discovery:** R52 / R83's "g.ki = 2816/32 = 88" comments were stale
from a pre-K_STEP=64 era — the actual `g.ki = fast_k / K_STEP =
2816 / 64 = 44` for K=2880, so R52's `INSTANTIATE_K_GRP(88)` is dead
code that no metric shape ever reaches, and R83's KI=88 FUSE attempt
also never actually triggered (it fell through `default` in the
dispatcher switch).

| run | weighted score | gpt_oss geomean |
|-----|---------------|-----------------|
| baseline (R83 commit 329fad1) | 882-884 (±2 noise) | 1.093 |
| R84 KI=44 FUSE attempt        | 842 / 843 / 843 (mean 842.7) | (regressed) |
| post-revert (FUSE_DISABLE gate kept, no kernel change) | 880 / 883 / 882 | 1.093 |

KI=44 FUSE: 28 VGPR spills + 116 B/lane scratch → -40 score. **REVERT**.

## Part A: PMC pair test — FUSE on vs FUSE_DISABLE=1

Added a runtime-readable env gate `BF16_FUSE_DISABLE` to
`dispatch_grouped` (host code, static const lambda IIFE — read once
per template instantiation, zero hot-path overhead). Production
behavior unchanged when env not set. Setting the env to "1" routes
all RCR FUSE-eligible shapes (K%128==K_STEP) through the non-FUSE
main + `grouped_ktail_kernel_mfma32x32_M{2,4}` RMW K-tail kernel,
exactly the path FUSE was added (R3) to eliminate. Same target shape
(gpt_oss-Down-B4-M2048, B=4 M=2048 N=2880 K=2880, fwd+bwd, 13 iter
post-warmup):

| run                               |  kernel                          | n  | dur_us | LDSBC% | MfmaU% | vgpr | Occ% |
|-----------------------------------|----------------------------------|----|--------|--------|--------|------|------|
| **FUSE on** (production)          | `grouped_kernel<RCR,0,FUSE>`     | 26 | 3806   |  0.0   | **42.9** | 128  | 15.9 |
| **FUSE_DISABLE=1** (R84 probe)    | `grouped_kernel<RCR,0>`          | 26 | 3499   |  0.0   | **46.3** | **124**  | 15.8 |
| **FUSE_DISABLE=1** (R84 probe)    | `grouped_ktail_kernel_M4<RCR,64>`| 26 | 1358   |  0.0   |  2.0     | 28   | 47.7 |
| (var-K kernel — unchanged) FUSE on | `grouped_var_k_kernel<RCR>`      | 14 | 4184   | 16.0   | 42.5     | 128  | 16.4 |
| (var-K kernel — unchanged) FUSE off| `grouped_var_k_kernel<RCR>`      | 14 | 4233   | 16.0   | 42.8     | 128  | 16.6 |

**Findings:**

1. **FUSE epilog is NOT the dominant binding constraint.** Removing
   FUSE lifts main-loop MfmaU from 42.9% → 46.3% (+3.4pp) and frees
   4 VGPRs (128 → 124). But the standalone K-tail RMW kernel costs
   1358 us — fwd+dA total time goes from 3806 us (FUSE) to 4857 us
   (FUSE_DISABLE = 3499 + 1358), **+27% wall**. FUSE is doing exactly
   what it was designed for (eliminating the K-tail RMW launch).
2. **Main-loop MfmaU caps at 46.3% even without FUSE.** Far below
   DSV3 KI=64 RRR's 74%. The remaining ~28pp gap is NOT explained by
   the FUSE epilog — it's structural (layout / KI specialization /
   register tile shape).
3. **var-K kernel is unaffected** (PMC stats identical between FUSE
   on and off, as expected — it doesn't go through the FUSE path).
4. **FUSE epilog adds 4 VGPRs of live state through the main loop.**
   Recoverable lever, but small (~+3.4pp MFMA util best case = ~+7%
   ratio = ~+5 score on gpt_oss). Not worth a multi-round epilog
   surgery when KI specialization (path 2) might give 2-3× more.

### Verdict

**Lever 1 (FUSE epilog live-state trim) FALSIFIED as a primary
lever.** A 4-VGPR live-state weight is real but not where the bulk
of the 31pp MfmaU gap to DSV3 lives. Pivot to lever 2.

## Part B: discovery — R52 / R83's `g.ki` mapping was stale

Cross-referenced the dispatcher's `case 88` against the actual
`g.ki = g.fast_k / K_STEP` computation (kernel_bf16_dynamic.cpp
line 1520). With `K_STEP = 64` (line 7), gpt_oss K=2880 hits
`fast_k = 2816, g.ki = 44` — NOT `88`. The R52 comment claims
"g.ki = 2816 / 32 = 88", reading a stale `K_STEP=32` value that
existed in an earlier era of the codebase.

Audit of the existing KI specs against actual `g.ki` for the metric
suite (K = `fast_k / 64`):

| metric K | family               | fast_k | g.ki | hits spec? |
|----------|----------------------|--------|------|-----------|
| 7168     | DSV3-GateUP          | 7168   | 112  | ✓ KI=112  |
| 4096     | DSV3-Down dA / Qwen3-GateUP | 4096   |  64  | ✓ KI=64   |
| 3072     | (Qwen3-GateUP B-side variant) | 3072   |  48  | ✓ KI=48   |
| 2880     | **gpt_oss-{GateUP,Down}**     | 2816   |  **44**  | ✗ falls to KI=0 |
| 2048     | DSV3-Down            | 2048   |  32  | ✗ falls to KI=0 |
| 1536     | Qwen3-Down           | 1536   |  24  | ✗ falls to KI=0 |

`INSTANTIATE_K_GRP(56)` (g.ki=56 → K=3584): NO metric shape.
`INSTANTIATE_K_GRP(88)` (g.ki=88 → K=5632): NO metric shape (DEAD).
`INSTANTIATE_K_GRP(128)`, (172), (224), (256), (296), (448), (462),
(832): all > 112; no metric shape.

**Three metric-shape K values land on KI=0 generic** (K=2880, 2048,
1536) — all the K%128==64 / K%64-aligned-but-not-spec'd shapes. This
is the actual KI-specialization gap on the metric.

R83's KI=88 FUSE template instantiation was therefore dead from
the start; R83's "flat metric" finding is consistent with the
template never being reached. The right experiment was always
KI=44 FUSE (this round).

## Part C: R84 lever — `grouped_kernel<RCR, 44, FUSED=true>`

Added `INSTANTIATE_K_GRP`-style explicit instantiation for KI=44
+ FUSED=true RCR + dispatcher case 44. Build report:

| kernel                           | VGPRs | Spill | Scratch B/lane | Occ |
|----------------------------------|-------|-------|----------------|-----|
| `grouped_kernel<RCR,0,FUSE>` (existing) | 250 | 0  | 0   | 2 |
| `grouped_kernel<RCR,44,FUSE>` (R84 attempt) | 256 | **28** | **116** | 2 |
| `grouped_kernel<RCR,88,FUSE>` (R83 attempt — dead but built) | 256 | 9 | 40 | 2 |

KI=44 / FUSE spills WORSE than KI=88 / FUSE despite half the unroll
factor. Two factors compound:

1. KI=44 has more `main_loop_iter` lambda live-range overlap with the
   FUSE K-tail epilog block — the half-unroll keeps more partial-state
   regs alive across the lambda boundary than KI=88's full-unroll
   does (LLVM eagerly spills loop-carried-but-also-fuse-block-live
   regs at the lambda exit).
2. The KI=44 unroll lands in a register-allocator bucket where the
   compiler chooses scratch over re-materialization more aggressively
   than at KI=88 (R83) or KI=0 (production) — the body's MMA-tile
   pre-issue pattern at 22 unrolled main_loop_iter pairs (44 / 2) is
   pessimised vs the "fully simple" KI=0 (1 iter pair, dynamic loop)
   or "fully aligned" KI=88 / KI=64 / KI=112 (full unroll, single
   basic block).

**Metric (3 runs):**

| run | score |
|-----|-------|
| 1   |  842  |
| 2   |  843  |
| 3   |  843  |

`-40 score`. **REVERT.** The MFMA-pipe headroom KI specialization
was supposed to unlock is dwarfed by the spill traffic.

### Verdict

KI specialization on the FUSE template (lever 2) **FALSIFIED for
KI=44**. Remaining KI specialization candidates for the FUSE path:

* KI=44 **non-FUSE** (R52-style spec, no epilog block): no current
  shape uses the non-FUSE path at K=2880 (FUSE eligible takes
  precedence). Could become reachable IF FUSE is disabled for K=2880
  via a finer eligibility predicate, BUT R84 part A showed
  FUSE_DISABLE = +27% wall — net loss.
* KI=44 **FUSE with epilog state hoist** (would require trimming the
  FUSE epilog's live-range overlap with the unrolled main_loop_iter
  lambda): multi-round kernel surgery, high risk (R39 saw spill at
  KI=44 FUSED=true; R83 at KI=88; R84 at KI=44 → re-confirmed the
  pattern).

The KI specialization avenue for K=2880 forward / dA appears closed
in single-round form. The FUSE template's main-loop unroll strategy
(KI=0 + #pragma unroll 2) is at a local min.

## Part D: what to keep — `BF16_FUSE_DISABLE` env probe gate

Kernel changes are **fully reverted** (no KI=44 spec, no dispatcher
case 44). The single permanent change is the `BF16_FUSE_DISABLE`
env-runtime gate (similar to existing `BF16_RRR_FUSE_PROBE`). Read
once per process via static const lambda; zero hot-path overhead.
3-run baseline metric with the gate compiled in (env unset): 880 /
883 / 882 — same ±2 noise band as without the gate.

This gate stays in production for future PMC pair tests — the cost
of adding it is one `getenv` per template instantiation per process,
the value is being able to A/B test any FUSE-vs-non-FUSE hypothesis
in 30 min without touching kernel source.

## Part E: direction for R85

PMC + R83 + R84 jointly ruled out the easy single-round levers on
the FUSE / KI dimension for gpt_oss K=2880. Remaining surface:

1. **dB var-K LDS swizzle on the 16% LDSBC remainder** (R82's
   originally planned R84). var-K kernel has LDSBC=16% — not the
   dominant bottleneck (MfmaU=42.5% is) but real. A custom swizzle
   that drops LDSBC to ~5% might lift MfmaU by ~2-3pp = ~+3 score
   on gpt_oss. Multi-round (R74 falsified one swap; needs a
   targeted padded-32x16 prototype). NOT a one-round win.
2. **Layout pivot** — DSV3 RRR FUSE=false at KI=64 hits MfmaU=74%
   while gpt_oss RCR FUSE at KI=0 hits 42.9%. The 31pp gap might
   come from the rt_16x32_s vs rt_32x16_s register-tile shape
   difference (forward grouped_kernel uses rt_16x32_s on B for RCR,
   rt_32x16_s for RRR). Investigating whether the gpt_oss path
   could route through RRR (via a different transpose pattern at
   the Primus dispatcher) is worth a probe but the H4 reroute
   already does the opposite (RRR → RCR for K-tail). Likely
   structural — not a one-round win either.
3. **Accept the 882 plateau on BF16 grouped and pivot to FP8** —
   the FP8 grouped task is concurrent and may yield single-round
   wins more easily. (Out of scope for this task body — the BF16
   line continues regardless.)

R85 candidate: **PMC walk on the var-K kernel's LDS access pattern**
(no kernel change, ~30 min) — narrow rocprofv3 to just the
`grouped_var_k_kernel` invocation and pull `SQ_LDS_BANK_CONFLICT`
+ `SQ_LDS_IDX_ACTIVE` raw counters per dispatch to identify which
LDS chunk pattern is conflicting. If the conflicts cluster on a
specific access (B-tile load vs A-tile load vs accumulator
spill), a targeted pad-tile swap (smaller surface than R74's
full-tile-replacement) becomes the R86 candidate.

## Files touched

* `/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
  — `BF16_FUSE_DISABLE` env-runtime probe gate added (kept for
  future PMC pair tests). KI=44 FUSED=true instantiation +
  dispatcher case 44 + `launch_one_grouped_fuse<L,KI>` template
  REVERTED.
* This round note (Primus-Turbo).
* `/tmp/r6_pmc_target.py`, `/tmp/r7_compare_pmc.py`,
  `/tmp/r6_pmc_counters.txt`, `/tmp/r7_pmc_{fuse,nofuse}_out/...`
  — PMC artifacts (offline, not committed).
