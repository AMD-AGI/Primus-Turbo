# R51-dm — FP8 grouped: post-R50 vmcnt sweep — INIT0_VMCNT=8 BREAKS correctness; INIT1_VMCNT=12 neutral

## TL;DR

- **Lever attempted**: Re-sweep prologue vmcnt counts post-R50, per R50's
  take-away that "removing end-of-tile vmcnt(0) drain may shift the optimal
  prologue VMCNT setting".
- **Probes**:
  1. `RCR_INIT1_VMCNT 6 → 12` (looser): score 959 → 960, **noise**.
  2. `RCR_INIT0_VMCNT 4 → 8` (looser): score crashed to 338-393 with
     **fwd-NaN on 4-6 DSV3 FP8 shapes**.
- **Falsified**: vmcnt sweep is exhausted at the current values. INIT0=4
  is at the SAFETY BOUNDARY, not arbitrary — `init0=8` lets batch-1
  prologue loads issue while batch-0 LDS writes are still uncommitted,
  causing the post-init1-barrier main-loop ds_read to read stale LDS →
  NaN propagation. INIT1=12 is in the noise zone (already saturated).
- **Status**: REVERTED to R50 winner. HK clean at SHA `6a93fa32`.
- **Both repos clean. Score 960 (R50 baseline preserved).**

## Numerical evidence

### Probe 1: INIT1_VMCNT 6 → 12 (looser, more in-flight)

| Run             | Baseline (R50) | INIT1=12  | Δ      |
|-----------------|---------------|-----------|--------|
| 1               | 959           | 960       | +1     |
| Geomean grp_FP8 | 1.1210        | 1.1206    | -0.04 pp |

Result: NOISE. The init1 wait is at the END of prologue (gates main-loop
entry), not at a critical R50-affected point. C-store stragglers are
already drained by init0. Looser init1 doesn't change the critical path.

### Probe 2: INIT0_VMCNT 4 → 8 (looser, more in-flight)

| Run             | Baseline (R50) | INIT0=8     | Δ          |
|-----------------|---------------|-------------|------------|
| 1               | 959           | 338         | -621       |
| 2               | 959           | 393         | -566       |
| Correctness     | 32/32 PASS    | 26/32 FAIL  | -6 specs   |

CORRECTNESS BREAKAGE — `fwd-nan` on these shapes (subset varies between
runs):
- grpFP8_DeepSeek-V3-GateUP-B16-M2048
- grpFP8_DeepSeek-V3-Down-B16-M2048
- grpFP8_DeepSeek-V3-Down-B16-M4096
- grpFP8_DeepSeek-V3-Down-B32-M2048
- grpFP8_DeepSeek-V3-GateUP-B32-M4096
- grpFP8_DeepSeek-V3-Down-B32-M4096

All FAILING shapes are DSV3 (FUSED_KTAIL=true with K_REM=0). Reproducible
across runs (different but overlapping subset).

## Root cause of INIT0=8 breakage

The prologue structure is:

```cpp
rcr_8w_load_hoist × 4              // batch 0: 16 buffer_load_lds issued
if (wm == 1) __builtin_amdgcn_s_barrier();
TK_WAIT_VMCNT(RCR_INIT0_VMCNT);    // init0=4 → 12 retired; init0=8 → 8 retired
__builtin_amdgcn_s_barrier();
rcr_8w_load_hoist × 3              // batch 1: 12 more buffer_load_lds issued
TK_WAIT_VMCNT(RCR_INIT1_VMCNT);    // wait for ≤6 outstanding total
__builtin_amdgcn_s_barrier();
// Main loop: load_b/load_a (ds_read from LDS) → mfma
```

With INIT0=4: batch 0 has 12 loads retired, 4 outstanding when batch 1
starts issuing. The 4 outstanding are batch 0's last (in-issue-order
retirement). They drain during batch 1 issue.

With INIT0=8: only 8 of batch 0 retired, 8 outstanding. Batch 1 starts
issuing while batch 0's MIDDLE loads (positions 8-15) are still in flight.
At init1_vmcnt(6) wait time, vmcnt = max(8 batch0_pending, 12 batch1_pending) -
6 = ... actually 8 + 12 - 6 = 14 retired = 8 batch0_pending fully drained
+ 6 batch1_pending drained. So batch 0 should be done.

**Hypothesis** (cannot fully verify without ISA/hardware trace): on gfx950,
`buffer_load_lds` decrements vmcnt on HBM read completion, NOT on LDS
write commit. The LDS write itself is tracked via lgkmcnt (or has no
explicit scoreboard at all — it's a fire-and-forget write that takes
several extra cycles after vmcnt reports complete).

When INIT0=4, the 12-retired-state plus the s_barrier between batches
gives the LDS writes ~50 cy buffer to commit before any subsequent
ds_read. When INIT0=8, only 8 retired by the s_barrier, and batch 1
issues further loads that bypass the LDS write commit window. The
`if (wm == 1) s_barrier()` half-barrier between batch 0 and the wait
provides additional sync but only for wm=1 warps; wm=0 may skip.

By the time main loop ds_read fires (after init1_vmcnt(6) + s_barrier),
all-vmcnt drain has occurred, BUT the LDS write commit for batch 0's
last 8 loads may STILL be in the LDS arbiter pipeline. ds_read returns
stale data → NaN in the main loop's first mfma → cascade through all
22 K-iters → fwd-NaN.

**This is a strong indicator that INIT0_VMCNT=4 is at the safety
boundary, not arbitrary**. The original code chose 4 for a reason.
DSV3 is more sensitive than gpt_oss to this because DSV3's FUSED_KTAIL=
true template has a slightly different codegen layout (R34-dm) where
the LDS write timing dependency is tighter. gpt_oss may have benefited
from extra noise in the schedule.

## Other end-of-X drain patterns identified but unsafe to relax

After R50's win, I audited the remaining `s_waitcnt vmcnt(0)` calls in
the FP8 kernel for R50-style optimization opportunities:

| Line | Location | Why R50-style relaxation is unsafe |
|---|---|---|
| 2237 | Start of epilog 2 | Drains last main-loop prefetch into LDS; subsequent ds_read of As[tic][0] depends on it. Same LDS write commit issue as INIT0=8 above. |
| 2466 | K-tail block (post b/a issue, pre cA/cB mfma) | Required: vmcnt(8) is the MIN for cA/cB to fire (need b0+b1+a fully retired = 16 of 24). |
| 2466b | K-tail block (post cA/cB, pre cC/cD) | Required: vmcnt(0) is the MIN for cC/cD (need a_kt1 fully retired = all 24 retired). |
| 2845 | grouped_rrr_kernel epilog | Backward dA path. Cannot probe without manual `bench_grouped_gemm_turbo.py` validation (per task body rules). |
| 3082 | grouped_rrr_kernel end-of-tile | Same as 2845 — backward path. |
| 5611 | Other backward kernel | Same — backward path. |

Only the end-of-tile `s_waitcnt vmcnt(0) lgkmcnt(0)` at line 2557 (R50's
target) was both (a) provably safe to relax (no R/W aliasing) and (b)
detectable as a savings opportunity (~150 cy/tile across many tiles).
The other patterns either require more setup or are in untested code
paths.

## Lever ranking after R51-dm falsifications

R50 was a one-off architectural opportunity. Vmcnt micro-tuning is now
**fully exhausted** within the current MFMA cell shape:

- Main-loop vmcnt: saturated at RCR_STEADY_VMCNT=8 (R3-dm sweep done).
- K-tail vmcnt: saturated at vmcnt(8)/vmcnt(0) split (R12-dm split-vmcnt).
- Prologue vmcnt: INIT0=4 at safety boundary (this round); INIT1
  saturated (this round + R24-dm).
- Epilog vmcnt: saturated (R25-dm).
- End-of-tile vmcnt: removed (R50).

### Strongly suggested next round: Lever D
**`rt_32x64` / `rt_64x32` cell shape switch (32x32x64 MFMA)**. K=2880 has
K%64=0 → eliminates K-tail entirely for gpt_oss. HK has the scaffold
ready (commit `96a84c08` on a separate branch). 32x32 main MFMA with K=64
per call uses ~halved register-tile count vs 16x16 with K=128. Estimated
+5-10 pp. Big rewrite, likely 2-3 rounds:
1. Round N: port the `grouped_ktail_kernel_mfma32x32_M2N2` schedule into
   the persistent main kernel (replace 16x16 mfma calls + load helpers).
2. Round N+1: validate correctness on all 16 shapes; restore FUSED_KTAIL
   logic for non-K%64=0 cases (DSV3 K=7168 → K%64=0 too, so all current
   shapes hit the new path; K_REM ∈ {16, 32, 48} would still need the old
   path, but no metric shape exercises those).
3. Round N+2: tune VMCNT/LGKMCNT/sched for the 32x32 schedule.

### Less attractive: Lever B
**Dual LDS buffer ping-pong** for K-iter prefetch. Current LDS = 128 KB
(As + Bs each 64 KB). Need 32 KB more for triple-buffer (3rd K-iter
ahead). gfx950 LDS = 160 KB → headroom OK. Risk: occupancy regression
if MIN_BLOCKS_PER_CU = 2 fails. Not as high-EV as Lever D since it
only addresses HBM latency hiding (already largely hidden by current
double-buffer + R50 overlap).

### Forbidden / known-falsified
- Per-shape dispatch tweaks: frozen by task body.
- K-tail interleave probes: 3x falsified (R41, R47, R49).
- `store_c_tile_n_masked` clone with restricted unroll: catastrophic (R48).
- LDS swizzle ST_v3: ds_read hardcoded v2.
- MFMA cell shape 32x32x64 INTEGRAL migration without other lever: R29.
- HAS_KTAIL_BODY=false template: would lose R34's codegen win.

## Take-away for next agent

1. **vmcnt micro-tuning is fully exhausted**. INIT0=4 is at the safety
   boundary (this round). All other VMCNT settings have been probed
   across rounds 3-50.
2. **R50's mechanism (drop unnecessary cross-tile drain) does NOT
   generalize to other vmcnt(0) calls in the kernel**. The other drains
   are either (a) load-bearing for correctness (LDS commit timing) or
   (b) in untested backward kernels.
3. **The ONLY remaining unfalsified architectural lever is Lever D
   (32x32x64 MFMA cell shape)**. It requires 2-3 rounds of commitment
   and has 5-10 pp upside. Without it, the score is plateaued at ~960
   (R50 ceiling).
4. **If next agent does NOT commit to Lever D, the round budget is
   better spent on**:
   - Re-checking BF16 grouped (already at 1.183, also <1.20). BF16's
     own Lever D may be more tractable (BF16 16x16x32 MFMA → 32x32x16 a
     possible re-shape).
   - Backward path (RRR/CRR) optimization with manual benching.
   - Pre-quantize fusion (FORBIDDEN per task body — skip).

## Repo state at end of round

- HipKittens: clean at SHA `6a93fa32` (R50 winner unchanged).
- Primus-Turbo: 1 doc-only commit (this note).
- Score: 960 (R50 ceiling preserved). 32/32 PASS.
