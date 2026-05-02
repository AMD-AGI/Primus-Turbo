# Round 33 — FP8 grouped: cluster analysis + explicit STOP/PIVOT signal

## Summary

R33 did NOT attempt a kernel-level probe. Instead, this round documents
the cluster-level breakdown of the 24-shape FP8 grouped score and
makes an explicit recommendation for the user/orchestrator: **the
"micro-spill / anti-CSE / loop-body shuffle" lever class is exhausted;
remaining levers (A async-LDS, D-Qwen 32x32x64) require multi-round
commitment with non-trivial code rewrites and uncertain payoff**. We
have spent 5 rounds (R28→R32) inside the saturation band and the pattern
is now decisive.

## R33 metric snapshot

```text
Baseline (this round, single trial): 980
Goals:
  (1) grp_BF16  vs triton  >= 1.20  : 1.1873  progress=0.989  FAIL
  (2) grp_FP8   vs triton  >= 1.20  : 1.1647  progress=0.971  FAIL
```

24-shape FP8 ratios sorted ascending (key data, NOT picked by intuition):

```text
1.083 grpFP8_gpt_oss_20B-GateUP-B32-M4096    [WORST]
1.093 grpFP8_gpt_oss_20B-Down-B32-M2048
1.099 grpFP8_gpt_oss_20B-Down-B32-M4096
1.108 grpFP8_gpt_oss_20B-GateUP-B32-M2048
1.115 grpFP8_gpt_oss_20B-GateUP-B4-M4096
1.120 grpFP8_gpt_oss_20B-Down-B4-M4096
1.127 grpFP8_gpt_oss_20B-GateUP-B4-M2048
1.141 grpFP8_Qwen3-235B-A22B-Down-B16-M4096
1.144 grpFP8_Qwen3-235B-A22B-GateUP-B16-M4096
1.147 grpFP8_Qwen3-235B-A22B-Down-B16-M2048
1.152 grpFP8_DeepSeek-V3-GateUP-B16-M2048
1.161 grpFP8_Qwen3-235B-A22B-Down-B32-M4096
1.167 grpFP8_Qwen3-235B-A22B-GateUP-B32-M2048
1.175 grpFP8_Qwen3-235B-A22B-GateUP-B32-M4096
1.177 grpFP8_DeepSeek-V3-GateUP-B32-M2048
1.179 grpFP8_DeepSeek-V3-GateUP-B16-M4096
1.185 grpFP8_Qwen3-235B-A22B-Down-B32-M2048
1.189 grpFP8_DeepSeek-V3-GateUP-B32-M4096
1.200 grpFP8_Qwen3-235B-A22B-GateUP-B16-M2048    [target]
1.215 grpFP8_gpt_oss_20B-Down-B4-M2048
1.221 grpFP8_DeepSeek-V3-Down-B32-M2048           [PASS]
1.243 grpFP8_DeepSeek-V3-Down-B16-M2048           [PASS]
1.262 grpFP8_DeepSeek-V3-Down-B16-M4096           [PASS]
1.276 grpFP8_DeepSeek-V3-Down-B32-M4096           [PASS]
```

## Cluster analysis

| Cluster | Cases | Median ratio | Template |
|---|---|---|---|
| gpt_oss | 8 | ~1.10 (worst, K_REM=64 K-tail) | mix `<0,F,T>` + `<0,T,T>` |
| Qwen3 | 8 | ~1.17 (mixed, K_REM=0) | `<0,F,T>` |
| DSV3-GateUP | 4 | ~1.18 (uniform, K_REM=0) | `<0,F,T>` |
| DSV3-Down | 4 | ~1.25 (PASSING) | `<0,F,T>` |

Key observations:

1. **gpt_oss is the dominant drag** (7/8 below 1.15). It's the only
   cluster with K_REM=64 → exercises the FUSED_KTAIL block (24
   buffer_loads + 2 vmcnt + 4 mfma per output tile).

2. **R30-R32 work on `<0,T,T>` template only addressed gpt_oss-Down
   (N=2880 masked) — half of the gpt_oss cases**. gpt_oss-GateUP
   (N=5760 aligned) takes `<0,F,T>` template, same as DSV3+Qwen, and
   was NOT touched by the anti-CSE work.

3. **Qwen + DSV3-GateUP are also below target (1.14-1.19)**, going
   through the K-aligned `<0,F,T>` template (34 dw spill, NO K-tail
   block). These shapes have NO FUSED_KTAIL overhead — their gap is
   structural to the main K-loop itself, not to the K-tail epilog.

4. **DSV3-Down is the only fully-passing cluster** — N=4096/7168 are
   power-of-2 aligned, K=2048 is moderate, and M_per_group is large
   (so amortization is good). This is the "easy" subset.

## Why the K-tail pre-issue idea was rejected (data, not intuition)

K-tail HBM pre-issue (move `load_b_kt(b0,0)` + `load_b_kt(b1,1)` from
the K-tail block into Epilog 2's tail) was the only remaining
"micro-optimization" lever I had not falsified. Estimated gains via
careful cycle accounting:

* Per-tile K-tail critical path: ~143 cy (24-cy issue + 71-cy vmcnt(8)
  wait + 32-cy cA/cB mfma + ~16 cy vmcnt(0) hidden + 32 cy cC/cD mfma)
* Pre-issue saves the `b0+b1` retirement window (~64 cy) by giving them
  a head start during Epilog 2's 64-cy mfma window
* Per-CU saving = 90 tiles × 64 cy = 5760 cy
* Per-CU total cycles ~= 1.68 ms × 2 GHz = 3.36 M cy
* % savings = 5760 / 3.36M = **0.17 %** → ~+0.2 pp on gpt_oss ratio →
  ~+0.07 pp on FP8 geomean → ~**0-1 pts metric** (within ±2 pts noise)

Additionally, this optimization has a **correctness risk**: DSV3+Qwen
also use FUSED_KTAIL=true template (R34-dm regalloc trick), and
unconditional pre-issue would add 64 KB/tile of wasted HBM traffic to
their K-aligned path. ~5.76 MB/WG × 256 WGs = 1.5 GB extra HBM traffic
→ +0.28 ms / 1.39 ms kernel time = ~20% slowdown on DSV3-GateUP. To
gate this off requires `if constexpr (FUSED_KTAIL) if (g.fast_k < g.k)
{ ... }` runtime branch inside Epilog 2, which disrupts the existing
tight scheduled barrier structure.

**Verdict**: Negative-EV. ~0-1 pt expected upside, ~5-10 pt downside if
correctness gate misfires. Not implemented.

## R28→R32 saturation chronology (the long view)

| Round | Lever | Result |
|---|---|---|
| R28 | C-1 saturation acknowledgment (no kernel change) | doc-only |
| R29 | C-1 SCOUT (LLVM optremarks → spill source localized to L2223 + L2313) | doc-only |
| R30 | anti-CSE asm-volatile on offsets | -42% scratch_load, -2 pts FALSIFIED |
| R31 | sched_barrier(0) at K-iter boundary | NO codegen change FALSIFIED |
| R32 | IR-level loop-iter dep (asm-volatile clobbered laneid) | -42% scratch_load (same as R30), -1 pt FALSIFIED |
| R33 | Cluster analysis + STOP signal | doc-only |

R30 + R31 + R32 collectively eliminated the entire "anti-CSE / anti-hoist
/ loop-body offset shuffle" lever class. Three independent attacks all
yielded zero metric improvement because:

* (a) the prologue spill_store cost (30 × 80 cy at function entry,
  once per work-group) is **unaffected** by any loop-body offset
  manipulation — it's bound to the fast-path's CSE-hoisted offsets in
  the kittens shared `store(...)` helper, and modifying that helper
  would affect all kernels (high blast radius).
* (b) the spill RELOAD cost is **L1-cached** after the first pass
  through the loop (~21 cy effective vs ~80 cy VMEM cost), so the -42%
  reload reduction observed in R30 + R32 only saves ~21 × (285-166) ≈
  2500 cy/tile = 0.07% of kernel time = within noise.

## Remaining levers (from original task brief), with re-cost estimates

### Lever A — async global→LDS (`global_load_lds_*` intrinsic)

* **Predicted gain**: +5-10 pp on FP8 geomean (per task brief)
* **Cost**: Major K-loop rewrite (3-5 round commitment minimum). Must
  also restructure the Epilog 1/2 + K-tail block + LDS double-buffer
  scheme. Correctness risk is HIGH (LDS write completion semantics
  vs vmcnt vs lgkmcnt are tricky).
* **EV**: Positive in expectation if the +5pp lower bound holds, but
  variance is high.
* **When to choose**: If user has 5+ rounds of patience for FP8
  grouped specifically and is willing to absorb potential correctness
  regressions during the rewrite.

### Lever B — dual/triple LDS ping-pong

* **Predicted gain**: +3-6 pp (per task brief)
* **Status**: STRUCTURALLY BLOCKED. Current LDS = 139796 bytes/WG.
  Doubling to triple-buffer = 200KB+, but gfx950 LDS cap is ~160 KB/CU
  for this kernel. Cannot fit. Requires reducing tile size (BLOCK_SIZE
  256→192 or 128), which changes the entire mfma layout and is a
  larger rewrite than Lever A.
* **EV**: Negative — infeasible without first shrinking tile.

### Lever D-Qwen — 32x32x64 mfma cell shape on Qwen GateUP

* **Predicted gain**: +3-5 pp on Qwen GateUP cluster (4/24 cases) ≈
  +0.5-1 pp on geomean
* **Cost**: Single-warp mfma_323264 vs mfma_1616128 microbench gate
  (1 round) → if ≥3 pp wins, port FP8 grouped kernel to use
  `rt_32x64_s` cells (2-3 rounds of careful porting + correctness
  validation).
* **EV**: Modest positive if microbench wins, but the original
  16-shape Lever D (R56-R64-dm) was extensively falsified. The
  "Qwen GateUP is different" hypothesis has not been verified.
* **When to choose**: As a cheap 1-round experiment to revive the
  lever class. R33 did NOT do this microbench because the rocprofv3
  setup hung (multipass collection of 10 counters timed out at 4+
  minutes). Could retry with fewer counters or a custom standalone
  microbench kernel.

### Lever F — Qwen-Down K=1536 specialized config

* **Predicted gain**: +0.5 pp on geomean (4 cases × +5pp upper bound /
  24 = 0.83 pp)
* **Cost**: 1 round to find a "K < 2048" generic dispatcher rule that
  doesn't violate the "no model-specific config" rule.
* **EV**: Marginal. Worth trying only if other levers all fail.

## Recommendation

**This is the explicit STOP/PIVOT decision point**. The saturation is
now confirmed across 5 rounds with conclusive ASM-level evidence. Three
paths forward:

1. **STOP and lock at 977-981.** Best-known is 981 (R27), current
   trajectory is 977-980. Document this as the natural ceiling under
   the current architectural constraints (single-launch persistent
   kernel + no quantize fuse + no dispatcher fallback + 256-VGPR /
   140KB-LDS / 1-WG-per-CU constraints). User can decide if 981 is
   acceptable as the FP8 grouped score.

2. **Pivot to Lever D-Qwen** (1-round microbench gate). Concrete
   action: write a standalone single-warp mfma microbench (32x32x64
   vs 16x16x128 with same A/B data), measure throughput. If 32x32x64
   wins by ≥3 pp → port the kernel (3-5 rounds). If not → falsify and
   PIVOT to option 3.

3. **Pivot to Lever A** (multi-round commitment). Major K-loop rewrite
   using `global_load_lds_*`. Predicted +5-10 pp, but high
   correctness risk. Should be done as a separate exploration branch
   that can be reverted if it hits a wall.

**My recommendation**: option 2 is the lowest-risk, highest-EV per
round of effort. Microbench is data-only and tells us in 1 round
whether the lever has any chance. R34 should be a microbench round.

## Files touched this round

* `/workspace/code/Primus-Turbo/analysis/_notes/round-33-fp8-grouped-cluster-analysis-and-explicit-stop-signal.md` (new)

## HipKittens

NO CHANGES this round. HK working tree is clean of probe code.

## Metric

Single trial baseline: 980. No probe attempted, no after-metric.
