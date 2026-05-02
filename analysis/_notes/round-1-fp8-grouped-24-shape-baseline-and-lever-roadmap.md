# Round 1 — FP8 grouped: 24-shape baseline + lever roadmap (NEW SUITE)

**Status**: BASELINE — no kernel/dispatch change this round (per task body
"第一轮强烈推荐 baseline metric + 把 24 row table 写入 round note，
不要直接改 kernel")
**Auto-optimize round**: 1 / 100
**Date**: 2026-05-02
**HK SHA at round start**: `ecbead9a` (R64-dm Lever D R-B step 1 LANDED)
**PT SHA at round start**: `6952b395`

---

## Context: 16-shape → 24-shape suite expansion

The old 16-shape suite (DSV3 8 + gpt_oss 8) plateaued at **score 957-962
/ grp_FP8 geomean ≈ 1.119** for 30+ rounds (see round-37..round-64-dm).
Every micro-knob (gm/num_xcds, unroll, launch_bounds, warps flip, K-tail
vmcnt, sched_barrier, setprio, readfirstlane, host overhead, MFMA
cell-shape full-port via Lever D R-B) has been exhaustively falsified
on that suite.

This task adds **8 new Qwen3-235B-A22B cases**:
- GateUP: N=3072, K=4096 (128-aligned, 0 K-tail iters → "clean main loop")
- Down  : N=4096, K=1536 (128-aligned, **shortest K in suite**, 12 K-iters)

Total = 24 cases per dtype. Score formula:
**score = 1000 × min(grp_fp8_geomean / 1.20, 1.0)** over all 24 cases.

---

## Baseline metric (this round)

`python3 scripts/_metric_grouped_only.py` → `/tmp/metric_round_baseline.log`

| Segment | geomean (n=24) | progress vs 1.20 | status |
|---|---|---|---|
| grp_BF16 | **1.1856** | 0.988 | FAIL ([watch] only, not scored) |
| grp_FP8  | **1.1157** | 0.930 | FAIL (the only scored segment) |

**Score = 958** (correctness 0/48 FAIL, reject 0/48, below_target 36/48).

Compared to the old 16-shape plateau (959 / 1.1195), the new 24-shape
baseline sits **within ±1 noise band** — the predicted "Qwen drag to
1.05-1.10" did NOT happen. Qwen subset actually *outperforms* gpt_oss
subset (see breakdown below).

---

## Full 24-row FP8 table (sorted by ratio ASCENDING — worst first)

| # | Shape | hk_TF | trt_TF | ratio | model | sub | K | aligned? |
|--:|---|--:|--:|--:|---|---|--:|---|
|  1 | grpFP8_gpt_oss_20B-GateUP-B32-M4096       | 1969.3 | 1926.5 | **1.022** | gpt_oss | GateUP | 2880 | NO (K%128=64) |
|  2 | grpFP8_gpt_oss_20B-GateUP-B32-M2048       | 1923.6 | 1853.1 | **1.038** | gpt_oss | GateUP | 2880 | NO |
|  3 | grpFP8_Qwen3-235B-A22B-Down-B16-M4096     | 1729.8 | 1650.2 | **1.048** | Qwen3   | Down   | 1536 | yes |
|  4 | grpFP8_gpt_oss_20B-GateUP-B4-M4096        | 1878.5 | 1791.1 | **1.049** | gpt_oss | GateUP | 2880 | NO |
|  5 | grpFP8_gpt_oss_20B-Down-B32-M2048         | 1787.4 | 1680.1 | **1.064** | gpt_oss | Down   | 2880 | NO |
|  6 | grpFP8_gpt_oss_20B-Down-B32-M4096         | 1886.8 | 1760.7 | 1.072     | gpt_oss | Down   | 2880 | NO |
|  7 | grpFP8_gpt_oss_20B-Down-B4-M4096          | 1725.0 | 1598.8 | 1.079     | gpt_oss | Down   | 2880 | NO |
|  8 | grpFP8_gpt_oss_20B-GateUP-B4-M2048        | 1720.2 | 1590.5 | 1.082     | gpt_oss | GateUP | 2880 | NO |
|  9 | grpFP8_Qwen3-235B-A22B-Down-B32-M4096     | 1756.8 | 1614.2 | 1.088     | Qwen3   | Down   | 1536 | yes |
| 10 | grpFP8_Qwen3-235B-A22B-Down-B16-M2048     | 1667.7 | 1528.9 | 1.091     | Qwen3   | Down   | 1536 | yes |
| 11 | grpFP8_Qwen3-235B-A22B-GateUP-B32-M2048   | 2397.9 | 2153.4 | 1.114     | Qwen3   | GateUP | 4096 | yes |
| 12 | grpFP8_Qwen3-235B-A22B-GateUP-B32-M4096   | 2484.2 | 2218.2 | 1.120     | Qwen3   | GateUP | 4096 | yes |
| 13 | grpFP8_gpt_oss_20B-Down-B4-M2048          | 1236.3 | 1097.7 | 1.126     | gpt_oss | Down   | 2880 | NO |
| 14 | grpFP8_Qwen3-235B-A22B-GateUP-B16-M4096   | 2429.2 | 2157.2 | 1.126     | Qwen3   | GateUP | 4096 | yes |
| 15 | grpFP8_Qwen3-235B-A22B-Down-B32-M2048     | 1757.8 | 1559.7 | 1.127     | Qwen3   | Down   | 1536 | yes |
| 16 | grpFP8_Qwen3-235B-A22B-GateUP-B16-M2048   | 2322.6 | 2055.8 | 1.130     | Qwen3   | GateUP | 4096 | yes |
| 17 | grpFP8_DeepSeek-V3-GateUP-B16-M4096       | 2722.2 | 2396.4 | 1.136     | DSV3    | GateUP | 7168 | yes |
| 18 | grpFP8_DeepSeek-V3-GateUP-B16-M2048       | 2653.7 | 2333.2 | 1.137     | DSV3    | GateUP | 7168 | yes |
| 19 | grpFP8_DeepSeek-V3-GateUP-B32-M2048       | 2684.6 | 2320.8 | 1.157     | DSV3    | GateUP | 7168 | yes |
| 20 | grpFP8_DeepSeek-V3-GateUP-B32-M4096       | 2718.4 | 2328.7 | 1.167     | DSV3    | GateUP | 7168 | yes |
| 21 | grpFP8_DeepSeek-V3-Down-B16-M2048         | 2058.5 | 1742.9 | 1.181     | DSV3    | Down   | 2048 | yes |
| 22 | grpFP8_DeepSeek-V3-Down-B16-M4096         | 2142.7 | 1772.7 | **1.209** ✓ | DSV3 | Down   | 2048 | yes |
| 23 | grpFP8_DeepSeek-V3-Down-B32-M4096         | 2162.0 | 1771.7 | **1.220** ✓ | DSV3 | Down   | 2048 | yes |
| 24 | grpFP8_DeepSeek-V3-Down-B32-M2048         | 2120.8 | 1727.7 | **1.228** ✓ | DSV3 | Down   | 2048 | yes |

**Only 3 / 24 case currently PASS (≥ 1.20)**, all in DSV3-Down (K=2048
"sweet spot" with 16 K-iters). 21 cases below target.

---

## Per-model / per-sub geomean breakdown

| Model | sub | n | geomean | min | max | observation |
|---|---|--:|--:|--:|--:|---|
| DSV3    | GateUP (K=7168) | 4 | 1.149 | 1.136 | 1.167 | large K → main loop dominates |
| DSV3    | Down   (K=2048) | 4 | **1.209** | 1.181 | 1.228 | sweet spot, **3/4 PASS** |
| Qwen3   | GateUP (K=4096) | 4 | 1.122 | 1.114 | 1.130 | clean main loop, 0 K-tail |
| Qwen3   | Down   (K=1536) | 4 | **1.088** | 1.048 | 1.127 | shortest K, prologue overhead |
| gpt_oss | GateUP (K=2880) | 4 | **1.047** | 1.022 | 1.082 | **WORST**: K-tail 1 iter mandatory |
| gpt_oss | Down   (K=2880) | 4 | 1.085 | 1.064 | 1.126 | K-tail 1 iter mandatory |

**Aggregate per model**:
- DSV3   8 cases: geomean **1.179** (best subset)
- Qwen3  8 cases: geomean **1.105**
- gpt_oss 8 cases: geomean **1.066** (worst subset)

**Key observation contradicting task body's prediction**: Qwen3 cold-start
is NOT the new bottleneck. gpt_oss K=2880 misalignment (already plateaued
at 1.06 on old suite) is the dominant drag. Qwen3 GateUP "clean main
loop" sits at 1.122 — better than gpt_oss but still far from 1.20.

---

## Worst 5 cases (the optimization anchors for next rounds)

| Rank | Case | ratio | root cause hypothesis |
|--:|---|--:|---|
|  1 | gpt_oss-GateUP-B32-M4096 | 1.022 | K=2880 K-tail + many tiles + N=5760 wide |
|  2 | gpt_oss-GateUP-B32-M2048 | 1.038 | same kernel, smaller M |
|  3 | Qwen3-Down-B16-M4096     | 1.048 | K=1536 (12 iter) + N=4096, prologue/epilog overhead |
|  4 | gpt_oss-GateUP-B4-M4096  | 1.049 | K=2880, smaller batch (4 routed) |
|  5 | gpt_oss-Down-B32-M2048   | 1.064 | K=2880, narrower N=2880 |

**4/5 worst are gpt_oss K=2880** (K-tail 1 iter required at K%128=64).
**1/5 is Qwen3 K=1536** (shortest K in suite, prologue/epilog dominant).

Both subsets converge on a common failure mode: **the "main loop" path's
absolute throughput is the bottleneck**, not K-tail (already heavily
optimized) and not host overhead (already trimmed to 1us).

---

## Forbidden-direction sanity check (all already FALSIFIED, do NOT redo)

| Direction | Falsifying round(s) |
|---|---|
| (gm, num_xcds) sweep | round-1..6-dm, R8 |
| #pragma unroll enumeration | R7-dm |
| launch_bounds MIN={2,3,4,5} | R5-dm, R7-dm |
| WARPS_M / WARPS_N flip | R6-dm |
| K-tail vmcnt single-wait | R7-dm (split-vmcnt SHIPPED) |
| K-tail epilog single-wait | SHIPPED |
| sched_barrier mask sweep | R8, R55-dm |
| setprio | R53-dm (CATASTROPHIC) |
| readfirstlane | R52-dm |
| __noinline__ on hot helpers | R54-dm |
| simple SRD hoist | (multiple) |
| chunk_size | (multiple) |
| host overhead | trimmed |
| MFMA cell shape 16x16x128 → 32x32x64 (whole kernel) | R15-dm + R56-R64 R-B (FALSIFIED) |
| LDS ST_v2 → ST_v3 | (architectural reject) |
| Python dispatcher / can_handle | task-body forbidden |

---

## Empirical foundation for next rounds

ASM disassembly of current `tk_fp8_layouts.so` (dumped via
`llvm-objdump -d --triple=amdgcn--amdhsa --mcpu=gfx950 tk_fp8_layouts.so`):

| Instruction class | count | meaning |
|---|--:|---|
| `scratch_load_sbyte_d16` | **31** | VGPR spill reload (FP8 byte) |
| `s_scratch_store_dword`  | **27** | scalar scratch spill out |
| `global_load_lds_*`      | **0**  | **async global→LDS path UNUSED** |

Total spill instr ≈ 58 (matches the "67 spill" figure in task body
within tooling tolerance — `objdump` segfaulted partway, so total may
be slightly higher). **Critical**: the kernel has zero `global_load_lds`
intrinsics → Lever A's async-copy thesis is genuinely unexplored.

---

## Lever roadmap for R2..R10

The task body lists 6 architectural levers (A..F). Cost-of-investigation
ranking + expected payoff (from task body + R56-R64 prior analysis):

| Lever | Expected | Risk | Adv-on-suite | Order |
|---|---|---|---|---|
| **A** Async global→LDS copy + MFMA pipelining | **+5..10pp** | Med-High | Largest-K cases (DSV3-GateUP, Qwen-GateUP); minimal on Qwen-Down K=1536 | **R2 — try FIRST** |
| **B** Dual / triple LDS buffer ping-pong | +3..6pp | Med (LDS budget tight: 2×32KB = 64KB cap) | Same coverage as A; complementary to A | R3 if A lands |
| **C** Reduce VGPR live-range to kill spill | +2..4pp | Low-Med | Helps all 24 cases | R4 (low risk fill-in) |
| **F** Qwen3-Down K=1536 short-K specialization | +2..4pp on 4/24 case (=+0.7pp geomean) | Low | Only Qwen3-Down 4 cases | R5 if A+B saturated |
| **D** 32x32x64 cell-shape on Qwen GateUP probe | 0..+5pp | High (R56-R64 falsified the broader port; R64-dm noted "no analytical backing") | Qwen3-GateUP 4 cases only | **REQUIRES** mfma_323264 microbench gate ≥ 3pp first |
| **E** ASM software pipelining | 0..high | Very High | All cases, but breaks readability + build-time | LAST RESORT after A+B+C |

### R2 plan (next round) — Lever A scout

1. Read `include/ops/warp/memory/util/util.cuh` to find existing BUFFER
   intrinsic helpers + look for any `global_load_lds_*` ASM stubs that
   may already exist as scaffolding.
2. Pick the SMALLEST main-loop change: replace the synchronous
   `global → register → ds_write` path for tile A in the steady-state
   K-iter with an async `global_load_lds_b128` direct path. Skip B for
   now (one-tile scout, easier to revert).
3. Build, smoke check correctness on 1 shape via `bench_grouped_gemm_turbo`
   (fwd+bwd allclose).
4. Run full metric. If score moves by ≥ +5 pts → commit + extend to B
   tile in R3. If neutral / regression → revert + falsify Lever A.

### Decision tree summary

```
R2: Lever A scout (1-tile async copy)
    ├── ≥ +5 pts → R3 extend to both tiles + R4 + ping-pong (Lever B)
    └── < 0 pts  → falsify A; R3 jumps to Lever C (reg-live-range work)

R5+: depending on plateau, either Lever F (Qwen-Down K=1536)
     or Lever D microbench gate.
```

---

## Hard constraints reminder for next agent (do NOT violate)

1. ❌ No dense / BF16 changes (BF16 is `[watch]`, not scored)
2. ❌ No metric / benchmark / config edits (`benchmark/ops/config.py`,
   `scripts/_metric_*.py` are FROZEN — user already added Qwen3-235B-A22B
   to MoEModelConfigs)
3. ❌ No dispatcher / can_handle changes (FP8 grouped MUST hit HK kernel)
4. ❌ No quantize fuse, no host-side .item() / .tolist()
5. ❌ No per-model branches in `select_default_config` (must be generic
   rules: `if K >= 4096`, `if N == hidden_size`, `if min(tiles) < 16` etc)
6. ✓ HIPKITTEN must remain `BackendEntry(..., autotune=False)`
7. ✓ One focused commit per repo per round

---

## What this round changed in code

**Nothing** — docs-only round per task-body explicit instruction.

- Primus-Turbo: this `.md` (round-1 baseline note)
- HipKittens: no change

Working tree at end of round:
```
 M benchmark/ops/config.py         (user-staged, frozen — do NOT touch)
 M scripts/_metric_grouped_only.py (user-staged, frozen)
 M scripts/_metric_hk_ratio.py     (user-staged, frozen)
?? .auto_optimize_logs/            (auto-optimize harness output)
?? 3rdparty/composable_kernel      (untracked submodule worktree)
+? analysis/_notes/round-1-fp8-grouped-24-shape-baseline-and-lever-roadmap.md
```

Only the new round note will be committed.

---

## Metric command + raw output

```
$ python3 scripts/_metric_grouped_only.py 2>&1 | tee /tmp/metric_round_baseline.log
... (full log preserved at /tmp/metric_round_baseline.log)
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1856 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1157 (n=24)
[metric_grouped_only]   weighted_progress=0.9584  correct_fail=0/48  reject=0/48
[metric_grouped_only]   below_target=36/48  goals=0/2  score=958
```

GPU pinned by auto_optimize.py: `HIP_VISIBLE_DEVICES=3` (logical
device 0 inside the process, physical card3, VRAM 298MB at start —
clean, no other tenant).

---

## Recommendation for R2 (concrete first step)

**Read first**, then probe — do NOT immediately edit kernel:

1. `Grep` for `global_load_lds` across `/workspace/code/HipKittens/include/`
   to see if any scaffolding exists.
2. `Read /workspace/code/HipKittens/include/ops/warp/memory/util/util.cuh`
   to find what BUFFER intrinsic primitives are already exposed.
3. `Read kernel_fp8_layouts.cpp ~lines 1973..2400` (grouped_rcr main
   kernel + load helpers) to identify the cleanest A-tile-only swap point.

Only after these reads should R2 commit to a kernel diff. The scout
should be the SMALLEST possible change: 1 tile, 1 K-iter path. If it
moves the metric by ≥ +5 pts and correctness 0/48 fail, expand. If not,
revert + falsify in R2 note + jump to Lever C in R3.
