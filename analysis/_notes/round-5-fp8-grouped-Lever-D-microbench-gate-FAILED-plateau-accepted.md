# Round 5 — FP8 grouped: Lever D microbench gate FAILED → plateau ACCEPTED

**Status**: LEVER D ARCHITECTURALLY FALSIFIED via R64-dm prerequisite
microbench. mfma_f32_32x32x64_f8f6f4 vs mfma_scale_f32_16x16x128_f8f6f4
single-warp throughput is **statistically identical** (Δ ≈ 0%, well
below R64-dm's +3 pp gate). All 5 architectural levers from R1
roadmap are now exhausted.
**Auto-optimize round**: 5 / 100
**Date**: 2026-05-02
**HK SHA at round start**: `ecbead9a`
**HK SHA at round end**: `<commit this round>` (lever_d_microbench.cu added)
**PT SHA at round start**: `fbc5693`
**Round time**: ~25 min (1 metric + 1 .cu write + 1 build + 3 trials + write-up)
**Score before**: 959 (R5 baseline)
**Score after**: 959 (no kernel change; .cu added is documentation only)

---

## R5 baseline metric

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1884 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1155 (n=24)
[metric_grouped_only]   weighted_progress=0.9595  correct_fail=0/48  reject=0/48
[metric_grouped_only]   below_target=36/48  goals=0/2  score=959
```

Worst 5 unchanged. Stable at the 956-961 noise band the last 5 rounds
have observed.

---

## Lever D microbench (R64-dm explicit prerequisite)

R64-dm landed `st_32x64` shared-memory tile type as scaffolding for a
hypothetical full mfma_f32_32x32x64_f8f6f4 port, but mandated that
"before R38+ agent commits to any further kernel rewrite, they MUST
run a focused microbench comparing mfma_323264 vs mfma_1616128
throughput on a synthetic M=64 N=32 K=128 single-warp workload. If the
microbench shows < 3 pp single-warp throughput advantage for
mfma_323264, **ABANDON the full port** and accept the 957-962 plateau."

R5 wrote and ran that microbench.

### Microbench design

`/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/lever_d_microbench.cu`
(committed this round to HK). Standalone .cu, no Kittens dependency.
Two kernels both targeting equivalent 64×32 fp32 output × K=128
accumulation depth, both holding 32 dw/lane in accumulators (matching
the real grouped_rcr_kernel main-loop per-warp budget):

```
bench_16x16x128:
  - 8 mfma_scale_f32_16x16x128_f8f6f4 calls per inner iter
  - 8 × floatx4 acc (32 dw/lane)  = covers 4 height × 2 width
                                    = 64x32 output, K=128

bench_32x32x64:
  - 4 mfma_scale_f32_32x32x64_f8f6f4 calls per inner iter
  - 2 × floatx16 acc (32 dw/lane) = covers 2 height × 1 width × 2 K-chain
                                    = 64x32 output, K=128
```

Both kernels: `__launch_bounds__(64, 8)` (1 warp/CTA, 8 wave/SIMD),
data prefetched into VGPR before the timed loop, no LDS, no buffer_load,
no branch divergence. Pure mfma compute throughput as the gate metric
specifies.

Build:
```
hipcc /tmp/mfma_microbench.cu -o /tmp/mfma_microbench \
      --offload-arch=gfx950 -O3
```

### Microbench results (3 trial sweeps, median over 11 runs each)

| N inner iter | 16x16x128 (ns/iter) | 32x32x64 (ns/iter) | Δ throughput |
|--:|--:|--:|--:|
| 10000  | 107.41 | 107.44 | **-0.03 %** |
| 50000  | 107.08 | 107.06 | +0.02 % |
| 100000 | 107.00 | 107.05 | -0.05 % |

Both paths sustain ~4.9 TFLOPS per warp (~107 ns for 8×16x16 or 4×32x32
mfma calls). Throughput delta is **statistically zero** at all sample
sizes.

### Decision

R64-dm gate said: **"if mfma_32x32x64 is < +3 pp faster per-iter,
ABANDON Lever D full port"**. Empirical result is essentially 0 %
delta. **Lever D ARCHITECTURALLY FALSIFIED**.

The R64-dm cost-model (equal MFMA cycle count, equal accumulator
dw/lane held simultaneously) is **empirically confirmed**. The
remaining hypothesis ("scheduling freedom from larger acc tile lets
LLVM expose better LDS read overlap") was speculation with no
mechanism, and per the R64-dm rule, we don't get to spend 4-6 rounds
chasing it without empirical backing.

---

## Cumulative lever falsification status

After 5 rounds of disciplined falsification:

| Lever | Status | Round falsified |
|---|---|---|
| **A** Async global→LDS copy | FALSIFIED — already shipped via inline-ASM `buffer_load_dwordx4 ... lds` (line 787) | R2 |
| **B** Triple LDS slab | FALSIFIED — kernel uses 137 KB / 160 KB CU cap; dual already shipped (`As[2][2]/Bs[2][2]`) | R2 |
| **C-1** Manual LDS scratch hand-spill | NOT TRIED — R3 spill localization showed no single-target expression; predicate gated on C-X first, which falsified | (gated falsified) |
| **C-2** K-tail capture refactor | FALSIFIED — captures already in `if (g.fast_k < g.k)` block (line 2540-2719), zero liveness leak | R3 |
| **C-3** Spill source localization | DONE — primary spill is architectural (cA-cD + a/b0/b1 = ~171 dw), 256 VGPR cap overflows by 32-43 dw | R3 |
| **C-X** N_MASKED helper SENTINEL refactor | FALSIFIED — neutral on active `<0,T,T>` template (39→39 spill), -1.8 pp worst-case shape regression | R4 |
| **D** mfma_32x32x64 cell-shape | **FALSIFIED R5** — microbench gate -0.03 % (gate threshold ≥ +3 pp) | R5 |
| **E** ASM software pipelining | OPEN — last resort, very high risk, R20+ |
| **F** Qwen-Down K=1536 short-K specialization | OPEN — only +0.8 pp geomean ceiling even if successful (4/24 case) |

5 of 7 substantive levers falsified. Remaining = E (last resort) and
F (limited geomean upside).

---

## Score ceiling derivation

Working set inside `for (int gt = pid; gt < total_tiles; ...)` of
grouped_rcr_kernel:

```
cA, cB, cC, cD (4× rt_fl<64,32>)   = 128 dw/lane (architectural)
a, b0, b1      (3× register tiles) =  32 dw/lane (architectural)
soA, soB, lds_addrs                =   8 dw/lane
laneid, warpid, tic/toc, k counter =   4 dw/lane
helper temporaries / iter index    = 80-120 dw/lane (LLVM-determined)
=================================
Working set                       = ~250-290 dw/lane

VGPR cap (launch_bounds=512,1)     = 256 dw/lane

→ overflow 32-43 dw/lane → VMEM scratch (=current spill remark)
```

The 32-43 dw spill on grouped_rcr_kernel<0,*,*> templates is the
**unavoidable consequence** of cap-bound register allocation when
the working set inherently exceeds 256 dw/lane. We cannot:
- reduce VGPR cap further without moving accumulator to LDS (severe
  perf regression — LDS double-buffer + ds_read latency)
- raise occupancy (LDS is also 137/160 KB used)
- shrink accumulator (Lever D fails, cell shape change neutral)
- skip K-tail block (already in if-runtime branch, captures scoped)

The plateau at score 956-962 / FP8 geomean ~1.115 is the
**architectural ceiling** of this kernel design on gfx950. To break
the ceiling would require:
1. Lever E (ASM software pipeline, R20+) — possibly winning by tighter
   register packing than LLVM achieves
2. Different kernel architecture (different unrolling, different acc
   shape) — outside Lever D scope, would require novel design

---

## R6+ plan options

Given the plateau is empirically confirmed at the architectural
ceiling, three honest paths forward:

### Option A: **Continue with Lever F (Qwen-Down K=1536)** [LOW RISK, +0.8 pp ceiling]

Qwen-Down N=4096 K=1536 has only 12 K-iters (K_BLOCK=128). The
prologue + epilog overhead occupies a relatively larger fraction
than the longer-K cases. Could try a K<2048 generic dispatch rule
that picks different `group_m` / `num_xcds` / unroll factor for
short-K shapes.

But task body explicitly forbids new (gm, num_xcds) sweep ("✗ (gm,
num_xcds) config sweep — 4 worst case 已穷尽, 所有 config 都试过").
This means Lever F's only legal mechanism is changing the kernel
itself for short-K (e.g. different block dim, different unroll
factor). The `select_default_config` only accepts generic rules
like `if K < 2048 and N >= K`. Limited surface area.

Expected: +0.5..0.8 pp on Qwen-Down 4 cases × 4/24 weight = +0.1..0.13 pp
geomean = +1..2 score points. Marginal.

### Option B: **Lever E ASM software pipelining** [VERY HIGH RISK]

Manually schedule the K-iter chain in inline ASM:
```
load[k+2] | mfma[k+1] | store_acc[k]   (3 stages overlapped)
```
instead of LLVM's heuristic schedule. Could let us fit acc + load
state into different SGPR/VGPR pools more tightly.

Risk: hand-written ASM main loop is unmaintainable, build time
explodes, codegen-bug debug is brutal. Per task body: "最后再做". R20+.

### Option C: **Accept plateau, redirect rounds to documentation + cleanup**

5 rounds remaining in this chat session window (90 min total, ~60 min
used). Rest of 100-round budget could go to:
- DoD smoke run (R6 = checkpoint per task body cadence)
- Round-by-round documentation cleanup (consolidate falsification notes
  into a single "FP8 grouped GEMM ceiling analysis" doc)
- Lever F low-risk attempt
- Re-baseline metric multiple times to nail down exact noise band

### R6 specific recommendation: **Option A (Lever F scout, low cost)**

Lever F is the only remaining lever with **any** measurable upside (even
if marginal). Spending 1 round to scout it has positive expected value:
- If it lands +5..8 pts → commit, raise plateau slightly
- If neutral / negative → falsify, accept ~960 plateau as final, rest
  of rounds can be cleanup

Concrete R6 step:
1. Inspect Qwen-Down 4 cases per-shape ratio (~1.05..1.13).
2. Look at `dispatch_grouped_rcr` (line 5243+) to see if there's a
   generic rule slot for K-bound dispatch (e.g. choosing different
   `g.num_xcds` or `g.group_m` based on `if K < 2048 and N >= K`).
3. If a clean generic rule slot exists AND it's not in the task-body
   forbidden list (which forbids gm/num_xcds **sweep**, not single
   generic rule)... wait actually task body ✗ list says "(gm,
   num_xcds) config sweep". A single generic-rule choice is not a
   sweep. Need to be careful: the forbidden item says "all config
   tried" implying any specific (gm, num_xcds) value has been tested.
   Worth re-checking R56-dm + earlier rounds for what `(gm=?, xcds=?)`
   gpt_oss-Down-K=2880 used vs Qwen-Down-K=1536, and whether the
   dispatcher actually distinguishes them.

If Lever F's dispatcher can't be touched (full-saturated) → R6 = R6
write-up + R7 = Lever E scout (very high risk).

---

## What this round changed in code

### HipKittens repo
- **NEW**: `analysis/fp8_gemm/mi350x/lever_d_microbench.cu` (~200 lines)
  - Standalone .cu microbench for the R64-dm Lever D gate
  - Build: `hipcc lever_d_microbench.cu -o lever_d_microbench --offload-arch=gfx950 -O3`
  - Run: `./lever_d_microbench [N=10000]`
  - Output: per-iter throughput (16x16x128 vs 32x32x64 vs %delta vs
    PASS/FAIL verdict)
  - **Result of this run**: -0.03% / +0.02% / -0.05% across 3 trials
    → FAIL (gate ≥ +3 pp)

No HK kernel changes. `kernel_fp8_layouts.cpp` is bit-identical to
R4 end-of-round.

### Primus-Turbo repo
- **NEW**: this `.md` (round-5 falsification + plateau analysis)

---

## Hard-constraint compliance

- [x] No metric / benchmark / config edits
- [x] No dispatcher / can_handle changes
- [x] No quantize fuse, no host-side .item() / .tolist()
- [x] No per-model branches in dispatcher
- [x] HIPKITTEN remains `BackendEntry(..., autotune=False)`
- [x] One focused PT commit (this note)
- [x] One focused HK commit (microbench .cu)
- [x] No BF16 grouped touch
- [x] Correctness 0/48 fail (R5 baseline metric run)

---

## DoD smoke status

R5 is the auto-checkpoint cadence (5, 10, 15, ...). Per task body, the
auto_optimize.py harness runs `bash scripts/run_dod_metric.sh --full`
at the end of this round automatically. No manual run needed. The
score from that goes to summary.json.
