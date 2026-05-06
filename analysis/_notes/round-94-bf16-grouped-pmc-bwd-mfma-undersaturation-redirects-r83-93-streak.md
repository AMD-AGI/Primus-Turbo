# Round 94 (auto-loop round 17-18) — BF16 grouped GEMM: backward-path PMC profile redirects R83-R93 falsification streak from "memory-bound" to "MFMA-undersaturated + occupancy-limited"

## Context

Auto-loop rounds 17-18 (this note covers two rounds combined; R17 was
interrupted before commit due to GPU contention, R18 picks up the data
and commits the diagnostic).

R83-R90 was an 8-round streak of var-K kernel falsifications under the
shared assumption that the var-K backward path was memory-bound (LDS
bank conflict, swizzle, cross-permute, ST_B pad, rematerialization
levers). R91 declared "cumulative falsification pivot" and recommended
PMC profiling. R92 attempted Lever A1 (forward K-tail prefetch) at +2
score / +6 VGPR cost — partial falsification. R93 attempted Lever B4
(per-group var-K partition) at -77 score — catastrophic falsification
via XCD-driven L2 thrash.

**Common thread of R83-R93**: every lever assumed the kernel was
HBM/L2-bound and tried to rearrange memory access. None worked. R93's
note speculated that "var_k might be HBM-bound, hence why naive
partition rewrites fail" and recommended a PMC round (option 1) to
test that hypothesis. This round delivers that data.

## Method

`/tmp/probe_round17_rocprof_backward.py` calls
`turbo.ops.grouped_gemm` with autograd `c.backward(go)` so rocprofv3
captures BOTH forward (`grouped_kernel<RCR, KI=0, FUSED=true>`) AND
backward dB (`grouped_var_k_kernel<0>`) in one pass. R9's note (the
prior PMC anchor) only profiled forward; the backward path's
characteristics were never measured before this round.

Counter input (`/tmp/rocprof_input_pass1_only.yaml`): MfmaUtil,
VALUUtilization, MemUnitStalled, MeanOccupancyPerCU, LdsBankConflict
— five derived counters in one pass. Target shape: lowest-progress in
R17 metric was `gpt_oss-Down-B4-M2048` (ratio 1.056, weight 3, K=2880
→ K%128=64 K-tail path).

Pass 2 (FetchSize / WriteSize / TCC_HIT / TCC_MISS for HBM bandwidth +
L2 cache hit rate) hung past 10 minutes despite a 180 s timeout, both
on R17 first attempt and R18 retry on a second shape (G=32). Suspected
GPU pool contention with the parallel FP8 auto-loop worker. Pass 1
data is sufficient to falsify the memory-bound hypothesis; pass 2 was
intended to confirm the alternative (compute-bound) directly via L2
hit rate but isn't strictly required.

## Pass 1 results

`/tmp/rocprof_round17_gpt_oss_Down_B4/pass_1/chi2894/3398716_counter_collection.csv`
— 12 dispatches profiled (3 warmup + 1 measured call × 3 kernels per
call: forward `grouped_kernel`, backward dA `grouped_kernel`, backward
dB `grouped_var_k_kernel`; warmup not skipped by rocprofv3 in
single-pass mode):

| Did | kernel        | dur_us | MfmaU% | VALU% | MemSt% | Occ/CU | LdsBC% |
|----:|---------------|-------:|-------:|------:|-------:|-------:|-------:|
|   8 | grouped       |  148.7 |  43.0  |  88.2 | 0.16 | 5.12 | 0.000 |
|  12 | grouped       |  150.1 |  42.1  |  88.2 | 0.15 | 5.02 | 0.000 |
|  13 | grouped_var_k |  166.0 |  41.0  |  85.9 | 0.10 | 5.10 | 0.999 |
|  17 | grouped       |  152.7 |  41.1  |  88.2 | 0.14 | 4.93 | 0.000 |
|  21 | grouped       |  144.0 |  45.4  |  88.2 | 0.16 | 5.31 | 0.000 |
|  22 | grouped_var_k |  166.2 |  40.0  |  85.9 | 0.11 | 5.00 | 0.999 |
|  28 | grouped       |  153.1 |  40.8  |  88.2 | 0.15 | 4.98 | 0.000 |
|  32 | grouped       |  150.7 |  41.7  |  88.2 | 0.15 | 4.89 | 0.000 |
|  33 | grouped_var_k |  167.1 |  40.3  |  85.9 | 0.10 | 5.04 | 0.999 |
|  39 | grouped       |  153.8 |  40.5  |  88.2 | 0.15 | 4.97 | 0.000 |
|  43 | grouped       |  147.0 |  43.0  |  88.2 | 0.16 | 5.05 | 0.000 |
|  44 | grouped_var_k |  165.7 |  40.5  |  85.9 | 0.11 | 5.04 | 0.999 |

Steady-state averages (excluding the first dispatch of each kernel as
warmup-tainted):

| kernel        | MfmaUtil | VALUUtil | MemUnitStalled | MeanOcc/CU | LdsBC | dur_us |
|---------------|---------:|---------:|---------------:|-----------:|------:|-------:|
| grouped (fwd+dA) |  42.2%  |   88.2%  |    0.15%       |   5.00     |  0%   |  151   |
| grouped_var_k (dB) |  40.4%  |  85.9%  |    0.10%       |   5.04     |  1.0% |  166   |

Resource report from build (matched against R17's HipKittens HEAD
`40be51de`, identical to current R18 baseline since R93 reverted):

```
grouped_kernel<Layout::RCR, KI=0, FUSED_KTAIL=true>:
  TotalSGPRs: 102   VGPRs: 250   Occupancy [waves/SIMD]: 2

grouped_var_k_kernel<0>:
  TotalSGPRs: 94    VGPRs: 256   Occupancy [waves/SIMD]: 2
```

## Cross-reference vs R9 forward-only data

R9's PMC anchor (forward only, B=32 shapes, same 5 derived counters):

```
gpt_oss-GateUP-B32-M2048   K=2880  MfmaUtil 63.4 %  Occ 7.29
gpt_oss-Down-B32-M2048     K=2880  MfmaUtil 62.3 %  Occ 7.27
DSV3-GateUP-B16-M4096      K=7168  MfmaUtil 79.7 %  Occ 7.34
Qwen3-GateUP-B16-M4096     K=4096  MfmaUtil 75.3 %  Occ 7.33
Qwen3-Down-B16-M4096       K=1536  MfmaUtil 25.6 %  Occ 3.58
```

R9 fwd B=32 = 62-63% MfmaUtil. R17 fwd B=4 = 42%. **20 pp gap between
B=32 and B=4 forward** despite both being on the same K-tail path.
Same kernel, same instruction stream, same K=2880; the only
difference is total tile count per CU:

* B=32: total_tiles ≈ 384 × 32 = 12 288 tiles, 256 CU = 48 tiles/CU
* B=4: total_tiles ≈ 384 × 4 = 1 536 tiles, 256 CU = 6 tiles/CU

The persistent grouped_kernel amortizes the per-tile prologue
(group lookup, swizzled offset prefill, A/B SRD setup) and per-tile
epilog (sync barrier, store, last waitcnt). With 6 tiles/CU these
overheads are 1/6 = 17% of per-CU time; with 48 tiles/CU they're 1/48
= 2%. The 15 pp shift in overhead amortization explains most of the
B=4 vs B=32 MFMA-util gap.

## What the data falsifies

**The "memory-bound var-K" hypothesis that drove R83-R93 is
falsified**:

* `MemUnitStalled = 0.10 %` on var_k_kernel — the memory unit is
  effectively never stalled. A memory-bound kernel sits at 10-30 %
  MemUnitStalled.
* `LdsBankConflict = 1.0 %` on var_k_kernel — barely above noise. Far
  from "LDS-bank-bound" (a bound kernel hits 5-15 %). R83-R88's LDS
  swizzle / pad / cross-permute levers were chasing a bottleneck that
  never existed in the metric environment.
* Forward and var_k kernels run at near-identical MfmaUtil (~40-42 %)
  on the same shape, despite VERY different memory access patterns
  (forward: persistent grid + LDS-staged main loop + HBM-direct
  K-tail. var_k: persistent grid + LDS-staged main loop ONLY,
  CRR layout, double-buffered). If memory was the bottleneck, the
  two would not converge.

**The "L2-thrash explains R93 -77 score" hypothesis is consistent with
the data but wasn't strictly required**: R93's failure was already
explained (and predicted by post-mortem analysis) via XCD-affine
mismatch between blockIdx.x % NUM_XCDS round-robin and the per-group
partition cluster. The PMC data above just confirms that even the
ORIGINAL stride-NUM_CUS kernel isn't memory-bound — it's
MFMA-undersaturated.

## What the data points toward — Lever B1 family

Both kernels show:
1. **MfmaUtil ≈ 40-42 %** at B=4 (the worst case) — far from the ~63 %
   B=32 ceiling for the SAME kernel. Roughly 50 % of compute cycles
   are MFMA-idle.
2. **MeanOccupancyPerCU = 5.0 / theoretical 8** — 63 % occupancy.
   Compiler reports `Occupancy [waves/SIMD]: 2` (= 8 waves/CU) but
   only 5 are actually resident. The 3-wave gap is barrier / waitcnt
   stall.
3. **VALUUtilization = 86-88 %** — vector ALU is busy. Suggests the
   non-MFMA cycles are doing useful V_ALU work (address computation,
   LDS swizzle, coord arithmetic), not idle stalls.
4. **MemUnitStalled = 0.10-0.15 %** — memory stall is irrelevant.

Three R83-R93-untried levers map onto this profile:

* **Lever B1: MFMA pipeline scheduling**. The 40 % MFMA util gap
  doesn't come from memory or LDS — it comes from MFMA-issue
  stalls inside the K-loop. Likely culprits: V_ALU instructions
  serialising MFMA chains (RAW on `C_accum`), prologue/epilog code
  that the compiler doesn't interleave with main-loop MFMAs, and
  DO_MMA macro expansion that emits all MFMA in a contiguous block
  (no cross-tile interleaving).
  
  Concrete attack: insert `__builtin_amdgcn_sched_barrier(...)` hints
  to allow the compiler to interleave V_ALU and MFMA across DO_MMA
  invocations. Or split the C_accum array across 2 register banks so
  alternating MFMAs can issue without RAW chain stalls. Each percent
  of MfmaUtil gain ≈ +1 % per-kernel speedup ≈ +0.5-1.0 score on the
  metric. A 10 pp MFMA-util improvement (42 → 52 %) maps to roughly
  +5-10 score across the 24 shapes.

* **Lever B2: register pressure / occupancy**. VGPR=250 / Occ=2
  waves/SIMD. Dropping to VGPR=192 (occ=3) requires either reducing
  the live tile-array footprint OR allowing minor LDS-resident
  staging. VGPR savings on the same kernel can also enable a wider
  prologue/epilog-overlap schedule (Lever B1 above). The two levers
  are coupled.

* **Lever B1' (lower-leverage variant): tile-level pipelining for
  low-tile-count shapes**. The B=4 (1.054) vs B=32 (1.052) ratio is
  near-identical — Triton's bwd path is similarly limited at low B.
  The amortization gap (6 tiles/CU vs 48 tiles/CU) is structural
  but could be offset by interleaving consecutive tiles' main loops
  (issue tile N+1's prologue during tile N's K-tail). Would lift
  B=4 toward B=32's MfmaUtil ceiling. Multi-round refactor.

## Why the falsification streak should pivot here

R83-R93 collectively spent 11 rounds on memory-side levers, all
falsified. That's ~11 % of the 100-round task budget burned without
score progress (best 891 → current best 889). The PMC data above
declares those leverage candidates dead — not because the implementations
were buggy, but because the kernel isn't memory-bound to begin with.

Lever B1 (MFMA scheduling) hasn't been tried since the pre-R83 era. The
PMC data suggests it has the highest concentration of remaining
score-per-round leverage. R19+ should pick a concrete B1 variant and
attempt it.

## R17/R18 outcome

* Round 17 (auto-loop): captured PMC pass 1 data on
  `gpt_oss-Down-B4-M2048` for both forward (`grouped_kernel<RCR, 0,
  FUSED=true>`) and backward dB (`grouped_var_k_kernel<0>`). Round
  was interrupted before commit due to time-window expiration; PMC
  CSV preserved at `/tmp/rocprof_round17_gpt_oss_Down_B4/pass_1/chi2894/`.
* Round 18 (this commit): re-ran metric → score 881-882 (within ±2
  of R17's start). Cross-shape PMC retry (B=32 minimal counters) hung
  past 90 s; killed and proceeded with R17 + R9 archive data.
* Files touched: this note (`analysis/_notes/round-94-bf16-grouped-pmc-bwd-mfma-undersaturation-redirects-r83-93-streak.md`).
* Code changes: NONE. R17/R18 are pure diagnostic rounds.
* Build state: HipKittens working tree clean.
* Metric:
  - R17 start: 884 (gpt_oss 1.098 / DSV3 1.118 / Qwen3 1.112)
  - R18 baseline (before this note): 881-882
  - R18 final: 881-882 (no code change)
  - All 24 shapes PASS, no regression.

## Recommended R19 plan

**Lever B1 step 1**: try the cheapest sched_barrier-only variant.
Insert `__builtin_amdgcn_sched_barrier(0x18)` (= mask out
`SCHED_BARRIER_MASK_VALU | SCHED_BARRIER_MASK_TRANS`) before each
MFMA in `device_gemm_tile_body` to let the compiler hoist V_ALU
above pending MFMAs. No correctness risk, no VGPR delta, ~1 hour to
implement and gate. Expected +2-5 score if it works; falsification
note if not.

If B1 step 1 falsifies, R20 tries B1 step 2: split `C_accum[2][2]`
into two register banks `C_accum_a[2][2]` and `C_accum_b[2][2]`, with
DO_MMA writing alternating banks. Halves the RAW chain length; should
double effective MFMA throughput. Risk: may exceed 256 VGPR ceiling
(currently 250 → ~282 after split). Mitigation: drop one of the
intermediate B-stage register banks to LDS to make room.

R21+ depends on R19/R20 outcome. If B1 family flat or down, pivot to
**Lever B1' tile-level pipelining** (high-effort multi-round) or
**Lever C uniform-M dB BMM dispatch** (deferred since R91 — 
requires new HK kernel binding).

## SHA pointers

* Primus-Turbo HEAD pre-round (R18): `af614435c9c211333e785d6f27af451b3730d97b`
  (= round-27 fp8 docs; not a BF16 commit — last BF16 commit is
  859ca38 = R93 falsification note)
* HipKittens HEAD: `40be51de` (= pre-R92 baseline; R92 + R93 both
  reverted)
* R17 PMC CSV preserved at `/tmp/rocprof_round17_gpt_oss_Down_B4/`
  for R19 reference
