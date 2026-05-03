# Round 18 — FP8 grouped GEMM: end-to-end Python overhead is hidden, Qwen3 (gm=4, xcd=8) signal is noise

## Today's metric (HEAD `7e714b2`, post-warm)

```
[metric_fused_wall] geomean=1.3393  progress=0.992  FAIL
[metric_fused_wall] correct_fail=0/24  reject=0/24  below_target=14/24  goals=10/24  score=992
```

Bottom-of-metric (sorted ascending) — within the same noise band as R17:

| shape | ratio |
| --- | --- |
| Qwen3-Down-B16-M2048   (K=1536) | 1.250 |
| Qwen3-GateUP-B16-M2048 (K=4096) | 1.256 |
| Qwen3-Down-B16-M4096   (K=1536) | 1.269 |
| Qwen3-GateUP-B32-M2048 (K=4096) | 1.271 |
| gpt_oss-Down-B32-M2048 (K=2880) | 1.271 |
| Qwen3-Down-B32-M2048   (K=1536) | 1.280 |
| Qwen3-GateUP-B16-M4096 (K=4096) | 1.285 |
| Qwen3-Down-B32-M4096   (K=1536) | 1.291 |
| gpt_oss-Down-B32-M4096 (K=2880) | 1.297 |
| DSV3-Down-B16-M2048    (K=2048) | 1.297 |

## What this round did

Two parallel investigations, both falsifications, both eliminate
Primus-side levers definitively.

### (1) End-to-end dispatch overhead probe (`/tmp/probe_r17_dispatch_overhead.py`)

Mirror of R17's attribution probe but at a deeper level: time the
**full** `grouped_gemm_fp8_impl` call chain (custom_op + dispatcher +
can_handle + execute + kernel) vs the bare `execute()` body vs just
`hk.grouped_rcr_dscale(...)`. All three measure the SAME shape
(Qwen3-Down-B16-M2048, B=16 M=2048 N=4096 K=1536) under identical
conditions:

```
grouped_gemm_fp8_impl (FULL public path)        :   220.73 us / call
GroupedGEMMFP8HipKittenBackend.execute (DIRECT) :   222.29 us / call
hk.grouped_rcr_dscale (KERNEL+OUT_REUSE)        :   222.65 us / call
hk.grouped_rcr_dscale + fresh empty             :   223.23 us / call

Overhead breakdown:
  custom_op + dispatcher overhead:   -0.07 us / call
  execute body overhead:              0.22 us / call
  kernel + alloc:                   223.22 us / call
  TOTAL public API:                 223.37 us / call
```

**The custom_op + dispatcher + can_handle + execute body together add
< 1 µs / call** when measured end-to-end with the kernel. The "noise"
in the differences (-0.07, +0.22, +0.36) is well below the 220 µs
kernel wall and is dominated by per-iter `cuda.Event` overhead.

Compare to a Python-only probe of the execute body (no kernel):

```
hk.grouped_dscale('rcr')                 :     36.2 ns / call
hk.grouped_rcr_dscale (direct attr)      :     19.8 ns / call
select_default_config(Qwen3-Down K=1536) :    542.1 ns / call
torch.empty((M_total, N))                :   1600.6 ns / call
FULL Python body of execute() (no kernel):   2776.6 ns / call
MINIMAL Python body (pre-resolved everything): 1846.8 ns / call
```

The 2.78 µs Python work is **fully overlapped with kernel async
queueing** — the kernel `hipModuleLaunchKernel` returns to Python in
~5-15 µs while the actual SIMD work runs ~210 µs in the background.
Subsequent Python steps (the next call's `select_default_config`,
`torch.empty`, etc.) execute **during** the prior kernel's GPU
runtime, so they cost zero wall.

This invalidates a long-standing assumption: even if I could trim
ALL ~2.8 µs of Python in the execute body to zero, the metric
would not move. **Python overhead has been below the kernel-wall
horizon since R10/R11.** R14/R16 host-side trims were correct on
paper but were below the metric's noise floor by construction.

### (2) Qwen3-Down-B16-M2048 metric-aligned config sweep falsified
(`/tmp/probe_r18_qwen3_down_m2048_metric_aligned.py` →
 `/tmp/probe_r18_verify_qwen3_xcd_signal.py`)

R12 swept Qwen3-Down K=1536 family at steady-state 200-iter timing
and found `(gm=4, num_xcds=8)` was joint with `(gm=4, num_xcds=0)`
(binding default → 8) — bit-equivalent kernels per the binding's
`0 → 8` fallback. R23 noted that the metric script's
`WARMUP=10, ITERS=50, per-iter cuda.Event` regime can produce
different winners than the steady-state bench (R23's gpt_oss
GateUP-B4 win for `(gm=1, xcd=4)` was found in this regime).

Re-swept Qwen3-Down-B16-M2048 in the metric-aligned regime first
(5 trials):

```
(gm, xcd)      p20 ms (med)  spread%     TFLOPS   Δ vs (4,0)
( 4,  0)           0.2231 ms    5.63%   1848.27    +0.000%   ← high spread!
( 4,  8)           0.2202 ms    0.76%   1872.79    +1.309%
( 4, 16)           0.2205 ms    1.20%   1870.07    +1.165%
( 4,  4)           0.2281 ms    0.25%   1807.44    -2.259%
```

The `(4, 8)` cell appeared to win by +1.31 % — but the baseline
`(4, 0)` had **5.63 % run-to-run spread** vs `(4, 8)`'s 0.76 %,
suggesting the gap was a single-trial noise event on the baseline.
Tight 10-trial verify across 6 Qwen3 shapes:

```
shape                                 (4,0) ms   (4,8) ms     Δ%   (4,0) TF   (4,8) TF
Qwen3-Down-B16-M2048    K=1536          0.2200     0.2197 +0.145%    1873.81    1876.54
Qwen3-Down-B32-M2048    K=1536          0.4326     0.4333 -0.166%    1906.21    1903.04
Qwen3-GateUP-B16-M2048  K=4096          0.3313     0.3335 -0.682%    2489.21    2472.35
Qwen3-GateUP-B32-M2048  K=4096          0.6618     0.6635 -0.266%    2492.22    2485.61
Qwen3-Down-B16-M4096    K=1536          0.4453     0.4453 -0.013%
Qwen3-Down-B32-M4096    K=1536          0.8802     0.8792 +0.123%
```

Signal range collapses from `+1.31 %` (5-trial) to `[-0.68 %,
+0.15 %]` (10-trial) — pure noise, with `(4, 8)` actually
*regressing* on Qwen3-GateUP shapes. R12's falsification stands
across both timing regimes. The `(4, 0) ↔ (4, 8)` bit-equivalence
guarantee from the binding (kernel sees `BLOCK_SWIZZLE_NUM_XCDS=8`
either way) means any timing difference IS noise — there is no
underlying performance signal to find.

## What both falsifications jointly imply

1. **Python-side dispatch is no longer measurable** in the metric.
   Even at the most aggressive theoretical cut (eliminate ALL ~2.8 µs
   of execute-body Python), the metric would not move because the
   kernel wall is 100× larger and Python overlaps with kernel async.
   R14 (-0.4-1.4 % wall savings on B=4 small shapes) and R16
   (-0.05-0.4 pp on var-K dispatch) were the last meaningful host
   trims; further trim attempts cannot register above 0.5 score.

2. **Per-shape config tuning is exhausted** for the Qwen3 family.
   R6 / R12 / R23 / this round's regime-specific re-tests have all
   converged on the same conclusion: the binding default and `(gm=2,
   xcd=8)` for M=4096 are the local optima for Qwen3-Down K=1536;
   `(gm=1, xcd=4)` is the local optimum for Qwen3-GateUP K=4096
   (R10 / R45 already wired). All remaining cells are within
   ±0.5 pp run-to-run noise.

3. **The remaining ratio gap (1.250 → 1.350 = 8.0 pp on the worst
   shape) lives entirely in HK kernel internals**, as R17 attribution
   already showed (fwd_x = 1.19, bwd_x = 1.30 on Qwen3-Down-B16-M2048).

## What's NOT actionable in `primus_turbo` anymore

The exhausted lever inventory after R18:

| Lever family | Status | Last touched |
| --- | --- | --- |
| Forward execute Python trim | Sub-noise | R11 / R14 |
| Var-K execute Python trim | Sub-noise | R16 |
| FusedActFunc dispatch shortcut | Sub-noise | R14 |
| FP8 weight quant cache | LANDED | R9 |
| FP8 activation quant cache | LANDED | R10 |
| FP8 grad_out delayed-scale cache | LANDED | R11 |
| H4 transpose cache | LANDED | R9 |
| FP8 RCR per-shape config rules | All wins below noise | R6 / R12 / R23 |
| FP8 RRR per-shape config rules | LANDED for Qwen3 family | R44 |
| FP8 var-K per-shape config rules | LANDED for m_total≥16384 | R39 |
| Var-K Qwen3 family config sweep | FALSIFIED | R15 |
| Qwen3 K=1536 metric-aligned sweep | FALSIFIED (this round) | R18 |
| End-to-end dispatch overhead | Sub-µs (this round) | R18 |

## What IS actionable (next chat)

The R17 next-step plan stands and is the **only** remaining direction:

* **Highest ceiling — HK source `BLOCK_K=64` template specialization
  for shallow-K shapes**. Multi-round (2-4 chats):
  1. Template `kernel_fp8_layouts.cpp` types on BK (or duplicate the
     LDS slab + register tile types under a `BK_64` namespace).
  2. Add `grouped_rcr_kernel<...>` instantiations for the BK=64
     variant.
  3. Update `dispatch_grouped_rcr` (line 6395 onward) to choose the
     BK template based on K/N/m_total — needs a Python config rule on
     the Primus-Turbo side too.
  4. Correctness probe + sweep + Primus dispatch wire-up.

* **Lower ceiling — HK source `grouped_rcr_kernel` BUFFER vs FLAT
  store mode for shallow-K shapes**. The R19/R20 BUFFER reroute won
  big on long-K shapes; whether shallow-K Qwen3-Down would benefit
  from FLAT (lower kernel-side latency at the cost of less coalescing)
  has not been probed since the kernel template change. Single-round
  HK-side experiment.

Both require fresh chat sessions (the current chat has spent its
budget on Primus-side audits and the SKILL.md context is loaded with
Primus rules, not HK rules).

## Round 18 round summary

* Target: lowest-ratio Qwen3-Down-B16-M2048 (1.250) and the rest of
  the bottom 5.
* Outcome: **two falsifications**:
  1. End-to-end dispatch+execute Python overhead is **< 1 µs / call**
     (kernel wall is 220 µs, Python is overlapped via kernel async).
  2. Metric-aligned (gm=4, xcd=8) signal **collapses to noise** at
     10-trial verify across 6 Qwen3 shapes (range -0.68 % to +0.15 %).
* Code changes: none in `primus_turbo/`. Two new probe scripts under
  `/tmp/`: dispatch-overhead probe + metric-aligned config sweep.
  Probes are intentionally not committed (one-shot diagnostic, not
  reusable test infrastructure unlike R17's attribution probe).
* Metric: 992 (within noise band 980-1000 since R11). No regression.
* Next chat: HK source kernel surgery (no further Primus-Turbo lever
  exists; see "What IS actionable" above).
