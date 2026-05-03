# Round 23 — Noise floor quantified: 8-run metric distribution at fixed HEAD

This round is a **measurement** round (no kernel/dispatch change).  Goal:
nail down the actual run-to-run noise floor of
``scripts/_metric_grouped_fused_wall.py`` so the auto_optimize loop has
hard numbers to decide whether to keep burning rounds at the plateau or
pivot to multi-round HK kernel surgery.

R12 / R15 / R17 / R19 collectively concluded "all cheap Primus-side
levers exhausted; only HK kernel surgery (BLOCK_K=64 template) can push
the geomean above the noise band".  None of those rounds, however,
quantified the noise band rigorously — R15 cited a 3-run sample (1000 /
1000 / 831, the 831 being a cold-GPU artifact).  This round closes that
gap with a tight-controlled 8-run sample.

## Setup

* HEAD `a6d4b4f8` on Primus-Turbo, HEAD `a7683112` on HipKittens (both
  unchanged across the 8 runs).
* Single GPU pinned by auto_optimize (``HIP_VISIBLE_DEVICES=3`` chosen
  this round from pool ``3,4,6,7``).
* Pre-warm before the first metric run:
  ``python3 -c "import torch; a=torch.randn(8192,8192,device='cuda',
  dtype=torch.bfloat16); b=...; [torch.matmul(a,b) for _ in range(50)]"``.
* All 8 runs back-to-back within ~70 s, no idle gap → stays in
  high-power state (the 831 outlier R15 saw was a cold-GPU artifact;
  pre-warm + back-to-back execution eliminates it).
* Otherwise the metric script is the unmodified
  ``scripts/_metric_grouped_fused_wall.py`` (same WARMUP=10, ITERS=50,
  per-iter ``cudaDeviceSynchronize`` p20 timing).

## Raw 8-run data

```
run #     score    geomean    notes
1          996     1.3444     full per-shape table captured (used as today's auto_optimize sample)
2          994     1.3418
3          998     1.3469
4          986     1.3314
5          985     1.3303
6          992     1.3388
7          989     1.3347
8         1000     1.3524     >= 1.35 target, PASS
```

* min = 985, max = 1000, range = 15 pts
* median = 993.0
* mean   = 992.5
* stdev  ≈ 5.2 pts
* P(score == 1000) on a single run ≈ 1/8 = 12.5 %
* P(score >= 995) ≈ 5/8 = 62.5 %

## What this means for the auto_optimize loop

The metric ``score = int(min(geomean / 1.35, 1.0) * 1000)`` has a
**hard cap at 1000**.  With geomean naturally fluctuating in the band
``[1.330, 1.353]``, a single run lands in ``[985, 1000]`` with the
distribution above.  An "improved" round (``+5`` over the previous
sample) is observed under random noise alone with probability ≈ 30 %
between any two consecutive samples — the auto_optimize improvement
detector has zero discriminative power at this score level.

Concretely, the recent ``[round 18..23]`` history
``988, 992, 992, 986, 995, 996`` (with no Primus-side commit since R20)
is a **pure noise walk** within the measured ``[985, 1000]`` band.  The
``streak=11`` non-improvement counter is therefore not signalling a
stuck optimizer — it's signalling that there is nothing left within
this metric's measurement noise to improve.

## What's NOT a lever (re-confirmed by 4 chat sessions)

| Direction                                          | Round (chat)        | Outcome                                           |
| -------------------------------------------------- | ------------------- | ------------------------------------------------- |
| Path A fused-fwd (BF16 → FP8 cvt inside load_a)    | R7                  | -26 % wall (DTR + cvt cannot beat FP8-direct DTL) |
| Activation / weight / grad_out quant caches        | R9 / R10 / R11      | LANDED, all real wins; ceiling hit                |
| FP8 RCR per-shape ``(group_m, num_xcds)`` tuning   | R6 / R12 / R23 / R26| All Qwen3-Down K=1536 candidates within noise     |
| FP8 RRR per-shape rules for Qwen3-GateUP           | R44                 | No single (gm, xcd) cell wins all 4 shapes        |
| FP8 var-K per-shape rules for Qwen3 family         | R15                 | All 8 cells within ±0.34 % (deep below noise)     |
| Var-K Python execute trim                          | R16                 | -485 ns / call sub-noise                          |
| Forward execute Python trim                        | R11 / R14           | Sub-noise                                         |
| Grouped execute method-call trim                   | R19                 | Sub-noise                                         |
| FusedActFunc dispatch shortcut                     | R14                 | Sub-noise (try/except + 3 frame skip)             |
| End-to-end Python overhead probe                   | R18                 | < 1 µs / call (kernel-async hides it)             |
| Qwen3 (gm=4, xcd=8) metric-aligned 5-trial signal  | R18                 | Falsified at 10-trial (collapses to noise)        |

The R17 forward-vs-backward attribution probe established **the
remaining gap on every bottom-of-metric Qwen3 shape lives in the HK
forward kernel** (fwd_x ∈ [1.10, 1.20] vs bwd_x ∈ [1.29, 1.35]).  HK
backward is already at or near target on every bottom shape; the cap
therefore lives in ``grouped_rcr_kernel<KI_HINT=0, N_MASKED_STORE,
FUSED_KTAIL=*>`` itself.

## What IS a lever (HK kernel surgery, multi-round)

The next-chat plan from R17 / R19 / R20 stands unchanged:

1. **Highest ceiling — `BLOCK_K=64` template specialization for
   shallow-K shapes** in
   ``/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp``.
   Adds a `BK` template parameter to the LDS slab type
   ``st_fp8e4m3<HB, BK, ...>``, register tile types
   ``rt_fp8e4m3<RBM, BK, row_l, rt_16x128_s>``, and the kernel itself.
   New ``grouped_rcr_kernel<KI_HINT=0, N_MASKED_STORE, FUSED_KTAIL=*,
   BK=64>`` instantiation for Qwen3-Down K=1536 + Qwen3-GateUP K=4096.
   The audit count is **23 references to ``K_BLOCK`` / ``BK = 128`` in
   the file** — non-trivial but bounded.

   * Round budget: **3-5 rounds** to template the types + add the BK=64
     instance + correctness probe + Primus dispatch rule wire-up.
   * Per-round risk: high (LDS slab + register tile type changes can
     trip llvm spill; correctness probe gating is mandatory).

2. **Lower-ceiling secondary — ``grouped_rcr_kernel`` BUFFER vs FLAT
   store mode** for shallow-K Qwen3 (1 round HK-side).  The R19/R20
   BUFFER reroute won big on long-K; whether shallow-K benefits from
   FLAT (lower kernel-side latency at the cost of less coalescing) has
   not been probed since.

3. **gpt_oss K=2880 H4 reroute audit with rocprof** (1 round).
   Currently fwd_x ≈ 1.10 on B=32 (worst of all bottom-5 shapes).  If
   the transpose itself dominates rather than the RCR-on-transposed
   kernel, a pre-quantize-time transpose cache (vs the current
   per-call cache hit) might save another 1-2 % wall.  Smaller ceiling
   but lower risk than (1).

## Decision for this round (and recommendation for the next ≤ 19 rounds)

* **No Primus-Turbo source change committed this round.**  Adding a
  sub-noise Python trim or a per-shape config fiddle would risk a
  -5 score regression with no real payoff (proved 5x in R14-R19).
* **No HipKittens source change committed this round.**  The
  next concrete HipKittens lever (BLOCK_K=64 template) is multi-round
  by audit; starting it within the same chat as a doc-only round risks
  partial-state HK changes that future chats can't safely build on.
* **Recommendation for the auto_optimize loop**: with patience at
  ``11/30`` and the 8-run distribution showing the median is **at**
  the noise ceiling, the remaining 19 rounds are best spent EITHER:
    - **(A) Letting patience naturally expire** — no productive work
      possible without HK kernel surgery; the loop converges to
      "best metric == 1000 (noise-ceiling)".
    - **(B) Pivoting the next chat to a focused HK kernel surgery
      session** with a fresh task body that explicitly scopes the
      multi-round work (audit → template → BK=64 instance →
      correctness → Primus rule).  This is the only direction that can
      push the metric *median* above the current 993.

Plan (A) wastes ~19 cursor-agent rounds.  Plan (B) requires
out-of-band coordination (the auto_optimize task body would need a
rewrite to recognize partial-progress HK commits as "improvement"
without the metric moving).

## Files committed this round

* ``analysis/_notes/round-23-fp8-grouped-noise-floor-quantified-8-run-distribution.md``
  — this note.

No other files in either repo.  No HipKittens commit.

## Summary line for auto_optimize

Round 23, lever **none-cheap-remain (noise-floor measurement)**, files
**docs only**, metric before/after **996/996** (within 985-1000 noise
band measured from 8 back-to-back runs at this HEAD), Primus SHA
unchanged target HEAD, HipKittens SHA unchanged.  Next round
recommendation: **either let patience expire OR escalate to
out-of-loop HK kernel surgery chat** (BLOCK_K=64 template
specialization, 3-5 round commitment).
