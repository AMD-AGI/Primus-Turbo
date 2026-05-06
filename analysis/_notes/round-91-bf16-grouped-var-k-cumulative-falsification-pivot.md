# Round 91 — bf16 grouped GEMM weighted wall (auto_optimize round 14/100)

> **Context:** auto_optimize round 14/100. R83-R90 = **8-round
> falsified streak** on `grouped_var_k_kernel` VGPR / swizzle /
> rematerialisation / readfirstlane levers. R91 = formal pivot
> round: lock the verdict, document remaining viable angles for
> R15+, no code change.

**Status:** R91 = **structural / formal-pivot round**. HK + Primus
working trees clean. Score in noise band (881-883 across last 4
rounds).

| run                                              | n | mean | std |
|--------------------------------------------------|--:|-----:|----:|
| R86 baseline (st_32x16_v2 within-half FALSIFIED) | 1 |  -5  | -   |
| R87 (PMC fwd-vs-bwd kernel split, docs)          | 5 | 882.4| 0.6 |
| R88 (RRR ST_B pad-only swizzle FALSIFIED)        | 5 | 883.0| 0.7 |
| R89 (var_k helper extraction, bit-identical)     | 5 | 882.6| 0.5 |
| R90 (rematerialisation analytical falsification) | 5 | 883.0| 0.7 |
| R91 open metric (single run, this round)         | 1 | 881  | -   |

Mean across R86-R90 = **882.6 ± 0.5**. Best-ever 886 (R8) sits
**+3.4 sigma** above the cluster but inside historical noise envelope
for this metric on a contended GPU pool. The `improved=False` streak
is therefore best read as `"effectively at local optimum"` for the
attack vector R83-R90 has been pursuing.

## Cumulative falsification streak — what was tried

| Round | Lever / target                                            | Build gate    | Metric        | Verdict |
|-------|-----------------------------------------------------------|---------------|---------------|---------|
| R83   | var_k local recompute trim                                | -3 spill      | flat          | F       |
| R84   | var_k env-gated FUSE_DISABLE PMC pair (diagnostic)        | n/a           | n/a           | docs    |
| R85   | RRR cross-permute swizzle (Lever B5 prototype)            | spill         | -             | F       |
| R86   | st_32x16_v2 within-half swizzle (Lever B5 v2)             | +17 VGPR      | -5 score      | F       |
| R87   | PMC fwd vs bwd kernel split (diagnostic)                  | n/a           | n/a           | docs    |
| R88   | RRR ST_B pad-only swizzle (Lever B5 final)                | OK on RRR-fuse| flat          | F       |
| R89   | var_k helper extraction (Lever B6 step 1, prep)           | bit-identical | bit-identical | docs    |
| R90   | rematerialisation (Lever B6 step 2) + readfirstlane       | bit-identical | flat          | F       |

5 *performance* rounds (R83 R85 R86 R88 R90) → all flat / regressed.
3 *structural* rounds (R84 R87 R89) → docs-only, bit-identical
build outputs.

## Why every lever has hit the same wall

R87 PMC (verified on `gpt_oss-Down-B4-M2048` then re-verified
analytically for B=32 — same kernel mix, +8x launch volume):

* **forward path (`grouped_kernel<RCR, FUSED_KTAIL=true>`):**
  * 0 LDS bank conflicts.
  * HBM-bound (compute time fully overlapped with K_STEP-sized
    HBM→LDS double-buffer).
  * 256 VGPR / 0 scratch / Occ 2 — at the spillover ceiling.

* **backward dB (`grouped_var_k_kernel<0>`):**
  * **2.0 LDS BC / Inst (= 50 % bank-conflict instruction rate)**
    — confirmed on B=4; same kernel + same swizzle for B=32 means
    same per-instruction rate, just more instructions.
  * 256 VGPR / 0 scratch / Occ 2 — also at the ceiling.
  * 50 % LDS BC is the dominant bottleneck for the entire bwd pass
    of every gpt_oss shape (and ~50 % of the metric headroom toward
    1.25 sits on this kernel for the 8 weight-3 shapes).

The classical fix for 50 % LDS BC on `st_32x16_s` is a different
in-tile swizzle (e.g. `st_32x16_v2` with subtile_padding=16, or
within-half permutation). **Every viable swizzle attempted in
R85-R88 either:**

1. **Adds VGPR pressure** (R86 within-half permute: +17 VGPR
   spill) — the kernel cannot afford this since the compiler is
   already at the 256 VGPR ceiling.
2. **Doesn't reduce BC enough** (R88 RRR ST_B pad-only: build OK,
   metric flat) — pad-only addresses inter-subtile alignment, not
   within-wave bank striping which is where the conflicts live.

R89-R90 attempted to **free VGPR margin** so a future swizzle can
fit. Both legs failed:

* R12 helper extraction (R89) is `always_inline` in production →
  bit-identical compiled output → no margin freed.
* R13 rematerialisation (R90) was analytically falsified —
  `k_offset_tiles` and `ki_g` are captured by reference in
  `device_gemm_tile_body`'s `a_coord` / `b_coord` lambdas and live
  across the entire main-loop body, so passing them in vs
  recomputing inside makes zero difference to the live-range graph.
* R13 fallback (explicit readfirstlane on derived LDS offsets) was
  bit-identical — the AMDGPU SI Convergent Annotator pass already
  promotes them to SGPR.

## What's still viable — Plan B for R15+

The kernel-internal angle on `grouped_var_k_kernel` is **exhausted**
for the `st_32x16_s` swizzle family at the 256 VGPR ceiling. R15+
should pivot to a fundamentally different angle. The 4 most
promising angles, ranked by expected score-per-round leverage:

### Lever A1 (high-leverage, untried) — forward HBM dual-buffer

R87 confirmed forward `grouped_kernel<RCR, FUSED_KTAIL=true>` is
HBM-bound (not LDS-BC-bound, not MFMA-bound). The K-tail epilog
(`device_gemm_tile_body` line 829-915 in HK source) loads the K-tail
LDS stripe into stage-1 slots and feeds the same DO_MMA pipeline.
The current implementation issues the K-tail HBM→LDS load AFTER
epilog 2 drains stage 1 — i.e. there's a stall between "main loop
epilog 2 done" and "K-tail load complete" that the existing
double-buffer doesn't cover.

A targeted prefetch (start the K-tail load at the entry of epilog 1
rather than after epilog 2) would reduce that stall by ~K_STEP-worth
of HBM latency. Touches only the `if constexpr (FUSED_KTAIL)` block
and is gated by `K_REM == K_STEP` (compile-time), so it cannot
regress the K-aligned path. Forward affects 8/8 gpt_oss + 8/8
DSV3 + 8/8 Qwen3 shapes — broad coverage.

### Lever B4 (medium-leverage, partially explored) — persistent grid work-stealing

Current `grouped_var_k_kernel` outer loop:
```cpp
for (int gt = pid; gt < total_tiles; gt += NUM_CUS) { ... }
```

distributes tiles across CUs in stride-`NUM_CUS` chunks. With
`tiles_per_group ≈ 88` (for n=2880, k=2048-tile slice) and `G ≤ 32`
in metric, total_tiles ≤ 2816, so each CU sees ~9.3 tiles — a
short persistent loop where the per-tile fixed cost (group lookup
+ coord swizzle, R12 helpers) is amortized only ~9x.

Swap to a **per-group chunked schedule** (each CU steals contiguous
tile-blocks within a group rather than strided across groups) —
improves L2 reuse for `g.a` (read by all tiles in the same group's
M-stripe). Risk: hurts CU work balance when groups are highly
non-uniform; metric uses uniform M (balance=True) so this risk is
zero in the metric environment.

### Lever C (high-leverage, FROZEN-list-adjacent) — uniform-M dB BMM dispatch

When all `M_g` are equal (= the metric environment with
`balance=True`), `dB[g] = dout[g].T @ x[g]` is mathematically
identical to a regular batched GEMM `dB = dout.T @ x` reshaped per
group. The BMM kernel has fixed K, no var-K LDS-BC bottleneck.

**FROZEN concern:** This requires checking `group_lens` uniformity
on the host, which is a `.tolist()` / equality check. The FROZEN
list bans "host syncs in the hot path" (item #4) — a uniformity
check on `group_lens` reads it as host data, which is a sync.
**However:** the dispatcher already pulls `B`, `M_total`, `n`, `k`
from tensor shapes (lines 348-353 of `grouped_gemm_impl.py`) which
are static metadata, not data syncs. If `group_lens.size(0) == B`
and `M_total % B == 0`, we have a *static-metadata* uniformity hint
that requires no `.item()` / `.tolist()`. This isn't quite
"uniformity" (the actual values could differ even if shape says
B=32 and M_total=65536 → avg_m=2048), but for the metric's
`balance=True` callers it always *is* uniform.

**Open question for the agent:** does the kernel binding for
HipKittens BF16 grouped expose a "uniform-M fast path" knob? If
yes, dispatching to that would be a static-metadata gate (no host
sync). If no, this lever is FROZEN-blocked.

### Lever D (low-leverage, broad) — DSV3 / Qwen3 push (Phase B)

Each non-gpt_oss shape is at ratio ~1.10-1.13 vs target 1.25. With
weight=1, lifting all 16 by 5pp (= +0.04 progress each) yields:
16 × 0.04 / 40 = +1.6 % weighted progress = **+16 score**. That
exceeds the +5 commit gate by 3x with margin for noise. The
DSV3/Qwen3 path is K%128==0 (no K-tail), so it routes through the
forward main kernel + native RRR + var_k for dB — different
bottleneck profile than gpt_oss.

A future PMC pass on a DSV3/Qwen3 worst-shape (e.g.
DSV3-Down-B16-M4096 ratio 1.104) would identify the actual
bottleneck and make Lever D actionable. Currently uninstrumented
on the K%128==0 path.

## R15 recommended priority

**Lever A1** (forward HBM K-tail prefetch). Specific:

1. **Where**: `device_gemm_tile_body`, the
   `if constexpr (FUSED_KTAIL)` block (line ~829 onward), HK source.
2. **Concrete change**: hoist the K-tail `G::load(...)` calls
   from after-epilog-2 to start of epilog 1, gated on
   `FUSED_KTAIL && K_REM == K_STEP`. The K-tail data is
   stage-independent of epilog 2 (writes to slot `tic^1`), so
   issuing the load ~150 cycles earlier is correctness-preserving.
3. **Build gate**: -Rpass-analysis on
   `grouped_kernel<RCR, KI=*, FUSED_KTAIL=true>`. Expect VGPR
   delta within ±2 (the additional in-flight HBM reads cost some
   `vmcnt` tracking VGPRs but the prefetch itself doesn't consume
   register state).
4. **Numerics**: `check_allclose` on `out` for one gpt_oss-GateUP
   shape (any K%128 != 0 K-tail-touching shape).
5. **Affected shapes**: All 4 gpt_oss-Down + all 4 gpt_oss-GateUP
   forward calls (8/24 metric shapes), since K%128 != 0 only on
   gpt_oss family in the metric.
6. **Expected score**: +5 to +12 score if forward is the dominant
   gpt_oss-Down stage (R87 PMC suggests forward is HBM-bound; a
   K-tail prefetch buying ~10-20 % forward = ~3-5 % full-pass).

If A1 falsifies, R16+ proceeds to Lever B4. Lever C is FROZEN-list
contingent on the uniform-M kernel binding question above. Lever D
needs a DSV3/Qwen3 PMC round first (mirror of R87 for K%128==0).

## R91 commit summary

* **HipKittens**: no change.
* **Primus-Turbo**: this note. No code or kernel change.
* **Metric (R91 single run)**: 881 (within noise band 881-883).
  No commit-gate test needed — no code change.
* **Round 14/100 verdict**: structural pivot. No metric movement
  expected; commit gate `≥ +5` not applicable to docs round.
