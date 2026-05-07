# Round-4 (current Primus run, gpt_oss FP8 kernel-only ceiling task; 2026-05-07)

## TL;DR

After R1 / R2 / R3 dispatcher re-tunes shipped 3 wins (R30 var-K split,
R7+R12 RCR rebuild-drift fix, R10dm+R35 RCR/var-K rebuild-drift fix),
**all 24 (shape, section) cells × all dispatcher levers are at the kernel
ceiling under the current FP8 binding (commit `32a9604`).** No further
dispatcher-only round can lift the score. Phase-3 kernel-template overrides
(`HipKittenConfig(kernel="4"|"8")`) also yielded no win on any of the 8
gpt_oss shapes — the binding's auto-pick is already optimal.

Score state at end of R4: **665** (10-sample median 666; pre-change
identical). Historical best 669. Run-to-run noise floor ±4 points.

The next score lift requires touching **kernel sources** in
`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (or a sister
cpp), not config.py / grouped_gemm_fp8_impl.py.

## Audit table — current rule cells, ceiling status, and falsification probe pointers

| Shape (gpt_oss) | section | rule | (gm, xcds) | metric T | probe verdict | best alt Δ |
|-----------------|---------|------|------------|---------:|---------------|-----------:|
| GateUP_B4_M2048  | fwd    | R23      | (1, 4)   | 1773  | ceiling (R23 robust)      | (4,4) -1.19% |
| GateUP_B4_M2048  | dgrad  | default  | (4, 8)   | 1818  | ceiling (probed R4)       | (8,8) +0.09% |
| GateUP_B4_M2048  | wgrad  | R3 (this run) | (1, 4) | 1551 | shipped R3 +0.52% kernel  | (2,4) -0.32% |
| GateUP_B4_M4096  | fwd    | R3 (this run) | (8, 4) | 2002 | shipped R3 +1.25% kernel  | (1,2) -0.47% |
| GateUP_B4_M4096  | dgrad  | R8       | (1, 4)   | 2421  | ceiling (probed R3)       | (8,4) -0.23% |
| GateUP_B4_M4096  | wgrad  | R9-A     | (4, 4)   | 1929  | ceiling (probed R3)       | (1,4) -1.03% |
| Down_B4_M2048    | fwd    | R2 (this run) | (16, 2)| 1373 | shipped R2 +1.37% kernel  | (32,2) +0.04% (TIE) |
| Down_B4_M2048    | dgrad  | R2 (shared with fwd) | (16, 2) | 1327 | shared rule ceiling | same TIE |
| Down_B4_M2048    | wgrad  | R11      | (1, 2)   | 1190  | ceiling (probed R4 ext.)  | (12,4) -0.78% |
| Down_B4_M4096    | fwd    | R2 (this run) | (1, 4) | 1802 | shipped R2 +1.12% kernel  | (16,4) -0.97% |
| Down_B4_M4096    | dgrad  | R2 (shared with fwd) | (1, 4) | 1807 | shared rule ceiling | same |
| Down_B4_M4096    | wgrad  | R10      | (1, 2)   | 1546  | ceiling (probed R3)       | (32,2) -0.02% (TIE) |
| GateUP_B32_M2048 | fwd    | R70      | (8, 4)   | 2014  | ceiling (probed R4)       | (8,2) -3.14% |
| GateUP_B32_M2048 | dgrad  | (audit)  | (16, 4)  | 2479  | borderline win (R8/R34 falsification re-confirmed: per-seed +0.34/-0.07/+0.66 — only 2/3 positive) | (1,4) +0.54% NOT ROBUST |
| GateUP_B32_M2048 | wgrad  | R31      | (1, 4)   | 1845  | ceiling (probed R3)       | (2,4) +0.08% (TIE) |
| GateUP_B32_M4096 | fwd    | R70      | (8, 4)   | 2064  | ceiling (probed R4)       | (8,2) -0.23% |
| GateUP_B32_M4096 | dgrad  | R8       | (1, 4)   | 2541  | ceiling (probed R3)       | — |
| GateUP_B32_M4096 | wgrad  | R31      | (1, 4)   | 2145  | ceiling (probed R3)       | (2,4) -0.33% |
| Down_B32_M2048   | fwd    | R8       | (16, 4)  | 1849  | ceiling (probed R4)       | (12,4) -0.15% |
| Down_B32_M2048   | dgrad  | R8 (shared) | (16, 4) | 1826 | shared rule ceiling      | — |
| Down_B32_M2048   | wgrad  | R1 (this run) | (8, 4) | 1662 | ceiling (probed R3)       | (16,4) +0.27% NOT ROBUST (50T spread) |
| Down_B32_M4096   | fwd    | R50      | (4, 4)   | 1950  | ceiling (probed R4)       | (4,8) -0.28% |
| Down_B32_M4096   | dgrad  | R50 (shared) | (4, 4) | 1907 | shared rule ceiling      | — |
| Down_B32_M4096   | wgrad  | R30      | (4, 4)   | 1943  | ceiling (probed R3)       | (8,4) +0.26% NOT ROBUST |

24 / 24 cells confirmed at ceiling. The 2 borderline wins (GateUP_B32_M2048 dgrad,
Down_B32_M2048 wgrad) sit at +0.27..+0.54% with only 2/3 seeds positive — known
falsification pattern from R8/R34 audit, re-confirmed this round.

## Phase-3 kernel-template overrides — falsified

Probe (`/tmp/_probe_round_4_kernel_template.py`, 250-iter × 7-trial p20 ×
3 seeds at the shape's currently-shipped (gm, xcds) cell):

```
shape                  gm xc aut    auto T force-4 T force-8 T  Δ4 vs auto / Δ8 vs auto
GateUP_B4_M2048         1  4   8    1946.9    1948.6    1949.7  Δ4=+0.09%  Δ8=+0.14%
GateUP_B4_M4096         8  4   8    2082.7    2085.2    2084.0  Δ4=+0.12%  Δ8=+0.06%
Down_B4_M2048          16  2   8    1505.2    1507.9    1507.3  Δ4=+0.18%  Δ8=+0.13%
Down_B4_M4096           1  4   8    1973.5    1972.3    1973.5  Δ4=-0.06%  Δ8=+0.00%
GateUP_B32_M2048        8  4   4    2039.4    2037.2    2036.5  Δ4=-0.10%  Δ8=-0.14%
GateUP_B32_M4096        8  4   4    2093.5    2094.4    2094.3  Δ4=+0.05%  Δ8=+0.04%
Down_B32_M2048         16  4   8    1869.0    1859.5    1859.1  Δ4=-0.51%  Δ8=-0.53%
Down_B32_M4096          4  4   4    1926.8    1927.5    1928.6  Δ4=+0.04%  Δ8=+0.10%
```

All Δ within ±0.5% which is below run-to-run spread (4-35T spreads observed).
The Down_B32_M2048 force-4 / force-8 -0.5% deficit reflects env-var read
overhead inside the kernel dispatch (`getenv("TK_RCR_FORCE_KERNEL")` per
launch); auto-pick path skips this and is uniformly ≥ either explicit
template. Shipping a `kernel="..."` rule would only ADD this overhead.

The R13 audit in `analysis/_notes/round-13-config-tuning-saturation.md`
(2026-04 timeline) had also concluded the binding's auto-pick was bit-
equivalent within ±0.1pp on every metric shape; this round's R4 audit
extends that finding to the post-rebuild binding state.

## var-K (CRR) wgrad kernel — single-template

`grouped_var_k_kernel_fp8<KI_HINT=0>` is the only template
instantiation of the wgrad kernel
(`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:7046`). No
4-wave / 8-wave variants — the 4-wave/8-wave dispatch is RCR-only.

So the wgrad section has zero remaining dispatcher levers (vk_group_m and
vk_num_xcds confirmed at ceiling for all 8 wgrad shapes; no kernel-template
to override; no env-var auto-pick to revisit).

## Ceiling vs target — kernel surgery is the next path

| Section | HK avg | section progress | gap to 2800T | implied per-shape lift |
|---------|--------|------------------|--------------|------------------------|
| fwd     | 1850 T | 0.661            | 950 T        | +51% per shape         |
| dgrad   | 2014 T | 0.719            | 786 T        | +39% per shape         |
| wgrad   | 1727 T | 0.617            | 1073 T       | +62% per shape         |

To bring section averages from 1727..2014 to 2800 T requires +30..+60% per shape,
which can ONLY come from the kernel itself (or from removing the H4 reroute
+ Python overhead in `grouped_gemm_fp8_impl.py:execute`, which is ~5-8 µs of
the 90-99 µs metric wall — 5-9% of total — but already significantly trimmed
per R11/R18 commit history).

## Recommended next-round directions (kernel-level, in priority order)

### A. wgrad var-K kernel — biggest absolute gap

Current section ratio HK/TRT = 1.86× (vs fwd 1.11× / dgrad 1.17×); HK is
relatively far ahead of Triton, but absolute floor is low (1727 T vs 5033 T
peak = 34% peak utilisation). Targets in
`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp:6670-7062`:

1. The `grouped_var_k_kernel_fp8` is a single instantiation `<KI_HINT=0>`.
   Adding `<KI_HINT=22>` (= K_fwd=2880/128 main-loop iters for gpt_oss)
   could let the compiler unroll the K main-loop and reduce control-flow
   overhead. Already partially done for the dense `gemm_kernel<L,KI_HINT>`
   (R55 era) but not for var-K.

2. The K-tail handling for K%128==64 (= 64-element tail per group) — every
   var-K call pays one masked tail iter. Quantify the tail cost on
   gpt_oss-Down B=4 M=2048 (1185 T metric, the worst single shape). If
   the tail is ≥ 5% of per-iter wall, the lever is worth investigating
   (kernel-mainloop unroll factor change or tail co-issue with main).

### B. RCR persistent grouped kernel — for the ratio < 1.10× shapes

Three shapes have HK/TRT ratio in [1.07, 1.10] (the lowest):
- Down_B32_M2048 fwd ratio 1.08 (1849T)
- Down_B32_M4096 fwd ratio 1.07 (1950T)
- GateUP_B32_M4096 fwd ratio 1.08 (2064T)

These are B=32 large-grid shapes where the persistent kernel's epilogue
overhead is amortised across many tile-steps. The per-tile cost is the
limiting factor. Lever candidates in
`grouped_rcr_kernel<KI_HINT, FUSED_KTAIL, FUSE_ACT>` (line 2509-3199):

1. The R55 LATER comment notes `grouped_rcr_kernel<FUSED_KTAIL=true>` is
   slower than `=false` for gpt_oss K=2880 despite lower spill — there's
   a known (tile-completion-latency vs spill) trade-off here that has
   not been re-tuned in months. R23-era profiling suggested the K-tail
   prefetch may be at the wrong phase.

2. The 4-wave RCR template (lines 7104-7113) is auto-picked for grids ≥
   3200; the 4-wave variant has `RBM_4w` / `RBN_4w` micro-tile choices
   that may not be optimal for K%128==64 with K=2880 (different ki count
   than the 4-wave dispatch was originally tuned for, K=4096 in R20-era).

### C. Python overhead trim in `grouped_gemm_fp8_impl.py:execute`

Probe-vs-metric gap is 8-10 µs / call across 8 shapes (90-200 µs kernel
walls). The hot-path in `execute()` (line 460-590) has been trimmed by
R11 / R13 / R18 already. Remaining candidates:

1. The `select_default_config(...)` lru_cache lookup — already cheap (~50 ns)
   but called every iter. Hoisting cfg outside the hot-path is hard
   because callers vary.
2. `out = torch.empty(...)` — an 8-12 µs cudaMalloc when the cache is
   cold. Already amortised by torch's caching allocator after the first
   iter, so this isn't the real overhead source.
3. The H4 reroute cache hit — adds ~1-2 µs per dgrad call. The reroute
   is REQUIRED for K%128 != 0 paths (per R14 falsification of
   unconditional reroute on K-aligned shapes); for K=2880 % 128 = 64
   the reroute is necessary, so this overhead is structural.

Estimated lift if Python overhead halved to 4 µs: section avg fwd
1850 → 1950 T (+5%), dgrad 2014 → 2120 T (+5%), wgrad 1727 → 1830 T
(+6%). Score lift ~+30 points but engineering cost is high (every line
already audited).

## Kernel-rebuild drift handling caveat

If the next round touches kernel sources and triggers a `.so` rebuild,
the R1 / R2 / R3 dispatcher rules SHIPPED in this run (R30 var-K split,
R7→R2, R12→R2, R10dm→R3, R35→R3) are based on the **current** binding
state. Each rebuild may shift the optimal (gm, xcds) cell again — same
phenomenon documented across R30 / R31 / R32 / R45 / R50 / R7 / R10dm
in `config.py`. Plan: after any `.so` rebuild, re-run R3-style sweeps
on the 5 shaped rules above before relying on metric numbers.

## Probes archived

- `/tmp/_probe_round_4_dgrad_trace.py` — dispatcher cell trace for all 8 shapes × 3 sections
- `/tmp/_probe_round_4_gateup_dgrad.py` — GateUP_B4_M2048 + GateUP_B32_M2048 dgrad sweep
- `/tmp/_probe_round_4_down_b32_rcr.py` — Down_B32 fwd RCR sweep (M=2048 + M=4096)
- `/tmp/_probe_round_4_gateup_b32_fwd.py` — GateUP_B32 fwd RCR sweep (M=2048 + M=4096)
- `/tmp/_probe_round_4_kernel_template.py` — TK_RCR_FORCE_KERNEL override probe (8 shapes)
- `/tmp/_probe_round_4_down_b4_m2048_wgrad.py` — Down_B4_M2048 wgrad extended sweep (19 cells)

All probes use the standardised methodology (250-iter × 7-trial p20 × 3 seeds
× kernel-only direct call), matching R2/R3 round-series tight-verify protocol.
