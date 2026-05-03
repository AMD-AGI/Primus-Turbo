# Round-35 — BF16 grouped GEMM: chat-window consolidation (no change)

## Status
**DOCS-ONLY consolidation**. Chat session at ~88 min of 90 min window
— insufficient budget for paired 5-run (~3 min) + correctness probe
(~25 s) + new rule attempt + commit. Using this round to consolidate
R29-R34 findings for the R36 cold-start agent.

## Metric (baseline only, R35 has no change)
```
score = 879/1000  (HEAD 160a18d)

[metric_bf16_weighted] Per-family geomean (un-weighted):
  gpt_oss_20B         geomean_ratio = 1.0854  (target 1.25, weight 3x)
  DeepSeek-V3         geomean_ratio = 1.1220  (target 1.25, weight 1x)
  Qwen3-235B-A22B     geomean_ratio = 1.1159  (target 1.25, weight 1x)

Lowest-ratio shape: gpt_oss-GateUP-B32-M2048 @ ratio 1.048
```

## Cross-round outcome summary (R24-R34)
```
R24 (LAND): 4-rule dB var-K aggregate   → ACCEPTED (>+5)
R25:  Qwen3-GateUP single-family rule   → sub-noise
R26:  3-rule aggregate                  → noise
R27:  DSV3-GateUP dB var-K (gm=2, xcds=8) → +0.25% (sub-noise)
R28:  bf16_transpose_3d (BK,BN)=(128,128) K==N → +3.4 (sub-+5)
R29:  5-lever aggregate                 → -0.2 (noise)
R30:  R28 re-test + H4 wall profile     → non-reproducible
R31:  H4 gate tighten (bypass RRR)      → -79 (N-tail is SLOW PATH,
                                            H4 is a fast-path switch,
                                            NOT an overhead)
R32:  post-H4 bwd dA "tiles_n==12"      → noise (VACUOUS PREDICATE —
                                            tiles_n = n//256 = 11)
R33:  post-H4 bwd dA "tiles_n==12"(2,4) → noise (VACUOUS)
R34:  post-H4 bwd dA tiles_n==11,       → -1.6 (genuine firing rule,
      k==5760, (gm=1, xcds=4)               sub-noise)
R35:  docs-only, chat-window closing
```

## Confirmed facts (R36+ should treat as axioms)
1. **BF16 dispatch surface is fully exhausted** at current metric
   precision (stdev ~4, +5 gate). R34 tested the last uncovered
   gpt_oss-GateUP bwd dA post-H4 scope with a correctly-firing rule;
   still sub-noise.
2. **H4 transpose is a fast-path switch, NOT an overhead**. R31 data:
   bypassing it drops score -79. The native RRR N-tail path
   (`grouped_ntail` launcher for `N_kernel % 256 != 0`) runs at
   ~0.6× RCR fuse throughput on gpt_oss K=2880 shapes. H4 transpose
   costs ~7% wall but avoids this slowness.
3. **R28 BF16-specific transpose heuristic is non-reproducible**:
   R30 paired 5-run confirmed +3.4 was session-warmup bias.
4. **R32/R33 "rules" were VACUOUS** — `tiles_n = n // 256` is floor-
   div, so `tiles_n == 12` never matched (actual is 11). Their
   falsifications are pure noise data, NOT real tests. Discount them.
5. **Python-side dispatch overhead is trimmed** (R18, etc.).
   `grouped_gemm_impl.py::execute` body is tight. No more Python
   savings available.

## R36+ main line — C++ kernel work on native RRR N-tail

**Target file**: `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`

**Target launcher**: `grouped_ntail` (N_kernel % 256 != 0 path).
Specifically the RRR variant for gpt_oss shapes where `K_kernel = 2880
(K%128=64)` AND `N_kernel % 256 != 0`.

**Current state**: R31 bench showed this path at 0.6× RCR fuse
throughput — that's why H4 transpose (reroute to RCR) is the fast path.

**Fix hypothesis** (pick ONE per round):
- (a) **VGPR spill** in the wider epilogue — check occupancy
  via `--save-temps` + `llvm-readobj` on the compiled .hsaco.
- (b) **Masked store / divergent branch** on the tail tile
  (the final N block that covers < 256 columns). May benefit
  from a separate specialized tail kernel.
- (c) **Sub-optimal tile schedule** — the persistent launcher
  may be issuing tail tiles LAST where they can't overlap with
  compute-bound full tiles. A reorder might help.

**Workflow**:
1. Read `kernel_bf16_dynamic.cpp`, find `grouped_ntail` launcher.
2. Pick (a), (b), or (c). Small, focused edit.
3. `cd /workspace/code/HipKittens/analysis/bf16_gemm/mi350x`
   `make -j8 tk_bf16_layouts.so 2>&1 | tail -20`
4. Correctness: `python3 /tmp/probe_round29_correctness.py` (9 shapes,
   fwd+bwd vs Triton ref).
5. Metric: `python3 scripts/_metric_grouped_bf16_weighted_wall.py`
   → compare to 879 baseline.
6. Paired 5-run if single run looks promising.
7. **Commit on both repos** if ≥ +5 AND correctness PASS.
8. Expected headroom: drop H4 entirely (save ~7% wall on gpt_oss
   fwd) → score +30-40 (gpt_oss geomean 1.086 → ~1.15).

## R36+ fallback — (gm, xcds) cell sweep at R34 scope (LOW upside)
If C++ blocked, R34's `tiles_n==11 ∧ k==5760` scope still has
cells (2,4), (4,4), (8,4), (16,4) untested. Each needs paired 5-run
(~3 min). Upside is probably ≤ +3 since (1,4) was -1.6 — cell may
just prefer the (4,8) default.

## Chat window + commit discipline
R35 commit = docs-only. R36+ is **cold-start** (new chat session, no
prior file reads visible). Must infer from git log + this
consolidation note.
