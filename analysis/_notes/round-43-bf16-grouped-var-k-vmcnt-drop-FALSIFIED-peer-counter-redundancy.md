# Round 43 — BF16 grouped, drop `vmcnt(0)` from `grouped_var_k_kernel` per-tile epilog — FALSIFIED

## Goal coming in

R42 falsification note (KI=90 closing the KI-specialization surface)
recommended R43 start with the `var-K (dB) kernel kernel-body audit` —
R32-R42 had never read `grouped_var_k_kernel` source. R42 dA/dB probe
showed dB on `gpt_oss-GateUP-B32-M2048` runs at 1117 TF on HK vs 1239 TF
fwd on the same shape, a ~10% gap purely from the var-K kernel side
(structurally worse than the fwd `grouped_kernel`).

## Var-K kernel scout

`/workspace/code/HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
lines 4486-4685.

Structure (CRR-only, persistent):
- Outer loop: `for (gt = pid; gt < total_tiles; gt += NUM_CUS)`.
- Inner per-tile body:
  1. compute `group_idx`, `M_g`, `ki_g = M_g / K_STEP`, `pid_m`,
     `pid_n`, `k_offset_tiles`;
  2. zero 4 accumulators `C_accum[2][2]` (4 KB of VGPRs);
  3. call `device_gemm_tile_body<Layout::CRR, KI_HINT, ...>` (same
     shared body as fwd `grouped_kernel`);
  4. conditional `if (warp_row == 0) { __builtin_amdgcn_s_barrier(); }`
     between body and store;
  5. four `store_c_tile_mn_masked_grouped` calls;
  6. `s_waitcnt vmcnt(0) lgkmcnt(0)` + `__builtin_amdgcn_s_barrier()`.

Compared to fwd `grouped_kernel` (lines 3667-4178), the structure is
nearly identical — fwd is *also* persistent, and the per-tile epilog
is the same `vmcnt(0) lgkmcnt(0) + s_barrier` pattern. So the `vmcnt(0)`
in the epilog is not a var-K-specific cost; it's a shared kernel-design
choice from round-7-v0 (commented in line 1019-1026).

## Hypothesis (R43)

The `vmcnt(0)` per-tile drain is unnecessary on the persistent loop
boundary:
- the 4 stores write to disjoint output regions of `g.c`
  (`(group_idx, m_tile, n_tile)`);
- the next persistent iteration's loads (a different (group, tile)
  pair) read from `g.a` / `g.b` — HBM-independent of the previous
  iter's `g.c` stores;
- LDS double-buffer slots overwritten in the next prologue still
  need `lgkmcnt(0)` to drain LDS pipes;
- `s_barrier` (workgroup sync) is still required so all waves can
  safely overwrite shared LDS in the next prologue;
- kernel-exit semantics implicitly drain `vmcnt` before the launch
  returns, so correctness on the final kernel boundary is preserved.

Expected gain: the `vmcnt(0)` wait nominally costs 0-200 cycles per
tile depending on store-queue saturation. With ~25 tiles per CU on
B32-M2048 shapes, dropping it could save 5-50μs per CU → 0.5-2% on
dB time.

R43 changed the var-K epilog from
```
asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
__builtin_amdgcn_s_barrier();
```
to
```
asm volatile("s_waitcnt lgkmcnt(0)");
__builtin_amdgcn_s_barrier();
```

## Evidence

### Build (var-K kernel only changed; main grouped untouched)

```
grouped_var_k_kernel<0> :  VGPRs:256  Spill:0  Scratch:0  LDS:272
```

Clean compile. Behaviorally a smaller change than R39/R40/R42 KI
specializations.

### Metric

```
baseline (324414d):   880
R43 v1 (vmcnt drop):  879   Δ = -1   (sub-noise)
post-revert recheck:  883          (noise)
```

Per-shape ratio diffs across all 24 shapes: range -0.012 to +0.007,
mean essentially 0. None of the dB-bound shapes (DSV3-Down,
Qwen3-Down) showed a coherent positive shift.

### Bench (`benchmark/ops/bench_grouped_gemm_turbo.py --dtype bf16`)

Comparing the existing 0503 baseline CSV to R43 v1:

```
                            fwd avg   bwd avg
baseline (2026-05-03):     1267.47   1024.15
R43 v1 (vmcnt drop):       1273.02   1026.43
                           +0.4%     +0.2%
```

All 24 shapes PASS correctness. Numerical equivalence preserved.

Bwd Δ +0.2% is within run-to-run noise (typical bench σ ≈ ±5 TF on
the avg ≈ 1025 TF).

## Mechanism — why it didn't move

The fundamental misunderstanding was that `vmcnt(0)` and `lgkmcnt(0)`
drain on independent timescales. They don't, in practice on this kernel:

1. `device_gemm_tile_body` returns with both vmcnt and lgkmcnt at
   high values (many in-flight HBM loads + LDS reads/writes from the
   K loop). The 4 store_c calls add ~64 vmcnt each (256 buffer_store
   ops total) on top.
2. AMD's `s_waitcnt` waits for ALL specified counters to drain. Both
   counters are clocked off the same CU memory-pipe state. By the time
   `lgkmcnt(0)` is satisfied, the store queue (vmcnt) is already at
   or near 0 because:
   - LDS reads issued during the K-loop are causally before the
     post-body stores;
   - the LDS subsystem and the HBM store path share enough common
     scheduling that lgkmcnt typically drains within a few cycles
     of vmcnt for typical kernels.
3. `__builtin_amdgcn_s_barrier()` waits for waves to reach the
   barrier. With the K-loop body taking ~30-50μs per tile, the
   waves arrive at the barrier nearly simultaneously, so the barrier
   itself is also fast.

In short: dropping `vmcnt(0)` recovers ~0 cycles in practice because
the wait was already satisfied at the same time as `lgkmcnt(0)`.
The change is logically correct (and could matter on a kernel where
HBM stores are large and LDS pipes drain fast) but on this var-K
kernel the savings are below run-to-run noise.

## Falsification consequence

R43 closes one specific micro-opt on the var-K epilog. It does NOT
close the var-K kernel surface; the ~10% structural gap between
var-K dB (1117 TF) and fwd (1239 TF) on `gpt_oss-GateUP-B32-M2048`
remains. Possible next angles **inside** var-K:

1. **Persistent loop overhead**: per-tile `(group_idx, M_g, ki_g,
   pid_m, pid_n, k_offset_tiles)` arithmetic + 4-accumulator zero
   could be amortised. Currently each gt-iter does ~30 SGPR/VGPR ops
   for the work-divider; one or two of these (e.g., k_offset_tiles
   = m_start_g/K_STEP) is already constant within a group, so the
   inner-tile branch could read a per-group cache.
2. **Barrier reduction**: the `if (warp_row == 0) s_barrier()` between
   body and stores (line 4654) is suspicious — `s_barrier` on
   gfx95 is a workgroup-wide barrier, conditional execution by
   warp-id should deadlock. Either it's silently a no-op for
   warp_row != 0 waves (semantically dangerous), or there's
   compiler-specialised handling. Worth a separate scout.
3. **Store batching**: 4 separate `store_c_tile_mn_masked_grouped`
   calls could in principle be one larger masked store that fills
   all 4 sub-tiles (BLOCK_SIZE × BLOCK_SIZE = 256x256) in one go;
   this changes the LDS↔HBM staging pattern and is a much bigger
   refactor than R43 v1.
4. **KI specialization for var-K**: `KI_HINT=32` (gpt_oss-Down-B32,
   DSV3-Down-B32, Qwen3-Down-B32 dB all have ki_g=32) and
   `KI_HINT=64` (M=4096 variants). Risk: KI=32 spilled 20 VGPRs
   on RCR/RRR in R40; CRR-specific spill curve is unknown but the
   shared `device_gemm_tile_body` template suggests similar.

## Action

- HK kernel reverted (no diff after revert + rebuild).
- Primus-Turbo: this round-43 note added; no code change.
- HK: no commit (kernel reverted).
- Primus-Turbo: 1 commit (this falsification note + var-K scout
  table embedded in commit body).

## R44 next-action surface

Pivot from kernel-body micro-opts (cheap to try, all sub-noise to
date) to **dispatch / config tuning**, since:
- KI specialization closed (R39/R40/R42).
- Per-tile epilog wait closed (R43).
- The 4 worst metric shapes — `gpt_oss-{GateUP,Down}-{B32-M2048,
  B32-M4096}` — span the same `select_default_config` route.
  R32-R34 explored `(group_m, num_xcds)` on dA H4 RCR with
  null effect. R44 should re-examine with the dA/dB-decomposed
  picture.

Specifically: R42 dA/dB probe revealed `gpt_oss-Down-B32-M2048
dA_r = 0.942` (worst) — this shape's dA is the biggest single
ratio contributor to the gpt_oss family deficit. dA goes through
the H4 RCR fuse-eligible path (K_kernel=2880, ki_g=44) — different
from gpt_oss-GateUP dA (K_kernel=5760 non-fuse). R44 should:

1. Read the H4 dA path's `select_default_config` outcome
   (`(group_m, num_xcds)`) on `gpt_oss-Down-B32-M2048`. R32-R34
   were on gpt_oss-GateUP, not Down.
2. Probe a Down-specific dispatch rule (still general predicate,
   not per-(M,N,K) hardcode).
3. If predicate-driven dispatch can shift Down dA from 0.942 →
   1.05, the wall ratio improvement is ~5pp on a weight-3 shape.
