# MXFP8 Backward — Performance & Correctness Status

**Date:** 2026-08-01 (all numbers measured this session)
**Device:** MI355X (`gfx950`, 256 CU, 8 XCD), node `smci355-ccs-aus-n04-33`, container `xiaoming-dev`
**Model shape:** DeepSeek-V3 — H=7168, I=2048, E=256, K=8, EP8 intra-node, T=8192/rank
**Code:** `feat/xiaompen/mega_moe_flydsl_mxfp8` @ `bbb5a85e` + uncommitted campaign work
**Bench:** `benchmark/ops/bench_mega_moe_bwd_only.py`, CUDA events, `_amax` across ranks, warmup=8 / iters=25

Companion to `mxfp8_fwd_breakdown_note.md` (forward). Campaign log:
`agent/workspace/mega_moe_combine_reduce_flydsl_gfx950_20260801/logs/`.

---

## 1. Current performance

Three legs share persistent leaf tensors, so the fp8 weight-quant cache hits across iters (mirrors
training). `bwd-only` times `y.backward(grad_y)` with the forward outside the event window.

| Routing | dtype | fwd+bwd | fwd-only | bwd-only |
|---|---|---|---|---|
| load_balanced | fp8 | **14.11** | 5.38 | **8.60** |
| load_balanced | bf16 | 20.47 | 7.23 | 13.91 |
| load_balanced | bf16/fp8 | 1.45× | 1.34× | **1.62×** |
| round_robin | fp8 | **13.42** | 5.12 | **8.28** |
| round_robin | bf16 | 19.77 | 7.01 | 13.43 |
| round_robin | bf16/fp8 | 1.47× | 1.37× | **1.62×** |

The backward is where fp8 pays off most (1.62× vs 1.34× on the forward), and the ratio is identical
under a heavily skewed routing, so the speedup does not depend on balanced experts.

Cross-check `fwd-only + bwd-only` vs `fwd+bwd`: fp8 13.98 vs 14.11 (+1.0%), i.e. the three legs are
self-consistent and nothing is hidden between them.

## 2. Backward internal structure

From the campaign's round-1/2 budget (see `logs/optimize.md`); not re-measured this session:

| Leg | Time (ms) | Share | Note |
|---|---|---|---|
| STEP3 combine (fc1-dgrad GEMM + PUSH + reduce) | ~2.69 | ~31% | PUSH is at the XGMI limit |
| dW1 GEMM (variable-K, LOCAL) | ~2.08 | ~24% | ~39% of peak |
| L2 dgrad + dW2 + quant | remainder | — | includes ~0.43 ms fused dual-quant |

STEP3's PUSH leg moves ~484 MB/rank in ~1.97 ms ≈ 246 GB/s/rank ≈ 1.97 TB/s aggregate = the XGMI
ceiling. Rounds 3 and 5 established this independently, so **the remaining lever there is fewer
bytes, not better scheduling or a different CU split**. The largest single traffic item is the local
round trip where the GEMM epilogue writes 469 MB to L2Y and the PUSH reads the same 469 MB back.

## 3. Correctness — nondeterministic by default, and the fix is free

> **Correction (round 7).** Section 3 below conflated two independent things. Its *bitwise*
> determinism data stands. Its accuracy claim does not: the swinging dx SNR was never the fp8
> path's fault. Scored against an **analytic** reference instead of the bf16 op, fp8 measures
> y=22.3, dx=21.9, d_topk_w=23.1, dW1=19.5, dW2=19.7 dB — reproducible to 0.1 dB across runs, at
> `PT_COMBINE_GATE_DELAY=0`, with zero catastrophic rows. The old 9.1–19.7 dB spread came from the
> **bf16 reference** corrupting 1–30 of its own dx rows per step (4–9x inflation, varying ranks);
> its forward is clean, only its backward combine is affected. So the fp8 combine's bitwise
> nondeterminism sits at the quantization-noise level and does **not** cost accuracy, and
> `GATE_DELAY` is a reproducibility knob, not an accuracy fix. The gate now uses the analytic
> reference (`benchmark/ops/bench_mega_moe_fused_fp8_bwd.py`); the bf16 op is kept only for timing.

### 3.1 The defect

With identical seeded inputs, 6 runs compared bitwise against run 0
(`rounds/round-6/artifacts/check_determinism.py`):

| Output | rows differing | max\|Δ\| |
|---|---|---|
| `y` (forward L2 combine) | 23.2% (T=2048), 13.1% (T=8192) | 4.1e-01 |
| `dx` (backward STEP3 combine) | 18.5% (T=2048), 11.3% (T=8192) | 3.4e+01 |
| `d_topk_w` | deterministic | 0 |
| `dW1` / `dW2` | 100% | 3.1e-02 / 1.6e-02 |

The magnitudes rule out rounding: this is corruption, not imprecision. Round 6 localized it with a
permutation-invariant stage bisect — the whole L1 dispatch → L1 GEMM → SwiGLU → L2 GEMM chain is
bitwise deterministic, and divergence is introduced at the combine PUSH / `comb` handoff, where the
GEMM-done gate releases the push before L2Y is readable.

**The bf16-referenced SNR gate was not a correctness check.** Seven runs of one unchanged tree gave
dx SNR 9.1–19.7 dB (10.6 dB spread, 3 of 7 below the 15 dB threshold). Round 7 traced that spread to
the reference rather than the fp8 path — see the correction above.

### 3.2 A post-gate stall closed it

> **Removed.** `PT_COMBINE_GATE_DELAY` has been deleted from the kernel. Once round 7 showed it
> buys bitwise reproducibility but not accuracy, an off-by-default knob that nobody should enable
> was not worth carrying. The measurements below stand as the evidence that the gate releases the
> push too early; reinstate the stall from git history if you pick up that root-cause work.

The knob stalled `N × s_sleep(127)` after the GEMM-done gate, before the push read L2Y.
Determinism vs N (T=2048, runs=6, rows differing):

| N | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| `y` | 23.2% | 17.0% | 2.5% | **0** | **0** | **0** | **0** |
| `dx` | 18.5% | 14.0% | 0.4% | **0** | **0** | **0** | **0** |

N=4 is the minimum clean value. Because that is only 2× the last failing value, it was stressed
further: **T=2048 runs=12 and T=8192 runs=6 both give `diff_elems=0`** for `y` and `dx`, while the
`N=0` control at T=8192 still fails (13.1% / 11.3%).

### 3.3 The fix costs nothing measurable

Interleaved A/B, order `0 → 4 → 8 → 4 → 0`, T=8192, `--only fp8`, means of repeated arms:

| | N=0 (incorrect) | N=4 | N=8 |
|---|---|---|---|
| LB fwd+bwd | 14.064 | 14.133 (+0.49%) | 14.165 (+0.72%) |
| LB bwd-only | 8.634 | 8.649 (+0.17%) | 8.652 (+0.21%) |
| RR fwd+bwd | 13.510 | 13.530 (+0.15%) | 13.540 (+0.22%) |
| RR bwd-only | 8.272 | 8.296 (+0.28%) | 8.295 (+0.28%) |

**Read these as "below the measurement floor", not as measured costs.** The spread between two runs
of the *same* arm (N=4 LB `fwd-only` 5.479 vs 5.391 = 1.6%; `bwd-only` 8.695 vs 8.602 = 1.1%) exceeds
every arm-to-arm difference in the table. Corroborating: N=8 must cost at least as much as N=4
physically, yet measures *lower* on LB `fwd-only` — a real effect cannot invert.

This also corrects an earlier reading. `N=32` costs +9.4% on LB fwd+bwd (15.39 vs 14.06, reproduced
twice), which suggested correctness was expensive. It is not — 32 is 8× more stall than needed, and
the entire 9.4% was waste.

N=8 would have been the sane setting (4× margin at the same unmeasurable cost) had the stall been
kept as a default.

### 3.4 What the stall does *not* fix

`dW1`/`dW2` stay nondeterministic at every N including 32, at a magnitude of one bf16 ULP
(3.1e-02 / 1.6e-02). This is a **second, independent source**: pool rows are assigned by two levels
of atomics (per-block LDS `atomic_add` over a cross-block global `atomic_add`), so the row
permutation differs every run and the variable-K grouped GEMM accumulates in a different order. It
was masked by the combine's much larger corruption until the stall removed that. Almost certainly
benign for training, but it should be tracked separately.

### 3.5 Status of the stall as a fix

It is a **mitigation, not a root-cause fix** — it closes the race window by wall-clock separation
rather than by a correct release/acquire pairing. Round 6 empirically eliminated the entire
memory-visibility family first: 12 hypotheses including acquire strength (`PT_MXFP8_GEMM_INV=1`),
pool-scale read strategy, `cache_modifier=19` on the payload, coherent L2Y reads, paired cross-rank
release/acquire, flag scope `agent`→`sys`, and `s_waitcnt` after `buffer_inv`. The corrupted fraction
was pinned at ~24% across arms with substantially different timing, and `l2_writeback()` on the
GEMM-role release made it *worse* (24% → 38%). A genuine visibility bug would have moved.

## 4. Repro

```bash
docker exec xiaoming-dev bash -lc '
cd /perf_apps/xiaoming/MegaMoE
export PYTHONPATH=/perf_apps/xiaoming/MegaMoE MEGA_BENCH_TIMEOUT_S=1800
W=agent/workspace/mega_moe_combine_reduce_flydsl_gfx950_20260801/rounds/round-6/artifacts

# performance (fp8 vs bf16, both routings)
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_mega_moe_bwd_only.py \
  --num-processes 8 --num-tokens 8192 --routing-mode both --only both --warmup 8 --iters 25

# accuracy, against the analytic reference (no bf16 in the verdict)
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_mega_moe_fused_fp8_bwd.py \
  --num-processes 8 --num-tokens 2048

# bitwise determinism (vs run 0)
MASTER_PORT=$((20000+RANDOM%20000)) python3 $W/check_determinism.py \
  --num-processes 8 --num-tokens 2048 --runs 6
'
```

Both are 8-process spawns; kill the whole tree on interrupt or the ranks keep holding VRAM.

## 5. Open items

| P | Item | Notes |
|---|---|---|
| **P1** | root-cause the GEMM→push gate | costs reproducibility, not accuracy (round 7); the stall that proved it is removed, visibility family already ruled out |
| **P2** | track `dW1`/`dW2` accumulation-order nondeterminism | independent source, one bf16 ULP |
| **P3** | kill the L2Y local round trip in STEP3 | 469 MB written + 469 MB re-read; the only lever that attacks the XGMI/HBM floor |
| **P4** | group-aware L2 swizzle for the combine GEMM | plain `GROUP_M` degrades monotonically (round 4) — groups span per-expert B slabs |
| — | `num_combine_cu` re-sweep | exhausted (round 3): 28 optimal, PUSH is traffic-bound |
| — | segment-driven PUSH | refuted (round 5): loses ILP from the compile-time trip count |
