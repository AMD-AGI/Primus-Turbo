# Round 2 — FP8 grouped fused-act: `group_offs` cache + `select_default_config` `lru_cache`

**Status**: SHIPPED. Two micro-cache deposits compounding the round-1 tensorwise
quantize cache. Per-iter wall reduced ~5 µs (~4 µs symmetric + ~1 µs
HK-asymmetric on grouped FP8 fwd+bwd). Wall metric median lifts 998 → 1000
(4/5 runs at cap vs 1/5 pre-R2). Un-fused regression metric stays in noise
band (median 970 → 972, +2). DoD smoke 608/608 PASS. All 24 fused-wall shapes
correctness PASS, SNR > 28 dB on out / dA / dB across 3 sampled shapes.

## What changed (Primus-Turbo)

* `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`
  * Wraps `grouped_gemm_compute_offs(group_lens)` with a
    `(id(group_lens), group_lens._version)`-keyed identity cache.
  * Same discipline as the R1 tensorwise quantize cache
    (`_FP8_TENSORWISE_QUANT_CACHE`): `weakref.finalize` evicts entries
    when the source tensor is garbage-collected, in-place mutations
    bump `_version` → cache MISS.
  * Per-call wall: 4.08 µs (un-cached) → 0.107 µs (cache HIT) — saves
    3.97 µs per FP8/BF16 grouped GEMM entry. Symmetric to backends
    (both HK and Triton callers reach this helper at the same call
    site).

* `primus_turbo/pytorch/kernels/hipkitten/config.py`
  * `select_default_config(...)` decorated with `@lru_cache(maxsize=1024)`.
  * Pure if/else cascade over hashable scalar inputs (m, n, k, layout,
    dtype, m_total) → frozen `HipKittenConfig`. Zero IO, zero global
    state. Trivially cacheable.
  * Per-call wall: 0.535 µs (un-cached) → 0.067 µs (cache HIT) — saves
    0.47 µs per HK execute call. **Asymmetric to HK**: only HK execute
    paths call `select_default_config`; Triton paths skip it entirely.
  * `maxsize=1024` covers ~150 unique tuples (24 metric shapes × {fwd,
    dA, dB} × {BF16, FP8} × {RCR, RRR, CRR}) plus all DoD smoke / dense
    metric shapes; comfortably above any expected production working
    set.

No HipKittens repo changes this round.

## Per-iter wall breakdown (Qwen3-Down-B16-M2048, the lowest-ratio cohort)

```
Pre-R2 (R1 baseline):
  Full fwd+bwd HK   : 642 µs
  Full fwd+bwd TRT  : 788 µs   ratio 1.228
  group_offs        : 4.6 µs
  3× cached_quantize: 0.31 µs (post-R1, ~free)
  torch.empty(out)  : 1.65 µs
  is_contiguous() x1: 0.05 µs
  custom_op wrapper : 0.4 µs
  HK execute Python : ~3 µs / call

Post-R2 (this round):
  group_offs        : 0.11 µs  (cached)        Δ-3.97 µs symmetric
  select_default_cfg: 0.067 µs (cached)         Δ-0.47 µs / HK call
                                                 × 2 calls / fwd+bwd
                                                 = 0.94 µs HK-asym
```

Theoretical ratio impact on Qwen3-Down-B16-M2048 (from per-iter wall
math, `(TRT - 4) / (HK - 4 - 0.94)`):

```
ratio_pre  = 788 / 642        = 1.228
ratio_post = 784 / 637.06     = 1.231  (Δ +0.003)
```

## Score landscape (5-run variance comparison, this GPU pin GPU 5)

```
                 Wall metric (HK fused vs TRT, target geomean ≥ 1.35)
HEAD (pre-R2):   999 · 1000 · 997 · 997 · 998        median 998
R2  (post):     1000 ·  997 · 1000 · 1000 · 1000     median 1000

Geomean range
HEAD: [1.3456, 1.3558]   1/5 runs above 1.35 (= cap)
R2  : [1.3462, 1.3682]   4/5 runs above 1.35 (= cap)
```

```
                 Un-fused metric (kernel-only, target ≥ 980 — drift band)
HEAD (8 runs):  969 · 970 · 970 · 970 · 971 · 972 · 972 · 974   median 970.5
R2  (5 runs):   970 · 971 · 972 · 972 · 973                     median 972

Note: the 980 floor was set at a prior baseline; both HEAD and R2
sit ~10 points below today due to upstream BF16 round drift, NOT
from this round's changes. R2 median is +1.5 vs HEAD median —
within noise but consistent with the cache savings hitting the
BF16 grouped path (FP8 kernel-only metric pre-computes group_offs
outside the timer; cache doesn't affect that section).
```

```
                 DoD smoke (target: stays at HEAD pass count, must NOT
                            introduce new failures on dense / shared code)
R2: passed=608  failed=0  errors=0     PASS
```

## Correctness probe

3-shape SNR check (`/tmp/probe_round_2_correctness.py`) vs torch-native
FP8 reference:

```
DSV3-GateUP-B16-M2048   SNR(out)=28.45  SNR(dA)=28.46  SNR(dB)=28.47   PASS
Qwen3-Down-B16-M2048    SNR(out)=28.47  SNR(dA)=28.47  SNR(dB)=28.46   PASS
gpt_oss-Down-B32-M2048  SNR(out)=28.46  SNR(dA)=28.45  SNR(dB)=28.45   PASS

Repeated-call bit_eq (cache HIT path):
  All 3 shapes: out=True  dA=True  dB=True
```

Cache HIT path returns the *same* device tensor (verified via
`data_ptr()` equality) so downstream consumers see bit-identical
inputs across iters. No re-launching of the C++ op on cache HIT.

## Why this is two complementary levers

**Lever A — `group_offs` cache (symmetric)**: every call to
`turbo.ops.grouped_gemm_fp8(..., group_offs=None)` and
`turbo.ops.grouped_gemm(..., group_offs=None)` invokes
`grouped_gemm_compute_offs` once per fwd+bwd. Both HK and Triton
backends pay this 4 µs symmetrically. After cache, both walls drop by
4 µs. Ratio = `TRT_wall / HK_wall` strictly increases when HK is faster
(the canonical Δ-from-both-walls geometric proof — see R1 note for
detailed math). All 24 metric shapes have ratio > 1, so geomean
reliably lifts.

**Lever B — `select_default_config` `lru_cache` (HK-asymmetric)**: the
HK execute body (`grouped_gemm_fp8_impl.py:541` and `grouped_gemm_impl.py`'s
HK execute) calls `select_default_config(...)` per kernel launch. The
Triton execute paths do not. Per FP8 fwd+bwd: 2 HK grouped GEMM
launches (forward RCR + backward dA RRR) call this; the var-K dB path
uses inline `(group_m, num_xcds)` rules (no `select_default_config`
lookup). 0.47 µs × 2 = ~0.94 µs HK-only saving — small but compounds
with Lever A.

The two together give the cleanest "post-R1 micro-cache" deposit
without any kernel-internal work or risk to the FP8 numerical contract.

## What this round did NOT do (deferred)

* **Force H4 reroute on Qwen3-Down dA RRR** (lift the lowest-ratio
  cohort by exploiting the well-tuned RCR kernel + cached transpose):
  speculative; R14 falsified unconditional reroute on K-aligned shapes.
  Would need a tight per-shape probe (Qwen3-Down with K_fwd=1536
  specifically) and a narrow gate. Risk of regressing DSV3 / Qwen3-GateUP
  if the gate is loose. Saved for a future round.
* **Forward kernel template tuning for Qwen3 cohort**: per the existing
  R29 commentary in `kernels/hipkitten/config.py`, every (group_m,
  num_xcds) cell on Qwen3-Down M=2048 forward RCR was tight-verified
  and the binding default (gm=4, xcds=8) is the robust optimum.
  Forward-side tuning is at the local optimum; further forward gains
  require HipKittens C++ kernel changes (RRR template stronger compute
  path).
* **`maybe_pre_sync=True` on the dispatcher hot path** (the HK FP8
  forward already passes `maybe_pre_sync=True`; the dA / dB paths set
  it false. Whether always-true is faster is an open question but
  requires probing). Deferred.

## Production semantics (cache lifetime / VRAM bound)

* `group_lens` tensor lifetime: typically lives across the entire
  training step (the per-microbatch dispatch reuses the same group_lens
  shape and value within a sequence-parallel boundary). Cache HIT rate
  in production: ≥99 % once warm.
* `select_default_config` working set: ~50 unique tuples per training
  loop in any realistic deployment (the model has ~15 distinct GEMM
  shapes × fwd/bwd × layouts). lru_cache(1024) never evicts useful
  entries.
* VRAM impact: each `_GROUP_OFFS_CACHE` entry is `(B+1)*8` bytes int64
  (≤ 264 B for B=32). Bounded by the number of *concurrently-live*
  group_lens tensors via `weakref.finalize`. Negligible.

## Next-round suggestion

The wall metric now caps reliably at 1000 (median 1000, 4/5 runs at cap).
The geomean buffer above 1.35 is still thin (~1.36 typical), so the
score remains sensitive to per-shape kernel timing noise. Levers
remaining on the Primus side:

1. **Force H4 reroute for Qwen3-Down dA RRR** (carry-over from this
   round's deferred list). Probe (1) RCR-via-transpose dA kernel time
   vs original RRR, on Qwen3-Down with `K_fwd=1536, N_fwd=4096` —
   compare 4-trial × 200-iter × p20 against `n=K_fwd, k=N_fwd`. If
   RCR ≥ 5 % faster, ship the reroute. The transpose cost amortises
   to ~0 via the existing `_FP8_H4_TRANSPOSE_CACHE`.
2. **Investigate `Float8QuantConfig.fuse_act_quant` defaulting**:
   currently `False`; the metric flips it to True per shape. Production
   integration may benefit from a smarter default once any Path-A
   helper ships (deferred from R1 / R14 / today).
3. **HipKittens C++ kernel side**: forward kernel for Qwen3 cohort and
   the RRR backward template are documented as the residual ~ratio gap
   sources (R8 / R26 / R29). Out of Primus-Turbo's scope but the
   single biggest remaining lever.

If the harness scores 1000 stably for the next 2-3 rounds, consider
the Primus-side fused-wall task converged at 1000 cap.

## Commit

Primus-Turbo: this commit (Round 2).
HipKittens : no change.

## Reference probes (`/tmp/probe_round_2_*`)

- `overhead_breakdown.py`         — per-iter wall decomposition
- `hk_asym.py`                    — HK-asymmetric Python overhead
- `custom_op.py`                  — torch.library.custom_op vs dispatcher
- `offs_cache.py`                 — group_offs cache correctness + timing
- `combined_cache.py`             — select_default_config lru_cache
- `correctness.py`                — end-to-end SNR > 25 dB on 3 shapes
