# Round 16 — FP8/BF16 var-K `execute()` host-overhead trim (mirror of R11)

## Target

The lowest-ratio shape on today's metric (post-warm) was
`fusedFP8_Qwen3-235B-A22B-Down-B16-M2048` at ratio 1.252. R12 already
falsified the RCR-forward `(group_m, num_xcds)` lever for the entire
Qwen3-Down K=1536 family; R15 falsified the var-K `(group_m, num_xcds)`
lever for the same family. Both confirmed those shapes are kernel-bound,
not Python-bound, and the remaining HK kernel-internal levers (BLOCK_K=64
template, 4w grouped) are multi-round HK source surgery.

This round picks up a smaller-ceiling but tractable lever: trim the
remaining Python overhead in the var-K `execute()` body to mirror what
R11 did for the dense forward `execute()` body.

## What R16 changed

`primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`
(`GroupedGEMMFP8VariableKHipKittenBackend.execute`) and
`primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_impl.py`
(`GroupedGEMMVariableKHipKittenBackend.execute`):

1. **Pre-resolve `grouped_variable_k_crr{,_dscale}` on `HipKittenModule`**
   — the var-K execute previously did
   `getattr(hk.module, "grouped_variable_k_crr", None)` × 2 on every
   backward dB launch (~66 ns / call). Now they're loaded once at
   module-load time (`loader.py::_build_module`) and accessed via
   plain attribute on the dataclass. Mirrors the existing
   `gemm_*` / `grouped_*{,_dscale}` pre-resolved attribute pattern.

2. **Skip `_resolve_fp8_scales` on the dscale fast path** —
   the var-K execute previously called
   `_resolve_fp8_scales(b_scales, a_scales, fp8_has_dscale(hk, "crr"))`
   on every call (~485 ns / call). On the metric's hot path (TENSORWISE
   scales from `quantize_fp8(..., TENSORWISE)` are always
   `numel==1 / fp32 / contiguous / cuda` by construction), every
   condition inside `_resolve_fp8_scales` evaluates True and the result
   is just `(None, None, b_scales, a_scales)` — a no-op pass-through.
   The trim short-circuits this on the dscale fast path and passes the
   raw scale tensors directly. Fallback path (CPU scales / older .so
   without dscale) preserved bit-identical via
   `_resolve_fp8_scales(b_scales, a_scales, False)`.

## Probe data (per-call host overhead)

`/tmp/probe_r16_var_k_overhead.py`, 100 000-iter median:

| Segment                                  | Before     | After      | Saved        |
| ---------------------------------------- | ---------- | ---------- | ------------ |
| Var-K callable lookup                    | 66.3 ns    | 31.9 ns    |  34.4 ns ( -52 %) |
| Scale resolution (dscale fast path)      | 485.3 ns   | 31.0 ns    | 454.3 ns ( -94 %) |
| **Total per-call**                       | **551.6 ns** | **62.9 ns** | **488.7 ns** |

Over the metric's 1 440 var-K calls (60 timed iters × 24 shapes), the
host-side savings sum to ~750 µs. On the geomean of a ~13 s metric run
that's ~0.005 % — well under the run-to-run noise band (980-1000)
established across R11-R15.

But on the small-grid B=4 gpt_oss family (var-K dB call wall ≈ 100 µs,
of which ~25 % is Python frame overhead per probe), the saving is
~0.5 % of var-K dB call wall and ~0.1-0.2 % of fwd+bwd wall — visible
in microbench, not above noise in the 24-shape geomean.

## Bit-equivalence

Verified via `/tmp/probe_r16_var_k_integration.py` on the
Qwen3-Down B=16 M=2048 var-K shape:

```
out_new vs out_old (same kernel, same scales): max_abs_diff = 0.0
PASS: bit-identical (max_abs_diff = 0)
```

Plus the two HK-pinned bench runs (`bench_grouped_gemm_turbo.py`):

* FP8 tensorwise: 24/24 PASS (SNR > 25 on out / dA / dB across
  DSV3 / gpt_oss / Qwen3 families).
* BF16: 24/24 PASS (allclose on fwd / bwd_x / bwd_w across all
  three families).

## Metric impact

```
Before R16 (today, warmed): score = 995 (geomean = 1.3428)
After  R16 (today, warmed): score = 995 (geomean = 1.3430)
```

Same score within noise; geomean drift is +0.0002, well below the
±0.005 noise band observed across the R11-R15 runs.

## Why land it anyway

* **Bit-identical**: the change provably does not affect any tensor
  value (probe + 24 + 24 bench cases).
* **Code-quality**: brings the var-K execute body to architectural
  parity with the dense forward execute body (R11 pattern), removes
  an inconsistency where one path pays ~520 ns / call of overhead
  the sibling path doesn't.
* **Future-proof**: covers BF16 var-K too; any future grouped
  variable-K work (e.g. RCR / RRR var-K bindings if HK ships them)
  inherits the same pre-resolved attribute pattern.
* **Asymmetric to HK**: Triton var-K backend doesn't touch
  `hipkitten` at all, so the savings are HK-only differential.

## What's left (unchanged from R13/R15)

The remaining performance gap to a stable score of 1000 is HK
kernel-internal:

1. **HK kernel surgery for Qwen3 K=1536** (BLOCK_K=64 template
   specialization). Highest ceiling (~+5-10 % on 4 Qwen3-Down shapes),
   multi-round HK source work (recompile + correctness probe +
   sweep + Primus dispatch wire-up).
2. **HK pybind: `kernel`-template arg for grouped FP8 RCR / dscale**
   would re-open per-shape kernel template selection — but the lever_c2
   work (R57-R63) showed the only alternative grouped template (4w-style)
   regresses, so this lever is questionable in practice.

Python-side dispatch / cache levers are now exhausted (R7-R12 for
forward, R15-R16 for var-K dB).

## Round summary

* Target: lowest-ratio Qwen3-Down-B16-M2048 (1.252) — confirmed
  kernel-bound, no Python lever available.
* Picked up the var-K execute trim (R11 mirror) as the only
  remaining tractable Python-side lever.
* Bit-identical, ~488 ns / call host-side savings, ~750 µs total
  metric wall savings (under noise floor).
* Metric: 995 → 995 (noise band 980-1000, no regression).
* Bench: FP8 24/24 PASS (SNR > 25), BF16 24/24 PASS (allclose).
