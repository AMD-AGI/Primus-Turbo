# MXFP8 fused GEMM+combine (L2) perf note (MI355X / gfx950)

Perf of the mega-MoE **L2 (down-proj) fused GEMM + combine PUSH + top-k reduce**, vs the bf16
fused kernel (`grouped_gemm_combine_bf16_kernel.py`). AMD Instinct MI355X, DeepSeek-V3 MoE L2:
`T=8192, H=7168, I=2048, E=256, K(top-k)=8`, EP8 (`G=32` experts/rank), `BLOCK_M=BLOCK_N=256`.

Benchmarks: `benchmark/ops/bench_grouped_gemm_combine_mxfp8.py`, `benchmark/ops/bench_grouped_gemm_comine.py`.

## TL;DR — fp8 gives **no** L2 win. Ship **L1 fp8 fused + L2 bf16 fused**.

The L2 combine is a **bf16** reduce-scatter of the (residual) MoE output; it is genuinely
**XGMI-bandwidth-bound at ~300 GB/s** (see the bandwidth correction below — the earlier "43 GB/s /
117 MB" figure was wrong). fp8-ing the combine **payload** does halve the bytes and, *in isolation*,
combine drops 2.70 → 1.44 ms (1.85×). **But it cannot be turned into a fused L2 win:** the mxfp8
**quantization of the L2 GEMM output is expensive compute (~1–2 ms)** and, wherever it is placed in
the fused kernel, it costs more than the combine bytes it saves. Three placements were implemented
and measured (all bit-correct, cos ≈ 0.9996 vs bf16 fused), all **slower** than bf16 fused:

| L2 fused variant | correct? | vs bf16 fused |
|---|---|---|
| bf16 fused (reference) | — | 1.00× (2.7–2.9 ms) |
| fp8, **quant-in-combine** (3-role, 4-lane amax in combine) | cos 0.9996 | **0.99×** |
| fp8, **separate quant role** (4-role, GEMM→quant→combine→reduce) | cos 0.9996 | **0.76×** |
| fp8, **quant in GEMM epilogue** (32-lane butterfly amax) | cos 0.9996 | **0.76×** |

Recommendation: **keep L2 as `grouped_gemm_combine_bf16`**; use fp8 only at L1 (`comm="fp8_fused"`,
validated 1.4× vs bf16 fused). The fp8-combine correctness + byte-lever are validated and the
approaches above are exhausted; do not re-attempt fp8 L2 combine without a fundamentally cheaper
GEMM-output quantization.

## Combine bandwidth correction (the "43 GB/s wall" was wrong)

The earlier note claimed combine moves `~117 MB/rank` → ~43 GB/s and called it a "pattern/uncached
wall". **That byte count was the final reduced output `y[T,H]` size (8192·7168·2 = 117 MB), NOT the
combine transfer volume.** Combine pushes each token's **K=8** expert-output rows (pre-reduction), so
the real volume is ≈ **895 MiB/rank** (valid rows ≈ T·K ≈ 65 k × H × 2 B). Measured directly:

| | value |
|---|---|
| combine remote (XGMI) rows/rank | ~57.7 k |
| combine bytes/rank (bf16) | ~788 MiB (XGMI-only) / ~895 MiB (incl. self) |
| combine_only time | 2.70 ms |
| **combine XGMI bandwidth** | **~300 GB/s** (bf16 bench reports 297; probes 309–321) — near the XGMI limit, same order as the L1 dispatch push (377 GB/s) |

So combine is **bandwidth-bound**, not pattern-bound. That makes the fp8 byte-halving a *real* lever
in isolation (below) — it just can't be realized in the fused kernel.

## The byte lever, in isolation (combine_only, bf16 vs fp8 payload)

Pushing fp8 (`H` bytes) + E8M0 (`H/32` bytes) instead of bf16 (`2H` bytes) — pure bandwidth test,
no quant, no accuracy:

| ncu | bf16 combine | fp8 combine | speedup |
|---|---|---|---|
| 128 | 2.67 ms (309 GB/s) | 1.44 ms (295 GB/s) | **1.85×** |

Byte ratio 0.516; bandwidth ~preserved → halving bytes ~halves time. Real, if quant were free.

## Why it doesn't become a fused win — the quant is the wall

To realize the byte lever the combine must stay a pure fp8 **copy** (XGMI-bound, few CUs). That
requires the L2 GEMM output to already be mxfp8. Quantizing it is the problem:

- **Standalone** rowwise mxfp8 quant is cheap (`quantize_rowwise_mxfp8_flydsl` ≈ 0.57 ms) but a
  separate kernel → **no overlap** (decoupled fp8 L2 ≈ gemm+quant+combine+reduce serial ≈ 3.7 ms).
- **quant-in-combine**: the amax+cvt runs on the combine CUs → combine becomes compute-bound (a CU
  sweep confirms quant-combine is compute-bound: 32 CU→3.6 ms, 96 CU→1.6 ms; the pure copy is flat
  ~1.5 ms). Giving it more CUs steals from the GEMM (zero-sum) and, past ncu≈64, the extra spinning
  combine blocks starve the GEMM → co-scheduling livelock/hang. Net ≈ 0.99×.
- **separate quant role**: adds a pipeline stage + a 2nd device-wide `l2_invalidate` + a local
  fp8-L2Y round-trip + a 4-way CU split (most splits hang). Net 0.76×.
- **GEMM epilogue quant** (the "right" place, in-register): the MFMA 32×32 output layout has each
  lane owning one column and 16 rows, so a per-32-col-block amax needs a **32-lane butterfly
  reduction** (5 shuffles × 16 rows × tiles) + scattered write-through byte stores. This makes the
  GEMM epilogue heavy → gemm+epilogue ≈ 3 ms → fused 4.6 ms. Net 0.76×.

In all cases the mxfp8 quant of the GEMM output is ~1–2 ms in-kernel (the standalone 0.57 ms is not
achievable fused because the GEMM output is in MFMA layout, not the rowwise layout the cheap quant
wants — the layout transpose is the cost). That exceeds the ≤1.3 ms of combine bytes saved.

## Reduce is cheap; combine is the only real L2 knob (and it's shared bf16/fp8)

`reduce_only` ≈ 0.18–1.05 ms (fp8 dequant reduce = 0.18 ms). The L2 floor is `max(gemm, combine)`;
with bf16 combine (2.7) that's combine-bound at ~2.7–2.9 ms — the bf16 fused number, which stands.

## Files

- Production L2 (recommended): `grouped_gemm_combine_bf16_kernel.py` (`grouped_gemm_combine_bf16`).
- `grouped_gemm_combine_fp8_kernel.py`: experimental **dead-end** (bf16 GEMM + mxfp8 epilogue quant
  + fp8 combine + fp8 dequant reduce). Bit-correct (cos 0.9996) but 0.76× and has an intermittent
  reduce-flag liveness stall under back-to-back timing calls. Kept as a reference for the exhausted
  approach; NOT wired into any forward path. (`gemm_bf16_nt_tile` grew a backward-compatible
  optional `store_c` param so this kernel can inject its quantizing epilogue.)
- `grouped_gemm_combine_mxfp8_kernel.py`: earlier fp8-GEMM + bf16-combine fused (4.68 ms, l2_writeback
  serialized) — superseded by this analysis; also not the win.

## Compute-heavy caveat

If tokens/expert grows so `gemm ≳ combine`, the fp8 GEMM would start to matter at L2 — re-measure
that regime before choosing any fp8 L2 path. For DSv3 EP8 shapes here, L2 is combine-bound and bf16
fused is the pick.
