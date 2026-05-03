# Round 7 — Path A fused-fwd FALSIFIED: DTR + in-register cvt loses to DTL by ~40 %

## TL;DR

Re-enabled the R6 ``_hk_fused_act_forward`` impl + wired the saved
forward-scale into the dB un-fused fallback so it skips the redundant
``amax`` pass via ``quantize_fp8_tensorwise_impl(scale=...)``. With both
paths active, the fused-fwd metric **regressed -26 %** (940 → 696) on
the 16 K-aligned shapes where the fused path actually runs:

| Shape (K%128==0)                         | un-fused HK ratio | fused-fwd HK ratio | Δ      |
| ---------------------------------------- | ----------------- | ------------------ | ------ |
| DSV3-GateUP-B16-M2048                    | 1.302             | 0.832              | -36 %  |
| DSV3-Down-B16-M2048                      | 1.241             | 0.810              | -35 %  |
| DSV3-GateUP-B16-M4096                    | 1.402             | 0.822              | -41 %  |
| DSV3-Down-B16-M4096                      | 1.340             | 0.790              | -41 %  |
| Qwen3-GateUP-B16-M4096                   | 1.239             | 0.778              | -37 %  |
| Qwen3-Down-B32-M4096                     | 1.231             | 0.795              | -35 %  |
| (… 10 more K-aligned shapes, all -30…-41 %)             |                    |        |

The 8 K-misaligned shapes (gpt_oss, K=2880, K%128==64) fall back to the
un-fused path inside ``_hk_fused_act_forward``'s gate so their ratios
are unchanged.

Root cause is **architectural**, not implementation-level: the
fused-fwd load helper cannot match the un-fused FP8-DTL throughput on
this kernel's load_a critical path.

## DTL vs DTR — fundamental gap

The un-fused FP8 ``rcr_8w_load_hoist`` uses **DTL** (Direct Tile Load):

```
buffer_load_dwordx4 ... offen lds       ; HBM bytes → LDS in 1 hardware path
                                        ; no VGPR round-trip, no per-lane store
```

Per pass / lane: 16 B FP8 HBM read → 16 B LDS write. Single instruction,
no register cost (the load goes through the LDS-DMA path).

The fused-act ``rcr_8w_load_hoist_fused_act`` (R5a deposit) uses **DTR**
(Direct Tile Read) + cvt + ds_write:

```
raw_buffer_load_b128                    ; HBM → VGPR (16 B BF16 lo)
raw_buffer_load_b128                    ; HBM → VGPR (16 B BF16 hi)
cvt_bf16x4_to_fp8x4    × 4              ; VGPR cvt: 32 B BF16 → 16 B FP8
ds_write_b128                           ; VGPR → LDS (16 B FP8)
```

Per pass / lane: **32 B BF16 HBM read** (2× the DTL) + an explicit
VGPR / cvt / LDS-write critical path. The cvt + ds_write add ~20-30
cycles of latency that the DTL-based path doesn't pay.

## Why the fwd-only quantize-launch saving cannot offset

* Quantize launch elimination: ~10-20 µs / call (max_abs + per-elem
  multiply + cvt). On big GEMMs this is < 1 % of fwd wall.
* DTR-based load_a slowdown: ~30-40 % per kernel call (measured).

For HBM-traffic-bound or load_a-critical-path-bound shapes (which
covers all K-aligned metric shapes), the +30-40 % kernel slowdown
overwhelms the -10-20 µs launch saving by 1-2 orders of magnitude.

## Path B (BF16 LDS staging) — already falsified at R3

R3 falsification note (round-3-fused-act-fwd-kernel-surgery-plan.md)
covered this: BF16 LDS staging requires 2× the LDS allocation of the
FP8 LDS-staged tile, and the existing ``ST_v2`` LDS budget on MI355X is
already saturated (64 KB / CU; the FP8 ``As[2][2] + Bs[2][2]`` =
16 + 16 = 32 KB; doubling A's tile to BF16 would push us past the 64 KB
budget and drop occupancy from 2 waves/SIMD → 1, halving throughput).

So both R3-falsified-Path-B (LDS-staged BF16) and R6/R7-falsified-Path-A
(DTR + in-register cvt) are blocked. There is **no** in-kernel fused-act
path that beats the un-fused un-fused FP8-input ``grouped_rcr_kernel``
on this MI355X kernel architecture.

## What stays in the codebase

* HK ``.so`` deposit (commit a7683112 in HipKittens):
  ``grouped_rcr_kernel<...,FUSE_ACT=true>`` template specialisation +
  ``dispatch_grouped_rcr_fused_act`` + ``grouped_rcr_fused_act_dscale``
  pybind binding + the R5a load helper + R4 cvt builtin. **No metric cost
  to leaving these in** — the FUSE_ACT=false instantiations are the only
  ones the dispatcher launches today, and their codegen hasn't measurably
  drifted from pre-R6 (metric 940 vs best 938 = noise-band).
* Primus-Turbo ``Float8QuantConfig.fuse_act_quant`` flag +
  ``FP8GroupedGemmTensorFusedActFunc`` autograd Function + the
  ``_hk_fused_act_*`` per-GEMM hook scaffolding from R2.
* Primus-Turbo dB un-fused fallback's
  ``quantize_fp8_tensorwise_impl(scale=...)`` enhancement — kept (small
  correctness-preserving improvement; will help if any future fused
  path is enabled and saves a forward scale).

## What's removed / disabled

* ``_hk_fused_act_forward`` body — reverted from the R7 attempt back to
  ``raise NotImplementedError(...)``. The autograd Function falls back
  to the un-fused path on every shape. Same effective behaviour as R6
  pre-PHASE-1 baseline.

## Where to look for residual wins

The metric is at 938-940 (best of 100 rounds). Target 1000. Remaining
gap = ~6-7 %. Path A doesn't help on K-aligned shapes (covered above).
**Promising directions for R8+:**

1. **dB var-K kernel** (``grouped_var_k_kernel_fp8``, line ~6682 of
   ``kernel_fp8_layouts.cpp``). Different load pattern (``a.T``, K-major
   accumulation across groups). Need to inspect whether load_a uses DTL
   too — if not, fused-dB might pay off where fused-fwd doesn't. Phase 3
   was always the most promising of the three sub-paths; it just got
   deprioritized in R1-R7 in favour of fwd-first.
2. **Lowest-ratio shapes** in the current metric:
   * gpt_oss-Down-B32-M2048: 1.149
   * Qwen3-235B-A22B-Down-B16-M2048: 1.172
   * Qwen3-235B-A22B-GateUP-B16-M2048: 1.183
   These are NOT fused-act bound — they're un-fused ratio limits. R8+
   should look at the un-fused kernel itself (``grouped_rcr_kernel`` +
   ``grouped_tail_kernel`` interaction on small-M shapes).
3. **Reverse direction** — if Triton itself has a small per-launch
   overhead we're not seeing (a known issue at R12), narrowing the gap
   on small shapes (gpt_oss-B4) could give ~0.5-1 % score uplift.

## Falsification log (R1-R7 fused-act sub-thread)

| Round | What                                            | Outcome                                            |
| ----- | ----------------------------------------------- | -------------------------------------------------- |
| R1    | max_abs_bf16_to_fp8_scale HK kernel + binding   | DEPOSITED. Standalone amax+scale 2-23 % SLOWER     |
|       |                                                 | than C++ quantize_fp8 due to launch overhead       |
| R2    | Per-GEMM fused-act hook scaffolding             | SCAFFOLDING ONLY. No perf change                   |
| R3    | Surgery plan; falsify Path B (LDS budget)       | DESIGN. Selected Path A (DTR + in-register cvt)    |
| R4    | cvt_bf16x4_to_fp8x4 HK builtin + numerical probe| DEPOSITED. SNR 25+ dB; 22 VGPRs per cvt batch      |
| R5a   | rcr_8w_load_hoist_fused_act DTR helper          | DEPOSITED. SNR 340 dB end-to-end probe; 24 VGPRs   |
| R6    | grouped_rcr_kernel<FUSE_ACT=true> template +    | DEPOSITED. Build clean; 256 VGPRs / 2 waves        |
|       | dispatcher + binding                            | (same as un-fused)                                 |
| R7    | End-to-end Primus wire + bwd skip-amax          | **FALSIFIED**. Net wall -26 % vs un-fused          |

The R1-R6 deposits compile cleanly + are bit-numerically equivalent at
the helper level. The R7 failure is at the **kernel-perf level**, not the
correctness level. End-to-end correctness was 24/24 PASS even at -26 %
wall — the saved fwd scale flows correctly through dB and the output
matches torch reference.

## Decision (R7)

* Disable the Primus call-site (revert to ``NotImplementedError``).
* Document the falsification (this note).
* Pivot the fused-act sub-thread to dB var-K (Phase 3) for R8+ —
  different kernel, different load pattern, may not have the DTL-vs-DTR
  block. If R8 dB inspection also says blocked, fused-act as a whole is
  falsified for this kernel architecture and we move to non-fused-act
  optimization directions (kernel-internal: Mfma scheduling, ktail
  cooperative tiling, var-K epilog spill trim, etc.).
