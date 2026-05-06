# Round-80 — BF16 grouped wall — H4 elimination for gpt_oss-GateUP via native RRR (LANDED, +9 score)

**Date**: 2026-05-05  **HEAD before**: `188559618e49bfbe14`  **score before**: 874 / 1000  **best before**: 874
**HEAD after** : `<this commit>`  **score after** : **883 / 1000  (+9)**

## Lever — extend native-RRR ceil_div N coverage to RRR + drop the dedicated ntail kernel + tighten H4 gate

Path A from R78 (revived from R31's 47-round-old backlog): "H4-elimination via
RRR N-tail MFMA". R78 was diagnostic; R79 burned a side-quest (R28 transpose
re-attempt, FALSIFIED at +1 score). R80 finally lands Path A's first concrete
step — for the **K-tail-aligned half** of the gpt_oss family (GateUP shapes).

### What the kernel comment said for 47 rounds

`kernel_bf16_dynamic.cpp:4283-4291` (R5-era):

> RRR/CRR : B is [G, K, N] — N lives on the COLUMN axis, so an OOB N column
> lands at byte-offset ``row*N_stride + col_oob``, which is still inside the
> per-group SRD (just wraps to the next K row's valid columns) — no clamp
> triggers, garbage data feeds the MMA. Until we add a column-mask path on
> the B load itself (Phase 6+), RRR/CRR keep the legacy ``bpc = fast_n /
> BLOCK_SIZE`` and N-tail flows through ``grouped_tail_kernel``.

The comment was **literally true** but the conclusion ("garbage data feeds the
MMA → can't extend bpc") missed an MFMA-lane-mapping subtlety: for
`mfma_f32_32x32x16_bf16`, lane[i % 32] holds B[k=(i/32)*8..(i/32)*8+7,
n=i%32]. **Each lane holds B values for exactly one N column.** The MFMA
reads B[k, n_in] only from lanes whose held N == n_in. Garbage in OOB-N lanes
is consumed only by lanes computing OOB output cells — which the existing
`store_c_tile_n_masked` path already drops at C-store time.

So the "B-load column mask" the comment said was needed is **not** required
for in-bounds output correctness. The masked C-store handles it.

### Three-line HK kernel change

`kernel_bf16_dynamic.cpp`:

1. Extend `g.bpc = ceil_div(g.n, BLOCK_SIZE)` from RCR-only to `RCR || RRR`
   (line 4297-4322).
2. Extend `layout_supports_main_n` from RCR-only to `RCR || RRR` (line 4471).
   This makes `need_tail_run` return false for K-aligned RRR (no scalar tail).
3. Extend the LDS K-tail RMW kernel grid from `ceil_div(g.fast_n, TBN)` to
   `ceil_div(g.n, TBN)` for RRR (line 4595-4604), so the partial last
   col-tile gets its K-tail correction. The kernel body already handles
   `col >= g.n` correctly (early-return at line 2204; per-col guards at
   line 2278-2298). Mirrors RCR's R11 grid extension.
4. DROP the dedicated `grouped_ntail_kernel_lds_rrr<64>` launch (line 4607-
   4619). Main + LDS K-tail now cover every cell, mirroring RCR's R11
   removal.
5. Extend `lds_handles_all` to include RRR with `K_rem == 64` (line 4636-
   4655) so the scalar tail kernel doesn't double-write over the LDS
   K-tail RMW output for Down-shape paths that take native RRR (none in
   the metric — see Primus-side gate below — but defensive).

### Primus-side gate (one-liner)

`primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_impl.py`:

Change the H4 reroute gate from "fire if K%K_BLOCK!=0 OR N%BLOCK_SIZE!=0"
to "fire if K%K_TWO_TILE!=0" (= K%128 != 0). Net effect on the metric's 24
shapes:

| family | K_RRR (=N_fwd) | N_RRR (=K_fwd) | Pre-R80 path | Post-R80 path | Δ |
|---|---|---|---|---|---|
| DSV3 | aligned | aligned | native RRR | native RRR | unchanged |
| Qwen3 | aligned | aligned | native RRR | native RRR | unchanged |
| gpt_oss-GateUP | 5760 (K%128=0) | 2880 (N%256=64) | **H4 reroute → RCR** | **native RRR** | -1× transpose |
| gpt_oss-Down | 2880 (K%128=64) | 2880 (N%256=64) | H4 reroute → RCR | H4 reroute → RCR | unchanged |

gpt_oss-Down can't take native RRR yet because its K-tail RMW kernel does
an extra bf16 round-trip (load existing C, fp32 add K-tail acc, store bf16)
that exceeds `check_allclose` tolerance vs Triton (Triton fuses the K-tail
in fp32 within a single reduction). The fuse path (no round-trip) is RCR-
only at this stage, so K-tail-inducing RRR shapes still take H4-RCR.

## Correctness probe

`/tmp/r80_native_rrr_correctness.py` runs the metric's `_check_hk_vs_triton_small`
gate on all 8 gpt_oss shapes after a DSV3 warmup:

```
PASS gpt_oss-GateUP B=4  M=2048  reason=''
PASS gpt_oss-Down   B=4  M=2048  reason=''
PASS gpt_oss-GateUP B=32 M=2048  reason=''
PASS gpt_oss-Down   B=32 M=2048  reason=''
PASS gpt_oss-GateUP B=4  M=4096  reason=''
PASS gpt_oss-Down   B=4  M=4096  reason=''
PASS gpt_oss-GateUP B=32 M=4096  reason=''
PASS gpt_oss-Down   B=32 M=4096  reason=''
```

GateUP shapes verify the new native-RRR path; Down shapes verify the
fallback-to-H4 path still works.

## Backward path bench (gpt_oss-GateUP excerpt)

`benchmark/ops/bench_grouped_gemm_turbo.py --dtype bf16` — 24/24 PASS:

```
gpt_oss_20B-GateUP B=4  M=2048  fwd 1217.4 TF  bwd  863.7 TF  PASS
gpt_oss_20B-GateUP B=4  M=4096  fwd 1212.6 TF  bwd 1059.8 TF  PASS
gpt_oss_20B-GateUP B=32 M=2048  fwd 1235.8 TF  bwd  995.7 TF  PASS
gpt_oss_20B-GateUP B=32 M=4096  fwd 1259.5 TF  bwd 1085.8 TF  PASS
```

Average across all 24 shapes: fwd=1265.6 TF, bwd=1026.5 TF. All correctness
checks PASS.

## Metric numbers

```
                                           pre-R80    post-R80   Δratio
gpt_oss_20B-GateUP-B4-M2048   ratio=       1.075      1.121      +0.046
gpt_oss_20B-GateUP-B4-M4096   ratio=       1.106      1.121      +0.015
gpt_oss_20B-GateUP-B32-M2048  ratio=       1.046      1.104      +0.058   ← target
gpt_oss_20B-GateUP-B32-M4096  ratio=       1.085      1.102      +0.017
gpt_oss_20B-Down-*  (4 shapes)             unchanged  unchanged  ~0
DSV3 / Qwen3       (16 shapes)             unchanged  unchanged  ~0

per-family geomean  gpt_oss   1.0761  →  1.0932  (+1.6 %)
                    DSV3      1.1253  →  1.1220  (-0.3 %, noise)
                    Qwen3     1.1099  →  1.1129  (+0.3 %, noise)

weighted score      874  →  883  (Δ +9, prior best 874)
correctness         all 24 PASS, 0 reject
```

## Side metric guard

* `_metric_grouped_fused_wall.py`: 1000/1000 (no FP8 fused-act path touched).
* `_metric_grouped_only.py`: 977 vs baseline 979 — within ±2 noise band on
  the FP8 portion (FP8 path unchanged by this round; small variance per
  re-run, see `/tmp/r80_grouped_only_baseline_compare.log`). The R80
  changes themselves leave the BF16 `grp_BF16` segment effectively
  unchanged (1.1787 → 1.1789).

## Why this gate is general (no per-shape hardcode)

The condition `a.shape[1] % K_TWO_TILE != 0` is a **predicate on K
alignment**, equivalent to "does the HK RRR K-tail kernel need to run".
It's symmetric with the existing RCR fuse gate (`K_rem == K_STEP &&
fuse_ktail_eligible`). It does NOT mention M, N, B, or any specific
family. Future shapes with K_RRR%128==0 will automatically take the
native-RRR fast path; future shapes with K_RRR%128!=0 will continue to
take H4-RCR until R81+ implements the RRR fuse path.

## Direction for R81

Two natural follow-ons, both targeting **gpt_oss-Down dA** (current ratio
~1.05; the only remaining family-low):

1. **RRR K-tail FUSE** — extend `fuse_ktail_eligible` to RRR. The R29-era
   RRR fuse work was archived at SNR ~19 dB (vs 44.5 dB legacy) due to a
   "phantom-read" cross-warp G::load LDS visibility bug. The current
   kernel may have evolved enough (post-R55 LDS-staged ntail landing,
   post-R56 K-tail RMW restructuring) that the bug no longer reproduces.
   Re-attempting under the BF16_RRR_FUSE_PROBE flag (line 4388-4395)
   and re-checking SNR is a low-blast-radius first step. If SNR ≥ 40 dB
   on the metric's downsized gate, gpt_oss-Down dA gets ~393 µs H4
   transpose savings (matching GateUP's gain) → ~+5-10 score.

2. **dB var-K LDS swizzle** — gpt_oss-Down B=32 dB still hits the
   217M-LDS-bank-conflict regime per R68 PMC. R74's `st_64x32_padded_b128_s`
   swap caused 66 VGPR spills + 24/24 dB FAIL (incompatible sub-tile
   padding). A more careful swizzle-only (no shape change) attempt could
   eliminate the conflicts without touching VGPRs. Independent of (1).

R81 picks (1) — re-attempt RRR fuse — as the cheaper test (single-flag
flip + SNR probe).
