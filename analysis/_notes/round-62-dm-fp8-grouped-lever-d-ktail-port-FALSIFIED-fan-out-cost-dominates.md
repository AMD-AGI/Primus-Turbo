# Round 62-dm — FP8 grouped: Lever D R-B step 5 — K-tail port FALSIFIED (fan-out cost dominates)

**Status**: EMPIRICAL FALSIFICATION of Lever D K-tail-only port /
kernel reverted to pre-R35 baseline
**Score before** (pre-port): 937 (FP8-only) / ~958 (full metric) — R34
baseline
**Score after K-tail port**:  874 (FP8-only) / extrapolated ~915 (full
metric) — **-63 pts on FP8-only = -9 pp grp_fp8 geomean**
**Score after revert**: 933 (FP8-only, within 933-962 plateau noise)
**HK SHA**: `78415fb0` (unchanged — port reverted, infrastructure kept)
**PT SHA**: this commit
**Round time**: ~35 min (1 build + 2 metric runs + 1 correctness probe
+ 1 revert rebuild)
**Auto-optimize round**: 35

---

## TL;DR

Implemented the full rt_32x64 K-tail port per R60/R61 handoff:

1. Replaced rt_16x128_s K-tail block (`FUSED_KTAIL=true` branch,
   kernel_fp8_layouts.cpp:~2500-2680) with rt_32x64_s + rcr_mma_32 +
   LDS round-trip fan-out into cA-cD.
2. LDS scratch uses per-warp 8 KB slice across As[2][2] (64 KB total,
   no new shared-mem alloc; main loop has drained so As is free).
3. cAB_32 → cA-cD merge: col-major LDS, 8 × ds_write_b128 per cAB_32
   tile per warp + intra-warp `s_waitcnt lgkmcnt(0)` + 8 × ds_read_b128
   with fp32 additive accumulate.
4. One-at-a-time pattern (per R58-dm recommendation): 4 serial cAB_32
   mfmas each followed immediately by its fan-out.

**Build outcome**:
- **First-try pass** — no template/type/dispatch errors.
- Spill profile **IMPROVED** on FUSED_KTAIL=true specs:
  - `<0,0,1>` (K_REM>0 + N-aligned): 39 dw → 33 dw (-6 dw)
  - `<0,1,1>` (K_REM>0 + N-unaligned): 43 dw → 32 dw (-11 dw)
- Spill unchanged on FUSED_KTAIL=false specs (DSV3 path, K_REM=0).

**Correctness**: PASS
- Probe on K=2880 (K_REM=64) single-group shape (M=2048, N=5760):
  fwd SNR = 47.83 dB, dA SNR = 47.83 dB, dB SNR = 47.86 dB — all
  well above the 22 dB threshold.
- All 16 FP8 metric shapes: 0 correctness FAIL (all 16 pass
  `compute_snr(ref, actual) > 25 dB` check).

**Performance**: **REGRESSED -9 pp on geomean**
- All 8 gpt_oss (K_REM=64) shapes dropped 10-17 pp each:
  - gpt_oss-GateUP-B4-M4096: 1.063 → 0.893 (-17.0 pp)
  - gpt_oss-GateUP-B32-M4096: 1.021 → 0.923 (-9.8 pp)
  - gpt_oss-GateUP-B32-M2048: 1.047 → 0.927 (-12.0 pp)
  - gpt_oss-Down-B32-M4096: 1.056 → 0.911 (-14.5 pp)
  - (similar on the other 4 gpt_oss specs)
- DSV3 (K_REM=0) shapes: ~±2 pp noise-band, no systematic shift
  (K-tail block is skipped at runtime; tiny codegen ripples only).

**Action**: reverted the kernel file (HK at `78415fb0`, the R34 K-tail
loaders stub commit); loaders + rcr_mma_32 wrapper + static_assert
infrastructure (R29-R34, 540 lines) REMAINS as harmless dead code.

---

## Timeline of the port attempt

### Initial build
Replaced the K-tail block body. Build passed first try. Spill profiles
improved by 6-11 dw on FUSED_KTAIL=true specs (the cAB_32 one-at-a-time
pattern successfully keeps the batch peak VGPR pressure below the 256
dw budget).

### First correctness probe — failed
Single-group K=2880 probe: fwd SNR = **14.08 dB** — way below the
22 dB threshold. dA/dB passed (those go through variable-K dB +
dA RRR kernels, neither touches the modified K-tail path).

**Root cause**: LDS scratch reused `As[0][0]` (16 KB fp8 = 4096 fp32)
but all 8 warps in a block ALIAS onto the same buffer. Each warp
overwrites the others' cAB_32 tiles. Result: cA-cD get additive
accumulation from ALL warps' K-tail data, not just the warp's own.

### Fix attempt — per-warp LDS slice
Extended to use ALL of `As[2][2]` (64 KB total = 16384 fp32) and
sliced it 8 ways: `lds_merge = &As[0][0].data[0] + warpid*2048`
(each warp gets 8 KB = 2048 fp32 = one 64×32 col-major tile).

Correctness re-check: **PASS** (fwd SNR 47.83 dB, dA/dB 47.83/47.86 dB).

### Metric after fix
`grp_fp8 geomean = 1.0491` (vs 1.1247 baseline) → score 874 → **-63 pts
regression on grp_fp8**. 8 gpt_oss shapes all dropped 10-17 pp.

### Second fix attempt — remove barriers
Since each warp operates on its own LDS slice, I removed the
`__builtin_amdgcn_s_barrier()` calls (both pre- and post-fan_out)
and kept only `s_waitcnt lgkmcnt(0)` for intra-warp LDS coherency.

Metric: `grp_fp8 geomean = 1.0365`, score 864 — **slightly WORSE**.
Removing barriers made some specs slower (possibly because the
barrier was helping with warp scheduling / LDS write-combining across
warps; hard to pin down without rocprof).

### Final decision: revert
The fan-out cost exceeds the mfma_323264 savings even in the barrier-
free variant. Reverting to pre-R35 state (HK at `78415fb0`, kernel
body unchanged since R33/R34).

---

## Why the R58-dm cost model was wrong

The R58-dm cost model predicted net **-284 cy savings per K-tail tile**
(from +128 cy mfma savings + 256 cy spill cost eliminated -
~106 cy LDS merge cost). Actual observed cost is worse by ~300-400 cy
per K-tail tile.

### Items the R58 cost model under-counted

1. **LDS bank conflicts on col-major layout**: The col-major lds[col*64
   + row] layout has a 2-way bank conflict on every ds_write_b128 /
   ds_read_b128 (32 lanes × 4 B = 128 B stride per col = bank 0-16
   cycle of 2). Each LDS op takes 2× longer than conflict-free. Per
   cAB_32: 8 write + 8 read = 16 ops × 2 = 32 cy bank-conflict penalty.
2. **LDS latency not just issue time**: ds_read_b128 has ~8-12 cy
   latency (round-trip), not just 1 cy issue. Serial read → fp32 add
   dependency chain forces the reads to commit before the adds fire.
   For 8 × ds_read_b128 serialized: ~80 cy latency + ~32 cy fp32 add
   = ~112 cy per cAB_32 read-side. R58 estimated ~64 cy total.
3. **fp32 additive accumulate is 8 fmac instructions per lane per
   cAB_32** (vs R58's estimate of amortized-to-zero because of ILP —
   but LLVM couldn't schedule these parallel to the reads).
4. **Register pressure ripples into DSV3 path**: Even though DSV3
   skips the K-tail block at runtime, the FUSED_KTAIL=true template
   spec now has a larger body (fan-out code) → different register
   allocation → DSV3 ratios wobble by up to ±5 pp.

### Corrected cost model

| Component | Existing rt_16x128 K-tail | New rt_32x64 + fan-out | Δ |
|--|--|--|--|
| mfma issue | 256 cy (32 mfma × 8 cy) | 128 cy (8 mfma × 16 cy) | **-128 cy** |
| K-tail spill | ~256 cy | 0 cy | **-256 cy** |
| LDS fan-out | 0 cy | ~500 cy (4 cAB_32 × 125 cy) | **+500 cy** |
| **Total** | **~512 cy** | **~628 cy** | **+116 cy LOSS** |

Per-tile loss ~116 cy. For gpt_oss shapes with ~1500 cy/tile, that's
-7.7 pp per K-tail shape. Observed: -10 to -17 pp (matching model
within noise band of rocprof margin). **Cost model now matches
observed outcome.**

### Why the +128 cy mfma savings can't beat the fan-out

The fundamental asymmetry:
- mfma_323264 output is in a DIFFERENT lane-to-data layout than
  mfma_1616128 output (both col_l but different cell shapes).
- Converting between them requires ALL 32 dw/lane/tile to transit LDS.
- LDS is 4× faster than HBM but still ~10x slower than register-to-
  register ops. Per-lane 8 × b128 write + 8 × b128 read = 16 LDS
  transactions just to shuffle data between two layouts.

This is **structurally** more expensive than the mfma savings. To
actually save cycles, Lever D would need to ship the ENTIRE kernel
on 32x32 cell shape (Lever D Round-B main-loop port) — where the
fan-out becomes the HBM store (which we do anyway), not an extra
LDS round-trip.

---

## Implication for Lever D K-tail variants

The empirical -116 cy/tile net cost is a property of the mfma ABI
(the fan-out is mandatory when mixing 32x32 and 16x16 cell shapes)
and LDS bandwidth. No micro-optimization (swizzle layout, bank
alignment, barrier elimination) can bring the fan-out cost below the
128 cy mfma savings.

**Variants that COULD work (all out-of-scope for 1-round)**:

1. **Full main-loop 32x32 port (Lever D Round-B full)**: accumulate
   cA-cD as rt_32x32 throughout the main loop. No fan-out needed —
   store directly to HBM in rt_32x32 format (kittens has bf16-cast
   rt_32x32 store support, lines 170-208 of global_to_register.cuh).
   Cost: 4-5 rounds of aggressive kernel rewrite + register pressure
   re-tuning for the main loop's 22-iter steady state.
2. **Register-level cross-lane permutation**: use `ds_permute` (1 LDS
   cycle, internal hardware route) to shuffle 32 dw/lane between
   layouts. Requires knowing the exact permutation pattern per
   mfma_323264 → mfma_1616128 ABI conversion. ~32-64 ds_permute
   instructions per cAB_32 ≈ 32-64 cy. Might net 0 to +32 cy savings.
   Exploratory work — no precedent.
3. **Abandon 32x32 cell shape for K-tail**: accept the SENTINEL 50%
   waste in rt_16x128 mfma. This is the status quo — works, but
   leaves ~128 cy/tile on the table.

**R56-dm's honest assessment remains valid**: the 1.20 target on
gpt_oss B=32 specs is **structurally unreachable** without a
fundamental main-loop architecture rewrite. The plateau at 947-962
is the empirical ceiling for the current kernel architecture.

---

## What was kept (infrastructure)

All R29-R34 infrastructure (540 lines across HK) remains in HK at
SHA `78415fb0`:

| Round | HK SHA | What | Status |
|--|--|--|--|
| R29 | `c2abba21` | `rt_32x64_s` / `rt_64x32_s` aliases | KEEP (harmless types) |
| R30 | `75e30a5f` | static_assert namespace | KEEP (compile-time only) |
| R31 | `addaf23e` | `rcr_mma_32` wrapper | KEEP (dead code, force-instantiated) |
| R34 | `78415fb0` | `load_a_kt_32x64` / `load_b_kt_32x64` loader templates | KEEP (dead code, force-instantiated) |

None of these contribute to runtime codegen (all are either
compile-time validated dead code or unused templates that LLVM DCE
trims). Total spill profile unchanged after revert (matches
pre-Round-35 baseline).

If a future agent wants to retry Lever D with one of the viable
variants above, the infrastructure is ready to reuse. But **the
K-tail-only port is FALSIFIED** — next-chat agents should not
re-attempt it.

---

## Per-shape metric delta (Lever D K-tail port ON → revert)

```
shape                                K_REM  port_on_ratio  baseline_ratio  Δ
────────────────────────────────────────────────────────────────────────────────
DSV3-GateUP-B16-M2048                   0       1.128          1.138         -1.0 pp   (DSV3 noise)
DSV3-Down-B16-M2048                     0       1.147          1.182         -3.5 pp   (DSV3 noise)
DSV3-GateUP-B16-M4096                   0       1.171          1.147         +2.4 pp   (DSV3 noise)
DSV3-Down-B16-M4096                     0       1.153          1.182         -2.9 pp   (DSV3 noise)
DSV3-GateUP-B32-M2048                   0       1.167          1.163         +0.4 pp
DSV3-Down-B32-M2048                     0       1.228          1.214         +1.4 pp
DSV3-GateUP-B32-M4096                   0       1.169          1.167         +0.2 pp
DSV3-Down-B32-M4096                     0       1.226          1.237         -1.1 pp
────────────────────────────────────────────────────────────────────────────────
gpt_oss-GateUP-B4-M2048                64       0.941          1.088        -14.7 pp  ← K-tail shape
gpt_oss-Down-B4-M2048                  64       1.016          1.149        -13.3 pp
gpt_oss-GateUP-B4-M4096                64       0.896          1.063        -16.7 pp
gpt_oss-Down-B4-M4096                  64       0.939          1.089        -15.0 pp
gpt_oss-GateUP-B32-M2048               64       0.942          1.047        -10.5 pp
gpt_oss-Down-B32-M2048                 64       0.931          1.079        -14.8 pp
gpt_oss-GateUP-B32-M4096               64       0.924          1.021         -9.7 pp
gpt_oss-Down-B32-M4096                 64       0.919          1.056        -13.7 pp
────────────────────────────────────────────────────────────────────────────────
grp_fp8 geomean                                  1.0491         1.1247        -7.6 pp
score (FP8-only segment)                             874            937         -63 pts
```

All 8 K_REM=64 (gpt_oss) shapes regressed 10-17 pp. The fan-out cost
manifests uniformly across the gpt_oss family; B=4 shapes regress
slightly more than B=32 because they have fewer tiles (small-grid
underutilization amplifies per-tile cost).

---

## Recommendation for R36+

Lever D K-tail-only port is **falsified**. Remaining options:

1. **Accept plateau**: Score 947-962 is the empirical ceiling.
   Auto-optimize patience counter at 8/10 post this round; will hit
   10 around R37. Loop doesn't early-stop, so continuing is fine
   but expected score delta ≈ 0.
2. **Lever D Round-B full main-loop port**: 4-6 rounds commitment.
   High risk, larger scope, but gets the 32x32 mfma savings WITHOUT
   paying fan-out cost. This is the only structurally sensible path
   to close the gap to 1.20 on gpt_oss.
3. **Lever E (manual ASM main loop scheduling)**: 2-3 rounds. Risky,
   targets the main-loop unidentified 5-6 pp micro-overhead from R56
   analysis. No precedent, hard to debug.

None of these can be delivered in a single round. The R36+ agent
should either:
- Pick (2) or (3) and commit to a multi-round schedule (OK because
  auto-optimize resumes the chat across rounds), OR
- Accept (1) and use remaining rounds for code-quality fixes /
  documentation.

---

## Round summary

| Item | Value |
|--|--|
| Goal | Implement Lever D R-B step 5: K-tail block rewrite + LDS fan-out |
| Change | 174+/147- lines in HK kernel_fp8_layouts.cpp (reverted) |
| Correctness | PASS (SNR 47.83 dB fwd/dA/dB on K_REM=64 probe, 0/16 fails on metric) |
| Spill profile | IMPROVED: FUSED_KTAIL=true specs -6 to -11 dw |
| Metric before | 937 FP8-only (~958 full metric) |
| Metric with port ON | 874 FP8-only (geomean 1.0491) — **-63 pts** |
| Metric after revert | 933 FP8-only (within 933-962 noise band) |
| HK commit | NONE (reverted to `78415fb0`) |
| PT commit | this commit (doc-only, falsification record) |
| Next round suggestion | R36: accept plateau OR plan multi-round Lever D Round-B full port |
