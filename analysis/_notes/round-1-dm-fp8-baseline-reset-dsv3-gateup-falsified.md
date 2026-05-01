# Round-1 (death-march, fresh run) — FP8 grouped baseline reset + DSV3-GateUP config-rule falsified

**Date**: 2026-05-01
**Repo / HEAD**: Primus-Turbo `ea6994d` (round-27 FP8 K-tail hoist falsified); HipKittens `a0644b80` (round-19 mirror)
**Score tracker state**: baseline=811, historical_best=811, patience=30, unimproved=0
**Target**: all 16 `grp_FP8` cases ratio ≥ 1.20× Triton → geomean ≥ 1.20 → score=1000
**Result**: no kernel / config change shipped — baseline re-anchored + one config-tuning lever falsified.

---

## 1. Fresh-run baseline (today, GPU 3, 2 consecutive `_metric_grouped_only.py` runs)

```
run 1 (cold):  score=822  grp_FP8 geomean=0.9861
run 2 (warm):  score=813  grp_FP8 geomean=0.9761

noise band:    ±0.5pp geomean  (~±10 score points)
round-1 baseline ≈ 817 ± 10 (=0.98 ± 0.005 geomean)
```

All 16 FP8 shapes correctness PASS, 0 rejects, 16/16 below target. 16/16 watch BF16
cases also correctness PASS. Segment filter = `grp_fp8`; BF16 excluded from score.

### Per-shape ratios (run 2, representative)

| shape                                    | hk_tflops | trt_tflops | ratio |
|------------------------------------------|-----------|------------|-------|
| grpFP8_DeepSeek-V3-GateUP-B16-M2048      | 1378.5    | 1388.0     | 0.993 |
| grpFP8_DeepSeek-V3-Down-B16-M2048        | 1156.4    | 1181.7     | 0.979 |
| grpFP8_DeepSeek-V3-GateUP-B16-M4096      | 1648.5    | 1630.1     | 1.011 |
| grpFP8_DeepSeek-V3-Down-B16-M4096        | 1359.6    | 1431.9     | 0.950 |
| grpFP8_DeepSeek-V3-GateUP-B32-M2048      | 1393.8    | 1399.8     | 0.996 |
| grpFP8_DeepSeek-V3-Down-B32-M2048        | 1189.3    | 1217.9     | 0.977 |
| grpFP8_DeepSeek-V3-GateUP-B32-M4096      | 1665.6    | 1607.5     | 1.036 |
| grpFP8_DeepSeek-V3-Down-B32-M4096        | 1392.1    | 1435.4     | 0.970 |
| grpFP8_gpt_oss_20B-GateUP-B4-M2048       | 1039.9    | 1120.3     | 0.928 **lowest** |
| grpFP8_gpt_oss_20B-Down-B4-M2048         | 782.1     | 799.6      | 0.978 |
| grpFP8_gpt_oss_20B-GateUP-B4-M4096       | 1235.4    | 1309.2     | 0.944 |
| grpFP8_gpt_oss_20B-Down-B4-M4096         | 1090.6    | 1171.3     | 0.931 |
| grpFP8_gpt_oss_20B-GateUP-B32-M2048      | 1197.0    | 1214.7     | 0.985 |
| grpFP8_gpt_oss_20B-Down-B32-M2048        | 1033.5    | 1044.6     | 0.989 |
| grpFP8_gpt_oss_20B-GateUP-B32-M4096      | 1405.3    | 1436.7     | 0.978 |
| grpFP8_gpt_oss_20B-Down-B32-M4096        | 1204.9    | 1230.6     | 0.979 |

**Observations**:

1. `gpt_oss-GateUP-B4-M2048` at 0.928 is the lowest ratio and the task body's
   canonical round target (K=2880, K_REM=64, N=5760, N_REM=128 — both K-tail
   and N-tail active). Sibling `gpt_oss-Down-B4-M4096` at 0.931 is effectively
   tied.
2. DSV3-Down family (K=2048, N=7168, both 128/256-aligned → no K-tail, no N-tail)
   sits at 0.950-1.018 — isolates pure main-loop throughput regression vs Triton
   persistent grouped. Already anchored at `(gm=32, xcd=2)` by round-20 / 67 / 68.
3. DSV3-GateUP family (K=7168, N=4096) is at 0.993-1.036 — near-break-even vs
   Triton. Falls through to the default FP8 rule `(gm=4, xcd=None=8)` with no
   specific rule match (see §3 probe result).
4. All 8 gpt_oss cases (K=2880) cluster at 0.928-0.989 — the K-tail + N-tail
   cost is already amortised by the round-3 path-B fuse (commit 07354791) +
   round-12 column-masked C-store; remaining gap is main-loop throughput and
   per-tile completion-latency variance.

## 2. Prior-round falsifications carried forward (do NOT retry)

From HipKittens `analysis/_notes/`:

* **Round-6 (FP8)** — Naïve dense 2-tile main-loop body port to grouped_rcr_kernel
  regressed −144 score (−28.7pp grp_FP8 geomean). `RCR_MAIN_UNROLL=2` +
  `RCR_TWO_TILE_MID_VMCNT` tuned for dense prologue, grouped prologue is longer
  (binary-search + group-by-M swizzle). Only open 2-tile option left:
  grouped-specific `MID_VMCNT` sweep with `unroll=1`.
* **Round-12 (FP8)** — rocprof on `gpt_oss-GateUP-B4-M2048` confirmed the 10.1 ms
  per-iter gap is entirely in the GEMM kernel, not quantize scaffolding. Even if
  HK matched Triton GEMM exactly, total time → 255µs → ratio ≈ 1.00. Hitting 1.20×
  requires HK to be **20% faster than Triton** on raw GEMM, i.e. kernel rewrite.
* **Round-12 (FP8)** — MFMA cell-shape probe falsified: gfx950 `v_mfma_f32_16x16x32_fp8`
  is actually a K=128-wide MFMA, already at par with `_32x32x64_f8f6f4`. Switching
  to 32×32 cells does NOT save MFMA cycles.
* **Round-15 (FP8)** — Lowering `RCR_TWO_TILE_MIN_KI` 28→20 for gpt_oss ki=22 was
  within ±0.23 % noise on all 8 shapes. The 2-tile schedule does not amortise
  at ki=22.
* **Round-27 (FP8)** — K-tail `load_a_kt(a_kt1)` single-load hoist regressed
  −4.9pp grp_FP8 due to VGPR live-range pressure on the 256-VGPR ceiling kernel.
* **Rounds 5, 8, 14, 16, 22, 24, 25 (BF16 + FP8)** — All micro VMCNT /
  sched_barrier / chunk_size / BN=128 / config-knob variants saturated or
  falsified. Per task body: do not re-litigate.
* **BF16 segment** — explicit `[watch]` only; score does not move on BF16
  ratio changes. K_STEP=64→32 port deprioritised (round-25 doc); do NOT
  spend rounds on BF16 this run.

## 3. Round-1 probe — DSV3-GateUP FP8 (gm, xcd) sweep — **FALSIFIED**

### Hypothesis

DSV3-GateUP family (tiles_n=16, tiles_m ∈ {8, 16}, k=7168) has **no** explicit FP8
rule in `primus_turbo/pytorch/kernels/hipkitten/config.py`; falls through to the
binding default `(gm=4, xcd=None → BLOCK_SWIZZLE_NUM_XCDS=8)`. BF16 does have a
rule for the same tile geometry (`tiles_m==8, tiles_n==16, k<=7168 → gm=1, xcd=4`,
round-10). Worth checking if an analogous FP8 rule wins 0.5-1.5pp and lifts the
4 DSV3-GateUP ratios from 0.99-1.04 → ≥1.00.

### Method

`/tmp/probe_dsv3_gateup_fp8_round1.py` — 8-candidate `(gm, xcd)` sweep
(`(4,None), (4,4), (4,2), (2,4), (2,8), (1,4), (1,2), (8,4)`) on the 4 DSV3-GateUP
grouped FP8 shapes. ITERS=80 per trial, REPEATS=5 p20-min. Default `(4, None)`
as baseline; delta in %.

### Result — no winner

```
shape                                config     tflops  Δ vs (4,None)
DeepSeek-V3-GateUP-B16-M2048         (4,None)   1364.75    +0.00%
                                     (4,4)      1370.59    +0.43%   *noise
                                     (4,2)      1347.37    -1.27%
                                     (2,4)      1361.87    -0.21%
                                     (2,8)      1365.15    +0.03%
                                     (1,4)      1366.10    +0.10%
                                     (1,2)      1364.79    +0.00%
                                     (8,4)      1345.04    -1.44%

DeepSeek-V3-GateUP-B16-M4096         (4,None)   1611.78    +0.00%
                                     (4,4)      1608.08    -0.23%
                                     (4,2)      1600.91    -0.67%
                                     (2,4)      1617.89    +0.38%
                                     (2,8)      1619.12    +0.46%
                                     (1,4)      1608.74    -0.19%
                                     (1,2)      1622.16    +0.64%   *noise-ish
                                     (8,4)      1612.74    +0.06%

DeepSeek-V3-GateUP-B32-M2048         (4,None)   1395.71    +0.00%
                                     (4,4)      1395.03    -0.05%
                                     (4,2)      1378.67    -1.22%
                                     (2,4)      1395.76    +0.00%
                                     (2,8)      1384.72    -0.79%
                                     (1,4)      1394.97    -0.05%
                                     (1,2)      1395.31    -0.03%
                                     (8,4)      1395.88    +0.01%   *flat

DeepSeek-V3-GateUP-B32-M4096         (4,None)   1663.41    +0.00%
                                     (4,4)      1662.68    -0.04%
                                     (4,2)      1661.11    -0.14%
                                     (2,4)      1660.65    -0.17%
                                     (2,8)      1660.19    -0.19%
                                     (1,4)      1651.91    -0.69%
                                     (1,2)      1650.94    -0.75%
                                     (8,4)      1651.66    -0.71%   *default wins
```

### Verdict

**No candidate wins across all 4 shapes by more than ≈0.5pp** (noise-band ≈0.3%).
The best winner per-shape is different for each shape:
* B16-M2048: `(4, 4)` +0.43%
* B16-M4096: `(1, 2)` +0.64% (sibling `(2, 8)` at +0.46%)
* B32-M2048: effectively flat — default is within 0.1pp of top
* B32-M4096: default is the **top** by 0.04-0.75pp (all other configs lose)

No single rule-worthy transfer. The default `(gm=4, xcd=None)` is at or near a
local optimum across the family — BF16's `(gm=1, xcd=4)` rule does **not** port
to FP8. Adding a shape-specific rule on a sub-0.5pp margin would be noise-chasing
(exactly what round-23 note on FP8 gpt_oss-GateUP-B4-M2048 warned against:
"candidate's min still beats the default's max only marginally — but the median
gap is consistent across all 7 repeats" = the minimum bar, and this sweep does
not clear even the median bar).

Why the transfer fails: FP8 RCR main-loop VGPR pressure (256 VGPR ceiling, 67
spill dwords at FUSED_KTAIL=true) is **categorically tighter** than BF16 RCR
(256 VGPR, typical occupancy 2 w/SIMD with 2 pad-dwords spill). BF16's (gm=1,
xcd=4) benefit comes from L2 reuse on the B-pack when walking the full N-row;
FP8's per-tile completion latency is MFMA-bound (`rcr_mma` ~60 % of kernel
cycles per round-12 rocprof), so tile schedule order matters less than the MFMA
issue-rate — the default's natural 8-XCD spread already saturates the issue
queue.

## 4. Available "big" levers (round-2+)

Per task body §"Allowed big levers", all kernel-internal (no dispatch fallback,
no `_pad_2d`, no quantize fuse):

| lever | scope | est. yield | round investment |
|-------|-------|-----------:|-----------------:|
| A. Main-loop pipeline port from BF16 (deeper ks unroll, s_setprio, sched_barrier placement) | grouped_rcr_kernel lines 2161-2189 | +1-2pp | 1 round per knob |
| B. LDS bank conflict elimination on FP8 K_BLOCK=128 LDS (ds_read_b128 swizzle / padding, cross-ref `ST_v2` vs BF16 `st_16x32_s` / `st_32x16_s`) | grouped_rcr_kernel LDS staging | +2-5pp | 2-3 rounds |
| C. Register-tile RBM×RBN / WARPS_N re-split (FP8 grouped currently 64×32 × WARPS_N=2 × WARPS_M=4) | kernel template + rt_* types | +2-4pp | 3-4 rounds (VGPR re-derivation) |
| D. K-tail epilog amortise across multi-tile-M (persistent wg processes multi BM rows per outer iter; K-tail fuse runs once per M-slab) | grouped_rcr_kernel FUSED_KTAIL block | +1-3pp on K=2880 | 2 rounds |
| E. **Direct HBM→reg main loop** (round-6 option c) — remove A-tile LDS staging → remove 8 cross-warp `s_barrier` / K-iter | grouped_rcr_kernel main loop | +5-7pp | 4-8 rounds |
| F. dB CRR (var-K) optimization | dispatch_grouped_var_k_fp8 line 5532 | fwd metric INVISIBLE | off-target |
| G. XCD swizzle / grid-walk order for B=4 shapes | persistent grid + chiplet_transform_chunked | +1-2pp gpt_oss-B4 | 1-2 rounds |

## 5. Round-2 suggestion

**Pivot away from config tuning entirely** (§3 falsified another config-rule
lever; task body explicitly cautions against burning rounds here). Start
lever **A (main-loop pipeline micro-port)** or **G (XCD swizzle for B=4)** as
the 1-round WIP probe, with the following concrete first sub-step:

* **A1**: Audit the FP8 `grouped_rcr_kernel` main-loop (line 2161-2189)
  `s_barrier` + `s_setprio` + `s_waitcnt` sequence and diff against the BF16
  `grouped_kernel` / `device_gemm_tile_body` main_loop_iter (`kernel_bf16_dynamic.cpp`
  line 600-686). Target: identify at least one BF16-tuned pattern that is
  simpler in FP8 (e.g. fewer s_barriers, different s_setprio cadence, or a
  different `vmcnt` drain position). Ship a single small knob change + metric
  verify. Estimated +0.3-1.5pp grp_FP8 geomean = +3-15 score.

* Parallel track: keep the **multi-round lever E (direct HBM→reg main loop)**
  as the round-3+ backbone work. Lever E is the only documented path to +5-7pp,
  which is what's needed to take the round-1 baseline 0.98 → 1.05 (still
  far from 1.20 target, but the first 5-7pp is the best-evidenced hope).

## 6. Files touched

* `analysis/_notes/round-1-dm-fp8-baseline-reset-dsv3-gateup-falsified.md`
  (this file) — Primus-Turbo side.
* HipKittens side: mirror note at
  `analysis/_notes/round-1-dm-dsv3-gateup-fp8-config-falsified.md` (planned
  in same commit sequence).

No kernel, dispatch, config, or test code changed this round.

## 7. Verification

```bash
# 2× metric runs on pinned GPU 3, noise band ±10 score:
cd /workspace/code/Primus-Turbo
python3 scripts/_metric_grouped_only.py  # 822 (geomean 0.9861)
python3 scripts/_metric_grouped_only.py  # 813 (geomean 0.9761)

# DSV3-GateUP (gm, xcd) sweep:
python3 /tmp/probe_dsv3_gateup_fp8_round1.py  # 8 cfg × 4 shapes, ±1.5% all noise
```
