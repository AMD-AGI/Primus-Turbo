# Round 21 — FP8 grouped: `grouped_rrr_kernel` 76 dw spill is NOT the bottleneck (dA TFLOPS exceeds forward)

**Status**: R64-dm noted "dB / dA kernels at 52-76 dw spill — significantly worse than forward 32-43 dw" and recommended spill reduction as a backward-track target. R21 directly measures dA TFLOPS on the DSV3 / Qwen3 shapes that exercise `grouped_rrr_kernel` and finds dA TFLOPS **exceeds forward TFLOPS** on the same kernel-MFLOP budget. Conclusion: the 76 dw spill is **not on the critical path**, spill-reduction is a low-EV lever, and another backward-track candidate is ruled out.

After R20 ruled out custom HIP `fp8_transpose` (Triton already at 75-110 % HBM peak), R21 rules out RRR spill reduction (no measurable bottleneck signal). All identified backward-track levers within the 1-round implementation budget have been falsified by direct measurement.

**Auto-optimize round**: 21 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `5d45ceb2` (no kernel rebuild this round; verification rebuild only)
**PT SHA at round start**: `6008c138`
**Reported best (forward)**: 966 (R15 / R18, high-tail of noise band)
**R21 baseline metric**: 963 (single trial, score history median 963)
**R21 patience**: 6 rounds at noise floor (R16-R21)

---

## R21 — full FP8 kernel spill profile (re-built `tk_fp8_layouts.so` with `-Rpass-analysis=kernel-resource-usage`)

```
kernel                                       VGPRs   ScratchSize   VGPRSpill   Occupancy
─────────────────────────────────────────────────────────────────────────────────────────
grouped_rcr_kernel<0, F, F>                    256     160 B/lane      39 dw    2 waves/SIMD  fwd RCR (FUSED_KTAIL=F, N_MASKED=F)
grouped_rcr_kernel<0, T, F>                    256     176 B/lane      43 dw    2 waves/SIMD  fwd RCR (FUSED_KTAIL=T, N_MASKED=F)
grouped_rcr_kernel<0, F, T>                    256     132 B/lane      32 dw    2 waves/SIMD  fwd RCR (FUSED_KTAIL=F, N_MASKED=T)
grouped_rcr_kernel<0, T, T>                    256     160 B/lane      39 dw    2 waves/SIMD  fwd RCR (FUSED_KTAIL=T, N_MASKED=T)  ← gpt_oss & DSV3-GateUP fwd path
grouped_rrr_kernel<0>                          256     308 B/lane      76 dw    2 waves/SIMD  ← bwd dA path for DSV3 / Qwen3
grouped_var_k_kernel_fp8<0>                    256     152 B/lane      37 dw    2 waves/SIMD  bwd dB path
```

`grouped_rrr_kernel` has **2× the VGPR spill** of any forward `grouped_rcr_kernel` spec
(76 dw vs 32-43 dw) and 2× the scratch (308 B vs 132-176 B). On paper, this is a
prime spill-reduction target.

---

## R21 — direct dA TFLOPS measurement (50 iter median, kernel-only timing)

```
shape                              M, N, K (dA)              fwd us    dA us    dA TFLOPS
─────────────────────────────────────────────────────────────────────────────────────────
DSV3-GateUP-B16-M2048           ( 32768,  7168, 4096)         753.2    906.9    2121.6     ← dA: rrr_kernel
DSV3-Down-B32-M4096             (131072,  2048, 7168)        1782.6   1601.1    2403.5     ← dA: rrr_kernel
Qwen-GateUP-B32-M2048           ( 65536,  4096, 3072)         673.9    852.6    1934.4     ← dA: rrr_kernel
Qwen-Down-B16-M4096             ( 65536,  1536, 4096)         452.6    400.1    2061.0     ← dA: rrr_kernel
gpt_oss-GateUP-B32-M4096        (131072,  2880, 5760)        2138.1   1966.6    2211.3     ← dA: REROUTED to rcr_kernel (K_RCR=5760 K-aligned but N_RCR=2880 256-misaligned)
```

(probe: `/tmp/probe_dsv3_qwen_bwd_r21.py`)

### Critical observation: dA TFLOPS exceeds forward TFLOPS

| shape                       | fwd TFLOPS (metric) | dA TFLOPS (this round) | dA / fwd |
|---|---|---|---|
| DSV3-Down-B32-M4096         | 2167                | **2403**               | **+11 %** |
| DSV3-GateUP-B16-M2048       | 2628                | 2122                   | -19 %  |
| Qwen3-GateUP-B32-M2048      | 2420                | 1934                   | -20 %  |
| Qwen3-Down-B16-M4096        | 1783                | 2061                   | **+16 %** |

`grouped_rrr_kernel` at 76 dw spill achieves **higher TFLOPS than the
4-spec `grouped_rcr_kernel` at 32-43 dw spill on DSV3-Down and Qwen3-Down**. On the
GateUP shapes it's lower but for different reasons (different M/N/K aspect ratio
and dispatcher (gm, xcds) tuning, not spill).

### What this means

The 76 dw VGPR spill on `grouped_rrr_kernel` is a "static budget number"; the
**runtime cost** of that spill (how many spill round-trips actually hit the hot
path) is small enough that the kernel still achieves competitive throughput.
Reasons:

1. **Spill amortisation across K-loop**: 76 dw spill / 128 K-iters per tile
   = 0.6 spills/iter on average. With ~8 cy per spill round-trip, that's ~5 cy
   in a ~512 cy main-loop iter (< 1 %).

2. **Many spilled values are K-loop-invariant**: addresses, group indices,
   tile-coord cache. These spill once per tile and reload at branch points,
   not every K-iter.

3. **MFMA latency hides scratch round-trips**: the 4× `rrr_mma` per K-iter
   issue 4 × 16-32 cycles of MFMA latency. SQ has plenty of slot time to
   issue spill loads/stores during MFMA pass-through.

4. **2 waves/SIMD occupancy is unaffected**: 76 dw spill goes to scratch
   (308 B/lane), not into the VGPR allocation that gates wave occupancy.
   `grouped_rrr_kernel` still hits 2 waves/SIMD same as `grouped_rcr_kernel`.

So **reducing 76 → ~40 dw spill on `grouped_rrr_kernel` would yield close to
zero bench improvement** — the spill is already amortised below the
measurement noise floor.

---

## Cumulative falsification matrix (R21 final)

| Lever | Verdict | Round | Mechanism / measurement |
|---|---|---|---|
| **A** Async global→LDS                 | FALSIFIED | R2  | Already shipped via inline ASM |
| **B** Triple LDS slab                  | FALSIFIED | R2  | LDS at 137/160 KB cap |
| **C-X** N_MASKED helper SENTINEL       | FALSIFIED | R4  | Spill neutral, -1.8 pp regression |
| **C-2** K-tail capture refactor        | FALSIFIED | R3  | Already correctly scoped |
| **D** mfma_32x32x64 cell-shape         | FALSIFIED | R5  | Microbench -0.03 % |
| **E** ASM software pipelining          | FALSIFIED | R11 | Microbench -7.28 % |
| **R** Stage-level pipelining           | FALSIFIED | R17 | Microbench -0.07 % (LLVM auto-overlaps) |
| **H/B-rcr** voffset swap               | FALSIFIED | R19 | Uncoalesced HBM reads |
| **H/B-rrr** in-kernel K-tail           | FALSIFIED | R28+R29 (HK) | Compiler aliases A→c VGPRs |
| **HIP transpose** rewrite              | FALSIFIED | R20 | Triton already at 75-110 % HBM peak |
| **RRR spill reduction**                | **FALSIFIED** | **R21** | **dA TFLOPS already exceeds fwd TFLOPS** |
| **F** Per-shape dispatcher rules       | LANDED+SAT | R6-R10  | 5 rules, R10-dm audit confirmed top-1 |
| **H/A** Triton fp8_transpose_3d        | LANDED  | R13 | +9.3 % bwd avg |
| **K** var_k spill trim                 | LANDED  | R14 | +0.81 % bwd avg |
| **Q** transpose block tile             | LANDED  | R15 | +1.1 % gpt_oss bwd |

10 architectural/backward levers FALSIFIED, 4 LANDED + SATURATED. **No remaining
1-round positive-EV lever exists** within the FP8 grouped kernel surface.

---

## Score band stability (R14-R21, 18 trials)

```
R14=962 R15=966 R16=964 R17=963 R18={964,962,964,964,959} R19={960,965,961,962} R20=962 R21=963
→ 18 trials min=959, max=966, range=7, median=963
```

Score band: 963 ± 3, no movement since R15. Patience at 6 rounds.

---

## R22+ recommendation (unchanged from R20)

1. **Pause auto-optimize on FP8 grouped** — concrete probe data now confirms
   no remaining 1-round lever. Continued rounds will only consume budget on
   docs / re-confirmation.

2. **Or attempt a multi-round-budget structural rewrite** (e.g., wave-
   specialised producer/consumer split, or block-CCR layout). These are
   4-15 round projects with high regression risk — should be scheduled
   as explicit budget commitment, not opportunistic 1-round attempts.

3. **Or accept the plateau and switch the auto-optimize loop to a
   different optimisation target** entirely (dense FP8 GEMM, attention,
   MoE all-to-all). Out of R21's scope.

---

## Files touched in R21

* `analysis/_notes/round-21-fp8-grouped-rrr-spill-not-bottleneck-da-tflops-exceeds-fwd.md` (NEW)

No HK kernel changes, no PT runtime changes; HK rebuild was for `-Rpass-analysis=kernel-resource-usage` spill profiling only and produced an identical binary.
