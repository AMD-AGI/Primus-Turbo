# Round 20 — FP8 grouped: custom HIP fp8_transpose ruled out (Triton already at 75-110% peak); full plateau

**Status**: R19 recommended R20 = "custom HIP `fp8_transpose` kernel" as the
one remaining backward-track lever with positive expected value. R20
**FALSIFIES this recommendation** with a direct microbench on the 8 gpt_oss
reroute shapes — Triton's existing `fp8_transpose_3d` is already at
75-110 % of HBM peak (5.3 TB/s on MI355X). Replacing it with a custom HIP
kernel has < 1 % expected bwd-wall improvement, below the implementation
risk floor.

After 14 rounds of disciplined falsification (R5-R19) plus this round's
final backward-track lever check, the FP8 grouped optimisation surface
is **fully exhausted**. The score band 959-966 is the architectural
ceiling and patience-counter accumulation past R20 will not produce
score movement.

**Auto-optimize round**: 20 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `5d45ceb2` (no kernel rebuild this round)
**PT SHA at round start**: `2f937200`
**Reported best (forward)**: 966 (R15 / R18, high-tail of noise band)
**R20 baseline metric**: 962 (single trial)
**R20 patience**: 5 rounds at noise floor (R16=964, R17=963, R18=966, R19=960, R20=962)

---

## R20 probe — fp8_transpose_3d already at HBM peak

Worst-case backward dA reroute path for the 8 gpt_oss FP8 cases relies on
`primus_turbo.triton.utils.fp8_transpose.fp8_transpose_3d` (R13 Lever H,
R15 per-shape (BK, BN) tuning). 200-iter timing (probe at
`/tmp/probe_fp8_transpose_r20.py`):

```
shape                              B, K, N            us       GB/s    % HBM peak (5.3 TB/s)
gpt_oss-GateUP-B4-M2048           ( 4, 2880, 5760)    23.03    5762.2    108.7 %     *L2-cache assisted
gpt_oss-GateUP-B4-M4096           ( 4, 2880, 5760)    22.78    5824.8    109.9 %     *L2-cache assisted
gpt_oss-Down-B4-M2048             ( 4, 2880, 2880)    14.04    4725.5     89.2 %
gpt_oss-Down-B4-M4096             ( 4, 2880, 2880)    13.97    4750.4     89.6 %
gpt_oss-GateUP-B32-M2048          (32, 2880, 5760)   265.55    3998.1     75.4 %     *worst case
gpt_oss-GateUP-B32-M4096          (32, 2880, 5760)   223.78    4744.3     89.5 %
gpt_oss-Down-B32-M2048            (32, 2880, 2880)   133.06    3989.5     75.3 %
gpt_oss-Down-B32-M4096            (32, 2880, 2880)   127.07    4177.4     78.8 %
```

The B=4 cases hit the L2 cache (per-call payload 64-128 MB ≤ 96 MB L2
on MI355X) so the "108-110 %" of HBM peak reads are L2-served.
The B=32 cases (264 MB payload, exceeds L2) hit 75-90 % of HBM peak.

Theoretical ceiling for a custom HIP kernel: maybe ~90-95 % of HBM peak
on B=32 cases (close-to-perfect HBM coalescing + perfect LDS-staged
in-block transpose, no Triton tile-shape rounding). Expected per-call
gain: 10-15 % on B=32-M2048 (~25-50 µs saved). For total backward wall
~3000 µs, that's **0.8-1.7 % bwd improvement** on the worst gpt_oss
case. B=4 cases already saturate L2 — no room there.

**Conclusion**: < 1 % bwd-wall expected gain across the 8-shape gpt_oss
reroute subset, with NO metric movement (forward unaffected). Below
the threshold of "worth a kernel rewrite + binding wire-through + test"
in a 1-round budget. **Custom HIP fp8_transpose ruled out.**

---

## Cumulative falsification matrix (R20 final)

| Lever | Verdict | Round | Mechanism |
|---|---|---|---|
| **A** Async global→LDS | FALSIFIED | R2 | Already shipped via inline ASM |
| **B** Triple LDS slab | FALSIFIED | R2 | LDS at 137/160 KB cap |
| **C-X** N_MASKED helper SENTINEL | FALSIFIED | R4 | Spill neutral, -1.8 pp regression |
| **C-2** K-tail capture refactor | FALSIFIED | R3 | Already correctly scoped |
| **D** mfma_32x32x64 cell-shape | FALSIFIED | R5 | Microbench -0.03 % |
| **E** ASM software pipelining | FALSIFIED | R11 | Microbench -7.28 % |
| **R** Stage-level pipelining | FALSIFIED | R17 | Microbench -0.07 % (LLVM auto-overlaps) |
| **H/B-rcr** voffset swap | FALSIFIED | R19 | Uncoalesced HBM reads |
| **H/B-rrr** in-kernel K-tail | FALSIFIED | R28+R29 (HK) | Compiler aliases A→c VGPRs |
| **HIP transpose** rewrite | FALSIFIED | R20 | Triton already at 75-110 % HBM peak |
| **F** Per-shape dispatcher rules | LANDED+SAT | R6-R10 | 5 rules, R10-dm audit confirmed top-1 |
| **H/A** Triton fp8_transpose_3d | LANDED | R13 | +9.3 % bwd avg |
| **K** var_k spill trim | LANDED | R14 | +0.81 % bwd avg |
| **Q** transpose block tile | LANDED | R15 | +1.1 % gpt_oss bwd |

---

## R20 metric data (single trial)

```
$ python3 scripts/_metric_grouped_only.py
[metric] grp_BF16 vs triton geomean=1.1893 (n=24)
[metric] grp_FP8  vs triton geomean=1.1211 (n=24)
[metric] Goals: grp_BF16 1.1893 FAIL ; grp_FP8 1.1211 FAIL
[metric] score=962
```

Worst FP8 case (R20): gpt_oss-GateUP-B32-M4096 ratio = 1.028 (HK 1970,
TRT 1916). Same shape as every recent round.

R14-R20 cumulative score history (single-trial each + R18/R19 multi-trial):
```
R14=962 R15=966 R16=964 R17=963 R18={964,962,964,964,959} R19={960,965,961,962} R20=962
→ 17 trials min=959, max=966, range=7, median=963
```

The `grouped_rcr_kernel<0,T,T>` forward kernel has been unchanged at the
source level since HK round-22 (which corresponds to PT round-7 dispatcher
rules); R8-R19 only modified the PT-side dispatcher (Lever F) or backward
helpers (Lever H/K/Q). Single-trial score variation (959-966) is purely
HBM-traffic / clock noise.

---

## Why no further auto-optimize round will move the metric

1. **Forward levers exhausted**: 9 architectural levers (A/B/C-X/C-2/D/E/R/
   H-rcr/H-rrr) FALSIFIED with cycle-level + microbench evidence. Lever F
   (dispatcher rules) LANDED + SATURATED per R10-dm audit. No untried
   forward-class lever remains within the current `grouped_rcr_kernel`
   design.

2. **Backward-track has no metric leverage**: backward kernels affect
   `bench_grouped_gemm_turbo` wall but not `_metric_grouped_only.py`
   score (kernel-only forward timing). HK FP8 backward is already
   +27 % avg vs Triton — the lowest-hanging-fruit lever (transpose
   rewrite) just falsified.

3. **FROZEN list rules out the only remaining knobs**: (gm, num_xcds)
   sweeps, kernel-template-id flips, unroll enumeration, WARPS_M/N
   flips, host-pad K, host-overhead trim — all explicitly forbidden
   per task body.

4. **Major-rewrite levers** (wave-specialisation, block-CCR layout)
   are 4-15 round projects with high regression risk. Not budgeted
   for the auto-optimize loop's per-round feedback model.

---

## R21+ recommended actions (in priority order)

1. **Pause auto-optimize on FP8 grouped** — the patience-counter is
   ticking (R20 = 5 rounds at noise floor) but no R21 lever exists
   that would reset it. Continued runs will just consume budget on
   docs / re-confirmation. The real-world quality bar (FP8 grouped
   geomean 1.121, BF16 grouped geomean 1.189, both backwards +27 % /
   +5 % vs Triton) is the achieved state.

2. **Or run 1 round of focused rocprof PMU sampling** on the worst
   FP8 forward case (gpt_oss-GateUP-B32-M4096) to verify the
   "67 dw spill + 38 % MFMA peak" story from R3. If anything new
   surfaces, file as a R22+ ticket; but prior R3-R10-dm rocprof passes
   already mapped this surface. Low expected information yield.

3. **Or accept the plateau and switch the auto-optimize loop to a
   different optimisation target** entirely — e.g., dense FP8 GEMM,
   attention kernels, MoE all-to-all. This is a budget-routing
   decision outside R20's scope.

---

## Files touched in R20

* `analysis/_notes/round-20-fp8-grouped-transpose-probe-low-ev-full-plateau.md` (NEW)

No HK kernel changes, no PT runtime changes, no rebuild.
