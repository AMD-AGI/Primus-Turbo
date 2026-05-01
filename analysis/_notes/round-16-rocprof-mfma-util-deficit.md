# Round 16 — rocprof PMC breakdown localizes 8pp MfmaUtil deficit

## TL;DR

- HK FP8 grouped main loop runs MFMA at **33.6%** utilisation while Triton runs at **41.9%** on the same shape. Closing this 8pp gap is worth ~20% kernel wall-time (matches the 0.834 ratio gap on `grpFP8-Down-B4-M4096`).
- The HK kernel does **+33% more SQ_BUSY_CYCLES** for the same grid because its main loop spends more cycles on non-MFMA work (address compute / s_barrier / s_waitcnt / load staging) than Triton's persistent kernel.
- **Knob sweep confirmed the gap is structural, not tuning**: `RCR_MAIN_UNROLL ∈ {1, 2, 4}` all produced identical MfmaUtil (33.55±0.05%) — compiler already at its local optimum given the source structure.
- BF16 has the same shape of gap (4-5pp MfmaUtil deficit) but smaller in magnitude — consistent with BF16 ratios being closer to 1.0.

## What we measured

`rocprofv3 --pmc MfmaUtil OccupancyPercent SIMD_UTILIZATION GPU_UTIL MemUnitStalled SQ_BUSY_CYCLES` over 50 hot-loop iterations of the HK and Triton FP8/BF16 grouped GEMM on representative gpt_oss shapes.

### FP8 — `Down-B4-M4096` (worst case, ratio 0.834)

| counter            | HK     | Triton | delta (HK − TRT) |
|--------------------|--------|--------|------------------|
| MfmaUtil           | 33.6%  | 41.9%  | **−8.3 pp**      |
| OccupancyPercent   | 22.4%  | 20.9%  | +1.5 pp          |
| SIMD_UTILIZATION   | 89.4%  | 84.9%  | +4.5 pp          |
| MemUnitStalled     | 23.1%  | 42.0%  | **−18.9 pp**     |
| SQ_BUSY_CYCLES     | 12.2 M | 9.19 M | **+33 % more**   |
| GPU_UTIL           | 100 %  | 100 %  | 0                |

### FP8 — `GateUP-B32-M4096` (B=32 sanity check, ratio 0.863)

| counter            | HK      | Triton  | delta            |
|--------------------|---------|---------|------------------|
| MfmaUtil           | 36.1%   | 42.9%   | **−6.8 pp**      |
| MemUnitStalled     | 20.9%   | 42.8%   | **−21.9 pp**     |
| SQ_BUSY_CYCLES     | 187 M   | 156 M   | +20 % more       |

### BF16 — `GateUP-B4-M2048` (ratio 1.006, near tie)

| counter            | HK     | Triton | delta            |
|--------------------|--------|--------|------------------|
| MfmaUtil           | 50.8%  | 55.1%  | −4.3 pp          |
| MemUnitStalled     | 13.5%  | 31.3%  | −17.8 pp         |
| SQ_BUSY_CYCLES     | 14.9 M | 13.0 M | +14 % more       |

### BF16 — `Down-B32-M2048` (ratio 1.089, HK wins)

| counter            | HK     | Triton | delta            |
|--------------------|--------|--------|------------------|
| MfmaUtil           | **55.4%**  | 50.5%  | **+4.9 pp**  |
| MemUnitStalled     | 13.3%  | 26.0%  | −12.7 pp         |
| SQ_BUSY_CYCLES     | 60.4 M | 66.2 M | **−9 % less**    |

The pattern is unambiguous: **whenever HK loses, MfmaUtil is below Triton; whenever HK wins, MfmaUtil is above Triton.** The metric ratio tracks MfmaUtil rank order across shapes.

## Diagnosis: HK main loop has more "auxiliary" work per K-iter

Triton's persistent kernel has *more* memory stalls (40-43% vs HK's 13-23%) yet still completes in fewer SQ cycles. That can only happen if Triton issues MFMAs more densely between non-MFMA instructions. Concretely:

- HK: 33.6% × 12.2 M = **4.10 M MFMA cycles**, 8.10 M non-MFMA cycles.
- TRT: 41.9% × 9.19 M = **3.85 M MFMA cycles**, 5.34 M non-MFMA cycles.
- HK does roughly the same amount of MFMA work but **+50% more non-MFMA SQ work** (8.10 M vs 5.34 M).

The HK FP8 RCR grouped main-loop body (`kernel_fp8_layouts.cpp:2092-2120`) has 4 MFMAs surrounded by:

- 4× `load_b` / `load_a` (LDS reads)
- 4× `rcr_8w_load_hoist` (HBM prefetch for k+2)
- 8× `__builtin_amdgcn_s_barrier()`
- 4× `s_waitcnt lgkmcnt(0)` + 1× `TK_WAIT_LGKM(...)` + 1× `TK_WAIT_VMCNT(...)`
- 8× `__builtin_amdgcn_s_setprio(0/1)` (priority hints around each MFMA)

That's ~28 visible asm-level instructions per K-iter for 4 MFMAs → **MFMA is ~14% of the iteration's instruction count**. The 33.6% MfmaUtil comes from MFMA latency overlapping with the surrounding ops. Triton's main loop has fewer non-MFMA instructions and lets MFMA dominate.

## Falsified knob: `RCR_MAIN_UNROLL` is saturated

```
HK FP8 Down-B4-M4096, varying RCR_MAIN_UNROLL:
  UNROLL=1   MfmaUtil = 33.55%   SQ_BUSY = 12.23 M    spill = 0
  UNROLL=2   MfmaUtil = 33.63%   SQ_BUSY = 12.22 M    spill = 0   (baseline)
  UNROLL=4   MfmaUtil = 33.55%   SQ_BUSY = 12.17 M    spill = 0   (no extra spills in grouped_rcr_kernel)
```

The compiler already reaches the same instruction schedule regardless of the `#pragma unroll` hint — the body is small enough that the compiler can pipeline two iters' loads/MFMAs without explicit unrolling. So this knob is dead. Reverted to UNROLL=2 (the long-standing baseline; matches dense kernel, no behaviour change).

## What the data rules out

1. **Tile size (BN=128 vs 256)** — round 15 already showed Triton picks BN=256 too. Now reinforced: HK's deficit is *not* in geometry but in the inner loop's instruction mix.
2. **Memory bandwidth saturation** — HK's MemUnitStalled is *lower* than Triton's. HK is not HBM-starved; it's instruction-issue-limited.
3. **Wave occupancy** — HK has *higher* OccupancyPercent than Triton (22.4 vs 20.9). Adding more occupancy will not help.
4. **Compile-time `#pragma unroll`** — falsified above.

## What the data points at (next-round candidates, ordered by leverage)

1. **Reduce `s_setprio` density.** 8 setprio instructions per K-iter (2 per MFMA) is heavy. Round-2 falsified removing them *together with* sched_barrier, but never tested removing s_setprio in isolation. Each setprio is 1 cycle of SGPR work; cutting them in half (e.g. only the (1) before MFMA, drop the (0) after) saves ~4 cycles per K-iter × 22 K-iters × 768 tiles = ~67 K SQ cycles per kernel — about 0.5% if it doesn't break MFMA priority. **Risk: the (0)→(1) toggle is what gives MFMA priority over the next iter's load-issue; removing the post-(0) might let the next setprio(1) fire too late.**
2. **Drop one of the 8 `s_barrier`s per K-iter.** Each s_barrier forces all warps in the WG to wait. The 4 paired barriers around s_waitcnt might be collapsible if we move to a single barrier per LDS-fence point. Round-7 falsified BF16 K-tail single-wait (different code path), but main-loop barrier reduction has not been swept on FP8.
3. **Hoist k+2 prefetches out of the K-iter body.** Currently `rcr_8w_load_hoist` runs interleaved with MFMAs; if its address computation could be moved to a single batched chunk at the top of the body, the compiler might pack the 4 MFMAs more tightly.
4. **Shorten the LDS-stage round-trip.** The 2-stage LDS pipeline (As/Bs[2][N]) plus per-iter swap (`tic ^= 1, toc ^= 1`) costs registers and SGPRs. A direct HBM→register path (round 5/6 attempted, regressed at the time) might be revisitable now that we can measure MfmaUtil delta directly.

Each of (1)-(4) needs to be guarded by:
- **Mechanism evidence**: rocprof MfmaUtil must rise by ≥2pp.
- **Metric evidence**: ≥1.5pp median shift over ≥10 metric runs.

## Score impact this round

- Metric this round: **795** (unchanged from round-15 best). Notes-only commit; no kernel change shipped.
- The diagnostic itself does not move the score, but localises the remaining gap quantitatively so future rounds can target the right thing.

## Probe artefacts

- `/tmp/rocprof_round16/{hk,trt}_{fp8,bf16,b32}_counter_collection.csv` — raw rocprof CSVs.
- `/tmp/bench_one_shape.py`, `/tmp/bench_bf16_one.py` — minimal hot-loop drivers.
- `/tmp/analyze_rocprof.py` — kernel-name filter + counter aggregation.

## Round-16 commits

- Primus-Turbo: this notes file. No kernel change.
- HipKittens: none (UNROLL=4/1 experiments reverted to UNROLL=2 baseline; falsified at the rocprof level, no commit warranted).
