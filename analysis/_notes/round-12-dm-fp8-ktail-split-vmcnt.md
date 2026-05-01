# Round 12 (death-march) — FP8 grouped K-tail split-vmcnt + load reorder

**Score: 811 → 824 (+13, beats prior best 820 by +4).** First improvement in 7 rounds.

## Target shape

Per round-12 metric baseline (entry score=811, GPU 3 idle), worst FP8
ratio was `grpFP8_gpt_oss_20B-GateUP-B4-M2048 @ 0.924` (B=4, M=2048,
N=2880, K=2880 — K-tail K_REM=64, N-tail N%256=64). 7 of 8 worst-ratio
shapes were gpt_oss B=4 small-batch + B=32 cases that all hit the
`FUSED_KTAIL=true` path. Single-target attack.

## Change

`HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` —
`grouped_rcr_kernel` FUSED_KTAIL block (lines 2425-2438).

The K-tail epilog issues 24 `buffer_load_b128` total
(8 a + 8 a_kt1 + 4 b0 + 4 b1) then a single `vmcnt(0)` + 4 mfmas:

```
load_a_kt(a,     0);   // 8 buffer_load → a (M slab 0)
load_a_kt(a_kt1, 1);   // 8 buffer_load → a_kt1 (M slab 1)
load_b_kt(b0,    0);   // 4 buffer_load → b0
load_b_kt(b1,    1);   // 4 buffer_load → b1
asm volatile("s_waitcnt vmcnt(0)");
rcr_mma(cA, a,     b0);
rcr_mma(cB, a,     b1);
rcr_mma(cC, a_kt1, b0);
rcr_mma(cD, a_kt1, b1);
```

After: reorder so `a_kt1` issues LAST and split the wait into two stages
so the cA/cB mfmas overlap with the in-flight a_kt1 vmem drain:

```
load_a_kt(a,     0);   // 8 b128
load_b_kt(b0,    0);   // 4 b128
load_b_kt(b1,    1);   // 4 b128
load_a_kt(a_kt1, 1);   // 8 b128 (LAST — drains last)
asm volatile("s_waitcnt vmcnt(8)");   // wait for first 16 (a + b0 + b1)
rcr_mma(cA, a,     b0);    // ~32 cyc — overlaps with a_kt1 drain
rcr_mma(cB, a,     b1);    // ~32 cyc — overlaps with a_kt1 drain
asm volatile("s_waitcnt vmcnt(0)");   // a_kt1 likely already drained
rcr_mma(cC, a_kt1, b0);
rcr_mma(cD, a_kt1, b1);
```

vmcnt is in-issue-order retirement on AMDGCN (same semantics relied on
by the main loop's `RCR_STEADY_VMCNT=8` mid-iter wait at line 2199).
After issuing 24 ops, `vmcnt(8)` returns when 16 have retired — those
are the FIRST 16 issued (a, b0, b1), which is exactly what cA/cB need.
The remaining 8 (a_kt1) are still in flight; they drain during the
2 mfmas before the second `vmcnt(0)`.

## Results

GPU 3 (pinned, idle, 0 KFD VRAM). Triple-run verify:

```
metric runs         : 824 / 828 / 821     (mean ≈ 824, stdev ≈ 3)
entry baseline      : 811 / 816 / 813     (mean ≈ 813, stdev ≈ 3)
delta               : +11..+13            (well outside ±10 noise band)
```

Per-shape grpFP8 ratio improvement (round-12 entry → post-edit):

| shape                          | before | after | Δ     |
|--------------------------------|-------:|------:|------:|
| gpt_oss-GateUP-B4-M2048        | 0.924  | 0.962 | +3.8  |
| gpt_oss-Down-B4-M2048          | 0.972  | 1.009 | +3.7  |
| gpt_oss-GateUP-B4-M4096        | 0.935  | 0.965 | +3.0  |
| gpt_oss-Down-B4-M4096          | 0.936  | 0.964 | +2.8  |
| gpt_oss-GateUP-B32-M2048       | 0.981  | 0.991 | +1.0  |
| gpt_oss-Down-B32-M2048         | 0.985  | 0.994 | +0.9  |
| gpt_oss-GateUP-B32-M4096       | 0.967  | 0.980 | +1.3  |
| gpt_oss-Down-B32-M4096         | 0.974  | 0.994 | +2.0  |
| DSV3-* (8 shapes)              |    n/a |   n/a | within noise |

DSV3 K=2048 has K_REM=0 → does not trigger `FUSED_KTAIL=true`, so
unchanged (as expected). gpt_oss K=2880 has K_REM=64 → all 8 shapes
hit the optimization. Correctness: 0/16 FAIL across all 3 runs (fwd +
dA + dB SNR > 25 dB on every shape).

## Resource usage

`grouped_rcr_kernel<0, *, true>` (FUSED_KTAIL=true) build remarks
identical to baseline:

```
VGPRs: 256   AGPRs: 0   Occupancy: 2 waves/SIMD   LDS: 139796 B
```

Pure instruction reorder; no register-pressure delta.

## Why this works

The 4 K-tail mfmas (~32 cyc each = 128 cyc) and the 24 buffer_loads
(~40-100 cyc latency) were strictly serialized in the original code
via the single `vmcnt(0)` barrier. With ki_dyn=22 main-loop iters per
output tile, the K-tail epilog fraction is ~5 % of total wall on
K=2880 shapes. The split lets us hide ~32-64 cyc of the buffer_load
drain behind 1-2 mfma latencies.

For B=4 grids (~5760 output tiles per shape, ~40 tiles per CU
persistent), this saves ~1280-2560 cyc per CU × 144 CUs / GPU clock
= ~7-14 µs per kernel launch; on a ~750 µs gpt_oss-Down-B4-M2048
baseline, that's ~1-2 % wall. Matches the measured +1-3 % per shape.

The B=32 cases see smaller absolute improvement (+0.9-2.0 pp) because
their per-CU tile count is higher (~80-160 tiles/CU), so the K-tail
epilog's fixed-cost saving is amortized over more main-loop work.

## Why correctness holds

vmcnt(N) on AMDGCN GFX950 retires VMEM ops in issue order: when the
counter reaches N, the **first (24-N) ops issued** are guaranteed
retired. This is the same property the main loop's
`RCR_STEADY_VMCNT=8` wait at line 2199 depends on (where multiple
prefetches from prior iters are outstanding and the wait must drain
the older ones). The K-tail uses the same property, just with a
different threshold.

Re-issued in dependency-aware order:
- cA = a · b0 needs a (1st issue) + b0 (2nd issue) — both retire at vmcnt ≤ 16.
- cB = a · b1 needs a (1st) + b1 (3rd) — both retire at vmcnt ≤ 12.
- cC = a_kt1 · b0 — a_kt1 is 4th (last), retires at vmcnt = 0.
- cD = a_kt1 · b1 — same.

vmcnt(8) safely covers cA/cB; vmcnt(0) covers cC/cD.

Empirical correctness validation: SNR ≥ 25 dB on all 16 grouped FP8
shapes × {fwd, dA, dB} = 48 SNR checks all PASS, across 3 metric runs.

## Why no host-side change

`primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`
is INVARIANT (RED LINE — single-launch persistent architecture).
`GroupedGEMMFP8HipKittenBackend.execute` still issues exactly one
kernel launch per dispatch call.
`primus_turbo/pytorch/kernels/hipkitten/config.py` per-shape rules
unchanged.

## Falsification check

The 4 worst-shape per-shape `(gm, xcd)` rules audited in round-10-dm
were verified at saturation plateau against the round-19 BUFFER-store
kernel. This change does NOT alter the kernel's per-tile completion
latency in a way that would re-open the config space (the K-tail epilog
runs once per output tile, after the main-loop scheduling is complete).
The (gm=14, xcd=4) rule for gpt_oss-GateUP-B4-M4096 documented in
round-11-dm remains correct.

## Falsification banks updated

This round REOPENS one round-25 inventory entry — the round-25
"K-tail epilog single-load merge" plan (multi-round structural project)
was previously unimplemented. Round-12 implements a smaller-scope
variant (split-vmcnt instead of full load hoist into Epilog 2).
Remaining structural projects (AGPR migration, MFMA cell-shape
16x16x128→32x32x64) still on the table for future rounds.

## Commits

- HipKittens: `perf(grouped-fp8): split K-tail vmcnt to overlap M-slab-1
  drain with cA/cB mfmas (+13 score)`
- Primus-Turbo: this round-12 doc.

## Next-round suggestion

The DSV3 cluster (4 shapes at ratio 0.95-0.98) is now the main score
gap. Per task body's first task, these are aligned shapes (no K-tail,
no N-tail) so the 2-3 % gap is pure main-loop throughput. The
split-vmcnt optimization here doesn't apply to them (no FUSED_KTAIL).

Round-13 candidates:

1. **Same split-vmcnt pattern in main-loop** — could the per-K-tile
   `RCR_STEADY_VMCNT=8` wait similarly be split to overlap one mfma
   with the next-tile B prefetch drain? Round-25 docs say
   `RCR_STEADY_VMCNT` is saturated; needs re-verification post-this-change.

2. **DSV3-Down rocprof PMC** — task body's "First task" was to localize
   the missing TFLOPS on DSV3-Down via rocprof. Now that the gpt_oss
   gap is closed, DSV3 (ratio 0.95-0.98 on aligned shapes) is the
   bottleneck. Worth doing the diagnostic.

3. **MFMA cell-shape `16x16x128` → `32x32x64`** (round-25 doc §2,
   2-3 rounds). Triton uses 32x32x64 on the same shapes. Larger cell
   reduces MFMA count by 2× and matches Triton's main-loop schedule.

Recommend (2) — diagnostic first, then optimization. The DSV3 cluster
is now the dominant score gap (gpt_oss is 0.96-1.01, DSV3-Down is
0.95-0.97); pulling DSV3 from 0.96 → 1.05 would push score past 850.
