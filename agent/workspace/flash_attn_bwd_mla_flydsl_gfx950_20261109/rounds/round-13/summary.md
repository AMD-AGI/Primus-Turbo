# Round 13 — ACCEPTED — 2026-11-10 03:05 — LLVM backend option sweep for the dkdv compile

## Hypothesis
The re-profile puts the steady-state gap in stall, and two of its largest items are placement
decisions of the register allocator / scheduler (spill reloads right before use in GEMM1a; long
MFMA runs). The builder already passes two cl::opts (enable-post-misched, lsr-drop-solution);
other allocator and GCN-scheduler options may place spills and reloads better.

## Single change (config sweep = one round)
`flash_attn_bwd_mla.py`: builder knob `llvm_opts` (tuple of (name, value), merged into the
dkdv compile's llvm_options; default () = unchanged). `ops_bwd._dkdv_kwargs` (MLA only) now
passes `amdgpu-disable-unclustered-high-rp-reschedule=True` and
`greedy-regclass-priority-trumps-globalness=True`. Snapshot: kernel_snapshot/.

## Screening (ISA, artifacts/isa_r13.txt; option availability: artifacts/llvm_opt_probe.py)
Thirteen options, one compile each (dev/isa_sweep.py). Identical ISA counts to r9 (98 spilled
VGPRs, 396 B, hot loop 3 / 2 reloads / vmcnt(0)): exhaustive-register-search, clustered
low-occupancy reschedule off, relaxed occupancy, enable-local-reassign, rewrite-mfma-form stage
on, schedule-metric-bias 0 / 100. Different:

| option | spill | scratch B | masked loop | hot loop |
|---|---|---|---|---|
| amdgpu-use-amdgpu-trackers | 66 | 268 | 9 / 1 | 8 / 1 |
| disable-unclustered-high-rp-reschedule (U) | 98 | 396 | 17 / 4 | 4 / 1 |
| greedy-regclass-priority-trumps-globalness (G) | 98 | 396 | 17 / 2 | 3 / 2 |
| U + G (artifacts/isa_r13u.txt) | 98 | 396 | 17 / 2 | 3 / 2 |
| sched-strategy max-memory-clause | 104 | 420 | 18 / 5 | 5 / 4 |
| sched-strategy max-ilp | 140 | 536 | 34 / 13 | 20 / 18 |
| greedy-reverse-local-assignment | 137 | 540 | 41 / 22 | 20 / 16 |

## Timing sweeps (dev/sweep_proc.py, one process per config, 3 rotated rounds, B2 S4K / B1 S8K /
B1 S16K)
artifacts/sweep_r13t.raw: trackers -21.76 % gm (despite 66 spills), trackers + U -20.87 %,
trackers + G -22.52 %, U +1.47 %, G +1.15 %.
artifacts/sweep_r13u.raw:

| config | B2 S4K d% | B1 S8K d% | B1 S16K d% | gm d% |
|---|---|---|---|---|
| U | +2.20 | +1.16 | +0.51 | +1.29 |
| G | +1.43 | +0.80 | +1.12 | +1.12 |
| U + G | +2.11 | +1.96 | +1.99 | +2.02 |

## Correctness
Full set 7 / 7 PASS in all three candidate runs; SNR identical to the same-session r9 reference
to 0.01 dB (dQ 44.68-47.02, dK 49.54-49.66, dV 51.08-51.38 dB); dK/dV bitwise repeat True.
D128 is untouched (builder default `llvm_opts=()`; only the MLA `ops_bwd` passes options).

## Results (n02-29 GPU 0, full set, rotated vs aiter a16)
Order: candidate run1, r9 reference (`dev/bench_over.py '{"llvm_opts": []}'`), run2, run3.

| shape | bwd ms run1 | run2 | run3 | r9 ref (same session) | R9 run3 | Step run3 | d vs r9 ref | step ratio vs aiter |
|---|---|---|---|---|---|---|---|---|
| b1_s4k | 2.652 | 2.722 | 2.700 | 2.747 | 2.749 | 737.74 | +1.406 % | 0.854 |
| b1_s8k | 8.880 | 8.923 | 8.928 | 9.095 | 9.118 | 879.07 | +1.484 % | 0.902 |
| b1_s16k | 32.718 | 32.784 | 32.782 | 33.392 | 33.502 | 953.65 | +1.474 % | 0.922 |
| b1_s32k | 126.314 | 126.207 | 126.530 | 129.106 | 129.337 | 989.52 | +1.604 % | 0.914 |
| b1_s64k | 504.006 | 504.360 | 504.441 | 512.115 | 513.657 | 996.73 | +1.199 % | 0.902 |
| b2_s4k | 5.120 | 5.132 | 5.124 | 5.236 | 5.260 | 769.63 | +1.711 % | 0.868 |
| b2_s16k | 64.756 | 64.825 | 64.872 | 65.813 | 66.264 | 962.04 | +1.128 % | 0.915 |

Step geomean: run1 896.18 (ratio 0.8992), run2 892.38 (0.8964), run3 892.66 (0.8964); median
892.658 (+1.795 % vs R9 876.918; +1.429 % vs the same-session r9 reference 880.08, ratio 0.8840).
Trend row uses run3 (the median): Fwd 1203.816 (kernel unchanged), Bwd 812.035 (ratio 0.8722),
Step 892.658.

## Decision
ACCEPTED: +1.4 % combined-step against the same-session reference, every shape +1.1..+1.7 %.

## Real-training transfer check
- Bucket: K1 (codegen: compiler options baked into the dkdv binary)
- Cache key (if any): FlyDSL's compile cache key includes the compile hints (llvm_options too)
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +1.429 % combined-step vs same-session reference (median of 3)
- Estimated real-training gain: +1.1..+1.7 % on the MLA attention backward at these shapes; the
  options travel with the compiled kernel, and spill count / scratch size are unchanged from R9,
  so the R9 process-bimodality risk is unchanged (not seen in the 4 full-bench processes here).
- Decision: ACCEPT-as-real

## Next direction
U + G move schedule, not spills (ISA counts equal to r9 except the masked loop's vmcnt(0) 4 -> 2).
The GEMM1a K-pack reloads remain; next try to lower the GEMM1 register peak (all 12 Q A-fragments
are live together in GEMM1a).
