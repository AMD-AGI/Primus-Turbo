# Round 14 — ROLLED BACK — 2026-11-10 03:18 — second LLVM option sweep on top of U + G

## Hypothesis
Round 13 got +1.4 % from two scheduler / allocator options without changing the spill count.
More options from the same layers (machine scheduler, greedy allocator, AMDGPU scheduler
strategies) on top of U + G may move code further in the same direction.

## Single change (config sweep = one round)
`ops_bwd._dkdv_kwargs` llvm_opts = (U, G, X) for one extra option X per config, via the
round-13 knob (no source change). Availability probe: artifacts/llvm_opt_probe2.py; missing in
FlyDSL's LLVM: consider-local-interval-cost, amdgpu-premra-rematerialize,
amdgpu-disable-rematerialization-sched-stage, amdgpu-schedule-optimize-ilp. Defaults read back:
misched-regpressure True, amdgpu-mfma-vgpr-form True, amdgpu-spill-vgpr-to-agpr True,
misched-postra False, dag-maps-huge-region 500, misched-limit 256.

## Screening (ISA, artifacts/isa_r14.txt; artifacts/isa_r14_hashes.txt = md5 of the final ISA
without comments / directives)
Byte-identical to U + G (no timing needed): amdgpu-mfma-vgpr-form=False,
amdgpu-spill-vgpr-to-agpr=False, misched-cyclicpath=False, misched-postra=True, misched-limit=1024,
dag-maps-huge-region=4000, regalloc-eviction-max-interference-cutoff=100,
split-threshold-for-reg-with-hint=0.

| + option on U + G | spill | scratch B | masked loop | hot loop | ISA |
|---|---|---|---|---|---|
| none (r13) | 98 | 396 | 17 / 2 | 3 / 2 | ref |
| misched-cluster=False | 98 | 396 | 17 / 2 | 3 / 2 | differs |
| misched-fusion=False | 98 | 396 | 17 / 2 | 3 / 2 | differs |
| enable-misched=False | 100 | 404 | 17 / 2 | 4 / 3 | differs |
| amdgpu-enable-rewrite-partial-reg-uses=False | 98 | 396 | 17 / 2 | 3 / 2 | differs |
| misched-regpressure=False | 131 | 524 | 38 / 20 | 17 / 12 | not timed |
| amdgpu-sched-strategy iterative-ilp | 138 | 520 | 25 / 17 | 46 / 24 | not timed |
| amdgpu-sched-strategy iterative-minreg | 162 | 596 | 53 / 18 | 39 / 21 | not timed |
| amdgpu-sched-strategy iterative-maxocc | 219 | 732 | 71 / 37 | 51 / 23 | not timed |
| amdgpu-igrouplp-exact-solver=True | compile did not finish in 2 min of CPU; killed | | | | |

## Correctness
SNR gate passed for all timed configs (dQ 47.00-47.01 / dK 49.74 / dV 51.28 dB, unchanged).

## Results (dev/sweep_proc.py, one process per config, 3 rotated rounds; artifacts/sweep_r14t.raw)

| config | B2 S4K ms | d% | B1 S8K ms | d% | B1 S16K ms | d% | gm d% |
|---|---|---|---|---|---|---|---|
| r13 (U + G) | 5.083 | +0.00 | 8.870 | +0.00 | 32.668 | +0.00 | +0.00 |
| + misched-cluster=False | 5.143 | -1.16 | 8.964 | -1.06 | 33.010 | -1.04 | -1.08 |
| + misched-fusion=False | 5.085 | -0.05 | 8.867 | +0.04 | 32.648 | +0.06 | +0.02 |
| + enable-misched=False | 5.150 | -1.31 | 8.874 | -0.05 | 32.543 | +0.38 | -0.33 |
| + rewrite-partial-reg-uses=False | 5.094 | -0.22 | 8.845 | +0.28 | 32.639 | +0.09 | +0.05 |

Full set not run: no candidate above noise.

## Decision
ROLLED BACK (no source change; r13 defaults kept). Rollback count: 1 (round 13 accepted).

## Finding
Turning the pre-RA machine scheduler off entirely costs only 0.33 % (spills 98 -> 100), so the
hot-loop order is essentially the order FlyDSL emits; compiler options have little room left.
amdgpu-mfma-vgpr-form=False and amdgpu-spill-vgpr-to-agpr=False change nothing in this kernel.

## Real-training transfer check
- Bucket: K1 (codegen: compiler options)
- Cache key (if any): FlyDSL compile cache key includes the compile hints
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: +0.05 % (best variant, sweep, 3 shapes; noise)
- Estimated real-training gain: 0 % (rolled back)
- Decision: ROLLBACK

## Next direction
Kernel-side scheduling: the re-profile's largest per-tile item is the MFMA stall in back-to-back
runs (+739 cycles vs aiter), i.e. VALU / LDS work that is not overlapped with MFMA.
