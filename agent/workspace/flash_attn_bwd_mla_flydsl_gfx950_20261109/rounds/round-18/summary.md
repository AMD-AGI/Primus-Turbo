# Round 18 — ACCEPTED (correctness fix, user-authorized cost) — 2026-11-10 15:18 — dS slots alternate per trip on the GQA = 1 path (DS_PAR)

## Machine change
n02-29's allocation expired; this round and the following ones run on smci355-ccs-aus-n01-33,
container xiaoming-dev, GPU 0 (torch 2.14.0+rocm10.1.0, primus_turbo 0.6.1.dev4 from
/opt/venv site-packages, flydsl 0.3.4.1). Toolchain parity: the r16 package and the r18 working
tree compile on n01-33 to the same dkdv ISA hashes as on n02-29 (c587bf720b, ae8b324271;
artifacts/isa_r18.txt). Absolute numbers move a little between nodes (r16 step 929.77 here vs
927.33 on n02-29), so this round compares against r16 re-measured in the same session.

## Hypothesis
Round 17 found a formal WAR race on the MLA path: with one q head per kv head,
`head_local % G3S_SLOTS` is always 0, so the deferred GEMM3 of trip t-1 (top of trip t) reads dS
slot 0 and trip t writes its own dS into slot 0 later in the same trip, with no barrier between
the two (r15 ATT margin about 2200 cycles against about 100 cycles of barrier skew; never
observed). Alternating the slot by trip parity puts every rewrite of a slot behind the next
trip's leading barrier. Expected cost: small; two loop-carried VGPR bases take the place of the
two pinned ones, plus one SGPR step and three adds per trip.

User decision (fix_now): fix it in this round and accept it even at some performance cost,
reporting the cost.

## Single change
`flash_attn_bwd_mla.py` (artifacts/ds_par.diff; ops_bwd.py and __init__.py unchanged from r16):
`DS_PAR = G3_QCARRY and GQA_GROUP_SIZE == 1`. The q loop carries (dS write base, deferred GEMM3
read base, step d): trip 0 writes slot 0 while its priming GEMM3 (all stores clip) reads slot 1;
each trip steps the write base by +d and the read base by -d, then flips d. The dS writes and
the dS zero-fill use the carried write base with slot offset 0, the in-loop deferred GEMM3 and
the post-loop tail GEMM3 use the carried read base (`_gemm3(..., sbase=)`). Asserts: no G3_SPLIT,
odd group sizes above 1 rejected, and the slot stride's lowest set bit lies above every XOR mask
applied to the bases afterwards.

## Correctness
- dev/bitwise_r18.py (r16 package vs r18, 3 repeats each; artifacts/bitwise_r18_n0133.log),
  B1 S1024 / 2112 / 4096 / 4224, B2 S4096, B1 S8192: dK and dV bitwise equal to r16 on every
  shape and repeat; dQ SNR of r18 vs r16 60.9-61.4 / 55.3-55.6 / 52.6-52.9 / 52.3-52.5 /
  52.8-53.0 / 49.5-49.6 dB against r16's own repeat-to-repeat 60.9-61.4 / 55.4-55.5 / 53.1-53.3 /
  52.7-53.1 / 53.3 / 49.7 dB (bf16 atomic order).
- dev/d128_regress.py: REGRESS PASS (ws_dk / ws_dv bitwise for det True / False and the
  MLA-shaped D128 case; the img line is non-bitwise by design since round 2, excluded from the
  gate). artifacts/d128_regress_r18.log.
- Full set 7 / 7 PASS in all three r18 runs; dQ / dK / dV SNR identical to r16 to 0.01 dB.

## ISA (artifacts/isa_r18.txt, hot_reload_window.txt; dev/isa_reload_cover.py)
| | md5 | spills | scratch | masked loop scratch_load / vmcnt(0) | hot loop scratch_load / vmcnt(0) | hot loop instrs | hot reload MFMA cover (offset:cover) |
|---|---|---|---|---|---|---|---|
| r16 | c587bf720b | 94 | 380 B | 16 / 1 | 3 / 2 | 604 | 0:36 16:38 240:5 |
| r18 | ae8b324271 | 98 | 396 B | 17 / 2 | 3 / 1 | 611 | 0:35 16:18 208:0 |

The hot loop gains 7 instructions (the step adds, one s_sub, a few s_nop / v_mul_f32 shuffles);
its 3 scratch reloads stay 3. "Cover" counts the MFMAs between a reload and the first
`s_waitcnt vmcnt(k)` that guarantees it (gfx950's vmcnt completes in issue order).

## Results
Per-process A/B (dev/sweep_proc.py, now with a per-config `__pkg__` package root; 6 rotated
rounds, one process per config; artifacts/sweep_r18ab.raw):

| config | B2 S4K ms | B1 S8K ms | B1 S16K ms | gm |
|---|---|---|---|---|
| r16 | 4.927 | 8.392 | 31.044 | 0 |
| r18 | 5.000 (-1.46 %) | 8.551 (-1.86 %) | 31.621 (-1.82 %) | -1.71 % |

Every one of the 6 rounds orders the two the same way on every shape.

Full set, rotated r18 / r16 / r18 / r16 / r18 (artifacts/full_r18_run{1,2,3}, full_r16_run{1,2};
artifacts/run_full_ab.sh, per_shape_ab.txt):

| shape | r18 bwd ms run1/2/3 | r16 bwd ms run1/2 | bwd d | r18 step (run2) | r16 step (mean) | step d | step ratio vs aiter (r18) |
|---|---|---|---|---|---|---|---|
| b1_s4k | 2.495/2.501/2.504 | 2.440/2.445 | -2.33 % | 786.11 | 800.35 | -1.78 % | 0.904 |
| b1_s8k | 8.586/8.573/8.592 | 8.432/8.423 | -1.84 % | 908.33 | 920.67 | -1.34 % | 0.915 |
| b1_s16k | 31.708/31.738/31.768 | 31.208/31.403 | -1.36 % | 979.68 | 989.72 | -1.01 % | 0.941 |
| b1_s32k | 123.023/123.384/123.004 | 120.623/120.695 | -1.92 % | 1007.60 | 1025.34 | -1.73 % | 0.928 |
| b1_s64k | 492.232/492.577/493.016 | 483.949/483.830 | -1.76 % | 1014.53 | 1028.76 | -1.38 % | 0.919 |
| b2_s4k | 5.017/5.003/5.009 | 4.933/4.935 | -1.50 % | 783.85 | 792.65 | -1.11 % | 0.874 |
| b2_s16k | 63.389/63.676/63.580 | 62.850/62.883 | -1.12 % | 975.76 | 985.07 | -0.94 % | 0.923 |

Step geomean: r18 918.84 / 917.41 / 916.89 (median 917.409, ratio vs aiter 0.9145); r16 930.33 /
929.21 (mean 929.769, ratio 0.927). Bwd geomean: r18 840.501 (median run, ratio 0.8989) vs r16
855.185. Cost of the fix: step -1.329 %, bwd -1.717 %; fwd unchanged (+0.114 %, noise).

## Attribution
The fix itself adds almost nothing to the hot loop; the cost is the register allocator's
response to it. Spills go 94 -> 98, and GEMM1a's third K-pack scratch reload of the trip moves
into a register that frees up only 2 instructions before the MFMA that reads it: in r16 the
reload (`v[36:39]`, offset 240) is issued 5 MFMAs ahead of its `vmcnt(0)`; in r18 the reload
(`v[28:31]`, offset 208) is followed directly by `vmcnt(0)` and the MFMA, so its whole scratch
latency is exposed once per trip, and the second reload's cover halves (38 -> 18 MFMAs). This is
consistent with a near-uniform 1.1-2.3 % bwd loss across shapes (the hot loop dominates every
shape). Hypothesis level for the exact cycle cost: not ATT-measured.

## Decision
ACCEPTED as a correctness fix under the user's fix_now decision. Mechanically this round would
roll back (step -1.33 %, no core shape below -5 %); it is the first accepted round below its
predecessor, which Rule 9's monotonic check must read as this documented exception. r18 is the
new baseline: later rounds compare against r18 on n01-33. Rollback counter reset to 0.

## Real-training transfer check
- Bucket: K1 (kernel-internal LDS slot addressing)
- Cache key (if any): no cache
- id(...) audit: n/a
- 4-step pen-and-paper trace hit rate: n/a
- Benchmark gain on this round: -1.33 % combined-step (cost of the fix)
- Estimated real-training gain: -1.33 % combined-step (the kernel and its hot loop are the same
  in training; the race it closes exists in training too)
- Decision: ACCEPT-as-real (correctness fix with a measured cost)

## Next direction
Win the cost back on the r18 base without reopening the race. Round 19: knob re-sweep on r18
(g3d 2..6, g3_st_n, g3_st_at, g3_dbat, mfma_tie, g2d, g1_ks_outer, the two LLVM options), screened
first by ISA with dev/isa_reload_cover.py (target: no hot-loop reload with cover 0, spills <= 94)
and then timed per process. If no knob restores the cover, look at freeing the 4 extra spilled
VGPRs (for example deriving the read base from the write base instead of carrying both).
