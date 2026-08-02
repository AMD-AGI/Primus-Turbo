# MI355X / mega MoE fp8 / flydsl — reusable tips

## Kernel

- **Issue whole-cache coherence ops from one lane, not one per wave.** `l2_invalidate()`
  (`buffer_inv sc1`) invalidates the issuing CU's vector L1 and its XCD's L2. A workgroup lives on
  one CU inside one XCD, so every wave issuing it is redundant — waves 2..N find an
  already-invalidated cache and only serialize on the L2 port. Guard with
  `if thread_index == 0: l2_invalidate(); s_waitcnt(0)` then `barrier()` to release the rest.
  Worth -6.5% on the L1 GEMM stage, -3.8% end-to-end. The cost is *issue serialization*, not lost
  cache state — do not attack it by making the acquire rarer (see anti-patterns).

- **Producer and consumer in the same workgroup should never round-trip through global scratch.**
  The SwiGLU kernel wrote `act_bf16` to global and re-read it a few instructions later in the
  quant loop, 44% of its traffic. Remapping threads so one thread owns a complete 1x32 MXFP8 block
  lets the activation stay in registers all the way into the quantizer. -12.6% on the kernel.

- **`fx.Array[fx.Int8, ...]` is not Storable in flydsl 0.2.4.** Use `fx.Int32` slots and pack the
  byte (`value & 0xFF`). Only concrete types like `Int32`/`BFloat16`/`Float8` work in LDS structs.

- **`@fx.struct` breaks under `from __future__ import annotations`.** It needs live type objects to
  compute field layout; stringized annotations leave shape variables unresolved. Do not add that
  import to a module containing flydsl structs.

- **`gemm_mxfp8_nt_tile` hardcodes `full_barrier=False`**, a partial `wave_m == 1` barrier that is
  only valid when the workgroup enters with fresh LDS. Any persistent / multi-tile-per-workgroup
  scheme must switch it to a full barrier — and even then expect further ordering hazards.

- **The compiler will not hoist a load across a `buffer_store` it cannot prove is non-aliasing.**
  A producer loop that writes LDS and output buffers between passes gets *zero* cross-pass
  prefetch, however obvious it looks: the tell in ATT is one `s_waitcnt vmcnt(N)` per pass, each at
  ~100% stall rate, together dominating the profile. Issuing pass p+1's loads explicitly before
  consuming pass p's halved VMEM-wait (27% -> 14% of stall) for ~12 VGPRs.

- **`fx.arith.divf(a, b, fastmath="afn,arcp")`** replaces the ~10-VALU IEEE divide expansion
  (`v_div_scale` x2, `v_rcp`, 3 `v_fma`, `v_div_fmas`, `v_div_fixup`) with `v_rcp_f32`. `arcp`
  alone is a **no-op** — the backend keeps the IEEE expansion unless `afn` is set too. Do not use
  `fast`: it also implies nnan/ninf, which lets the compiler delete an `ACTIVATION_CLAMP` min/max
  for no extra speed. Worth -17% on the ALU-bound forward SwiGLU but only -1.6% on the
  memory-bound backward one, where `v_exp_f32` still dominates the line.

- **Workgroup-uniform metadata (indexed off `block_idx`) should be `buffer_load`ed by every lane**,
  not staged into LDS by lane 0 behind a barrier. The LDS route costs a barrier plus the
  `v_readfirstlane_b32` the read-back lowers to — 8% of stall. `pack_kern` and
  `_compile_colwise_quant_grouped` already read directly; `_compile_rowcol_dual_grouped` does not,
  so anything derived from it inherits the slow path.

- **Zeroing via multiply by a 0/1 mask produces `-0.0` for negative inputs**, which encodes as fp8
  `0x80` instead of `0x00`. Any output whose pad rows are written unconditionally will mismatch
  while `amax`-derived scales still agree — a confusing signature. Use `arith.select`.

- **LDS *per thread* is what caps occupancy, so shrinking `BT` does not raise it.** A 32-row MX
  block staged as packed bf16 pairs is 128 B/thread, which pins the fused swiglu-bwd dual-quant at
  4 of 8 waves/SIMD on the 160 KB LDS regardless of block size. Halving it needs either re-reading
  `l1`/`dact` (+33% traffic: an F-tiled split makes the gate half read gate+up+d and the up half
  gate+d) or register residency, which spills. Optimize for *tolerating* 4 waves/SIMD.

## Measurement

- **`import pkg.sub.mod as m` can hand back a function, not the module.** When the package
  `__init__` does `from pkg.sub.mod import mod_name` it rebinds that name, shadowing the submodule;
  `import ... as m` then resolves via `getattr` and returns the re-export. Monkeypatching `m.attr`
  silently does nothing and A/B legs come out identical (the tell: baseline == optimized to 3
  decimals). Take the module from `sys.modules["full.dotted.name"]` and assert the patch took.

- **The fused fp8 MoE forward is run-to-run nondeterministic** — repeating an identical config
  changes ~3-5% of output elements with max|d| ~0.4 (cross-rank combine accumulates in arrival
  order). A single SNR reading therefore cannot compare two kernel variants; a 1 dB gap between
  two configs is well inside the self-noise. Compare *distributions* of repeated SNR readings, and
  do bit-exactness checks at a stage output (e.g. L1 / SwiGLU result) rather than the final output.

- **A/B both legs in one process, interleaved.** Cross-session numbers on this node drift by more
  than the effects being measured (~0.05 ms). `PT_MXFP8_GEMM_INV` is part of the compile cache key,
  so it can be flipped in-process; `PT_COMBINE_GEMM_ONLY` is *not*, and flipping it in-process
  silently reuses the stale compiled kernel — that one needs a fresh process.

- **Skewed routing changes absolute latency far more than it changes speedups.** A 32x expert
  imbalance more than doubles FULL forward time at T=8192 (4.66 -> 10.16 ms) while kernel-internal
  gains hold at ~1.03x. Always include a skewed leg, but read relative gains, not absolute ms.

- **The first run of an isolated stage probe reads several percent slow** (cold clocks). A 4x
  repeat gave 0.609/0.609/0.610/0.610 where the first single shot read 0.630 — enough to invert an
  accept/rollback decision. Never judge a round on one probe sample; this compounds with the
  cross-session node drift noted above.

- **`bench_mega_moe_bwd_only.py` has a ±0.03-0.06 ms (0.4-0.7%) band** at EP8/T=8192. A stage that
  is ~7% of backward cannot produce a resolvable e2e signal from a <5% stage gain. Use the isolated
  probe plus an ATT stall delta for attribution and read bwd-only as a no-regression check.

- **A stall class's share overstates what removing it buys.** Deleting a barrier worth 6.6% of
  stall bought 0.3% of wall clock: barrier stall is partly covered by the other workgroups resident
  on the CU, and removing already-hidden stall pays nothing. Prefer hotspots whose stall rate is
  ~100% (nothing is covering them) over ones with a big share but partial coverage.

## Profiling (rocprofv3 / ATT)

- **Every FlyDSL kernel lowers to its Python function name**, so a kernel called `kern` appears as
  `kern_0` next to a dozen unrelated ones and cannot be selected by an ATT `kernel_include_regex`.
  Name kernels uniquely up front. When reading `kernel_trace.csv`, group by `Kernel_Id`, never by
  `Kernel_Name`.

- **`rocprofv3 --stats` merges output across ranks** under `torch.multiprocessing.spawn`, making
  per-kernel aggregates meaningless (and mixing warmup with steady state). Use `--kernel-trace` and
  group per dispatch.

- **The `rocprof-trace-decoder` .so is not in the ROCm 7.2.1 image.** Fetch the release installer
  and copy it into `/opt/rocm/lib` before any ATT run; the installer fails unless its `--prefix`
  directory already exists.

- **Classifying ATT stalls needs `s_waitcnt`'s operands**, not just the opcode, to separate `vmcnt`
  from `lgkmcnt`. Matching on the opcode alone silently buries every memory wait in a "SALU" bucket
  and hides the real bottleneck.

## Anti-patterns (do not retry)

- **Making the L1 acquire rarer via fewer/fatter workgroups (NB>1).** Serializes the software
  pipeline — NB workgroups block on one pool block and inter-tile overlap is lost. 0.85-0.91x.
  The acquire's cost is per-issue latency, so reduce *issues per workgroup*, not workgroups.

- **Quad-butterfly amax to coalesce the SwiGLU loads.** The redundant per-lane scale computation
  costs more than the coalescing saves: 4.5% slower.

- **Tile-locality XCD remapping** on this fused kernel: up to +67% regression. Block ordering is
  dependency-driven, so a locality-driven remap fights the scoreboard.

- **SwiGLU launch-geometry tuning.** 23 `(BT, grid_x)` configs, all flat within 3% noise — the
  kernel is bandwidth-bound. Not worth a round.
