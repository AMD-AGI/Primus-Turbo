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
