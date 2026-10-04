# gfx1250_mla_fwd -- provenance

The five kernel files are aiter's gfx1250 FlyDSL forward prefill (`aiter/ops/flydsl/kernels/`,
aiter commit 6963ae9d, 2026-09-17):

| file here | aiter source |
|---|---|
| `fmha_fwd_prefill_a16w16_m32x8.py` | `fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py` |
| `fmha_b16_buffer_managers.py` | `fmha_gfx1250/fmha_b16_buffer_managers.py` |
| `kernels_common.py` | `kernels_common.py` (unmodified) |
| `tensor_shim.py` | `tensor_shim.py` |
| `buffer_ops.py` | `buffer_ops.py` (unmodified) |

Import edits only: the files were flattened into one package, so `..x` / `aiter.ops.flydsl.kernels`
imports became `.x` (11 sites, two of them function-local in `tensor_shim.py`), and
`get_lds_capacity_bytes` (aiter `jit/utils/chip_info.py`) was inlined into the kernel file, so
the package depends only on flydsl, torch, numpy and the stdlib.

Kernel changes over aiter, in order (each measured on gfx1250; 4 and 5 at DeepSeek-V3 shapes,
b2 s4096 h128, bf16, causal):

1. Longest-first (LPT) remap of the workgroup id: under causal masking the work per q tile
   grows with its index, so the heaviest tiles are dispatched first (`_lpt_block_id`).
2. Packed exp2 argument: one `v_pk_fma` produces two softmax exponent arguments.
3. Speculative / stale-max softmax disabled (it recomputes many tiles on real data).
4. Softmax scale in fp32: aiter multiplied Q by `softmax_scale` rounded to bf16 (0.1352338 ->
   0.1347656 at the DeepSeek-V3 scale; o 41.5 dB, lse 57 dB vs an fp32 reference). Q now stays
   unscaled, `c = softmax_scale * log2(e)` in fp32 is the multiplier of the exp2 FMA
   `p = exp2(fma(s, c, -m))`, the running max is carried in log2 units, and
   `lse = (m + log2(l)) * ln(2)`. o 54.5 dB, lse 145 dB, the same as aiter's ASM kernel.
5. `RESCALE_THRESHOLD = 1.0` (aiter: 8.0): the running max may stay stale by at most one
   natural-log unit. With 8 the dominant P weight is no longer exactly 1.0 when rounded to bf16
   for the PV GEMM (o 53.2 dB at b2 s4096 h128); 1.0 keeps o at 54.2 dB and is 1.4% faster than
   an exact max (0.0) on random inputs, 3.3% on DeepSeek-V3 training activations.
6. Only the `m32x8` kernel is kept (the `m32x2` small-grid and `m16x8` variants carried the same
   bf16-scale issue and are never selected at head dim 192).
7. `MODULE_KNOBS`: FlyDSL keys its compile cache by function sources and closure scalars, not by
   module globals, so builds differing only in a module-level knob (such as item 5) would share a
   cached binary. Both kernels reference a closure tuple of every module-level scalar knob, which
   puts their values into the key. Codegen is unchanged (same ISA hash as without it).
8. Split per-tile barrier (`SPLIT_TILE_BARRIER = True`, `HI_SKEW_SLEEP = 0`, `BARRIER_FENCE =
   True`): the tile-top `tensor_wait(0)` + full workgroup barrier is split into a raw
   `s_barrier_signal` / `s_barrier_wait` pair (id -1, workgroup release / acquire fences around
   them, each site wrapped in `sched_barrier(0)`). The prologue signals after the Q reads and
   tile 0's TDM slice retire; the tile top only waits; mid tile, after softmax and before
   rescale / PV, each wave drains its K/V reads of slot t (`s_wait_dscnt(0)`) and its TDM(t+1)
   slice (`tensor_wait(0)`) and signals; one wait follows the loop (#signal == #wait == n + 1
   for every wave). A wave may run up to one PV (register-only) ahead of its slowest peer, so
   the SIMD partners (wave i / i + 4) stop running QK and softmax in lockstep. Same LDS slots,
   no new addresses; outputs bitwise equal to item 7. 2.8% faster at b2 s4096 h128.
9. Per-wave skip of fully masked diagonal tiles (`SKIP_MASKED_TILES = True`, only when
   `mask_right`): each wave splits its right-boundary sub-loop `[clean_hi, n_tiles)` at the
   wave-uniform `wave_end = clamp(ceildiv(max(wave_q_max, -1) + 1, n_block), clean_hi, n_tiles)`,
   `wave_q_max = min(last packed row of the wave // gqa, q_len - 1) + causal_off +
   window_right`, into full tiles `[clean_hi, wave_end)` and sync-only tiles
   `[wave_end, n_tiles)` that keep only the wave's share of the item-8 protocol (wait, its
   TDM(t+1) slice, drained signal, pointer swap) and bypass m / d / O. Every kv of a skipped
   tile lies past the band edge of every valid row of the wave, so the skipped body was an
   exact no-op: o / lse bitwise equal to item 8. With the knob off the kernel compiles to item
   8's binary. 2.2% faster at b2 s4096 h128 (4.4% of the wave-tiles are skipped there).
