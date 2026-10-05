# gfx1250_mla_fwd -- provenance

The five kernel files are aiter's gfx1250 FlyDSL forward prefill (`aiter/ops/flydsl/kernels/`,
aiter commit 6963ae9d, 2026-09-17), distributed there under the MIT license:

| file here | aiter source |
|---|---|
| `fmha_fwd_prefill_a16w16_m32x8.py` | `fmha_gfx1250/fmha_fwd_prefill_a16w16_m32x8.py` |
| `fmha_b16_buffer_managers.py` | `fmha_gfx1250/fmha_b16_buffer_managers.py` |
| `kernels_common.py` | `kernels_common.py` (unmodified) |
| `tensor_shim.py` | `tensor_shim.py` |
| `buffer_ops.py` | `buffer_ops.py` (unmodified; aiter's copy of FlyDSL's `kernels/common/buffer_ops.py`) |

Each file carries Primus-Turbo's FlyDSL (Apache-2.0) header and names its aiter / FlyDSL origin
below it. `interface.py` (host entry) and `__init__.py` were written for Primus-Turbo.

Import edits only: the files were flattened into one package, so `..x` / `aiter.ops.flydsl.kernels`
imports became `.x` (11 sites, two of them function-local in `tensor_shim.py`), and
`get_lds_capacity_bytes` (aiter `jit/utils/chip_info.py`) was inlined into the kernel file, so
the package depends only on flydsl, torch, numpy and the stdlib. The files are ruff-formatted;
formatting changed no statement and no instruction of the compiled kernel.

## Kernel changes over aiter

In order. Each was measured on gfx1250 (MI455X); the training shape is DeepSeek-V3's
b2 s4096 h128, qk 192 / v 128, bf16, causal. Code comments refer to the stages by the labels in
brackets: `r16` is the aiter kernel with items 1-3 (the comments `r6 g14` / `r13ns` belong to that
series), `fwd_c0` / `fwd_c0t9` are items 4 / 5, and `fwd_r1_a` ... `fwd_r5_a_g16` are items 8-11.
Every item after 4 leaves o / lse bitwise equal to the item before it unless it says otherwise.

1. [r16] Longest-first (LPT) remap of the workgroup id: under causal masking the work per q tile
   grows with its index, so the heaviest tiles are dispatched first (`_lpt_block_id`).
2. [r16] Packed exp2 argument: one `v_pk_fma` produces two softmax exponent arguments.
3. [r16] Speculative / stale-max softmax disabled (`SPEC_STALE_MAX = False`): on real activations
   its trigger fired on 13-25% of the tiles, each of which recomputes S.
4. [fwd_c0] Softmax scale in fp32: aiter multiplied Q by `softmax_scale` rounded to bf16
   (0.1352338 -> 0.1347656 at the DeepSeek-V3 scale; o 41.5 dB, lse 57 dB vs an fp32 reference).
   Q now stays unscaled, `c = softmax_scale * log2(e)` in fp32 is the multiplier of the exp2 FMA
   `p = exp2(fma(s, c, -m))`, the running max is carried in log2 units, and
   `lse = (m + log2(l)) * ln(2)`. o 54.5 dB, lse 145 dB, the same as aiter's gfx1250 ASM kernel.
5. [fwd_c0t9] `RESCALE_THRESHOLD = 1.0` (aiter: 8.0): the running max may stay stale by at most
   one natural-log unit. With 8 the dominant P weight is no longer exactly 1.0 when rounded to
   bf16 for the PV GEMM (o 53.2 dB at the training shape); 1.0 keeps o at 54.2 dB and is
   1.4% faster than an exact max (0.0) on random inputs, 3.3% faster on DeepSeek-V3 training
   activations.
6. Only the `m32x8` kernel is kept (the `m32x2` small-grid and `m16x8` variants carried the same
   bf16-scale issue and are never selected at head dim 192).
7. `MODULE_KNOBS`: FlyDSL keys its compile cache by function sources and closure scalars, not by
   module globals, so builds differing only in a module-level knob (such as item 5) would share a
   cached binary. Both kernels reference a closure tuple of every module-level scalar knob, which
   puts their values into the key. Codegen is unchanged (same ISA hash as without it).
8. [fwd_r1_a] Split per-tile barrier (`SPLIT_TILE_BARRIER = True`, `HI_SKEW_SLEEP = 0`,
   `BARRIER_FENCE = True`): the tile-top `tensor_wait(0)` + full workgroup barrier is split into
   a raw `s_barrier_signal` / `s_barrier_wait` pair (id -1, workgroup release / acquire fences
   around them, each site wrapped in `sched_barrier(0)`). The prologue signals after the Q reads
   and tile 0's TDM slice retire; the tile top only waits; mid tile, after softmax and before
   rescale / PV, each wave drains its K/V reads of slot t (`s_wait_dscnt(0)`) and its TDM(t+1)
   slice (`tensor_wait(0)`) and signals; one wait follows the loop (#signal == #wait == n + 1
   for every wave). A wave may run up to one PV (register-only) ahead of its slowest peer, so
   the SIMD partners (wave i / i + 4) stop running QK and softmax in lockstep. Same LDS slots,
   no new addresses; outputs bitwise equal to item 7. The protocol (RAW / WAR per slot, signal /
   wait counts) was checked by a CPU model. 2.8% faster at the training shape.
9. [fwd_r2_c] Per-wave skip of fully masked diagonal tiles (`SKIP_MASKED_TILES = True`, only when
   `mask_right`): each wave splits its right-boundary sub-loop `[clean_hi, n_tiles)` at the
   wave-uniform `wave_end = clamp(ceildiv(max(wave_q_max, -1) + 1, n_block), clean_hi, n_tiles)`,
   `wave_q_max = min(last packed row of the wave // gqa, q_len - 1) + causal_off +
   window_right`, into full tiles `[clean_hi, wave_end)` and sync-only tiles
   `[wave_end, n_tiles)` that keep only the wave's share of the item-8 protocol (wait, its
   TDM(t+1) slice, drained signal, pointer swap) and bypass m / d / O. Every kv of a skipped
   tile lies past the band edge of every valid row of the wave, so the skipped body was an
   exact no-op: o / lse bitwise equal to item 8. With the knob off the kernel compiles to item
   8's binary. 2.2% faster at the training shape (4.4% of the wave-tiles are skipped there).
10. [fwd_r3_a] QK K loads issued and consumed d-step-major (`QK_DMAJOR = True`,
    `QK_KV_FENCE = True`): the 48 K `ds_load_b128` of a tile were issued kv-tile-major while
    LLVM ordered the QK WMMAs d-step-major, and `s_wait_dscnt` retires in order, so a wave waited
    for 26-38 of 48 loads before its first WMMAs. `load_k_to_reg(dmajor=True)` issues them
    (d-step, kv tile, half) and `_qk_gemm` runs d-step-major with `sched_barrier(0)` between
    d-step groups and kv pairs, so every wait releases at most one d-step. Same addresses and
    per-chain accumulation order: o / lse bitwise equal to item 9; with both knobs off the kernel
    compiles to item 9's binary. 0.5% faster at the training shape (1.1% at b1 s4096 h64).
11. [fwd_r4_a, fwd_r5_a_g16] Head-grouped LPT dispatch (`KV_GROUP = 16`, `KV_HEAD_CHUNK = 16`, in
    `_lpt_block_id`): pure LPT hands the first `gy*gz` dispatched workgroups the last q block of
    every (batch, head), so at the training shape the 256 resident workgroups stream 256 different
    heads' K/V and every later q block of a head re-reads it after eviction. When
    `gx % KV_GROUP == 0` and `gy*gz % KV_HEAD_CHUNK == 0` (a runtime-uniform select), windows of
    `KV_GROUP` LPT ranks are dispatched `KV_HEAD_CHUNK` heads at a time, so at b2 s4096 h128 and at
    the folded launch [1, 4096, 256, D] every 256-id dispatch block holds 16 heads x all 16 q
    blocks and a head's K/V is shared in L2. Other grids (small ones included) keep pure LPT. The
    remap is a bijection of the block id (checked on the CPU for gx 1..64, gy*gz 1..1024), every
    workgroup still computes one (batch, head, q block): o / lse bitwise equal to item 10. About
    13% faster than item 10 at the training shape, 0.99x the time of aiter's gfx1250 ASM forward
    in the same process. (fwd_r4_a was the first version, `KV_GROUP = 4`, `KV_HEAD_CHUNK = 64`.)

## Small grids and partial tiles

- No host-side small-grid fallback is needed. At b1 s1024 h8 and b1 s2048 h16 (gx = 4 / 8, not a
  multiple of `KV_GROUP`) the kernel already takes the pure-LPT path of item 11 and runs as fast as
  a build with `KV_GROUP = 1` (1.000x, same-process A/B); the build before items 9-11 is 2.5-3.5%
  slower there.
- The Turbo gate accepts seqlen_q % 64 == 0, so a 256-row workgroup may be partly past seqlen_q.
  Each 32-row wave is then either fully valid or fully past the end: an empty wave's Q TDM has
  dim0 = 0 and its O stores are skipped by explicit predicates, but its masked LSE lanes store at
  byte offset 0x7FFFFFFF and rely on the buffer descriptor dropping out-of-range stores. A CPU
  replay of every Q / K / V / O / LSE access checks the index math for these shapes, and a
  sentinel-guarded run on gfx1250 confirmed that the masked LSE stores are dropped (seqlen_q 64,
  128, 192 and seqlen_q < seqlen_kv rectangles, batched and folded; o 54.2-55.1 dB and lse
  142.9-146.5 dB against fp32, bitwise equal to the same call on ordinary allocations).

## Version requirement

The kernels use gfx1250 FlyDSL ops whose API changes between releases (TDM, `s_wait_dscnt` /
tensorcnt waits, split barriers, `ds_load_tr16`) and were validated with flydsl 0.3.4.x only.
`gfx1250_mla_version.FLYDSL_REQUIREMENT` (`>=0.3.4,<0.3.5`) is checked when the package is
imported (ImportError otherwise) and by the attention gate before any kernel module is imported
(the call falls back to another backend). After a FlyDSL upgrade, rebuild compile-only and compare
the kernel's code hash before widening the requirement.
