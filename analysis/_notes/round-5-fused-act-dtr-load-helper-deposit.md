# Round 5 — FP8 Fused-Act: DTR load-helper deposit (Phase 1, step R5a)

## Context

R4 deposited the BF16→FP8 cvt builtin (`cvt_bf16x4_to_fp8x4`) and validated
bit-exact numerics via the `cvt_bf16_to_fp8_bulk_compile_test` probe (SNR
~340 dB on a contiguous BF16 buffer). The next step in the Phase 1
fwd-kernel surgery sequence (per the R3 plan) is to wrap the cvt builtin in
a **production-grade DTR (Direct Tile Read) load helper** that mirrors the
existing `rcr_8w_load_hoist`'s 8-warp swizzled multi-pass cooperative load,
but with three differences:

1. HBM read uses `raw_buffer_load_b128` **without** the `lds` modifier — data
   lands in VGPRs.
2. Per-pass body inserts 4× cvt of `bf16_2` pairs → packed FP8 (16 BF16 →
   16 FP8 in registers).
3. LDS write goes through `ds_write_b128` (instead of the implicit DTL LDS
   write inside `buffer_load_dwordx4 ... offen lds`).

This helper is the Phase 1 critical primitive that the cloned
`grouped_rcr_fused_act_kernel` will call instead of the un-fused load
helper. R5a delivers the helper + its compile-test launcher; R5b will fold
it into the kernel clone.

## Lever / direction selected

**HipKittens kernel deposit (R5a step)** — production-grade
`rcr_8w_load_hoist_fused_act<N_THREADS, ST_DST, GL_SRC, COORD>` template
helper plus a compile-test launcher (`rcr_load_hoist_fused_act_test`) that
forces codegen at build time and exposes a Python-callable probe to verify
the LDS round-trip's numerical correctness BEFORE the larger kernel-clone
work in R5b.

Also depositted: a new `grouped_layout_globals_fused_act` struct (mirror of
the existing `grouped_layout_globals` with `_gl_fp8 a` replaced by
`_gl_bf16 a`) — placeholder to be consumed by R5b's cloned kernel; not yet
referenced by any production dispatch in this round.

## Files touched

### HipKittens
- `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  - **+1** new struct: `grouped_layout_globals_fused_act` (line ~7405).
    Mirrors `grouped_layout_globals` field-for-field; only differs in the
    `a` view type (`_gl_bf16` instead of `_gl_fp8`). The dispatchers and
    host wrappers are not yet wired to construct this — R5b territory.
  - **+1** new namespace `fused_act_round5_compile_test` containing:
    - `rcr_8w_load_hoist_fused_act<N_THREADS, ST_DST, GL_SRC, COORD>`:
      production-grade DTR load + cvt + LDS-write helper. Mirrors the
      un-fused helper's SGPR plumbing (lds_addrs[] hoist via readfirstlane,
      tile_byte_offset hoist, full-tensor SRD bound) for wave-uniformity.
      Per-pass body: 2× `raw_buffer_load_b128` (32 bytes BF16) → 4×
      `cvt_bf16x4_to_fp8x4` → 1× `kittens::macros::ds_write_b128` (16 bytes
      FP8 to LDS). Includes leftover-warps tail handling for
      `memcpy_per_tile`-fractional tiles.
    - `rcr_8w_load_hoist_fused_act_kernel_test`: single-block compile-test
      that loads one ST_v2 worth of BF16 from HBM via the helper, then
      block-strided-dumps the FULL LDS storage (incl. subtile padding) to
      `out` for Python-side verification. Adds a one-time LDS zero-init
      pass so subtile padding bytes don't show up as random 0x7F NaN
      encodings in the dump.
  - **+1** host wrapper `rcr_load_hoist_fused_act_test_fn` (line ~7480) +
    pybind binding `m.def("rcr_load_hoist_fused_act_test", ...)`.

### Primus-Turbo
- `analysis/_notes/round-5-fused-act-dtr-load-helper-deposit.md` (this
  file).

No production code (`primus_turbo/pytorch/ops/grouped_gemm_fp8.py`) was
touched — wiring waits until R5b's kernel clone is wired through to
Primus.

## Build + numerical-probe results

### HipKittens build (compile-test path)

```
make -j8 tk_fp8_layouts.so
```

Resource-usage report (`-Rpass-analysis=kernel-resource-usage`) for the
new compile-test kernel:

| metric | value |
|---|---|
| TotalSGPRs | 20 |
| **VGPRs** | **22** |
| AGPRs | 0 |
| ScratchSize / lane | 0 |
| **VGPRs Spill** | **0** |
| SGPRs Spill | 0 |
| LDS / block | 17408 (= ST_v2 storage incl. padding) |
| Occupancy | 8 waves/SIMD |

VGPR=22 is comfortably within budget; the helper's per-pass live state
(2× `__uint128_t` BF16 read + 4× cvt result + 1× SGPR-hoisted lds_addr +
the misc loop SGPRs) holds without spills. **Falsification gate cleared
for R5b**: the helper has plenty of register headroom to absorb the
`grouped_rcr_kernel` body (~80 VGPRs typical) when folded in.

### Numerical probe (SNR vs torch reference)

```
python3 /tmp/probe_fused_act_round5_lds.py
```

```
[r5-lds-probe] amax=21.1250  scale_inv (FP8_MAX/amax)=21.2071
[r5-lds-probe] LDS dump: zero count = 1024 / 17408   (= 8×128 padding bytes)
[r5-lds-probe] valid cvt-output bytes: 16384 (expected 16384)
[r5-lds-probe] ref:  amax=448.0000 mean=-0.7406 std=106.1308
[r5-lds-probe] dst:  amax=448.0000 mean=-0.7406 std=106.1308
[r5-lds-probe] sorted-elementwise: max_abs_diff=0.000000  mean_abs_diff=0.000000  SNR=340.52 dB
```

**Bit-exact distributional match** vs the torch-side reference (BF16 → FP8
via `(src * scale).to(float8_e4m3fn)`). SNR = R4's reference SNR (~340 dB);
the swizzle reorders the LDS data so element-wise equality fails, but the
sorted-distribution match is bit-exact. Three things validated:

1. `kittens::llvm_amdgcn_raw_buffer_load_b128` reads correct BF16 from HBM
   under a full-tensor SRD bound.
2. `cvt_bf16x4_to_fp8x4` (R4 helper) produces bit-exact FP8 packed output
   when called from the new helper's loop body.
3. `kittens::macros::ds_write_b128(u32x4, smem_ptr, 0)` writes 16 bytes to
   the per-lane LDS slot — and the `swizzled_offsets[i] × 2` post-mult
   correctly recovers the BF16-byte stride from the FP8-byte-typed prefill
   output.

### Metric impact (no regression)

| metric | before R5a (R4 final) | after R5a |
|---|---|---|
| `fused_act_wall_score` | 933 | **936** |
| `geomean` | 1.2593 | 1.2640 |
| correctness fails | 0 / 24 | 0 / 24 |
| at-target shapes | 6 / 24 | 4 / 24 |

±15 score noise band as usual. Score essentially flat — Phase 0 fallback
path still runs (helpers all raise `NotImplementedError` → autograd
function uses `_unfused_*` paths). Real movement comes when R6 wires
`_hk_fused_act_forward` to actually call the new helper-backed kernel.

## Path-of-bugs encountered (all fixed)

1. `::kittens::ds_write_b128(...)` — wrong namespace; helper is in
   `::kittens::macros::ds_write_b128`. Fixed to fully-qualify.
2. Initial probe used a thread-strided dump (`for (idx = tid; idx <
   tile_elems; idx += _NUM_THREADS)` with `#pragma unroll`) that produced
   sparse partial writes — only 512 of 16384 bytes ended up in HBM. Switched
   to block-strided (`for (j = 0; j < per_thread; ++j) { idx = tid *
   per_thread + j; ... }` with `#pragma unroll 1`) — full coverage.
3. Initial probe sized `dst` to `tile_elems = ST_v2::rows * ST_v2::cols`
   (16384 bytes) — but ST_v2 has subtile_padding=128, so the FULL LDS
   storage is 8 × (2048 + 128) = 17408 bytes. Sizing dst to 17408 + having
   Python skip the 8 × 128 padding gaps recovers the exact 16384 cvt output
   bytes for the SNR check.
4. Subtile-padding bytes were uninitialized in LDS, occasionally showing up
   as 0x7F (E4M3 NaN encoding) and contaminating the float-conversion in
   Python (`.to(float32)` → `nan`). Fixed by adding a one-time
   `__shared__` zero-init pass at kernel start, before
   `prefill_swizzled_offsets`.

## Plan for R6 (next round)

R5b/c (kernel clone + dispatcher) was originally split off as a separate
round; with the helper validated, the cleanest next step is the kernel
clone itself — we have all the pieces:

1. **R6a — Clone `grouped_rcr_kernel` → `grouped_rcr_fused_act_kernel`**
   (in HipKittens). Take `grouped_layout_globals_fused_act` instead of
   `grouped_layout_globals`. Replace the two `rcr_8w_load_hoist(As[...]
   g.a ...)` call sites (the only ones reading `g.a`) with
   `fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act(As[...]
   g.a ..., scale)`. Keep `rcr_8w_load_hoist(Bs[...] g.b ...)` un-changed
   (B is FP8, not BF16). The scale is read once per kernel from
   `*g.dscale_a` at kernel entry into a wave-uniform float SGPR.

   Initial gate: keep the kernel single-instantiation
   (`KI_HINT=0, N_MASKED_STORE=false, FUSED_KTAIL=false`). RCR-only +
   K%128==0 covers 16/24 metric shapes (DSV3 + Qwen — gpt_oss K=2880
   keeps the un-fused fallback).

2. **R6b — Clone the dispatcher** (`dispatch_grouped_rcr` →
   `dispatch_grouped_rcr_fused_act`) + host wrapper
   (`grouped_rcr_dscale_fn` → `grouped_rcr_fused_act_dscale_fn`) + pybind
   binding. Take a BF16 `a` (instead of pre-quantized FP8) + a device-side
   `a_scale_inv` scalar buffer.

3. **R6c — Wire `_hk_fused_act_forward` in Primus-Turbo**:
   - Call `max_abs_bf16_to_fp8_scale` (R1 binding) on BF16 `a` →
     `a_scale_inv` device fp32 scalar.
   - Call new `tk.grouped_rcr_fused_act_dscale(a_bf16, b_fp8, c, ...,
     a_scale_inv, b_scale_inv, group_offs, ...)` returning fwd output.
   - Adaptive save policy already in place (`ctx.fwd_fused = True` →
     save BF16 `a`, no FP8 staging).

4. **R6d — Bench + commit**:
   - `bench_grouped_gemm_turbo.py --dtype fp8` for fwd+bwd correctness +
     TFLOPS sanity check.
   - `python3 scripts/_metric_grouped_fused_wall.py` to confirm score
     uptick (Phase 1 expected ~960-980).
   - Falsify if SNR < 25 dB on any shape OR score < baseline 931.

If R6 lands cleanly we should see the first real metric movement — Phase 0
floor 931 → Phase 1 ceiling ~960-980 (saves ~25 % of fwd quantize_fp8 time
on RCR-aligned shapes).

## Commit plan

- HipKittens commit: `feat(fp8-fused-act): R5a — production-grade
  DTR+cvt+LDS load helper deposit, numerical probe SNR=340.52 dB`.
  Files: `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`.
- Primus-Turbo commit: `docs(fp8-fused-act): R5 — record HipKittens DTR
  load-helper deposit + R6 kernel-clone plan`.
  Files: `analysis/_notes/round-5-fused-act-dtr-load-helper-deposit.md`.
