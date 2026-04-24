# HIP Wgrad v2 — Strided-Load / LDS-Transpose Kernel Plan

Concrete 3-day implementation plan for closing the final 0.4–0.5 ms gap in
HIP wgrad (balanced) and the 4–5× gap on unbalanced MoE.

**Do NOT start kernel coding from scratch.** The 128×128×128 aborted attempt
(documented in
`kernel-agents/knowledge_base/skills/mfma_16x16x128_f8f6f4_operand_layout.json`)
showed that blind MFMA-layout guessing gives 8.6 dB garbage. Reuse the
existing working tile struct and modify it in small testable steps.

## Scope

Let "strided-load" mean the kernel accepts explicit `stride_a_m`,
`stride_a_k` (in bytes), `stride_b_n`, `stride_b_k` rather than implicit
`row-major-contig-with-inner-K`. For the wgrad case, we additionally need
**LDS-transpose**: the kernel reads `[M_block, N_tile]` from source but
stores `[N_tile, M_block]` to LDS, so downstream MFMA feeding sees the
expected operand layout.

## What already works (baseline to NOT regress)

- `GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950` at gpt_oss_20B
  fwd shape: 1.35 ms / 1608 TFLOPS / 1.137× Triton.
- `turbo_grouped_gemm_mxfp8_kernel` (our grouped variant) with per-expert
  flat-grid dispatch.
- Fast Triton fp8 permute (`_permute_fp8.py`) — kept as fallback.

## Exact touch-points in the kernel

[`csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h`](../../../csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h)

Replace hardcoded `k` (= M-row stride in bytes) with a kernel parameter
`stride_a_m` / `stride_b_n` in these methods:

| Line  | Method | Change |
|-------|--------|--------|
| 145   | `load_a_gmem_to_smem_half_srd::soff`   | `... * k + extra` → `... * stride_a_m + extra` |
| 165   | `load_b_gmem_to_smem_half_srd::soff`   | similarly `stride_b_n` |
| 217   | `precompute_base_soff`                  | `(i * 64 + warp_id * MFMA_SIZE_M) * k` → `* stride_a_m` (for A), need separate helper for B |
| 304   | `compute_ldg_offsets`                   | `ldg_row * stride + ldg_col * 16` — already takes `stride` as arg, but callers pass `k` — OK |
| 533   | kernel body                             | pass `stride_a_m` / `stride_b_n` to `compute_ldg_offsets` |

## Day 1 — stride parameterization (no LDS transpose)

**Goal**: make the existing kernel accept explicit strides. At the fwd shape
with `stride_a_m = k` this is a no-op and must pass bit-exact.

### Concrete changes

1. **Add `stride_a_m`, `stride_b_n` (bytes) to the kernel signature.**
   Default values via constexpr overload preserve the existing call sites.

2. **Thread `stride_a_m` / `stride_b_n` through:**
   - `GemmTile` constructor (store as member)
   - All `load_a_*` / `load_b_*` methods' `soff` computation
   - `precompute_base_soff` split into `precompute_base_soff_a` (uses
     `stride_a_m`) and `precompute_base_soff_b` (uses `stride_b_n`)
   - `compute_ldg_offsets_a` / `_b` variants (or keep shared helper
     if the stride arg is wired through correctly)

3. **Kernel body (line 533 area)**: pass `k` as `stride_a_m` and `stride_b_n`
   initially. No behavior change.

4. **Correctness test**: `test_phase_a.py` on balanced MoE must still pass
   28.46 dB. PMC counters (MFMA util, SQ_WAIT_INST_LDS) must match within
   5% of baseline.

### Day 1 deliverable
- New header `turbo_gemm_mxfp8_kernel_v2.h` with strided-params tile struct.
- New kernel `turbo_grouped_gemm_mxfp8_v2.hip` that uses v2 tile.
- `test_v2_fwd.py` confirms no regression vs v1 at fwd shape.

## Day 2 — LDS-transpose integration

**Goal**: let the kernel load `[M_row_block, N_col_tile]` from source but
write to LDS in `[N_col_tile, M_row_block]` layout. Downstream MFMA loads
from LDS as usual.

### Concrete changes

1. **`load_gmem_to_smem_srd`** writes `Bytes` contig per-thread to a per-thread
   `lds_addr`. For transposed store, change `lds_addr` per thread so thread
   `t` writes its data at the TRANSPOSED LDS offset:
   - Thread holds data from `(m_row_t, k_col_t)` in source
   - Writes to LDS at `(k_col_t, m_row_t)` with swizzle preserved

2. **New helper `load_gmem_to_smem_srd_transposed<Bytes>`** that takes
   `lds_addr_transposed` computed differently from the non-transposed path.

3. **Per-thread ldg → lds address mapping**:
   - Input tile `[M_row=64, N_col=128]` row-major, byte-stride N
   - Each thread loads 16 bytes (16 consecutive N-col bytes) at `(m_row, n_col_start)`
   - Transposed LDS target: 16 bytes must be written to 16 DIFFERENT LDS rows
     (each column position of N becomes a row in LDS). This means
     **buffer_load_lds with 16-byte writes CANNOT do this** — each byte goes
     to a different LDS row (different cache line).
   - **Alternative**: load to VGPR first, then `ds_write` byte-by-byte to
     transposed LDS position. Costs VGPR pressure but is correct.

### Day 2 deliverable
- Strided + LDS-transpose load path in v2 kernel.
- Benchmark vs v1 at fwd shape (no-transpose mode) — confirm no regression.
- Benchmark vs v1 at wgrad shape (transpose mode) — target 1.10× the
  current v1+Triton-permute (aka 1.5 ms kernel time).

### Risk
The `buffer_load_lds` hardware path is designed for contig writes.
Transposed LDS stores break this. A VGPR-intermediate path **will**
slow the kernel by ~20-40% at wgrad shape (vs non-transposed). Expected
v2 kernel time at wgrad shape: 1.8-2.0 ms (vs current 1.7 ms via permute).
Still **eliminates the 240 µs permute overhead** for a net e2e win.

## Day 3 — variable-K dispatch

**Goal**: single-launch HIP wgrad that handles arbitrary per-expert `M_g`
without padding.

### Concrete changes

1. **Kernel takes `group_offs[G+1]` and computes per-tile `m_start_g`,
   `M_g` from the expert index** (already done in
   `turbo_grouped_gemm_mxfp8_kernel`).

2. **K-loop iterates `M_g / 128` times** per tile, with a masked tail iter
   for `M_g % 128 != 0`. Tail handling reuses the `EVEN_K=False` pattern
   from the fwd kernel's Triton sibling (`grouped_gemm_mxfp8_kernel.py:170-179`).

3. **No M_g >= 384 requirement** if we can skip the pipelined prologue
   for small experts (simple un-pipelined fallback path).

4. **Host-side**: single `grouped_gemm_mxfp8_hip_wgrad_v2(lhs, rhs, scales,
   group_offs)` that takes the `[M_total, L]` tensors directly. No permute.

### Day 3 deliverable
- `grouped_gemm_mxfp8_hip_wgrad_v2` handles all 8 real shapes from
  `gem_shape_summary.txt` at >=0.95× Triton perf on balanced and
  >=1.0× Triton on unbalanced (where Triton loses to launch overhead).
- Wire into `FP8GroupedGemmMXHipFunc.backward` as the primary wgrad path.
- Drop the Triton-fallback branch (HIP now handles unbalanced natively).

## Test harness (reuse existing)

- `test_phase_a.py` — fwd correctness
- `test_phase_b.py` — dgrad correctness
- `test_variable_k.py` — wgrad kernel-only bench
- `test_wgrad_padded.py` — unbalanced correctness sweep
- `test_real_shapes.py` — all 8 production shapes
- `bench_fwd_bwd.py` — full step bench (HIP vs Triton)

## Risks & mitigations

1. **Hardware `buffer_load_lds` cannot transposed-store**. Mitigation:
   VGPR-intermediate path. Costs VGPR pressure but works.
2. **AMDGCN lane-layout for MFMA operands when loaded transposed**.
   Mitigation: reuse the existing XOR-swizzle
   (`swizzle_col_(row, col) = col ^ (row >> 1)`) pattern from the
   current kernel — it's already correct for the operand layout.
3. **Register pressure blowing past 256 VGPR and dropping occupancy**.
   Mitigation: if it hits, fall back to a smaller per-warp tile (64×64
   instead of 128×128) but same 4-warp workgroup.
4. **Time-budget overrun**. Mitigation: ship Day 1 as a standalone
   improvement (stride-parameterized kernel is useful even without
   transpose — enables tuned non-wgrad callers).

## Alternative path — quant with pre-permuted output (ATTEMPTED, didn't pay off)

**Implemented 2026-04-24 as `quant_mxfp8_dual_jagged_permuted` in
`primus_turbo/triton/quantization/mxfp8_quant_kernels.py`.** Bit-exact
correctness vs `dual_jagged + fp8_permute_M_to_GN` post-permute. But the
fused kernel is **SLOWER** at the gpt_oss_20B grad_out shape:

| config                                  |    us | speedup |
|-----------------------------------------|------:|--------:|
| dual_jagged + post-permute (baseline)   | 1092  | 1.000×  |
| dual_jagged_permuted (fused)            | 1301  | 0.839×  |

### Root cause

Triton's blocked layout for the tile `[GROUP=32, BLOCK_L=128]` fp8 splits
threads along the OUTER axis first. Even after `tl.trans(xq_c)` swaps the
tile's logical axes, stores to the permuted `[G, L, M_g]` output with M_g
as the innermost dim are not coalesced (threads write 32-byte stripes at
different l_idx positions → different cache lines).

The existing fp8 post-permute kernel (`_permute_fp8.py::fp8_permute_M_to_GN`)
uses an LDS-tile transpose pattern that hits ~85% of HBM peak, which is
cheaper than the layout penalty of the fused approach.

### Status
Kernel kept in the source as a reference implementation (lines 807+ in
`mxfp8_quant_kernels.py`) with a detailed comment explaining the failure
mode. **Not wired into autograd.** Future work would need an explicit
`tl.BlockedEncoding` tuned for inner-M_g stores to beat the post-permute.

## Alternative path — quant with pre-permuted output (recommended — SUPERSEDED)

While drafting Day 1, realized **stride parameterization alone doesn't enable
direct col-quant read**: the HIP kernel's reduction axis is structurally the
"inner (K_kern) dim of A", so to reduce along M (wgrad), M must BE inner in
the source seen by the kernel. Without LDS transpose, that requires either
(a) source physically permuted, or (b) source physically stored that way.

Option (b) is cheaper than the v2 kernel rewrite:

**`quant_mxfp8_dual_jagged_permuted`** — modify the existing Triton quant
kernel to optionally emit `col_fp8` in `[G, L, M_g]` layout directly, instead
of `[M_total, L]`. The quant kernel already reads `x[M_total, L]` and has
`group_offs`, so it can compute per-expert destination offsets during the
write.

### Savings
- Eliminates the 240 µs runtime permute overhead per step (both a and
  grad_out permutes become free since pre-permuted at quant time).
- Extra HBM cost at quant time: **zero** — quant kernel already writes the
  fp8 output once; just writes to a different layout.
- Net e2e step delta: expect to match Triton (~1.18× pure Triton with prequant
  becomes ~1.22-1.25×).

### Engineering cost
- Modify `quant_mxfp8_dual_jagged` (or add sibling) to accept a
  `permute_col=True` flag and thread through per-expert offset math.
- Update `grouped_gemm_mxfp8_hip_variable_k` to skip permute when source is
  already in `[G, L, M_g]` layout.
- **~1 day**, vs 3 days for the strided-load kernel v2.

### Decision
**Prefer this alternative.** The kernel v2 work would be better spent on
unbalanced support (variable-K kernel) where pre-permuted layout can't help
(each expert has different M_g).

## Success criteria

- Kernel-only wgrad ≥ 1.00× Triton at gpt_oss_20B balanced
- Kernel-only wgrad ≥ 1.05× Triton at Entry -10 (max M_g=16053, very unbalanced)
- Hybrid+prequant step ≥ 1.25× pure Triton (currently 1.19×)
- All shapes in `test_real_shapes.py` pass 25 dB SNR
- No regression on fwd / dgrad (v2 kernel used only for wgrad)
