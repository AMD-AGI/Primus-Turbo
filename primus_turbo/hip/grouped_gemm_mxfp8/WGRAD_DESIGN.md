# HIP MX-FP8 Wgrad Kernel — Design Notes

Status: **v0 correctness shipped, v1 fused kernel deferred.**

Production hybrid autograd uses **Triton variable-K wgrad** (HIP path is only
competitive once v1 fused kernel is built). v0 is provided as a correctness
oracle for the v1 work.

## v0 → v1 progression (Python shims)

Current shipped: **v1 single-launch** in
`primus_turbo/hip/grouped_gemm_mxfp8/__init__.py::grouped_gemm_mxfp8_hip_wgrad`.

### v0 (initial, 2026-04-23)

Approach:
1. ONE global permute-contig: `go_bf.view(G, M_g, N).permute(0, 2, 1).contiguous()` → `[G, N, M_g]`. Same for `a`.
2. ONE batched rowwise quant each (flatten leading dim): `quant_mxfp8_rowwise(view(G*N, M_g))`.
3. **G per-expert fwd-kernel launches**, role-swapped: `A_k = go[g]` shape `[N, M_g]`, `B_k = a[g].unsqueeze(0)` shape `[1, K, M_g]`.

Perf on gpt_oss_20B: **7.733 ms (281 TFLOPS)** — dominated by 32 kernel launches (~50us × 32 = 1.6ms launch overhead alone).

### v1 (single-launch, 2026-04-23)

**Key insight**: for balanced MoE, M_g is uniform across experts, so the fwd grouped kernel (which assumes uniform K_kern across groups) can handle all G experts in **one call** by stacking: `group_offs_stacked = [0, N, 2N, ..., G*N]`, A_kern = `[G*N, M_g]`, B_kern = `[G, K, M_g]`.

Perf on gpt_oss_20B: **4.100 ms (530 TFLOPS)** — **1.88× over v0, 0.404× Triton** (1.658 ms). Correctness preserved: 28.46 dB vs bf16; inf dB vs Triton (bit-exact).

Optimization attempts that did NOT help:
- `quant_mxfp8_dual` (both row+col outputs): 5.04 ms — wasted row output writes.
- `quant_mxfp8_colwise_for_variable_k` + fp8 permute (save bf16 transpose): 4.84 ms — colwise quant kernel is slower than rowwise.

### Constraints
- Balanced MoE only (uniform M_g across experts)
- M_g ≥ 384 (fwd kernel prologue minimum)
- M_g % 128 == 0 (fwd kernel BLOCK_SIZE_K assumption)

## v2 attempt (aborted): 128×128×128 custom kernel

Attempted 2026-04-24. Wrote a standalone 128×128×128 tile kernel using the
builtin ``__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`` with
compiler-managed registers + single-buffered LDS (~36 KB → occ=3-4, per the
memory-bound occupancy guidance in
``rocm-docs-markdown/hip_skill/references/hip-docs/understand/performance_optimization.md``).

Correctness failed (8.6 dB SNR) because the MFMA operand lane-layout for
``16×16×128 f8f6f4`` on gfx950 is **not** a flat ``row = lane%16, cols =
(lane/16) * 32 + [0..32)`` mapping. The 256-sibling kernel
(``turbo_gemm_mxfp8_kernel.h``) uses a 2-stripe split per lane
(cols ``(lane/16)*16 + [0..16)`` and ``(lane/16 + 4)*16 + [0..16)``) fed
via paired ``ds_read_b128`` with an XOR-swizzled LDS descriptor. Even after
matching the 2-stripe layout in the custom kernel, SNR stayed at 8.6 dB —
the actual ISA lane mapping involves additional swizzling the 256 sibling
resolves via the ``lds_offsets[]`` / ``swizzle_col_`` helpers.

**Lesson**: writing a custom MFMA-based kernel from scratch for
``f32_16x16x128_f8f6f4`` requires either (a) reusing the 256-sibling's LDS
descriptor + ``ds_read_pinned`` pattern (deep copy of ~400 lines of phase
code), or (b) using CK-tile primitives (``static_distributed_tensor`` +
``tile_window`` + ``intrin_mfma_scale_f32_16x16x128f8f6f4<16,16,0,0>``).
Either path is ~2-3 days of careful work. The attempt and its failure-
mode is the v2 scope estimate in the table below.

## v2.b attempt (aborted): hipBLASLt MX-FP8 grouped GEMM

Patched the C++ binding (`csrc/pytorch/grouped_gemm/hipblaslt_grouped_gemm.cpp`)
to accept `granularity == "MX_BLOCKWISE"` and set
`HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`. Also patched
`csrc/kernels/grouped_gemm/hipblaslt_grouped_gemm.cu` to advance per-expert
A_scale/B_scale pointers (the original wrapper's ``// TODO support variable
scale mode`` shared scale ptrs across experts).

### Forward result (single-GEMM-per-expert via hipBLASLt loop)

| metric | value |
|---|---|
| Correctness | **28.43 dB** vs bf16 (= fp8 floor); 47.86 dB vs Triton |
| HIP (256-tile) | 1.352 ms / 1608 TFLOPS / **1.137× Triton** |
| Triton | 1.538 ms / 1414 TFLOPS / 1.00× |
| **hipBLASLt** | **9.327 ms / 233 TFLOPS / 0.165× Triton** ← **6× SLOWER** |

Root cause: Primus-Turbo's `hipblaslt_grouped_gemm.cu` wrapper does
**per-expert `hipblasLtMatmul` calls** with 4-stream pingpong, not a single
Tensile-fused grouped kernel. 32 launches × per-launch overhead dominates.
The actual `hipblaslt_ext::GroupedGemm` API supports single-launch grouped
matmul but isn't wired up in Primus-Turbo.

### Wgrad result

**Architecturally cannot work** with `VEC32_UE8M0` scale mode:
- hipBLASLt's `VEC32_UE8M0` groups scales along the **innermost dim** of each
  operand, "stored in the same order as A" (per
  `rocm-docs-markdown/hipblaslt_skill/.../api-reference.md:602`).
- hipBLASLt MX-FP8 also requires `opA=T, opB=N`. With opA=T, the innermost
  dim of A (post-storage) is the **output row dim** (m_hbl), NOT the
  reduction (K_hbl).
- For fwd, m_hbl ≡ M (output) and K_hbl ≡ K (reduction); innermost of A as
  stored ([m, k] row-major) is K = reduction → scales along reduction ✓.
- For wgrad, K_hbl ≡ M_g (reduction). To get scales along M_g, we'd need
  M_g as innermost of A's storage — i.e., A stored [k_out, M_g] row-major
  → opA=N, which **violates the MX-FP8 opA=T constraint**.

Empirically: routing MX wgrad through the patched binding gives **7.37 dB SNR**
(garbage), as expected.

Backend registration was **reverted** for both `GroupedGEMMFP8HipblasltBackend`
(fwd, slow) and `GroupedGEMMFP8VariableKHipblasltBackend` (wgrad, broken).
The C++ patches are kept (harmless, unblocks future work when single-launch
hipblaslt_ext::GroupedGemm is wired up).

## v2 — Strided-load HIP fwd kernel (design, unimplemented)

The last 0.4-0.5 ms e2e gap in the balanced path and the 4-5× gap in the
unbalanced path both come from the same root cause: the HIP fwd kernel
assumes **row-major contiguous A and B with inner stride = 1**. For wgrad,
the pre-col-quant data is `[M_total, L]` with L inner, but we need the
kernel to reduce along M — so we currently permute to `[G, L, M_g]` before
each call (189+377 MB HBM write+read per step = ~240 µs).

### What needs to change

The kernel template
``GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950`` uses BufferSRD
with an implicit row-stride = K (the inner dim). See
``csrc/kernels/gemm/turbo/turbo_gemm_mxfp8_kernel.h`` methods:

- ``compute_ldg_offsets(uint32_t (&ldg)[2], const uint32_t stride)`` —
  computes per-thread offsets from `lane_id` + `stride`. Strides are
  currently = K.
- ``load_a_gmem_to_smem_half_srd<H>(a_srd, ldg_offsets, …, extra_soffset)`` —
  issues ``buffer_load_ushort/_dword`` with a single ``soffset`` per call.
  This treats the source as row-major contiguous from ``a_base_ptr`` with
  row-stride = K.

For wgrad strided-load, these need **two** stride parameters:
``stride_m`` (outer, = number of elements between consecutive rows in the
source = L_total for our col-quant) and ``stride_k`` (inner, = 1 or a
transposed-access stride). Each ``buffer_load`` must encode the per-row
offset as ``row_idx * stride_m``.

### Minimum-scope rewrite

Option 1 — **Add a `stride_m` parameter** alongside `a_srd` in all ldg
helpers. Pass it through from the kernel launcher. Replace implicit
`row * K` with `row * stride_m`. Same goes for scale loads. Effort:
~2 days once MFMA lane layout is understood.

Option 2 — **Use CK-tile's `tile_window` + `static_distributed_tensor`** to
replace Primus-Turbo's hand-pinned LDS path. Tile-window handles arbitrary
strides naturally. Effort: ~3 days, because the 4-warp × 256×256 phase-
scheduling code has to be rewritten in CK's coordinate system.

### Projected impact

Eliminates the 566 MB HBM traffic (~240 µs at 5.3 TB/s peak).

| path             | current e2e | projected e2e |
|------------------|------------:|--------------:|
| step no-prequant | 9.38 ms     | ~9.05 ms (match Triton) |
| step prequant k1 | 6.34 ms     | ~5.90 ms (match Triton) |

For unbalanced shapes, strided loads also eliminate the permute layout
mismatch — a single kernel launch could handle variable M_g directly
(like Triton's variable-K kernel). Closes the 4-5× unbalanced gap.

### Work partitioning

1. **Day 1**: understand the existing ``ds_read_pinned`` + ``swizzle_col_`` layout
   (the MFMA lane mapping that blocked the earlier 128×128×128 attempt). Write
   a single-GEMM strided-load variant with correctness test.
2. **Day 2**: integrate stride into the 4-phase pipeline (``phase_mfma_lds``,
   ``phase_mfma_lds_ldg``, ``phase_mfma_only``). PMC-validate it matches the
   original at the fwd shape.
3. **Day 3**: wire into variable-K dispatch (per-expert base pointers + M_g
   loop). Correctness on gpt_oss_20B. Perf bench.

Skill reference: `kernel-agents/knowledge_base/skills/mfma_16x16x128_f8f6f4_operand_layout.json`
(documents the layout pitfalls from the earlier failed attempt).

## v1.5 — Padded per-expert HIP wgrad for unbalanced MoE (2026-04-24)

`grouped_gemm_mxfp8_hip_variable_k_padded` in ``__init__.py`` handles arbitrary
per-expert M_g (M_g % 16 != 0, M_g % 128 != 0, M_g < 384) via pad-to-128
(min 384) and per-expert HIP fwd kernel calls. Zero-token experts are skipped.

### Results on real gpt_oss_20B unbalanced shapes (tbench timing)

| shape                                | HIP pad ms | Triton ms | HIP/Tri |
|--------------------------------------|-----------:|----------:|--------:|
| balanced (M_g=2048)                  | 8.33       | 1.72      | 0.21×   |
| Entry -5 (more uniform)              | 8.34       | 1.87      | 0.22×   |
| Entry -10 (max=16053)                | 8.49       | 2.06      | 0.24×   |
| **warmup (4 non-zero experts)**      | **2.81**   | **1.85**  | **0.66×** |

All pass correctness (28.46 dB vs bf16 = fp8 floor).

**Use when**: pure-HIP code paths required OR catastrophic few-non-zero-expert
cases (warmup). For general unbalanced MoE, Triton variable-K wins by ~4-5×
because it is a single persistent-kernel launch.

Autograd default keeps the existing routing:
- Aligned M_g % 16 == 0 and balanced M_g % 128 == 0 → HIP path
- Else → Triton fallback (both fwd/dgrad and wgrad)

`grouped_gemm_mxfp8_hip_variable_k_padded` is available as an opt-in entry
point for users who need all-HIP builds.

## v1.4 — HIP wgrad wired as autograd default (2026-04-24)

After v1.3 the kernel-only HIP wgrad is essentially equivalent to Triton
(1.67 ms vs 1.69 ms, measured via `torch.utils.benchmark.Timer`). Per user
directive "use HIP for wgrad, don't use Triton", the autograd backward now
routes wgrad through `grouped_gemm_mxfp8_hip_variable_k` when balanced-MoE
constraints are met (M_g uniform, ≥384, %128==0). Unbalanced shapes fall back
to Triton (HIP v1 constraint).

Also removed a `torch.equal(group_offs, expected)` assertion in the HIP
entry point that was forcing a CPU-GPU sync per call — caller now bears
the responsibility of passing balanced `group_offs`.

### E2E step-time impact (gpt_oss_20B, balanced)

| path                  | step_no_prequant | step_prequant_k1 |
|-----------------------|-----------------:|-----------------:|
| Triton wgrad (prior)  | 8.89 ms          | 5.88 ms          |
| **HIP wgrad (now default)** | **9.38 ms**      | **6.34 ms**      |
| Δ                      | **+0.49 ms**      | **+0.46 ms**     |

HIP e2e is 0.4-0.5 ms *slower* despite equivalent kernel-only time. Root
cause (by arithmetic): HIP path writes the permuted operands to HBM (566 MB
total: 189 MB for a, 377 MB for grad_out) then reads them back in the kernel.
At ~5.3 TB/s that's ~213 µs extra HBM traffic vs Triton's strided-in-kernel
reads. Plus 3 extra kernel launches (permute a, permute go, scale permutes)
~30 µs. Total ~240 µs inherent HIP overhead, consistent with measured gap.

### Closing the last gap requires a kernel-level change

To recover the 0.4 ms, the HIP fwd kernel (reused for wgrad via stacked-M)
needs to accept strided A/B tensors directly, eliminating the pre-permute.
This means modifying the kernel template's `BufferSRD`/`ds_read` patterns
to index with arbitrary stride — non-trivial given the tight register-pinned
layout. Defer to v2 (purpose-built variable-K kernel).

## v1.3 — fast fp8 permute (2026-04-24) — HIP wgrad 0.44× → 0.97× Triton

Profile of v1 (3.8 ms total):

| step | measured |
|---|---:|
| `lhs.view(g,M_g,L).permute(0,2,1).contiguous()` (189 MB fp8) | **754 us** |
| `rhs.view(g,M_g,R).permute(0,2,1).contiguous()` (377 MB fp8) | **1534 us** |
| scale permutes (small) | ~30 us |
| HIP fwd kernel (at wgrad shape) | 1498 us |

PyTorch's generic `.permute().contiguous()` runs the fp8 transposes at ~12%
of HBM peak. A Triton LDS-tile transpose kernel hits ~85% of peak:

| op | torch | Triton | speedup | HBM peak |
|---|---:|---:|---:|---:|
| a fp8 permute  (189 MB) | 754 us | **83 us**  | **9.09×** | 71 us |
| go fp8 permute (377 MB) | 1534 us | **168 us** | **9.23×** | 142 us |
| scale permute  (small) | 10 us | 13 us | 0.8× (keep torch) | — |

After wiring the Triton permute into `grouped_gemm_mxfp8_hip_variable_k`:

| variant | ms | TFLOPS | vs Triton |
|---|---:|---:|---:|
| Triton variable-K (baseline) | 1.66 | 1310 | 1.000× |
| HIP v1 (bf16 permute path) | 4.10 | 530 | 0.40× |
| **HIP v1.3 (fast fp8 permute)** | **1.71** | **1268** | **0.97×** |

Files:
- New `_permute_fp8.py` — LDS-tile transpose kernel `fp8_permute_M_to_GN`.
- `__init__.py::grouped_gemm_mxfp8_hip_variable_k` — now uses the Triton permute.

The HIP wgrad is now essentially at parity with Triton wgrad. Hybrid autograd
keeps Triton as the default wgrad (still marginally faster kernel-only), but
the `grouped_gemm_mxfp8_hip_variable_k` entry point and the
`GroupedGEMMFP8VariableKHipBackend` are now competitive for users who want
all-HIP code paths.

## Why v1 can't close the last gap

Remaining 2.48× to Triton comes from the **fwd kernel at the wgrad-reshaped shape**:
- Wgrad reshape: M_kern = G*N = 184320, N_kern = K = 2880, K_kern = M_g = 2048
- HIP fwd kernel's 256×256×128 tile is tuned for (M=65536, N=5760, K=2880) — the original fwd shape
- At the wgrad shape: 2880 is tile-unaligned (2880 % 256 = 64), short K_kern (2048 vs 2880) means less compute/tile amortization
- Measured: ~1610 TFLOPS at fwd shape → ~690 TFLOPS at wgrad shape (43% efficiency drop)

Triton's variable-K wgrad kernel is purpose-tuned for this exact shape pattern.

## Why deferred

The HIP fwd + dgrad win came from collapsing Triton's ~14 s_waitcnt per MFMA
cluster down to 1 (confirmed in AMDGCN dump: 27 total waitcnt across 512
MFMAs, 16× fewer/MFMA than Triton). That's a **compute-bound-kernel** win.

Wgrad is **memory-bound** on this shape (KB note
`tl_dot_scaled_gfx950.md`: stall ratio ~0.91 on Triton's variable-K kernel).
For a memory-bound path, waitcnt optimization doesn't move wall-clock — HBM
bandwidth caps the throughput. Expected outcome of a HIP port:

- Best case: match Triton's 1291 TFLOPS (1.33× bf16)
- No realistic path to 1.5× bf16 unless we can cut HBM traffic

Given the cost (1-2 days of kernel work + register/LDS engineering) and
the hybrid autograd already delivers 1.495× step-ratio with prequant on
gpt_oss_20B, the HIP wgrad is a future PR, not a session-scope item.

## Math

```
dB[g, n, k] = Σ_{m ∈ [offs[g], offs[g+1])} grad_out[m, n] * A[m, k]
```

Per expert g: output shape `[N, K]`, reduction along M_g (variable).

## Inputs (pre-quanted by Triton `quant_mxfp8_dual_jagged`)

| Tensor              | Shape                  | dtype     | Layout notes |
|---------------------|------------------------|-----------|--------------|
| `a_col`             | `[M_total, K]`         | fp8 e4m3  | col-quantised; data row-major (K inner) |
| `grad_out_col`      | `[M_total, N]`         | fp8 e4m3  | col-quantised; data row-major (N inner) |
| `a_scale_col`       | `[K, total_sc]`        | uint8 e8m0| jagged: expert g owns cols `[scale_offs[g], scale_offs[g+1])` |
| `grad_out_scale_col`| `[N, total_sc]`        | uint8 e8m0| same jagged layout |
| `group_offs`        | `[G+1]`                | int64     | M-space prefix sums |
| `scale_offs`        | `[G+1]`                | int64     | scale-space prefix sums (= `cumsum(ceil(M_g, 32))`) |

Output: `dB [G, N, K]` bf16 / fp16.

## Proposed kernel structure

Mirror the fwd HIP kernel's `GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950`
with these **specific deltas**:

### Flat-grid dispatch
- `total_tiles = Σ_g ceil(N/256) * ceil(K/256)`
- `block_to_expert[bid]` → expert index
- `tile_offs_within_expert[g]` → per-expert tile prefix
- Inside expert g: `local_tile = bid - tile_offs[g]`, `pid_n = local / tiles_k`, `pid_k = local % tiles_k`

### Base pointers
- `a_col_base = a_col + pid_k * 256 * K`  — read K=256 rows × M_total cols but **we want M_g cols only**
- `grad_out_base = grad_out_col + pid_n * 256 * N` — read N=256 rows × M_total cols, then slice M

Wait: `a_col` is stored as `[M_total, K]` row-major. `a_col[m, k]` = `a_col + m*K + k`. For wgrad's B_kern operand (want rows=K, inner=M, 128-reduction): we need `a_col[m, k]` for fixed k-stripe (16 K cols), varying m (128 M rows). That's a **strided read** (stride = K).

### Operand layout — the hard part
MFMA 16×16×128 needs:
- A operand: 16 rows × 128 K-reduction cols per warp, packed as `int32x8` per lane (32 fp8 / lane)
- B operand: same shape

For wgrad:
- "A operand" = grad_out[16 N-rows, 128 M-cols] — storage: grad_out_col stored `[M, N]` so reading 16 N-cols × 128 M-rows is **transposed** vs. native row-major layout.

**Two approaches**:

1. **LDS transpose**: cooperative load `grad_out_col[m_blk:m_blk+128, n_tile:n_tile+256]` to LDS with a transposing store pattern (e.g., column-store into row-organised LDS), then `ds_read` consumes it row-major for MFMA. Adds 2 barriers per K-iter but preserves the kernel's compute-pipeline structure.
2. **Operand shuffle post-load**: load with native pattern + use `ds_bpermute` / `v_perm` to shuffle into MFMA format. No LDS transpose store, but costs VALU.

LDS transpose is the standard choice and proven in flash-attention-style wgrad patterns (CUTLASS "stream-k-wgrad"). Budget ~1-2 extra LDS barriers per 128-M K-iter.

### Scale loading
Per-expert scale base pointers:
```
a_scale_base = a_scale_col + pid_k * 256 * total_sc + scale_offs[g]
grad_out_scale_base = grad_out_scale_col + pid_n * 256 * total_sc + scale_offs[g]
```

For 128-M reduction step: fetch 4 scale bytes (= 128/32) per row, 16 rows per warp sub-tile. This is the **SAME access pattern** as the fwd kernel's scale loads — reuse `preshuffle_scale_16x4_padded_kernel` with jagged base pointer.

Preshuffle step:
1. For each expert g, preshuffle `a_scale_col[:, scale_offs[g]:scale_offs[g+1]]` into expert-local uint32 buffer
2. Same for `grad_out_scale_col`
3. Per-expert scale_cols = `scale_offs[g+1] - scale_offs[g]` — need padding to multiple of 4 per expert
4. Total workspace = `sum_g (K + N) * ceil(scale_cols_g, 4) * 4 bytes`

### Variable-K reduction loop
Loop iterates `m_blk` from `group_offs[g]` to `group_offs[g+1]` step 128:
- If remaining M_g < 128, the last iter is a masked load (BufferSRD returns 0 for OOB rows)
- Scale for partial tail: e8m0 0x00 = 2^-127 ≈ 0 — as long as scale buffer is zero-padded
- K_iter count per expert = `ceil(M_g / 128)` — **variable per expert** (this is the "variable-K" moniker)

Critical: the kernel's prologue/epilogue pipeline (double-buffered LDS) assumes ≥3 iters. For small experts (M_g < 384 = 3×128), the kernel must fall back to a simpler "no-pipeline" path. Either:
- Reject small experts (host-side filter)
- Separate kernel variant for the short-K case

## Estimated engineering

| Step                                        | Effort |
|---------------------------------------------|--------|
| LDS-transpose tile load + ds_read reshape  | 0.5-1d |
| Jagged scale preshuffle + base ptr math    | 0.5d   |
| Variable-K loop + <384 fallback             | 0.5d   |
| Correctness harness vs Triton wgrad         | 0.5d   |
| PMC + one round of tuning                   | 0.5d   |
| **Total**                                   | **~2-3d** |

## Expected outcome (grounded in Triton PMC)

| metric                    | Triton wgrad | HIP wgrad (projected) |
|---------------------------|--------------|-----------------------|
| Stall ratio (WAIT_LDS/LDS)| ~0.91 (mem-bound) | ~0.85 (marginal improvement — LDS transpose adds fences) |
| TFLOPS                    | 1291         | **1291–1500** (≤1.16×)|
| Memory BW utilization     | ~80% of peak | ~80% (same ceiling)  |

Bottom line: HIP wgrad is a **parity play**, not a speedup play. Ship only
if the caller cares about pure-HIP code paths (regulatory/platform reasons)
or to unlock prequant savings on the col-quant path (currently the col-quant
of A+grad_out is shared across dgrad+wgrad, so switching wgrad backend does
not change quant cost — parity-only from that angle too).

## Recommendation

**Keep the hybrid** (HIP fwd + HIP dgrad + Triton wgrad) until a specific
memory-bandwidth improvement (e.g., scale compression, a new gfx950 LDS
instruction) changes the ceiling analysis.

If you do write it, start with balanced MoE only (M_g = M/G, a simple
non-jagged path — no flat-grid dispatcher needed) to validate the LDS
transpose + scale preshuffle, then generalise to unbalanced.
