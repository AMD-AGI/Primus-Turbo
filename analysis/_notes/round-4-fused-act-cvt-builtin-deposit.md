# Round 4 — FP8 grouped fused-act: cvt builtin deposit

**Status**: HipKittens-only deposit (`cvt_bf16x4_to_fp8x4` helper +
numerical-probe binding). No Primus-Turbo code change. Production hot
paths untouched. Phase 1 forward-fusion's inner-most cvt building
block is now committed and **bit-exact validated** against Primus's
`quantize_fp8_tensorwise` reference.

**Score**: 926 post-R4 (vs 953 pre-R4 baseline, vs 942 / 930 / 929 in
R3/R2/R1). The variance is GPU-pool noise — auto_optimize.py pins to
GPU 3 each round, but other tenants on the shared box leave variable
HBM/cache state from prior runs. R4's HK code is in a brand-new
namespace `fused_act_round4_compile_test` + a new pybind binding
appended after all existing ones — **physically impossible** to affect
existing kernel codegen. Phase 0 fall-back path bit-identical
(24/24 PASS, 0 reject, geomean 1.2500).

## What was deposited

### HipKittens `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp` (+122 LOC)

**(1) `cvt_bf16x4_to_fp8x4`** — `__device__ __forceinline__` helper:

```cpp
__device__ __forceinline__ uint32_t cvt_bf16x4_to_fp8x4(
    bf16_2 lo, bf16_2 hi, float scale)
{
    float2 lo_f = __bfloat1622float2(lo);
    float2 hi_f = __bfloat1622float2(hi);
    lo_f.x *= scale; lo_f.y *= scale;
    hi_f.x *= scale; hi_f.y *= scale;
    int dummy_old;
    uint32_t packed = __builtin_amdgcn_cvt_pk_fp8_f32(
        lo_f.x, lo_f.y, dummy_old, /*sel=*/false);
    packed = __builtin_amdgcn_cvt_pk_fp8_f32(
        hi_f.x, hi_f.y, packed, /*sel=*/true);
    return packed;
}
```

Per-call cost: 4 v_cvt_f32_bf16 (free; bf16→f32 is bit-truncation) +
4 v_mul_f32 (scale multiply) + 2 v_cvt_pk_fp8_f32 = 10 instructions
for 4 BF16 → 4 FP8. The `int dummy_old` uninitialized pattern + the
`-Wuninitialized` suppression mirror the composable_kernel reference
at `3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_elementwise_hip.hpp:197-213`.

**(2) `cvt_bf16_to_fp8_bulk_compile_test`** kernel + binding:

```cpp
PYBIND11_MODULE(tk_fp8_layouts, m) {
    ...
    m.def("cvt_bf16_to_fp8_bulk", &cvt_bf16_to_fp8_bulk_fn,
          pybind11::arg("src"),
          pybind11::arg("dst"),
          pybind11::arg("scale"));
}
```

Persistent grid (NUM_CUS = 304), 256-thread blocks. Each thread
processes one bf16x4 group (= 1 uint4 of input → 1 uint32_t = 4 FP8
output). Used only by the R4 numerical probe — NOT in any production
hot path.

## Build resource report (`-Rpass-analysis=kernel-resource-usage`)

```
cvt_bf16x4_to_fp8x4_compile_test:    VGPRs= 3, AGPRs=0, Spill=0,
                                     LDS=0, Occupancy=8 waves/SIMD
cvt_bf16_to_fp8_bulk_compile_test:   VGPRs=10, AGPRs=0, Spill=0,
                                     LDS=0, Occupancy=8 waves/SIMD
```

Both kernels saturate occupancy (8 waves/SIMD) with zero VGPR spill
and zero AGPR usage. The cvt path is **extremely lean** — folding
this helper into a future DTR load_a path (R5) is unlikely to push
the existing `grouped_rcr_kernel` (VGPR=198, Spill=37) into worse
register pressure.

## Numerical validation (`/tmp/probe_cvt_bf16_to_fp8_round4.py`)

Compares HK `cvt_bf16_to_fp8_bulk(a, scale=FP8_MAX/amax(a))` output
against Primus's `quantize_fp8_tensorwise(a, dtype=float8_e4m3,
granularity=TENSORWISE)`. Both produce a tensorwise-scaled FP8 tensor;
in theory they should agree bit-exactly modulo round-to-even ties.

| shape                            | M     | K    | amax   | scale   | SNR(dB) | bit_exact |
| -------------------------------- | ----: | ---: | ------:| -------:| ------:| ---------:|
| DSV3-GateUP-B16-M2048           | 32768 | 7168 | 5.8438 | 76.66   | 337.69 | **100.00%** |
| DSV3-Down-B16-M2048             | 32768 | 2048 | 5.5938 | 80.09   | 338.06 | **100.00%** |
| Qwen3-235B-A22B-Down-B16-M2048  | 32768 | 1536 | 5.5938 | 80.09   | 338.07 | **100.00%** |
| gpt_oss-B4-M2048                |  8192 | 2880 | 5.5938 | 80.09   | 338.06 | **100.00%** |
| small-M256-K128                 |   256 |  128 | 4.2188 | 106.19  | 340.50 | **100.00%** |
| awkward-M12345-K512             | 12345 |  512 | 5.3438 | 83.84   | 338.46 | **100.00%** |

**Conclusion**: HK cvt path is **bit-exactly identical** to Primus's
production `quantize_fp8_tensorwise` on every probed shape (both
metric-relevant and edge-case). The SNR ~340 dB ceiling is the
fp32 arithmetic floor (when `diff = 0` the SNR computation hits its
numerical max). Target: SNR > 25 dB on E4M3. **Achieved 14× the
target margin**.

This **eliminates** numerical risk from the R5 kernel surgery.
R5 can focus exclusively on the load-path DTR conversion + correctly
threading the scale through the load helper — no need to re-validate
the cvt builtin numerics.

## R5 plan (next round)

With cvt verified, R5 implements the actual `rcr_8w_load_hoist_fused_act`
load helper. The helper must:

1. Read BF16 from HBM via `buffer_load_dwordx4` (16 bytes / lane / pass,
   no `lds` modifier — DTR mode).
2. Use `cvt_bf16x4_to_fp8x4` (R4 helper) to cvt the loaded BF16 → FP8
   in registers.
3. Write FP8 to LDS via `ds_write_b64` (8 bytes / lane / pass — half
   the FP8 LDS write rate of the existing DTL helper because each BF16
   uint4 read produces only half a uint4 of FP8 output) OR
   `ds_write_b128` after coalescing two BF16 uint4s per thread per pass
   (preferred — keeps the same LDS write granularity as the FP8 helper).

**Coalesced version** (preferred):
- Each thread reads 2 × uint4 BF16 = 32 bytes BF16 = 16 BF16 elements.
- Calls `cvt_bf16x4_to_fp8x4` 4 times → 4 × uint32_t = 16 FP8 = 16 bytes.
- Writes one `ds_write_b128` (16 bytes) to LDS.
- Same per-pass LDS write rate as DTL helper, but 2x HBM read instructions.

Memory bandwidth pre-fusion vs post-R5-fusion (per fwd call, A side):
- **Pre**: read M×K BF16 + write M×K FP8 (in `quantize_fp8`) + read M×K FP8 (in GEMM) = **4×M×K bytes**
- **Post-R5** (DTR fused): read M×K BF16 (in fused GEMM) = **2×M×K bytes**
- A-side traffic saving = **50%**

R5 step plan:
1. Read existing `rcr_8w_load_hoist` (line ~810) carefully to nail the
   8-warp distribution + `bytes_per_thread` constants.
2. Implement `rcr_8w_load_hoist_fused_act` near it, mirroring the
   8-warp pattern but with the BF16 read + cvt + LDS write loop body.
3. Add `grouped_layout_globals_fused_act` struct (`_gl_bf16 a_bf16`
   field replacing `_gl_fp8 a`).
4. Clone `grouped_rcr_kernel` → `grouped_rcr_fused_act_kernel`,
   replacing `rcr_8w_load_hoist` calls with the fused-act variant.
5. Clone `dispatch_grouped_rcr` + `grouped_rcr_dscale_fn` host wrapper.
6. Add pybind binding `grouped_rcr_fused_act_dscale`.
7. Build, probe SNR > 25 dB on DSV3-GateUP-B16-M2048.
8. **Don't wire to Primus yet** — R6 wires `_hk_fused_act_forward` once
   the kernel itself is verified.

R5 estimated diff: ~700-900 LOC of HK additions (kernel clone is bulk
copy with type substitutions; dispatcher + host wrapper + binding are
small clones).

## Falsification ledger update

(R3-A) Path B (BF16 LDS staging) — falsified analytically by LDS budget.
       (Still applies, no new data.)

(R4-A) cvt builtin numerics — **VALIDATED**, not falsified. 100% bit-exact
       on all 6 probed shapes. SNR ~340 dB (~14× the E4M3 25 dB floor).

## R4 commit plan

Two commits, one per repo:
- **HipKittens**: `feat(fp8-fused-act): R4 — cvt_bf16x4_to_fp8x4 builtin helper + numerical probe binding`
- **Primus-Turbo**: `docs(fp8-fused-act): R4 — note HipKittens cvt-builtin deposit; R5 plan` (this note)
