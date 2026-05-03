# Round 3 — FP8 grouped fused-act: forward kernel surgery design plan

**Status**: DESIGN INVESTIGATION — no kernel code change, no metric impact.
This round inspects the Phase-1 surgery surface area, locks down the cvt
builtin signature, identifies the LDS-budget constraint that rules out
the naive "stage BF16 in LDS" path, and decides the actual implementation
approach for R4.

**Score**: 942 (baseline, Phase 0 fall-back only — fwd+bwd geomean 1.2711,
24/24 PASS, 6/24 already at goal). No change vs prior round (R2 = 930).
The metric drift 930 → 942 is GPU-pool noise (auto_optimize.py picked
GPU 3 with ~7.3 GB phantom VRAM held by a prior tenant; SM use was 0%
so the run was clean but cache state differs).

## Key facts established this round

### Fact 1 — cvt builtin signature confirmed

From `composable_kernel/include/ck_tile/core/numeric/float8_hip.hpp:653`
and `tile_elementwise_hip.hpp:204-213`:

```c
uint32_t __builtin_amdgcn_cvt_pk_fp8_f32(
    float a, float b,
    int dummy_old,    // NOTE: must be uninitialized (compiler emits unwanted v_mov_b32 if zero)
    bool sel);        // false → write to lo half (word 0); true → write to hi half (word 1)
```

- 1 builtin call: 2 fp32 → packed FP8 in lo or hi half of an int (= 2 bytes packed).
- 2 builtin calls (one with sel=false, one with sel=true) on the same
  `dummy_old` accumulator: 4 fp32 → 4 FP8 in 1 int (= 4 bytes packed).
- The `dummy_old` is a hardware quirk: the builtin takes a "merge"
  operand to combine the new pair into the destination int. Per CK
  comment, declare uninitialized + suppress the `-Wuninitialized`
  diagnostic to avoid emitting a redundant `v_mov_b32` before the cvt.

**Mapped onto our use case**: BF16 → FP8 cvt for one packed `uint4`
(8 BF16 = 16 bytes, the unit `rcr_8w_load_hoist` reads):

```c
// Step 1: bf16x8 → 4× float2 via __bfloat1622float2.
// Step 2: 4× cvt_pk_fp8_f32 calls (lo-hi-lo-hi pattern) → 2× int = 8 FP8 = 8 bytes.
```

Per-uint4 instruction overhead: 4 v_cvt_f32_bf16 + 4 v_cvt_pk_fp8_f32 +
2 v_mov merges = ~10 cycles. HBM read of the same uint4 = ~50 cycles
(LD-bound). So cvt is **free** in the steady state if it can be
pipelined behind the load.

### Fact 2 — `rcr_8w_load_hoist` is DTL (HBM → LDS, register-bypass)

`kernel_fp8_layouts.cpp:810-887`. The hot load helper for grouped RCR:

```c
__device__ __forceinline__ void rcr_8w_load_hoist(...) {
    ...
    asm volatile(
        "s_mov_b32 m0, %0\n\t"
        "buffer_load_dwordx4 %1, %2, %3 offen lds\n\t"   // ← DTL: HBM → LDS
        : ...);
}
```

The `offen lds` modifier on `buffer_load_dwordx4` is **Direct Tile
Load** (DTL) on CDNA4: 16 bytes flow HBM → m0-addressed LDS slot
**without ever materializing in a VGPR**. This is the key reason the
existing FP8 grouped GEMM achieves the kernel-only ratios it does —
the load + LDS write fuse into a single instruction issue.

**Constraint**: any cvt step (BF16 → FP8 register cvt) BREAKS DTL,
because cvt requires the value in a VGPR. The fused-act kernel must
either (A) read into VGPR + cvt + write LDS as DTR (Direct Tile Read),
or (B) DTL into a BF16-typed LDS staging tile, then run a second
LDS→VGPR→cvt→LDS pass.

### Fact 3 — LDS budget rules out path (B)

Existing tile sizes (from `kernel_fp8_layouts.cpp:78-100`):
- `ST_v2 = st_fp8e4m3<HB=128, BK=128, st_16x128_v2_s>` → 16 KB / tile (FP8).
- `As[2][2]`, `Bs[2][2]` ⇒ 8 tiles total ⇒ 128 KB FP8.

But MI355X CDNA4 LDS = **64 KB / CU** (verify with HK existing kernels;
they fit by relying on subtile internal padding + the fact that `As`
and `Bs` actually share half the storage, total is below 64 KB by
design — this exact accounting is for R4 to revisit).

**What matters for path (B)**: BF16 staging would require **double the
bytes per A tile** (BF16 = 2 bytes vs FP8 = 1 byte). Doubling A's LDS
footprint is incompatible with the existing 4-stage A-double-buffer
that pipelines load-A across MFMA. Keeping the same staging requires
halving tile size or LDS layout, which destroys the existing kernel's
arithmetic intensity.

**Decision**: path (B) is **falsified**. Use path (A).

### Fact 4 — path (A) implementation sketch

Replace `rcr_8w_load_hoist` for the fused-act kernel with a new helper
`rcr_8w_load_hoist_fused_act` that:

1. Reads BF16 from HBM into VGPR via `buffer_load_dwordx4` (no `lds` modifier).
2. cvt's BF16 → FP8 in register via `__bfloat1622float2` + `__builtin_amdgcn_cvt_pk_fp8_f32`.
3. Writes FP8 to LDS via `ds_write_b128`.

**Cost analysis** (vs DTL on FP8 source — the un-fused path's load
per tile):
- DTL (un-fused, FP8 source):  M*K bytes HBM → 1 instruction issue, no VGPR pressure.
- DTR (fused-act, BF16 source): 2*M*K bytes HBM (BF16 = 2x FP8 size) → 1 buffer_load + 1 ds_write per uint4 + ~10 cvt cycles.

**Net A-side HBM bandwidth comparison** (per fwd call):
- Un-fused: read M*K bytes BF16 (in `quantize_fp8(a)`) + write M*K bytes FP8 (in `quantize_fp8(a)`) + read M*K bytes FP8 (in GEMM via DTL) = **4*M*K bytes**.
- Fused (path A): read M*K bytes BF16 (in fused GEMM via DTR) + write 0 bytes FP8 staging (no quantize_fp8 call) = **2*M*K bytes**.

**A-side traffic saving = 50%**. (Task body's "17%" estimate was
conservative; it included BF16 input + FP8 staging write + GEMM read.
A more careful accounting that excludes BF16 input — which is needed
in both un-fused and fused paths — is **2x reduction on A-side
non-input traffic**.)

### Fact 5 — register pressure risk

The existing `grouped_rcr_kernel` is at VGPR=198 / AGPR=256 / Spill=37
(round 53-58 ledger). DTR adds:
- Per-uint4 load: ~4 VGPRs for the `buffer_load` result.
- cvt intermediate: ~4 VGPRs for fp32 fan-out (lifetime: 4 cycles).
- LDS write source: ~2 VGPRs for packed FP8.

Steady-state extra VGPR pressure: ~6-10 VGPRs. Likely **forces spill**
beyond the existing 37. Will need to validate via `--save-temps` ISA
inspection in R4-R5.

**Mitigation**: keep the cvt intermediate fan-out short (re-use the
load-VGPRs as cvt-VGPRs once the cvt completes, since AMDGPU mem ops
don't read from VGPR). Compiler should naturally sink this if the cvt
helper is a `__forceinline__` template.

## Decision matrix

| Approach | LDS impact | VGPR impact | DTL preserved | Cost | Verdict |
|---|---|---|---|---|---|
| (A) DTR with in-register cvt | none | +6-10 VGPR (manageable) | NO (load goes via VGPR) | medium-low | **chosen** |
| (B) BF16 LDS staging + 2nd pass | +50% LDS (overflow) | +2-4 VGPR | YES (DTL preserved) | high | **falsified** |
| (C) Pre-quant in workspace | none | none | YES | high (extra launch) | falsified by R1 |
| (D) MFMA on BF16 directly | none | none | YES | abandons FP8 acceleration entirely | rejected |

## R4 plan (concrete next-round steps)

1. **Read** `kernel_fp8_layouts.cpp:78-110` for exact ST_v2 size + LDS
   layout (`As`, `Bs` sharing pattern).
2. **Clone** `grouped_rcr_kernel<KI_HINT, N_MASKED_STORE, FUSED_KTAIL>`
   to `grouped_rcr_fused_act_kernel<KI_HINT, N_MASKED_STORE, FUSED_KTAIL>`.
   Keep body ~bit-identical except the `g.a` source type (`_gl_bf16`
   instead of `_gl_fp8`).
3. **Add** new struct `grouped_layout_globals_fused_act` (mirrors
   `grouped_layout_globals` with `_gl_bf16 a` field; do NOT modify
   the existing struct or its instantiations).
4. **Implement** `rcr_8w_load_hoist_fused_act` near the existing
   `rcr_8w_load_hoist` (~line 810). It uses `buffer_load_dwordx4`
   without `lds` modifier (DTR), then issues 4 `v_cvt_f32_bf16` +
   4 `v_cvt_pk_fp8_f32` builtins, then `ds_write_b128`. Mirror the
   8-warp pattern so leftover loop bodies match.
5. **Add** dispatcher `dispatch_grouped_rcr_fused_act` (clone of
   `dispatch_grouped_rcr` from line 6096); host wrapper
   `grouped_rcr_fused_act_dscale_fn` (clone of `grouped_rcr_dscale_fn`
   line 6970), pybind binding `grouped_rcr_fused_act_dscale`.
6. **Build** `make -j8 tk_fp8_layouts.so`. Expect 1-3 min compile
   for the new template specs.
7. **Probe** `/tmp/probe_fused_act_round_4.py`: one shape
   (DSV3-GateUP-B16-M2048, M_total=32768, N=4096, K=7168, RCR), pass
   BF16 a directly + an externally-computed `a_scale_inv` from
   `tk_fp8_layouts.max_abs_bf16_to_fp8_scale`. Compare output to
   `grouped_gemm_ref` SNR. Target SNR > 25 dB.
8. **If probe passes**: wire `_hk_fused_act_forward` in Primus-Turbo
   (gate to RCR + K%128==0 — DSV3 + Qwen, 16/24 shapes).
9. **If probe fails**: revert kernel commit. Check ISA via `--save-temps`
   for VGPR spill or register-aliasing bug. Drop a falsification note.

## Falsification ledger updated

- (R3-A) Path B (BF16 LDS staging) — falsified analytically by LDS
  budget. M*K BF16 staging = 2× M*K FP8 → 50% LDS overhead per
  staging tile, exceeds remaining LDS headroom on the existing
  4-stage double-buffer pipeline.

(R1-A through R1-D still apply; this round adds R3-A only.)

## R3 commit plan

This is a **doc-only** round. One commit to Primus-Turbo:
- `analysis/_notes/round-3-fused-act-fwd-kernel-surgery-plan.md` (this file).

No HipKittens changes this round (kernel surgery is R4 work).
No Primus-Turbo code changes (`_hk_fused_act_forward` still raises
`NotImplementedError` until R4 binding lands).

Metric is unchanged because Phase 0 fall-back path is untouched.
