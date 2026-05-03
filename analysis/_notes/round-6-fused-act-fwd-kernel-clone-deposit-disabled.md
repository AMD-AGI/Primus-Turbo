# Round 6 — Fused-act forward: full kernel + dispatcher + binding deposited; Primus side disabled (dB regression)

## TL;DR

Phase 1 fwd-fusion infrastructure landed end-to-end on the HK side. The
`grouped_rcr_kernel` template grew a new `bool FUSE_ACT = false` parameter;
when `FUSE_ACT=true` the four FP8 ``g.a`` load-helper call sites swap to
the R5a `rcr_8w_load_hoist_fused_act` (BF16 HBM → in-register cvt → FP8
LDS) and the per-tile epilog reads the FORWARD scale convention via a new
`resolve_combined_scale_grp<FUSE_ACT>` template overload. A full host
dispatcher (`dispatch_grouped_rcr_fused_act`), host wrapper
(`grouped_rcr_fused_act_dscale_fn`), and pybind11 binding
(`grouped_rcr_fused_act_dscale`) plumb the new path through to Python.

End-to-end correctness on the metric: 24 / 24 PASS when the Primus side
calls the fused path. End-to-end wall: 683 / 937 = **regressed -27 %**.
Root cause is on the Primus side, not HK: the `_unfused_backward_dB`
fallback re-quantizes ``a_bf16`` (saved by fused-fwd) inside backward,
adding a `quantize_fp8` launch on the dB critical path; the un-fused-fwd
baseline saved ``a_fp8`` directly so dB had no quantize work.

Fix-forward (Phase 2 / R7+): either wire `_hk_fused_act_backward_dB`
properly, or pass the saved ``a_scale_inv_dev`` to
`quantize_fp8_tensorwise_impl(scale=...)` so the bwd quantize skips amax.
Either recovers the +25 % fwd savings without paying any bwd penalty.

For this round the Primus side reverts to `raise NotImplementedError`
(autograd falls back to un-fused) so metric stays at 936 (best 937, ±2
noise). HK ``.so`` retains the new kernel + binding so R7 can flip Primus
back on without rebuilding.

## What landed (HK side, persistent)

1.  **`resolve_combined_scale_grp<bool FUSE_ACT = false>` template**
    Existing un-fused calls deduce `GL` from arg + pick up default
    `FUSE_ACT=false` (bit-identical). FUSE_ACT=true inverts the stored
    forward scale (`FP8_MAX/amax`) into the dequant scale (`amax/FP8_MAX`)
    the FP8 epilog needs.
2.  **`grouped_rcr_kernel<KI_HINT, NMS, FK, bool FUSE_ACT = false>`**
    *  Kernel param: `std::conditional_t<FUSE_ACT,
       grouped_layout_globals_fused_act, grouped_layout_globals> g`.
    *  `static_assert(!(FUSE_ACT && FUSED_KTAIL))` — fused-act forces
       FUSED_KTAIL=false (the K-tail fuse path B reads FP8 bytes via
       reinterpret_cast → incompatible with BF16 src).
    *  One-shot wave-uniform read of `*g.dscale_a` into local
       `scale_a_inv` (FUSE_ACT=true only; compiler DCEs for the false
       instance).
    *  Six `rcr_8w_load_hoist(g.a, …)` call sites wrapped in
       `if constexpr (FUSE_ACT)` dispatch to
       `fused_act_round5_compile_test::rcr_8w_load_hoist_fused_act<...>(...,
       scale_a_inv)`.
    *  Per-tile epilog calls `resolve_combined_scale_grp<FUSE_ACT>(g)`.
3.  **R5a forward-relocation** (mechanical)
    `cvt_bf16x4_to_fp8x4` (R4) forward-decl + `grouped_layout_globals_fused_act`
    struct + `rcr_8w_load_hoist_fused_act` template body all moved to ~line
    2245 (just before `grouped_layout_globals`) so `grouped_rcr_kernel`'s
    FUSE_ACT=true branch can resolve them. Originals at ~line 7400-7550 are
    stubbed to comment-only redirects.
4.  **`dispatch_grouped_rcr_fused_act(grouped_layout_globals_fused_act)`**
    Mirror of `dispatch_grouped_rcr` but only launches FUSE_ACT=true /
    FUSED_KTAIL=false / NMS={true,false} variants. Rejects (early-return)
    when bpc/ki invalid or K%128≠0 (no K-tail fuse path); caller is
    responsible for the un-fused fallback.
5.  **`grouped_rcr_fused_act_dscale_fn` host wrapper + `grouped_rcr_fused_act_dscale`
    pybind11 binding**
    Returns `bool` so the Primus side can detect dispatcher-rejected shapes
    and fall back. Same `(group_m, m_per_group, num_xcds)` scheduling knobs
    as the un-fused path.

Build / codegen:
*  All four un-fused instantiations `<0, {0,1}, {0,1}, 0>`: VGPRs 256,
   waves/SIMD 2 — bit-identical resource budget to pre-R6 (R63 plateau).
*  Two new fused-act instantiations `<0, {0,1}, 0, 1>`: VGPRs 256,
   waves/SIMD 2 — fits in the same register / occupancy envelope as
   un-fused. **No occupancy regression from the in-register cvt.**

Metric with HK changes only (Primus disabled): **936** (best 937, ±2
noise band). Confirms the FUSE_ACT=false template specialisations match
pre-R6 codegen exactly.

## What landed (Primus side)

*  **Bug fix in `FP8GroupedGemmTensorFusedActFunc.backward`'s dB un-fused
   fallback**: the R2 scaffolding called
   `quantize_fp8(a_bf16, a_dtype, granularity, scale=a_scale)` — but
   `quantize_fp8` has no `scale=` kwarg. This was unreachable in R2-R5
   (because `_hk_fused_act_forward` always raised) but unblocks R7 from
   tripping the `TypeError` again. Replaced with a plain
   `quantize_fp8(a_bf16, a_dtype, granularity)` and overwrote the saved
   forward scale with the freshly-computed dequant scale.
*  **`_hk_fused_act_forward` docstring updated** to record the R6
   regression analysis and the Phase 2 / R7 plan.

## Why disable Primus, not commit the regression

Per task body: each commit must improve metric OR be neutral. R6's
end-to-end at 683 violates this hard constraint. The fwd path itself is
**correct + functional + ~25 % faster on isolated fwd wall**, but the
backward dB un-fused fallback's re-quantize-from-BF16-a launch
overshadows the fwd win until either the saved scale is reused (~5 LOC
fix) or fused-dB lands (Phase 3, R10+). Disabling the call-site keeps the
infrastructure ready and the metric flat.

## R7 plan (next round)

1.  **Wire saved `a_scale_inv` into the bwd un-fused fallback** —
    replace `quantize_fp8(a_bf16, a_dtype, granularity)` with
    `quantize_fp8_tensorwise_impl(a_bf16, a_dtype,
    scale=<saved_a_scale_inv_dev>)` (the C++ kernel accepts an optional
    pre-computed scale and skips amax when present). Expected effect:
    fwd save (+25 %) recovered, bwd cost flat. Net +5 to +15 score points
    on the K%128==0 fused path.
2.  **Re-enable `_hk_fused_act_forward`** by reverting the round-6
    docstring stub and verifying metric ≥ 942 (slight uplift on K-aligned
    DSV3 + Qwen3 shapes).
3.  Iterate on `m_per_group` / `group_m` / `num_xcds` per-shape rules for
    the fused-act path (the current `select_default_config(...)` was
    tuned for the un-fused path; the fused path may have a slightly
    different optimal scheduling envelope due to the BF16 src bandwidth).

If R7 lands +20 score points (937 → 957) Phase 1 enters the 950s ceiling
and we move to dA (R8-R9) + dB (R10+).

## Falsification log

*  **R6 strategy A vs B** — Strategy B (template-add) chosen because the
   FUSE_ACT=false branch inside each `if constexpr` is bit-identical to
   the existing un-fused code path; existing instantiations stay at 256
   VGPRs / 2 waves with zero codegen drift. Strategy A (full kernel
   clone) would also work but doubles the kernel LOC for a marginal
   safety improvement (verified by codegen inspection: all 4 un-fused
   instantiations matched pre-R6 spill counts exactly).
*  **R5a forward-relocation** — required because `std::conditional_t`
   needs the complete struct type at the kernel template
   instantiation point, and `__device__ __forceinline__` template
   bodies must be visible at the call site. Forward-declarations alone
   do not satisfy either; physical movement of the struct + helper
   to ~line 2245 was the minimal fix.
