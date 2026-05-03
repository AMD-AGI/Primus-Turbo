# Round 2 — FP8 grouped fused-act: scaffolding refinement (per-GEMM hooks)

**Status**: SCAFFOLDING REFINED — no perf change (Phase 0 fallback path
remains bit-identical to ``FP8GroupedGemmTensorFunc``). This is
infrastructure work that unblocks shipping Phase 1 (forward fusion)
in a single follow-up round without touching backward, and Phase 2 (dA)
without touching forward, etc.

**Score**: 929 (baseline ~929-931 noise band, 22/24 below target 1.35,
geomean 1.2540, 24/24 correctness PASS, 0 reject). **No metric
movement expected** — all three fused-act helpers still raise
`NotImplementedError`.

## Why this round

R1 (`cf2d5c33`) landed a single `_fused_act_grouped_fp8_forward` helper
that bundled three concerns into one fall-back / fuse switch:
- forward kernel call,
- backward dA reuses the same `a_fp8` materialized inside the helper,
- backward dB always un-fused.

This is too coarse for the **three-phase delivery plan** the task body
requires:
- Phase 1 = forward fused only.
- Phase 2 = forward + dA fused.
- Phase 3 = forward + dA + dB fused (no FP8 staging buffer end-to-end).

With the R1 helper, you cannot ship Phase 1 without also touching
backward semantics (because forward-fused decided to NOT materialize
`a_fp8`, but backward dB still needs it). The save policy was tangled.

R2 splits the helper into three independent hooks, each with its own
try/except guard in the autograd Function. The save policy adapts via
`ctx.fwd_fused`:

| Stage         | fwd fused?   | saved tensors                                     |
| ------------- | ------------ | ------------------------------------------------- |
| Phase 0 (today) | No         | `a_fp8, b_fp8, a_scale_inv, b_scale_inv, ...`     |
| Phase 1       | Yes          | `a (BF16), b_fp8, a_scale_inv, b_scale_inv, ...`  |

Once `_hk_fused_act_forward` lands and starts succeeding, the autograd
Function automatically saves BF16 `a` (no FP8 staging) and backward
either:
- (Phase 1) re-quantizes `a_fp8` lazily inside dB fallback using the
  saved `a_scale_inv` (skips max_abs — cheap), OR
- (Phase 3) re-cvts BF16 `a` directly inside the fused dB var-K kernel
  (no quantize_fp8 launch at all).

## Files touched

- Primus-Turbo `primus_turbo/pytorch/ops/grouped_gemm_fp8.py`:
  - Remove `_fused_act_grouped_fp8_forward`.
  - Add three independent hooks raising `NotImplementedError`:
    - `_hk_fused_act_forward(a, b_fp8, b_scale_inv, ...)` — Phase 1.
    - `_hk_fused_act_backward_dA(grad_out, b_fp8, b_scale_inv, ...)` — Phase 2.
    - `_hk_fused_act_backward_dB(a_bf16, grad_out, a_scale_inv, ...)` — Phase 3.
  - Add `_unfused_forward` / `_unfused_backward_dA_dB` (the bit-identical
    Phase 0 fall-back implementations factored out of the original Func).
  - `FP8GroupedGemmTensorFusedActFunc.forward`: try `_hk_fused_act_forward`
    first; on `NotImplementedError` fall back to `_unfused_forward`. Set
    `ctx.fwd_fused = True/False` and choose the save policy accordingly.
  - `FP8GroupedGemmTensorFusedActFunc.backward`: independent try/except
    around `_hk_fused_act_backward_dA` and `_hk_fused_act_backward_dB`.
    Each falls back independently. The dB fallback materializes whichever
    of `a_fp8` / `grad_out_fp8` is missing (depending on which upstream
    paths succeeded).

- Primus-Turbo `scripts/_metric_grouped_fused_wall.py`:
  - Switch from forward-only timing to forward+backward timing.
    FLOPs now `6 * M * N * K` = 2 fwd + 2 dA + 2 dB. Triggers actual
    backward dispatch (no `requires_grad=False` short-circuit).
  - Default `METRIC_FUSED_WALL_TARGET = 1.35` (was 1.30) — calibrated
    against Phase 0 fwd+bwd geomean ~1.246. Targets Path A's full
    ~12 % wall saving across the entire step. Override via env
    `METRIC_FUSED_WALL_TARGET` if needed during tuning.
  - Module docstring + `_bench_grouped_fp8_fused_wall` docstring rewritten
    to explain why fwd+bwd timing is the right metric for Path A
    (forward-only metric would let the agent stop after ~3 % wall saving
    instead of pushing through the harder dB var-K fusion).

- Primus-Turbo `analysis/_notes/round-2-fused-act-scaffolding-refinement.md`
  (this note).

No HipKittens change this round. The R1 deposit (`max_abs_bf16` +
`max_abs_bf16_to_fp8_scale` bindings, HK commit `4b24c70b`) remains the
prep infra used by Phase 1 / 2 / 3. The next HK change is **R3**:
clone `grouped_rcr_kernel<...>` into `grouped_rcr_fused_act_kernel<...>`
with `_gl_bf16 a_bf16` field + cvt-on-load inside `load_a_tile`.

## Verification

- `python3 scripts/_metric_grouped_fused_wall.py` — score=929,
  correct_fail=0/24, reject=0/24, geomean=1.2540, below_target=22/24.
  All `<135%`-marked rows are between 1.149 and 1.348.
- `_metric_grouped_only.py` regression check: deferred to R5 (every-5
  schedule per task body); v2 scaffolding only touches the
  `fuse_act_quant=True` path which `_metric_grouped_only.py` does NOT
  exercise.

## Round 3 plan

1. Open `HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`,
   inspect `grouped_rcr_kernel<KI_HINT, N_MASKED_STORE, FUSED_KTAIL>`
   (~lines 2317-2978) and the `grouped_layout_globals` struct.
2. Decide clone-vs-template-add (clone is safer, ~700 LOC dup but no
   risk to existing path; template-add is denser but `if constexpr`
   inside `load_a_tile` may interact with KI_HINT specialization).
3. Implement `grouped_rcr_fused_act_kernel<...>` + host wrapper +
   pybind11 binding `grouped_rcr_fused_act_dscale(a_bf16, b_fp8, c,
   a_scale_inv, b_scale_inv, group_offs, ...)`.
4. Wire `_hk_fused_act_forward`:
   - Call `max_abs_bf16_to_fp8_scale(a, scale_buf, done_buf, FP8_MAX)`
     (R1 binding) → `a_scale_inv`.
   - Call new HK binding with BF16 `a` + the `a_scale_inv`.
   - Initially gate to RCR + K%128==0 (16/24 shapes — DSV3 + Qwen).
5. Probe `/tmp/probe_fused_act_round_3.py` — single shape SNR check
   vs `grouped_gemm_ref`. Then metric.

Expected R3 metric: 940-960 (Phase 1 only saves quant(a) HBM traffic
in fwd; backward dA + dB still un-fused, full grad_out quant + var-K
read of `a_fp8`).

## Falsification ledger (R1 still applies, no new entries this round)

- (R1-A) torch.amax(a.abs()) → C++ scale-apply: -150 % vs default.
- (R1-B) Stream-overlap quantize(a) ‖ quantize(b): -0.4 to -3.5 %.
- (R1-C) HK_max_abs + python FP8_MAX/amax + apply: -3 to -63 %.
- (R1-D) HK_max_abs_to_fp8_scale + apply: -2 to -23 %.

The win path (R3+) remains: fuse the BF16→FP8 cvt INTO the GEMM kernel,
not as a separate launch.
