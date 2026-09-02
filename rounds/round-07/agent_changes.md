# Round 07 (optimize round 7, round 8 overall) — change record

`git` is forbidden this round, so this file stands in for `agent_diff.txt`.

## Files changed (three, one subsystem)

### 1. `primus_turbo/triton/normalization/rmsnorm_kernel.py`
* All eight norm kernels (four forward, four backward) take `LD_CM` / `ST_CM`
  `tl.constexpr` cache-policy strings, applied to the global loads and stores. Default
  `""`, which lowers to exactly the round-7 instruction stream.
* The two residual backward kernels take `DUAL_DX` / `DXR_ptr`: under `DUAL_DX` the
  already-computed `dx` block is stored a second time, to the residual's own buffer,
  while it is still in registers.
* New `dgamma_reduce_kernel`: folds a tall `(n_parts, H)` fp32 partial buffer to
  `(ceil(n_parts / ROWS_PER_PROG), H)`, each program owning a fixed contiguous row range
  stepped in `BLOCK_N` tiles, so the sum order is fixed at any height.

### 2. `primus_turbo/pytorch/kernels/normalization/rmsnorm_impl.py`
* `_FWD_LD_CM = ""`, `_FWD_ST_CM = ".cs"`, `_BWD_LD_CM = ".cg"`, `_BWD_ST_CM = ".cs"`,
  threaded to all eight launches. Raced in the proj unit's own composition, not on the
  kernels alone.
* `rmsnorm_bwd_residual_impl(..., dual_dx=False)` allocates the second output and returns
  `(dx, dx_residual_or_None, dgamma)`.
* `_finalize_dgamma` sends a buffer taller than `_FINALIZE_TRITON_MAX_PARTS` through
  `_narrow_dgamma_partials` instead of falling back to `torch.sum` + a dtype cast.
* `_pick_bwd_config`'s full-wave grid is `_BWD_GRID_MULT * _num_cus()` (2), no longer
  clamped by `_FINALIZE_TRITON_MAX_PARTS` — the fold removed the reason for the clamp.
  On this device both expressions are 512, so the deployed grid is unchanged.

### 3. `primus_turbo/pytorch/ops/normalization.py`
* `_RMSNormResidualFunction.backward` asks for `dual_dx=True` and returns two distinct
  tensors, so autograd no longer clones `dx` for the second consumer.

## Correctness
`_r8_norm.py corr` over five shapes (`32768x2880` bf16, `2097152x64` bf16, `4096x2880`
fp32, `111x50` fp16, `262144x64` bf16): forward, `dx`, the residual `dx`, `dx == dxr`,
and `dgamma` all **bit-identical** to the round-7 build, including on the shapes that
take the fold (`dgamma_rel = 0.000e+00`). `tests/pytorch/ops/test_normalization.py`
138 passed, twice. Bench gate PASS on all four scored runs, `snr_rms [98.73, 100.22]`
and `snr_proj 28.5` unchanged.

## Measurements
Palindrome A/B inside the proj unit (`_r8_norm.py ab`), cumulative from the round-7
build: dual-dx `0.9891`, + dgamma fold `0.9856`, + cache policy `0.9919` of that.
Cache-policy arms, twice, consistent ranking: `f.st+b.both` 0.9919/0.9912, `fb.both`
0.9942, `fb.wt_st` 0.9932, `f.st+b.ldonly` 1.0106.

Scored bench, four clean runs: spd **1.039281 / 1.042931 / 1.040574 / 1.039569**
(mean 1.040589) against round 7's 1.039271 / 1.040679 / 1.039331 / 1.039170 / 1.039388
(mean 1.039568). `ratio_proj` 0.829686 / 0.820441 / 0.821687 / 0.828478 (mean 0.825073)
against round 6's 0.8365-0.8413. A fifth run, discarded: spd 0.938279 with all three
parts degraded together (proj 1.561, perm 1.079, mlp 0.985) — node contention, re-run
per the campaign's own rule.

## Probes
Archived to `rounds/round-07/probes/`, removed from the repo root.
