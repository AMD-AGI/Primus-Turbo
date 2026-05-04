# Round 71 — BF16 grouped fwd FUSE-off + zeros-init FALSIFIED (M4 K-tail bug is genuine, not uninit memory)

**Status:** FALSIFIED — zeros init does NOT rescue M4 K-tail; bug is in kernel axis/accum/lane mapping, not in pre-state of C.
**Score:** baseline 875 → with-patch 344 (clean revert → 875).
**No code commits this round;** HK + Primus working trees reverted.

## R71 Hypothesis (from R69 falsification note)

R69 falsified `FUSE-off for g.ki<48` due to 8/8 gpt_oss `fwd-allclose` FAIL
when activating the previously-dead-code `grouped_ktail_kernel_mfma32x32_M4`.
Two candidate root causes:

- **H1 (uninit memory):** M4 K-tail does an RMW (`C += partial_acc`). If the
  Primus dispatcher allocates `torch.empty` for `out`, and if the main
  non-fuse kernel does NOT write every cell in [0, M_total) × [0, g.n), the
  M4 RMW reads garbage for those cells → incorrect final C.
- **H2 (kernel axis/accum bug):** The M4 kernel's SRD construction, lane-to-
  row/col mapping, or accumulator layout is incorrect for forward-pass
  globals. (The kernel was originally designed for dA globals only, per R69
  note.) Main writes every cell correctly, but M4's per-lane writes clobber
  or mis-accumulate into the wrong cells.

## R71 experiment (H1 eliminator)

Paired change:
- HipKittens `kernel_bf16_dynamic.cpp` `fuse_ktail_eligible` gate: `g.ki >= 2`
  → `g.ki >= 48` (same as R69). Routes gpt_oss K=2880 fwd through non-fuse
  main + external M4.
- Primus `grouped_gemm_impl.py` L332: `torch.empty` → `torch.zeros` for the
  HK forward output. **Zero-initialize every cell in `out` before the HK
  kernel is invoked.** If H1 is correct, main writes the cells it actually
  covers + zeros in the uncovered cells → M4 RMW on zero = just the K-tail
  partial acc on uncovered cells → slightly wrong (missing main's K=[0, fast_k)
  contribution on those cells), OR **correct** if main DID cover those cells
  (then zeros is overwritten by main's full K=[0, fast_k) acc + M4's K-tail
  RMW). The key: if H1 was the bug, zeros removes the "garbage" factor.

## Result

- **correct_fail = 8/24** — **same 8 gpt_oss shapes still fail fwd-allclose.**
  H1 is ruled out.
- **score 344** (vs baseline 875). Additional regression on DSV3 (1.122 →
  1.086) and Qwen3 (1.113 → 1.061) from the added `torch.zeros` cudaMemset
  cost on the HK fwd output (~ 100 µs / call on the big shapes) — expected,
  and confirms the zeros patch is NOT a free no-op for non-gpt_oss shapes.

**Conclusion: M4 K-tail bug is NOT a pre-state-of-C issue. The kernel's own
per-lane RMW logic (or the globals it reads) is incorrect for forward-pass
inputs.** Main kernel's `store_c_tile_n_masked` (kernel_bf16_dynamic.cpp
L303-363) DOES cover every cell in [0, M_total) × [0, g.n) via the
predicated per-lane path when `n1 > n_limit`; H1 was a reasonable guess
but was not the bug.

## Pending M4 root-cause candidates (not tried in R71)

- **H2a (row-lane mapping):** M4's `rt_32x16` per-lane row/col decomposition
  may assume dA-style globals (A is the 2nd fwd operand transposed for the
  dA path). Applying it to fwd globals may read A rows with the wrong
  stride.
- **H2b (SRD base for A):** The `make_buffer_resource` for A may use the dA
  batch stride rather than the fwd batch stride. Group offsets computed
  correctly in R41 kernels_grouped_ktail_kernel_mfma32x32_M4 for dA's
  access pattern may not match fwd's.
- **H2c (accumulator polarity):** M4 may write the final accumulated C with
  an implicit `=` (not `+=`) — which would work for dA (first kernel to
  write C) but NOT for fwd (where main's K=[0, fast_k) contribution must
  survive in C).

Fastest discriminator (R72): in a controlled probe, set M4 to ALWAYS use
`C = 0 + tail_acc` (overwrite) and run just the non-fuse + M4 pair for a
single gpt_oss shape downsized to 1 CU. If the fwd result matches
`sum_{K in [fast_k, fast_k+K_REM)} A * B` (pure K-tail partial matmul with
no K=[0, fast_k) contribution), H2c is true — M4 is overwriting, not
accumulating. Then the fix is a trivial `+=` in the M4 store.

## R72 suggestion

Read the M4 kernel store loop (C-write region). Look for `=` vs `+=` on the
final write. If it's `=`, change to `+=` (with a read-before-write of the
bf16 cell). Rebuild. Re-run R69 gate (`g.ki >= 48`). Expected +5-10 score
if H2c was the bug.

Failing that: write a minimal Python probe that calls the non-fuse main +
M4 pair on a tiny shape (B=1, M=128, N=128, K=128) and compare per-cell
output to fp32 reference. The residual pattern (all cells wrong vs. some
cells right) narrows H2a / H2b vs. H2c.
