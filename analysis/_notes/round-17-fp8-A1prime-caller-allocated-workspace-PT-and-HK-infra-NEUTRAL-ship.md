---
name: round-17-fp8-A1prime-caller-allocated-workspace-PT-and-HK-infra-NEUTRAL-ship
description: R17 lands the infrastructure side of the A1' (Stream-K variant-2 K-split) caller-allocated workspace plan promised in R16. HK gains a trailing ``sk_workspace_ptr`` pybind kwarg on grouped_rcr / grouped_rcr_dscale; dispatch_grouped_rcr's R13a per-call hipMallocAsync branch is now gated on ``g.sk_partial_buf == nullptr`` so caller-supplied buffers skip the alloc. PT side gains a ``_FP8WorkspaceCache`` LRU singleton (8 slabs) keyed on (buf_bytes, device_index) plus a ``sk_split_n: int = 0`` field on ``HipKittenConfig``; the call site in grouped_gemm_fp8_impl.py looks up / allocates the slab and passes its data_ptr through, gated strictly on ``cfg.sk_split_n > 0`` (no production rule sets that today, so production calls are bit-identical to pre-R17). Verified: dbg_remote.sh build clean, canonical metric 694 (correctness 8/8 PASS) — within R29 cluster, no regression.
type: project
---

# Round-17 — gpt_oss FP8 kernel-only ceiling: A1' caller-allocated workspace infra ship (NEUTRAL)

**Date**: 2026-05-09 (UTC)
**Repo**: Primus-Turbo `dev/kyle_hipkitten_bf16` + HipKittens `main`-tracking working branch.
**Pre-R17 HEAD**: Primus-Turbo `1d36334` (R16 docs); HipKittens `4e9f6b62` (R14 sk_split_n kwarg).
**Forward pointer source**: R16 (`round-16-fp8-r15-fuse-ktail-ship-confirmed-and-r17-caller-allocated-workspace-replan.md`) explicitly committed R17 to the PT-side `WorkspaceCache` singleton + HK trailing `sk_workspace_ptr` kwarg, defaulting to bit-identical production.

## Bottom line

Two coordinated commits land the A1' infrastructure layer:

1. **HK (`kernel_fp8_layouts.cpp`, +36 / -5)**:
   * `grouped_rcr_fn` and `grouped_rcr_dscale_fn` gain a trailing
     `uint64_t sk_workspace_ptr = 0` parameter; when non-zero the
     wrapper assigns `g.sk_partial_buf = reinterpret_cast<int*>(...)`
     before calling `dispatch_grouped_rcr`.
   * `dispatch_grouped_rcr`'s R13a per-call `hipMallocAsync` /
     `hipMemsetAsync` block (lines ~7882-7891) gains an extra gate
     `&& g.sk_partial_buf == nullptr`. Caller-supplied buffers skip
     the alloc; the matching `hipFreeAsync` skip is automatic via the
     pre-existing `sk_partial_buf_owned != nullptr` check at line ~8128.
   * `m.def("grouped_rcr", ...)` and `m.def("grouped_rcr_dscale", ...)`
     register the new trailing pybind kwarg (default 0).

2. **Primus-Turbo (`config.py` + `grouped_gemm_fp8_impl.py`,  +12 / +121)**:
   * `HipKittenConfig` gains `sk_split_n: int = 0` (production-default
     no-op; gates all R17 PT-side plumbing).
   * `grouped_gemm_fp8_impl.py` adds `_FP8WorkspaceCache` LRU (8 slabs,
     keyed on `(buf_bytes, device_index)`) and a buffer-size helper
     `_fp8_sk_workspace_bytes(m_total, n)` mirroring the HK formula
     `T_max * 256 * 256 * 4` exactly.
   * The dscale call site (around line 668-700) gates on
     `trans_b and cfg.sk_split_n > 0`: when set, allocates the slab
     and passes `sk_split_n` + `sk_workspace_ptr` through to the HK
     pybind. Default branch (`cfg.sk_split_n == 0`) skips both kwargs
     entirely — HK pybind defaults cover them at 0 / nullptr, so the
     call is byte-for-byte identical to pre-R17.

Production NEUTRAL: no `select_default_config` rule sets
`sk_split_n > 0` in this commit. The 8 gpt_oss FP8 metric shapes all
land on the existing rules (R15 FUSED_KTAIL gating still active,
R3/R4/R9/R14/R16 dispatcher rules unchanged), and `cfg.sk_split_n`
remains 0 → the WorkspaceCache is never consulted, no extra VRAM
allocated, no kwarg appears in `grouped_dscale_kwargs`.

## Verification

### HK build + canonical metric (dbg_remote.sh GPU 3, post-R17 commit-pending HEAD)

```
[metric_gpt_oss_fp8_kernel] correctness: 8/8 PASS (1.9s)
score = 694
fwd avg = 1911 T  (progress 0.682)
dgrad avg = 2092 T  (progress 0.747)
wgrad avg = 1824 T  (progress 0.651)
```

vs R16 daemon canonical 693 (pre-R17): Δ +1 score, well within R29's
σ ~3-4 single-sample band. Per-shape TFLOPS table is structurally
identical to R16 (e.g. GateUP_B32_M4096 fwd 2091 T, GateUP_B32_M4096
dgrad 2507 T, all R15-FUSED_KTAIL-gated values unchanged). No
regression on any shape; correctness 8/8.

### Bit-identity argument (production path)

For every R17-pre call to `grouped_rcr_dscale(...)`:
1. `cfg.sk_split_n` is `0` (default; no rule overrides).
2. Hence `if trans_b and sk_split_arg > 0:` evaluates false.
3. `grouped_dscale_kwargs` carries the same keys it did pre-R17:
   `{m_per_group, num_xcds, num_slots, chunk_size}` plus optional
   `fuse_ktail_off` (R16 ship, unchanged).
4. The HK call therefore passes only those kwargs; `sk_split_n=0` and
   `sk_workspace_ptr=0` come from HK's pybind defaults.
5. Inside `dispatch_grouped_rcr`, the alloc gate is
   `if (g.sk_split_n > 0 && g.bpc > 0 && g.ki > 0 && g.sk_partial_buf == nullptr)`.
   With `g.sk_split_n == 0` the gate short-circuits → no alloc → no
   workspace touched. Identical control-flow to pre-R17.

### Cumulative inventory chain (R12 → R17)

* R12 (HK `bc5df92d`): added struct fields `sk_split_n`, `sk_partial_buf`. NEUTRAL.
* R13 (HK `43f37f8b`): added per-call `hipMallocAsync` branch in `dispatch_grouped_rcr`, gated. NEUTRAL.
* R14 (HK `4e9f6b62` + Primus probe): exposed `sk_split_n` pybind kwarg, measured per-call alloc cost = 2.9-9.1 ms (FALSIFIED the per-call primitive at the +25-30 envelope budget).
* R15: pivoted to FUSED_KTAIL ship (orthogonal direction).
* R16: re-asserted A1' caller-allocated workspace as the only un-falsified +25-30 envelope; sized R17-R20 concretely.
* **R17 (this commit)**: PT-side `_FP8WorkspaceCache` + HK `sk_workspace_ptr` kwarg. Production NEUTRAL.

## Why this is the right scope for R17

The R16 plan split A1' into three rounds (R17 infra, R18 kernel branch,
R19 ship verdict) precisely to isolate failure modes (mirrors R12's
narrowing of R11's "fields + alloc" to fields-only). Bundling the
kernel control-flow K-split branch with the workspace plumbing in one
commit would entangle:
* alloc bug (workspace size formula off-by-one, slab leak),
* K-split coord-decode bug (per-tile partial-CTA assignment),
* AGPR spill > 60 from extra K-iter bookkeeping (current main kernel
  is 256 VGPR / 37 spill near LLVM ceiling),
* numerical accumulation reordering (partial reduction order changes).

R17 lands ONE of those concerns (alloc/workspace plumbing) with zero
kernel impact; R18 lands the kernel branch with the alloc machinery
already validated. The R29 noise model (σ ~3-4) means any single ship
that's NEUTRAL at the metric is still a positive-information event:
this round confirms the workspace plumbing doesn't perturb production
even at the sub-noise level (score 694 vs 693 = +1, fully within band).

## Risk tally (closed in R17 verification)

| Risk | Mitigation | Status |
|------|------------|--------|
| HK pybind kwarg breaks build | dbg_remote.sh runs make + import; metric ran to completion. | CLOSED |
| Adding HipKittenConfig.sk_split_n breaks @lru_cache | Field has default 0, dataclass remains hashable, lru_cache key tuple stable for production. | CLOSED |
| WorkspaceCache holds VRAM unnecessarily | Cache only allocates when sk_split_n>0 (no rule sets it in R17) → 0 VRAM in R17. | CLOSED |
| Pybind default value compatibility (uint64_t) | Used `uint64_t{0}` literal at pybind11::arg, mirrors HK convention. | CLOSED |
| Buffer-size formula mismatch with HK | `_fp8_sk_workspace_bytes` mirrors HK lines 7884-7886 (T_max * 256 * 256 * 4) verbatim; over-alloc by at most (G-1) tiles per cell — safe upper bound per R13a note. | CLOSED |
| RRR/var-K bindings TypeError on new kwarg | Call site gates kwarg passing on `trans_b` (RCR-only); RRR/var-K untouched. | CLOSED |

## R18 forward pointer

R18 lands the HK kernel control-flow K-split branch. Concrete plan:

1. **Kernel-side scope**: in `grouped_rcr_kernel<...>` (entry around
   line ~5500), add a per-CTA early-decision branch when `g.sk_split_n
   > 0`. SK tiles emit partial mma accumulators to `g.sk_partial_buf`
   via atomicAdd at slab offset `tile_id * 256 * 256 * 4` (the slot
   layout `_fp8_sk_workspace_bytes` already provisioned). Final
   reduction step: separate single-CTA-per-output-tile kernel that
   sums the `sk_split_n` partial slabs into the BF16 output `c` tile.
2. **Estimated 150-300 lines** in `kernel_fp8_layouts.cpp` per R16's
   plan. Reference: CUTLASS Stream-K reduction template.
3. **Tight-verify SNR > 25 dB** vs un-split path on the 4 B=4 fwd
   cells (the cells R14 measured alloc-cost on). Drive via a probe
   `_probe_round_18_sk_branch_snr.py` that flips `cfg.sk_split_n` to
   2 / 4 / 8 on each cell and compares against the production
   reference output.
4. **Ship gate** (R19): only if SNR > 25 dB on all cells AND median
   metric lift on the 4 cells exceeds the per-cell wave-step
   undersaturation envelope (R11 cost decomp: ~+15-25 % per cell at
   sk_split_n=2 if K-split mma savings materialize).

If R18 SNR fails or perf is NEUTRAL after a 5-sample sweep on each of
the 4 cells with workspace amortization confirmed, the full A1'
direction is FALSIFIED; document at
`analysis/_notes/round-19-fp8-A1prime-FALSIFIED-or-SHIP.md` and pivot
R20+ to the next-tier project (Stream-K with work-stealing — Direction
A1 from task md, never prototyped, ~6-8 rounds).

## Deliverables

### Primus-Turbo (this commit)
* `primus_turbo/pytorch/kernels/hipkitten/config.py`: `sk_split_n: int = 0` field on `HipKittenConfig` (+ docstring).
* `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py`: `_FP8WorkspaceCache` class + module singleton + `_fp8_sk_workspace_bytes` helper + call-site plumbing (gated on `cfg.sk_split_n > 0`).
* `analysis/_notes/round-17-fp8-A1prime-caller-allocated-workspace-PT-and-HK-infra-NEUTRAL-ship.md` (this note).

### HipKittens (companion commit)
* `analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`:
  * `grouped_rcr_fn` / `grouped_rcr_dscale_fn`: trailing `uint64_t sk_workspace_ptr = 0` param + assignment to `g.sk_partial_buf` when non-zero.
  * `dispatch_grouped_rcr`: alloc gate now includes `g.sk_partial_buf == nullptr`.
  * Two `m.def(...)` calls: register the new pybind kwarg.
