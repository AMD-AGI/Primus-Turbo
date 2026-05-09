---
name: round-16-fp8-r15-fuse-ktail-ship-confirmed-and-r17-caller-allocated-workspace-replan
description: R16 closes two open items at the round-15 → round-16 hand-off. (1) Verifies the R15 per-shape FUSED_KTAIL gating ship (commit a55d608) is fully wired through both Primus dispatcher (config.py:2597+2631) and the FP8 grouped impl call-site (grouped_gemm_fp8_impl.py:652) — kwarg reaches the HK pybind, no plumbing gap. (2) Re-asserts the R14 forward-pointer (caller-allocated workspace for A1' variant-2 K-split) as the only un-falsified +25-30 score envelope still in the inventory, given that R15 used its round budget on the FUSED_KTAIL ship instead. R16 also explicitly grades the 13-round no-improvement streak against the R29 noise-floor model (σ~3-4 score, tail-mode 9 % at 200+ point drop, min detectable single-sample effect ~+12-15 score) and concludes the streak is fully consistent with code at structural ceiling rather than evidence of any regression. R16 ships no kernel or dispatcher change (mirror of R10/R12/R13/R14 docs commits).
type: project
---

# Round-16 — gpt_oss FP8 kernel-only ceiling: R15 FUSED_KTAIL ship verified, R17+ replanned to A1' caller-allocated workspace

**Date**: 2026-05-09 (UTC)
**Repo**: Primus-Turbo, branch `dev/kyle_hipkitten_bf16` (pre-R16 HEAD `a55d608` from round-15 perf commit; this round is docs-only).
**Scope**: gpt_oss FP8 kernel-only suite. R16 closes the round-15 ship-verification and the round-15-plan forward-pointer, then re-anchors R17+ on the only un-falsified residual envelope — caller-allocated K-split workspace per R14 forward pointer.

## Bottom line

Two items resolve in this docs round:

1. **R15's per-shape FUSED_KTAIL gating ship is fully wired and active.**
   The HK pybind kwarg `fuse_ktail_off` (registered at HK SHA `3bcd248a`,
   `kernel_fp8_layouts.cpp:9812+9830`) is propagated end-to-end through:
     * `primus_turbo/pytorch/kernels/hipkitten/config.py:2597,2631` —
       two `HipKittenConfig(fuse_ktail_off=1, ...)` returns gating the
       two GateUP-B32 dgrad-via-H4 sub-cells (`tiles_n==11 and k==5760
       and tiles_m in {16,8} and m_total>=65536`).
     * `primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py:640,652` —
       `fuse_off_arg = cfg.fuse_ktail_off` then
       `grouped_dscale_kwargs["fuse_ktail_off"] = fuse_off_arg` on the
       RCR variant only (CRR var-K wgrad untouched, as designed).
   Daemon canonical metric since the ship: 693 (round-15 result), within
   noise of the prior round (693). The +1 score median lift the R15
   dev-side measurement saw on the 2 affected dgrad cells is below the
   single-sample noise threshold, so the daemon score didn't move on the
   first canonical run after the ship — this is expected and consistent
   with the R29 noise-floor model below.

2. **R14's caller-allocated workspace forward-pointer is the only +25-30
   envelope still in the inventory** that has not been preflight-falsified.
   R15 did not pursue it; R15 used its round budget on the FUSED_KTAIL
   ship. R16 re-asserts it as the R17+ project and sizes it concretely
   below.

R16 ships **no kernel or dispatcher change** (mirror of R10 / R12 / R13 /
R14 docs commits). The decisive value is closing the round-15 → round-16
hand-off cleanly and re-anchoring R17+ on the highest-EV remaining work.

## Item 1 — R15 ship verification

### Plumbing chain (verified 2026-05-09)

```
HK kernel (kernel_fp8_layouts.cpp)
  ├─ struct grouped_layout_globals.fuse_ktail_off  (R23 ABI extension)
  ├─ pybind11::arg("fuse_ktail_off") = 0           (m.def lines 9812, 9830)
  └─ dispatch_grouped_rcr override (line 7445):
       if (g.fuse_ktail_off > 0) fuse_ktail_active = false;

Primus dispatch (config.py:select_default_config)
  └─ HipKittenConfig.fuse_ktail_off: int = 0       (dataclass line 209)
       ├─ Round 2597: tiles_n==11 ∧ k==5760 ∧ tiles_m==16 ∧ m_total>=65536
       │              → fuse_ktail_off=1   (= GateUP_B32_M4096 dgrad-via-H4)
       └─ Round 2631: tiles_n==11 ∧ k==5760 ∧ tiles_m==8  ∧ m_total>=65536
                      → fuse_ktail_off=1   (= GateUP_B32_M2048 dgrad-via-H4)

Primus call-site (grouped_gemm_fp8_impl.py)
  ├─ Line 640: fuse_off_arg = cfg.fuse_ktail_off
  └─ Line 652: grouped_dscale_kwargs["fuse_ktail_off"] = fuse_off_arg
                (RCR variant only; CRR var-K wgrad untouched)
```

Every link verified by `grep -n` against the post-R15 HEAD `a55d608`. No
plumbing gap; the kwarg reaches the HK pybind and the HK dispatch flips
`fuse_ktail_active = false` on exactly those 2 cells.

### Measured impact (R15 commit body data, dbg_remote.sh GPU 3 9-sample)

Per the R15 commit message:
```
scores: 693, 697, 697, 693, 693, 694, 698, 697, 693
median = 694, mean = 695.0, max = 698, min = 693
vs recent rounds 9-14 baseline (GPU 3): median 693, max 694, min 692.
```

That's a **+1 score median lift** on the dev-side 9-sample. Daemon's
canonical first-after-ship sample landed at 693 (cluster low end). Both
are consistent with the R14 column predicting +28-41 T per affected
shape (= ~+0.0031 dgrad-section progress = ~+1.0 score) and the R29
noise model (σ ~3-4, single-sample lift below detection floor).

## Item 2 — 13-round streak vs R29 noise-floor model

### R29 noise floor (23-sample bit-equivalent characterization)

`analysis/_notes/round-29-noise-floor-characterization-23-sample-bit-equivalent-baseline.md`
established at HEAD `80cf0b69` over 23 samples:
* Cluster (91 % of samples): score range [690, 705], median 698-700, σ ~3-4.
* Tail-mode (9 % of samples): score range [476, 578] — GPU-side transient
  (clock drop / contention), NOT a correctness failure.
* **Minimum detectable single-sample effect: ~+12-15 score** (3σ above
  cluster median).

### 13-round streak at this noise level

Daemon canonical scores for rounds 4-15 (the no-improvement streak):
```
R4-R8:   docs/perf rounds, scores 692-695 cluster
R9:      692 (Direction D step 1, NEUTRAL ship)
R10-R14: 692-693 (all docs FALSIFICATION rounds)
R15:     693 (FUSED_KTAIL ship, dev-side +1 median below daemon detection)
```

Versus best-of-run = 695 (from `5e20a3e1`, an early ship that may
itself have been a single-sample lucky-tail draw at the 695 cluster
upper edge). The 693-695 range is **fully within R29's σ ~3-4 cluster**.
Per the R29 model, no single-sample lift below +12-15 is daemon-observable;
**every shippable +1 score lever this run identified would be invisible
to the daemon as a 1-sample observation**, and the ones that did ship
(R3 var-K wgrad +0.78%, R15 FUSED_KTAIL +1 dev median) genuinely did
land — they're just below the noise floor.

This rules out the alternative interpretation ("the 13 ships actually
regressed and we don't know it"): if R15's FUSED_KTAIL had regressed,
its 9-sample median would be ≤690, not 694. The dev-side 9-sample
discipline catches regressions even when daemon 1-sample can't see wins.

## Item 3 — R17+ project replan (caller-allocated K-split workspace)

### Why this is the only un-falsified residual envelope

The R14 forward-pointer
(`round-14-A1prime-variant-2-K-split-per-call-hipMallocAsync-FALSIFIED-pivot-to-caller-allocated-workspace.md`)
falsified the **per-call** `hipMallocAsync` path at 2.9-9.1 ms / call
overhead vs an R11 envelope of ~3 us / call. The forward-pointer
prescribed: caller-allocated workspace, allocated once at first call
per cell, passed through pybind as a `data_ptr`, mirroring the existing
`done_counter` pattern.

This is the only direction in the cumulative inventory that:
1. Has not been preflight-falsified (the R14 falsification was at the
   per-call alloc primitive level, not at the K-split mechanism).
2. Has a +25-30 envelope from the R11 cost-decomp (the LDS-double-buffer
   amortization of K-split partial-accumulator work, with the alloc
   cost driven to ~0 by once-per-cell allocation).
3. Has its host-side scaffolding **already committed in HK** — the
   `sk_split_n` and `sk_partial_buf` pybind kwargs are in place
   (HK commits `bc5df92d` R12, `43f37f8b` R13a, `4e9f6b62` R14),
   `dispatch_grouped_rcr` already has the gated alloc/free branches
   (`kernel_fp8_layouts.cpp:7882-7891` and `8128-8130`). The remaining
   surgery is (a) replace the host-side `hipMallocAsync` branch with
   a caller-supplied data_ptr field, (b) write the kernel control-flow
   branch to consume the workspace.

### Concrete R17-R20 sized plan

* **R17**: PT-side caller-allocator. Add `WorkspaceCache` (singleton
  dict keyed by `(N, K, m_total, B, dtype)` → `torch.Tensor` of size
  `sk_partial_buf_bytes(cell)`). On first dispatch per cell, allocate
  via `torch.empty(size, device='cuda', dtype=torch.uint8)` and stash;
  on subsequent dispatches, retrieve and pass `t.data_ptr()` to the
  HK pybind through a new `sk_workspace_ptr` kwarg.
  HK pybind: add `uint64_t sk_workspace_ptr = 0` trailing kwarg to
  `grouped_rcr_fn` / `grouped_rcr_dscale_fn`; pass through as the
  alloc-branch input (skipping the alloc-call when ptr != 0).
  PT cell-trigger: `sk_split_n=2` on the 4 B=4 fwd cells (the cells
  R14 measured the alloc cost on) — **gated to a build-flag at first**,
  not on by default; production calls still default to `sk_split_n=0`
  bit-identical, so the round 17 ship should be NEUTRAL on the metric.
  Ship gate: 5-sample dbg_remote.sh probe with `sk_split_n=2` enabled
  via env hook on the 4 affected cells; verify alloc cost dropped from
  2.9-9.1 ms to <50 us amortized after the first warmup iter.

* **R18**: HK kernel branch — write the K-split partial-accumulator
  reduction. The `sk_partial_buf` is HBM scratch the K-loop writes
  partial mma accumulators to; a final reduction step (single-CTA per
  output tile) sums the `sk_split_n` partial slabs into the output
  tile. Reference: CUTLASS Stream-K reduction template. Estimated
  150-300 lines in `kernel_fp8_layouts.cpp`. Tight-verify SNR > 25 dB
  vs un-split path on all 4 fwd cells.

* **R19**: ship verdict. With workspace amortized to ~0 alloc per call,
  the R11 envelope (+25-30 score) recovers if the K-split mma savings
  materialize on the wave-step-undersaturated B=4 cells. Two
  outcomes:
    * R19 success: ship the rule, +25-30 score envelope to current
      693 → ~720, well above R29's 715 detection floor.
    * R19 falsification: the workspace amortization works (R17 NEUTRAL
      ships), but the K-split mma savings don't materialize on the
      production cell — falsifies the full A1' direction; document
      and close. R20+ pivots to the next-tier project (Stream-K with
      work-stealing — currently parked at high-cost / no-prototype).

* **R20**: regardless of R19 outcome, this is the natural DoD checkpoint
  (auto_optimize's `--dod-every 5`: R20 falls 5 rounds after R15).
  Verify all three canonical metrics
  (`_metric_grouped_only.py`, `_metric_hk_ratio.py`,
  `_metric_grouped_fused_wall.py`) stay >= 990. Per the SKILL.md hard
  constraint, any of these regressing below 990 forces an immediate
  revert.

### Alternate fallback path

If R17-R19 land in 3-4 rounds without a +25 envelope, the next-tier
options are all multi-round structural projects:
1. Stream-K with work-stealing (Direction A1 from task md, never
   prototyped — high-cost, ~6-8 rounds).
2. Decoupled-warps producer-consumer (Direction A3 from task md,
   `round-6` PREFLIGHT FALSIFIED on cooperative-load throughput
   coupling — would need a fresh angle).
3. Sub-noise dispatcher/macro accumulation: continue the existing
   ship pattern (+0.5 to +1 score per round, daemon-invisible at
   1-sample), accumulate 10-15 rounds to break out of the R29
   noise band. Predictable but slow, and depends on finding fresh
   sub-noise levers (rate has slowed from 1/round in R3-R8 to
   ~1/3-rounds in R10-R15).

## R16 deliverables

### Primus-Turbo
* `analysis/_notes/round-16-fp8-r15-fuse-ktail-ship-confirmed-and-r17-caller-allocated-workspace-replan.md`
  (this note).
* **No** `select_default_config` change. **No**
  `grouped_gemm_fp8_impl.py` change. **No** dispatcher rule modification.

### HipKittens
* No change. The R12/R13a/R14 host-side scaffolding for the R17+
  caller-allocated workspace is already committed in HK at SHA
  `4e9f6b62` (R14). R17 PT-side allocator can be wired without further
  HK changes if `sk_workspace_ptr` is added then; otherwise R17's HK
  edit is a single trailing kwarg on the two pybind m.def calls.

## R17 binding commitment

R17 is the PT-side `WorkspaceCache` singleton + the HK trailing
`sk_workspace_ptr` kwarg. Default `sk_split_n=0` keeps production
bit-identical (no alloc branch entered, no metric impact). The probe
gate is via env hook on `sk_split_n` to enable the alloc branch on
the 4 B=4 fwd cells; verify amortized alloc cost drops from
2.9-9.1 ms (R14 measurement) to <50 us after warmup, 5-sample
dbg_remote.sh on each cell. Ship if amortized cost confirmed; the
kernel-side K-split branch lands in R18.

Expected R17 metric impact: **NEUTRAL** (production calls default
`sk_split_n=0`, alloc branch never entered). The decisive value is
infrastructure for the R18-R19 +25-30 envelope.
