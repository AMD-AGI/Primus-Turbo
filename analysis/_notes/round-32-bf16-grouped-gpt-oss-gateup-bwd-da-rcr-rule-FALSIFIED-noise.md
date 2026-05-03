# Round 32 — gpt_oss-GateUP bwd dA RCR dispatch rule FALSIFIED (paired 5-run mean -1.4)

## Goal

R31 established that for gpt_oss, H4 is a mandatory fast-path
switch — removing it drops GateUP ratio from 1.05 to 0.67
(native RRR N-tail is 0.6× the RCR fuse throughput on K_fwd=2880).
R31 note recommended R32 pivot to HK kernel C++ work to fix the
native-RRR N-tail slow path (multi-round).

With only 32 min remaining in the chat window, the C++ path was
infeasible. R32 instead targeted the **post-H4 RCR cfg selection**
on the 4 gpt_oss-GateUP shapes (the lowest-progress family in the
metric at ratio 1.049-1.109). After H4 transposes b
[B, N_fwd=5760, K_fwd=2880] → [B, K_fwd=2880, N_fwd=5760], the
bwd dA kernel runs with shape (m=2048 or 4096, n=2880, k=5760).

Currently falls to default (gm=4, xcds=8). R10 / R26 precedent
for small-tiles_m + shallow-tiles_n + mid-K RCR is (gm=1, xcds=4).

## Change probed

Added to `primus_turbo/pytorch/kernels/hipkitten/config.py`:

```python
if (
    layout == "rcr"
    and tiles_n == 12
    and k == 5760
    and m_total is not None
):
    return HipKittenConfig(layout=layout, group_m=1, num_xcds=4, kernel=None)
```

Scope: `tiles_n==12 ⇔ N_kernel ∈ [2816, 3071]` + `k==5760` uniquely
captures gpt_oss-GateUP bwd dA. Qwen3-GateUP fwd has tiles_n==12
but k=4096, not 5760. No other metric-grouped or dense family hits
this predicate.

## Correctness verification

Downsized allclose on 9 canonical shapes: 9/9 PASS (group_m / num_xcds
are scheduling knobs only, bit-identical output).

## Metric verification — paired 5-run

```
baseline: 880, 879, 886, 890, 880 → mean 883.0 (range 879-890, σ ~4.4)
applied:  879, 878, 885, 884, 882 → mean 881.6 (range 878-885, σ ~2.9)
Δ = -1.4
```

Sub-noise, slightly negative. Not a landable change. Single-run
pre/post verify (879 applied vs 884 baseline) had suggested the
same direction but with noise swamping the signal.

## Why (gm=1, xcds=4) didn't win this geometry

The R10 precedent targets dense-LLaMA-like M_per_group=2048 with
narrow-K (≤ 7168). This bwd dA geometry has k=5760 (= N_fwd,
"wide-K" by the dispatch table's standards) AND tiles_n=12 which
is narrower than R10's tiles_n=16. The combination may have
different L2-residence characteristics (B-tile is 2880×5760
bf16 = 33 MB per group vs 2048×7168 = 29 MB — similar rough size
but different aspect ratio) that make the (gm=1, xcds=4) cfg's
"all-B-columns-one-M-row" walk less efficient than default's
(gm=4, xcds=8) batch-4-M-tiles pattern.

A full per-cell sweep probe crashed with a GPU memory fault during
setup (known HK BF16 cold-start issue); time budget didn't allow
debugging a new probe harness. Filed as backlog for R33+.

## Decision

**Revert. Documentation-only commit.** The single-rule (gm=1, xcds=4)
for tiles_n==12 ∧ k==5760 is now characterized as -1.4 paired-5,
not landable alone.

## Compounded R31 + R32 insights

This round further consolidates R29's conclusion that the BF16
dispatch surface is exhausted:

  * R24 (LANDED, +5.4): 4-rule dB var-K aggregate on fresh default cells.
  * R25-R27 (FALSIFIED): incremental single-family rules (fwd RCR, dA
    RRR, dB var-K) all sub-noise alone.
  * R28 (FALSIFIED): bf16_transpose_3d block-shape BF16-tune +3.4 alone.
  * R29 (FALSIFIED): 5-rule bundle net -0.2.
  * R30 (DOC): H4 wall profile characterized.
  * R31 (FALSIFIED, -79): H4 gate tighten exposed native-RRR N-tail is
    0.6× RCR fuse throughput; H4 is a mandatory fast-path switch,
    not an overhead.
  * R32 (FALSIFIED, -1.4): post-H4 RCR (gm=1, xcds=4) for tiles_n==12
    ∧ k==5760 is sub-noise.

**Score plateau 878-892** is a hard ceiling given the current HK
kernel binary. The remaining lever is HK C++ kernel work on the
native-RRR N-tail handler (R31's lever B, 80 % lever per R31
estimate).

## Files

* `analysis/_notes/round-32-bf16-grouped-gpt-oss-gateup-bwd-da-rcr-rule-FALSIFIED-noise.md` — this note.
* `/tmp/probe_round32_gpt_oss_gateup_bwd_da.py` — crashed setup
  probe (not committed).
* No production change (`config.py` reverted to HEAD d9c1620).

## Suggested R33 next step

Chat window will almost certainly expire before R33 starts. On
resume, the new chat will see commit log up to R32 and must infer
current state from notes.

**R33 main line** (C++ kernel work, high upside, multi-round):

1. Read `HipKittens/analysis/bf16_gemm/mi350x/kernel_bf16_dynamic.cpp`
   to identify the `grouped_ntail` launcher code path (per R31
   note) for native RRR with N_kernel % 256 != 0. The R31 metric
   data showed this path runs at 0.6× the RCR fuse throughput on
   K_fwd=2880 shapes.
2. Compare to the `grouped_ntail` RCR launcher (the fast one). The
   RRR and RCR tail kernels share code via `template<Layout L>`
   (kernel_bf16_dynamic.cpp lines ~1200-1500 area per the
   layout-dispatched template); the difference is in the
   per-layout LDS subtile shape and the B-tile load pattern.
3. Hypothesize the slow-path root cause (likely: LDS bank conflicts
   or uncoalesced HBM load due to row-major B in RRR N-tail vs
   col-major B in RCR N-tail).
4. Profile with rocprofv3 on a minimal (`bench_bf16_vs_torch.py`
   style) harness.

**R33 alt** (lower risk, smaller upside, ~+1-2 score):
try CUDA graph capture around the bwd dA H4 + grouped_rcr call
to amortize Python dispatch overhead. But this crosses the FROZEN
line on "no caching across forward calls" since CUDA graph
captures are de facto per-shape caches.

**R33 bottom line**: accept 880-892 plateau for the 24-shape wall
metric. Pivot to DoD failed=608 convergence (orthogonal to metric
but reduces cross-backend risk) or kernel C++ work.
