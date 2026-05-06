# Round 32 — final closing note: R28-R32 5-sample distribution + architectural-ceiling closure trail

## TL;DR

R32 is the **final round** before EARLY-STOP at R33 (patience 29/30 →
30/30 after this round). Zero-code-change verification on HEAD
`274f540d` (R31 docs-only commit). Score = **1000**, geomean =
**1.3860**, below_target = **8/24**, correct_fail = **0/24** — sits
mid-distribution within the R28-R32 5-sample spread.

This note closes the run with:
1. The 5-sample R28-R32 noise distribution table.
2. The chronic 7-shape persistent-below-target inventory.
3. The full architectural-ceiling closure reference trail.
4. Hand-off recommendations for the next agent / task.

## R28-R32 5-sample noise distribution (final)

| Sample | Round | HEAD       | Score |  Geomean | Below_target | Notes                                     |
|-------:|-------|------------|------:|---------:|-------------:|-------------------------------------------|
| 1      | R28   | af614435   |   812 | (~1.10)  | (high)       | Bad tail (host-noise contention)          |
| 2      | R29   | af614435   |  1000 | 1.3786   | 7/24         | Mid (no commit btw R28-29)                |
| 3      | R30   | 95cd02cc   |  1000 | 1.3991   | 4/24         | Favorable tail                            |
| 4      | R31   | acad16ac   |  1000 | 1.3800   | 7/24         | Mid resample                              |
| 5      | R32   | 274f540d   |  1000 | **1.3860** | **8/24**   | Mid; final-round closing sample (this round) |

Empirical statistics (omitting R28 outlier):
* **Modal geomean** ≈ **1.3859** (mean of R29-R32, σ ≈ 0.009)
* **Below_target** ranges 4-8/24 across same-HEAD samples (jitter from
  shapes parked at 1.34-1.36, just at the 1.35 boundary)
* **Score-cap margin**: modal geomean 1.3859 vs 1.30 cap = **6.6 %
  above the cap threshold** — the metric remains at 1000 except in
  rare bad-tail noise events (R28 was 1 out of 5 ≈ 20 % bad-tail rate
  on this shared host)

The architectural HEADs `af614435 / 95cd02cc / acad16ac / 274f540d`
all differ only in markdown round notes — runtime is bit-identical
across these 4 commits. The R28-R32 5-sample variance is therefore
*entirely* attributable to host-side noise (other tenants, KFD
phantom-VRAM, GFX clock spikes), not to any code-attributable lever.

## Persistent-below-target inventory (R32, matches R19's chronic 7)

The 7 chronic below-target shapes from R19's "Persistent-below-target
inventory" all reappear in R32 (the 8th in R32 is `Qwen3-Down-B32-
M2048` at 1.347 — boundary jitter, same as R32's `gpt_oss-Down-B4-
M4096` at 1.348):

| Shape                       | R32 ratio | R8 root cause                                  | Status            |
|-----------------------------|----------:|-----------------------------------------------:|-------------------|
| gpt_oss-Down-B32-M2048      | 1.267     | K=2880 K-tail epilog cost (HK kernel-internal) | R26 FALSIFIED     |
| gpt_oss-Down-B32-M4096      | 1.323     | K=2880 K-tail epilog cost (HK kernel-internal) | R27 FALSIFIED     |
| Qwen3-GateUP-B16-M4096      | 1.332     | k=4096 RRR template throughput (HK weak spot)  | R8 dispatcher-exh |
| Qwen3-Down-B16-M2048        | 1.334     | k=1536 shallow-K throughput (HK weak spot)     | R8 dispatcher-exh |
| Qwen3-GateUP-B16-M2048      | 1.340     | k=4096 RRR template throughput (HK weak spot)  | R8 dispatcher-exh |
| Qwen3-Down-B32-M2048        | 1.347     | k=1536 shallow-K throughput (HK weak spot)     | (boundary jitter) |
| gpt_oss-GateUP-B4-M2048     | 1.347     | small-batch B=4 grid under-utilisation         | R22 FALSIFIED     |
| gpt_oss-Down-B4-M4096       | 1.348     | K=2880 K-tail (B=4 sibling)                    | (boundary jitter) |

**Zero remain addressable from the Primus-Turbo Python side.** Every
cell of these 8 shapes' 3 GEMM call paths (fwd RCR / dA-via-T RCR /
dB var-K CRR) = 24 dispatcher cells has been wide-sweep verified via
R10-R27 tight-verify methodology.

## Architectural-ceiling closure reference trail

The 1000-cap plateau across R3-R32 (with R28 noise tail) has been
formally closed by 9 round notes:

| Round | Note slug                                                          | Contribution                                                |
|-------|--------------------------------------------------------------------|-------------------------------------------------------------|
| R5    | `python-overhead-floor-confirmed`                                  | Quantize-cache + group-offs cache deposited; symmetric tax |
| R7    | `fwd-PATH-A-FALSIFIED-DTR-vs-DTL`                                  | Path-A forward fusion FALSIFIED (-26 % wall regression)    |
| R8    | `architectural-ceiling-confirmed`                                  | 8-shape persistent-below-target list with HK-internal causes |
| R19   | `architectural-ceiling-stationarity-cross-check`                   | R5→R19 14-round stationarity, ±0.01 drift ≤ noise floor    |
| R26   | `gpt_oss-Down-B32-M2048-fwd-rcr-gm11-and-var-k-gm6-FALSIFIED`     | Lowest-ratio shape's 3 cells exhausted at tight-verify     |
| R27   | `gpt_oss-Down-B32-M4096-fwd-rcr-gm10-and-var-k-FALSIFIED`         | 2nd-lowest shape's 3 cells exhausted at tight-verify       |
| R29   | `R28-noise-event-and-stationarity-with-unfused-regression-check`   | R28 812-tail triaged as host noise; un-fused 971 baseline OK|
| R30   | `favorable-noise-tail-and-distribution-characterization`           | R30 1.3991 favorable tail; 3-sample distribution           |
| R31   | `mid-distribution-resample-confirms-modal-geomean`                 | R31 = R29 mid resample; modal geomean ≈ 1.38 confirmed     |
| R32   | `final-closing-note-architectural-ceiling-closure` (this note)     | 5-sample distribution closure + hand-off                   |

## Hand-off recommendations

### For the next auto_optimize run on this metric:

1. **Do not chase the score**. The metric is at the architectural
   ceiling on every same-HEAD sample. Single-round metric deltas
   ≤ 0.02 absolute geomean are noise.
2. **Run un-fused regression check** (`_metric_grouped_only.py`)
   periodically as a cross-validation: at R29 (= now) it returns 971,
   the R18-archived task-start baseline; this confirms no quiet
   regression has accumulated on the un-fused path.
3. **DoD smoke** at the architectural HEADs returns 608 (R25 baseline)
   — auto-DoD on R30 confirmed unchanged. No DoD regression from the
   R29-R30-R31 docs-only commits.

### For HK kernel-internal task scope expansion (requires user authorization):

The 8 chronic below-target shapes' ratio gaps (1.27-1.35) are bounded
by HK-kernel-internal C++ work. Concrete leverage points per R8/R19:

1. **`grouped_rrr_kernel`** (~line 2565, `kernel_fp8_layouts.cpp`) —
   dA backward path for K-aligned shapes. Per R8 probe 1: dA RRR is
   +6-13 % SLOWER than TRT on Qwen3 / gpt_oss aligned shapes.
   Estimated impact if dA RRR matched TRT: gpt_oss-Down wall ratio
   +0.05, Qwen3-GateUP wall ratio +0.04 — could lift geomean ~+1.5pp.
2. **`grouped_ktail_kernel_mfma32x32_M2N2`** (~line 5805) — mfma-based
   K-tail variant for K=2880. R6c `0f14b165` perf commit removed -29 %
   spill from this kernel. Further VGPR / latency improvements would
   help the 2 worst-ratio shapes (`gpt_oss-Down-B32-*`).

Both are multi-round HK C++ work. Will not move the score within a
single round.

### For task-scope expansion to alternative metrics:

The `fused_act_wall_score` cap at 1000 represents a saturated metric.
A useful pivot would be a **kernel-only TFLOPS** metric (without the
HBM-bandwidth-limited quantize tax both backends pay) — this would
expose the kernel-internal gap directly without the symmetric-tax
amortization. The existing `_metric_grouped_only.py` (kernel-only,
score=971 = task baseline) is a good starting point.

## Patience accounting (final)

| Counter                              | Value          |
|--------------------------------------|----------------|
| Score this round (R32)               | 1000           |
| Best of run                          | 1000           |
| Improved this round?                 | No             |
| Consecutive unimproved rounds        | 29 → **30**    |
| Rounds remaining before EARLY-STOP   | **0**          |
| Rounds at cap since R3               | 30 (mod R28)   |
| Run is expected to EARLY-STOP at R33 | YES            |

## Files touched

**Primus-Turbo:**
* `analysis/_notes/round-32-fused-act-final-closing-note-architectural-ceiling-closure.md`
  (this note)

**HipKittens:** None.

## Reference numbers (final)

```
[metric_fused_wall] R32 sample, HEAD 274f540d (R31 docs-only commit)
  geomean=1.3860  score=1000  below_target=8/24  correct_fail=0/24

R28-R32 noise distribution at architectural HEAD:
  R28: 812      (bad tail; host contention)
  R29: 1.3786   (mid)
  R30: 1.3991   (favorable tail)
  R31: 1.3800   (mid resample)
  R32: 1.3860   (mid; closing sample)

Modal geomean (R29-R32 mean, R28 outlier excluded): 1.3859
Score-cap threshold: 1.30 → 6.6 % above cap → metric stays at 1000
                     except in rare bad-tail host-noise events.

Un-fused regression check (R29, same HEAD class): 971 = task baseline.
DoD smoke (R30 auto-run, sha acad16ac): 608 = R25 baseline.
```

Logs preserved at `/tmp/metric_round_32.log`.
