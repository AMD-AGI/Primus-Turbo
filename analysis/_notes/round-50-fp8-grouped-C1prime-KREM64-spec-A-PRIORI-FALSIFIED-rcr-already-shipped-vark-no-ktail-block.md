round-50-fp8-grouped-C1prime-KREM64-spec-A-PRIORI-FALSIFIED-rcr-already-shipped-vark-no-ktail-block.md
============================================================================================================

Round: 50 / 100
Date: 2026-05-10
Pre-SHA: b2056b22 (R49 docs commit)
Task: gpt_oss_fp8_kernel_score (8 fp8 shapes, kernel-only TFLOPS)

## TL;DR

R49 (this run) handed off **C-1' per-spec KREM_HINT=64 constexpr template
specialisation** as the next experiment, with two steps:

  1. Add `KREM_HINT` template parameter to ``grouped_rcr_kernel`` and
     constexpr-fold the K-tail per-lane validity masks.
  2. Apply the same to ``grouped_var_k_kernel_fp8``.

**Both halves A-PRIORI FALSIFIED this round** by reading the current
HipKittens HEAD and tracing the wgrad call path:

  * **Step 1 (grouped_rcr_kernel)** — **ALREADY LANDED**. The parallel
    auto_optimize run on the 24-shape bf16+fp8 grouped suite committed
    the constexpr KREM=64 collapse at HK ``6c52d017`` (in our HK
    HEAD's ancestry; verified via ``git merge-base --is-ancestor``).
    Both ``grouped_rcr_kernel`` template-spec sites carry the change
    (lines 3451-3491 and 4273-4311). The change is exactly the form
    R49's forward-pointer specified: replaces dynamic
    ``K_REM = g.k - g.fast_k`` + 2 per-lane masks
    ``b128_lo_valid`` / ``b128_hi_valid`` with a single
    ``constexpr int KREM = 64`` + ``const bool both_valid =
    (laneid < 32)``. Resource-report delta from the parallel R49 doc:
    ``TotalSGPRs 81 → 79`` (-2), V/A/spill/scratch bit-identical,
    binary md5 differs. Net metric Δ = -7 score / -0.016 geomean over
    3 KREM-build runs vs baseline (within ±10 score / ±0.020 geomean
    noise band — refactor flat on perf). Source:
    ``analysis/_notes/round-49-fp8-grouped-KREM-spec-LANDED.md``.

  * **Step 2 (grouped_var_k_kernel_fp8)** — **N/A — no K-tail block
    exists in this kernel**. The var-K wgrad kernel processes the
    contraction axis (M for wgrad CRR) per-group with a runtime K
    that varies group-to-group; it has no analogue of the
    ``FUSED_KTAIL`` if-branch nor a ``K_REM`` constexpr to collapse.
    Verified by ``awk 'NR>=8418 && NR<=8839 && /K_REM|fast_k|b128_lo|
    laneid|FUSED_KTAIL|tail|K_TAIL|k_lane|partial|mask|valid|krem|
    k_rem|K%/{print}'`` over the entire kernel body — zero matches.
    The R49 forward-pointer's "step 2" reflects an incomplete read of
    the var-K kernel structure when the plan was written.

So the R49 plan, taken as written, is **moot**: half is in main and
already empirically sub-noise (parallel R49 measured -7..+0 score
within noise band on the 24-shape suite, and the gpt_oss 8-shape
subset is structurally identical), the other half does not have a
target site to apply to.

## Why the parallel R49 is informative for our task

The parallel run's empirical evidence on the same source file is
directly transferrable: the K-tail block fires once per output tile.
For gpt_oss shapes (tile counts 88-352, output tiles per CU 1-2),
the K-tail block executes ~2-3 times per CU. Even saving 50 cycles
per K-tail call → ~150 cy/CU = ~0.075 µs/CU = ~0.0003 % of kernel
wall — below the 692-696 noise floor of our metric. The fact that
the parallel R49 measured ≈flat after applying the change is fully
consistent with our PMC bottleneck being the **main loop spill**
(37 VGPR words × ~30 cy/scratch-load × 22 K-iters per tile ≈ 24000
cy/tile) — K-tail micro-optimization is structurally unable to move
that needle.

## Falsification register update (gpt_oss FP8 task)

| Lever                                            | Status        | Source              |
|--------------------------------------------------|---------------|---------------------|
| C-1 (restrict / lifetime hints)                  | SATURATED     | R12, R54            |
| C-3 `+a` inline-asm AGPR hint                    | FALSIFIED     | parallel R48 (bf16) |
| C-3 step 1 art_base typedef (R47 plan)           | A-PRIORI FALS | R49                 |
| C-4 `-mllvm -amdgpu-mfma-vgpr-form=0`            | FALSIFIED     | R48 this run        |
| Other 6 amdgpu mllvm flags                       | A-PRIORI FALS | R48 this run        |
| **C-1' KREM=64 collapse on grouped_rcr_kernel**  | **SHIPPED**   | parallel R49 (HK 6c52d017) |
| **C-1' KREM=64 collapse on grouped_var_k_kernel**| **N/A**       | **R50 (this round)** |
| C-2 warp-tile restructure (4w occ=1)             | NOT STARTED   | R51+ pending        |
| K-tail micro-knobs (vmcnt / reorder)             | SATURATED     | R3-R55, R31b        |
| RCR_KTAIL_VMCNT ≠ 8                              | FALSIFIED     | R31b                |
| Direction D step 1 var-K SALU coord-decode       | SHIPPED       | HK b3a5c8db (R9)    |

## R51+ forward-pointer — Stream-K (Direction A1)

The remaining structural levers in the task spec's NEW DIRECTIONS
list are A1 (Stream-K), A3 (decoupled-warps), C-2 (warp-tile to
4w occ=1), and F (larger tiles). Of these, **A1 has the most
infrastructure already in place**:

  * HK ``bc5df92d`` — Stream-K scaffolding fields (``sk_split_n``,
    ``sk_partial_buf``)
  * HK ``43f37f8b`` — Stream-K host-side allocator (gated, NEUTRAL)
  * HK ``4e9f6b62`` — pybind kwarg ``sk_split_n`` for alloc-cost probe
  * HK ``49ffb984`` — caller-allocated workspace via
    ``sk_workspace_ptr`` pybind kwarg

That is, the host plumbing for Stream-K (workspace allocation, CTA
counts, partial-output buffer) is already present and gated off.
What remains is the kernel-side change: when ``sk_split_n > 1``,
have ``grouped_rcr_kernel`` (or a new sibling template spec)
partition the N-reduction axis, accumulate partials to the
``sk_partial_buf`` workspace, and run a small-grid reduction kernel
to fold the partials into final output.

Concrete R51 step (single-round, NEEDS-METRIC):

  1. **Read the existing scaffolding** — confirm ``sk_split_n``
     plumbing reaches ``g.sk_split_n`` inside ``grouped_rcr_kernel``;
     confirm ``g.sk_partial_buf`` is a valid HBM pointer when
     ``sk_split_n > 1``.
  2. **Probe whether enabling sk_split_n=2 on the slowest fwd cell
     (Down_B4_M2048: 1565 T, gap 1235 T to 2800 target) yields any
     ratio at all** — even if ratio < 1 because the kernel currently
     ignores the field, that's a baseline; if ratio = 1 for free,
     the kernel is already partial-Stream-K aware.
  3. **If kernel is unaware**, write a ``_probe_round_51_streamk_*.py``
     that calls the binding with ``sk_split_n=2``, captures the
     output, and SNR-checks against ``sk_split_n=1``. If SNR > 25 dB,
     the kernel is already split-aware (existing scaffolding is more
     advanced than the round comments suggested) — measure TFLOPS.
     If SNR < 25 dB or NaN, the kernel doesn't yet honor the split
     and that's the R52+ work item.

Why this is worth a round given patience 8/40 and round 50/100: the
infrastructure investment from R12-R17 (4 commits) suggests the
parallel/prior session believed Stream-K was the productive
direction but stopped at scaffolding. A 30-minute probe round to
audit whether the scaffolding is functional or stub-only narrows
the next 5-10 rounds of R51+ work (kernel change vs. partial
re-scaffold).

## Why this is NOT a give-up round

The plateau (5+ rounds at 691-693) is real but the patience
streak (8 / 40) and round budget (50 / 100) both leave substantial
runway. The structural levers that *can* break the plateau (Stream-K,
C-2 warp-tile, decoupled-warps) are exactly what the patience budget
exists for. This round formally **retires C-1' from the search
space** (one more structural axis closed: K-tail micro-opt is now
fully exhausted across both rcr and var-K), pointing R51 at the
A1 (Stream-K) direction with a concrete 30-minute probe step that
either unlocks the scaffolding-already-functional fast path or
defines the kernel-edit work for R52-R55.

## What this round changed

* ``analysis/_notes/round-50-fp8-grouped-C1prime-KREM64-spec-A-PRIORI-FALSIFIED-rcr-already-shipped-vark-no-ktail-block.md``
  — this file (Primus-Turbo only; HK working tree clean).
* No code changes in either repo.
* No metric / test edits.

## Scoring expectation

NEUTRAL round, daemon-canonical metric expected within the 691-699
window characterised by R29 (23-sample bit-equivalent baseline). No
code shipping ⇒ score should land near R47/R48/R49 (693/692/691).
This is explicitly a planning + falsification round; the value is
removing C-1' from the search space + pointing R51 at a concrete
single-round audit step on the Stream-K scaffolding (Direction A1).

## Attribution

* HipKittens HEAD: ``49ffb984`` — UNCHANGED this round
* Primus-Turbo: only this doc note
* No ``config.py`` / ``dispatch.py`` / kernel changes
* No metric / test edits
