# Round 16 — FP8 grouped: noise-band confirmation + R17 stage-level pipelining proposal

**Status**: NO KERNEL CHANGE — R16 = 5-trial baseline confirmation that
forward path remains at the architectural plateau established at R5/R11,
plus a concrete (untried) R17 candidate: stage-level software pipelining
(pre-issue K-tail HBM loads inside epilog 2).

**Auto-optimize round**: 16 / 100
**Date**: 2026-05-02
**HK SHA at round start / end**: `0f14b165` (unchanged — last HK commit was
R14 `grouped_var_k_kernel_fp8` epilog cleanup, which is **backward-only**
and does NOT touch `grouped_rcr_kernel` forward binary)
**PT SHA at round start**: `97cbda86`
**Reported best**: 966 (R15)
**R16 5-trial median**: **962** (range 960-964)

---

## 5-trial baseline (R16 lock-in)

```
trial 1: score=961  grp_FP8=1.1218  grp_BF16=1.1856
trial 2: score=960  grp_FP8=1.1176  grp_BF16=1.1879
trial 3: score=964
trial 4: score=962
trial 5: score=964
median : 962        range 960-964   spread = 4 score points
```

The reported best=966 (R15 commit 97cbda86) sits **+4 above the R16
median 962**, well within the +4 spread observed in this 5-trial sample.
**Conclusion**: best=966 is a high-tail noise sample, not a structural
improvement over the R5/R11 plateau. The "improved=Yes" status that
auto_optimize.py recorded for R15 reflects a single-trial fluctuation
toward the upper noise edge, not a genuine kernel-perf gain.

This is consistent with the forward-kernel-binary audit (next section):
`grouped_rcr_kernel` codegen has been bit-identical since R5 (HK
`ecbead9a`).

---

## Forward kernel binary audit (R5 → R16)

HK commits between R5 baseline (`ecbead9a`) and R16 head (`0f14b165`):

| HK SHA | Round | What it touched | Forward kernel impact |
|---|---|---|---|
| `ecbead9a` | R5 / dm-R64 | `infra(fp8 grouped)`: `st_32x64` shared-memory tile type | infra-only, DCE'd → 0 |
| `78415fb0` | dm-R64 | `infra(fp8 grouped)`: K-tail `rt_32x64` loaders | infra-only, force-instantiate stub → 0 |
| `addaf23e` | dm-R64 | `infra(fp8 grouped)`: `rcr_mma_32` wrapper | infra-only, DCE'd → 0 |
| `75e30a5f` | dm-R64 | `infra(fp8 grouped)`: rt_fp8e4m3 type-check | infra-only → 0 |
| `9ee90e2c` | R5 | `test`: Lever D microbench `.cu` | test file only → 0 |
| `e2890b43` | R11 | `test`: Lever E microbench `.cu` | test file only → 0 |
| `0f14b165` | R14 | `perf`: `grouped_var_k_kernel_fp8` epilog `b0_keep`/`b1_keep` removal | **backward kernel only** (`grouped_var_k_kernel_fp8` ≠ `grouped_rcr_kernel`) → 0 forward |

**Net forward-kernel change since R5: 0 bytes.**  Per R12 round note's
explicit verification (`Forward kernel binary bit-identical to R11
commit be2e7a30`), and confirmed by inspecting the diff scope of every
HK commit listed above. The plateau the metric measures IS the kernel
plateau.

**Score history (24-shape suite, R5 onward):**

```
R 5 ↺ 959 [F→T spec switch landed earlier]
R 6 ↺ 958 [Qwen-Down M=4096 dispatcher rule, no fwd binary change]
R 7 ↺ 960 [Qwen-GateUP M=2048 dispatcher rule]
R 8 ↺ 961 [DSV3-GateUP 2 dispatcher rules]
R 9 ↺ 962 [BF16 Qwen-Down M=2048 fix]
R10 ↺ 960 [Qwen-GateUP-B32-M4096 dispatcher rule]
R11 ↺ 963 [Lever E microbench falsified]
R12 ↺ 962 [Backward decomposition; Lever H planned]
R13 ↺ 962 [Lever H landed; backward-only]
R14 ↺ 962 [Lever K landed; backward-only]
R15 ↺ 966 [Lever Q landed; backward-only — high-tail noise]
R16 ↺ 962 [this round, 5-trial median]
```

The R6-R10 dispatcher rules genuinely shifted some sub-shape ratios
(which produced the +1..+3 score migrations in that window). R11-R15
are all explicitly backward, leaving forward kernel binary unchanged —
the +4 step at R15 is noise.

---

## Cumulative lever falsification (canonical, citing prior rounds)

After 11 rounds of disciplined falsification on the new 24-shape suite:

| Lever | Verdict | Falsified | Mechanism |
|---|---|---|---|
| **A** Async global→LDS | FALSIFIED | R2 | Already shipped via inline ASM `buffer_load_dwordx4 ... offen lds` (line 787) |
| **B** Triple LDS slab | FALSIFIED | R2 | Dual already shipped (`As[2][2]/Bs[2][2]`); LDS at 137/160 KB cap |
| **C-X** N_MASKED helper SENTINEL | FALSIFIED | R4 | Neutral on active `<0,T,T>` template; -1.8 pp regression on Down-B32-M4096 |
| **C-2** K-tail capture refactor | FALSIFIED | R3 | Captures already in if-branch (line 2540-2719) |
| **C-1** LDS hand-spill of cold regs | DEFERRED | R3-R4 | "No single targetable expression"; gated on C-X (failed) |
| **D** mfma_32x32x64 cell-shape | FALSIFIED | R5 | Microbench gate -0.03 % (gate ≥ +3 pp) |
| **E** ASM software pipelining (iter-level) | FALSIFIED | R11 | Microbench gate -7.28 % × 5 trials |
| **F** Per-shape dispatcher rules | LANDED | R6-R10 | 5 rules landed; saturated |

R12-R15 backward levers (H/K/Q) are NOT in the forward-path
falsification table because they target different kernels
(`grouped_var_k_kernel_fp8` and the Triton transpose helper, not
`grouped_rcr_kernel`).

---

## Architectural ceiling reaffirmed

R3 derivation (still empirically valid):

```
Working set inside main loop body (per-warp, dw/lane):
  cA, cB, cC, cD (4× rt_fl<64,32>) = 128
  a, b0, b1 (3× register tiles)    =  32
  soA, soB, lds_addrs              =   8
  laneid, warpid, tic/toc, k       =   4
  helper temporaries / iter index  = 80-120 (LLVM-determined)
                                   ─────
                            total  = ~250-290 dw/lane

  VGPR cap (launch_bounds=512, 1)  = 256 dw/lane
                                   ─────
  Overflow → VMEM scratch          = 32-43 dw/lane (= measured spill)
```

Worst FP8 cases (R16 measurement, sorted ascending):

| # | Shape | ratio | template |
|--:|---|--:|---|
|  1 | gpt_oss-GateUP-B32-M4096 | **1.026** | `<0,T,T>` (39 spill) |
|  2 | gpt_oss-GateUP-B32-M2048 | **1.046** | `<0,T,T>` |
|  3 | gpt_oss-GateUP-B4-M4096  | 1.067 | `<0,T,T>` |
|  4 | gpt_oss-Down-B32-M4096   | 1.069 | `<0,T,T>` |
|  5 | gpt_oss-Down-B32-M2048   | 1.071 | `<0,T,T>` |
|  6 | gpt_oss-GateUP-B4-M2048  | 1.080 | `<0,T,T>` |
|  7 | Qwen3-Down-B16-M4096     | 1.082 | `<0,F,T>` (32 spill, K=1536) |
|  8 | gpt_oss-Down-B4-M4096    | 1.085 | `<0,T,T>` |

**6/8 worst are gpt_oss `<0,T,T>` (FUSED_KTAIL=true + N_MASKED_STORE=true).**
The remaining 2 split between gpt_oss-Down-B4-M2048 (`<0,T,T>`, K=2880)
and Qwen3-Down-B16-M4096 (`<0,F,T>`, K=1536, prologue/epilog overhead).

---

## R17 candidate: stage-level software pipelining (untried)

**Idea**: pre-issue the K-tail HBM loads (currently first instructions
inside `if (g.fast_k < g.k)` block at line 2588+) **before** epilog 2
starts (line 2484). This overlaps the K-tail HBM round-trip
(~80 cy) with epilog 2's 4 mfma's (~64-128 cy compute), saving the
serialized 80 cy on K-misaligned (gpt_oss) shapes.

**Why this is NOT covered by the R11 Lever E falsification**:
- R11 microbench tested **iter-level** software pipelining
  (`load[k+1] | mfma[k]` inside the main loop K-iter chain). Result:
  -7.28 %, falsified.
- This R17 idea is **stage-level** software pipelining (overlap one
  stage's load with another stage's compute, where the stages are
  "epilog-2" and "K-tail"). Different mechanism.
- The architectural difference: iter-level requires duplicating the
  per-iter LDS double-buffer state (the +16 dw/lane that killed R11's
  hand-rolled path); stage-level reuses the already-existing K-tail
  state (no duplication, since K-tail is a separate code block with
  its own register tiles).

**Implementation sketch**:
1. Hoist `K_tail_base_bytes`, `b_group_byte_base`, `a_srsrc_kt`,
   `b_srsrc_kt`, `laneid`, `row_lane`, `k_lane_byte` (lines 2588-2615)
   to **before line 2484** (epilog 2 start). These are uniform values.
2. Issue the 4× `buffer_load_b128` calls (currently inside K-tail
   lambdas at lines 2627-2673) as raw inline ASM at the **top of
   epilog 2**, with a runtime predicate `g.fast_k < g.k`. On the fast
   path (DSV3+Qwen K%128==0), the predicate is false → no load issued.
   On the slow path (gpt_oss), 4 buffer_load_b128's go in flight.
3. Inside K-tail block (line 2535+), the loads are already in flight
   under the same predicate — just `s_waitcnt vmcnt(N)` then mfma.

**Expected gain**:
- gpt_oss subset (8/24 cases): ~+2-3 % main-kernel TFLOPS, depending
  on how much of the 80 cy actually overlaps with epilog 2 (epilog 2
  is ~256 cy / 64 mfma cycles + 4 barrier waits — plenty of room).
- Geomean impact: 8/24 weight × +2.5 % ≈ +0.83 pp on grp_FP8 geomean
  (1.122 → ~1.131) → +6 score points.
- DSV3+Qwen subsets: 0 % impact (predicate-gated, no work issued).

**Risks**:
1. Hoisting K-tail constants to epilog-2 scope **adds liveness** on
   the fast path. Worst case: spill grows from 32 → 35 dw on `<0,F,T>`
   template, regressing 16/24 cases. **Mitigation**: hoist ONLY the
   minimal constants needed (laneid, row_lane, k_lane_byte = ~3 dw);
   compute SRDs lazily inside the K-tail block.
2. LLVM may not honor the manual issue ordering and reorder the
   buffer_load to its original location. **Mitigation**: use
   `asm volatile` to lock the load placement.
3. Fast-path wastes bandwidth if predicate evaluates true at runtime.
   **Mitigation**: gate by template parameter `FUSED_KTAIL` AND
   runtime `g.fast_k < g.k` — skips entirely when both conditions
   exclude the load.

**Microbench-first protocol** (before committing the kernel rewrite):

R17 should write a synthetic `lever_stage_pipeline_microbench.cu`:
- Path (a): current sequence (mfma chain, then HBM load, then mfma).
- Path (b): pre-issued (HBM load, then mfma chain that overlaps,
  then mfma using loaded data).
- Single warp, ~10K iters, gate threshold ≥ +3 pp.
- If gate fails → falsify R17 idea, accept plateau.

**Multi-round commitment estimate**: 1 round microbench + 1-2 rounds
implementation = 2-3 rounds total. Smaller than Lever D R-B full port
(8-12 rounds per R29 estimate) and concrete (specific code lines to
modify, specific spill targets to validate).

---

## Why this round is docs-only

1. R15's score=966 was high-tail noise (this round's 5-trial sample
   confirms R15 result was 4 above the median 962). Acting on that
   sample as if it were a real improvement would lead to wrong
   regression detection in R16+.
2. Every architectural lever (A/B/C-X/D/E) has been falsified with
   strong empirical evidence. No incremental forward-path lever
   remains within the existing kernel architecture.
3. R17's stage-level pipelining idea is concrete but requires a
   microbench gate first (per R5/R11 protocol). Doing the microbench
   in R17 (not this round) keeps the gate decision evidence-based.
4. R12, R3, R5, R11 all set the precedent for docs-only rounds when
   no actionable lever is in scope; this round follows that pattern.

---

## What this round changed in code

**Nothing.**

- **HipKittens repo**: no change (HK SHA stays at `0f14b165`).
- **Primus-Turbo repo**: this `.md` is the only diff.

Working tree at end of round (excluding harness side-effects):
```
?? analysis/_notes/round-16-fp8-grouped-noise-band-confirmation-and-r17-stage-pipeline-proposal.md
```

Existing user-staged frozen files (`benchmark/ops/config.py`,
`scripts/_metric_*.py`) and untracked harness output
(`.auto_optimize_logs/`, `grouped_gemm_turbo_*.csv`, `3rdparty/composable_kernel`)
are NOT touched by this round.

---

## Hard-constraint compliance check

- [x] No metric / benchmark / config edits
- [x] No dispatcher / can_handle changes
- [x] No quantize fuse, no host-side `.item()` / `.tolist()`
- [x] No per-model branches in `select_default_config`
- [x] HIPKITTEN remains `BackendEntry(..., autotune=False)`
- [x] One focused PT commit (this docs note)
- [x] Zero HK commits this round
- [x] No BF16 grouped touch
- [x] Correctness 0/48 fail across 5 baseline trials

---

## DoD smoke status

R16 is not on the 5/10/15 cadence (R15 was; next is R20). DoD harness
will run automatically at R20. Last DoD score recorded: 608 (SHA
`97cbda86`).

---

## R17 explicit recommendation

Cold-start instructions for the next agent (whether resumed or fresh):

1. **DO NOT** rerun the metric expecting >966 — the 5-trial median
   is 962 with 4-pt spread. Score variation is metric-side noise on a
   bit-identical kernel binary.
2. **DO NOT** retry any of A / B / C-2 / C-X / D / E levers — all
   falsified with documented evidence (cite R2 / R3 / R4 / R5 / R11).
3. **DO** attempt the stage-level pipelining microbench:
   - Write `analysis/fp8_gemm/mi350x/lever_stage_pipeline_microbench.cu`
     mirroring the R5/R11 microbench skeletons.
   - Path (a) = current ordering, Path (b) = pre-issued K-tail load.
   - Gate threshold ≥ +3 pp single-warp throughput delta.
   - If gate passes → R18 = implement in `grouped_rcr_kernel` epilog
     2 (lines 2484-2506) with conservative liveness mitigation
     (hoist only laneid / row_lane / k_lane_byte).
   - If gate fails → falsify Lever R (this round's "stage pipelining"
     designation) and accept plateau as final.
4. **Backward-side option** (parallel track, no metric impact): If
   stage-level pipelining is gated out, fall back to R15's R16 plan
   recommendation — Lever H Direction B (fused HK transpose+RCR
   kernel for dA path). This is the only documented untried backward
   lever with non-trivial expected gain (+5-8 % bwd TFLOPS on
   gpt_oss reroute subset). It does NOT move the metric (forward-
   only) but improves real-world training wall-time and the user
   tracks both via `bench_grouped_gemm_turbo.py` output.

---

## Round-end summary (本轮目标 / 改了什么 / before-after metric / commit SHA / 下一轮建议)

- **本轮目标**: confirm that R15's score=966 is high-tail noise (not a
  structural improvement); reaffirm the plateau structure documented in
  R5/R11; propose ONE concrete untried lever for R17 (stage-level
  pipelining).
- **改了什么**: nothing in code. Only this `.md` round note.
- **Before-after metric**: 5-trial baseline median = **962** (range
  960-964, R15 reported best 966 = high-tail). Bit-identical forward
  kernel binary since R5.
- **Commit SHA**: filled at commit time (PT only).
- **下一轮建议**: R17 = write `lever_stage_pipeline_microbench.cu` per
  the section above; gate at +3 pp; if pass, implement; if fail,
  falsify R17 and switch to backward-track (Lever H Direction B).
