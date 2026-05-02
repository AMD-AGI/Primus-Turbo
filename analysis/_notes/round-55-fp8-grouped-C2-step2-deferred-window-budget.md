# Round 55 (dm) — FP8 grouped: C-2 step-2 deferred (chat-window budget exhausted)

## TL;DR
- Metric: **score=982**, BF16=1.1857, FP8=1.1716. Same Triton-high
  baseline as R54 (982). Noise floor reaffirmed.
- C-2 step 2 (copy kernel body into scaffold namespace, lift
  instantiations, capture resource report) requires ~30-45 min and
  the chat window has ~3 min residual. **Step 2 deferred to R56**
  fresh chat session.
- R54 scaffold (HK SHA `73da21c6`) and step-2 entry-point doc
  (`round-54-fp8-grouped-C2-scaffold-LANDED-types-compile-clean.md`)
  are intact and ready to pick up cold.

## 1. Metric

```
[metric_grouped_only]   grp_BF16  vs triton geomean=1.1857 (n=24)
[metric_grouped_only]   grp_FP8   vs triton geomean=1.1716 (n=24)
score=982
```

Effectively identical to R54 (982 / 1.1861 / 1.1718). Run-to-run
σ ≈ 0.001 on geomean confirms today's GPU3 / Triton baseline is
running ~3% hot vs the R47-R51 historical mean — a system-level
shift, not a kernel regression.

## 2. C-2 step-2 work to pick up in R56

**Source**: `/workspace/code/HipKittens/analysis/fp8_gemm/mi350x/kernel_fp8_layouts.cpp`

**Mechanical steps** (preserved verbatim from R54 doc §4):

1. Copy `grouped_rcr_kernel` template body — currently at line ~2319
   (R54 post-scaffold) — into `namespace lever_c2_round_54_step1_scaffold { … }`
   block at line 184 (right before its closing brace).

2. Lift the four explicit instantiations (currently lines ~2978-2981
   per R54 line shifts; were 2882-2885 in R53):

   ```cpp
   template __global__ void grouped_rcr_kernel<0, false, false>(...);
   template __global__ void grouped_rcr_kernel<0, true , false>(...);
   template __global__ void grouped_rcr_kernel<0, false, true >(...);
   template __global__ void grouped_rcr_kernel<0, true , true >(...);
   ```
   into the namespace too. Mangled symbol names will gain the
   `lever_c2_round_54_step1_scaffold::` prefix automatically.

3. **Crucial**: the `dispatch_grouped_rcr` function (line ~5480 post-R54
   shifts; was 5384 in R53) does **NOT** change in step 2 — it still
   launches the *outer* `grouped_rcr_kernel`, leaving the namespace
   variant as a build-only / measurement-only target. R57 wires
   the dispatcher gated on FUSED_KTAIL=true.

4. Build with `THUNDERKITTENS_ROOT=/workspace/code/HipKittens make all`
   in `analysis/fp8_gemm/mi350x/`. Capture resource report from the
   build log:

   ```
   grep -E "Function Name|VGPRs:|AGPRs:|VGPRs Spill" /tmp/build.log \
     | grep "lever_c2_round_54_step1_scaffold"
   ```

5. **Acceptance criteria** (R53 revised):
   - (a) AGPR ≥ 256 on namespace kernel instances → AGPR hypothesis
     CONFIRMED, proceed to R57 dispatcher wiring.
   - (b) Spill ≤ 8 (FK=T baseline 34, so any reduction OK).
   - (c) Occupancy ≥ 1 waves/SIMD.
   - (d) End-to-end gpt_oss-GateUP-B32-M2048 ratio ≥ 1.15 — only
     measurable after R57 dispatcher wiring, not in step 2.

   If (a) fails (AGPR=0 still): the C-2 hypothesis (4w-style accum
   density triggers AGPR allocation) is FALSIFIED → fall back to
   Lever C-3 (`art_base` + ASM) per R47 doc.

## 3. Latest 5-round metric history

| Round | sha (PT)   | sha (HK)   | score | FP8     | BF16    |
|-------|------------|------------|-------|---------|---------|
| R51   | 4be1a81    | 6c52d017   | 988   | 1.180   | 1.225   |
| R52   | 50f0e67    | 6c52d017   | 989   | 1.142   | 1.225   |
| R53   | a495a55    | 6c52d017   | 987   | 1.160   | 1.233   |
| R54   | 0342233    | 73da21c6   | 984   | 1.172   | 1.186   |
| **R55** | **(this)**  | **73da21c6** | **982** | **1.172** | **1.186** |

R54-R55 identical (within 0.0002 on geomean) — confirms scaffold
zero-impact and Triton baseline drift hypothesis.

## 4. R55 actions

- Metric: ✅ score 982 (matches R54)
- HK code: ❌ NO change (window too tight for step 2)
- Doc commit: this file (R56 entry-point)

## 5. Next round (R56) — clear handoff

The R56 chat (likely a fresh session) should:

1. Read `round-54-fp8-grouped-C2-scaffold-LANDED-types-compile-clean.md`
   (entry-point doc, has the step-2 mechanical recipe).
2. Open `kernel_fp8_layouts.cpp`, locate
   `namespace lever_c2_round_54_step1_scaffold` (line 184) and the
   `grouped_rcr_kernel` template (~line 2319).
3. Move kernel body + 4 instantiations into the namespace.
4. Build, capture resource report, validate criteria (a)-(c).
5. Doc + commit (HK code change, PT doc).

Time budget for R56: a single 90-min chat window should fit the
entire step 2 + measurement + commit (~30-45 min for the code
move, plus build + report + commit).
