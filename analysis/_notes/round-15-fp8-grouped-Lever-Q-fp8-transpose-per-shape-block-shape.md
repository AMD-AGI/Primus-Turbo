# Round 15 — FP8 grouped backward Lever Q: per-shape `(BK, BN)` block-shape selection in `fp8_transpose_3d`

## TL;DR

- **R15 metric (forward) median 963** (5-trial: 960, 964, 964, 963, 963) —
  noise-band stable vs R14 961-963 / R13 962-964.
- **Lever Q landed**: `fp8_transpose_3d` (R13 Lever H helper) now picks
  `(BK, BN)` per (K, N) shape via a 3-branch heuristic instead of the
  fixed `(128, 128)`. The heuristic source is the R14 microbench sweep
  at `/tmp/tri_transpose_sweep.py` over the 4 gpt_oss reroute shapes:
  - `K > N` (gpt_oss-GateUP K=5760, N=2880) → `(BK=128, BN=256)` —
    fewer N-blocks (12 vs 23) to keep per-program work-set in L2.
  - `K == N` (gpt_oss-Down K=N=2880) → `(BK=256, BN=128)` — half the
    K-blocks (12 vs 23), N-blocks unchanged.
  - `K < N` (no FP8 metric shape currently triggers; the H4 reroute
    only fires for gpt_oss K=2880 ≤ N=2880-5760) → `(BK=128, BN=128)`
    default fallback for forward-compatibility.
- **R15 took two iterations**:
  1. **Lever Q v1 (autotune)**: wrapped `_fp8_transpose_3d_kernel` in
     `@triton.autotune` with a 9-config grid `(BK, BN) ∈ {64,128,256}^2`
     and `key=["B","K","N"]`. Per-iter wall similar to v2 below, but
     metric 5-trial range widened to 958-964 (vs R14 961-963), the 30 ms
     first-call autotune sweep colliding with the bench warmup phase
     and inflating one of the 5 trials. **Reverted**.
  2. **Lever Q v2 (static heuristic)**: replaced autotune with a
     branch on `K vs N` in the Python wrapper. No runtime cache lookup,
     no first-call sweep cost. **Landed**.
- **Backward bench** (24-shape FP8 e4m3 tensorwise, all PASS):

| metric | R14 (post-Lever-K) | R15 (post-Lever-Q) | delta |
|---|--:|--:|--:|
| Avg forward TFLOPS  | 1338.94 | 1339.25 | +0.02 % (noise) |
| Avg backward TFLOPS | 1365.56 | **1366.46** | **+0.07 %** (24-case avg) |
| **gpt_oss subset B32 bwd avg** | 1294 | **1308** | **+1.1 %** |
| Correctness (SNR ≥ 25 dB) | 24/24 | 24/24 | matched |

- **gpt_oss B32 per-case bwd TFLOPS**:

| shape | R14 bwd | R15 bwd | delta |
|---|--:|--:|--:|
| gpt_oss-Down-B32-M2048 | 1057 | **1078** | **+1.9 %** |
| gpt_oss-Down-B32-M4096 | 1270 | **1287** | +1.3 % |
| gpt_oss-GateUP-B32-M2048 | 1288 | **1306** | +1.4 % |
| gpt_oss-GateUP-B32-M4096 | 1559 | **1561** | +0.1 % |

- **BF16 backward bench** (sanity, transpose path not used in BF16):
  avg fwd 1266.81 / bwd 943.32 — unchanged vs R14 1268 / 944.
- **No HK kernel change**; R15 commit is Primus-Turbo only.

## R15 falsified probe — H4 reroute K-only revert

Before landing Lever Q, R15 tested whether the R18 H4 extension
(reroute also fires when `b.shape[-1] % BLOCK_SIZE != 0`, i.e. the
gpt_oss-GateUP N_RRR=2880 case) is still beneficial after R13 Lever H
shrunk the transpose cost from 1078 µs → 138 µs.

- **Hypothesis**: R18's "30 % bwd wall went to ntail+scalar" was
  measured against the PyTorch transpose (1078 µs at ~1 TB/s).  With
  Triton transpose at 138 µs (peak HBM), maybe the trade-off has
  flipped and native RRR + `grouped_ntail_kernel_lds_rrr<64>` is
  cheaper for K_RRR-aligned + N_RRR-misaligned shapes.
- **Test**: change `grouped_gemm_fp8_impl.py:364` from
  `if not trans_b and ((a.shape[1] % K_BLOCK) != 0 or (b.shape[-1] %
  BLOCK_SIZE) != 0):` to `if not trans_b and (a.shape[1] % K_BLOCK)
  != 0:` and re-run `bench_grouped_gemm_turbo.py --dtype fp8`.
- **Result**: avg backward TFLOPS 1365.56 → **1261.13 (−7.6 %)** across
  all 24 shapes.  The 4 gpt_oss-GateUP cases regress hard.  R18 H4
  extension remains net positive even after Lever H lowered the
  transpose cost.
- **Verdict**: **falsified**.  Reverted line 364 to the existing
  condition. R18 extension stays.

This rules out the "shrink reroute coverage" lever for any future
round and locks in transpose+RCR as the chosen path for both K_RRR-
and N_RRR-misaligned gpt_oss cases.

## Static heuristic vs autotune

`@triton.autotune` 在第一次 call (per unique key) does an actual sweep
over all configs and selects the winner.  With `key=["B","K","N"]`
and the bench/metric calling sequence (multiple shapes back-to-back),
the first call for each shape inside the `bench.Timer.timeit(100)`
loop hits the autotune sweep — a 30 ms blocking call that lands inside
one of the 100 timed iterations.

torch.utils.benchmark uses the **median** of measurements when
reporting `m.median`, but the metric script's score aggregation runs
many independent `bench.Timer` instances and the autotune outliers do
shift the lower percentiles of the 5-trial median sample (R15 v1:
958-964 range vs static-heuristic 960-964 range, with the 958 outlier
matching the autotune-warmup fingerprint).

Static heuristic is the right tool here: only 3 (K, N) regimes ever
occur in the FP8 metric/bench, the per-regime winner was found to
1× the per-call wall in the R14 sweep, and the cost is one Python
branch instead of a Triton runtime cache lookup.

The 9-config autotune sweep itself is preserved as documentation in
the commit history (this round's first iteration) — re-enable
`@triton.autotune` if a future shape doesn't fit the 3-branch
heuristic.

## Lever stack update (R15)

| Lever | Round | Path | Verdict |
|---|---|---|---|
| A: async global→LDS | R1 | fwd | already shipped |
| B: dual/triple LDS ping-pong | R2 | fwd | LDS capacity bound |
| C: VGPR live-range reduction | R3..R5 | fwd | architectural VGPR cap |
| D: 32×32×64 MFMA cell | R5 | fwd | ≈ 0 % delta |
| E: ASM software pipelining | R11 | fwd | -7.28 % microbench, falsified |
| F: per-shape dispatcher rules | R6..R10 | fwd | 5 rules landed; plateau |
| G: `grouped_rrr` spill compression | R12 | dA | DSV3-only, no ratio gap |
| H: drop `b.contiguous()` round-trip | R13 | dA | **landed**, +9.3 % avg bwd |
| I: quantize/elementwise glue trim | R14 audit | bwd | quantize-fusion forbidden; defer |
| J: `var_k` outer-loop spill | R12 | dB | re-localized to epilog by R14 |
| K: `var_k` epilog redundant copies | R14 | dB | **landed**, +0.81 % avg bwd |
| **Q: per-shape (BK,BN) for fp8_transpose_3d** | **R15** | **dA** | **landed**, +1.1 % gpt_oss B32 bwd |
| H-Direction-B: fused HK transpose+RCR | R12 plan | dA | 6-9 % bwd potential, multi-round risk |
| H4-revert: K-only reroute | R15 probe | dA | **falsified**, -7.6 % avg bwd |

Forward path: 6 levers exhausted, plateau at 961-964 noise band.
Backward path: 3 net-positive levers landed (H R13 + K R14 + Q R15,
**+10.5 % avg backward TFLOPS** cumulative since R12 baseline 1238.55).
1 backward lever falsified this round (H4 K-only revert).

## R16 candidate levers

1. **Lever H Direction B (P1)** — write a fused HK
   `dispatch_grouped_rcr_btranspose` kernel that consumes
   `b: [B, K, N]` directly and produces grad_a inside the RCR fuse
   epilog without external Triton transpose.  Eliminates the remaining
   ~106 μs / iter Triton transpose call on the gpt_oss reroute path
   (~5 % bwd wall on the 8 gpt_oss cases). Risk: medium-high; new HK
   binding, requires correctness probe across all 24 shapes.  R12
   round-note proposed this; R13/R14/R15 levered around it; with
   R15's confirmation that K-only reroute is wrong (H4 extension still
   needed), Direction B is the only remaining backward architectural
   lever.

2. **`var_k_kernel_fp8` further spill trim (P3)** — R14 reduced spill
   52 → 37 dw via epilog dead-copy removal.  Forward `grouped_rcr` floor
   is 39 dw; var_k now sits 2 dw below that, suggesting any further
   trim has to come from inside the K-iter inner-loop (currently
   byte-identical to dense `gemm_kernel<CRR,0>` which has 12 dw spill).
   The 25 dw gap is real but the inner loop is already shared with
   dense — refactor would be a dense kernel change too.

3. **Forward path noise edge re-test (P3)** — Qwen3-GateUP-B16-M4096
   FP8 noise edge (R10 deferred, ratio 1.108 vs target 1.20).  7-trial
   tight verify might land +0.2-0.4 pp if the post-R15 transpose
   change accidentally shifted the optimum.  Marginal, not worth a
   dedicated round unless H Direction B falsifies.

## Round-end summary (本轮目标 / 改了什么 / before-after metric / commit SHA / 下一轮建议)

- **本轮目标**: 接 R14 backward agenda — R14 round-note 推荐 R15 = Lever H
  Direction B (fused HK transpose+RCR kernel)。1-round 内做不完，先做 (a)
  H4 reroute K-only revert probe (验证 R18 extension 是不是仍 beneficial)
  + (b) Lever Q (fp8_transpose_3d 加 per-shape BK/BN heuristic)。
- **改了什么**:
  - `primus_turbo/triton/utils/fp8_transpose.py`:
    - 加 `_select_block_shape(K, N)` 3-branch heuristic (K>N → (128,256);
      K==N → (256,128); K<N → (128,128) default fallback)。
    - `fp8_transpose_3d(b)` wrapper now drops the previous `bk=, bn=`
      kwargs (was unused outside the R14 microbench helper) and uses
      the heuristic.
  - 没动 HK kernel（kernel_fp8_layouts.cpp 跟 R14 commit 0f14b165 一致）。
  - H4 reroute K-only revert probe falsified (-7.6 % bwd avg) — line 364
    of `grouped_gemm_fp8_impl.py` 保持不变。
- **Before-after metric**: 962 → 963 (5-trial median，noise band 960-964
  内 stable)。FP8 backward avg TFLOPS 1365.56 → **1366.46 (+0.07 %)**;
  gpt_oss B32 subset bwd avg **+1.1 %** (per-shape +0.1..+1.9 %)。
  24/24 PASS。BF16 unchanged（transpose path 不参与 BF16）。
- **Commit SHA**: filled at commit time.
- **下一轮建议**: R16 攻 Lever H Direction B — 写新 HK kernel
  `dispatch_grouped_rcr_btranspose` 直接 consume [B, K, N] + 内部 LDS
  swizzle transpose + RCR mma + N_MASKED_STORE epilog。预期 +5-8 % bwd
  TFLOPS on gpt_oss reroute subset。Risk medium-high；一轮内做完 +
  bench 24-shape correctness 比较挤，可能 R16 写 + R17 verify 分两轮。
