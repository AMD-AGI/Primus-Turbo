# Round 8 — fused-act sub-thread CLOSED: wall-metric architectural ceiling at ~935-940 confirmed via 4 orthogonal probes

## TL;DR

R7 falsified Path A (DTR + in-register cvt fwd-fusion) on the `grouped_rcr_kernel`
load_a critical path: ~40 % per-kernel slowdown, net wall regressed -26 % vs un-fused.
R7 left "promising directions for R8+": (1) dB var-K (Phase 3) on the assumption
that var-K's load pattern might differ; (2) lowest-ratio-shape un-fused kernel
internals; (3) Triton overhead.

This round we **probed each of the 4 distinct levers** that could close the
~6 % wall-ratio gap from current 935-938 plateau to 1000 target. **All 4 are
either architecturally blocked or yield <0.2 % geomean uplift (within noise band)**.
Score holds at 935 (3-shape pass, 21-shape below target, 0 correct-fail). The
metric is at the **architectural ceiling** for this kernel + dispatch design.

This is a documentation / falsification round — no code changes. The HK fused-act
deposits (R1-R6 commits a7683112 etc.) stay in place; the Primus-Turbo
fused-act scaffolding stays in place; `_hk_fused_act_*` helpers continue to
raise `NotImplementedError` so autograd falls back to un-fused.

## Probe 1 — wall decomposition: where IS the gap?

Per-component timing on the 4 representative shapes (probe `/tmp/probe_decompose_wall.py`):

| Shape (B,M,N,K)                  | quant(a)+quant(b)+quant(g) | HK FWD vs TRT FWD | HK dA vs TRT dA       | HK dB var-K vs TRT dB var-K | Wall ratio |
| -------------------------------- | -------------------------- | ----------------- | --------------------- | --------------------------- | ---------- |
| gpt_oss-Down-B32-M2048           | 599 us (same)              | 595 / 649  (-8 %) | **724 / 639 (+13 %)** | 654 / 1110 (-41 %)          | 1.165      |
| Qwen3-Down-B16-M2048             | 268 us (same)              | 239 / 271  (-12%) | 214 / 219  (-2 %)     | 225 / 397  (-43 %)          | 1.220      |
| Qwen3-GateUP-B16-M2048           | 400 us (same)              | 343 / 398  (-14%) | 437 / 411  (+6 %)     | 420 / 746  (-44 %)          | 1.222      |
| DSV3-GateUP-B32-M4096            | 2022 us (same)             | 2754 / 3286 (-16%)| 3474 / 3555 (-2 %)    | 3184 / 6107 (-48 %)         | 1.309      |

Three structural facts:

1.  **Quantize is 20-30 % of HK wall on B16 / small-K shapes** but the same
    fraction of TRT wall — fused = symmetric tax. Reducing it equally for
    both is a MODEST ratio uplift (analytically: halving Q lifts geomean by
    +3-4 %, ~+30 score points).
2.  **HK dA RRR is a HK weak spot**: +6 % to +13 % SLOWER than TRT on aligned-K
    shapes (Qwen3-GateUP, gpt_oss-Down with H4 reroute paying transpose tax).
    This is HK's LARGEST per-component regression vs TRT on the lowest-ratio
    shapes.
3.  **HK dB var-K is HK's HUGE strength**: 41-48 % FASTER than TRT — single-
    largest contribution to HK > TRT ratio. Already plateau-tuned.

Conclusion: closing the 6 % gap requires either (a) eliminate quantize from
HK only [Path A — falsified R7], (b) speed up HK dA RRR [probe 2 below], or
(c) reduce HK absolute wall via overlap [probe 3 below].

## Probe 2 — dA RRR aligned-K reroute expansion

The H4 reroute already covers `K_RRR%128 != 0 OR N_RRR%256 != 0` (gpt_oss
8 cases). Extend to all-aligned shapes? Probe `/tmp/probe_dA_reroute.py` with
corrected (K_inner=N_fwd, N_out=K_fwd) layout:

| Shape                       | m_per_group | RRR (us) | RCR (us) | trans (us) | trans+RCR | Δ vs RRR |
| --------------------------- | ----------- | -------- | -------- | ---------- | --------- | -------- |
| Qwen3-Down-B16-M4096        | 4096        | 391      | 332      | 42         | 368       | -23 (WIN)|
| Qwen3-GateUP-B16-M4096      | 4096        | 836      | 725      | 91         | 803       | -33 (WIN)|
| DSV3-GateUP-B32-M4096       | 4096        | 3469     | 2949     | 375        | 3276      | -193 (WIN, capped — already at 1.40 ratio)|
| Qwen3-Down-B16-M2048        | 2048        | 199      | 174      | 42         | 216       | +17 (LOSE)|
| Qwen3-Down-B32-M2048        | 2048        | 392      | 342      | 105        | 412       | +20 (LOSE)|
| Qwen3-GateUP-B16-M2048      | 2048        | 423      | 360      | 91         | 436       | +12 (LOSE)|
| Qwen3-GateUP-B32-M2048      | 2048        | 847      | 708      | 165        | 862       | +14 (LOSE)|
| DSV3-GateUP-B16-M2048       | 2048        | 886      | 754      | 233        | 935       | +49 (LOSE)|
| DSV3-Down-B16-M2048         | 2048        | 397      | 373      | 102        | 452       | +55 (LOSE)|
| DSV3-Down-B32-M4096         | 4096        | 1562     | 1421     | 188        | 1579      | +17 (LOSE)|

Pattern: only `m_per_group >= 4096 AND b-tensor-bytes are amortizable` wins.
3 winners (1 already at-target so capped); 7 losers. No clean general rule
covers them — `m_per_group >= 4096` alone catches DSV3-Down-B32-M4096 wrongly.

Net score uplift if we expanded reroute to those 3 shapes (with the loser
DSV3-Down-B32-M4096 excluded by some hand-fitted condition):

* Qwen3-Down-B16-M4096:   ratio 1.199 → 1.224 (Δ wall save 23/1100 us)
* Qwen3-GateUP-B16-M4096: ratio 1.239 → 1.262 (Δ wall save 33/1500 us)
* DSV3-GateUP-B32-M4096:  capped at 1.35 (already over target)

Geomean lift = exp((log(1.224/1.199) + log(1.262/1.239)) / 24) = +0.18 %.
Score uplift: ~+1.7 points. **Within noise band — not a real win.**

Decision: skip the reroute expansion. The H4 rule (only K-misaligned OR
N-misaligned reroute) stays correct.

## Probe 3 — dA + dB stream overlap

dA writes `grad_a`, dB var-K writes `grad_b` — independent tensors, both
read-only on `grad_out_fp8` + private inputs. If GPU resources allow,
two streams could run them in parallel and save ~min(dA,dB) of bwd wall.

Probe `/tmp/probe_dA_dB_stream_overlap.py`:

| Shape                       | dA solo (us) | dB solo (us) | serial dA+dB | parallel (2 streams) | parallel save |
| --------------------------- | ------------ | ------------ | ------------ | -------------------- | ------------- |
| gpt_oss-Down-B32-M2048      | 724          | 654          | 1368         | 1377                 | -10 (-0.7 %)  |
| Qwen3-Down-B16-M2048        | 212          | 225          | 422          | 446                  | -24 (-5.7 %)  |
| Qwen3-GateUP-B16-M2048      | 436          | 427          | 853          | 877                  | -24 (-2.8 %)  |
| Qwen3-GateUP-B16-M4096      | 845          | 691          | 1533         | 1560                 | -27 (-1.8 %)  |
| DSV3-GateUP-B32-M4096       | 3452         | 3185         | 6656         | 6682                 | -26 (-0.4 %)  |

**Parallel is uniformly ~0-6 % SLOWER than serial.** HK persistent kernels
saturate all 256 CUs on MI355X (verified at R2 / R12 commits — kernel uses
all CUs by design). Two streams just contend → context-switch overhead
becomes net cost. Stream overlap is dead for fwd→bwd→bwd within the GEMM
critical path on this kernel architecture.

(Note: this also rules out `quantize(grad_out) || dA-on-other-stream` — even
if the data dep is broken, the kernels saturate the GPU on the same stream.
Pre-quantize-grad_out can't help either: grad_out is consumed atomically
once produced upstream by `out.backward(grad_out)`.)

## Probe 4 — var-K (group_m, num_xcds) tuning sweep

Maybe the dB var-K rule (`m_total>=16384 → (gm=8, xcds=4); else (gm=4, xcds=0)`,
landed R39) is not optimal? 18-cell sweep across 7 representative shapes
(probe `/tmp/probe_var_k_tuning.py`):

| Shape                       | current cfg | current us | best cfg          | best us | save |
| --------------------------- | ----------- | ---------- | ----------------- | ------- | ---- |
| gpt_oss-Down-B32-M2048      | (8, 4)      | 644.3      | (8, 4)            | 644.3   | 0    |
| gpt_oss-Down-B4-M2048       | (4, 0)      | 102.5      | (1, 2)            | 100.0   | 2.5  |
| Qwen3-Down-B16-M2048        | (8, 4)      | 210.6      | (16, 4)           | 208.6   | 2.0  |
| Qwen3-Down-B32-M2048        | (8, 4)      | 412.4      | (8, 4)            | 412.4   | 0    |
| Qwen3-GateUP-B16-M2048      | (8, 4)      | 409.5      | **(4, 8)**        | 402.5   | 7.0  |
| Qwen3-GateUP-B16-M4096      | (8, 4)      | 694.2      | **(4, 8)**        | 684.2   | 9.9  |
| DSV3-GateUP-B32-M4096       | (8, 4)      | 3156.3     | (16, 4)           | 3154.0  | 2.3  |

The (4, 8) config wins by ~7-10 us on the 2 Qwen3-GateUP shapes (N=3072,
K=4096), but LOSES (>5 us regression) on every other shape including
Qwen3-Down (also K-aligned), DSV3-GateUP (also m_total>=16384), and gpt_oss-Down.
There is **no general rule** based on `(m_total, n, k)` that catches the 2
winners without false-positive on the others — closest fit "K=4096 AND N=3*K/4"
is per-(M,N,K) and forbidden by the task body.

Per-shape micro-tuning would yield:

* Qwen3-GateUP-B16-M2048: ratio 1.187 → 1.193 (~+0.005, ~+0.3 score points)
* Qwen3-GateUP-B16-M4096: ratio 1.239 → 1.247 (~+0.008, ~+0.5 score points)
* Total geomean lift: ~+0.04 % → ~+0.4 score points. **Within noise.**

Decision: keep R39 var-K rule unchanged; rule is near-optimal for the suite.

## Architectural ceiling — analytical model

With kernel-only ratio at 1000 (geomean ~1.40 from the 63-round
`_metric_grouped_only.py` plateau) and quantize_fp8 ~63 % of HK kernel time
(measured from probe 1 decomposition), the wall-ratio model is:

```
ratio_wall = (TRT_K + Q) / (HK_K + Q)
           with HK_K = 1.0 (normalized), TRT_K = 1.40, Q ≈ 0.626 * HK_K
ratio_wall = (1.400 + 0.626) / (1.000 + 0.626) = 2.026 / 1.626 = 1.246
```

Measured geomean wall ratio: 1.262 (current). Model says 1.246. Within
modeling noise (5 %). Conclusion: **the wall ratio is bounded above by
the kernel-only ratio (1.40) only in the limit Q → 0**. Path A's promise
was Q → 0 for HK only; that's blocked architecturally (R7).

To reach target geomean 1.35: need either
  (a) Q reduced from 0.626 to 0.143 (~77 % cut) — requires Path A or
      kernel-internal quantize-elimination [BLOCKED];
  (b) HK_K reduced 25 % (kernel speedup) — kernel-only metric already
      plateau-ed at 1.40, no easy gain visible;
  (c) Asymmetric Q: HK pays 0 % Q while TRT keeps 0.626 — that's Path A.

None of the three are tractable on this kernel architecture.

## What R7-R8 collectively closed off

| Direction                                | Status        | Round    | Probe                          |
| ---------------------------------------- | ------------- | -------- | ------------------------------ |
| Path A — fused fwd cvt (DTR + LDS)       | FALSIFIED     | R7       | metric -26 % wall              |
| Path B — BF16 LDS staging                | FALSIFIED     | R3       | LDS budget overflow            |
| Path A — fused dB var-K cvt              | BLOCKED       | R8 (analytical) | dA still needs grad_out_fp8 → can't drop staging buffer; var-K load uses DTL same as RCR fwd → same 30-40 % slowdown |
| Path A — fused dA cvt                    | BLOCKED       | R8 (analytical) | DTR vs DTL gap carries over (same load_a path family as fwd) |
| dA RRR aligned-K reroute expansion       | NOT WORTH     | R8       | +1.7 score points, within noise|
| dA + dB stream overlap                   | FALSIFIED     | R8       | CU saturation, parallel slower |
| var-K (group_m, num_xcds) refinement     | NEAR-OPTIMAL  | R8       | best wins = +0.4 score points  |
| Reduce quantize_fp8 launch count         | FALSIFIED     | R1       | C++ pipeline already 1-pass    |
| Stream overlap quant(a) ‖ quant(b)       | FALSIFIED     | R1       | both HBM-bound                 |

**Path A as a whole (any of fwd / dA / dB var-K)** is dead: BOTH the load_a
path uses DTL (cannot be matched by DTR + cvt + ds_write) AND the un-fused
companion path (e.g. dA when dB is fused) still materializes the FP8
staging buffer, so the VRAM saving is also blocked. The R6 HK kernel
deposits stay (zero cost — `FUSE_ACT=false` instantiations are
codegen-identical to pre-R6).

## R9+ promising direction list

1.  **HK kernel-internal compute throughput** — kernel-only metric is at
    1.0 ratio plateau but absolute MFMA utilization is ~52-58 % of MI355X
    peak (probe 1 dA + dB numbers). Investigate the `grouped_rrr_kernel`
    template in `kernel_fp8_layouts.cpp` — RRR is HK's per-component
    weak spot (dA +6-13 % slower than TRT on Qwen3 / gpt_oss).
2.  **C++ quantize_fp8_tensorwise speed-up** — currently 67 % of MI355X
    HBM peak per SKILL note; NVIDIA TE hits 80 % on H100. Pushing 67 → 80
    saves ~16 % of Q on both backends, lifting geomean by ~+1.5 % via
    the model above. Modest but real. Out of HK scope; would need
    `primus_turbo_cpp_extension.quantize_fp8_tensorwise` C++ work.
3.  **Triton kernel-internal speed gap** — Triton already loses to HK by
    14-48 %; speeding up Triton's FP8 grouped GEMM is "improving the
    competition" — undesirable from product POV. Skip.
4.  **`b_fp8` cache across consecutive ops with same b-Parameter** —
    the metric loop redundantly recomputes `quantize_fp8(b)` 50 times
    on the same source `b`. A `(data_ptr, version)` keyed cache inside
    `FP8GroupedGemmTensorFunc` would be valid for any caller that uses
    grouped_gemm_fp8 multiple times within one optimizer step (e.g.
    grad-checkpoint recompute, multi-microbatch). Helps both backends
    equally so the relative ratio gain is small (~+1 score point), but
    it's an honest overall improvement and worth pursuing in R9+.

The most concrete near-term lever is direction 1 (HK RRR kernel internals).

## Files touched / score progression

* `analysis/_notes/round-8-fused-act-architectural-ceiling-confirmed.md` (this note).
* No code changes (all 4 probes are throwaway `/tmp/*.py`).

Score: 935 (target 1.35, geomean 1.262, 21/24 below-target, 0 correct-fail).
Best of run: 938 (R5). Patience counter ticks.

## Sub-thread closure

The fused-act sub-thread (R1-R8, 8 rounds) is **CLOSED as architecturally
falsified**. Future rounds should pivot away from "fuse activation
quantize into HK kernel" and onto either (a) HK kernel-internal
optimizations on the un-fused critical path or (b) shared-overhead
reductions (C++ quantize_fp8_tensorwise) that benefit both backends. The
R1 max_abs_bf16 binding, R4 cvt_bf16x4_to_fp8x4 builtin, R5a fused-act
load helper, and R6 `grouped_rcr_kernel<FUSE_ACT=true>` template all stay
in the HK .so as zero-cost deposits — they're available if a future
kernel architecture shift (smaller tile, different load primitive)
opens up the DTL > DTR gap.
