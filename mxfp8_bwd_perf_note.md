# MXFP8 Backward — Performance & Correctness Status

**Date:** 2026-08-01; §1 and §2.1 re-measured 2026-08-02
**Device:** MI355X (`gfx950`, 256 CU, 8 XCD) — node `smci355-ccs-aus-n04-33` / container `xiaoming-dev`
(08-01), node `smci355-ccs-aus-n05-29` / container `xiaoming-perf` (08-02). The two are not
interchangeable; see §1.1.
**Model shape:** DeepSeek-V3 — H=7168, I=2048, E=256, K=8, EP8 intra-node, T=8192/rank
**Code:** `feat/xiaompen/mega_moe_flydsl_mxfp8` @ `b9103a5c` (was `bbb5a85e` + uncommitted work on 08-01)
**Bench:** `benchmark/ops/bench_mega_moe_bwd_only.py`, CUDA events, `_amax` across ranks, warmup=8 / iters=25

Companion to `mxfp8_fwd_breakdown_note.md` (forward). Campaign log:
`agent/workspace/mega_moe_combine_reduce_flydsl_gfx950_20260801/logs/`.

---

## 1. Current performance

Three legs share persistent leaf tensors, so the fp8 weight-quant cache hits across iters (mirrors
training). `bwd-only` times `y.backward(grad_y)` with the forward outside the event window.

Means of four runs, pooled over `803951e0` and `b9103a5c` (§1.2 shows the two are one population):

| Routing | dtype | fwd+bwd | fwd-only | bwd-only |
|---|---|---|---|---|
| load_balanced | fp8 | **13.62** | 5.31 | **8.42** |
| load_balanced | bf16 | 20.20 | 7.12 | 13.74 |
| load_balanced | bf16/fp8 | 1.48× | 1.34× | **1.63×** |
| round_robin | fp8 | **13.07** | 5.06 | **8.12** |
| round_robin | bf16 | 19.53 | 6.90 | 13.31 |
| round_robin | bf16/fp8 | 1.49× | 1.37× | **1.64×** |

The backward is where fp8 pays off most (1.63× vs 1.34× on the forward), and the ratio is unchanged
under a heavily skewed routing, so the speedup does not depend on balanced experts.

Spread between repeats of one arm reached 1.35% (RR fp8 `bwd-only`, 8.158 vs 8.049) and 2.1% (LB
bf16 `fwd-only`, 7.063 vs 7.211). **Treat anything under ~1.5% as noise**, and do not read a single
run as a result.

Cross-check `fwd-only + bwd-only` vs `fwd+bwd`: fp8 13.73 vs 13.62 (+0.8%), bf16 20.86 vs 20.20
(+3.3%). The three legs are self-consistent; bf16 hides somewhat more between them.

### 1.1 Against the 08-01 table — do not diff across nodes

The previous session measured (`bbb5a85e` + uncommitted, node `n04-33`):

| Routing | dtype | fwd+bwd | fwd-only | bwd-only |
|---|---|---|---|---|
| load_balanced | fp8 | 14.11 | 5.38 | 8.60 |
| load_balanced | bf16 | 20.47 | 7.23 | 13.91 |
| round_robin | fp8 | 13.42 | 5.12 | 8.28 |
| round_robin | bf16 | 19.77 | 7.01 | 13.43 |

Every cell is 0.9–3.5% faster now, which overstates the progress: **bf16 moved ~1.3% as well**, and
no commit since has touched the bf16 path. That 1.3% is the node/session change, not code. Netting
it out leaves, on load_balanced, about −2.2% `fwd+bwd` and −0.9% `bwd-only` — and **+0.2% on
`fwd-only`, i.e. no measurable forward gain at all** once the drift is removed.

So **A/B on one node within one session**; a cross-session diff on this bench carries a >1% floor
that will swallow or invent a result of the size this campaign is chasing. Keeping the bf16 leg in
every run is what makes the drift visible — it is the control, not redundant data.

### 1.2 `803951e0` → `b9103a5c` — a no-op, confirmed as one

`b9103a5c` renames the fused swiglu-bwd dual-quant kernels (`kern` → `swiglu_bwd_dual_kern`) so they
can be picked out of a profile; it claims byte-exact parity. Two runs each side, same node, same
session, means:

| Routing | dtype | Δ fwd+bwd | Δ fwd-only | Δ bwd-only |
|---|---|---|---|---|
| load_balanced | fp8 | +0.37% | +0.38% | +0.23% |
| load_balanced | bf16 | −0.36% | −0.39% | −0.10% |
| round_robin | fp8 | −0.08% | +0.16% | −0.28% |
| round_robin | bf16 | +0.06% | −0.13% | −0.45% |

Everything is inside ±0.5%, well under the 1.5% noise floor, and **the signs disagree between the
two routings** — fp8 up and bf16 down on load_balanced, the reverse on round_robin. A real effect
cannot flip like that, so this is noise, as a pure rename should be. Recorded because a rename does
change the FlyDSL cache key (kernels lower to their Python function name), which is worth confirming
costs nothing rather than assuming it.

Earlier the same day, `192ba6aa` (SwiGLU backward fused into the grad_l1 dual-quant) measured LB
`bwd-only` 8.511 → 8.414, −1.1%, with bf16 flat to ±0.1% across the pair. That is the right shape
for a real gain but only marginally above the floor, so read it as consistent with the −0.149 ms
the commit itself reports rather than as an independent confirmation.

## 2. Backward internal structure

From the campaign's round-1/2 budget (see `logs/optimize.md`); not re-measured this session:

| Leg | Time (ms) | Share | Note |
|---|---|---|---|
| STEP3 combine (fc1-dgrad GEMM + PUSH + reduce) | ~2.69 | ~31% | PUSH is at the XGMI limit |
| dW1 GEMM (variable-K, LOCAL) | ~2.08 | ~24% | ~39% of peak |
| L2 dgrad + dW2 + quant | remainder | — | includes ~0.43 ms fused dual-quant |

STEP3's PUSH leg moves ~484 MB/rank in ~1.97 ms ≈ 246 GB/s/rank ≈ 1.97 TB/s aggregate = the XGMI
ceiling. Rounds 3 and 5 established this independently, so **the remaining lever there is fewer
bytes, not better scheduling or a different CU split**. The largest single traffic item is the local
round trip where the GEMM epilogue writes 469 MB to L2Y and the PUSH reads the same 469 MB back.

> **Unresolved (08-02).** STEP1 pushes the same ~484 MB/rank, yet its *entire fused stage* measures
> 1.686 ms — less than the 1.97 ms this paragraph assigns to STEP3's PUSH alone. Both cannot hold.
> The 1.97 ms was not re-derived this session, so treat "246 GB/s/rank = the XGMI ceiling" as
> unverified, and do not build a roofline on it until someone re-measures STEP3's PUSH in isolation.

### 2.1 STEP1 (dispatch(dy) + fc2 dgrad) — measured 08-02

`benchmark/ops/bench_step1_fp8_vs_bf16.py`, EP8 T=8192, two runs agreeing to 0.001 ms:

| | Time | TFLOPS | Share of its own backward |
|---|---|---|---|
| STEP1 fp8 | 1.686 ms | 1212 | 19.8% |
| STEP1 bf16 | 2.408 ms | 849 | 17.5% |

fp8 is 1.43× here against 1.63× for the backward as a whole, so **STEP1 is the weakest fp8 leg** and
is what pulls the average down. Accuracy: `grad_swiglu` SNR 30.89 dB vs a per-group bf16 reference.

Truncating the pushed rows (`MEGA_MOE_PROBE_PUSH_SKIP_MOD`; wrong results, timing only) gives the
derivative *on the fused stage* rather than on an isolated PUSH: 100% of rows 1.686 ms, 75% 1.523,
50% 1.400. Extrapolated to zero, the GEMM floor is ~1.12–1.15 ms — so **~0.54 ms, a third of the
stage, is exposed PUSH and two thirds is GEMM**. Against the gfx950 peaks (MXFP8 5.0 PFLOPS, BF16
2.5) the fp8 stage reaches 24% of its own dtype's peak while the bf16 stage reaches 34%, and even
with the PUSH removed entirely fp8 only reaches 36%. The mxfp8 grouped GEMM failing to convert its
2× nominal advantage — not the comm — is what caps the ratio at 1.43×.

**This partially un-refutes dedup, for the backward only.** The same probe on the forward L1 bought
0.077 ms for a third of the rows, inside that leg's noise (`mxfp8_fwd_breakdown_note.md` §4c); on
STEP1 the same fractional cut is worth ~0.21 ms, ~2.7× more sensitive. The reason is shape: the
dgrad GEMM is half the FLOPs of the forward L1 (N=I=2048 vs 2I=4096) against identical PUSH bytes,
so here the PUSH has much less GEMM to hide behind. A forward-refuted optimization does not stay
refuted in the backward.

## 3. Correctness — nondeterministic by default, and the fix is free

> **Correction (round 7).** Section 3 below conflated two independent things. Its *bitwise*
> determinism data stands. Its accuracy claim does not: the swinging dx SNR was never the fp8
> path's fault. Scored against an **analytic** reference instead of the bf16 op, fp8 measures
> y=22.3, dx=21.9, d_topk_w=23.1, dW1=19.5, dW2=19.7 dB — reproducible to 0.1 dB across runs, at
> `PT_COMBINE_GATE_DELAY=0`, with zero catastrophic rows. The old 9.1–19.7 dB spread came from the
> **bf16 reference** corrupting 1–30 of its own dx rows per step (4–9x inflation, varying ranks);
> its forward is clean, only its backward combine is affected. So the fp8 combine's bitwise
> nondeterminism sits at the quantization-noise level and does **not** cost accuracy, and
> `GATE_DELAY` is a reproducibility knob, not an accuracy fix. The gate now uses the analytic
> reference (`benchmark/ops/bench_mega_moe_fused_fp8_bwd.py`); the bf16 op is kept only for timing.

### 3.1 The defect

With identical seeded inputs, 6 runs compared bitwise against run 0
(`rounds/round-6/artifacts/check_determinism.py`):

| Output | rows differing | max\|Δ\| |
|---|---|---|
| `y` (forward L2 combine) | 23.2% (T=2048), 13.1% (T=8192) | 4.1e-01 |
| `dx` (backward STEP3 combine) | 18.5% (T=2048), 11.3% (T=8192) | 3.4e+01 |
| `d_topk_w` | deterministic | 0 |
| `dW1` / `dW2` | 100% | 3.1e-02 / 1.6e-02 |

The magnitudes rule out rounding: this is corruption, not imprecision. Round 6 localized it with a
permutation-invariant stage bisect — the whole L1 dispatch → L1 GEMM → SwiGLU → L2 GEMM chain is
bitwise deterministic, and divergence is introduced at the combine PUSH / `comb` handoff, where the
GEMM-done gate releases the push before L2Y is readable.

**The bf16-referenced SNR gate was not a correctness check.** Seven runs of one unchanged tree gave
dx SNR 9.1–19.7 dB (10.6 dB spread, 3 of 7 below the 15 dB threshold). Round 7 traced that spread to
the reference rather than the fp8 path — see the correction above.

### 3.2 A post-gate stall closed it

> **Removed.** `PT_COMBINE_GATE_DELAY` has been deleted from the kernel. Once round 7 showed it
> buys bitwise reproducibility but not accuracy, an off-by-default knob that nobody should enable
> was not worth carrying. The measurements below stand as the evidence that the gate releases the
> push too early; reinstate the stall from git history if you pick up that root-cause work.

The knob stalled `N × s_sleep(127)` after the GEMM-done gate, before the push read L2Y.
Determinism vs N (T=2048, runs=6, rows differing):

| N | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| `y` | 23.2% | 17.0% | 2.5% | **0** | **0** | **0** | **0** |
| `dx` | 18.5% | 14.0% | 0.4% | **0** | **0** | **0** | **0** |

N=4 is the minimum clean value. Because that is only 2× the last failing value, it was stressed
further: **T=2048 runs=12 and T=8192 runs=6 both give `diff_elems=0`** for `y` and `dx`, while the
`N=0` control at T=8192 still fails (13.1% / 11.3%).

### 3.3 The fix costs nothing measurable

Interleaved A/B, order `0 → 4 → 8 → 4 → 0`, T=8192, `--only fp8`, means of repeated arms:

| | N=0 (incorrect) | N=4 | N=8 |
|---|---|---|---|
| LB fwd+bwd | 14.064 | 14.133 (+0.49%) | 14.165 (+0.72%) |
| LB bwd-only | 8.634 | 8.649 (+0.17%) | 8.652 (+0.21%) |
| RR fwd+bwd | 13.510 | 13.530 (+0.15%) | 13.540 (+0.22%) |
| RR bwd-only | 8.272 | 8.296 (+0.28%) | 8.295 (+0.28%) |

**Read these as "below the measurement floor", not as measured costs.** The spread between two runs
of the *same* arm (N=4 LB `fwd-only` 5.479 vs 5.391 = 1.6%; `bwd-only` 8.695 vs 8.602 = 1.1%) exceeds
every arm-to-arm difference in the table. Corroborating: N=8 must cost at least as much as N=4
physically, yet measures *lower* on LB `fwd-only` — a real effect cannot invert.

This also corrects an earlier reading. `N=32` costs +9.4% on LB fwd+bwd (15.39 vs 14.06, reproduced
twice), which suggested correctness was expensive. It is not — 32 is 8× more stall than needed, and
the entire 9.4% was waste.

N=8 would have been the sane setting (4× margin at the same unmeasurable cost) had the stall been
kept as a default.

### 3.4 What the stall does *not* fix

`dW1`/`dW2` stay nondeterministic at every N including 32, at a magnitude of one bf16 ULP
(3.1e-02 / 1.6e-02). This is a **second, independent source**: pool rows are assigned by two levels
of atomics (per-block LDS `atomic_add` over a cross-block global `atomic_add`), so the row
permutation differs every run and the variable-K grouped GEMM accumulates in a different order. It
was masked by the combine's much larger corruption until the stall removed that. Almost certainly
benign for training, but it should be tracked separately.

### 3.5 Status of the stall as a fix

It is a **mitigation, not a root-cause fix** — it closes the race window by wall-clock separation
rather than by a correct release/acquire pairing. Round 6 empirically eliminated the entire
memory-visibility family first: 12 hypotheses including acquire strength (`PT_MXFP8_GEMM_INV=1`),
pool-scale read strategy, `cache_modifier=19` on the payload, coherent L2Y reads, paired cross-rank
release/acquire, flag scope `agent`→`sys`, and `s_waitcnt` after `buffer_inv`. The corrupted fraction
was pinned at ~24% across arms with substantially different timing, and `l2_writeback()` on the
GEMM-role release made it *worse* (24% → 38%). A genuine visibility bug would have moved.

## 4. Repro

```bash
docker exec xiaoming-dev bash -lc '
cd /perf_apps/xiaoming/MegaMoE
export PYTHONPATH=/perf_apps/xiaoming/MegaMoE MEGA_BENCH_TIMEOUT_S=1800
W=agent/workspace/mega_moe_combine_reduce_flydsl_gfx950_20260801/rounds/round-6/artifacts

# performance (fp8 vs bf16, both routings)
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_mega_moe_bwd_only.py \
  --num-processes 8 --num-tokens 8192 --routing-mode both --only both --warmup 8 --iters 25

# accuracy, against the analytic reference (no bf16 in the verdict)
MASTER_PORT=$((20000+RANDOM%20000)) python3 benchmark/ops/bench_mega_moe_fused_fp8_bwd.py \
  --num-processes 8 --num-tokens 2048

# bitwise determinism (vs run 0)
MASTER_PORT=$((20000+RANDOM%20000)) python3 $W/check_determinism.py \
  --num-processes 8 --num-tokens 2048 --runs 6
'
```

Both are 8-process spawns; kill the whole tree on interrupt or the ranks keep holding VRAM.

The 08-02 numbers were taken from the other node, whose repo path and container differ:

```bash
ssh smci355-ccs-aus-n05-29 "docker exec xiaoming-perf bash -lc '
cd /perf_apps/xiaoming/MegaMoE-dev
export PYTHONPATH=/perf_apps/xiaoming/MegaMoE-dev MEGA_BENCH_TIMEOUT_S=1800
export MASTER_PORT=\$((9000+RANDOM%500))

# sec.1 regression (fp8 vs bf16, both routings) -- run it at least twice
python benchmark/ops/bench_mega_moe_bwd_only.py \
  --num-processes 8 --num-tokens 8192 --routing-mode both --only both --warmup 8 --iters 25

# sec.2.1 STEP1 isolate; prefix MEGA_MOE_PROBE_PUSH_SKIP_MOD=2 or 4 for the PUSH derivative
python benchmark/ops/bench_step1_fp8_vs_bf16.py --num-processes 8 --num-tokens 8192
'"
```

`torch.multiprocessing.spawn` children carry the generic `spawn_main` cmdline and do not match a
`pgrep` on the script name, so killing the parent leaves 8 GPU-holding orphans that silently block
every later run. Reap by `spawn_main`; the wrappers under
`agent/workspace/mxfp8_fwd_opt/run_probe_guarded.sh` already do.

## 5. Open items

| P | Item | Notes |
|---|---|---|
| **P1** | root-cause the GEMM→push gate | costs reproducibility, not accuracy (round 7); the stall that proved it is removed, visibility family already ruled out |
| **P2** | track `dW1`/`dW2` accumulation-order nondeterminism | independent source, one bf16 ULP |
| **P3** | kill the L2Y local round trip in STEP3 | 469 MB written + 469 MB re-read; the only lever that attacks the XGMI/HBM floor |
| **P5** | XGMI dedup on STEP1 | refuted on the forward, but §2.1 measures ~2.7× more push sensitivity here; ~0.21 ms of 1.686 |
| **P6** | why the mxfp8 grouped GEMM stops at ~36% of peak | §2.1 — the cap on STEP1's fp8/bf16 ratio, and dW1 sits at a similar 39% |
| **P7** | re-derive STEP3's isolated PUSH time | §2 says 1.97 ms at the XGMI ceiling; STEP1's whole stage is 1.686 ms for the same bytes |
| **P4** | group-aware L2 swizzle for the combine GEMM | plain `GROUP_M` degrades monotonically (round 4) — groups span per-expert B slabs |
| — | `num_combine_cu` re-sweep | exhausted (round 3): 28 optimal, PUSH is traffic-bound |
| — | segment-driven PUSH | refuted (round 5): loses ILP from the compile-time trip count |
