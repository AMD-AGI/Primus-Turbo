# MXFP8 Forward — Performance Breakdown & Optimization Targets

**Date:** 2026-08-01 (supersedes the 2026-07-29 revision)
**Device:** MI355X (`gfx950`, 256 CU, 8 XCD), node `smci355-ccs-aus-n05-29`, container `xiaoming-perf`
**Model shape:** DeepSeek-V3 — H=7168, I=2048, E=256, K=8, EP8 intra-node, T=8192/rank
**Routing:** `load_balanced` → pool P = 65536 rows/rank, 32 experts × 2048 rows, 256 tiles @ BM=256
**Code:** `feat/xiaompen/mega_moe_flydsl_mxfp8` (single-stream, side-stream overlap **reverted**)

Timing: CUDA events, `_amax` across ranks, warmup=8-10 / iters=25-30.

> **Environment note.** The previous notes' repro block is stale. `/tmp/ptdev_deps` is gone
> and the `rocm/primus:v26.3` base image ships flydsl `0.1.1.dev409`, which is too old for this
> tree (`ImportError: TargetAddressSpace`). Use the `tasimage/primus:v26.3_turbo_perf` image
> (flydsl 0.2.4) and this repo's own `primus_turbo/lib` on `LD_LIBRARY_PATH`. See §8.

---

## 1. Top-level breakdown (fp8 vs bf16)

Source: `benchmark/ops/bench_fwd_breakdown_compare.py`, L1 dispatch/preshuffle CU = 16/16, L2 `num_combine_cu` = 32.

| Stage | fp8 (ms) | bf16 (ms) | bf16/fp8 | % of fp8 FULL |
|-------|----------|-----------|----------|---------------|
| L1 = dispatch + fc1 (NT) | **2.543** | 3.947 | 1.55× | **54.2%** |
| SwiGLU (+ mxfp8 quant) | **0.253** | 0.256 | 1.01× | 5.4% |
| L2 = fc2 + combine (NT) | **1.879** | 2.737 | 1.46× | **40.0%** |
| **FULL forward** | **4.695** | 6.952 | **1.48×** | 100% |

Isolated sum 4.674 vs FULL 4.695 (+0.021 ms) — stages are serial on one stream, no hidden overlap.

> **Superseded.** These are the pre-optimization numbers, kept as the baseline the rest of this
> note reasons against. Two fixes have landed since, both bit-exact: SwiGLU (§6) now runs
> **0.216–0.218 ms**, and the L1 acquire (§5) brings L1 to **2.229–2.236 ms**. Current
> **FULL = 4.48–4.51 ms**. L2 is unchanged.

### 1a. Confirmation pass

The table above and the current numbers were measured in different sessions, and this node drifts
by more than the effects involved, so the gain was re-confirmed by toggling both changes back to
their pre-optimization form **inside one process** (round-1 SwiGLU module reinstated,
`PT_MXFP8_GEMM_INV=1`) and interleaving the legs:

| Leg | FULL fwd (ms) | vs baseline |
|---|---|---|
| baseline (old SwiGLU, `inv=1`) | 4.700 | — |
| + SwiGLU only (§6) | 4.653 | −0.047 |
| + L1 acquire only (§5) | 4.518 | −0.182 |
| **both (shipped)** | **4.484** | **−0.215 (−4.6%)** |

Near-additive, matching the per-change attribution. Across the validation set the speedup is
1.029–1.047× (geomean 1.037×) with uniform routing and 1.023–1.030× (geomean 1.027×) under a 32×
expert imbalance; no shape regresses, so the gain is not an artifact of the benchmark's even
`tokens_per_expert`. Official gate passes on the shipped tree: T=2048 SNR 20.76 dB (1.25× vs
bf16), T=8192 SNR 22.18 dB (1.53× vs bf16). Harness:
`agent/workspace/mxfp8_fwd_opt/{bench_confirm,sweep_confirm_shapes}.py`.

> **Careful with single SNR readings.** The fused fp8 forward is run-to-run **nondeterministic** —
> rerunning one identical config changes ~3–5% of output elements with max|Δ| ≈ 0.4, because the
> cross-rank combine accumulates in arrival order. Two variants can differ by ~1 dB SNR purely
> from this. Comparing repeated-SNR *distributions* instead: baseline 17.00–17.82 dB vs shipped
> 17.06–17.24 dB on the probe's data — fully overlapping, i.e. both changes are accuracy-neutral.
> Bit-exactness must be checked at a stage output (L1 / SwiGLU result), not on the final output.

---

## 2. Where each stage actually binds

Both L1 and L2 are single fused grids running `comm PUSH ∥ GEMM` behind a scoreboard, so
stage time ≈ `max(PUSH, GEMM) + tail`. Isolating the legs (env at **process start**) gives:

| Stage | GEMM leg | PUSH leg | max | FULL | tail | **binds on** |
|-------|----------|----------|-----|------|------|--------------|
| L1 (dc/pc = 16/16) | **1.917** | 1.753 | 1.917 | 2.40 | **0.48** | **GEMM** |
| L2 (cc = 32) | 1.396 | **1.730** | 1.730 | 1.95 | **0.22** | **PUSH** |

This corrects the previous note, which called both stages "balanced". They are not, and they
bind on **opposite** resources — which is why the CU-split knobs behave the way they do (§4).

---

## 3. Roofline: how far is each leg from its own limit?

### 3a. GEMM legs vs. this codebase's own standalone mxfp8 grouped GEMM

`agent/workspace/mxfp8_fwd_opt/bench_ref_gemm.py` runs `grouped_gemm_mxfp8_flydsl_kernel` at the
identical shapes on one GPU (no comm), i.e. the best this repo's own MXFP8 GEMM does:

| Leg | shape (M,N,K,G) | fused role | standalone ref | ref TFLOPS | gap |
|-----|-----------------|-----------:|---------------:|-----------:|-----|
| L1 fc1 | 65536, 4096, 7168, 32 | 1.917 ms | **1.564 ms** | 2461 | **+23%** |
| L2 fc2 | 65536, 7168, 2048, 32 | 1.396 ms | **0.973 ms** | 1978 | **+43%** |

The fused roles get ~224 CU instead of 256; CU-normalising the reference gives 1.79 ms (L1) and
1.11 ms (L2), so **~0.13 ms (L1) and ~0.28 ms (L2) of the gap is real inefficiency**, not CU count.

### 3b. PUSH legs vs. XGMI

Both PUSHes move exactly the same payload: 65536 rows × (7168 B fp8 + 224 B E8M0) = **484 MB/rank**,
of which 7/8 = **424 MB** crosses XGMI.

| Leg | CUs | time | XGMI egress |
|-----|-----|------|-------------|
| L1 PUSH | 16 | 1.753 ms | 242 GB/s |
| L1 PUSH | 32 | 1.501 ms | 283 GB/s |
| L2 PUSH | 32 | 1.717 ms | 247 GB/s |
| L2 PUSH | 64 | 1.640 ms | 259 GB/s |

Per-GPU XGMI capability on MI355X is ~1 TB/s aggregate, so the PUSH legs run at roughly **25% of
link peak** and **saturate on CU count around 32** (adding CUs past that buys ~1%). They are neither
link-bound nor CU-bound past 32 — the limiter is the per-row scatter (each row goes to a different
peer slot), i.e. outstanding-request / address-divergence bound.

---

## 4. Knob sweeps — two hypotheses tested and **refuted**

### 4a. Tile-locality swizzle (XCD remap / GROUP_M / GROUP_N) — refuted, harmful

Every standalone FlyDSL GEMM in this repo uses `xcd_remap_pid` for per-XCD L2 reuse; the mega-MoE
L2 combine has the machinery but ships it **off** (`PT_COMBINE_NUM_XCD=1`, `GROUP_M=0`), and the L1
dispatch kernel has none. That looked like a missed optimization. It is not:

| L2 config (xcd/gm/gn) | L2 ms | vs base |
|---|---:|---|
| 1 / 0 / 0 (ship default) | **1.899** | — |
| 8 / 0 / 0 | 3.063 | **+61%** |
| 1 / 4 / 0 | 1.944 | +2% |
| 8 / 4 / 0 | 2.982 | +57% |
| 8 / 8 / 4 | 3.176 | +67% |

| L1 `GROUP_M` | 1 | 2 | 4 (default) | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| L1 fused ms | **2.344** | 2.355 | 2.397 | 2.504 | 2.530 |

**Why:** in these fused kernels the GEMM tile order is not free — it *is* the pipeline order. In L2
the PUSH role can only ship a pool row once **all** `n_blocks` tiles of that `block_m` are done, so
any swizzle that scatters a row's N-tiles across XCDs delays row completion and serialises comm
behind compute. In L1 the GEMM tiles consume the preshuffle role's per-`block_m` signal, so
row-major (`GROUP_M`→1) matches the order data becomes ready. **Dependency order dominates cache
locality here.** Do not retry XCD/group swizzle on these two kernels.

### 4b. Give the comm role more CUs — refuted for L1

L1 PUSH gets faster with more CUs in isolation (1.753 → 1.501 at 32), but in FULL mode L1 is
GEMM-bound, so the CUs are worth more to the GEMM:

| L1 dc/pc | 8/16 | **16/16** | 16/8 | 24/16 | 24/8 | 32/16 | 24/24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 ms | 2.992 | **2.377** | 2.385 | 2.691 | 2.660 | 2.887 | 2.680 |

Same for L2: `cc` = 32 (1.913) beats 48 (1.972) and 64 (2.006) in FULL even though PUSH-only
prefers 64. **The shipped 16/16 and cc=32 splits are already optimal.** This closes out the old P4.

### 4c. Dedup the redundant XGMI rows — refuted for BOTH stages

A token routed to K experts that land on the same peer is pushed once **per expert**: only 66.1%
of the 524288 (token, expert) pairs are distinct (token, expert-rank) pairs, so ~34% of both
PUSHes is redundant. L1 could drop it with a local copy (identical bytes, `dedup_src_row` is
already allocated in the symmetric layout); L2 would need a pre-combine weight-sum.

Measured the ceiling before implementing either, via a compile-time knob that pushes only 2/3 of
the rows (`MEGA_MOE_PROBE_PUSH_SKIP_MOD` / `MEGA_MOE_PROBE_COMBINE_SKIP_MOD`, §8). Results are
wrong by construction; only the timing is read. One session, idle node, measured at `bbb5a85e`
(the §4d re-measure at `3af98659` leaves the structure these numbers rest on unchanged):

| | baseline | −1/3 of the pushed rows | Δ |
|---|---:|---:|---|
| L1 PUSH (isolated) | 1.768 | 1.309 | −0.459 |
| L1 GEMM-only (push-independent → noise floor) | 1.740 | 1.686 | −0.054 |
| **L1 fused** | **2.173** | **2.096** | **−0.077** |
| L2 PUSH (isolated) | 1.729 | 1.164 | −0.565 |
| L2 no-reduce | 1.745 | 1.681 | −0.064 |
| **L2 full** | **1.900** | **1.858** | **−0.042** |
| **FULL** | **4.501** | **4.460** | **−0.041** |

Both isolated PUSHes shrink exactly as predicted (the L2 floor was computed at 1.136 ms, measured
1.164). **Neither fused stage follows.** Removing 34% of *all* forward XGMI traffic is worth
~0.06 ms of 4.5 ms — and the L1 figure is inside the noise floor of its own GEMM-only leg.

**Why the "L2 binds on PUSH so cut its bytes" inference was wrong:** the PUSH-only leg runs with
the GEMM idle, so its 32 CUs own the whole memory system; `max(GEMM, PUSH)` never described the
fused stage. Decomposing L2 properly (§3c) puts 73% of it in the GEMM and only 19% in exposed
PUSH, and that exposure is drain, not bandwidth — bytes are close to irrelevant on this path.
**Do not retry dedup, on either stage, in any form that only removes bytes.**

### 4d. L2 is not PUSH-bound — corrected decomposition

Superseding the §3 reading. Idle node, `cc`=32, re-measured at `3af98659` (L2 full 1.874):

| component | ms | share | from |
|---|---:|---:|---|
| GEMM | 1.391 | 74% | `PT_COMBINE_GEMM_ONLY` |
| exposed PUSH | 0.355 | 19% | no-reduce 1.746 − GEMM |
| reduce tail | 0.128 | 7% | full 1.874 − no-reduce 1.746 |

`ea9880d7` (b128 top-k reduce) took the reduce tail from 0.155 to 0.128; GEMM and exposed PUSH
were unmoved, and L1 is unchanged (fused 2.177, exposed push 0.433).

The PUSH's 1.729 ms is **79% hidden**. What is left is pipeline drain: a row cannot ship until all
`n_blocks` of its `block_m` are done (the same dependency §4a is about), so cutting bytes only
shortens the last slice — 34% fewer bytes bought 0.064 of the 0.358.

---

## 5. ~~Confirmed defect: whole-L2 `buffer_inv` in every L1 GEMM workgroup~~ — **FIXED** (round 6)

**Original finding.** `dispatch_grouped_gemm_mxfp8_kernel.py`, GEMM role, ran unconditionally per tile:

```python
fx.gpu.barrier()
l2_invalidate()      # -> "buffer_inv sc1", invalidates the ENTIRE L2
fx.gpu.barrier()
```

It is the acquire that makes the peer-pushed pool rows and the preshuffled A-scale visible. But
`buffer_inv` has no address range, so it also evicts the **local, read-only fc1 weights** that the
same tile is about to stream — and there is **one workgroup per output tile**, i.e. ~4096
whole-L2 invalidations per launch. The standalone grouped GEMM (§3a) issues **zero**.

Measured with the new `PT_MXFP8_GEMM_INV` probe (default `1` = current behaviour; `0` skips the
invalidate — **timing only, output is not coherent**):

| | inv=1 (ship) | inv=0 | delta |
|---|---:|---:|---|
| L1 GEMM-only | 1.954 ms | 1.804 ms | **−7.7%** |
| L1 fused | 2.384 ms | 2.152 ms | **−9.7%** (−0.232 ms) |
| **FULL fwd** | **4.759 ms** | **4.502 ms** | **−5.4% (−0.257 ms)** |

The FULL gain exceeds the GEMM-only gain because in FULL mode the invalidate also wipes the
concurrently-running comm and preshuffle roles' working set.

The L2 combine has the same call but in the **COMBINE role** (once per pushed `block_m`, ~256/launch,
16× fewer) — much less damaging, and L2 binds on PUSH anyway.

**Root cause was not what it looked like.** The premise above — that the damage is the *whole-L2*
scope evicting B — is wrong, and the fix that follows from it (fewer, fatter workgroups) was
tried and **rejected**: see round 5, where NB=2/4 N-tiles per workgroup gave 0.905x/0.854x. The
L1 grid is a pipeline, and committing a workgroup to several tiles of one `block_m` serializes
them behind that pool block's arrival.

The actual problem is that `l2_invalidate()` sat in straight-line code executed by **all 256
threads**, so every workgroup issued `buffer_inv sc1` **once per wave, four times**. A workgroup
occupies one CU inside one XCD and `buffer_inv` invalidates that CU's L1 and that XCD's L2, so
the last three are redundant — they find an already-invalidated cache, destroy no extra data, and
just serialize on the L2 port in the workgroup's critical path.

**Fix shipped:** one lane issues it, `s_waitcnt vmcnt(0)` waits for it to land, and the trailing
`s_barrier` releases the other waves (`gemm_inv=2`, now the default).

| | L1 | FULL fwd |
|---|---:|---:|
| before (`gemm_inv=1`) | 2.383 ms | 4.678 ms |
| after (`gemm_inv=2`) | 2.229 ms | 4.510 ms |
| incoherent bound (`gemm_inv=0`) | 2.149 ms | 4.501 ms |

**L1 −6.5%, FULL −3.8%, bit-identical output**, capturing ~93% of the no-acquire bound. The
residual is ~0.02 ms, so this item is closed. Two follow-ups are ruled out and should not be
retried: fewer workgroups (round 5, above) and coherent `sc1` A loads — the latter is
unreachable from the DSL, since A moves through `fx.rocdl.BufferCopyLDS128b`
(`CopyOpCDNA3BufferCopyLDSType.get(128)`), which has no cache-modifier operand.

Note the L2 combine has the same per-wave redundancy in its COMBINE role and has **not** been
touched — same one-line shape, ~256 workgroups/launch, so a much smaller prize, but free.

---

## 6. ~~Confirmed waste: SwiGLU round-trips the activation through HBM~~ — **FIXED** (round 2)

**Original finding.** `swiglu_mxfp8_kernel.py` computed the activation, stored it to a global
`act_bf16` scratch, `s_barrier()`ed, then re-loaded it from global to run the mxfp8 block quant:

```python
buffer_store(act_v, act_rsrc, act_row + col)   # -> act_bf16 [P, I] global scratch
fx.rocdl.s_barrier()
words, biased = _quant_block_words(act_rsrc, act_row + b * _BLK)   # re-read from global
```

Traffic per call: read `l1` 537 MB + write `act_bf16` 268 MB + re-read `act_bf16` 268 MB +
write fp8 134 MB + scale traffic ~13 MB ≈ **1.22 GB in 0.248 ms**, i.e. **44% of the bytes were
avoidable**. The round-trip existed only because the quant block (`_BLK=32`) spans 4 lanes while
the activation loop was `_VEC=8`-wide per lane.

**Fix shipped.** Rather than stage the row in LDS, the thread mapping was changed so one thread
owns a whole 1x32 block (a workgroup keeps `ROWS = ceildiv(BT, n_blk)` = 4 rows resident, so all
256 threads stay busy). The SwiGLU result then feeds `_mxfp8_words_from_f32_subvecs` straight from
registers; only the 1-byte E8M0 scales go through LDS, which also removed the `scale_raw`
round-trip. Output is **bit-identical** to the old kernel (the f32 activation is round-tripped
through bf16 before the amax, exactly as storing/reloading bf16 did). End-to-end SNR vs bf16 is
unchanged at 22.31 dB.

| | bytes | isolated | in the fused fwd |
|---|---|---|---|
| before | 1.221 GB | 0.223 ms | 0.248 ms |
| after | 0.675 GB | 0.196 ms | 0.216–0.218 ms |

**−0.031 ms on the stage (−12.6%); FULL 4.695 → ~4.66 ms.**

**What it revealed.** Bytes fell 45% but time only fell 12%, so achieved bandwidth *dropped* from
5.5 to 3.4 TB/s. This kernel was never bandwidth-bound — it is bound by the per-block quant ALU
chain (`exp`, the sigmoid divide, `1/scale`, `cvt_pk_fp8`). Two follow-ups confirmed that and were
rejected: restoring perfect coalescing via a `ds_swizzle` quad-butterfly amax was **4.5% slower**
(4 lanes then redundantly compute the E8M0 scale, divide included), and a 23-point `(BT, grid_x)`
sweep was **flat within ~3% noise**. See `rounds/round-{2,3,4}/summary.md`.

Further gains here now require cheapening the quant math (fast reciprocal for the sigmoid, bit
trick for `2^-exp`), which changes numerics and forfeits the bit-exact gate — an accuracy
decision, not a perf one.

**That ceiling was measured, and the reciprocal half of it is now SHIPPED (round 7).** The single
biggest ALU item was the *per-element* IEEE divide in the SiLU (`divf(g, denom)`), which AMDGPU
expands to ~10 VALU (`v_div_scale` ×2, `v_rcp`, 3 `v_fma`, `v_div_fmas`, `v_div_fixup`) rather
than the 2 of `v_rcp` + `v_mul`. Sweeping the fastmath flags on that one divide:

| variant | SwiGLU ms | Δ | end-to-end SNR |
|---|---:|---:|---:|
| IEEE `divf` (old default) | 0.220 | — | 20.78 dB |
| `arcp` | 0.218 | −0.002 | — |
| **`afn,arcp` — SHIPPED** | **0.182** | **−0.038** | **20.57–20.66 dB** |
| `fast` | 0.181 | −0.039 | 20.72 dB |
| divide replaced by a multiply (bound) | 0.173 | −0.047 | wrong by construction |

The divide was worth **0.047 ms of the 0.220 (21%)**; the shipped reciprocal captures 81% of it.
Two things the sweep settles, both worth knowing before touching fastmath elsewhere:

* **`arcp` alone is a no-op** — the backend keeps the full IEEE expansion unless `afn` is set too.
  Testing only `arcp` would have wrongly cleared the divide as "not the bottleneck"; it took the
  deliberately-wrong multiply variant to establish that it was.
* **`fast` buys nothing over `afn,arcp`** and additionally implies `nnan`/`ninf`, which lets the
  compiler assume the `ACTIVATION_CLAMP` min/max never sees a NaN — a robustness regression a
  clean benchmark cannot show, for zero extra speed.

Cost: ~1 ULP on the divide, so SwiGLU is **no longer bit-exact** against the round-1 snapshot;
`test_swiglu.py --ref 1` will now fail its bit-exact leg by design. End-to-end SNR moves less than
this gate's own session spread (a plain rebuild measured 19.97 and 20.78 on different days).

**−0.038 ms on the stage (−17%); FULL ~4.50 → ~4.43 ms.** What is left here is the `1/scale` in
`_mxfp8_words_from_f32_subvecs`: `scale` is an exact power of two, so replacing that divide with
an exponent negation (`254 - biased`, shifted) is *bit-exact*, and would also remove the
`0 * Inf = NaN` an all-zero 32-block currently produces. Amortised over 32 elements, so small —
unmeasured.

Note the *deeper* fusion — folding SwiGLU into the L1 GEMM epilogue the way L2 does with
`StoreCQuantMxfp8CShuffle` — is **not** as attractive as it first looks: the backward saves the raw
`l1` [P, 2I] bf16 (`swiglu_backward_flydsl_kernel` consumes it), so L1 must write those 537 MB
regardless. Fusion would only save the 537 MB re-read plus a launch, and it requires re-tiling N so
each tile owns matched gate/up column pairs.

---

## 7. Ranked optimization backlog

| P | Target | Expected | Evidence | Risk |
|---|--------|----------|----------|------|
| **P2** | L2 GEMM role: close the +28% CU-normalised gap vs standalone (CShuffle mxfp8 quant epilogue, BN/BM not autotuned) | up to −0.28 ms on the GEMM leg | §3a, §4d | med — **now unblocked and top-ranked**: the GEMM is 73% of L2, and the old "wait for P3" caveat died with P3 |
| **P5** | L1 tail 0.48 ms (20% of L1): pipeline fill/drain of comm→preshuffle→GEMM | −0.1 to −0.2 ms | §2 | med |
| **P6** | L1 `GROUP_M` 4 → 1/2 | −0.04 ms | §4a | **low** — one-line default |
| **P7** | L2 COMBINE role: same per-wave `buffer_inv` redundancy as §5, untouched | small (~256 WGs/launch) but free | §5, round 6 | **low** — same one-line fix |
| ~~P1~~ | L1 GEMM `buffer_inv`: issue from one lane, not all 4 waves | **DONE, −0.18 ms FULL**, bit-exact | §5, round 6 | shipped |
| ~~P4~~ | SwiGLU: drop the global `act_bf16` round-trip | **DONE, −0.031 ms**, bit-exact | §6, round 2 | shipped |
| ~~P8~~ | SwiGLU: serve the SiLU divide with `afn,arcp` (rcp+mul, not the IEEE expansion) | **DONE, −0.038 ms** (stage −17%) | §6, round 7 | shipped — **not** bit-exact (~1 ULP) |
| ~~X~~ | L1: fewer/fatter GEMM workgroups to cut invalidates | **refuted, 0.85–0.91x** | §5, round 5 | do not retry |
| ~~X~~ | L1: coherent `sc1` A loads instead of the invalidate | **blocked** — no cache modifier on `BufferCopyLDS128b` | §5, round 5 | needs a flydsl change |
| ~~X~~ | SwiGLU: recover coalescing with a quad-butterfly amax | **refuted, 4.5% slower** | §6, round 3 | do not retry |
| ~~X~~ | SwiGLU: `(BT, grid_x)` launch geometry | **refuted, flat within noise** | §6, round 4 | do not retry |
| ~~X~~ | XCD / GROUP_M / GROUP_N tile swizzle on L1 or L2 | **refuted, up to +67%** | §4a | do not retry |
| ~~X~~ | Re-tune dispatch/preshuffle/combine CU splits | **refuted, already optimal** | §4b | do not retry |
| ~~P3~~ | L2 PUSH bytes (incl. the (token, expert-rank) dedup that would have cut 34%) | **refuted, −0.04 ms** | §4c | do not retry — L2 is not PUSH-bound (§4d) |
| ~~X~~ | L1 (token, dst-rank) dedup via `dedup_src_row` | **refuted, −0.02 ms (inside noise)** | §4c | do not retry |
| ~~X~~ | Fold x-quant into the grid | not worth it — x-quant is 0.038 ms | §1 | — |

**Banked so far: 4.695 → 4.48–4.51 ms (−4.3%)** from P4 + P1, both bit-exact. The L1 stage is now
within ~0.08 ms of its no-acquire bound, so the forward's remaining headroom has shifted almost
entirely to L2 (1.90 ms) and the L1 pipeline tail (P5).

With P3 and both dedup variants refuted (§4c), **every remaining byte-reduction lever on the comm
path is closed.** What is left in L2 is the GEMM itself (P2, 73% of the stage) and two drains —
0.358 ms of PUSH and 0.155 ms of reduce (§4d) — neither of which responds to moving fewer bytes.

---

## 8. Reproduce

Campaign harness lives in `agent/workspace/mxfp8_fwd_opt/`:
`run.sh` (container runner), `bench_fwd_fp8.py` (fp8-only legs, ~20 s/point vs ~156 s for the full
compare), `bench_ref_gemm.py` (standalone GEMM roofline), `test_swiglu.py` (SwiGLU bit-exact +
perf gate), `sweep_swiglu_geom.py`, `sweep_l1_inv.py` (L1 acquire gate), `sweep_locality.sh`,
`sweep_cu.sh`, `sweep2.sh`, `probe_inv.sh`, `probe_fwd_overlap.py` (per-stage overlap
decomposition, §4d), `run_probe_guarded.sh` (timeout + worker reaping).
Per-round kernel snapshots and write-ups are under `rounds/round-N/`.

```bash
# One-time: the base image's flydsl is too old; use the turbo_perf image.
ssh smci355-ccs-aus-n05-29 'sudo docker run -d --name=xiaoming-perf --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --privileged --group-add video \
  -v /perf_apps/xiaoming:/perf_apps/xiaoming -w /perf_apps/xiaoming/MegaMoE-dev \
  --entrypoint bash tasimage/primus:v26.3_turbo_perf -lc "sleep infinity"'

cd /perf_apps/xiaoming/MegaMoE-dev
R=agent/workspace/mxfp8_fwd_opt

# Full fp8-vs-bf16 breakdown (§1)
$R/run.sh /tmp/bd.log "python benchmark/ops/bench_fwd_breakdown_compare.py \
  --num-processes 8 --num-tokens 8192 --warmup 10 --iters 30"

# fp8-only legs incl. L1 GEMM/PUSH isolation (§2)
$R/run.sh /tmp/legs.log "python $R/bench_fwd_fp8.py --num-processes 8 --num-tokens 8192 --legs"

# GEMM roofline (§3a)
$R/run.sh /tmp/ref.log "python $R/bench_ref_gemm.py"

# SwiGLU gate (§6) — bit-exact vs the round-1 snapshot, plus A/B timing
$R/run.sh /tmp/sg.log "python $R/test_swiglu.py --ref 1 --reps 3"
$R/run.sh /tmp/geom.log "python $R/sweep_swiglu_geom.py"

# L1 acquire gate (§5) — gemm_inv=2 vs 1 bit-exact, plus the gemm_inv=0 bound
$R/run.sh /tmp/inv.log "python $R/sweep_l1_inv.py"

# End-to-end accuracy on the real fused op (SNR vs bf16, gate >=18 dB)
$R/run.sh /tmp/snr.log "python benchmark/ops/bench_mega_moe_fused_fp8.py"

# L2 leg isolation — env MUST be set at process start (§2, see caveat below)
$R/run.sh /tmp/l2g.log "PT_COMBINE_GEMM_ONLY=1 python $R/bench_fwd_fp8.py --num-processes 8"
$R/run.sh /tmp/l2p.log "PT_COMBINE_PUSH_ONLY=1 python $R/bench_fwd_fp8.py --num-processes 8"

# The buffer_inv probe (§5)
$R/run.sh /tmp/inv.log "bash $R/probe_inv.sh"

# Stage decomposition + the dedup ceiling probes (§4c, §4d). SKIP_MOD=N pushes (N-1)/N of the
# rows, so 3 ~= the 66.1% a dedup would leave. Output is WRONG by construction; read only timing.
G="bash $R/run_probe_guarded.sh $R/probe_fwd_overlap.py"
$R/run.sh /tmp/base.log "$G --mode full"
$R/run.sh /tmp/l1d.log  "MEGA_MOE_PROBE_PUSH_SKIP_MOD=3 $G --mode full"
$R/run.sh /tmp/l2d.log  "MEGA_MOE_PROBE_COMBINE_SKIP_MOD=3 $G --mode full"
$R/run.sh /tmp/l2nr.log "$G --mode noreduce"   # full - noreduce = reduce tail
```

Two traps this campaign lost runs to, both worth knowing before adding a knob:

* **The jit disk cache keys on the `_compile` arguments, not on the environment.** A knob read
  from `os.environ` *inside* a traced closure silently reuses the previous binary — the first
  L1 probe skipped 100% of the pushes and still reported the baseline 1.775 ms. Thread new knobs
  through `_compile` as arguments (as `push_only` / `gemm_only` already are).
* **`torch.multiprocessing.spawn` children carry a generic `spawn_main` cmdline**, so killing the
  parent (or `pgrep`-ing the script name) leaves 8 orphans holding one GPU each and every later
  run blocks. `run_probe_guarded.sh` wraps the probe with a hard timeout and reaps by
  `spawn_main`; the node is shared, so check `ps -eo pid,ppid,args | grep spawn_main` and match
  the parent against your own container before killing anything.

**Measurement caveat.** `grouped_gemm_combine_fp8_kernel._compile()` reads
`PT_COMBINE_GEMM_ONLY` / `PT_COMBINE_PUSH_ONLY` *inside* an `lru_cache`d body without them being
part of the cache key. Flipping them in-process after the first L2 call silently re-runs the FULL
kernel and yields `L2_gemm ≈ L2_push ≈ L2_full`. Always isolate the L2 legs in a fresh process.
(The L1 kernel is fine — it passes `push_only`/`gemm_only` as explicit `_compile` arguments.)

---

## 9. Constraints (unchanged)

| Allowed | Not allowed |
|---------|-------------|
| Single CUDA stream | Extra streams for prep / quant overlap |
| Comm ∥ GEMM inside one fused kernel grid (L1, L2) | Side-stream STEP3-style overlap |
| CU split tuning, tile mapping, cache-op placement | Multi-stream w1/w2/x-quant overlap |
