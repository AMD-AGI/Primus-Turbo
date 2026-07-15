# gfx950 mega-MoE backward (FlyDSL) — reusable tips

## MoE dgrad: do it as NT via a static weight transpose, NOT as an in-kernel NN transpose-read
On gfx950, the mxfp8 scaled MMA (`v_mfma_scale_f32_16x16x128_f8f6f4`, `MfmaScale16x16x128`)
accumulates in **VGPR** and has **no AGPR variant**. The per-tensor fp8 NN kernel's fast B
path (`S2RLoaderTr(inline_asm=True)` + `ds_read_b64_tr_b8`) requires **AGPR** accumulation
(pinned `agpr_alloc>0`), so it CANNOT be combined with the scaled MMA — the fused kernel
fails to compile (`asm_mma=True` + inline tr8). The only NN option for mxfp8 is the intrinsic
`ds_read_tr8_b64` transpose-read, which is no faster than NT (the NN dgrad GEMM is already
near fp8 roofline, ~2109 TFLOPS at M=8192).
=> For MoE dgrad `grad_act = dispatch(dy) @ w2`, transpose the (static) weight
`w2 [G,H,I] -> [G,I,H]` (cached, free) and reuse the near-roofline **NT** tile / the forward
L1 fused NT kernel `dispatch_grouped_gemm_mxfp8` with `dy` as tokens. Validated: identical
grad (cos 0.99921 vs bf16), 1.06x vs bf16 fused, zero new kernel. A bespoke NN mxfp8 tile
(built + validated) was strictly dominated and removed. Matches `mxfp8_grouped_kernel.py`.

## Fused backward STEP1 is comm/overlap-bound, not GEMM-bound
The fused dispatch(dy)+dgrad STEP1 cost (~2.2-2.3 ms) is dominated by the cross-rank fp8
dispatch push + padded-pool GEMM overlap, not the per-tile GEMM (near roofline). Tuning the
tile (tr8 co-schedule etc.) gives ~0. Levers are the CU split (ndcu=16/pscu=16 best, same as
fwd L1; pscu=8 exposes the preshuffle) and reducing padded work.

## Fence granularity in the fused STEP1: hard to reclaim (parallelism trade + masking)
The L2 fences cost ~0.37 ms (17%) at STEP1: gemm-role full-L2 `buffer_inv` = 0.143 ms (2048
per-tile invalidates), preshuffle-role inv+wbl2 = 0.242 ms (~256 each). Isolate with a `diag`
compile toggle (`NO_GEMM=1`, `GEMM_ONLY=2`, `NO_FENCE=4`, `NO_GEMM_FENCE=5`, `COMM_ONLY=8`).
- The acquire (invalidate) only needs to drop stale lines ONCE per workgroup for a single-owner,
  gated reader (preshuffle owns each pool-block, gemm gated per block_m). Hoisting the preshuffle
  invalidate to once/workgroup is correct but NEUTRAL — the gemm's 2048 full-L2 nukes dominate
  device L2 and mask it.
- Cutting gemm invalidate COUNT by putting >1 tile per workgroup (loop block_n) is a DEAD END:
  it collapses the 2048-way parallelism that hides memory latency (256 wg × 8 serial GEMMs →
  3.5 ms, −63%), and sequential `gemm_mxfp8_nt_tile` calls need a barrier between them (shared
  LDS race). Invalidate saving (≤0.143 ms) ≪ serialization cost.
- Parallelism-preserving option = invalidate ONCE PER XCD (first wg on each XCD invalidates, rest
  skip). Needs a hardware XCD/SE-id read (no FlyDSL intrinsic; only `workgroup_id`/`wave_id`
  exist → raw `s_getreg HW_ID` inline-asm + arch bit-decode) + per-XCD flag + atomic CAS/spin.
  High correctness risk on a distributed kernel for ~7% on a comm-bound shape. Not worth it
  unless STEP1 stops being comm-bound.

## STEP1: don't add comm CUs or batch the push to speed the full kernel
Measured on the fused dispatch(dy)+dgrad STEP1 (EP8 MI355X, T=8192 H=7168 I=2048 E=256 K=8):
- comm-only (diag=8) DOES scale with comm CU: ndcu 8/16/24/32 -> 2.367/1.644/1.499/1.404 ms
  (CU-limited in isolation, diminishing returns).
- But the FULL kernel gets WORSE with more comm CU: ndcu 16/24/32 -> 2.170/2.345/2.521 ms. The
  fused kernel shares a fixed ~256-CU budget between comm and gemm; comm CUs are stolen from gemm,
  starving it of CU residency + HBM bandwidth, and that loss exceeds the comm-stage saving.
  ndcu=16/pscu=16 is the sweet spot (same as forward L1).
- "Send 2 tokens per warp-iter" (tok_unroll=2: issue U rows' b128 loads before storing, deeper
  MLP) is NEUTRAL (full 2.170->2.161, comm-only 1.644->1.636): the push is already XGMI-bandwidth-
  bound, so more in-flight loads per warp don't help. (Implemented as a forward-safe `tok_unroll=1`
  default kwarg on `dispatch_fp8_copy_tile`; the tok_unroll>1 path uses a clamped-index load + a
  guarded store so it is numerically identical to the tok_unroll=1 path.)
=> STEP1 is comm-bandwidth/contention-bound. CU split and per-send granularity are dead ends; the
   real lever is reducing the cross-rank push VOLUME / padded-pool work, not moving CUs around.

## FlyDSL gotcha: branch fns passed to _emit_if_then / scf_if_dispatch MUST take 0 args
`ReplaceIfWithDispatch._call_branch` injects `result_names` (an empty tuple `()` when there are no
carried results) into any branch fn that ACCEPTS an argument. A branch fn written with default
params (`def _store(dest_row=dest_row): ...`) receives `dest_row=()` -> `() * Int32 + col` raises
`TypeError: can only concatenate tuple (not "Int32") to tuple`. Write branch fns with ZERO params
and capture values via closure (they run immediately, so loop late-binding is not a problem), or
wrap in a 0-arg helper that takes the values as its own args. This matches the `_one_scale`/`_signal`
idiom already in `ep_fp8.py`.

## Benchmark idiom hides missing-fence bugs — validate fence changes by REASONING
The STEP1 bench reuses the same `dy`/routing every iter, so stale-L2 == fresh-L2 and a kernel
with NO fences (`diag=4`) still passes cos≈0.999. Comparing accuracy on a FRESH `dy_acc` the
timing loop never pushed is a stricter gate, but pool_fp8 (~470 MB) ≫ L2 and the inter-run
`torch.cuda.synchronize` evicts warm lines, so even that can't reliably force the stale
condition. => Do NOT accept a fence-removal/coarsening based on the bench passing; prove it from
the coherence argument (single-owner + gated read + one invalidate drops stale, writeback stays
per-block before each sentinel). A coarsened fence that still invalidates once is safe; skipping
invalidation entirely is not (fails in real training with fresh per-step data).

## Preshuffle-role RELEASE: coalesced write-through beats the whole-L2 l2_writeback (-16.8%, BIG win)
The preshuffle role's release fence (`l2_writeback` = `buffer_wbl2 sc1` whole-L2 flush + `s_waitcnt
vmcnt(0)`) is the STEP1-bwd's main remaining cost. Replace it with: write pool_scale_ps via the shared
LDS transpose `_emit_lds_repack(is_a=True)` (both DRAM sides coalesced b128) using an sc1 WRITE-THROUGH
store (`st_cm=16`), DROP the l2_writeback, and keep the gemm role's `l2_invalidate` acquire (write-
through publishes to the coherent point; gemm invalidate+refill sees it fresh). Measured STEP1-bwd wall
2.17 -> 1.81 ms (-16.8% LB, ~-19% RR), cos PASS on a FRESH dy (LB+RR). Near the comm floor (1.72).
- WHY (rocprofv3, single-rank technique): the old writeback was DRAIN/STALL-bound, not write-bw-bound.
  Full-kernel diag0 baseline vs write-through: SQ_WAIT_ANY -27.9%, GRBM_GUI_ACTIVE -13.8%, TCC_HIT
  -19.6%, TCC_WRREQ_STALL ~flat. `buffer_wbl2` is DEVICE-scope: it flushed the whole L2 and stalled the
  CONCURRENT gemm waves + churned L2. Write-through streams the (coalesced) stores and drops the flush.
- Coalescing is REQUIRED first (scattered write-through = 2.39 ms, tips below). Coalesce-only (cached +
  keep writeback) already gave -6.3% (contiguous dirty lines -> cheaper flush + efficient b128 store).
- PROFILING PITFALL: the diag1 ISOLATION (comm+preshuffle, gemm-exit) is the WRONG lens for a device-
  scope fence — it shows write-through as WORSE (store latency not hidden, no concurrent gemm to benefit
  from dropping the flush). Always profile the FULL kernel (diag0) for device-scope fence changes.
- The LDS transpose tile (64*K128 i32 = 14 KB @ K=7168) fits in a bwd-local shared struct alongside the
  128 KB fp8 ping-pong (-> 142 KB, still 1 block/CU: no occupancy loss). flydsl allows only ONE
  SharedAllocator per kernel, so put the tile IN the same struct (not a 2nd allocate()).
- Switches: PT_MXFP8_PS_COALESCE=1, PT_MXFP8_PS_RELEASE=1 (now default in dispatch_grouped_gemm_mxfp8_bwd).

## STEP1-bwd CU split: re-tune AFTER a role gets cheaper — the optimum MOVES (ndcu=16 -> 24)
The old "ndcu=16/pscu=16 best" was measured on the pre-write-through kernel. After Round 3 made the
preshuffle role cheap, the optimum shifted to **ndcu=24, pscu=8** (-5.2% load_balanced, -2.0%
round_robin, cos PASS both). Lessons:
- ndcu MUST be a MULTIPLE OF 8 (= #XCDs on MI355X). ndcu=20 is a hard CLIFF (2.25 ms vs 1.72 at 24,
  reproduced twice) — uneven comm-block distribution across XCDs. Only sweep ndcu in {16,24,32,...}.
- ndcu=24 beat 16 and 32 (16->1.82, 24->1.72, 32->1.88): 24 lowers the comm floor without over-
  starving the (now-cheaper) gemm; 32 starves it.
- pscu is INSENSITIVE once preshuffle is cheap (pscu 4/8/12/16 at ndcu=24 all ~1.73); pick the small
  end (pscu=8) — it frees CUs for the more gemm-exposed round_robin case (RR: 24/8=1.567 vs 24/12=1.605).
- GENERAL LESSON: whenever a pipeline role's cost changes materially, RE-SWEEP the CU split — a prior
  "optimal" split is stale. The benefit is comm-volume-dependent but 24/8 improved BOTH tested dists.

## The STEP1 comm (push) floor is XGMI-WRITE-BANDWIDTH-bound (counter-confirmed) — no fence lever
An earlier tip asserted STEP1 is comm-bandwidth-bound from wall-time alone; rocprofv3 EA/GMI counters
now PROVE it. Single-rank, comm-only (diag8): `TCC_EA0_WRREQ_GMI_CREDIT_STALL` = 163.8M >> GRBM_GUI_
ACTIVE = 31.3M (~12 XGMI-write-credit stall-cycles per 32B remote write); remote writes dominate
(`TCC_EA0_WRREQ_WRITE_GMI_32B` 13.3M vs `..._WRITE_DRAM_32B` 1.9M = 88% remote). The comm's own
`l2_writeback` is WHERE the XGMI transfer is paid, so it is NOT removable overhead — do NOT try the
Round-3 write-through trick on the comm push (it's bandwidth-bound, not drain-bound; profiled, ruled
out). The raw-scale push is only ~3% of GMI volume (diag8 vs diag16), so streaming/dropping it can't
move the floor (matches the tok_unroll-neutral result). To profile the push: single-rank diag8/diag16 +
`--pmc TCC_EA0_WRREQ_WRITE_GMI_32B TCC_EA0_WRREQ_GMI_CREDIT_STALL TCC_EA0_WRREQ_WRITE_DRAM_32B`. Below
the ~1.72 ms comm floor needs LESS XGMI VOLUME (fp8->fp4 dy push = precision tradeoff; fewer remote
tokens = routing; or faster links) — not a kernel fence change.

## rocprofv3 on a distributed spin-wait kernel: profile ONE rank, not all
Profiling all 8 ranks with `--pmc` serializes every dispatch and DESYNCS the cross-rank scoreboard
handshake -> preshuffle/gemm gate timeouts, GPUs spin at 100%, garbage counters. Instead run 8 ranks
manually (prof_single_rank.py, same MASTER_PORT) and wrap ONLY rank 0 with rocprofv3; ranks 1-7 run
unprofiled so rank 0's peers push fast -> rank 0's spin stays short -> 0 gate timeouts, clean counters.
Also: detached runs that hang leave ORPHAN python children (ppid=1) holding GPU mem at 100% util that
`pkill -f bench_...` misses -> `pkill -9 python` before every measurement. Kernel-trace DURATIONS are
spin-inflated (bf16 shows 14 ms vs 2.37 ms bench) -> use bench CUDA-event wall for timing, PMC for why.

## Preshuffle-role ACQUIRE: a glc (L1-bypass) read replaces buffer_inv (small win, coherent)
The preshuffle role's acquire of the peer-pushed raw pool_scale can drop `l2_invalidate` (buffer_inv
sc1) and instead read raw with cache_modifier=1 (glc, globally-coherent L1-bypass). This is coherent
because the COMM role already `l2_writeback`s the pushed scale to the device-coherent point BEFORE the
sys-scope scoreboard signal the preshuffle gates on — so a glc read observes it fresh. Validated on a
FRESH dy the timing loop never pushed (cos PASS, load_balanced AND round_robin, repeated). Win is
small (~-1% of the STEP1 wall) because this is only the CHEAP acquire fence. IMPORTANT distinction vs
the "don't remove fences" tip: glc is NOT a naive cached-read fence removal — glc IS the coherence
mechanism (the read goes to the coherent point). A plain cached (cm=0) read WOULD read stale. Do NOT
use sc1 (16/17) on a LOAD (measured 23.8 ms — sc1-on-load is catastrophic). `PT_MXFP8_PS_READ_CM=1`
is now the default in `dispatch_grouped_gemm_mxfp8_bwd`.

## FRESH-dy coherence gate: the SCALE region IS L2-resident, so scale-fence bugs ARE catchable
An earlier tip said even a fresh dy_acc "can't reliably force the stale condition" — that is true for
pool_FP8 (~470 MB ≫ L2) but NOT for pool_SCALE (small, L2-resident). To catch a SCALE-fence bug:
run the fp8 kernel on the timing dyq (warms L2 with its scale), then IMMEDIATELY (only cuda.sync +
group.barrier between, no cache flush) run it on a DIFFERENT dy_acc and compare vs bf16(dy_acc); a
missing/insufficient scale acquire reads the stale (dyq) scale -> cos collapses. Implemented as
`PT_MXFP8_ACC_FRESH` (default 1) in `bench_dispatch_grouped_gemm_mxfp8_nn.py`, running the fp8 acc
FIRST (before the big bf16 push evicts warm scale lines). Calibrated: the proper-fence baseline PASSES
(cos 0.999), so it is not a false-positive gate. Use it to gate ALL scale-fence changes.

## Do NOT fold the mxfp8 A-scale preshuffle into the gemm workgroup (2-stage) — dead end
Tried removing the separate preshuffle stage + its 2 L2 fences (~0.24 ms) by having each gemm WG
preshuffle its own tile's A-scale into a write-through global scratch, then read it back via the
fast `ScaleS2R` (`PT_MXFP8_BWD_2STAGE=2`, FUSED_PS). Correct (cos 0.999 on fresh dy) but REGRESSED
to 3.66 ms vs 2.15 ms 3-stage (−69%). Three compounding costs the dedicated preshuffle role avoids:
1. **8× redundant** — `n_blocks = N/BLOCK_N` gemm tiles share one `block_m`'s A-scale, but each
   tile re-preshuffles all 256 rows (2048 tiles vs the role's 256 block_m).
2. **write-through is uncached / HBM-latency** — to survive concurrent device-wide `buffer_inv sc1`
   from other WGs' acquires, the 224 KB (4×-inflated broadcast) scratch MUST be sc1 write-through
   (~458 MB HBM-latency stores, no L2 coalescing). A cached store + `l2_writeback` is the very fence
   being removed; a plain store is unsafe (another WG's device-wide invalidate drops the dirty line).
3. **not overlapped** — serial prologue before every tile's K-loop; the role overlaps with comm+gemm.
The raw on-the-fly variant (`=1`, `ScaleS2RRaw`) is also net-negative (~2.8–3.1 ms, scattered loads).
=> The mxfp8 A-scale transform is intrinsically **per-block_m**; a dedicated preshuffle ROLE that
dedups to once/block + uses cached stores + overlaps is MORE efficient than the 2 fences it costs.
The 3-stage pipeline (comm → preshuffle-role → gemm) is the efficient structure; stop trying to
collapse it to 2 stages. LDS-resident preshuffle is also infeasible: one tile's broadcast A-scale
is 224 KB (raw 56 KB) vs only ~32 KB LDS headroom (gemm already uses 128 KB of the 160 KB/CU).

## fp8 STEP1 dgrad does not transfer alone — needs fp8 dW2 wgrad
The bf16 STEP1 produces both `grad_swiglu` and `dispatch_l2_grad` (bf16 scattered dy);
`dispatch_l2_grad` feeds `dW2`. fp8 STEP1 pushes dy fp8-along-H, so keeping dW2 bf16 needs a
dequant (~0.2 ms) that erases the STEP1 win. The coherent unit is STEP1 dgrad + dW2 wgrad;
dgrad quantizes along H, wgrad contracts over the pool/token axis (quantize along pool), so
they need different quantizations of dy (no free pool reuse). Use `grouped_quantize_fp8_with_trans`
(dual-axis) + `grouped_gemm_fp8_variable_k_impl(MX_BLOCKWISE)` for dW2, mirroring
`MXFP8GroupedGEMMFunction.backward` (validated). fp8 weight-grad prefers E5M2 + SNR gate.

## dW2 colwise (re)quant: wider packed-i32 loads REGRESS it (occupancy) — do not retry
The colwise transpose-(re)quant (`quant_colwise_trans_flydsl`, the dW2 `a`/`b` operands) is
LATENCY-bound and its scalar `vec_width=1` i8 path already MAXES occupancy (VGPR≈36, ~16 waves/CU,
LDS-limited). Widening the strided column read to VW=4 (each thread owns 4 columns, read as one
packed i32, unpack via `cvt_f32_fp8(.., byte)`; MB coupled to VW*MB=4 so LDS is held constant) makes
the requant **+80% slower** (0.337→0.607 ms, full dW2 +20%), byte-identical output. Root cause: the
per-column block-amax needs all 32 vals/column LIVE, so VW=4 holds 4×32 f32 + 32 hoisted i32 loads
→ VGPR ~3-4x → VGPR-occupancy-limited → the latency-bound kernel loses the resident waves that are
its ONLY latency-hiding mechanism. LDS was held constant but VGPR is the binding constraint. Wider
loads are the WRONG lever here; VW=2 has the same mechanism (less severe, still a regression risk).
The (re)quant is at its practical single-pass limit — the dW2 wgrad wins are the fp8 GEMM (near
roofline) and producer-fusion (emit colwise-fp8 from swiglu_backward), NOT the standalone quant.
