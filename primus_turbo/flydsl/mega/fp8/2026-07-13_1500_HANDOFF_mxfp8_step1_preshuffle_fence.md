# HANDOFF PROMPT — optimize fused STEP1 mxfp8 dgrad by cutting the preshuffle role's L2 fences

> Paste this whole file as the opening prompt of a new session. It is self-contained.
> Target GPU: MI355X / gfx950. Run everything in the `xiaoming-dev` docker container.

## Your goal
Reduce the wall time of the fused **STEP1 backward** kernel (dispatch(dy) PUSH + fc2 dgrad
grouped mxfp8 GEMM, NT-reuse) in `primus_turbo/flydsl/mega/fp8/dispatch_grouped_gemm_mxfp8_bwd_kernel.py`,
without breaking correctness. The baseline is **2.17 ms** (fp8) vs 2.29 ms (bf16) = 1.13x.
The proven headroom is **~0.5 ms** (getting to ~1.65 ms = 1.4x). The lever is the **preshuffle
role's device-scope L2 fences**, NOT the transpose compute.

## What is already proven (data, 8xMI355X EP8 T=8192 H=7168 I=2048, ndcu=pscu=16, load_balanced)
Breakdown via in-kernel DIAG (env `PT_MXFP8_BWD_DIAG`):
- comm-only (diag=8):                 **1.635 ms**  (comm floor)
- comm+preshuffle, NO fences (diag=5):**1.705 ms**  => preshuffle transpose+write = +0.056 ms (TINY)
- comm+preshuffle, WITH fences(diag=1):**1.986 ms** => preshuffle's L2 fences = **+0.281 ms** (the cost)
- full 3-stage (diag=0):              **2.170/2.174 ms** (baseline; gemm role adds +0.184)
- skip preshuffle role entirely (two_stage=3): **1.644 ms** (=comm floor) => removing the
  preshuffle stage lets the gemm fully hide under comm. Headroom = 2.174 - 1.644 = **0.53 ms**.

So: the preshuffle *transpose* is tiny; the slowness is its 2 device-scope L2 fences per tile
(16 rounds/CU => 32 fence ops/CU). Splitting the fence (see `prims.py`):
- `l2_invalidate()` = `buffer_inv sc1` (no drain, CHEAP).
- `l2_writeback()`  = `buffer_wbl2 sc1 + s_waitcnt vmcnt(0)` (whole-L2 flush + full vmem drain, EXPENSIVE).

## The breakthrough so far (acquire side) — cache_modifier=1 (glc)
The preshuffle role's ACQUIRE `l2_invalidate` (see the peer-pushed raw pool_scale) can be replaced
by an **uncached glc read** of the raw scale + dropping the invalidate. Validated:
`PT_MXFP8_PS_READ_CM=1` (glc, L1-bypass read; skips the acquire invalidate):
  **2.144 ms, cos=0.99910 PASS** (correct AND slightly faster than 2.174).
  - cm=3 (glc+nt) ~2.167 (baseline-ish); cm=16/17 (sc1) = 23.8 ms (sc1 on a LOAD is catastrophic).
This is only the CHEAP acquire fence, so the win is small (-0.03 ms), but it VALIDATES the
"uncached data + scoreboard-only, no bulk fence" direction for the acquire.
IMPORTANT lesson: `cache_modifier=1`=glc (L1-bypass, for coherent READS); `=16`=sc1 (write-through,
for STORES). Do NOT use 16 on a load (it silently read stale earlier -> cos FAIL, and is slow).

## THE MAIN REMAINING LEVER — the release-side writeback (expensive fence)
The big cost is the preshuffle `l2_writeback` (release of `pool_scale_ps` to the gemm role) +
the gemm role's own `l2_invalidate` (acquire of pool_fp8 + pool_scale_ps). To cash the 0.28 ms:
1. **Coalesce the preshuffle WRITE first.** `preshuffle_a_scale_tile` (`ep_fp8.py`) currently does
   a SCATTERED broadcast write (stride-256, 4x data). Cached, that is cheap (L2 absorbs it), but
   any uncached/write-through store of a scattered pattern is slow (measured: write-through gave
   2.39 ms). The efficient transpose is `_emit_lds_repack` (`utils/gemm_helper.py`) — LDS wave-lane
   transpose with BOTH DRAM sides coalesced. Rewrite the preshuffle-role write to use an LDS-tiled
   coalesced transpose (reuse `_emit_lds_repack` logic; the preshuffle workgroup has 128 KB LDS
   free — it is not doing the GEMM).
2. THEN make the release coherent WITHOUT the whole-L2 `buffer_wbl2`: a coalesced glc/uncached
   store to the device-coherent point + `s_waitcnt vmcnt(0)` before the SENTINEL, and have the
   gemm read `pool_scale_ps` with glc (cache_modifier=1) so its `l2_invalidate` can drop the
   pool_scale_ps part. (The gemm still needs pool_fp8 fresh — that is a separate acquire; the
   gemm reads fp8 via `buffer_load_lds` DMA. Handle it or keep one acquire for fp8 only.)
3. Alternative if coalesced-store coherence is hard: reduce writeback FREQUENCY — batch several
   tiles' writes then one `l2_writeback` + delayed SENTINELs (trades some comm∥gemm overlap; must
   measure the net).

## Available experiment switches (all env, DEFAULT = baseline, safe)
In `dispatch_grouped_gemm_mxfp8_bwd_kernel.py`:
- `PT_MXFP8_BWD_DIAG`: 0 normal | 1 comm+ps | 2 gemm-only | 4 no-fence | 5 comm+ps no-fence | 8 comm-only | 16 data-only.
- `PT_MXFP8_BWD_2STAGE`: 0 3-stage(default) | 1 RAW_2STAGE | 2 FUSED_PS | 3 SKIP_PS_DIAG(garbage, timing) | 4 LDS_STREAM(Plan A, correct but slow).
- `PT_MXFP8_LDS_KT`: streaming A-scale window for two_stage=4.
- `PT_MXFP8_PS_READ_CM`: preshuffle raw-scale read CPOL; !=0 skips the acquire invalidate (1=glc validated, correct+small-win).

## How to run (hang-safe — the bf16 kernel has an INTERMITTENT L1 scoreboard-liveness stall)
The bench runs bf16 FIRST; it occasionally hangs (pre-existing bf16 bug, `MEGA dispatch GEMM gate
timeout` from `dispatch_grouped_gemm_bf16_kernel.py`) — NOT your fault. Retry with a fresh port.
```bash
docker exec xiaoming-dev bash -lc 'cd /mnt/shared/xiaoming/Primus-Turbo && \
  for t in 1 2 3; do MEGA_BENCH_TIMEOUT_S=90 MASTER_PORT=$((8900+t)) PYTHONPATH=$PWD \
  PT_MXFP8_PS_READ_CM=1 \
  timeout 130 python benchmark/ops/bench_dispatch_grouped_gemm_mxfp8_nn.py \
    --num-processes 8 --mode load_balanced --iters 30 --num-dispatch-cu 16 --num-preshuffle-cu 16 \
    2>&1 | grep -E "fp8 NT-reuse" && break || echo "try $t hung (bf16 stall), retry"; done'
```
Read the `fp8 NT-reuse` line: `ms | ... | Nx vs bf16 | cos=... PASS/FAIL`.

## HARD RULES / pitfalls
- **Correctness is cross-XCD coherence.** A wrong fence removal SILENTLY reads stale. ALWAYS check
  `cos` — the bench compares against bf16 using a FRESH `dy_acc` the timing loop never pushed
  (stale-read guard). `cos >= 0.99` = PASS. Do not trust a single PASS for a coherence change:
  re-run 2-3x and also try `--mode round_robin` before believing it.
- **cache_modifier values**: 1=glc(read, L1-bypass), 16=sc1(store write-through), 17=glc+sc1
  (23 ms on load — avoid). Verify semantics; do not reuse a store modifier on a load.
- **One variable per round; snapshot + measure each** (see `.cursor/rules/iteration_rules.mdc`).
- **Do NOT redo**: Plan A (stage A-scale into LDS in the gemm, streaming) — measured NET-NEGATIVE
  (2.9 ms; the per-window barrier perturbs the fp8 pipeline). broadcast-direct-push — lost (fwd
  campaign). FUSED_PS — 3.66 ms. invalidate-once and scattered write-through — both regressed.

## Files
- Kernel: `primus_turbo/flydsl/mega/fp8/dispatch_grouped_gemm_mxfp8_bwd_kernel.py` (preshuffle role
  ~L205-260, gemm role ~L245-350, switches at top of `_compile` + backward fn env reads).
- Transpose: `preshuffle_a_scale_tile` in `ep_fp8.py` (scattered — rewrite to coalesced);
  `_emit_lds_repack` in `utils/gemm_helper.py` (the coalesced LDS-transpose reference).
- Fences/scoreboard: `prims.py` (`l2_invalidate`, `l2_writeback`, `ld`/`st`).
- Cache modifier is the raw buffer-intrinsic aux immediate (`buffer_ops.buffer_load/store`).
- Campaign log + full data: `agent/workspace/gemm_mxfp8_lds_scale_gfx950_20260713/logs/optimize.md`.

## First concrete step
1. Reproduce the breakdown (diag 8/1/5/0) and `PT_MXFP8_PS_READ_CM=1` (expect 2.144, cos PASS).
2. Make `PT_MXFP8_PS_READ_CM=1` the default after hardening its coherence (2-3 runs + round_robin).
3. Then take on the release-side writeback: coalesce the preshuffle write (LDS transpose), then
   remove `l2_writeback` with a glc/coalesced release + glc gemm read; measure cos + wall each step.
