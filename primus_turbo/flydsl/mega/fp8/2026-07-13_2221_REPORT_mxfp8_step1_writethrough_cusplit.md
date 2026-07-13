# REPORT / HANDOFF — fused STEP1 bwd mxfp8 dgrad: preshuffle-fence campaign DONE (-22%)

> Supersedes `2026-07-13_1500_HANDOFF_mxfp8_step1_preshuffle_fence.md`. Self-contained. Target GPU:
> MI355X / gfx950 (OCP fp8). Run in the `xiaoming-dev` container. Kernel:
> `primus_turbo/flydsl/mega/fp8/dispatch_grouped_gemm_mxfp8_bwd_kernel.py` (a fork of the forward
> kernel so the forward stays byte-identical).

## Result

Fused STEP1 backward = dispatch(dy) cross-rank PUSH + fc2 dgrad grouped mxfp8 GEMM (NT-reuse),
8xMI355X EP8, T=8192 H=7168 I=2048 E=256 K=8. Metric = fp8 STEP1-bwd wall (ms), gate = cos>=0.99 vs bf16.

| round | change | LB wall | vs baseline | commit |
|-------|--------|---------|-------------|--------|
| R1 | baseline (committed a262499) | 2.211 ms | — | a262499 |
| R2 | glc acquire (preshuffle: skip buffer_inv, coherent glc read) | 2.196 | -0.7% | 769a799 |
| R3 | **coalesced LDS-transpose WRITE-THROUGH release; drop whole-L2 l2_writeback** | **1.81** | **-16.8%** | 769a799 |
| R4 | drop gemm-role l2_invalidate | (not shipped) | REJECTED | — |
| R5 | **CU split ndcu=24/pscu=8 (was 16/16)** | **1.724** | **-22.0%** | e861838 |

round_robin also improved (16/16 1.60 -> 24/8 1.567, -2.0%). cos PASS on LB AND RR every accepted round
(fresh-dy gate + coherence argument). **1.724 ms is at the XGMI-write-bandwidth comm floor** — see below.

### R6 — same R2+R3 ported to the FORWARD L1 kernel (`dispatch_grouped_gemm_mxfp8_kernel.py`)
The forward is the SAME 3-stage pipeline and is wired into the PRODUCTION MoE forward
(`mega_moe_fused_mxfp8.py`). Ported the glc acquire + coalesced write-through release (same env
switches, default on; local `_make_fwd_shared_storage_coalesce`). Forward L1 bench (fresh-x gate):
LB 2.610 -> 2.317 ms (**-11.2%**), RR 2.391 -> 2.148 ms (**-10.2%**); cos_vs_decoupled-ref = **1.00000**
(bit-exact on FRESH x = strongest coherence proof), cos_vs_bf16 ~0.999. Smaller % than the bwd (fwd
GEMM is 2x, N=4096, so the fence is a smaller fraction). Forward CU split left at 16/16 (module default;
a forward-specific CU re-sweep like R5 is future work — N differs so re-sweep, don't assume 24/8).

## Accepted code state (defaults now on)
- `PT_MXFP8_PS_READ_CM=1` (R2): preshuffle acquire = glc coherent read of raw pool_scale, no buffer_inv.
- `PT_MXFP8_PS_COALESCE=1`, `PT_MXFP8_PS_RELEASE=1` (R3): preshuffle writes pool_scale_ps via the shared
  LDS transpose `_emit_lds_repack(is_a=True)` (both DRAM sides coalesced b128) with sc1 WRITE-THROUGH
  (st_cm=16), and DROPS the whole-L2 `l2_writeback`. Gemm role's `l2_invalidate` acquire is UNCHANGED
  (write-through publishes to the coherent point; gemm invalidate+refill sees it fresh). The 14 KB
  transpose tile shares the bwd storage struct (`_make_bwd_shared_storage_coalesce`) -> still 1 block/CU.
- `num_dispatch_cu=24, num_preshuffle_cu=8` (R5): default launch split (bwd fn + bench argparse).
- Diagnostic-only (default OFF): `PT_MXFP8_GEMM_ACQ=1` (skip gemm invalidate, INCORRECT timing probe).

## How to run (hang-safe)
The bench runs bf16 FIRST; it (and occasionally the fp8 path) hits an INTERMITTENT scoreboard-liveness
stall — retry with a fresh port. Also `pkill -9 python` before each run: detached runs that hang leave
ORPHAN spawn children holding the GPUs at 100%.
```bash
docker exec xiaoming-dev bash -lc 'cd /perf_apps/xiaoming/Primus-Turbo && pkill -9 python; sleep 3; \
  for t in 1 2 3; do MEGA_BENCH_TIMEOUT_S=90 MASTER_PORT=$((9000+RANDOM%900)) PYTHONPATH=$PWD \
  LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages timeout -k 15 240 \
  python benchmark/ops/bench_dispatch_grouped_gemm_mxfp8_nn.py --num-processes 8 --mode load_balanced \
    --iters 30 2>&1 | grep -E "fp8 NT-reuse" && break || echo "retry $t"; done'
```
Read `fp8 NT-reuse`: `ms | ... | Nx vs bf16 | cos=... PASS/FAIL`. Bench harness this session:
`PT_MXFP8_BWD_FORK=1` (default) points the fp8 line at the bwd fork; `PT_MXFP8_ACC_FRESH=1` (default) is
the FRESH-dy coherence gate (compare on a dy the timing loop never pushed — the committed bench had
lost this; it is restored). Env container note: repo Python + prebuilt dev23 `_C` are linked via
`agent/workspace/gemm_mxfp8_lds_scale_gfx950_20260713/setup_env.sh` (+ LD_LIBRARY_PATH to the venv).

## Profiler evidence (rocprofv3; the campaign was profiler-driven)
- **Profiling a distributed spin-wait kernel**: profiling all 8 ranks with `--pmc` desyncs the cross-rank
  scoreboards (gate timeouts, garbage). Profile ONLY rank 0 (`prof_single_rank.py` + `run_profile_1rank.sh`);
  peers run unprofiled so rank 0's spin stays short -> clean counters. Kernel-trace DURATIONS are
  spin-inflated -> use bench CUDA-event wall for timing, PMC for the "why".
- **R3 (why write-through won)**: full-kernel diag0 baseline vs write-through: SQ_WAIT_ANY -27.9%,
  GRBM_GUI_ACTIVE -13.8%, TCC_HIT -19.6%, TCC_WRREQ_STALL ~flat. The old `buffer_wbl2` is a DEVICE-wide
  L2 flush that stalled the concurrent gemm waves + churned L2 — drain-bound, not write-bandwidth-bound.
  PITFALL: the diag1 ISOLATION (comm+ps, gemm-exit) falsely shows write-through as WORSE (store latency
  unhidden; a device flush only hurts under concurrency). Profile the FULL kernel for device-scope fences.
- **comm floor (why it's a hard floor)**: comm-only (diag8) EA/GMI counters: `WRREQ_GMI_CREDIT_STALL`
  163.8M >> busy 31.3M (~12 XGMI-write-credit stalls per 32B remote write); 88% of writes are remote
  (GMI 13.3M vs DRAM 1.9M); scale push is only ~3% of GMI volume. The push is XGMI-WRITE-BANDWIDTH-bound;
  the comm's own l2_writeback IS the XGMI transfer (not removable overhead).

## Rejected / DO NOT RETRY
- **Drop the gemm-role l2_invalidate (R4)**: ceiling only -0.09 ms (-5%); UNVERIFIABLE — pool_fp8
  (~470 MB) >> L2 so the fresh-dy gate can't force the stale condition (the probe "passed" cos while
  incorrect); and it touches the SHARED forward loaders (G2SLoader/ScaleS2R). Not worth it while comm-bound.
- **write-through / drop the COMM push l2_writeback**: comm is XGMI-bandwidth-bound (counter-proven), NOT
  drain-bound -> no overhead to remove. Different from the preshuffle release.
- **ndcu not a multiple of 8**: ndcu=20 is a hard CLIFF (2.25 ms, reproduced) — uneven comm blocks across
  the 8 XCDs. Only sweep ndcu in {16,24,32}. ndcu=32 starves the gemm (1.88).
- (from prior tips, still true) FUSED_PS (fold preshuffle into gemm) 3.66 ms; LDS-stream A-scale 2.9 ms;
  more comm CUs on the PRE-R3 kernel. scattered write-through 2.39 ms (must coalesce first).

## Real-training transfer (Rule 11 re-attribution)
All accepted rounds are K1 kernel-side / launch-config changes (fence granularity, scale layout, CU
split) — no id()-keyed activation/grad cache, no compile-time M_per_group, tokens_per_expert stays a
runtime tensor. R5 verified on BOTH load_balanced and round_robin (not distribution-overfit).
S_real = full -22% (transfers 1:1); W_real = 0; R_real = 0. Inflation gap ~0.

## The one remaining lever (next campaign, out of this fence scope)
Below the ~1.72 ms comm floor needs LESS XGMI push VOLUME:
1. **fp4 dy push** (fp8 -> fp4 halves the XGMI write volume). PRECISION/numerics change: dy for dgrad
   may be too lossy in fp4 — needs an SNR gate (E5M2/fp4 recipe; see the wgrad tip about fp4 + SNR) and a
   real accuracy campaign, not just cos. Biggest potential (~halve the comm floor) but highest risk.
2. **dynamic ndcu(remote_rows)**: pick the CU split at launch from the actual remote-token count (24 is
   the static optimum for this shape/comm volume; a lighter shape wants fewer comm CUs).
3. faster interconnect (HW).

## Files / key locations
- Kernel: `dispatch_grouped_gemm_mxfp8_bwd_kernel.py` (preshuffle role ~L300-345 write/release; gemm
  acquire ~L416; env reads + `_make_bwd_shared_storage_coalesce` + geometry near top of `_compile`).
- Transpose: `_emit_lds_repack` (+ `rd_cm`/`st_cm`) in `utils/gemm_helper.py`; `preshuffle_a_scale_tile`
  (old scattered path) in `ep_fp8.py`; comm role `dispatch_fp8_copy_tile` (its l2_writeback ~L251) in `ep_fp8.py`.
- Fences: `prims.py` (`l2_invalidate`=buffer_inv sc1; `l2_writeback`=buffer_wbl2 sc1 + drain).
- Bench: `benchmark/ops/bench_dispatch_grouped_gemm_mxfp8_nn.py` (PT_MXFP8_BWD_FORK / PT_MXFP8_ACC_FRESH).
- Campaign logs/profiles/snapshots/run-scripts: `agent/workspace/gemm_mxfp8_lds_scale_gfx950_20260713/`.
- Reusable tips: `agent/historical_experience/gfx950/mega_moe_bwd/flydsl/tips.md`.
