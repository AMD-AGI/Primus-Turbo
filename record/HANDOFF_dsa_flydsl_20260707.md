# HANDOFF — FlyDSL DeepSeek-V4 sparse-MLA fwd+bwd optimization (2026-07-07)

> Read this first in the next session. It is the single source of truth for where
> the FlyDSL fused single-latent (K==V) sparse-MLA attention stands, what's fast,
> what's gated, what's a dead end, and the ranked next levers.
> Deeper per-topic detail: the three `record/dsa_*_20260707.md` docs. Rolling state:
> memory `project_flydsl_env_fix.md` + `project_csa_flydsl_goal.md`.

---
## 0. ENVIRONMENT — DO THIS FIRST (else nothing compiles)
`pip install --force-reinstall --no-deps flydsl==0.2.2`
Verify: `python -c "import flydsl; print(flydsl.__version__)"` == 0.2.2 AND
`hasattr(flydsl.expr.rocdl,'raw_ptr_buffer_load_lds')` == True.
Machine: MI355X gfx950 (CDNA4), bf16. flydsl index type is 32-bit → use explicit
i64 for offsets/tensors that can exceed 2^31 (bytes OR elements).

---
## 1. HEADLINE RESULTS (MI355X, real adapter structured topk, sink on, warmed)

### Forward (TFLOP/s, higher better)
| shape        | flydsl(tr16) | triton_v2 | gluon_v2 | tr16/gluon | tr16/triton |
|--------------|-------------:|----------:|---------:|-----------:|------------:|
| H128 K512    | 375          | 380       | 414      | 0.91x      | 0.99x       |
| H64  K512    | 329          | 377       | 413      | 0.80x      | 0.87x       |
| H128 K2048   | 388          | 403       | 440      | 0.88x      | 0.96x       |

### Backward (ms, lower better / TFLOP/s)
| shape        | flydsl bwd     | triton_v2 | gluon_v2 | flydsl/triton |
|--------------|---------------:|----------:|---------:|--------------:|
| H128 K512    | 14.5 ms (174)  | 9.2 (273) | 8.4 (301)| 0.64x         |
| H64  K512    | 8.8 ms (144)   | 6.8 (187) | 6.9 (184)| 0.77x         |
| H128 K2048   | 33.5 ms (186)  | 26.4 (236)| 24.2 (258)| 0.79x        |

Correctness: fwd out SNR 47.5 dB vs M=32 ref / 82 dB vs gluon; bwd dq 51-80 dB,
dkv 67-70 dB vs triton. **All 138 CSA tests pass** with every flydsl path enabled.

### The arc this session
- fwd: was 131 (naked M=16) / 278 (old M=32 default) → **375-388** (tr16 shared gather).
- bwd: was 0.32-0.36x triton (H128) → **0.64-0.79x** (M=16 dq kernel).

---
## 2. HOW TO ENABLE / BENCH (all optimizations are ENV-GATED, default OFF)

Production default is STILL the old M=32 fwd + M=32 dq bwd (safe). Turn on the wins:
```
export PRIMUS_DSA_FLYDSL_FWD_TR16=1        # fwd: M=16 shared-gather tr16 kernel (the 375 win)
export PRIMUS_DSA_FLYDSL_BWD_DQ_M16=1      # bwd: M=16 dq kernel (the 2.9x dq win)
```
Benchmarks (in output/):
- `bench_3way.py`  — fwd+bwd flydsl vs triton_v2 vs gluon (the table above). Sets TR16=1 itself; set BWD_DQ_M16=1 in env.
- `harness_tr16.py` — fwd tr16 vs M=32 ref, SNR + TFLOP/s (bit-exact regression guard).
- `harness_bwd.py`  — bwd dq/dkv/dsink SNR vs triton + TFLOP/s.
- `bench_h2h.py` / `bench_fused_fwd.py` — older fwd-only harnesses.
gluon import needs `sys.path` to `/apps/tas/yaoc/agent_work/mi355x/flydsl-dpskv4-attn/Primus` (already in the harnesses).
ALWAYS warm gluon (its @triton.autotune needs ≥2 runs) — harnesses do this.

### All env knobs
| env | default | effect |
|---|---|---|
| PRIMUS_DSA_FLYDSL_FWD_TR16 | 0 | use tr16 M=16 fwd (WIN, turn ON) |
| PRIMUS_DSA_FLYDSL_BWD_DQ_M16 | 0 | use M=16 dq bwd (WIN, turn ON) |
| PRIMUS_DSA_TR16_DMA | 0 | fwd DMA gather (correct, SLOWER — dead end) |
| PRIMUS_DSA_TR16_PIPE | 0 | fwd 2-buf softmax‖QK pipeline (correct, ~0 gain) |
| PRIMUS_DSA_TR16_QLDS | 0 | fwd Q in LDS (occ2 but slower) |
| PRIMUS_DSA_TR16_QK_PF | 3 | fwd QK LDS prefetch depth |
| PRIMUS_DSA_TR16_BLOCK_K | 32 | fwd tile keys (32 optimal; 64 halves perf) |
| PRIMUS_DSA_M16_WAVES | 0(auto) | fwd/dq waves per workgroup (auto = max) |
| PRIMUS_DSA_BWD_DQ_QK_PF | 1 | bwd dq QK prefetch depth |
| PRIMUS_DSA_FLYDSL_BWD_INTERM_MFMA | 0 | bwd FlyDSL dkv-interm (correct, SLOWER — gated ref) |
| PRIMUS_DSA_INTERM_DBLOCK | 128 | bwd interm D-block size |

---
## 3. KEY FILES
Kernels (`primus_turbo/flydsl/attention/kernels/sparse_mla_v2/`):
- `dsa_fwd_tr16_kernel.py`  — **fwd WIN**. M=16 16x16x32 QK + ds_read_tr PV + SHARED gather.
  Also has gated DMA + 2-buffer pipeline scaffolding (both correct, neither wins — see §5).
- `dsa_bwd_dq_m16_kernel.py` — **bwd dq WIN**. Same recipe: S=Q·Kᵀ + dP=dO·Kᵀ (shared kv
  A-op), elementwise dS, dQ=dS·K via ds_read_tr, shared gather.
- `dsa_bwd_dkv_interm_kernel.py` — dkv-interm FlyDSL port (gated OFF, correct-but-slower).
- `dsa_fwd.py` / `dsa_bwd.py` — launchers + dispatch + env gating + kernel caches.
- `dsa_fwd_kernel.py` (M=32 fwd, old default), `dsa_fwd_pipe_kernel.py` (M=32+DMA),
  `dsa_fwd_m16_kernel.py` (FAILED early M=16, no ds_read_tr — kept for history),
  `dsa_bwd_dq_kernel.py` (M=32 dq, old default), `adapter.py` (topk/kv bridging).
Records (`record/`): this file + `dsa_fwd_pathB_m16_tr16_`, `dsa_bwd_dq_m16_`,
  `dsa_bwd_dkv_interm_` (all _20260707.md).
Checkpoint: `output/flydsl_v2_ckpt/dsa_fwd_tr16_kernel_shared373.py` (clean 373 fwd).
gluon reference: `/apps/tas/yaoc/agent_work/mi355x/flydsl-dpskv4-attn/Primus/primus/
  backends/megatron/core/transformer/v4_attention_kernels/_gluon_v2/dsa_{fwd,bwd_dq,
  bwd_dkv_interm}_*.py`.

---
## 4. THE WINNING RECIPE (why it works — reuse this pattern)
The one idea behind both wins: **M=16 16x16x32 + a per-workgroup SHARED kv tile +
ds_read_tr hardware transpose for the second GEMM.**
- topk depends only on (token), so ALL waves in a CTA gather the SAME kv rows →
  gather ONCE into shared LDS, read the QK A-operand FROM LDS (not redundant per-wave
  HBM). This single change flipped naked-M=16 fwd 131 → 373. Redundant HBM traffic was
  the real bottleneck, NOT occupancy.
- The second GEMM (PV in fwd, dQ=dS·K in bwd) reads the SAME row-major shared tile via
  `ds_read_tr16_b64` (hardware 4x4 transpose). Lane map (verified): row-major V[key][d],
  `k_row=(L//16)*8 + (L%16)//4`, `d_col=dt*16 + (L%4)*4`, second read at +4*V_STRIDE,
  shuffle to vec8 → 16x16x32 A-operand A[d=L%16, key=(L//16)*8+e].
- M=16 acc [16,512]f32 = 128 VGPR (vs M=32's 256). bwd dq: VGPR 512→256, spill 196→55.

---
## 5. DEAD ENDS — do NOT re-try these (each cost real time this session)
1. **Naked M=16 without shared gather** — 131 TFLOP/s. Redundant per-wave gather. (The
   old `dsa_fwd_m16_kernel.py` also dropped ds_read_tr → doubly bad.)
2. **Occupancy-2 via Q-in-LDS / amdgpu-waves-per-eu** — reaches occ2 (VGPR 226) but
   perf DROPS. acc is VALU-rescaled each tile so it MUST stay in arch VGPR (AGPR idle).
   "Occupancy is necessary but NOT sufficient" — proven twice now.
3. **fwd DMA gather (raw_ptr_buffer_load_lds)** — correct (99/47 dB) but 105-118 vs coop
   376. Tried 4 variants incl. exact M=32-style (uniform per-wave lds_ptr via readfirstlane,
   1 wave-batch = 1 key row, unpadded stride, swizzle round-trip). topk-SCATTERED rows get
   NO throughput edge from DMA; its only benefit is async overlap, which M=16's small
   per-wave compute can't hide (M=32 CAN — that's why DMA gives M=32 +6-16% but M=16 zero).
4. **fwd softmax‖QK 2-buffer pipeline** — correct but nets ~0 (372 vs 376): coop gather is
   synchronous, its 2 barriers/tile cost ≈ the overlap saved. Would need async DMA (see #3).
5. **fwd BLOCK_K=64** — halves perf (191). BLOCK_K=32 is optimal.
6. **bwd dkv-intermediate FlyDSL MFMA (single-wave)** — correct (dkv 69 dB) but bwd
   14.6→22.6 ms. Single wave/token serializes the qT/doT transpose staging (128h×512d
   scalar LDS stores) that Triton spreads over 4 warps. D_BLOCK sweep confirmed staging-bound.

---
## 6. NEXT LEVERS (ranked by expected value)
1. **Make tr16 fwd + M=16 dq bwd the PRODUCTION DEFAULT** (currently env-gated). They
   beat the M=32 defaults (fwd 1.35x, bwd 2.9x dq), match SNR, 138 tests pass. Lowest
   risk, banks the wins. Gate: num_heads%16==0 && topk%32==0 (covers H64/H128 production).
   User was asked twice and chose "keep optimizing" over flipping default — CONFIRM intent.
2. **Multi-wave dkv-intermediate rewrite** — the only path to close bwd's remaining ~25%.
   Split heads or D across waves + vectorized transpose staging (mirror Triton's 4-warp
   layout). The single-wave port (dead end #6) is correct and a good starting reference;
   the head-contraction output GEMM math is already validated. Big effort, high value:
   would take bwd H128 from 0.64x → ~0.8x triton.
3. **bwd dq spill reduction** — still spills 55 VGPR at occ=1 (q_packs+do_packs+dq_accs
   =256). Lower value (occ2 didn't help fwd), but if a cheap win exists it's here.
4. **Fuse dq + dkv-intermediate** — dq already has Q/dO/dS/P in registers; it round-trips
   dS/P through HBM for dkv-interm to read back. A fused kernel eliminates that. Highest
   ceiling, highest complexity (different output tilings: [head,d] vs [key,d]).
5. **fwd last ~10% to gluon** — would need a structurally different (non-scattered) gather;
   likely not worth it vs the bwd gaps.

---
## 7. GOTCHAS (flydsl-specific, learned the hard way)
- flydsl packs a tensor's BYTE size as i32 in its CABI → any single tensor arg must be
  < 2^31 bytes. interm [T,TOPK,d_qk] = 2.4 GB at H128 K512 overflows. Workaround: chunk
  the launch over T (see dsa_bwd.py interm path). Applies to any big output tensor.
- Multi-line `if/else` rebinding of a var inside a traced kernel body is NOT propagated
  by the flydsl AST rewriter → use a single-line ternary or `const_expr` guards. (A
  `_kv_swz` identity call even blocked a codegen fast-path once — gate with const_expr.)
- `const_expr(...)` for build-time branches; `range_constexpr(...)` for unrolled loops.
- Read compiled VGPR/spill/occupancy: load newest `/root/.flydsl/cache/*<kernel>*/*.pkl`,
  `pickle.load`, regex `vgpr_count|vgpr_spill_count|agpr_count` in `artifact.ir`.
- rocprofv3 kernel-trace for per-kernel µs: `rocprofv3 --kernel-trace -d DIR -- python x.py`
  then query the sqlite `rocpd_kernel_dispatch` + `rocpd_info_kernel_symbol` tables.
- The standalone flydsl JIT dependency-walker can hit RecursionError on tiny probe
  kernels (infra bug, not your code) — don't waste time; use the real kernels + SNR harness.

---
## 8. GIT STATE
Branch: `optimize/optimize_dpskv4_attention_csa_kernel_with_flydsl_202606261028`.
This session's commits (newest first):
  ca2a8f3 FlyDSL dkv-interm MFMA port (gated off, correct-but-slower)
  a5e8ef1 bwd dq m16: QK_PF=1 default
  0fdc869 M=16 bwd dQ kernel (shared gather + ds_read_tr): dQ 2.9x
  f5c39a5 tr16 M=32-style DMA + pipeline (gated); document ceiling
  1107080 tr16 gated DMA + softmax‖QK pipeline; QK_PF=3
  b7cdd43 M=16 ds_read_tr fwd (tr16) shared gather: 1.34x over M=32
All committed; working tree has only pre-existing unrelated changes. Production
default UNCHANGED (M=32 paths) — wins are opt-in via env until §6.1 is decided.
