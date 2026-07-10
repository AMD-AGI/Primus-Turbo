# MXFP8 fused dispatch+GEMM perf note (MI355X / gfx950)

Perf of the mega-MoE **L1 fused cross-rank dispatch PUSH + grouped mxfp8 GEMM**
(`dispatch_grouped_gemm_mxfp8_kernel.py`, used by `mega_moe_fused_mxfp8_forward(comm="fp8_fused")`),
vs the decoupled fp8 path and the bf16 fused kernel. AMD Instinct MI355X, DeepSeek-V3 MoE
L1: `T=8192, H=7168, I=2048, E=256, K(top-k)=8`, EP8 (`G=32` experts/rank), `BLOCK_M=BLOCK_N=256`.

Benchmark: `benchmark/ops/bench_dispatch_grouped_gemm_mxfp8.py --num-processes 8 --iters 20`.

## TL;DR

- The default fp8 fused path is a **3-stage single-kernel pipeline** — COMM (clean-push raw
  fp8 + E8M0) → PRESHUFFLE role (raw→broadcast scale, once/pool-block) → GEMM (preshuffled
  mxfp8), all overlapped and gated per pool-block by the sys-scope scoreboard.
- **fp8 fused ≈ 2.3–2.6 ms**, i.e. **~1.4–1.5× vs the bf16 fused kernel** and **~1.1–1.2× vs
  the decoupled fp8 path** (push + GEMM run as two serial kernels). Accuracy unchanged
  (cos = 1.0 vs the decoupled mxfp8 ref; e2e MoE forward 22.97 dB).
- Even *without* fusion, fp8 already wins: decoupled fp8 (~2.87 ms) is ~1.2× faster than the
  bf16 fused kernel (~3.55 ms). The fusion buys the remaining comm/compute overlap on top.

## Measured (max over ranks, iters=20, ndcu=16, pscu=16)

| stage | load_balanced | round_robin |
|---|---|---|
| dense_gemm roofline | 2.95 ms / 1388 TF | 2.78 ms / 1383 TF |
| **bf16** gemm_only | 3.16 ms | 2.89 ms |
| **bf16** dispatch_only | 2.09 ms (394 GB/s XGMI) | 2.08 ms |
| **bf16** fused | 3.58 ms | 3.40 ms |
| **fp8** gemm_only (preshuffle+GEMM) | 1.74 ms / 2348 TF | 1.67 ms / 2305 TF |
| **fp8** dispatch_only (push) | 1.13 ms (377 GB/s, 1.94× fewer bytes) | 1.12 ms |
| **fp8 decoupled** (= push + GEMM, serial) | 2.87 ms | 2.79 ms |
| **fp8 fused (role pipeline)** | **2.56 ms / 1597 TF** | **2.29 ms / 1679 TF** |
| fused vs bf16 fused | **1.40×** | **1.48×** |
| fused vs fp8 decoupled | **1.12×** | **1.22×** |
| accuracy (cos vs ref / vs bf16) | 1.00000 / 0.99921 PASS | 1.00000 / 0.99921 PASS |

## The pipeline (why 3 roles)

The blocker to a fast fused kernel is the E8M0 **scale layout**: the fast MMA scale loader
(`ScaleS2R`/`ScaleBComb`, one coalesced dwordx4/K-iter) needs a **broadcast** layout, but a
fast (XGMI-saturating) push needs the **raw** [row, K/32] layout (coalesced per token). Those
two are transposes of each other, so a raw→broadcast **preshuffle** must happen somewhere.

Grid = `num_dispatch_cu` (comm) + `num_preshuffle_cu` (preshuffle) + `worst_case_tiles*n_blocks`
(gemm), role selected by `block_index`:

1. **COMM** (`dispatch_fp8_copy_tile`): clean-push pre-quantized fp8 + **raw** E8M0 to peer
   `pool_fp8`/`pool_scale` (coalesced, ~377 GB/s), L2 write-back, `atomic_add` the peer
   scoreboard.
2. **PRESHUFFLE role** (`preshuffle_a_scale_tile`): wait `scoreboard >= expected`, `l2_invalidate`,
   transpose that pool-block's A-scale raw→broadcast into `pool_scale_ps` **once**
   (non-redundant), `l2_writeback`, then `st` a **SENTINEL** (`1<<20`) on the scoreboard.
3. **GEMM**: wait `scoreboard >= SENTINEL`, `l2_invalidate`, run `gemm_mxfp8_nt_tile(preshuffled=True)`
   reading `pool_scale_ps` + host-preshuffled weight scale.

The scoreboard sys-scope acquire/release + device-scope L2 fences carry cross-rank/cross-XCD
visibility, so — unlike decoupled — **no host sync + standalone L2 invalidate** is needed
(that bubble is also saved in the real forward).

## Why the other fused variants lost (differential; each = one placement of the transpose)

| variant | L1 fused | why |
|---|---|---|
| quant-in-push (comm quantizes bf16 + writes broadcast scale over XGMI) | ~7.1 ms | quant + **scattered broadcast XGMI write** collapse comm to ~78 GB/s |
| clean-push raw + on-the-fly `ScaleS2RRaw` gemm | ~5.1 ms | scattered per-lane scale reads **in the MMA hot loop** → gemm 4.5 ms (2.8×) |
| clean-push + broadcast scale over XGMI (Option D) | ~3.2 ms | broadcast XGMI write (~185 GB/s) exposes comm above the gemm |
| clean-push + per-tile local preshuffle in gemm (Option C') | ~3.2 ms | preshuffle done redundantly by all `n_blocks`(16) tiles/block_m (~1.4 ms) |
| **preshuffle role (this)** | **~2.3–2.6 ms** | raw push (fast, hides) + preshuffle **once/block** + fast preshuffled gemm, overlapped |

Takeaway: the transpose is cheapest done **once per pool-block, locally, as its own pipeline
stage** — not folded into the XGMI write (scattered → BW collapse) nor the MMA hot loop
(scattered reads → MMA stall) nor redundantly per gemm tile.

## Config / tuning

- `num_dispatch_cu` (comm CUs) and `num_preshuffle_cu` (preshuffle CUs) both steal from the
  gemm. Sweep (load_balanced, iters=20): `ndcu=16, pscu=16` → **2.57 ms** (best); `pscu=8`
  → 3.29 ms (preshuffle can't keep up, exposed); `pscu=24` → 2.59 ms; `ndcu=12` → 2.95 ms
  (comm starved). Defaults `ndcu=16, pscu=16`.
- The caller must **zero `symm.scoreboard`** (cross-rank, barrier-bracketed) before each launch:
  the per-block sentinel handoff has no in-kernel self-reset (the forward + benchmark do this).

## Known follow-ups

- End-to-end whole-forward timing (fp8_fused vs fp8 vs bf16) to confirm the step-level win
  (fused additionally eliminates decoupled's host sync + cross-rank barrier, ~0.2–0.3 ms).
- A cross-rank scoreboard-signal race appears at high CU counts (`ndcu≥32`) with the fast
  clean-push comm; the default `ndcu=16` config does not trip it, but it should be root-caused
  before pushing the CU split higher.
