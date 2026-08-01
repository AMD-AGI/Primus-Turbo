# MXFP8 Forward — Bottleneck Analysis & Optimization Plan

**Shape:** EP8, T=8192, H=7168, I=2048, E=256, K=8, load_balanced  
**Source:** `bench_logs_e2e/fp8_fwd_n04.log`, `bf16_fwd_n04.log`, WIP branch `mxfp8_moe_perf_summary.md`

## Baseline (n04, MI355X)

| Stage | fp8 (ms) | bf16 (ms) | bf16/fp8 | % of fp8 FULL |
|-------|----------|-----------|----------|---------------|
| L1 dispatch+fc1 | 2.408 | 4.173 | 1.73× | **49%** |
| SwiGLU+mxfp8 quant | ~0.35 | ~0.35 | ~1× | **7%** |
| L2 fc2+combine | 2.129 | 2.850 | 1.34× | **44%** |
| **FULL forward** | **4.889** | **6.924** | **1.42×** | 100% |

fp8 already beats bf16 by ~42%. The two dominant legs are **L1** and **L2** (roughly equal).

## L2 decomposition (fp8, K=I=2048)

| Leg | Time | Notes |
|-----|------|-------|
| GEMM-only (`PT_COMBINE_GEMM_ONLY=1`) | 1.590 ms | ~1286 TFLOPS |
| PUSH-only (`PT_COMBINE_PUSH_ONLY=1`) | 1.572 ms | comm-bound |
| FULL L2 | 2.131 ms | **balanced** max(GEMM, PUSH) + ~0.5 ms reduce tail |

Production pins `num_combine_cu=32` (e2e beats 48 by ~5%). Autotune candidates for fwd: 24–64.

## L1 structure

Single-grid pipeline: **comm PUSH ∥ preshuffle ∥ GEMM** (scoreboard-gated). Serial prefix on main stream:

1. `quantize_rowwise_mxfp8_flydsl(x)` — per-forward x quant before kernel
2. w1 scale preshuffle (cached on static weights)

L1 is already **1.82×** vs bf16 (fp8 PUSH bytes + ~2× GEMM).

## Constraint

**Single CUDA stream only** — no side-stream overlap for comm∥compute (same rule as dW1 path).
Cross-stage / prep overlap via extra streams is **not allowed**.

## Serial pipeline (production)

```
w1 prep → L1 (prologue → x quant → dispatch+GEMM) → SwiGLU → w2 prep → L2
```

## Optimization backlog (priority)

| P | Item | Status | Notes |
|---|------|--------|-------|
| ~~P0~~ | side-stream w1/w2/x-quant overlap | **reverted** | violates single-stream constraint |
| **P1** | L2 `num_combine_cu` sweep | validated | **32 optimal** @ n04 |
| **P2** | L1 kernel: comm∥GEMM within single grid | existing | already in fused dispatch kernel |
| **P3** | L2 GEMM tile / CShuffle epilogue micro-opts | pending | main lever for fwd |
| **P4** | L1 x-quant fusion into comm role (single grid) | pending | remove serial x-quant prefix |

SwiGLU is ~7% — already trimmed; not the lever.

## Validation

```bash
docker exec xiaoming-dev bash -lc '
export PYTHONUNBUFFERED=1 PYTHONPATH=/perf_apps/xiaoming/MegaMoE-dev:/tmp/ptdev_deps
export LD_LIBRARY_PATH=/workspace/Primus-Turbo/primus_turbo/lib:/workspace/Primus-Turbo/build/lib.linux-x86_64-cpython-312
export MASTER_PORT=$((9000 + RANDOM % 500))
cd /perf_apps/xiaoming/MegaMoE-dev
python benchmark/ops/bench_fwd_breakdown_compare.py --num-processes 8 --num-tokens 8192 --warmup 10 --iters 30
python benchmark/ops/training/bench_mega_moe_fp8.py --num-processes 8 --num-tokens 8192 --stage fwd --mode load_balanced --warmup 10 --iters 30
'
```

Acceptance: FULL fwd ≤ 4.7 ms (from 4.89) with correctness unchanged.
