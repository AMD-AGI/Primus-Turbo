# Report to the image maintainers: hipBLASLt on gfx1250 — two defects

**Image:** `fa-tune:deps`, ROCm SDK wheels `_rocm_sdk_libraries_gfx1250` / `_rocm_sdk_devel`,
torch `2.11.0+rocm7.14.0a20260625`, hipBLASLt version string `100401`.
**Hardware:** single gfx1250 (`AMD Radeon Graphics`, DID `0x75c1`), VR-limited —
`rocm-smi` reports a 1100 MHz DPM ceiling, a sysfs sample under sustained load read 943 MHz.
**All measurements below are on this one machine at that clock**, and no figure here is
converted to any other clock.

Two independent problems. The first has a one-line workaround and is probably a packaging
slip. The second has no workaround available to a user and is the one that costs real
throughput.

---

## Defect 1 — the Tensile library index is one directory above the files

### Symptom

Any matmul raises `HIPBLAS_STATUS_INVALID_VALUE` out of `hipblasLtMatmulAlgoGetHeuristic`,
preceded by:

```
rocblaslt error: Cannot read ".../hipblaslt/library/TensileLibrary_lazy_gfx1250.dat"
                 (or .zlib variant): No such file or directory
rocblaslt error: Could not load ".../hipblaslt/library/TensileLibrary_lazy_gfx1250.dat"
hipModuleLoad failed: .../hipblaslt/library/Kernels.so-000-gfx1250.hsaco  error: file not found
```

The only escape a user finds is `TORCH_BLAS_PREFER_HIPBLASLT=0`, i.e. falling back to rocBLAS.

### Cause

The loader looks in `.../hipblaslt/library/`. That directory contains exactly one entry —
the subdirectory `gfx1250/` — and all 402 files, index included, live inside it:

```
looked for:  _rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/TensileLibrary_lazy_gfx1250.dat
actually at: _rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250/TensileLibrary_*.dat.zlib
```

### Workaround (no file changes)

```
HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
```

Note for anyone reproducing: do **not** point it at
`_rocm_sdk_devel/lib/hipblaslt/library/gfx1250` — that path core-dumps.

### Impact, measured

| | TFLOP/s | ms |
|---|--:|--:|
| rocBLAS (`TORCH_BLAS_PREFER_HIPBLASLT=0`) | 27.64 | 39.78 |
| **hipBLASLt after the path fix** | **68.74** | **15.99** |

End to end on llama-3.1-8B (b=4, s=8192, 1 GPU, 32 layers), changing only the BLAS backend:
**244 → 2,027 tokens/s, 8.3×**. (Both arms are 3-step, non-steady-state, n=1 — the ratio is
large enough that the caveat does not change the conclusion, but it is not a steady-state
measurement.)

---

## Defect 2 — the NN transpose combination has no plain-GEMM tuning library

This is the expensive one, and unlike Defect 1 a user cannot work around it in configuration.

### Symptom

Some GEMMs run 16–21× slower than others **of identical shape**, having been assigned a
`MT32x16x32` macro tile — a grid of 16 to 65 **million** 64-thread workgroups, 592–681 ms for
a single call where the appropriate solution takes 0.8 ms.

### Evidence, from hipBLASLt's own logs

`HIPBLASLT_LOG_LEVEL=4` plus `TENSILE_DB=0x6`. Same M/N/K, same 76 MB workspace, the only
difference is the transpose combination:

```
NN:  ProblemMap: Contraction_l_Ailk_Bljk...   sol-tag = GridBased
     Object key: 4096, 32768, 1, 14336
     nearest entry in the table ->  M=1024, N=1, B=1, K=512
     Considered 11 (31.43%) of entries        ->  sol-idx 104,  MT32x16x32

TN:  ProblemMap: Contraction_l_Alik_Bljk...   sol-tag = Prediction
                                              ->  sol-idx 237,  MT256x256x128
```

The NN path has no prediction model, only a sparse lookup table, and the nearest entry it can
find to N=32768 is **N=1**. `MT32x16x32` is exactly the tile one would choose for a GEMV, and
it is being applied to a 4096×32768×14336 problem.

Workspace is not the cause: sweeping `HIPBLASLT_WORKSPACE_SIZE` across 128 MB and 1 GB changes
the timings by less than 0.1%.

### Root cause, from the shipped files

Of the bf16 libraries under `library/gfx1250/`:

```
Ailk_Bljk (NN)                      Alik_Bljk (TN)
  BB_BB_Bias_UA_Type_BB_HPA           BB_BB_Bias_UA_Type_BB_HPA
  BB_BB_HA_Bias_Grad_UA_...           BB_BB_HA_Bias_Grad_UA_...
  BB_BB_HA_Bias_SAV_UA_...            BB_BB_HA_Bias_SAV_UA_...
  BB_BB_HA_Grad_UA_...                BB_BB_HA_Grad_UA_...
                                      BB_BB_UA_Type_BB_HPA          <-- plain GEMM
                                      BB_BB_UA_Type_BB_HPA_CU96     <-- CU variant
                                      BB_BB_UA_Type_BB_HPA_CU192    <-- CU variant
```

**`BB_BB_UA_Type_BB_HPA` — the plain bf16 GEMM tuning library, no Bias/Grad/SAV suffix —
ships for TN in three variants and for NN in none.** File counts overall: 68 vs 40.

A plain GEMM issued as NN therefore has to fall back on the Bias/SAV libraries, whose
GridBased tables never covered this region of the shape space.

### Why this matters beyond one workload

In PyTorch training, the NN combination is **the dgrad of every `nn.Linear`** — that is, a
large fraction of every training step of every model on this image, not a corner case.

### Measured impact

Same card, same clock, same process. Simply re-laying out the operand so the call lands on a
different solution:

| shape | as issued | after re-layout | |
|---|--:|--:|--:|
| `[32768,14336]×[14336,4096]` | 55.5 ms / 69 TF/s | 2.4 ms / 1628 TF/s | 20.1× |
| `[14336,32768]×[32768,4096]` | 55.6 ms / 69 TF/s | 3.3 ms / 1160 TF/s | 8.4× |
| `[128256,32768]×[32768,4096]` | 513.6 ms / 67 TF/s | 23.0 ms / 1498 TF/s | 11.2× |

(Both speedup columns include the cost of the extra copy the re-layout requires.)

End to end on the 8-layer llama-3.1-8B configuration, seed pinned, **n=9 per arm**:
**6,128 → 11,604 tokens/s, 1.894×**, MFU 34% → 65.4%, and run-to-run variance drops from
3.91% to 0.42%.

That last number is worth a sentence of its own: the missing coverage was also making the
machine's end-to-end throughput **trimodal** — repeated identical runs landed on one of three
distinct values — because the GridBased nearest-neighbour search does not always resolve the
same way. Routing around it made the distribution unimodal.

**This is not a hardware limit.** On the same card at the same clock, the NT path measures
**1502 TFLOP/s**.

---

## Defect 2b — `hipblaslt-bench` cannot run on this image

`hipblaslt-bench` (present at `_rocm_sdk_devel/bin/`) fails at startup:

```
rocblaslt error: Cannot read ".../library/gfx1250/TensileLibrary_lazy_gfx1250.dat"
hipBLASLt status error: Expected HIPBLAS_STATUS_SUCCESS, received HIPBLAS_STATUS_INVALID_VALUE
```

It expects the *lazy* layout with a master index. The image ships the non-lazy layout: one
`.dat.zlib` per problem type, no master index.

Consequence: **a user cannot enumerate solutions, benchmark them, or produce a tuning override
file** — the normal path for working around exactly the kind of gap in Defect 2 is closed.

---

## What we are asking for

1. **Ship `BB_BB_UA_Type_BB_HPA` (and its CU variants) for `Contraction_l_Ailk_Bljk`.**
   This is the whole of Defect 2.
2. **Fix the library path**, or ship the index where the loader looks (Defect 1).
3. **Make `hipblaslt-bench` usable** on the shipped layout, so users can self-diagnose.

## How to reproduce quickly

```python
import torch, time
M, K, N = 32768, 14336, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)          # contiguous (K,N) -> NN
bt = b.t().contiguous()                                              # same matrix, N-major

def bench(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it

for name, fn in (("mm(a, b)      NN", lambda: torch.mm(a, b)),
                 ("linear(a, bt) NT", lambda: torch.nn.functional.linear(a, bt))):
    s = bench(fn)
    print(f"{name}  {s*1e3:7.2f} ms  {2*M*N*K/s/1e12:7.1f} TF/s")
```

Run with `HIPBLASLT_TENSILE_LIBPATH` set to the `gfx1250/` subdirectory (per Defect 1).
Add `HIPBLASLT_LOG_LEVEL=4 TENSILE_DB=0x6` to see `GridBased` versus `Prediction` in the log.
