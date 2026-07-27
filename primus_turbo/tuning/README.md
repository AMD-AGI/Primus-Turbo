# primus_turbo/tuning

Offline autotuning for Primus-Turbo. For every shape it decides **which backend to use**
and **that backend's internal config**, writes the answer to disk, and leaves the runtime
with nothing but a table lookup — no profiling, no benchmarking. Inference therefore never
stalls on a first call, and the kernels can be captured into a CUDA graph directly.

## Assets

Results live in `configs/<framework>/<arch>/<op>.json`, e.g.
`configs/pytorch/gfx950/gemm_fp8.json`. Each dispatcher loads its own file lazily on the
first dispatch, selected by arch, with no environment variable involved. Loading is skipped
while auto-tune is on, and a lookup miss falls back to the default backend running its fixed
default config.

An entry is `{key, backend, backend_config, perf}`:

| Field | Meaning |
| --- | --- |
| `key` | shape, dtypes, transposes, granularity — what the lookup matches on |
| `backend` | the winner, e.g. `TRITON`, `CK`, `HIPBLASLT`, `FLYDSL` |
| `backend_config` | that backend's own flat config dict, or `null` if it has no internal autotune |
| `perf` | `{time_ms, tflops, gbps}`, informational only |

`backend_config` is opaque to the framework: the `backend` field decides who interprets it.
Triton block-wise stores `{BLOCK_M, BLOCK_N, num_warps, num_stages, ...}`, while CK and
hipBLASLt tune internally and store `null`.

## GEMM offline tune

### Shape list

`m` is the token count, `n` and `k` are the weight dims:

```json
{"mnk": [[16, 4096, 4096], [64, 4096, 4096], [256, 4096, 4096]]}
```

```python
import json

mnk = [[16, 4096, 4096], [64, 4096, 4096], [256, 4096, 4096]]
json.dump({"mnk": mnk}, open("my.json", "w"))
```

### Running

```bash
# built-in smoke-test shape (4096^3)
python -m primus_turbo.tuning.offline_tune_gemm

# your own shapes
python -m primus_turbo.tuning.offline_tune_gemm --shapes my.json

# spread those shapes over 8 GPUs
python -m primus_turbo.tuning.offline_tune_gemm --shapes my.json --gpus 8
```

`--gpus N` deals the shapes round-robin to N single-GPU workers and merges their output.
**At most one worker per GPU**: asking for more than the visible device count is rejected,
because two workers sharing a GPU contend for it, mismeasure, and bake the wrong winner into
the asset. Fewer shapes than GPUs simply leaves the extra ones idle.

Output goes to `configs/pytorch/<arch>/`, the canonical path the runtime auto-loads from.
`--out-dir` writes somewhere else instead, which is handy for inspection but will not be
loaded at runtime (the sharded run uses it internally).

### What gets swept

Every shape runs forward and backward, so the two gradient GEMMs are tuned as well.

| Asset | Grid |
| --- | --- |
| `gemm.json` | dtype (bf16, fp16) |
| `gemm_fp8.json` | dtype × format (E4M3, E5M2, HYBRID) × granularity (TENSORWISE, ROWWISE, BLOCKWISE, MX_BLOCKWISE) |
| `gemm_fp4.json` | dtype × preshuffle; format and granularity are fixed by `Float4QuantConfig`, and the whole precision is skipped off gfx950 |

## Adding a precision

Write its `_jobs_*` builder and add one row to the `_PRECISIONS` table in
`offline_tune_gemm.py`. Both the sweep and the shard merge walk that table, so the precision
is described in exactly one place.
