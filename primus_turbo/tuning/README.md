# primus_turbo/tuning

Offline autotuning for Primus-Turbo. For every shape it decides **which backend to use**
and **that backend's internal config**, writes the answer to disk, and leaves the runtime
with nothing but a table lookup — no profiling, no benchmarking. A first call therefore never
stalls, and the kernels can be captured into a CUDA graph directly.

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

## Running

One driver per op family, same CLI for both:

```bash
# built-in smoke-test shape
python -m primus_turbo.tuning.offline_tune_gemm
python -m primus_turbo.tuning.offline_tune_grouped_gemm

# your own shapes, spread over 8 GPUs
python -m primus_turbo.tuning.offline_tune_gemm --shapes my.json --gpus 8
```

`--gpus N` deals the shapes round-robin to N single-GPU workers and merges their output.
**At most one worker per GPU**: asking for more than the visible device count is rejected,
because two workers sharing a GPU contend for it, mismeasure, and bake the wrong winner into
the asset. Fewer shapes than GPUs simply leaves the extra ones idle.

Output goes to `configs/pytorch/<arch>/`, the canonical path the runtime auto-loads from.
`--out-dir` writes somewhere else instead, which is handy for inspection but will not be
loaded at runtime (the sharded run uses it internally).

Every shape is run forward and backward, so the gradient GEMMs are tuned as well.

## Dense GEMM

`m` is the token count, `n` and `k` are the weight dims:

```json
{"mnk": [[16, 4096, 4096], [64, 4096, 4096], [256, 4096, 4096]]}
```

| Asset | Grid |
| --- | --- |
| `gemm.json` | dtype (bf16, fp16) |
| `gemm_fp8.json` | dtype × format (E4M3, E5M2, HYBRID) × granularity (TENSORWISE, ROWWISE, BLOCKWISE, MX_BLOCKWISE) |
| `gemm_fp4.json` | dtype × preshuffle; format and granularity are fixed by `Float4QuantConfig`, and the whole precision is skipped off gfx950 |

## Grouped GEMM

`g` is the number of experts **one rank owns** and `m` the rows **per expert**, so `a` is
`[g * m, k]` and `b` is `[g, n, k]` with `n`/`k` the per-expert weight dims:

```
g = num_experts / EP
m = bs * seq * topk / num_experts
```

```json
{"gmnk": [[8, 2048, 4096, 4096], [8, 256, 4096, 4096]]}
```

The sweep uses load-balanced groups, which is what the dispatchers profile against anyway.
Note the lookup key records the *total* row count, so for a runtime lookup to hit, `g * m`
must equal that workload's `sum(group_lens)` — with the numbers above that is
`bs * seq * topk / EP`, but capacity limits or padding can change it, so use the number your
model actually produces.

Forward/dgrad and the variable-K wgrad are separate dispatchers, hence the `_vk` assets; one
sweep fills both.

| Asset | Grid |
| --- | --- |
| `grouped_gemm.json`, `grouped_gemm_vk.json` | dtype (bf16, fp16) |
| `grouped_gemm_fp8.json`, `grouped_gemm_fp8_vk.json` | dtype × format × granularity, as above |
| `grouped_gemm_fp4.json`, `grouped_gemm_fp4_vk.json` | dtype; skipped off gfx950 |

## Extending

To add a precision to an existing family, write its `_jobs_*` builder and add one row to that
driver's `_PRECISIONS` table — both the sweep and the shard merge walk it, so the precision is
described in one place.

To add a family, write an `offline_tune_<family>.py` that declares a `Family` and calls
`main`. Sweeping, perf annotation, sharding and the CLI all live in `_driver.py`.
