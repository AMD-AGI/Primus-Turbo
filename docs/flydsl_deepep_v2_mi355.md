# FlyDSL DeepEP V2 without TDM, and bringing it up on MI355X (gfx950)

Status: **bring-up branch, not supported.** `flydsl_v2` ships for MI455X (gfx1250) (PR #583). On this
branch the same backend runs on every GPU and picks its code path per GPU (design:
[`flydsl_deepep_v2_unified.md`](flydsl_deepep_v2_unified.md)); off gfx1250 it is experimental. This
document has the register-copy path, the benchmark used to measure it, and what is known so far. On
MI355X V2 would have to beat the C++ DeepEP (`cpp`, the default there), and nothing has been measured
there yet.

## TDM on gfx1250, a register copy elsewhere

V2 moves every token row global -> LDS -> remote heap with the Tensor Data Mover (TDM): one
`tensor_load_2d` of a row into LDS, then one `tensor_store_2d` per destination rank. gfx942 / gfx950
(MI300X / MI355X) have no TDM.

The kernels have a second path for this step, the **register (VGPR) copy**: a warp loads the row into
registers in 8 KiB pieces (`FANOUT_PIECE_BYTES`) and stores it to each destination: `_fanout_row` in
the dispatch (each piece loaded once for all destinations), V1's `_copy_row` in the combine. Everything else in V2 is the same on both paths: the
fixed slots in each destination's heap, the row counts, the per-row combine flags, the reduce queue.
Off gfx1250 the register copy is the only path. On gfx1250 it is off unless asked for.

A warp copy through registers caps how many bytes each CU has in flight, so per CU it moves less than
TDM. With enough CUs that cap stops mattering (results below).

## Switches

| Variable | Effect |
| --- | --- |
| `PRIMUS_TURBO_DEEPEP_V2_TDM=0` | `flydsl_v2` takes the register copy on gfx1250 too. Read when `intranode_v2_kernel` is imported; `flydsl_intranode.v2_code_path()` shows the path. |
| `PRIMUS_TURBO_DEEPEP_ARCH_TRAITS="field=value,..."` | Replaces the tuning fields of this GPU's `ArchTraits` (`v2_copy_piece_bytes`, `combine_loads_in_flight`, `spin_timeout_cycles`, `trap_on_timeout`), for tuning on MI355X without editing `arch.py`. Read at import. |

No switch is needed off gfx1250: the runtime accepts `flydsl_v2` on every GPU (with one warning per
process where it is experimental), and `tests/pytorch/deep_ep/test_intranode.py` runs its `flydsl_v2`
tests everywhere.

Give each setting its own FlyDSL compile cache (`FLYDSL_RUNTIME_CACHE_DIR`) when switching between them,
so that no kernel compiled for the other path is picked up.

## MI455X: V2 with and without TDM

2× MI455X (gfx1250), EP2, Primus's MI455X DeepSeek-V3 proxy shape: 8192 tokens per rank, hidden 7168
bf16, top-8 of 128 experts, routing limited to the top 4 of 8 expert groups; 512 MB L2 flush before each
iteration, slowest rank, median of 30 iterations, mean of 3 rounds, then the mean of 2 runs (runs
within 1.6%). Image `amdprimus/amdprimus:tas-20261019`, FlyDSL 0.3.4.1, code at `60e5868f` (PR #583) plus
the switches. µs; dispatch = notify, its host wait and the dispatch.

| CUs | Dispatch: TDM / register copy / V1 | Combine: TDM / register copy / V1 | Register copy against TDM (dispatch / combine) |
| ---: | --- | --- | --- |
| 32 | 331 / 757 / 1691 | 347 / 604 / 1069 | 2.29× / 1.74× slower |
| 64 | 215 / 428 / 969 | 195 / 306 / 574 | 1.99× / 1.57× slower |
| 128 | 183 / 261 / 582 | 150 / 171 / 330 | 1.43× / 1.14× slower |
| 160 | 180 / 230 / 501 | 147 / 161 / 308 | 1.27× / 1.10× slower |
| 256 | 175 / 185 / 356 | 137 / 139 / 240 | 1.06× / 1.02× slower |

- Correct: `test_intranode.py -k flydsl_v2` with `PRIMUS_TURBO_DEEPEP_V2_TDM=0`, 8 tests pass on 2 GPUs,
  and every benchmark configuration's outputs were checked.
- Most of V2's gain over V1 is its structure, not TDM: with the register copy it is still 1.7–2.6×
  faster than V1 at every CU count.
- TDM is what lets V2 reach the link with few CUs. Without it V2 needs about 256 CUs to come within
  2–6% of TDM; at Primus's default of 32 CUs the dispatch is 2.3× slower.
- With a cached handle (the backward pass's dispatch) the register copy costs 2.55× at 32 CUs, 1.64× at
  128 and 1.08× at 256.

## Bringing V2 up on MI355X

Not done. What it takes, in order:

1. Correctness on gfx950. The wave64 branches of the V2 kernels (`_ballot`, `_popc`, `_lanes_below`)
   and gfx9's memory ordering (`ArchTraits` for gfx9: a store to the uncached heap is acknowledged only
   from memory) have run with V2 on gfx942 only: the CI's `test_intranode.py` at EP8 on 8× MI300X
   passes (`flydsl_deepep_v2_unified.md`, validation table). Run the unit tests on 2 GPUs, then 8:

   ```bash
   NCCL_CUMEM_ENABLE=0 FLYDSL_RUNTIME_CACHE_DIR=/tmp/v2_gfx950_cache \
     python3 tests/pytorch/deep_ep/test_intranode.py -v -k flydsl_v2
   ```

   `NCCL_CUMEM_ENABLE=0` only matters on the MI455X image's ROCm SDK (issue #584).

2. EP8. A MI355X node has 8 GPUs; V2 has run at EP2 and EP4 on MI455X (4 GPUs per node) and at EP8 only
   in the gfx942 CI (correctness tests, register copy). Check the heap
   size (`Buffer.get_token_buffer_size_hint`) and the channel split at 8 ranks.

3. Performance against `cpp`, at the same shape, at EP2 and EP8:

   ```bash
   PRIMUS_TURBO_EP_FORCE_CURRENT_STREAM=1 \
   BACKENDS=cpp,flydsl_v2,flydsl CU_LIST=32,64,128,160,256 OUT_JSON=/tmp/ops_mi355_ep2.json \
   FLYDSL_RUNTIME_CACHE_DIR=/tmp/v2_gfx950_cache PYTHONPATH=benchmark/ops/deep_ep_v2:. \
     python3 -m torch.distributed.run --nproc_per_node=2 benchmark/ops/deep_ep_v2/bench_ops.py
   ```

   `--nproc_per_node=8` for EP8. `bench_ops.py` documents its other variables (`TOKENS_LIST`,
   `ROUTING`, `CHECK`, `ITERS`, `ROUNDS`, ...); add `rccl` to `BACKENDS` for the all-to-all rows.
   V2 is only worth adapting if it beats `cpp` with the CUs training can spare: without TDM it needs
   many CUs, and those are taken from the GEMMs.

4. If it does: tune the register copy for gfx950 (piece size, warps per destination), enable the tests
   in CI (gfx942 / gfx950), and decide the default there.
