# Round 08 (optimize round 8) — fused fc1 GLU: row-merged store + the cache policy that follows from it

Git is forbidden this round, so this file stands in for `agent_diff.txt`.

## What is on disk

Two files, one mechanism: the fused SwiGLU epilogue's output stores now close a
64 B write run in one store instead of half-filling two, and the cache policy that
was tuned for the old 32 B run is retired because the wider run inverts its sign.

### 1. `primus_turbo/flydsl/utils/gemm_epilogue_helper.py` — `StoreCSwiGLU`

* import `_permlane16_swap` from `gemm_helper` (the same cross-lane primitive
  `StoreCPerTensorRowN` already uses for the plain stores).
* `__init__`: `self.row_merge = not ilv and n_tiles_b % 2 == 0 and out_ty is fx.BFloat16`,
  plus the merged run's geometry `merge_row = (lane_id // 32) * 8` and
  `merge_col = lane_id % 32`.
* `store_pair`: new `_emit_merged()`. For each fragment pair it packs the lane's four
  rows into two dwords by row pair (`self._pack`, i.e. `cvt_pk_bf16_f32` for bf16),
  moves the odd fragment's dword into the other 32-lane half with one
  `v_permlane16_swap_b32` per pair, and stores the run the two fragments used to
  half-fill. Values, their predicate and the abs-max fold are all taken **before** the
  swap, so only the address changes.
* the three streams (l1 gate band, l1 up band, act) take `_emit_merged` when
  `row_merge` holds and the pre-existing scalar `_emit` otherwise.

Why: with the plain column map a lane owns one column per n-fragment, so a fragment's
16 lanes cover 32 B of a 64 B run and the run only closes if L2 merges the next
fragment's store into it.

### 2. `primus_turbo/flydsl/grouped_gemm/grouped_gemm_fp8_glu_kernel.py` — NT fused GLU `_build`

`cstore_aux=2, glu_act_aux=2` → `0, 0`, comment rewritten. The non-temporal hint was
measured at 1.11x back when a lane pair wrote 32 B — half a line either way. With the
merged store each run is a 64 B half-line, and evicting it early splits the 128 B line
it shares with its neighbour run.

## Measured

### PMC, fc1 GLU dispatch (`kernel_grouped_nt_persistent`, deployed M=131072, G=32, warm)

| build | TCP_TCC_WRITE_REQ | TCC_EA0_WRREQ | dispatch µs |
|---|---|---|---|
| round 8 shipped | 70.853 M (32.0 B/req) | 52.858 M | 2266.2 |
| + row-merged store | **35.389 M** (64.0 B/req) | 44.114 M | 2162.8 |
| + l1 store at default policy | 35.389 M | 38.393 M | 2148.1 |
| + act store at default policy (**shipped**) | 35.389 M | **35.581 M** | **2105.0** |

35.389 M x 64 B = 2.265 GB = exactly l1 [131072, 5760] + act [131072, 2880] in bf16, so
the request path is now at the floor for a 2 B-per-lane store. Every other dispatch in
the unit already had EA0 ≈ TCP (quantize 11.796/11.796, wgrad 16.589/16.589, fc1 dgrad
11.796/11.797); fc1 GLU was the only one writing more to DRAM than it asked L2 for, and
it no longer is.

### Scored bench

Round-8 baseline, two runs on the clean tree at the head of this session (same node,
same session, so this is the honest before/after pair rather than a cross-session
reference): spd 1.040197 / 1.041097, ratio_mlp 0.971050 / 0.970856, mean spd 1.040647 —
within 0.006% of the 1.040589 memory records for round 8.

Row-merge only, three runs: spd 1.044418 / 1.045052 / 1.045211, ratio_mlp 0.963125 /
0.960410 / 0.962103.

Final tree (row-merge + default policy), three runs:

| | spd | step_ms | ratio_mlp | ratio_proj | ratio_perm |
|---|---|---|---|---|---|
| 1 | 1.047533 | 675.0149 | 0.956863 | 0.822603 | 0.800254 |
| 2 | 1.047291 | 675.1708 | 0.957436 | 0.822184 | 0.801478 |
| 3 | 1.047865 | 674.8007 | 0.955538 | 0.823345 | 0.807740 |

mean spd **1.047563** (spread 0.055%) against this session's own baseline pair at
1.040647, i.e. **+0.66%**; ratio_mlp mean **0.956612** against 0.970953, i.e. **-1.48%
of the mlp unit**. Gate PASS on every run,
`snr_mlp_bal`/`snr_mlp_ragged` 23.72-23.74 and `snr_rms`/`snr_proj`/`snr_perm`
unchanged. proj and perm untouched and inside their round-8 bands.

### Correctness

The permutation moves addresses, not values. Four cross-process runs of a direct probe
on the fused GLU with real fp8 inputs gave byte-identical `l1`, `act` and folded amax on
both arms; the fold is taken pre-swap over the same (value, predicate) set, so the amax
is the same float. The aux bits are cache policy and cannot change a stored value. The
bench's own byte-determinism gate and the fp8 SNR floor passed on all runs.

### Sibling paths (red line 6)

`row_merge` requires the non-interleaved column map, an even `n_tiles_b` and a bf16
output. The mxfp4 quantised GLU (`StoreCSwiGLUQuant`) sets `skip_act` and requires the
in-mainloop `cst` store, so both merged emits are skipped, and it runs with `ilv != 0`
anyway; fp16 output is excluded by the dtype gate; odd `n_tiles_b` falls back to the
scalar path. The row bound is still the band SRD's `num_records` and the column
predicate is the run's own first column, so `band_drop` and the masked case behave as
before. The aux change is a constant inside the fp8 NT fused GLU `_build` only — the NN
dGLU's `cstore_aux=2` is untouched (its CShuffle store already writes 59.8 B/request and
shows no amplification: 25.272 M requests against 24.001 M DRAM writes).

## Rejected this round, with numbers

* **L2 tile swizzle on the fused entries** — `(num_xcd, group_m, group_n)` A/B at the
  deployed M was flat inside ~0.5%, agreeing with round 8's H7/H8. Not the lever.
* **g2s cache-policy hints on the operand loads** (the directive's axis) — priced by
  subtraction before building: the necessary operand reads are 0.92 GB, which at the
  4.0-4.7 TB/s this kernel actually achieves is ~230 µs of the 2.4 ms dispatch, and
  round 6 already measured intra-workgroup no-allocate hints at -0.95%. The write side
  carried 2.27 GB of necessary traffic and 0.56 GB of amplification, so that is where
  the round went. `BufferCopyLDS128b` still has no `cache_modifier`; the underlying
  `raw_ptr_buffer_load_lds` does carry `aux`, so the axis remains reachable if a future
  round wants it.
