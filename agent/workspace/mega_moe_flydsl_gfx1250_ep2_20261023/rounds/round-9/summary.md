## Hypothesis

Both wgrads ran the variable-K kernel on an operand first copied into pool order by torch
(`pool_rows_of`: `token_buffer[pool_row_to_recv_token.clamp(...)]`): dW2's A (the dispatched dy) and dW1's
B (the re-dispatched x). At T=8192 the two `vectorized_gather_kernel`s cost 224 + 232 us per step plus ~13 us
of index prep each (round-8 trace), ~0.47 ms of an ~11.3 ms step, all of it HBM traffic (~0.9 GB per gather).
The variable-K kernel already loads its operands with TDM; TDM gather mode (CDNA5 ISA 4.10.3.2) loads 16 rows
per instruction from a row-index list, so the kernel can read the gathered operand straight from the dispatch
token buffer through `pool_row_to_recv_token`.

## Single change

`grouped_gemm_bf16_variable_k_flydsl_kernel(..., a_index=, b_index=)` (gfx1250): with an index, that
operand's k-tile rows are read through the int32 row table with TDM gather loads. Every wave gathers
`tile_k / num_waves` rows (2 x 16-row ops at tile_k = 128); the next k-tile's indices are loaded one tile ahead
into a register cell and broadcast with `v_readlane`. Rows past the expert's count re-read its last row (the
other operand's TDM zero-fills them, so they add exact zeros); an empty expert has no row to re-read, so it
reads table entry 0 (its `m_start` can be the table's end) and its accumulators are zeroed in the epilogue.
Gathering is limited to the 2-buffer, non-persistent pipeline the Mega shapes use. Without an index the trace
is unchanged.

Callers: `stage2_backward` passes the token buffer with `a_index=handle.pool_row_to_recv_token`;
`dispatch_only_gfx1250` returns the dispatch token buffer itself and `stage1_backward` passes it with
`b_index`; `pool_rows_of` is removed (no other user).

## Results

### Validation
- Time: 2026-10-23 06:58 (UTC)
- Validation level: full (EP2 pytest 5 passed, EP1 pytest 19 passed, FLYDSL grouped-gemm pytest 87 passed /
  36 skipped (reduce_num_cu, pre-existing), quick bench all PASS), bench repeated 2x plus a same-session A/B
  against round-8 (HEAD, changes stashed)
- The first validation (`artifacts_pre_bounds_fix/`: 1276.623 TF, reps 1275.225 / 1276.281 / 1275.427,
  same-session round-8 1248.641) ran before the empty-expert table read was bounded; the fix does not touch
  the Mega shapes' work (their table is a slice of a larger workspace), and everything was re-run after it
- Isolated check (`artifacts/test_vk_gather.py`, GPU 0): gathered A and gathered B vs the dense kernel on the
  materialized gather, ragged counts with empty experts (middle and last) and out-of-range padding markers:
  max |diff| = 0, no NaN, empty experts exactly zero
- Raw artifacts: `artifacts/test_ep.log`, `artifacts/test_ep1.log`, `artifacts/test_grouped_gemm_flydsl.log`,
  `artifacts/quick_validation.log`, `artifacts/bench.csv`, `artifacts/quick_rep{2,3}.log`,
  `artifacts/quick_ab_r8.log`, `artifacts/vk_gather_isolated.log`, `artifacts/vk_gather_cost_split.txt`,
  `artifacts/kernel_trace_last_iter_r8_vs_r9.txt`

| Shape | R8 step TF (run 1) | R8 same session | R9 step TF (run 1; reps 2-3) | R8 same-session bwd ms | R9 bwd ms (run 1) |
|---|---|---|---|---|---|
| dsv3_t8192_random | 1532.3 | 1528.4 | 1559.9; 1560.8, 1560.3 | 7.321 | 7.104 |
| dsv3_t8192_skew1 | 988.7 | 988.7 | 1009.9; 1006.9, 1007.0 | 11.316 | 10.965 |
| dsv3_t4096_random | 1313.7 | 1306.7 | 1337.4; 1324.8, 1322.7 | 4.446 | 4.324 |
| Aggregate (geomean) | 1257.834 | 1254.526 | 1281.979; 1276.913, 1276.132 | | |

Mean of the three R9 runs 1278.341 TF: +1.630% over round-8 run 1, +1.898% over the same-session round-8 run
(the pre-fix set gave +2.182% same-session). Every shape gains 1.7-2.1% same-session (mean of runs); the
forward is unchanged (aggregate fwd 1210.6 vs 1217.6-1222.7 TF), the backward carries it (1278.3 ->
1306.6-1317.0 TF). Min SNR 48.5 dB, unchanged.

Kernel trace, rank 0, T=8192 random, last iteration (`artifacts/kernel_trace_last_iter_r8_vs_r9.txt`): the two
gathers and their index prep are gone; dW2 variable-K 1059 -> 1202 us, dW1 2118 -> 2241 us. That run's other
kernels are ~5% slower too (L1 fwd dispatch_gemm 1951 -> 2060, L2 fwd gemm_combine 1550 -> 1628), so part of
the variable-K delta is clock drift; the swiglu_bwd-to-dW1 span still drops 6386 -> 6231 us.

Isolated gather-mode cost (`artifacts/vk_gather_cost_split.txt`, GPU 0, DSV3 EP2 rank shapes): dense 1076 /
2024 us; gathered with identity rows 1135 / 2098, sorted random rows 1144 / 2091, random rows 1111 / 2069 us
(dW2 / dW1). The overhead (~35-75 us, 3-5%) is gather mode itself, not row locality.

## Decision

`ACCEPTED` — every run (before and after the bounds fix) beats round-8 on every shape, +1.9% same-session;
bit-exact against the dense kernel on the materialized operand.

## Attribution

~0.47 ms of torch gathers per T=8192 step removed, ~0.1-0.15 ms of it given back as TDM gather-mode overhead in
the two wgrad kernels; net bwd -0.22 ms (random), -0.35 ms (skew1), -0.12 ms (T=4096) against the same-session
round-8 run.

Probe carried over from round-8's next direction (not a round): pinning the fold sum's `dst_rank` to SGPRs did
not remove the remote-store waterfalls (the per-segment `map` offset is reloaded into VGPRs); pinning
`combine_token_addr` did, but the L1 dgrad gemm_combine moved -11 us (2448 -> 2437), below noise. Reverted.

## Real-training transfer check
- Bucket: K2 (memory traffic removed from the production backward)
- Production path: stage2_backward (dW2) and stage1_backward (dW1) — every training step's backward
- Skewed routing: dsv3_t8192_skew1 +2.1% same-session (+1.9% mean of runs)
- Decision: ACCEPTED

## Next direction

Gather-mode TDM costs 3-5% of each wgrad kernel. Before picking a fix, split it with the ISA and a counter
run: the per-op descriptor build (16 `v_readlane` + SALU packing, 2 ops per wave per k-tile) vs the TDM
engine's gather throughput (8 KB per op vs 64 KB per tile op). Elsewhere: the prologue's ~70 us idle before
the first dispatch_gemm.
