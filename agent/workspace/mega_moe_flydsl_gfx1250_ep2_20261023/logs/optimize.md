# mega_moe_bf16 FLYDSL Optimization Log

## Basic Information
- Target operator: fused_mega_moe BF16 (dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine, fwd+bwd)
- Implementation language: FlyDSL
- Backend: FLYDSL
- Target GPU: gfx1250 (MI455X), EP2 on GPUs 0,1
- Campaign: agent/workspace/mega_moe_flydsl_gfx1250_ep2_20261023
- Start time: 2026-10-23 04:57 (UTC)
- Current status: Optimizing (round-2)

## Baseline
- Time: 2026-10-23 05:02 (UTC)
- Backend: FLYDSL
- GPU: gfx1250
- Commit: ca0a9c2f
- Validation level: full (EP2 test 5 passed, EP1 tests 19 passed)

| Shape | Fwd ms | Fwd+bwd ms | Fwd TF | Bwd TF | Step TF | Check |
|---|---|---|---|---|---|---|
| dsv3_t8192_random | 4.609 | 12.920 | 1253 | 1389 | 1340.4 | PASS |
| dsv3_t8192_skew1 | 6.698 | 18.991 | 862 | 939 | 911.9 | PASS |
| dsv3_t4096_random | 2.406 | 7.190 | 1200 | 1207 | 1204.3 | PASS |
- Combined Step aggregate (geomean): 1137.514 TF
- Fwd aggregate: 1089.939, Bwd aggregate: 1163.258
- Detailed data: rounds/round-1/summary.md
- Quick baseline log: rounds/round-1/artifacts/quick_validation.log

## Optimization History

### round-2 — 8 waves per workgroup (4x2) for the gfx1250 fused kernels
- Time: 2026-10-23 05:09 (UTC)
- Validation level: full
- Hypothesis: dispatch/combine/reduce in the fused launch are per-wave latency-bound at 4 waves/CU; 8 waves double the bytes in flight
- Changes: dispatch_gemm_kernel.py, gemm_combine_kernel.py default m_warp 2 -> 4 (n_warp 2)
- Result: Combined Step 1137.514 -> 1178.823 TF (+3.632%) ✅ (shapes +4.91% / +3.82% / +2.18%)
- Test: PASS (EP2 5, EP1 19)
- Decision: accept
- Detailed data: rounds/round-2/summary.md

### round-3 — combine chunk 16 -> 32 / 64 fold rows
- Time: 2026-10-23 05:13 (UTC)
- Validation level: quick
- Hypothesis: restore combine_bf16_tile's 4-row unroll per wave with 8-wave workgroups
- Changes: gemm_combine_kernel.py _FOLD_CHUNK_ROWS (config sweep)
- Result: 1176.666 (32) / 1177.542 (64) vs 1178.823 TF ❌ (noise)
- Test: PASS
- Decision: rollback (back to round-2)
- Detailed data: rounds/round-3/summary.md
- Notes: dispatch remote bandwidth is cache-policy independent (238-240 GB/s at 64 CUs) and sublinear in CUs;
  dispatch_bf16_tile serializes a remote store-retire wait + barrier + flag per expert (32 per launch).

### round-4 — dispatch-only publishes once per comm block
- Time: 2026-10-23 05:40 (UTC)
- Validation level: full
- Hypothesis: dispatch_bf16_tile's per-expert store retire + barrier + flag is a serialized remote round trip x32;
  the dispatch-only launch has no consumer of the per-expert flags before it ends
- Changes: ep_intranode.dispatch_bf16_tile experts_per_publish (default 1); gfx1250 dispatch launcher passes
  all experts for dispatch-only, 1 for dispatch+GEMM
- Result: 1194.345 vs 1178.823 TF ✅ (+1.317%; +1.72 / +1.62 / +0.62% per shape)
- Test: PASS (EP2 5, EP1 19, bench SNR >= 48.5 dB)
- Decision: accepted
- Detailed data: rounds/round-4/summary.md
- Notes: dispatch_only 490 -> 393 us at EP2; grouping the fused dispatch+GEMM publishes is noise (2-4) or worse (8+)

### round-5 — dispatch-only takes every CU
- Time: 2026-10-23 05:50 (UTC)
- Validation level: full
- Hypothesis: the dispatch-only launch shares no CU with a GEMM, and its push rate grows with comm CUs
- Changes: dispatch_only_gfx1250 default num_dispatch_cu 64 -> multi_processor_count - 1
- Result: 1203.385 vs 1194.345 TF ✅ (+0.757%; +0.91 / +0.18 / +1.19% per shape)
- Test: PASS (EP2 5, EP1 19, bench SNR >= 48.5 dB)
- Decision: accepted
- Detailed data: rounds/round-5/summary.md
- Notes: dispatch-only + gather 600 -> 413 us at EP2 (flat past ~192 CUs)

### round-6 — gemm_combine grid holds this call's tiles only
- Time: 2026-10-23 06:15 (UTC)
- Validation level: full (+ repeated bench)
- Hypothesis: ~7600 masked worst-case GEMM workgroups queue ahead of the tail combine (timeline: GEMM ends
  ~1020 us, tail combine starts 1155-1510 us)
- Changes: gemm_combine_kernel.py runtime grid (pool rows // BLOCK_M tiles + combine WGs); tail start from the
  device tile count
- Result: 1237.297 vs 1203.385 TF ✅ (+2.818%; rep 1234.672); +3.80 / +1.30 / +3.37% per shape
- Test: PASS (EP2 5, EP1 19, bench SNR >= 48.5 dB)
- Decision: accepted
- Detailed data: rounds/round-6/summary.md
- Notes: FlyDSL launch takes a runtime fx.Int32 grid, so no recompile per tile count

### round-7 — dispatch+GEMM grid holds this call's tiles only
- Time: 2026-10-23 06:25 (UTC)
- Validation level: full (+ repeated bench)
- Hypothesis: the dispatch+GEMM launch's masked worst-case tiles drain after the live ones
- Changes: dispatch_gemm_kernel.py runtime grid (comm WGs + pool rows // BLOCK_M tiles, or the dispatch-only waiter); MAX_TILES constexpr dropped
- Result: 1239.985 vs 1237.297 TF ✅ (+0.217%; reps 1239.146 vs 1234.672); +0.75 / +0.32 / -0.41% per shape
- Test: PASS (EP2 5, EP1 19, bench SNR >= 48.5 dB)
- Decision: accepted (small; forward -0.1 ms on T=8192 random)
- Detailed data: rounds/round-7/summary.md
- Notes: trailing masked tiles are cheap; round-6's gain came from the tail combine starting on time

### round-8 — pin the fold sum's pool rows to SGPRs (gfx1250)
- Time: 2026-10-23 06:12 (UTC)
- Validation level: full (+ 3 repeated benches + same-session A/B without the change)
- Hypothesis: one pool row index the compiler cannot prove wave-uniform puts each of its fold-sum loads behind
  a readfirstlane waterfall that drains loadcnt (6 per pass, 4 passes per fold row)
- Changes: ep_intranode.py combine_bf16_tile.push_rows readfirstlanes the loaded pool rows on gfx1250;
  symm_buffer.py SymBuffer cache signature also hashes ep_intranode.py (prerequisite, no codegen change)
- Result: 1257.834 vs 1239.985 TF ✅ (+1.439%; mean of 4 runs 1252.228, +0.987%; same-session no-pin 1242.921);
  isolated L2 forward gemm_combine 1442 -> 1383 us
- Test: PASS (EP2 5, EP1 19, bench SNR >= 48.5 dB)
- Decision: accepted
- Detailed data: rounds/round-8/summary.md
- Notes: FlyDSL's disk-cache key follows helpers in the launcher's own directory only, so edits to
  bf16/ep_intranode.py never reached the bf16_gfx1250 kernels' key; the first round-8 bench ran the round-7 binary

### round-9 — wgrads read the dispatched operand through the pool row table (TDM gather)
- Time: 2026-10-23 06:42 (UTC)
- Validation level: full (+ FLYDSL grouped-gemm pytest + 3 repeated benches + same-session A/B vs round-8)
- Hypothesis: the two torch pool gathers before dW2 / dW1 (224 + 232 us + index prep at T=8192) are pure HBM
  traffic; the variable-K kernel's TDM can gather the rows itself from the dispatch token buffer
- Changes: grouped_gemm_bf16_kernel_gfx1250.py variable-K takes a_index / b_index (TDM gather mode, 16 rows
  per op, indices prefetched a k-tile ahead, tail rows re-read the last row, empty experts zeroed);
  fused_mega_moe_gfx1250_impl.py passes pool_row_to_recv_token; dispatch_only_gfx1250 returns the token
  buffer; pool_rows_of removed
- Result: 1276.623 vs 1257.834 TF ✅ (+1.494%; mean of 4 runs 1275.889; same-session round-8 1248.641, +2.182%);
  +2.3 / +2.1 / +2.0% per shape same-session, all from the backward
- Test: PASS (EP2 5, EP1 19, grouped-gemm FLYDSL 87, bench SNR >= 48.5 dB; isolated gather vs dense max |diff| 0)
- Decision: accepted
- Detailed data: rounds/round-9/summary.md
- Notes: gather mode costs 3-5% of the wgrad kernel by itself (identity rows as slow as random ones), so the
  net is ~2/3 of the removed gathers
- Correction (2026-10-23 06:58 UTC): an empty last expert read the row table one entry past its end (the
  table read is an unbounded global load; Mega's table is a slice of a larger workspace, so the bench above was
  safe). Empty experts now read entry 0. Re-validated: EP2 5, EP1 19, grouped-gemm FLYDSL 87, isolated max |diff|
  0; 1281.979 TF (reps 1276.913, 1276.132; mean 1278.341) vs same-session round-8 1254.526 (+1.898%). Pre-fix
  artifacts in rounds/round-9/artifacts_pre_bounds_fix/

### EP4 validation of rounds 2-9 (GPUs 0-3)
- Time: 2026-10-26 02:14 (UTC)
- Method: every round commit's tree benched at EP4 (one shape per torchrun, own FlyDSL cache dir per commit);
  baseline benched first and last; round 8 / round 9 repeated alternately on T=8192 random
- Result (fwd+bwd ms, base -> round 9): T=8192 random 14.903 -> 11.553 (-22.5%), skew1 30.995 -> 27.456
  (-11.4%), T=4096 random 8.184 -> 6.784 (-17.1%); geomean step TF 882.3 -> 1064.7 (+20.7%; EP2 +12.7%).
  Largest step: round 6 (-1.36 ms on T=8192 random, 3.1x its EP2 gain); round 5 -0.40 and round 8 -0.32 ms
  also exceed EP2
- Test: bench SNR >= 48.64 dB (39 runs); EP pytest 4/5 in one process, test_staged_bias_gpt_oss hangs after the
  others (symmetric-buffer re-creation rendezvous, same at baseline) and passes alone
- Detailed data: ep4_validation/summary.md
- Notes: round 9 at T=8192 random is bimodal at EP4 (11.53 ms in 3 of 5 runs, 11.79 / 12.08 otherwise);
  forward-only timing at EP4 swings by up to 1 ms between runs, fwd+bwd does not
