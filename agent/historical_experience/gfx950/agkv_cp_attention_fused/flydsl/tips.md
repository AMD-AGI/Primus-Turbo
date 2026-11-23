# gfx950 / agkv_cp_attention_fused / flydsl tips

Mode `fused` of the all-gather-KV context-parallel attention (`attention_agkv.py`): the forward is one persistent
FlyDSL kernel with the K/V all-gather inside it (`agkv_fused_fwd`), the backward one persistent kernel as well
(`agkv_fused_bwd`: dK / dV, their reduce-scatter and dQ; `PRIMUS_TURBO_AGKV_FUSED_BWD=0` runs mega's). 8 x MI355X
XGMI mesh, zigzag cp 8, mbs 1, `FLUX_ATTN_FLYDSL=1`.

### 2026-11-23 03:55 UTC final — STATE OF THE BRANCH (5d24cb67c)
- `bench_agkv_cp.py --modes mega fused`, us, max over ranks; floor = attention fwd + bwd with no exchange; hidden = (naive - mode) / (naive - floor) of fwd + bwd, over 100 % where the mode beats the floor

| shape | mega fwd | fused fwd | mega bwd | fused bwd | mega fwd+bwd | fused fwd+bwd | floor | hidden mega / fused |
|---|---|---|---|---|---|---|---|---|
| llama3.1-8b 16K | 562 | 415 | 1282 | 1137 | 1854 | 1537 | 1461 | 50 % / 90 % |
| llama3.1-8b 64K | 4335 | 4011 | 14018 | 13785 | 18106 | 17726 | 17665 | 81 % / 97 % |
| qwen3-30b-a3b 16K | 461 | 337 | 1154 | 1024 | 1641 | 1322 | 1419 | 58 % / 118 % |
| qwen3-30b-a3b 32K | 1157 | 1129 | 4000 | 3489 | 5092 | 4628 | 4893 | 75 % / 134 % |
| qwen3-235b-a22b 16K | 628 | 646 | 2164 | 1904 | 2837 | 2524 | 2567 | 53 % / 108 % |
| qwen3-235b-a22b 32K | 2193 | 2124 | 7696 | 6826 | 9873 | 8984 | 9636 | 68 % / 189 % |
| llama3.1-70b 32K | 2239 | 2149 | 7344 | 6985 | 9524 | 9098 | 8948 | 61 % / 90 % |

- Takeaway: the exchange is hidden (fused fwd+bwd is within 150 us of the floor, below it where the persistent backward beats the stock flux backward); what is left is in the attention kernels themselves. The only shape whose fused forward still trails mega is qwen3-235b 16K (+18 us)
- Applicability: re-measure against the floor before any further comm work; a change that does not move the attention body itself has at most ~150 us left on these shapes

### 2026-11-23 03:55 UTC fwd parts — ACCEPTED (5d24cb67c)
- Context: the persistent forward merges the parts of a q chunk (own keys, remote first / second halves) through workspace slots by LSE
- Signal: 8 parts -> one part per half: qwen3-235b 16K 682 -> 646 us, qwen3-30b 32K 1172 -> 1129, llama3.1-8b 16K 426 -> 415, llama3.1-70b 32K 2164 -> 2149; error against mega 3.8e-3 -> 2.4e-3
- Takeaway: every part costs a slot written and read back by the merge (16 MB per slot at Hq 64, c 1024); that traffic outweighs the finer load balance, so split as coarsely as the arrival order allows
- Applicability: any split-K / split-KV attention merged through global memory; count slot bytes before adding parts

### 2026-11-23 bwd dQ claim hint — BUG FIXED (e9f53c7a6)
- Context: the persistent backward hands out a head group's dQ units once all its items are done; a hint tells work-groups where to start looking
- Signal: the hint could be advanced past a head group whose units were not all claimed, so that group's dQ was never computed (silent wrong dQ on some ranks)
- Takeaway: advance a shared scan hint only for the hint's own group and only once that group is fully claimed; never from a look-ahead slot
- Applicability: any persistent kernel with look-ahead claiming over counters; test with every rank and both chunks, not rank 0 only

### 2026-11-23 04:29 UTC bwd atomic dQ — REJECTED
- Context: replace the dQ band partials + in-kernel band sum with atomics into one image (probe, timing only, current BLOCK_KV 128 tile)
- Signal: fused bwd, us, current / bf16 pk atomics / fp32 atomics: llama3.1-8b 16K 1131 / 1588 / 1815, llama3.1-8b 64K 13893 / 25630 / 26236, qwen3-235b 32K 6833 / 7236 / 13513
- Takeaway: per rank the dQ contributions are ~16 GB at llama3.1-8b 64K; written as partials and summed they cost ~2 ms, as atomics ~14 ms (~1.2 TB/s atomic throughput). The single-GPU a16 win (17-33 % on square shapes) comes from its 256-wide band halving the contributions and from split-K partials being far larger there, not from atomics being cheap
- Applicability: atomics only pay where the partial workspace itself does not fit or the band is wide; at cp 8 the per-rank partials are small enough that split-K wins

### 2026-11-23 06:17 UTC bwd BLOCK_KV 256 — REJECTED
- Context: halve the dQ partial traffic by giving the embedded flux dK/dV body BLOCK_KV 256 (BLOCK_Q 32, kv-halves pair body), split-K dQ kept
- Signal: correct (rel err 3.6-4.4e-3 against mega) but llama3.1-8b 64K 13790 -> 22345 us, qwen3-235b 32K 6847 -> 10454; 1261-2930 VGPRs spilled, ~1 KB scratch (BLOCK_KV 128: 0 spills, ~504 VGPRs). a16's knobs (mfma_tie 3, q_pref, g3_st_at 0 / g3_st_n 1) still spill 1500-2500
- Takeaway: the stock a16 body at 256 already sits at 512 VGPRs with a few spills; the split-K store addressing plus the persistent kernel's state push it over. Making it spill-free is register-allocation work in the flux body, for at most ~7 % (1 ms at llama3.1-8b 64K)
- Applicability: check `vgpr_spill_count` (`spills.py`) of the embedded body before timing any tile change in a persistent kernel

### 2026-11-23 14:11 UTC Primus end to end — DEADLOCK FOUND, numbers
- Context: Primus llama3.1-8B pretrain, cp 8 (`cp_comm_type all_gather`), mbs 1, GBS 8, 32 layers, mock data, distributed optimizer with `overlap_grad_reduce` / `overlap_param_gather`; ms/iter is the median of iterations 4-20
- Signal (hang): with the runner's `GPU_MAX_HW_QUEUES=2`, fused hung in iteration 2 (rank 0 `agkv_fused_bwd` flag wait timeout, the peers never launched their backward). Bisect, 32 layers GBS 8: both overlaps off passes; reserving 8 CUs from the persistent grid still hangs; `GPU_MAX_HW_QUEUES=8` passes (20 iterations, every config). Mega passed with 2 queues but ran 6 % slower than with 8
- Signal (numbers, `GPU_MAX_HW_QUEUES=8` everywhere; TE = Transformer Engine attention with its own CP):

| seq | TE p2p | TE all_gather | agkv mega | agkv fused | fused tok/s/GPU | max alloc TE / fused |
|---|---|---|---|---|---|---|
| 32K | 5102 | 3009 | 2817 | 2737 | 11972 | 75.4 / 81.2 GB |
| 64K | 9547 | 7249 | 7074 | 7027 | 9327 | 93.5 / 114.5 GB |

  Loss matches TE at iteration 1 and within bf16 noise at 10 / 20 (32K: 8.274-8.281, 5.977-5.997)
- Takeaway: on this ROCm the RCCL streams (`TORCH_NCCL_HIGH_PRIORITY=1`) share the compute stream's hardware queue when queues are few; a rank whose RCCL kernel is queued ahead of its attention kernel waits for a peer whose attention kernel, queued ahead of its RCCL kernel, spins for it. Any kernel that waits on its peers' kernels needs the collectives on other queues; leaving CUs free does not help
- Applicability: run agkv (fused or mega) with `GPU_MAX_HW_QUEUES>=8`; the Primus example config sets it in its `env:` block and `PrimusTurboAttention` warns below 8. Fused saves the gathered K / V for the backward: +6 GB (32K) / +21 GB (64K) per GPU over TE at 32 layers
