# RL attention (batch-invariant)

On-policy RL needs the trainer to recompute exactly the log-probabilities the rollout engine
sampled with. Ordinary attention kernels do not give that. The rollout runs paged decode and prefill
with batch-dependent tiling and split-KV, the trainer runs a different, dense kernel, and the two
round differently. RL attention runs one kernel family on both sides, Triton or (on gfx950) FlyDSL.
A token's forward output is **bitwise identical** whether it is computed in a rollout prefill, in a
rollout decode step, or in the training forward, and whatever else shares its batch.

Supported: bf16 / fp16, GQA / MQA, head dims up to 256, causal (bottom-right aligned), sliding
window, per-head attention sinks (with their gradient), packed variable-length sequences, a
vLLM-layout paged KV cache, and visibility masks. The backward is deterministic (no atomics).
Tested on MI355X (gfx950). The Triton kernels are derived from aiter's `unified_attention` and
one-kernel backward (MIT); the FlyDSL forward is forked from Turbo's gfx950 dual-wave flash-attention
forward (see [Backends](#backends)).

## Usage

Through the normal attention interface:

```python
from primus_turbo.pytorch.ops.attention import flash_attn_varlen_func
from primus_turbo.pytorch.ops.attention.rl_attention import create_block_mask

# training: forward + backward
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                             causal=True, sink=sink, batch_invariant=True)

# rollout: paged KV cache [num_blocks, block_size, Hkv, D], forward only
out = flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q, None, max_seqlen_q, max_seqlen_k,
                             causal=True, sink=sink, block_table=block_table, seqused_k=seq_lens)
```

`flash_attn_func` and `flash_attn_varlen_func` run RL attention when any of `batch_invariant=True`,
`block_mask`, `kernel_options` or `block_table` is given, or when
`PRIMUS_TURBO_ATTN_BATCH_INVARIANT=1` is set. The environment switch reaches a trainer that calls
`flash_attn_*` from inside a framework. Without these, backend selection is unchanged.

- `causal=True` with `window_size=(left, 0)` is causal attention with a sliding window.
  `causal=False` is full attention over each sequence's keys.
- `sink` is a per-head logit.
- LSE comes back as `[B, Hq, Sq]` (dense) or `[Hq, total_q]` (varlen).
- Unsupported arguments raise: dropout, bias, ALiBi, `return_attn_probs`, right windows, and LSE on
  the paged path. `deterministic` has no effect, since the backward is always deterministic.

The same kernel has dedicated entry points: `rl_paged_attention_func` (rollout),
`rl_flash_attn_varlen_func` (THD) and `rl_flash_attn_func` (BSHD).

## Visibility masks

`create_block_mask(lo, hi)` gives each query token one contiguous, inclusive range of keys
`[lo, hi]` of its own sequence. Unlike `torch.nn.attention.flex_attention.create_block_mask`, it
takes the bounds directly, not a `mask_mod`. Bounds are **global key indices**: sequence `s` owns
keys `cu_seqlens_k[s] .. cu_seqlens_k[s+1]-1`. For a paged cache, `cu_seqlens_k` is the exclusive
cumulative sum of `seq_lens`; it is computed once per mask and reused by every layer. With `i` the
global index of a query token:

| Mask | lo | hi |
|---|---|---|
| Causal | start of i's sequence | i |
| Bidirectional span (e.g. an image) | span start | span end |
| Sliding window of W keys | max(seq start, i - W + 1) | i |
| No keys (output 0) | hi + 1 | any |

`create_block_mask(..., window_left=W)` narrows every row to `[max(lo, hi - W), hi]`. Keys
outside a row's interval are exact no-ops. Tiles are always walked in increasing key position, so a
mask never changes the order of additions. The backward skips work outside the intervals.
`mask.validate(cu_seqlens_q, cu_seqlens_k)` checks that no interval reaches outside its sequence.

## Kernel options

`kernel_options` (a dict, as in FlexAttention) sets kernel parameters per call.
`PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS="KEY=VALUE,..."` sets process-wide defaults, and a call
overrides them key by key. 0 or `auto` means the default.

| Key | Changes bits | Meaning |
|---|---|---|
| `TILE_SIZE` | yes | KV tile per inner step (default 64, 32 for D > 128) |
| `SEGMENT_SIZE` | yes | KV segment length, the split-KV unit (default 1024) |
| `BACKEND` | yes | `TRITON` (default) or `FLYDSL` (gfx950), see [Backends](#backends) |
| `MODE` | no | `auto`, `fused` (one pass) or `split` (split-KV + reduce) |
| `SPLIT_SEGS` | no | `FLYDSL`: segments per split-KV program (default 1) |
| `PACK` | no | `FLYDSL`: `auto`, `on` or `off`, rows as (token, head) of one query group |
| `BLOCK_M`, `num_warps`, `num_stages`, `waves_per_eu` | no | Schedule only |

**Keys that change bits must be identical in the rollout and the training process**; setting the
same environment variable in both is the simplest way. The others may differ per call and per
process. `num_warps` is clamped to `BLOCK_M / 16` when `BLOCK_M > 16`, because warps that also
split the columns of the score tile change the bits. The backward uses its own fixed configuration.

### Tuning the schedule

The defaults are measured on MI355X for common model shapes. For other shapes, tune the schedule
keys offline for the settings you run:

```bash
python benchmark/ops/training/tune_rl_attention.py --phase decode prefill train \
    --model llama3-8b --backend TRITON FLYDSL [--case BATCH,CTX[,Q_LEN] ...] [--tp-size N] -o tuned.json
```

For each setting and backend it searches `MODE`, `BLOCK_M`, `num_warps`, `num_stages`,
`waves_per_eu`, `SPLIT_SEGS` and `PACK` (coordinate descent; `--exhaustive` times the full grid),
keeps only candidates whose output is **bitwise equal to the default schedule's**, and prints the
fastest as a `kernel_options` string. It needs a 3% gain (`--min-gain`) to leave a default.
`TILE_SIZE` and `SEGMENT_SIZE` change bits, so they are not searched; pass the values your training
process uses with `--tile-size` / `--segment-size`. Training settings tune the forward only.
The search is also available in Python as
`primus_turbo.pytorch.ops.attention.rl_attention_tune.tune(run, search_space(backend, max_seqlen_q))`,
where `run(options)` makes the call.

## Backends

The two backends implement the same contract but give different bits, so **rollout and training must
use the same `BACKEND`**. A backend that cannot take a call raises; there is no silent fallback and no
per-shape auto-tuning across backends.

| | TRITON | FLYDSL |
|---|---|---|
| Hardware | any ROCm GPU Triton supports | gfx950 (MI350 / MI355) |
| dtypes, head size | bf16 / fp16, D <= 256 | bf16, D = 64 or 128, `softmax_scale` = 1/sqrt(D) |
| Paged cache | any page size | pages that are multiples of 64 |
| Masks, window, sinks, varlen, GQA | yes | yes |
| Split-KV for decode | yes (`SEGMENT_SIZE`) | yes (`SEGMENT_SIZE`, see [the segment cost](#known-issue-cost-of-segments-on-flydsl)) |
| Backward | Triton, deterministic | the same Triton backward |
| Options used | all | `SEGMENT_SIZE` (a multiple of 64), `BLOCK_M` (64 / 128); `TILE_SIZE` must be 0 or 64; `MODE` and the warp / stage / wave options are ignored |

**FLYDSL** (`primus_turbo/flydsl/attention/rl_attention_fwd.py`, helpers in
`primus_turbo/flydsl/utils/rl_attn_helper.py`) is a fork of the dual-wave, software-pipelined gfx950
forward. The stock kernel is batch invariant across batch composition, but not across prefill,
chunked prefill and decode: its lazy rescale moves the reference of every row in a wave when any row
outruns its own, so a row's bits depend on which rows share its wave (measured on MI355X once the
rescale fires, and with a sliding window even on random inputs). The fork changes:

- **Per-row rescale.** The rebase stays a wave-uniform branch, but rows that did not outrun their own
  reference take an exact identity (shift 0, rescale 1). The D <= 64 score bound, computed over the
  whole packed K, is off.
- **fp32 score scaling.** Q stays unscaled in bf16 and each score is
  `fma(q.k, log2(e) * softmax_scale, -reference)` in fp32. The stock kernel rounds the pre-scaled Q
  to bf16, which moves logits in the hundreds by whole units: max error vs fp32 was 0.15-1.2 on such
  inputs, against 0.015 now (the Triton kernel's level). **This is the main reason the RL forward is
  slower than the stock FlyDSL forward**: +8% (D=128) and +21% (D=64) on causal training shapes,
  about 0 with a sliding window. It costs instructions, not occupancy: the compiled kernel uses 250
  VGPRs instead of 230 at D=128 (170 instead of 150 at D=64), with no spills and the same 2 waves
  per SIMD (see the register table below).
- **Visibility intervals.** A workgroup walks its rows' tile range with the windowed pipeline; tiles
  every row of a wave sees entirely skip the mask. Causal and window attention are converted to
  intervals for everything but plain causal self-attention, and the two paths give the same bits.
- **Paged cache.** Each 64-key tile lies in one page: one block-table lookup per tile, with a buffer
  descriptor sized to the keys below the sequence length, so unused cache slots are never read.
- **Packed decode rows.** With at most 16 query tokens per sequence, a workgroup's rows are
  (token, head) pairs of one query group, so each K/V tile is read once per KV head.
- **Fixed segments and split-KV.** When a sequence has more keys than `SEGMENT_SIZE`, each row's
  state restarts at every fixed absolute segment and is merged, in order and with exact IEEE
  arithmetic, into an outer state carried beside it. Decode with too few workgroups for the GPU then
  runs one program per segment and merges the partial states in a second launch with the same
  arithmetic and the same normalize / store code, so the bits equal the training forward's. Rows
  whose keys fit in one segment are unchanged (merging into an empty state is exact).
- Visibility metadata (key starts, intervals, per-32-row bounds) is cached per step on the identity
  and version of its input tensors.

Speed on MI355X (bf16, TP 1, `benchmark/ops/training/bench_rl_attention.py`; "turbo" is the default,
not batch-invariant `flash_attn_varlen_func` backend; SEGMENT_SIZE 1024, so every case below except
the 4k decodes of short contexts runs the segmented FLYDSL build):

| Case | aiter / turbo | TRITON | FLYDSL |
|---|---|---|---|
| Training fwd, Llama-3-8B 8 x 4096 | 1.04 ms | 2.57 ms | 1.43 ms |
| Training fwd + bwd, same | 6.04 ms | 11.06 ms | 9.96 ms |
| Training fwd, Qwen3-32B 2 x 16384 | 7.14 ms | 17.10 ms | 10.69 ms |
| Prefill 2048 on 16k, Llama-3-8B | 572 us | 1963 us | 862 us |
| Decode 1 x 4k, Llama-3-8B | 67 us | 74 us | 60 us |
| Decode 64 x 4k, Llama-3-8B | 173 us | 175 us | 240 us |
| Decode 1 x 32k, Llama-3-8B | 67 us | 76 us | 125 us |
| Decode 8 x 32k, Llama-3-8B | 178 us | 195 us | 370 us |
| Decode 8 x 4k, GPT-OSS-120B (window 128, sinks) | | 77 us | 34 us |

### Known issue: cost of segments on FLYDSL

Split-KV decode has to give the same bits as the training forward, so the training and prefill
kernel must compute the same per-segment partial states and merge them in the same order, rather
than run one online softmax across the whole sequence (the two round differently). It therefore
carries a second, outer softmax state per row, about 64 more registers per lane at D=128 (32 at
D=64), in a kernel tuned to just fit its register budget. Measured on MI355X, for launches with a
sequence longer than `SEGMENT_SIZE` (shorter ones run the unsegmented build, with the same bits):

| | Without segments | With segments |
|---|---|---|
| Training fwd, D=128, 8 x 4096 | 1.08 ms | 1.43 ms (+32%; at 2 waves / SIMD and 128-row blocks it was +68%) |
| Training fwd, D=64, 8 x 4096 | 1.24 ms | 1.38 ms (+11%) |
| Prefill 2048 on 16k, D=128 | 562 us | 862 us (+53%) |
| Packed decode 64 x 4k, D=128 | 229 us | 240 us (+5%) |

The segment length does not matter (16 or 32 tiles cost the same); the cost is the extra live state.
Register use from the compiler's ISA metadata (`FLYDSL_DUMP_IR=1 FLYDSL_DEBUG_DUMP_ASM=1`, then
`.vgpr_count` / `.agpr_count` / `.vgpr_spill_count` in `15_final_isa.s`; on gfx950 VGPRs and AGPRs
share a 512-entry pool per SIMD lane, so waves per SIMD = 512 / (VGPR + AGPR), capped by
`waves_per_eu`):

| Build | VGPR | AGPR | Spilled VGPRs (scratch) | LDS | Waves / SIMD |
|---|---|---|---|---|---|
| D=128 stock FlyDSL forward | 230 | 0 | 0 | 68 KB | 2 |
| D=128 RL forward | 250 | 0 | 0 | 68 KB | 2 |
| D=128 RL + segments, 2 waves, 128-row blocks | 256 (cap) | 0 | 46 (188 B) | 68 KB | 2 |
| D=128 RL + segments, 1 wave, 64-row blocks (used) | 300 | 44 | 0 | 68 KB | 1 |
| D=64 stock / RL / RL + segments | 150 / 170 / 200 | 0 | 0 | 34 KB | 2 |

So at D=128 the segment cost is a register spill at 2 waves per SIMD (+68%) or, in the configuration
used, half the occupancy without spills (+32%); at D=64 nothing spills and occupancy is unchanged.
Because the backward (Triton) is about 8x the forward, the training forward cost is about +4% of
attention forward + backward. Possible fixes, not yet tried:

- **Outer state in LDS.** It is only touched at segment boundaries (every 16 tiles), so it could live
  in shared memory instead of registers: about 64 KB per 256-thread workgroup on top of the 68 KB the
  D=128 kernel uses, so a 160 KB CU would hold one workgroup; the gain over the current one wave per
  SIMD is uncertain. Fewer rows per workgroup shrink it proportionally.
- **Outer state in AGPRs.** Pin it to the accumulation registers, which the pipeline uses less, so it
  does not compete with the score and P registers.
- **Larger `SEGMENT_SIZE`.** Fewer launches pay at all (e.g. 4096: training on sequences up to 4k is
  unaffected), at the price of less decode parallelism for long contexts.
- **Segment-parallel prefill.** For long chunked prefills, run the split programs plus the merge, as
  decode does, instead of the single walk; costs the partial-state traffic but uses all CUs.
- **Block-table prefetch** for the paged path (+14% on D=128 prefill, independent of segments).

## How the invariance is achieved (TRITON)

A row's result depends only on the order of its floating-point reductions. The kernels fix that
order per model and make everything else a pure scheduling choice:

- **Fixed reduction order.** Keys are reduced in tiles of `TILE_SIZE`, in increasing absolute
  position, inside fixed absolute segments of `SEGMENT_SIZE`. Segments are merged in order. Nothing
  depends on the batch: upstream sizes segments from the batch.
- **One code path for every mode.** The fused kernel walks the segments itself. The split kernel
  computes one segment per program and a reduce kernel merges them in the same order. Both run the
  same `@triton.jit` tile, merge, sink and epilogue helpers.
- **Masking that cannot reorder.** The running max stays `-inf` until a row sees a key (upstream
  resets it to 0, which made a fully masked leading tile round differently depending on block
  grouping). Masks apply per element on the absolute tile grid. Sinks fold once after all keys.
- **Paged = contiguous.** Contiguous K/V is read through the same kernels with page size 1, and key
  loads are always masked by sequence length.

## Testing methodology

Tests live in `tests/pytorch/ops/test_rl_attention.py` (Triton backend) and
`tests/pytorch/ops/test_rl_attention_flydsl.py` (FlyDSL backend, gfx950; every comparison stays within
the backend), with an fp32 reference in `tests/pytorch/ref/rl_attention_ref.py`:

```bash
pytest tests/pytorch/ops/test_rl_attention.py tests/pytorch/ops/test_rl_attention_flydsl.py
```

### Two kinds of checks

- **Accuracy** compares against the fp32 PyTorch reference with `torch.testing.assert_close`:
  atol / rtol 2e-2 for outputs, 5e-2 for gradients. This shows the kernel computes attention
  correctly.
- **Bitwise invariance** compares two kernel results with `torch.equal(a.view(torch.int16),
  b.view(torch.int16))`. Every bit must match, with no tolerance; this is the RL guarantee itself. A
  tolerance would hide exactly the 1-ulp differences that break on-policy RL.

### Inputs chosen to expose differences

- **Shapes that stress tile edges.** Most tests use sequence lengths `[1, 77, 300, 129]`. 1 behaves
  like decode; the others end partway through a KV tile, so the masked last tile is always
  exercised. Head shapes `(Hq, Hkv, D)` cover GQA ratios 1-8, a group size that is not a power of
  two (6:2), and D = 64 / 128 / 256.
- **Short segments.** A fixture sets `SEGMENT_SIZE=128` through
  `PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS`, as a deployment would, so split-KV and multi-segment fused
  passes happen at test sizes.
- **Adversarial paged caches.** K/V are scattered into a cache with shuffled physical blocks (page
  sizes 1, 16, 64). Unused slots hold large garbage (about 100x), so a missing key mask shows up
  immediately.
- **Masks that mix every case.** Generated masks combine bidirectional image spans inside causal
  sequences, random non-causal ranges, and empty rows, which must output exactly zero.
- **Seeded data.** Inputs come from seeded CPU generators, so failures reproduce.

### Bitwise invariance (the RL guarantees)

Each test changes one thing a token's result must not depend on, and requires identical bits:

| Guarantee | Test | What varies |
|---|---|---|
| Layout | `test_paged_split_fused_contiguous_bitwise`, `test_visibility_paged_split_fused_bitwise` | Paged cache (each page size) vs contiguous K/V; split-KV vs fused |
| Launch | `test_launch_invariance` | 16 (`BLOCK_M`, `num_warps`) launches given through `kernel_options`, with and without masks |
| Schedule (FLYDSL) | `test_schedule_knobs_do_not_change_bits` | `BLOCK_M`, `waves_per_eu`, `MODE`, `SPLIT_SEGS`, `PACK` on training, decode and a 37-token chunk |
| Batch | `test_batch_invariance`, `test_visibility_batch_invariance` | A 200-token sequence alone vs packed among others (`[1, 5]`, `[333, 64, 17]`, six of 129), with masks and sinks |
| Mask form | `test_causal_intervals_equal_builtin` | Causal / sliding window as `[lo, hi]` intervals vs the built-in masks |
| Rollout vs training | `test_causal_rollout_matches_training`, `test_visibility_rollout_matches_training`, `test_flash_attn_varlen_func_paged_rollout_matches_training` | See below |
| Options route | `test_env_options_and_per_call_override` | The same options through the environment vs per call (and a different `TILE_SIZE` must change the bits) |
| Interface | `test_flash_attn_*_routes_to_rl_kernel`, `test_env_switch_routes_flash_attn` | `flash_attn_*` vs the `rl_*` entry points |

The **rollout-vs-training** tests simulate an RL step end to end at the kernel level:

1. **Training side:** one varlen forward over the whole sequence (prompt + response).
2. **Rollout side:** the prompt is prefilled into a paged cache, either all at once or in 37-token
   chunks (chunked prefill). The response is then decoded one token per step. Every step is batched
   with another sequence that is also decoding, so tile and program assignment differ from training.
   Each step runs in both `fused` and `split` mode.
3. **Check:** every rollout token's output equals the training output at the same position, bit for
   bit.

They run with causal masks (each window and page size), with a bidirectional image span in the
prompt plus sinks, and with both sides going through `flash_attn_varlen_func`.

### Backward

The backward never has to match inference. It must be correct and deterministic:

- dq / dk / dv / dsink match the fp32 reference's autograd, for causal, windowed, masked and sink
  cases.
- Two runs give bitwise-identical gradients.
- Short intervals scattered over long sequences check that the pruned walks skip only work that
  does not matter.

### Error handling

Unknown or malformed kernel options, unsupported arguments, a paged `block_table` without
`seqused_k`, and masks reaching outside their sequence all raise `ValueError`.

### Not covered

These tests check invariance at the kernel level. End-to-end on-policy matching also needs every
other op in the model (GEMMs, norms, and tensor-parallel all-reduces) to be batch invariant and
identical between the engine and the trainer, and the engine must call this kernel with the same
bit-changing options.
