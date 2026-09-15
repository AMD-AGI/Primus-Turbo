# Beat-target provenance

`op.target.beat` is non-null, so this arm exists. It is the bar the evolve loop
must clear by the spec's margin.

## Library and version

| | |
|---|---|
| **Library** | `torch.nn.attention.flex_attention` (PyTorch built-in, Inductor/Triton-generated) |
| **torch** | `2.11.0+rocm7.14.0a20260625` |
| **triton** | `3.6.0` |
| **Backward `num_warps`** | **4** (chosen by measurement -- see below) |

These strings are produced at runtime by `impl.py::library_version()` and are
re-printed by `benchmark.py` and `validation.py` on every run, so the version
recorded here cannot drift away from the version actually measured.

This library was available and ran. Nothing was skipped.

## Why flex_attention

It is the only attention implementation in the image that (a) is a real
library rather than something written for this job, (b) supports GQA
(`enable_gqa=True`) with `heads_q_per_kv = 4`, and (c) supports a custom
bottom-right causal mask. aiter was rejected as the bar: the corpus's
`knowledge/pitfalls/` records two separate aiter attention paths on this class
of shape that are fast *and wrong* (a `BLOCK_N1=256`-alone config that is 1.31x
faster with dq at 9.59 dB, and `mha_fused_bwd.py` returning dk at -0.22 dB with
`out` perfect). A bar that is only fast because it is incorrect is not a bar.

## `num_warps`: measured, not inherited -- this mattered a lot

Inductor's default heuristic gives the flex-attention backward template
`num_warps=8`. Measured in this job, on this node, each arm in its own process,
job shape `b4_s8192_hq32_hkv8_d128`:

| `num_warps` | fwd ms | bwd ms | bwd TFLOP/s | bwd spread |
|---|---|---|---|---|
| 8 (Inductor default) | 5.4364 | **30.6392** | 179.45 | 4.20% |
| **4 (adopted)** | 3.7857 | **11.6542** | 471.78 | 2.49% |

**2.63x apart.** Taking the default would have handed the loop a straw man: the
baseline already beats `num_warps=8` by 3.06x at round 0, so the 50% margin
would have been met before any optimization happened and the target would have
meant nothing. `num_warps=4` is set explicitly via `kernel_options` and is
overridable by `BEAT_BWD_NUM_WARPS` for anyone who wants to re-check the sweep.

No further tuning of the beat arm was attempted. This is a deliberate stopping
point: the goal was to remove an obviously-broken default, not to hand-tune the
bar into something the loop cannot reach. If a later round suspects the bar is
still soft, the sweep should be redone and this file updated.

## Caveat: block-sparse vs dense causal

`op.reference.logic` specifies dense bottom-right causal masking. flex_attention
implements causality through a `BlockMask`, which skips whole blocks that are
entirely masked and applies the element-wise predicate only on the diagonal
blocks. The arithmetic is equivalent; **the work is not identical**. On shapes
where the sequence length is a clean multiple of the block size the difference
is negligible, but on a ragged shape flex does relatively more masked-out work
inside the partial diagonal block.

This shows up in the measurements: on `ragged_tail` (Sq=Skv=1000) the beat arm
is 0.6707 ms against the baseline's 0.3135 ms, where on the clean `min_seqlen`
(512) it is *faster* than the baseline (0.1426 vs 0.1806 ms). The comparison is
still apples-to-apples in the sense that matters -- both compute the same
mathematical function on the same inputs and are timed the same way -- but the
block-granularity difference should be kept in mind before reading too much
into any single non-power-of-two shape.

## Known flakiness, and what was done about it

`knowledge/pitfalls/` records that flex_attention's BlockMask backward
**memory-faults roughly 2 runs in 12 (rc=139) when B > 1**. The `B=None`
broadcast form has not been seen to fault. This arm therefore builds its mask
with `B=None, H=None` -- the mask does not depend on batch or head anyway, so
this costs nothing and avoids the known-faulting form. No rc=139 was observed
in any run during Op Setup; the retry count is zero.

The mask and the compiled callable are cached per shape and built at first
call, never inside a timing loop. The `q/k/v` transposes into flex's
`(B, H, S, D)` layout are views, so no bytes move.
