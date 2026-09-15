# Facts

Things this job has measured and can build on. Every entry names the round that
measured it. The bottleneck line at the top is what the next round reads first.

## CURRENT BOTTLENECK (round 1, confidence MEDIUM)

**Register spill to scratch at 1 wave/SIMD, on a part that is hard power-capped.**
`bwd_kernel_causal` is 99.28% of the backward and holds 1024 VGPRs with **326
spilled, 1308 B/lane of scratch** and LDS at only 65,536 of 327,680 B (20%). At
`num_warps=4` wave32 the flat `131072 / NUM_THREADS` budget is exactly 1024, so one
wave per SIMD is forced and there is no second wave to cover a scratch reload: every
spill is exposed latency. The card is separately measured pinned at 2497-2502 W of a
2500 W limit with the GFX clock held 29% below maximum, which is why the direction is
*delete* work rather than overlap it.

**Confidence is medium and the reason matters.** This is an inference from a static
resource census plus the corpus, not a reading. gfx1250 has **no byte counter at any
level** and its **entire WMMA FLOP counter family reads exactly 0**, so there is no
counter-derived roofline on this part; and `rocprofv3 --stats --kernel-trace` recorded
**zero dispatches** on this job's benchmark, so round 1 never got a per-kernel
instrument working at all. A deep round with six analyses behind it should not treat
this as settled -- it is one agent's reading from one survey, and `r1.i6.g6` (get PC
sampling or ATT working) is in the pool precisely because the instrument is missing.

What round 1's measurement does support: both wins were *deletions* and both paid
(13.2% together), and `num_warps=8` -- which halves the lane budget to 512 against a
kernel already spilling -- measured **2.36x slower**, not merely flat. That is the
shape of a register-pressure limit. What it does not support is any claim about where
inside the kernel the traffic is.

## r1.i1.g1 -- `waves_per_eu: 1 -> 0` in the vendored one-kernel backward config

EXECUTED round 1 (arm A). `_DEFAULT_ONEKERNEL_CONFIG` pins `waves_per_eu: 1`.
`bwd_kernel_causal` already occupies all 1024 VGPRs and spills 326, so it gets one
wave per SIMD whatever the hint says; the hint's only effect is to constrain the
register allocator. Setting it to 0 (`_ZERO_MEANS_UNSET`, i.e. let the backend
choose) measured **9.571 ms vs 9.996 ms, -4.3%** on `b4_s8192_hq32_hkv8_d128`.
`waves_per_eu: 2` is catastrophic at 50.6 ms, which is the same story from the other
side: the kernel cannot fit two waves and being asked to try wrecks it.

## r1.i2.g2 -- `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1` around the backward launch

EXECUTED round 1 (arm B). The Triton AMD backend can lower a transposed dot operand
either through LDS or inside the thread's own registers; the pass is enabled by
default **only for gfx942** (`triton/backends/amd/compiler.py:29`), so gfx1250 never
gets it. The fused backward transposes on nearly every dot. Measured **9.418 ms vs
9.996 ms, -5.8%**, and with g1 **8.696 ms, -13.0%** -- superadditive, so the two are
not competing for the same resource.

The knob is read at compile time and is cache-invalidating, so it must be set and
restored *around the launch*: left set process-wide it also changes how
`validation.py` compiles `beat` and `baseline`, which would move the gate ratio
without moving the kernel.

### Both of the above, as shipped

Round 1 built each alone from `op/current` (B from the same base as A, not on top of
it), then the merge because neither lost. Graded on a cleared Triton cache, unit tests
first, order `001 · 000 · 000 · 001` twice, every shape, GPU[1] verified idle:

| | bwd ms | bwd TF/s | |
| --- | --- | --- | --- |
| round 0 incumbent, re-measured in the same session | 10.0099 | 549.2 | stored figure 548.9 -- 0.05% apart |
| A alone (`r1.i1.g1`) | 9.5873 | 573.5 | 1.043x |
| B alone (`r1.i2.g2`) | 9.4248 | 583.4 | 1.063x |
| **merge, shipped** | **8.683** | **633.2** | **1.1529x** |
| `op/beat` flex_attention, same session | 11.680 | 470.7 | target = 1.50x = 706.0 |

`score` 0.8968. Correctness unchanged to two decimals on every shape (min 52.04 dB
over all four tensors and all four gated shapes). The cold cache moved nothing
(8.609-8.697 ms cold vs 8.700-8.703 warm), so no stale binary was ever involved.
Every edge shape improved too, 5.4% to 16.5% -- the spec-shape win was not paid for
out of them.

## The launch-configuration surface is closed (round 1)

Not an idea, a boundary. Round 1 swept tiles, warps and the documented HIP knobs and
found a sharp local optimum that the shipped config already sits on. Specifically:

* `num_warps=8` is **2.36x slower** (22.5-23.6 ms), and mechanically so -- the per-lane
  VGPR budget on gfx1250 is `131072 / NUM_THREADS`, so 4 warps wave32 gives exactly the
  1024 the census shows and 8 warps gives 512 to a kernel already spilling 326. The
  occupancy precondition `current_vgpr x new_waves <= 1024` says no wave count above 1
  is legal here. **Do not import HipKittens' `gqa_d128` finding that 8 warps beat 4 by
  1.332x** -- that is gfx950, where the 4-warp arm spilled and extra warps rescued it.
* `waves_per_eu: 2` is catastrophic at 50.6 ms.
* `BLOCK_N1` and `BLOCK_M2` are **dead as env keys**: `dense_fused_backward` overwrites
  them from `fused_backward_tile(seqlen_k)` after the config is read. A sweep over them
  measures nothing and reports no error. `BLK_SLICE_FACTOR` is not overwritten, which is
  why it survives in the pool as `r1.i5.g5`.

The next 10.3% to the gate is not on this surface.

## `validation.py` cannot judge a vendoring candidate without a candidate-side fix (round 1)

Pre-existing, hits every round, and reproduces with an **unmodified copy of
`op/current`** as the candidate. Three distinct failures, all fixed in the shipped
`impl.py` / vendored tree, all documented in `rounds/001/1-opt/opt.md` s3:

1. `op/baseline` guards on the name `primus_turbo` being absent from `sys.modules`, so
   any candidate that vendors the package trips "an installed primus_turbo was imported
   before this module". Fix: move the vendored modules to a private `_op_evolve_<n>.`
   prefix rather than deleting them.
2. torch's opaque-type registry is process-global and keyed by qualname, so the second
   `primus_turbo` tree in the process dies on `Float8QuantConfig`. Fix: make
   `torch._C._register_opaque_type` idempotent -- do NOT unregister, which perturbs a
   registry the dispatcher reads.
3. **Use-after-free in the dispatcher.** Both trees declare
   `primus_turbo::attention_triton_forward_impl`; two `torch.library` fragments on one
   `OperatorEntry` means the second one's collection destroys the schema under the
   first's live `OpOverload`. Fix: register under `primus_turbo_opevolve::`.

Any future round that keeps the vendored layout inherits all three for free by starting
from `op/current`. A round that restructures the import must re-check them.
