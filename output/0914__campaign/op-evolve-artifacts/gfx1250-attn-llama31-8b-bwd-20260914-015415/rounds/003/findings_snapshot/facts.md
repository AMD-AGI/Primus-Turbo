# Facts

Things this job has measured and can build on. Every entry names the round that
measured it. The bottleneck line at the top is what the next round reads first.

## CURRENT BOTTLENECK (round 3, confidence LOW -- and the drop from MEDIUM is the point)

**Unresolved between instruction issue and the memory path, and round 3 removed the proxy
that rounds 1-2 were using to decide.** The direction of round 2's line survives -- the
time is in the dk/dv loop body, the dq pass is cheaper and cleaner -- but the *reason*
named in it does not, and the next round must not build on it as written.

What still stands, measured:

- **All of it is in one kernel.** `torch.profiler`, round 3: `bwd_kernel_causal` is
  **99.2%** of the backward (8.5832 ms/iter), `_bwd_preprocess` 0.5%, the host-side LSE
  gather and allocations 0.4% combined. There is no cheap slice anywhere else.
- **The dk/dv pass is the expensive half and owns all the spilling.** Deletion probe,
  round 3: dk/dv alone **5.906 ms** at 263 spills; dq alone **3.858 ms** at
  `vgpr_count 971` with **zero spills and zero scratch**; fused 8.678. Fusion costs the
  allocator almost nothing -- the two passes are sequential and barely interfere.
- **The fusion is load-bearing.** One pid does dk/dv for K-tile `pid` and dq for Q-tile
  `pid`, and at `seqlen 8192` the two are exact complements: every program runs 264 loop
  trips. Split apart, the halves sum to 9.764 ms against 8.678 fused.

**What round 3 refuted, and why the confidence is now LOW.** Round 2 left
`vgpr_spill_count` standing as the within-tile predictor of time. **It is not one.** At a
FIXED tile, fixed source and fixed launch config, `AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=1`
compiles to **187 spills -- the lowest figure this job has ever produced, 30% below the
incumbent's 266 -- and runs at 15.839 ms against 8.679, +82%.** `AMDGCN_USE_BUFFER_OPS=0`
is a second such point (250 spills, +74%). Both change *addressing*, not register
pressure, and both cost ~12% more instructions for 75-82% more time. No spill-count and no
instruction-count model explains that, and it sits badly with "the kernel is not waiting on
memory (154 GB/s)". **Use the DISCARD half of `r2.i3.g10` as a heuristic if you like; do
not justify a candidate by a spill count alone.**

**And the 37% overhead figure is an instruction count, not a time share.** gfx1250 is hard
power-capped -- 2497-2502 W of 2500, GFX clock held at 1699-1703 MHz against `MAX_CLK`
2400, i.e. 29% below maximum, measured on this part in the corpus. Under a cap, bytes price
at full value and cycles/instructions at **20-45%**. The 517 "register-pressure overhead"
instructions per dk/dv iteration are 363 `s_set_vgpr_msb` and 84 `v_nop`, which switch
almost nothing: at that coefficient they are **6-14% of time, not 37%**. The part of that
figure worth chasing is the **77 `scratch_load` per iteration**, because those move bytes.

**Why this cannot be settled right now, and what to do instead.** The measurement that
would separate the two families is PC sampling, and it **does not run on this kernel** --
see `r3.i3.g14` in `dead_ends.md`. There is no instruction-level dynamic instrument on this
part at all. So the next round should either (a) take a candidate that pays under *both*
readings -- anything that removes scratch bytes qualifies, since it removes instructions
too -- or (b) discriminate by deletion probe, which is the technique that worked twice in
round 3 and needs no profiler. **Do not spend a row re-attempting PC sampling.**

One correction the next round must carry, because it was wrong in `pool.md` for a round:
**this kernel uses 65,536 B of LDS, not zero.** `group_segment_fixed_size: 0` in the ELF is
0 because Triton passes the LDS *dynamically at launch*; `shared = 65536` in the kernel's
own metadata JSON is the real figure, and it is 20% of the 327,680 available, with ~262 KB
free. Round 1 read this correctly, round 2 retracted it on the static field, and round 3
restored it. The 656 `ds_*` ops the census has always shown are real traffic against a real
allocation.

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

## Where the time is inside `bwd_kernel_causal` (round 2, static, no profiler)

`r1.i6.g6` asked for an instrument. `rocprofv3` is still dead on this part
(`dead_ends.md`) and it turned out not to matter: the compiled `.amdgcn` that Triton
writes into `TRITON_CACHE_DIR` answers the question for free. Find the backward
branches, map the loop regions, count instructions per region. No GPU, no counters.

Measured on round 1's shipped build (the incumbent), `rounds/002/1-opt/raw/`:

`vgpr_count 1024`, `vgpr_spill_count 266`, `sgpr_count 107`, `sgpr_spill_count 67`,
`private_segment_fixed_size 1064` B/lane, `group_segment_fixed_size 0`.

| loop region | instrs | wmma | `scratch_load` | `scratch_store` | `ds_*` | global |
| --- | --- | --- | --- | --- | --- | --- |
| dk/dv pass, masked inner | 1643 | 128 | **76** | 0 | 108 | 16 |
| dk/dv pass, unmasked inner | 1400 | 128 | **77** | 0 | 108 | 16 |
| dq pass, masked inner | 1091 | 96 | **0** | 0 | 88 | 8 |
| dq pass, unmasked inner | 807 | 96 | **0** | 0 | 88 | 8 |

**Every spill reload is in the dk/dv pass. The dq pass has none.** And the stores sit
outside the loops -- these are loop-invariant values spilled once and re-loaded on
every iteration (~119 dwords per iteration of the 266 spilled). That is the first
statement in this job about WHICH HALF of the kernel is the problem.

It matches the arithmetic exactly. At `BLOCK_N1=256`, `HEAD_DIM=128`, wave32,
`num_warps=4` (128 lanes): the dk/dv pass holds `dk` 256 + `dv` 256 + `k` 128 + `v`
128 = **768 VGPRs of loop-invariant state**; the dq pass holds `dq` 256 + `q` 128 +
`do` 128 = 512 and fits.

Histogram of ONE unmasked dk/dv iteration (128 wmma = the four dots at this tiling):

| count | instruction | |
| --- | --- | --- |
| **363** | `s_set_vgpr_msb` | **26% -- gfx1250's high-VGPR bank switch** |
| 128 | `v_wmma_f32_16x16x32_bf16` | the actual matrix work, 9.1% |
| 96 | `v_perm_b32` | in-thread transpose / bf16 packing |
| 84 | `v_nop` | hazard padding |
| 77 | `scratch_load_*` | spill reloads |
| 64 | `v_pk_mul_f32`, 64 `v_exp_f32`, 64 `v_cvt_pk_bf16_f32` | the softmax |
| 48 | `v_fma_f32` | the `*RCP_LN2 - m*RCP_LN2` fold, already fused |
| 41 | `v_add_nc_u32` / `v_add3_u32` | pointer-tensor increments |
| 1400 | total | |

`s_set_vgpr_msb` + `v_nop` + `scratch_load` = **517 of 1400, 37% of the dominant loop
is register-pressure overhead and not arithmetic.** gfx1250's flat 1024-VGPR file is
addressed 256 at a time and an instruction naming a register above `v255` needs the
MSB mode set first; at 1024 VGPRs in use that is roughly one per two VALU. This is a
tax that no amount of instruction deletion touches -- only lowering `vgpr_count` does.
It also explains 633 TF/s against a 2236-2667 TF/s bf16 GEMM: the kernel is
**instruction-bound on overhead**, not memory-bound.

Free corollary, checked in TTGIR: the `dk_pe = dk` / `dq_pe = dq` aliases (taken when
`PE_HEAD_DIM == 0`) are **already eliminated** -- the `scf.for` carries
`(dk, dv, curr_m, qT_ptrs, do_ptrs)` and no `_pe` value. Nobody needs to propose
removing them.

## Time tracks `vgpr_spill_count`, and nothing else does (round 2)

| build | `vgpr_spill_count` | scratch B/lane | bwd ms |
| --- | --- | --- | --- |
| round 0 | 326 | 1308 | 9.983 |
| `r2.i2.g9` | 325 | 1288 | 9.302 |
| `r2.i1.g8` | 307 | 1228 | 9.347 |
| round 1 ship | **266** | 1064 | **8.691** |

Monotone over 22% of spill range. Loop-body instruction count does **not** predict
time: round 2's arm A deleted exactly the 32 `v_pk_mul_f32` it targeted and got 7.5%
SLOWER, because the deletion cost 41 spills. This is the job's only quantitative
predictor and it is readable from a compile, with no GPU -- see `r2.i3.g10`.

## Measurement floor, measured (round 2)

Arms cannot share a process (three vendored trees segfault, `dead_ends.md`), so they
share a session: one process per arm, back to back, palindromically, inside one
`docker exec`. The incumbent's two slots read **8.6924 / 8.6899** in one session and
**8.6873 / 8.6945** in another -- a **0.03-0.08% floor**, tighter than round 1's
in-process 0.7-2.5%. A byte-identical copy of the incumbent's directory read 8.686 /
8.681 in the same shape, so the directory is not a variable. **Anything below ~0.5%
is noise; anything above 1% is real.**

## r2.i3.g10 -- gate a candidate on `vgpr_spill_count` before spending GPU time on it

**Built and used in round 2** (`rounds/002/1-opt/census_gate.py`). It compiles a
candidate, reads `vgpr_count`, `vgpr_spill_count`, `sgpr_spill_count`,
`private_segment_fixed_size` and a per-region instruction histogram out of the
`.amdgcn` Triton leaves in `TRITON_CACHE_DIR`, and prints DISCARD / MEASURE against
the incumbent's 266 spills. It needs a compile, not a timing run: no idle GPU, no
palindromic ordering, seconds instead of minutes. It paid for itself the same round by
killing route row 2 (`r2.i4.g11`) before a line of it was written.

**Its stated law is true only at a fixed tile, and that is the correction round 2 owes
the next round.** The section "Time tracks `vgpr_spill_count`, and nothing else does"
above is written from four points that are *all* at `BLOCK_N1=BLOCK_M2=256`. The gate
itself falsified the general form: at `BLOCK_N1=BLOCK_M2=128` the kernel compiles to
`vgpr_count=771`, **`vgpr_spill_count=0`, `sgpr_spill_count=0`, scratch 0** -- a
perfect score on the metric -- and round 1 timed that tile at **11.8-13.3 ms** against
the incumbent's 8.691. So:

- **within one tile**, fewer spills is faster, monotonically, and no other static
  number in the census predicts anything;
- **across tiles**, spills predict nothing, because shrinking the tile trades
  residency for Q/dO re-reads and the traffic wins.

Use it as a screen, never as an objective. A candidate that changes `BLOCK_N1`,
`BLOCK_M2` or the number of m-loop passes is outside its domain and has to be timed.
