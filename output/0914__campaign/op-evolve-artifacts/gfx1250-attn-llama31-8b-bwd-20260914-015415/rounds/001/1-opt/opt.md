# Round 1 — optimisation log (fast round)

Op: bf16 attention fwd+bwd, gate on the BACKWARD. Shape `b4_s8192_hq32_hkv8_d128`,
bshd, GQA 32/8, bottom-right causal. Device: logical GPU 1 in container
`op-evolve-gfx1250-attn-llama31-8b-bwd`, gfx1250, wave32. Every number below was
produced through `op/runjob.sh` on that device; raw logs in `raw/`.

Incumbent, as `op/validation.py` measures it (`raw/session.txt`, and the framework's
own run on `op/baseline`): backward **10.026 ms**, **1.166x** `op/beat/`, gate is
1.50x. Short by 22.3% of the target.

## 0. What round 1 inherited

`job_context/findings/` contained only the framework's empty `route.md`. No
`facts.md`, no `dead_ends.md`, no `pool.md`. Nothing was inherited; everything
below was established this round.

## 1. Where the time is

Expectation stated before measuring: the backward is a single fused kernel, so the
dispatch breakdown should be ~1 kernel and host-side work should be irrelevant.
Confirmed:

| kernel | ms | share |
|---|---|---|
| `bwd_kernel_causal` | 9.926 | 99.28% |
| `_bwd_preprocess` | 0.044 | 0.44% |
| everything else (gather/scatter/elementwise) | ~0.03 | ~0.3% |

(`raw/survey3.txt`, torch.profiler CUDA breakdown.) **Host-side and
launch-overhead candidates are dead for this op.** Anything that moves the number
has to move `bwd_kernel_causal` itself.

### rocprofv3 did not work here — recorded, not worked around silently

`rocprofv3 --stats --kernel-trace` on `op/benchmark.py` reported
`output generation :: 0.006 sec` and produced **no output files and zero kernel
dispatches**, both through `runjob.sh` and through a bare `docker exec`
(`raw/survey2.txt`). A control run on a plain torch matmul with the default
rocpd/sqlite format DID produce `*_results.db`; `--output-format csv` produced
nothing even for the control. Unresolved. The dispatch breakdown above therefore
comes from `torch.profiler`, and the static census below from the Triton cache.
First attempt also wrote to host `/tmp`, which is not bind-mounted into the
container — all profiler output now goes under `rounds/001/scratch/`.

### Free, code-free instrument: static register census

From the compiled `.amdgcn` metadata in the Triton cache:

| kernel | vgpr | vgpr spills | sgpr | sgpr spills | scratch B/lane | LDS | warps |
|---|---|---|---|---|---|---|---|
| `bwd_kernel_causal` | **1024** | **326** | 107 | 65 | **1308** | 65536 | 4 (wave32) |
| `attn_fwd` | 262 | 0 | — | 0 | 0 | — | 4 |

The backward sits at the top of gfx1250's flat 1024-VGPR file, spills 326 of them,
and runs 1 wave/SIMD. That is the structural story of this kernel.

Headroom calibration: 549 TFLOP/s on the backward FLOP basis, against 2236–2667
TFLOP/s for bf16 GEMM on this chip per the corpus. ~20–25% of achievable. The gap
is structural, not a knob.

## 2. What was tried, and what it measured

All sweeps: `op/benchmark.py`'s own timing path, spec shape, one process per
configuration, separate `TRITON_CACHE_DIR` per environment setting.
Session floor spread on the backward: **0.7–2.5%**, so "lost" below means worse by
more than ~2.5%.

### Falsified prediction, recorded as such

I predicted `num_warps=8` would relieve the 326-VGPR spill and win 10–25%.
It measured **22.5–23.6 ms, 2.36x SLOWER** (`raw/warps8.txt`, `raw/sweep3.txt`).
`num_warps=2` and `16` are also far worse. The spill is not the binding
constraint, or relieving it this way costs more than it saves.

### Tile / warp surface — a sharp local optimum (`raw/sweep1.txt`, `raw/sweep2.txt`)

| change | bwd ms |
|---|---|
| default: tile 256, BLOCK_M1 32, warps 4, stages 1 | **9.996** |
| tile 128 | 11.8–13.3 |
| tile 512 | 54–59 |
| BLOCK_M1 16 | 12.1 |
| BLOCK_M1 64 | 22–23 |
| num_warps 8 | 21–23 |
| num_stages 2 | 12.2–12.9 |

Every neighbour is worse, most of them by a lot. The vendored config is already at
a sharp optimum on this surface; there is nothing left in it.

### Harness trap found: two config keys are dead on this path

`PRIMUS_TURBO_FUSED_MHA_BWD_TUNE=BLOCK_N1=128,BLOCK_M2=128` returned **exactly**
10.0068 ms — identical to the default to four decimals. Root cause:
`dense_fused_backward` re-reads `fused_backward_tile(seqlen_k)` and overwrites
`BLOCK_N1`/`BLOCK_M2` *after* reading the config, so those two env keys cannot
take effect and `_verify_launch` cannot catch it. Measured tile variants only after
monkeypatching `FB.fused_backward_tile` in the sweep harness. Anyone reading a tile
sweep that used the env knob alone was reading the default, six times.

### Knob sweep (`raw/sweep3.txt`, env matrix)

| setting | bwd ms | verdict |
|---|---|---|
| default | 9.996 | — |
| **`waves_per_eu: 1 -> 0`** | **9.571** | −4.3%, kept (arm A) |
| **`TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1`** | **9.418** | −5.8%, kept (arm B) |
| both together | **8.696** | **−13.0%, superadditive** |
| `schedule_hint="attention"` | 9.633 | wins alone, but 8.941 with ITT vs 8.696 without — **hurts in combination, not shipped** |
| `schedule_hint="memory-bound-attention"` | 9.598 | subsumed |
| `kpack=2` | 10.016 | noise |
| `matrix_instr_nonkdim=32` | ~10.0 | noise |
| `TRITON_HIP_USE_BLOCK_PINGPONG=1` | ~10.0 | noise |
| `TRITON_HIP_USE_ASYNC_COPY=0/1` | ~10.0 | noise |
| `AMDGCN_USE_BUFFER_OPS=0` | 16.07 | catastrophic — buffer ops are load-bearing |
| `waves_per_eu: 2` | 50.6 | catastrophic |

### Structural things ruled out without writing code

* **Causal load imbalance** — not a problem here. The one-kernel design already
  pairs dk/dv work (∝ 32−pid) against dq work (∝ pid+1) so every program does a
  constant amount. No split-k / load-balance rewrite is available to win.
* **The two-kernel in-tree path** — the vendored file's own comment table records
  it losing badly at this shape on gfx1250. Not revisited.

## 3. Three harness bugs, all blocking, all fixed in the candidate

Neither is an optimisation; without them no candidate can be validated at all.

1. **No directory other than `op/baseline` could be validated.** `validation.py`
   loads candidate, beat and baseline into ONE process. The vendored `impl.py`
   imports `primus_turbo` at module level and leaves it in `sys.modules`;
   `op/baseline/impl.py` then raises *"an installed primus_turbo was imported
   before this module"*. Reproduced with the **unmodified `op/current`** as the
   candidate (`raw/` `imports` run) — this is a pre-existing framework bug, not
   something this round introduced, and it will hit every round.
   Fix, in the candidate's own `impl.py`: import the vendored tree once per
   directory under a stripped-and-restored `sys.modules`, asserting every module
   resolved inside this directory's `vendor/`.
2. **`Type '...Float8QuantConfig' is already registered as an opaque type.`**
   `primus_turbo.pytorch.core.low_precision` calls `register_opaque_type` at
   import and torch's registry is process-global and keyed by qualname, so
   whichever of the two `primus_turbo` trees is imported second dies. Fix: make
   the C-level registration idempotent (skip when
   `torch._C._is_opaque_type_registered(name)`), so both copies of the same class
   from two identical copies of the same file can register. The types are fp8
   quantisation configs; nothing in the bf16 attention path touches them.

   The first attempt at this fix *unregistered* the types our import had added,
   to leave the process as found. I suspected that of causing bug 3 below and
   replaced it; bug 3 then recurred anyway, so the unregistration was not the
   cause. Recorded because the replacement is still the better of the two -- not
   touching a registry the dispatcher reads beats putting it back.

3. **The dispatcher faulting on the first forward of the speed phase.** It wore three
   faces, all of them the signature of reading a freed C++ object:
   `schema_.has_value() INTERNAL ASSERT FAILED ... Tried to access the schema for .`
   (surfacing as `ValueError: vector::reserve` until `TORCH_SHOW_CPP_STACKTRACES=1`),
   `MemoryError: std::bad_alloc`, and bare SIGSEGV (rc 139). It was intermittent: it
   survived one full arm-A validation and two full arm-B validations, then took AB
   twice, then A. Correctness always passed; only the speed phase died.

   **Root cause, third theory and the one that held:** the vendored tree and the
   installed `primus_turbo` that `op/baseline` imports both declare
   `primus_turbo::attention_triton_forward_impl`. Two `torch.library` fragments
   defining one op name share a single `OperatorEntry`; when the second fragment is
   collected the schema is destroyed under the first tree's still-live `OpOverload`,
   and the next call through that stale handle reads freed memory. `validation.py`'s
   speed order is `cand > beat > baseline > baseline > beat > cand`, so baseline
   registers and dies between the candidate's first and last turn -- which is exactly
   where the fault always landed, and why it never touched the correctness phase, and
   why whether it fired at all depended on when the collector ran.

   The fix is one line in the vendored `attention_triton_impl.py`: register under
   `primus_turbo_opevolve::` instead. Only the registry key changes -- the op is
   reached through the Python symbol -- so no kernel and no timing is affected.

   Two earlier theories were wrong and are recorded as wrong, because both looked
   right for a while. (a) That fix 1's deletion of `primus_turbo` from `sys.modules`
   dropped the last reference to the defining module: so the modules were MOVED to a
   private `_op_evolve_<n>.` prefix instead of deleted, and it failed again. (b) That
   fix 2's opaque-type unregistration perturbed a registry the dispatcher reads: the
   unregistration was dropped, and it failed again. A probe
   (`rounds/001/scratch/schematest.py`) had already shown the op's kernel surviving
   load, `gc.collect()` and the whole correctness phase -- which was true, and which
   pointed away from the answer, because the destruction happens later and in
   baseline's process turn, not the candidate's.

   Cost of this bug: four wasted validation runs and two hours. The lesson worth
   carrying is that a segfault, a `bad_alloc` and an internal assert in the same
   place are ONE bug, not three, and that "it passed last time" is not evidence
   against a use-after-free.

Fixes 1 and 2 are in `impl.py`; fix 3 is the one-line namespace change in the
vendored `attention_triton_impl.py`. None of them touches a kernel. All of them are
pre-existing framework problems, not problems this round's ideas created: an
UNMODIFIED copy of `op/current` reproduces bug 1, so before this round
`validation.py` could not judge any directory except `op/baseline`.

### One more, found by reading the compiler rather than by measuring

`TRITON_HIP_USE_IN_THREAD_TRANSPOSE` is read at **compile** time
(`triton/backends/amd/compiler.py:268`, `add_in_thread_transpose`), defaults to
on only for gfx942, and **is cache-invalidating** (`get_cache_invalidating_env_vars()`
reports it once set). Setting it at module import — the obvious way — would also
change how `validation.py` compiles `beat`'s and `baseline`'s kernels in the same
process, corrupting both ratios in the candidate's favour. Arm B therefore sets and
restores it around the fused backward launch only.

## 4. Arms

Built from `op/current` each, B from the same base as A and not on top of it.

* **A — `r1.i1.g1`** `waves_per_eu: 1 -> 0` in the vendored config table.
* **B — `r1.i2.g2`** `TRITON_HIP_USE_IN_THREAD_TRANSPOSE=1`, scoped to the launch.
* **AB** — both.

## 5. Results

Every number below is `op/validation.py` on the arm directory, through
`job_context/op/runjob.sh` into the pinned container on `HIP_VISIBLE_DEVICES=1`.
Logs: `raw/valA.txt`, `raw/valB.txt`, `raw/valAB.txt`, `raw/valship.txt`. The
physical card was checked idle before the set (`rocm-smi --showuse` GPU[1] 0%,
`--showmeminfo vram` 174.9 MB); a second tenant was visible on GPU[0] at 100% and
280 GB throughout, which is why the pin matters.

| arm | idea | bwd ms | vs `beat` | vs `baseline` | spread% | correctness |
| --- | --- | --- | --- | --- | --- | --- |
| incumbent `op/baseline` (framework's own run) | -- | 10.0274 | 1.166x | -- | -- | pass |
| A | `r1.i1.g1` `waves_per_eu` 1 -> 0 | 9.5873 | 1.216x | 1.043x | 0.53 | pass, >= 52.2 dB |
| B | `r1.i2.g2` in-thread transpose, scoped | 9.4248 | 1.238x | 1.063x | 0.55 | pass, >= 52.2 dB |
| **AB** | **both** | **8.7026** | **1.340x** | **1.150x** | 0.36 | pass, >= 52.2 dB |
| shipped `rounds/001/op` (= AB, re-validated) | both | 8.6997 | 1.340x | 1.151x | 0.42 | pass, >= 52.2 dB |

Neither arm lost -- the floor spread on the backward is 0.4-0.8% and both are 4%+
clear of it -- so the merge was required, not optional. It is **superadditive**:
A alone saves 4.2%, B alone saves 5.9%, together they save 13.2% where a purely
additive pair would save 9.9%. The ad-hoc sweep that first suggested this predicted
8.696 ms; validation measured 8.7026 and 8.6997. That agreement across two different
harnesses is the reason to believe the number.

**The ITT scoping did what it was supposed to.** `beat` measured 11.6624 / 11.6668 /
11.6648 / 11.6543 ms across the A, B, AB and shipped runs -- four runs, 0.11% apart,
with and without the knob live in the same process. If the knob had leaked into
`beat`'s compilation the ratio would have moved without the kernel moving, and that
is precisely the kind of number that is a discrepancy with my name on it.

B was measured four times in total across the session (9.4128 / 9.4142 / 9.4147 /
9.4248) and A twice (9.5873 / 9.5936). Nothing here rests on a single run.

**Still short.** 1.340x against a 1.50x gate: 10.7% of the target remains. The
launch-configuration surface is now exhausted -- §2's sweeps show a sharp optimum
and `num_warps=8` is 2.36x *slower* for a mechanical reason (§6). What is left is
structural, and §6 says where.

## 6. What the corpus says, and why row 4 of the route changed because of it

Read after the measurements, against the static census from §1.

**The `num_warps=8` result was predictable and I should have predicted it.** gfx1250
has a flat per-lane VGPR budget of `131072 / NUM_THREADS`; at `num_warps=4` wave32
that is `131072/128 = 1024`, exactly the census figure. Doubling warps halves it to
512 against a kernel already spilling at 1024. The corpus's occupancy precondition
`current_vgpr x new_waves <= 1024` says no wave count above 1 is legal here. The
warp knob is not an escape from the spill; it is the fastest way to make it worse.
Also recorded: HipKittens measures 8 warps beating 4 by 1.332x on `gqa_d128` -- on
gfx950, where the 4-warp arm spilled and extra warps rescued it. That finding does
not port, and importing it is how one would have got this wrong twice.

**The structural gap, in one table.** See `findings/pool.md` `r1.i7.g7`: this kernel
runs LDS at 65,536 of 327,680 B (20%) while spilling 326 VGPRs / 1308 B per lane to
scratch; AITER's hd128 backward runs 163,840 of 163,840 (100%) with zero spill,
HipKittens' 140,288 of 160,000 with zero spill, FlyDSL's 116,736 with 20 B. It holds
in registers what all three of them hold in LDS, and 262 KB of LDS is idle. At
1 wave/SIMD -- which the VGPR budget forces -- there is no second wave to cover an
exposed scratch reload, so every spill is latency on the critical path. That is
consistent with 549 (now 632) TFLOP/s against a 2667 TFLOP/s healthy bf16 GEMM roof.

**Why "remove work" and not "overlap work".** gfx1250 is hard power-capped: measured
2497-2502 W against a 2500 W limit with the GFX clock held at ~1700 of 2400 MHz, 29%
below maximum, under sustained bf16 GEMM. Under a cap, deleting instructions pays and
rescheduling them does not -- the corpus prices barrier-splitting and interleaving
rungs at approximately zero while marking up the fetch-*removing* ones. This is also
a warning about how I read counters on this part: a change can save 10% of cycles and
0% of time with every counter showing a win.

**Two traps that bear directly on §1's failed instrument.** `rocprofv3` exits 0 and
writes no CSV when given an unknown counter -- a silent failure, which is consistent
with what §1 hit. On gfx1250 only 13 of 51 counters read non-zero, the entire WMMA
FLOP family reads exactly 0 (marked emulated), and there is no byte counter at any
level, so there is no counter-derived roofline on this part at all. Stochastic PC
sampling and ATT are reported to work; that is `r1.i6.g6` and it is the only route to
knowing where inside the kernel the time goes.

**One more, filed for the next session rather than acted on:** a degraded gfx1250
host reads as a slow kernel and not as an error -- the same binaries measured 19%
lower (2236 vs 2667 TFLOP/s) on a long-uptime host, only on the data-moving cases,
with nothing in the benchmark output indicating a problem. If a number moves and the
code did not, check `dmesg` for `SMU: No response` before believing it.

### explored.consulted

Every file opened under `knowledge/` this round. Nothing under `knowledge/` was
modified.

* `knowledge/INDEX.md`
* `knowledge/arch/gfx1250/gfx1250.md`
* `knowledge/arch/gfx1250/isa.md`
* `knowledge/arch/gfx1250/profiling-surface.md`
* `knowledge/optimization/techniques/0-register-pressure-and-occupancy.md`
* `knowledge/optimization/techniques/2-lds-bank-conflicts.md`
* `knowledge/optimization/techniques/3-pipelining-and-scheduling.md`
* `knowledge/optimization/routes/1-metrics-to-techniques.md`
* `knowledge/pitfalls/measurement-traps.md`
* `knowledge/pitfalls/compiler-and-toolchain.md`
* `knowledge/backends/README.md`
* `knowledge/backends/hip/README.md`
* `knowledge/backends/aiter/attention/recipes/fmha_v3_bwd_hd128_bf16.md`
* `knowledge/backends/flydsl/attention/recipes/hd128.md`
* `knowledge/backends/hipkittens/attention/recipes/gqa_d128.md`
* `knowledge/backends/hipkittens/gemm/recipes/bf16_gfx1250_ladder.md`
* `knowledge/backends/hipkittens/tiles-and-register-control.md`

`knowledge/backends/triton/` does not exist -- there is no Triton-specific backend
card in the corpus, which is why §3's in-thread-transpose finding had to come from
reading `triton/backends/amd/compiler.py` in the image instead.

## 7. What shipped

`rounds/001/op` is arm AB, re-validated in place at 8.6997 ms / 1.340x. Against
`op/current` it differs in exactly three files:

* `impl.py` -- the scoped `TRITON_HIP_USE_IN_THREAD_TRANSPOSE` context manager
  around the fused backward launch, plus harness fixes 1 and 2.
* `vendor/primus_turbo/triton/attention/fused_mha_bwd_kernel.py` -- `waves_per_eu`
  `1 -> 0`, one line.
* `vendor/primus_turbo/pytorch/kernels/attention/attention_triton_impl.py` -- harness
  fix 3, the op namespace, two strings.

It does not pass the 1.50x gate. Nothing in round 1 was going to: the round started
at 1.166x, the whole launch-configuration surface is worth 13.2%, and the remaining
10.7% has to come from the spill.

## 8. The graded measurement — cold cache, unit tests, back-to-back references

Everything in §5 was measured on a warm Triton cache. A backend serving a stale
kernel would have this round measure the previous round's binary and report it as
this round's, with no error anywhere, so the graded set was taken again from a
**cleared cache** (`rm -rf /root/.triton/cache`, 31 MB removed), in one session, on
the pinned device, with the incumbent re-measured beside the candidate rather than
read from a file. Raw: `raw/bench_final.txt`, `raw/bench_beat.txt`.

**Unit tests, first thing after the clear** (`op/ut/correctness.py` on the working
copy, which is also the round's first cold build): **PASSED**, 4 gated shapes, all
four tensors ≥ 50 dB — 52.04 to 54.09 dB. `sq_gt_skv` fails at dq −15.95 dB, which
is the pre-existing diagnostic defect documented in `op/ut/shapes.py` and
`PROVENANCE.md`, is not gated, and reads identically on the incumbent.

**Order**: `001 · 000 · 000 · 001`, twice, every shape, 30 iters, 3 s warmup. GPU[1]
verified idle before (0% use, 174.9 MB VRAM) and a second tenant sitting on GPU[0]
at 100% and 283 GB the whole time — which is the entire reason the runner pins
`HIP_VISIBLE_DEVICES=1`.

| shape | r001 bwd ms (4 runs) | r000 bwd ms (4 runs) | r001 TF/s | r000 TF/s | gain |
| --- | --- | --- | --- | --- | --- |
| **b4_s8192_hq32_hkv8_d128** (spec) | 8.6089 / 8.6821 / 8.6832 / 8.6972 | 9.9964 / 10.0039 / 10.0159 / 10.0176 | **633.2** | **549.2** | **1.1529** |
| sq_lt_skv | 0.7243 / 0.7264 / 0.7271 / 0.7287 | 0.8040 / 0.8047 / 0.8057 / 0.8084 | 354.7 | 320.1 | 1.1079 |
| min_seqlen | 0.1570 / 0.1583 / 0.1838 / 0.1872 | 0.1786 / 0.1789 / 0.1796 / 0.1803 | 126.5 | 120.0 | 1.0537 |
| ragged_tail | 0.2707 / 0.2709 / 0.2712 / 0.2728 | 0.3142 / 0.3154 / 0.3161 / 0.3188 | 302.5 | 259.7 | 1.1648 |
| sq_gt_skv (diagnostic) | 0.3418 / 0.3438 / 0.3439 / 0.3449 | 0.3957 / 0.3967 / 0.3972 / 0.3972 | 250.1 | 216.6 | 1.1544 |

Only the spec shape is scored — `op.shape.mode: single`, and `state.yaml`'s round-0
row carries that one figure. The other four are the correctness set; they are listed
because a win on the scored shape paid for out of the edge shapes would be visible
here, and it is not: every shape improves, by 5.4% to 16.5%.

Two checks on the measurement itself. **The re-measured incumbent lands on its stored
figure**: 549.2 TF/s now against 548.9 recorded at setup, 0.05% apart, which is the
best evidence available that the session is comparable to the one that set the
baseline. And **the cold cache changed nothing**: 8.6089-8.6972 ms here against
8.6997-8.7026 warm in §5, so no stale binary was involved either way.

**The target, and why it had to be measured too.** `op.target.tflops` is null on
purpose; the target is relative — `op/beat/` "MEASURED IN THE SAME RUN, must be
beaten by 50%". So `beat` was measured in the same session and on the same device,
palindromically against the candidate (`beat · 001 · beat · 001`, `raw/bench_beat.txt`):

| | bwd ms | bwd TF/s |
| --- | --- | --- |
| `beat` (flex_attention) | 11.6720 / 11.6887 | **470.7** |
| target = 1.50x beat | 7.781 | **706.0** |
| this round | 8.6792 / 8.6857 | 633.2 |

`score` = 633.2 / 706.0 = **0.8968**. Gain over the incumbent = **1.1529x**. The gate
still fails, by 10.3% of the target.

**Bound: power**, medium confidence. Not measured directly this round -- it is read
off the census plus the corpus. The kernel spills 326 VGPRs / 1308 B per lane to
scratch at 1 wave/SIMD, where nothing covers a reload, on a part measured pinned at
2497-2502 W of a 2500 W cap with clocks held 29% below maximum. Under a cap what pays
is deleting work, and both of this round's wins delete work rather than overlap it:
in-thread transpose removes an LDS round trip per transposing dot, and
`waves_per_eu: 0` removes an occupancy constraint the compiler was paying registers
to honour. The reason confidence is not high is that this part has no byte counter at
any level and its whole WMMA FLOP counter family reads zero, so "power" here is an
inference from the static resource report and the clock/wattage pair, not a reading.
