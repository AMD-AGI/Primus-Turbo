{{preamble}}

## Step 1: pin what will be profiled

Every later step refers to what you write here and rediscovers nothing. Fill in
`{{out_dir}}/kernel.yaml`, following `{{schemas_dir}}/kernel.yaml`. The discovery run's own
output -- the commands and the counter CSV you read -- goes in
`{{out_dir}}/1-kernel-selection/raw/`, with a `provenance.yaml` beside it.

Read [`{{knowledge_dir}}/profiling/0-kernel-profiling.md`](../../../../knowledge/profiling/0-kernel-profiling.md),
the section "Attributing a measurement to the right kernel". Nothing else from the corpus.

### Which kernels

Discover what actually ran.

**`--kernel-trace` MUST NOT be used on this stack, and `--stats` goes with it.** On this
card (gfx1250, rocprofv3 1.3.2) a `--kernel-trace` pass records **zero dispatches**: it
exits 0 and writes a stats table with nothing in it, which reads exactly like "the op never
ran". You cannot rank from it, and believing its emptiness is worse than not running it.

Rank from a `--pmc` pass instead. Counter collection works on this card, and every `--pmc`
row carries the dispatch's descriptors and timestamps for free -- which is all the ranking
needs. One pass, five counters, one hardware pass:

```bash
rocprofv3 --pmc SQ_WAVES,SQ_CYCLES,SQ_BUSY_CYCLES,SQ_ITEMS,GRBM_GUI_ACTIVE \
  --output-format csv --output-file discover -d $SCRATCH -- <benchmark>
```

The four `SQC_ICACHE_*` counters are validated here too, but say nothing about time; leave
them to step 2 rather than risking a second hardware pass in the step everything depends on.

**Then assert the CSV exists and contains the counters you asked for. Exit status is not a
signal on this part**: a `--pmc` list naming one counter gfx1250 does not define writes no
CSV at all and still exits 0, which looks identical to a clean run. Glob for the file --
`<dir>/discover_counter_collection.csv` in some builds,
`<dir>/pass_1/discover_counter_collection.csv` in others.

It is long-format, one row per counter per dispatch, with columns

```text
"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id","Grid_Size",
"Kernel_Id","Kernel_Name","Workgroup_Size","LDS_Block_Size","Scratch_Size","VGPR_Count",
"Accum_VGPR_Count","SGPR_Count","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"
```

Compute, per `Kernel_Name`, filtering to the rows of **one** `Counter_Name` first so each
dispatch is counted once:

- **duration** = sum of `End_Timestamp - Start_Timestamp` over its dispatches, in ns.
- **share of time** = that sum over the same sum across every kernel in the profile. This is
  what `pct_of_time` means below.
- **a second, independent ranking** from the per-dispatch `GRBM_GUI_ACTIVE` sums. If the two
  orderings disagree, rank by the timestamps and say in `kernel.yaml` that they disagreed.
- **dispatch count**, and the descriptors every `--pmc` row carries anyway: `Grid_Size`,
  `Workgroup_Size`, `VGPR_Count`, `SGPR_Count`, `LDS_Block_Size`, `Scratch_Size`. Record them
  in `kernel.yaml`. A non-zero `Scratch_Size` is a register spill and is a finding in its own
  right, not bookkeeping.

These durations are taken under the profiler, where dispatches are serialised and the clock
differs. **They rank kernels; they are never the round's score.** That stays
`benchmark-results.md`.

Select **the kernels implementing the op**, not every dispatch in the process. A PyTorch
driver also launches `distribution_elementwise_grid_stride_kernel` for `randn` and
`__amd_rocclr_fillBufferAligned` for allocation -- together under 8% of the time and never
the answer. The name tells you who emitted it; the table in that document maps name shape to
producer. If you cannot tell them apart, fall back to a 10% share threshold and say in
`kernel.yaml` that you did.

**Then cut that list to the bottleneck: take kernels in descending share until the
cumulative reaches 80%, and carry at most three.** Every later step runs once per kernel, so
a long list is paid for in wall clock and in the plan's attention, and the round is here to
optimise what dominates. A round that carries seven kernels spends its counters step on a
tail it will not act on.

Rank by share alone, **never by `ours`**. The second-largest block in one backward measured
here was PyTorch's `AccumulateGrad`, at 23.4% across three dispatches -- larger than two of
the op's own kernels. A rule that kept only the op's kernels would have hidden it, and a
bottleneck this round cannot edit is still one it must know about before it plans around it.

Carry a fourth only if you can say what it buys, and say that in `kernel.yaml`. Cutting a
kernel here does not delete it: it goes in `excluded` with its share and
`reason: below_cumulative_cut`, so the next round sees what was set aside rather than never
looked at.

List what you excluded and why. "Why is this kernel not in the analysis" is a question the
next round will ask, and silence reads as an oversight.

Two things to get right:

**The `-k` index is not an identity.** It is assigned within one profile, ordered by time
share, so the same number means different kernels in different profiles. Record it beside its
profile id. Directories are named by the kernel's slug, never by index -- otherwise round 3's
`kernel-0` and round 7's silently compare two different kernels.

**A kernel from a runtime-loaded code object may have no name in a thread trace.** If a later
step needs a name the trace does not carry, it comes from the `Kernel_Name` column of the
`--pmc` pass you just ran. Set `name_available_in_trace` accordingly.

### Which shapes

`benchmark.py` measured every shape; profiling analyses **two**: the one furthest from its
target and the one closest.

**Rank by distance from each shape's own target, not by absolute TFLOPS.** Throughput is not
comparable across shapes. Measured on this part: 8192 cubed reaches 1326 TFLOP/s while moving
2.33x its compulsory HBM traffic, and 8192x8192x128 reaches 390 TFLOP/s at 1.02x. The second
is the slower number and the one with nothing left to win, so a TFLOPS ranking picks it every
round and hides the shape that has 2.33x of traffic to remove.

Every later verdict states which shape it is about.

## Reply with

```yaml
kernels: [{slug: ..., pct_of_time: ..., producer: ..., ours: true}]
excluded: [{name: ..., pct_of_time: ..., reason: ...}]
shapes: [{name: ..., reason: furthest_from_target|closest_to_target, ratio: ...}]
cumulative_pct_carried: 80.2      # what `kernels` sums to, so the cut is auditable
fallback_threshold_used: false
```
