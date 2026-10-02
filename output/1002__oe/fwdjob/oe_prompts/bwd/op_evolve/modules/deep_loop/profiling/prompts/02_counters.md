{{preamble}}

## Step 2: counters

Read [`{{knowledge_dir}}/profiling/0-kernel-profiling.md`](../../../../knowledge/profiling/0-kernel-profiling.md).
That file only.

Collect per-kernel counters with `rocprofv3` for each kernel in `{{out_dir}}/kernel.yaml`,
one `-k`-equivalent filter at a time. Results go to
`{{out_dir}}/2-kernel-profiling/kernel-<slug>/`.

What that document settles and you should not re-derive: which block a counter comes from,
how many instances it sums over (`GRBM_*` per XCD, `SQ_*_CU_CYCLES` per CU), and whether a
counter reports bytes or requests. Apply the normalisation it gives; an un-normalised counter
is off by the instance count and looks plausible.

**Its counter NAMES are gfx942/gfx950 and mostly do not exist here. Do not copy a list out of
it.** The surface that was swept on this card is
[`{{knowledge_dir}}/arch/gfx1250/profiling-surface.md`](../../../../knowledge/arch/gfx1250/profiling-surface.md);
this step's list comes from there and from nowhere else.

**Validated on this card** -- non-zero against a kernel provably doing the work:

```
SQ_WAVES   SQ_CYCLES   SQ_BUSY_CYCLES   SQ_ITEMS   GRBM_GUI_ACTIVE
SQC_ICACHE_REQ   SQC_ICACHE_MISSES   SQC_ICACHE_MISSES_DUPLICATE
```

That is wall-clock cycles, wave count and instruction-cache behaviour. Nothing about memory,
the matrix unit, occupancy or stalls.

⚠ **`SQC_ICACHE_HITS` is accepted and returns zero** in the same run where `REQ` and `MISSES`
are both sensible (`profiling-surface.md:105`). If you collect it, a zero from it is **not a
measurement** and must be labelled as such wherever it appears.

**Rejected names fail the WHOLE pass, silently.** A `--pmc` list naming one counter gfx1250
does not define writes **no** `*_counter_collection.csv` and still exits 0 -- indistinguishable
from a clean run. Every wait/stall counter is rejected, and so is anything naming MFMA:
**gfx1250 has WMMA, not MFMA, so `SQ_VALU_MFMA_BUSY_CYCLES` does not exist here** and neither
do `SQ_WAIT_ANY`, `SQ_ACTIVE_INST_ANY` or `SQ_BUSY_CU_CYCLES`.

**Accepted-but-silently-zero -- do not put these in a list.** The whole
`SQ_VALU_WMMA_FLOP_*` family, `SQ_INSTS_VEC32_VALU_WMMA`, `SQ_INST_CYCLES_VALU_WMMA`, and
every memory counter (`CHC_*`, `GL1C_*`, `CHA_*`, `GLARBC_*`). Measured: a kernel executing
exactly 4,096,000 `v_wmma_f32_16x16x32_bf16` instructions reports all three WMMA counters as
exactly 0 while `SQ_WAVES` reports 4096, which is exact. If one of these reaches a CSV anyway,
write beside every zero that **a zero from this counter is not a measurement**.

**There is no byte-reporting counter at any level on this part.** No traffic figure, no
measured arithmetic intensity, no cache hit rate comes from counters here. Say so rather than
substituting something that looks like one.

**Assert the output.** After every pass, check the CSV exists and that `Counter_Name` contains
each counter you asked for. Exit status is not a signal.

**Split the counter list.** A `--pmc` list needing more than one hardware pass aborts with
`signal 6` or leaves dangling correlation ids, and writes an `agent_info.csv` with no
counters -- which looks identical to "the counter does not exist". Collect in several passes
rather than diagnosing this each time.

⚠ **DO NOT RUN PC SAMPLING. MEASURED 2026-09-24, round 17.**

An earlier revision of this file recommended stochastic PC sampling here as "the only rich
source of *why* on this card". **That recommendation is withdrawn — it was written from the
docs, not from a measurement, and the measurement went the other way.**

Round 17's own profiling step ran it at the production shape and it faulted:

```
Memory access fault by GPU node-2 on address 0x7ed0642e4000.
    Reason: Page not present or supervisor privilege.
GPU core dump skipped because PC Sampling active
... rocprofv3 caught signal 6 ...
Timeout while waiting for queue sync: 4 kernels still active
```

with a 131-line `no-retry page fault` burst in `dmesg`
(`PERMISSION_FAULTS: 0x5`, `RW: 0x1`, `MORE_FAULTS: 0x1`, client TCP). The card survived
that time. It did not on **2026-09-11**, where PC sampling put MES into an unrecoverable
state and cost a physical power cycle — the only profiler-attributable wedge this campaign
has.

Note the two aggravating details, both visible above:
- **`GPU core dump skipped because PC Sampling active`** — the failure destroys its own
  post-mortem. You cannot diagnose what it broke.
- **`Timeout while waiting for queue sync: 4 kernels still active`** — that is the queue
  path that becomes `failed to suspend all gangs` and then `unrecoverable`. It came close.

2026-09-30, after the 09-29 reflash: rocprofv3 1.3.2 now **rejects every PC-sampling
configuration at startup** ("not supported on any of the agents") although `-L` lists one. That is
not a licence to retry it: the ban stands.

**So: counters cannot answer the stall question here** (every wait and stall counter is rejected
by this chip) **and PC sampling must not be attempted. The instruction trace (step 4) answers it**:
since the 09-29 reflash ATT gives per-instruction stall and idle cycles for this op's FlyDSL kernels
(verified 2026-09-30). In `analysis.md`, state the counter limitation and point at step 4. Do not
substitute a static instruction count for it -- this job has measured static metrics improving on
arms that lost, again and again (h26, h78).

**Cycles.** `GRBM_GUI_ACTIVE / 8` per kernel is this card's cycle ruler (8 XCDs). Run the profiled
benchmark with `--warmup-seconds 0` under `--pmc`: the unsynchronised warmup floods the profiler and
looks like a hang.


Write `analysis.md` with the counters, their normalised values, and what each says -- and,
for anything you collected from the silently-zero set, the explicit note that its zero is not
a measurement. No verdict yet -- the regime comes in step 5 and the bound in step 6.

## Reply with

```yaml
kernels_done: [<slug>, ...]
passes: 3
counters_absent: []        # names asked for that produced no column, if any
highlights: [{metric: ..., value: ..., note: ...}]   # at most 6 lines
pc_sampling: {status: not_attempted, reason: forbidden_faults_this_card}   # see above
status: ok
```
