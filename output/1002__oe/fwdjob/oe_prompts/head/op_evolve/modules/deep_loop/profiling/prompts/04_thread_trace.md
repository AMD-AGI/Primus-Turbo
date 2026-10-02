{{preamble}}

## Step 4: instruction trace

Read [`{{knowledge_dir}}/profiling/4-thread-trace.md`](../../../../knowledge/profiling/4-thread-trace.md).
That file only. Output to `{{out_dir}}/4-thread-trace/`.

A thread trace answers a different question from everything above: not *which* unit is busy
but *what each instruction did on each cycle*, for one compute unit. It costs minutes and
hundreds of megabytes per dispatch.

**ATT serialises dispatches. Its timings must NEVER be used for ranking**, for a share of
time, or for any comparison against `benchmark-results.md`. It answers *what the instructions
did*, not *how long the kernel takes*. The ranking is step 1's and stays step 1's.

**The decoder library is already on disk. Pin it and install nothing.**

```
att_library_path: /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib
```

That directory holds `librocprof-trace-decoder.so` -> `.so.0.2`. **You may not pip install,
download, or otherwise add anything to this container** -- it is shared and the job spec sets
`owned: false`. If the pinned path does not resolve, this step fails: write
`status: failed` with the path you tried and stop.

**Smallest safe capture, and no larger.** One kernel, one iteration, one CU:

```yaml
jobs:
  - kernel_include_regex: "<the exact kernel name from kernel.yaml>"
    kernel_iteration_range: "[3]"        # one dispatch, past warm-up
    advanced_thread_trace: true
    att_library_path: /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib
    att_target_cu: 1
    att_shader_engine_mask: "0x1"
    att_simd_select: "0xf"
    att_buffer_size: "0x2000000"         # 32 MB. Raise ONLY if the trace truncates,
                                         # one doubling at a time.
    output_directory: <scratch>
    output_file: att
```

Start at that buffer size rather than the corpus's 384 MB: this card has one GPU and a
capture that overruns is paid for by the whole job. A truncated trace is a re-run; a wedged
card is a human power-cycle.

**PMC and ATT cannot share a job.** Never add `--pmc` to this run, and do not use
`--att-perfcounters` / `--att-perfcounter-ctrl` / `--att-activity`: all three are accepted on
this stack and emit no counter file, silently.

**Narrow the capture from the start**, with an input file using `kernel_include_regex` and
`kernel_iteration_range`. Unfiltered, one HIP program produced 374 MB and two directories,
one of them the runtime's memset; filtered, the same GEMM produced one directory of 21 MB.

**Confirm which directory is yours.** One directory is one dispatch, and the number in its
name is the application's dispatch id -- not an index into your kernels. Gaps are normal.
Use the name from `kernel.yaml`; if the kernel came from a runtime-loaded code object it has
no name in the trace, and `tools/att_views.py --kernel-name` is how you supply it.

`tools/att_views.py` imports `matplotlib`. As of 2026-09-24 it IS available under
`PYTHONPATH=/home/lihuzhan/.local/pyviz` (measured: Agg renders a valid PNG). It is **not in the container's own site-packages and
must not be installed into it**. Render under that PYTHONPATH if you want charts; otherwise parse the
`stats_*.csv` the decoder writes and report the numbers in `analysis.md` instead; record
`charts: []` rather than a path to a chart that was never drawn. Three limits to state
explicitly in `analysis.md`:

- **These timings are serialised by the tracer** and are not comparable to anything in
  `benchmark-results.md`, nor usable to rank kernels.

- **This is one compute unit**, selected by `--att-target-cu`, out of hundreds. Every
  per-wave view describes that one.
- **The overlap view is per-wave.** Latency is also hidden between waves, which it does not
  show: a kernel reading 0% there can still run at full speed if occupancy covers it. Read it
  against the wave-states view.

The trace's `ui_output_*` stays in `{{scratch_dir}}` and is discarded. Keep the charts, the
`stats_*.csv` you parsed, and `analysis.md`.

**If the decoder library is missing, do NOT install it.** The library present on this
machine is
`/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/librocprof-trace-decoder.so`
(0.2.x); a second copy sits under `_rocm_sdk_core/lib`. Pointing `--att-library-path` at that
directory is the only sanctioned way to satisfy the decoder. Downloading a release, unpacking
one, or writing anything into `/opt/venv` is forbidden: the container is shared and the job
spec sets `owned: false`. If the pinned path does not resolve, or the decoder reports
`INVALID_SHADER_DATA` (version skew against rocprofv3 1.3.2), record `status: failed` with the
message and stop. "The library is not where I expected" is not the same finding as "ATT is
unavailable here", and neither is a licence to install.

## Reply with

```yaml
kernels_done: [<slug>, ...]
dispatch_dirs: [{kernel: <slug>, dir: ui_output_..._dispatch_6, size_mb: 21}]
findings: [{observation: ..., view: overlap|wave_states|hotspot|...}]
charts: []                     # set only if you actually rendered one under the PYTHONPATH above
decoder_path: /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib
installed_anything: false
status: ok|failed
```

---

## MEASURED ON THIS MACHINE, 2026-09-24 -- read this before spending a step here

ATT was probed directly on this card while it was otherwise idle. Four runs, and the card
was **completely unharmed every time**: zero `MES ... failed to respond`, zero
`suspend all gangs`, zero `unrecoverable`, **zero page faults**, `docker exec` responsive
throughout. **ATT does not wedge this card.** The standing ban was inherited from PC
sampling, which is a different mechanism -- a per-wave interrupt that writes wave state to
a save area -- and it does not apply here.

But ATT is **not usable on this job's kernels**, and that is the operative fact:

| workload | result |
|---|---|
| a trivial `torch` elementwise kernel | **34 files, 11 MB, a real `.att` trace plus `ui_output_*_dispatch_*/`** |
| this op's FlyDSL kernels (`profdrv.py`), with `--kernel-include-regex` | 9 files, **every one a `*_code_object_id_*.out` ELF dump**, no trace |
| the same, **without** any regex filter | identical -- 9 files, no trace |

So the filter is not the problem; **ATT captures nothing for FlyDSL JIT-compiled kernels
on this stack**, while capturing a torch kernel in the same container minutes apart. This
also explains the 2026-09-11 records, whose output was likewise ELF-dumps-only: those runs
were not "unarmed", they hit this same limitation.

**Therefore:** attempting ATT on this op costs a step and returns nothing. Record
`status: skipped`, `reason: att_captures_no_flydsl_kernels`, and cite this block. If a
future round changes how kernels are loaded (away from runtime `hipModuleLoad` of a JIT'd
module), re-probe with the torch control alongside it.

Two traps found while probing, worth keeping:
- **`--att-buffer-size` is in BYTES, not MB.** `--att-buffer-size 64` aborts with
  `F core.cpp:108] Invalid buffer size: 64`, and rocprofv3 then sits in its own SIGABRT
  handler until killed -- which reads exactly like a hung GPU and is not one. Use
  `67108864` for 64 MiB.
- Always verify a capture **armed** before quoting it as a pass: a directory containing
  only `*_code_object_id_*.out` is a NULL result, not a clean one.
