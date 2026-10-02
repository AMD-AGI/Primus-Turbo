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

**Smallest safe capture, and no larger: one call of the op, one CU, only the kernels you want.**
This exact form was verified on this card on 2026-09-30 (after the 09-29 reflash) for FlyDSL JIT
kernels (the backward job's) and for the ASM `.co`, with a clean `dmesg` every time. Run it in its own process, with a fresh
FlyDSL cache directory, through the job's own benchmark so the kernels see their production inputs:

```bash
FLYDSL_RUNTIME_CACHE_DIR=$(mktemp -d /tmp/flycache_att.XXXX) \
rocprofv3 --att \
  --att-library-path /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib \
  --att-target-cu 1 \
  --kernel-include-regex "<regex from kernel.yaml, QUOTED>" \
  --output-format csv --output-file att -d {{scratch_dir}}/att_<slug> \
  -- /opt/venv/bin/python3 {{job_context_dir}}/op/benchmark.py --arm-path cur={{profile_target}} \
     --shapes prod --iters 1 --warmup-seconds 0 --block 1 --lead 0
```

(run from `{{job_context_dir}}/op`; the container env must carry `ARCH=gfx1250` and
`FLYDSL_GPU_ARCH=gfx1250` as for every run here). For this op `"fmha_fwd_prefill"` captures our prod
kernel `kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0` (one dispatch directory per call). For `op/beat` the
ASM kernel is `aiter::fmha_bf16_pertokenBf16_hd128_128x256_mask` (`"fmha_bf16"`); it comes from a
runtime-loaded `.co`, so pass that name to `tools/att_views.py --kernel-name`. **Quote the regex.** `--iters 1 --warmup-seconds 0 --block 1 --lead 0` keeps the run to one build call plus one
timed call per kernel; a warmup window under the tracer only multiplies dispatches. Start with no
buffer-size flag; if the trace truncates, pass `--att-buffer-size` **in bytes** (e.g. `67108864` for
64 MiB), one doubling at a time.

What you get per captured dispatch: `stats_ui_output_agent_<pid>_dispatch_<N>.csv` (one row per
instruction: `Hitcount`, `Latency`, `Stall`, `Idle`, `Vaddr`, `Instruction`) and a
`ui_output_agent_<pid>_dispatch_<N>/` directory with per-wave timelines (`se*_wv*.json`). A capture that
produced only `*_code_object_id_*.out` files **armed nothing**: that is a NULL result, not a clean one --
record it as failed with the file listing.

Summarise each CSV with the operator's parser (read-only, stdlib only):
`python3 /home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/tools/attsum.py <csv> 30`
-- per-class hit/latency/stall/idle, latency per WMMA, and the top stall sites with their `Vaddr`.
Then map the top sites back to the ISA (`*_final_isa.s` from a compile-only dump of the same tree) and
to source lines. Two reading rules learned on this card (backward job, 2026-09-30):
- **The `Latency` column charges every WMMA 8 cycles** even when the next instruction issues a cycle
  later, so class shares from it overstate WMMA. For a per-iteration budget use wall time: issue-time
  differences between successive loop heads in the wave timelines.
- **`stall = 7` on back-to-back WMMAs is not a dependency** -- it is one WMMA per 8 cycles.

Comparable reads on disk: **none for this op's FlyDSL kernels on the new firmware** -- this capture is
the first. The beat's ATT (`{{job_context_dir}}/profiling/beat/4-thread-trace/`, A0 2026-09-27, OLD
firmware) has the ASM kernel's instruction mix; its stall and clock picture predates the reflash. The
champion's cycle census and light-speed ablations (PMC, not ATT: skeleton 28%, softmax 19%, per-tile
barrier 5%) are in `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__roofline/REPORT.md` §6 and §9.
What a useful read looks like (same recipe, the backward job):
`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/notes/report_{dkdv,dqg,asm}.md`.

**PMC and ATT cannot share a job.** Never add `--pmc` to this run, and do not use
`--att-perfcounters` / `--att-perfcounter-ctrl` / `--att-activity`: all three are accepted on
this stack and emit no counter file, silently.

**Narrow the capture from the start** with `--kernel-include-regex`. Unfiltered, one HIP program
produced 374 MB and two directories, one of them the runtime's memset; filtered, one directory of 21 MB.

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

## MEASURED ON THIS MACHINE -- ATT is the primary diagnostic here

**2026-09-24 (old firmware):** ATT captured nothing for FlyDSL JIT kernels (ELF dumps only) while it did
capture a torch kernel; this job's rounds 10, 15 and 20 skipped this step on that finding. **That finding
is superseded.** **2026-09-30, after the 09-29 reflash (VBIOS 700E):** the recipe above captures FlyDSL JIT
kernels (the backward job's `k_dkdv` / `k_dqg`) and the ASM `.co` with full per-instruction statistics,
every run rc 0 with no GPU fault line in `dmesg`
(`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/output/0930__bwd/probe/P1-RESULTS.md`). The backward's
hand campaign that day went from 83.5% to 103.8% of its ASM bar, and every accepted lever was found or
confirmed from an ATT read (a `s_wait_loadcnt` stall pointing at operand latency, a `s_wait_dscnt` behind
a readback burst, VALU and `v_nop` counts between WMMAs). This op's kernels load the same way
(runtime-loaded FlyDSL JIT code objects) but have not been captured on this firmware yet. So: **run this
step; do not skip it**, and check that it armed. A `status: skipped` here needs a measured reason from
this round.

ATT does not wedge this card (probed 2026-09-24 and 2026-09-30). **PC sampling is a different mechanism
and stays forbidden** (step 2); do not substitute it for, or add it to, this step.

Traps worth keeping:
- **`--att-buffer-size` is in BYTES, not MB.** `--att-buffer-size 64` aborts with
  `F core.cpp:108] Invalid buffer size: 64`, and rocprofv3 then sits in its own SIGABRT handler until
  killed -- which reads exactly like a hung GPU and is not one.
- Always verify a capture **armed** before quoting it: a directory containing only
  `*_code_object_id_*.out` is a NULL result.
- `--kernel-trace` still records zero dispatches on rocprofv3 1.3.2 -- take kernel names from step 1's
  `--pmc` csv, not from a kernel trace.
- One process on the card at a time: check `ls /sys/class/kfd/kfd/proc` is empty before the capture and
  that your process is gone after it.
