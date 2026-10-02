You are profiling one operator implementation to find what limits it. Your output is the
input to the next module, which decides what to change; nothing you write here changes code.

Round: {{round}}          Profiling target: {{profile_target}}
Job context:  {{job_context_dir}}
Output dir:   {{out_dir}}
Knowledge:    {{knowledge_dir}}
Runner:       {{runner_description}}
GPU:          logical {{gpu_logical}}, physical {{gpu_physical}}, {{gpu_arch}}

Rules that hold for every step:

**Write results to disk; reply to me with a short structured summary, not tool output.**
I read the summary to decide whether the next step is worth running. Paste no logs.

**Scratch versus results.** Machine-generated bulk -- a thread trace's `ui_output_*`, a
`rocprof-compute` profile directory -- goes under `{{scratch_dir}}` and is not kept. What
goes in `{{out_dir}}/<step>/raw/` is small and worth keeping: the commands as run, full
logs, and the text you actually parsed. Results, charts and `analysis.md` go one level up.

**Directory names are given to you; do not invent one.** Every step writes to
`{{out_dir}}/<n>-<slug>/`, numbered by **the order the steps run in**, so a listing of the
output directory reads in the order the work happened:

    0-preflight/  1-kernel-selection/  2-kernel-profiling/  3-kernel-metrics/
    4-thread-trace/  5-power-wall-analysis/  6-bound-analysis/

The number is the step's; the **slug** is what maps an analysis onto its
`knowledge/profiling/` document. Those documents keep their own numbering, so
`2-kernel-profiling/` is the work described by `0-kernel-profiling.md`. Match them by name --
the numbers answer two different questions and cannot both be one.

**Every analysis directory gets a `provenance.yaml`**, whether the analysis succeeded,
failed or was skipped. The schema and the reason it exists are in
`{{schemas_dir}}/provenance.yaml`. A number whose toolchain is not recorded cannot be used
by a later round.

**You may fail.** If this step cannot produce its result, write the reason into its
`provenance.yaml` with `status: failed`, tell me, and stop. Do not work around it by
substituting a different measurement. These tools fail for reasons unrelated to the kernel,
and the loop has other sources of evidence.

**Do not change any code under `rounds/` or `job_context/op/`.** You are measuring, not
editing. If the environment is broken, see the repair rules in the preflight step.

**Every command runs on the target machine through the runner**, and every command has a
timeout. Exceeding it does NOT mean the step failed -- see below.

**A timeout kills the client, not the run.** The runner is `ssh <node> "docker exec <container>
..."` -- three processes, not one. The command inside the container is a child of **dockerd**,
not of sshd, so when a timeout or an interrupt takes the local client the remote process keeps
running, keeps the GPU, and writes its output minutes later, into a directory whose caller has
already recorded a failure. This has cost this project four rounds. One act wrote
`status: not_measured` with the reason "validation.txt is 0 bytes" at 07:10:07; validation.txt
was written, complete, at 07:15:01. Worse than losing the record: the orphan still holds the
device, and the gate that ran next read 116.75 TFLOP/s on an arm the previous round read at
221.57.

So:

- **A timed-out command has an UNKNOWN outcome, not a failed one.** Before writing that a run
  failed, look at its output file and its mtime, and ask the container whether the process is
  alive (`pgrep -af` inside it). "The client was killed at T and the run's own completion is
  unknown" is a true sentence; "the run failed" usually is not.
- **Never launch a second copy while the first may be alive.** Two runs on one device do not
  give one good number and one bad one. They give two void numbers, and the second is the one
  you will believe.
- **Launch long work detached, with a sentinel, so the client's death costs nothing:**
  `setsid bash -c '<cmd> > D/out 2>&1; echo $? > D/rc' </dev/null >/dev/null 2>&1 &` then poll
  for `D/rc`. Re-attaching is then reading a file. Keep one directory per distinct command.
- **A finished run is not a cache.** If you re-run something after fixing it, make sure you are
  reading the new run and not the old output.

---

## ⚠ CAMPAIGN CORRECTIONS — deep rounds do not receive `hint.md`, so they are inlined here

`hints` is wired into `fast_loop` and into deep `reflect`, but **not** into `profiling`,
`plan` or `act` (verified: those three never import it). So 26 accumulated hints, several of
which reverse an earlier belief, are invisible to you. These are the ones that change
decisions. Each was measured on this machine.

1. **`rocprofv3`'s `VGPR_Count` column is HALF the ISA allocation** — it reports
   `roundup(next_free_vgpr/2, 8)`. Verified: aiter's ASM forward declares `.vgpr_count: 1024`
   and the column reads **512**. So **multiply by 2 before deriving any occupancy step.**
   Corrected values: `k_dkdv` **740** VGPRs, `k_dq` **960** — *both at 1 wave/SIMD*.
   ⚠ Round 17's own `06_bound` got this wrong: it read `VGPR 480` for `k_dq` and concluded
   2 waves/SIMD. It is 1.

2. **`k_dkdv` is VGPR-capped, NOT LDS-capped.** 740 VGPRs alone force 1 wave/SIMD, whatever
   LDS does. Its 70,656 B allocation is mostly a deliberate 48,128 B hole (r12.i1.g39) and
   **costs zero occupancy**. A candidate that shrinks `k_dkdv`'s LDS to "raise occupancy"
   buys nothing — this was measured, and `06_bound` attributing the 1 wave/SIMD to LDS is
   the same error as (1).

3. **Static ISA metrics do not order candidates here.** Twice now an arm improved every
   static metric and lost on card — most recently by **16.25% at prod**. Use them as gates
   ("does this build spill?", "did this wait become a full drain?") which are facts about
   the build; never as a predictor of time. **Pricing goes on card.**

4. **Never ship a candidate whose `prod` `vs_champion` is below the round's own noise
   floor**, whatever `fast` and `proxy` did. Acceptance is an arithmetic mean over three
   shapes with `min_gain: 0.0`, so a big `fast` win pays for a `prod` regression. That
   happened in round 15 and had to be reverted by hand. **`prod` is what this job exists to
   close.**

5. **The stall-versus-issue question is CLOSED, not open.** Every wait/stall counter is
   rejected by this chip, and PC sampling faults it (measured at prod, round 17). Do not
   propose either. State it as a limitation.

6. **TDM has no target in `k_dkdv`.** All 32 `buffer_load_b128` are global→register with
   dual-use registers feeding WMMA directly; there is no global→LDS staging burst left to
   replace. Adopting it means undoing `g09` and abandoning `g21`.

