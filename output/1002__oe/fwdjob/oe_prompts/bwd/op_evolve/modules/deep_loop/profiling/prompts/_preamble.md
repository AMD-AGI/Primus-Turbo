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

## ⚠ CAMPAIGN CORRECTIONS (bwd job, A0, rewritten 2026-09-30) — inlined because this step does not read `hint.md`

`hints` is wired into `fast_loop`, deep `reflect` and `route.md`'s operator tables, but **not** into the
`profiling`, `plan` or `act` prompts. The job's operator file is `{{job_context_dir}}/hint.md`; **read h74-h82
there before you start** (h54 is the older one-page index; h76 is the current champion's map). What changes
decisions, all measured on this card (A0, one gfx1250, container `fa-repro`, reflashed 2026-09-29):

1. **The champion is s6** (adopted by refactor h75): prod 5.30 ms = **1.038x the ASM bar**; proxy ~1.13x. The
   target is now **1.20x beat on proxy AND prod** (fast is reported only). Structure: `k_delta` -> `k_dkdv` (main
   stream) concurrent with `k_dqg` (side stream); both kernels take their big operands through a **3-stage TDM
   LDS ring whose next-iteration B operands are read back into carried VGPRs** (k_dkdv: Q/dO; k_dqg: K/V).
   Any "TDM has no target in k_dkdv" or "k_dkdv 740 / k_dq 960 VGPR" statement in older findings is obsolete.
2. **`rocprofv3`'s `VGPR_Count` column is HALF the ISA allocation** (reads 512 for aiter's 1024). Read
   `.vgpr_count` from the ISA. s6: k_dkdv **713**, k_dkdv_sp 707, k_dqg **881** -- all **1 wave/SIMD**
   (VGPR-capped, not LDS-capped). A second wave needs <= 512 VGPR per wave.
3. **Static ISA metrics do not order candidates.** Arms that improved every static metric lost on the card
   (ku2 +8.2%, trim2 +1.3%, earlier -16%). Use them as gates (spill, scratch, a wait turned into a full drain);
   **price on the card** with the blocked benchmark.
4. **`prod` ranks; fast/proxy are sentinels.** `min_gain` is 0.007 (floor 0.24-0.66%; blocked A/A on A0
   0.01-0.07%). Never ship a candidate whose prod `vs_champion` is inside the noise, whatever fast did.
5. **The stall question is OPEN again: ATT works on this card since the reflash** (per-instruction
   Hitcount/Latency/Stall/Idle for FlyDSL JIT kernels and the ASM `.co`; recipe in h81 and in profiling step 4).
   Every s6 lever was found from an ATT read. ATT tells you WHERE cycles go; it never ranks. **PC sampling
   stays forbidden** (it wedged MES; rocprofv3 now rejects every PC-sampling config anyway). `--kernel-trace`
   still records 0 dispatches: take kernel names/timestamps from the `--pmc` csv.
6. **What bounds k_dkdv was measured by ablation** (h77): Q/dO already in VGPRs -18.5%; TDM source pinned in L2
   -1.0% (**L2 bandwidth is not the limit**); +16 WMMA/iteration (fused dQ matrix work) only +4%. Size a lever with
   an ablation before building it. Closed on A0 (h78): lse_late, ku2, flip, trorder_b, trim2, TDM without
   carried operands, k_dq U2. The fused 4-wave w4f is correct but parked at 6.51 ms (h79; 4.24 ms without atomics).
7. **k_dqg only runs at proxy and prod.** toy/fast dispatch k_dq_sp, so a toy/fast pass does not test a k_dqg
   change; its first real run is a serialised proxy validation.
8. **Machine rules.** The fp32 reference (`forward_reference`, `op/eager`) is **never computed on the card**: use
   `op/refcache_util.py` (`cached_forward` / `cached_backward`) or `op/refcache/*.pt` (it faulted a card on
   2026-09-28 and cost a power cycle on 2026-09-22). One shape per process. Compile-only first with BOTH
   `ARCH=gfx1250` and `FLYDSL_GPU_ARCH=gfx1250`; spill or scratch > 0 is a kill. A fresh
   `FLYDSL_RUNTIME_CACHE_DIR` per process and a new directory per arm (kernels.py arms are module-level switches,
   which the JIT cache key ignores). Leave the card idle when you finish: no scoring process left running.
