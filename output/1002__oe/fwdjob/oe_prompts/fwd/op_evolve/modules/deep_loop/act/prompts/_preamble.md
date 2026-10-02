You are implementing one change to an operator and measuring it. You planned this change in
the previous step, so you know why it was chosen -- but the round is judged on what you
deliver, not on the intent.

Round: {{round}}                 Candidate: {{candidate_id}}
Job context:  {{job_context_dir}}
Working copy: {{round_op_dir}}            # rounds/<n>/op/ -- edit here
Output dir:   {{out_dir}}                 # rounds/<n>/3-act/
Plan:         {{plan_yaml}}
Profiling:    {{profiling_dir}}
Beat profile: {{beat_profile_dir}}       # op/beat/ profiled once per job; absent if never taken
Knowledge:    {{knowledge_dir}}
Schemas:      {{schemas_dir}}
Runner:       {{runner_description}}

Rules that hold for every step:

**Never edit `{{job_context_dir}}/op/current/`.** It is a copy of the best round so far and
is regenerated only when a round is accepted. You work in `{{round_op_dir}}`, which starts as
a copy of it.

**One candidate.** `{{candidate_id}}`, and nothing else. If it turns out to be infeasible you
stop -- you do not take the next candidate from the pool. This round's profiling chose its
kernel and its shapes for this candidate; another one would be measured against the wrong
survey.

**Correctness is the obligation, speed is the measurement.** A faster kernel that computes
the wrong answer is not a result. Use the same precision check `op/validation.py` uses, never
one you write for yourself: iterating against your own tolerance converges on passing your own
bar, and the gate will then fail you for reasons you will misread.

**Every build and every measurement goes through the runner above.** Not by hand, and not on
whatever GPU happens to be reachable from where you are running. The controller may have GPUs
of its own -- it does here -- and using them silently moves the round onto a different machine
with a different toolchain, which makes its numbers incomparable with the profile they are
supposed to answer and with the champions the acceptance rule re-measures. `op/validation.py`
is executed through this runner, so a number obtained any other way is not the number that
decides the round.

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

**The framework runs `op/validation.py` itself when you are done**, through this same
runner, against your working copy, and its exit code is the one that decides whether the
job stops. Report yours as you measured it. If the two differ that is recorded rather than
resolved in your favour, so a number adjusted to look better is a discrepancy with your
name on it, and an honest failing run costs you nothing.

**Every measurement runs on an idle device and carries provenance.** Same discipline as
1-Profiling: `rocm-smi --showpids` and `--showuse` before and after, against the *physical*
GPU, and a `provenance.yaml` beside the numbers following
`{{knowledge_dir}}/../modules/deep_loop/profiling/schemas/provenance.yaml`. A number whose toolchain is
unrecorded cannot be compared against the next round.

**Write as you go, not at the end.** Build failures and what you think caused each one go
into `{{out_dir}}/act.md` when they happen. There is no attempt limit and no timeout here;
the bound is your judgement, and this record is what makes it visible -- to you as much as to
anyone. The same root cause three times means the direction is the problem, not the attempt.

**Reply with the YAML block each step asks for.** Prose above it is fine.

---

## ⚠ CAMPAIGN CORRECTIONS (fwd job, A0, written 2026-10-02) — inlined because this step does not read `hint.md`

`hints` is wired into `fast_loop`, deep `reflect` and `route.md`'s operator tables, but **not** into the
`profiling`, `plan` or `act` prompts. The job's operator file is `{{job_context_dir}}/hint.md`; **read h49 (this
host), h47 (rulers), h48 (pending refactor), h50 (no fp32 reference on the card) and h45/h46 there before you
start.** What changes decisions, measured on this card (A0, one gfx1250, container `fa-repro`, reflashed
2026-09-29) unless marked B0 (`PT/` = `/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo/`):

1. **The champion is r13ns** (round 16, refactor h44; `op/current`): the vendored aiter FlyDSL forward with
   longest-first dispatch, packed exp argument and row-sum, and speculation OFF (`SPEC_STALE_MAX=False`). proxy
   and prod dispatch `fmha_fwd_prefill_a16w16_m32x8` (rocprofv3 `kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0`):
   8 waves x 32 rows, 256-row WG, ~464 VGPRs in the ISA (the `VGPR_Count` column reads half: 232) -> 2 waves/SIMD,
   320 KiB LDS per WG. `fast` does not fill the device and takes the small-grid `m32x2` kernel (`impl.py`). Refactor h48
   (adopt `rounds/019/op`, +1.5% on real data after a GEMM burst) runs at the head of round 21; round 20's plan
   already builds on `rounds/019/op`. **Round 20 on A0 redoes B0's interrupted act**: `rounds/020/op` already holds
   B0's build of its candidate (`rounds/019/op` + the g65 fold: compiled, ut 16/16 and validation correctness PASS
   on B0, never timed; the bitwise check vs `rounds/019/op` faulted the card in its fp32 reference and never
   finished). B0's notes: `rounds/020/3-act.stale-*/act.md`.
2. **The job's ruler under-weights the training gap.** A0 2026-10-02 (`PT/output/1002__e2e/RESULT-realab.md`):
   blocked randn (this job's `benchmark.py`, which acceptance and validation use) r16/ASM time **1.08**; the 6
   real training q/k/v dumps right after a GEMM burst (training operating point, ~1.28 GHz) **1.349**: ASM barely
   moves with the clock, FlyDSL slows ~17%. Rank on the gate's ruler, but ALSO measure every candidate against
   the champion on the real dumps after a burst (h47) and report both; prefer cutting issued work and power.
3. **Where the cycles go** (A0 2026-09-30, `PT/output/0930__roofline/REPORT.md` §6, §9): prod m32x8 1.98e6
   cycles/SIMD = 53% of the matrix floor (ASM 1.43-1.45e6 = 72%, but power-capped at an effective 1.2-1.36 GHz
   against our 1.46-1.70). Ablations of r13ns: skeleton (LDS/TDM/sync/prologue/epilogue) 28%, softmax 19% (exp
   itself 2%), per-tile WG barrier 5%, causal imbalance 4%, mask ~0. K and V sit in separate 64 KB LDS segments,
   QK reads only K's and PV only V's, and the per-tile barrier phase-locks the 2 waves/SIMD, so one LDS read port
   works at a time (256 B/clk/CU = 2 waves x 512 B per WMMA, plus the TDM writes). A cycle cut that raises WMMA
   density lowers the clock: price time on the card, not cycles.
4. **Static ISA metrics do not order candidates.** A0 round 5's QK/softmax software pipeline improved the static
   metrics and lost 15.2% at prod (h28, L30 in h13). Use them as gates (spill, scratch, a wait turned into a full
   drain); **price on the card** with the blocked benchmark.
5. **Acceptance in this spec**: `min_gain` 0.007 over the plain mean of fast/proxy/prod (no `gain_weights`);
   bands fast 0.90, proxy 0.98, prod 0.993; validation's target is proxy AND prod each >= 1.00x the ASM beat. The
   best-ever records of rounds 18/19 belong to unpromoted arms (h48 retires them). **prod ranks**; fast (~13 us,
   m32x2) is launch-bound and its median moves with arm position.
6. **Rulers, measured on B0 and kept here**: blocked timing (h40); candidate vs champion in a process WITHOUT
   beat (h31); an in-process A/A copy (h35); one shape per process; a fresh `FLYDSL_RUNTIME_CACHE_DIR` per
   process and a new directory per arm (h46: the JIT key ignores module-level constants; fa-repro sets no cache
   dir, so processes that set none share `/root/.flydsl/cache`). The burst ruler needs the IMAGE hipBLASLt: this
   job's env, `_env.py` and `eager/impl.py` assign `HIPBLASLT_TENSILE_LIBPATH=/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250`,
   which makes the burst slow and leaves the clock high (ruler void). Re-assign
   `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250` after the
   arms are loaded and check the 10-GEMM burst takes ~21-25 ms, as `PT/output/1002__e2e/tools/realab.py` does. Dumps:
   `/home/lihuzhan/_prof_dump/qkv_call0{672,673,674,680,688,703}.pt` ([B,S,H,D] bf16).
7. **Speculation is dead** (h43-h45): never re-add stale-max / guessed-max / trigger-and-redo softmax; real
   training scores (std 21-53) recompute 13-25% of tiles, and the randn ruler rewards them anyway.
8. **Profiler state since the 09-29 reflash**: ATT gives per-instruction Hitcount/Latency/Stall/Idle for FlyDSL
   JIT kernels and the ASM `.co` (verified 2026-09-30 on the backward job's kernels; this op's kernels have not
   been captured on A0 yet -- confirm the capture armed). B0's rounds 10/15/20 skipped step 4 on the obsolete
   2026-09-24 finding. PMC works (`GRBM_GUI_ACTIVE`/8 = cycles; `--warmup-seconds 0` under `--pmc`).
   `--kernel-trace` still records 0 dispatches. **PC sampling stays forbidden.**
9. **Machine rules.** The fp32 reference (`op/eager` `forward_reference`) is **never computed on the card**: round
   20's act on B0 faulted the card inside it (`adv_m32x8.py`, 2026-09-28), and the same path cost a power cycle on
   2026-09-22. Use `op/refcache_util.py` `reference()` (refcache at fast/proxy/prod, the CPU for everything else)
   or compare candidate vs champion bitwise (both bf16 kernels); since 2026-10-02 `forward_reference` itself
   moves CUDA inputs to the CPU. Never re-run a script that faulted the card. Compile-only first with BOTH
   `ARCH=gfx1250` and `FLYDSL_GPU_ARCH=gfx1250`; spill or scratch > 0 is a kill; bounds-prove new index math on
   the CPU. One GPU client: the bwd job, e2e runs and labs never share the card with this job; check
   `ls /sys/class/kfd/kfd/proc` before trusting a number, and leave the card idle when you finish.
