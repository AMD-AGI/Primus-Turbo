---
name: gpu-kernel-campaign
description: Running an unattended multi-GPU kernel-optimization campaign — measurement discipline, fleet orchestration, and the failure modes that waste a day. Use when benchmarking or tuning GPU kernels (Triton/ASM/CUDA/HIP), running sweeps across several GPUs, driving autonomous optimization loops (op-evolve, claude -p campaigns), or diagnosing "my change should have had an effect and did not" / "the GPU looks idle" / "the card seems wedged".
---

# GPU kernel campaign: measurement discipline and fleet orchestration

Every rule here cost real time on a 4×gfx1250 campaign. They are ordered by how much.

## 1. The one rule: measure X directly, don't infer it from a proxy

In one day, **nine** "the change should have worked and didn't" investigations had their root
cause in the measurement chain, and **zero** in the change under test. The four most expensive
shared one shape — a fact that could be measured directly was inferred from a proxy instead:

| Wanted to know | Proxy used | Why the proxy lied | Direct answer |
|---|---|---|---|
| Is the card busy? | ledger file mtime | an exhausted sweep still appends `round_complete` every cycle | run a matmul; `rocm-smi --showuse` |
| Has the card recovered? | process left D state | leaving D state ≠ able to compute | run a 4096³ matmul on that card |
| Are all cards dead? | one matmul with a 150 s timeout | the timeout was mostly runtime startup | time the phases separately |
| Did my kernel path run? | absence of a vendor log banner | the launcher set the vendor's log level to ERROR | have the gate write its own trace file |

Three GPUs sat idle for hours and it was **the human operator who noticed, not the
monitoring** — because the monitoring watched a proxy.

**Before trusting any "is X working" signal, ask whether X can be measured directly.**
It usually can, and usually faster than building the proxy.

## 2. Absence is not evidence

When something "should have had an effect and didn't", check the measurement chain *first*:

- **Print the imported module's `__file__`.** An editable install registers a MetaPathFinder
  via a `.pth`; `sys.meta_path` is consulted *before* `sys.path`, so `PYTHONPATH` cannot
  shadow it. A whole day of end-to-end runs measured an image's pre-built package instead of
  the checkout, while `PYTHONPATH` looked correct in every log.
- **Check log levels before concluding a code path didn't run.** A launcher exporting
  `<VENDOR>_LOG_LEVEL=ERROR` silently removes the banner you are grepping for.
- **Check stdout capture.** Under a training launcher, `print()` may never reach the log.
  Have the probe write to a file it opens itself.

## 3. A diagnostic that changes whether the program runs is not a diagnostic

Inside an `autograd.Function` that `torch.compile` traces, each of these is a **fatal** error,
not a soft graph break: `os.getpid()`, `open()`, `os.environ[...] = ...`, and calling a
`@torch.compiler.disable`d function. Three probe designs in a row killed the training run.

Guard side-effecting diagnostics with `torch.compiler.is_compiling()` — it folds at trace time
and the tracer never sees the I/O behind it.

The same trap in shipped code: **AOTAutograd traces the backward while compiling the forward**,
so an `os.environ` write on a backward launch path lands in the traced graph. Unit tests that
don't compile the model will never see it.

## 4. Contention corrupts correctness signals, not just timings

A measurement that lands on a card another stream is using **does not raise**. It records a
wrong number — *and can produce a false SQNR/accuracy failure*. Same config, same seed:
contended 42–48 dB, clean 53.7–53.9 dB.

The accuracy gate is usually the only correctness defence. A false failure there makes you
discard a good config. **Reproduce every accuracy failure in an exclusive window before
believing it.** Real failures reproduce; contention artifacts do not.

Fencing one card is not enough when the others are loaded — a shared board power budget moved
single-card timings by up to 50% (11.2 ms to 17.1 ms for one config). Mitigations that work:
**interleave the A/B arms** so drift hits both equally, and prefer **best-of-N** over median
(it is closest to the undisturbed card). Verified afterwards on a quiet box: interleaving
widened the spread but did not shift the median — the discipline held.

## 5. Degraded ≠ wedged. Check which, and don't reboot on an inference

- `MES(...) ring buffer is full` — routine backpressure. Matching a bare `MES(` reports a
  healthy card as faulted.
- `MES(...) failed to respond to msg=...` — **degraded**: the card still finishes kernels.
- `wait for reset ack`, `ring gfx timeout`, `GPU reset begin` — **unrecoverable**: the
  driver's own reset never completes; a reboot is required.

A false wedge is worse than no detector: the watchdog then refuses to restart dead streams,
so it costs a whole day of a card that is still producing.

**Never conclude "the box is down" from a timeout.** A cold process can spend ~80 s in runtime
init and allocation before its first kernel. Time the phases separately:

```python
t0=time.time(); import torch;            t1=time.time()   # ~1 s
torch.cuda.init();                        t2=time.time()   # was 58–172 s here
a=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16); torch.cuda.synchronize(); t3=time.time()
b=a@a; torch.cuda.synchronize();          t4=time.time()   # first: compile
# then 5 more: steady state is the only number that means "the card works"
```

Test **each card separately** — one dead card among four looks like a dead box.

## 6. Orphans, self-matching patterns, and signals that lie

- **`pkill -f <pattern>` matches the killer's own command line** and takes your shell with it.
  Hit four times in one day. Kill by explicit PID; never by a pattern you are typing.
- **Agent/background children become `ppid=1` orphans and keep running.** One orphaned bash
  loop respawned GPU jobs for 2h15m; three cleanups killed only its children, and each time
  the reappearance was misdiagnosed as "the kill pattern missed". `ps -eo pid,ppid,args | awk '$2==1'`.
- **`timeout 5 kill -0 $p` runs `/bin/kill`, not the shell builtin** — different semantics,
  reports a live process as gone.
- **`kill -0` returning failure may be EPERM, not ESRCH.** A container running as root owns
  processes an unprivileged host user cannot signal; `2>/dev/null` turns that into "no such
  process". Containers also have their own PID namespace — host pid ≠ container pid, and a
  kill in one does nothing in the other.
- **`grep -c` / `pgrep -c` print `0` *and* exit 1 when there are no matches.** `|| echo 0`
  then yields `"0\n0"` and every arithmetic test on it fails. Read stdout, ignore the status.
- **`kill -9` cannot touch a D-state process.** Check `ps -o stat,wchan` before theorising.
- **A pattern that names one run also names its longer-named sibling.** `pgrep -f
  repro_l8b_turbo_conv.yaml` matches `repro_l8b_turbo_conv_8L_long.yaml`; `pgrep -f e2e.sh`
  matches every run that script ever launches. This is not the self-match above -- the pattern
  is about something real, it is just not *unique*. Three times in one day a watchdog armed for
  one run adopted the next one, once killing it. Give each run an identifier nothing else can
  contain (a marker env var, a PID) and match that. When you must match a name, anchor it:
  `-f "yaml$"`, or a tag that is not a prefix of any other tag.
- On a wedged card, `rocm-smi`, `ps ... wchan`, `pgrep`, and `torch.cuda.device_count()` all
  **hang** — the commands instinct reaches for are exactly the ones that die. Bounded `dmesg`
  and `/sys` reads are safe.

## 7. Attend long jobs on a timer, and make the monitor silent-until-broken

The two ways to get this wrong cost a day between them, and they are opposite mistakes.

- **Do not just wait.** A hung run looks exactly like a slow one. One hang burned 16 minutes
  of wall clock before anyone looked, on a card that then needed a human power cycle — the
  operator was away and the loop kept queueing work behind a corpse.
- **Do not narrate progress.** A monitor that emits one event per training step buries every
  useful line; the operator's words were "I can't see the useful output any more". Progress
  is not news.

So: poll on a timer (30 s to a few minutes), and **print nothing while the job is healthy.**
Speak only for: a card-fault signature, a stall past a threshold, an unexpected exit, or
completion. A healthy long job should produce zero monitor output from start to finish.

- **Silence is not success.** A filter that greps only the success marker stays quiet through
  a crashloop. Before arming, ask: *if this died right now, would my filter emit anything?*
  Widen the alternation until the answer is yes — some noise beats missing a corpse.
- **Check the card-fault signatures before the stall reading**, because they decide whether
  the answer is "restart it" or "stop and ask a human": `ring buffer is full`,
  `failed to respond`, `SIGBUS`, `wait for reset ack`. Baseline the count when the watch
  arms and compare against that, not against zero — old lines from a previous incident are
  still in `dmesg`.
- **mtime alone cannot tell a hang from a crash.** A run that dies without writing its
  completion marker leaves a log that simply stops, so a mtime-only watchdog reports
  STALLED forever and never exits. Also check that a process still holds the card
  (`/sys/class/kfd/kfd/proc/` — a plain directory read, safe on a misbehaving card, unlike
  `rocm-smi` or `ps`). No process and no completion marker means it died; say so and exit.
- **De-duplicate the stall warning.** One line when it starts, not one per poll.

## 8. When you orchestrate in parallel, assume a delegated agent will take the GPU

A subagent or workflow told to "analyse" will write and run probes if that is the shortest
path to an answer, and nothing in the prompt stops it. Launching an analysis track alongside
a measurement queue, believing the analysis is CPU-only, is how a clean queue becomes a
contended one.

This cost a real conclusion: a card fault during an unpatched baseline run was written up as
decisive evidence that the workaround under test was innocent — until the KFD holders turned
out to be the analysis track's own probes, one of which hung and pinned the card. The
baseline was genuinely unpatched; the measurement was still worthless.

- **Say it explicitly in the prompt** ("read files only; do not run anything on the GPU"), or
  fence the agent (`HIP_VISIBLE_DEVICES=` empty), or serialise the tracks.
- **Before trusting any card-fault attribution, list the KFD holders and identify every one.**
  A holder you cannot name is a reason to discard the measurement, not a curiosity.
- **A hang is not a failed candidate.** One knob value that never returns occupies the card
  indefinitely; a retry wrapper does not rescue it. Give every candidate a hard timeout, and
  treat "no output at all" as distinct from "a bad number".
- **The health probe can say HEALTHY while the card cannot work.** `degraded=0 wedged=0`
  describes driver flags, not whether a kernel can launch. A card occupied by hung processes
  passes the probe and fails a one-line GEMM. Confirm with actual work, bounded by a timeout.

## 9. Queue and watchdog design

- **Idempotent tags must not contain a monotonic field.** A tag carrying a round number is
  never "already done", so once the grid is exhausted the queue re-runs only that one
  candidate — 40 rounds of a single repeated measurement while looking healthy.
- **A completion marker refreshes the ledger mtime even when the round did nothing.** Any
  liveness check reading mtime will call an exhausted queue healthy forever.
- **Distinguish `parked` from `stalled`.** A stream held by your own exclusive-window sentinel
  looks exactly like a hung one. The sentinel file is ground truth for "this was on purpose".
- **Watch every card, including ones running one-shot jobs.** A watchdog that only restarts
  streams it has registered will let a finished one-shot leave its card idle indefinitely.
- **Give each stream its own compiler cache dir** (`TRITON_CACHE_DIR=/tmp/cache_g<N>`); a
  shared cache causes lock contention.
- **Stop by sentinel, never by signal.** Have the loop poll a STOP file between candidates so
  nothing is killed mid-kernel. Repeatedly SIGKILLing processes with work in flight is a
  plausible contributor to GPU wedges. Install the sentinel-removal trap on `EXIT INT TERM`
  *before* the wait loop, or a timeout leaves the queue parked forever.
- **Unique port per run** for anything using `torchrun`: one orphaned elastic agent holding
  the default port made 12 consecutive runs fail before step 1 — and they were recorded as
  completed, because the bookkeeping did not require a minimum number of steps.

## 10. What makes a result trustworthy

- **Gate every tensor separately** (out, dq, dk, dv against an fp32 reference). A config was
  observed 1.31× "faster" with dq at 9.59 dB while dk/dv stayed perfect — an output-only check
  actively rewards it.
- **Verify the config reached the kernel, not that the config source returned it.** Those are
  different claims. Read back the compiled kernel's options.
- **A verifier that cannot introspect must say UNVERIFIABLE, not fail.** One that fires on
  healthy runs is one people learn to switch off.
- **Two methods disagreeing means the gap is not established.** Don't change code on a number
  only one method produces.
- **No ledger row, no evidence.** Every claimed figure should be traceable to a stored record.
- **An independent confirmation from a different harness in a different process is worth more
  than either result alone.** Two of the day's accepted knobs were found twice this way.

## 11. An operator-level win is not an end-to-end win, and both numbers can be right

Measured 2026-09-15, gfx1250 single card. A prebuilt ASM attention backward beat the
vendored Triton one by **1.74x** at the operator level: 17.686 -> 10.160 ms, n=5, 0.5%
spread, SQNR bit-identical. Across 32 layers that is 241 ms off a 17.6 s training step.

Twenty steps of real training, the only difference being an env var that disables it:

```
ASM backward ON    1858 tps   mfu 34.49%   (n=13, spread 0.4%)
ASM backward OFF   1984 tps   mfu 36.83%   (n=13, spread 0.2%)
```

**Turning the faster kernel off made training 6.78% faster**, at thirty times the noise
floor. The step got about 1.2 s LONGER where it should have got 241 ms shorter -- roughly
1.4 s per step that operator-level timing cannot see.

Neither measurement is wrong. They measure different things:

- **Operator level** measures the kernel. The same tensors are reused every iteration, so
  the caching allocator hands back the same blocks and allocation costs nothing.
- **End to end** measures the kernel *plus what it demands of the allocator and the memory
  system*. Here: an fp32 dq_acc the alternative does not need (it does not accumulate
  atomically), and dk/dv allocated per q head rather than per kv head because the kernel's
  grid races otherwise. 1.07 GB per layer per step, ~34 GB of churn across 32 layers.

### What to take from it

**A kernel that needs scratch the alternative does not need has a hidden cost that only e2e
shows.** Before claiming an op-level win, ask what the kernel allocates that the thing it
replaces does not. If the answer is "a large fp32 accumulator" or "a larger output that gets
reduced afterwards", assume the e2e number will disagree until measured.

**Cache the scratch, never the returned tensors.** Reusing buffers across calls removes the
churn. But anything handed to autograd must be fresh -- a reused gradient buffer gets
overwritten by the next layer's backward before it is accumulated. Watch for the path where
no reduction happens (rep == 1 under GQA) and the scratch would be returned directly.

**The wider lesson is about what the campaign had evidence for.** Every result in this
campaign's first two days rested on operator-level measurement alone, because e2e could not
run at all -- a TypeError on an unaccepted `enable_gqa` kwarg had forced `converters: []`,
and the converter is what installs the attention under test. The first thing running e2e
revealed was that the headline result was negative in training. If a campaign cannot run
end-to-end, that is not a scheduling detail to defer; it is the thing blocking every
conclusion it produces.

## 12. When the infrastructure becomes the work, stop

Several hours went into adding streams, discovering idle ones, fixing the monitor, then
finding the monitor untrustworthy. If a measurement takes 8–13 minutes, a 6-minute inspection
cadence cannot close any kernel-level diagnosis — the loop spends more time restarting probes
than measuring. Match the cadence to the measurement, or batch the diagnosis and let it run.

## 13. The sweep harness is part of the experiment, and it has three failure modes

A sweep is a long unattended loop that launches code you have not run before. Three things
about the harness itself will quietly ruin a campaign day.

**A sweep with no per-candidate timeout is a card-wedging device.** Some configs do not fail,
they hang. A hung candidate is not a slow candidate: the subprocess never returns, the loop
never advances, nothing in the log says anything is wrong, and on a machine where recovery
means an AC cycle you lose the rest of the session. Give every candidate a wall clock, kill the
process *group* on expiry -- the child owns a GPU context, and a half-torn-down context leaves
a KFD holder that looks exactly like a wedged card -- and record the timeout as a **terminal
verdict, not a retry**. A crash is worth retrying because faults here are non-deterministic; a
hang is reproducible, so retrying it just spends the timeout again. A knob value known to hang
belongs outside the grid, not inside it with a timeout as the net.

**A knob your parser accepts is not necessarily a knob the backend reads.** Today's tuner
accepted `waves_per_eu` while that Triton's `Config.__init__` had no such parameter, so it
would have been passed through as a kernel constexpr to a kernel that never declared it. That
is the precise shape of a sweep that returns a flat line and gets written up as "this knob does
nothing on this architecture". Before trusting any axis, prove the value reaches the compiler:
read it back from `compiled.metadata`, or from `best_config`, or find the mechanism it travels
by. This is the same discipline as §1, applied to your own instrument.

**Do not touch the working tree while a measurement is reading from it.** Subprocess candidates
import the library from the checkout, so a `git checkout` or merge mid-sweep silently splits
the run into two halves measured against two versions of the code. Shared dispatch files are
the dangerous ones -- a backend-registration change does not look like it affects an attention
sweep, and it does. If a branch operation cannot wait, note the candidate index at the moment
of the switch and re-run that window afterwards; a resumable ledger makes that cheap, but only
if you remember the window existed.

## 14. Output discipline: the terminal is a report, not a log

A long campaign generates far more text than it generates decisions. If every grep, every
config dump and every commit message lands on screen at full length, the user stops reading --
and the one line that mattered ("the card wedged", "that number is wrong") scrolls past
unnoticed. Noise here is not a cosmetic problem; it is how findings get lost.

There is no Claude Code setting that hides file diffs and shows only filenames. `/config
verbose` controls output truncation and defaults to off already. So this is a discipline, not
a toggle.

**Default to quiet, and make loudness deliberate.**

- Edit files through Bash heredocs, `sed`, or short scripts rather than the Edit/Write tools.
  Bash edits render as a one-line description; Edit renders the full diff. Same result, a
  fraction of the screen.
- Every inspection command carries its own limiter: `| head -20`, `grep -c` instead of `grep`,
  `-q` when only the exit status matters, `>/dev/null` for the rest. Never `cat` a whole file
  to find one line.
- Long commit messages go in a file and get used with `git commit -F`, not typed into the
  command line where the whole body is echoed before it runs.
- Report results, not steps. "Sweep done, winner is X at Y ms" -- not a play-by-play of each
  candidate. Progress belongs in the silent monitor (§7), which speaks only when something
  breaks.
- One table per finding, not one per intermediate measurement. If a number does not change a
  decision, it belongs in the committed document, not on screen.

**What still deserves the screen**, in full and immediately: a card fault or anything needing
human action; a result that reverses an earlier conclusion; a number that contradicts something
already reported to the user; and the final answer to whatever was asked. Concision applies to
process, never to bad news.

## 15. Never put an autotuner on the online path, and distrust loops that swallow GPU errors

Two rules that came from one wedged card, and that generalise past the library involved.

**Tuning that can be done offline must not happen online.** An autotuner works by launching
configurations nobody has validated -- that is its job. Offline, a configuration that faults
costs you that process. Inside a training step it costs the step, the run, and on a
single-card box a power cycle. So measure on an idle card, write a table keyed on the *exact*
problem shape, and have the online path do nothing but look the shape up. A shape with no
entry falls back to the path that already works; that fallback is the feature, not a gap.
Run each shape's tuning in its own subprocess with a hard timeout, so one faulting shape costs
that shape and not the table. Gate every entry on correctness before it is written, not after
it is used -- a config that is fast and wrong must never reach a file the online path trusts.

**`except Exception: continue` is only safe when the exception is recoverable.** A GPU launch
failure is not. Once a HIP or CUDA context takes an unspecified launch failure it is dead;
catching the Python exception does not revive it, so a benchmarking loop that swallows the
error and moves to the next candidate keeps running against a corpse. The failure then
surfaces in unrelated work far away, and the real cause is dozens of iterations back. When you
read a tuning loop, check what it does with a *launch* error specifically -- "this config did
not compile" and "the device is gone" arrive through the same `except`, and only one of them
is a reason to continue.

**And the meta-lesson, which is the one that would actually have saved the card.** The estimate
document written an hour earlier already said, in as many words, that one shape had been
substituted rather than measured. That shape was then handed to a live autotuner mid-step.
Writing an assumption down is not acting on it: when your own notes say a case is unverified,
the next step is to verify it somewhere cheap, not to let the expensive path discover it.

## 16. A watchdog must be bound to the run it was started for

A monitor whose liveness check is a pattern like `pgrep -f e2e.sh` does not watch a run; it
watches a *name*. When that run ends and the next one starts, the old monitor is still alive,
still matching, and now supervising an experiment it knows nothing about -- including its
authority to kill it. Today one did exactly that: it outlived its run, attached itself to the
next one, and reaped it.

Bind the watch to the instance. Capture the PID at launch and check that PID, or put the run's
unique tag in the pattern, and have the monitor exit when its own run is done rather than when
*a* run is done. The same applies to the fault check: a monitor that greps `dmesg | tail` has
no idea whether those lines predate it, so stamp the boot time or the line count at arm time
and only react to what appears after.

This matters more than it looks. A watchdog that kills the wrong process produces a failure
that looks exactly like the failure it was watching for, and you will spend the next twenty
minutes investigating the card instead of the harness. Check the ordering before you believe
your own instrument: today the fault genuinely preceded the kill, but that had to be
established from timestamps, not assumed.

## 17. Score every run on correctness before you score it on speed

The fastest run of the day had `loss: nan` from step 10. It was reported as the best result,
and two separate explanations were built on top of it -- a bimodal distribution, then an
anomalous first run -- before anyone looked at the loss column. Eleven end-to-end runs had been
scored on throughput alone.

A broken computation is usually a *fast* computation. NaNs propagate without the work that
produces real numbers, denormals can take different paths, a kernel that skips tiles skips
their cost too. So the corrupted run does not look like an outlier to be investigated; it looks
like the best result you have, and it will anchor every explanation you build afterwards. This
is worse than losing the run: it actively steers the analysis.

**Make the harness do it.** Do not rely on remembering to look. The run script should read its
own log, count the non-finite steps, and print that count on the same line as the result --
with an explicit "discard this run" when it is non-zero. The discipline that survives is the
one that is mechanical.

**Watch what a correctness gate can actually see.** Two gates today could not have caught this:
a *mean* SQNR averages away a small wrong region, which is exactly the shape of a kernel that
misses tiles; and `float('nan') < threshold` is `False`, so an all-NaN result passes a
naive comparison. When the failure you fear is "part of the output was never written", the
instrument is a NaN prefill and `isfinite()`, not a summary statistic. Pass the output buffer
in pre-filled and check what the kernel left untouched.

**And a clean isolated test does not clear a kernel.** 720 calls across three processes came
back bit-identical to the reference while the same kernel had produced a NaN in training an
hour earlier. Isolation removes the conditions -- allocator state, concurrent streams, memory
pressure, whatever it was -- that the fault needs. "I could not reproduce it" is not "it does
not happen"; it means the reproduction is not yet built.


## 18. When one run turns out to be bad, check every run you have

A single corrupted result is rarely single. It is a sample from a rate you have not measured,
and the rate is the thing that matters. The day a reported result turned out to be a NaN run,
scanning every logged run in the repository -- 74 of them -- found eight more, and the split
was 8 of 39 runs with the patch under test against 0 of 35 without it. That is a Fisher
one-sided p of 0.004 and it reframed the whole project: not "one run went bad" but "this change
corrupts about one run in five".

The scan is cheap, it is pure CPU, and it works on logs you already have. Do it the moment you
find the first instance, before building any theory about that instance. Specifically:

- **Classify, do not just count.** Split the population by the thing you suspect (was the patch
  installed? which backend? which config?) and compare rates. A raw count tells you little; a
  contingency table tells you whether you have a suspect.
- **Check the exposure, not only the run count.** A group with fewer runs but more steps has
  more opportunity, not less, and saying so is what turns a suggestive split into an argument.
  Today the clean group had more total steps AND zero events, which is much stronger than the
  headline 21% vs 0%.
- **Look for confounds in the same pass.** If the two groups differ in configuration, step
  count, or date, say so before quoting the p-value.

The corollary is uncomfortable and worth stating: every conclusion drawn from that population
before the scan was drawn without a correctness filter, and some of them are now unsupported.
Go back and mark them, including the ones that were reported upward.

## 19. A sampling detector has a coverage, and the coverage is usually the answer

To catch a rare fault you will be tempted to check every Nth operation. Work out what that
actually covers before you trust a clean result. Sampling one call in seven over 6,840 calls
inspects 14% of them -- so even a run that *did* fail would most likely be reported clean, and
"no hits" would be read as evidence when it is nearly noise.

Raising it to every call is usually worse, not better: a full check means a device
synchronisation per operation, which serialises the pipeline and can suppress precisely the
timing-dependent fault you are hunting (see §3).

The way out is usually to separate *observing* from *synchronising*. Fold every result into a
running accumulator asynchronously -- NaN and Inf are absorbing, so once one enters it never
leaves -- and synchronise on the accumulator only every N operations. That is 100% coverage at
1/N of the syncs. Re-arm the accumulator after a hit so a second event is still visible.

State the coverage next to the result whenever you report one. "No non-finite output detected"
and "no non-finite output detected, at 14% coverage, in one run, against a 21% per-run failure
rate" are different sentences, and only the second one is honest.

## 20. Price the risk per unit of what actually fails

On a box where jobs die, find out *when* they die before designing the experiment schedule.
Here, every wedge that killed a run happened during training startup at step 0, and nothing
died mid-run all day: 3 of 14 startups failed, 21%, while the total minutes of running time
cost nothing. Risk was priced per **startup**, not per minute.

That inverts the natural instinct. Wanting to "use less of the card", I had been running short
jobs and running more of them -- which is the worst combination available when the startup is
the hazard. The right move is fewer, longer runs: a 60-step config instead of 20 gives three
times the post-warmup samples for the same single roll of the die, and several configurations
measured inside one process cost one startup instead of five.

So before planning a day's queue, ask what the failure is attached to. If it is startup, batch
work into runs. If it is thermal or time-dependent, do the opposite. Either way, say what the
per-unit rate is -- "about 5 runs to catch this fault once, at 21% wedge risk per startup, so
roughly one power cycle" is a cost a human can decide about; "it might take a few tries" is not.

## 21. Language: English in the repository, the user's language in what people read

Two different audiences, so two different defaults. Getting this backwards is a recurring slip
worth a rule.

**English, always, for anything that lives in the repository as engineering artefact:** commit
messages, code, comments, docstrings, identifiers, log and error strings, config keys, test
names, and the prose inside source files. These are read by everyone who touches the codebase
and by tooling that assumes ASCII; a commit subject in another language is unsearchable for
half the team and looks like an accident in `git log`.

**The user's language for documents written to be read by people:** progress reports, HTML
write-ups, analysis documents, weekly summaries, and the explanations in chat. If the user
writes to you in Chinese, these default to Chinese.

The line is not "source file versus markdown" — it is *who the reader is*. A findings document
under `output/` that a colleague will read follows the user's language. A README that ships with
the library is documentation for whoever clones it, so English.

**Watch for mixed strings specifically.** The failure mode is not writing a whole message in the
wrong language; it is dropping one word in. A Chinese word inside an otherwise English commit
subject reads as a typo and survives review because the sentence still parses. Before committing,
it is cheap to check: `git log -1 --format=%B | grep -P "[^\x00-\x7F]"` should return nothing.
