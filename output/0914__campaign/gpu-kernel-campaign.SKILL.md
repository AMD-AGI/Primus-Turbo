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
- On a wedged card, `rocm-smi`, `ps ... wchan`, `pgrep`, and `torch.cuda.device_count()` all
  **hang** — the commands instinct reaches for are exactly the ones that die. Bounded `dmesg`
  and `/sys` reads are safe.

## 7. Queue and watchdog design

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

## 8. What makes a result trustworthy

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

## 9. When the infrastructure becomes the work, stop

Several hours went into adding streams, discovering idle ones, fixing the monitor, then
finding the monitor untrustworthy. If a measurement takes 8–13 minutes, a 6-minute inspection
cadence cannot close any kernel-level diagnosis — the loop spends more time restarting probes
than measuring. Match the cadence to the measurement, or batch the diagnosis and let it run.
