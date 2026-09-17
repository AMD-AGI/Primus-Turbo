# Incident: GPU wedged, 2026-09-13 ~13:30 UTC

**The card is wedged and needs a host reboot.** No GPU work can run until then. This is
wedge number five; `HARDWARE-ISSUE.md` records four in the eight days before today.

## Signature

```
[32982] amdgpu 0001:01:00.0: MES(0, 0) failed to respond to msg=INVALIDATE_TLBS
[33159] amdgpu 0001:01:00.0: MES(7, 0) failed to respond to msg=REMOVE_QUEUE
[33164] amdgpu 0001:01:00.0: wait for reset ack
```

Onset ~11 minutes before detection. The driver's own reset does not complete — it stops at
`wait for reset ack`, the state Phase 1 documented as requiring a reboot.

## Two commands that HANG on a wedged card — do not use them to diagnose one

- `ps -eo pid,stat,comm,wchan` — hung. Processes stuck in the driver make reading process
  state block.
- `sudo dmesg | grep ...` piped in the same shell as other device reads — hung.
- `rocm-smi` was already known to hang this way.

**Only `dmesg` read alone, with a hard `timeout`, is safe.** Everything that walks device or
process state is not. This matters because the natural instinct on a suspected wedge is to
run exactly the commands that will hang.

## What was running, and the probable contribution

`forever_queue.sh` with four workers, plus repeated exclusive-window probes. Those probes
took the GPU by `pkill -9`-ing the queue **mid-kernel** and restarting it — several times in
the preceding half hour.

**Repeatedly SIGKILLing processes with work in flight is a plausible contributor, and that
was my measurement pattern, not the queue's design.** The card is independently fragile
(five wedges in nine days), so this is not a sole cause, but it was avoidable.

**Corrected practice:** never take an exclusive window by killing the queue. Either enqueue
the measurement, or have the loop poll a `STOP` sentinel between candidates so it can be
stopped *between* kernels rather than during one.

## Driver reload was attempted and is NOT possible — a reboot is genuinely required

The cheaper recovery was tried before escalating to a reboot, and it fails at the first step:

```
docker stop fa-tune-0913
  -> cannot stop container: tried to kill container, but did not receive an exit event
lsmod | grep amdgpu
  -> refcount still 4
/sys/class/kfd/kfd/proc/  -> 1 process still held
```

The container's processes are stuck in D-state inside the driver, so the container will not
exit, the `/dev/kfd` handle is never released, the module refcount never drops, and
`modprobe -r amdgpu` cannot succeed. `rmmod -f` is not an option: force-unloading a module
with threads still inside it risks a kernel panic, which is strictly worse than a reboot.

This confirms what Phase 1 documented rather than merely repeating it.

## Why this was not rebooted automatically

Only the operator's own sessions are on the box, so no third party would be disrupted — but
one of them is **a tmux session open since 04:13** whose contents are not visible from here.
A reboot kills it, and whatever is in it cannot be recovered. That is a decision for whoever
owns that session, not for an autonomous run, so everything below is prepared and the reboot
is left to a human.

## Recovery

1. Reboot the host.
2. **`sudo modprobe amdgpu`** — this node blacklists amdgpu on the kernel command line, so
   it does NOT load on its own. Without it there is no `/dev/kfd` and torch reports
   "No CUDA GPUs are available".
3. `cat /sys/class/drm/card*/device/pp_dpm_sclk` — the VR throttle has returned after every
   reboot so far (1100 MHz ceiling).
4. `rocm-smi --showpids` must show no KFD processes and VRAM 0% before starting.
5. The container `fa-tune-0913` and the image `fa-tune:deps` (full torchtitan dependency set)
   survive; only driver state is lost.

## What was lost

Nothing committed. Sweep ledgers in the container are intact and every queue is resumable
(each skips tags already present). The one probe that did not finish is the
Triton-GEMM-vs-rocBLAS test, which is the first thing to re-run because it decides whether
end-to-end validation is possible at all.
