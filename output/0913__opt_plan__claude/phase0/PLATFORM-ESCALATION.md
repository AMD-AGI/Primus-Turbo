# Escalation: gfx1250 node `heliosr-1b114-c07-1` is VR-throttled, and it is the single
# largest variable in the MI455X attention effort

**To:** platform / BMC / firmware owners for this shelf
**From:** the MI455X Llama-3.1-8B attention optimisation effort
**Date:** 2026-09-13
**Asking for:** (1) a decision on whether this card's clock ceiling can be restored, or a
different gfx1250 node; and (2) a container image with a working BLAS for gfx1250. Neither
is something we can act on from inside the OS.

**Updated 2026-09-13 after a full day on the card.** Issue 1 (VR throttle) is unchanged —
a reboot did *not* clear it. Issue 3 below is new, was found by measurement, and is
currently the larger of the two for anyone trying to benchmark end-to-end.

The detailed technical report already exists and we are not duplicating it:

> `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/output/0911__fa_gfx1250_phase1/HARDWARE-ISSUE.md`
> (written 2026-09-11 by the op-evolve Phase-1 probe, with dmesg, DPM tables and
> paired measurements attached)

This note adds the one thing that report could not: **what the throttle costs this
project**, in the project's own numbers.

---

## The finding, in one line

`MAX_CLK` reads **1100 MHz**. The same physical card held **1699–1703 MHz** under load on
2026-09-04. The kernel prints `GPU is throttled, expect performance decrease. VR.` at
amdgpu init on every boot observed since, and the sclk DPM table is truncated to two
levels (500 MHz, 1100 MHz).

That is a factor of **~1.65**.

---

## Why it decides this project rather than merely slowing it

We are trying to close a gap on MI355X. MI455X currently runs Llama-3.1-8B BF16 at 19,795
tokens/GPU/s against MI355X's 21,351 — about 7% behind end-to-end, with the entire deficit
in the attention path (755 ms/step vs 315 ms). The GEMM path is already **1.51× faster**
than MI355X, so attention is the whole story.

The measured bf16 compute roof on this card, in its throttled state, is
**1002.7 TFLOP/s**. Attention at this shape needs 246.29 TFLOP per training step. That
gives a hard floor on what the attention path can cost:

| target | attention ms/step | required TFLOP/s | vs the 1002.7 roof |
|---|---:|---:|---|
| the original 30,000 tps ask | 192 | 1,282.8 | **128% — impossible by construction** |
| parity with MI355X | ~315 | 782 | 78% of roof — above any known FA implementation |
| what we think is reachable, throttled | ~560 | 440 | 44% of roof — plausible |
| the same kernel, un-throttled | ~340 | 725 | 44% of an un-throttled roof |

An attention kernel cannot exceed the dense-GEMM roof of the same silicon at the same
clock. So at 1100 MHz:

- **The 30,000 tps target is not merely ambitious, it is unreachable.** We have already
  re-baselined the project's goals against the measured roof rather than carry a target
  the hardware cannot produce.
- **Even a perfect kernel does not reach parity with MI355X.** Our best case throttled is
  roughly 22,300 tps (104% of MI355X); un-throttled the same kernel work lands near
  25,900 (121%).

**So restoring the clock is worth about 3,600 tps on its own — more than the entire kernel
campaign is expected to deliver, and it requires no engineering from us.** We would rather
know now than discover it in week six.

---

## The second problem, which is about whether we can run at all

The same report documents that this node **wedged four times in eight days**, twice within
2.5 hours of one work session. Each time the signature is the same: MES firmware stops
answering, the driver's reset path cannot complete (`wait for reset ack`), tasks pile up
in D-state, and a host reboot is required.

Our plan is an unattended multi-round tuning campaign, roughly 40 rounds over a day and a
half. A node that needs a reboot every few hours cannot host that. We have built the
harness to be resumable and to fsync its ledger after every candidate precisely because we
expect this, but that limits the damage rather than fixing it.

We have also removed `rocprofv3` PC sampling from our tooling entirely — it faulted the
GPU on all three attempts and was the proximate cause of one of the wedges — so we are not
contributing that particular stress.

---

## What we are asking for

In priority order:

1. **Can the VR throttle be cleared on this card?** The report's specific asks are
   BMC-side input-power / VR-fault telemetry for this shelf and a PPTable dump compared
   against a known-good gfx1250. Neither is determinable from inside the OS.
2. **If it cannot be cleared, can we have a different gfx1250 node** whose clock ceiling
   is intact? Everything we measure on a throttled card has to be re-measured later
   anyway, so this changes the schedule, not just the numbers.
3. **When did the throttle start?** Last known-good is 2026-09-04; first bad is
   2026-09-10 22:38. Older `/var/log/kern.log` rotations or BMC event history for
   09-05..09-09 would bracket it, and would tell us whether this is a degrading part or a
   configuration change.
4. **Is the wedging one root cause or several?** Firmware-side MES logs, or an AMD triage
   of the 09-04 and 09-11 dmesg together, would settle it. We can supply everything from
   our side.

---

## What we are doing meanwhile

Not blocking on this. All GPU-free work is being completed now — dispatch fixes, the
tuning harness, the correctness reference, the job specification, and documentation — so
that the moment a healthy card is available the measurement campaign starts the same day
rather than beginning from scratch.

One consequence worth flagging back to whoever reads our results later: **every absolute
performance number we publish from this node is conditional on its clock state**, and we
are labelling them that way. There is already one unexplained 1.675× discrepancy in the
record between two measurements of the same kernel at the same shape — which is the
throttle ratio to within noise, and which we intend to settle with a paired measurement
the moment we have a card we trust.


---

## Problem 3 (NEW, 2026-09-13) — the container image has no working BLAS for gfx1250

This is the one that blocks end-to-end work today.

### The evidence

`amdprimus/amdprimus:gfx1250-20260910` does not ship hipBLASLt's Tensile library for this
architecture:

```
rocblaslt error: Cannot read ".../_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/
                 TensileLibrary_lazy_gfx1250.dat": No such file or directory
```

Any fp32 `torch.matmul` therefore raises `HIPBLAS_STATUS_INVALID_VALUE`. Setting
`TORCH_BLAS_PREFER_HIPBLASLT=0` falls back to rocBLAS, which runs — but at this speed
(measured on an idle card, `rocm-smi` confirming zero KFD processes, bf16):

| GEMM | time | achieved |
|---|---:|---:|
| 8192 x 8192 x 8192 | 40.12 ms | **27.4 TFLOP/s** |
| 32768 x 4096 x 6144 | 68.77 ms | 24.0 TFLOP/s |
| 32768 x 4096 x 14336 | 161.00 ms | 23.9 TFLOP/s |
| 32768 x 14336 x 4096 | 147.20 ms | 26.1 TFLOP/s |

### Why we are confident this is broken rather than "the card is just slow"

Two numbers from the *same card, same session, same afternoon*:

- our tuned flash-attention kernel sustains **215 TFLOP/s**
- a Triton GEMM established a **1002.7 TFLOP/s** bf16 roof (already in the throttled state)

**A dense GEMM running 8x slower than a flash-attention kernel is backwards.** Attention
does strictly more work per FLOP — softmax, masking, online rescaling — and should be the
slower of the two. rocBLAS here is ~37x off the roof the same silicon demonstrably reaches.

### What it costs

Every end-to-end training measurement on this image is GEMM-bound by a margin that makes
everything else invisible. Concretely, Llama-3.1-8B BF16 at mbs 4 / seq 8192:

- 245 tokens/s, 133.7 s/step, 14 TFLOPS overall, MFU 4.5%
- 100% GPU utilisation doing almost no useful matrix work
- the Triton-attention path and the flex path land within **0.4%** of each other, because
  neither is what the step is waiting on

For scale, the attention path at this shape is 1.14 s of that 133.7 s step — under 1%.

This also means the JIRA's MI455X figures (GEMM 655 ms/step, 1.51x faster than MI355X;
19,795 tokens/GPU/s) **cannot have been produced on this image**. Whatever environment
produced them has a functioning BLAS.

### The ask

A gfx1250 image with a working hipBLASLt (i.e. one that actually contains
`TensileLibrary_lazy_gfx1250.dat`), or guidance on which existing image has one. We checked
four images on this host — `amdprimus/amdprimus:gfx1250-20260910`,
`primus-turbo:gfx1250-20260831-extended-v2`, `gemma4-mi455x:gfx1250-20260910`,
`recommendation-amdprimus0815:triton-3.8.0-amd` — and they all carry the same torch build
(`2.11.0+rocm7.14.0a20260625`).

Kernel-level optimisation work is **not** blocked by this and is proceeding: the attention
kernel is measured, tuned and 1.66x faster. Only end-to-end validation is blocked.
