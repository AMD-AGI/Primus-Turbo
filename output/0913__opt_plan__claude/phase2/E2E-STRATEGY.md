# End-to-end validation: what is broken, and what to do instead

## The problem, measured

A real training step on this node runs at **245 tokens/s = 133.7 s/step**. The entire
attention path at that shape is **1.14 s** — under 1% of the step. The Triton path and the
flex path land within **0.4%** of each other because neither is what the step waits on.

Root cause: **the container image has no working BLAS for gfx1250.**

| GEMM (bf16, idle card) | achieved |
|---|---:|
| 8192³ | **27.4 TFLOP/s** |
| 32768 × 4096 × 14336 | 23.9 |

Against two numbers from the *same card, same session*: the tuned attention kernel sustains
**315 TFLOP/s**, and a Triton GEMM established a **1002.7 TFLOP/s** roof. **A dense GEMM
running 8× slower than a flash-attention kernel is backwards** — attention does strictly
more work per FLOP. hipBLASLt's `TensileLibrary_lazy_gfx1250.dat` is simply absent from the
image, and `TORCH_BLAS_PREFER_HIPBLASLT=0` falls back to a rocBLAS path measured above.

**Consequence: an end-to-end tps number on this image cannot validate attention work, and
cannot falsify a regression either.** It also means the JIRA's MI455X figures (GEMM 655 ms,
19,795 tps) were not produced on this image.

## Can it be fixed? The untested hypothesis

**`torch.compile` with inductor generates Triton GEMMs**, and Triton on this card reaches
1002.7 TFLOP/s. If the model compiles, the linear layers stop going through the broken BLAS
and E2E becomes meaningful.

Two things make this plausible and one makes it uncertain:

- The MI455X repro configs set `compile.enable: false`, with a comment saying inductor's
  autotune on `TransformerBlock` triggered `hipErrorLaunchFailure` and killed the GPU. That
  was on an older image; it has not been retried since.
- flex attention already goes through inductor on this card and works, at 31.3 ms — so
  inductor *can* produce working kernels here.
- Against: the failure that stopped it was a GPU kill, i.e. the expensive kind.

**This probe was written and did not run — the card wedged first.** It is cheap (minutes)
and it decides whether E2E is available at all. It is Item 0 for tomorrow.

## If the probe fails: what a defensible validation looks like instead

Do **not** fall back to "run training and read tps". On a GEMM-bound step that number is
insensitive to attention by construction, so a green result proves nothing and a red result
cannot be attributed.

Use a **composition check** instead, which is what the JIRA itself did:

1. **Per-layer attention time from a profiled step.** Sum the attention kernels' device time
   over one real step. This is the quantity the optimisation actually changes.
2. **× 32 layers** gives the attention path per step, directly comparable to the JIRA's
   "FA path 755 ms" row.
3. **Report it against the step's other components** (GEMM, elementwise, optimizer) so the
   share is visible and nobody reads a 1.45× kernel win as a 1.45× training win.

This is strictly better than a tps number on a broken-BLAS image, because it measures the
thing that changed rather than a sum dominated by something that did not.

**It is not a substitute for a real E2E run on a healthy stack.** Two things a composition
check cannot catch, and they must be closed separately:

- **Transfer failure.** A microbenchmark win that depends on a warm config cache, a fixed
  input, or a shape that training does not actually issue. Mitigated here by the fact that
  a training step issues exactly one attention shape (MBS=GBS=4, no grad accumulation, no
  activation checkpointing → 32 forward + 32 backward calls of one shape), and by having
  swept eight shapes rather than one.
- **Integration failure.** Wrong gradients reaching the optimizer, a dispatch path that
  silently falls back, memory growth. Mitigated by the four-tensor SQNR gate, by
  launch-level verification of which kernel ran, and by the 400-rep bitwise determinism run.

## Recommendation

1. **Tomorrow, first:** run the `torch.compile`-GEMM probe. Minutes, and it decides the rest.
2. **If it works:** E2E is back. Run the repro config with compile on, both attention paths,
   and report tps. This is the only thing that closes the loop.
3. **If it does not:** ship on the composition check, and raise the missing gfx1250 BLAS as
   a platform blocker alongside the VR throttle. State plainly in any report that E2E tps
   was not validated and why — do not let a composition number be read as an E2E number.
4. **Either way**, ask the platform owners for an image with a working hipBLASLt. The BLAS
   gap is worth ~37× on GEMM; no amount of attention work is visible behind it.
