# Round 1 (fast) -- gfx1250 FlyDSL attention backward

Written as I go.

## 0. What I inherited

- `findings/` holds only the framework's empty `route.md`. **No `facts.md`, no `pool.md`,
  no `dead_ends.md`, no deep-round profile.** This is round 1; there is nothing in the
  job to recycle, so the sources are (b) my own survey, (c)/(d) the corpus, and the
  Op Setup review.
- Baseline (setup session, `op_setup_v000.md`): fast 3.04 / proxy 36.44 / prod 57.24 TF/s,
  i.e. **0.061x / 0.063x / 0.080x beat**. Beat = aiter prebuilt gfx1250 ASM bwd.
- Setup measured the bf16 roof at 1002.7 TF/s; baseline is at 5.7% of it, beat at 72%.
- Setup's review names four deliberate baseline weaknesses, in its own order:
  1. **no causal tile-skipping in either main kernel** -- "roughly 2x is sitting there";
  2. one wave (32 lanes) per workgroup, no multi-wave / async pipeline;
  3. `k_dkdv` pads a 16-wide contraction to the WMMA's 32 -- half that GEMM wasted;
  4. no bank-conflict swizzle on LDS staging.
- Precision headroom is thin: 52.5-52.8 dB against a 50 dB gate. Any approximation
  is out of scope this round.

## 1. Environment notes (cost me the first 15 minutes, recording so nobody repeats it)

- **`op/runner_util.run_detached` only redirects the LAST command of the string.** It
  builds `bash -c '<inner> > out 2>&1; echo $? > rc'`, so with `a; b; c` only `c`'s output
  is captured, and with `cd X && Y` the `cd`'s error is silently lost. My first survey
  returned `rc=1` with a **0-byte `out`** and nothing else -- which is exactly the failure
  mode the round brief warns about, except here the run genuinely had not started.
  **Wrap every multi-command payload in `{ ...; }`.**
- **The container's `/tmp` is NOT the host's `/tmp`.** Mounts are `[/home/lihuzhan]` only,
  so the round's scratch dir does not exist inside the container. Bulk profiling output
  therefore goes to a mounted dir and is moved to the scratch dir afterwards.
- `rocprofv3` is present (`/opt/venv/bin/rocprofv3`). **`rocprof-compute` is NOT installed
  in `fa-repro`** -- so no roofline/SoL tooling here without building it; counters via
  `rocprofv3` only.
- GPU idle before starting (`rocm-smi --showpids`: no KFD PIDs).
- **The runner does NOT inject `runtime.env` into the command.** `TORCH_BLAS_PREFER_HIPBLASLT`
  reads `UNSET` inside a `drive.py` payload, and torch then prefers hipBLASLt, whose gfx1250
  Tensile library is **missing from this image**. My first real survey attempt therefore died
  in `forward_reference`'s `torch.matmul` with
  `Memory access fault by GPU node-2 ... Reason: Page not present`, and **rocprofv3 then hung
  in its signal handler with the process still alive and still holding the card** -- `rc` was
  never written. I checked `pgrep -af` inside the container (the brief's rule), found PID 28344
  alive, killed it, and confirmed `rocm-smi --showpids` clean before relaunching.
  **Every `drive.py` call in this job must pass `--env TORCH_BLAS_PREFER_HIPBLASLT=0`.**

## 2. The survey

### 2a. rocprofv3 is not usable on this op, and I am recording that as the round's instrument finding

Two separate failures, both now pinned down, so no later round pays for them again:

1. **Under `rocprofv3`, the benchmark faults the GPU.** `benchmark.py` calls
   `forward_reference`, which does fp32 `torch.matmul`; under rocprofv3 that dies in a
   rocBLAS Tensile kernel with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` /
   `Memory access fault by GPU node-2`. The identical command WITHOUT rocprofv3 runs clean
   (`raw/baseline_proxy.txt`). So it is the profiler, not the code.
2. **With the matmul removed, `rocprofv3 --stats --kernel-trace` exits 0 and writes no
   files at all.** Empty `-d` directory, `output generation :: 0.002 sec`, rc 0. This is
   exactly the trap `knowledge/arch/gfx1250/profiling-surface.md:128-141` documents --
   *"no CSV, no warning, no diagnostic ... exit status is not a signal"*.

And even had it worked, it would not have told me much: of 51 counters defined on gfx1250,
**13 return non-zero, and they are wall cycles, wave count and I-cache only**
(`profiling-surface.md:60-61, 95-103`). *"gfx1250 exposes no counter that reports bytes, at
any level of the hierarchy"* (`:84`), and the entire `SQ_VALU_WMMA_FLOP_*` family reads
**exactly 0** on a kernel proven to issue 4.096M WMMAs (`:117-126`). `rocprof-compute` is
not installed in `fa-repro` either. **There is no counter-based route to a bound verdict on
this op on this machine.** What is left is stochastic PC sampling (`:143-173`) and
arithmetic.

**So I took the split with CUDA events instead** (`raw/split.py`), which is the number I
actually wanted. Cost: ten minutes. A round that had "asked for counters" here would have
got nothing.

### 2b. Where the time is (median of 21, synthetic o/lse, per-launch events)

    shape   delta            dkdv               dq                total
    fast    0.0599 ( 3.2%)   1.5595 (83.0%)     0.2604 (13.9%)    1.8799 ms
    proxy   0.0666 ( 0.7%)   7.0526 (73.7%)     2.4515 (25.6%)    9.5707 ms
    prod    0.1179 ( 0.1%)  64.3936 (67.1%)    31.3932 (32.7%)   95.9047 ms

The 95.905 ms total reproduces `benchmark.py`'s 96.053 ms end-to-end, so the split driver
is measuring the real thing. **`k_delta` is noise. `k_dkdv` is the kernel.**

### 2c. The prediction I made first, and what it bought

Before measuring I predicted `dkdv ~= 3x dq`, from counting WMMAs alone: per (16 queries x
32 keys) `k_dq` issues 8 (S,P) + 8 (dQ) = 16, while `k_dkdv` issues 2x(8 + 16) = 48.
Measured at proxy: **7.0526 / 2.4515 = 2.88x** against a predicted 3.00x.

That agreement is worth more than the split itself: **a static WMMA count predicts this
op's runtime to within 4%, so on this kernel I can price a change by counting matrix
instructions and do not need a profiler to do it.** Given 2a, that is the only cost model
available here, and it is now a validated one. (It degrades at `prod`, 2.05x, and inflates
at `fast`, 5.99x -- see 2d, which is why.)

### 2d. A second thing the split says, which nobody had written down

`k_dkdv`'s grid is `(Skv/16, Hkv, B)`:

    fast   (64, 2, 1) =    128 workgroups   <-- on a 256-CU part
    proxy  (256, 8, 1) =  2048 workgroups
    prod   (512, 8, 4) = 16384 workgroups

**At the `fast` shape the dominant kernel launches 128 single-wave workgroups onto 256 CUs.
More than half the machine is idle, and each CU that is busy runs one 32-lane wave.** That
is why `fast` reads 3.0 TF/s while `prod` reads 57.2, and why the WMMA model over-predicts
there. It also has a direct consequence for the candidate below, which I did not expect and
which changed how I read it -- see 4b.
