# Environment

Written by `tools/check_runner.py`. Observed values, not the ones the job
file asked for.

| Check | Result | Detail |
| --- | --- | --- |
| reach              | ok | ctheliosp-1b112-a37-1.mnb.dcgpu (kfd) via docker |
| torch gemm         | ok | torch   2.11.0+rocm7.14.0a20260625 \| gemm    4096^3 bf16  1.116 ms  123.2 TFLOP/s |
| shared path        | ok | verified both directions at /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-attn-llama31-8b-bwd-20260914-015415 |
| counters           | ok | GRBM_COUNT, 27 rows |
| decoder            | ok | reachable; the profiling agent may fetch it |

`run_torch_gemm.sh` beside this file is what the GEMM check ran, left so it
can be repeated by hand.

---

## Observed by job setup, 2026-09-14

Everything below was observed directly on this machine, in the pinned image. These
are measured values, not the values the job file asked for; where the two differ it
is said so.

### Access path

| Item | Observed |
| --- | --- |
| ssh hops | **none.** `runtime.runner.ssh` is null and `runner.type` is `docker`, so the controller, the node and the container host are one machine. |
| node | `ctheliosp-1b112-a37-1.mnb.dcgpu`, Linux 6.16.1-0_fbk2_brcmrdma5_35_g5ba27bd1d6b9 |
| docker | server 29.7.2 |
| image | `docker.io/amdprimus/amdprimus:gfx1250-20260910`, image id `sha256:6e656de79e6c…` |
| image pull | **FAILS — `pull access denied … may require 'docker login'`.** The image is present locally and its only RepoDigest equals its image id, i.e. it was loaded/built here and never came from a registry this daemon can reach. `docker run` works because the image is local. **If it is ever `docker rmi`'d this job cannot recover it.** |
| container | `op-evolve-gfx1250-attn-llama31-8b-bwd` — did not exist before setup; created by `check_runner.py` and left running. Runner-owned (`docker.container` is null), so the runner may start and provision it. |
| docker options | all nine from the spec accepted verbatim: `--ipc=host --network=host --device=/dev/kfd --device=/dev/dri --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN --security-opt=seccomp=unconfined --group-add=video` |
| mount | `/home/lihuzhan/code/2026_0910__op-evolve/op-evolve` at the same absolute path both sides (the whole checkout — it already contains `artifacts/`, `tools/` and `knowledge/`). Verified in both directions by the shared-path check above. |
| sudo | **available, passwordless** (`sudo -n true` succeeds). Spec asked for `sudo: true`; matches. |

### GPU

| Item | Observed | Spec asked for |
| --- | --- | --- |
| GPU present | yes — `/dev/kfd` plus 32 render nodes; **4** gfx1250 cards on this node (kfd topology nodes 2–5) | — |
| gpu_arch | `gfx1250`, `gfx_target_version 120500` on every card | `gfx1250` ✅ |
| gpu_id | **1 is usable.** `HIP_VISIBLE_DEVICES=1` gives torch exactly one device, `gcnArchName gfx1250` | `1` ✅ |
| wave_front_size | 32 | 32 ✅ (wave32 — the premise of the whole `num_warps` argument) |
| lds_size_in_kb | 320 | 320 ✅ |
| simd_count | 1024 → 256 CUs / 128 WGPs (`simd_per_cu 4`, `array_count 32`) | 256 CUs ✅ |
| memory | 432.0 GiB per card | — |
| `max_engine_clk_fcompute` | 2400 MHz | — |
| driver | amdgpu 7.1.1.31300009 | — |
| GPU pool | `runtime.gpu_pool: []`, so profiling runs every concurrent analysis on GPU 1. Note the spec comment says "single card on this node"; there are **four**. The pool is a fencing choice, not a hardware fact. | — |

### Software

| Item | Observed |
| --- | --- |
| rocm-smi | **installed and runs**: `/opt/venv/bin/rocm-smi`, ROCM-SMI 4.0.0+feb9c98, ROCM-SMI-LIB 7.8.0 |
| ROCm | **7.14.60850** (`hipconfig --version`). Not at `/opt/rocm` — it is the pip wheel SDK at `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel`. |
| torch | **2.11.0+rocm7.14.0a20260625**, `torch.version.hip 7.14.60850`. Imports, sees the GPU, and executes a bf16 matmul on GPU 1. |
| triton | 3.6.0 |
| rocprofv3 | 1.3.2, at `/opt/venv/bin/rocprofv3`. Collected `GRBM_COUNT` and wrote 27 rows — so `CAP_SYS_ADMIN` + `seccomp=unconfined` are effective. |
| `rocprof-compute` | **absent.** Only `rocprofv3` is installed. |
| aiter | **not installed** (`ModuleNotFoundError: No module named 'aiter'`). This directly corroborates the job description's claim that the `turbo_attention` patch is applied and then dies at the first step. |

### Three things a later round will otherwise be surprised by

**1. The card is NOT at the 1100 MHz the job file was written against.**
Under a sustained 45 s GPU load, GPU 1 held **1701–1702 MHz** across three samples at
100% utilisation (idle reads 2400 MHz). The job file's header states op-evolve Phase 1
measured `MAX_CLK 1100 MHz`, against 1699–1703 MHz on 2026-09-04 — and 1701 MHz is that
earlier, un-throttled figure to within 1 MHz.

⚠ **This is necessary but not sufficient evidence.** The load used was `torch.matmul`,
which on this image runs at ~113–124 TFLOP/s (see below) — a VALU-rate, low-power load.
A matrix-core-saturating kernel draws far more power and could still hit a VR limit that
this load never approaches. What is established is that the card is not *statically*
capped at 1100 MHz. What is not established is the clock under a real attention kernel.

Consequence if it holds: **every absolute figure in the job file is understated by
~1.55x**, including the 1002.7 TFLOP/s roof, the four seed anchors, and the
"20.8 ms = 37% of roof" arithmetic. It would also explain the header's unresolved
c1325c7e contradiction (220.6 vs 131.7 TFLOP/s = 1.675x). The paired re-measurement the
header says setup owes is therefore not hygiene — it is mandatory, and it must record
the clock under the kernel being measured.

**2. `check_runner.py`'s GEMM number is NOT a roof — it is runtime hazard 3, reproduced.**
The table above records `4096^3 bf16 1.116 ms 123.2 TFLOP/s`. Setup re-measured:
`4096^3 → 123.8`, `8192^3 → 113.5 TFLOP/s`. Flat in size, which is the signature the job
file describes. `torch.matmul` goes through hipBLASLt, and runtime hazard 3 says outright:
never use hipBLASLt as a roof, a reference or a GEMM baseline on this image (it measured
91.5 against Triton's 1002.7, a 10.96x gap). **The check passing means "a GPU GEMM runs
through the runner", which is all it claims. Do not quote 123 TFLOP/s as this card's
capability anywhere.** A Triton GEMM roof was not measured here — writing an untuned
Triton GEMM and reporting its result as "the roof" would be worse than reporting none.

**3. The seed kernel and the mandated harness are not in the pinned image.**
The SEED block says to seed from `primus_turbo/triton/attention/fused_mha_bwd_kernel.py`
and `primus_turbo/pytorch/kernels/attention/attention_fused_bwd_impl.py`, and
`op.reference.api` says to use `tools/gfx1250/tune_attention.py` and
`tools/gfx1250/exclusive.sh` and "do not write a new one". **None of those four files
exist in the image.** The image's Primus-Turbo is at commit `f857e429` and carries only
the in-tree unfused `attention_triton_impl.py` / `attention_kernel.py`, and has no
`tools/gfx1250/` at all.

All four files DO exist in the host checkout at
`/home/lihuzhan/code/2026_0903__turbo/Primus-Turbo` (commit `d2f75576`) — which is **not
mounted into the container**, since the only mount is the op-evolve checkout.
`import primus_turbo` inside the container resolves to `/workspace/Primus-Turbo`, so a
`PYTHONPATH` alone is not enough either; the two trees would mix.
**Op Setup must resolve this before it can build `op/baseline`.** The clean route is to
vendor the needed files into `op/` (which lives under the already-mounted job directory)
and pin how `primus_turbo` is imported; the alternative is adding the turbo checkout to
`container_mounts`. Setup did not choose, because it is Op Setup's structure to decide.

### Minor

- `ContainerLayer._provision()` globs `/opt/rocm*/libexec/rocprofiler-compute/requirements.txt`.
  ROCm is not at `/opt/rocm` here, so that glob misses and provisioning prints
  "no rocprof-compute requirements found". Harmless — it is reported, not raised — and
  moot anyway, since `rocprof-compute` is not installed.
- `dmesg` was grepped for `MES(` / `GPU Hang` / `wait for reset ack` before and after
  every check, per runtime hazard 2. **Zero matches both times.** `rocm-smi` was called
  only after that grep came back clean, and never hung.
- Per runtime hazard 1, **no PC sampling was run**. Per hazard 5, `tools/clock_probe.py`
  was **not** run — the clock above came from `rocm-smi --showgpuclocks` sampled beside an
  independent load.
- At setup time two unrelated containers (`fa-e2e`, `fa-repro`) were running on this node
  on the same image. All four GPUs read **0% utilisation** before and after setup's own
  load, so neither was using GPU 1 — but nothing prevents them from doing so mid-round.
  There is no GPU lock (runtime hazard 6); check the card is idle before every round.

### Measurement floor spread — NOT measured

`evolve.min_gain` is 0.0. The framework's own guidance is to set it from the spread this
machine shows, recorded here. **Setup could not measure it**: there is no `op/baseline`
to repeat-measure yet. The 0.26% cross-process spread quoted in the job file is carried
from a prior campaign on a different seed and a different clock state. **Op Setup owes
this number**: run `op/baseline` in ≥5 separate processes and record the spread here.
