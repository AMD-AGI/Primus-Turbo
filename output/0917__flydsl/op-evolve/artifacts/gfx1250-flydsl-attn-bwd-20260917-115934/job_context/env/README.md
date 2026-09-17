# Environment

Written by `tools/check_runner.py`. Observed values, not the ones the job
file asked for.

| Check | Result | Detail |
| --- | --- | --- |
| reach              | ok | heliosr-1b114-c07-1 (kfd) via docker |
| torch gemm         | ok | torch   2.11.0+rocm7.14.0a20260625 \| gemm    4096^3 bf16  5.060 ms  27.2 TFLOP/s |
| shared path        | ok | verified both directions at /home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934 |
| counters           | ok | GRBM_COUNT, 29 rows |
| decoder            | ok | reachable; the profiling agent may fetch it |

`run_torch_gemm.sh` beside this file is what the GEMM check ran, left so it
can be repeated by hand.

---

## Added by Job Setup, 2026-09-17

Everything below was observed on the machine, not read off the job file. Where an
observed value differs from what the spec asks for, the difference is called out.

### Access path

| Item | Observed |
| --- | --- |
| ssh hops (`runtime.runner.ssh`) | **none** -- the key is absent from the spec, so this is a *local* docker runner. Controller, node and container are all `heliosr-1b114-c07-1`. Nothing to verify, and nothing that can go down between the loop and the card. |
| host | `heliosr-1b114-c07-1` |
| runner type | `docker` |

### Container

| Item | Observed |
| --- | --- |
| container | `fa-repro`, **Up** at check time (3 h old) |
| image | `fa-tune:deps`. The spec's `docker.image` is `null` and stays `null`: `container:` is set, so the runner attaches to a container the **user owns** and never creates, restarts or `pip install`s into it. Nothing was pulled. |
| `docker.options` | `[]` in the spec, and **not applied either way** -- `options` is only read on the create path, which is dead here. The options the container actually runs with are below, and they came from whoever started it. |
| binds | `/home/lihuzhan:/home/lihuzhan` |
| devices | `/dev/kfd`, `/dev/dri` (both `rwm`) |
| other | `ipc=host`, `net=host`, `privileged=false`, `seccomp=unconfined`, `label=disable`, `group_add=[video, 993]`, `shm=16 GiB` |
| job dir inside the container | present at the identical absolute path (the single `/home/lihuzhan` bind covers it) |

### GPU

| Item | Spec asks | Observed |
| --- | --- | --- |
| GPU present | yes | yes, exactly **one** (`torch.cuda.device_count() == 1`) |
| `gpu_arch` | `gfx1250` | **`gfx1250`**, agreed by all three: `rocm-smi` GFX Version, `rocminfo` `amdgcn-amd-amdhsa--gfx1250`, torch `gcnArchName` |
| `gpu_id` | `0` | usable -- allocation and a bf16 GEMM both ran on device 0 |
| marketing name | -- | `AMD Eng Sample: 100-000001046-04`, SKU `M4500001`, device id `0x75c1` |
| `gpu_pool` | `[]` | correct: there is no second card to pool |

### Software

| Item | Observed |
| --- | --- |
| torch | `2.11.0+rocm7.14.0a20260625` |
| `torch.version.hip` | `7.14.60850` |
| ROCm, per `rocm-smi` | **driver version `7.1.1.31300009`**. Note `rocm-smi` is `/opt/venv/bin/rocm-smi`: this container has **no `/opt/rocm`** at all, so there is no `/opt/rocm/.info/version` to quote. ROCm arrives through the `_rocm_sdk_libraries_gfx1250` wheel inside the venv. |
| `rocprofv3` | `1.3.2` (`git feb9c98f`) |
| flydsl in the image | `0.2.4` at `/opt/venv/lib/python3.12/site-packages/flydsl` |
| flydsl this job must use | **`0.3.2`** at `/home/lihuzhan/.local/flydsl032/flydsl`. Already installed -- no `pip install --target` step is needed. |
| aiter | **not installed in the container.** It is imported off `/home/lihuzhan/code/aiter-src`. |
| sudo | `sudo -n true` **succeeds**, so sudo *is* available. The spec's `runtime.sudo: false` is kept -- it is a policy ("do not use sudo"), not a statement about the machine. |

### Two things that will bite whoever writes `op/baseline/` and `op/validation.py`

**1. `sys.path`, in this order, before `import flydsl` / `import aiter`:**

    sys.path.insert(0, "/home/lihuzhan/.local/flydsl032")   # 0.3.2, ahead of the image's 0.2.4
    sys.path.insert(0, "/home/lihuzhan/code/aiter-src")      # aiter is not installed at all

Verified in-container: `flydsl.__version__ == "0.3.2"`,
`flydsl.__file__ == "/home/lihuzhan/.local/flydsl032/flydsl/__init__.py"`,
`flydsl.expr.buffer_ops` **absent** (which is why `primus_turbo` must never be imported in
the same process), and `import aiter` succeeds off `/home/lihuzhan/code/aiter-src`.

**2. Do NOT set `TORCH_BLAS_PREFER_HIPBLASLT=1`.** hipBLASLt's gfx1250 Tensile library is
**missing from this image** (`TensileLibrary_lazy_gfx1250.dat`, `Kernels.so-000-gfx1250*.hsaco`).
Measured here, 4096^3 bf16 `torch.matmul`:

| `TORCH_BLAS_PREFER_HIPBLASLT` | Result |
| --- | --- |
| `1` (torch's own preference) | **RuntimeError**: `HIPBLAS_STATUS_INVALID_VALUE` from `hipblasLtMatmulAlgoGetHeuristic` -- the matmul does not run at all |
| `0` | runs, 5.039 ms, 27.3 TFLOP/s |
| unset | runs, 5.060 ms, 27.2 TFLOP/s (this is what `check_runner` got) |

So set it to `0` explicitly, as the baseline kernels already do.

### ⚠ The `torch gemm` row above is NOT a roof for this card

`27.2 TFLOP/s` is the **rocBLAS fallback** bf16 path, and it is broken-slow on this image.
The card's measured bf16 compute roof in this same degraded state is **1002.7 TFLOP/s**
(Triton, `output/0911__fa_gfx1250_phase1/`), and HBM is **6.46 TB/s**, ridge 155 FLOP/byte.
The check's number proves *a GPU GEMM executes*, which is all it claims. Never use it as a
ceiling, a reference or a baseline.

### Clock state -- every figure in this job needs its own witness

This card is **VR-throttled**, and the DPM table is the only tell (`THROTTLE_STATUS` reads
`N/A` on this part). Observed at setup, with the card idle:

| Item | Observed |
| --- | --- |
| supported `sclk` | `500Mhz`, `1100Mhz` -- a two-entry table, truncated. An unthrottled part of this type reports up to 2400 MHz. |
| current `sclk` | `1100Mhz` (also `1100Mhz` immediately after the 4096^3 GEMM) |
| `fclk` / `mclk` / `socclk` | `1100Mhz` / `1900Mhz` / `1350Mhz` |
| socket power | 851 W |
| junction temp | 43 C |

Absolute TFLOP/s on this box move ~1.65x with the throttle, and the clock has been seen to
drift 1100 -> 967 MHz *inside a single timing window*. This is why `op.target` is a purely
relative gate measured in the same run (`beat_measured_same_run: true`) and why
`op.target.tflops` / `roofline_pct` must stay `null`. Interleave arms ABAB and record a
clock witness beside every figure. Raw output: `../logs/step4_clock_witness.log`.

### Kernel log during these checks

A `dmesg` monitor was armed before the GPU work and the window was **clean** -- no fault, no
`MES` message, no reset, during `check_runner` or any check above.

Two things in the buffer are worth knowing and neither is a blocker:

- **A GPU fault ~2.9 h before setup** (kernel ts ~5044, uptime at setup 15406):
  `GCVM_L2_PROTECTION_FAULT` / `no-retry page fault`, `Faulty UTCL2 client ID: TCP`,
  `PERMISSION_FAULTS: 0x5`, from a `python3`, followed by
  `MES(0,0) failed to respond to msg=REMOVE_QUEUE` and `msg=SUSPEND`. This is the event the
  HINTS file already records (timing `dkdv_heads="kv"` at s=8192). It **recovered**: the
  card serves GEMMs and `rocprofv3` counters now. Treat it as the documented precedent that
  a known out-of-bounds path has no floor worth measuring, not as a live condition.
- **CPU-side corrected machine checks**, roughly every 20-30 min since boot (72 lines, 10
  events): `CPU:8`, `cache level: L3/GEN, mem/io: IO`, every one
  `Corrected error, no action required`. Pre-existing, unrelated to the GPU and to this job.
  Worth mentioning to whoever owns the box.

**Safety rules that apply for the whole run** (from this machine's own history):
never enable `rocprofv3` **PC sampling** -- it wedged MES to the point of needing a reboot on
2026-09-11, while plain ATT and PMC sweeps were clean; never call an autotuner on a training
path; launch every newly compiled kernel at a toy shape first, in its own process, with
`AMD_SERIALIZE_KERNEL=3`, because a bad TDM descriptor or a wave-size mismatch **hangs**
rather than raising.
