# Llama-3.1-8B BF16 pretraining on MI455X (gfx1250) with Primus + Primus-Turbo `dev/lhz/llama31_attn_opt`

Single MI455X, torchtitan backend of Primus, Llama-3.1-8B, BF16, MBS = GBS = 4, seq 8192, no TP/PP/CP/FSDP,
no activation checkpointing, `torch.compile` off, mock data. Measured on host A0 (heliosr-1b114-c07-1) on
2026-10-05.

## TL;DR

| what | result |
|---|---|
| 20 iterations, wall clock from `docker exec` to exit | **53 s** = ~8 s startup + ~10 s step 1 (FlyDSL JIT) + ~29 s steps 2-20 + ~6 s teardown |
| steady-state step time (median of steps 6-20) | **1,361.9 ms/step** |
| throughput | **24,061 tokens/s** (4 x 8,192 tokens per step) |
| peak memory | 382.9 GiB = 88.64 % of HBM |
| loss | 12.26 at step 1 -> 5.05 at step 20, all finite; bitwise identical across two runs |

Everything lives in this directory of the branch, `docs/gfx1250_llama31_8b_e2e/`:

| file | purpose |
|---|---|
| `run_l8b_e2e.sh` | host-side launcher: `./run_l8b_e2e.sh [steps=20] [tag]` |
| `l8b_mi455x.yaml.in` | Primus experiment config template (comments explain every non-default) |
| `nkfix/nkfix.py`, `nkfix/transpose_triton.py` | runtime workaround for the untuned backward GEMMs (section 3.2) |
| `primus-nkfix-hook.patch` | 14-line Primus patch that installs nkfix in the trainer process |
| `docker/` | `Dockerfile` + pinned requirement/constraint files to rebuild `fa-tune:deps` (section 4) |

The three things that matter most, in order: (1) the backward GEMM workaround (nkfix) -- without it a step
takes ~16 s instead of ~1.4 s; (2) the image's hipBLASLt library path; (3) a gfx1250 attention backend --
Primus-Turbo `main` has none that works by default; the branch adds FlyDSL kernels.

## 1. Your symptoms vs. what we hit

We ran **one GPU only**, so anything about RCCL is outside our data.

| your symptom | same as ours? | cause we found | what fixes it |
|---|---|---|---|
| backward many times slower than forward (compute) | **yes** | The image's hipBLASLt ships the tuned plain-bf16 library only for the forward's TN layout. Every Linear dgrad (`Cijk_Ailk_Bljk`) and wgrad (`Cijk_Ailk_Bjlk`) falls to an untuned table whose nearest entry is a GEMV tile, `MT32x16x32`: 50-80 TF/s vs 1.5-1.9 PF/s forward, 11-32x slower per call; backward GEMMs took ~50x the forward GEMMs per step (~16 s/step) | nkfix (section 3.2): re-lays out the backward operands so every GEMM hits the forward's tuned kernel. 10.3x faster step. Check your trace for `MT32x16x32` kernel names to confirm |
| backward slower than forward (RCCL) | **unknown** | We never ran multi-GPU. Note: with FSDP2 the backward moves ~3x the forward's bytes by design (re-all-gather + fp32 reduce-scatter). The GEMM problem above also makes ranks run at different speeds (hipBLASLt picks the NN fallback per process), which collectives then absorb as waiting | Fix the GEMMs first, then profile collectives per rank. The image ships a private RCCL (`/opt/rccl-gfx1250`, ref `rccl/gfx1250_merge`); launch through `runner/primus-cli` so Primus' `MI455X.sh` env applies |
| version incompatibilities, no e2e result | **yes** -- the stock image never completed a training step for us | (a) the stock image lacks torchtitan's deps (tyro, torchdata, tabulate, tensorboard, wandb). `primus-cli ... train pretrain` re-installs torchtitan with its declared deps on every launch when PyPI is reachable, so this bites with plain `torchrun` or offline: Primus' patch runner then *skips* the turbo-attention patch ("missing dependency: tensorboard") and torchtitan's own attention passes `enable_gqa` to `TurboAttention.forward` -> `TypeError` (the same TypeError appears whenever the converter is on but `use_turbo_attention` is false); (b) the image ships its own editable `primus-turbo 0.4.1.dev12` whose import hook **wins over `PYTHONPATH`**, so your checkout is silently ignored; (c) flydsl: the image and Primus-Turbo pin 0.2.4, aiter pins 0.3.2, and Primus-Turbo does not import under 0.3.x; (d) Primus-Turbo `main` has no working default gfx1250 attention backend (section 2) | section 4 (image) + section 5 (setup). Everything in this tutorial runs on the image's own torch / triton / flydsl 0.2.4 |
| page fault / GPU hang, root-caused to hipBLASLt | **yes, several variants** | (a) default library search path is one directory too high -> every matmul raises `HIPBLAS_STATUS_INVALID_VALUE`; the common workaround `TORCH_BLAS_PREFER_HIPBLASLT=0` (rocBLAS) is 8x slower and its fp32 Tensile kernels page-faulted the card; (b) a non-image hipBLASLt library (we tried a host-built copy) was live in a run that went NaN from step 4 and in another that wedged the GPU in step 1; (c) fp32 GEMMs on the card (e.g. test references) faulted in `Cijk_Ailk_Bljk_*` kernels (`GCVM_L2_PROTECTION_FAULT`) | Use only the image library and set it explicitly: `TORCH_BLAS_PREFER_HIPBLASLT=1 HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250`. Never the `_rocm_sdk_devel/.../gfx1250` copy (incomplete, SIGSEGV). No fp32 GEMMs on the card. With the image library: 12 e2e launches since 2026-09-28 (8 on a 4-GPU box, 4 here) without a GPU fault -- too few to prove low risk; one earlier wedge on the image library, on this card before its 2026-09-29 firmware update, is unexplained |

Other hangs we hit that look GPU-side: `converters: []` (torchtitan flex attention -> inductor autotune -> MES
"failed to respond" wedge), `compile.enable: true` (inductor autotune -> `hipErrorLaunchFailure`), plain
`8B` flavor with the converter off (SDPA MATH materialises [4,32,8192,8192] -> SIGBUS), and starting a run while another process
still holds the GPU. A wedged MI455X needs an AC power cycle; never `modprobe -r amdgpu`.

## 2. What goes wrong with Primus-Turbo `main` (and what the branch changes)

Branch `dev/lhz/llama31_attn_opt` = `main` @ `8cda13c7` + attention-only commits. Items 2.1 is fixed by the
branch; 2.2-2.5 apply to **both** and are handled by this tutorial's setup.

1. **Attention (fixed by the branch).** On gfx1250, `main`'s default `FLYDSL` backend names gfx950 kernels.
   For Llama BSHD with batch 4 dispatch falls through to AITER/CK: `ImportError` (no aiter in the image) or,
   with aiter, CK's backward rejects the call mid-step. With batch 1 or SBHD the gfx950 gate (`cc >= (9,5)`)
   also accepts gfx1250 and raises `requires gfx950+ (uses ds_read_tr16_b64)`. The only working path on `main`
   is `PRIMUS_TURBO_ATTN_BACKEND=TRITON`, ~4x slower forward and ~3.4x slower backward than the branch's
   FlyDSL kernels (36-shape geomean). Estimated e2e on `main` + Triton: ~1.9-2.05 s/step (~16-17k
   tokens/s; not measured on `main`) vs 1.36 s/step on the branch. The branch's FlyDSL backend covers bf16, head_dim 128, Hq/Hkv in
   {1,2,4,8,16}, Sq % 64 == 0, Skv % 32 == 0; other shapes still fall to AITER/CK -- pin TRITON for those models.
   Unset any leftover `PRIMUS_TURBO_ATTN_BACKEND` on the branch: a pin beats the default.
2. **Backward GEMMs (both).** See section 1, row 1. Not fixed in any branch; nkfix is a runtime workaround.
3. **hipBLASLt library/env (both).** See section 1, row 4. Also: a `bash -lc` login shell can read a
   profile script that forces `TORCH_BLAS_PREFER_HIPBLASLT=0` (we had one in our container); use `bash -c` and
   set both variables explicitly, then check `torch.backends.cuda.preferred_blas_library()` in-process.
4. **The image's own primus-turbo shadows yours (both).** `pip uninstall -y primus_turbo` in the container,
   and do not use Primus' `REBUILD_PRIMUS_TURBO=1` hook (it builds Primus-Turbo `main` for gfx942;gfx950 and
   pulls `triton>=3.7.0`, which does not support gfx1250, over the image's triton).
5. **Config (both).** `flavor: 8B_flex` + `converters: ["primus_turbo"]` + `enable_primus_turbo: true` +
   `use_turbo_attention: true`; `compile.enable: false`; `debug.seed` set (torchtitan does not seed a
   1-GPU run); keep `use_turbo_float8_linear` / `use_turbo_mx_linear` / `use_turbo_async_tp` false for BF16
   (Primus defaults them to true; never exercised on gfx1250). The run peaks at ~89 % HBM: do not raise MBS
   or seq without activation checkpointing.

## 3. What the branch and the workaround do

### 3.1 FlyDSL attention (branch `dev/lhz/llama31_attn_opt`)
`primus_turbo/flydsl/attention/gfx1250/`: forward (adapted from aiter's gfx1250 FlyDSL prefill kernel) and
an original backward (in-kernel GQA reduction, deterministic, split-K for small grids), ported to the
image's flydsl 0.2.4. `flash_attn_func` / `TurboAttention` dispatch to it automatically on gfx1250 -- no env
var needed. Op level at b4 s8192 hq32 hkv8 d128 causal: forward 1.29 ms, backward 6.62 ms (aiter's prebuilt
ASM: 1.19 / 5.36 ms). 49-53 dB SNR vs an fp32 reference.

### 3.2 nkfix (backward GEMM layout workaround)
`nkfix.py` installs a `TorchDispatchMode` on `aten::mm` that rewrites the backward GEMMs into the
forward's physical layout (dgrad: B -> N-major; wgrad: A contiguous + B N-major), with bounded temporary
copies (<= 256 MiB, `NKFIX_CHUNK_BYTES`) and a Triton transpose (`transpose_triton.py`; **keep it next to
`nkfix.py`** -- without it nkfix falls back to torch copies and the step goes from 1,362 to 1,689 ms).
It is installed by `primus-nkfix-hook.patch` when `NKFIX_ENABLE=1`. Overhead ~70 ms/step; optional
non-finite checking `NKFIX_CHECK=1` adds ~33 ms/step.

## 4. Docker image: `fa-tune:deps` from `amdprimus/amdprimus:gfx1250-20260910`

`fa-tune:deps` is the stock image plus one `docker commit` layer (2026-09-13, 656 MB) that only adds
torchtitan's dependencies; there was never a Dockerfile. `docker/Dockerfile` rebuilds an equivalent image
from the pinned base, with every added package pinned to the A0 freeze (`docker/*.txt`), and removes the
stock image's own primus-turbo:

```bash
cd docker && docker build -t fa-tune:deps -f Dockerfile .
```
The base `amdprimus/amdprimus:gfx1250-20260910` (digest `sha256:6e656de7...`) is private on Docker Hub:
`docker login`, or `docker save` / `docker load` it from a host that has it.

Container (one GPU here; add your devices as usual):
```bash
docker run -d --name fa-repro --network host --ipc host --shm-size 16g \
  --device /dev/kfd --device /dev/dri --group-add video --group-add $(getent group render | cut -d: -f3) \
  --cap-add CAP_SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
  -v $HOME:$HOME fa-tune:deps sleep infinity
```
`$HOME` is mounted at the same path, so host checkouts are used in place.

What the image already has and we rely on: Python 3.12 venv `/opt/venv`; ROCm 7.14.0a20260625 as pip wheels
(no `/opt/rocm`); torch 2.11.0+rocm7.14.0a20260625; triton 3.6.0+rocm7.14.0a20260625 (keep it);
flydsl 0.2.4; tuned hipBLASLt (`_rocm_sdk_libraries_gfx1250`, 2026-09-09); RCCL 2.30.4 at
`/opt/rccl-gfx1250`.

## 5. Setup (host checkouts, used inside the container)

```bash
# Primus (we use main @ e7968675, 2026-09-09) + the nkfix hook
git clone https://github.com/AMD-AGI/Primus.git && cd Primus && git checkout e7968675
git submodule update --init third_party/torchtitan          # torchtitan v0.2.2 @ 73a0e697
git apply /path/to/Primus-Turbo/docs/gfx1250_llama31_8b_e2e/primus-nkfix-hook.patch

# Primus-Turbo, this branch (gfx1250 FlyDSL attention + this tutorial); build the C++ extension in the container
git clone -b dev/lhz/llama31_attn_opt https://github.com/AMD-AGI/Primus-Turbo.git && cd Primus-Turbo
git submodule update --init 3rdparty/hipify_torch
docker exec -w $PWD fa-repro bash -c 'git config --global --add safe.directory "*"; \
  GPU_ARCHS=gfx1250 PRIMUS_TURBO_BUILD_CK=0 MAX_JOBS=64 python3 setup.py build_ext --inplace'   # ~2 min
# do not pip install it; the launcher puts the checkout on PYTHONPATH

# Llama-3.1-8B tokenizer + config (gated repo; mock data needs only these files)
#   tokenizer.json tokenizer_config.json special_tokens_map.json config.json generation_config.json
#   e.g. torchtitan scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer config --hf_token ...
```
`primus-cli direct ... train pretrain` re-installs `third_party/torchtitan` editable on every launch (its
torchtitan hook); that is expected.

## 6. Run

```bash
# from the Primus-Turbo checkout of this branch (TURBO defaults to it)
export PRIMUS=/path/to/Primus HF_ASSETS=/path/to/llama31_8B WORKSPACE=/path/to/runs CT=fa-repro
docs/gfx1250_llama31_8b_e2e/run_l8b_e2e.sh 20 my_first_run
```
The launcher (read it, it is short) refuses to start if another process holds the GPU, then runs in the
container, via `bash -c`:
```bash
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIPBLASLT_TENSILE_LIBPATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
export NKFIX_ENABLE=1 NKFIX_CHECK=0          # NKFIX_CHECK=1: non-finite check, +~33 ms/step
export FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl_cache_<tag>
export PYTHONPATH=$TURBO/docs/gfx1250_llama31_8b_e2e/nkfix:$TURBO
cd $PRIMUS && bash runner/primus-cli direct --log_file ... -- train pretrain --config <generated yaml>
```
and prints a summary (median steady-state tokens/s and ms/step, peak memory, loss finiteness, wall time).

What to check in the log:
- `primus_turbo <path>` printed at the top must be your branch checkout (and `flydsl 0.2.4`).
- `[nkfix] installed` and **no** `[nkfix] triton transpose unavailable` (that means `transpose_triton.py` is missing).
- `Primus-Turbo Attention successfully installed for LLaMA3, ...` (the turbo_attention patch was applied; if it
  was skipped for a missing dependency, the run dies with an `enable_gqa` TypeError).
- the first start on a fresh host can sit minutes before step 1 (torchtitan compiles its block-mask builder for
  `8B_flex`); keep `TRITON_CACHE_DIR` persistent. With a warm cache step 1 comes ~17 s after launch.
- steps 1-2 are slow (FlyDSL JIT on first use, allocator growth); from step 3 on torchtitan's `tps` should
  read ~24k. A steady ~19.4k means nkfix is running without `transpose_triton.py`; ~2k means nkfix is not
  installed at all.

## 7. Expected results

Our run `l8b_branch_1005b` (2026-10-05, A0, sclk 2.36 GHz max, driver amdgpu 7.1.0-2412954):

```
step:  1  loss: 12.25957  grad_norm: 42.3893  memory: 313.66GiB(72.61%)  tps: 3,214
step:  2  loss: 26.02536  grad_norm: 69.7816  memory: 382.94GiB(88.64%)  tps: 15,580
step:  3  loss: 19.83216  grad_norm: 91.1363  memory: 382.94GiB(88.64%)  tps: 23,990
step: 10  loss:  9.32729  grad_norm: 42.6789  memory: 382.94GiB(88.64%)  tps: 24,001
step: 20  loss:  5.05053  grad_norm:  8.0958  memory: 382.94GiB(88.64%)  tps: 24,070
steady state (steps 6-20): median 24,061 tokens/s = 1361.9 ms/step
rc=0 wall=53 s (container start to exit, incl. model init and JIT)
```

| configuration (same machine, same config) | ms/step | tokens/s | source |
|---|--:|--:|---|
| **this tutorial: branch FlyDSL attention + nkfix (`NKFIX_CHECK=0`)** | **1,362** | **24,061** | measured 2026-10-05 |
| same, re-run from `docs/gfx1250_llama31_8b_e2e/` exactly as committed | 1,360 | 24,099 | measured 2026-10-05 |
| same, but `transpose_triton.py` missing (nkfix torch copies) | 1,689 | 19,406 | measured 2026-10-05 |
| aiter prebuilt ASM attention + nkfix (`NKFIX_CHECK=1`, our A/B kit) | 1,351 | 24,262 | measured 2026-10-02 |
| FlyDSL fwd r16 + newer bwd "s6" (`NKFIX_CHECK=1`, not yet in the branch) | 1,349 | 24,283 | measured 2026-10-02 |
| `main` + `PRIMUS_TURBO_ATTN_BACKEND=TRITON` + nkfix | ~1,900-2,050 | ~16,000-17,000 | estimate (op-level and in-training attention times); not measured on `main` |
| no nkfix (any attention) | ~16,000 | ~2,000 | measured 2026-09-28 (B0) |
| `TORCH_BLAS_PREFER_HIPBLASLT=0` (rocBLAS), no nkfix | ~134,000 | ~245 | measured 2026-09-13 (A0, 1.1 GHz clock cap) |

Absolute numbers depend on the platform state: A0 ran at a 1.1 GHz cap until a VBIOS/SMU update on
2026-09-29, after which these numbers were taken. Check `cat /sys/class/drm/card*/device/pp_dpm_sclk`.
torchtitan's `mfu` column is meaningless on this part (it falls back to the A100's peak FLOPs); use tokens/s.

## 8. Safety notes (MI455X)

- One GPU client at a time per card; start only when `ls /sys/class/kfd/kfd/proc` is empty. Use a fresh
  `MASTER_PORT` per run (an orphaned `pt_elastic` can hold the old one).
- Wedge risk is per startup: prefer one longer run over many short ones.
- On a hang: `timeout 20 sudo dmesg | tail` is safe; `rocm-smi`, `docker stop` and `torch.cuda.device_count()`
  can hang. `GPU reset begin` / `wait for reset ack` / `MES ... unrecoverable` = needs an AC power cycle. Send one
  SIGTERM by PID; never SIGKILL mid-kernel; never `modprobe -r amdgpu`.
- No in-process autotune (`PRIMUS_TURBO_AUTO_TUNE`, inductor max-autotune) and no fp32 GEMMs on the card.
- If a loss or grad_norm turns NaN/inf, rerun with `NKFIX_CHECK=1` and discard that run's timings.
