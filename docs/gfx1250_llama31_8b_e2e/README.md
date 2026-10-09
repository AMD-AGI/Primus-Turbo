# Llama-3.1-8B BF16 pretraining on one MI455X (gfx1250)

This directory reproduces Llama-3.1-8B BF16 pretraining on a single AMD Instinct MI455X
(gfx1250) with [Primus](https://github.com/AMD-AGI/Primus) (TorchTitan backend) and
Primus-Turbo. The experiment is Primus' MI455X recipe,
`examples/torchtitan/configs/MI455X/llama3.1_8B-BF16-pretrain.yaml`: local batch = global
batch = 4, sequence length 8192 (32,768 tokens per step), no tensor, pipeline or context
parallelism, no FSDP, no activation checkpointing, `torch.compile` off, mock data. This
directory adds the container image (`docker/`) and a launcher, `run_llama31_8b_e2e.sh`, that checks the
setup, runs the recipe and summarizes the log.

## Requirements

- **Primus with the MI455X Llama-3.1-8B recipe.** Besides the recipe, it provides the
  `primus_turbo.enable_mm_layout_workaround` switch that the recipe turns on, and MI455X
  settings (`runner/helpers/envs/MI455X.sh`) that point hipBLASLt at the image's gfx1250
  library.
- **Primus-Turbo on flydsl 0.3.4.1**, i.e. this checkout, with its C++ extension built in the
  container (section 3). The checkout's `setup.py` pins the flydsl release, and the launcher
  checks that the container has that release.
- **The image built from `docker/`**: the gfx1250 Primus base image plus flydsl 0.3.4.1 and
  TorchTitan's dependencies.
- **One MI455X**, and access to the gated `meta-llama/Llama-3.1-8B` repository on Hugging Face
  for the tokenizer.

On gfx1250 the run depends on three pieces that a stock setup lacks:

1. **Attention kernels.** Primus-Turbo's FlyDSL flash attention for gfx1250
   (`primus_turbo/flydsl/attention/gfx1250/`), which `flash_attn_func` and `TurboAttention`
   select by themselves on gfx1250 (section 6.1).
2. **Fast backward GEMMs.** `primus_turbo.pytorch.core.mm_layout_workaround` (section 6.2),
   which the recipe turns on with `primus_turbo.enable_mm_layout_workaround: true`. Without it
   a training step takes more than ten times as long.
3. **The hipBLASLt library path.** `HIPBLASLT_TENSILE_LIBPATH` must point at the image's
   gfx1250 Tensile library, with `TORCH_BLAS_PREFER_HIPBLASLT=1`. Primus' `MI455X.sh` sets
   both, and so does `run_llama31_8b_e2e.sh`.

## Files

| file | purpose |
|---|---|
| `run_llama31_8b_e2e.sh` | host-side launcher: checks the setup, runs the recipe in the container, prints a summary |
| `docker/Dockerfile`, `docker/requirements.txt`, `docker/constraints.txt` | the container image (section 2) |

The experiment config is the Primus recipe alone; its comments explain every non-default
setting.

## 1. What goes wrong without these pieces

Observed on MI455X with the base image of section 2, on one GPU (multi-GPU training is not
covered here).

| area | what happens | handled by |
|---|---|---|
| attention | Without the gfx1250 kernels, Llama's attention falls back to AITER, i.e. CK: `ImportError` without aiter (the image has none); with aiter, CK's fmha backward rejects the call (`invalid argument for fmha_bwd`) after the forward has run. The only working alternative is `PRIMUS_TURBO_ATTN_BACKEND=TRITON`, which is several times slower than the FlyDSL kernels. | the gfx1250 FlyDSL kernels, the default on gfx1250 |
| flydsl version | The base image has flydsl 0.2.4. Under a release outside the kernels' requirement (`FLYDSL_REQUIREMENT` in `primus_turbo/flydsl/attention/gfx1250/flydsl_version.py`) the gfx1250 kernels decline, Primus-Turbo logs `fallback backend AITER is selected`, and the run fails as above. | the image installs flydsl 0.3.4.1; the launcher checks it against that requirement and against the checkout's pin |
| backward GEMMs | The image's hipBLASLt has tuned bf16 kernels only for the layout of a Linear's forward GEMM (`Cijk_Alik_Bljk`). The input-gradient (`Cijk_Ailk_Bljk`) and weight-gradient (`Cijk_Ailk_Bjlk`) GEMMs fall back to a small untuned tile (`MT32x16x32` in the kernel name) that reaches a few percent of the forward GEMM's throughput, and a training step takes more than ten times as long. | `mm_layout_workaround` (section 6.2), switched on by the recipe |
| hipBLASLt library | The ROCm wheels keep the gfx1250 Tensile library one directory below where hipBLASLt looks, so without `HIPBLASLT_TENSILE_LIBPATH` every hipBLASLt GEMM fails with `HIPBLAS_STATUS_INVALID_VALUE`. Falling back to rocBLAS (`TORCH_BLAS_PREFER_HIPBLASLT=0`) avoids the error but is much slower. The copy of the library under `_rocm_sdk_devel` is incomplete; use the one under `_rocm_sdk_libraries_gfx1250`. | Primus' `MI455X.sh` and the launcher set both variables for the image's library |
| the base image's Primus-Turbo | The base image ships an editable Primus-Turbo without the gfx1250 kernels. Primus' launcher puts site-packages ahead of `PYTHONPATH`, so that copy silently replaces your checkout. | the Dockerfile removes it; the launcher checks |
| TorchTitan dependencies | The base image lacks torchdata, tyro, tensorboard and tomli. If they are not installed when training starts, Primus skips its turbo-attention patch ("missing dependency"), and torchtitan's own attention then calls `TurboAttention` with `enable_gqa`: `TypeError`. | the Dockerfile installs them |

Settings that fail on gfx1250 (the recipe avoids all of them; its comments have the details):

- `converters: []`: torchtitan then uses flex attention, which always goes through
  `torch.compile`, and Inductor's autotuning fails on gfx1250 with this image.
- `compile.enable: true`: Inductor's Triton autotuning fails (`hipErrorLaunchFailure`), and
  GEMMs in compiled code would bypass the GEMM workaround.
- flavor `8B` without the converter: SDPA falls back to its math backend, which materializes a
  [4, 32, 8192, 8192] score tensor in every layer; they do not fit in memory.
- the converter without `use_turbo_attention: true`: the `enable_gqa` TypeError above.
- `use_turbo_float8_linear`, `use_turbo_mx_linear` and `use_turbo_async_tp` default to true in
  Primus and are not validated on gfx1250; this BF16 recipe turns them off.
- no `debug.seed`: torchtitan does not seed a single-GPU run.

Two more traps: Primus' `REBUILD_PRIMUS_TURBO=1` hook rebuilds Primus-Turbo main for
gfx942;gfx950 unless `GPU_ARCHS` is set, and that build installs `triton>=3.7.0`, which does not
support gfx1250, over the image's Triton; and a login shell (`bash -lc`) may source profile
scripts that override `TORCH_BLAS_PREFER_HIPBLASLT`. When in doubt, check
`torch.backends.cuda.preferred_blas_library()` in the training process.

## 2. Docker image

The commands from here on run on the host, in a work directory that will hold the Primus and
Primus-Turbo checkouts, the tokenizer files and the training logs. The container mounts that
directory at the same path.

```bash
docker build -t llama31-8b-mi455x Primus-Turbo/docs/gfx1250_llama31_8b_e2e/docker
```

The base image, `amdprimus/amdprimus:gfx1250-20260910`, is pinned by digest; pulling it needs
access to that repository. On top of it the image installs TorchTitan's missing dependencies and
flydsl 0.3.4.1 in place of 0.2.4 (the PyPI wheel has no dependencies, and nothing in the image
depends on flydsl), and removes the base image's own Primus-Turbo. `constraints.txt` pins every
package this can install or replace, i.e. the dependency closure of `requirements.txt`, torch,
triton and the ROCm wheels included, so the build is reproducible and keeps the base image's
torch, triton and ROCm as they are. The build fails unless flydsl is 0.3.4.1 and no
`primus_turbo` is left installed.

What the base provides and this setup relies on: a Python 3.12 venv in `/opt/venv`; ROCm
7.14.0a20260625 as pip wheels (there is no `/opt/rocm`); torch 2.11.0+rocm7.14.0a20260625; triton
3.6.0+rocm7.14.0a20260625 (keep it: Triton 3.7.0 does not support gfx1250); hipBLASLt with a tuned
gfx1250 library (`_rocm_sdk_libraries_gfx1250`).

Start a container:

```bash
docker run -d --name llama31-8b-mi455x --network host --ipc host --shm-size 16g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$PWD:$PWD" llama31-8b-mi455x
```

The work directory is mounted at the same path because the launcher hands host paths to the
container; it checks that the container sees them.
Training uses the container's first GPU (Primus sets `HIP_VISIBLE_DEVICES` itself); on a host
with several GPUs, pass only the one you want, e.g. `--device /dev/dri/renderD128` instead of
`--device /dev/dri`.

## 3. Setup

The checkouts live on the host and are used in place inside the container.

```bash
# Primus with the MI455X Llama-3.1-8B recipe, and its TorchTitan submodule (v0.2.2).
git clone https://github.com/AMD-AGI/Primus.git
git -C Primus submodule update --init third_party/torchtitan

# Primus-Turbo: this checkout (its setup.py must pin flydsl==0.3.4.1). Build its C++ extension
# inside the container, which takes a few minutes. Do not pip install it: the launcher puts the
# checkout on PYTHONPATH.
git -C Primus-Turbo submodule update --init --recursive
docker exec -w "$PWD/Primus-Turbo" llama31-8b-mi455x bash -c \
  'GPU_ARCHS=gfx1250 PRIMUS_TURBO_BUILD_CK=0 MAX_JOBS=64 python3 setup.py build_ext --inplace'

# Llama-3.1-8B tokenizer and config files; mock data needs nothing else. The repository is
# gated: accept its license on Hugging Face first. The files land in $PWD/hf/Llama-3.1-8B.
docker exec -w "$PWD/Primus" llama31-8b-mi455x python3 third_party/torchtitan/scripts/download_hf_assets.py \
  --repo_id meta-llama/Llama-3.1-8B --assets tokenizer config --hf_token "$HF_TOKEN" --local_dir "$PWD/hf"
```

`primus-cli direct ... train pretrain` reinstalls `third_party/torchtitan` in editable mode at
every launch; that is expected.

## 4. Run

```bash
export PRIMUS=$PWD/Primus HF_ASSETS=$PWD/hf/Llama-3.1-8B
Primus-Turbo/docs/gfx1250_llama31_8b_e2e/run_llama31_8b_e2e.sh 20 first_run     # [steps] [tag]
```

Optional settings (see the script header): `TURBO` (default: the checkout the script is in),
`CT` (container, default `llama31-8b-mi455x`), `WORKSPACE` (logs and Primus' output, default
`llama31_8b_runs` next to the Primus-Turbo checkout), `MM_LAYOUT_WORKAROUND=0` (run without the
GEMM workaround),
`SKIP_GPU_IDLE_CHECK=1`.

The launcher

1. checks, without touching the GPU, that the Primus checkout has the recipe and the TorchTitan
   submodule; that the Primus-Turbo checkout has the gfx1250 attention, the workaround and a
   built C++ extension; that `HF_ASSETS` holds the tokenizer; and that the container has a
   flydsl that meets the kernels' requirement and matches the checkout's pin, no installed
   Primus-Turbo, and the image's hipBLASLt gfx1250 library. It refuses to start while another
   process holds a GPU (`/sys/class/kfd/kfd/proc` is not empty);
2. runs the recipe unchanged, inside the container and through `bash -c`:

   ```bash
   export TORCH_BLAS_PREFER_HIPBLASLT=1
   export HIPBLASLT_TENSILE_LIBPATH=<site-packages>/_rocm_sdk_libraries_gfx1250/lib/hipblaslt/library/gfx1250
   export PRIMUS_HF_ASSETS_PATH=$HF_ASSETS PRIMUS_WORKSPACE=$WORKSPACE PRIMUS_EXP_NAME=<tag>
   export FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl_cache_<tag>   # fresh per run
   export TRITON_CACHE_DIR=/tmp/triton_cache_llama31_8b             # kept between runs
   export PYTHONPATH=$TURBO
   cd $PRIMUS && bash runner/primus-cli direct --log_file ... -- train pretrain \
     --config examples/torchtitan/configs/MI455X/llama3.1_8B-BF16-pretrain.yaml \
     --training.steps <steps>
   ```

   The recipe reads `PRIMUS_HF_ASSETS_PATH`, `PRIMUS_WORKSPACE` and `PRIMUS_EXP_NAME` from the
   environment; `--training.steps` is a Primus config override, and `MM_LAYOUT_WORKAROUND=0`
   adds `--primus_turbo.enable_mm_layout_workaround false`;
3. prints a summary: steps logged, loss, peak memory, the median steady-state tokens/s and
   ms/step (steps 6 onwards), whether loss and grad_norm stayed finite, and the log checks of
   section 5. It exits with the training's status, or 2 when training succeeded but a log check
   failed.

The training log is `$WORKSPACE/<tag>.log`.

## 5. What to check in the log

The launcher checks the first four lines below by itself.

- `Primus-Turbo Attention successfully installed for LLaMA3, ...`: Primus' attention patch was
  applied (when it is skipped for a missing dependency, the run dies with the `enable_gqa`
  TypeError).
- `[Patch:torchtitan.primus_turbo.mm_layout_workaround] Primus-Turbo aten::mm layout workaround installed on gfx1250.`
- no `mm layout workaround: Triton is unavailable`: without Triton the operand transposes fall
  back to slower torch copies.
- no `fallback backend ... is selected`: a Primus-Turbo op left its default backend. For
  attention it means the gfx1250 kernels declined: check the flydsl version and the shape.
- `MI455X: HIPBLASLT_TENSILE_LIBPATH=...`, printed by Primus' `MI455X.sh`.
- `[Primus:Runtime] Applying CLI overrides: ...`: the overrides the launcher passed, plus the
  tokenizer path Primus resolved.
- Steps 1-2 are slow (FlyDSL JIT on first use, allocator growth); from step 3 on torchtitan's
  `tps` settles. A run without the GEMM workaround (`MM_LAYOUT_WORKAROUND=0`) shows what it is
  worth: steps more than ten times as long.
- With the recipe's `8B_flex` flavor, the first launch in a fresh container can sit for minutes
  before step 1 while torchtitan compiles its block-mask builder; the Triton cache keeps that
  for later runs in the same container.
- torchtitan's `mfu` column assumes an A100's peak FLOP/s; ignore it on MI455X.

## 6. What the pieces do

### 6.1 FlyDSL attention for gfx1250

The forward is adapted from aiter's gfx1250 FlyDSL prefill kernel. The backward was written for
Primus-Turbo: GQA is reduced inside the kernel (one workgroup owns a key/value tile and
accumulates the gradients of every query head that shares it), every output element is written
once without atomics, so the backward is deterministic, and split-K variants keep small grids
busy. Its main loops are fed from an LDS ring that the Tensor Data Mover fills two iterations
ahead, and the dQ kernels run on a side stream, concurrently with the dK/dV kernels.

The kernels take bf16, head_dim 128, Hq/Hkv in {1, 2, 4, 8, 16}, Sq a multiple of 64 and Skv a
multiple of 32 (neither zero), bottom-right causal (which needs Sq <= Skv) or full attention,
and q and k of at most 1 GiB each; any memory layout (inputs are made contiguous BSHD) and any
softmax scale. Dropout, bias, ALiBi, sink, sliding window and returning the softmax are not
supported. They need flydsl>=0.3.4.1,<0.3.5 (`FLYDSL_REQUIREMENT` in `flydsl_version.py`;
`interface.py` has the exact shape rules). Other calls go to the other backends, and on gfx1250 the first of
them is AITER/CK, whose backward fails: pin `PRIMUS_TURBO_ATTN_BACKEND=TRITON` for models
outside this range. A pinned backend overrides the default, so do not leave one set for Llama.

`tests/pytorch/ops/test_attention_flydsl_gfx1250.py` checks outputs and gradients against an
fp32 reference computed on the CPU; `PRIMUS_TURBO_TEST_LARGE=1` adds the Llama-3.1-8B training
shape and, with `--deterministic-only`, 500 repeated forwards at sequence length 8192 that must
agree bit for bit.

### 6.2 Backward GEMM layout workaround

`primus_turbo.pytorch.core.mm_layout_workaround` installs a `TorchDispatchMode` on `aten.mm` that
hands hipBLASLt the same logical operands in the forward GEMM's physical layout: the input
gradient becomes `mm(A, Bn)` and the weight gradient `mm(A.contiguous(), Bn)`, with `Bn` a copy of
B in N-major order. All other calls, the forward GEMM included, are left alone, so the backward
GEMMs run on the tuned forward-layout kernels. The copies use a tiled Triton transpose; GEMMs
whose copies would exceed `chunk_bytes` (default 256 MiB) are split along their output columns
and rows, never along K, with scratch buffers kept per (device, stream, dtype). The module
docstring has the details.

It is opt-in and thread-local: it covers the GEMMs issued on the thread that enabled it,
including the backward passes that thread runs, and only eager ones (GEMMs that Inductor
generates are not rewritten). While enabled, every aten op on that thread takes a Python round
trip through the mode, a few microseconds of host time per op. Primus enables it with
`primus_turbo.enable_mm_layout_workaround: true`, on gfx1250 only, right before the training
loop. Another training script can do the same:

```python
from primus_turbo.pytorch.core.mm_layout_workaround import enable_mm_layout_workaround

enable_mm_layout_workaround()  # once, on the thread that runs the training loop, after building the model
```

The workaround should go once hipBLASLt ships tuned kernels for these layouts. To see whether a
stack still needs it, look for `MT32x16x32` in the GEMM kernel names of a profiled backward pass.

## 7. Not validated on gfx1250 with this image

fp32 GEMMs on the GPU (the tests compute their fp32 references on the CPU), and autotuning
(`PRIMUS_TURBO_AUTO_TUNE`, Inductor max-autotune). The recipe and the tests avoid both.
