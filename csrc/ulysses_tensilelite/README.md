# TENSILELITE backend of the Ulysses fused GEMM + all-to-all

The fused hops of [attn-a2a-demo](https://github.com/DDEle/attn-a2a-demo) as a backend of `ulysses_qkv_gemm_a2a` /
`ulysses_a2a_o_gemm`: TensileLite GemmA2A / A2AGemm kernels that move the data with SDMA, gfx950 only, bf16, forward
and backward (the dgrads on the fused hops, the wgrads on hipBLASLt). The Python side is `TensileLiteUlyssesA2ABackend`
in `primus_turbo/pytorch/kernels/attention/ulysses_a2a_impl.py`.

| path | what |
|---|---|
| `src/` | the extension `primus_turbo.pytorch._ulysses_tl`: SDMA queues (KFD), IPC exchange, fused-hop launch, boundary barrier (adapted from the demo's `csrc/`, see the file headers; `src/3rdparty/` is verbatim) |
| `logic/<hop>/` | the tuned TensileLite logic of each hop, one file per hop |
| `tuning/<hop>.yaml` | the tuning config of each hop (its search space and the shipped shapes) |
| `kernels.py` | generates the kernel libraries from `logic/` and checks them against the pins |
| `manifest.json` | the rocm-libraries commit, the compiler and the sha256 of every generated file |

The four hops: `fwd_gemm_a2a` (forward QKV, TN), `fwd_a2a_gemm` (forward o_proj, TN), `bwd_gemm_a2a` (o_proj dgrad,
NN: the weight read as stored), `bwd_a2a_gemm` (QKV dgrad, NN).

## Sources

| what | from | changes here |
|---|---|---|
| `src/` (except `src/3rdparty/`) | the native code (`csrc/`) of [DDEle/attn-a2a-demo](https://github.com/DDEle/attn-a2a-demo) at `7a7343ef47`, MIT, Copyright Advanced Micro Devices, Inc. | Turbo's license header; the module is named by `TORCH_EXTENSION_NAME`; `FusedHop.for_selection` takes `trans_a` (`false` selects the NN layout of the backward dgrads); includes `3rdparty/` instead of `third_party/`; reformatted by Turbo's clang-format. Otherwise the demo's code, token for token. |
| `src/3rdparty/` | the demo's `csrc/third_party/`: `SdmaQueue.hpp` from `projects/hipblaslt/tensilelite/client/include/` of [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) at `15ff8cb10ed`, the barrier kernel's origin (see `src/3rdparty/README.md`) | reformatted by clang-format only |
| `_TLComm` / `_TLHop` in `ulysses_a2a_impl.py` | the demo's hop driver, `attn_demo/hops.py` at `7a7343ef47` | split into kernel selection, buffer allocation and peer mapping, so that the ranks can agree after each step |
| TensileLite (host library, kernels, tools) | [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) at `511cc84bb2f`, an earlier revision of [rocm-libraries PR #14168](https://github.com/ROCm/rocm-libraries/pull/14168) (the commit the demo builds at `7a7343ef47`) | built by `tools/build_tensilelite.sh`, not vendored |
| `tuning/` | the search spaces of the demo's tuning configs (`tuning/` at `7a7343ef47`); the backward adds `DirectToLds: 0` | the problem sizes of the shapes shipped here |
| `logic/` | tuned on MI355X: the two world-size-8 entries of `fwd_a2a_gemm` are the demo's own logic; every other entry was tuned for these shapes with the configs in `tuning/` | merged per hop with `TensileMergeLibrary` |

## Use

`GlobalBackendManager.set_ulysses_a2a_backend(BackendType.TENSILELITE)`, or `PRIMUS_TURBO_ULYSSES_A2A_BACKEND=TENSILELITE`
when the code sets nothing. Every rank of the Ulysses group must request the same backend. The forward of a call picks
the backend for its forward and backward. A call TENSILELITE cannot take runs on FLYDSL, with one warning per reason:

1. the extension is built (see Build);
2. the GPU is gfx950 and the tensors are bf16 (no FP8), and the call is not inside CUDA graph capture;
3. the generated library of every hop the call needs (forward, and backward when an input requires grad) has its pinned
   files, and has a tuned kernel for the shape;
4. the SDMA queues to every peer, the buffers and the mapping of the peers' buffers can be set up.

The ranks agree on 3 and 4 (one `all_gather_object` after each step, the first time a shape is seen) before any of them
launches a kernel; the decision is then cached. Once TENSILELITE runs, an error raises.

## Build

The TensileLite toolkit is built outside `setup.py`, at the commit `manifest.json` pins, into one prefix:

```
tools/build_tensilelite.sh [PREFIX]   # default $TENSILELITE_HOME, else /opt/tensilelite; about 5 minutes, 280 MB
```

It needs network access and, for its system packages, root (`SKIP_APT=1` skips them). Then `pip install` builds the
backend when `GPU_ARCHS` contains gfx950 and the toolkit's `STAMP.json` is the manifest's commit, arch and compiler:
the extension, a copy of the toolkit's `libtensilelite-host.so.1` in `primus_turbo/lib/`, and the kernel libraries in
`primus_turbo/lib/ulysses_tl/<hop>/` with a copy of the manifest. `TENSILELITE_HOME` points at the toolkit;
`PRIMUS_TURBO_BUILD_ULYSSES_TL=0` skips the backend. A missing or mismatched toolkit, or an extension that fails to
compile, skips the backend with a warning and never fails the install.

By hand:

```
python csrc/ulysses_tensilelite/kernels.py [--toolkit DIR] [--check] [--force] [--update-pins] [--strict]
```

**Fail-closed.** `TensileCreateLibrary` output is byte-identical across runs and directories. A library is installed
only when all of its files have the pinned sha256, one that does not is removed, and another compiler generates nothing.
At run time a hop whose files are not the pinned ones runs on FLYDSL.

## Adding a shape

Needs W gfx950 GPUs (W = Ulysses world size) and the toolkit.

1. The TensileLite `sizes` `[M, N, 1, K]` of each hop, N = tokens per rank:

   | hop | `sizes` |
   |---|---|
   | `fwd_gemm_a2a` | `[(hq + 2 * hkv) * d, N, 1, hidden]`, `a2aExtent` = M |
   | `fwd_a2a_gemm` | `[hidden, N, 1, hq * d]` |
   | `bwd_gemm_a2a` | `[hq * d, N, 1, hidden]`, `a2aExtent` = M |
   | `bwd_a2a_gemm` | `[hidden, N, 1, (hq + 2 * hkv) * d]` |

2. Copy `tuning/<hop>.yaml`, set its `ProblemSizes` (and `world`), and tune:

   ```
   export PYTHONPATH=$TENSILELITE_HOME/python:$TENSILELITE_HOME/tensilelite
   HIP_VISIBLE_DEVICES=0,1,2,3 python3 $TENSILELITE_HOME/tensilelite/bin/Tensile my_tune.yaml /work/tune/<name> \
     --prebuilt-client $TENSILELITE_HOME/bin/tensilelite-client
   ```

   A few minutes per hop for a few shapes; the tuning client checks every winner on the CPU. If it stops with
   "Insufficient rotating buffer size" (large shapes), set `RotatingBufferSize: 0`.
3. Merge `/work/tune/<name>/3_LibraryLogic/*.yaml` into `logic/<hop>/` with
   `python3 $TENSILELITE_HOME/tensilelite/bin/TensileMergeLibrary logic/<hop> <new logic dir> <out dir>`. A shape that
   is in both keeps the new entry. The tool pairs files by name, so the new file must have the name of the one in
   `logic/<hop>/`; otherwise the output holds the new shapes only.
4. `kernels.py --update-pins`, reinstall, and run `tests/pytorch/ops/test_ulysses_fused_a2a.py` with
   `PRIMUS_TURBO_TEST_ULYSSES_TENSILELITE_SHAPE=hq,hkv,d,s_local,hidden,world` on W GPUs. To try libraries before
   committing them, `PRIMUS_TURBO_ULYSSES_TENSILELITE_LIB_{PRE,POST,PRE_BWD,POST_BWD}` point at `TensileCreateLibrary`
   outputs instead (their sha256 is not checked).
5. Commit the logic and the manifest.

## Moving the TensileLite commit

Set `rocm_libraries_commit` (and `toolchain`, for a new compiler) in `manifest.json`, rebuild the toolkit, run
`kernels.py --update-pins` (retune where the new commit does not accept the logic), check whether `src/` must change,
and rerun the tests on MI355X.
