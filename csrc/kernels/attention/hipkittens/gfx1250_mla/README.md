# gfx1250 MLA attention backward on HipKittens kernels

DeepSeek-V3 MLA attention (head dims qk 192 / v 128, bf16, MHA, bottom-right causal) backward on HIP kernels for
gfx1250 (MI455X): a drop-in for the FlyDSL MLA backward (`primus_turbo/flydsl/attention/gfx1250_mla`) with the same
tensors, whose gradients it reproduces bit for bit. The forward stays on FlyDSL. The Python side is
`primus_turbo/pytorch/kernels/attention/attention_hipkittens_gfx1250_mla_impl.py` (`hk_mla_bwd`).

## Use

`PRIMUS_TURBO_MLA_BWD_IMPL=hk` (default `flydsl`) runs the backward of the FlyDSL MLA attention on these kernels. It
is read through `primus_turbo.common.envs` at the first MLA backward (`envs.set_env()` changes it for later eager
backwards); `torch.compile` reads it while tracing, and a compiled graph keeps that setting. A call the kernels cannot
take runs on FlyDSL, with one warning per reason. It must pass, in order:

1. the FlyDSL MLA rules (bf16, 4-D, qk 192 / v 128, MHA, causal, seqlen_q and seqlen_kv multiples of 64,
   seqlen_q <= seqlen_kv, 32-bit descriptor extents);
2. the HipKittens rules (`hk_mla_unsupported_reason`): seqlen_q % 384 in {0, 256} and
   batch * heads * seqlen_kv <= 512 * 4096;
3. the code objects are built (`primus_turbo/lib/hk_gfx1250_mla`).

## Kernels

| kernel | work |
|---|---|
| `k_delta` | the row constants NL = -log2(e) * lse and ND = -scale * delta, delta = rowsum(dO * O) |
| `k_dkdv96ds` | dK, dV over 96-row kv blocks (two 48-row waves on a 3-stage Q / dO TDM ring), then 64-row tail blocks; stores every bf16 dS^T tile it feeds the dK WMMA into the dS scratch |
| `k_dqds_c` | dQ = dS K from that scratch: 384-query workgroups (4 waves sharing one K ring) plus one merged 256-query head tile |

**Numerics.** The kernels run the FlyDSL backward's operations in its order, so dQ, dK and dV are bitwise those of
the FlyDSL backward. Every output element is written by exactly one workgroup, so the backward is deterministic.

**dS scratch.** One block per backward call from the torch caching allocator, on the current stream: the layout of
`include/dslayout.h`, about batch * heads * seqlen_q * seqlen_kv bytes under the causal mask: 4.33 GB at b2 s4096
h128, 33 GB at b1 s16128 h128 (FlyDSL needs none). When it or the gradients cannot be allocated, the call runs the
FlyDSL backward, and no scratch at least as large is tried again on that device.

## Build

`setup.py` runs `hsaco.build()` when `GPU_ARCHS` contains gfx1250 (`PRIMUS_TURBO_BUILD_HK_MLA=0` skips it); the code
objects go to `primus_turbo/lib/hk_gfx1250_mla` and ship in the wheel. By hand:

```
python csrc/kernels/attention/hipkittens/gfx1250_mla/hsaco.py [--check] [--force] [--strict] [--cxx PATH] [--bin-dir DIR]
```

It builds into this checkout's `primus_turbo/lib/hk_gfx1250_mla`, which an in-place or editable build uses; for an
installed package, pass that package's `primus_turbo/lib/hk_gfx1250_mla` as `--bin-dir`. The runtime loads a code
object only when its sha256 is the pin `_MLA_KERNELS` in `attention_hipkittens_gfx1250_mla_impl.py` holds (the
manifest's, kept equal by a test), so a stale one left by an older build runs nothing: the backward runs on FlyDSL.

The compiler is `--cxx`, `$PRIMUS_TURBO_HK_CXX`, `amdclang++` on `PATH`, or `$ROCM_PATH` / `$ROCM_HOME` / `/opt/rocm`
`llvm/bin/amdclang++`. Each kernel is one device-only code object built with exactly the manifest's flags from this
directory alone (nothing from the csrc extension or `3rdparty/hipkittens`); `-cuid=<kernel>` makes the build
byte-identical in any directory.

**Fail-closed.** `manifest.json` pins the sha256 of every code object for its `toolchain` (the
`amdprimus/amdprimus:tas-20261019` image). Only a code object with the pinned bytes is installed, one without them is
removed, and another compiler builds nothing; a missing code object sends the backward to FlyDSL, and a failed build
never fails the install.

## Changing a kernel

1. Edit `src/` or `include/`, or a kernel's flags in `manifest.json` (for a new compiler, set `toolchain` to its
   `--version` line first).
2. `hsaco.py --force` reports the sha256 of every code object that changed. Set it as the kernel's `hsaco_sha256` in
   `manifest.json` and in `_MLA_KERNELS` of `attention_hipkittens_gfx1250_mla_impl.py` (with its block size and
   argument layout if the interface changed); `hsaco.py --force --strict` must then pass.
3. On an MI455X, `hsaco.py --check` must print `ok`, then run

   ```
   pytest tests/pytorch/ops/test_attention_hipkittens_gfx1250_mla.py tests/pytorch/ops/test_attention_flydsl_gfx1250_mla.py
   ```

## Performance

MI455X (GPU clock held at 1.3 GHz on the test node), DeepSeek-V3 MBS 2 (b2 s4096 h128), SBHD, causal, bf16, through
the impl layer training uses; medians of 3 processes x 10 interleaved rounds of 20 back-to-back calls each:

| | FlyDSL (ms) | HipKittens backward (ms) | speedup |
|---|---|---|---|
| backward | 4.600 | 2.828 | 1.63x |
| forward + backward (autograd) | 5.676 | 3.916 | 1.45x |

With the L2 flushed before every call the backward takes 2.808 ms against FlyDSL's 4.573; with 20 ms of idle GPU
before every call (the clocks of a training step) 2.936 against 4.728.

DeepSeek-V3 proxy on one MI455X (4 layers, 32 experts, MBS 2, GBS 16; medians of 3 runs of iterations 4-20):
1076.7 -> 1019.6 ms per iteration (-5.30%); the loss of every iteration is unchanged (the gradients are FlyDSL's), and
at this shape so is the peak memory: the dS scratch fits under the step's peak.
