# gfx1250 MLA attention on HipKittens kernels

DeepSeek-V3 MLA attention (head dims qk 192 / v 128, bf16, MHA, bottom-right causal) forward and backward on HIP
kernels for gfx1250 (MI455X): a drop-in for the FlyDSL MLA kernels (`primus_turbo/flydsl/attention/gfx1250_mla`) with
the same tensor layouts. The forward is bitwise equal to FlyDSL's, so either backward takes either forward's out / lse.
The Python side is `primus_turbo/pytorch/kernels/attention/attention_hipkittens_impl.py` (`hk_mla_fwd`,
`hk_mla_bwd`).

## Use

| setting | values | effect |
|---|---|---|
| `PRIMUS_TURBO_MLA_FWD_IMPL` | `flydsl` (default), `hk` | forward kernels |
| `PRIMUS_TURBO_MLA_BWD_IMPL` | `flydsl` (default), `hk` | backward kernels |
| `PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES` | bytes, unset = no limit | dS scratch budget of the `hk` backward |

The two `IMPL` settings are read when `attention_flydsl_gfx1250_mla_impl` is imported; the passes are independent. A
pass `hk` cannot take runs on FlyDSL, with one warning per reason. It must pass, in order:

1. the FlyDSL MLA rules (bf16, 4-D, qk 192 / v 128, MHA, causal, seqlen_q and seqlen_kv multiples of 64,
   seqlen_q <= seqlen_kv, 32-bit descriptor extents);
2. the HipKittens rules (`hk_mla_unsupported_reason`): forward: seqlen_q == seqlen_kv and q / k under 2 GiB;
   backward: seqlen_q % 384 in {0, 256}, batch * heads * seqlen_kv <= 512 * 4096, and a finite softmax scale of at
   least 2^-100 (its log2 is folded into the row constants, and dV accumulates bf16(scale * P));
3. the pass's code objects are built (`primus_turbo/lib/hk_gfx1250_mla`).

## Kernels

| pass | kernel | work |
|---|---|---|
| forward | `k_mla_fwd` | 256-query workgroups (4 waves, one per SIMD) over a 4-stage TDM ring of 64-row K / V tiles; one q block per workgroup, longest first, or one persistent workgroup per CU when its static schedule covers the shape (`hk_mla_persistent_workgroups`), whose ring runs on across its blocks |
| backward | `k_delta` | delta = rowsum(dO * O) and the row constants of the dK / dV kernel: NL = -log2(e) * lse + log2(scale), ND = -delta |
| backward | `k_dkdv96ds` | dK, dV over 96-row kv blocks (two 48-row waves on a 3-stage Q / dO TDM ring), then 64-row tail blocks; stores every bf16 dS^T tile it feeds the dK WMMA into the dS scratch; workgroups take their blocks from a ticket counter (zeroed before each launch), longest first |
| backward | `k_dqds_c` | dQ = dS K from that scratch: 384-query workgroups (4 waves sharing one K ring) plus one merged 256-query head tile |

**Numerics.** The forward runs the FlyDSL forward's operations in its order. The backward starts the dP chains from ND
and its softmax yields scale * P, so dS = (scale * P) * (dP - delta) is one multiply; dV accumulates scale * P and is
divided by scale once at the end. Every gradient is within 0.3 dB of the FlyDSL backward's SNR against an fp32
reference. Every output element is written by exactly one workgroup, so both passes are deterministic.

**dS scratch.** One block per backward call from the torch caching allocator, on the current stream: guard | layout
(`include/dslayout.h`) | guard, 4.33 GB at b2 s4096 h128. Over the budget the call runs `k_delta` once, then the
dK / dV and dQ kernels per chunk of heads, every chunk on the same scratch (bitwise equal to the unchunked call). When
not even one head group fits, or the scratch cannot be allocated, the call runs the FlyDSL backward.

## Build

`setup.py` runs `hsaco.build()` when `GPU_ARCHS` contains gfx1250 (`PRIMUS_TURBO_BUILD_HK_MLA=0` skips it); the code
objects go to `primus_turbo/lib/hk_gfx1250_mla` and ship in the wheel. By hand:

```
python csrc/kernels/attention/hipkittens/gfx1250_mla/hsaco.py [--check] [--force] [--strict] [--cxx PATH]
```

The compiler is `--cxx`, `$PRIMUS_TURBO_HK_CXX`, `amdclang++` on `PATH`, or `$ROCM_PATH` / `$ROCM_HOME` / `/opt/rocm`
`llvm/bin/amdclang++`. Each kernel is one device-only code object built with exactly the manifest's flags from this
directory alone (nothing from the csrc extension or `3rdparty/hipkittens`); `-cuid=<kernel>` makes the build
byte-identical in any directory.

**Fail-closed.** `manifest.json` pins the sha256 of every code object for its `toolchain` (the
`amdprimus/amdprimus:tas-20261019` image). Only a code object with the pinned bytes is installed, one without them is
removed, and another compiler builds nothing; a missing code object sends its pass to FlyDSL, and a failed build never
fails the install.

**Generated kernels.** `k_mla_fwd` and `k_dkdv96ds` are marker shells: the HIP source keeps the workgroup set-up, the
TDM descriptors and the uniform control flow, and inline-asm markers (`; HKGEN <region>`) stand where the per-tile
work goes. `hsaco.py` checks the sha256 of the compiler's own `.s` (`asm_input_sha256`), applies
`asm/<kernel>.s.patch` (exact context, no fuzz) and reruns the driver's own assembler and linker. The patch is what the
generators write (`tools/fwdgen`, `tools/bwdgen` on `tools/asmkit`): software-pipelined WMMA / LDS / softmax
schedules on a static register plan, with `s_set_vgpr_msb`, the expert-mode dependency waits and the hazard `v_nop`s
derived from the instruction stream. A generator checks its regions against the shell (VGPR writes only above every
VGPR the shell references, SGPR writes only in the declared clobbers, no open hazard on any path) and compares the code
object with the intended instructions. Without the generator's switch (`-DHKGEN_REF=1` / `-DHKGEN=0`) the same source
compiles the per-tile work in HIP.

## Changing a kernel

1. Edit `src/` / `include/`, or a kernel's flags in `manifest.json`.
2. When `k_mla_fwd` or `k_dkdv96ds` (or a header, flag or the toolchain they depend on) changed, regenerate their
   patches; this also writes their `asm_input_sha256` / `hsaco_sha256` (`--check` verifies that the committed ones are
   current, byte for byte):

   ```
   python csrc/kernels/attention/hipkittens/gfx1250_mla/tools/regen_asm_patch.py [k_mla_fwd] [k_dkdv96ds] [--check]
   ```

3. For the other kernels, `hsaco.py --force` prints the sha256 of a code object that does not match (with a new
   compiler, set `toolchain` to its `--version` line first); update `hsaco_sha256` (and `args` / `kernarg_size` /
   `block`, with `_MLA_KERNELS` in `attention_hipkittens_impl.py`, if an interface changed), then
   `hsaco.py --force --strict` must pass.
4. On an MI455X, `hsaco.py --check` must print `ok` for both passes, then run

   ```
   pytest tests/pytorch/ops/test_attention_hipkittens_gfx1250_mla.py tests/pytorch/ops/test_attention_flydsl_gfx1250_mla.py
   ```

## Performance

MI455X, DeepSeek-V3 MBS 2 (b2 s4096 h128), SBHD, causal, bf16, through the impl layer training uses; medians of 3
processes x 10 interleaved rounds of 20 back-to-back calls each:

| | FlyDSL (ms) | HipKittens (ms) | speedup |
|---|---|---|---|
| forward | 0.707 | 0.624 | 1.13x |
| backward | 2.895 | 1.776 | 1.63x |
| forward + backward (autograd) | 3.745 | 2.513 | 1.49x |

With the L2 flushed before every call the forward takes 0.646 ms against FlyDSL's 0.719 and the backward 1.794
against 3.000; with 20 ms of idle GPU before every call (the clocks of a training step) 0.634 against 0.767 and 1.798
against 3.065.

DeepSeek-V3 proxy on one MI455X (4 layers, 32 experts, MBS 2, GBS 16; medians of 3 runs of 20 iterations):
774.6 -> 735.9 ms per iteration (-5.0%), loss at iteration 20 1.90874 -> 1.90870, peak memory +4 GB (the dS scratch).
