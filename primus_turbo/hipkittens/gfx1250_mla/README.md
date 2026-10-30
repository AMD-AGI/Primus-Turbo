# gfx1250 MLA attention on HIP kernels

DeepSeek-V3 MLA attention (head dims qk 192 / v 128, bf16, MHA, bottom-right causal) forward and backward on HIP
kernels for gfx1250 (MI455X). They are a drop-in for the FlyDSL MLA kernels
(`primus_turbo/flydsl/attention/gfx1250_mla`): same tensor layouts, the forward is bitwise equal, and either
backward takes either forward's out / lse.

## Use

| setting | values | effect |
|---|---|---|
| `PRIMUS_TURBO_MLA_FWD_IMPL` | `flydsl` (default), `hk` | forward kernels |
| `PRIMUS_TURBO_MLA_BWD_IMPL` | `flydsl` (default), `hk` | backward kernels |
| `PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES` | bytes (e.g. `2.2e9`), unset = no limit | dS scratch budget of the `hk` backward |

The two `IMPL` settings are read at import by `primus_turbo/pytorch/kernels/attention/attention_flydsl_gfx1250_mla_impl.py`
(the passes are independent), the scratch budget when the first `hk` backward runs. The entry points are
`fwd.hk_attn_fwd` and `bwd.hk_attn_bwd`. A pass the HIP kernels cannot take runs on FlyDSL, with one warning per
reason (not under torch.compile, which cannot trace a log call). A call must pass, in order:

1. the FlyDSL MLA rules (one device, bf16, 4-D, qk 192 / v 128, MHA, causal, seqlen_q and seqlen_kv multiples of 64,
   seqlen_q <= seqlen_kv, 32-bit descriptor extents);
2. the HIP kernels' rules (`limits.py`): forward: seqlen_q == seqlen_kv and q / k smaller than 2 GiB; backward:
   seqlen_q % 384 in {0, 256}, seqlen_kv % 96 in {0, 32, 64}, batch * heads * seqlen <= 512 * 4096, and a finite
   softmax scale of at least 2^-100 (its log2 is folded into the row constants, and dV accumulates bf16(scale * P));
3. the pass's code objects: built and byte-identical to the ones `manifest.json` pins, and a HIP runtime `hiprt` can
   use (the one torch loaded, mapped once).

## Kernels

| pass | kernel | work |
|---|---|---|
| forward | `k_mla_fwd` | 256-query workgroups (4 waves, one per SIMD) over a 4-stage TDM ring of 64-row K / V tiles; QK, online softmax, PV per tile, the diagonal tiles specialized. One q block per workgroup, longest first; or, when its static schedule covers the shape, one persistent workgroup per CU (`fwd.persistent_workgroups`) |
| backward | `k_delta` | delta = rowsum(dO * O) and the row constants of the dK / dV kernel: NL = -log2(e) * lse + log2(scale), ND = -delta |
| backward | `k_dkdv96ds` | dK, dV over 96-row kv blocks (two 48-row waves on a 3-stage Q / dO TDM ring), then 64-row tail blocks; stores every bf16 dS^T tile it feeds the dK WMMA into the dS scratch; workgroups take their blocks from a ticket counter (zeroed on the host before each launch), longest first |
| backward | `k_dqds_c` | dQ = dS K from that scratch: 384-query workgroups (4 waves sharing one K ring) plus one merged 256-query head tile |

**Numerics.** The forward runs the FlyDSL forward's operations in its order, so out and lse are bitwise equal. The
backward starts the dP chains from ND and its softmax yields scale * P, so dS = (scale * P) * (dP - delta) is one
multiply; dV accumulates scale * P and is divided by scale once at the end. Every gradient is within 0.3 dB of the
FlyDSL backward's SNR against an fp32 reference (dV rounds scale * P to bf16 instead of P: 0.02-0.07 dB lower).
Every output element is written by exactly one workgroup (the ticket only hands out work), so both passes are
deterministic.

**dS scratch.** One block per backward call from the torch caching allocator, on the current stream: guard | layout
(`hip/include/dslayout.h`, `limits.ds_layout_bytes`) | guard, 4.33 GB at b2 s4096 h128. Over the budget the call runs
`k_delta` once, then the dK / dV and dQ kernels per chunk of heads, every chunk on the same scratch (bitwise equal to
the unchunked call; two chunks at b2 s4096 h128: 2.16 GB, 0.75% slower). When not even one head group fits, or the
scratch cannot be allocated, the call runs the FlyDSL backward (one warning).

## Build

`setup.py` runs `hsaco.build()` when `GPU_ARCHS` contains gfx1250 (`PRIMUS_TURBO_BUILD_HK_MLA=0` skips it). By hand:

```
python primus_turbo/hipkittens/gfx1250_mla/hsaco.py [--check] [--force] [--strict] [--cxx PATH]
```

The compiler is `--cxx`, `$PRIMUS_TURBO_HK_CXX`, `amdclang++` on `PATH`, or `$ROCM_PATH` / `$ROCM_HOME` / `/opt/rocm`
`llvm/bin/amdclang++`. Each kernel is one device-only code object built with exactly the manifest's flags; the sources
are self-contained (the HIP runtime and the package's own headers: they share nothing with the csrc build or
`3rdparty/hipkittens`). `-cuid=<kernel>` makes the build byte-identical in any directory.

**Fail-closed.** `manifest.json` pins the sha256 of every code object as validated with its `toolchain`. A build that
produces other bytes is discarded (a code object replaces the one in `bin/` only when it is the pinned one), a missing
or foreign code object is never loaded (`hsaco.unavailable_reason`, checked again by `hiprt` before loading), and that
pass runs on FlyDSL. Another compiler than `toolchain` builds nothing (it cannot reproduce the pins), and a failed
build never fails the install. The wheel ships `hip/` with the code objects, so an
installed package can rebuild them in place.

**Generated kernels.** `k_mla_fwd` and `k_dkdv96ds` are marker shells: compiled with their manifest flags, the HIP
source keeps the workgroup set-up, the TDM descriptors and the uniform control flow, and inline-asm markers
(`; HKGEN <region>`) stand where the per-tile work goes. Their manifest entries have `asm_patch` and
`asm_input_sha256`: `hsaco.py` checks the sha256 of the compiler's own `.s` (`-save-temps=obj`), applies
`hip/asm/<kernel>.s.patch` (exact context, no fuzz) and re-runs the driver's own assembler and linker. The patch is
the code the generators write (`tools/fwdgen`, `tools/bwdgen` on `tools/asmkit`): software-pipelined WMMA / LDS /
softmax schedules on a static register plan, with `s_set_vgpr_msb`, the expert-mode dependency waits and the hazard
`v_nop`s derived from the instruction stream. A generator checks its regions against the shell (generated VGPR writes
only above every VGPR the shell's code references, SGPR writes only in the declared clobbers, no open hazard on any
path), assembles them and compares the code object with the intended instructions. Without the generator's switch
(`-DHKGEN_REF=1` / `-DHKGEN=0`) the same source compiles the per-tile work in HIP: the reference the generators learn
the VGPR-MSB operand fields from.

## Changing a kernel

1. Edit `hip/src` / `hip/include`, or a kernel's flags in `manifest.json`.
2. When `k_mla_fwd` or `k_dkdv96ds` (or a header, flag or the toolchain they depend on) changed, regenerate their
   patches; this also writes their `asm_input_sha256` / `hsaco_sha256`:

   ```
   python primus_turbo/hipkittens/gfx1250_mla/tools/regen_asm_patch.py [k_mla_fwd] [k_dkdv96ds] [--check]
   ```

   `--check` verifies that the committed patches and sha256 values are current, byte for byte.
3. For the other kernels, `python primus_turbo/hipkittens/gfx1250_mla/hsaco.py --force` prints the sha256 of a code
   object that does not match (with a new compiler, set `toolchain` to its `--version` line first); update
   `hsaco_sha256` (and `args` / `kernarg_size` / `lds` / `vgpr` if an interface changed), then
   `hsaco.py --force --strict` must pass.
4. On an MI455X, check that `hsaco.py --check` prints `ok` for both passes, then run

   ```
   pytest tests/pytorch/ops/test_attention_hipkittens_gfx1250_mla.py tests/pytorch/ops/test_attention_flydsl_gfx1250_mla.py
   ```

   The numeric tests skip when the code objects are unavailable. They cover the fp32 reference, bitwise equality with
   the FlyDSL forward, the FlyDSL backward's SNR, `k_delta` against the FlyDSL `k_delta` (denormal inputs included),
   determinism, NaN-filled outputs and scratch, the persistent forward grid, head chunking and torch.compile.

## Files

| path | content |
|---|---|
| `fwd.py`, `bwd.py` | entry points and launch plans |
| `limits.py` | shape rules (standard library only) |
| `hsaco.py` | build and verify the code objects (standard library only) |
| `hiprt.py` | load and launch them through the HIP runtime torch loaded (ctypes) |
| `manifest.json` | toolchain, flags, per-kernel sha256 pins and launch metadata (block, LDS, VGPRs, kernel arguments) |
| `hip/src`, `hip/include` | kernel sources and headers |
| `hip/asm` | assembly patches of the generated kernels |
| `tools/` | `regen_asm_patch.py`, the generators `fwdgen/` and `bwdgen/` and their assembly toolkit `asmkit/` |
| `bin/*.hsaco` | build output (not in git; shipped in the wheel) |

## Performance

MI455X, b2 s4096 h128 (DeepSeek-V3 MBS 2), SBHD, causal, bf16; medians of 10 interleaved rounds of 20 back-to-back
calls (with the L2 flushed before every call: within 1%):

| | FlyDSL (ms) | HIP (ms) | speedup |
|---|---|---|---|
| forward | 0.794 | 0.777 | 1.02x |
| backward | 3.493 | 2.281 | 1.53x |
| forward + backward (autograd) | 4.302 | 3.082 | 1.40x |

HIP backward per launch: `k_delta` 0.058, ticket fill 0.005, `k_dkdv96ds` 1.609, `k_dqds_c` 0.493 ms. The kernels are
power-capped at this shape: a saved cycle converts to about a third of a cycle of time.
