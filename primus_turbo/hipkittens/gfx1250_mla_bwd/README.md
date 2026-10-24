# gfx1250 MLA backward on HIP kernels

The DeepSeek-V3 MLA attention backward on gfx1250 (MI455X) with three HIP kernels: head dims qk 192 /
v 128, bf16, MHA, bottom-right causal. The forward and the tensor layouts are the FlyDSL MLA ones
(`primus_turbo/flydsl/attention/gfx1250_mla`); only the backward kernels change.

## Use

`PRIMUS_TURBO_MLA_BWD_IMPL=hk` selects it (default `flydsl`). The setting is read at import by
`primus_turbo/pytorch/kernels/attention/attention_flydsl_gfx1250_mla_impl.py`; the entry point is
`impl.hk_attn_bwd`.

Each backward call makes three launches on the caller's current stream:

| launch | kernel | work |
|---|---|---|
| `delta` | HIP `k_delta` | delta, and the row constants NL = -LOG2E·lse, ND = -scale·delta: 64-query workgroups of 16 rows × 16 lanes, bitwise equal to the FlyDSL `k_delta` (below) |
| `dkdv` | HIP `k_dkdv96ds` | dK, dV over 96-row kv blocks (two 48-row waves on a 3-stage Q/dO TDM ring), then 64-row tail blocks; stores every bf16 dS^T tile it feeds the dK WMMA into the dS scratch |
| `dq` | HIP `k_dqds_c` | dQ = dS·K from that scratch: 384-query workgroups (4 waves sharing one K ring) plus one merged 256-query head tile |

**dS scratch.** One block per call from the torch caching allocator, on the current stream:
guard | layout (`hip/include/dslayout.h` v1, `limits.ds_layout_bytes`) | guard. That is 4.33 GB at
b2 s4096 h128. `PRIMUS_TURBO_MLA_BWD_HK_MAX_SCRATCH_BYTES` (bytes, e.g. `2.2e9`, read at import) or
`hk_attn_bwd(max_scratch_bytes=...)` caps it. Over the cap, the call runs `delta` once, then
(`dkdv`, `dq`) per chunk of heads, every chunk on the same scratch. Two chunks at b2 s4096 h128 use
2.16 GB and take 0.75% longer. When not even one head group fits, the call runs the FlyDSL backward.

**Gate** (`hk_mla_bwd_unsupported_reason`). A call must pass, in order:

1. the FlyDSL MLA rules: one device, bf16, 4-D, qk 192 / v 128, MHA, causal, and the FlyDSL
   backward's shape rules (seqlen_q a multiple of 64, seqlen_kv a multiple of 64,
   seqlen_q <= seqlen_kv, 32-bit descriptor extents);
2. the HIP kernels' shape rules (`limits.shape_unsupported_reason`): seqlen_q % 384 in {0, 256},
   seqlen_kv % 96 in {0, 32, 64}, and batch·heads·max(seqlen) <= 512·4096 (the extent the index
   proofs replay);
3. the required code objects (`k_dkdv96ds`, `k_dqds_c`): built, and byte-identical to the ones
   `manifest.json` pins.

A call that fails any rule runs the FlyDSL backward, with one warning per reason. `k_delta` is the
manifest's one `optional` kernel: when its code object is missing or not the pinned one, the `delta`
launch runs the FlyDSL `k_delta` instead (one warning) and the HIP `k_dkdv96ds` / `k_dqds_c` still
run; the results are the same bits.

**k_delta.** Same arithmetic per row as the FlyDSL `k_delta`: per lane two chains over its 8
elements (even, odd), their sum, then a 1/2/4/8 xor butterfly over the row's 16 lanes. It is built
with fp contraction off (`#pragma clang fp contract(off)`): FlyDSL multiplies and adds separately,
and a fused multiply-add rounds differently once a bf16 × bf16 product underflows into the fp32
denormal range. fp32 denormals are preserved in both. The tests cover denormal inputs, underflowing
products, signed zeros and an lse whose NL overflows.

**Results.** Every output element is written by exactly one workgroup (no atomics), so dQ / dK /
dV are deterministic. They are also bitwise equal to the FlyDSL backward's (same dS rounding, same
accumulation order), and independent of the head-group launch targets and of head chunking.

## Files

| path | content |
|---|---|
| `hip/src/k_delta.hip`, `hip/src/k_dkdv96ds.hip`, `hip/src/k_dqds_c.hip` | the three kernels |
| `hip/include/` | their headers; `cdna5/` vendors a subset of HipKittens (MIT, commit in the manifest) |
| `hip/asm/k_dkdv96ds.s.patch` | assembly patch of k_dkdv96ds (below) |
| `manifest.json` | compiler flags, toolchain, per-kernel sha256 pins, launch metadata (block, LDS, VGPRs, kernel arguments) |
| `hsaco.py` | builds and verifies the code objects (standard library only) |
| `hiprt.py` | loads and launches them through the HIP runtime torch loaded (ctypes) |
| `impl.py`, `limits.py` | launch plan and shape rules |
| `tools/regen_asm_patch.py` | regenerates the assembly patch |
| `bin/*.hsaco` | build output (not in git; shipped in the wheel) |

## Build

`setup.py` runs `hsaco.build()` when `GPU_ARCHS` contains gfx1250; `PRIMUS_TURBO_BUILD_HK_MLA_BWD=0`
skips it. To build or check by hand:

```
python primus_turbo/hipkittens/gfx1250_mla_bwd/hsaco.py [--check] [--force] [--strict] [--cxx PATH]
```

The compiler is `--cxx`, `$PRIMUS_TURBO_HK_CXX`, `amdclang++` on `PATH`, or
`$ROCM_PATH` / `$ROCM_HOME` / `/opt/rocm` `llvm/bin/amdclang++`.

- Each kernel is one device-only code object, built with exactly the manifest's flags (never
  Turbo's csrc flags). `-cuid=<kernel>` makes the build byte-identical in any directory.
- **Fail-closed.** `manifest.json` pins the sha256 of every code object as validated with its
  `toolchain` (AMD clang 24.0.0git, ROCm/llvm-project 86230635). A build that produces other bytes
  is deleted. A missing or foreign code object is never loaded (`hsaco.unavailable_reason`,
  checked again by `hiprt` before loading), so the backward runs on FlyDSL; for the optional
  `k_delta` only that launch does (`hsaco.kernel_unavailable_reason`). A failed build never fails
  the install.
- **Assembly patch.** `k_dkdv96ds` also has `asm_patch` and `asm_input_sha256`. After compiling,
  `hsaco.py` checks the sha256 of the compiler's own `.s` (`-save-temps=obj`). It then applies
  `hip/asm/k_dkdv96ds.s.patch` (exact context, no fuzz) and re-runs the driver's own assembler and
  linker commands (from `-###`). Any mismatch means the kernel is not built, so the backward runs
  on FlyDSL.

Why the patch: the compiler puts the 24 Q^T `ds_load_tr16_b128` of an iteration into the b_do
registers the dV WMMAs have just read. In expert scheduling mode such a load first has to drain
the VALU queue (`s_wait_alu depctr_va_vdst(0)`), and LLVM has no option against it.

The patch moves 5 of the 8 affected 8-VGPR destination tuples per loop trip (the other 3 sit in
v0-v255, which has no free tuple), and the WMMA operands that read them, to registers that are
free over the value's live range. Each new tuple stays in the same 256-VGPR block, so every
`s_set_vgpr_msb` stays valid. The patch then recomputes the loop's `va_vdst`
waits with LLVM's own rules. Effect: va_vdst(0) drains per loop trip drop from 15 to 9; the
dK/dV kernel is 3.1% faster and the backward 2.2% faster; results are bitwise unchanged. Only
register numbers and `s_wait_alu` lines change; the tool's docstring has the rules and the model.

## Changing a kernel

1. Edit `hip/src` / `hip/include` (or a kernel's flags in `manifest.json`).
2. Regenerate the assembly patch whenever `k_dkdv96ds` or its headers, its flags, or the toolchain
   change. Without it, `hsaco.py` refuses: `compiler .s sha256 ... != asm_input_sha256`.

   ```
   python primus_turbo/hipkittens/gfx1250_mla_bwd/tools/regen_asm_patch.py
   ```

   The tool compiles the kernel exactly as `hsaco.py` does and finds the full loop and its Q^T
   tr16 loads by the kernel's structure. It re-targets them, recomputes the waits and runs static
   checks: only VGPR numbers and waits change, each operand keeps its 256-VGPR block, dataflow is
   unchanged, no new open hazard, no `s_wait_alu` inside a hard clause, and every wait is at least
   as strict as LLVM's model. It then writes the patch, assembles it through `hsaco.py`'s build
   path, and prints the new `asm_input_sha256` / `hsaco_sha256`.

   If the loop changed shape, or a check fails, the tool stops. Then either adapt the tool, or
   drop `asm_patch` / `asm_input_sha256` from the manifest (the kernel loses the ~3%).
   `regen_asm_patch.py --check` verifies that the committed patch and shas are current, byte for
   byte.
3. Update `manifest.json`: the new sha256 values (`hsaco.py --force` prints the sha of a code
   object that does not match), `toolchain` if the compiler changed, and `args` / `kernarg_size`
   / `lds` / `vgpr` if a kernel's interface changed.
4. Run `python primus_turbo/hipkittens/gfx1250_mla_bwd/hsaco.py --force --strict`, then the gates on
   an MI455X:

   ```
   pytest tests/pytorch/ops/test_attention_flydsl_gfx1250_mla.py
   ```

   The HIP numeric tests skip when the code objects are unavailable, so check first that
   `hsaco.py --check` prints `ok` and no `optional k_delta` line. They cover the fp32 reference,
   bitwise equality with the FlyDSL backward and of `k_delta` with the FlyDSL `k_delta` (denormal
   inputs included), determinism, a NaN-filled scratch at a production-like shape, and head
   chunking. The static checks are not a proof; run the GPU gates after every patch regeneration.

## Performance

MI455X, b2 s4096 h128, SBHD, causal, bf16:

| | time (ms) |
|---|---|
| forward (FlyDSL) | 0.742 |
| backward, `hk` | 2.231 |
| per kernel: k_delta / k_dkdv96ds / k_dqds_c | 0.057 / 1.700 / 0.484 |
| backward, FlyDSL | 3.196 |
| autograd fwd + bwd, `hk` / FlyDSL | 3.234 / 4.254 |

The `hk` backward is 1.43x faster than the FlyDSL one, and forward + backward is 1.32x faster.
