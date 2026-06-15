# asm_kernels — Hand-Optimized AMDGCN Kernels for gpt-oss MoE 20B

AMDGCN assembly kernels for MI355X (gfx950) tuned for the gpt-oss MoE 20B
training workload in Primus-Turbo.

Each subfolder is a self-contained kernel family with a flat, consistent layout:

```
<kernel>/
  base.co / base*.hsaco   Triton-generated reference binary (unmodified)
  patch_*.s               Assembly patch files (one per optimization step)
  build.sh                Applies patches via tools/patch_co.py → final.co
  final.co                Optimized output (produced by build.sh)
  check.py                Correctness check vs Triton (cosine similarity)
  bench.py                Performance benchmark vs Triton (latency / TFLOPS / BW)
```

---

## Kernels

| Folder | Kernel function | Description |
|--------|----------------|-------------|
| `grouped_gemm_wgrad/` | `grouped_variable_k_dot_scaled_kernel` | FP8 WGRAD grouped GEMM (dot-scaled) |
| `grouped_gemm_dgrad/` | `_grouped_fp8_persistent_gemm_kernel` | FP8 DGRAD grouped GEMM (trans_b=True) |
| `grouped_gemm_fwd/`   | `_grouped_fp8_persistent_gemm_kernel` | FP8 FWD grouped GEMM (trans_b=False) |
| `swiglu_bwd/`         | `swiglu_with_mask_bwd_kernel`         | SwiGLU backward with expert mask |
| `fused_router/`       | `fused_scaling_group_sum_routing_kernel` | MoE router (softmax + top-k) |

---

## How Base Binaries Were Obtained

All base binaries are **unmodified** Triton-JIT-compiled kernels. They serve
as ELF donor files: `patch_co.py` replaces only the `.text` section while
preserving the original kernel descriptor, metadata YAML, and symbol table.

### WGRAD kernel (`grouped_gemm_wgrad/base.co`)

The dot_scaled WGRAD base binary comes from the
[mawad-amd/ggemm-asm](https://github.com/mawad-amd/ggemm-asm/tree/wgrad-asm-optimization)
repository (`kernels/dot_scaled_compiled.co`). It was produced by running
Triton's `tl.dot_scaled` API (`grouped_vark_dot_scaled.py` in that repo),
which triggers JIT compilation and caches the resulting `.co` in
`~/.triton/cache/`. The binary uses `v_mfma_f32_32x32x64_f8f6f4` (MI355X
native dot_scaled MFMA) — 8× the FLOPs per instruction vs the legacy 16×16
opcode.

### All other kernels (DGRAD, FWD, SwiGLU, Router)

Triton writes compiled `.hsaco` binaries to its cache at:
`~/.triton/cache/<hash>/<kernel_name>.hsaco`

To extract the binary for a specific kernel and shape, run the Triton kernel
once to trigger JIT compilation, then copy the resulting `.hsaco`.

### Disassembly

To disassemble any base binary to human-readable AMDGCN:

```bash
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 base.co
```

To convert the raw disassembly to a reassembleable `.s` file (with labels
and metadata regenerated):

```bash
python3 ../tools/disasm_to_asm.py raw.dis patch_new.s
```

---

## Tools

| File | Description |
|------|-------------|
| `tools/patch_co.py` | Assembles a `.s` file and splices the new `.text` section into an existing `.co` ELF, preserving all metadata and kernel descriptors |
| `tools/disasm_to_asm.py` | Converts raw `llvm-objdump` output into a complete reassembleable `.s` file with `.amdgcn_target`, `.amdhsa_kernel` metadata, and branch labels |

### `patch_co.py` usage

```bash
python3 ../tools/patch_co.py <donor.co> <modified.s> <output.co>
```

The donor ELF provides metadata (kernel descriptor, symbol table, AMDGPU
metadata YAML). The `.s` file provides the new machine code. The output ELF
is binary-identical to the donor except for the `.text` section content.

If the new `.text` is shorter than the original, it is padded with `s_nop 0`
to preserve code object size.

### `disasm_to_asm.py` usage

```bash
# Step 1: raw disassembly
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 base.co > raw.dis

# Step 2: convert to reassembleable .s
python3 ../tools/disasm_to_asm.py raw.dis patch_template.s
```

The output `.s` contains `.amdgcn_target "amdgcn-amd-amdhsa--gfx950"`,
`.amdhsa_kernel` block extracted from the ELF metadata YAML, all branch
targets replaced with `.L0`/`.L1`/... symbolic labels, and the instruction
stream ready for `llvm-mc`.

---

## General Workflow: Adding a New Optimization

1. **Disassemble** the base binary to get a starting `.s`:

   ```bash
   /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 base.co > /tmp/raw.dis
   python3 ../tools/disasm_to_asm.py /tmp/raw.dis patch_vN.s
   ```

2. **Verify round-trip** (the unmodified `.s` should produce the same binary):

   ```bash
   python3 ../tools/patch_co.py base.co patch_vN.s /tmp/rt.co
   python3 check.py --co /tmp/rt.co   # should PASS with cos_sim=1.0
   ```

3. **Edit** `patch_vN.s` with your optimization.

4. **Build** the optimized `.co`:

   ```bash
   bash build.sh
   ```

5. **Check correctness**:

   ```bash
   python3 check.py        # tests final.co vs Triton, cosine similarity >= 0.99999
   ```

6. **Benchmark**:

   ```bash
   python3 bench.py        # reports latency ms, TFLOPS/BW, speedup vs Triton
   ```

---

## Per-Kernel Documentation

| README | Content |
|--------|---------|
| `grouped_gemm_wgrad/README.md` | cbsz bug root cause, two-step patch chain, beta=1 trampoline |
| `grouped_gemm_dgrad/README.md` | DGRAD kernel shapes, optimization patches applied |
| `grouped_gemm_fwd/README.md` | FWD kernel shapes, round-trip validation, adding future patches |
| `swiglu_bwd/README.md` | Two-phase optimization: NOP removal and IEEE div → v_rcp_f32 |
| `fused_router/README.md` | Routing kernel optimization: fast reciprocal and sort scheduling |
