# grouped_gemm_fwd — FP8 FWD Grouped GEMM

**Kernel:** `_grouped_fp8_persistent_gemm_kernel`
**Architecture:** gfx950 (MI355X)
**Operation:** `out = A @ B * scale` (FWD: trans_b=False)

---

## Base Kernel Origin

`base_gate_up_5760.hsaco` and `base_down_2880.hsaco` are Triton-JIT-compiled
binaries for the two FWD GEMM shapes in the MoE FFN forward pass.

They share the same kernel name as the DGRAD binaries
(`_grouped_fp8_persistent_gemm_kernel`) but are compiled with `trans_b=False`
and different tile/split-k configurations that Triton auto-selects per shape.

The binaries were extracted from the training container's Triton cache at:
`~/.triton/cache/<hash>/<kernel_name>.hsaco`

by running the FWD kernel for each shape to trigger JIT compilation and
then copying the resulting `.hsaco` files.

---

## Production Shapes

| File | A shape | B shape | Output shape | Context |
|------|---------|---------|--------------|---------|
| `base_gate_up_5760.hsaco` | `[131072, 2880]` | `[32, 2880, 5760]` | `[131072, 5760]` | Gate/up-projection FWD |
| `base_down_2880.hsaco` | `[131072, 5760]` | `[32, 5760, 2880]` | `[131072, 2880]` | Down-projection FWD |

Token dimension `131072 = 32 groups × 4096 tokens/group` at batch size 1, seqlen 4096.

---

## Status: No Patches Yet

No hand-tuned assembly patches exist for the FWD kernels. The base binaries
are reference-quality Triton outputs. The `build.sh` provides a round-trip
toolchain validation mode to verify that `tools/patch_co.py` and
`tools/disasm_to_asm.py` can process these binaries correctly before any
optimization work begins.

---

## Toolchain Validation (Round-Trip)

Before writing any optimizations, verify the toolchain works end-to-end:

```bash
bash build.sh --round-trip
```

This disassembles each base binary → converts to `.s` → reassembles →
produces `rt_gate_up_5760.hsaco` and `rt_down_2880.hsaco`. Then:

```bash
python3 check.py   # should PASS with cos_sim=1.0 for round-trip binaries
```

If the round-trip produces correct results, the disassembly output is a valid
starting point for editing.

---

## Adding FWD Optimizations

```bash
# 1. Disassemble the target shape
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 base_gate_up_5760.hsaco > /tmp/raw.dis

# 2. Convert to reassembleable .s
python3 ../tools/disasm_to_asm.py /tmp/raw.dis patch_gate_up_5760.s

# 3. Verify round-trip (must match before any edits)
python3 ../tools/patch_co.py base_gate_up_5760.hsaco patch_gate_up_5760.s /tmp/rt.hsaco
python3 check.py   # PASS

# 4. Edit patch_gate_up_5760.s with your optimization

# 5. Build and validate
bash build.sh
python3 check.py
python3 bench.py
```

Once a `patch_gate_up_5760.s` or `patch_down_2880.s` file exists in this
folder, `build.sh` (without `--round-trip`) will automatically apply it and
produce the corresponding `final_*.hsaco`.

---

## Optimization Opportunities

Potential areas for FWD kernel improvements (similar analysis to DGRAD):

- **Persistent grid scheduling:** The kernel already uses a persistent grid;
  verify that grid size is saturating all CUs.
- **Prefetch distance:** Check whether double-buffering or software pipelining
  can better hide HBM latency for the operand tiles.
- **Split-K tuning:** Triton's auto-tuned `SPLIT_K` may not be optimal for
  the specific MI355X memory subsystem configuration.
- **Scale application:** If the FP8 scale is applied after MFMA (v_pk_mul_f32),
  verify it is not redundant with any implicit hardware scaling.

---

## Files

| File | Description |
|------|-------------|
| `base_gate_up_5760.hsaco` | Triton reference binary for gate/up-projection FWD (unmodified) |
| `base_down_2880.hsaco` | Triton reference binary for down-projection FWD (unmodified) |
| `build.sh` | Round-trip validator + optimization build driver |
| `check.py` | Cosine-similarity correctness check vs Triton |
| `bench.py` | TFLOPS/bandwidth/latency benchmark |
