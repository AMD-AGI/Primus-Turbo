# grouped_gemm_dgrad — FP8 DGRAD Grouped GEMM

**Kernel:** `_grouped_fp8_persistent_gemm_kernel`
**Architecture:** gfx950 (MI355X)
**Operation:** `out = A @ B^T * scale` (DGRAD: trans_b=True)

---

## Base Kernel Origin

`base_down_2880.hsaco` and `base_gate_up_5760.hsaco` are Triton-JIT-compiled
binaries for two DGRAD shapes used in the MoE FFN backward pass.

They were extracted from the training container's Triton cache at:
`~/.triton/cache/<hash>/<kernel_name>.hsaco`

by running the DGRAD kernel for each shape to trigger JIT compilation and
then copying the resulting `.hsaco` files.

Both files are stored unmodified as the patch chain starting point.

---

## Production Shapes

| File | A shape | B shape (before transpose) | Output shape | Context |
|------|---------|---------------------------|--------------|---------|
| `base_down_2880.hsaco` | `[131072, 2880]` | `[32, 2880, 2880]` | `[131072, 2880]` | DGRAD through down-projection |
| `base_gate_up_5760.hsaco` | `[131072, 5760]` | `[32, 2880, 5760]` | `[131072, 2880]` | DGRAD through gate/up-projection |

Token dimension `131072 = 32 groups × 4096 tokens/group` at batch size 1, seqlen 4096.

---

## Patch Chain

Each shape has an independent single-step patch:

```
base_down_2880.hsaco     + patch_down_2880.s     -> final_down_2880.hsaco
base_gate_up_5760.hsaco  + patch_gate_up_5760.s  -> final_gate_up_5760.hsaco
```

The patch `.s` files contain hand-optimized AMDGCN assembly. Edit the
relevant `.s` to add new optimizations, then rebuild.

---

## Build

```bash
bash build.sh                  # build both shapes
bash build.sh --down-only      # down_dgrad (K=2880, N=2880) only
bash build.sh --gate-up-only   # gate_up_dgrad (K=5760, N=2880) only
```

Produces `final_down_2880.hsaco` and/or `final_gate_up_5760.hsaco`.

---

## Correctness Check

```bash
python3 check.py
```

Tests `final_down_2880.hsaco` and `final_gate_up_5760.hsaco` against Triton
for both DGRAD shapes. Passes cosine-similarity threshold ≥ 0.99999.

---

## Benchmark

```bash
python3 bench.py
```

Reports latency (ms), TFLOPS, and memory bandwidth (GB/s) for each shape,
with speedup vs the Triton reference.

---

## Adding a New Optimization

1. Disassemble the base binary you want to optimize:

   ```bash
   /opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx950 base_down_2880.hsaco > /tmp/raw.dis
   python3 ../tools/disasm_to_asm.py /tmp/raw.dis patch_down_2880.s
   ```

2. Verify round-trip (unmodified `.s` must reproduce original binary):

   ```bash
   python3 ../tools/patch_co.py base_down_2880.hsaco patch_down_2880.s /tmp/rt.hsaco
   python3 check.py   # should PASS
   ```

3. Edit `patch_down_2880.s` with your changes.

4. Rebuild and validate:

   ```bash
   bash build.sh --down-only
   python3 check.py
   python3 bench.py
   ```

---

## Files

| File | Description |
|------|-------------|
| `base_down_2880.hsaco` | Triton reference binary for K=2880, N=2880 (unmodified) |
| `base_gate_up_5760.hsaco` | Triton reference binary for K=5760, N=2880 (unmodified) |
| `patch_down_2880.s` | Optimized assembly for down_dgrad shape |
| `patch_gate_up_5760.s` | Optimized assembly for gate_up_dgrad shape |
| `final_down_2880.hsaco` | Optimized output for down_dgrad (produced by build.sh) |
| `final_gate_up_5760.hsaco` | Optimized output for gate_up_dgrad (produced by build.sh) |
| `build.sh` | Applies patches to produce final `.hsaco` files |
| `check.py` | Cosine-similarity correctness check vs Triton |
| `bench.py` | TFLOPS/bandwidth/latency benchmark |
