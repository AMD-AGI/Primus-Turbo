# MI300X / MI325X (`gfx942`, CDNA3) Optimization Guide

`MI300X` and `MI325X` both target `gfx942`. They share the same ISA, CU layout, and core optimization constraints. The biggest SKU-level differences are HBM capacity and bandwidth.

## Core Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | CDNA3 |
| LLVM / offload-arch | `gfx942` |
| Compute Units | `304` active (`8 XCD x 38`) |
| Wavefront size | `64` |
| LDS / CU | `64 KiB` |
| LDS banks | `32` |
| HBM | `192 GiB` HBM3 (`MI300X`) |
| L2 cache | `32 MiB` total (`4 MiB / XCD`) |
| Infinity Cache | `256 MiB` |
| Matrix peak (official rated) | FP16/BF16 `1307.4 TFLOPS`, FP8 `2614.9 TFLOPS`, FP64 `163.4 TFLOPS` |
| Max engine clock | `2100 MHz` |
| FP8 ecosystem | FNUZ |

### Variant Notes

| Item | MI300X | MI325X |
|------|--------|--------|
| offload-arch | `gfx942` | `gfx942` |
| CU / Architecture | `304` / CDNA3 | `304` / CDNA3 |
| VRAM | `192 GiB HBM3` | `256 GiB HBM3E` |
| Peak HBM BW | `5.3 TB/s` class | `6.0 TB/s` class |

## Per-Clock, Per-CU Throughput

| Computation | FLOPS / clock / CU |
|-------------|--------------------|
| Vector FP64 | `128` |
| Vector FP32 | `256` |
| Vector FP16 | `256` |
| Matrix FP64 | `256` |
| Matrix FP32 | `256` |
| Matrix FP16 / sparsity | `2048` |
| Matrix BF16 / sparsity | `2048` |
| Matrix FP8 / sparsity | `4096` |
| Matrix INT8 / sparsity | `4096` |

## ISA Capability Summary

MI300X's CDNA3 ISA matters in practice because it defines both what you can optimize for and what you should not assume exists:

| Area | MI300X ISA capability |
|------|------------------------|
| Core ISA families | SALU, VALU, packed math, VGPR indexing, DPP / SDWA, scalar + vector memory ops, flat / global / scratch, LDS + GWS |
| Matrix core | Dense and `2:4` sparse `MFMA` with `AccVGPR` accumulation and explicit `V_ACCVGPR_*` moves |
| Matrix datatypes | `FP64`, `FP32`, `FP16`, `BF16`, `INT8`, `BF8`, `FP8`, and CDNA3 `TF32` / `XF32` style reduced-precision matrix paths |
| Low-precision convert | Packed `BF8` / `FP8` conversion instructions in the vector ISA |
| Memory staging | `Memory Buffer Load to LDS` exists, but LDS is still only `64 KiB` with `32` banks |
| Missing vs CDNA4 | No `FP6` / `FP4` / block-scaled `MXFP`, no `MFMA Transpose Load from LDS` |

## Optimization Implications from the ISA

- On MI300X, the default tensor-core mental model is classic `MFMA` plus careful buffer-to-LDS staging.
- `TF32` / `XF32` matrix support exists on CDNA3, so mixed-precision FP32-like workloads have a real hardware path here.
- Sparse `2:4` matrix support is a meaningful optimization direction when the algorithm can enforce that structure.
- FP8 is available, but the software-visible format must still be validated carefully because `gfx942` stacks often surface FP8 through FNUZ-flavored types.
- Because the ISA does not provide CDNA4 block-scaled low-precision or transpose-load helpers, compensate with better layout, staging, and scheduling rather than trying to imitate MI355X-specific patterns directly.

## Partition and NPS Modes

MI300X-class devices expose `8 XCDs` and support multiple partition modes:

| Mode | Description | Typical use |
|------|-------------|-------------|
| `SPX` | Full card as one GPU | Default single-card tuning |
| `DPX` | `2` partitions of `4` XCDs | Dual-tenant or split inference |
| `CPX` | `8` independent partitions | Strong isolation / multi-tenant |

NPS changes memory interleaving and locality. If the deployment target uses `CPX`, `DPX`, or non-default NPS settings, re-profile under that exact mode before trusting results from `SPX`.

## Compilation

```bash
# HIP
hipcc -O3 --offload-arch=gfx942 kernel.cpp

# Multi-target
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 kernel.cpp

# CK
# AMDGPU_TARGETS=gfx942
```

## Good Starting Points

| Path | Parameter | Starting point |
|------|-----------|----------------|
| Triton | `BLOCK_SIZE` | `128` or `256` |
| Triton | `num_warps` | `4-8` |
| HIP | Thread block | `256` threads |
| CK BF16 GEMM | `block_m, block_n, block_k` | `256, 128, 64` |

## Healthy Ranges

| Metric | Good | Needs attention |
|--------|------|-----------------|
| Occupancy | `> 50%` | `< 25%` |
| HBM BW utilization | `> 60%` on memory-bound kernels | `< 30%` |
| MFMA utilization | `> 70%` on compute-bound kernels | `< 40%` |
| LDS bank conflicts | `< 5%` | `> 20%` |
| Register spill | `0` scratch waves | Any scratch |

## Hardware-Specific Implications

- `304` CUs means small grids often underutilize the machine. Persistent kernels or merged launches can matter more than tiny instruction tweaks.
- Wavefront size is `64`, not `32`. CUDA-derived reduction trees, shuffle logic, and occupancy mental models must be adjusted.
- LDS is `64 KiB` per CU. Double buffering is valuable but easy to overuse; always re-check occupancy after enlarging tiles or staging buffers.
- CDNA3 FP8 commonly uses `FNUZ`. If you port code or validation from `gfx950`, treat format conversion as a correctness risk first and a performance topic second.
- Cache hierarchy is deep enough that traversal order and L2 reuse matter. Re-check data order before assuming the kernel is purely compute-bound.
- If the profiler suggests register pressure or LDS pressure, believe it quickly: MI300X does not have CDNA4's extra LDS headroom or scale-aware low-precision ISA escape hatches.

## System Tuning Notes

### BIOS / platform

| Setting | Recommended value | Why |
|---------|-------------------|-----|
| Above 4G Decoding | Enabled | Required for GPU BAR |
| SR-IOV | Enabled | Virtualization support |
| SMT Control | Disable for compute-heavy benchmarking | Reduces host-side noise |
| xGMI Max Speed | `32 Gbps` | Full fabric bandwidth |
| IOMMU | Enabled | Required for ROCm |

### Runtime

```bash
# Reduce launch variance where supported
export HIP_FORCE_DEV_KERNARG=1
export HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0

# Optional deterministic clocks for benchmarking
rocm-smi --setperfdeterminism 1900
```

## Multi-GPU Notes

```bash
export NCCL_MIN_NCHANNELS=112

# Often useful when benchmarking CPX deployments
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1
export RCCL_MSCCLPP_THRESHOLD=$((2*1024*1024*1024))
export MSCCLPP_READ_ALLRED=1
```

## Common Mistakes

- Building without `-O3`.
- Porting a kernel that assumes warp size `32`.
- Optimizing around the wrong CU count.
- Treating CDNA3 FP8 as interchangeable with CDNA4 FP8.
- Increasing LDS use without checking occupancy or bank conflicts.
- Reusing partition-mode measurements on a differently partitioned deployment.

## Cross References

- Portable ISA digest: `isa-summary.md`
- Cross-generation comparison: `../hardware-comparison.md`
- CDNA4 counterpart: `../gfx950/optimization-guide.md`

## Original Source PDFs

- `amd-instinct-mi300-cdna3-instruction-set-architecture.pdf`
- `amd-cdna-3-white-paper.pdf`
