###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Precompile Primus-Turbo's FlyDSL GEMMs into FlyDSL's disk cache, with or without a GPU.

    python -m primus_turbo.flydsl.precompile [--cache-dir DIR] [--arch gfx950]

FlyDSL compiles a kernel on its first call (minutes for a model's shape set, on every rank), then reuses its disk cache
(``FLYDSL_RUNTIME_CACHE_DIR``). This fills that cache ahead of time -- e.g. in a Dockerfile ``RUN`` step, where no GPU is
visible -- so a run starts on cached kernels (and ``FLYDSL_RUNTIME_RUN_ONLY=1`` can prove it compiles nothing).

The cache key is FlyDSL's toolchain fingerprint, the target arch, the kernel function and its captured globals, the
argument dtypes / shapes / strides and a few FLYDSL_* env vars -- not the device -- so compiling against CPU tensors in
FlyDSL's COMPILE_ONLY mode writes the same entries a GPU run looks up (checked byte-for-byte). Without a GPU the
wrappers' device queries are stubbed for the duration: the stream object (only its type enters the key), the CUDA-graph
capture check (False) and the CU count (``--ncu``, which only picks a persistent tile count; production runs one tile
per WG). Shapes: the pinned MXFP4 shapes (``mxfp4_pinned.PINNED``) and the MXFP6 shapes below.
"""

import argparse
import contextlib
import os
import sys
import time

# Flux MXFP6 (A6W6) GEMM shapes (M, N, K) -> bias variants: forward and MXFP6 backward.
MXFP6_SHAPES = {
    # forward: single blocks at 16384 rows, joint streams at 8192 (QKV carries a bias)
    (16384, 12288, 3072): (False,),
    (16384, 9216, 3072): (True,),
    (8192, 12288, 3072): (False,),
    (8192, 9216, 3072): (True,),
    (8192, 3072, 3072): (False,),
    (8192, 3072, 12288): (False,),
    # MXFP6 backward (dgrad / wgrad; the fallback when the MXFP4 backward is off) and the forward-FP4-off linear2
    (16384, 3072, 3072): (False,),
    (16384, 3072, 9216): (False,),
    (16384, 3072, 12288): (False,),
    (8192, 3072, 9216): (False,),
    (3072, 3072, 8192): (False,),
    (3072, 3072, 16384): (False,),
    (9216, 3072, 8192): (False,),
    (9216, 3072, 16384): (False,),
    (12288, 3072, 8192): (False,),
    (12288, 3072, 16384): (False,),
    (3072, 12288, 8192): (False,),
    (3072, 12288, 16384): (False,),
}


@contextlib.contextmanager
def _device(ncu):
    """Yield the device to build operands on; without a GPU, stub the wrappers' device queries (see module doc)."""
    import torch

    if torch.cuda.is_available():
        yield "cuda"
        return
    import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as FK

    class _Stream:
        cuda_stream = 0

    saved = (torch.cuda.current_stream, torch.cuda.is_current_stream_capturing, list(FK._MXFP4_NCU))
    torch.cuda.current_stream = lambda *a, **k: _Stream()
    torch.cuda.is_current_stream_capturing = lambda: False
    FK._MXFP4_NCU[:] = [ncu]
    try:
        yield "cpu"
    finally:
        torch.cuda.current_stream, torch.cuda.is_current_stream_capturing, FK._MXFP4_NCU[:] = saved


def _mxfp4(dev):
    import torch

    import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as FK
    from primus_turbo.flydsl.gemm.mxfp4_pinned import PINNED

    for M, N, K in sorted(PINNED):
        a = torch.zeros(M, K // 2, dtype=torch.uint8, device=dev)
        b = torch.zeros(N, K // 2, dtype=torch.uint8, device=dev)
        sa = torch.zeros(M * K // 128, dtype=torch.int32, device=dev)
        sb = torch.zeros(N * K // 128, dtype=torch.int32, device=dev)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        FK.gemm_mxfp4_flydsl_kernel(
            a, sa, b, sb, out_dtype=torch.bfloat16, scales_prepacked=True, k=K, out=out
        )
        yield f"mxfp4 {M}x{N}x{K}"


def _mxfp6(dev):
    import torch

    from primus_turbo.flydsl.gemm.gemm_mxfp6_kernel import gemm_mxfp6_persistent

    for (M, N, K), variants in sorted(MXFP6_SHAPES.items()):
        steps = K // 128 + 2  # the mxfp6_c0c1_256_padk2 blob: 256-row tiles x (K/128 + 2 guard) steps
        a = torch.zeros(M // 256 * steps * 24576, dtype=torch.uint8, device=dev)
        b = torch.zeros(N // 256 * steps * 24576, dtype=torch.uint8, device=dev)
        sa = torch.zeros(M // 256 * steps * 1024, dtype=torch.uint8, device=dev)
        sb = torch.zeros(N // 256 * steps * 1024, dtype=torch.uint8, device=dev)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        for has_bias in variants:
            bias = torch.zeros(N, dtype=torch.bfloat16, device=dev) if has_bias else None
            gemm_mxfp6_persistent(
                a, None, b, None, sa, sb, out=out, tpw=1, bias=bias, layout="aiter", m=M, n=N, k=K
            )
            yield f"mxfp6 {M}x{N}x{K}{' bias' if has_bias else ''}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache-dir", help="FlyDSL cache dir (default: FLYDSL_RUNTIME_CACHE_DIR or FlyDSL's)")
    p.add_argument("--arch", default="gfx950", help="target arch when no GPU is visible")
    p.add_argument("--ncu", type=int, default=256, help="CU count to assume when no GPU is visible")
    p.add_argument("--only", choices=("mxfp4", "mxfp6"), help="one family only")
    args = p.parse_args(argv)
    if args.cache_dir:
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = args.cache_dir
    import torch

    if not torch.cuda.is_available():
        # Must be set before FlyDSL compiles anything: compile without launching, for this arch.
        os.environ["FLYDSL_COMPILE_COMPILE_ONLY"] = "1"
        os.environ.setdefault("FLYDSL_COMPILE_ARCH", args.arch)
        # Kernel construction also queries the arch (flydsl.runtime.device.get_rocm_arch), which without a GPU falls
        # back to a default rather than the target; pin it too so the kernels are built for --arch.
        os.environ.setdefault("FLYDSL_GPU_ARCH", args.arch)
    t0, n = time.time(), 0
    with _device(args.ncu) as dev:
        for fam, gen in (("mxfp4", _mxfp4), ("mxfp6", _mxfp6)):
            if args.only and fam != args.only:
                continue
            for what in gen(dev):
                n += 1
                print(f"[precompile] {what} ({time.time() - t0:.0f} s)", flush=True)
    print(
        f"[precompile] {n} FlyDSL GEMMs into {os.environ.get('FLYDSL_RUNTIME_CACHE_DIR', '~/.flydsl/cache')} "
        f"on {dev}, {time.time() - t0:.0f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
