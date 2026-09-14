"""Is gfx1250 dense GEMM actually slow, or is the tuned hipBLASLt simply off the search path?

Yesterday's platform escalation records dense bf16 GEMM at 27.4 TFLOP/s against a measured
Triton roof of 1002.7 (at 1100 MHz), and attributes it to a missing gfx1250 Tensile library.
But the image was built with TUNED_HIPBLASLT=1 and its build step asserts >=46 gfx1250 bf16
solutions are present -- they are, under _rocm_sdk_libraries_gfx1250/lib, which is NOT on
LD_LIBRARY_PATH. The stock _rocm_sdk_devel copy is. So the escalation may be describing a
packaging bug rather than a platform gap, and that is a five-minute question.

Run this once per library configuration; the caller varies LD_LIBRARY_PATH and
TORCH_BLAS_PREFER_HIPBLASLT and compares.
"""
import json, os, sys

import torch

SHAPES = [
    # square anchor, the figure the escalation quotes
    ("8192^3", 8192, 8192, 8192),
    # the three that actually dominate a Llama-3.1-8B step at mbs4 seq8k
    ("qkv_proj", 32768, 6144, 4096),
    ("mlp_up", 32768, 14336, 4096),
    ("mlp_down", 32768, 4096, 14336),
]


def timed_ms(fn, iters=10, warmup=3):
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts, s, e = [], torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        flush.zero_()
        torch.cuda.synchronize()
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    torch.manual_seed(0)
    out = {
        "prefer_hipblaslt": os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT", "(unset)"),
        "ld_library_path_head": os.environ.get("LD_LIBRARY_PATH", "").split(":")[0],
        "torch": torch.__version__,
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "gemms": {},
    }
    for name, m, n, k in SHAPES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        try:
            ms = timed_ms(lambda: torch.mm(a, b))
            out["gemms"][name] = {
                "m": m, "n": n, "k": k, "ms": ms,
                "tflops": 2 * m * n * k / (ms * 1e-3) / 1e12,
            }
        except Exception as exc:  # a missing Tensile library surfaces here, not as slowness
            out["gemms"][name] = {"m": m, "n": n, "k": k, "error": f"{type(exc).__name__}: {exc}"[:200]}
        del a, b
        torch.cuda.empty_cache()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
