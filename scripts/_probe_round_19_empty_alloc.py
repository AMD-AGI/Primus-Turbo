#!/usr/bin/env python3
"""R19 — probe torch.empty alternatives for the output buffer alloc.

R18 stage profile pinned the alloc cost at 1.68 µs (44% of remaining
host overhead). Try every viable empty-alloc API and time them in a
tight Python-only loop.

Goal: pick the fastest API that's bit-equivalent (uncontracted
contiguous buffer of right (size, dtype, device)). The bench is a
plain Python loop; allocations don't queue GPU work so wall time ==
host CPU time.
"""
import os
import sys
import time

os.environ.setdefault("PRIMUS_TURBO_HIPKITTEN_PATH", "/workspace/code/HipKittens")
sys.path.insert(0, "/workspace/code/Primus-Turbo")

import torch  # noqa: E402

import primus_turbo.pytorch as turbo  # noqa: F401  E402


def _bench_py(fn, iters=20000):
    fn()  # warmup
    deltas = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        deltas.append(time.perf_counter_ns() - t0)
    deltas.sort()
    return deltas[len(deltas) // 5] / 1000.0  # ns → µs


def main():
    print(f"[probe] R19 — output buffer alloc API benchmark")
    print(f"[probe] device: {torch.cuda.get_device_name(0)}")
    print()

    # Target shape: Down-B4-M2048 fwd output: (8192, 2880) bf16 cuda.
    M_total, N = 8192, 2880
    out_dtype = torch.bfloat16

    # Reference tensor for "based-on" alloc APIs.
    base = torch.empty((M_total, 2880), dtype=torch.float8_e4m3fn, device="cuda")
    dev = base.device

    print(f"  target: shape=({M_total}, {N})  dtype={out_dtype}  device={dev}")
    print()

    # ---- A: current public torch.empty -----------------------------------
    def a_current():
        return torch.empty((M_total, N), dtype=out_dtype, device=dev)
    print(f"  [A] torch.empty(size, dtype=, device=):           {_bench_py(a_current):.3f} µs")

    # ---- B: torch.empty with positional --------------------------------
    def b_pos():
        return torch.empty(M_total, N, dtype=out_dtype, device=dev)
    print(f"  [B] torch.empty(M, N, dtype=, device=):           {_bench_py(b_pos):.3f} µs")

    # ---- C: tensor.new_empty -------------------------------------------
    def c_new_empty():
        return base.new_empty((M_total, N), dtype=out_dtype)
    print(f"  [C] base.new_empty(size, dtype=):                 {_bench_py(c_new_empty):.3f} µs")

    # ---- D: torch._C._VariableFunctions.empty (legacy) ------------------
    try:
        _empty = torch._C._VariableFunctions.empty
        def d_c_var():
            return _empty([M_total, N], dtype=out_dtype, device=dev)
        print(f"  [D] torch._C._VariableFunctions.empty(...):       {_bench_py(d_c_var):.3f} µs")
    except (AttributeError, Exception) as e:
        print(f"  [D] torch._C._VariableFunctions.empty: SKIP {e}")

    # ---- E: torch.empty with str device ---------------------------------
    def e_str_dev():
        return torch.empty((M_total, N), dtype=out_dtype, device="cuda")
    print(f"  [E] torch.empty(size, dtype=, device='cuda'):     {_bench_py(e_str_dev):.3f} µs")

    # ---- F: torch.empty no device kw (default) --------------------------
    # Won't work — would create CPU tensor. Skip.

    # ---- G: AT::empty via _C low-level (if exposed) ----------------------
    try:
        empty_strided = torch._C._VariableFunctions.empty_strided
        # contiguous strides for (M, N): (N, 1)
        def g_strided():
            return empty_strided([M_total, N], [N, 1], dtype=out_dtype, device=dev)
        print(f"  [G] empty_strided(size, stride, dtype=, device=): {_bench_py(g_strided):.3f} µs")
    except (AttributeError, Exception) as e:
        print(f"  [G] empty_strided: SKIP {e}")

    # ---- H: torch.empty with memory_format -------------------------------
    def h_memfmt():
        return torch.empty((M_total, N), dtype=out_dtype, device=dev,
                           memory_format=torch.contiguous_format)
    print(f"  [H] torch.empty(... memory_format=cf):            {_bench_py(h_memfmt):.3f} µs")

    # ---- I: cached size tuple ---------------------------------------
    sz = (M_total, N)
    def i_cached_size():
        return torch.empty(sz, dtype=out_dtype, device=dev)
    print(f"  [I] torch.empty(<cached tuple>, ...):             {_bench_py(i_cached_size):.3f} µs")

    print()

    # ---- Verify dtype/shape/device for each ------------------------------
    print("[probe] sanity check (shape, dtype, device, is_contig):")
    for name, fn in [
        ("A current", a_current),
        ("C new_empty", c_new_empty),
        ("E str dev", e_str_dev),
    ]:
        t = fn()
        print(f"  [{name}]  shape={tuple(t.shape)}  dtype={t.dtype}  device={t.device}  contig={t.is_contiguous()}")


if __name__ == "__main__":
    main()
