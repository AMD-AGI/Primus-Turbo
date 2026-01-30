"""Compare amax computation methods."""

import torch
import torch.utils.benchmark as benchmark
import triton
import triton.language as tl

from primus_turbo.triton.quantization.quantization_tensorwise import fast_amax


@triton.jit
def _amax_kernel_v2(
    x_ptr,
    partial_amax_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Two-level reduction: each block computes partial max."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_amax = tl.max(tl.abs(x))
    tl.store(partial_amax_ptr + pid, local_amax)


def triton_amax(x: torch.Tensor) -> torch.Tensor:
    """Compute amax using Triton with two-level reduction."""
    n_elements = x.numel()
    BLOCK_SIZE = 4096  # Larger block for better efficiency
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)

    # First level: each block computes partial max
    partial_amax = torch.empty(n_blocks, dtype=torch.float32, device=x.device)
    _amax_kernel_v2[(n_blocks,)](x, partial_amax, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Second level: reduce partial results (very small, can use PyTorch)
    return partial_amax.max()


def main():
    M, N = 8192, 8192
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    print("=" * 60)
    print(f"Comparing amax computation methods for {M}x{N} tensor")
    print("=" * 60)

    # Method 1: PyTorch abs().amax()
    def pytorch_amax():
        return x.abs().amax()

    # Method 2: Triton two-level (local)
    def triton_amax_fn():
        return triton_amax(x)

    # Method 3: Our fast_amax from quantization_tensorwise
    def our_fast_amax():
        return fast_amax(x)

    # Verify correctness
    ref = pytorch_amax()
    tri = triton_amax_fn()
    ours = our_fast_amax()
    print(f"\nCorrectness check:")
    print(f"  PyTorch abs().amax(): {ref.item():.6f}")
    print(f"  Triton two-level:     {tri.item():.6f}")
    print(f"  Our fast_amax:        {ours.item():.6f}")

    # Warmup
    for _ in range(20):
        pytorch_amax()
        triton_amax_fn()
        our_fast_amax()
    torch.cuda.synchronize()

    # Benchmark
    t1 = benchmark.Timer(stmt="fn()", globals={"fn": pytorch_amax})
    t2 = benchmark.Timer(stmt="fn()", globals={"fn": triton_amax_fn})
    t3 = benchmark.Timer(stmt="fn()", globals={"fn": our_fast_amax})

    m1 = t1.timeit(100)
    m2 = t2.timeit(100)
    m3 = t3.timeit(100)

    print(f"\nBenchmark results:")
    print(f"  PyTorch abs().amax(): {m1.mean * 1e3:.4f} ms")
    print(f"  Triton two-level:     {m2.mean * 1e3:.4f} ms")
    print(f"  Our fast_amax:        {m3.mean * 1e3:.4f} ms")

    # Breakdown: how much time we save per quantization
    print(f"\nTime saved per quantization: {(m1.mean - m3.mean) * 1e3:.4f} ms")
    print(f"  Forward has 2 quantizations: {2 * (m1.mean - m3.mean) * 1e3:.4f} ms saved")
    print(f"  Backward has 1 quantization: {1 * (m1.mean - m3.mean) * 1e3:.4f} ms saved")


if __name__ == "__main__":
    main()
