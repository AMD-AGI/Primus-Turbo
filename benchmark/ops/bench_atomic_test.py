"""Test atomic_max overhead in cast+transpose kernel."""

import torch
import torch.utils.benchmark as benchmark
import triton
import triton.language as tl

M, K = 8192, 8192
dtype = torch.bfloat16
device = "cuda"
fp8_dtype = torch.float8_e4m3fn
fp8_max = torch.finfo(fp8_dtype).max


# TE-style kernel with atomic_max (using TE's block sizes)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),  # TE's config
    ],
    key=["M", "N"],
)
@triton.jit
def te_style_kernel(
    x_ptr,
    y_ptr,
    y_t_ptr,
    scale_ptr,
    amax_ptr,
    scale_inv_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_ytm,
    stride_ytn,
    FP8_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    scale = tl.load(scale_ptr)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_scaled = x * scale
    x_fp8 = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX).to(y_ptr.dtype.element_ty)

    # Store regular
    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y_ptrs, x_fp8, mask=mask)

    # Store transposed
    y_t_ptrs = y_t_ptr + rn[None, :] * stride_ytm + rm[:, None] * stride_ytn
    tl.store(y_t_ptrs, x_fp8, mask=mask)

    # Atomic amax (like TE)
    amax = tl.max(tl.abs(x))
    tl.atomic_max(amax_ptr, amax, sem="relaxed")

    if pid == 0:
        tl.store(scale_inv_ptr, 1.0 / scale)


# Without atomic_max (using TE's block sizes)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),  # TE's config
    ],
    key=["M", "N"],
)
@triton.jit
def no_atomic_kernel(
    x_ptr,
    y_ptr,
    y_t_ptr,
    scale_ptr,
    scale_inv_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_ytm,
    stride_ytn,
    FP8_MAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    scale = tl.load(scale_ptr)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rn[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    x_scaled = x * scale
    x_fp8 = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX).to(y_ptr.dtype.element_ty)

    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    tl.store(y_ptrs, x_fp8, mask=mask)

    y_t_ptrs = y_t_ptr + rn[None, :] * stride_ytm + rm[:, None] * stride_ytn
    tl.store(y_t_ptrs, x_fp8, mask=mask)

    if pid == 0:
        tl.store(scale_inv_ptr, 1.0 / scale)


if __name__ == "__main__":
    # Test
    a = torch.randn((M, K), dtype=dtype, device=device)
    scale = torch.tensor([fp8_max / a.abs().max().item()], dtype=torch.float32, device=device)
    a_fp8 = torch.empty((M, K), dtype=fp8_dtype, device=device)
    a_t = torch.empty((K, M), dtype=fp8_dtype, device=device)
    amax = torch.zeros(1, dtype=torch.float32, device=device)
    scale_inv = torch.empty(1, dtype=torch.float32, device=device)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_N"]),)

    def run_with_atomic():
        amax.zero_()
        te_style_kernel[grid](
            a,
            a_fp8,
            a_t,
            scale,
            amax,
            scale_inv,
            M,
            K,
            a.stride(0),
            a.stride(1),
            a_fp8.stride(0),
            a_fp8.stride(1),
            a_t.stride(0),
            a_t.stride(1),
            FP8_MAX=fp8_max,
        )

    def run_no_atomic():
        no_atomic_kernel[grid](
            a,
            a_fp8,
            a_t,
            scale,
            scale_inv,
            M,
            K,
            a.stride(0),
            a.stride(1),
            a_fp8.stride(0),
            a_fp8.stride(1),
            a_t.stride(0),
            a_t.stride(1),
            FP8_MAX=fp8_max,
        )

    # Warmup
    for _ in range(10):
        run_with_atomic()
        run_no_atomic()
    torch.cuda.synchronize()

    t1 = benchmark.Timer(stmt="fn()", globals={"fn": run_with_atomic}).timeit(100).mean * 1e3
    t2 = benchmark.Timer(stmt="fn()", globals={"fn": run_no_atomic}).timeit(100).mean * 1e3

    print(f"cast+trans+atomic_max:  {t1:.4f} ms")
    print(f"cast+trans (no atomic): {t2:.4f} ms")
    print(f"atomic_max overhead:    {t1 - t2:.4f} ms")

    # Verify correctness
    run_with_atomic()
    torch.cuda.synchronize()
    print(f"\nComputed amax: {amax.item():.4f}, expected: {a.abs().max().item():.4f}")
