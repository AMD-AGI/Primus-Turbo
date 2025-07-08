import torch

__all__ = ["gemm"]


class GemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        transA: bool,
        transB: bool,
        out_dtype: torch.dtype,
    ):
        assert a.dim() == 2 and b.dim() == 2, "Only 2D GEMM is supported"
        # FWD
        # out    = a * b
        # [M, N] = [M, K] * [K, N]
        out = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(a, b, out_dtype, transA, transB, False)
        # Save for bwd
        if a.requires_grad or b.requires_grad:
            ctx.save_for_backward(a, b)
            ctx.transA = transA
            ctx.transB = transB
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b = ctx.saved_tensors

        # AGrad
        # grad_a = grad_out * b^T
        grad_a = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(
            grad_out, b, a.dtype, False, not ctx.transB, ctx.transA
        )

        # BGrad
        # grad_b = a^T * grad_out
        grad_b = torch.ops.primus_turbo_cpp_extension.hipblaslt_gemm(
            a, grad_out, b.dtype, not ctx.transA, False, ctx.transB
        )

        return grad_a, grad_b, None, None, None


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    transA: bool = False,
    transB: bool = False,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2, "Only 2D tensors are supported"
    if out_dtype is None:
        out_dtype = torch.result_type(a, b)
    return GemmFunction.apply(a, b, transA, transB, out_dtype)
