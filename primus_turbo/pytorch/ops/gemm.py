import torch

from primus_turbo.pytorch.kernels.gemm.gemm_csrc_impl import gemm_csrc_impl
from primus_turbo.pytorch.kernels.gemm.gemm_triton_impl import gemm_triton_imlp

__all__ = ["gemm"]


class GemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return gemm_triton_imlp(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = gemm_triton_imlp(grad_output, b, "NT") if ctx.needs_input_grad[0] else None
        grad_b = gemm_triton_imlp(a, grad_output, "TN") if ctx.needs_input_grad[1] else None
        return grad_a, grad_b


# TODO: out
# @torch.compile(fullgraph=True, mode="max-autotune")
def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ """
    return GemmFunction.apply(a, b)


# def gemm_fp8():
#     """ """
#     pass
