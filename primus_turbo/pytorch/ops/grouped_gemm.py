import torch

from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_csrc_impl import (
    grouped_gemm_csrc_impl,
    grouped_gemm_variable_k_csrc_impl,
)

__all__ = ["grouped_gemm"]


@torch.compile
def compute_group_offs(group_lens: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [torch.tensor([0], device=group_lens.device, dtype=group_lens.dtype), group_lens.cumsum(0)]
    )


class GroupedGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        group_lens: torch.Tensor,  # [B,] int64
        group_offs: torch.Tensor,  # [B+1,] int64
        trans_b: bool = False,
    ):
        out = grouped_gemm_csrc_impl(
            a,
            b,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=trans_b,
        )
        ctx.save_for_backward(a, b, group_lens, group_offs)
        ctx.trans_a = False
        ctx.trans_b = trans_b
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b, group_lens, group_offs = ctx.saved_tensors
        grad_a = grouped_gemm_csrc_impl(
            grad_out,
            b,
            group_lens,
            group_offs,
            trans_a=False,
            trans_b=not ctx.trans_b,
        )

        lhs, rhs = (grad_out, a) if ctx.trans_b else (a, grad_out)
        grad_b = grouped_gemm_variable_k_csrc_impl(
            lhs,
            rhs,
            group_lens,
            group_offs,
            trans_a=True,
            trans_b=False,
        )
        return grad_a, grad_b, None, None, None, None


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    group_lens: torch.Tensor,
    group_offs: torch.Tensor | None = None,
    trans_b: bool = False,
) -> torch.Tensor:
    """
    Grouped GEMM.

    Args:
        a (torch.Tensor): Shape [sum(group_lens), K], DType float16/bfloat16.
        b (torch.Tensor): Shape [G, K, N] (or [G, N, K] if trans_b=True), DType float16/bfloat16.
        group_lens (torch.Tensor): Rows per expert of shape [G], int64. sum(group_lens) == a.size(0).
        group_offs (torch.Tensor | None): Exclusive prefix-sum of group_lens, shape [G+1].
                                          If None, it will be computed internally.
        trans_b (bool): If True, treat each b[g] as transposed.

    Returns:
        torch.Tensor: Output of shape [sum(group_lens), N], same dtype/device as `a`.

    Example:
        >>> G, K, N = 3, 128, 64
        >>> group_lens = torch.tensor([32, 16, 48], dtype=torch.long, device="cuda")
        >>> a = torch.randn(group_lens.sum().item(), K, device="cuda", dtype=torch.bfloat16)
        >>> b = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16)  # or [G, N, K] with trans_b=True
        >>> out = grouped_gemm(a, b, group_lens)  # [96, 64]
        >>> out.shape
        torch.Size([96, 64])
    """
    if group_offs is None:
        group_offs = compute_group_offs(group_lens)

    return GroupedGemmFunc.apply(a, b, group_lens, group_offs, trans_b)
