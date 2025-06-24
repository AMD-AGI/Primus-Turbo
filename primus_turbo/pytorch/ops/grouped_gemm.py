import torch


# TODO:
class GroupedGemmFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,  # [M, K],
        b: torch.Tensor,  # [B, N, K]
        seg_lens: torch.Tensor,  # [B,] int64
    ):
        # TODO:
        raise NotImplementedError("GroupedGemmFunc forward is not implemented")

    @staticmethod
    def backward(ctx):
        # TODO:
        raise NotImplementedError("GroupedGemmFunc backward is not implemented")


# TODO:
def grouped_gemm(a, b, seg_lens):
    return GroupedGemmFunc.apply(a, b, seg_lens)
