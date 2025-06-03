import torch


# Demo
def gemm_csrc_impl(a: torch.Tensor, b: torch.Tensor, layout: str = "NN"):
    assert layout in ["NN", "NT", "TN"], f"Unsupported layout: {layout}"
    if layout == "NN":
        a_mat, b_mat = a, b
    elif layout == "NT":
        a_mat, b_mat = a, b.transpose(-1, -2)
    elif layout == "TN":
        a_mat, b_mat = a.transpose(-1, -2), b

    return torch.ops.primus_turbo_cpp_extension.gemm(a_mat, b_mat)
