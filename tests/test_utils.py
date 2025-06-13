import torch


def is_ROCM():
    return torch.cuda.is_available() and torch.version.hip


# TODO: Need to check again whether these values are reasonable.
# TODO: fp8
def get_tolerances(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-5, atol=1e-5)
    elif dtype == torch.float16:
        return dict(rtol=1e-2, atol=1e-2)
    elif dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


###################################################################


# Relative Error
# Note: x is ref
def relative_error(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    return (torch.norm(x - y) / torch.norm(x)).detach().item()


# MSE Error
def mean_squared_error(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    return torch.mean((x - y) ** 2).item()


# Max Abs Error
def max_abs_error(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    return torch.max(torch.abs(x - y)).item()


# Cosine Similarity
def cosine_similarity(x: torch.Tensor, y: torch.Tensor):
    x, y = x.flatten().float(), y.flatten().float()
    return torch.nn.functional.cosine_similarity(x, y, dim=0).item()


# Symmetric Similarity
def symmetric_similarity_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# SNR
# Note: x is ref
def compute_snr(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    signal_power = torch.norm(x).pow(2)
    noise_power = torch.norm(x - y).pow(2)
    return 10 * torch.log10(signal_power / (noise_power + 1e-12)).detach().item()
