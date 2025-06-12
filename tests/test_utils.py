import torch


# TODO: Need to check again whether these values are reasonable.
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
    return (torch.norm(x - y) / torch.norm(x)).item()


# Max Abs Error
def max_abs_error(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    return torch.max(torch.abs(x - y)).item()


# MSE Error
def mean_squared_error(x: torch.Tensor, y: torch.Tensor):
    x, y = x.float(), y.float()
    return torch.mean((x - y) ** 2).item()


# Cosine Similarity
def cosine_similarity(x: torch.Tensor, y: torch.Tensor):
    x, y = x.flatten().float(), y.flatten().float()
    cosim = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    return 1 if cosim > 1 else cosim


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
    return 10 * torch.log10(signal_power / (noise_power + 1e-12)).item()


def ulp_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    if x.dtype in [ torch.float16, torch.bfloat16 ]:
        x_bits = x.view(torch.int16)
        y_bits = y.view(torch.int16)
    elif x.dtype in [torch.float32]:
        x_bits = x.view(torch.int32)
        y_bits = y.view(torch.int32)
    else:
        raise ValueError("Not support dtype.")

    def to_ordered(bits : torch.Tensor):
        if bits.element_size() == 4:
            magic_number = 0x80000000
        elif bits.element_size() == 2:
            magic_number = 0x8000
        else:
            raise ValueError("Not support bits.")

        return torch.where(bits < 0, magic_number - bits, bits)

    return (to_ordered(x_bits) - to_ordered(y_bits)).abs()
