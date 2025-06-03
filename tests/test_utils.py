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
