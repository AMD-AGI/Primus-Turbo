from functools import partial

import torch
import torch.nn.functional as F


def glu(x, act_func):
    x = torch.chunk(x, 2, dim=-1)
    return act_func(x[0]) * x[1]


def activation_func_with_probs(x, probs, act_func):
    dtype = x.dtype
    res = act_func(x) * probs
    return res.to(dtype)


def activation_with_probs_ref(x: torch.Tensor, probs: torch.Tensor, activation_type: str) -> torch.Tensor:
    if activation_type == "swiglu":
        return activation_func_with_probs(x, probs, partial(glu, act_func=F.silu))
    elif activation_type == "geglu":
        return activation_func_with_probs(x, probs, partial(glu, act_func=F.gelu))
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")
