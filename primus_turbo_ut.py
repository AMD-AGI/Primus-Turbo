import os
import random

import numpy as np
import torch
import transformer_engine.pytorch as te

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
    ScalingStrategy,
)


def set_global_seed(seed: int = 42):
    # Set the random seed for numpy and random
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)

    # Set the random seed for CUDA (if applicable)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_global_seed(42)


def make_group_offs(group_lens):
    group_lens = group_lens.to(torch.long)
    offs = torch.zeros((group_lens.numel() + 1,), device=group_lens.device, dtype=torch.long)
    offs[1:] = torch.cumsum(group_lens, dim=0)
    return offs


def compute_snr(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    ref = ref.float()
    test = test.float()
    signal_power = (ref**2).mean()
    noise_power = ((ref - test) ** 2).mean()
    if noise_power == 0:
        return float("inf")
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


def te_weights_to_3d(te_layer, E):
    return torch.stack([getattr(te_layer, f"weight{e}") for e in range(E)], dim=0)


def te_grads_to_3d(te_layer, E):
    return torch.stack([getattr(te_layer, f"weight{e}").grad for e in range(E)], dim=0)


E = 8
in_features = 2048
out_features = 8192
device = "cuda"
dtype = torch.bfloat16

S = 8192 * 2


# TE
te_layer = te.GroupedLinear(
    num_gemms=E,
    in_features=in_features,
    out_features=out_features,
    bias=False,
    device=None,
    params_dtype=dtype,
).to(device)


# Turbo
class TurboGL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((E, out_features, in_features), device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if int(os.environ.get("TEST_BLOCKWISE", "0")) == 1:
            print("test block-wise")
            granularity = ScalingGranularity.BLOCKWISE
        else:
            print("test tensor-wise")
            granularity = ScalingGranularity.TENSORWISE

        self.cfg = Float8QuantConfig(
            format=Format.E4M3,
            granularity=granularity,
            strategy=ScalingStrategy.DYNAMIC,
            scale_dtype=ScaleDtype.FP32,
            block_size=128,
        )

    def forward(self, a, lens):
        return turbo.ops.grouped_gemm_fp8(a, self.weight, lens, None, trans_b=True, config=self.cfg)


turbo_layer = TurboGL()

# copy weight
with torch.no_grad():
    turbo_layer.weight.copy_(te_weights_to_3d(te_layer, E))

g = torch.cuda.CUDAGraph()

tb_input = torch.zeros((S, in_features), device=device, dtype=dtype)
group_lens = torch.tensor([0] * E, device=device, dtype=torch.long)

for i in range(5):

    te_input = torch.randn((S, in_features), device=device, dtype=dtype)
    tb_input.copy_(te_input)

    if int(os.environ.get("TEST_ROUTING_MODE", "0")) == 0:
        print("balance routing")
        grp_lens = [S // E] * E
    elif int(os.environ.get("TEST_ROUTING_MODE", "0")) == 1:
        print("only first two experts")
        grp_lens = [8192, 8192, 0, 0, 0, 0, 0, 0]
    elif int(os.environ.get("TEST_ROUTING_MODE", "0")) == 2:
        print("dynamic routing")
        grp_lens = [0] * E
        for s in range(S):
            grp_lens[random.randint(0, E - 1)] += 1

    group_lens.copy_(torch.tensor(grp_lens, device=device, dtype=torch.long))

    # forward
    y_te = te_layer(te_input, group_lens.tolist())
    loss_te = y_te.float().square().mean()
    te_layer.zero_grad(set_to_none=True)
    loss_te.backward()

    if int(os.environ.get("TEST_HIPGRAPH", "0")) == 1:
        print("enable hipgraph")
        if i == 0:
            # Warmup run to JIT compile Triton kernels before graph capture
            y_tb = turbo_layer(tb_input, group_lens)
            loss_tb = y_tb.float().square().mean()
            turbo_layer.zero_grad(set_to_none=True)
            loss_tb.backward()
            del y_tb, loss_tb
            torch.cuda.synchronize()
            # Now capture graph
            with torch.cuda.graph(g):
                y_tb = turbo_layer(tb_input, group_lens)
                loss_tb = y_tb.float().square().mean()
                turbo_layer.zero_grad(set_to_none=True)
                loss_tb.backward()
            # Replay immediately to get actual results for first iteration
            g.replay()
        else:
            g.replay()
    else:
        print("disable hipgraph")
        y_tb = turbo_layer(tb_input, group_lens)
        loss_tb = y_tb.float().square().mean()
        turbo_layer.zero_grad(set_to_none=True)
        loss_tb.backward()

    # compare forward, grads
    with torch.no_grad():
        fwd_abs = (y_te - y_tb).abs().mean().item()
        fwd_rel = ((y_te - y_tb).abs().mean() / (y_te.abs().mean() + 1e-6)).item()
        fwd_snr = compute_snr(y_te, y_tb)
        print("FWD mean_abs_diff", fwd_abs, "mean_rel_diff", fwd_rel, "SNR", f"{fwd_snr:.2f} dB")

        print(y_te.reshape([-1])[:100], "\n", y_tb.reshape([-1])[:100])

        te_g = te_grads_to_3d(te_layer, E)
        tb_g = turbo_layer.weight.grad
        wg_abs = (te_g - tb_g).abs().mean().item()
        wg_rel = ((te_g - tb_g).abs().mean() / (te_g.abs().mean() + 1e-6)).item()
        wg_snr = compute_snr(te_g, tb_g)
        print("WGRAD mean_abs_diff", wg_abs, "mean_rel_diff", wg_rel, "SNR", f"{wg_snr:.2f} dB")
