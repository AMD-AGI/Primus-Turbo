"""Compare candidate gradients with the pinned baseline before a training run.

The baseline file path is supplied by the workflow's immutable source bundle.
Timing results are deliberately left to the PyTorch-profiled training workload.
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch

from primus_turbo.flydsl.attention.flash_attn_bwd import flydsl_varlen_backward
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import flash_attn_sbhd_flydsl_forward_impl
from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func


def relative_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def main():
    spec = importlib.util.spec_from_file_location("attention_campaign_baseline", sys.argv[1])
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    GlobalBackendManager.set_attn_backend(BackendType.FLYDSL, PrecisionType.BF16_FP16_FP32)
    torch.manual_seed(30279)
    results = []
    cases = [(512, -1, 16), (512, 128, 16), (8192, -1, 1), (8192, 128, 1)]
    cases = [(seq, window, spike, False) for seq, window, spike in cases]
    cases.append((8192, -1, 1, True))
    for seq, window, spike, deterministic in cases:
        batch, hq, hkv, dim = 4, 64, 8, 64
        q = torch.randn(seq, batch, hq, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq, batch, hkv, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        if spike > 1:
            q[::127] *= spike
            k[::131] *= spike
        sink = torch.randn(hq, device="cuda")
        out, lse = flash_attn_sbhd_flydsl_forward_impl(
            q, k, v, return_lse=True, window_size=(window, 0), sink=sink
        )
        lse = lse.view(batch, seq, hq).permute(0, 2, 1)
        dout = torch.randn_like(out)
        args = (dout, q, k, v, out, lse, batch, seq, seq, hq, hkv, dim, 0.125)
        kwargs = dict(sbhd=True, window_left=window, sink=sink, deterministic=deterministic)
        ref = baseline.flydsl_varlen_backward(*args, **kwargs)
        repeat = baseline.flydsl_varlen_backward(*args, **kwargs)
        candidate = flydsl_varlen_backward(*args, **kwargs)
        torch.cuda.synchronize()
        gradients = {}
        for name, expected, repeated, actual in zip(("dq", "dk", "dv", "dsink"), ref, repeat, candidate):
            assert torch.isfinite(expected).all(), f"baseline {name} is nonfinite"
            assert torch.isfinite(actual).all(), f"candidate {name} is nonfinite"
            noise = relative_l2(repeated, expected)
            error = relative_l2(actual, expected)
            if deterministic:
                assert torch.equal(repeated, expected), ("baseline deterministic repeat", name, noise)
            tolerance = 0.0 if deterministic else max(0.002, 4 * noise)
            gradients[name] = dict(
                relative_l2=error,
                baseline_repeat_relative_l2=noise,
                tolerance=tolerance,
                exact=torch.equal(actual, expected),
            )
            assert error <= tolerance, (seq, window, spike, name, gradients[name])
        # Exercise the public autograd path as well as the raw backward entry:
        # saved-Q gradients must still flow into the original Q/K/V inputs.
        original_q = q.clone()
        for tensor in (q, k, v, sink):
            tensor.requires_grad_(True)
        public_out = flash_attn_func(
            q.permute(1, 0, 2, 3),
            k.permute(1, 0, 2, 3),
            v.permute(1, 0, 2, 3),
            causal=True,
            window_size=(window, 0),
            sink=sink,
            deterministic=deterministic,
        )
        assert torch.equal(public_out.permute(1, 0, 2, 3), out), "public forward output changed"
        public_grads = torch.autograd.grad(public_out, (q, k, v, sink), dout.permute(1, 0, 2, 3))
        assert torch.equal(q, original_q), "public attention changed original Q"
        autograd_gradients = {}
        for name, expected, actual in zip(("dq", "dk", "dv", "dsink"), ref, public_grads):
            error = relative_l2(actual, expected)
            assert torch.isfinite(actual).all() and error <= gradients[name]["tolerance"], (
                seq,
                window,
                name,
                error,
            )
            autograd_gradients[name] = dict(relative_l2=error, exact=torch.equal(actual, expected))
        results.append(
            dict(
                sequence=seq,
                window=window,
                spike=spike,
                deterministic=deterministic,
                gradients=gradients,
                exact_public_output=True,
                autograd_gradients=autograd_gradients,
            )
        )
        print("ATTENTION_GRADIENT_CASE " + json.dumps(results[-1]), flush=True)
        del q, k, v, out, lse, dout, args, kwargs, ref, repeat, candidate
        del original_q, public_out, public_grads
    Path("/results/attention_backward_preflight.json").write_text(
        json.dumps(dict(passed=True, cases=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
