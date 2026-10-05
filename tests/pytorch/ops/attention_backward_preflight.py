"""Compare candidate gradients with the pinned baseline before a training run.

The baseline file path is supplied by the workflow's immutable source bundle.
Timing results are deliberately left to the PyTorch-profiled training workload.
"""

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch

from primus_turbo.flydsl.attention.flash_attn_bwd import flydsl_varlen_backward
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import flash_attn_sbhd_flydsl_forward_impl


def relative_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def main():
    spec = importlib.util.spec_from_file_location("attention_campaign_baseline", sys.argv[1])
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    torch.manual_seed(30279)
    results = []
    cases = [(512, -1, 16), (512, 128, 16), (8192, -1, 1), (8192, 128, 1)]
    if os.getenv("PRIMUS_TURBO_ATTN_Q_PREP") in ("forward", "forward_hybrid", "lse"):
        cases += [(513, -1, 16), (513, 128, 16)]
    for seq, window, spike in cases:
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
        if os.getenv("PRIMUS_TURBO_ATTN_Q_PREP") == "lse":
            from primus_turbo.flydsl.attention.flash_attn_bwd import _prescale_q_lse

            original_q, original_lse = q.clone(), lse.clone()
            scaled_q = torch.empty_like(q)
            scaled_lse = _prescale_q_lse(q, scaled_q, lse, 0.125, torch.cuda.current_stream())
            torch.cuda.synchronize()
            assert torch.equal(q, original_q) and torch.equal(lse, original_lse)
            assert torch.equal(scaled_q, torch.mul(q, 0.125 * math.log2(math.e))), "Q/LSE Q rounding"
            assert torch.equal(scaled_lse, torch.mul(lse, -math.log2(math.e))), "Q/LSE LSE rounding"
            del original_q, original_lse, scaled_q, scaled_lse
        dout = torch.randn_like(out)
        args = (dout, q, k, v, out, lse, batch, seq, seq, hq, hkv, dim, 0.125)
        kwargs = dict(sbhd=True, window_left=window, sink=sink)
        ref = baseline.flydsl_varlen_backward(*args, **kwargs)
        repeat = baseline.flydsl_varlen_backward(*args, **kwargs)
        if os.getenv("PRIMUS_TURBO_ATTN_Q_PREP") == "forward" or (
            os.getenv("PRIMUS_TURBO_ATTN_Q_PREP") == "forward_hybrid" and window < 0
        ):
            original_q = q.clone()
            saved_out, saved_lse, scaled_q = flash_attn_sbhd_flydsl_forward_impl(
                q,
                k,
                v,
                return_lse=True,
                window_size=(window, 0),
                sink=sink,
                return_scaled_q=True,
            )
            torch.cuda.synchronize()
            assert torch.equal(q, original_q), "forward modified its Q input"
            assert torch.equal(scaled_q, torch.mul(q, 0.125 * math.log2(math.e))), "saved Q rounding mismatch"
            assert torch.equal(out, saved_out), "saving Q changed forward output"
            assert torch.equal(lse, saved_lse.view(batch, seq, hq).permute(0, 2, 1)), "saving Q changed LSE"
            saved_args = (dout, scaled_q, k, v, out, lse, batch, seq, seq, hq, hkv, dim, 0.125)
            candidate = flydsl_varlen_backward(*saved_args, **kwargs, q_is_scaled=True)
            del original_q, saved_out, saved_lse, scaled_q, saved_args
        else:
            candidate = flydsl_varlen_backward(*args, **kwargs)
        torch.cuda.synchronize()
        gradients = {}
        for name, expected, repeated, actual in zip(("dq", "dk", "dv", "dsink"), ref, repeat, candidate):
            assert torch.isfinite(expected).all(), f"baseline {name} is nonfinite"
            assert torch.isfinite(actual).all(), f"candidate {name} is nonfinite"
            noise = relative_l2(repeated, expected)
            error = relative_l2(actual, expected)
            tolerance = max(0.002, 4 * noise)
            gradients[name] = dict(
                relative_l2=error,
                baseline_repeat_relative_l2=noise,
                tolerance=tolerance,
                exact=torch.equal(actual, expected),
            )
            assert error <= tolerance, (seq, window, spike, name, gradients[name])
        results.append(dict(sequence=seq, window=window, spike=spike, gradients=gradients))
        print("ATTENTION_GRADIENT_CASE " + json.dumps(results[-1]), flush=True)
        del q, k, v, out, lse, dout, args, kwargs, ref, repeat, candidate
    Path("/results/attention_backward_preflight.json").write_text(
        json.dumps(dict(passed=True, cases=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
