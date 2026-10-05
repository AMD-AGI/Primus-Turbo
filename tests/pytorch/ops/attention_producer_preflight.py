"""Check producer Q cache, autograd routing, and packed-QKV gradients on the runner."""

import json
import math
from pathlib import Path

import torch

from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType
from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func
from primus_turbo.pytorch.ops.rope import fused_qkv_rmsnorm_rope


def relative_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def main():
    GlobalBackendManager.set_attn_backend(BackendType.FLYDSL, PrecisionType.BF16_FP16_FP32)
    torch.manual_seed(30279)
    results = []
    for seq, window in [(512, -1), (512, 128), (8192, -1), (8192, 128)]:
        qkv = torch.randn(seq, 4, 8, 640, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        original = qkv.detach().clone()
        qg = torch.ones(64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kg = torch.ones_like(qg, requires_grad=True)
        angles = torch.randn(seq, 1, 1, 32, device="cuda")
        freqs = torch.cat((angles, angles), dim=-1).contiguous()
        sink = torch.randn(64, device="cuda", requires_grad=True)
        dout = torch.randn(4, seq, 64, 64, device="cuda", dtype=torch.bfloat16)

        def execute(prepared, qkv=qkv, qg=qg, kg=kg, freqs=freqs, window=window, sink=sink, dout=dout):
            produced = fused_qkv_rmsnorm_rope(
                qkv, qg, kg, freqs, [512, 64, 64], 1e-5, return_scaled_q=prepared
            )
            q, k, v = produced[:3]
            qs = produced[3] if prepared else None
            if prepared:
                assert not qs.requires_grad
                assert torch.equal(qs, torch.mul(q, 0.125 * math.log2(math.e))), "producer Q rounding"
            out = flash_attn_func(
                q.permute(1, 0, 2, 3),
                k.permute(1, 0, 2, 3),
                v.permute(1, 0, 2, 3),
                causal=True,
                window_size=(window, 0),
                sink=sink,
                q_for_backward=qs.permute(1, 0, 2, 3) if prepared else None,
            )
            gradients = torch.autograd.grad(out, (qkv, qg, kg, sink), dout)
            return tuple(t.detach() for t in (q, k, v, out)), gradients

        values, reference = execute(False)
        _, repeated = execute(False)
        actual_values, actual = execute(True)
        torch.cuda.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(values, actual_values)), "producer changed Q/K/V/O"
        assert torch.equal(qkv, original), "producer modified packed QKV"
        gradients = {}
        for name, ref, rep, got in zip(
            ("dqkv", "dq_gamma", "dk_gamma", "dsink"), reference, repeated, actual
        ):
            assert torch.isfinite(got).all()
            noise, error = relative_l2(rep, ref), relative_l2(got, ref)
            tolerance = max(0.002, 4 * noise)
            gradients[name] = dict(
                relative_l2=error,
                baseline_repeat_relative_l2=noise,
                tolerance=tolerance,
                exact=torch.equal(got, ref),
            )
            assert error <= tolerance, (seq, window, name, gradients[name])
        if seq == 512 and window == -1:
            # The nondifferentiable cache must not force a full zero-Q allocation
            # in normal backward, and unused K/V gradients must remain correct.
            raw = fused_qkv_rmsnorm_rope(qkv, qg, kg, freqs, [512, 64, 64], 1e-5)
            baseline_unused = torch.autograd.grad(raw[0].float().sum(), (qkv, qg, kg))
            cached = fused_qkv_rmsnorm_rope(qkv, qg, kg, freqs, [512, 64, 64], 1e-5, return_scaled_q=True)
            candidate_unused = torch.autograd.grad(cached[0].float().sum(), (qkv, qg, kg))
            assert all(torch.equal(a, b) for a, b in zip(baseline_unused, candidate_unused))
        results.append(
            dict(sequence=seq, window=window, exact_q_cache=True, exact_forward=True, gradients=gradients)
        )
        print("ATTENTION_PRODUCER_CASE " + json.dumps(results[-1]), flush=True)
    Path("/results/attention_producer_preflight.json").write_text(
        json.dumps(dict(passed=True, cases=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
