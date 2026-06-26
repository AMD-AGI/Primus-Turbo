###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Correctness + bandwidth/latency test for the Mega-MoE SwiGLU epilogue
(primus_turbo.flydsl.mega.swiglu_kernel).

Covers forward and backward, the optional per-row scale, and the gate gradient.
Prints achieved latency (us) and effective HBM bandwidth (GB/s) per shape.

Run as a perf report:
  PYTHONPATH=<...>/Primus-Turbo python -m pytest -s \
      tests/pytorch/ops/test_mega_moe_epilogue.py
or directly:
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_moe_epilogue.py
"""
import pytest
import torch
import torch.nn.functional as F

from primus_turbo.flydsl.mega.swiglu_kernel import (
    ACTIVATION_CLAMP,
    swiglu,
    swiglu_backward,
)

torch.manual_seed(0)
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# (M tokens, I intermediate); I must be a multiple of 1024.
_SHAPES = [(1024, 2048), (8192, 2048), (8192, 4096), (16384, 4096)]
_CLAMP = ACTIVATION_CLAMP
_COS_NAMES = ("cos", "gg_cos")  # output 0 = activation/grad, output 1 = gate grad


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _bench(fn, warmup=10, iters=50):
    """Mean latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0 / iters


def _check(tag, fn, ref, M, I, bytes_io, **fields):
    """Run ``fn``, assert cos > 0.999 vs ``ref`` (a tensor or a tuple of tensors),
    then bench ``fn`` and print one uniform perf line."""
    outs = fn()
    outs = outs if isinstance(outs, tuple) else (outs,)
    refs = ref if isinstance(ref, tuple) else (ref,)
    for name, out, r in zip(_COS_NAMES, outs, refs):
        cos = _cos(out, r)
        assert cos > 0.999, f"{tag} {name}={cos}"
        fields[name] = f"{cos:.6f}"
    lat = _bench(fn)
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    bw = bytes_io / (lat * 1e-6) / 1e9
    print(f"[{tag}] M={M:6d} I={I:5d} {extra} lat={lat:8.2f}us bw={bw:7.1f}GB/s")


# --- references -------------------------------------------------------------
def _ref_fwd(acc1, I, scale=None):
    gate = acc1[:, :I].float().clamp(-_CLAMP, _CLAMP)
    up = acc1[:, I:].float().clamp(-_CLAMP, _CLAMP)
    out = F.silu(gate) * up
    if scale is not None:
        out = out * scale.float().unsqueeze(1)
    return out.bfloat16()


def _ref_bwd(dact, acc1, I, scale=None, want_gate=False):
    acc = acc1.float().clone().requires_grad_(True)
    gate = acc[:, :I].clamp(-_CLAMP, _CLAMP)
    up = acc[:, I:].clamp(-_CLAMP, _CLAMP)
    act = F.silu(gate) * up
    sc = None
    if scale is not None:
        sc = scale.float().clone().requires_grad_(True)
        act = act * sc.unsqueeze(1)
    act.backward(dact.float())
    return (acc.grad.bfloat16(), sc.grad) if want_gate else acc.grad.bfloat16()


# --- tests ------------------------------------------------------------------
@pytest.mark.parametrize("M,I", _SHAPES)
@pytest.mark.parametrize("with_scale", [False, True])
def test_fwd(M, I, with_scale):
    acc1 = torch.randn(M, 2 * I, dtype=torch.bfloat16, device="cuda")
    scale = torch.rand(M, dtype=torch.float32, device="cuda") if with_scale else None
    fn = lambda: swiglu(acc1, scale=scale)
    # read 2*I bf16 + (scale), write I bf16
    bytes_io = M * (2 * I * 2 + I * 2) + (M * 4 if with_scale else 0)
    _check("fwd ", fn, _ref_fwd(acc1, I, scale), M, I, bytes_io, scale=int(with_scale))


# NOTE: the device-bounded grid-stride path (num_tile_blocks/BM) is no longer
# invocable standalone -- swiglu now reads the bound from the active symm workspace;
# it is exercised by the fused mega e2e test.


@pytest.mark.parametrize("M,I", _SHAPES)
@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("with_gate", [False, True])
def test_bwd(M, I, with_scale, with_gate):
    if with_gate and not with_scale:
        pytest.skip("gate gradient only meaningful with per-row scale")
    acc1 = torch.randn(M, 2 * I, dtype=torch.bfloat16, device="cuda")
    dact = torch.randn(M, I, dtype=torch.bfloat16, device="cuda")
    scale = torch.rand(M, dtype=torch.float32, device="cuda") if with_scale else None
    # return_gate -> swiglu_backward allocates+returns grad_gate (paired with want_gate ref)
    fn = lambda: swiglu_backward(dact, acc1, scale=scale, return_gate=with_gate)
    ref = _ref_bwd(dact, acc1, I, scale, want_gate=with_gate)
    # read 2*I bf16 (acc1) + I bf16 (dact) + scale, write 2*I bf16 (dacc1)
    bytes_io = M * (2 * I * 2 + I * 2 + 2 * I * 2) + (M * 4 if with_scale else 0)
    _check("bwd ", fn, ref, M, I, bytes_io, scale=int(with_scale), gate=int(with_gate))


if __name__ == "__main__":
    print("=== Mega-MoE SwiGLU epilogue: correctness + perf ===")
    for M, I in _SHAPES:
        for ws in (False, True):
            test_fwd(M, I, ws)
        for ws in (False, True):
            for wg in (False, True):
                if wg and not ws:
                    continue
                test_bwd(M, I, ws, wg)
    print("=== done ===")
