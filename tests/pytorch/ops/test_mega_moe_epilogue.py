###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Correctness + bandwidth/latency test for the Mega-MoE SwiGLU epilogue
(primus_turbo.flydsl.mega.mega_moe_epilogue).

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

from primus_turbo.flydsl.mega.mega_moe_epilogue import (
    ACTIVATION_CLAMP,
    swiglu,
    swiglu_backward,
)

torch.manual_seed(0)

# (M tokens, I intermediate); I must be a multiple of 1024.
_SHAPES = [
    (1024, 2048),
    (8192, 2048),
    (8192, 4096),
    (16384, 4096),
]


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _bench(fn, warmup=10, iters=50):
    """Returns mean latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0 / iters


# --- references ------------------------------------------------------------
def _ref_fwd(acc1, I, clamp, scale=None):
    gate = acc1[:, :I].float().clamp(-clamp, clamp)
    up = acc1[:, I:].float().clamp(-clamp, clamp)
    out = F.silu(gate) * up
    if scale is not None:
        out = out * scale.float().unsqueeze(1)
    return out.bfloat16()


def _ref_bwd(dact, acc1, I, clamp, scale=None, want_gate=False):
    acc = acc1.float().clone().requires_grad_(True)
    gate = acc[:, :I].clamp(-clamp, clamp)
    up = acc[:, I:].clamp(-clamp, clamp)
    act = F.silu(gate) * up
    sc = None
    if scale is not None:
        sc = scale.float().clone().requires_grad_(True)
        act = act * sc.unsqueeze(1)
    act.backward(dact.float())
    dacc1 = acc.grad.bfloat16()
    if want_gate:
        return dacc1, sc.grad
    return dacc1


# --- forward ----------------------------------------------------------------
@pytest.mark.parametrize("M,I", _SHAPES)
@pytest.mark.parametrize("with_scale", [False, True])
def test_fwd(M, I, with_scale):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = "cuda"
    acc1 = torch.randn(M, 2 * I, dtype=torch.bfloat16, device=dev)
    scale = torch.rand(M, dtype=torch.float32, device=dev) if with_scale else None

    out = swiglu(acc1, None, I, M, scale=scale)
    ref = _ref_fwd(acc1, I, ACTIVATION_CLAMP, scale)
    cs = _cos(out, ref)
    assert cs > 0.999, f"fwd cos={cs}"

    lat = _bench(lambda: swiglu(acc1, None, I, M, scale=scale))
    # read 2*I bf16 + (scale), write I bf16
    bytes_io = M * (2 * I * 2 + I * 2) + (M * 4 if with_scale else 0)
    bw = bytes_io / (lat * 1e-6) / 1e9
    print(
        f"[fwd ] M={M:6d} I={I:5d} scale={int(with_scale)} " f"cos={cs:.6f} lat={lat:8.2f}us bw={bw:7.1f}GB/s"
    )


# --- forward, bounded (no-sync grid-stride) ---------------------------------
@pytest.mark.parametrize("M,I", _SHAPES)
def test_fwd_bounded(M, I):
    """Exercise the device-bounded grid-stride path (num_tile_blocks/BM) used by the
    fused mega pipeline; rows [0, num_tile_blocks*BM) must match the unbounded result."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    dev = "cuda"
    BM = 256
    assert M % BM == 0, "M must be a multiple of BM for the bounded test"
    acc1 = torch.randn(M, 2 * I, dtype=torch.bfloat16, device=dev)
    num_tile_blocks = torch.tensor([M // BM], dtype=torch.int32, device=dev)  # m_real = M

    out = swiglu(acc1, None, I, M, num_tile_blocks=num_tile_blocks, BM=BM)
    ref = _ref_fwd(acc1, I, ACTIVATION_CLAMP)
    cs = _cos(out, ref)
    assert cs > 0.999, f"fwd bounded cos={cs}"

    lat = _bench(lambda: swiglu(acc1, None, I, M, num_tile_blocks=num_tile_blocks, BM=BM))
    bytes_io = M * (2 * I * 2 + I * 2)
    bw = bytes_io / (lat * 1e-6) / 1e9
    print(f"[fwdb] M={M:6d} I={I:5d} BM={BM:4d} " f"cos={cs:.6f} lat={lat:8.2f}us bw={bw:7.1f}GB/s")


# --- backward ---------------------------------------------------------------
@pytest.mark.parametrize("M,I", _SHAPES)
@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("with_gate", [False, True])
def test_bwd(M, I, with_scale, with_gate):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if with_gate and not with_scale:
        pytest.skip("gate gradient only meaningful with per-row scale")
    dev = "cuda"
    acc1 = torch.randn(M, 2 * I, dtype=torch.bfloat16, device=dev)
    dact = torch.randn(M, I, dtype=torch.bfloat16, device=dev)
    scale = torch.rand(M, dtype=torch.float32, device=dev) if with_scale else None

    if with_gate:
        gg = torch.zeros(M, dtype=torch.float32, device=dev)
        dacc1, gg = swiglu_backward(dact, acc1, I, scale=scale, grad_gate=gg)
        dacc1_ref, gg_ref = _ref_bwd(dact, acc1, I, ACTIVATION_CLAMP, scale, want_gate=True)
        cs_g = _cos(gg, gg_ref)
        assert cs_g > 0.999, f"gate-grad cos={cs_g}"
    else:
        dacc1 = swiglu_backward(dact, acc1, I, scale=scale)
        dacc1_ref = _ref_bwd(dact, acc1, I, ACTIVATION_CLAMP, scale)
        cs_g = float("nan")
    cs = _cos(dacc1, dacc1_ref)
    assert cs > 0.999, f"bwd cos={cs}"

    def run():
        if with_gate:
            torch.zero_(gg)
            swiglu_backward(dact, acc1, I, scale=scale, grad_gate=gg)
        else:
            swiglu_backward(dact, acc1, I, scale=scale)

    lat = _bench(run)
    # read 2*I bf16 (acc1) + I bf16 (dact) + scale, write 2*I bf16 (dacc1)
    bytes_io = M * (2 * I * 2 + I * 2 + 2 * I * 2) + (M * 4 if with_scale else 0)
    bw = bytes_io / (lat * 1e-6) / 1e9
    gtxt = f"{cs_g:.6f}" if with_gate else "  n/a  "
    print(
        f"[bwd ] M={M:6d} I={I:5d} scale={int(with_scale)} gate={int(with_gate)} "
        f"cos={cs:.6f} gg_cos={gtxt} lat={lat:8.2f}us bw={bw:7.1f}GB/s"
    )


if __name__ == "__main__":
    print("=== Mega-MoE SwiGLU epilogue: correctness + perf ===")
    for M, I in _SHAPES:
        for ws in (False, True):
            test_fwd(M, I, ws)
    for M, I in _SHAPES:
        test_fwd_bounded(M, I)
    for M, I in _SHAPES:
        for ws in (False, True):
            for wg in (False, True):
                if wg and not ws:
                    continue
                test_bwd(M, I, ws, wg)
    print("=== done ===")
