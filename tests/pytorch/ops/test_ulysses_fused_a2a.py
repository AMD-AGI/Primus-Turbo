###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses GemmA2A / A2AGemm fused projections against the unfused reference.

Reference: fp32 ``F.linear`` -> Turbo's ``AttentionCPA2AHelper`` packing -> ``all_to_all_single``
-> unpacking, i.e. the ``flash_attn_usp_func`` sequence order.

The pytest entry launches this file under torchrun for every world size the node has GPUs for:

    pytest tests/pytorch/ops/test_ulysses_fused_a2a.py --dist-only
    torchrun --nproc_per_node 8 tests/pytorch/ops/test_ulysses_fused_a2a.py
"""

import argparse
import itertools
import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from primus_turbo.pytorch.core.utils import is_gfx950

# (hq, hkv) as multiples of n: GQA 32/8, MHA 40/40, hkv == n (Qwen3 64/4 at n = 4)
HEAD_CONFIGS = {
    "gqa": lambda n: (32, 8),
    "mha": lambda n: (40, 40),
    "hkv_eq_n": lambda n: (16 * n, n),
}


def _split_qkv_reference(qkv, hq, hkv, d, weight_layout):
    lead = qkv.shape[:2]
    if weight_layout == "interleaved":
        qkv = qkv.view(*lead, hkv, hq // hkv + 2, d)
        q = qkv[..., : hq // hkv, :].reshape(*lead, hq, d)
        return q, qkv[..., hq // hkv, :], qkv[..., hq // hkv + 1, :]
    q, k, v = torch.split(qkv, [hq * d, hkv * d, hkv * d], dim=-1)
    return q.reshape(*lead, hq, d), k.reshape(*lead, hkv, d), v.reshape(*lead, hkv, d)


def _helper(b, seq, hq, hkv, d, n, qkv_format):
    from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
        get_attention_cp_a2a_helper,
    )

    return get_attention_cp_a2a_helper(b, seq, hq, hkv, d, d, 0 if qkv_format == "sbhd" else 1, n)


def _a2a(t, group):
    """all_to_all_single with autograd: the reference backward is the mirrored all-to-all."""
    from torch.distributed.nn.functional import all_to_all_single

    return all_to_all_single(torch.empty_like(t), t.contiguous(), group=group)


def reference_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout):
    n = group.size()
    s_local, b = x.shape[:2] if qkv_format == "sbhd" else x.shape[1::-1]
    helper = _helper(b, s_local * n, hq, hkv, d, n, qkv_format)
    qkv = F.linear(x, w_qkv)
    q, k, v = _split_qkv_reference(qkv, hq, hkv, d, weight_layout)
    return helper.splits_qkv_after_a2a(_a2a(helper.combine_qkv_before_a2a(q, k, v), group))


def reference_a2a_o_gemm(o, w_o, residual, group, qkv_format):
    n = group.size()
    if qkv_format == "sbhd":
        seq, b, heads, d = o.shape
    else:
        b, seq, heads, d = o.shape
    helper = _helper(b, seq, heads * n, heads * n, d, n, qkv_format)
    o_gathered = helper.reshape_o_after_a2a(_a2a(helper.reshape_o_before_a2a(o), group))
    y = F.linear(o_gathered.reshape(*o_gathered.shape[:2], -1), w_o)
    return y if residual is None else y + residual


def _close(name, out, ref, failures, tag, scale=None):
    """bf16 allclose; ``scale`` (default ``|ref|``) is what ``rtol`` is relative to, e.g. the summed
    magnitudes when ``ref`` is a sum of terms that cancel."""
    from tests.pytorch.test_utils import get_tolerances

    tol = get_tolerances(torch.bfloat16)
    if out.shape != ref.shape:
        failures.append(f"{tag} {name}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        return
    scale = ref.float().abs() if scale is None else scale
    diff = (out.float() - ref.float()).abs()
    excess = diff - tol["atol"] - tol["rtol"] * scale
    if (excess > 0).any():
        i = excess.argmax().item()
        failures.append(
            f"{tag} {name}: max abs err {diff.max().item():.4g}, {(excess > 0).sum().item()} elements out of "
            f"tolerance, worst got {out.flatten()[i].item():.5g} expected {ref.flatten()[i].item():.5g}"
        )


def _randn(*shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=gen).mul_(scale).bfloat16()


def _fp32_leaf(t):
    return t.detach().float().requires_grad_()


def check_qkv(group, hq, hkv, d, b, s_local, hidden, qkv_format, weight_layout, repeats, failures):
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import ulysses_qkv_gemm_a2a

    rank = group.rank()
    w_qkv = _randn((hq + 2 * hkv) * d, hidden, seed=1234, scale=0.02).requires_grad_()
    tag = f"[qkv hq={hq} hkv={hkv} d={d} b={b} s/n={s_local} {qkv_format} {weight_layout}]"
    for it in range(repeats):
        shape = (s_local, b, hidden) if qkv_format == "sbhd" else (b, s_local, hidden)
        x = _randn(*shape, seed=100 * it + rank).requires_grad_()
        w_qkv.grad = None
        outs = ulysses_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout)
        x_ref, w_ref = _fp32_leaf(x), _fp32_leaf(w_qkv)
        refs = reference_qkv_gemm_a2a(x_ref, w_ref, hq, hkv, d, group, qkv_format, weight_layout)
        grads = [_randn(*r.shape, seed=7 * it + 31 * j + rank) for j, r in enumerate(refs)]
        for name, out, ref in zip("qkv", outs, refs):
            _close(name, out, ref, failures, f"{tag} call {it}")
        torch.autograd.backward(outs, grads)
        torch.autograd.backward(refs, [g.float() for g in grads])
        _close("dx", x.grad, x_ref.grad, failures, f"{tag} call {it}")
        _close("dw_qkv", w_qkv.grad, w_ref.grad, failures, f"{tag} call {it}")


def check_o_gemm(group, hq, d, b, s_local, hidden, qkv_format, with_residual, repeats, failures):
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import ulysses_a2a_o_gemm

    n, rank = group.size(), group.rank()
    w_o = _randn(hidden, hq * d, seed=4321, scale=0.02).requires_grad_()
    tag = f"[o_proj hq={hq} d={d} b={b} s/n={s_local} {qkv_format} residual={with_residual}]"
    seq = s_local * n
    for it in range(repeats):
        o_shape = (seq, b, hq // n, d) if qkv_format == "sbhd" else (b, seq, hq // n, d)
        y_shape = (s_local, b, hidden) if qkv_format == "sbhd" else (b, s_local, hidden)
        o = _randn(*o_shape, seed=200 * it + rank).requires_grad_()
        res = _randn(*y_shape, seed=300 * it + rank).requires_grad_() if with_residual else None
        w_o.grad = None
        y = ulysses_a2a_o_gemm(o, w_o, res, group, qkv_format)
        o_ref, w_ref = _fp32_leaf(o), _fp32_leaf(w_o)
        res_ref = _fp32_leaf(res) if with_residual else None
        y_ref = reference_a2a_o_gemm(o_ref, w_ref, res_ref, group, qkv_format)
        _close("y", y, y_ref, failures, f"{tag} call {it}")
        dy = _randn(*y_shape, seed=400 * it + rank)
        y.backward(dy)
        y_ref.backward(dy.float())
        _close("do", o.grad, o_ref.grad, failures, f"{tag} call {it}")
        _close("dw_o", w_o.grad, w_ref.grad, failures, f"{tag} call {it}")
        if with_residual:
            _close("d_residual", res.grad, res_ref.grad, failures, f"{tag} call {it}")


class _Layer:
    """QKV projection -> elementwise stand-in for attention (``q + k * v`` per GQA group, so dq, dk
    and dv are all non-zero) -> output projection + residual."""

    def __init__(self, group, hq, hkv, d, hidden, qkv_format, seed):
        self.group, self.hq, self.hkv, self.d, self.qkv_format = group, hq, hkv, d, qkv_format
        self.w_qkv = _randn((hq + 2 * hkv) * d, hidden, seed=seed, scale=0.02).requires_grad_()
        self.w_o = _randn(hidden, hq * d, seed=seed + 1, scale=0.02).requires_grad_()

    def _attn(self, q, k, v):
        return q + (k * v).repeat_interleave(self.hq // self.hkv, dim=2)

    def __call__(self, x):
        from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
            ulysses_a2a_o_gemm,
            ulysses_qkv_gemm_a2a,
        )

        args = (self.hq, self.hkv, self.d, self.group, self.qkv_format, "interleaved")
        self.qkv = ulysses_qkv_gemm_a2a(x, self.w_qkv, *args)
        self.o = self._attn(*self.qkv)
        for t in (*self.qkv, self.o):
            t.retain_grad()
        return ulysses_a2a_o_gemm(self.o, self.w_o, x, self.group, self.qkv_format)

    def check(self, x, y, failures, tag):
        """Each op of ``y = self(x)`` against the fp32 reference fed the same bf16 inputs and grads."""
        group, fmt = self.group, self.qkv_format
        x_ref, wq_ref, wo_ref = _fp32_leaf(x), _fp32_leaf(self.w_qkv), _fp32_leaf(self.w_o)
        refs = reference_qkv_gemm_a2a(x_ref, wq_ref, self.hq, self.hkv, self.d, group, fmt, "interleaved")
        for name, out, ref in zip("qkv", self.qkv, refs):
            _close(name, out, ref, failures, tag)
        torch.autograd.backward(refs, [t.grad.float() for t in self.qkv])
        o_ref, res_ref = _fp32_leaf(self.o), _fp32_leaf(x)
        y_ref = reference_a2a_o_gemm(o_ref, wo_ref, res_ref, group, fmt)
        _close("y", y, y_ref, failures, tag)
        y_ref.backward(y.grad.float())
        _close("do", self.o.grad, o_ref.grad, failures, tag)
        # autograd sums the two dx terms in bf16; they cancel in deep layers, where dy has grown
        scale = x_ref.grad.abs() + res_ref.grad.abs()
        _close("dx", x.grad, x_ref.grad + res_ref.grad, failures, tag, scale=scale)
        _close("dw_qkv", self.w_qkv.grad, wq_ref.grad, failures, tag)
        _close("dw_o", self.w_o.grad, wo_ref.grad, failures, tag)


def check_scenarios(group, failures, hq=32, hkv=8, d=128, b=2, s_local=300, hidden=1024):
    """Chained layers and a non-default stream."""
    rank = group.rank()
    for qkv_format in ("sbhd", "bshd"):
        shape = (s_local, b, hidden) if qkv_format == "sbhd" else (b, s_local, hidden)
        layers = [_Layer(group, hq, hkv, d, hidden, qkv_format, seed=10 * i) for i in range(3)]

        # back to back layers: every op of a layer overlaps the bank / flag reuse of the previous one
        xs = [_randn(*shape, seed=500 + rank).requires_grad_()]
        for layer in layers:
            xs.append(layer(xs[-1]))
            xs[-1].retain_grad()
        xs[-1].backward(_randn(*shape, seed=600 + rank))
        for i, layer in enumerate(layers):
            layer.check(xs[i], xs[i + 1], failures, f"[layers {qkv_format}] layer {i}")

        layer = layers[0]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            layer.w_qkv.grad = layer.w_o.grad = None
            x = _randn(*shape, seed=700 + rank).requires_grad_()
            y = layer(x)
            y.retain_grad()
            y.backward(_randn(*shape, seed=800 + rank))
        torch.cuda.current_stream().wait_stream(side)
        layer.check(x, y, failures, f"[side stream {qkv_format}]")


def _poison_banks(group):
    """Fill both recv banks of every workspace with bf16 NaN: a kernel that reads a recv row before
    its producer wrote it this epoch (or a padding row into a live output) shows up as NaN."""
    from primus_turbo.flydsl.ulysses.workspace import _WORKSPACES, FLAG_BYTES

    torch.cuda.synchronize()
    dist.barrier(group=group)
    for ws in _WORKSPACES.values():
        banks = ws.symm_mem.get_buffer(ws.rank, (ws.bank_bytes,), torch.int16, storage_offset=FLAG_BYTES // 2)
        banks.fill_(0x7FC0)
    torch.cuda.synchronize()
    dist.barrier(group=group)


def check_poisoned(group, failures, hq=32, hkv=8, d=128, b=2, s_local=300, hidden=1024):
    """Forward + backward of both ops, the banks poisoned before every call."""
    for qkv_format in ("sbhd", "bshd"):
        for it in range(2):
            # s/n = 512: dw tiles inside the A2AGemm
            for s in (s_local, 512):
                _poison_banks(group)
                check_qkv(group, hq, hkv, d, b, s, hidden, qkv_format, "interleaved", 1, failures)
            _poison_banks(group)
            check_o_gemm(group, hq, d, b, s_local, hidden, qkv_format, True, 1, failures)


def run_suite(quick):
    group = dist.group.WORLD
    n = group.size()
    rank = group.rank()
    failures = []
    head_names = ["gqa"] if quick else list(HEAD_CONFIGS)
    dims = [128] if quick else [64, 128]
    batches = [1] if quick else [1, 2]
    formats = ["sbhd"] if quick else ["sbhd", "bshd"]
    layouts = ["interleaved"] if quick else ["interleaved", "concat"]
    # s/n = 300 is not a multiple of any tile (separate wgrad); 512 runs the dw tiles in the A2AGemm
    # where the rank's QKV columns divide by 256
    for head_name, d, b, qkv_format, weight_layout, s_local in itertools.product(
        head_names, dims, batches, formats, layouts, [300, 512]
    ):
        hq, hkv = HEAD_CONFIGS[head_name](n)
        if hq % n or hkv % n:
            continue
        check_qkv(group, hq, hkv, d, b, s_local, 1024, qkv_format, weight_layout, 3, failures)
    for head_name, d, b, qkv_format, with_residual in itertools.product(
        head_names, dims, batches, formats, [True] if quick else [True, False]
    ):
        hq, _ = HEAD_CONFIGS[head_name](n)
        if hq % n:
            continue
        check_o_gemm(group, hq, d, b, 300, 1024, qkv_format, with_residual, 3, failures)
    # GLM-4.7 widths: over two waves of dw tiles (1120 QKV, 960 o_proj) change both launch orders
    check_qkv(group, 96, 8, 128, 1, 512, 5120, "sbhd", "interleaved", 2, failures)
    check_o_gemm(group, 96, 128, 1, 512, 5120, "sbhd", True, 2, failures)
    check_scenarios(group, failures)
    check_poisoned(group, failures)

    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if rank == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"ulysses fused a2a: all cases passed (n={n})")
    return not any(gathered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    ok = run_suite(args.quick)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


@pytest.mark.multigpu
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()), reason="the Ulysses FlyDSL kernels need gfx950"
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ulysses_fused_a2a(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", __file__]
    result = subprocess.run(cmd, cwd=repo_root, env={**os.environ, "PYTHONPATH": repo_root}, timeout=1800)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
