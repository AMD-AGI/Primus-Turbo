###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaAttn ops against the unfused fused-a2a path they replace: ``ulysses_qkv_attn_func`` =
``ulysses_qkv_gemm_a2a`` + ``flash_attn_func``, ``ulysses_attn_layer_func`` = that + ``ulysses_a2a_o_gemm``.
The forward runs the same tiles in the same order and the
backward the same kernels, so with the deterministic attention backward (no dQ atomics) everything is
bitwise equal (the layer output ``y`` within bf16 rounding: its GEMM tiles differ from the A2AGemm's).
The default (atomic dQ) backward differs run to run in either path; it only has to run.

    pytest tests/pytorch/ops/test_mega_attn.py --dist-only
    torchrun --nproc_per_node 4 tests/pytorch/ops/test_mega_attn.py
"""

import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

# (hq, hkv) as multiples of n: GQA 32/8 (2 kv heads per rank at n = 4), Qwen3-30B-A3B 32/4
HEAD_CONFIGS = [lambda n: (32, 8), lambda n: (8 * n, n)]


def _randn(*shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=gen).mul_(scale).bfloat16()


def _reference(x, w, hq, hkv, d, group, qkv_format, seq_chunk):
    """Deterministic attention backward."""
    from primus_turbo.pytorch.ops import flash_attn_func
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import ulysses_qkv_gemm_a2a

    q, k, v = ulysses_qkv_gemm_a2a(x, w, hq, hkv, d, group, qkv_format, "interleaved", seq_chunk)
    if qkv_format == "sbhd":
        q, k, v = (t.permute(1, 0, 2, 3) for t in (q, k, v))
    o = flash_attn_func(q, k, v, causal=True, deterministic=True)
    return o.permute(1, 0, 2, 3) if qkv_format == "sbhd" else o


def check_qkv_attn(group, hq, hkv, d, s_local, hidden, qkv_format, seq_chunk, repeats, failures):
    from primus_turbo.pytorch.ops.attention.mega_attn import ulysses_qkv_attn_func

    rank = group.rank()
    tag = f"[qkv_attn hq={hq} hkv={hkv} s/n={s_local} {qkv_format} chunk={seq_chunk}]"
    w = _randn((hq + 2 * hkv) * d, hidden, seed=11, scale=0.02).requires_grad_()
    for it in range(repeats):
        shape = (s_local, 1, hidden) if qkv_format == "sbhd" else (1, s_local, hidden)
        x = _randn(*shape, seed=100 * it + rank).requires_grad_()
        o = ulysses_qkv_attn_func(x, w, hq, hkv, d, group, qkv_format, seq_chunk, deterministic=True)
        do = _randn(*o.shape, seed=200 * it + rank)
        dx, dw = torch.autograd.grad(o, (x, w), do)
        o_ref = _reference(x, w, hq, hkv, d, group, qkv_format, seq_chunk)
        dx_ref, dw_ref = torch.autograd.grad(o_ref, (x, w), do)
        for name, out, ref in (("o", o, o_ref), ("dx", dx, dx_ref), ("dw", dw, dw_ref)):
            if out.shape != ref.shape or not torch.equal(out, ref):
                diff = (out.float() - ref.float()).abs().max().item() if out.shape == ref.shape else "shape"
                failures.append(f"{tag} call {it} {name}: not bitwise, max abs diff {diff}")
        o = ulysses_qkv_attn_func(x, w, hq, hkv, d, group, qkv_format, seq_chunk)
        (dx,) = torch.autograd.grad(o, (x,), do)
        if not torch.isfinite(dx).all():
            failures.append(f"{tag} call {it} dx (atomic dQ): not finite")


def check_layer(group, hq, hkv, d, s_local, hidden, qkv_format, seq_chunk, repeats, failures):
    from primus_turbo.pytorch.ops.attention.mega_attn import ulysses_attn_layer_func
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import ulysses_a2a_o_gemm

    rank = group.rank()
    tag = f"[layer hq={hq} hkv={hkv} s/n={s_local} {qkv_format} chunk={seq_chunk}]"
    w = _randn((hq + 2 * hkv) * d, hidden, seed=11, scale=0.02).requires_grad_()
    w_o = _randn(hidden, hq * d, seed=12, scale=0.02).requires_grad_()
    shape = (s_local, 1, hidden) if qkv_format == "sbhd" else (1, s_local, hidden)
    for it in range(repeats):
        x = _randn(*shape, seed=100 * it + rank).requires_grad_()
        res = _randn(*shape, seed=300 * it + rank).requires_grad_()
        y = ulysses_attn_layer_func(
            x, w, w_o, res, hq, hkv, d, group, qkv_format, seq_chunk, deterministic=True
        )
        dy = _randn(*y.shape, seed=200 * it + rank)
        grads = torch.autograd.grad(y, (x, w, w_o, res), dy)
        o_ref = _reference(x, w, hq, hkv, d, group, qkv_format, seq_chunk)
        y_ref = ulysses_a2a_o_gemm(o_ref, w_o, res, group, qkv_format, seq_chunk)
        grads_ref = torch.autograd.grad(y_ref, (x, w, w_o, res), dy)
        if y.shape != y_ref.shape or not torch.allclose(y.float(), y_ref.float(), atol=2e-2, rtol=2e-2):
            diff = (y.float() - y_ref.float()).abs().max().item() if y.shape == y_ref.shape else "shape"
            failures.append(f"{tag} call {it} y: max abs diff {diff}")
        for name, out, ref in zip(("dx", "dw", "dw_o", "dres"), grads, grads_ref):
            if out.shape != ref.shape or not torch.equal(out, ref):
                diff = (out.float() - ref.float()).abs().max().item() if out.shape == ref.shape else "shape"
                failures.append(f"{tag} call {it} {name}: not bitwise, max abs diff {diff}")
        y = ulysses_attn_layer_func(x, w, w_o, res, hq, hkv, d, group, qkv_format, seq_chunk)
        (dx,) = torch.autograd.grad(y, (x,), dy)
        if not torch.isfinite(dx).all():
            failures.append(f"{tag} call {it} dx (atomic dQ): not finite")


def run_suite(layer_only=False):
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    failures = []
    for make in HEAD_CONFIGS:
        hq, hkv = make(n)
        for qkv_format in ("sbhd", "bshd"):
            if not layer_only:
                # s/n = 1024 in chunks of 256; s/n = 1000 in chunks of 200 (no tile multiple); contiguous
                for s_local, seq_chunk in ((1024, 256), (1000, 200), (1024, 0)):
                    check_qkv_attn(group, hq, hkv, 128, s_local, 1024, qkv_format, seq_chunk, 2, failures)
            for s_local, seq_chunk in ((1024, 256), (1024, 128), (1024, 0)):
                check_layer(group, hq, hkv, 128, s_local, 1024, qkv_format, seq_chunk, 2, failures)
    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if rank == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"mega_attn: all cases passed (n={n})")
    return not any(gathered)


def main():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    ok = run_suite(layer_only=os.environ.get("MEGA_ATTN_LAYER_ONLY") == "1")
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


@pytest.mark.multigpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_mega_attn(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", __file__]
    result = subprocess.run(cmd, cwd=repo_root, env={**os.environ, "PYTHONPATH": repo_root}, timeout=3600)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
