###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""EP1 MegaMoE numerical tests with a local PyTorch oracle (no DeepEP dependency)."""

import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from primus_turbo.pytorch.core.utils import is_gfx950
from tests.pytorch.ops.gradient_contract_helpers import caller_thread_backward  # noqa: F401
from tests.pytorch.test_utils import compute_snr

pytestmark = pytest.mark.usefixtures("caller_thread_backward")


@pytest.fixture
def local_group():
    if not is_gfx950():
        pytest.skip("MegaMoE numerical regression requires gfx950")
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group("nccl", init_method=f"file://{directory}/store", rank=0, world_size=1)
        try:
            yield dist.group.WORLD
        finally:
            dist.destroy_process_group()


def reference(x, w1, w2, idx, probs, b1=None, b2=None):
    # Use the same BF16 operation boundaries as the independent existing MoE
    # reference; routing here is simple indexing and needs no dispatcher library.
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w1.shape[0]):
        tokens, slots = (idx == expert).nonzero(as_tuple=True)
        h = x[tokens] @ w1[expert].T
        if b1 is not None:
            h = h.float() + b1[expert].float()
        gate, up = h.chunk(2, dim=-1)
        act = (F.silu(gate) * up * probs[tokens, slots, None]).to(x.dtype)
        y = act @ w2[expert].T
        if b2 is not None:
            y = (y.float() + b2[expert].float() * probs[tokens, slots, None]).to(x.dtype)
        out = out.index_add(0, tokens, y.float())
    return out.to(x.dtype)


@pytest.mark.parametrize("mode", ["fused", "staged", "staged_bias", "fp8_bias"])
@pytest.mark.parametrize("compile_graph", [False, True])
def test_mega_moe_local_forward_backward(local_group, mode, compile_graph):
    from primus_turbo.flydsl.mega.bf16.symm_buffer import get_symm_buffer_for_mega_moe
    from primus_turbo.pytorch.ops.moe.fused_mega_moe import (
        fused_mega_moe,
        fused_mega_moe_stage1,
        fused_mega_moe_stage2,
    )
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
        fused_mega_moe_fp8_stage1,
        fused_mega_moe_fp8_stage2,
    )

    torch.manual_seed(802)
    tokens, hidden, inter, experts, topk = 256, 1024, 1024, 4, 2
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16).requires_grad_()
    w1 = (
        torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.0625
    ).requires_grad_()
    w2 = (torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.0625).requires_grad_()
    logits = torch.randn(tokens, experts, device="cuda").softmax(-1)
    p, idx = logits.topk(topk, dim=-1)
    p = p.detach().requires_grad_()
    bias = "bias" in mode
    b1 = (
        torch.randn(experts, 2 * inter, device="cuda", dtype=torch.bfloat16).requires_grad_()
        if bias
        else None
    )
    b2 = torch.randn(experts, hidden, device="cuda", dtype=torch.bfloat16).requires_grad_() if bias else None
    inputs = [x, w1, w2, p] + ([b1, b2] if bias else [])
    refs = [v.detach().clone().requires_grad_() for v in inputs]
    symm_options = {}
    if mode.startswith("fp8"):
        from primus_turbo.flydsl.mega.fp8.symm_buffer import get_symm_buffer_for_mega_moe

        symm_options["use_mxfp8"] = True
    symm = get_symm_buffer_for_mega_moe(
        local_group,
        num_experts=experts,
        num_max_tokens_per_rank=tokens,
        num_topk=topk,
        hidden=hidden,
        intermediate_hidden=inter,
        **symm_options,
    )
    try:
        grad = torch.randn_like(x).float()
        grad = (grad / grad.norm() * 1e-3).bfloat16()

        def forward(x, w1, w2, p, b1, b2):
            if mode == "fused":
                return fused_mega_moe(local_group, x, idx, p, w1, w2)
            if mode.startswith("fp8"):
                h, dw, handle, state = fused_mega_moe_fp8_stage1(x, idx, p, w1, local_group)
                return fused_mega_moe_fp8_stage2(h, dw, handle, state, idx, p, w2, local_group, None, b1, b2)
            h, dw, handle = fused_mega_moe_stage1(x, idx, p, w1, local_group)
            return fused_mega_moe_stage2(h, dw, handle, idx, p, w2, local_group, None, b1, b2)

        runner = torch.compile(forward, backend="aot_eager") if compile_graph else forward
        for _ in range(2):
            y = runner(x, w1, w2, p, b1, b2)
            grads = torch.autograd.grad(y, inputs, grad)
            ref = reference(refs[0], refs[1], refs[2], idx, refs[3], *(refs[4:] if bias else []))
            expected = torch.autograd.grad(ref, refs, grad)
            for label, actual, want in zip(
                ["out", "dx", "dw1", "dw2", "dprobs", "db1", "db2"], [y, *grads], [ref, *expected]
            ):
                snr = compute_snr(want.float(), actual.float())
                assert snr > (20 if mode.startswith("fp8") else 40), label
                assert F.cosine_similarity(want.float().flatten(), actual.float().flatten(), dim=0) > (
                    0.995 if mode.startswith("fp8") else 0.99
                )
    finally:
        symm.destroy()
