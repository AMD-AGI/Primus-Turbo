###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""beta = 1 (``accumulate``) on the gfx1250 FlyDSL variable-K grouped wgrad, into an fp32 ``main_grad``.

This is what Megatron's ``gradient_accumulation_fusion`` asks of the expert wgrad: ``main_grad[g] +=
dy[rows_g].T @ x[rows_g]`` with ``main_grad`` fp32, instead of a bf16 dW and a separate ``main_grad.add_``.

The shapes and routing are those of a Muse Spark expert layer as one GPU of the EP4 run sees it: 64 local
experts, top-8 routing of a 4096-token micro-batch, fc1 ``[64, 8192, 4096]`` (gate and up of a 4096 FFN
over the 4096 latent) and fc2 ``[64, 4096, 4096]``, two micro-batches accumulated into the same
``main_grad``. The end-to-end test drives them through ``grouped_gemm`` with the FlyDSL backend pinned and
``fuse_bgrad_accum_pattern="megatron"``, as Primus does. The kernel-level tests compare the two ways of
bringing the old ``out`` in (``accum_impl`` "tdm" through LDS, "reg" through registers): they add the same
fp32 accumulators to the same fp32 values, so they must agree bit for bit.
"""

import pytest
import torch
import torch.nn.functional as F

import primus_turbo.pytorch as pt
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager, PrecisionType
from primus_turbo.pytorch.core.utils import is_gfx1250

pytest.importorskip("flydsl", reason="the FlyDSL gfx1250 grouped GEMM lives behind flydsl")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 (MI455X) GPU"
)

# Muse Spark expert layer, one GPU of the EP4 run
NUM_EXPERTS = 64
TOPK = 8
TOKENS = 4096  # sequence 4096, micro-batch 1
LATENT = 4096  # expert input width
FFN = 4096
MICRO_BATCHES = 2

IMPLS = ["tdm", "reg"]


def _vk():
    from primus_turbo.flydsl.grouped_gemm import grouped_gemm_bf16_kernel_gfx1250

    return grouped_gemm_bf16_kernel_gfx1250


def _routing(seed, num_experts=NUM_EXPERTS, tokens=TOKENS, topk=TOPK):
    """Rows per expert of a softmax top-k router over N(0, 1) logits, after the token permute."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, num_experts, device="cuda", generator=g)
    experts = logits.softmax(-1).topk(topk, dim=-1).indices
    lens = torch.bincount(experts.flatten(), minlength=num_experts).to(torch.int64)
    offs = torch.zeros(num_experts + 1, dtype=torch.int64, device="cuda")
    offs[1:] = lens.cumsum(0)
    return lens, offs


def _fp32_wgrad(dy, x, offs):
    """``out[g] = dy[rows_g].T @ x[rows_g]`` in fp32."""
    o = offs.tolist()
    out = torch.zeros(len(o) - 1, dy.shape[1], x.shape[1], dtype=torch.float32, device="cuda")
    for e in range(len(o) - 1):
        if o[e + 1] > o[e]:
            out[e] = dy[o[e] : o[e + 1]].float().T @ x[o[e] : o[e + 1]].float()
    return out


def _max_rel_err(got, expected):
    scale = expected.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return ((got - expected).abs() / scale).max().item()


@pytest.fixture
def flydsl_grouped_gemm():
    """Pin the bf16 grouped GEMM to FlyDSL, as Primus's ``turbo_grouped_gemm_backend: bf16:flydsl`` does."""
    saved = GlobalBackendManager._grouped_gemm_backend
    GlobalBackendManager.set_grouped_gemm_backend(BackendType.FLYDSL, PrecisionType.BF16_FP16_FP32)
    yield
    GlobalBackendManager._grouped_gemm_backend = saved


def _expert_mlp_step(w1, w2, seed, fuse):
    """One micro-batch of the expert MLP, forward and backward. Returns what the fp32 reference needs."""
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    lens, offs = _routing(seed)
    m = int(offs[-1])
    x = torch.randn(m, LATENT, device="cuda", generator=g).to(torch.bfloat16).requires_grad_()
    dy = torch.randn(m, LATENT, device="cuda", generator=g).to(torch.bfloat16)
    pattern = "megatron" if fuse else None
    h = pt.ops.grouped_gemm(x, w1, lens, offs, trans_b=True, fuse_bgrad_accum_pattern=pattern)
    h.retain_grad()
    gate, up = h.chunk(2, dim=-1)
    act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    act.retain_grad()
    y = pt.ops.grouped_gemm(act, w2, lens, offs, trans_b=True, fuse_bgrad_accum_pattern=pattern)
    y.backward(dy)
    return x.detach(), h.grad, act.detach(), act.grad, dy, offs


def _expert_weights():
    g = torch.Generator(device="cuda").manual_seed(7)
    w1 = (torch.randn(NUM_EXPERTS, 2 * FFN, LATENT, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(NUM_EXPERTS, LATENT, FFN, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    ws = []
    for w in (w1, w2):
        w = torch.nn.Parameter(w)
        w.main_grad = torch.zeros(w.shape, dtype=torch.float32, device="cuda")
        w.grad_added_to_main_grad = False
        ws.append(w)
    return ws


def test_muse_spark_expert_wgrads_into_main_grad(flydsl_grouped_gemm):
    """Two micro-batches of fc1 / fc2 through ``grouped_gemm`` with the wgrad fused into fp32 main_grad,
    against an fp32 reference of the same gradients and against today's bf16 dW + ``main_grad.add_``."""
    w1, w2 = _expert_weights()
    sep1, sep2 = _expert_weights()  # the same values: same seed
    ref1 = torch.zeros_like(w1.main_grad)
    ref2 = torch.zeros_like(w2.main_grad)
    for mb in range(MICRO_BATCHES):
        x, dh, act, dact, dy, offs = _expert_mlp_step(w1, w2, mb, fuse=True)
        assert w1.grad_added_to_main_grad and w2.grad_added_to_main_grad
        ref1 += _fp32_wgrad(dh, x, offs)
        ref2 += _fp32_wgrad(dy, act, offs)
        # today's path: the wgrad returns a bf16 dW and Megatron's hook adds it into main_grad
        _expert_mlp_step(sep1, sep2, mb, fuse=False)
        for w in (sep1, sep2):
            w.main_grad.add_(w.grad)
            w.grad = None
    for got, sep, ref in ((w1.main_grad, sep1.main_grad, ref1), (w2.main_grad, sep2.main_grad, ref2)):
        # the dgrads that feed the wgrads are identical in both runs, so the reference is exact for both
        assert _max_rel_err(got, ref) < 1e-5
        assert (got - ref).abs().sum() <= (sep - ref).abs().sum()
        assert _max_rel_err(sep, ref) < 2**-7


CASES = [
    # (OUT_M, OUT_N, rows per expert)
    pytest.param(2 * FFN, LATENT, None, id="muse_fc1"),  # None: the Muse Spark top-8 routing above
    pytest.param(LATENT, FFN, None, id="muse_fc2"),
    pytest.param(512, 512, [100, 0, 50, 0], id="avg_m_le_128"),  # 64-deep k-tiles: "auto" takes "reg"
    pytest.param(4096, 7168, [2048] * 4, id="three_buffer_ring"),  # DeepSeek-V3 dW1 key, 64-deep: "reg"
]


def _problem(out_m, out_n, counts, seed=0):
    if counts is None:
        lens, offs = _routing(seed)
    else:
        lens = torch.tensor(counts, dtype=torch.int64, device="cuda")
        offs = torch.zeros(len(counts) + 1, dtype=torch.int64, device="cuda")
        offs[1:] = lens.cumsum(0)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    m = int(offs[-1])
    a = torch.randn(m, out_m, device="cuda", generator=g).to(torch.bfloat16)
    b = torch.randn(m, out_n, device="cuda", generator=g).to(torch.bfloat16)
    return a, b, lens, offs, _fp32_wgrad(a, b, offs)


def _old_out(shape, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=g, dtype=torch.float32) * 16


def _accumulate(a, b, offs, lens, out, impl):
    _vk().grouped_gemm_bf16_variable_k_flydsl_kernel(
        a, b, offs, masked_k=lens, out=out, accumulate=1, accum_impl=impl
    )


def _impls_for(out_m, out_n, counts):
    vk = _vk()
    avg_m = TOKENS * TOPK // NUM_EXPERTS if counts is None else max(1, sum(counts) // len(counts))
    tm, tn, tk, mw = vk._pick_variable_k_config(out_m, out_n, avg_m)[:4]
    return IMPLS if vk._vk_accum_tdm_fits(tm, tn, tk, mw, vk._vk_lds_pad(tm, tn, 2)) else ["reg"]


@pytest.mark.parametrize("impl", IMPLS + ["auto"])
@pytest.mark.parametrize("out_m, out_n, counts", CASES)
def test_accumulates_once_per_call(out_m, out_n, counts, impl):
    """Three calls add three times -- the first included, which compiles the variant (a compile that also
    ran the launch would add twice) -- to within fp32 rounding of the exact sum."""
    if impl != "auto" and impl not in _impls_for(out_m, out_n, counts):
        pytest.skip(f"accum_impl={impl!r} does not fit this tile")
    a, b, lens, offs, ref = _problem(out_m, out_n, counts)
    old = _old_out((lens.numel(), out_m, out_n))
    out = old.clone()
    _accumulate(a, b, offs, lens, out, impl)
    torch.cuda.synchronize()
    assert _max_rel_err(out, old + ref) < 1e-5
    for _ in range(2):
        _accumulate(a, b, offs, lens, out, impl)
    assert _max_rel_err(out, old + 3 * ref) < 3e-5
    # experts with no valid row keep their old value exactly
    for e in (lens == 0).nonzero().flatten().tolist():
        assert torch.equal(out[e], old[e])


@pytest.mark.parametrize("out_m, out_n, counts", CASES)
def test_tdm_and_reg_agree(out_m, out_n, counts):
    impls = _impls_for(out_m, out_n, counts)
    if len(impls) < 2:
        pytest.skip("only 'reg' fits this tile")
    a, b, lens, offs, _ = _problem(out_m, out_n, counts)
    old = _old_out((lens.numel(), out_m, out_n))
    outs = []
    for impl in impls:
        out = old.clone()
        _accumulate(a, b, offs, lens, out, impl)
        outs.append(out)
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("impl", IMPLS)
def test_backend_takes_accumulating_calls(impl, monkeypatch):
    """The registry's FlyDSL wgrad backend admits an fp32 beta = 1 call, swaps the operands for
    ``trans_c`` as the op's wgrad does, and adds into ``out``; it declines a bf16 ``out``."""
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
        GroupedGEMMVariableKFlyDSLBackend as backend,
    )

    vk = _vk()
    entry = vk.grouped_gemm_bf16_variable_k_flydsl_kernel
    monkeypatch.setattr(
        vk, "grouped_gemm_bf16_variable_k_flydsl_kernel", lambda *a, **k: entry(*a, **k, accum_impl=impl)
    )
    x, dy, lens, offs, _ = _problem(LATENT, LATENT, None)
    ref = _fp32_wgrad(dy, x, offs)  # [E, out, in], the weight's own layout
    old = _old_out(tuple(ref.shape))
    out = old.clone()
    kw = dict(trans_a=True, trans_b=False, trans_c=True, num_cu=None, inplace_add_to_out=True, out=out)
    assert backend.can_handle(x, dy, lens, offs, **kw)
    assert not backend.can_handle(x, dy, lens, offs, **{**kw, "out": out.to(torch.bfloat16)})
    backend.execute(x, dy, lens, offs, **kw)
    assert _max_rel_err(out, old + ref) < 1e-5


def test_refuses_what_it_cannot_add_into():
    vk = _vk()
    run = vk.grouped_gemm_bf16_variable_k_flydsl_kernel
    assert vk.grouped_gemm_bf16_variable_k_accum_supported(2 * FFN, LATENT, TOKENS * TOPK // NUM_EXPERTS)
    assert not vk.grouped_gemm_bf16_variable_k_accum_supported(5760, 2880, 2048)  # GPT-OSS: ragged tiles
    a, b, lens, offs, _ = _problem(512, 512, [300])

    def f32(*shape):
        return torch.zeros(*shape, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="fp32"):
        run(a, b, offs, masked_k=lens, accumulate=1)
    with pytest.raises(ValueError, match="fp32"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512).to(torch.bfloat16), accumulate=1)
    with pytest.raises(ValueError, match="accum_impl"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512), accumulate=1, accum_impl="lds")
    with pytest.raises(ValueError, match="contiguous"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512).transpose(1, 2), accumulate=1)
    # without accumulate the overwrite epilogue stores bf16, so an fp32 out would get reinterpreted bits
    with pytest.raises(ValueError, match="operands' dtype"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512))
    a, b, lens, offs, _ = _problem(512, 320, [300])
    with pytest.raises(ValueError, match="whole output tiles"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 320), accumulate=1)
