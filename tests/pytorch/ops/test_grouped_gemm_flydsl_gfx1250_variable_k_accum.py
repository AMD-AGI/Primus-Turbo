###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""beta = 1 (``accumulate``) on the gfx1250 FlyDSL variable-K grouped wgrad, into an fp32 or bf16 ``main_grad``.

This is what Megatron's ``gradient_accumulation_fusion`` asks of the expert wgrad: ``main_grad[g] +=
dy[rows_g].T @ x[rows_g]``, instead of a bf16 dW and a separate ``main_grad.add_``.

The shapes and routing are those of a Muse Spark expert layer as one GPU of the EP4 run sees it: 64 local
experts, top-8 routing of a 4096-token micro-batch, fc1 ``[64, 8192, 4096]`` (gate and up of a 4096 FFN
over the 4096 latent) and fc2 ``[64, 4096, 4096]``, two micro-batches accumulated into the same
``main_grad``. The end-to-end test drives them through ``grouped_gemm`` with the FlyDSL backend pinned and
``fuse_bgrad_accum_pattern="megatron"``, as Primus does. The kernel-level tests compare the two ways of
bringing the old ``out`` in (``accum_impl`` "tdm" through LDS, "reg" through registers): they add the same
fp32 accumulators to the same fp32 values, so they must agree bit for bit.

A bf16 ``main_grad`` (Kimi K3.1 Pro keeps its main_grads in bf16) gets the old ``out`` added to the fp32
accumulators and the sum rounded once, as MegaMoE's wgrad does: that is the fp32 accumulate's result rounded
to bf16, bit for bit. The ``_round_first`` hook rounds the accumulators before the add, which is today's
bf16 dW plus ``main_grad.add_(dW)`` bit for bit.
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


def _expert_weights(main_grad_dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(7)
    w1 = (torch.randn(NUM_EXPERTS, 2 * FFN, LATENT, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(NUM_EXPERTS, LATENT, FFN, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    ws = []
    for w in (w1, w2):
        w = torch.nn.Parameter(w)
        w.main_grad = torch.zeros(w.shape, dtype=main_grad_dtype, device="cuda")
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


def test_muse_spark_expert_wgrads_into_bf16_main_grad(flydsl_grouped_gemm):
    """The same two micro-batches into a bf16 main_grad: each fused wgrad rounds the sum once, so main_grad ends up
    within a bf16 rounding per micro-batch of the fp32 reference, and at least as close to it as today's bf16 dW +
    ``main_grad.add_``, which rounds twice per micro-batch."""
    w1, w2 = _expert_weights(torch.bfloat16)
    sep1, sep2 = _expert_weights(torch.bfloat16)
    ref1 = torch.zeros(w1.shape, dtype=torch.float32, device="cuda")
    ref2 = torch.zeros(w2.shape, dtype=torch.float32, device="cuda")
    for mb in range(MICRO_BATCHES):
        x, dh, act, dact, dy, offs = _expert_mlp_step(w1, w2, mb, fuse=True)
        assert w1.grad_added_to_main_grad and w2.grad_added_to_main_grad
        ref1 += _fp32_wgrad(dh, x, offs)
        ref2 += _fp32_wgrad(dy, act, offs)
        _expert_mlp_step(sep1, sep2, mb, fuse=False)
        for w in (sep1, sep2):
            w.main_grad.add_(w.grad)
            w.grad = None
    for got, sep, ref in ((w1.main_grad, sep1.main_grad, ref1), (w2.main_grad, sep2.main_grad, ref2)):
        assert got.dtype == torch.bfloat16
        assert _max_rel_err(got.float(), ref) < MICRO_BATCHES * 2**-8
        assert (got.float() - ref).abs().sum() <= (sep.float() - ref).abs().sum() * 1.001


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


def _accumulate(a, b, offs, lens, out, impl, **kw):
    _vk().grouped_gemm_bf16_variable_k_flydsl_kernel(
        a, b, offs, masked_k=lens, out=out, accumulate=1, accum_impl=impl, **kw
    )


def _impls_for(out_m, out_n, counts, out_dtype=torch.float32):
    vk = _vk()
    avg_m = TOKENS * TOPK // NUM_EXPERTS if counts is None else max(1, sum(counts) // len(counts))
    tm, tn, tk, mw = vk._pick_variable_k_config(out_m, out_n, avg_m)[:4]
    c_eb = out_dtype.itemsize
    return IMPLS if vk._vk_accum_tdm_fits(tm, tn, tk, mw, vk._vk_lds_pad(tm, tn, 2), c_eb) else ["reg"]


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


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("impl", IMPLS)
def test_backend_takes_accumulating_calls(impl, out_dtype, monkeypatch):
    """The registry's FlyDSL wgrad backend admits a beta = 1 call into an fp32 or a bf16 ``out``, swaps the
    operands for ``trans_c`` as the op's wgrad does, and adds into ``out``; it declines an fp16 ``out``."""
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
    old = _old_out(tuple(ref.shape)).to(out_dtype)
    out = old.clone()
    kw = dict(trans_a=True, trans_b=False, trans_c=True, num_cu=None, inplace_add_to_out=True, out=out)
    assert backend.can_handle(x, dy, lens, offs, **kw)
    assert not backend.can_handle(x, dy, lens, offs, **{**kw, "out": out.to(torch.float16)})
    assert not backend.can_handle(
        x, dy, lens, offs, **{**kw, "out": _misaligned(tuple(out.shape), out_dtype)}
    )
    backend.execute(x, dy, lens, offs, **kw)
    tol = 1e-5 if out_dtype == torch.float32 else 2**-8
    assert _max_rel_err(out.float(), old.float() + ref) < tol


def test_dispatcher_key_tells_the_out_dtypes_apart():
    """A tuned choice for an fp32 main_grad must not serve a bf16 one: the backends do not take the same outs."""
    from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import (
        GroupedGEMMVariableKKernelDispatcher as dispatcher,
    )

    x, dy, lens, offs, _ = _problem(512, 512, [300])
    kw = dict(a=x, b=dy, group_lens=lens, group_offs=offs, trans_a=True, trans_b=False, trans_c=True)
    kw.update(num_cu=None, maybe_pre_sync=False)
    keys = {
        dispatcher.make_key(
            **kw, inplace_add_to_out=True, out=torch.empty(1, 512, 512, dtype=dt, device="cuda")
        )
        for dt in (torch.float32, torch.bfloat16)
    }
    assert len(keys) == 2
    assert dispatcher.make_key(**kw) not in keys


MUSE_WGRAD_OUTS = [(2 * FFN, LATENT), (LATENT, FFN)]  # fc1, fc2
MUSE_WGRAD_KEYS = [(m, n, avg) for m, n in MUSE_WGRAD_OUTS for avg in (512, 1024, 2048, 4096, 8192)]


def test_muse_wgrad_keys_reach_accumulating_calls_only():
    """The wgrad fc1 / fc2 keys apply to accumulate=1 calls; overwrites keep theirs."""
    vk = _vk()
    for key in MUSE_WGRAD_KEYS:
        want = vk._KWARGS_KEYS["vk_accum"][key]
        assert set(want) <= {"group_m", "num_xcd", "xcd_band", "lds_cosched"}, (
            "only knobs the accumulate path keeps"
        )
        assert vk._keyed_kwargs("vk_accum", key, NUM_EXPERTS, 0, {"out": None, "accumulate": 1}) == want
        assert vk._keyed_kwargs("vk", key, NUM_EXPERTS, 0, {}) == {}
        assert vk._keyed_kwargs("vk_accum", key, NUM_EXPERTS, 0, {"accumulate": 1, "group_m": 16}) == {}


@pytest.mark.parametrize("out_m, out_n", MUSE_WGRAD_OUTS, ids=["fc1", "fc2"])
def test_muse_wgrad_key_is_bit_identical(out_m, out_n):
    """The keyed knobs move a schedule, never a reduction order."""
    a, b, lens, offs, _ = _problem(out_m, out_n, None)
    old = _old_out((lens.numel(), out_m, out_n))
    keyed, untuned = old.clone(), old.clone()
    run = _vk().grouped_gemm_bf16_variable_k_flydsl_kernel
    run(a, b, offs, masked_k=lens, out=keyed, accumulate=1)
    run(a, b, offs, masked_k=lens, out=untuned, accumulate=1, group_m=16)
    assert torch.equal(keyed, untuned)


def test_refuses_what_it_cannot_add_into():
    vk = _vk()
    run = vk.grouped_gemm_bf16_variable_k_flydsl_kernel
    supported = vk.grouped_gemm_bf16_variable_k_accum_supported
    assert supported(2 * FFN, LATENT, TOKENS * TOPK // NUM_EXPERTS)
    assert not supported(5760, 2880, 2048)  # GPT-OSS: ragged tiles
    # the out dtypes it adds into: fp32 (the default) and the 16-bit operand dtypes
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        assert supported(6144, 3584, 585, dtype)
        assert not supported(5760, 2880, 2048, dtype)
    assert not supported(6144, 3584, 585, torch.float64)
    a, b, lens, offs, _ = _problem(512, 512, [300])

    def f32(*shape):
        return torch.zeros(*shape, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="fp32"):
        run(a, b, offs, masked_k=lens, accumulate=1)
    # fp32 or the operands' dtype (bf16 here); an fp16 out would be added as reinterpreted bits
    with pytest.raises(ValueError, match="fp32"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512).to(torch.float16), accumulate=1)
    with pytest.raises(ValueError, match="accum_impl"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512), accumulate=1, accum_impl="lds")
    for dtype in (torch.float32, torch.bfloat16):
        with pytest.raises(ValueError, match="contiguous"):
            run(a, b, offs, masked_k=lens, out=f32(1, 512, 512).to(dtype).transpose(1, 2), accumulate=1)
    # a main_grad is a view into a gradient bucket: a view off the 16-byte grain is refused, not misread
    for dtype in (torch.float32, torch.bfloat16):
        with pytest.raises(ValueError, match="16-byte"):
            run(a, b, offs, masked_k=lens, out=_misaligned((1, 512, 512), dtype), accumulate=1)
    # the persistent walk takes a 16-bit out only
    with pytest.raises(ValueError, match="persist"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512), accumulate=1, persist=2)
    # the round-first hook exists for a 16-bit out only, and only when accumulating
    with pytest.raises(ValueError, match="_round_first"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512), accumulate=1, _round_first=True)
    with pytest.raises(ValueError, match="_round_first"):
        run(a, b, offs, masked_k=lens, _round_first=True)
    # without accumulate the overwrite epilogue stores bf16, so an fp32 out would get reinterpreted bits
    with pytest.raises(ValueError, match="operands' dtype"):
        run(a, b, offs, masked_k=lens, out=f32(1, 512, 512))
    a, b, lens, offs, _ = _problem(512, 320, [300])
    for dtype in (torch.float32, torch.bfloat16):
        with pytest.raises(ValueError, match="whole output tiles"):
            run(a, b, offs, masked_k=lens, out=f32(1, 512, 320).to(dtype), accumulate=1)


# A bf16 out. Kimi K3.1 Pro's expert wgrads: fc1 [6144, 3584] (gate and up of a 3072 FFN over the 3584 hidden) and
# fc2 [3584, 3072], here at 8 of its 224 local experts with empty, one-row and ragged ones.
KIMI_ROWS = [585, 0, 1, 300, 1170, 257, 2000, 37]
BF16_CASES = [
    # (OUT_M, OUT_N, rows per expert); 128-deep k-tiles stage the whole old C in one ring buffer, 64-deep ones
    # in two halves
    pytest.param(2 * FFN, LATENT, None, id="muse_fc1"),
    pytest.param(6144, 3584, KIMI_ROWS, id="kimi_fc1"),
    pytest.param(3584, 3072, KIMI_ROWS, id="kimi_fc2"),
    pytest.param(512, 512, [100, 0, 50, 0], id="avg_m_le_128"),  # 64-deep k-tiles
    pytest.param(4096, 7168, [2048] * 2, id="three_buffer_ring"),  # 64-deep, three buffers
]


def _misaligned(shape, dtype):
    """A contiguous view one element past a 16-byte boundary, as a main_grad can sit in its gradient bucket."""
    n = shape[0] * shape[1] * shape[2]
    return torch.zeros(n + 16, dtype=dtype, device="cuda")[1 : 1 + n].view(shape)


def _same_bits(x, y):
    bits = torch.int32 if x.element_size() == 4 else torch.int16
    return x.dtype == y.dtype and torch.equal(x.view(bits), y.view(bits))


def _old_bf16(shape):
    return _old_out(shape).to(torch.bfloat16)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("out_m, out_n, counts", BF16_CASES)
def test_bf16_round_first_matches_separate_add(out_m, out_n, counts, impl):
    """``_round_first``: the accumulators rounded to bf16, then added -- today's dW and ``main_grad.add_(dW)``, bit
    for bit, which pins the accumulate path's accumulators to the overwrite path's."""
    if impl not in _impls_for(out_m, out_n, counts, torch.bfloat16):
        pytest.skip(f"accum_impl={impl!r} does not fit this tile")
    a, b, lens, offs, _ = _problem(out_m, out_n, counts)
    old = _old_bf16((lens.numel(), out_m, out_n))
    separate = old.clone()
    separate.add_(_vk().grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=lens))
    fused = old.clone()
    _accumulate(a, b, offs, lens, fused, impl, _round_first=True)
    assert _same_bits(fused, separate)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("out_m, out_n, counts", BF16_CASES)
def test_bf16_rounds_the_fp32_accumulate_once(out_m, out_n, counts, impl):
    """Into bf16 the sum is the fp32 accumulate's, rounded once: bit for bit the fp32 path's result on the same old
    values, rounded to bf16 afterwards."""
    if impl not in _impls_for(out_m, out_n, counts, torch.bfloat16):
        pytest.skip(f"accum_impl={impl!r} does not fit this tile")
    a, b, lens, offs, _ = _problem(out_m, out_n, counts)
    old = _old_bf16((lens.numel(), out_m, out_n))
    wide = old.float()
    _accumulate(a, b, offs, lens, wide, "auto")
    once = old.clone()
    _accumulate(a, b, offs, lens, once, impl)
    assert _same_bits(once, wide.to(torch.bfloat16))


@pytest.mark.parametrize("out_m, out_n, counts", BF16_CASES)
def test_bf16_accumulates_close_to_fp32(out_m, out_n, counts):
    """One call is within a bf16 rounding of the exact sum and at least as close to it as the separate add; three
    calls add three times; experts with no valid row keep their old value exactly."""
    a, b, lens, offs, ref = _problem(out_m, out_n, counts)
    old = _old_bf16((lens.numel(), out_m, out_n))
    exact = old.float() + ref
    once = old.clone()
    _accumulate(a, b, offs, lens, once, "auto")
    assert _max_rel_err(once.float(), exact) < 2**-8
    separate = old.clone()
    separate.add_(_vk().grouped_gemm_bf16_variable_k_flydsl_kernel(a, b, offs, masked_k=lens))
    assert (once.float() - exact).abs().sum() <= (separate.float() - exact).abs().sum() * 1.001
    for _ in range(2):
        _accumulate(a, b, offs, lens, once, "auto")
    assert _max_rel_err(once.float(), old.float() + 3 * ref) < 3 * 2**-8
    for e in (lens == 0).nonzero().flatten().tolist():
        assert _same_bits(once[e], old[e])


@pytest.mark.parametrize("out_m, out_n, counts", BF16_CASES)
def test_bf16_stagings_agree(out_m, out_n, counts):
    """The three stagings of the old C -- "tdm" with the whole tile in one ring buffer (128-deep k-tiles), "tdm" in
    two halves (64-deep) and "reg" -- add the same accumulators to the same old values: bit-identical."""
    a, b, lens, offs, _ = _problem(out_m, out_n, counts)
    old = _old_bf16((lens.numel(), out_m, out_n))
    outs = []
    for impl in _impls_for(out_m, out_n, counts, torch.bfloat16):
        for kw in ({}, {"BLOCK_K": 64}):
            out = old.clone()
            _accumulate(a, b, offs, lens, out, impl, **kw)
            outs.append(out)
    assert all(_same_bits(out, outs[0]) for out in outs[1:])


# The k-loop knobs of the Kimi K3.1 wgrad sets, without their persistent seam: the deep ring with the split barrier
# and co-scheduled fragment reads, head_pipe, the branch-free TDM issue, and a 4 x 2 / 2 x 4 cluster.
KLOOP = dict(BLOCK_K=64, num_buffers=4, split_bar=1, lds_cosched=2, wmma_b2b=1, head_pipe=8, tdm_deal=3)
KIMI_CLUSTER = {
    (6144, 3584): dict(mc_m=4, mc_n=2, mc_early=0, mc_tile_sync=1, num_xcd=8, xcd_band=256),
    (3584, 3072): dict(mc_m=2, mc_n=4, mc_early=0, mc_tile_sync=1, num_xcd=8, xcd_band=128),
}


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("out_m, out_n", list(KIMI_CLUSTER), ids=["kimi_fc1", "kimi_fc2"])
def test_k_loop_knobs_keep_the_accumulate_bit_identical(out_m, out_n, out_dtype):
    """head_pipe, tdm_deal and the cluster move loads and matrix ops, never a k-step: the old C is loaded at the
    last k-tile's head as without them, and the sum is the untuned one bit for bit -- the Kimi K3.1 key's included,
    which the call without a tuning argument takes."""
    a, b, lens, offs, _ = _problem(out_m, out_n, KIMI_ROWS)
    old = _old_out((lens.numel(), out_m, out_n)).to(out_dtype)
    want = old.clone()
    _accumulate(a, b, offs, lens, want, "auto", group_m=16)  # a tuning argument: the key is off
    assert _vk()._keyed_kwargs(
        "vk_accum", (out_m, out_n, sum(KIMI_ROWS) // len(KIMI_ROWS)), 8, 0, {"out": old}
    )
    for kw in (KLOOP, dict(KLOOP, **KIMI_CLUSTER[(out_m, out_n)]), {}):
        got = old.clone()
        _accumulate(a, b, offs, lens, got, "auto", **kw)
        assert _same_bits(got, want), kw


# The persistent tile walk at 8 experts: 64 workgroups (whole clusters, whole rounds on both outputs).
PERSIST_SETS = [
    # 128-deep k-tiles, two buffers: the whole old C in slot 0
    dict(persist=64),
    # 64-deep, four buffers: the halves in slots 0 and 1
    dict(KLOOP, persist=64),
    # + the next tile's k-tiles 0 and 1 in the epilogue, the stagger and the decode carry
    dict(KLOOP, persist=64, vk_ovl=5, vk_stagger=127, vk_seam=6),
    # + no seam wait (the next tile waits for this tile's stores) and the first-use head order
    dict(KLOOP, persist=64, vk_ovl=7, vk_stagger=127, vk_seam=14),
]


@pytest.mark.parametrize("out_m, out_n", list(KIMI_CLUSTER), ids=["kimi_fc1", "kimi_fc2"])
def test_bf16_persistent_accumulate_is_bit_identical(out_m, out_n):
    """A bf16 out takes the persistent walk too: the old C sits in the C staging slots and comes in at the last
    k-tile's head unless that k-tile reads its slot -- the routing reaches every case (no part, the first, the second
    half late) -- and the sum is the untuned one bit for bit, with the cluster and the seam's prefetch as well."""
    a, b, lens, offs, _ = _problem(out_m, out_n, KIMI_ROWS)
    old = _old_bf16((lens.numel(), out_m, out_n))
    want = old.clone()
    _accumulate(a, b, offs, lens, want, "auto", group_m=16)
    for kw in PERSIST_SETS + [dict(PERSIST_SETS[-1], **KIMI_CLUSTER[(out_m, out_n)])]:
        got = old.clone()
        _accumulate(a, b, offs, lens, got, "auto", **kw)
        assert _same_bits(got, want), kw


def test_bf16_staging_fits_where_fp32_does_not():
    """A 64-deep k-tile cannot hold half an fp32 C tile, so fp32 takes "reg" there; half a bf16 one fits, and at
    128 deep the whole bf16 tile does."""
    vk = _vk()
    pad = vk._vk_lds_pad(256, 256, 2)
    assert vk._vk_accum_tdm_fits(256, 256, 128, 2, pad, 4)
    assert not vk._vk_accum_tdm_fits(256, 256, 64, 2, pad, 4)
    assert vk._vk_accum_tdm_fits(256, 256, 64, 2, pad, 2)
    assert vk._vk_accum_tdm_fits(256, 256, 128, 4, pad, 2)  # the whole tile needs no particular warp split
