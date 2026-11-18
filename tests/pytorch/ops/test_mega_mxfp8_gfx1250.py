###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegaMoE in MXFP8 on gfx1250, one GPU (EP1): the stages, the MX rows on the wire and the combine-only launch
included, bit for bit against the routed experts' grouped-path kernels run in the Mega pool order on the same
quantized operands (``check_stages``, which the EP2+ suite runs on every rank); the staged ops against an fp32
reference, run to run and with two layers interleaved; a bf16 combine after a combine-only launch on one heap, the
zero routing weight of the pool's padding rows, and the contract violations the stages refuse."""

import os
import sys

import pytest
import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.flydsl.utils.glu_activation import GLUActivation
from primus_turbo.pytorch.core.utils import is_gfx1250

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
import mxfp8_ref as R  # noqa: E402

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernels")

from tests.pytorch.ops import test_mega_bf16_gfx1250_fused_moe as _bf16  # noqa: E402

_reference, _snr = _bf16._reference, _bf16._snr
ep1_group = _bf16.ep1_group  # the one-rank group fixture

SWIGLU = GLUActivation.swiglu()  # DeepSeek-V3's activation: plain SiLU-SwiGLU


def _inputs(T, H, I, E, topk, seed, skew=False, local=None):
    """Tokens routed over ``E`` experts, and the weights of ``local`` (default all) of them."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, generator=g, device="cuda").to(torch.bfloat16)
    scores = torch.randn(T, E, generator=g, device="cuda")
    if skew:  # hot experts and two empty ones
        scores[:, : E // 4] += 2.0
        scores[:, -2:] -= 100.0
    topk_w, topk_idx = scores.softmax(-1).topk(topk, dim=-1)
    local = local or E
    w1 = (torch.randn(local, 2 * I, H, generator=g, device="cuda") * 2 / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(local, H, I, generator=g, device="cuda") * 2 / I**0.5).to(torch.bfloat16)
    dy = torch.randn(T, H, generator=g, device="cuda").to(torch.bfloat16)
    return x, topk_idx, topk_w, w1, w2, dy


def _recv_rows(t, group):
    """Every rank's ``t`` at the receive rows ``src_rank * T_max + i`` that ``pool_row_to_recv_token`` numbers (zero
    past each rank's own count): what the dispatch delivers to any rank."""
    from primus_turbo.flydsl.mega.gfx1250.symm_buffer import get_symm_buffer_for_mega_moe

    t_max = int(get_symm_buffer_for_mega_moe().num_max_tokens_per_rank)
    buf = t.new_zeros((t_max,) + t.shape[1:])
    buf[: t.shape[0]] = t
    out = t.new_empty((group.size() * t_max,) + t.shape[1:])
    torch.distributed.all_gather_into_tensor(out, buf, group=group)
    return out


def _diff(got, want, what):
    """None when ``got`` and ``want`` are the same bytes, else what differs."""
    a, b = got.contiguous().view(torch.uint8).cpu(), want.contiguous().view(torch.uint8).cpu()
    if a.shape != b.shape:
        return f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    bad = (a != b).nonzero()
    return f"{what}: {bad.shape[0]} bytes differ, first at {bad[0].tolist()}" if bad.numel() else None


def _eq(got, want, what):
    assert (message := _diff(got, want, what)) is None, message


def _fold(pool, h, T):
    """The combine at EP1: token t's pool rows summed in f32 in ``recv_token_to_pool_rows`` order, rounded to bf16."""
    rows = h.recv_token_to_pool_rows[:T].long()
    acc = torch.zeros(T, pool.shape[1], dtype=torch.float32, device=pool.device)
    for k in range(rows.shape[1]):
        ok = rows[:, k] >= 0
        acc[ok] += pool[rows[ok, k]].float()
    return acc.to(torch.bfloat16)


def _gate_scatter(grad_gate, h, T, topk):
    """``grad_topk_weights``: each real pool row's gate gradient at its route."""
    rows = h.recv_token_to_pool_rows[:T].long()
    out = torch.zeros(T * topk, dtype=torch.float32, device=grad_gate.device)
    p = rows[rows >= 0]
    out[h.pool_row_to_route[p].long()] = grad_gate[p]
    return out.view(T, topk)


def _ref_cols(t, h):
    """The pool's column copy of ``t`` by mxfp8_ref from its MX rows (``t``'s dispatch): codes, packed scales. An
    expert's tokens are its whole 256-row pool segment (padding rows zero), as the kernels see the pool."""
    q, s = R.quant_rows(t.cpu())
    P = h.pool_row_to_route.numel()
    seg = torch.diff(h.num_tokens_per_expert_prefix.cpu())
    out = R.quant_grouped_fp8(q, s, h.pool_row_to_recv_token[:P].cpu(), seg)
    return out[2], R.pack_scales(out[3])


def _cols(c, P, shuffled):
    """A column copy's written part (the pool's ``P`` tokens): codes in row-major order, packed scales."""
    codes = R.unpreshuffle_b16(c[0].cpu()) if shuffled else c[0].cpu()
    return codes[:, :P], c[1].cpu()[:, : P // 128]


def check_stages(group, T, H, I, E, topk, seed, skew=False):
    """The MX stages on this rank bit for bit against the grouped path's kernels run in this rank's pool order on
    the same quantized operands: the delivered x rows, l1, the activation's column copy, the x / dy column copies,
    grad_gate, grad_l1's copies, dW1, dW2 (also into a filled bf16 main_grad). Each is per pool row (dW: per local
    expert), so it holds on every rank at any EP size; y, dx and d topk_w fold rows of several ranks and are
    checked bit for bit at EP1 only. ``E`` experts in all, ``E / group.size()`` per rank; a collective: the ranks
    reduce their mismatch counts before any of them asserts (a rank failing alone would leave its peers waiting)."""
    mismatches = []

    def _eq(got, want, what):  # collects instead of asserting
        if (message := _diff(got, want, what)) is not None:
            mismatches.append(message)

    from primus_turbo.flydsl.gemm.gemm_mxfp8_kernel_gfx1250 import (
        grouped_gemm_mxfp8_gfx1250,
        grouped_gemm_mxfp8_variable_k_gfx1250,
    )
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import DispatchHandle
    from primus_turbo.flydsl.quantization import mxfp8_quant_gfx1250 as Q
    from primus_turbo.pytorch.kernels.fused_mega_moe import fused_mega_moe_gfx1250_fp8_impl as MX

    world = group.size()
    x, topk_idx, topk_w, w1, w2, dy = _inputs(T, H, I, E, topk, seed + 101 * group.rank(), skew, E // world)
    l1, dw, handle, x_col, _ = MX.stage1_forward(x, w1, group, topk_idx, topk_w)
    y, act_col = MX.stage2_forward(l1, w2, handle, topk_idx, dw, SWIGLU)
    gl_row, gl_col, grad_gate, dW2 = MX.stage2_backward(dy, l1, dw, w2, handle, act_col, SWIGLU)
    dx, gtw, dW1 = MX.stage1_backward(gl_row, gl_col, grad_gate, x_col, w1, handle, topk_idx)
    h = DispatchHandle(*handle)
    P, lens, offs = h.pool_row_to_route.numel(), h.num_tokens_per_expert, h.num_tokens_per_expert_prefix
    rmap = h.pool_row_to_recv_token[:P]  # padding rows: out of range -> zero rows, as the pool's
    x_recv, dy_recv = _recv_rows(x, group), _recv_rows(dy, group)

    def gemm(a, w):
        return grouped_gemm_mxfp8_gfx1250(a[0], a[1], w[0], w[1], offs, offs, P)

    def wgrad(a, b, **kw):
        return grouped_gemm_mxfp8_variable_k_gfx1250(a[0], a[1], b[0], b[1], offs, **kw)

    W1, W1t = MX._weights(w1)[:2], MX._weights(w1)[2:]
    W2, W2t = MX._weights(w2)[:2], MX._weights(w2)[2:]
    # forward: rows quantized from bf16 x by the grouped path's gather-quant
    xr = Q.grouped_gather_quant_mxfp8(x_recv, rmap, lens, offs, total_m=P)
    xr_mega = MX._send(x, h, 0, cols=False)  # the forward's rows once more: what the dispatch delivered
    _eq(xr_mega[0][:P], xr[0][:P], "x rows")
    _eq(xr_mega[1].view(-1)[: P * H // 32], xr[1].view(-1)[: P * H // 32], "x row scales")
    r_l1 = gemm(xr, W1)
    _eq(l1, r_l1, "l1")
    act = Q.grouped_glu_quant_mxfp8(r_l1, dw, lens, offs, "silu", None, layout=Q.SHUF_COL)
    for got, want, what in zip(_cols(act_col, P, True), _cols(act[2:4], P, True), ("act columns", "scales")):
        _eq(got, want, what)
    # the column copy of x for dW1, from the dispatched fp8 rows
    for got, want, what in zip(_cols(x_col, P, True), _ref_cols(x_recv, h), ("x columns", "x column scales")):
        _eq(got, want, what)
    # backward
    dyr = Q.grouped_gather_quant_mxfp8(dy_recv, rmap, lens, offs, total_m=P)
    gp, *g = Q.grouped_dglu_quant_mxfp8(gemm(dyr, W2t), r_l1, dw, lens, offs, "silu", None)
    _eq(grad_gate, gp, "grad_gate")
    _eq(gl_row[0][:P], g[0][:P], "grad_l1 rows")
    _eq(gl_row[1].view(-1)[: P * 2 * I // 32], g[1].view(-1)[: P * 2 * I // 32], "grad_l1 row scales")
    for got, want, what in zip(
        _cols(gl_col, P, False), _cols(g[2:4], P, False), ("grad_l1 columns", "scales")
    ):
        _eq(got, want, what)
    if world == 1:
        _eq(y, _fold(gemm(act, W2), h, T), "y")
        _eq(dx, _fold(gemm(g, W1t), h, T), "dx")
        _eq(gtw, _gate_scatter(gp, h, T, topk), "grad_topk_weights")
    dy_col = MX._send(dy, h, MX._COLS_A)[2:]  # the backward's own dispatch of dy, again (same pool)
    for got, want, what in zip(
        _cols(dy_col, P, False), _ref_cols(dy_recv, h), ("dy columns", "dy column scales")
    ):
        _eq(got, want, what)
    _eq(dW2, wgrad(dy_col, (act[2], act[3])), "dW2")
    _eq(dW1, wgrad((g[2], g[3]), x_col), "dW1")
    # beta = 1 into a bf16 main_grad: rounded once, round(main_grad + the fp32 wgrad)
    main_grad = torch.randn_like(w2)
    want = (main_grad.float() + wgrad(dy_col, act_col, out_dtype=torch.float32)).to(torch.bfloat16)
    if MX._wgrad(dy_col, act_col, h, main_grad) is not None:
        mismatches.append("dW2 into main_grad: a gradient was returned")
    _eq(main_grad, want, "dW2 into main_grad")
    count = torch.tensor([len(mismatches)], device=x.device)
    torch.distributed.all_reduce(count, group=group)
    assert int(count) == 0, f"{int(count)} mismatches over the ranks; on rank {group.rank()}: {mismatches}"


@pytest.mark.parametrize(
    "T, H, I, E, topk, skew",
    [
        (512, 1024, 512, 8, 4, False),
        (1024, 7168, 2048, 32, 8, True),  # DeepSeek-V3's experts (EP1 slice), hot and empty experts
    ],
)
def test_stages_vs_grouped_path_bitwise_ep1(ep1_group, T, H, I, E, topk, skew):
    check_stages(ep1_group, T, H, I, E, topk, T + H, skew)


def _op_forward(group, x, topk_idx, topk_w, w1, w2, activation=SWIGLU, fuse=False):
    """The staged MX op forward on fresh leaves: ``(y, leaves)``; with ``fuse`` the wgrads go into a zeroed bf16
    ``main_grad`` (``fuse_wgrad_accum_pattern``). The EP2+ suite steps through this too."""
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
        fused_mega_moe_fp8_stage1,
        fused_mega_moe_fp8_stage2,
    )

    leaves = [t.clone().requires_grad_() for t in (x, topk_w, w1, w2)]
    pattern = "megatron" if fuse else None
    if fuse:
        for w in leaves[2:]:
            w.main_grad = torch.zeros_like(w)
            w.grad_added_to_main_grad = False
    stage1 = fused_mega_moe_fp8_stage1(  # (l1, dispatch_weights, handle, state)
        leaves[0], topk_idx, leaves[1], leaves[2], group, pattern, activation=activation
    )
    y = fused_mega_moe_fp8_stage2(
        *stage1, topk_idx, leaves[1], leaves[3], group, activation, fuse_wgrad_accum_pattern=pattern
    )
    return y, leaves


def _op_grads(y, leaves, dy, fuse=False):
    """The backward of ``_op_forward``: ``(y, dx, d topk_w, dW1, dW2)``, the wgrads from main_grad with ``fuse``."""
    y.backward(dy)
    assert not fuse or all(w.grad_added_to_main_grad for w in leaves[2:])
    wgrads = [w.main_grad if fuse else w.grad for w in leaves[2:]]
    return [y.detach(), leaves[0].grad, leaves[1].grad] + wgrads


def _op_step(group, x, topk_idx, topk_w, w1, w2, dy, activation=SWIGLU, fuse=False):
    """Forward + backward as a training step of its own: the MX weight cache is keyed by address, which fresh
    weight clones may reuse."""
    from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation

    advance_weight_generation()
    return _op_grads(*_op_forward(group, x, topk_idx, topk_w, w1, w2, activation, fuse), dy, fuse)


@pytest.mark.parametrize("fuse", [False, True])
def test_staged_op_ep1(ep1_group, fuse):
    """``fused_mega_moe_fp8_stage1/2`` on gfx1250 against the fp32 reference (the grouped MLP's 20 dB gate), and the
    wgrads into a bf16 ``main_grad`` with ``fuse_wgrad_accum_pattern``."""
    x, topk_idx, topk_w, w1, w2, dy = _inputs(1024, 7168, 2048, 32, 8, 11)
    got = _op_step(ep1_group, x, topk_idx, topk_w, w1, w2, dy, fuse=fuse)
    refs = [t.detach().float().requires_grad_() for t in (x, topk_w, w1, w2)]
    ref_y = _reference(refs[0], topk_idx, *refs[1:], SWIGLU)
    ref_y.backward(dy.float())
    want = [ref_y.detach()] + [t.grad for t in refs]
    for name, g, w in zip(("y", "dx", "d topk_w", "dW1", "dW2"), got, want):
        assert (snr := _snr(g, w)) > 20, f"{name} SNR {snr:.1f} dB"


def test_repeat_bitwise_ep1(ep1_group):
    """Every output, the wgrads included, bit for bit in three runs on the same inputs, with a call on other inputs
    between the first two: the prologue puts each pool segment's rows in receive-token order, so the wgrads'
    contraction order and their 32-row MX column blocks do not depend on the run."""
    first = _inputs(1024, 7168, 2048, 32, 8, 21, skew=True)
    other = _inputs(1024, 7168, 2048, 32, 8, 22)
    want = _op_step(ep1_group, *first)
    _op_step(ep1_group, *other)
    for run in (2, 3):
        for got, w, what in zip(_op_step(ep1_group, *first), want, ("y", "dx", "d topk_w", "dW1", "dW2")):
            _eq(got, w, f"run {run} {what}")


def test_interleaved_layers_ep1(ep1_group):
    """Two layers on one heap, forward A, forward B, backward B, backward A (A's backward after B's prologue and
    dispatches): each layer bit for bit as it is alone."""
    from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation

    layers = [_inputs(512, 1024, 512, 8, 4, seed, skew=seed % 2 == 0) for seed in (31, 32)]
    alone = [_op_step(ep1_group, *inputs) for inputs in layers]
    advance_weight_generation()  # one step: both layers' weights in one generation of the MX weight cache
    runs = [_op_forward(ep1_group, *inputs[:5]) for inputs in layers]
    got = [_op_grads(*run, inputs[5]) for run, inputs in zip(reversed(runs), reversed(layers))]  # B, then A
    for outputs, want, layer in zip(reversed(got), alone, "AB"):
        for g, w, what in zip(outputs, want, ("y", "dx", "d topk_w", "dW1", "dW2")):
            _eq(g, w, f"layer {layer} {what}")


def test_bf16_combine_after_combine_only_ep1(ep1_group):
    """A bf16 ``gemm_combine_nt_gfx1250`` on the heap a combine-only launch just used completes and is right: the
    combine-only launch tops every pool block's flag up to its epoch."""
    from primus_turbo.flydsl.mega.gfx1250.bf16.gemm_combine_kernel import (
        combine_only_gfx1250,
        gemm_combine_nt_gfx1250,
    )
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        DispatchHandle,
        run_dispatch_prologue,
    )

    T, H, I, E, topk = 333, 1024, 256, 8, 4
    x, topk_idx, topk_w, w1, w2, _ = _inputs(T, H, I, E, topk, 41, skew=True)
    topk_idx = topk_idx.to(torch.int64)
    h, _ = run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
    h = DispatchHandle(*h)
    P = h.pool_row_to_route.numel()
    g = torch.Generator(device="cuda").manual_seed(43)
    pool = torch.randn(P, H, generator=g, device="cuda").to(torch.bfloat16)
    # NaN at the padding rows: the combine-only launch reads none of them
    pool[h.pool_row_to_recv_token[:P].long() >= T] = float("nan")
    act = torch.randn(P, I, generator=g, device="cuda").to(torch.bfloat16)
    expert = h.tile_to_expert.long().repeat_interleave(256)[:P]
    l2 = torch.zeros(P, H, dtype=torch.bfloat16, device="cuda")
    for e in range(E):
        rows = expert == e
        l2[rows] = (act[rows].float() @ w2[e].float().T).to(torch.bfloat16)
    # the kernel's GEMM sums in another order than torch's, so a bf16 L2 term can round the other way (one ulp, at
    # most |term| / 128), and then the folded sum too
    want, slack = _fold(l2, h, T).float(), _fold(l2.abs(), h, T).float() / 32
    for _ in range(2):
        _eq(combine_only_gfx1250(pool, h, topk_idx), _fold(pool, h, T), "combine-only y")
        y = gemm_combine_nt_gfx1250(act, w2, h, topk_idx)
        assert ((y.float() - want).abs() <= slack).all(), "bf16 combine after a combine-only launch"


def test_padding_rows_zero_weight_ep1(ep1_group):
    """The prologue writes routing weight 0 at the pool's padding rows, whatever an earlier call left there (the
    MX wgrads contract over whole 256-row segments, padding rows included); the real rows keep theirs."""
    from primus_turbo.flydsl.mega.gfx1250.dispatch_prologue_kernel import (
        DispatchHandle,
        run_dispatch_prologue,
    )
    from primus_turbo.flydsl.mega.gfx1250.symm_buffer import get_symm_buffer_for_mega_moe

    T, H, I, E, topk = 333, 1024, 256, 8, 4
    x, topk_idx, topk_w, w1, _, _ = _inputs(T, H, I, E, topk, 51, skew=True)
    run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
    get_symm_buffer_for_mega_moe().weight_recv_buf.fill_(float("nan"))
    h, dw = run_dispatch_prologue(x, w1, ep1_group, topk_idx, topk_w)
    h = DispatchHandle(*h)
    token = h.pool_row_to_recv_token[: dw.numel()].long()
    real = token < T
    assert bool(torch.isfinite(dw).all()) and not bool(dw[~real].any()), "padding rows: weight 0"
    _eq(dw[real], topk_w.reshape(-1)[h.pool_row_to_route[real].long()], "real rows: their routing weight")


def test_contract_rejected_ep1(ep1_group):
    """What the gfx1250 MX stages cannot compute raises before any cross-rank work: a shape off the grid (no
    padding fallback), more than 256 experts, a token count past the combine buffer's 32-bit offsets, the bf16
    kernels' default activation (``None``), a non-positive clamp limit; and an expert bias in stage2."""
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8 import (
        fused_mega_moe_fp8_stage1,
        fused_mega_moe_fp8_stage2,
    )

    x, topk_idx, topk_w, w1, w2, _ = _inputs(64, 1024, 256, 8, 4, 61)
    fc1_bias = torch.zeros(w1.shape[:2], dtype=torch.bfloat16, device="cuda")
    for bad_w1 in (w1[:, :, :640].contiguous(), w1[:, :192].contiguous()):  # hidden % 512, intermediate % 128
        with pytest.raises(ValueError, match="hidden % 512"):
            fused_mega_moe_fp8_stage1(
                x[:, : bad_w1.shape[2]].contiguous(), topk_idx, topk_w, bad_w1, ep1_group
            )
    with pytest.raises(ValueError, match="256 experts"):
        fused_mega_moe_fp8_stage1(x, topk_idx, topk_w, w1[:1].expand(257, -1, -1), ep1_group)
    # 37504 tokens x 64 routes x hidden 1024 is past the combine buffer's 2**31 elements
    many, routes = x[:1].expand(37504, -1).contiguous(), topk_idx[:1, :1].expand(37504, 64)
    with pytest.raises(ValueError, match="tokens per rank"):
        fused_mega_moe_fp8_stage1(many, routes, None, w1, ep1_group)
    with pytest.raises(ValueError, match="clamp limit must be positive"):
        fused_mega_moe_fp8_stage1(x, topk_idx, topk_w, w1, ep1_group, activation=GLUActivation.swiglu(0.0))
    with pytest.raises(NotImplementedError, match="SiLU-SwiGLU"):
        fused_mega_moe_fp8_stage1(x, topk_idx, topk_w, w1, ep1_group, activation=None)
    l1, dw, handle, state = fused_mega_moe_fp8_stage1(x, topk_idx, topk_w, w1, ep1_group, activation=SWIGLU)
    with pytest.raises(NotImplementedError, match="fc1 bias"):
        fused_mega_moe_fp8_stage2(
            l1, dw, handle, state, topk_idx, topk_w, w2, ep1_group, SWIGLU, fc1_bias=fc1_bias
        )
    with pytest.raises(NotImplementedError, match="fc2 bias"):
        fused_mega_moe_fp8_stage2(
            l1, dw, handle, state, topk_idx, topk_w, w2, ep1_group, SWIGLU, fc2_bias=fc1_bias
        )
    with pytest.raises(NotImplementedError, match="SiLU-SwiGLU"):
        fused_mega_moe_fp8_stage2(l1, dw, handle, state, topk_idx, topk_w, w2, ep1_group, None)
