###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import copy
import gc
import importlib

import pytest
import torch
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.utils import run_and_get_code
from torch.utils.checkpoint import checkpoint

from primus_turbo.pytorch.core.delayed_scaling import (
    DelayedScaling,
    DelayedScalingState,
)
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    ScalingStrategy,
)
from primus_turbo.pytorch.core.quantized_tensor import QuantizedTensor
from primus_turbo.pytorch.ops import gemm_fp8, grouped_gemm_fp8
from primus_turbo.pytorch.ops.utils import _get_fp8_dtype

_gemm_fp8_module = importlib.import_module("primus_turbo.pytorch.ops.gemm_fp8")


def _delayed(format=Format.E4M3):
    return Float8QuantConfig(format=format, strategy=ScalingStrategy.DELAYED)


def _fwd_bwd(a0, b0, grad_c, trans_a, trans_b, config, state=None, fn=gemm_fp8):
    a = a0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    c = fn(a, b, trans_a, trans_b, a0.dtype, config, scaling_state=state)
    c.backward(grad_c)
    return c.detach(), a.grad, b.grad


def _assert_bitwise_equal(got, expected, names=("out", "a.grad", "b.grad")):
    for name, x, y in zip(names, got, expected):
        assert torch.equal(x, y), f"{name} differs"


def _amax(x):
    return x.float().abs().amax()


def _operands(m, n, k, trans_a, trans_b, scale=1.0):
    dtype = torch.bfloat16
    a0 = torch.randn((k, m) if trans_a else (m, k), dtype=dtype, device="cuda") * scale
    b0 = torch.randn((n, k) if trans_b else (k, n), dtype=dtype, device="cuda") * scale
    grad_c = torch.randn(m, n, dtype=dtype, device="cuda") * scale
    return a0, b0, grad_c


@pytest.mark.parametrize("layout, nt_bwd", [("NT", True), ("NT", False), ("NN", False), ("TN", False)])
@pytest.mark.parametrize("format", [Format.E4M3, Format.HYBRID])
def test_delayed_matches_dynamic_with_fresh_amax(monkeypatch, layout, nt_bwd, format):
    """Before the first step a delayed GEMM is the dynamic one and records exact amaxes; after a
    step on the same operands its scales are the dynamic ones, so it is bit-identical again."""
    monkeypatch.setattr(_gemm_fp8_module, "_deter_use_nt_layout_gemm_in_bwd", lambda trans_a, trans_b: nt_bwd)
    torch.manual_seed(0)
    trans_a, trans_b = layout[0] == "T", layout[1] == "T"
    a0, b0, grad_c = _operands(1032, 1536, 1024, trans_a, trans_b)
    dynamic = _fwd_bwd(a0, b0, grad_c, trans_a, trans_b, Float8QuantConfig(format=format))

    recipe = DelayedScaling()
    state = recipe.make_state(_delayed(format))
    _assert_bitwise_equal(_fwd_bwd(a0, b0, grad_c, trans_a, trans_b, _delayed(format), state), dynamic)
    expected_amax = torch.stack([_amax(a0), _amax(b0), _amax(grad_c)])
    assert torch.equal(torch.stack(state.amax), expected_amax)

    recipe.step()
    assert state.ready
    assert torch.equal(torch.stack(state.amax), torch.zeros(3, device="cuda"))
    fp8_max = torch.tensor(
        [torch.finfo(_get_fp8_dtype(format, stage)).max for stage in (True, True, False)], device="cuda"
    )
    assert torch.equal(torch.stack(state.scale), fp8_max / expected_amax)

    _assert_bitwise_equal(_fwd_bwd(a0, b0, grad_c, trans_a, trans_b, _delayed(format), state), dynamic)
    assert torch.equal(torch.stack(state.amax), expected_amax)


def test_delayed_casts_with_the_previous_steps_scale():
    """After a step, the GEMM casts with the stored scales, not the current operands' amaxes."""
    torch.manual_seed(0)
    a0, b0, grad_c = _operands(512, 1024, 1024, False, True)
    recipe = DelayedScaling()
    state = recipe.make_state(_delayed())
    _fwd_bwd(a0, b0, grad_c, False, True, _delayed(), state)
    recipe.step()

    # Operands 4x larger: a dynamic cast rescales, the delayed one saturates at the old scale.
    big = [t * 4 for t in (a0, b0, grad_c)]
    delayed_out, _, _ = _fwd_bwd(*big, False, True, _delayed(), state)
    dynamic_out, _, _ = _fwd_bwd(*big, False, True, Float8QuantConfig())
    assert not torch.equal(delayed_out, dynamic_out)
    assert torch.equal(torch.stack(state.amax), torch.stack([_amax(t) for t in big]))


@pytest.mark.parametrize("format", [Format.E4M3, Format.HYBRID])
def test_delayed_compile(format):
    """fullgraph compile across warm-up and later steps: outputs, gradients, amaxes and scales
    bit-identical to eager, in two graphs (before and after the state turns ready)."""
    torch.manual_seed(0)
    steps = [_operands(1032, 1536, 1024, False, True, scale=s) for s in (1.0, 2.0, 0.5)]
    runs = {}
    for mode in ("eager", "compiled"):
        torch._dynamo.reset()
        counters.clear()
        fn = torch.compile(gemm_fp8, fullgraph=True) if mode == "compiled" else gemm_fp8
        recipe = DelayedScaling()
        state = recipe.make_state(_delayed(format))
        results = []
        for a0, b0, grad_c in steps:
            results.append(_fwd_bwd(a0, b0, grad_c, False, True, _delayed(format), state, fn=fn))
            results.append((torch.stack(state.amax).clone(),))
            recipe.step()
            results.append((torch.stack(state.scale).clone(),))
        runs[mode] = results
        if mode == "compiled":
            assert counters["stats"]["unique_graphs"] == 2
    for got, expected in zip(runs["compiled"], runs["eager"]):
        _assert_bitwise_equal(got, expected)


def test_delayed_compile_keeps_outputs_aligned():
    """After step() packs the states, Inductor still takes the casts' and the GEMM's outputs as
    16-byte aligned, so the kernels that read the GEMM output keep their vectorised loads."""
    torch._dynamo.reset()
    a0, b0, _ = _operands(512, 1024, 1024, False, True)
    recipe = DelayedScaling()
    first = recipe.make_state(_delayed())  # noqa: F841 (the second state's views follow its rows)
    state = recipe.make_state(_delayed())
    recipe.step()

    def fn(a, b):
        return gemm_fp8(a, b, False, True, a.dtype, _delayed(), scaling_state=state) * 2

    # The FX graph cache key leaves out input alignment, so a graph cached for unaligned views
    # would be reused.
    with inductor_config.patch(alignment_asserts=True, force_disable_caches=True):
        _, code = run_and_get_code(torch.compile(fn, fullgraph=True), a0, b0)
    assert "assumed to be not aligned" not in "".join(code)


def test_delayed_running_max_over_micro_batches():
    """The amax folds every cast since the last step; the step takes the max and resets it."""
    torch.manual_seed(0)
    recipe = DelayedScaling()
    state = recipe.make_state(_delayed())
    small = _operands(256, 512, 512, False, True)
    large = _operands(256, 512, 512, False, True, scale=3.0)
    for ops in (small, large, small):
        _fwd_bwd(*ops, False, True, _delayed(), state)
    expected = torch.stack([torch.maximum(_amax(x), _amax(y)) for x, y in zip(small, large)])
    assert torch.equal(torch.stack(state.amax), expected)
    recipe.step()
    assert torch.equal(torch.stack(state.scale), torch.full_like(expected, 448.0) / expected)
    assert not torch.stack(state.amax).any()


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_delayed_step_keeps_scale_for_unusable_amax(bad):
    """A zero (GEMM not run) or non-finite amax keeps the previous scale; the others update."""
    recipe = DelayedScaling()
    state = recipe.make_state(_delayed())
    for t, v in zip(state.amax, (2.0, 4.0, 8.0)):
        t.fill_(v)
    recipe.step()
    before = torch.stack(state.scale).clone()
    state.amax[0].fill_(bad)
    state.amax[1].fill_(1.0)
    recipe.step()
    after = torch.stack(state.scale)
    assert after[0] == before[0]
    assert after[1] == 448.0
    assert after[2] == before[2]  # its amax was reset to 0 by the first step


def test_delayed_step_packs_states_into_flat_buffers():
    """step() makes the states views of one flat buffer; a state made later joins at the next
    step without changing the others' scales."""

    def storage(t):
        return t.untyped_storage().data_ptr()

    recipe = DelayedScaling()
    first = recipe.make_state(_delayed())
    for t, v in zip(first.amax, (2.0, 4.0, 8.0)):
        t.fill_(v)
    recipe.step()
    assert storage(first.scale[0]) == storage(first.scale[2]) != storage(first.amax[0])
    before = torch.stack(first.scale).clone()

    second = recipe.make_state(_delayed())
    assert not second.ready and storage(second.scale[0]) != storage(first.scale[0])
    second.amax[0].fill_(1.0)
    recipe.step()
    assert torch.equal(torch.stack(first.scale), before)
    assert second.ready and second.scale[0] == 448.0
    assert storage(second.scale[0]) == storage(first.scale[0])
    # Inductor treats a cast's outputs as unaligned if its scale or amax view is.
    for t in (*first.scale, *first.amax, *second.scale, *second.amax):
        assert t.dim() == 0 and t.data_ptr() % 16 == 0


def test_delayed_activation_checkpoint_recomputes_identically():
    """Scales only change in step(), so a checkpointed forward recomputes the same FP8 operands."""
    torch.manual_seed(0)
    a0, b0, grad_c = _operands(512, 1024, 1024, False, True)
    results = []
    for use_checkpoint in (False, True):
        recipe = DelayedScaling()
        state = recipe.make_state(_delayed())
        for _ in range(2):  # warm-up, then ready
            a = a0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)

            def f(a, b, state=state):
                return gemm_fp8(a, b, False, True, a0.dtype, _delayed(), scaling_state=state)

            c = checkpoint(f, a, b, use_reentrant=False) if use_checkpoint else f(a, b)
            c.backward(grad_c)
            amax = torch.stack(state.amax).clone()
            recipe.step()
        results.append((c.detach(), a.grad, b.grad, amax))
    _assert_bitwise_equal(results[1], results[0], ("out", "a.grad", "b.grad", "amax"))


def test_delayed_same_gemm_twice_before_backward():
    """One state used by two calls in a forward: both backwards run, the amax holds the max."""
    torch.manual_seed(0)
    recipe = DelayedScaling()
    state = recipe.make_state(_delayed())
    w = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    for _ in range(2):
        h = gemm_fp8(x, w, False, True, x.dtype, _delayed(), scaling_state=state)
        y = gemm_fp8(h, w, False, True, x.dtype, _delayed(), scaling_state=state)
        y.float().sum().backward()
        assert torch.equal(state.amax[0], torch.maximum(_amax(x), _amax(h)))
        recipe.step()


def test_delayed_step_between_forward_and_backward_is_an_error():
    recipe = DelayedScaling()
    state = recipe.make_state(_delayed())
    a0, b0, grad_c = _operands(256, 512, 512, False, True)
    _fwd_bwd(a0, b0, grad_c, False, True, _delayed(), state)
    recipe.step()
    a = a0.clone().requires_grad_(True)
    c = gemm_fp8(a, b0, False, True, a0.dtype, _delayed(), scaling_state=state)
    recipe.step()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        c.backward(grad_c)


def test_delayed_fused_bgrad_accum():
    """Megatron's fused weight-gradient accumulation (an untraced path) takes the state too."""
    torch.manual_seed(0)
    a0, b0, grad_c = _operands(512, 1024, 1024, False, True)
    previous = torch.randn(b0.shape, dtype=torch.float32, device="cuda")

    def run(config, state=None):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        b.main_grad = previous.clone()
        b.grad_added_to_main_grad = False
        c = gemm_fp8(a, b, False, True, a0.dtype, config, "megatron", scaling_state=state)
        c.backward(grad_c)
        return c.detach(), a.grad, b.main_grad

    dynamic = run(Float8QuantConfig())
    recipe = DelayedScaling()
    state = recipe.make_state(_delayed())
    for _ in range(2):  # warm-up, then ready with the same operands' scales
        _assert_bitwise_equal(run(_delayed(), state), dynamic, ("out", "a.grad", "main_grad"))
        recipe.step()


def test_delayed_recipe_forgets_dead_states_and_registers_copies():
    recipe = DelayedScaling()
    kept = recipe.make_state(_delayed())
    dropped = recipe.make_state(_delayed())
    del dropped
    gc.collect()
    assert recipe.states() == [kept]

    clone = copy.deepcopy(kept)
    assert recipe.states() == [kept, clone]
    clone.amax[0].fill_(2.0)
    recipe.step()
    assert clone.scale[0] == 224.0 and kept.scale[0] == 1.0
    assert clone.scale[0].data_ptr() != kept.scale[0].data_ptr()


def test_delayed_state_from_meta_device_starts_over():
    state = DelayedScaling().make_state(_delayed(), device="meta")
    state.to("cuda")
    assert state.device.type == "cuda" and not state.ready
    assert torch.equal(torch.stack(state.scale), torch.ones(3, device="cuda"))
    assert not torch.stack(state.amax).any()


def test_delayed_validation():
    recipe = DelayedScaling()
    state = recipe.make_state(_delayed())
    a, b, _ = _operands(256, 512, 512, False, True)
    with pytest.raises(ValueError, match="scaling_state"):
        gemm_fp8(a, b, False, True, a.dtype, _delayed())
    with pytest.raises(ValueError, match="scaling_state"):
        gemm_fp8(a, b, False, True, a.dtype, Float8QuantConfig(), scaling_state=state)
    a_q = QuantizedTensor.quantize(a, _get_fp8_dtype(Format.E4M3, True), ScalingGranularity.TENSORWISE)
    with pytest.raises(ValueError, match="pre-quantized"):
        gemm_fp8(a_q, b, False, True, a.dtype, _delayed(), scaling_state=state)
    with pytest.raises(ValueError, match="DELAYED"):
        recipe.make_state(Float8QuantConfig())
    with pytest.raises(AssertionError, match="TENSORWISE"):
        Float8QuantConfig(granularity=ScalingGranularity.ROWWISE, strategy=ScalingStrategy.DELAYED)
    with pytest.raises(ValueError, match="DYNAMIC only"):
        group_lens = torch.tensor([256], dtype=torch.int64, device="cuda")
        grouped_gemm_fp8(a, b.unsqueeze(0), group_lens, config=_delayed())

    source, globals_ = _delayed(Format.HYBRID).__fx_repr__()
    assert eval(source, globals_) == _delayed(Format.HYBRID)


def _fp8_max(format=Format.E4M3):
    return torch.tensor(
        [torch.finfo(_get_fp8_dtype(format, stage)).max for stage in (True, True, False)], device="cuda"
    )


def test_delayed_refresh_interval_records_every_nth_step():
    """refresh_interval=3 records in steps 0, 3, 6 only and keeps the scales in between."""
    torch.manual_seed(0)
    steps = [_operands(512, 1024, 1024, False, True, scale=1.0 + i) for i in range(7)]
    recipe = DelayedScaling(refresh_interval=3)
    state = recipe.make_state(_delayed())
    expected_scale = None
    for i, (a0, b0, grad_c) in enumerate(steps):
        assert state.recording == (i % 3 == 0)
        _fwd_bwd(a0, b0, grad_c, False, True, _delayed(), state)
        amax = torch.stack([_amax(a0), _amax(b0), _amax(grad_c)])
        assert torch.equal(torch.stack(state.amax), amax if i % 3 == 0 else torch.zeros(3, device="cuda"))
        recipe.step()
        if i % 3 == 0:
            expected_scale = _fp8_max() / amax
        assert torch.equal(torch.stack(state.scale), expected_scale)


def test_delayed_step_that_does_not_record_casts_like_one_that_does():
    """Skipping the amax update leaves the cast unchanged: same outputs and gradients, bitwise."""
    torch.manual_seed(0)
    warm, later = _operands(512, 1024, 1024, False, True), _operands(512, 1024, 1024, False, True, 3.0)
    results = []
    for interval in (2, 1):
        recipe = DelayedScaling(refresh_interval=interval)
        state = recipe.make_state(_delayed())
        _fwd_bwd(*warm, False, True, _delayed(), state)
        recipe.step()
        assert state.recording == (interval == 1)
        results.append(_fwd_bwd(*later, False, True, _delayed(), state))
    _assert_bitwise_equal(*results)


def test_delayed_refresh_interval_compile():
    """fullgraph compile with refresh_interval=2: bit-identical to eager, in three graphs (warm-up,
    recording and not recording); the default interval keeps two (test_delayed_compile)."""
    torch.manual_seed(0)
    steps = [_operands(1032, 1536, 1024, False, True, scale=s) for s in (1.0, 2.0, 0.5, 4.0, 0.25)]
    runs = {}
    for mode in ("eager", "compiled"):
        torch._dynamo.reset()
        counters.clear()
        fn = torch.compile(gemm_fp8, fullgraph=True) if mode == "compiled" else gemm_fp8
        recipe = DelayedScaling(refresh_interval=2)
        state = recipe.make_state(_delayed())
        results = []
        for a0, b0, grad_c in steps:
            results.append(_fwd_bwd(a0, b0, grad_c, False, True, _delayed(), state, fn=fn))
            results.append((torch.stack(state.amax).clone(),))
            recipe.step()
            results.append((torch.stack(state.scale).clone(),))
        runs[mode] = results
        if mode == "compiled":
            assert counters["stats"]["unique_graphs"] == 3
    for got, expected in zip(runs["compiled"], runs["eager"]):
        _assert_bitwise_equal(got, expected)


def test_delayed_state_added_between_refreshes_gets_its_scales():
    """A state made in a step that does not record still records its warm-up amaxes, and the next
    step() turns them into scales; the other states keep theirs."""
    torch.manual_seed(0)
    a0, b0, grad_c = _operands(512, 1024, 1024, False, True)
    recipe = DelayedScaling(refresh_interval=4)
    first = recipe.make_state(_delayed())
    _fwd_bwd(a0, b0, grad_c, False, True, _delayed(), first)
    recipe.step()
    kept = torch.stack(first.scale).clone()

    second = recipe.make_state(_delayed())
    big = [t * 2 for t in (a0, b0, grad_c)]
    _fwd_bwd(*big, False, True, _delayed(), first)
    _fwd_bwd(*big, False, True, _delayed(), second)
    assert not torch.stack(first.amax).any()
    recipe.step()
    assert torch.equal(torch.stack(first.scale), kept)
    assert second.ready and not second.recording
    assert torch.equal(torch.stack(second.scale), _fp8_max() / torch.stack([_amax(t) for t in big]))
    assert not torch.stack(second.amax).any()


@pytest.mark.parametrize("interval", [0, -1, 2.0, True])
def test_delayed_refresh_interval_validation(interval):
    with pytest.raises(ValueError, match="refresh_interval"):
        DelayedScaling(refresh_interval=interval)


def test_delayed_step_without_states_is_a_no_op():
    DelayedScaling().step()
    assert isinstance(DelayedScaling().make_state(_delayed()), DelayedScalingState)
