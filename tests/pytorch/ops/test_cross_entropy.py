###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.cross_entropy import cross_entropy


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [1, 33, 1024, 32769, 128256])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided_vocab", "2d"])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_loss_and_gradient(dtype, vocab, layout, smoothing):
    torch.manual_seed(31)
    x = (torch.randn(2, 3, vocab * (2 if layout == "strided_vocab" else 1), device="cuda") * 8).to(dtype)
    if layout == "transpose":
        x = x.transpose(0, 1)
    elif layout == "strided_vocab":
        x = x[..., ::2]
    elif layout == "2d":
        x = x.reshape(-1, vocab)
    x.requires_grad_()
    original = x.detach().clone()
    target = torch.randint(vocab, x.shape[:-1], device="cuda")
    target.flatten()[0] = -100
    upstream = torch.randn_like(target, dtype=torch.float32)
    upstream.flatten()[1] = 0
    y = cross_entropy(x, target, label_smoothing=smoothing)
    (dx,) = torch.autograd.grad(y, x, upstream)
    ref_x = original.float().requires_grad_()
    ref_y = F.cross_entropy(
        ref_x.reshape(-1, vocab), target.reshape(-1), reduction="none", label_smoothing=smoothing
    ).reshape(target.shape)
    (ref_dx,) = torch.autograd.grad(ref_y, ref_x, upstream)
    torch.testing.assert_close(y, ref_y, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        dx, ref_dx.to(dtype), rtol=0.008 if dtype == torch.bfloat16 else 2e-5, atol=2e-6
    )
    torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)
    assert y.dtype == torch.float32
    assert torch.count_nonzero(dx.reshape(-1, vocab)[0]).item() == 0


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_buffer_ownership_and_broadcast_gradient(overwrite, all_ignored):
    x = torch.zeros(2, 3, 128256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.full((2, 3), -100 if all_ignored else 7, device="cuda", dtype=torch.int64)
    before = x.detach().clone()
    saved_ptrs = []

    def pack(tensor):
        saved_ptrs.append(tensor.data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        loss = cross_entropy(x, target, overwrite_input=overwrite)
    torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)
    (grad,) = torch.autograd.grad(loss.sum(), x, retain_graph=True)
    assert (saved_ptrs[0] == x.data_ptr()) == overwrite
    if overwrite:
        assert grad.data_ptr() == x.data_ptr()
    else:
        torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)
    if all_ignored:
        assert torch.count_nonzero(loss).item() == 0
        assert torch.count_nonzero(grad).item() == 0
    else:
        assert grad[0, 0, 7].item() == -1.0
    with pytest.raises(RuntimeError, match="one backward"):
        torch.autograd.grad(loss.sum(), x)


def test_noncontiguous_targets():
    x = torch.randn(3, 2, 33, device="cuda", requires_grad=True)
    target = torch.randint(33, (2, 3), device="cuda").T
    loss = cross_entropy(x, target)
    ref = F.cross_entropy(x.flatten(0, 1), target.reshape(-1), reduction="none").reshape(3, 2)
    torch.testing.assert_close(loss, ref)


def test_validation():
    x = torch.zeros(2, 3, 33, device="cuda", dtype=torch.bfloat16)
    y = torch.zeros(2, 3, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="contiguous"):
        cross_entropy(x.transpose(0, 1), y.T, overwrite_input=True)
    with pytest.raises(ValueError, match="target.shape"):
        cross_entropy(x, y.reshape(-1))
    with pytest.raises(TypeError, match="int64"):
        cross_entropy(x, y.int())
    with pytest.raises(TypeError, match="BF16 or FP32"):
        cross_entropy(x.half(), y)
    with pytest.raises(ValueError, match="label_smoothing"):
        cross_entropy(x, y, label_smoothing=1.1)


def test_vocab_stride_beyond_int32():
    # 8 GiB padding makes a signed-i32 wrap land inside allocated storage with
    # a wrong sentinel value, rather than faulting the GPU. Only 17 logits are
    # used; the large backing allocation tests address arithmetic, not compute.
    if torch.cuda.mem_get_info()[0] < 10 * 1024**3:
        pytest.skip("64-bit strided-address regression requires 10 GiB free")
    boundary = 2**31
    storage = torch.empty(2 * boundary + 1, device="cuda", dtype=torch.bfloat16)
    storage[0] = 42
    x = storage.as_strided((1, 17), (0, 2**27), storage_offset=boundary)
    x.zero_()
    x.requires_grad_(True)
    target = torch.tensor([16], device="cuda", dtype=torch.int64)
    reference = torch.zeros(1, 17, device="cuda", requires_grad=True)
    loss = cross_entropy(x, target)
    expected = F.cross_entropy(reference, target, reduction="none")
    (grad,) = torch.autograd.grad(loss.sum(), x)
    (ref_grad,) = torch.autograd.grad(expected.sum(), reference)
    torch.testing.assert_close(loss, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(grad, ref_grad.to(x.dtype), rtol=0.008, atol=2e-6)
