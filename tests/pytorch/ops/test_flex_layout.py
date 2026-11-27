###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``flex/_layout.py``: which input layouts reach the backend uncopied.

The gate used to be ``t.is_contiguous()``, which passes through a ``bhsd``-contiguous
tensor and materialises everything else. Everything else includes the layout Megatron
actually hands attention -- ``sbhd``-contiguous storage viewed as ``[B, S, H, D]`` --
so three copies were made per layer, per step, on the one caller that matters, while
the direct ``flash_attn_func`` path this layer is compared against copied nothing.

These tests pin the distinction that bug turned on: passthrough is decided by whether
the backend can *address* the bytes (``_infer_qkv_format``'s three orders), not by
whether they happen to be contiguous. They are pure stride bookkeeping, so they run on
CPU with no accelerator.
"""

import pytest
import torch

from primus_turbo.pytorch.ops.attention.attention_utils import _infer_qkv_format
from primus_turbo.pytorch.ops.attention.flex._layout import (
    _backend_format,
    from_backend_layout,
    to_backend_layout,
    to_backend_layout_qkv,
)

B, S, HQ, HKV, D = 4, 64, 8, 2, 16


def _bhsd_from_bshd(b=B, h=HQ, s=S, d=D):
    """A bhsd-shaped view of ``bshd``-contiguous storage (a synthetic bshd caller)."""
    return torch.randn(b, s, h, d).transpose(1, 2)


def _bhsd_from_sbhd(b=B, h=HQ, s=S, d=D):
    """A bhsd-shaped view of ``sbhd``-contiguous storage (what Megatron hands us).

    Mirrors ``PrimusTurboAttention.forward``: ``.contiguous()`` on sbhd storage, then
    ``permute(1, 0, 2, 3)`` to a bshd *shape*, which ``flex_attention_bshd`` transposes
    once more on the way in.
    """
    return torch.randn(s, b, h, d).permute(1, 0, 2, 3).transpose(1, 2)


def _bhsd_contig(b=B, h=HQ, s=S, d=D):
    """A genuinely ``bhsd``-contiguous tensor."""
    return torch.randn(b, h, s, d)


def _shares_storage(a, b):
    return a.data_ptr() == b.data_ptr() and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


# ---------------------------------------------------------------------------
# _backend_format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda: torch.randn(B, S, HQ, D), "bshd"),
        (lambda: torch.randn(S, B, HQ, D).permute(1, 0, 2, 3), "sbhd"),
        (lambda: torch.randn(B, HQ, S, D).transpose(1, 2), "bhsd"),
    ],
)
def test_backend_format_matches_infer_qkv_format(make, expected):
    """The gate must classify exactly what the kernel path classifies.

    A layout this function waves through but ``_infer_qkv_format`` rejects would turn a
    copy we skipped into an assertion inside the kernel call, so the two have to agree
    order for order -- not merely both be "some sensible classifier".
    """
    t = make()
    assert _backend_format(t) == expected
    assert _infer_qkv_format(t, t, t) == expected


def test_backend_format_rejects_non_unit_last_stride():
    """``_infer_qkv_format`` asserts on stride[-1] != 1; here it must be a soft None."""
    t = torch.randn(B, S, HQ, D * 2)[..., ::2]
    assert t.stride(-1) != 1
    assert _backend_format(t) is None


def test_backend_format_rejects_wrong_rank():
    assert _backend_format(torch.randn(B, S, HQ)) is None


# ---------------------------------------------------------------------------
# to_backend_layout_qkv -- the passthrough decision
# ---------------------------------------------------------------------------


def test_sbhd_view_passes_through_uncopied():
    """The regression this file exists for: Megatron's layout must not be materialised."""
    q, k, v = (_bhsd_from_sbhd(h=h) for h in (HQ, HKV, HKV))
    q_be, k_be, v_be = to_backend_layout_qkv(q, k, v)
    for src, be in ((q, q_be), (k, k_be), (v, v_be)):
        assert _shares_storage(src, be), "sbhd input was copied; the backend can address it as it lies"
        assert be.shape == (src.shape[0], src.shape[2], src.shape[1], src.shape[3])
    assert _infer_qkv_format(q_be, k_be, v_be) == "sbhd"


def test_bshd_view_passes_through_uncopied():
    q, k, v = (_bhsd_from_bshd(h=h) for h in (HQ, HKV, HKV))
    q_be, k_be, v_be = to_backend_layout_qkv(q, k, v)
    for src, be in ((q, q_be), (k, k_be), (v, v_be)):
        assert _shares_storage(src, be)
    assert _infer_qkv_format(q_be, k_be, v_be) == "bshd"


def test_bhsd_contiguous_still_passes_through():
    """The behaviour the old ``is_contiguous()`` gate already had must be preserved."""
    q, k, v = (_bhsd_contig(h=h) for h in (HQ, HKV, HKV))
    q_be, k_be, v_be = to_backend_layout_qkv(q, k, v)
    for src, be in ((q, q_be), (k, k_be), (v, v_be)):
        assert _shares_storage(src, be)
    assert _infer_qkv_format(q_be, k_be, v_be) == "bhsd"


def test_batch_one_lands_bshd_contiguous():
    """B == 1 keeps the sbhd-only FlyDSL / HipKittens backends eligible -- see the docstring.

    The guarantee is the *layout of the result*, not that bytes moved. At B == 1 the sbhd
    and bshd orders are the same bytes, so this input is already bshd-contiguous and
    ``.contiguous()`` is a no-op; the invariant those backends need still holds.
    ``test_batch_one_bhsd_is_copied`` covers the case where the copy is real.
    """
    q, k, v = (_bhsd_from_sbhd(b=1, h=h) for h in (HQ, HKV, HKV))
    q_be, k_be, v_be = to_backend_layout_qkv(q, k, v)
    for be in (q_be, k_be, v_be):
        assert be.is_contiguous()
    assert _infer_qkv_format(q_be, k_be, v_be) == "bshd"


def test_batch_one_bhsd_is_copied():
    """A bhsd-contiguous B == 1 input is not bshd bytes, so here the copy must happen."""
    q, k, v = (_bhsd_contig(b=1, h=h) for h in (HQ, HKV, HKV))
    q_be, k_be, v_be = to_backend_layout_qkv(q, k, v)
    for src, be in ((q, q_be), (k, k_be), (v, v_be)):
        assert not _shares_storage(src, be)
        assert be.is_contiguous()
        torch.testing.assert_close(be, src.transpose(1, 2), rtol=0, atol=0)


def test_mixed_layouts_materialise_all_three():
    """Per-tensor passthrough could hand the backend an sbhd q beside a bshd k.

    Each is addressable alone; the combination trips ``_infer_qkv_format``'s
    agreement assert inside the kernel call. So a disagreement must materialise
    everything rather than pass through the ones that happen to match.
    """
    q = _bhsd_from_sbhd(h=HQ)
    k = _bhsd_from_bshd(h=HKV)
    v = _bhsd_from_bshd(h=HKV)
    q_be, k_be, v_be = to_backend_layout_qkv(q, k, v)
    # What must hold is that the three now agree, in bshd. Only q's bytes actually move:
    # k and v were already bshd-contiguous, so ``.contiguous()`` is a no-op on them, and
    # demanding a fresh allocation there would be asserting on the wrong thing.
    for be in (q_be, k_be, v_be):
        assert be.is_contiguous()
    assert not _shares_storage(q, q_be)
    assert _infer_qkv_format(q_be, k_be, v_be) == "bshd"


def test_unaddressable_layout_materialises():
    q = _bhsd_from_sbhd(h=HQ)[..., ::2]  # stride[-1] == 2
    k = _bhsd_from_sbhd(h=HKV)[..., ::2]
    v = _bhsd_from_sbhd(h=HKV)[..., ::2]
    q_be, k_be, v_be = to_backend_layout_qkv(q, k, v)
    for be in (q_be, k_be, v_be):
        assert be.is_contiguous()


# ---------------------------------------------------------------------------
# Values, not just strides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [_bhsd_from_sbhd, _bhsd_from_bshd, _bhsd_contig])
def test_conversion_is_value_preserving(make):
    """Whichever branch is taken, ``[b, h, s, d]`` must land at ``[b, s, h, d]`` unchanged."""
    q, k, v = (make(h=h) for h in (HQ, HKV, HKV))
    for src, be in zip((q, k, v), to_backend_layout_qkv(q, k, v)):
        torch.testing.assert_close(be, src.transpose(1, 2), rtol=0, atol=0)


@pytest.mark.parametrize("make", [_bhsd_from_sbhd, _bhsd_from_bshd, _bhsd_contig])
def test_round_trip_restores_bhsd(make):
    """``from_backend_layout`` undoes the shape change and always yields contiguous bhsd."""
    t = make()
    back = from_backend_layout(to_backend_layout(t))
    assert back.shape == t.shape
    assert back.is_contiguous()
    torch.testing.assert_close(back, t, rtol=0, atol=0)


def test_single_tensor_helper_agrees_with_triple():
    """``to_backend_layout`` is kept for single-tensor callers; it must not drift."""
    for make in (_bhsd_from_sbhd, _bhsd_from_bshd, _bhsd_contig):
        t = make()
        one = to_backend_layout(t)
        three = to_backend_layout_qkv(t, t, t)[0]
        assert _shares_storage(one, three) == _shares_storage(t, three)
        assert one.stride() == three.stride()
