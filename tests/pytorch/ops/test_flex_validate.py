###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``flex/_validate.py``: argument validation for the flex entry points."""

import pytest
import torch

from primus_turbo.pytorch.ops.attention.flex._validate import (
    _normalise_explicit_softcap,
    _validate_and_adapt_bias,
    _validate_cu_seqlens,
    _validate_dropout_p,
    _validate_explicit_alibi_slopes,
    _validate_explicit_sink,
    _validate_max_seqlen,
    _validate_qkv_varlen,
    _validate_window_size,
)

from .flex_test_utils import _cu_from_seqlens, _make_thd


def test_validate_explicit_alibi_slopes_ok():
    slopes = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
    out = _validate_explicit_alibi_slopes(slopes, hq=3, device=torch.device("cpu"))
    assert out.shape == (3,)
    assert out.dtype == torch.float32
    assert torch.allclose(out, slopes)


def test_validate_explicit_alibi_slopes_rejects_non_tensor():
    with pytest.raises(ValueError):
        _validate_explicit_alibi_slopes([1.0, 0.5, 0.25], hq=3, device=torch.device("cpu"))


def test_validate_explicit_alibi_slopes_rejects_2d():
    slopes = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        _validate_explicit_alibi_slopes(slopes, hq=3, device=torch.device("cpu"))


def test_validate_explicit_alibi_slopes_rejects_wrong_length():
    slopes = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(ValueError):
        _validate_explicit_alibi_slopes(slopes, hq=8, device=torch.device("cpu"))


def test_validate_explicit_alibi_slopes_rejects_non_fp32():
    slopes = torch.zeros(4, dtype=torch.float16)
    with pytest.raises(ValueError):
        _validate_explicit_alibi_slopes(slopes, hq=4, device=torch.device("cpu"))


def test_normalise_explicit_softcap_none_and_zero_disable():
    assert _normalise_explicit_softcap(None) == 0.0
    assert _normalise_explicit_softcap(0) == 0.0
    assert _normalise_explicit_softcap(0.0) == 0.0


def test_normalise_explicit_softcap_positive_kept():
    assert _normalise_explicit_softcap(30.0) == 30.0


def test_normalise_explicit_softcap_negative_raises():
    with pytest.raises(ValueError):
        _normalise_explicit_softcap(-1.0)


def test_normalise_explicit_softcap_nan_raises():
    with pytest.raises(ValueError):
        _normalise_explicit_softcap(float("nan"))


def test_validate_dropout_p_zero_and_valid_values():
    assert _validate_dropout_p(0.0) == 0.0
    assert _validate_dropout_p(0) == 0.0
    assert abs(_validate_dropout_p(0.1) - 0.1) < 1e-9
    assert abs(_validate_dropout_p(0.999) - 0.999) < 1e-9


def test_validate_dropout_p_rejects_one_and_above():
    with pytest.raises(ValueError):
        _validate_dropout_p(1.0)
    with pytest.raises(ValueError):
        _validate_dropout_p(1.5)


def test_validate_dropout_p_rejects_negative():
    with pytest.raises(ValueError):
        _validate_dropout_p(-0.1)


def test_validate_dropout_p_rejects_nan():
    with pytest.raises(ValueError):
        _validate_dropout_p(float("nan"))


def test_validate_dropout_p_rejects_non_number():
    with pytest.raises(ValueError):
        _validate_dropout_p([0.1])


def test_validate_explicit_sink_ok():
    sink = torch.zeros(4, dtype=torch.float32)
    out = _validate_explicit_sink(sink, hq=4, head_dim_qk=64, head_dim_v=64, device=torch.device("cpu"))
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_validate_explicit_sink_rejects_non_tensor():
    with pytest.raises(ValueError):
        _validate_explicit_sink([0.0] * 4, hq=4, head_dim_qk=64, head_dim_v=64, device=torch.device("cpu"))


def test_validate_explicit_sink_rejects_2d():
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros((2, 4), dtype=torch.float32),
            hq=4,
            head_dim_qk=64,
            head_dim_v=64,
            device=torch.device("cpu"),
        )


def test_validate_explicit_sink_rejects_wrong_length():
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(3, dtype=torch.float32),
            hq=8,
            head_dim_qk=64,
            head_dim_v=64,
            device=torch.device("cpu"),
        )


def test_validate_explicit_sink_rejects_non_fp32():
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float16),
            hq=4,
            head_dim_qk=64,
            head_dim_v=64,
            device=torch.device("cpu"),
        )


def test_validate_explicit_sink_rejects_mismatched_head_dim():
    # Sink kernel path requires head_dim_qk == head_dim_v.
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float32),
            hq=4,
            head_dim_qk=128,
            head_dim_v=64,
            device=torch.device("cpu"),
        )


def test_validate_explicit_sink_rejects_non_pow2_head_dim():
    # Sink kernel path requires a power-of-two head dim (48 is not).
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float32),
            hq=4,
            head_dim_qk=48,
            head_dim_v=48,
            device=torch.device("cpu"),
        )


def test_validate_and_adapt_bias_2d_ok():
    bias = torch.randn(16, 16, dtype=torch.float32)
    out = _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert out.shape == (16, 16)
    assert out.dtype == torch.bfloat16  # adapted to q's dtype
    assert out.is_contiguous()


def test_validate_and_adapt_bias_leading_singletons_squeezed():
    for shape in ((1, 16, 16), (1, 1, 16, 16)):
        bias = torch.randn(*shape, dtype=torch.float16)
        out = _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.float16, device=torch.device("cpu"))
        assert out.shape == (16, 16)


def test_validate_and_adapt_bias_rectangular_ok():
    bias = torch.randn(8, 16, dtype=torch.bfloat16)
    out = _validate_and_adapt_bias(bias, sq=8, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert out.shape == (8, 16)


def test_validate_and_adapt_bias_rejects_per_head_4d():
    # A genuine per-head bias cannot map to the kernel's single [Sq,Skv] bias.
    bias = torch.randn(2, 4, 16, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


def test_validate_and_adapt_bias_rejects_per_batch_3d():
    bias = torch.randn(2, 16, 16, dtype=torch.bfloat16)  # leading dim 2 != 1
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


def test_validate_and_adapt_bias_rejects_wrong_last_dims():
    bias = torch.randn(16, 8, dtype=torch.bfloat16)  # skv 8 != 16
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


def test_validate_and_adapt_bias_rejects_non_tensor():
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(
            [[0.0] * 16] * 16, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu")
        )


def test_validate_and_adapt_bias_rejects_non_float():
    bias = torch.zeros(16, 16, dtype=torch.int32)
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


# ---- _validate_max_seqlen -------------------------------------------------


def test_validate_max_seqlen_ok():
    assert _validate_max_seqlen("max_seqlen_q", 256) == 256


def test_validate_max_seqlen_rejects_float():
    with pytest.raises(ValueError):
        _validate_max_seqlen("max_seqlen_q", 256.0)


def test_validate_max_seqlen_rejects_bool():
    with pytest.raises(ValueError):
        _validate_max_seqlen("max_seqlen_q", True)


def test_validate_max_seqlen_rejects_non_positive():
    with pytest.raises(ValueError):
        _validate_max_seqlen("max_seqlen_q", 0)
    with pytest.raises(ValueError):
        _validate_max_seqlen("max_seqlen_q", -5)


# ---- _validate_window_size ------------------------------------------------


def test_validate_window_size_full_and_left_window():
    assert _validate_window_size((-1, -1)) == (-1, -1)
    assert _validate_window_size((256, 0)) == (256, 0)
    assert _validate_window_size([128, 0]) == (128, 0)  # list accepted, coerced to tuple


def test_validate_window_size_rejects_wrong_length():
    with pytest.raises(ValueError):
        _validate_window_size((1, 2, 3))


def test_validate_window_size_rejects_non_int():
    with pytest.raises(ValueError):
        _validate_window_size((128.0, 0))
    with pytest.raises(ValueError):
        _validate_window_size((True, 0))  # bool is not accepted as a window bound


def test_validate_window_size_rejects_non_sequence():
    with pytest.raises(ValueError):
        _validate_window_size(128)
    with pytest.raises(ValueError):
        _validate_window_size(torch.tensor([128, 0]))


# ---- _validate_qkv_varlen -------------------------------------------------


def test_validate_qkv_varlen_ok_mha():
    q = _make_thd(512, 8, 128)
    _validate_qkv_varlen(q, q.clone(), q.clone())  # no raise


def test_validate_qkv_varlen_ok_gqa():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 2, 128)
    _validate_qkv_varlen(q, k, k.clone())  # Hq=8, Hkv=2 -> ok


def test_validate_qkv_varlen_rejects_4d():
    q = torch.randn(1, 512, 8, 128, dtype=torch.float16)
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, q.clone(), q.clone())


def test_validate_qkv_varlen_rejects_fp32():
    q = _make_thd(512, 8, 128, dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        _validate_qkv_varlen(q, q.clone(), q.clone())


def test_validate_qkv_varlen_rejects_kv_total_mismatch():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 8, 128)
    v = _make_thd(256, 8, 128)  # total_v != total_k
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, k, v)


def test_validate_qkv_varlen_rejects_kv_head_mismatch():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 4, 128)
    v = _make_thd(512, 8, 128)  # Hv != Hk
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, k, v)


def test_validate_qkv_varlen_rejects_head_dim_qk_mismatch():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 8, 64)  # Dk != Dq
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, k, k.clone())


def test_validate_qkv_varlen_rejects_non_divisible_heads():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 3, 128)  # 8 % 3 != 0
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, k, k.clone())


# ---- _validate_cu_seqlens -------------------------------------------------


def test_validate_cu_seqlens_ok_causal():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    got = _validate_cu_seqlens(
        cu,
        cu,
        total_q=total,
        total_k=total,
        max_seqlen_q=max_s,
        max_seqlen_k=max_s,
        causal=True,
        device=torch.device("cpu"),
    )
    assert got == (256, 256)


def test_validate_cu_seqlens_ok_full_cross_lengths():
    # Non-causal cross attention: q and k may have different per-segment lengths
    # (same number of segments), which is allowed when causal=False.
    cu_q, max_q, total_q = _cu_from_seqlens([128, 256])
    cu_k, max_k, total_k = _cu_from_seqlens([300, 84])
    _validate_cu_seqlens(
        cu_q,
        cu_k,
        total_q=total_q,
        total_k=total_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        causal=False,
        device=torch.device("cpu"),
    )  # no raise


def test_validate_cu_seqlens_rejects_non_int32():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    cu_long = cu.to(torch.int64)
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu_long,
            cu_long,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_non_1d():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    cu2d = cu.view(1, -1)
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu2d,
            cu2d,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_too_short():
    cu = torch.zeros(1, dtype=torch.int32)  # numel < 2
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu,
            cu,
            total_q=0,
            total_k=0,
            max_seqlen_q=1,
            max_seqlen_k=1,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_nonzero_first():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    bad = cu.clone()
    bad[0] = 5  # first must be 0
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            bad,
            bad,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_non_monotone():
    bad = torch.tensor([0, 256, 128, 512], dtype=torch.int32)  # decreasing in the middle
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            bad,
            bad,
            total_q=512,
            total_k=512,
            max_seqlen_q=256,
            max_seqlen_k=256,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_last_ne_total():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu,
            cu,
            total_q=total + 1,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_length_mismatch():
    cu_q, max_q, total = _cu_from_seqlens([128, 128, 256])  # len 4
    cu_k, max_k, _ = _cu_from_seqlens([256, 256])  # len 3, same total 512
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu_q,
            cu_k,
            total_q=total,
            total_k=512,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=False,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_max_seqlen_too_small():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu,
            cu,
            total_q=total,
            total_k=total,
            max_seqlen_q=100,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_causal_len_mismatch():
    cu_q, max_q, total = _cu_from_seqlens([128, 128, 256])
    cu_k, max_k, _ = _cu_from_seqlens([256, 128, 128])  # same total/segments, different split
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu_q,
            cu_k,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_device_mismatch():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])  # on cpu
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu,
            cu,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("meta"),  # cpu != meta
        )
