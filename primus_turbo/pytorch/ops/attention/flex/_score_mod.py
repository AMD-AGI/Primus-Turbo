###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Recognise the ``score_mod`` forms the Turbo kernels can represent.

Only two ``score_mod`` shapes map onto a fixed kernel parameter: ALiBi (an additive
per-head linear bias in ``kv_idx - q_idx``) and logits soft-cap
(``cap * tanh(score / cap)``). Detection is *exact-representability* checking, not
curve fitting -- a candidate is accepted only after it is verified to reproduce the
closed form, so a ``score_mod`` is never silently approximated.

This module also carries the ALiBi sign-convention self-check, which pins down
empirically whether the backend's ``alibi_slopes`` follows flex's
``+slope * (kv - q)`` convention on the current build.
"""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ._cache import _ALIBI_CACHE, _CACHE_MISS, _SOFTCAP_CACHE, _cache_get, _cache_put
from ._config import _ALIBI_TOL, _ASSUMED_ALIBI_SIGN, _SOFTCAP_TOL
from ._probe import _call_score_mod


def _reference_attention_with_alibi(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    slopes: torch.Tensor,
    sign: float,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Dense fp32 reference for ``softmax(QK^T/sqrt(d) + sign*slope[h]*(kv-q)) V``."""
    q = q_bhsd.float()
    k = k_bhsd.float()
    v = v_bhsd.float()
    s_q, s_k = q.shape[-2], k.shape[-2]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    q_idx = torch.arange(s_q, device=q.device).view(s_q, 1)
    kv_idx = torch.arange(s_k, device=q.device).view(1, s_k)
    scores = scores + sign * slopes.float().view(1, -1, 1, 1) * (kv_idx - q_idx).float()
    if causal:
        scores = scores.masked_fill(kv_idx > q_idx, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def check_alibi_sign_convention(
    *,
    device: Any = None,
    dtype: torch.dtype = torch.float16,
    seqlen: int = 256,
    num_heads: int = 4,
    head_dim: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    """Empirically determine this build's ALiBi sign convention.

    The compat layer maps a flex ``score_mod`` of the form ``score + slope*(kv-q)``
    onto ``flash_attn_func(alibi_slopes=slope)``. Whether the kernel adds
    ``+slope*(kv-q)`` or ``-slope*(kv-q)`` is a property of the *installed aiter build*,
    not of this file -- get it wrong and ALiBi results are silently incorrect. This
    helper runs one small attention through the real backend and scores it against both
    fp32 references, so a new container/build can be validated in seconds instead of
    trusting a comment.

    Returns a dict with ``sign`` (``+1.0``/``-1.0``, or ``None`` if neither reference
    matches), ``plus_err`` / ``minus_err`` (relative L2), ``matches_assumption`` and the
    ``assumed_sign`` this layer is coded against. Requires a GPU (it calls the real
    kernel); it raises whatever the backend raises if one is unavailable.
    """
    from primus_turbo.pytorch.ops.attention.flash_attn_interface import flash_attn_func

    if device is None:
        device = "cuda"
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, seqlen, num_heads, head_dim)  # bshd for the Turbo backend
    q = torch.randn(shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    k = torch.randn(shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    v = torch.randn(shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    # Distinct, clearly separated slopes so the two sign hypotheses cannot alias.
    slopes = torch.tensor([2.0 ** -(i + 1) for i in range(num_heads)], dtype=torch.float32, device=device)

    out = flash_attn_func(q, k, v, causal=True, alibi_slopes=slopes)
    if isinstance(out, tuple):
        out = out[0]
    got = out.transpose(1, 2).float()  # -> bhsd for comparison

    q_b, k_b, v_b = (t.transpose(1, 2) for t in (q, k, v))

    def _rel_l2(sign: float) -> float:
        ref = _reference_attention_with_alibi(q_b, k_b, v_b, slopes, sign)
        return float((got - ref).norm() / ref.norm().clamp_min(1e-12))

    plus_err = _rel_l2(1.0)
    minus_err = _rel_l2(-1.0)
    # A match must be both small in absolute terms and decisively better than the other
    # hypothesis; otherwise report "unknown" rather than guessing.
    best_err, best_sign = min((plus_err, 1.0), (minus_err, -1.0))
    other_err = max(plus_err, minus_err)
    sign = best_sign if (best_err < 2e-2 and other_err > 10.0 * best_err) else None
    return {
        "sign": sign,
        "plus_err": plus_err,
        "minus_err": minus_err,
        "assumed_sign": _ASSUMED_ALIBI_SIGN,
        "matches_assumption": sign == _ASSUMED_ALIBI_SIGN,
        "dtype": str(dtype),
        "shape": {"seqlen": seqlen, "num_heads": num_heads, "head_dim": head_dim},
    }


def assert_alibi_sign_convention(**kwargs: Any) -> Dict[str, Any]:
    """Run :func:`check_alibi_sign_convention` and raise unless it matches the assumption.

    Intended as a one-line build gate (CI job, container smoke test, or the first thing
    a new environment runs) so a flipped-sign build fails loudly instead of quietly
    producing wrong ALiBi outputs.
    """
    report = check_alibi_sign_convention(**kwargs)
    if not report["matches_assumption"]:
        raise RuntimeError(
            "Turbo flex compat layer: the ALiBi sign convention of this build does not match the "
            f"assumed +slope*(kv-q) (assumed_sign={report['assumed_sign']}, measured "
            f"sign={report['sign']}, plus_err={report['plus_err']:.3g}, "
            f"minus_err={report['minus_err']:.3g}). ALiBi results would be silently wrong on this "
            "build."
        )
    return report


def _alibi_sample_points(q_len: int, kv_len: int) -> Tuple[Tuple[int, int], ...]:
    pts = set()
    for d in (0, 1, 2, 3, 5, 7):
        if d < kv_len:
            pts.add((0, d))  # delta = +d
        if d < q_len:
            pts.add((d, 0))  # delta = -d
    if q_len >= 2 and kv_len >= 2:
        # Far-corner points reuse small deltas at large absolute positions and so
        # catch score_mods that depend on absolute position, not just (kv - q).
        pts.add((q_len - 1, kv_len - 1))
        pts.add((q_len - 1, kv_len - 2))
        pts.add((q_len - 2, kv_len - 1))
    mq, mk = q_len // 2, kv_len // 2
    pts.add((mq, mk))
    if mk + 1 < kv_len:
        pts.add((mq, mk + 1))
    if mq + 1 < q_len:
        pts.add((mq + 1, mk))
    return tuple(sorted(pts))


def _detect_alibi_slopes(
    score_mod: Callable,
    *,
    B: int,
    Hq: int,
    q_len: int,
    kv_len: int,
    tol: float = _ALIBI_TOL,
) -> Optional[torch.Tensor]:
    """Return per-head ALiBi slopes iff ``score_mod`` is exactly representable as
    ``score + slope[h] * (kv_idx - q_idx)`` (additive, translation invariant,
    batch independent); otherwise ``None`` (caller routes to the custom path).
    """
    if q_len <= 0 or kv_len <= 0:
        return None

    pts = _alibi_sample_points(q_len, kv_len)
    nonzero = [(q, k) for (q, k) in pts if (k - q) != 0]

    slopes = []
    for h in range(Hq):
        # (1) additive in score with unit coefficient: score_mod(s) - s independent of s.
        c0 = _call_score_mod(score_mod, 0.0, 0, h, 0, 0)
        c1 = _call_score_mod(score_mod, 1.0, 0, h, 0, 0)
        if abs(c1 - c0 - 1.0) > tol:
            return None
        # bias at delta 0 must be ~0 for a pure (kv-q) bias.
        if abs(c0) > tol:
            return None

        if not nonzero:
            slopes.append(0.0)
            continue

        ref_q, ref_k = min(nonzero, key=lambda qk: abs(qk[1] - qk[0]))
        ref_delta = float(ref_k - ref_q)
        slope = _call_score_mod(score_mod, 0.0, 0, h, ref_q, ref_k) / ref_delta

        # (2) exact linear-in-(kv-q) fit across the sample grid.
        for q_i, k_i in pts:
            got = _call_score_mod(score_mod, 0.0, 0, h, q_i, k_i)
            expect = slope * float(k_i - q_i)
            if abs(got - expect) > tol * (1.0 + abs(expect)):
                return None

        # (3) batch independence (ALiBi slopes are per-head only).
        if B > 1:
            alt = _call_score_mod(score_mod, 0.0, B - 1, h, ref_q, ref_k)
            if abs(alt - _call_score_mod(score_mod, 0.0, 0, h, ref_q, ref_k)) > tol:
                return None

        slopes.append(slope)

    return torch.tensor(slopes, dtype=torch.float32)


def _detect_softcap(
    score_mod: Callable,
    *,
    B: int,
    Hq: int,
    q_len: int,
    kv_len: int,
    tol: float = _SOFTCAP_TOL,
) -> Optional[float]:
    """Return ``cap > 0`` iff ``score_mod`` is exactly a logits soft-cap
    ``cap * tanh(score / cap)`` (a function of the score alone, independent of
    batch/head/positions); otherwise ``None``.

    Detected explicitly for two reasons on this build:

    * The fixed Turbo dense kernels expose **no** softcap parameter (aiter dense
      ``mha_fwd``/``fmha_v3_fwd``/``mha_bwd`` lack it),
      so a soft-cap cannot be silently dropped.
    * :func:`_detect_alibi_slopes` only ever probes ``score=0``, where
      ``cap*tanh(0)=0``; a soft-cap would therefore be misread as a zero-slope
      (no-op) bias and handed to Turbo *without* the cap -- a silently wrong
      result. Recognising the soft-cap shape first prevents that.
    """
    if q_len <= 0 or kv_len <= 0:
        return None

    # (1) A pure soft-cap depends on the score only: constant across batch / head
    #     / query / key position. Any dependence rules it out (e.g. ALiBi, which
    #     varies with kv-q, falls through here to the ALiBi detector).
    ctx_points = [(0, 0, 0, 0)]
    if q_len > 1:
        ctx_points.append((0, 0, q_len - 1, 0))
    if kv_len > 1:
        ctx_points.append((0, 0, 0, kv_len - 1))
    if q_len > 1 and kv_len > 1:
        ctx_points.append((0, 0, q_len - 1, kv_len - 1))
    if Hq > 1:
        ctx_points.append((0, Hq - 1, 0, 0))
    if B > 1:
        ctx_points.append((B - 1, 0, 0, 0))
    for s in (0.5, 1.7, -1.3):
        base = _call_score_mod(score_mod, s, 0, 0, 0, 0)
        for b, h, qi, ki in ctx_points[1:]:
            if abs(_call_score_mod(score_mod, s, b, h, qi, ki) - base) > tol * (1.0 + abs(base)):
                return None

    def f(s: float) -> float:
        return _call_score_mod(score_mod, s, 0, 0, 0, 0)

    # (2) tanh passes through the origin.
    if abs(f(0.0)) > tol:
        return None

    # (3) Estimate the positive asymptote (the cap) from the saturating tail.
    cap = None
    prev = None
    for s in (4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0):
        y = f(s)
        if y <= 0.0:
            return None  # a soft-cap is strictly increasing / positive for s>0
        if prev is not None and abs(y - prev) <= tol * (1.0 + abs(y)):
            cap = y
            break
        prev = y
    if cap is None or cap <= tol:
        return None  # never saturated (identity/linear) or degenerate cap

    # Refine the asymptote where tanh is fully saturated, so the returned cap is
    # accurate rather than the first plateau sample.
    refined = f(64.0 * cap)
    if refined > 0.0:
        cap = refined

    # (4) Full-shape verification against cap*tanh(s/cap): near-linear region,
    #     knee, saturating tail, and odd symmetry f(-s) == -f(s).
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        for s in (frac * cap, -frac * cap):
            got = f(s)
            expect = cap * math.tanh(s / cap)
            if abs(got - expect) > tol * (1.0 + abs(expect)):
                return None

    return float(cap)


def _cached_detect_alibi_slopes(
    score_mod: Callable, *, B: int, Hq: int, q_len: int, kv_len: int
) -> Optional[torch.Tensor]:
    """Memoised :func:`_detect_alibi_slopes` keyed by ``score_mod`` object identity.

    Returns exactly what the uncached detector would (including ``None``); the cached
    slopes tensor is never mutated in place by callers (they ``.to(device)`` / read
    it), so sharing it is safe.
    """
    key = (B, Hq, q_len, kv_len)
    cached = _cache_get(_ALIBI_CACHE, score_mod, key)
    if cached is not _CACHE_MISS:
        return cached
    val = _detect_alibi_slopes(score_mod, B=B, Hq=Hq, q_len=q_len, kv_len=kv_len)
    _cache_put(_ALIBI_CACHE, score_mod, key, val)
    return val


def _cached_detect_softcap(
    score_mod: Callable, *, B: int, Hq: int, q_len: int, kv_len: int
) -> Optional[float]:
    """Memoised :func:`_detect_softcap` keyed by ``score_mod`` object identity."""
    key = (B, Hq, q_len, kv_len)
    cached = _cache_get(_SOFTCAP_CACHE, score_mod, key)
    if cached is not _CACHE_MISS:
        return cached
    val = _detect_softcap(score_mod, B=B, Hq=Hq, q_len=q_len, kv_len=kv_len)
    _cache_put(_SOFTCAP_CACHE, score_mod, key, val)
    return val


def _is_identity_score_mod(
    score_mod: Callable,
    *,
    B: int,
    Hq: int,
    q_len: int,
    kv_len: int,
    tol: float = _ALIBI_TOL,
) -> bool:
    """Return ``True`` iff ``score_mod`` leaves the score unchanged on a probe grid.

    Used only for the explicit-``alibi_slopes`` conflict check: a no-op / identity
    ``score_mod`` may coexist with explicit slopes, whereas *any* non-trivial
    ``score_mod`` alongside explicit slopes is ambiguous and rejected. The probe
    spans several scores and (batch, head, q, kv) positions -- including non-zero
    ``kv-q`` deltas -- so e.g. a (non-zero) ALiBi ``score_mod`` reads as non-identity.
    """
    b_pts = sorted({0, B - 1}) if B > 0 else [0]
    h_pts = sorted({0, Hq - 1}) if Hq > 0 else [0]
    q_pts = sorted({0, q_len // 2, q_len - 1}) if q_len > 0 else [0]
    kv_pts = sorted({0, kv_len // 2, kv_len - 1}) if kv_len > 0 else [0]
    for s in (0.0, 0.7, -1.3, 2.5):
        for b in b_pts:
            for h in h_pts:
                for qi in q_pts:
                    for ki in kv_pts:
                        got = _call_score_mod(score_mod, s, b, h, qi, ki)
                        if abs(got - s) > tol * (1.0 + abs(s)):
                            return False
    return True
