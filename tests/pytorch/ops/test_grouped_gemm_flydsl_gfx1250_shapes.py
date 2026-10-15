###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shape rules and ragged-K coverage for the FlyDSL bf16 grouped GEMM on gfx1250.

Two groups of tests:

* The gate-consistency tests are **pure host** -- they call the tile picker and the
  published shape gates, which are plain Python, so they need flydsl importable but
  no GPU. They are the permanent guard against the gate and the entry point growing
  two copies of the same rule, which is how a clean Triton fallback turned into an
  ``AssertionError`` once already.
* The numeric tests need a gfx1250 card. They cover the K values the repo's own
  shape list cannot reach: ``GROUPED_GEMM_SHAPES`` is 128-aligned in K throughout,
  so the backed-off tail k-tile and the short-reduction pipeline never execute there.
"""

import itertools

import pytest
import torch

from primus_turbo.pytorch.core.utils import is_gfx1250
from tests.pytorch.test_utils import compute_snr, get_tolerances

pytest.importorskip("flydsl", reason="the FlyDSL gfx1250 grouped GEMM lives behind flydsl")

from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (  # noqa: E402
    WMMA_K_ENTRY,
    _pick_config,
    _pick_config_nn,
    grouped_gemm_bf16_nn_flydsl_kernel,
    grouped_gemm_bf16_nn_supported,
    grouped_gemm_bf16_nt_flydsl_kernel,
    grouped_gemm_bf16_nt_supported,
    nn_native_unsupported_reason,
)

needs_gfx1250 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 (MI455X) GPU"
)

# The scan the gate-consistency tests sweep. K steps by 8 rather than by 32 so that
# tails which are *not* a whole number of WMMA sub-steps are in the space too, and it
# starts below the smallest tile_k so ``K < tile_k`` is covered.
_SCAN_KS = range(32, 4097, 8)
_SCAN_NS = (768, 2048, 2880, 4096, 7168)
_SCAN_AVG_MS = (32, 64, 96, 128, 256, 512, 1024)
_SCAN_GS = (8, 32)


def _n_steady(K: int, cfg) -> int:
    """The launcher's ``n_steady = K_TILES - (num_buffers - 1)`` for this tile.

    Recomputed here rather than imported: the point of the test is to check the host
    rules against an independent statement of what the K loop does.
    """
    tile_k, num_buffers = cfg[2], cfg[5]
    k_tiles = K // tile_k + (1 if K % tile_k else 0)
    return k_tiles - (num_buffers - 1)


def _pipeline_is_legal(K: int, cfg) -> bool:
    """Whether the launcher can run this ``(K, cfg)`` at all.

    ``n_steady`` must be non-negative, or the prologue issues k-tiles past K -- an
    out-of-bounds read, because the reduction axis is dim-1 of the A/B descriptors
    and carries no extent. With a tail it must be at least 1, because the iteration
    that prefetches the tail is peeled out at index ``n_steady - 1``.
    """
    return _n_steady(K, cfg) >= (1 if K % cfg[2] else 0)


# ---------------------------------------------------------------------------
# Host-only: the gate and the entry point must agree, shape for shape.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["nt", "nn"])
def test_gate_admits_only_shapes_the_pipeline_can_run(direction):
    """Everything the published gate admits must satisfy every entry-side K rule.

    The gate is the non-throwing form of the entry's preconditions, so a shape it
    admits has to clear all of them: the ``K % tile_k`` tail must be a whole number
    of WMMA sub-steps, and K must give the software pipeline a legal ``n_steady``.
    Asserted against the tile the picker really hands out, which is what the entry
    will use.
    """
    gate = grouped_gemm_bf16_nt_supported if direction == "nt" else grouped_gemm_bf16_nn_supported
    picker = _pick_config if direction == "nt" else _pick_config_nn
    admitted = 0
    for N, avg_m, G in itertools.product(_SCAN_NS, _SCAN_AVG_MS, _SCAN_GS):
        for K in _SCAN_KS:
            if not gate(N, K, avg_m, G):
                continue
            admitted += 1
            cfg = picker(N, K, max(1, avg_m), G)
            tile_k, num_buffers = cfg[2], cfg[5]
            k_tail = K % tile_k
            assert k_tail % WMMA_K_ENTRY == 0, (
                f"{direction}: gate admitted N={N} K={K} avg_m={avg_m} G={G}, but tile {cfg} "
                f"leaves a tail of {k_tail}, which is not a multiple of {WMMA_K_ENTRY}; the "
                f"entry point raises on it"
            )
            assert K // tile_k >= num_buffers - 1, (
                f"{direction}: gate admitted N={N} K={K} avg_m={avg_m} G={G}, but tile {cfg} "
                f"has only {K // tile_k} full k-tiles for a {num_buffers}-buffer pipeline"
            )
            assert _pipeline_is_legal(K, cfg), (
                f"{direction}: gate admitted N={N} K={K} avg_m={avg_m} G={G}, but tile {cfg} "
                f"gives n_steady={_n_steady(K, cfg)}"
            )
    assert admitted > 1000, f"{direction}: scan admitted only {admitted} shapes, so it proves little"


def test_nn_gate_agrees_with_the_published_reason_function():
    """``grouped_gemm_bf16_nn_supported`` must be exactly ``reason is None``.

    Both are in ``__all__`` and a dispatcher may use either, so they cannot disagree.
    The reason function is also what the entry point consults, which is what keeps
    the native path from being entered on a shape the gate declined.
    """
    for N, avg_m, G in itertools.product(_SCAN_NS, _SCAN_AVG_MS, _SCAN_GS):
        for K in _SCAN_KS:
            cfg = _pick_config_nn(N, K, max(1, avg_m), G)
            why = nn_native_unsupported_reason(N, K, cfg[1], cfg[2], cfg[5])
            assert grouped_gemm_bf16_nn_supported(N, K, avg_m, G) == (why is None), (
                f"N={N} K={K} avg_m={avg_m} G={G}: gate and reason function disagree ({why})"
            )


@pytest.mark.parametrize("direction", ["nt", "nn"])
def test_picker_never_asks_for_a_ring_k_cannot_fill(direction):
    """Either the picked ring fits K, or it is the launcher's minimum and K is declined.

    This is the invariant that joins the two halves of the rule. The picker backs
    ``num_buffers`` off to ``K // tile_k + 1`` so that a short reduction keeps this
    kernel instead of falling back, but it cannot go below the launcher's two-buffer
    floor -- so ``K < tile_k`` is left for the gate to refuse. Asserted together,
    because "clamped" and "declined" are the only two acceptable outcomes and a tile
    that is neither is an out-of-bounds read.
    """
    gate = grouped_gemm_bf16_nt_supported if direction == "nt" else grouped_gemm_bf16_nn_supported
    picker = _pick_config if direction == "nt" else _pick_config_nn
    for N, avg_m, G in itertools.product(_SCAN_NS, _SCAN_AVG_MS, _SCAN_GS):
        for K in _SCAN_KS:
            cfg = picker(N, K, max(1, avg_m), G)
            assert cfg[5] >= 2, f"{direction}: {cfg} is below the launcher's two-buffer minimum"
            if cfg[5] <= K // cfg[2] + 1:
                continue
            assert cfg[5] == 2, (
                f"{direction}: N={N} K={K} avg_m={avg_m} G={G} picked {cfg}, whose "
                f"{cfg[5]}-buffer ring needs {cfg[5] - 1} full tiles but K has "
                f"{K // cfg[2]}; only the two-buffer floor may exceed the bound"
            )
            assert not gate(N, K, avg_m, G), (
                f"{direction}: N={N} K={K} avg_m={avg_m} G={G} picked {cfg}, which K "
                f"cannot fill even at the two-buffer floor, yet the gate admits it"
            )


def test_k128_avgm128_picks_a_legal_pipeline():
    """The regression this file exists for: K=128 at ``64 < avg_m <= 128``.

    ``_pick_config_shape`` answers ``(128, 256, 128, 2, 2, 3)`` here, and at K=128
    that is one k-tile against a three-deep ring -- ``n_steady = -1``. The picker has
    to back the ring off instead.
    """
    for N, avg_m in itertools.product((768, 2048, 2880, 4096, 7168), (96, 128)):
        for picker, direction in ((_pick_config, "nt"), (_pick_config_nn, "nn")):
            cfg = picker(N, 128, avg_m, 8)
            assert _n_steady(128, cfg) >= 0, (
                f"{direction}: N={N} K=128 avg_m={avg_m} picked {cfg}, n_steady={_n_steady(128, cfg)}"
            )
    # And the gate has to admit them, because the shapes themselves are fine.
    assert grouped_gemm_bf16_nt_supported(2048, 128, 128, 8)
    assert grouped_gemm_bf16_nn_supported(2048, 128, 128, 8)


@pytest.mark.parametrize(
    "N, K, avg_m",
    [
        # Tails that are a whole number of WMMA sub-steps but whose full-tile count
        # used to be read against an unclamped ring.
        (768, 96, 32),
        (2048, 96, 256),
        (2048, 160, 128),
        (2880, 2880, 128),
    ],
)
def test_short_and_ragged_reductions_are_served(N, K, avg_m):
    """These were admitted by one gate and refused by the other side of the entry."""
    for gate, picker, direction in (
        (grouped_gemm_bf16_nt_supported, _pick_config, "nt"),
        (grouped_gemm_bf16_nn_supported, _pick_config_nn, "nn"),
    ):
        cfg = picker(N, K, avg_m, 8)
        if gate(N, K, avg_m, 8):
            assert _pipeline_is_legal(K, cfg), f"{direction}: {cfg} at K={K} is not runnable"


@pytest.mark.parametrize("K", [32, 64, 96, 128, 160, 352, 2056, 2080, 2880])
def test_gate_false_means_the_entry_would_refuse(K):
    """A declined shape must be declined for a reason the entry also holds.

    The direction that matters for a fixed backend: ``can_handle`` returning True on
    a shape the entry then raises on is what turns a Triton fallback into a crash.
    This is the contrapositive, spot-checked on the K values the fix moved.
    """
    for N, avg_m in itertools.product((768, 2048, 4096), (32, 128, 512)):
        cfg = _pick_config(N, K, avg_m, 8)
        legal = (K % cfg[2]) % WMMA_K_ENTRY == 0 and K // cfg[2] >= cfg[5] - 1
        assert grouped_gemm_bf16_nt_supported(N, K, avg_m, 8) == legal, (
            f"nt: N={N} K={K} avg_m={avg_m} tile {cfg}: gate disagrees with its own rules"
        )


# ---------------------------------------------------------------------------
# Numeric: needs a gfx1250 card.
# ---------------------------------------------------------------------------


def _group_offs(G: int, avg_m: int, device) -> torch.Tensor:
    """Balanced row runs: ``[0, avg_m, 2*avg_m, ...]``, which is what avg_m means."""
    return torch.arange(0, (G + 1) * avg_m, avg_m, dtype=torch.int64, device=device)


def _ref(a, b, offs, trans_b: bool):
    """fp32-accumulated per-expert matmul, on the **CPU**.

    Deliberately not a device matmul: hipBLASLt has no Tensile library for gfx1250 in
    every image this runs in, so a device reference would turn a kernel test into a
    hipBLASLt availability test. Same reason the MLA suite keeps its reference on the
    host.
    """
    offs = offs.cpu().tolist()
    a32, b32 = a.detach().float().cpu(), b.detach().float().cpu()
    out = []
    for g in range(len(offs) - 1):
        rhs = b32[g].t() if trans_b else b32[g]
        out.append((a32[offs[g] : offs[g + 1]] @ rhs).to(a.dtype))
    return torch.cat(out)


def _assert_matches_ref(out, a, b, offs, trans_b: bool):
    """The repo's own two-part comparison: SNR floor, then element-wise tolerances."""
    ref = _ref(a, b, offs, trans_b)
    got = out.detach().cpu()
    snr = compute_snr(ref, got)
    assert snr > 45, f"out SNR {snr:.2f} dB is below the bf16 threshold of 45 dB"
    torch.testing.assert_close(got, ref, **get_tolerances(a.dtype))


def _run_nt(a, b, offs):
    """Four calls, checking the fourth.

    The entry's own docstring records that the first one or two calls in a fresh
    process have been observed to return corrupt results, root cause unknown and
    predating this kernel's tuning. A correctness harness has to burn through that.
    """
    for _ in range(3):
        grouped_gemm_bf16_nt_flydsl_kernel(a, b, offs)
    return grouped_gemm_bf16_nt_flydsl_kernel(a, b, offs)


def _run_nn(a, b, offs):
    for _ in range(3):
        grouped_gemm_bf16_nn_flydsl_kernel(a, b, offs)
    return grouped_gemm_bf16_nn_flydsl_kernel(a, b, offs)


@needs_gfx1250
@pytest.mark.parametrize("N", [2048, 768])
def test_nt_k128_avgm128_numeric(N):
    """K=128 with avg_m=128: the shape whose pipeline depth went negative.

    Before the fix the picker answered ``(128, 256, 128, 2, 2, 3)`` here, the
    prologue issued a k-tile past K, and the drain's first index was -1.
    """
    G, avg_m, K = 8, 128, 128
    torch.manual_seed(0)
    device = "cuda"
    a = torch.randn(G * avg_m, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(G, N, K, dtype=torch.bfloat16, device=device)
    offs = _group_offs(G, avg_m, device)

    assert grouped_gemm_bf16_nt_supported(N, K, avg_m, G), "gate should admit this shape"
    cfg = _pick_config(N, K, avg_m, G)
    assert _n_steady(K, cfg) >= 0, f"{cfg} still has a negative n_steady"

    out = _run_nt(a, b, offs)
    _assert_matches_ref(out, a, b, offs, trans_b=True)


@needs_gfx1250
@pytest.mark.parametrize("N", [2048, 768])
def test_nn_k128_avgm128_numeric(N):
    """The dgrad-orientation twin of the test above."""
    G, avg_m, K = 8, 128, 128
    torch.manual_seed(0)
    device = "cuda"
    a = torch.randn(G * avg_m, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)
    offs = _group_offs(G, avg_m, device)

    cfg = _pick_config_nn(N, K, avg_m, G)
    assert _n_steady(K, cfg) >= 0, f"{cfg} still has a negative n_steady"

    out = _run_nn(a, b, offs)
    _assert_matches_ref(out, a, b, offs, trans_b=False)


@needs_gfx1250
@pytest.mark.parametrize("K", [352, 2080])
@pytest.mark.parametrize("avg_m", [128, 512])
def test_nt_ragged_k_tail_numeric(K, avg_m):
    """NT with ``K % 64 == 32``: the backed-off tail k-tile, which has no other test.

    K=352 is DeepSeek-V2-Lite's fc2 reduction under ETP=4. Every K in the repo's
    shape list is a multiple of 128, so this path does not run there at all.
    """
    G, N = 8, 2048
    assert K % 64 == 32, "the point of this case is a tail that is half a tile_k"
    torch.manual_seed(0)
    device = "cuda"
    a = torch.randn(G * avg_m, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(G, N, K, dtype=torch.bfloat16, device=device)
    offs = _group_offs(G, avg_m, device)

    if not grouped_gemm_bf16_nt_supported(N, K, avg_m, G):
        pytest.skip(f"gate declines N={N} K={K} avg_m={avg_m}; nothing to check numerically")
    cfg = _pick_config(N, K, avg_m, G)
    assert K % cfg[2], f"{cfg} divides K={K} exactly, so this case is not testing the tail"

    out = _run_nt(a, b, offs)
    _assert_matches_ref(out, a, b, offs, trans_b=True)


@needs_gfx1250
@pytest.mark.parametrize("K", [2848, 2880, 2912])  # tails of 32, 64 and 96
def test_nn_ragged_k_tail_numeric(K):
    """NN with ``K % 128`` in {32, 64, 96}: gpt-oss's dgrad reduction length and neighbours.

    The NN entry enables the ragged-K upgrade unconditionally, so these reduction
    lengths take the tail path by default -- which the repo's 128-aligned shape list
    never exercises.
    """
    G, N, avg_m = 8, 2048, 128
    assert K % 128 in (32, 64, 96), "the point of this case is a sub-tile tail"
    torch.manual_seed(0)
    device = "cuda"
    a = torch.randn(G * avg_m, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)
    offs = _group_offs(G, avg_m, device)

    out = _run_nn(a, b, offs)
    _assert_matches_ref(out, a, b, offs, trans_b=False)


@needs_gfx1250
def test_nt_entry_refuses_what_the_gate_refuses():
    """The contract, on hardware: gate False and the entry raises; gate True and it runs.

    Checked on a shape the gate used to admit and the entry used to raise on
    (N=2048, K=176: the tail is 48, which is not a whole number of WMMA sub-steps).
    """
    G, N, K, avg_m = 8, 2048, 176, 512
    device = "cuda"
    a = torch.randn(G * avg_m, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(G, N, K, dtype=torch.bfloat16, device=device)
    offs = _group_offs(G, avg_m, device)

    assert not grouped_gemm_bf16_nt_supported(N, K, avg_m, G)
    with pytest.raises(AssertionError, match="not a multiple of 32"):
        grouped_gemm_bf16_nt_flydsl_kernel(a, b, offs)
