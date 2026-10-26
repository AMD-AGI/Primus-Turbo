###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shape rules and ragged-K coverage for the FlyDSL bf16 grouped GEMM on gfx1250.

The gate-consistency tests are host-only and keep the published gates and the entry
point's own preconditions from drifting into two different rules. The numeric tests
need a gfx1250 card and cover the K tails that ``GROUPED_GEMM_SHAPES``, 128-aligned
in K throughout, never reaches.
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
    not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 GPU"
)

# K steps by 8, not by 32, so tails that are not a whole number of WMMA sub-steps are
# in the space, and starts below the smallest tile_k so ``K < tile_k`` is covered.
_SCAN_KS = range(32, 4097, 8)
_SCAN_NS = (768, 2048, 2880, 4096, 7168)
# 2048 covers the exact-key picks, which are all conditioned on it.
_SCAN_AVG_MS = (32, 64, 96, 128, 256, 512, 1024, 2048)
_SCAN_GS = (8, 32)

# NT keys are the forward weight's (N, K); NN keys are (N_out, K_red).
_KEYED_NT = ((4096, 7168), (7168, 2048))
_KEYED_NN = ((7168, 4096), (2048, 7168))
_KEYED_AVG_M = 2048


def _n_steady(K: int, cfg) -> int:
    """The launcher's ``n_steady = K_TILES - (num_buffers - 1)``, restated independently."""
    tile_k, num_buffers = cfg[2], cfg[5]
    k_tiles = K // tile_k + (1 if K % tile_k else 0)
    return k_tiles - (num_buffers - 1)


def _pipeline_is_legal(K: int, cfg) -> bool:
    """Whether the launcher can run this ``(K, cfg)``: a negative ``n_steady`` issues
    k-tiles past K, which the extent-less reduction axis cannot stop.
    """
    return _n_steady(K, cfg) >= (1 if K % cfg[2] else 0)


# --- Host-only: the gate and the entry point must agree, shape for shape. ---


@pytest.mark.parametrize("direction", ["nt", "nn"])
def test_gate_admits_only_shapes_the_pipeline_can_run(direction):
    """Everything the gate admits must satisfy every entry-side K rule, asserted
    against the tile the picker really hands out.
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
    """``grouped_gemm_bf16_nn_supported`` must be exactly ``reason is None``: both are
    in ``__all__`` and a dispatcher may use either.
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
    """Either the picked ring fits K, or it is the two-buffer floor and the gate declines
    K. A tile that is neither clamped nor declined is an out-of-bounds read.
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
    """K=128 at ``64 < avg_m <= 128``: one k-tile against a three-deep ring, so the
    picker has to back the ring off or ``n_steady`` goes negative.
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


def test_exact_key_picks_stay_inside_the_published_gate():
    """A pin returns before the general rules, so every pinned tile must still pass the gate."""
    for N, K in _KEYED_NT:
        for G in _SCAN_GS:
            cfg = _pick_config(N, K, _KEYED_AVG_M, G)
            assert grouped_gemm_bf16_nt_supported(N, K, _KEYED_AVG_M, G), f"NT N={N} K={K} G={G}: {cfg}"
            assert _n_steady(K, cfg) >= 0, f"NT N={N} K={K} G={G} picked {cfg}"
            assert K % cfg[2] % WMMA_K_ENTRY == 0, f"NT N={N} K={K}: tail of {K % cfg[2]} is not WMMA-aligned"
    for N, K in _KEYED_NN:
        for G in _SCAN_GS:
            cfg = _pick_config_nn(N, K, _KEYED_AVG_M, G)
            why = nn_native_unsupported_reason(N, K, cfg[1], cfg[2], cfg[5])
            assert why is None, f"NN N={N} K={K} G={G} picked {cfg}, which the gate refuses: {why}"
            assert grouped_gemm_bf16_nn_supported(N, K, _KEYED_AVG_M, G)
            # The 4-wave TDM dealing halves tile_k; each half must stay a whole number of WMMA k-steps.
            assert cfg[2] % (2 * WMMA_K_ENTRY) == 0, f"NN N={N} K={K}: tile_k={cfg[2]} is not split-safe"


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
    """A declined shape must be declined for a reason the entry also holds: a gate that
    admits what the entry raises on turns a fallback into a crash.
    """
    for N, avg_m in itertools.product((768, 2048, 4096), (32, 128, 512)):
        cfg = _pick_config(N, K, avg_m, 8)
        legal = (K % cfg[2]) % WMMA_K_ENTRY == 0 and K // cfg[2] >= cfg[5] - 1
        assert grouped_gemm_bf16_nt_supported(N, K, avg_m, 8) == legal, (
            f"nt: N={N} K={K} avg_m={avg_m} tile {cfg}: gate disagrees with its own rules"
        )


# --- Numeric: needs a gfx1250 card. ---


def _group_offs(G: int, avg_m: int, device) -> torch.Tensor:
    """Balanced row runs: ``[0, avg_m, 2*avg_m, ...]``, which is what avg_m means."""
    return torch.arange(0, (G + 1) * avg_m, avg_m, dtype=torch.int64, device=device)


def _ref(a, b, offs, trans_b: bool):
    """fp32-accumulated per-expert matmul, on the host: hipBLASLt has no gfx1250 Tensile
    library in every image this runs in.
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
    """Four calls, checking the fourth: the first calls in a fresh process are the ones
    the module docstring's known defect hits most often.
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
    """K=128 with avg_m=128: the shape whose pipeline depth went negative."""
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
    """NT with ``K % 64 == 32``: the backed-off tail k-tile. Every K in the repo's shape
    list is a multiple of 128, so this path never runs there.
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
    """NN with ``K % 128`` in {32, 64, 96}: the NN entry enables the ragged-K upgrade
    unconditionally, so these lengths take the tail path by default.
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
    """The contract, on hardware: gate False and the entry raises. N=2048, K=176 leaves
    a tail of 48, which is not a whole number of WMMA sub-steps.
    """
    G, N, K, avg_m = 8, 2048, 176, 512
    device = "cuda"
    a = torch.randn(G * avg_m, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(G, N, K, dtype=torch.bfloat16, device=device)
    offs = _group_offs(G, avg_m, device)

    assert not grouped_gemm_bf16_nt_supported(N, K, avg_m, G)
    with pytest.raises(AssertionError, match="not a multiple of 32"):
        grouped_gemm_bf16_nt_flydsl_kernel(a, b, offs)
