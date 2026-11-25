###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The Kimi K3.1 kwargs keys of the FlyDSL bf16 grouped GEMM on gfx1250: six operators at avg_m 293 / 585 / 1170.

The keys follow ``test_grouped_gemm_flydsl_gfx1250_kwargs_keys.py`` (the MiniMax-M3 ones) and add what is Kimi's
own: three row counts per shape served by the nearest-key rule, sets that use the knobs this kernel took from the
Kimi tuning (``mv_dec``, ``nt_half``, ``vk_seam`` bits 20-23), and an expert count (224 at EP4, 112 at EP8) above
the 32 the forward vec_dec keys are limited to.

* Host-only (needs flydsl, no GPU): which shapes are keyed, the row counts each key serves, the tile and the
  expert-count preconditions, the knobs the sets carry.
* Numeric, on a gfx1250 card: every key called the way the backend calls it, against an fp32 host reference and
  bit for bit against the same call with the key switched off.
"""

import inspect
import itertools
import math

import pytest
import torch

pytest.importorskip("flydsl", reason="the FlyDSL gfx1250 grouped GEMM lives behind flydsl")

from primus_turbo.flydsl.grouped_gemm import grouped_gemm_bf16_kernel_gfx1250 as gg  # noqa: E402
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (  # noqa: E402
    grouped_gemm_bf16_nn_flydsl_kernel,
    grouped_gemm_bf16_nt_flydsl_kernel,
    grouped_gemm_bf16_variable_k_flydsl_kernel,
)


def _have_gfx1250() -> bool:
    try:
        from primus_turbo.pytorch.core.utils import is_gfx1250

        return bool(torch.cuda.is_available() and is_gfx1250())
    except Exception:
        return False


needs_gfx1250 = pytest.mark.skipif(not _have_gfx1250(), reason="needs a gfx1250 (MI455X) GPU")

# Kimi K3.1's MoE layer: latent 3584, gated expert FFN 3072, so fc1 is (N, K) = (6144, 3584) and fc2 (3584, 3072).
# NT and variable-K keys take the forward (N, K); NN takes (output N, reduction K).
KEYED_AVG_M = (293, 585, 1170)
MEASURED_G = 224
SHAPES = (
    ("nt", "fwd_fc1", 6144, 3584),
    ("nt", "fwd_fc2", 3584, 3072),
    ("nn", "dgrad_fc1", 3584, 6144),
    ("nn", "dgrad_fc2", 3072, 3584),
    ("vk", "wgrad_fc1", 6144, 3584),
    ("vk", "wgrad_fc2", 3584, 3072),
)
_IDS = [name for _, name, *_ in SHAPES]
_ENTRY = {
    "nt": gg._grouped_gemm_bf16_nt_flydsl_kernel_entry,
    "nn": gg._grouped_gemm_bf16_nn_flydsl_kernel_entry,
    "vk": gg._grouped_gemm_bf16_variable_k_flydsl_kernel_entry,
}
_PUBLIC = {
    "nt": grouped_gemm_bf16_nt_flydsl_kernel,
    "nn": grouped_gemm_bf16_nn_flydsl_kernel,
    "vk": grouped_gemm_bf16_variable_k_flydsl_kernel,
}
_OPERANDS = {
    "nt": ("a", "b", "group_offs"),
    "nn": ("a", "b", "group_offs"),
    "vk": ("a", "b", "group_k_offsets"),
}
PASSTHROUGH = {
    "nt": {"out", "out_dtype", "m_tiles"},
    "nn": {"out", "out_dtype", "m_tiles"},
    "vk": {"out", "out_dtype", "masked_k", "expert_table"},
}


def _tuning_args(kind: str):
    params = inspect.signature(_ENTRY[kind]).parameters
    return [n for n in params if n not in _OPERANDS[kind] and n not in PASSTHROUGH[kind]]


def _key(kind, n, k, avg_m):
    return gg._KWARGS_KEYS[kind][(n, k, avg_m)]


# ---------------------------------------------------------------------------
# Host-only.
# ---------------------------------------------------------------------------


def test_only_the_kimi_rows_are_keyed():
    """At the Kimi (N, K) shapes, exactly the keys at 293 / 585 / 1170."""
    for kind, _, n, k in SHAPES:
        got = sorted(row[2] for row in gg._KWARGS_KEYS[kind] if row[:2] == (n, k))
        assert got == list(KEYED_AVG_M), (kind, n, k)


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_a_key_names_only_entry_point_arguments(kind, name, n, k):
    tuning = set(_tuning_args(kind))
    for avg_m in KEYED_AVG_M:
        for knob, value in _key(kind, n, k, avg_m).items():
            assert knob in tuning, (
                f"{name}@{avg_m}: {knob} is not a tuning argument of the {kind} entry point"
            )
            assert isinstance(value, int) and not isinstance(value, bool), f"{name}@{avg_m}: {knob}={value!r}"


@pytest.mark.parametrize("kind, name, n, k", SHAPES[:4], ids=_IDS[:4])
def test_forward_and_dgrad_sets(kind, name, n, k):
    """293: 160-row tiles, no cluster, the multi-slot decode; 585: the 1170 set plus the half-height grid."""
    k293, k585, k1170 = (_key(kind, n, k, m) for m in KEYED_AVG_M)
    assert k293["BLOCK_M"] == 160 and k293["mv_dec"] == 1 and (k293["mc_m"], k293["mc_n"]) == (1, 1)
    assert "tt_dec" not in k293 and not k293["nt_seam"] & 8
    assert k585 == dict(k1170, nt_half=1)
    assert "vec_dec" not in k1170 and k1170["tt_dec"] == 1


@pytest.mark.parametrize("kind, name, n, k", SHAPES[4:], ids=_IDS[4:])
def test_wgrad_sets(kind, name, n, k):
    """The MiniMax-M3 seam plus vk_seam bits 20 and 21; bits 22-23 (f = 3) and no stagger at 293."""
    k293, k585, k1170 = (_key(kind, n, k, m) for m in KEYED_AVG_M)
    assert k585 == k1170 and k1170["vk_seam"] == 538783 | (1 << 20) | (1 << 21)
    assert k293 == dict(k1170, vk_seam=k1170["vk_seam"] | (3 << 22), vk_stagger=0)


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_keys_serve_every_row_count_in_reach(kind, name, n, k):
    """Every avg_m from 293 / _AVG_M_REACH to 1170 x _AVG_M_REACH takes the key nearest on a log scale, a tie the
    smaller one (414 and 827 sit just below the geometric midpoints); MBS 1's 65536 // 224 = 292 takes 293."""
    lo = math.ceil(KEYED_AVG_M[0] / gg._AVG_M_REACH)
    hi = math.floor(KEYED_AVG_M[-1] * gg._AVG_M_REACH)
    for avg_m in range(lo - 20, hi + 20):
        got = gg._kwargs_key(kind, (n, k, avg_m))
        if avg_m < lo or avg_m > hi:
            assert got is None, avg_m
            continue
        want = 293 if avg_m <= 414 else 585 if avg_m <= 827 else 1170
        assert got == (n, k, want), avg_m
    assert gg._kwargs_key(kind, (n, k, 65536 // MEASURED_G)) == (n, k, 293)


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_an_untuned_call_gets_the_whole_key(kind, name, n, k):
    for avg_m in KEYED_AVG_M:
        key = (n, k, avg_m)
        want = dict(_key(kind, n, k, avg_m))
        assert gg._keyed_kwargs(kind, key, MEASURED_G, 0, {}) == want
        assert gg._keyed_kwargs(kind, key, MEASURED_G, 0, dict.fromkeys(PASSTHROUGH[kind])) == want


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_any_tuning_argument_switches_the_key_off(kind, name, n, k):
    params = inspect.signature(_ENTRY[kind]).parameters
    for arg in _tuning_args(kind):
        assert gg._keyed_kwargs(kind, (n, k, 585), MEASURED_G, 0, {arg: params[arg].default}) == {}, arg


@pytest.mark.parametrize("kind, name, n, k", SHAPES[:4], ids=_IDS[:4])
def test_forward_and_dgrad_keys_fire_at_every_expert_count(kind, name, n, k):
    """No set carries vec_dec, so the expert count the launchers drop it above does not decline them."""
    for G in (32, 33, 64, 112, 224, 256):
        for avg_m in KEYED_AVG_M:
            assert gg._keyed_kwargs(kind, (n, k, avg_m), G, 0, {}), (G, avg_m)


@pytest.mark.parametrize("kind, name, n, k", SHAPES[4:], ids=_IDS[4:])
def test_variable_k_keys_need_whole_persistent_rounds(kind, name, n, k):
    """256 workgroups over G x (n / 256) x (k / 256) tiles: fc1 (336 per expert) needs G % 16 == 0, fc2 (168) needs
    G % 32 == 0. EP4 (224) takes both, EP8 (112) only fc1; fc2 then runs the untuned path."""
    for G in (8, 16, 32, 112, 224):
        tiles = G * (n // 256) * (k // 256)
        fits = tiles % 256 == 0 and tiles // 256 >= 2
        assert bool(gg._keyed_kwargs(kind, (n, k, 585), G, 0, {})) == fits, G
    assert bool(gg._keyed_kwargs(kind, (n, k, 585), 112, 0, {})) == (name == "wgrad_fc1")


def _padded(avg_m, tile_m):
    return -(-avg_m // tile_m) * tile_m


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_a_pinned_tile_height_pads_no_more_than_the_default(kind, name, n, k):
    """The 293 forward / dgrad sets run 160-row tiles; they serve a row count only where they pad each expert's rows
    to no more than the default tile does, else the call takes the next-nearest key within reach, else none.
    Below 257 rows the 160-row tile pads to 320 against the default 256; the variable-K sets are not affected."""
    lo = math.ceil(KEYED_AVG_M[0] / gg._AVG_M_REACH)
    for avg_m in range(lo, 830):
        nearest = [row[2] for row in gg._kwargs_rows(kind, (n, k, avg_m))]
        want = None
        for key_m in nearest:
            keyed = _key(kind, n, k, key_m)
            if kind != "vk":
                pick = gg._pick_config if kind == "nt" else gg._pick_config_nn
                cfg = pick(n, k, avg_m, MEASURED_G)
                if cfg != pick(n, k, key_m) or _padded(avg_m, keyed.get("BLOCK_M") or cfg[0]) > _padded(
                    avg_m, cfg[0]
                ):
                    continue
            want = key_m
            break
        got = gg._keyed_kwargs(kind, (n, k, avg_m), MEASURED_G, 0, {})
        assert got == (dict(_key(kind, n, k, want)) if want else {}), avg_m
        if kind != "vk" and 160 < avg_m <= 256 and nearest[0] == 293:
            assert got != _key(kind, n, k, 293), avg_m
    for avg_m in (257, 293, 320, 414):
        assert gg._keyed_kwargs(kind, (n, k, avg_m), MEASURED_G, 0, {}) == _key(kind, n, k, 293), avg_m


@pytest.mark.parametrize("kind, name, n, k", SHAPES[:4], ids=_IDS[:4])
def test_a_prebuilt_m_tiles_of_another_height_skips_the_set(kind, name, n, k):
    """A caller's prebuilt m_tiles is checked against the tile height it was built for, so a set of another
    height (the 293 one, 160 rows) is left out rather than refused at launch; one built for 160 rows takes it."""
    rows = [293] * MEASURED_G
    offs = torch.tensor([0, *itertools.accumulate(rows)], dtype=torch.int64)
    for block_m, served in ((256, False), (160, True)):
        kw = {"m_tiles": gg.build_m_tile_map(offs, block_m)}
        got = gg._keyed_kwargs(kind, (n, k, 293), MEASURED_G, 0, kw)
        assert (got == _key(kind, n, k, 293)) == served, block_m


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_the_picker_still_hands_out_the_measured_tile(kind, name, n, k):
    """The sets were measured on the picker's 256 x 256 x 128 tile (the 293 forward / dgrad ones and the wgrad
    ones then set their own tile fields)."""
    for avg_m in KEYED_AVG_M:
        for G in (112, MEASURED_G):
            if kind == "nt":
                cfg = gg._pick_config(n, k, avg_m, G)
            elif kind == "nn":
                cfg = gg._pick_config_nn(n, k, avg_m, G)
            else:
                cfg = gg._pick_variable_k_config(n, k, avg_m, False)
            assert cfg == (256, 256, 128, 2, 2, 2), (avg_m, G)


# ---------------------------------------------------------------------------
# Numeric: on a gfx1250 card.
# ---------------------------------------------------------------------------


def _offs(rows, device="cuda") -> torch.Tensor:
    return torch.tensor([0, *itertools.accumulate(rows)], dtype=torch.int64, device=device)


def _assert_matches(got: torch.Tensor, ref: torch.Tensor, dtype) -> None:
    from tests.pytorch.test_utils import compute_snr, get_tolerances

    snr = compute_snr(ref, got)
    assert snr > 45, f"out SNR {snr:.2f} dB is below the bf16 threshold of 45 dB"
    torch.testing.assert_close(got, ref, **get_tolerances(dtype))


def _ref(kind, a, b, offs):
    bounds = offs.cpu().tolist()
    a32, b32 = a.detach().float().cpu(), b.detach().float().cpu()
    out = []
    for g in range(len(bounds) - 1):
        lo, hi = bounds[g], bounds[g + 1]
        if kind == "nt":
            out.append((a32[lo:hi] @ b32[g].t()).to(a.dtype))
        elif kind == "nn":
            out.append((a32[lo:hi] @ b32[g]).to(a.dtype))
        else:
            out.append((a32[lo:hi].t() @ b32[lo:hi]).to(a.dtype))
    return torch.stack(out) if kind == "vk" else torch.cat(out)


def _operands(kind, n, k, rows):
    """The call as Primus-Turbo's FlyDSL backend makes it: no tuning argument, masked_k on variable-K."""
    torch.manual_seed(0)
    M, G = sum(rows), len(rows)
    offs = _offs(rows)
    if kind == "nt":
        a = torch.randn(M, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(G, n, k, dtype=torch.bfloat16, device="cuda")
        return (a, b, offs), {"out_dtype": torch.bfloat16}
    if kind == "nn":
        a = torch.randn(M, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(G, k, n, dtype=torch.bfloat16, device="cuda")
        return (a, b, offs), {"out_dtype": torch.bfloat16}
    a = torch.randn(M, n, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(M, k, dtype=torch.bfloat16, device="cuda")
    masked_k = torch.tensor(rows, dtype=torch.int64, device="cuda")
    return (a, b, offs), {"masked_k": masked_k, "out_dtype": torch.bfloat16}


def _call_four_times(entry, *args, **kwargs):
    for _ in range(3):
        entry(*args, **kwargs)
    return entry(*args, **kwargs)


# The fewest experts at which every key fires (whole persistent rounds on the wgrad fc2 key), which keeps the fp32
# host reference affordable: a key is read on M_total // G, not on M_total.
NUMERIC_G = {"nt": 8, "nn": 8, "vk": 32}


def _check(kind, n, k, rows, reference=True, keyed_call=True):
    args, kw = _operands(kind, n, k, rows)
    G = len(rows)
    if keyed_call:
        assert gg._keyed_kwargs(kind, (n, k, sum(rows) // G), G, 0, kw), "the call would not take a key"
    keyed = _call_four_times(_PUBLIC[kind], *args, **kw)
    # persist=0 is a tuning argument: the key is off and the call runs the untuned path
    plain = _call_four_times(_PUBLIC[kind], *args, persist=0, **kw)
    if reference:
        _assert_matches(keyed.detach().cpu(), _ref(kind, *args), torch.bfloat16)
    torch.testing.assert_close(keyed, plain, rtol=0, atol=0)


@needs_gfx1250
@pytest.mark.parametrize("avg_m", KEYED_AVG_M)
@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_keyed_shape_numeric(kind, name, n, k, avg_m):
    """Balanced routing at the key's row count: right, and bit-identical to the untuned path."""
    _check(kind, n, k, [avg_m] * NUMERIC_G[kind])


@needs_gfx1250
@pytest.mark.parametrize("avg_m", (250, 414, 827, 1600))
@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_nearest_key_numeric(kind, name, n, k, avg_m):
    """Ragged routing at a row count between or past the keys: the key serving it (at 250 the 293 forward / dgrad
    sets give way, see the tile-height test), bit-identical to the untuned path (an empty expert, a one-row expert
    and a run that does not fill a tile)."""
    G = NUMERIC_G[kind]
    rows = [0, 1, 159, 256 + 7] + [avg_m] * (G - 5)
    rows.append(avg_m * G - sum(rows))
    assert min(rows) >= 0 and sum(rows) // G == avg_m
    _check(kind, n, k, rows, reference=False, keyed_call=False)
