###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The one-launch index tables of the FlyDSL bf16 grouped GEMM on gfx1250.

A launcher that is passed no ``m_tiles`` / ``expert_table`` builds the table it needs, and for device offsets it
does so in one kernel launch (``_m_tile_tables`` / ``_expert_table``) instead of with the PyTorch builders, which
stay the reference. These tests hold the two to the same int32 words:

* Host-only (needs flydsl, no GPU): CPU offsets get the PyTorch builders, as before.
* On a gfx1250 card: the map, the per-tile records behind it (``tt_dec``) and the expert table, bit for bit, over
  expert counts from 1 to 1024, the tile heights the kernels use, routings with empty experts, one hot expert or no
  rows at all, tile counts padded for a cluster, int32 and int64 offsets; and the keyed MiniMax-M3 calls -- a map
  on the forward, the per-tile records on dgrad, the expert table on wgrad -- run none of the PyTorch builders and
  return the same bits as with the tables prebuilt by them.

Listed in ``tests/conftest.py``'s ``_GFX1250_ENABLED_TESTS``, without which it never runs on gfx1250.
"""

import itertools

import pytest
import torch

pytest.importorskip("flydsl", reason="the FlyDSL gfx1250 grouped GEMM lives behind flydsl")

from primus_turbo.flydsl.grouped_gemm import grouped_gemm_bf16_kernel_gfx1250 as gg  # noqa: E402


def _have_gfx1250() -> bool:
    """A gfx1250 device is visible; see the same helper in test_grouped_gemm_flydsl_gfx1250_keys.py."""
    try:
        from primus_turbo.pytorch.core.utils import is_gfx1250

        return bool(torch.cuda.is_available() and is_gfx1250())
    except Exception:
        return False


needs_gfx1250 = pytest.mark.skipif(not _have_gfx1250(), reason="needs a gfx1250 (MI455X) GPU")

# The tile heights the pickers and the keys hand out.
BLOCK_MS = (64, 128, 160, 256)
GROUPS = (1, 3, 32, 33, 224, 1024)
AVG_M = 512


def _routings(G: int):
    """Rows per expert: balanced, log-normal, one hot expert among empty ones, random with empties, a pattern of
    runs either side of a tile boundary, and no rows at all."""
    gen = torch.Generator().manual_seed(G)
    w = torch.exp(torch.randn(G, generator=gen, dtype=torch.float64) * 0.3)
    skew = torch.floor(w / w.sum() * AVG_M * G).to(torch.int64).clamp_min(1)
    skew[torch.argmax(skew)] += AVG_M * G - int(skew.sum())
    hot = torch.zeros(G, dtype=torch.int64)
    hot[G // 2] = AVG_M * G + 7
    rand = torch.randint(0, 2 * AVG_M, (G,), generator=gen)
    rand[torch.rand(G, generator=gen) < 0.3] = 0
    runs = torch.tensor([0, 1, 255, 256, 257, 511, 4097, 0, 129, 2047, 2049, 3]).repeat(G // 12 + 1)[:G]
    return {
        "even": torch.full((G,), AVG_M),
        "lognorm30": skew,
        "hot": hot,
        "random": rand,
        "runs": runs,
        "no_rows": torch.zeros(G, dtype=torch.int64),
    }


def _offs(lens, dtype, device):
    return torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]).to(dtype=dtype, device=device)


def _builders(offs, block_m: int, upper: int):
    m_map = gg.build_m_tile_map(offs, block_m)
    return gg.build_m_tile_tt(m_map, offs, block_m, upper) if upper else m_map


def _uppers(lens, block_m: int):
    """0 (the map alone) and the launchers' tile counts for a cluster of 1, 2 and 4 M tiles."""
    up = gg.m_tile_upper_bound(int(lens.sum()), block_m, lens.numel())
    return sorted({0} | {up + -up % mc for mc in (1, 2, 4)})


# ---------------------------------------------------------------------------
# Host-only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (torch.int64, torch.int32))
def test_cpu_offsets_take_the_pytorch_builders(dtype):
    for G in (1, 3, 32):
        for name, lens in _routings(G).items():
            offs = _offs(lens, dtype, "cpu")
            for bm in (64, 256):
                for upper in _uppers(lens, bm):
                    got = gg._m_tile_tables(offs, bm, upper)
                    assert torch.equal(got, _builders(offs, bm, upper)), (G, name, bm, upper)
            for mk in (None, lens.to(dtype)):
                assert torch.equal(gg._expert_table(offs, mk), gg.build_expert_table(offs, mk).reshape(-1))


# ---------------------------------------------------------------------------
# Numeric, on a gfx1250 card.
# ---------------------------------------------------------------------------


@needs_gfx1250
@pytest.mark.parametrize("G", GROUPS)
def test_m_tile_tables_match_the_builders(G):
    for dtype in (torch.int64, torch.int32):
        for name, lens in _routings(G).items():
            offs = _offs(lens, dtype, "cuda")
            for bm in BLOCK_MS:
                for upper in _uppers(lens, bm):
                    got = gg._m_tile_tables(offs, bm, upper)
                    assert torch.equal(got, _builders(offs, bm, upper)), (dtype, name, bm, upper)


@needs_gfx1250
@pytest.mark.parametrize("G", GROUPS)
def test_expert_table_matches_the_builder(G):
    for dtype in (torch.int64, torch.int32):
        for name, lens in _routings(G).items():
            offs = _offs(lens, dtype, "cuda")
            for mk in (None, lens.cuda(), (lens // 2).to(device="cuda", dtype=torch.int32)):
                got = gg._expert_table(offs, mk)
                assert torch.equal(got, gg.build_expert_table(offs, mk).reshape(-1)), (dtype, name, mk)


# MiniMax-M3 at 512 rows per expert: the forward key decodes from the map, the dgrad keys from the per-tile records
# (`tt_dec`, cluster of 2 M tiles), the variable-K key reads the expert table. NN keys by (output N, reduction K).
KEYED = (("nt", 6144, 6144), ("nn", 6144, 6144), ("nn", 3072, 6144), ("vk", 6144, 6144))
KEYED_G = 8


def _call_four_times(entry, *args, **kwargs):
    """Four calls, checking the fourth (the NN entry records wrong results on a fresh process's first calls)."""
    for _ in range(3):
        entry(*args, **kwargs)
    return entry(*args, **kwargs)


@needs_gfx1250
@pytest.mark.parametrize("kind, n, k", KEYED, ids=["nt_fc1", "nn_fc1", "nn_fc2", "vk_fc1"])
def test_keyed_call_builds_its_table_in_one_launch(kind, n, k, monkeypatch):
    rows = [0, 1, 40, 255, AVG_M + 7, 2 * AVG_M, 3 * AVG_M]
    rows.append(KEYED_G * AVG_M - sum(rows))
    torch.manual_seed(0)
    M = sum(rows)
    offs = torch.tensor([0, *itertools.accumulate(rows)], dtype=torch.int64, device="cuda")
    bf16 = torch.bfloat16
    if kind == "vk":
        entry = gg.grouped_gemm_bf16_variable_k_flydsl_kernel
        a, b = torch.randn(M, n, dtype=bf16, device="cuda"), torch.randn(M, k, dtype=bf16, device="cuda")
        lens = torch.tensor(rows, dtype=torch.int64, device="cuda")
        kw = {"masked_k": lens, "out_dtype": bf16}
        assert gg._keyed_kwargs(kind, (n, k, AVG_M), KEYED_G, 0, kw)
        prebuilt = {"expert_table": gg.build_expert_table(offs, lens)}
    else:
        entry = (
            gg.grouped_gemm_bf16_nt_flydsl_kernel if kind == "nt" else gg.grouped_gemm_bf16_nn_flydsl_kernel
        )
        a = torch.randn(M, k, dtype=bf16, device="cuda")
        b = torch.randn(KEYED_G, *((n, k) if kind == "nt" else (k, n)), dtype=bf16, device="cuda")
        kw = {"out_dtype": bf16}
        keyed = gg._keyed_kwargs(kind, (n, k, AVG_M), KEYED_G, 0, kw)
        assert keyed and bool(keyed.get("tt_dec")) == (kind == "nn")
        pick = gg._pick_config if kind == "nt" else gg._pick_config_nn
        prebuilt = {
            "m_tiles": gg.build_m_tile_map(offs, keyed.get("BLOCK_M") or pick(n, k, AVG_M, KEYED_G)[0])
        }
    want = _call_four_times(entry, a, b, offs, **kw, **prebuilt)

    def refuse(*args, **kwargs):
        raise AssertionError("a launcher passed no table ran a PyTorch builder")

    for name in ("build_m_tile_map", "build_m_tile_tt", "build_expert_table"):
        monkeypatch.setattr(gg, name, refuse)
    got = _call_four_times(entry, a, b, offs, **kw)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
