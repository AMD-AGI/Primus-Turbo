###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The launch plan of the gfx1250 MLA backward (qk 192, v 128): k_delta_bshd, then k_dkdv (dK, dV and the bf16 dS^T
tiles into a scratch) and k_dqds (dQ = dS K from it), on all heads at once when the whole scratch is within the budget
PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB, else per head chunk, the chunks alternating between two streams.

The launch and chunking tests use meta tensors and run anywhere. The numeric tests need a gfx1250 card and compare
head chunkings with the whole scratch bitwise: every head's arithmetic is the same however the heads are chunked.
test_attention_flydsl_gfx1250_mla checks the gradients against an fp32 reference, and
test_attention_hipkittens_gfx1250_mla against the HipKittens kernels, bitwise.
"""

import contextlib
import inspect

import pytest
import torch

import primus_turbo.common.envs as envs
from primus_turbo.pytorch.core.utils import is_gfx1250

needs_gfx1250 = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx1250()), reason="needs a gfx1250 (MI455X) GPU"
)

D_QK, D_V = 192, 128
MLA_SCALE = 0.1352337788608801  # DeepSeek-V3: yarn mscale**2 / sqrt(192)
_BUDGET = "PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB"


def _bwd():
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd

    return bwd


@pytest.fixture(autouse=True)
def _default_budget(monkeypatch):
    """Every test starts from the default scratch budget, whatever the environment sets."""
    monkeypatch.delenv(_BUDGET, raising=False)
    envs.reset_cache(_BUDGET)


def _set_budget(monkeypatch, mib):
    monkeypatch.setenv(_BUDGET, str(mib))
    envs.reset_cache(_BUDGET)


def _force_chunks(monkeypatch, b, sq, skv, h, hc, streams=None):
    """Make the backward run chunks of hc heads: all heads in one scratch, or chunks on two streams (streams=1: one
    stream). The choice among chunkings is test_ds_chunkings'."""
    bwd = _bwd()
    streams = streams or (1 if hc == h else 2)
    plan = [(hc, streams, bwd.ds_scratch_bytes(b, sq, skv, 1) * hc * streams)]
    monkeypatch.setattr(bwd, "ds_chunkings", lambda *args: plan)


def _launches(dout, q, k, v, out, lse, stream=None):
    """(launcher, args) of every launch of the backward in issue order, without launching: k_delta_bshd, then k_dkdv
    and k_dqds per head chunk."""
    bwd = _bwd()
    delta_launch, bufs = bwd._delta_launch(dout, q, k, v, out, lse, MLA_SCALE, stream)
    return [delta_launch] + bwd._chunk_launches(dout, q, k, v, MLA_SCALE, stream, bufs)


def _chunks(launches):
    """[(H0, HC)] of the k_dqds launches."""
    out = []
    for fn, args in launches[1:]:
        if fn.func.__name__ == "launch_dqds":
            a = inspect.signature(fn.func).bind(*args).arguments
            out.append((a["H0"], a["HC"]))
    return out


def _meta(b, sq, skv, h):
    q = torch.empty(b, sq, h, D_QK, dtype=torch.bfloat16, device="meta")
    k = torch.empty(b, skv, h, D_QK, dtype=torch.bfloat16, device="meta")
    v = torch.empty(b, skv, h, D_V, dtype=torch.bfloat16, device="meta")
    out = torch.empty(b, sq, h, D_V, dtype=torch.bfloat16, device="meta")
    lse = torch.empty(b, h, sq, dtype=torch.float32, device="meta")
    return q, k, v, out, lse


# ---------------------------------------------------------------------------
# Launches and chunking (meta tensors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((1, 4096, 4096, 256), id="dsv3_b2_folded"),
        pytest.param((2, 4096, 4096, 128), id="dsv3_b2"),
        pytest.param((1, 384, 384, 2), id="s384"),
        pytest.param((1, 256, 448, 2), id="s256_skv448"),
        pytest.param((1, 512, 512, 2), id="s512_two_head_tiles"),
        pytest.param((1, 2048, 2048, 4), id="s2048"),
        pytest.param((1, 8192, 8192, 128), id="s8192_h128"),
        pytest.param((1, 128, 256, 2), id="s128_partial_head_tile"),
        pytest.param((1, 320, 320, 2), id="s320"),
        pytest.param((1, 64, 128, 2), id="s64"),
        pytest.param((1, 1984, 1984, 8), id="s1984"),
        pytest.param((1, 16448, 16448, 128), id="past_2_21_bhs"),
    ],
)
def test_launch_sequence(dims):
    """Every shape the gate takes runs k_delta_bshd, then k_dkdv and k_dqds per head chunk."""
    bwd = _bwd()
    b, sq, skv, h = dims
    assert bwd.shape_unsupported_reason(b, sq, skv, h) is None
    q, k, v, out, lse = _meta(b, sq, skv, h)
    launches = _launches(out, q, k, v, out, lse)
    names = [fn.func.__name__ for fn, _ in launches]
    nchunks = (len(names) - 1) // 2
    assert nchunks >= 1 and names == ["launch_delta"] + ["launch_dkdv", "launch_dqds"] * nchunks


@pytest.mark.parametrize("chunked", [False, True], ids=["whole", "chunked"])
@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((1, 4096, 4096, 256), id="dsv3_b2_folded"),
        pytest.param((1, 384, 576, 2), id="s384_skv576"),
        pytest.param((2, 256, 320, 5), id="b2_s256_skv320_h5"),
        pytest.param((1, 1024, 1024, 40), id="s1024_h40"),
        pytest.param((1, 512, 512, 2), id="s512_two_head_tiles"),
        pytest.param((2, 896, 1024, 3), id="b2_s896_skv1024_h3"),
        pytest.param((1, 64, 64, 2), id="s64_partial_head_tile"),
        pytest.param((1, 320, 576, 2), id="s320_head_tiles_partial_and_whole"),
    ],
)
def test_grids_cover_every_row_and_head(monkeypatch, dims, chunked):
    """The head chunks partition the heads (one chunk: the whole scratch; chunked: a budget of half of it); per
    chunk, the item lists of k_dkdv's persistent workgroups hold every 192-row kv block of every (batch, head) of
    the chunk once, and k_dqds's tiles cover every query, over the same heads and batches."""
    bwd = _bwd()
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_dkdv_items as items
    from primus_turbo.flydsl.attention.gfx1250_mla.common import num_cus

    b, sq, skv, h = dims
    q, k, v, out, lse = _meta(b, sq, skv, h)
    if chunked:
        _set_budget(monkeypatch, bwd.ds_scratch_bytes(b, sq, skv, h) // 2 >> 20)
    launches = _launches(out, q, k, v, out, lse)
    pairs = list(zip(launches[1::2], launches[2::2]))
    assert len(launches) == 1 + 2 * len(pairs) and (len(pairs) > 1) == chunked
    ncu = num_cus(q.device)
    h_next = 0
    for (dkdv_fn, dkdv_args), (dq_fn, dq_args) in pairs:
        a = inspect.signature(dkdv_fn.func).bind(*dkdv_args).arguments
        d = inspect.signature(dq_fn.func).bind(*dq_args).arguments
        assert d["H0"] == h_next and d["HG"] * d["ngz"] == b * d["HC"]
        # 192-row blocks, then one block of the last skv % 192 (64 or 128) rows: the last block ends at skv
        nblk = items.nblk_kv(skv)
        assert (nblk - 1) * 192 < skv <= nblk * 192
        lists = items.item_lists(b, sq, skv, d["HC"], ncu)
        assert sorted(x for lst in lists for x in lst) == [
            (hb, bid) for hb in range(b * d["HC"]) for bid in range(nblk)
        ]
        # at most one workgroup per CU, none without an item; a list ends with a row of zeros
        assert a["nwg"] == len(lists) == min(ncu, b * d["HC"] * nblk) and all(lists)
        assert a["nslot"] == max(len(lst) for lst in lists) + 1
        assert tuple(a["TBL"].shape) == (a["nwg"], a["nslot"], 4, items.ROW_DW)
        # 384-query tiles over [q_off, sq), 256-query head tiles from q_off down past query 0
        nhead = d["ny"] - d["ntile"]
        assert d["q_off"] + 384 * d["ntile"] == sq and 0 <= d["q_off"] <= 256 * nhead < d["q_off"] + 256
        h_next += d["HC"]
    assert h_next == h


@pytest.mark.parametrize(
    ("b", "sq", "skv", "hc", "ncu"),
    [
        (4, 384, 384, 5, 256),
        (1, 704, 704, 20, 256),
        (3, 448, 448, 12, 256),
        (1, 2240, 2240, 20, 304),
        (4, 384, 512, 5, 64),
    ],
)
def test_item_lists_leave_no_workgroup_empty(b, sq, skv, hc, ncu):
    """Head counts that are no multiple of the 8 XCDs: every workgroup of the grid still gets an item."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_dkdv_items as items

    lists = items.item_lists(b, sq, skv, hc, ncu)
    nblk = items.nblk_kv(skv)
    assert len(lists) == min(ncu, b * hc * nblk) and all(lists)
    assert sorted(x for lst in lists for x in lst) == [
        (hb, bid) for hb in range(b * hc) for bid in range(nblk)
    ]


def test_item_rows_of_the_last_block():
    """A wave whose kv rows pass seqlen_kv reads no K / V row and stores no dS^T block; one inside it does."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_dkdv_items as items
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_dkdv_sched

    sq, skv, h = 320, 448, 2  # the last block: rows [384, 448) of 192, waves 2 and 3 past seqlen_kv
    rows = items.item_rows([1, 1], [0, 2], sq, skv, h, 0, h)
    r = bwd_dkdv_sched.ROW
    assert rows.shape == (2, 4, items.ROW_DW)
    assert rows[0, :, r["ext"]].tolist() == [48, 48, 48, 48] and rows[1, :, r["ext"]].tolist() == [
        48,
        16,
        0,
        0,
    ]
    assert rows[1, 2, r["rds"] + 2] == 0 and rows[1, 3, r["rds"] + 2] == 0 and rows[1, 0, r["rds"] + 2] > 0
    # every tile of the last block is masked; the first block starts at query pair 0
    assert rows[1, 0, r["n"]] == rows[1, 0, r["nmask"]] and rows[0, 0, r["qps"]] == 0


@pytest.mark.parametrize(
    ("dims", "first_hc"),
    [
        # all heads on one scratch within the default budget
        pytest.param((1, 4096, 4096, 256), 256, id="dsv3_b2_folded"),
        pytest.param((2, 4096, 4096, 128), 128, id="dsv3_b2"),
        pytest.param((1, 1024, 1024, 16), 16, id="small"),
        # the whole scratch is over the budget: the fewest chunks, on two streams, that fit
        pytest.param((1, 8192, 8192, 128), 32, id="s8192_h128"),
        pytest.param((1, 32768, 32768, 64), 2, id="s32768_h64"),
        pytest.param((96, 2048, 2048, 16), 4, id="b96_s2048_h16"),
    ],
)
def test_ds_chunkings(dims, first_hc):
    bwd = _bwd()
    b, sq, skv, h = dims
    budget = envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20
    tries = bwd.ds_chunkings(b, sq, skv, h, budget)
    per_head = bwd.ds_scratch_bytes(b, sq, skv, 1)
    assert tries[0][0] == first_hc
    for hc, streams, nbytes in tries:
        # one stream for all heads at once and for the last resort, one head at a time; two for chunks
        assert (streams == 1) == (hc == h or (hc, nbytes) == (1, per_head))
        assert nbytes == per_head * hc * streams <= budget
    # then smaller and smaller scratches, for one that cannot be allocated, down to one head on one stream
    assert [x[2] for x in tries] == sorted((x[2] for x in tries), reverse=True) and tries[-1] == (
        1,
        1,
        per_head,
    )


def test_ds_chunkings_round_up_uneven_heads():
    bwd = _bwd()
    per_head = bwd.ds_scratch_bytes(1, 1024, 1024, 1)
    tries = bwd.ds_chunkings(1, 1024, 1024, 40, 1 << 40)
    # (20 heads on two streams take the scratch of all 40 on one: no retry at the same size)
    assert [(hc, streams) for hc, streams, _ in tries] == [
        (40, 1),
        (10, 2),
        (5, 2),
        (3, 2),
        (2, 2),
        (1, 2),
        (1, 1),
    ]
    # (odd head counts: 2 heads on two streams take more scratch than all 3 on one)
    assert bwd.ds_chunkings(1, 1024, 1024, 3, 1 << 40) == [
        (3, 1, 3 * per_head),
        (1, 2, 2 * per_head),
        (1, 1, per_head),
    ]
    # a budget below the whole scratch: the fewest chunks that fit, then the smaller ones
    assert [x[:2] for x in bwd.ds_chunkings(1, 1024, 1024, 40, 12 * per_head)] == [
        (5, 2),
        (3, 2),
        (2, 2),
        (1, 2),
        (1, 1),
    ]
    # below one head's scratch (0 included) only one head at a time is left
    assert bwd.ds_chunkings(1, 1024, 1024, 40, 0) == [(1, 1, per_head)]
    assert bwd.ds_chunkings(1, 1024, 1024, 1, 1 << 40) == [(1, 1, per_head)]


def test_zero_budget_runs_one_head_at_a_time(monkeypatch):
    _set_budget(monkeypatch, 0)
    q, k, v, out, lse = _meta(1, 384, 384, 3)
    assert _chunks(_launches(out, q, k, v, out, lse)) == [(0, 1), (1, 1), (2, 1)]


def test_scratch_is_the_last_allocation(monkeypatch):
    """The dS scratch is the backward's last allocation: when it fits, the row constants, the gradients and the
    item tables already did, so a scratch that fits never leaves the backward short of memory."""
    bwd = _bwd()
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_dkdv_items as items

    b, sq, skv, h = 1, 384, 384, 2
    q, k, v, out, lse = _meta(b, sq, skv, h)
    sizes = []

    def record(alloc):
        def fn(*args, **kwargs):
            t = alloc(*args, **kwargs)
            sizes.append(t.numel())
            return t

        return fn

    def table(*args):
        sizes.append("table")
        return item_table(*args)

    for owner, name in ((torch, "empty"), (torch, "empty_like"), (torch.Tensor, "new_empty")):
        monkeypatch.setattr(owner, name, record(getattr(owner, name)))
    item_table = items.item_table
    monkeypatch.setattr(items, "item_table", table)
    _launches(out, q, k, v, out, lse)
    scratch = bwd.ds_chunkings(b, sq, skv, h, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)[0][2] // 2
    assert sizes[-1] == scratch and sizes.count(scratch) == 1 and "table" in sizes, sizes


def test_item_tables_of_many_chunks_stay_cached(monkeypatch):
    """One head at a time over more heads than a cache counted in tables would hold: the second backward makes no
    table."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_dkdv_items as items

    _set_budget(monkeypatch, 0)
    q, k, v, out, lse = _meta(1, 384, 384, 130)
    made = []
    item_lists = items.item_lists
    monkeypatch.setattr(items, "item_lists", lambda *args: made.append(args) or item_lists(*args))
    monkeypatch.setattr(items, "_TABLES", items._TableCache(items.MAX_TABLE_BYTES))
    assert len(_chunks(_launches(out, q, k, v, out, lse))) == 130 and len(made) == 130
    _launches(out, q, k, v, out, lse)
    assert len(made) == 130


def test_item_table_cache_evicts_by_bytes_and_keeps_captured_tables():
    """The cache drops its least recently used tables once it is over its bytes, never the newest one, and never a
    table handed out while a CUDA graph was being captured: the graph keeps its address."""
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_dkdv_items as items

    def entry(nbytes):
        return (torch.empty(nbytes // 4, dtype=torch.int32, device="meta"), 1, 2)

    cache = items._TableCache(max_bytes=1000)
    a, b, c, d = entry(400), entry(400), entry(400), entry(4000)
    cache.put("a", a, capturing=True)
    cache.put("b", b, capturing=False)
    assert cache.get("a", False) is a  # a is now the most recently used
    cache.put("c", c, capturing=False)  # over the bytes: b, the least recently used, goes
    assert cache.get("b", False) is None and cache.get("c", False) is c and cache.nbytes == 800
    cache.put("d", d, capturing=False)  # larger than the cache: it alone stays
    assert list(cache.lru) == ["d"] and cache.nbytes == 4000
    assert cache.get("c", False) is None and cache.get("a", False) is a  # a graph captured a
    assert cache.get("d", True) is d and set(cache.captured) == {"a", "d"}
    # a key's first table stays (two threads made it at once): a graph may hold its address
    assert (
        cache.put("a", entry(400), capturing=True) is a and cache.put("d", entry(400), capturing=False) is d
    )
    assert cache.nbytes == 4000


def test_side_stream_is_joined_when_a_launch_raises():
    """The caller's stream waits for the side stream also when a launch raised: the kernels the side stream already
    runs still write the backward's buffers."""
    bwd = _bwd()
    calls = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            calls.append((self.name, other.name))

    main, side = Stream("main"), Stream("side")
    with pytest.raises(RuntimeError, match="launch"), bwd._joined(main, side):
        raise RuntimeError("launch")
    assert calls == [("side", "main"), ("main", "side")]
    calls.clear()
    with bwd._joined(main, None):
        pass
    assert calls == []


# ---------------------------------------------------------------------------
# Numerics (gfx1250)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _nan_filled_allocations():
    """Buffers the host code allocates start as NaN (the dS scratch included), so an element no launch writes, or a
    scratch block read before it is written, shows up as a mismatch or a NaN."""

    def nan(alloc):
        def fn(*args, **kwargs):
            t = alloc(*args, **kwargs)
            return t.fill_(float("nan")) if t.is_floating_point() and t.device.type == "cuda" else t

        return fn

    with pytest.MonkeyPatch.context() as mp:
        for owner, name in ((torch, "empty"), (torch, "empty_like"), (torch.Tensor, "new_empty")):
            mp.setattr(owner, name, nan(getattr(owner, name)))
        yield


def _problem(b, sq, skv, h, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(s, d):
        return torch.randn(b, s, h, d, device="cuda", generator=g).to(torch.bfloat16)

    q, k, v, do = r(sq, D_QK), r(skv, D_QK), r(skv, D_V), r(sq, D_V)
    from primus_turbo.flydsl.attention.gfx1250_mla import fwd

    out, lse = fwd.flash_attn_mla_fwd(q, k, v, MLA_SCALE, True)
    return do, q, k, v, out, lse


def _grads(problem):
    with _nan_filled_allocations():
        grads = _bwd().flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    torch.cuda.synchronize()
    return grads


def _whole(monkeypatch, problem):
    """The gradients with every head in one scratch."""
    b, sq, h, _ = problem[1].shape
    with monkeypatch.context() as mp:
        _force_chunks(mp, b, sq, problem[2].shape[1], h, h)
        return _grads(problem)


@needs_gfx1250
@pytest.mark.parametrize(
    ("dims", "hc"),
    [
        pytest.param((1, 384, 384, 2), 1, id="s384"),
        pytest.param((1, 384, 384, 3), -1, id="s384_one_head_at_a_time"),
        pytest.param((1, 256, 256, 3), 1, id="s256_head_tile_only"),
        pytest.param((2, 640, 640, 2), 1, id="b2_s640"),
        pytest.param((1, 256, 448, 2), 1, id="s256_skv448_tail64"),
        pytest.param((1, 384, 512, 2), 1, id="s384_skv512_tail32"),
        pytest.param((1, 768, 832, 4), 2, id="s768_skv832_tail64"),
        pytest.param((1, 1024, 1024, 40), 10, id="s1024_h40_four_chunks"),
        pytest.param((1, 1024, 1024, 40), 3, id="s1024_h40_uneven_chunks"),
        pytest.param((2, 2048, 2048, 8), 2, id="b2_s2048"),
        pytest.param((1, 512, 512, 2), 1, id="s512_two_head_tiles_only"),
        pytest.param((1, 896, 960, 2), 1, id="s896_skv960_two_head_tiles"),
        pytest.param((1, 64, 64, 2), 1, id="s64_partial_head_tile"),
        pytest.param((1, 192, 448, 2), 1, id="s192_skv448"),
        pytest.param((1, 64, 1024, 2), 1, id="s64_skv1024_dead_pairs_unmasked"),
        pytest.param((2, 576, 576, 4), 2, id="b2_s576"),
        pytest.param((1, 4160, 4160, 16), 4, id="s4160_h16"),
        pytest.param((1, 8192, 8192, 16), 4, id="s8192_h16"),
        pytest.param((1, 16448, 16448, 128), 16, id="s16448_h128_past_2_21_bhs"),
    ],
)
def test_head_chunks_equal_the_whole_scratch(monkeypatch, dims, hc):
    """Head chunks alternating between two streams, each stream's chunks on its own scratch, give the bits of all
    heads in one scratch, and write every element (NaN-filled buffers)."""
    problem = _problem(*dims)
    ref = _whole(monkeypatch, problem)
    _force_chunks(
        monkeypatch, *dims, abs(hc), streams=1 if hc < 0 else None
    )  # -1: one head at a time, one stream
    got = _grads(problem)
    for name, x, r in zip(("dq", "dk", "dv"), got, ref):
        assert torch.isfinite(x).all() and torch.isfinite(r).all(), name
        assert torch.equal(x, r), name


@needs_gfx1250
@pytest.mark.parametrize(
    ("dims", "hc"),
    [
        pytest.param((1, 1024, 1024, 40), 10, id="h40_four_chunks"),
        pytest.param((1, 4096, 4096, 256), 64, id="dsv3_b2_folded_four_chunks"),
    ],
)
def test_budget_below_the_whole_scratch_runs_chunks(monkeypatch, dims, hc):
    """The default budget runs all heads on one scratch; a budget of half the whole scratch runs four chunks on two
    streams, to the same bits."""
    bwd = _bwd()
    b, sq, skv, h = dims
    problem = _problem(*dims)
    assert bwd.ds_chunkings(b, sq, skv, h, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)[0][:2] == (h, 1)
    ref = _grads(problem)
    _set_budget(monkeypatch, -(-bwd.ds_scratch_bytes(b, sq, skv, h) // 2 >> 20))
    assert bwd.ds_chunkings(b, sq, skv, h, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)[0][:2] == (hc, 2)
    assert all(torch.equal(x, r) for x, r in zip(_grads(problem), ref))


@needs_gfx1250
def test_backward_is_deterministic():
    problem = _problem(1, 640, 640, 4, seed=3)
    a = _grads(problem)
    b = _grads(problem)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@needs_gfx1250
def test_chunked_backward_under_cuda_graphs(monkeypatch):
    """A captured chunked backward (its side stream forks from the capture stream and joins it again) replays to the
    eager bits, also after the cache dropped its item tables: the graph keeps their addresses, so the tables handed
    out during the capture stay."""
    bwd = _bwd()
    from primus_turbo.flydsl.attention.gfx1250_mla import bwd_dkdv_items as items

    dims = (1, 1024, 1024, 40)
    monkeypatch.setattr(items, "_TABLES", items._TableCache(items.MAX_TABLE_BYTES))
    _force_chunks(monkeypatch, *dims, 10)
    problem = _problem(*dims)
    eager = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(got, eager))
    # a cache of no bytes drops every table but the newest (another shape's): the four chunks' tables stay
    cache = items._TABLES
    tables = {key: entry[0].data_ptr() for key, entry in cache.captured.items()}
    assert len(tables) == 4
    cache.max_bytes = 0
    cache.put("another", (torch.zeros(8, dtype=torch.int32, device="cuda"), 1, 2), capturing=False)
    assert list(cache.lru) == ["another"]
    assert {key: entry[0].data_ptr() for key, entry in cache.captured.items()} == tables
    for x in got:
        x.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(got, eager))


@needs_gfx1250
def test_side_stream_per_caller_stream(monkeypatch):
    """Chunked backwards called on two streams run their chunks on two side streams of their own, and match."""
    bwd = _bwd()
    dims = (1, 1024, 1024, 8)
    _force_chunks(monkeypatch, *dims, 4)
    problem = _problem(*dims, seed=11)
    monkeypatch.setattr(bwd, "_SIDE_STREAMS", {})
    callers = [torch.cuda.Stream(), torch.cuda.Stream()]
    grads = []
    for s in callers:
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            grads.append(bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True))
    torch.cuda.synchronize()
    assert sorted(bwd._SIDE_STREAMS) == sorted((torch.cuda.current_device(), s.cuda_stream) for s in callers)
    assert all(torch.equal(x, y) for x, y in zip(*grads))


@needs_gfx1250
def test_scratch_oom_tries_smaller_chunks(monkeypatch):
    """A dS scratch that cannot be allocated sends the call to smaller head chunks, and that size is not tried again
    on the device."""
    bwd = _bwd()
    dims = (1, 1024, 1024, 8)
    problem = _problem(*dims, seed=7)
    ref = _grads(problem)
    tries = bwd.ds_chunkings(*dims, envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20)
    first = tries[0][2]
    assert tries[0][0] == 8 and any(nbytes < first for _, _, nbytes in tries)
    real_empty = torch.empty

    def empty(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        if size == (first // 2,) and kwargs.get("dtype") is torch.bfloat16:
            raise torch.OutOfMemoryError("test")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(bwd, "_DS_OOM_BYTES", {})
    monkeypatch.setattr(torch, "empty", empty)
    got = bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
    launches = _launches(*problem, stream=torch.cuda.current_stream())
    monkeypatch.setattr(torch, "empty", real_empty)
    torch.cuda.synchronize()
    assert bwd._DS_OOM_BYTES == {problem[1].device: first} and len(_chunks(launches)) > 1
    assert all(torch.equal(x, r) for x, r in zip(got, ref))


@needs_gfx1250
def test_scratch_oom_for_one_head_raises(monkeypatch):
    """With no memory even for one head's scratch the backward raises torch.OutOfMemoryError, saying so."""
    bwd = _bwd()
    problem = _problem(1, 384, 384, 2, seed=5)
    real_empty = torch.empty

    def empty(*args, **kwargs):
        size = args[0] if args else kwargs.get("size")
        if isinstance(size, tuple) and len(size) == 1 and kwargs.get("dtype") is torch.bfloat16:
            raise torch.OutOfMemoryError("test")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(bwd, "_DS_OOM_BYTES", {})
    monkeypatch.setattr(torch, "empty", empty)
    with pytest.raises(torch.OutOfMemoryError, match="even one head's dS scratch"):
        bwd.flash_attn_mla_bwd(*problem, MLA_SCALE, True)
