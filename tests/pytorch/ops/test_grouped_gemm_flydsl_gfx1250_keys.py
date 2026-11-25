###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The shape key table of the FlyDSL bf16 grouped GEMM on gfx1250, and the six DeepSeek-V3 shapes.

``test_grouped_gemm_flydsl_gfx1250_shapes.py`` covers the K rules -- which shapes the kernel may
run at all. This file covers the layer above it: which shapes get the tuned persistent-cluster
stack, and what a shape that misses it, or a caller that switches it off, is left with.

Two groups:

* Host-only. The key table, the whole-group sentinel, the two cluster gates and the ``head_pipe``
  gate are plain Python, so they need flydsl importable but no GPU. The load-bearing one is
  ``test_stack_off_resolves_to_the_pre_group_values``: three knobs of the group were once applied
  outside it, which cost a measured 0.86% / 0.57% / 0.44% on the three operators concerned when a
  caller switched the stack off, and nothing in the entry points stops that from coming back.
  The row counts near the key (``_shape_avg_m``) are host-only too, down to what the entry points
  launch, recorded with the launch itself replaced.
* Numeric, on a gfx1250 card: the six delivery shapes of the DeepSeek-V3 MoE layer, at the
  ``avg_m`` the key table is written for and at a row count it serves from nearby. These are the
  only shapes in the repository's test suite that take the keyed path, so without them the whole
  table is untested on hardware.

``tests/conftest.py`` lists this file in ``_GFX1250_ENABLED_TESTS``; on gfx1250 every test outside
that list is skipped, so an unlisted file never runs on the part it was written for.
"""

import inspect
import itertools
from collections.abc import Mapping

import pytest
import torch

import primus_turbo.common.envs as envs

pytest.importorskip("flydsl", reason="the FlyDSL gfx1250 grouped GEMM lives behind flydsl")

from primus_turbo.flydsl.grouped_gemm import grouped_gemm_bf16_kernel_gfx1250 as gg  # noqa: E402
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import (  # noqa: E402
    grouped_gemm_bf16_nn_flydsl_kernel,
    grouped_gemm_bf16_nt_flydsl_kernel,
    grouped_gemm_bf16_variable_k_flydsl_kernel,
)


def _have_gfx1250() -> bool:
    """A gfx1250 device is visible, answered without raising and without a top-level import.

    ``primus_turbo.pytorch`` probes the device compute capability while importing
    (``low_precision`` calls it at module scope) and that probe does not consult
    ``torch.cuda.is_available()``, so importing it anywhere in this file would make the whole
    module uncollectable on a machine with no GPU -- including the host-only group, which is
    exactly the machine it is meant to guard. Kept local and defensive for that reason; the
    FlyDSL kernel module itself imports cleanly without a device.
    """
    try:
        from primus_turbo.pytorch.core.utils import is_gfx1250

        return bool(torch.cuda.is_available() and is_gfx1250())
    except Exception:
        return False


needs_gfx1250 = pytest.mark.skipif(not _have_gfx1250(), reason="needs a gfx1250 (MI455X) GPU")

# The six operators of a DeepSeek-V3 MoE layer, G=32 experts, H=7168, F=2048 (so the fc1 weight is
# 2F x H for the gate/up pair), at the per-expert row count the key table is written for. The NT
# and NN keys are (N, K) of the call; the variable-K keys are (OUT_M, OUT_N).
KEYED_AVG_M = 2048
NT_SHAPES = {"fwd_fc1": (4096, 7168), "fwd_fc2": (7168, 2048)}
NN_SHAPES = {"dgrad_fc1": (7168, 4096), "dgrad_fc2": (2048, 7168)}
VK_SHAPES = {"wgrad_fc1": (4096, 7168), "wgrad_fc2": (7168, 2048)}
DELIVERY = (
    ("nt", "fwd_fc1", *NT_SHAPES["fwd_fc1"], (256, 320, 128, 2, 2, 2)),
    ("nt", "fwd_fc2", *NT_SHAPES["fwd_fc2"], (256, 320, 128, 2, 2, 2)),
    ("nn", "dgrad_fc1", *NN_SHAPES["dgrad_fc1"], (256, 256, 128, 2, 2, 2)),
    ("nn", "dgrad_fc2", *NN_SHAPES["dgrad_fc2"], (256, 256, 128, 2, 2, 2)),
    ("vk", "wgrad_fc1", *VK_SHAPES["wgrad_fc1"], (256, 256, 64, 2, 2, 4)),
    ("vk", "wgrad_fc2", *VK_SHAPES["wgrad_fc2"], (256, 256, 64, 2, 2, 4)),
)
_IDS = [f"{kind}-{name}" for kind, name, *_ in DELIVERY]
# The variable-K cluster gate reads the caller's M group; no caller passes one, so it is the default.
_VK_GROUP_M = inspect.signature(grouped_gemm_bf16_variable_k_flydsl_kernel).parameters["group_m"].default


def _picked_tile(kind: str, a: int, b: int, avg_m: int, G: int):
    """The tile the entry point will build the key lookup against."""
    key_m = gg._shape_avg_m(kind, a, b, avg_m)
    if kind == "nt":
        return gg._pick_config(a, b, avg_m, G, key_m=key_m)
    if kind == "nn":
        return gg._pick_config_nn(a, b, avg_m, G)
    # The variable-K picker answers a different ring depth inside the group than outside it, and
    # the lookup compares against the inside one -- see the ring-depth test below.
    return gg._pick_variable_k_config(a, b, avg_m, True, key_m)


def _keyed_group(kind: str, a: int, b: int, avg_m: int = KEYED_AVG_M) -> dict:
    """Every knob of the group as the table resolves it, including the ones it leaves off."""
    names = gg._STACK_VK if kind == "vk" else gg._STACK_NT
    entry = gg._STACK_KEYS[kind][(a, b, avg_m)]
    return {n: entry.get(n, gg._STACK_OFF[n]) for n in names}


# ---------------------------------------------------------------------------
# Host-only: the key table itself.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind, name, a, b, tile", DELIVERY, ids=_IDS)
def test_delivery_shapes_hit_the_stack_keys(kind, name, a, b, tile):
    """Each delivery shape is in the table, and the picker hands out the tile it was tuned on.

    ``_pick_stack`` returns ``None`` when ``cfg != keyed["cfg"]``, so a tile change anywhere in the
    pickers silently drops the whole stack rather than running it on a tile it was never measured
    on. That failure is invisible from the outside -- the kernel still computes the right answer,
    just at the untuned speed -- which is why it is asserted here rather than left to a benchmark.
    """
    key = (a, b, KEYED_AVG_M)
    assert key in gg._STACK_KEYS[kind], f"{kind} {name}: {key} is not in _STACK_KEYS"
    assert gg._STACK_KEYS[kind][key]["cfg"] == tile, (
        f"{kind} {name}: the table was tuned on {tile} but now holds {gg._STACK_KEYS[kind][key]['cfg']}"
    )
    for G in (8, 16, 32, 64):
        got = _picked_tile(kind, a, b, KEYED_AVG_M, G)
        assert got == tile, f"{kind} {name} G={G}: picker gives {got}, the table wants {tile}"


@pytest.mark.parametrize("kind, name, a, b, tile", DELIVERY, ids=_IDS)
def test_only_avg_m_2048_is_keyed(kind, name, a, b, tile):
    """The table holds avg_m = 2048 only, and the row counts within a factor ``_AVG_M_REACH`` of it read it at
    2048 (see the near-key tests below). At 512 rows the variable-K group still refuses itself: measured
    separately, its knobs are worth +34.8% there, but its ``split_bar`` needs more k windows than 512 rows give
    the deep ring, which no new row here can change (see the variable-K window test below).
    """
    for avg_m in (64, 128, 256, 512, 1024, 4096):
        assert (a, b, avg_m) not in gg._STACK_KEYS[kind], (
            f"{kind} {name}: a key appeared at avg_m={avg_m}; every gate and both cluster "
            f"checks have to be re-derived for a new row count, not just the tile"
        )


def test_variable_k_ring_depth_is_the_group_s_own():
    """The fourth ring slot exists only for the group, and only the group may see it.

    ``vk_ovl`` bit 1 prefetches the next tile's k-tiles 0 and 1 into the two slots past the C
    staging, so the depth is useless without the seam and was measured at -0.27% / +0.09% on its
    own. The picker therefore answers 3 outside the group and 4 inside it, and the lookup compares
    the tile including that slot -- so the two values are load-bearing in both directions.
    """
    for name, (out_m, out_n) in VK_SHAPES.items():
        off = gg._pick_variable_k_config(out_m, out_n, KEYED_AVG_M, False)
        on = gg._pick_variable_k_config(out_m, out_n, KEYED_AVG_M, True)
        assert off[5] == 3, f"{name}: the ungrouped ring depth is {off[5]}, not the measured 3"
        assert on[5] == 4, f"{name}: the grouped ring depth is {on[5]}, not the measured 4"
        assert off[:5] == on[:5], f"{name}: {off} and {on} differ in more than the ring depth"
        assert gg._pick_variable_k_config(out_m, out_n, KEYED_AVG_M)[5] == 3, (
            f"{name}: the default of `stack` is not the off value, so every caller that does not "
            f"pass it -- including the public gate -- would be answered the grouped depth"
        )


# ---------------------------------------------------------------------------
# Host-only: the whole-group sentinel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["nt", "nn", "vk"])
def test_any_one_knob_turns_the_whole_group_off(kind):
    """Setting any single member by hand has to drop the lookup for all of them.

    The members are interdependent -- ``ntp_flags``, ``nt_ovl``, ``vk_ovl`` and ``tdm_deal`` bit 1
    each need another one -- so a lookup that filled in the rest around a hand-set knob would
    build a combination that was never measured and, for several pairs, is rejected by the
    launcher outright. Probed one knob at a time, with the knob set to its own off value: that is
    the case where "it was already off anyway" is most tempting and still has to turn the group
    off, because off and unset are different things once the table can supply a value.
    """
    names = gg._STACK_VK if kind == "vk" else gg._STACK_NT
    a, b = next(iter({"nt": NT_SHAPES, "nn": NN_SHAPES, "vk": VK_SHAPES}[kind].values()))
    cfg = gg._STACK_KEYS[kind][(a, b, KEYED_AVG_M)]["cfg"]
    unset = tuple(-1 for _ in names)

    assert gg._pick_stack(kind, a, b, KEYED_AVG_M, names, unset, cfg, True) is not None
    assert gg._pick_stack(kind, a, b, KEYED_AVG_M, names, unset, cfg, False) is None
    for i, knob in enumerate(names):
        for value in (gg._STACK_OFF[knob], 0, 1):
            passed = list(unset)
            passed[i] = value
            got = gg._pick_stack(kind, a, b, KEYED_AVG_M, names, tuple(passed), cfg, True)
            assert got is None, f"{kind}: {knob}={value} left the lookup on and resolved {got}"


@pytest.mark.parametrize(
    "entry, names",
    [
        (grouped_gemm_bf16_nt_flydsl_kernel, "nt"),
        (grouped_gemm_bf16_nn_flydsl_kernel, "nt"),
        (grouped_gemm_bf16_variable_k_flydsl_kernel, "vk"),
    ],
    ids=["nt", "nn", "vk"],
)
def test_group_members_default_to_the_sentinel(entry, names):
    """Every member defaults to -1 in the signature, and -1 is the only value that means "unset".

    ``persist=0`` has to mean off rather than unset, so the group cannot use 0 as its sentinel; if
    a member's default drifted back to its off value the entry point would read it as hand-set and
    the table would never fire for any caller.
    """
    group = gg._STACK_VK if names == "vk" else gg._STACK_NT
    params = inspect.signature(entry).parameters
    for knob in group:
        assert knob in params, f"{entry.__name__}: {knob} is in the group but not in the signature"
        assert params[knob].default == -1, (
            f"{entry.__name__}: {knob} defaults to {params[knob].default}, not the -1 sentinel"
        )
        assert knob in gg._STACK_OFF, f"{knob} has no entry in _STACK_OFF to fall back to"
    off = gg._stack_off(group, tuple(-1 for _ in group))
    assert off == {n: gg._STACK_OFF[n] for n in group}
    passed_through = gg._stack_off(group, tuple(7 for _ in group))
    assert passed_through == {n: 7 for n in group}, "a hand-set value must reach the launcher as given"


# ---------------------------------------------------------------------------
# Host-only: the geometry gates.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NT_SHAPES))
def test_nt_cluster_gate_refuses_a_cluster_the_tile_space_cannot_cut(name):
    """A multicast cluster has to cut the N blocks, the M group and the XCD band into whole pieces.

    Both forward shapes run the 320-wide tile, which leaves 13 and 23 N blocks -- odd, so the
    4x2 cluster that is otherwise the better one on paper is refused here. That is the gate
    carrying a hardware limitation (a partial cluster has peers that never request the same
    block), not a tuning choice, so it is pinned on both shapes.
    """
    N, K = NT_SHAPES[name]
    tile = gg._pick_config(N, K, KEYED_AVG_M, 32)
    group_m, num_xcd, xcd_band = gg._TILE_ORDER_NT[(N, K, KEYED_AVG_M)]
    n_blocks_n = (N + tile[1] - 1) // tile[1]
    stack = _keyed_group("nt", N, K)

    assert n_blocks_n % 2 == 1, f"{name}: N blocks is {n_blocks_n}; this case needs an odd count"
    assert gg._nt_stack_cluster_ok(stack, n_blocks_n, group_m, num_xcd, xcd_band), (
        f"{name}: the shipped {stack['mc_m']}x{stack['mc_n']} cluster is refused on its own shape"
    )
    for mc_n in (2, 4):
        assert not gg._nt_stack_cluster_ok(dict(stack, mc_n=mc_n), n_blocks_n, group_m, num_xcd, xcd_band), (
            f"{name}: a 4x{mc_n} cluster was admitted on {n_blocks_n} N blocks"
        )
    assert not gg._nt_stack_cluster_ok(stack, n_blocks_n, 2, num_xcd, xcd_band), (
        f"{name}: GROUP_M=2 was admitted under a {stack['mc_m']}-deep cluster"
    )
    assert not gg._nt_stack_cluster_ok(stack, n_blocks_n, group_m, num_xcd, 18), (
        f"{name}: an XCD band of 18 was admitted under a cluster of {stack['mc_m'] * stack['mc_n']}"
    )


@pytest.mark.parametrize("name", sorted(VK_SHAPES))
def test_vk_cluster_gate_needs_whole_persistent_rounds(name):
    """The variable-K gate also asks for whole persistent rounds, so it is a function of G.

    ``persist=256`` is one workgroup per CU and the launcher walks the tile space in rounds of
    that size, so a tile count that is not a whole number of rounds would leave a partial round
    whose clusters have absent peers. With 256 experts' worth of tiles per round this refuses
    small expert counts: the fc1 shape needs G >= 4 and the fc2 shape G >= 8. Those are the
    shapes that take the fallback inside the entry point, so this gate is what makes that path
    reachable at all.
    """
    out_m, out_n = VK_SHAPES[name]
    tile = gg._pick_variable_k_config(out_m, out_n, KEYED_AVG_M, True)
    stack = _keyed_group("vk", out_m, out_n)
    n_blocks_n = (out_m + tile[0] - 1) // tile[0]
    n_blocks_k = (out_n + tile[1] - 1) // tile[1]
    group_m = _VK_GROUP_M

    for G in (8, 16, 32, 64):
        tiles = G * n_blocks_n * n_blocks_k
        assert gg._vk_stack_cluster_ok(stack, n_blocks_n, n_blocks_k, tiles, group_m), (
            f"{name}: G={G} gives {tiles} tiles, which the shipped cluster refuses"
        )
    for G in (1, 2):
        tiles = G * n_blocks_n * n_blocks_k
        assert not gg._vk_stack_cluster_ok(stack, n_blocks_n, n_blocks_k, tiles, group_m), (
            f"{name}: G={G} gives {tiles} tiles, which is not a whole number of "
            f"{stack['persist']}-tile rounds, yet the gate admits it"
        )
    assert not gg._vk_stack_cluster_ok(stack, n_blocks_n, n_blocks_k, stack["persist"], group_m), (
        f"{name}: a single persistent round was admitted; the walk needs at least two"
    )


@pytest.mark.parametrize(
    "kind, name, a, b, tile", [d for d in DELIVERY if d[0] != "vk"], ids=[i for i in _IDS if "vk" not in i]
)
def test_head_pipe_gate_matches_the_schedule_it_rides(kind, name, a, b, tile):
    """``head_pipe`` is admitted only on the schedule it was written against.

    It reads sub-step 0 of the next k-tile during the current one, which needs the split-barrier
    schedule (``lds_cosched >= 2``, ``split_bar == 1``, depth 1), the two-deep ring it counts
    buffers against, no K tail and no dead half-tile -- and the chain has to be short enough that
    its reads fit behind the matrix ops of the warp tile. Each of those is asserted by taking the
    shipped value and breaking one condition at a time, so the gate cannot quietly widen.

    Only the NT and NN entry points; the variable-K one runs a four-deep ring, so this gate would
    refuse its tile outright. It states its own preconditions as a raise instead -- see below.
    """
    lds_cosched = gg._pick_lds_cosched(kind, a, b, KEYED_AVG_M, True)
    split_bar, k_tail = 1, b % tile[2]
    head_pipe = _keyed_group(kind, a, b)["head_pipe"]
    assert head_pipe, f"{kind} {name}: the table no longer names a head_pipe chain"
    assert lds_cosched >= 2, f"{kind} {name}: lds_cosched resolves to {lds_cosched} inside the group"
    assert k_tail == 0, f"{kind} {name}: K={b} now leaves a tail of {k_tail} against tile_k={tile[2]}"

    assert gg._head_pipe_ok(head_pipe, tile, lds_cosched, split_bar, 1, k_tail, 0), (
        f"{kind} {name}: head_pipe={head_pipe} is refused on its own tile {tile}"
    )
    wm, wn = tile[0] // tile[3] // 16, tile[1] // tile[4] // 16
    over = wm * wn - 2 * (wm + wn) + 1
    assert not gg._head_pipe_ok(over, tile, lds_cosched, split_bar, 1, k_tail, 0), (
        f"{kind} {name}: a chain of {over} fits {wm}x{wn} warp tiles, which leaves no matrix ops "
        f"for its reads to hide behind"
    )
    assert not gg._head_pipe_ok(head_pipe, tile, 1, split_bar, 1, k_tail, 0)
    assert not gg._head_pipe_ok(head_pipe, tile, lds_cosched, 0, 1, k_tail, 0)
    assert not gg._head_pipe_ok(head_pipe, tile, lds_cosched, split_bar, 2, k_tail, 0)
    assert not gg._head_pipe_ok(head_pipe, tile, lds_cosched, split_bar, 1, 32, 0)
    assert not gg._head_pipe_ok(head_pipe, tile, lds_cosched, split_bar, 1, k_tail, 1)
    assert not gg._head_pipe_ok(head_pipe, (*tile[:5], 3), lds_cosched, split_bar, 1, k_tail, 0)


@pytest.mark.parametrize("name", sorted(VK_SHAPES))
def test_variable_k_group_resolves_the_schedule_head_pipe_needs(name):
    """The variable-K group has to resolve the schedule its own ``head_pipe`` demands.

    The launcher refuses ``head_pipe`` unless ``lds_cosched`` is 2 and ``split_bar`` is 1, and it
    refuses it by raising rather than by dropping the knob, so a table row whose schedule does not
    line up is a ``ValueError`` at the call rather than a slow kernel. Both come from tables keyed
    on the same shape, which is exactly the kind of agreement that drifts.
    """
    out_m, out_n = VK_SHAPES[name]
    cfg = gg._pick_variable_k_config(out_m, out_n, KEYED_AVG_M, True)
    assert gg._pick_lds_cosched("vk", out_m, out_n, KEYED_AVG_M, True) == 2
    assert gg._vk_auto_split_bar(cfg, out_m, out_n, KEYED_AVG_M) == 1
    assert _keyed_group("vk", out_m, out_n)["head_pipe"], f"{name}: the table names no head_pipe chain"


def test_a_variable_k_key_at_512_rows_would_be_a_dead_row():
    """Not every row count the knobs help at can be reached by adding a key, and 512 is one.

    The variable-K lookup requires ``split_bar == 1``, and ``split_bar`` resolves from the number
    of steady k windows an expert's rows give the ring: ``_VK_SPLIT_BAR_MIN_WINDOWS`` of them. A
    new key brings the deep ring with it (the ring depth is keyed on the same shape), and on the
    deep ring 512 rows leave 5 windows, below the threshold -- so the group would be refused with
    no diagnostic and the row would read as tuned while resolving to the untuned path forever,
    which is worse than no row at all. 1024 leaves 13 and clears it.

    That is why the measured +34.5% at 1024 rows is reachable by a key and the measured +34.8% at
    512 rows is not: reaching the latter needs the threshold or a keyed ``split_bar``, not a row.
    """
    deep = gg._pick_variable_k_config(*VK_SHAPES["wgrad_fc1"], KEYED_AVG_M, True)
    assert deep[5] == 4, "this case is about the ring a new key would bring with it"
    for name, (out_m, out_n) in VK_SHAPES.items():
        for avg_m, wanted in ((512, 0), (1024, 1), (KEYED_AVG_M, 1), (4096, 1)):
            got = gg._vk_auto_split_bar(deep, out_m, out_n, avg_m)
            assert got == wanted, (
                f"{name} avg_m={avg_m}: on the deep ring split_bar resolves to {got}, expected "
                f"{wanted} from {gg._vk_windows(deep, avg_m)} windows against a threshold of "
                f"{gg._VK_SPLIT_BAR_MIN_WINDOWS}"
            )
    # And no other row count is a key of its own: read at its own row count the picker hands out the shallow
    # tile. 512 to 8192 rows reach the deep ring only through the 2048 key (`_shape_avg_m`), and below 833 rows
    # the group then refuses itself on the window count above.
    for name, (out_m, out_n) in VK_SHAPES.items():
        for avg_m in (512, 1024, 4096):
            assert (out_m, out_n, avg_m) not in gg._VK_DEEP_RING_KEYS
            assert gg._pick_variable_k_config(out_m, out_n, avg_m, True) == (256, 256, 128, 2, 2, 2), (
                f"{name} avg_m={avg_m}: the fallback tile changed; its window count, and so the "
                f"split_bar the untuned path resolves, changes with it"
            )


# ---------------------------------------------------------------------------
# Host-only: what a caller who switches the stack off is left with.
# ---------------------------------------------------------------------------


def test_stack_off_resolves_to_the_pre_group_values():
    """Off the group, the three gated knobs must read exactly what they read before it existed.

    Three values were first written straight into the shared tables, where they also applied to
    callers who had switched the stack off. All three are tuned for the stack and are a cost
    without it, and all three were measured as such on the stack-off path at avg_m=2048:
    the forward fc2 epilogue chunk at -0.86%, and the two wgrad knobs together at -0.57% and
    -0.44% (the ``lds_cosched`` half of the pair is the expensive one on its own, -0.80% and
    -0.89%; the deeper ring recovers about half of that and is +0.09% to -0.27% alone). The
    numbers are what they are because the stack's ``head_pipe``, ``tdm_deal`` bit 1 and
    ``vk_ovl`` bit 1 all ride the split-barrier schedule that ``lds_cosched=2`` sets up, and none
    of them is present off the group.

    So the shared tables hold the pre-group values and the override tables hold the tuned ones.
    Asserted as literals rather than by comparing the two tables, because the point is the
    specific numbers that were measured, not that the two differ.
    """
    nt_fc2 = (*NT_SHAPES["fwd_fc2"], KEYED_AVG_M)
    assert gg._pick_epi_chunk("nt", *nt_fc2, False) == 4, "forward fc2 lost its ungrouped epi_chunk"
    assert gg._pick_epi_chunk("nt", *nt_fc2, True) == 1, "forward fc2 lost its grouped epi_chunk"
    nt_fc1 = (*NT_SHAPES["fwd_fc1"], KEYED_AVG_M)
    assert gg._pick_epi_chunk("nt", *nt_fc1, False) == gg._pick_epi_chunk("nt", *nt_fc1, True) == 1, (
        "forward fc1 never had an epi_chunk key; a difference here is a new, unmeasured one"
    )

    for name, (out_m, out_n), ungrouped in (
        ("wgrad_fc1", VK_SHAPES["wgrad_fc1"], 1),
        ("wgrad_fc2", VK_SHAPES["wgrad_fc2"], 0),
    ):
        key = (out_m, out_n, KEYED_AVG_M)
        assert gg._pick_lds_cosched("vk", *key, False) == ungrouped, (
            f"{name}: the ungrouped lds_cosched is no longer {ungrouped}"
        )
        assert gg._pick_lds_cosched("vk", *key, True) == 2, f"{name}: the grouped lds_cosched is not 2"
        assert gg._pick_variable_k_config(out_m, out_n, KEYED_AVG_M, False)[5] == 3
        assert gg._pick_variable_k_config(out_m, out_n, KEYED_AVG_M, True)[5] == 4

    for name, (N, K) in NN_SHAPES.items():
        key = (N, K, KEYED_AVG_M)
        assert gg._pick_lds_cosched("nn", *key, False) == gg._pick_lds_cosched("nn", *key, True) == 3
        assert gg._pick_epi_chunk("nn", *key, False) == gg._pick_epi_chunk("nn", *key, True) == 1


def test_the_nn_entry_point_cannot_read_a_stack_override():
    """The two override tables must stay empty for "nn", because nothing threads ``stack`` there.

    ``_pick_epi_chunk`` and ``_pick_lds_cosched`` take the group's verdict as an argument, and the
    NT and variable-K entry points pass it; the NN entry point does not, because its
    ``lds_cosched`` is read by ``_head_pipe_ok`` before the group is resolved. An "nn" row added
    to either table today would therefore be read as if the group had never fired and would never
    take effect -- a silent no-op rather than an error. There is no bug while both are empty, so
    this is the guard that keeps it that way.
    """
    assert gg._STACK_EPI_CHUNK["nn"] == {}, (
        "an nn epi_chunk override cannot take effect: wire `stack` through the NN entry point first"
    )
    assert gg._STACK_LDS_COSCHED["nn"] == {}, (
        "an nn lds_cosched override cannot take effect: wire `stack` through the NN entry point first"
    )
    for table in (gg._STACK_EPI_CHUNK, gg._STACK_LDS_COSCHED):
        for kind, rows in table.items():
            base = gg._EPI_CHUNK_KEYS if table is gg._STACK_EPI_CHUNK else gg._LDS_COSCHED_KEYS
            for key in rows:
                assert key in gg._STACK_KEYS[kind], (
                    f"{kind} {key}: an override on a shape the group never fires on is dead weight"
                )
                assert base[kind].get(key, 0 if table is gg._STACK_LDS_COSCHED else 1) != rows[key], (
                    f"{kind} {key}: the override equals the ungrouped value, so it does nothing"
                )


# ---------------------------------------------------------------------------
# Host-only: row counts near the key.
# ---------------------------------------------------------------------------

# Under expert parallelism the rows a GPU receives vary per step, so avg_m lands near 2048, not on it. Every table
# below is read at the avg_m `_shape_avg_m` resolves: 2048 within a factor _AVG_M_REACH of it (512-8192 rows),
# the call's own avg_m, which no table holds, outside.
NEAR = (512, 1024, 2047, 2049, 4096, 8192)
FAR = (256, 511, 8193, 16384)
# Every shape table the entry points read, written out so that a new table has to be added here and to the
# module's index together.
SHAPE_TABLES = {
    "nt": (
        "_ISSUE_FIRST_KEYS",
        "_EPI_CHUNK_KEYS",
        "_STACK_EPI_CHUNK",
        "_LDS_COSCHED_KEYS",
        "_STACK_LDS_COSCHED",
        "_STACK_KEYS",
        "_FWD_WIDE_N_KEYS",
        "_TILE_ORDER_NT",
        "_FWD_TDM_DEAL_KEYS",
    ),
    "nn": (
        "_ISSUE_FIRST_KEYS",
        "_EPI_CHUNK_KEYS",
        "_STACK_EPI_CHUNK",
        "_WMMA_B2B_KEYS",
        "_LDS_COSCHED_KEYS",
        "_STACK_LDS_COSCHED",
        "_STACK_KEYS",
        "_TILE_ORDER_NN",
        "_TDM_NN_KEYS",
        "_TDM_ASSIGN_NN_KEYS",
    ),
    "vk": (
        "_EPI_CHUNK_KEYS",
        "_STACK_EPI_CHUNK",
        "_WMMA_B2B_KEYS",
        "_LDS_COSCHED_KEYS",
        "_STACK_LDS_COSCHED",
        "_STACK_KEYS",
        "_VK_DEEP_RING_KEYS",
    ),
}
# The tables above that serve one kind only, keyed (N, K, avg_m) directly rather than per kind.
_ONE_KIND = frozenset(
    {
        "_FWD_WIDE_N_KEYS",
        "_TILE_ORDER_NT",
        "_FWD_TDM_DEAL_KEYS",
        "_TILE_ORDER_NN",
        "_TDM_NN_KEYS",
        "_TDM_ASSIGN_NN_KEYS",
        "_VK_DEEP_RING_KEYS",
    }
)


def _is_shape_key(row) -> bool:
    return isinstance(row, tuple) and len(row) == 3 and all(isinstance(v, int) for v in row)


def _rows(name: str, kind: str):
    table = getattr(gg, name)
    return set(table) if name in _ONE_KIND else set(table.get(kind, ()))


def test_every_shape_table_is_indexed():
    """`_shape_avg_m` reads one index of the (N, K, avg_m) rows of every shape table, and no table is missing from it.

    A table left out would be read at the call's own avg_m and miss near the key -- what every one of them did
    before -- and only the speed would show it. The module is scanned for tables of (N, K, avg_m) rows so that a
    new one cannot be added without being listed here.
    """
    for kind, names in SHAPE_TABLES.items():
        want = {}
        for name in names:
            for n, k, avg_m in _rows(name, kind):
                want.setdefault((n, k), set()).add(avg_m)
        assert {nk: set(ms) for nk, ms in gg._SHAPE_KEY_AVG_M[kind].items()} == want, kind
    listed = {name for names in SHAPE_TABLES.values() for name in names}
    for name, table in vars(gg).items():
        if name == "_KWARGS_KEYS" or not isinstance(table, (Mapping, frozenset, set)):
            continue
        if table and all(kind in ("nt", "nn", "vk") for kind in table):
            rows = [row for per_kind in table.values() for row in per_kind]
        else:
            rows = list(table)
        if rows and all(_is_shape_key(row) for row in rows):
            assert name in listed, f"{name} holds (N, K, avg_m) rows but is not read through _shape_avg_m"


@pytest.mark.parametrize("kind, name, a, b, tile", DELIVERY, ids=_IDS)
def test_row_counts_near_the_key_read_the_tables_at_it(kind, name, a, b, tile):
    """Every avg_m within a factor _AVG_M_REACH of 2048 reads the tables at 2048, and nothing further away does."""
    for avg_m in range(1, 8193):
        got = gg._shape_avg_m(kind, a, b, avg_m)
        near = max(avg_m, KEYED_AVG_M) / min(avg_m, KEYED_AVG_M) <= gg._AVG_M_REACH
        assert got == (KEYED_AVG_M if near else avg_m), avg_m
    assert [gg._shape_avg_m(kind, a, b, m) for m in NEAR] == [KEYED_AVG_M] * len(NEAR)
    assert [gg._shape_avg_m(kind, a, b, m) for m in FAR] == list(FAR)


def test_shapes_without_a_key_read_the_tables_at_their_own_avg_m():
    """MiniMax-M3, Muse Spark and any other (N, K): no shape table holds them, so nothing changes for them here."""
    for kind, n, k in (("nt", 6144, 6144), ("nn", 4096, 8192), ("vk", 8192, 4096), ("nt", 7168, 7168)):
        assert [gg._shape_avg_m(kind, n, k, m) for m in (363, 512, 1449, 2048)] == [363, 512, 1449, 2048]


def test_exact_keys_switch_reads_the_tables_at_the_call_s_own_avg_m(monkeypatch):
    """PRIMUS_TURBO_GROUPED_GEMM_EXACT_KEYS=1: only 2048 itself is keyed, and the cached pickers follow the switch."""
    out_m, out_n = VK_SHAPES["wgrad_fc1"]
    near = gg._pick_variable_k_config(out_m, out_n, 1700, True, gg._shape_avg_m("vk", out_m, out_n, 1700))
    assert near == (256, 256, 64, 2, 2, 4)
    monkeypatch.setenv("PRIMUS_TURBO_GROUPED_GEMM_EXACT_KEYS", "1")
    envs.reset_cache()
    for kind, name, a, b, tile in DELIVERY:
        assert [gg._shape_avg_m(kind, a, b, m) for m in NEAR] == list(NEAR), name
        assert gg._shape_avg_m(kind, a, b, KEYED_AVG_M) == KEYED_AVG_M, name
    exact = gg._pick_variable_k_config(out_m, out_n, 1700, True, gg._shape_avg_m("vk", out_m, out_n, 1700))
    assert exact == (256, 256, 128, 2, 2, 2)


@pytest.mark.parametrize("kind, name, a, b, tile", DELIVERY, ids=_IDS)
def test_near_the_key_every_lookup_answers_as_at_the_key(kind, name, a, b, tile):
    """Tile, ring depth, tile order, dealing, schedule, store chunk and the stack itself, knob by knob."""

    def answers(avg_m, G):
        key_m = gg._shape_avg_m(kind, a, b, avg_m)
        got = {
            "tile": _picked_tile(kind, a, b, avg_m, G),
            "epi_chunk": gg._pick_epi_chunk(kind, a, b, key_m, True),
        }
        got["epi_chunk_off"] = gg._pick_epi_chunk(kind, a, b, key_m, False)
        got["lds_cosched"] = gg._pick_lds_cosched(kind, a, b, key_m, True)
        got["lds_cosched_off"] = gg._pick_lds_cosched(kind, a, b, key_m, False)
        names = gg._STACK_VK if kind == "vk" else gg._STACK_NT
        got["stack"] = gg._pick_stack(kind, a, b, key_m, names, tuple(-1 for _ in names), got["tile"], True)
        if kind in ("nt", "nn"):
            got["issue_first"] = gg._pick_issue_first(kind, a, b, key_m)
            order = gg._TILE_ORDER_NT if kind == "nt" else gg._TILE_ORDER_NN
            got["order"] = order.get((a, b, key_m))
        if kind == "nt":
            got["tdm_deal"] = (a, b, key_m) in gg._FWD_TDM_DEAL_KEYS
        if kind == "nn":
            got["tdm_nn"] = (gg._pick_tdm_nn(a, b, key_m), gg._pick_tdm_assign_nn(a, b, key_m))
        if kind in ("nn", "vk"):
            got["wmma_b2b"] = gg._pick_wmma_b2b(kind, a, b, key_m)
        return got

    for G in (8, 16, 32, 64):
        keyed = answers(KEYED_AVG_M, G)
        assert keyed["tile"] == tile and keyed["stack"] is not None
        for avg_m in NEAR:
            assert answers(avg_m, G) == keyed, (G, avg_m)
        for avg_m in FAR:
            off = answers(avg_m, G)
            assert off["stack"] is None and off["lds_cosched"] == off["lds_cosched_off"] == 0, (G, avg_m)
            if kind != "nn":  # the NN tile at 2048 is the untuned one: no table keys it
                assert off["tile"] != tile, (G, avg_m)
    if kind == "vk":
        # The variable-K group needs split_bar, which the real row count decides: 512 rows leave the deep ring 5
        # k windows, below the threshold, so there the group refuses itself and the rest of the tables still apply.
        got = [gg._vk_auto_split_bar(tile, a, b, avg_m) for avg_m in NEAR]
        assert got == [0, 1, 1, 1, 1, 1], got


def _launched(kind: str, a: int, b: int, avg_m: int, G: int, monkeypatch) -> tuple:
    """The compile-time arguments the public entry point launches with for balanced rows, on CPU tensors.

    The launch itself is replaced by a recorder, so this runs everything the entry point resolves -- the picker, every
    table, the group and its gates -- without a device. Dropped from the record: the runtime arguments (pointers,
    stream, strides) and, on NT / NN, the M tile count, which follows the row count rather than the key.
    """
    seen = []
    monkeypatch.setattr(gg, "_require_gfx1250", lambda device: None)
    monkeypatch.setattr(gg, "_as_ptr", lambda t: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: 0)
    monkeypatch.setattr(gg, "_nt_launch", lambda args: seen.append(args[11:]))
    monkeypatch.setattr(gg, "_vk_launch", lambda args: seen.append(args[7:]))
    M, bf16 = G * avg_m, torch.bfloat16
    offs = _group_offs(G, avg_m, "cpu")
    if kind == "nt":
        grouped_gemm_bf16_nt_flydsl_kernel(
            torch.empty(M, b, dtype=bf16), torch.empty(G, a, b, dtype=bf16), offs
        )
    elif kind == "nn":
        grouped_gemm_bf16_nn_flydsl_kernel(
            torch.empty(M, b, dtype=bf16), torch.empty(G, b, a, dtype=bf16), offs
        )
    else:
        grouped_gemm_bf16_variable_k_flydsl_kernel(
            torch.empty(M, a, dtype=bf16), torch.empty(M, b, dtype=bf16), offs
        )
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize("kind, name, a, b, tile", DELIVERY, ids=_IDS)
def test_near_the_key_the_entry_point_launches_the_keyed_kernel(kind, name, a, b, tile, monkeypatch):
    """At both ends of what 2048 serves the entry point launches exactly the kernel the keyed row count launches --
    except the variable-K group at 512 rows, which refuses itself there (see the lookup test above) -- and one row
    past them what PRIMUS_TURBO_GROUPED_GEMM_EXACT_KEYS=1 launches at the ends themselves."""
    G = NUMERIC_G
    keyed = _launched(kind, a, b, KEYED_AVG_M, G, monkeypatch)
    below, above = _launched(kind, a, b, 511, G, monkeypatch), _launched(kind, a, b, 8193, G, monkeypatch)
    assert keyed not in (below, above)
    low = _launched(kind, a, b, 512, G, monkeypatch)
    assert (low == keyed) == (kind != "vk") and low != below
    for avg_m in (1024, 8192):
        assert _launched(kind, a, b, avg_m, G, monkeypatch) == keyed, avg_m
    monkeypatch.setenv("PRIMUS_TURBO_GROUPED_GEMM_EXACT_KEYS", "1")
    envs.reset_cache()
    assert _launched(kind, a, b, 512, G, monkeypatch) == below
    assert _launched(kind, a, b, 8192, G, monkeypatch) == above
    assert _launched(kind, a, b, KEYED_AVG_M, G, monkeypatch) == keyed


# ---------------------------------------------------------------------------
# Numeric: the six delivery shapes, on a gfx1250 card.
# ---------------------------------------------------------------------------

# G is the smallest expert count both variable-K shapes clear the persistent-round gate at, which
# keeps the fp32 host reference affordable: avg_m, not M_total, is what the key is read on.
NUMERIC_G = 8
NUMERIC_M = NUMERIC_G * KEYED_AVG_M


def _group_offs(G: int, avg_m: int, device) -> torch.Tensor:
    """Balanced row runs: ``[0, avg_m, 2*avg_m, ...]``, which is what avg_m means."""
    return torch.arange(0, (G + 1) * avg_m, avg_m, dtype=torch.int64, device=device)


def _assert_matches(got: torch.Tensor, ref: torch.Tensor, dtype) -> None:
    """The repository's two-part comparison: SNR floor, then element-wise tolerances.

    ``tests.pytorch.test_utils`` reaches ``primus_turbo.pytorch``, so it is imported here rather
    than at module scope, for the reason spelled out in ``_have_gfx1250``.
    """
    from tests.pytorch.test_utils import compute_snr, get_tolerances

    snr = compute_snr(ref, got)
    assert snr > 45, f"out SNR {snr:.2f} dB is below the bf16 threshold of 45 dB"
    torch.testing.assert_close(got, ref, **get_tolerances(dtype))


def _ref_rows(a, b, offs, trans_b: bool):
    """fp32-accumulated per-expert matmul of the row runs, on the **CPU**.

    Deliberately not a device matmul: hipBLASLt has no Tensile library for gfx1250 in every image
    this runs in, so a device reference would turn a kernel test into a hipBLASLt availability
    test. Same reason the MLA suite keeps its reference on the host.
    """
    bounds = offs.cpu().tolist()
    a32, b32 = a.detach().float().cpu(), b.detach().float().cpu()
    out = []
    for g in range(len(bounds) - 1):
        rhs = b32[g].t() if trans_b else b32[g]
        out.append((a32[bounds[g] : bounds[g + 1]] @ rhs).to(a.dtype))
    return torch.cat(out)


def _ref_variable_k(a, b, offs):
    """fp32-accumulated ``out[g] = a[rows_g].T @ b[rows_g]`` on the CPU, same reasoning as above."""
    bounds = offs.cpu().tolist()
    a32, b32 = a.detach().float().cpu(), b.detach().float().cpu()
    out = []
    for g in range(len(bounds) - 1):
        lo, hi = bounds[g], bounds[g + 1]
        out.append((a32[lo:hi].t() @ b32[lo:hi]).to(a.dtype))
    return torch.stack(out)


def _call_four_times(entry, *args, **kwargs):
    """Four calls, checking the fourth.

    The NN entry's own docstring records wrong results on the first one or two calls of a fresh
    process, root cause not located and predating the tuning, so a correctness harness has to burn
    those calls before it reads anything. Applied to all three entry points rather than only the
    NN one: the defect is not known to be specific to it.
    """
    for _ in range(3):
        entry(*args, **kwargs)
    return entry(*args, **kwargs)


def _assert_keyed_path(kind: str, a: int, b: int, G: int) -> None:
    """Fail unless this call will really take the keyed path.

    Every condition the entry point checks before resolving the group that does not need a device
    tensor: the key is present, the picker hands out the tile the key was tuned on, and both
    geometry gates admit the cluster. Without this a numeric case could pass while silently
    running the fallback, and the table would be untested on hardware.
    """
    tile = _picked_tile(kind, a, b, KEYED_AVG_M, G)
    assert tile == gg._STACK_KEYS[kind][(a, b, KEYED_AVG_M)]["cfg"]
    stack = _keyed_group(kind, a, b)
    lds_cosched = gg._pick_lds_cosched(kind, a, b, KEYED_AVG_M, True)
    if kind == "vk":
        n_blocks_n = (a + tile[0] - 1) // tile[0]
        n_blocks_k = (b + tile[1] - 1) // tile[1]
        group_m = _VK_GROUP_M
        assert gg._vk_stack_cluster_ok(stack, n_blocks_n, n_blocks_k, G * n_blocks_n * n_blocks_k, group_m)
        assert lds_cosched == 2 and gg._vk_auto_split_bar(tile, a, b, KEYED_AVG_M) == 1
    else:
        order = gg._TILE_ORDER_NT if kind == "nt" else gg._TILE_ORDER_NN
        group_m, num_xcd, xcd_band = order[(a, b, KEYED_AVG_M)]
        assert gg._nt_stack_cluster_ok(stack, (a + tile[1] - 1) // tile[1], group_m, num_xcd, xcd_band)
        assert gg._head_pipe_ok(stack["head_pipe"], tile, lds_cosched, 1, 1, b % tile[2], 0)


@needs_gfx1250
@pytest.mark.parametrize("name", sorted(NT_SHAPES), ids=sorted(NT_SHAPES))
def test_nt_delivery_shape_numeric(name):
    """The forward of a DeepSeek-V3 MoE layer, on the shape the key table is written for."""
    N, K = NT_SHAPES[name]
    _assert_keyed_path("nt", N, K, NUMERIC_G)
    torch.manual_seed(0)
    a = torch.randn(NUMERIC_M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(NUMERIC_G, N, K, dtype=torch.bfloat16, device="cuda")
    offs = _group_offs(NUMERIC_G, KEYED_AVG_M, "cuda")

    out = _call_four_times(grouped_gemm_bf16_nt_flydsl_kernel, a, b, offs)
    _assert_matches(out.detach().cpu(), _ref_rows(a, b, offs, trans_b=True), a.dtype)


@needs_gfx1250
@pytest.mark.parametrize("name", sorted(NN_SHAPES), ids=sorted(NN_SHAPES))
def test_nn_delivery_shape_numeric(name):
    """The input gradient of the same layer: the NN orientation of the two forward shapes."""
    N, K = NN_SHAPES[name]
    _assert_keyed_path("nn", N, K, NUMERIC_G)
    torch.manual_seed(0)
    a = torch.randn(NUMERIC_M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(NUMERIC_G, K, N, dtype=torch.bfloat16, device="cuda")
    offs = _group_offs(NUMERIC_G, KEYED_AVG_M, "cuda")

    out = _call_four_times(grouped_gemm_bf16_nn_flydsl_kernel, a, b, offs)
    _assert_matches(out.detach().cpu(), _ref_rows(a, b, offs, trans_b=False), a.dtype)


@needs_gfx1250
@pytest.mark.parametrize("name", sorted(VK_SHAPES), ids=sorted(VK_SHAPES))
def test_variable_k_delivery_shape_numeric(name):
    """The weight gradient of the same layer, which reduces along the ragged row axis."""
    out_m, out_n = VK_SHAPES[name]
    _assert_keyed_path("vk", out_m, out_n, NUMERIC_G)
    torch.manual_seed(0)
    a = torch.randn(NUMERIC_M, out_m, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(NUMERIC_M, out_n, dtype=torch.bfloat16, device="cuda")
    offs = _group_offs(NUMERIC_G, KEYED_AVG_M, "cuda")

    out = _call_four_times(grouped_gemm_bf16_variable_k_flydsl_kernel, a, b, offs)
    _assert_matches(out.detach().cpu(), _ref_variable_k(a, b, offs), a.dtype)


@needs_gfx1250
@pytest.mark.parametrize("knob", ["persist", "mc_m", "head_pipe"])
def test_switching_the_stack_off_still_computes_the_same_answer(knob):
    """A caller who switches the group off gets a slower kernel, not a different one.

    One shape per entry point is enough here: the fallback inside the variable-K entry point hands
    back ``lds_cosched``, the ring depth and ``split_bar`` after the gates have refused, and those
    three decide the LDS arena and the barrier schedule -- so an incomplete hand-back shows up as
    a wrong answer or a launch failure, not as a slowdown.
    """
    off = {knob: gg._STACK_OFF[knob]}
    torch.manual_seed(0)
    N, K = NT_SHAPES["fwd_fc2"]
    a = torch.randn(NUMERIC_M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(NUMERIC_G, N, K, dtype=torch.bfloat16, device="cuda")
    offs = _group_offs(NUMERIC_G, KEYED_AVG_M, "cuda")
    keyed = _call_four_times(grouped_gemm_bf16_nt_flydsl_kernel, a, b, offs)
    plain = _call_four_times(grouped_gemm_bf16_nt_flydsl_kernel, a, b, offs, **off)
    _assert_matches(plain.detach().cpu(), _ref_rows(a, b, offs, trans_b=True), a.dtype)
    torch.testing.assert_close(plain, keyed, rtol=0, atol=0)


@needs_gfx1250
def test_variable_k_stack_off_still_computes_the_same_answer():
    """The variable-K half of the test above, where the hand-back actually has work to do."""
    out_m, out_n = VK_SHAPES["wgrad_fc1"]
    torch.manual_seed(0)
    a = torch.randn(NUMERIC_M, out_m, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(NUMERIC_M, out_n, dtype=torch.bfloat16, device="cuda")
    offs = _group_offs(NUMERIC_G, KEYED_AVG_M, "cuda")
    keyed = _call_four_times(grouped_gemm_bf16_variable_k_flydsl_kernel, a, b, offs)
    ref = _ref_variable_k(a, b, offs)
    for knob in ("persist", "mc_m", "head_pipe"):
        plain = _call_four_times(
            grouped_gemm_bf16_variable_k_flydsl_kernel, a, b, offs, **{knob: gg._STACK_OFF[knob]}
        )
        _assert_matches(plain.detach().cpu(), ref, a.dtype)
        torch.testing.assert_close(plain, keyed, rtol=0, atol=0)


@needs_gfx1250
def test_ragged_rows_do_not_depend_on_the_stack():
    """An unbalanced routing has to answer the same whether the group fired or not.

    The cluster's peer test and the persistent walk both read the per-expert row spans, so an
    empty expert, a single-row expert and a run that does not fill a tile are where a cluster
    that shares the wrong block shows up. The balanced cases above cannot reach any of that.

    The row runs still add up to ``NUMERIC_M`` over ``NUMERIC_G`` experts, because the key is read
    on ``M_total // G``: redistributing the rows keeps the keyed path and changes only the spans.
    """
    N, K = NT_SHAPES["fwd_fc2"]
    rows = [0, 1, 40, 255, KEYED_AVG_M, KEYED_AVG_M + 7, 4 * KEYED_AVG_M]
    rows += [NUMERIC_M - sum(rows)]
    assert len(rows) == NUMERIC_G and min(rows) >= 0 and sum(rows) == NUMERIC_M
    _assert_keyed_path("nt", N, K, NUMERIC_G)
    offs = torch.tensor([0, *itertools.accumulate(rows)], dtype=torch.int64, device="cuda")
    torch.manual_seed(0)
    a = torch.randn(NUMERIC_M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(NUMERIC_G, N, K, dtype=torch.bfloat16, device="cuda")

    keyed = _call_four_times(grouped_gemm_bf16_nt_flydsl_kernel, a, b, offs)
    plain = _call_four_times(grouped_gemm_bf16_nt_flydsl_kernel, a, b, offs, persist=0)
    ref = _ref_rows(a, b, offs, trans_b=True)
    _assert_matches(keyed.detach().cpu(), ref, a.dtype)
    torch.testing.assert_close(plain, keyed, rtol=0, atol=0)


@needs_gfx1250
@pytest.mark.parametrize("avg_m", [512, 1024])
@pytest.mark.parametrize("kind, name, a, b, tile", DELIVERY, ids=_IDS)
def test_near_the_key_numeric(kind, name, a, b, tile, avg_m, monkeypatch):
    """Uneven rows at M_total // G = 512, the low end of what 2048 serves (where the variable-K group refuses itself
    and the rest of its tables still apply), and at 1024 (where it fires): right against the fp32 reference, and bit
    for bit the untuned call (PRIMUS_TURBO_GROUPED_GEMM_EXACT_KEYS=1), since every keyed value moves a schedule only.
    """
    rows = [0, 1, 40, 255, avg_m + 7, 2 * avg_m, 3 * avg_m]
    rows += [NUMERIC_G * avg_m - sum(rows)]
    assert len(rows) == NUMERIC_G and min(rows) >= 0
    assert gg._shape_avg_m(kind, a, b, avg_m) == KEYED_AVG_M
    _assert_keyed_path(kind, a, b, NUMERIC_G)
    offs = torch.tensor([0, *itertools.accumulate(rows)], dtype=torch.int64, device="cuda")
    M = sum(rows)
    torch.manual_seed(0)
    if kind == "nt":
        x = torch.randn(M, b, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(NUMERIC_G, a, b, dtype=torch.bfloat16, device="cuda")
        entry, ref = grouped_gemm_bf16_nt_flydsl_kernel, lambda: _ref_rows(x, w, offs, trans_b=True)
    elif kind == "nn":
        x = torch.randn(M, b, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(NUMERIC_G, b, a, dtype=torch.bfloat16, device="cuda")
        entry, ref = grouped_gemm_bf16_nn_flydsl_kernel, lambda: _ref_rows(x, w, offs, trans_b=False)
    else:
        x = torch.randn(M, a, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(M, b, dtype=torch.bfloat16, device="cuda")
        entry, ref = grouped_gemm_bf16_variable_k_flydsl_kernel, lambda: _ref_variable_k(x, w, offs)

    keyed = _call_four_times(entry, x, w, offs)
    monkeypatch.setenv("PRIMUS_TURBO_GROUPED_GEMM_EXACT_KEYS", "1")
    envs.reset_cache()
    plain = _call_four_times(entry, x, w, offs)
    _assert_matches(keyed.detach().cpu(), ref(), x.dtype)
    torch.testing.assert_close(keyed, plain, rtol=0, atol=0)
