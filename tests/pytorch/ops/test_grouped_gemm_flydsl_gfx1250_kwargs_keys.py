###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The kwargs-mode shape keys of the FlyDSL bf16 grouped GEMM on gfx1250: the six MiniMax-M3 shapes.

``test_grouped_gemm_flydsl_gfx1250_keys.py`` covers ``_STACK_KEYS``, the persistent-cluster group the
DeepSeek-V3 shapes resolve knob by knob. ``_KWARGS_KEYS`` is the other form: per (N, K, avg_m) the
whole set of entry-point arguments measured at that shape, which the three public entry points fill
in for a caller that passes no tuning argument -- as if it had passed them itself, so every gate of
the explicit path still applies. Primus-Turbo's own FlyDSL backend is such a caller.

Two groups:

* Host-only (needs flydsl, no GPU): which shapes are keyed, that any tuning argument switches the key
  off, that the key names only real arguments, the tile, whole-round and expert-count preconditions,
  and that the wrappers keep the entry points' signatures.
* Numeric, on a gfx1250 card: the six MiniMax-M3 shapes at avg_m = 512, called the way the backend
  calls them, against an fp32 host reference and bit for bit against the same call with the key
  switched off -- every keyed value moves a schedule, never an address or an accumulation order --
  and the forward at 64 experts, where the key declines.

Listed in ``tests/conftest.py``'s ``_GFX1250_ENABLED_TESTS``, without which it never runs on gfx1250.
"""

import inspect
import itertools

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
    """A gfx1250 device is visible; see the same helper in test_grouped_gemm_flydsl_gfx1250_keys.py."""
    try:
        from primus_turbo.pytorch.core.utils import is_gfx1250

        return bool(torch.cuda.is_available() and is_gfx1250())
    except Exception:
        return False


needs_gfx1250 = pytest.mark.skipif(not _have_gfx1250(), reason="needs a gfx1250 (MI455X) GPU")

# MiniMax-M3's MoE layer: hidden 6144, gated expert FFN 3072, so fc1 is (N, K) = (6144, 6144) and fc2
# (6144, 3072). NT and variable-K keys take the forward (N, K); NN takes (output N, reduction K).
KEYED_AVG_M = 512
MEASURED_G = 32
SHAPES = (
    ("nt", "fwd_fc1", 6144, 6144),
    ("nt", "fwd_fc2", 6144, 3072),
    ("nn", "dgrad_fc1", 6144, 6144),
    ("nn", "dgrad_fc2", 3072, 6144),
    ("vk", "wgrad_fc1", 6144, 6144),
    ("vk", "wgrad_fc2", 6144, 3072),
)
_IDS = [name for _, name, *_ in SHAPES]
# The tile every key was measured on (the variable-K keys then set BLOCK_K and num_buffers themselves).
MEASURED_TILE = (256, 256, 128, 2, 2, 2)
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
# What a call may pass and still get its key: the output, its dtype and the prebuilt index tables. Written out
# here rather than read from the module, so that widening the module's set fails the tests below.
PASSTHROUGH = {
    "nt": {"out", "out_dtype", "m_tiles"},
    "nn": {"out", "out_dtype", "m_tiles"},
    "vk": {"out", "out_dtype", "masked_k", "expert_table"},
}


def _tuning_args(kind: str):
    """Every keyword the entry point takes that is neither an operand nor a pass-through."""
    params = inspect.signature(_ENTRY[kind]).parameters
    return [n for n in params if n not in _OPERANDS[kind] and n not in PASSTHROUGH[kind]]


def _picked_tile(kind: str, n: int, k: int, avg_m: int, G: int):
    if kind == "nt":
        return gg._pick_config(n, k, avg_m, G)
    if kind == "nn":
        return gg._pick_config_nn(n, k, avg_m, G)
    return gg._pick_variable_k_config(n, k, avg_m, False)


# ---------------------------------------------------------------------------
# Host-only.
# ---------------------------------------------------------------------------


def test_only_the_minimax_m3_shapes_are_keyed():
    """Exactly the six shapes at avg_m = 512; another row count needs its own measurement."""
    want = {
        kind: {(n, k, KEYED_AVG_M) for kd, _, n, k in SHAPES if kd == kind} for kind in ("nt", "nn", "vk")
    }
    got = {kind: set(rows) for kind, rows in gg._KWARGS_KEYS.items()}
    assert got == want


def test_only_outputs_and_index_tables_pass_through():
    assert {kind: set(names) for kind, names in gg._KWARGS_PASSTHROUGH.items()} == PASSTHROUGH


def test_kwargs_keys_and_stack_keys_never_compete():
    """A shape is tuned one way or the other; both lookups on one key would make the result order-dependent."""
    for kind, rows in gg._KWARGS_KEYS.items():
        assert not set(rows) & set(gg._STACK_KEYS.get(kind, {})), kind


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_a_key_names_only_entry_point_arguments(kind, name, n, k):
    """Every knob is a tuning argument of its entry point, with an integer value.

    A knob at its default value stays in the set on purpose: the set is what was measured, and passing
    it is what marks the call as tuned.
    """
    tuning = set(_tuning_args(kind))
    for knob, value in gg._KWARGS_KEYS[kind][(n, k, KEYED_AVG_M)].items():
        assert knob in tuning, f"{name}: {knob} is not a tuning argument of the {kind} entry point"
        assert isinstance(value, int) and not isinstance(value, bool), f"{name}: {knob}={value!r}"


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_an_untuned_call_gets_the_whole_key(kind, name, n, k):
    """No tuning argument -- with or without the pass-through arguments -- gets the key's whole set."""
    key = (n, k, KEYED_AVG_M)
    want = dict(gg._KWARGS_KEYS[kind][key])
    assert gg._keyed_kwargs(kind, key, MEASURED_G, 0, {}) == want
    passthrough = dict.fromkeys(PASSTHROUGH[kind])
    assert gg._keyed_kwargs(kind, key, MEASURED_G, 0, passthrough) == want
    # the positional extras the entry points take: out_dtype, and masked_k first on variable-K
    n_pos = 2 if kind == "vk" else 1
    assert gg._keyed_kwargs(kind, key, MEASURED_G, n_pos, {}) == want
    assert gg._keyed_kwargs(kind, key, MEASURED_G, n_pos + 1, {}) == {}


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_any_tuning_argument_switches_the_key_off(kind, name, n, k):
    """One tuning argument, at any value, and the call gets exactly what it passed and nothing else.

    The members need each other (the launchers refuse several pairs outright), so a partial override
    of the set would launch a combination that was never measured, or none at all.
    """
    key = (n, k, KEYED_AVG_M)
    params = inspect.signature(_ENTRY[kind]).parameters
    for arg in _tuning_args(kind):
        assert gg._keyed_kwargs(kind, key, MEASURED_G, 0, {arg: params[arg].default}) == {}, arg


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_other_row_counts_are_not_keyed(kind, name, n, k):
    for avg_m in (64, 128, 256, 511, 513, 1024, 2048, 4096):
        assert gg._keyed_kwargs(kind, (n, k, avg_m), MEASURED_G, 0, {}) == {}, avg_m


@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_the_picker_still_hands_out_the_measured_tile(kind, name, n, k):
    """The keys were measured on one tile; a picker change would apply them to another one.

    Kept as an assertion rather than left to a benchmark: the answer would stay right, so the only
    symptom would be the speed -- or, for a cluster that no longer cuts the tile grid, a refusal.
    """
    for G in (8, 16, MEASURED_G, 64):
        assert _picked_tile(kind, n, k, KEYED_AVG_M, G) == MEASURED_TILE, G


@pytest.mark.parametrize(
    "kind, name, n, k", [s for s in SHAPES if s[0] == "vk"], ids=["wgrad_fc1", "wgrad_fc2"]
)
def test_variable_k_key_needs_whole_persistent_rounds(kind, name, n, k):
    """The variable-K launcher refuses a persistent grid that leaves a partial round, so the key declines it.

    256 workgroups over G x (n / 256) x (k / 256) tiles: whole rounds of two or more need G >= 4 on
    fc1 (576 tiles per expert) and G >= 8 on fc2 (288). Below that the call runs the untuned path.
    """
    key = (n, k, KEYED_AVG_M)
    persist = gg._KWARGS_KEYS[kind][key]["persist"]
    for G in (1, 2, 4, 8, 16, 32, 64):
        tiles = G * (n // 256) * (k // 256)
        fits = tiles % persist == 0 and tiles // persist >= 2
        assert bool(gg._keyed_kwargs(kind, key, G, 0, {})) == fits, G


@pytest.mark.parametrize(
    "kind, name, n, k", [s for s in SHAPES if s[0] != "vk"], ids=[s[1] for s in SHAPES if s[0] != "vk"]
)
def test_a_vec_dec_key_needs_at_most_32_experts(kind, name, n, k):
    """The NT and NN launchers drop vec_dec above 32 experts, where nt_seam bit 2 then refuses the set.

    So a key with vec_dec (the forward ones) declines there and the call runs the untuned path, e.g.
    MiniMax-M3 at EP2 with MBS 2 (64 experts, 512 rows each). The dgrad keys decode through tt_dec and
    fire at every expert count.
    """
    key = (n, k, KEYED_AVG_M)
    has_vec_dec = bool(gg._KWARGS_KEYS[kind][key].get("vec_dec"))
    assert has_vec_dec == (kind == "nt")
    for G in (8, 16, 32, 33, 64, 128):
        assert bool(gg._keyed_kwargs(kind, key, G, 0, {})) == (G <= 32 or not has_vec_dec), G


def test_the_wrappers_keep_the_entry_point_signatures():
    """Callers and tests read defaults off the public names (``inspect.signature`` follows ``__wrapped__``)."""
    for kind, public in _PUBLIC.items():
        assert inspect.signature(public) == inspect.signature(_ENTRY[kind]), kind
        assert public.__doc__ == _ENTRY[kind].__doc__, kind


# ---------------------------------------------------------------------------
# Numeric: the six MiniMax-M3 shapes, on a gfx1250 card.
# ---------------------------------------------------------------------------

# The smallest expert count at which both variable-K keys fire (see the whole-round test), which keeps
# the fp32 host reference affordable: the key is read on M_total // G, not on M_total.
NUMERIC_G = 8
NUMERIC_M = NUMERIC_G * KEYED_AVG_M


def _offs(rows, device="cuda") -> torch.Tensor:
    return torch.tensor([0, *itertools.accumulate(rows)], dtype=torch.int64, device=device)


def _assert_matches(got: torch.Tensor, ref: torch.Tensor, dtype) -> None:
    """SNR floor, then the element-wise tolerances (imported here, see _have_gfx1250)."""
    from tests.pytorch.test_utils import compute_snr, get_tolerances

    snr = compute_snr(ref, got)
    assert snr > 45, f"out SNR {snr:.2f} dB is below the bf16 threshold of 45 dB"
    torch.testing.assert_close(got, ref, **get_tolerances(dtype))


def _ref(kind, a, b, offs):
    """fp32-accumulated reference on the CPU (no device matmul; see test_grouped_gemm_flydsl_gfx1250_keys.py)."""
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
        args = (
            torch.randn(M, k, dtype=torch.bfloat16, device="cuda"),
            torch.randn(G, n, k, dtype=torch.bfloat16, device="cuda"),
            offs,
        )
        kw = {"out_dtype": torch.bfloat16}
    elif kind == "nn":
        args = (
            torch.randn(M, k, dtype=torch.bfloat16, device="cuda"),
            torch.randn(G, k, n, dtype=torch.bfloat16, device="cuda"),
            offs,
        )
        kw = {"out_dtype": torch.bfloat16}
    else:
        args = (
            torch.randn(M, n, dtype=torch.bfloat16, device="cuda"),
            torch.randn(M, k, dtype=torch.bfloat16, device="cuda"),
            offs,
        )
        kw = {"masked_k": torch.tensor(rows, dtype=torch.int64, device="cuda"), "out_dtype": torch.bfloat16}
    return args, kw


def _call_four_times(entry, *args, **kwargs):
    """Four calls, checking the fourth (the NN entry records wrong results on a fresh process's first calls)."""
    for _ in range(3):
        entry(*args, **kwargs)
    return entry(*args, **kwargs)


def _check(kind, n, k, rows):
    args, kw = _operands(kind, n, k, rows)
    G = len(rows)
    assert gg._keyed_kwargs(kind, (n, k, sum(rows) // G), G, 0, kw), "the call would not take the key"
    keyed = _call_four_times(_PUBLIC[kind], *args, **kw)
    # persist=0 is a tuning argument: the key is off and the call runs the untuned path
    plain = _call_four_times(_PUBLIC[kind], *args, persist=0, **kw)
    _assert_matches(keyed.detach().cpu(), _ref(kind, *args), torch.bfloat16)
    torch.testing.assert_close(keyed, plain, rtol=0, atol=0)


@needs_gfx1250
@pytest.mark.parametrize("kind, name, n, k", SHAPES, ids=_IDS)
def test_keyed_shape_numeric(kind, name, n, k):
    """Balanced routing at the key's row count: right, and bit-identical to the untuned path."""
    _check(kind, n, k, [KEYED_AVG_M] * NUMERIC_G)


@needs_gfx1250
@pytest.mark.parametrize("kind, name, n, k", [SHAPES[1], SHAPES[3], SHAPES[4]], ids=["nt", "nn", "vk"])
def test_keyed_shape_ragged_numeric(kind, name, n, k):
    """An empty expert, a one-row expert and runs that do not fill a tile, still M_total // G = 512.

    The cluster's peer reads and the persistent walk follow the per-expert spans, which the balanced
    case cannot reach; the key still fires because it is read on M_total // G.
    """
    rows = [0, 1, 40, 255, KEYED_AVG_M + 7, 2 * KEYED_AVG_M, 3 * KEYED_AVG_M]
    rows.append(NUMERIC_M - sum(rows))
    assert len(rows) == NUMERIC_G and min(rows) >= 0
    _check(kind, n, k, rows)


@needs_gfx1250
@pytest.mark.parametrize("kind, name, n, k", SHAPES[:2], ids=_IDS[:2])
def test_forward_above_32_experts_runs_untuned(kind, name, n, k):
    """64 experts at the key's row count: the forward key declines (see the host-only test) and the call runs.

    Had it fired, the launcher would refuse the call (nt_seam bit 2 without vec_dec). Bit-identical to the
    untuned path is the check; no host reference, which would cost minutes at 32768 rows.
    """
    G = 64
    args, kw = _operands(kind, n, k, [KEYED_AVG_M] * G)
    assert not gg._keyed_kwargs(kind, (n, k, KEYED_AVG_M), G, 0, kw)
    got = _call_four_times(_PUBLIC[kind], *args, **kw)
    plain = _call_four_times(_PUBLIC[kind], *args, persist=0, **kw)
    torch.testing.assert_close(got, plain, rtol=0, atol=0)
