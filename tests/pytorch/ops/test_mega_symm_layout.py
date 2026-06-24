###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for ``primus_turbo.flydsl.mega.symm_layout`` (mega-MoE two-heap SymLayout).

  A. layout: pool sizing matches the DeepGEMM ``get_num_max_pool_tokens`` formula,
     and every sub-buffer packs to an independently-recomputed 256B-aligned offset
     (the layout contract, no GPU needed), incl. the prod DSv3 shape.
  B. device: single-GPU, two side-by-side workspaces per heap; write a marker
     through every ``get_*_ptr`` (local + ``dst=peer``) and verify byte offset +
     peer translation, for both the cached main heap and the uncached signal heap.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_symm_layout.py
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch
from flydsl.expr import Int32, Int64
from flydsl.expr.buffer_ops import buffer_store

from primus_turbo.flydsl.mega import symm_layout as SLM
from primus_turbo.flydsl.mega.prims import addr_buffer_resource

# (world, num_experts, num_tokens, num_topk, hidden, intermediate)
_SHAPES = [
    (2, 8, 16, 4, 128, 256),
    (8, 32, 4096, 8, 256, 256),
    (8, 384, 8192, 6, 7168, 3072),  # prod DSv3
]

_ALIGN = 256
_BF16, _I32, _F32, _I64 = 2, 4, 4, 8


def _align(x, a=_ALIGN):
    return (x + a - 1) // a * a


def _expected_regions(shape):
    """Independent (name, heap, itemsize, numel) contract, mirrored from symm_layout."""
    world, E, T, K, H, I = shape
    epr = E // world
    nmpt = SLM.get_num_max_pool_tokens(world, T, K, epr)
    npb = nmpt // 8
    cs = K * T  # combine_slots
    return (
        nmpt,
        npb,
        [
            ("pool", "main", _BF16, nmpt * H),
            ("c_buffer", "main", _I32, world * E),
            ("signal", "main", _I32, world),
            ("origin_rank", "main", _I32, nmpt),
            ("origin_slot", "main", _I32, nmpt),
            ("weight_recv_buf", "main", _F32, nmpt),
            ("combine_gate", "main", _F32, cs),
            ("meta_scalars", "main", _I32, 8),
            ("grid_barrier_state", "main", _I32, 2),
            ("profile", "main", _I64, 8),
            ("act", "main", _BF16, nmpt * I),
            ("l2_token_buffer", "main", _BF16, nmpt * H),
            ("_ipc_barrier", "signal", _I32, world),
            ("scoreboard", "signal", _I32, npb),
            ("sb_consume", "signal", _I32, npb),
            ("sb_l2", "signal", _I32, npb),
            ("comb", "signal", _BF16, cs * H),
            ("barrier_local", "signal", _I32, cs),
        ],
    )


def _pack(regions):
    """Reference 256B-aligned per-heap packer -> (offsets{name}, totals{heap})."""
    offsets, cursors = {}, {"main": 0, "signal": 0}
    for name, heap, item, numel in regions:
        off = _align(cursors[heap])
        offsets[name] = off
        cursors[heap] = off + numel * item
    return offsets, {h: _align(c) for h, c in cursors.items()}


# ---------------------------------------------------------------------------
# A. layout contract
# ---------------------------------------------------------------------------
def _check_layout(shape):
    world, E, T, K, H, I = shape
    nmpt, npb, regions = _expected_regions(shape)

    # DeepGEMM pool formula
    assert nmpt % SLM._LCM_BLOCK_M == 0, "pool not LCM-aligned"
    assert npb == nmpt // SLM._MIN_BLOCK_M

    shape8 = (world, E, T, K, H, I, nmpt, npb)
    entries, totals = SLM.layout_spec(*shape8)
    exp_off, exp_tot = _pack(regions)
    main_b, sig_b = SLM.num_bytes(*shape8)
    assert (main_b, sig_b) == (exp_tot["main"], exp_tot["signal"]), "heap totals mismatch"

    for name, heap, item, numel in regions:
        e = entries[name]
        assert e.heap == heap, f"{name} heap {e.heap} != {heap}"
        assert e.offset == exp_off[name], f"{name} offset {e.offset} != {exp_off[name]}"
        assert e.itemsize == item and e.numel == numel, f"{name} dtype/numel mismatch"
        assert e.offset % _ALIGN == 0, f"{name} not 256B aligned"
    print(f"[layout] shape={shape} nmpt={nmpt} npb={npb} main={main_b}B signal={sig_b}B PASS")


# ---------------------------------------------------------------------------
# B. device offsets + peer map
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _build(dims_key):
    @flyc.kernel(known_block_size=[1, 1, 1])
    def k(MAIN: fx.Tensor, SIG: fx.Tensor, sl: SLM.SymLayout):
        def w(addr, val):
            buffer_store(fx.Int32(val), addr_buffer_resource(addr, num_records_bytes=4), 0)

        # cached main heap: local markers + one peer-translated marker
        w(SLM.get_pool_ptr(sl, index=5), 11)
        w(SLM.get_c_buffer_ptr(sl, index=3), 12)
        w(SLM.get_origin_slot_ptr(sl, index=7), 13)
        w(SLM.get_combine_gate_ptr(sl, index=2), 14)
        w(SLM.get_l2_token_ptr(sl, index=1), 15)
        w(SLM.get_pool_ptr(sl, dst_rank=fx.Int32(1), index=5), 111)  # -> peer main pool[5]
        # uncached signal heap
        w(SLM.get_scoreboard_ptr(sl, index=4), 21)
        w(SLM.get_comb_ptr(sl, index=6), 22)
        w(SLM.get_barrier_local_ptr(sl, index=3), 23)
        w(SLM.get_scoreboard_ptr(sl, dst_rank=fx.Int32(1), index=4), 121)  # -> peer signal scoreboard[4]

    @flyc.jit
    def launch(MAIN, SIG, sl, stream: fx.Stream = fx.Stream(None)):
        k(MAIN, SIG, sl).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    return launch


def _check_device(shape):
    world, E, T, K, H, I = shape
    nmpt = SLM.get_num_max_pool_tokens(world, T, K, E // world)
    npb = nmpt // SLM._MIN_BLOCK_M
    shape8 = (world, E, T, K, H, I, nmpt, npb)
    entries, _ = SLM.layout_spec(*shape8)
    nmain, nsig = SLM.num_bytes(*shape8)

    main = torch.zeros(2 * nmain, dtype=torch.int8, device="cuda")  # two ranks side by side
    sig = torch.zeros(2 * nsig, dtype=torch.int8, device="cuda")
    main_deltas = torch.tensor([0, nmain], dtype=torch.int64, device="cuda")
    sig_deltas = torch.tensor([0, nsig], dtype=torch.int64, device="cuda")
    sl = SLM.SymLayout(
        buffer_base=Int64(main.data_ptr()),
        buffer_offsets_ptr=Int64(main_deltas.data_ptr()),
        signal_base=Int64(sig.data_ptr()),
        signal_offsets_ptr=Int64(sig_deltas.data_ptr()),
        rank_idx=Int32(0),
        num_ranks=world,
        num_experts=E,
        num_max_tokens_per_rank=T,
        num_topk=K,
        hidden=H,
        intermediate_hidden=I,
        num_max_pool_tokens=nmpt,
        num_max_pool_blocks=npb,
    )
    _build(shape8)(main, sig, sl, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    # markers (<2^16) stored as i32 at a possibly 2-aligned byte offset (bf16 buffers);
    # read the low 16 bits via an int16 view at byte_off//2 so alignment never bites.
    m16, s16 = main.view(torch.int16), sig.view(torch.int16)

    def off(name):
        return entries[name].offset

    checks = [
        (m16, off("pool") + 5 * _BF16, 11),
        (m16, off("c_buffer") + 3 * _I32, 12),
        (m16, off("origin_slot") + 7 * _I32, 13),
        (m16, off("combine_gate") + 2 * _F32, 14),
        (m16, off("l2_token_buffer") + 1 * _BF16, 15),
        (m16, nmain + off("pool") + 5 * _BF16, 111),  # peer main pool[5]
        (s16, off("scoreboard") + 4 * _I32, 21),
        (s16, off("comb") + 6 * _BF16, 22),
        (s16, off("barrier_local") + 3 * _I32, 23),
        (s16, nsig + off("scoreboard") + 4 * _I32, 121),  # peer signal scoreboard[4]
    ]
    ok = True
    for buf, byte_off, val in checks:
        got = int(buf[byte_off // 2].item())
        if got != val:
            ok = False
            print(f"  MISMATCH @byte {byte_off}: got {got} want {val}")
    assert ok, "get_*_ptr offset/map mismatch"
    print(f"[device] shape={shape} all get_*_ptr + peer map land correctly PASS")


@pytest.mark.parametrize("shape", _SHAPES)
def test_layout(shape):
    _check_layout(shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU (run inside dev_primus)")
def test_device():
    torch.cuda.set_device(0)
    _check_device(_SHAPES[0])
    _check_device(_SHAPES[1])


def main():
    for s in _SHAPES:
        _check_layout(s)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        _check_device(_SHAPES[0])
        _check_device(_SHAPES[1])
    print("[symm_layout] ALL PASS")


if __name__ == "__main__":
    main()
