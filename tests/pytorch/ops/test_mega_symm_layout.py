###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for ``primus_turbo.flydsl.mega.sym_layout_mega`` (mega-MoE SymLayout).

  A. layout parity: the new module's main/signal offsets + totals match
     ``mega_moe_fused``'s ``slice_input_buffers`` / ``signal_spec`` byte-for-byte
     (no GPU needed), incl. the prod DSv3 shape.
  B. device: single-GPU, two side-by-side workspaces per heap; write a marker
     through every ``get_*_ptr`` (local + ``dst=peer``) and verify byte offset +
     peer translation, for both the cached main heap and the uncached signal heap.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_sym_layout_mega.py
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
from primus_turbo.pytorch.ops.moe.mega_moe_fused import _layout as _mmf_layout
from primus_turbo.pytorch.ops.moe.mega_moe_fused import (
    _signal_regions as _mmf_signal_regions,
)
from primus_turbo.pytorch.ops.moe.mega_moe_fused import (
    get_symm_buffer_size_for_mega_moe,
)

# (world, num_experts, num_tokens, num_topk, hidden, intermediate, block_m, pool_mult)
_SHAPES = [
    (2, 8, 16, 4, 128, 256, 256, 2),
    (8, 32, 4096, 8, 256, 256, 256, 2),
    (8, 384, 8192, 6, 7168, 3072, 256, 2),  # prod DSv3
]


# ---------------------------------------------------------------------------
# A. layout parity vs mega_moe_fused
# ---------------------------------------------------------------------------
def _check_parity(shape):
    world, E, T, K, H, I, BM, PM = shape
    dims = SLM.derive_dims(world, E, T, K, H, I, BM, PM)

    num_bytes, slice_input_buffers, signal_bytes, meta = get_symm_buffer_size_for_mega_moe(
        world, E, T, K, H, I, block_m=BM, pool_mult=PM
    )
    # main heap parity
    mine_main = SLM.main_offset_spec(dims)
    assert SLM.main_num_bytes(dims) == num_bytes, f"main total {SLM.main_num_bytes(dims)} != {num_bytes}"
    for name, (off, dtype, numel) in slice_input_buffers.items():
        moff, mitem, mnumel = mine_main[name]
        assert moff == off, f"{name} offset {moff} != {off}"
        assert mitem == dtype.itemsize and mnumel == numel, f"{name} dtype/numel mismatch"

    # signal heap parity (mega_moe_fused builds it inline in SymmBuffer via _signal_regions)
    sig_spec, sig_total = _mmf_layout(
        _mmf_signal_regions(dims["num_pool_blocks"], dims["combine_slots"], H, world)
    )
    mine_sig = SLM.signal_offset_spec(dims)
    assert SLM.signal_num_bytes(dims) == sig_total, "signal total mismatch"
    for name, (off, dtype, numel) in sig_spec.items():
        moff, mitem, mnumel = mine_sig[name]
        assert moff == off, f"signal {name} offset {moff} != {off}"
        assert mitem == dtype.itemsize and mnumel == numel, f"signal {name} dtype/numel mismatch"
    print(f"[parity] shape={shape} main={num_bytes}B signal={sig_total}B PASS")


# ---------------------------------------------------------------------------
# B. device offsets + peer map
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _build(dims_key):
    @flyc.kernel(known_block_size=[1, 1, 1])
    def k(MAIN: fx.Tensor, SIG: fx.Tensor, sl: SLM.SymLayout):
        def w(addr, val):
            buffer_store(fx.Int32(val), addr_buffer_resource(addr, num_records_bytes=4), 0)

        # cached main heap: local marker + peer-translated marker
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
    world, E, T, K, H, I, BM, PM = shape
    dims = SLM.derive_dims(world, E, T, K, H, I, BM, PM)
    nmain, nsig = SLM.main_num_bytes(dims), SLM.signal_num_bytes(dims)
    mlay, slay = SLM.main_offset_spec(dims), SLM.signal_offset_spec(dims)

    main = torch.zeros(2 * nmain, dtype=torch.int8, device="cuda")  # two ranks side by side
    sig = torch.zeros(2 * nsig, dtype=torch.int8, device="cuda")
    main_deltas = torch.tensor([0, nmain], dtype=torch.int64, device="cuda")
    sig_deltas = torch.tensor([0, nsig], dtype=torch.int64, device="cuda")
    sl = SLM.SymLayout(
        main_base=Int64(main.data_ptr()),
        main_delta_ptr=Int64(main_deltas.data_ptr()),
        sig_base=Int64(sig.data_ptr()),
        sig_delta_ptr=Int64(sig_deltas.data_ptr()),
        rank_idx=Int32(0),
        world=world,
        num_experts=E,
        num_tokens=T,
        num_topk=K,
        hidden=H,
        intermediate_hidden=I,
        block_m=BM,
        pool_mult=PM,
    )
    _build(tuple(sorted(dims.items())))(main, sig, sl, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    # markers (<2^16) are stored as i32 at a possibly 2-aligned byte offset (bf16 buffers);
    # read the low 16 bits via an int16 view at byte_off//2 so alignment never bites.
    m16, s16 = main.view(torch.int16), sig.view(torch.int16)

    checks = [
        (m16, mlay["pool"][0] + 5 * 2, 11),
        (m16, mlay["c_buffer"][0] + 3 * 4, 12),
        (m16, mlay["origin_slot"][0] + 7 * 4, 13),
        (m16, mlay["combine_gate"][0] + 2 * 4, 14),
        (m16, mlay["l2_token_buffer"][0] + 1 * 2, 15),
        (m16, nmain + mlay["pool"][0] + 5 * 2, 111),  # peer main pool[5]
        (s16, slay["scoreboard"][0] + 4 * 4, 21),
        (s16, slay["comb"][0] + 6 * 2, 22),
        (s16, slay["barrier_local"][0] + 3 * 4, 23),
        (s16, nsig + slay["scoreboard"][0] + 4 * 4, 121),  # peer signal scoreboard[4]
    ]
    ok = True
    for buf, byte_off, val in checks:
        got = int(buf[byte_off // 2].item())
        if got != val:
            ok = False
            print(f"  MISMATCH @byte {byte_off}: got {got} want {val}")
    assert ok, "get_*_ptr offset/map mismatch"
    print(f"[device] shape={shape} all get_*_ptr + peer map land correctly PASS")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU (run inside dev_primus)")
@pytest.mark.parametrize("shape", _SHAPES)
def test_parity(shape):
    _check_parity(shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU (run inside dev_primus)")
def test_device():
    torch.cuda.set_device(0)
    _check_device(_SHAPES[0])
    _check_device(_SHAPES[1])


def main():
    torch.cuda.set_device(0)
    for s in _SHAPES:
        _check_parity(s)
    _check_device(_SHAPES[0])
    _check_device(_SHAPES[1])
    print("[sym_layout_mega] ALL PASS")


if __name__ == "__main__":
    main()
