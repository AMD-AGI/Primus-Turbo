###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Single-GPU tests for ``primus_turbo.flydsl.mega.sym_layout.SymLayout``.

No multi-rank needed: one int8 arena holds two "ranks'" workspaces side by side.
We (A) sanity-check the host layout sizes against an independent re-derivation of
the mega_moe.cuh formulas, (B) write a distinct marker through every ``get_*_ptr``
and verify it lands at the independently-computed byte offset, and (C) check
``map(ptr, dst)`` translates a local ptr into the peer workspace.

Run inside dev_primus:
  PYTHONPATH=<...>/Primus-Turbo python tests/pytorch/ops/test_mega_sym_layout.py
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch
from flydsl.expr.buffer_ops import buffer_store

from primus_turbo.flydsl.mega import sym_layout as SL
from primus_turbo.flydsl.mega.prims import addr_buffer_resource

# small but valid config
_R, _E, _T, _K = 2, 8, 16, 4


# ---- independent re-derivation of the cuh byte offsets (NOT importing SL internals) ----
def _ref_offsets(sl):
    E = int(sl.num_experts)
    Epr = int(sl.num_experts_per_rank)
    NPB = int(sl.num_max_pool_blocks)
    R = int(sl.num_ranks)
    Trecv = R * int(sl.num_max_tokens_per_rank)
    a2 = ((NPB + 1) // 2) * 2
    off = {}
    off["barrier"] = 0
    off["send"] = 32
    off["recv"] = 32 + E * 8
    off["recv_sum"] = 32 + E * 16
    off["l1"] = 32 + (E * 2 + Epr) * 8
    off["l2"] = off["l1"] + a2 * 4
    off["src"] = off["l2"] + NPB * 8
    off["meta"] = off["src"] + Epr * R * Trecv * 4
    total = off["meta"] + int(sl.num_max_pool_tokens) * 12
    total = ((total + 15) // 16) * 16
    return off, total, Trecv


@functools.lru_cache(maxsize=None)
def _build(shape_key):
    @flyc.kernel(known_block_size=[1, 1, 1])
    def k(ARENA: fx.Tensor, sl: SL.SymLayout):
        def w(addr, val):
            buffer_store(fx.Int32(val), addr_buffer_resource(addr, num_records_bytes=4), 0)

        # one marker per sub-buffer accessor, at chosen indices (rank 0 == self)
        w(SL.get_grid_sync_count_ptr(sl, 2), 1001)
        w(SL.get_nvl_barrier_counter_ptr(sl), 1002)
        w(SL.get_nvl_barrier_signal_ptr(sl, 1), 1003)
        w(SL.get_expert_send_count_ptr(sl, 3), 1004)
        w(SL.get_expert_recv_count_ptr(sl, 1, 2), 1005)
        w(SL.get_expert_recv_count_sum_ptr(sl, 1), 1006)
        w(SL.get_l1_arrival_count_ptr(sl, 5), 1007)
        w(SL.get_l2_arrival_mask_ptr(sl, 3), 1008)
        w(SL.get_src_token_topk_idx_ptr(sl, 1, 1, 2), 1009)
        w(SL.get_token_src_metadata_ptr(sl, 4), 1010)
        # sym_map: write through a peer-translated grid-sync ptr (dst rank 1)
        w(SL.sym_map(sl, SL.get_grid_sync_count_ptr(sl, 0), fx.Int32(1)), 2001)

    @flyc.jit
    def launch(ARENA, sl, stream: fx.Stream = fx.Stream(None)):
        k(ARENA, sl).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    return launch


def _run():
    sl0 = SL.build_sym_layout(_R, _E, _T, _K)  # base=0, just for sizing & shapes
    nbytes = SL.get_num_bytes(sl0)
    off, ref_total, Trecv = _ref_offsets(sl0)

    # ---- A: host layout sanity vs independent formulas ----
    assert nbytes == ref_total, f"get_num_bytes {nbytes} != ref {ref_total}"
    assert int(sl0.num_experts_per_rank) == _E // _R
    print(f"[sym_layout host] num_bytes={nbytes} (matches cuh formula) PASS")

    # ---- B + C: device offsets ----
    arena = torch.zeros(2 * nbytes, dtype=torch.int8, device="cuda")  # two ranks side by side
    base0 = arena.data_ptr()
    offsets = torch.tensor([0, nbytes], dtype=torch.int64, device="cuda")  # self, peer delta
    sl = SL.build_sym_layout(_R, _E, _T, _K, base=base0, offsets_ptr=offsets.data_ptr(), rank_idx=0)
    _build((_R, _E, _T, _K))(arena, sl, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    a32 = arena.view(torch.int32)  # index by byte_off // 4
    Epr = int(sl.num_experts_per_rank)
    expect = {
        off["barrier"] + 2 * 4: 1001,  # grid_sync[2]
        16: 1002,  # nvl counter
        20 + 1 * 4: 1003,  # nvl signal[1]
        off["send"] + 3 * 8: 1004,  # expert_send[3]
        off["recv"] + (1 * Epr + 2) * 8: 1005,  # expert_recv[1,2]
        off["recv_sum"] + 1 * 8: 1006,  # recv_sum[1]
        off["l1"] + 5 * 4: 1007,  # l1[5]
        off["l2"] + 3 * 8: 1008,  # l2[3]
        off["src"] + (1 * _R * Trecv + 1 * Trecv + 2) * 4: 1009,  # src[1,1,2]
        off["meta"] + 4 * 12: 1010,  # meta[4]
    }
    ok = True
    for byte_off, val in expect.items():
        got = int(a32[byte_off // 4].item())
        if got != val:
            ok = False
            print(f"  MISMATCH @byte {byte_off}: got {got} want {val}")
    assert ok, "get_*_ptr offset mismatch"
    print("[sym_layout device] all get_*_ptr land at the correct byte offsets PASS")

    # map check: marker 2001 written into peer region (offset nbytes + grid_sync[0]=0)
    peer0 = int(arena.view(torch.int32)[nbytes // 4].item())
    assert peer0 == 2001, f"map() peer write landed wrong: {peer0}"
    print("[sym_layout device] map(ptr, dst) translated into peer workspace PASS")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU (run inside dev_primus)")
def test_sym_layout():
    torch.cuda.set_device(0)
    _run()


if __name__ == "__main__":
    torch.cuda.set_device(0)
    _run()
    print("[sym_layout] ALL PASS")
