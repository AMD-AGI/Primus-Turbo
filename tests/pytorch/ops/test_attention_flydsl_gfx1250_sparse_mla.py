###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Index helpers of the gfx1250 (MI455X) FlyDSL sparse-MLA backward's dKV gather.

The operator itself (every backend, forward and backward) is covered by ``test_sparse_mla_op``
in ``test_attention.py``.
"""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "n_tok, topk, num_kv, chunk", [(5, 64, 7, 2), (1, 8, 3, 1536), (4096, 576, 1025, 1536)]
)
def test_sparse_mla_inverted_topk_chunks(n_tok, topk, num_kv, chunk):
    """Joint CSR inverse of the dkv gather: row pointers equal ``searchsorted(keys, arange(...))``
    and every chunk's slice equals the per-chunk inverse (rows, keys and stable entry order)."""
    from primus_turbo.flydsl.attention.gfx1250_sparse_mla.bwd_kernel import (
        _inverted_topk,
        _inverted_topk_chunks,
    )

    gen = torch.Generator(device="cuda").manual_seed(n_tok)
    idx = torch.randint(-1, num_kv, (n_tok, topk), generator=gen, device="cuda", dtype=torch.int32)
    keys, inv_ptr, inv_data = _inverted_topk_chunks(idx, num_kv, chunk)
    n_chunks = -(-n_tok // chunk)
    bounds = torch.arange(n_chunks * num_kv + 1, device="cuda", dtype=keys.dtype)
    assert torch.equal(inv_ptr, torch.searchsorted(keys, bounds, out_int32=True))
    for c in range(n_chunks):
        t0, t1 = c * chunk, min(n_tok, (c + 1) * chunk)
        k_ref, p_ref, d_ref = _inverted_topk(idx[t0:t1], num_kv)
        ptr = inv_ptr[c * num_kv : (c + 1) * num_kv + 1].long()
        lo, hi = int(ptr[0]), int(ptr[-1])
        assert torch.equal(ptr - lo, (p_ref - p_ref[0]).long())
        assert torch.equal(keys[lo:hi].long() - c * num_kv, k_ref[int(p_ref[0]) :].long())
        assert torch.equal(inv_data[lo:hi] - t0 * topk, d_ref[int(p_ref[0]) :])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("n_tok, topk, num_kv, chunk", [(5, 64, 7, 2), (4096, 576, 1025, 1536)])
def test_sparse_mla_chunk_keys(n_tok, topk, num_kv, chunk):
    """Composite dkv-gather sort keys: ``chunk * num_kv + kv row`` per valid entry, -1 otherwise."""
    from primus_turbo.triton.attention.sparse_mla_dkv_gather import chunk_keys

    gen = torch.Generator(device="cuda").manual_seed(n_tok)
    idx = torch.randint(-1, num_kv, (n_tok, topk), generator=gen, device="cuda", dtype=torch.int32)
    cid = torch.arange(n_tok, device="cuda").div(chunk, rounding_mode="floor").unsqueeze(1)
    ref = torch.where(idx >= 0, idx + cid * num_kv, -1).to(torch.int16).reshape(-1)
    assert torch.equal(chunk_keys(idx, num_kv, chunk, torch.int16), ref)
