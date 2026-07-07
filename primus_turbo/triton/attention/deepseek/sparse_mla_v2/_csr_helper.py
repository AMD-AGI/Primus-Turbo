###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CSR inverted-topk builder for the non-atomic chunked-gather backward.

Vendored (unchanged) from the upstream Primus ``_gluon_dsa/_dsa_bwd_gather.py``
so the ``triton_v2`` sparse-MLA backward is self-contained in Primus-Turbo.
"""

import torch


def _build_inverted_topk_slice(topk_indices_slice, r_start, R_CHUNK, num_kv=None):
    """Build a CSR-style inverted index for a topk slice, excluding invalid (-1).

    Args:
        topk_indices_slice: [T, R_CHUNK] int32 — ``topk_indices[:, r_start:r_start+R_CHUNK]``.
          May contain -1 padding (last chunk shorter than R_CHUNK).
        r_start:  int — first rank index in this slice (documentation only).
        R_CHUNK:  int — number of ranks in this slice (constexpr width).
        num_kv:   int or None — number of KV tokens. When the KV buffer holds MORE
          rows than query tokens (V4 ``[local ++ pool]``, ``num_kv = S + P``), pass
          it so ``inv_ptr`` has length ``num_kv + 1``. Defaults to ``T``.

    Returns:
        inv_ptr:  [num_kv+1] int32 — CSR row pointers (kv_token -> range in inv_data).
        inv_data: [T*R_CHUNK] int32 — flat indices ``q*R_CHUNK + local_r`` sorted by KV
          token; invalid (-1) entries sort to the front and are skipped by inv_ptr[0].
    """
    T, RC = topk_indices_slice.shape
    n_kv = T if num_kv is None else int(num_kv)
    flat_kv = topk_indices_slice.reshape(-1).long()  # [T*R_CHUNK]; -1 marks invalid

    inv_data = torch.argsort(flat_kv, stable=True).to(torch.int32)  # [T*R_CHUNK]
    counts = torch.bincount(flat_kv + 1, minlength=n_kv + 1)  # [num_kv+1]; bin0 = #invalid
    inv_ptr = torch.cumsum(counts, dim=0).to(torch.int32)  # [num_kv+1]; inv_ptr[0]=#invalid

    return inv_ptr, inv_data
