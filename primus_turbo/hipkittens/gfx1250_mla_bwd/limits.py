###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shape rules of the HIP MLA backward kernels (pure Python, no torch: the adapter's gate imports it).

k_dkdv96ds tiles kv into 96-row blocks of two 48-row waves, then 64-row tail blocks; k_dqds_c tiles
queries into 384-query workgroups over [q_off, Sq) plus one merged 256-query head over [0, q_off).
The dS^T scratch between them is dslayout.h v1: per (batch, q head) region, kv16 row kt holds the
1 KiB blocks of the 32-query pairs that attend it (bottom-right causal: the dead ones are skipped).
"""

from typing import Optional

K3_ROWS = 96  # k_dkdv96ds kv rows per (two-wave) workgroup
DQ_MAIN_ROWS = 384  # k_dqds_c queries per workgroup
DQ_HEAD_ROWS = 256  # k_dqds_c merged head tile
DSL_BLK_B = 1024  # dslayout v1 block: 16 kv rows x 32 queries, bf16
# The largest extent the kernels' index proofs replay (b * heads * seqlen of [1, 4096, 512, D]).
MAX_BH_S = 512 * 4096


def k3_n96(skv: int) -> int:
    """96-row blocks: Skv // 96, one fewer when Skv % 96 == 32 (that remainder runs as two 64-row tails)."""
    n, r = skv // K3_ROWS, skv % K3_ROWS
    return n - 1 if r == 32 else n


def k3_n64(skv: int) -> int:
    """64-row tail blocks after the 96-row blocks: 0 / 1 / 2 for Skv % 96 == 0 / 64 / 32; -1 = unsupported."""
    return {0: 0, 64: 1, 32: 2}.get(skv % K3_ROWS, -1)


def k3_ok(skv: int) -> bool:
    return skv > 0 and k3_n64(skv) >= 0 and k3_n96(skv) >= 0


def k3_nblk(skv: int) -> int:
    return k3_n96(skv) + k3_n64(skv)


def dq_split(sq: int):
    """(q_off, nhead, ntile): 384-query tiles over [q_off, sq), the merged 256-query head over [0, q_off)."""
    q_off = sq % DQ_MAIN_ROWS
    if q_off not in (0, DQ_HEAD_ROWS):
        raise NotImplementedError(f"seqlen_q {sq}: seqlen_q % {DQ_MAIN_ROWS} must be 0 or {DQ_HEAD_ROWS}")
    return q_off, (1 if q_off else 0), (sq - q_off) // DQ_MAIN_ROWS


def ds_layout_bytes(b: int, sq: int, skv: int, hq: int, causal: bool) -> int:
    """dslayout.h dsl_total_bytes: b*hq regions of dsl_nblk_bh 1 KiB blocks (packed causal)."""
    nqp, nkt, m = sq // 32, skv // 16, (skv - sq) // 32
    if causal:

        def tri(k):
            p = k // 2
            return p * (p - 1) + (k % 2) * p

        skip = (tri(nkt) - m * nkt) if m <= 0 else (tri(nkt - 2 * m) if nkt > 2 * m else 0)
    else:
        skip = 0
    return b * hq * (nkt * nqp - skip) * DSL_BLK_B


def shape_unsupported_reason(b: int, sq: int, skv: int, hq: int, hkv: int, causal: bool) -> Optional[str]:
    """Why the HIP kernels cannot take this problem, beyond the FlyDSL MLA gate's rules, or None."""
    if hq != hkv:
        return f"MHA only (heads_q == heads_kv), got {hq} / {hkv}"
    if skv < sq:
        return f"seqlen_kv {skv} < seqlen_q {sq}"
    if sq % 32 or skv % 32:
        return f"seqlen_q / seqlen_kv must be multiples of 32, got {sq} / {skv}"
    if sq % DQ_MAIN_ROWS not in (0, DQ_HEAD_ROWS):
        return f"seqlen_q % {DQ_MAIN_ROWS} must be 0 or {DQ_HEAD_ROWS}, got {sq}"
    if not k3_ok(skv):
        return f"seqlen_kv % {K3_ROWS} must be 0, 32 or 64, got {skv}"
    if b * hq * max(sq, skv) > MAX_BH_S:
        return f"batch*heads*seqlen {b * hq * max(sq, skv)} exceeds the proven extent {MAX_BH_S}"
    return None
