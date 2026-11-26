###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""Static configs of the gfx1250 MXFP8 GEMMs (``gemm_mxfp8_kernel_gfx1250``): dense per (M, N, K), the routed
experts' grouped-M and variable-K modes per shape class. Measured on MI455X (DSv3 shapes); no runtime autotune."""

from typing import NamedTuple

DENSE, GROUPED_M, VARIABLE_K = 0, 1, 2  # the GEMM modes
_GRID_CUS = 256  # MI455X: the CU count of pick_config's grid-fill rule


class MXGemmConfig(NamedTuple):
    """One kernel instance: a ``tile_m x tile_n`` block per workgroup of 2x2 waves, ``num_buffers`` LDS slots of
    ``kpair`` 128-wide K steps each, and a ``cluster_m x cluster_n`` workgroup cluster that multicasts the shared A
    (along N) and B (along M) tiles (1x1: none). ``persist``: persistent workgroups, which load the next tile during
    the stores (beta = 1 into bf16: the old C's first rows too); it needs 2 slots of kpair 2 (else the call
    raises), and a call with fp32 out, or beta = 1 into partial N tiles, runs non-persistent."""

    tile_m: int = 256
    tile_n: int = 256
    num_buffers: int = 2
    kpair: int = 2
    cluster_m: int = 2
    cluster_n: int = 2
    persist: bool = False


C = MXGemmConfig
# (M, N, K) -> the fastest config measured where it is not ``pick_config``'s grid-fill rule. The table holds the
# benchmark shapes only, DSv3 at T = 8192 tokens: fwd (T, N, K), dgrad (T, K, N), wgrad (N, K, T); any other token
# count takes the rule's pick. The mainloop is bound by the global -> LDS traffic, most of it A's: the deep kpair-1
# pipeline and the larger multicast clusters win on the long K.
_TABLE: dict = {
    # MLA projections
    (1536, 7168, 8192): C(256, 256, 4, 1, 2, 2),  # q_down wgrad
    (8192, 24576, 1536): C(256, 256, 2, 2, 4, 1, persist=True),  # q_up fwd (many tiles, short K: persistent)
    (8192, 576, 7168): C(256, 128, 3, 2, 1, 1),  # kv_down fwd
    (576, 7168, 8192): C(128, 256, 3, 2, 1, 4),  # kv_down wgrad
    (8192, 32768, 512): C(256, 256, 2, 2, 4, 1, persist=True),  # kv_up fwd (many tiles, short K: persistent)
    (8192, 512, 32768): C(128, 128, 4, 2, 2, 2),  # kv_up dgrad
    (32768, 512, 8192): C(256, 256, 2, 2, 2, 2),  # kv_up wgrad
    (8192, 7168, 16384): C(256, 256, 4, 1, 2, 4),  # o_proj fwd
    (8192, 16384, 7168): C(256, 256, 4, 1, 4, 4),  # o_proj dgrad
    (7168, 16384, 8192): C(256, 256, 2, 2, 4, 1),  # o_proj wgrad
    # shared expert
    (8192, 4096, 7168): C(256, 256, 4, 1, 4, 4),  # fc1 fwd
    (4096, 7168, 8192): C(256, 256, 2, 2, 2, 2),  # fc1 wgrad
    (8192, 2048, 7168): C(256, 256, 4, 1, 4, 4),  # fc2 dgrad
    (7168, 2048, 8192): C(256, 256, 4, 1, 1, 4),  # fc2 wgrad
    # shared expert at T = 2048 tokens
    (2048, 7168, 4096): C(256, 256, 2, 2, 2, 2),  # fc1 dgrad
    (2048, 2048, 7168): C(128, 128, 4, 2, 2, 2),  # fc2 dgrad
    (7168, 2048, 2048): C(256, 256, 2, 2, 4, 1),  # fc2 wgrad
}

# beta = 1 into a bf16 out (a wgrad into main_grad), where the measured best is not ``pick_config``'s rule
_TABLE_ACC: dict = {
    (24576, 1536, 8192): C(256, 256, 2, 2, 1, 4, persist=True),  # q_up wgrad
    (7168, 16384, 8192): C(256, 256, 2, 2, 2, 2, persist=True),  # o_proj wgrad
}


def pick_config(M: int, N: int, K: int, beta1: bool = False) -> MXGemmConfig:
    """The dense config of (M, N, K): the measured one, else by grid fill: 256x256 while it gives at least ~3/4 of
    a wave (the deep kpair-1 pipeline on a long K), then 128x256, then 128x128 (two workgroups per CU). The entry
    point shrinks the cluster to what the grid divides. ``beta1`` (accumulating into a bf16 out): the old C fetched
    early wants kpair 2, so a kpair-1 pick becomes the persistent c2x2."""
    tiles = lambda tm, tn: -(-M // tm) * -(-N // tn)  # noqa: E731
    cfg = (beta1 and _TABLE_ACC.get((M, N, K))) or _TABLE.get((M, N, K))
    if cfg is None and tiles(256, 256) >= 3 * _GRID_CUS // 4:
        cfg = C(256, 256, 4, 1, 2, 2) if K >= 4096 else C()
    elif cfg is None:
        cfg = C(128, 256, 3, 2, 1, 4) if tiles(128, 256) >= 3 * _GRID_CUS // 4 else C(128, 128, 2, 2, 1, 1)
    return C(256, 256, 2, 2, 2, 2, persist=True) if beta1 and cfg.kpair == 1 else cfg


def pick_grouped_config(mode: int, N: int, K: int, bf16_out: bool = True) -> MXGemmConfig:
    """Routed experts. ``GROUPED_M`` (fwd / dgrad, ``N`` x ``K`` per expert): from K = 2048 the deep kpair-1
    pipeline in clusters of 2 m x 4 n tiles (B multicast inside an expert), else persistent kpair 2.
    ``VARIABLE_K`` (wgrad into ``[G, M, N]``): persistent, in 2x2 clusters except a narrow N into bf16
    (N <= 2048: 1x1)."""
    if mode == GROUPED_M:
        return C(256, 256, 4, 1, 2, 4) if K >= 2048 else C(256, 256, 2, 2, 1, 4, persist=True)
    if mode != VARIABLE_K:
        raise ValueError(f"pick_grouped_config: mode {mode} is not GROUPED_M or VARIABLE_K")
    c = 1 if N <= 2048 and bf16_out else 2
    return C(256, 256, 2, 2, c, c, persist=True)
