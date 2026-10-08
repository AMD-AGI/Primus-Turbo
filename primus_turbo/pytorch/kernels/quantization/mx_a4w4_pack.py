###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MX packers with a selectable output format, for MXFP4 (A4W4) backward GEMMs.

Every MXFP6 packer (plain, fused GELU modes, QK-norm+RoPE backward, AdaLN modulate, gate
multiply) is one kernel template; ``fmt`` picks what each direction emits:

  MX_FMT_FP6       0  the A6W6 blobs, exactly as the ``quantize_mxfp6_*`` ops
  MX_FMT_A4W4_GRAD 1  AITER f4gemm's A operand both ways: a gradient, contracted along its
                      columns by dgrad and along its rows by wgrad
  MX_FMT_A4W4_ACT  2  FP6 rows (the forward's operand) and f4gemm's B operand columns: an
                      activation for wgrad, or a weight for dgrad
  3 / 4               a gradient when only wgrad / only dgrad runs A4W4 (the other direction FP6)
  5 / 6 / 7           1 / 3 / 4 with stochastic rounding in the FP4 directions

The FP4 quantization is the one the A6W4 packers use (bit-exact with AITER's MXFP4: the
Hadamard normalised before the butterfly, bf16 rounding, RCEIL e8m0 scale). Only the store
layout differs, and it is AITER's A4W4 operand layout (``gemm_a4w4`` / ``gemm_a4w4_asm``
with ``bpreshuffle=True``):

  A codes   plain row-major fp4x2, [ceil(rows, 256), ceil(k, 256) / 2]
  B codes   ``shuffle_weight(layout=(16, 16))`` of that
  scales    ``shuffle_scale()``, [ceil(rows, 256), ceil(k, 256) / 32], both operands
"""

from typing import Optional, Tuple

import torch

MX_FMT_FP6 = 0
MX_FMT_A4W4_GRAD = 1
MX_FMT_A4W4_ACT = 2
MX_FMT_A4W4_GRAD_WGRAD_ONLY = 3  # FP6 rows (A6W6 dgrad), A4W4-A columns (wgrad)
MX_FMT_A4W4_GRAD_DGRAD_ONLY = 4  # A4W4-A rows (dgrad), FP6 columns (A6W6 wgrad)
# The three gradient formats with stochastic rounding in their FP4 directions:
# each code rounds up or down with probability proportional to distance, independently per
# launch (seed from a process-wide counter), so the packed gradient is unbiased.
MX_FMT_A4W4_GRAD_SR = 5
MX_FMT_A4W4_GRAD_WGRAD_ONLY_SR = 6
MX_FMT_A4W4_GRAD_DGRAD_ONLY_SR = 7
# Plain layout (Fp4Plain), the operands FlyDSL's MXFP4 GEMM takes: plain row-major codes AND plain row-major E8M0
# scales [rows, K/32], for both A and B. Gradient (both directions), its SR form, and an
# activation / weight (FP6 rows for the forward, plain FP4 columns).
MX_FMT_PLAIN_GRAD = 8
MX_FMT_PLAIN_ACT = 9
MX_FMT_PLAIN_GRAD_SR = 12
# Tilescale (MX_FMT_TS): codes plain, scales stored in the packed per-tile layout the tilescale GEMMs read (and FlyDSL's
# GEMM with scales_prepacked=True, so it does not repack them per launch). The layout belongs
# to the consuming GEMM operand, so the fmt carries it per direction (ts_fmt).
# The A4W4 tile-blob kernels (aiter `gemm_a4w4_blob_asm`): both operands in the A6W4 MXFP4
# tile blob (Fp4Blob: C0 codes rb*1024 + L*16 per 256x128 tile, scales row*4 + block, +2 guard K tiles).
MX_FMT_BLOB_GRAD = 16
MX_FMT_BLOB_ACT = 17
MX_FMT_BLOB_GRAD_SR = 18
MX_FMT_TS = 0x1000
MX_FMT_TS_SR = 0x800
TS_A = (False, 4, 0)  # every A operand: block_m is always 256, no interleave

# fmt bits 16-24: per-direction options of the FP4 directions, on top of any format above (see
# ``fp4_options``). All zero is the default emit. ``mx_fmt_base`` strips them.
MX_FMT_BASE_MASK = 0xFFFF
FP4_ROUND = {"rceil": 0, "m0": 1, "m1": 2, "m2": 3}  # m0-m2 = scale_rounding_mode 0-2
FP4_HADAMARD = {"h32": 0, "none": 1, "h16": 2}
MX_FMT_FP4_TILE2D = 1 << 24
MX_FMT_FP4_COL_SR = (
    1 << 25
)  # stochastic rounding of the (packed-FlyDSL) column direction only
# The column direction (packed FP4 tile, role B) K256-outer: codes [K/256, rows, 128] and the scale slab K256-outer
# (aiter tilescale "kouter"), so any 256-aligned range of the contraction is one contiguous byte range of both.
MX_FMT_COL_KOUTER = 1 << 26


def mx_fmt_base(fmt: int) -> int:
    return fmt & MX_FMT_BASE_MASK


def fp4_options(
    fmt: int,
    row_round: str = "rceil",
    col_round: str = "rceil",
    row_hadamard: str = "h32",
    col_hadamard: str = "h32",
    tile2d: bool = False,
    col_sr: bool = False,
) -> int:
    """``fmt`` with the FP4 directions' options set: scale rule (``FP4_ROUND``: RCEIL, the default, never
    saturates; m0 / m1 / m2 step the scale up at mantissa >= 1.75 / 1.5 / 1.8125 and saturate the rest to 6),
    Hadamard along the contraction axis (``FP4_HADAMARD``), and 2-D 32x32 block scaling (one amax per tile for
    both directions; needs no Hadamard), and SR of the column direction alone (``col_sr``: the backward copy of
    an activation / weight; packed-FlyDSL formats). The packer rejects options on an FP6 direction. Both operands of a GEMM
    must carry the same Hadamard choice for that GEMM's contraction."""
    assert not fmt >> 16, f"fmt {fmt:#x} already carries FP4 options"
    return (
        fmt
        | FP4_ROUND[row_round] << 16
        | FP4_ROUND[col_round] << 18
        | FP4_HADAMARD[row_hadamard] << 20
        | FP4_HADAMARD[col_hadamard] << 22
        | (MX_FMT_FP4_TILE2D if tile2d else 0)
        | (MX_FMT_FP4_COL_SR if col_sr else 0)
    )


def _ts_code(p):
    is_b, nt, ilv = p
    assert nt in (3, 4) and ilv in (0, 4), p
    return 0x8 | int(is_b) | (2 if nt == 3 else 0) | (4 if ilv == 4 else 0)


def ts_fmt(row=None, col=None, sr=False) -> int:
    """fmt for tilescale (packed per-tile) scales; ``row`` / ``col`` = (is_b, nt, ilv) of the GEMM operand each
    direction feeds, None for an FP6 direction (the forward rows of an activation / weight)."""
    return (
        MX_FMT_TS
        | (MX_FMT_TS_SR if sr else 0)
        | (_ts_code(col) << 4 if col else 0)
        | (_ts_code(row) if row else 0)
    )


def ts6_fmt(row_is_b: bool, col=None) -> int:
    """fmt whose row direction is MXFP6 in the A6W6 tilescale layout: codes as the K128-blocked
    C0 / C1 planes (``ts6_operand``), scales the tilescale slab at nt 4 / no interleave. ``col`` as in ``ts_fmt``."""
    return MX_FMT_TS | (_ts_code(col) << 4 if col else 0) | 0x1 | (int(row_is_b) << 1)


def with_ts6_row(fmt: int, row_is_b: bool) -> int:
    """``fmt`` (0 or a tilescale fmt with no row direction: an FP6-row activation / weight pack) with its row direction
    switched to ``ts6_fmt``'s layout; the column direction is kept."""
    ext, fmt = fmt & ~MX_FMT_BASE_MASK, mx_fmt_base(fmt)
    if fmt == 0:
        return ts6_fmt(row_is_b) | ext
    assert fmt & MX_FMT_TS and not fmt & 0xF and not fmt & MX_FMT_TS_SR, hex(fmt)
    return fmt | 0x1 | (int(row_is_b) << 1) | ext


def with_ts4_row(fmt: int, row_is_b: bool = True) -> int:
    """``fmt`` (0 or a tilescale fmt with no row direction) with its row direction switched to MXFP4 with K128-blocked
    codes ``[rows/16, K/128, 16, 64]`` (the tilescale FP4 "k128" layout: the A6W4 tilescale GEMM's weight operand),
    scales the tilescale slab at nt 4 / no interleave, round to nearest. The column direction is kept."""
    ext, base = fmt & ~MX_FMT_BASE_MASK, mx_fmt_base(fmt)
    code = 0x5 | (int(row_is_b) << 1)
    if base == 0:
        return MX_FMT_TS | code | ext
    assert base & MX_FMT_TS and not base & 0xF and not base & MX_FMT_TS_SR, hex(fmt)
    return fmt | code


def ts4_operand(codes: torch.Tensor, scales: torch.Tensor, rows: int, k: int):
    """A ``with_ts4_row`` row direction as the A6W4 tilescale GEMM takes it: codes ``[rows, k/2]`` uint8
    (K128-blocked), scales the flat uint8 slab. Rows and K must be multiples of 256."""
    assert rows % _TILE == 0 and k % _TILE == 0, (rows, k)
    return codes.view(torch.uint8).reshape(rows, k // 2), scales.view(torch.uint8).reshape(-1)


def ts6_operand(codes: torch.Tensor, scales: torch.Tensor, rows: int, k: int):
    """A ``ts6_fmt`` row direction as the A6W6 GEMM takes it: C0 [rows, k/2], C1 [rows, k/4] (K128-blocked, see
    ``gemm_mxfp6_kernel.kblk_planes``), scales the flat int32 slab. Rows and K must be multiples of 256."""
    assert rows % _TILE == 0 and k % _TILE == 0, (rows, k)
    flat = codes.view(torch.uint8).reshape(-1)
    n0 = rows * k // 2
    return (
        flat[:n0].view(rows, k // 2),
        flat[n0 : n0 + rows * k // 4].view(rows, k // 4),
        scales.view(torch.int32).reshape(-1),
    )


def ts_b_params(M: int, N: int, K: int):
    """(is_b, nt, ilv) of the B operand of the tilescale-layout MXFP4 GEMM [M, K] x [N, K]^T.

    The N tile is always 256 wide: the 192-wide one races (FlyDSL `_MXFP4_BLOCK_N_ALT_ON`), and the assembly
    kernels (aiter `gemm_a4w4_tilescale`) exist only for 256. Computed here rather than asked of FlyDSL so the
    aiter path imports no FlyDSL: the interleave is FlyDSL's `mxfp4_packed_scale_ilv(K, block_n=256)` -- 4 when the
    C store folds into the peel (at least 4 k-blocks of 256, an even count), else 0 (tested against it)."""
    kb = (K + 255) // 256
    return (True, 4, 4 if kb >= 4 and kb % 2 == 0 else 0)


def _ts_dir(fmt, col):
    fmt = mx_fmt_base(fmt)
    code = (fmt >> 4) & 0xF if col else fmt & 0xF
    if code == 0:
        return None
    if not code & 0x8:  # K128-blocked, scales at nt 4 / ilv 0: MXFP4 (with_ts4_row) or MXFP6 (ts6_fmt)
        return "k128fp4" if code & 0x4 else "kblk"
    return (bool(code & 1), 3 if code & 2 else 4, 4 if code & 4 else 0)


def ts_operand(codes: torch.Tensor, scales: torch.Tensor, rows: int, k: int):
    """A tilescale direction as the GEMM takes it: codes [rows, k/2], scales the flat int32 slab."""
    rp, kp = _ceil(rows, _TILE), _ceil(k, _TILE)
    return codes.view(rp, kp // 2)[:rows, : k // 2], scales.view(torch.int32).reshape(-1)


_TILE = 256


def _ops():
    return torch.ops.primus_turbo_cpp_extension


def _c(t):
    """Every operand contiguous, as the quantize_mxfp6_* wrappers make them (the ops assert it)."""
    return None if t is None else t.contiguous()


def _ceil(x: int, m: int) -> int:
    return -(-x // m) * m


def quantize_mx(x: torch.Tensor, axis: int, fmt: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """One direction: axis 1 (or -1) contracts columns (row direction), axis 0 contracts rows."""
    return tuple(_ops().quantize_mx(x.contiguous(), axis, fmt))


def quantize_mx_dual(x: torch.Tensor, fmt: int) -> Tuple[torch.Tensor, ...]:
    """``(row_codes, row_scales, col_codes, col_scales)`` from one pass over ``x``."""
    return tuple(_ops().quantize_mx_dual(x.contiguous(), fmt))


def quantize_mx_fused_dual(
    x: torch.Tensor,
    aux: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    mode: int,
    want_col_sum: bool,
    fmt: int,
) -> Tuple[torch.Tensor, ...]:
    """``quantize_mxfp6_fused_dual`` with an output format; returns the four blobs and the
    bias-gradient partial."""
    return tuple(_ops().quantize_mx_fused_dual(x.contiguous(), _c(aux), _c(bias), mode, want_col_sum, fmt))


def quantize_mx_qk_norm_rope_bwd(
    mixed_qkv, dq, dk, dv, cos, sin, wq, wk, rstd_q, rstd_k, want_col_sum: bool, fmt: int
) -> Tuple[torch.Tensor, ...]:
    """``quantize_mxfp6_qk_norm_rope_bwd`` with an output format; returns the four blobs,
    the bias-gradient partial and the dw_q / dw_k partials."""
    return tuple(
        _ops().quantize_mx_qk_norm_rope_bwd(
            *map(_c, (mixed_qkv, dq, dk, dv, cos, sin, wq, wk, rstd_q, rstd_k)), want_col_sum, fmt
        )
    )


def quantize_mx_ln_modulate(
    x, mean, rstd, scale, shift, want_col_sum: bool, fmt: int
) -> Tuple[torch.Tensor, ...]:
    return tuple(_ops().quantize_mx_ln_modulate(*map(_c, (x, mean, rstd, scale, shift)), want_col_sum, fmt))


def quantize_mx_gate_mul(x, gate, want_col_sum: bool, fmt: int) -> Tuple[torch.Tensor, ...]:
    return tuple(_ops().quantize_mx_gate_mul(x.contiguous(), gate.contiguous(), want_col_sum, fmt))


def quantize_mx_dual_out(
    x, row_packed, row_scale, col_packed, col_scale, fmt: int, row_c1=None, draws=1, draw_codes=0, draw_scales=0,
    col_prob=None,
) -> None:
    """``quantize_mx_dual`` into caller buffers (size them with ``mx_dir_sizes``). A direction whose two buffers are
    empty is not emitted. ``row_c1``: MXFP6 K128-blocked rows (``ts6_fmt``) with the C1 plane in its own buffer
    (``row_packed`` then holds the C0 plane). ``draws``: that many independent draws of an FP4 column direction from
    one read of ``x`` (stochastic rounding: draw d's seed derives from the launch seed and d), draw d at
    ``d * draw_codes`` / ``d * draw_scales`` bytes past ``col_packed`` / ``col_scale`` in the same allocations.
    ``col_prob`` (a round-to-nearest FP4 tile column, one draw): the column codes are emitted rounded down and
    ``col_prob`` gets each code's round-up probability -- 4 bits if it is ``col_packed``'s size (same layout), 2 bits
    if it is half of it (a code pair's byte at half the pair's offset); a receiver finishes the stochastic rounding
    with ``fp4_prob_round``."""
    _ops().quantize_mx_dual_out(
        x.contiguous(), row_packed, row_scale, col_packed, col_scale, fmt, row_c1, draws, draw_codes, draw_scales,
        col_prob,
    )


def mxfp6_tile_to_fp4_col(c0, c1, row_scale, R: int, K: int, col_packed, col_scale, fmt: int, sr: bool, seed: int):
    """The FP4 column (dgrad copy) of an [R, K] weight held as K128-blocked MXFP6 rows -- ``c0`` / ``c1`` planes and
    role-B ``row_scale`` slab, as a dual pack with ``fmt`` writes them -- into ``col_packed`` / ``col_scale`` (the
    sizes ``mx_dir_sizes(R, K, fmt, True)``): bitwise what that dual pack of the dequantized weight emits as its
    column direction, stochastically rounded from ``seed`` when ``sr`` (the dual pack's column seed is its launch seed
    ^ 0x5bd1e995)."""
    _ops().mxfp6_tile_to_fp4_col(c0, c1, row_scale, R, K, col_packed, col_scale, fmt, sr, seed & 0xFFFFFFFF)


def quantize_mx_dual_out_adam(
    param, grad, exp_avg, exp_avg_sq, remainder, row_packed, row_scale, col_packed, col_scale, fmt: int, *,
    lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, step: int, adamw: bool = True,
    bias_correction: bool = True, row_c1=None, draws=1, draw_codes=0, draw_scales=0,
) -> None:
    """``quantize_mx_dual_out`` of a bf16 parameter fused with its optimizer step: Transformer Engine's FusedAdam
    with ``store_param_remainders`` (fp32 master = ``param`` bits + int16 ``remainder``, fp32 moments), the same
    arithmetic, applied in place to ``param`` / ``remainder`` / ``exp_avg`` / ``exp_avg_sq``; the updated
    parameter is what gets packed. All operands contiguous, ``param``'s size; rows and columns multiples of 256."""
    _ops().quantize_mx_dual_out_adam(
        param, grad, exp_avg, exp_avg_sq, remainder, row_packed, row_scale, col_packed, col_scale, fmt, row_c1,
        draws, draw_codes, draw_scales, lr, beta1, beta2, eps, weight_decay, step, adamw, bias_correction
    )


def quantize_mx_fused_dual_out(
    x, aux, bias, mode: int, row_packed, row_scale, col_packed, col_scale, col_sum, fmt: int
) -> None:
    """``quantize_mx_fused_dual`` into caller buffers; ``col_sum`` as in the FP6 out-variant."""
    _ops().quantize_mx_fused_dual_out(
        x.contiguous(), _c(aux), _c(bias), mode, row_packed, row_scale, col_packed, col_scale, col_sum, fmt
    )


# Which layout each direction of a format emits: (row is A4W4, col is A4W4).
_DIR_FP4 = {
    0: (False, False),
    1: (True, True),
    2: (False, True),
    3: (False, True),
    4: (True, False),
    5: (True, True),
    6: (False, True),
    7: (True, False),
    8: (True, True),
    9: (False, True),
    12: (True, True),
}


def mx_fp4_dirs(fmt: int) -> Tuple[bool, bool]:
    """Whether the (row, column) directions of a ``fmt`` pack are FP4 (the ones ``fp4_options`` may set)."""
    fmt = mx_fmt_base(fmt)
    if fmt & MX_FMT_TS:
        return tuple(isinstance(_ts_dir(fmt, col), tuple) or _ts_dir(fmt, col) == "k128fp4" for col in (False, True))
    if fmt in (MX_FMT_BLOB_GRAD, MX_FMT_BLOB_GRAD_SR):
        return (True, True)
    if fmt == MX_FMT_BLOB_ACT:
        return (False, True)
    return _DIR_FP4[fmt]


def mx_dir_sizes(rows: int, k: int, fmt: int, col: bool) -> Tuple[int, int]:
    """Byte sizes ``(codes, scales)`` of one direction of a ``fmt`` pack of a ``rows x k``
    operand contracted along ``k`` (for the column direction pass the transposed extent)."""
    fmt = mx_fmt_base(fmt)
    if fmt in (MX_FMT_BLOB_GRAD, MX_FMT_BLOB_ACT, MX_FMT_BLOB_GRAD_SR):
        if fmt == MX_FMT_BLOB_ACT and not col:
            from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import mxfp6_pack_sizes

            return mxfp6_pack_sizes(rows, k)
        rt, kt = -(-rows // _TILE), -(-k // 128) + 2  # the A6W4 FP4 tile blob, +2 guard K tiles
        return rt * kt * 16384, rt * kt * 1024
    if fmt & MX_FMT_TS:
        p = _ts_dir(fmt, col)
        if p is None:
            from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import mxfp6_pack_sizes

            return mxfp6_pack_sizes(rows, k)
        rp, kp = _ceil(rows, _TILE), _ceil(k, _TILE)
        if p == "kblk":
            return rp * kp * 3 // 4, -(-rows // 256) * 256 * (kp // 128) * 4
        if p == "k128fp4":
            return rp * kp // 2, -(-rows // 256) * 256 * (kp // 128) * 4
        return rp * kp // 2, -(-rows // (64 * p[1])) * 256 * (kp // 128) * 4
    if _DIR_FP4[fmt][1 if col else 0]:
        rp, kp = _ceil(rows, _TILE), _ceil(k, _TILE)
        return rp * kp // 2, rp * kp // 32
    from primus_turbo.pytorch.kernels.quantization.mxfp6_pack import mxfp6_pack_sizes

    return mxfp6_pack_sizes(rows, k)


def a4w4_operand(codes: torch.Tensor, scales: torch.Tensor, rows: int, k: int):
    """View an A4W4-format direction as the 2-D tensors ``gemm_a4w4`` takes.

    ``rows`` and ``k`` are the operand's logical extent (rows of the operand, contraction
    length). The views cover the padded extent; rows are padded to 256, which AITER's kernels
    accept, and ``k`` is expected to be a multiple of 256.
    """
    rp, kp = _ceil(rows, _TILE), _ceil(k, _TILE)
    return codes.view(rp, kp // 2), scales.view(rp, kp // 32)


def plain_operand(codes: torch.Tensor, scales: torch.Tensor, rows: int, k: int):
    """View a plain-layout (fmt 8 / 9 / 12) direction as FlyDSL takes it: codes [rows, k/2] and
    E8M0 scales [rows, k/32], both row-major (the padded tail rows are dropped)."""
    rp, kp = _ceil(rows, _TILE), _ceil(k, _TILE)
    return codes.view(rp, kp // 2)[:rows, : k // 2], scales.view(torch.uint8).view(rp, kp // 32)[
        :rows, : k // 32
    ]


def a4w4_logical(codes: torch.Tensor, scales: torch.Tensor, rows: int, k: int, is_b: bool):
    """Undo the A4W4 layout: ``(codes [rows, k/2] uint8, scales [rows, k/32] uint8)`` in
    logical (row, group) order. For tests and debugging; the GEMM consumes the layout as is."""
    rp, kp = _ceil(rows, _TILE), _ceil(k, _TILE)
    c = codes.view(torch.uint8)
    if is_b:  # (N/16, Kb/32, 2, 16, 16) -> (N/16, 16, Kb/32, 2, 16)
        c = c.view(rp // 16, kp // 64, 2, 16, 16).permute(0, 3, 1, 2, 4).reshape(rp, kp // 2)
    else:
        c = c.view(rp, kp // 2)
    sp = kp // 32  # shuffle_scale stores (i0, j0, j2, i2, j1, i1); logical is (i0, i1, i2, j0, j1, j2)
    s = scales.view(torch.uint8).view(rp // 32, sp // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2)
    s = s.reshape(rp, sp)
    return c[:rows, : k // 2], s[:rows, : k // 32]


def a6w4_blob_logical(codes: torch.Tensor, scales: torch.Tensor, rows: int, k: int):
    """Decode AITER's A6W4 MXFP4 blob (what ``quantize_mxfp4_gemm_*`` writes) into the same
    logical ``(codes [rows, k/2], scales [rows, k/32])``. Mirrors mxfp4_emit's store address."""
    nk_pad = -(-k // 128) + 2  # MXFP6_GUARD_K_TILES
    dev = codes.device
    r = torch.arange(rows, device=dev).view(-1, 1)
    g = torch.arange(k // 32, device=dev).view(1, -1)
    tile_row, rem = r // 256, r % 256
    step, k_group = g // 4, g % 4
    tile = tile_row * nk_pad + step
    block = (rem // 16) * 64 + k_group * 16 + rem % 16
    base = tile * 16384 + block * 16
    byte = base.unsqueeze(-1) + torch.arange(16, device=dev)
    c = codes.view(torch.uint8)[byte].reshape(rows, k // 2)
    saddr = tile * 1024 + (rem // 128) * 512 + k_group * 128 + (rem % 16) * 8 + (rem % 128) // 16
    s = scales.view(torch.uint8)[saddr]
    return c, s
