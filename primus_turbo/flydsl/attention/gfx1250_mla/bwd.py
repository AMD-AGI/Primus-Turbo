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

"""Host side of the gfx1250 MLA flash-attention backward: shape rules, launch plan, entry point.

k_delta_bshd (bwd_delta_kernel: the row constants NL / ND), then per head chunk k_dkdv (bwd_dkdv_kernel:
dK and dV over 192-row kv blocks, and every bf16 dS^T tile it feeds its dK WMMA also into a scratch; persistent
workgroups that run the item lists of a table made once per shape, bwd_dkdv_items) and k_dqds (bwd_dq_kernel: dQ =
dS K from that scratch), so S and dP are computed once.

The whole dS^T scratch is about batch * heads * seqlen_q * seqlen_kv bytes under the causal mask (4.33 GB at b2 s4096
h128). The backward runs all heads on one scratch (the fastest plan) when that is within
PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB. Over the budget it runs the heads in chunks, alternating between the current
stream and a side stream with a scratch each (one chunk's kernels fill the CUs the other chunk's tail leaves idle; two
chunks' scratch is held at once): the fewest chunks that fit (ds_chunkings), and with a budget below two heads'
scratch one head at a time on one stream (that one head's scratch even when it is over the budget). The scratch is the
last buffer the backward allocates, from the caching allocator on the current stream; one that cannot be allocated
sends the call to smaller chunks. Every head's arithmetic is the same however the heads are chunked, so the gradients
are too.
"""

import contextlib
import inspect
import math
from typing import List, Optional, Tuple

import flydsl.compiler as flyc
import torch

import primus_turbo.common.envs as envs
from primus_turbo.common.logger import logger

from . import bwd_delta_kernel as _delta
from . import bwd_dkdv_items as _items
from . import bwd_dkdv_kernel as _dkdv
from . import bwd_dq_kernel as _dq
from .common import num_cus

# device -> the smallest dS scratch whose allocation failed there: a failed allocation costs the caching allocator a
# synchronize and its whole cache, so a scratch at least as large is not tried there again (except the smallest).
_DS_OOM_BYTES = {}
# (device index, caller stream) -> the side stream of its chunked backwards, so backwards on different streams stay
# independent
_SIDE_STREAMS = {}

# Parameter names of each launcher (_positional).
_PARAMS = {}
# flyc.compile's CompiledFunction per (launcher, tensor dtypes/ranks, device). Launching through it
# skips FlyDSL's per-call key derivation; all shape-dependent kernel arguments are runtime scalars.
_COMPILED = {}


def shape_unsupported_reason(batch: int, seqlen_q: int, seqlen_kv: int, num_heads: int) -> Optional[str]:
    """Why the backward cannot run these MHA sizes, or None. The forward takes every size this accepts."""
    if batch <= 0 or num_heads <= 0:
        return f"batch and num_heads must be positive, got {batch} / {num_heads}"
    # batch * heads is a dimension of k_delta_bshd's launch grid, and heads and batch (batch * heads and 1 for
    # b > 1 SBHD storage, which is folded) are dimensions of the forward's. They are 16-bit: at 65536 a launch
    # wraps to no workgroup at all, without an error. (Today the size rules below stop before that: they take
    # at most 58254, at seqlen_kv 64.)
    if batch * num_heads >= 1 << 16:
        return (
            "batch * heads must be under 65536 (a 16-bit dimension of the kernels' launch grids), got "
            f"{batch} * {num_heads}"
        )
    # multiples of 64: the forward's q tiles (4 waves x 16 rows), k_dqds's kv units, k_dkdv's last kv block
    if seqlen_q <= 0 or seqlen_q % 64:
        return f"seqlen_q must be a positive multiple of 64, got {seqlen_q}"
    if seqlen_kv <= 0 or seqlen_kv % 64:
        return f"seqlen_kv must be a positive multiple of 64, got {seqlen_kv}"
    if seqlen_q > seqlen_kv:
        # bottom-right causal leaves the first seqlen_q - seqlen_kv queries without keys
        return f"causal needs seqlen_q <= seqlen_kv, got {seqlen_q} > {seqlen_kv}"
    # The size limit is the 32-bit byte offsets of k / dk (k_dkdv) and of q / dq (k_dqds; k_dkdv too reaches a Q
    # tile by a 32-bit offset from q's 64-bit base): under 2 GiB each, so k's, as seqlen_q <= seqlen_kv. The other
    # tensors with 32-bit offsets or byte counts are smaller than q: o / dO (k_delta_bshd, k_dkdv's dO tiles), o and
    # lse (the forward), NL / ND (k_dkdv). The rest (q / k / v in the forward, K in k_dqds, the dS scratch) take
    # 64-bit addresses.
    if batch * num_heads * seqlen_kv * _dkdv.D_QK * 2 >= 1 << 31:
        return (
            "k must be under 2 GiB (the kernels' 32-bit byte offsets), got batch * seqlen_kv * heads = "
            f"{batch} * {seqlen_kv} * {num_heads} rows of {_dkdv.D_QK * 2} bytes"
        )
    # k_dkdv pads the last kv block to ROWS_B rows and leaves the dK / dV stores of its rows past seqlen_kv to the
    # buffer range check. Their 32-bit offsets must not wrap back into range for that, so k padded to whole blocks
    # stays under 4 GiB (dV's rows are shorter). Only seqlen_kv 64 gets there, where the padding triples k. The
    # rule is on batch * heads, like the one above: b > 1 SBHD storage reaches the kernels as [1, s, b * h, d], and
    # the gate (the op's shape) and the entry (the kernels') must agree.
    padded_kv = _items.nblk_kv(seqlen_kv) * _items.ROWS_B
    if batch * num_heads * padded_kv * _dkdv.D_QK * 2 >= 1 << 32:
        return (
            f"k padded to k_dkdv's {_items.ROWS_B}-row kv blocks must be under 4 GiB (32-bit store offsets), got "
            f"batch * rows * heads = {batch} * {padded_kv} * {num_heads} rows of {_dkdv.D_QK * 2} bytes"
        )
    return None


def _head_group(nh, target):
    """Heads per grid row: nh up to ``target``, else the largest multiple of 8 (one XCD per head under
    round-robin dispatch) that divides nh and is at most ``target``, else nh."""
    if nh <= target:
        return nh
    return next((d for d in range(target, 7, -8) if nh % d == 0), nh)


def ds_scratch_bytes(batch: int, seqlen_q: int, seqlen_kv: int, num_heads: int) -> int:
    """Bytes of the whole dS^T scratch of these MHA sizes (chunked, the backward holds two head chunks of it)."""
    return batch * num_heads * _items.ds_blocks_per_head(seqlen_q, seqlen_kv) * _items.DSL_BLK_B


def ds_chunkings(
    batch: int, seqlen_q: int, seqlen_kv: int, num_heads: int, budget: int
) -> List[Tuple[int, int, int]]:
    """[(heads per chunk, streams, scratch bytes)] the backward tries in order: the candidates within ``budget``
    bytes, each smaller than the one before (the next one is for a scratch that cannot be allocated). The candidates:
    all heads on one scratch; chunks of ceil(heads / 2^i) heads alternating between two streams with a scratch each
    (two chunks of half the heads take as much as all heads on one scratch, so the chunks start at a quarter); last,
    one head at a time on one stream. When nothing fits the budget, only the last candidate."""
    per_head = ds_scratch_bytes(batch, seqlen_q, seqlen_kv, 1)
    opts, hc = [(num_heads, 1, per_head * num_heads)], num_heads
    while hc > 1:
        hc = -(-hc // 2)
        opts.append((hc, 2, per_head * hc * 2))
    opts.append((1, 1, per_head))
    tries = []
    for opt in opts:
        if opt[2] <= budget and (not tries or opt[2] < tries[-1][2]):
            tries.append(opt)
    return tries or opts[-1:]


def _side_stream(device, stream):
    key = (device.index, stream.cuda_stream)
    side = _SIDE_STREAMS.get(key)
    if side is None:
        side = _SIDE_STREAMS[key] = torch.cuda.Stream(device=device)
    return side


def _delta_launch(dout, q, k, v, out, lse, softmax_scale, stream):
    """The k_delta_bshd launch and the buffers of the whole backward but the scratch: (NL, ND, dq, dk, dv)."""
    b, sq, h, _ = q.shape
    nl = torch.empty((b, h, sq), device=q.device, dtype=torch.float32)  # -LOG2E * lse
    nd = torch.empty_like(nl)  # -scale * rowsum(dO * O)
    bufs = (nl, nd, q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape))
    launch = _positional(
        _delta.launch_delta,
        DO=dout,
        O=out,
        LSE=lse,
        NL=nl,
        ND=nd,
        scale=float(softmax_scale),
        S=sq,
        H=h,
        n_rows=b * sq * h,
        nsblk=-(-sq // _delta.ROWS_DELTA),
        nbh=b * h,
        stream=stream,
    )
    return (_delta.launch_delta, launch), bufs


def _chunk_launches(dout, q, k, v, softmax_scale, stream, bufs):
    """The dS scratch (the backward's last allocation: when it fits, the backward fits) and the k_dkdv / k_dqds
    launches of each head chunk, in issue order. torch.OutOfMemoryError when even one head's scratch cannot be
    allocated."""
    nl, nd, dq, dk, dv = bufs
    b, sq, h, _ = q.shape
    skv = k.shape[1]
    ncu = num_cus(q.device)
    budget = envs.PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB << 20
    tries = ds_chunkings(b, sq, skv, h, budget)
    if tries[0][2] > budget:
        logger.warning(
            f"MLA backward: one head's dS scratch, {tries[0][2]} bytes, is over the {budget}-byte budget "
            "(PRIMUS_TURBO_MLA_BWD_DS_SCRATCH_MB); the heads run one at a time, each on that much scratch",
            once=True,
        )
    for i, (hc, streams, nbytes) in enumerate(tries):
        last = i == len(tries) - 1
        if not last and nbytes >= _DS_OOM_BYTES.get(q.device, math.inf):
            continue
        # k_dkdv's persistent workgroups run the item lists of a table made once per shape. A new one is a few MB on
        # the device, allocated before the scratch (every plan's tables hold the same items: without memory for
        # them no plan runs, so their torch.OutOfMemoryError is not caught here)
        tables = [
            _items.item_table(b, sq, skv, h, h0, min(hc, h - h0), ncu, q.device) for h0 in range(0, h, hc)
        ]
        try:
            ds = torch.empty((nbytes // 2,), device=q.device, dtype=torch.bfloat16)
            break
        except torch.OutOfMemoryError as e:
            if last:
                raise torch.OutOfMemoryError(
                    f"MLA backward: no memory on {q.device} for even one head's dS scratch, {nbytes} bytes (about "
                    f"batch * seqlen_q * seqlen_kv, here {b} * {sq} * {skv}); the FlyDSL gfx1250 MLA backward cannot "
                    "run without it: free memory, or take this sequence length to another attention backend "
                    f"(PRIMUS_TURBO_ATTN_BACKEND). {e}"
                ) from e
            _DS_OOM_BYTES[q.device] = nbytes
            logger.warning(
                f"MLA backward: no memory for a {nbytes}-byte dS scratch on {q.device}; smaller head chunks run (a "
                "scratch at least as large is not tried again)",
                once=True,
            )
    scale = float(softmax_scale)
    q_off, nhead, ntile = _dq.dq_split(sq)
    ds_blocks = _items.ds_blocks_per_head(sq, skv)
    scratches = ds.view(streams, -1)
    # The chunks alternate between the caller's stream and a side stream (meta tensors: one stream).
    side = stream if streams == 1 or q.device.type != "cuda" else _side_stream(q.device, stream)
    launches = []
    for i, h0 in enumerate(range(0, h, hc)):
        nh, st, ds_c = min(hc, h - h0), (stream, side)[i % 2], scratches[i % streams]
        hg_q = _head_group(nh, _dq.HEAD_GROUP)
        # MHA (Hq = Hkv = h, G = 1) on the heads [h0, h0 + nh). The kernels take only the scratch's base address:
        # k_dkdv a [blocks, 512] view (FlyDSL packs tensor sizes as int32), k_dqds the address itself.
        table, nwg, nslot = tables[i]
        dkdv = _positional(
            _dkdv.launch_dkdv,
            Q=q,
            K=k,
            V=v,
            DO=dout,
            NL=nl,
            ND=nd,
            DV_=dv,
            DK=dk,
            DS=ds_c.view(-1, _items.DSL_BLK_B // 2),
            TBL=table,
            scale=scale,
            Hq=h,
            nl_bytes=b * h * sq * 4,
            nslot=nslot,
            nwg=nwg,
            stream=st,
        )
        dqds = _positional(
            _dq.launch_dqds,
            DS=ds_c.data_ptr(),
            K=k,
            DQ=dq,
            Sq=sq,
            Skv=skv,
            Hq=h,
            Hkv=h,
            G=1,
            nkvt=skv // _dq.KV32,
            cshift=skv - sq,
            causal=1,
            q_off=q_off,
            ntile=ntile,
            HG=hg_q,
            nblk=ds_blocks,
            dq_bytes=b * sq * h * _dkdv.D_QK * 2,
            H0=h0,
            HC=nh,
            ny=ntile + nhead,
            ngz=b * nh // hg_q,
            stream=st,
        )
        launches += [(_dkdv.launch_dkdv, dkdv), (_dq.launch_dqds, dqds)]
    return launches


@contextlib.contextmanager
def _joined(stream, side):
    """Around the launches of a backward that also runs on ``side`` (None: it does not): ``side`` starts after what
    ``stream`` has issued, and ``stream`` goes on after ``side``, also when a launch raised: the kernels ``side``
    already runs still write the backward's buffers, which the caller's stream may reuse once they are freed."""
    if side is None:
        yield
        return
    side.wait_stream(stream)
    try:
        yield
    finally:
        stream.wait_stream(side)


def _on_device(device):
    """``device`` as the current device for the launches and their streams, when it is not already (meta / CPU
    tensors: no change)."""
    if device.type != "cuda" or device.index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(device)


def _stream_arg(launcher, args):
    """The stream ``launcher`` runs on, from its positional ``args`` (_positional)."""
    return args[_PARAMS[launcher].index("stream")]


def _positional(launcher, **kwargs):
    """``launcher``'s arguments in its parameter order (FlyDSL's compiled call takes them positionally)."""
    names = _PARAMS.get(launcher)
    if names is None:
        names = _PARAMS[launcher] = tuple(inspect.signature(launcher.func).parameters)
    if len(kwargs) != len(names):
        raise TypeError(f"{launcher.func.__name__} takes {names}, got {tuple(kwargs)}")
    return tuple(kwargs[n] for n in names)


def _launch(launcher, args):
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    key = (launcher, tuple((a.dtype, a.dim()) for a in tensors), tensors[0].device.index)
    fn = _COMPILED.get(key)
    if fn is not None:
        fn(*args)
        return
    fn = flyc.compile(launcher, *args)  # compiles and issues this launch
    if fn is not None:  # None under COMPILE_ONLY
        _COMPILED[key] = fn


def flash_attn_mla_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients (dq, dk, dv) of MLA flash attention on gfx1250, laid out like q, k, v.

    q [B, Sq, H, 192], k [B, Skv, H, 192], v [B, Skv, H, 128], out/dout [B, Sq, H, 128], contiguous
    bf16 on one device; lse [B, H, Sq] fp32 natural log, as the forward returns it. Causal only,
    bottom-right.
    """
    if not causal:
        raise ValueError("the gfx1250 MLA attention backward supports causal attention only")
    named = (
        ("dout", dout, _dkdv.D_V),
        ("q", q, _dkdv.D_QK),
        ("k", k, _dkdv.D_QK),
        ("v", v, _dkdv.D_V),
        ("out", out, _dkdv.D_V),
    )
    for name, t, dim in named:
        if t.dtype != torch.bfloat16 or t.dim() != 4 or t.shape[-1] != dim or not t.is_contiguous():
            raise ValueError(f"{name} must be a contiguous [B, S, H, {dim}] bfloat16 tensor")
    if any(t.device != q.device for t in (dout, k, v, out, lse)):
        raise ValueError("dout/q/k/v/out/lse must be on one device")
    b, sq, h, _ = q.shape
    skv = k.shape[1]
    if k.shape[:3] != (b, skv, h) or v.shape[:3] != (b, skv, h):
        raise ValueError(
            f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}: batch and heads must match (MHA)"
        )
    if out.shape[:3] != (b, sq, h) or dout.shape != out.shape or lse.shape != (b, h, sq):
        raise ValueError(
            f"out/dout/lse shapes {tuple(out.shape)}/{tuple(dout.shape)}/{tuple(lse.shape)} "
            f"do not match q {tuple(q.shape)}"
        )
    reason = shape_unsupported_reason(b, sq, skv, h)
    if reason is not None:
        raise ValueError(reason)
    lse = lse.contiguous().float()
    with _on_device(q.device):
        stream = torch.cuda.current_stream()
        # k_delta_bshd (the row constants every later launch reads) goes out first: the chunks' host work, the
        # scratch allocation included, then overlaps it.
        delta_launch, bufs = _delta_launch(dout, q, k, v, out, lse, softmax_scale, stream)
        _launch(*delta_launch)
        launches = _chunk_launches(dout, q, k, v, softmax_scale, stream, bufs)
        # Chunks on a side stream start after k_delta_bshd; the caller's stream goes on after both streams' chunks.
        streams = {_stream_arg(launcher, args) for launcher, args in launches} - {stream}
        with _joined(stream, streams.pop() if streams else None):
            for launcher, args in launches:
                _launch(launcher, args)
    return bufs[2:]
