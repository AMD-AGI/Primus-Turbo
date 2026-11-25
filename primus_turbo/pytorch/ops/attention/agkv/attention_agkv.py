###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""All-gather-KV context parallelism for causal GQA attention (the Llama 3 scheme), fused.

The sequence is cut into 2n chunks of c tokens and rank r holds chunks r and 2n-1-r (the zigzag order of
Megatron's ``get_batch_on_this_cp_rank``), so every rank does the same causal work. Every rank keeps all
heads; only K and V cross ranks: gathered in the forward, their gradients reduced back in the backward.
q0 (chunk r) sees the first halves of ranks 0..r, q1 (chunk 2n-1-r) every first half, the second halves of
ranks r+1..n-1 and its own second half.

The forward is one persistent FlyDSL kernel on the current stream (``flydsl/attention/agkv/fwd.py``): a
few blocks push this rank's K / V halves over XGMI into the peers' symmetric buffers and raise a flag per
(destination, half); every block then claims attention work of q0 and q1, their own keys first, then the
remote chunks, each once its flag is up, and the copies the backward needs out of the bank. Receiver p
keeps the remote halves as

    [ second halves of p+1..n-1 | first halves of p+1..n-1 | first halves of 0..p-1 ]

so q0's remote keys (the first halves of the lower ranks) are its tail and q1's are all of it; the forward
saves [ remote | own first | own second ] for the backward.

The backward is one persistent kernel as well (``flydsl/attention/agkv/bwd.py``): the dK / dV work-groups
of q0 and q1 store the remote-key gradients (q splits and q0's part summed) straight into the owners'
slots and raise their flags, dQ is reduced head group by head group as each group's items finish, and the
same kernel sums what this rank received.

The double-buffered symmetric buffers are reused call after call, which needs every rank of the group to
issue the same sequence of forward and backward calls (true of SPMD training); calls are counted on the
host, so the op cannot be graph captured. Kernels that wait on their peers' kernels need the collectives
of the same process on other hardware queues: run with ``GPU_MAX_HW_QUEUES>=8``.
"""

import functools
import math
from typing import Optional

import torch
import torch.distributed as dist

from primus_turbo.flydsl.attention.agkv import bwd as _bwd_kernel
from primus_turbo.flydsl.attention.agkv import comm as _comm
from primus_turbo.flydsl.attention.agkv import fwd as _fwd_kernel
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory

__all__ = ["attention_agkv"]

# blocks of the forward's push: (per peer link, at least, at most in all), always a multiple of the n - 1
# links, which keeps every block on one link (units alternate between destinations) so no link has fewer
# blocks than another. Fewer blocks leave the links idle, more slow the attention beside them; the
# attention needs the keys early (28 / 24 / 8 blocks at n = 8 / 4 / 2).
_FWD_PUSH_BLOCKS = (8, 1, 28)


def _push_grid(n, blocks):
    per_peer, least, most = blocks
    return (n - 1) * max(min(per_peer, most // (n - 1)), -(-least // (n - 1)), 1)


# --------------------------------------------------------------------------------------------------------
# symmetric exchange buffers
# --------------------------------------------------------------------------------------------------------


# flag arrays in the signal pad, n int64 each, indexed by sender
_FWD_FIRST, _FWD_SECOND, _BWD_FIRST, _BWD_SECOND = range(4)
_NUM_FLAG_ARRAYS = 4


class _Exchange:
    """One rank's symmetric buffers and flags for a process group.

    Buffer, per bank (two, used alternately by consecutive calls of each direction):
        forward   K, V remote halves, (2n-2-r)c tokens each, in the ``_remote_layout`` order
        backward  K, V partial-gradient slots, [first: n senders | second: n senders] x c tokens each

    Each direction counts its calls (epochs, from 1) and uses bank ``epoch & 1``; a sender raises its flag
    to the epoch once its data is in, a receiver waits until the flag reaches it. A rank's call e cannot
    finish before every peer has sent its call-e data, which a peer does only after finishing call e - 1,
    so when a peer writes bank e & 1 for call e + 2 this rank is past call e: two banks need no
    acknowledgement. Flags only grow, so a sender one call ahead does not hide a call from the receiver.
    """

    _cache = {}

    def __init__(self, group, nbytes):
        self.group = group
        self.n, self.r = group.size(), group.rank()
        self.symm = SymmetricMemory(
            group, nbytes, signal_pad_size=_NUM_FLAG_ARRAYS * self.n * 8, uncached_signal_pad=True
        )
        self.nbytes = nbytes
        self.epoch = {"fwd": 0, "bwd": 0}
        self.heaps = torch.tensor(self.symm.buffer_ptrs, dtype=torch.int64, device="cuda")
        self.flag_ptrs = torch.tensor(self.symm.signal_pad_ptrs, dtype=torch.int64, device="cuda")
        # forward push group counters, one per (destination, half)
        self.counters = torch.zeros(2 * self.n, dtype=torch.int64, device="cuda")

    @classmethod
    def get(cls, group, nbytes):
        key = (group.group_name, torch.cuda.current_device())
        old = cls._cache.get(key)
        if old is not None and old.nbytes >= nbytes:
            return old
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the all-gather-KV attention counts its calls on the host and cannot be graph captured"
            )
        # allocate before freeing: a heap that reuses just-freed memory cannot be exported over IPC
        ex = cls._cache[key] = cls(group, max(nbytes, 0 if old is None else 2 * old.nbytes))
        if old is not None:
            # peers may still be writing into the old heap until every rank is past its last call
            torch.cuda.synchronize()
            dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
            old.symm.destroy()
        return ex

    def begin(self, direction):
        """Epoch of a new call in ``direction``; its bank is ``epoch & 1``."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the all-gather-KV attention counts its calls on the host and cannot be graph captured"
            )
        self.epoch[direction] += 1
        return self.epoch[direction]

    def view(self, rank, byte_offset, shape, dtype):
        return self.symm.get_buffer(rank, shape, dtype, storage_offset=byte_offset // dtype.itemsize)


def _exchange_sizes(n, c, tok):
    fwd = 2 * (2 * n + 1) * c * tok
    bwd = 2 * 2 * n * c * tok
    return fwd, bwd


def _exchange(group, n, c, tok):
    fwd_bytes, bwd_bytes = _exchange_sizes(n, c, tok)
    return _Exchange.get(group, 2 * (fwd_bytes + bwd_bytes)), fwd_bytes, bwd_bytes


# --------------------------------------------------------------------------------------------------------
# copy unit tables
# --------------------------------------------------------------------------------------------------------


def _units_table(units, device):
    t = torch.tensor(units, dtype=torch.int64, device=device)
    return t, len(units)


def _phase(pieces, items):
    """Unit rows of ``items`` = ``(segment rows, ...)`` interleaved piece-major, so consecutive units
    (taken by consecutive blocks) go to different links."""
    rows = []
    for piece in range(max(len(seg) for seg in items) if items else 0):
        for seg in items:
            if piece < len(seg):
                rows.append(seg[piece])
    return rows


def _segment(sel, src_off, dst, dst_off, nbytes, group, group_size, flag):
    step = _comm.UNIT_BYTES
    return [
        [sel, src_off + o, dst, dst_off + o, min(step, nbytes - o), group, group_size, flag]
        for o in range(0, nbytes, step)
    ]


def _remote_layout(n, p):
    """Token offsets (in chunks) of the remote slots in receiver ``p``'s buffer, laid out

        [ second halves of senders p+1..n-1 | first halves of p+1..n-1 | first halves of 0..p-1 ]

    so q1's remote keys are the whole buffer and q0's (first halves of the lower ranks) its suffix."""
    first = {s: (n - 1 - p) + (s - p - 1) if s > p else 2 * (n - 1 - p) + s for s in range(n) if s != p}
    second = {s: s - p - 1 for s in range(p + 1, n)}
    return first, second


@functools.lru_cache(maxsize=64)
def _push_plan(n, r, c, tok, fwd_bytes, device):
    """Forward push units of rank ``r``: K (source 0) / V (source 1) halves into the peers' forward banks
    (V region at ``fwd_bytes // 2``) at their ``_remote_layout`` slots, the first half to every peer, the
    second half to the lower ones. A (destination, half) is one group; its flag is ``_FWD_FIRST`` /
    ``_FWD_SECOND`` of sender ``r``. Every rank's q0 needs only first halves, so they all go first, each
    destination served alike."""
    half = c * tok
    pieces = -(-half // _comm.UNIT_BYTES)

    def segments(dst, h):
        first, second = _remote_layout(n, dst)
        dst_off = (first if h == 0 else second)[r] * half
        group, flag = 2 * dst + h, (_FWD_FIRST if h == 0 else _FWD_SECOND) * n + r
        return [
            _segment(t, h * half, dst, t * (fwd_bytes // 2) + dst_off, half, group, 2 * pieces, flag)
            for t in range(2)
        ]

    peers = list(range(r + 1, n)) + list(range(r))
    firsts = [sg for p in peers for sg in segments(p, 0)]
    seconds = [sg for p in peers if p < r for sg in segments(p, 1)]
    rows = _phase(pieces, firsts) + _phase(pieces, seconds)
    return _units_table(rows, device) if rows else (None, 0)


@functools.lru_cache(maxsize=64)
def _save_plan(n, r, c, tok, device):
    """Save units of the forward: [ remote | own first | own second ] of K and V out of the bank and the
    local tensors, the own chunks first, then the remote chunks first halves before second halves (the
    order they arrive in), each behind its flag."""
    rem = (2 * n - 2 - r) * c
    rows = []
    for t in range(2):
        rows += _segment(_fwd_kernel.SAVE_SRC_K + t, 0, t, rem * tok, 2 * c * tok, 0, 0, -1)
    chunks = sorted(range(rem // c), key=lambda k: (_fwd_kernel.bank_chunk_flag(n, r, k) >= n, k))
    for k in chunks:
        flag = _fwd_kernel.bank_chunk_flag(n, r, k)
        for t in range(2):
            rows += _segment(
                _fwd_kernel.SAVE_SRC_BANK_K + t, k * c * tok, t, k * c * tok, c * tok, 0, 0, flag
            )
    return _units_table(rows, device)


# --------------------------------------------------------------------------------------------------------
# forward / backward
# --------------------------------------------------------------------------------------------------------


def _forward(q, k, v, group, save):
    """Returns ``out``, the lse of q0 / q1 and, if ``save``, K / V laid out
    [ remote slots (``_remote_layout``) | own first half | own second half ] for the backward."""
    n, r = group.size(), group.rank()
    c = q.shape[0] // 2
    tail_shape, dtype = k.shape[1:], k.dtype
    B, Hq, _ = q.shape[1:]
    tok = k[0].numel() * k.element_size()
    ex, fwd_bytes, _ = _exchange(group, n, c, tok)
    epoch = ex.begin("fwd")
    bank_off = (epoch & 1) * fwd_bytes
    units, num_units = _push_plan(n, r, c, tok, fwd_bytes, k.device)
    rem = (2 * n - 2 - r) * c
    kr, vr = (ex.view(r, bank_off + i * fwd_bytes // 2, (rem,) + tail_shape, dtype) for i in range(2))
    out = torch.empty_like(q)
    lse = torch.empty((2, B * c, Hq), device=q.device, dtype=torch.float32)
    if save:
        saved = [torch.empty((rem + 2 * c,) + tail_shape, device=k.device, dtype=dtype) for _ in range(2)]
        save_units, num_save = _save_plan(n, r, c, tok, k.device)
    else:
        saved, save_units, num_save = None, units, 0
    sk, sv = saved if save else (k, v)
    _fwd_kernel.forward(
        q,
        k,
        v,
        kr,
        vr,
        out,
        lse,
        sk,
        sv,
        ex.heaps,
        ex.flag_ptrs,
        units,
        num_units,
        ex.counters,
        save_units,
        num_save,
        n,
        r,
        bank_off,
        epoch,
        _push_grid(n, _FWD_PUSH_BLOCKS),
        torch.cuda.current_stream(),
    )
    return out, lse[0], lse[1], saved


def _backward(dout, q, kb, vb, out, lse0, lse1, group, scale):
    """Both chunks' backward in one persistent kernel that stores the remote-key gradients (q-split sum
    and q0's rows added) into the owners' backward banks, raises their flags itself and then sums the
    gradients this rank received (see ``flydsl/attention/agkv/bwd.py``)."""
    n, r = group.size(), group.rank()
    c = q.shape[0] // 2
    tail_shape, dtype = kb.shape[1:], kb.dtype
    tok = kb[0].numel() * kb.element_size()
    ex, fwd_bytes, bwd_bytes = _exchange(group, n, c, tok)
    epoch = ex.begin("bwd")
    bank_off = 2 * fwd_bytes + (epoch & 1) * bwd_bytes

    own = torch.empty((2, 2 * c) + tail_shape, device=kb.device, dtype=dtype)
    first, second = _remote_layout(n, r)
    owner = {k: (s, 0) for s, k in first.items()} | {k: (s, 1) for s, k in second.items()}
    table = [epoch]
    for k in range(2 * n - 2 - r):
        o, h = owner[k]
        slot = ex.symm.buffer_ptrs[o] + bank_off + (h * n + r) * c * tok
        flag = ex.symm.signal_pad_ptrs[o] + ((_BWD_FIRST if h == 0 else _BWD_SECOND) * n + r) * 8
        table += [slot, slot + bwd_bytes // 2, flag]
    for j in range(2):
        flag = ex.symm.signal_pad_ptrs[r] + ((_BWD_FIRST if j == 0 else _BWD_SECOND) * n + r) * 8
        table += [own[0, j * c].data_ptr(), own[1, j * c].data_ptr(), flag]
    dkv = torch.empty((2, 2 * c) + tail_shape, device=kb.device, dtype=dtype)
    table += [
        ex.symm.buffer_ptrs[r] + bank_off,
        ex.symm.signal_pad_ptrs[r],
        dkv[0].data_ptr(),
        dkv[1].data_ptr(),
    ]
    table += [own[0].data_ptr(), own[1].data_ptr()]
    dq = torch.empty_like(q)
    _bwd_kernel.backward(
        dout,
        q,
        kb,
        vb,
        out,
        lse0,
        lse1,
        scale,
        n,
        r,
        table,
        dq,
        bwd_bytes // 2,
        (_BWD_FIRST * n, _BWD_SECOND * n),
    )
    return dq, dkv[0], dkv[1]


class _AGKVAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, scale):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        ctx.group, ctx.scale = group, scale
        out, lse0, lse1, saved = _forward(q, k, v, group, any(ctx.needs_input_grad[:3]))
        kb, vb = saved if saved is not None else (None, None)
        ctx.save_for_backward(q, kb, vb, out, lse0, lse1)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, kb, vb, out, lse0, lse1 = ctx.saved_tensors
        dq, dk, dv = _backward(dout.contiguous(), q, kb, vb, out, lse0, lse1, ctx.group, ctx.scale)
        return dq, dk, dv, None, None


def attention_agkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Causal attention of this rank's zigzag tokens against the whole sequence.

    q [s_local, b, hq, d], k / v [s_local, b, hkv, d], sbhd, bf16, d in (64, 128), an even number of q
    heads per kv head; s_local holds this rank's two chunks, chunk r then chunk 2n-1-r, each a multiple of
    128 tokens and at least 256. Returns [s_local, b, hq, d] in the same token order.
    """
    assert q.dim() == 4 and k.shape == v.shape and q.shape[:2] == k.shape[:2], "q / k / v must be sbhd"
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "the all-gather-KV attention takes bf16 only"
    s_local, B, Hq, D = q.shape
    Hkv, n = k.shape[2], group.size()
    assert s_local % 2 == 0, f"local sequence {s_local} must hold two zigzag chunks"
    c = s_local // 2
    scale = 1.0 / math.sqrt(D)
    if softmax_scale is not None and abs(softmax_scale - scale) > 1e-6:
        raise ValueError(f"the all-gather-KV attention takes the default softmax scale 1/sqrt({D}) only")
    if not _fwd_kernel.supported(Hq, Hkv, D, c):
        raise ValueError(
            f"the all-gather-KV attention does not take hq {Hq}, hkv {Hkv}, d {D}, chunk {c}: it needs d in "
            "(64, 128), an even number of q heads per kv head and chunks of a multiple of 128 tokens, >= 256"
        )
    if (q.requires_grad or k.requires_grad or v.requires_grad) and not _bwd_kernel.supported(
        B, c, n, Hq, Hkv, D, scale
    ):
        raise ValueError(
            f"the all-gather-KV attention backward does not take b {B}, chunk {c}, cp {n}, hq {Hq}, "
            f"hkv {Hkv}, d {D}"
        )
    return _AGKVAttention.apply(q, k, v, group, scale)
