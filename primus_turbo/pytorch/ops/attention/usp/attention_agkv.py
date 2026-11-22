###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""All-gather-KV context parallelism for causal GQA attention (the Llama 3 scheme).

The sequence is cut into 2n chunks of c tokens and rank r holds chunks r and 2n-1-r (the zigzag order of
Megatron's ``get_batch_on_this_cp_rank``), so every rank does the same causal work. Every rank keeps all
heads; only K and V cross ranks: gathered in the forward, their gradients reduced back in the backward.
Each local q chunk is one bottom-right causal FlyDSL call against all the keys it sees.

Modes:

``naive``
    the plain implementation: one all-gather in rank order, an index copy of K and V into sequence order,
    both calls, then a scatter of dK / dV back into rank order and one reduce-scatter.

``serial``
    RCCL as in ``naive``, without the copies. A bottom-right causal call masks only its last c keys (its
    own chunk); every earlier key is seen by every query, so their order is free. The gathered buffer is

        [ second halves, slot j = chunk 2n-1-j | first halves, slot j = chunk j | own second half ]

    and both calls read contiguous ranges of it:

        q0 = chunk r          first-half slots 0..r (slot r is its own chunk)
        q1 = chunk 2n-1-r     second-half slots r+1..n-1, every first-half slot, own second half

    q1's keys end with the gathered region, so the backward writes dK / dV straight into the layout the
    reduce-scatter reads.

``overlap``
    the ``serial`` layout, exchanged on the SDMA engines instead of RCCL: every rank copies its halves
    straight into the peers' symmetric buffers (``hipMemcpyDeviceToDeviceNoCU``, one stream per peer) and
    raises a flag per peer with one more 8-byte copy behind them; each call waits (``hipStreamWaitValue64``)
    only for the slots it reads. The copies take no compute units, so they run under the attention kernels
    at full speed, which RCCL's kernels do not (they slow down several times beside FlyDSL attention). Forward: q0 starts once the first halves of the lower ranks are in, q1 once
    the rest is. Backward: q1's backward runs first and its final gradient slots go out during q0's
    backward; every rank sums the partial dK / dV slots it receives for its own two chunks.

    The calls stay as one per q chunk with the collective work around them; splitting into phases with a
    collective in between would line up every rank at the phase boundary, and the phases are not balanced
    across ranks (rank r's q0 sees (r+1) chunks, its q1 2n-r).

``mega``
    MegaMoE-style communication kernels (``agkv_comm_kernel``) instead of SDMA copies, and a forward that
    starts attending before any remote key is in. One push kernel per exchange step stores the halves
    (gradient slots) over XGMI from a few compute units into the peers' buffers and raises a flag per
    (destination, half); a one-block-per-XCD wait kernel holds a call until its slots are in. A few dozen
    blocks of stores move the data at about RCCL's speed beside the attention kernels, without the
    per-copy cost of the SDMA engines (~25 us a copy, a few copies in flight per engine).

    Receiver p keeps the remote halves as

        [ second halves of p+1..n-1 | first halves of p+1..n-1 | first halves of 0..p-1 ]

    so q0's remote keys (the first halves of the lower ranks) are its tail and q1's are all of it. Each q
    chunk runs its bottom-right causal call against its own keys first (q0: its own chunk, q1: both own
    chunks), then, once the remote slots it reads are in, one non-causal call against them; a merge
    kernel combines the two partial outputs by their LSE. Every rank sends all first halves before any
    second half. The forward saves [ remote | own ] for the backward: q1 against the remote keys
    (non-causal) first, whose gradient slots for the higher ranks are then final and go out; q0 (causal),
    after which the slots of the lower ranks go out, the push adding q0's part on the way; last q1 against
    its own two chunks (causal), under both pushes. One reduce kernel then sums the received slots and the
    own partial gradients of all calls (fp32, sender order) straight into dK / dV. Where that last call
    would be too small to fill the GPU, q1 runs as one causal call and only q0's backward hides the pushes.

    The non-causal calls use the flux FlyDSL attention (``FLUX_ATTN_FLYDSL=1``); the stock non-causal
    forward gives wrong outputs for these calls (SBHD, GQA, Sq != Skv).

Both ``overlap`` and ``mega`` reuse double-buffered symmetric buffers call after call, which needs every
rank of the group to issue the same sequence of forward and backward calls (true of SPMD training), and
count calls on the host, so they cannot be graph captured.

The forward keeps the gathered K / V for the backward rather than gathering again.
"""

import ctypes
import functools
import math
from typing import Optional

import torch
import torch.distributed as dist

from primus_turbo.common.flux import enabled as flux_enabled
from primus_turbo.flydsl.attention import agkv_comm_kernel as _comm
from primus_turbo.flydsl.attention.flash_attn_bwd import flydsl_varlen_backward as _flydsl_bwd
from primus_turbo.pytorch.core.pyhip_runtime_wrapper import (
    find_hip_runtime_library,
    get_hip_runtime_lib,
    hipMemcpyKindEnum,
)
from primus_turbo.pytorch.core.symm_mem import SymmetricMemory
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import _fwd_module as _flydsl_fwd_module
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import _score_bound as _flydsl_score_bound
from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
    flash_attn_sbhd_flydsl_backward_impl,
    flash_attn_sbhd_flydsl_forward_impl,
)

__all__ = ["AGKV_MODES", "attention_agkv"]

AGKV_MODES = ("naive", "serial", "overlap", "mega")

# blocks of a mega push launch: (per peer link, at least, at most in all), always a multiple of the n - 1
# links, which keeps every block on one link (units alternate between destinations) so no link has fewer
# blocks than another: at n = 8, 28 blocks ran the forward 30-50 us faster than 24 or 32 at 16K. Fewer
# blocks leave the links idle, more slow the attention beside them. The forward needs the keys early (28 /
# 24 / 8 blocks at n = 8 / 4 / 2; at n = 8, 0.5-3 % off fwd + bwd against 48); the backward's pushes run
# under attention and only need to finish before it does (21 / 9 / 8). The push after q0 without a split
# q1 has no attention beside it.
_FWD_PUSH_BLOCKS = (8, 1, 28)
_BWD_PUSH_BLOCKS = (3, 8, 1 << 30)
_EXPOSED_PUSH_BLOCKS = (8, 48, 1 << 30)


# mega splits q1's backward into the remote keys (non-causal) and its own two chunks (causal, last, under
# the gradient pushes) when that last call has at least this many (key, kv head) pairs: fewer leave the GPU
# under-occupied (2c = 2048 keys x 4 heads: +120 us on a 16K n = 8 fwd + bwd; x 8 heads: -35 us)
_BWD_SPLIT_MIN_KEY_HEADS = 16384


def _push_grid(n, blocks):
    per_peer, least, most = blocks
    return (n - 1) * max(min(per_peer, most // (n - 1)), -(-least // (n - 1)), 1)


def _fwd(q, k, v, softmax_scale):
    out, lse = flash_attn_sbhd_flydsl_forward_impl(
        q, k, v, softmax_scale=softmax_scale, causal=True, return_lse=True
    )
    return out, lse


def _bwd(dout, q, k, v, out, lse, softmax_scale, deterministic):
    Sq, B, Hq = q.shape[0], q.shape[1], q.shape[2]
    return flash_attn_sbhd_flydsl_backward_impl(
        dout,
        q,
        k,
        v,
        out,
        lse.view(B, Sq, Hq).permute(0, 2, 1),
        softmax_scale=softmax_scale,
        causal=True,
        deterministic=deterministic,
    )


class _Layout:
    """Index arithmetic of one rank; all lengths in tokens."""

    def __init__(self, s_local, group):
        self.n, self.r = group.size(), group.rank()
        assert s_local % 2 == 0, f"local sequence {s_local} must hold two zigzag chunks"
        self.c = s_local // 2
        n, r, c = self.n, self.r, self.c
        self.len0, self.len1 = (r + 1) * c, (2 * n - r) * c
        # [second halves | first halves | own second half]
        self.first = slice(n * c, 2 * n * c)
        self.second = slice(0, n * c)
        self.tail = slice(2 * n * c, (2 * n + 1) * c)
        self.keys0 = slice(n * c, n * c + self.len0)
        self.keys1 = slice((r + 1) * c, (2 * n + 1) * c)
        # naive: gathered rank-major [rank j: chunk j, chunk 2n-1-j] -> sequence order
        self.seq_order = [2 * g if g < n else 2 * (2 * n - 1 - g) + 1 for g in range(2 * n)]

    def first_slot(self, s):
        return slice((self.n + s) * self.c, (self.n + s + 1) * self.c)

    def second_slot(self, s):
        return slice(s * self.c, (s + 1) * self.c)


# --------------------------------------------------------------------------------------------------------
# symmetric exchange buffers; SDMA copies (overlap)
# --------------------------------------------------------------------------------------------------------

# flag arrays in the signal pad, n int64 each, indexed by sender
_FWD_FIRST, _FWD_SECOND, _BWD_FIRST, _BWD_SECOND = range(4)
_NUM_FLAG_ARRAYS = 4
_HIP_STREAM_WAIT_VALUE_GTE = 0


# flag values a sender copies from, per direction; refilled after a device synchronize every this many calls
_FLAG_TABLE = 4096


@functools.lru_cache(maxsize=None)
def _stream_wait_value():
    """``hipStreamWaitValue64``: holds a stream until a 64-bit word reaches a value, without a host round
    trip (ROCm runs it as a one-wave kernel)."""
    lib = ctypes.CDLL(find_hip_runtime_library())
    wait = lib.hipStreamWaitValue64
    wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint64]
    wait.restype = ctypes.c_int

    def wait_value(stream, ptr, value):
        err = wait(stream, ptr, value, _HIP_STREAM_WAIT_VALUE_GTE, 0xFFFFFFFFFFFFFFFF)
        if err != 0:
            raise RuntimeError(f"hipStreamWaitValue64 failed with hipError_t {err}")

    return wait_value


class _Exchange:
    """One rank's symmetric buffers, flags and streams for a process group.

    Buffer, per bank (two, used alternately by consecutive calls of each direction):
        forward   K, V gathered buffers, (2n+1)c tokens each, in the ``_Layout`` order
        backward  K, V partial-gradient slots, [first: n senders | second: n senders] x c tokens each

    Each direction counts its calls (epochs, from 1) and uses bank ``epoch & 1``; a sender raises its flag
    to the epoch once its data is in (``overlap``: an 8-byte SDMA copy queued behind its data copies on the
    same stream; ``mega``: the push kernel), a receiver waits until the flag reaches it. (``hipStreamWriteValue64`` is a kernel on ROCm; between the copies it
    makes the stream switch engines and doubles the copy time at 7 peers.) A rank's call e cannot finish
    before every peer has sent its call-e data, which a peer does only after finishing call e - 1, so when a
    peer writes bank e & 1 for call e + 2 this rank is past call e: two banks need no acknowledgement.
    Flags only grow, so a sender one call ahead does not hide a call from the receiver.
    """

    _cache = {}

    def __init__(self, group, nbytes):
        self.group = group
        self.n, self.r = group.size(), group.rank()
        self.symm = SymmetricMemory(
            group, nbytes, signal_pad_size=_NUM_FLAG_ARRAYS * self.n * 8, uncached_signal_pad=True
        )
        self.nbytes = nbytes
        self.lib = get_hip_runtime_lib()
        self.wait_value = _stream_wait_value()
        self.streams = [torch.cuda.Stream() if p != self.r else None for p in range(self.n)]
        self.save_stream = torch.cuda.Stream()
        self.epoch = {"fwd": 0, "bwd": 0}
        self.flag_values = {
            d: torch.arange(1, _FLAG_TABLE + 1, dtype=torch.int64, device="cuda") for d in self.epoch
        }
        self.heaps = torch.tensor(self.symm.buffer_ptrs, dtype=torch.int64, device="cuda")
        self.flag_ptrs = torch.tensor(self.symm.signal_pad_ptrs, dtype=torch.int64, device="cuda")
        # push group counters, see _GROUPS_*
        self.counters = torch.zeros(6 * self.n, dtype=torch.int64, device="cuda")
        self.comm_stream = torch.cuda.Stream(priority=-1)

    @classmethod
    def get(cls, group, nbytes):
        key = (group.group_name, torch.cuda.current_device())
        old = cls._cache.get(key)
        if old is not None and old.nbytes >= nbytes:
            return old
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the all-gather-KV overlap counts its calls on the host and cannot be graph captured"
            )
        # allocate before freeing: a heap that reuses just-freed memory cannot be exported over IPC
        ex = cls._cache[key] = cls(group, max(nbytes, 0 if old is None else 2 * old.nbytes))
        if old is not None:
            # peers may still be copying into the old heap until every rank is past its last call
            torch.cuda.synchronize()
            dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
            old.symm.destroy()
        return ex

    def _fill_flag_values(self, direction, first):
        self.flag_values[direction].copy_(torch.arange(first, first + _FLAG_TABLE, dtype=torch.int64))

    def begin(self, direction):
        """Epoch of a new call in ``direction``; its bank is ``epoch & 1``."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the all-gather-KV overlap counts its calls on the host and cannot be graph captured"
            )
        self.epoch[direction] += 1
        epoch = self.epoch[direction]
        if epoch % _FLAG_TABLE == 1 and epoch > 1:
            # no copy of the previous table range is still queued
            torch.cuda.synchronize()
            self._fill_flag_values(direction, epoch)
        return epoch

    def view(self, rank, byte_offset, shape, dtype):
        return self.symm.get_buffer(rank, shape, dtype, storage_offset=byte_offset // dtype.itemsize)

    def ptr(self, rank, byte_offset):
        return self.symm.buffer_ptrs[rank] + byte_offset

    def push(self, peer, dst_offset, src):
        """Copy contiguous ``src`` to byte ``dst_offset`` of ``peer``'s buffer on the SDMA engines."""
        self.lib.hipMemcpyAsync(
            self.ptr(peer, dst_offset),
            src.data_ptr(),
            src.numel() * src.element_size(),
            hipMemcpyKindEnum.hipMemcpyDeviceToDeviceNoCU,
            self.streams[peer].cuda_stream,
        )
        src.record_stream(self.streams[peer])

    def signal(self, peer, array, direction, epoch):
        """Raise my flag ``array`` on ``peer`` to ``epoch`` once my copies to it so far have landed."""
        self.lib.hipMemcpyAsync(
            self.symm.signal_pad_ptrs[peer] + (array * self.n + self.r) * 8,
            self.flag_values[direction].data_ptr() + (epoch - 1) % _FLAG_TABLE * 8,
            8,
            hipMemcpyKindEnum.hipMemcpyDeviceToDeviceNoCU,
            self.streams[peer].cuda_stream,
        )

    def wait(self, array, senders, epoch):
        """Hold the current stream until the flags ``array`` of ``senders`` reach ``epoch``."""
        stream = torch.cuda.current_stream().cuda_stream
        for s in senders:
            if s != self.r:
                self.wait_value(stream, self.symm.signal_pad_ptrs[self.r] + (array * self.n + s) * 8, epoch)

    def fork(self):
        ready = torch.cuda.current_stream().record_event()
        for st in self.streams:
            if st is not None:
                st.wait_event(ready)


def _exchange_sizes(lay, tok):
    fwd = 2 * (2 * lay.n + 1) * lay.c * tok
    bwd = 2 * 2 * lay.n * lay.c * tok
    return fwd, bwd


# --------------------------------------------------------------------------------------------------------
# mega: compute-unit pushes over XGMI (agkv_comm_kernel)
# --------------------------------------------------------------------------------------------------------

# push counter groups: forward [0, 2n), backward after q1 [2n, 4n), backward after q0 [4n, 6n)
_GROUPS_FWD, _GROUPS_BWD1, _GROUPS_BWD0 = 0, 2, 4


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


def _fwd_full(q, k, v):
    """Non-causal forward: the flux FlyDSL kernel built without the causal mask."""
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    mod = _flydsl_fwd_module(Hq, Hkv, D, False, Sq != Skv, True, -1, sbhd=True)
    out = torch.empty_like(q)
    lse = torch.empty((B * Sq, Hq), device=q.device, dtype=torch.float32)
    mod(
        q,
        k,
        v,
        out,
        B,
        Sq,
        seq_len_kv=Skv,
        stride_q_n=B * Hq * D,
        stride_kv_n=B * Hkv * D,
        sink=None,
        score_bound=_flydsl_score_bound(k, D),
        stream=torch.cuda.current_stream(),
        debug_counts=lse,
    )
    return out, lse


def _bwd_full(dout, q, k, v, out, lse, softmax_scale, deterministic):
    """Non-causal backward of queries ``q`` against keys ``k`` alone, ``out`` / ``lse`` over all their keys."""
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    scale = 1.0 / math.sqrt(D) if softmax_scale is None else softmax_scale
    lse_bhs = lse.view(B, Sq, Hq).permute(0, 2, 1)
    return _flydsl_bwd(
        dout,
        q,
        k,
        v,
        out,
        lse_bhs,
        B,
        Sq,
        Skv,
        Hq,
        Hkv,
        D,
        scale,
        sbhd=True,
        deterministic=deterministic,
        causal=False,
    )[:3]


def _remote_layout(n, p):
    """Token offsets (in chunks) of the remote slots in receiver ``p``'s buffer, laid out

        [ second halves of senders p+1..n-1 | first halves of p+1..n-1 | first halves of 0..p-1 ]

    so q1's remote keys are the whole buffer and q0's (first halves of the lower ranks) its suffix."""
    first = {s: (n - 1 - p) + (s - p - 1) if s > p else 2 * (n - 1 - p) + s for s in range(n) if s != p}
    second = {s: s - p - 1 for s in range(p + 1, n)}
    return first, second


@functools.lru_cache(maxsize=64)
def _mega_fwd_plan(n, r, c, tok, fwd_bytes, device):
    """Forward copy units of rank ``r`` and the flags its two remote calls wait for.

    Units copy K (source 0) / V (source 1) halves into the peers' forward banks (V region at
    ``fwd_bytes // 2``) at their ``_remote_layout`` slots: the first half to every peer, the second half
    to the lower ones. A (destination, half) is one group; its flag is ``_FWD_FIRST`` / ``_FWD_SECOND``
    of sender ``r``. Every rank's remote q0 call needs only first halves, so they all go first, each
    destination served alike (every rank is through its local calls at the same time)."""
    half = c * tok
    pieces = -(-half // _comm.UNIT_BYTES)

    def segments(dst, h):
        first, second = _remote_layout(n, dst)
        dst_off = (first if h == 0 else second)[r] * half
        group, flag = _GROUPS_FWD * n + 2 * dst + h, (_FWD_FIRST if h == 0 else _FWD_SECOND) * n + r
        return [
            _segment(t, h * half, dst, t * (fwd_bytes // 2) + dst_off, half, group, 2 * pieces, flag)
            for t in range(2)
        ]

    peers = list(range(r + 1, n)) + list(range(r))
    firsts = [sg for p in peers for sg in segments(p, 0)]
    seconds = [sg for p in peers if p < r for sg in segments(p, 1)]
    rows = _phase(pieces, firsts) + _phase(pieces, seconds)
    units, num_units = _units_table(rows, device) if rows else (None, 0)
    q0_flags = tuple(_FWD_FIRST * n + s for s in range(r))
    q1_flags = tuple(_FWD_FIRST * n + s for s in range(r + 1, n)) + tuple(
        _FWD_SECOND * n + s for s in range(r + 1, n)
    )
    return units, num_units, q0_flags, q1_flags


@functools.lru_cache(maxsize=64)
def _mega_bwd_plan(n, r, c, tok, bwd_bytes, device):
    """Backward copy units of rank ``r``: out of the remote-key gradients (``_remote_layout`` rows of
    dK / dV, sources 0 / 1), after q1's remote backward the slots of the higher owners, after q0's the
    first-half slots of the lower owners, each into slot ``(half, r)`` of the owner's backward bank
    (V region at ``bwd_bytes // 2``). Also returns the flags the owner's reduce waits for."""
    seg = c * tok
    pieces = -(-seg // _comm.UNIT_BYTES)
    first, second = _remote_layout(n, r)

    def segments(base, dst, h):
        group, flag = base * n + 2 * dst + h, (_BWD_FIRST if h == 0 else _BWD_SECOND) * n + r
        src_off = (first if h == 0 else second)[dst] * seg
        dst_off = (h * n + r) * seg
        return [
            _segment(t, src_off, dst, t * (bwd_bytes // 2) + dst_off, seg, group, 2 * pieces, flag)
            for t in range(2)
        ]

    after_q1 = [
        sg for s in range(r + 1, n) for sg in segments(_GROUPS_BWD1, s, 1) + segments(_GROUPS_BWD1, s, 0)
    ]
    after_q0 = [sg for s in range(r) for sg in segments(_GROUPS_BWD0, s, 0)]
    units1, num1 = _units_table(_phase(pieces, after_q1), device) if after_q1 else (None, 0)
    units0, num0 = _units_table(_phase(pieces, after_q0), device) if after_q0 else (None, 0)
    first_flags = tuple(_BWD_FIRST * n + s for s in range(n) if s != r)
    second_flags = tuple(_BWD_SECOND * n + s for s in range(r))
    return units1, num1, units0, num0, first_flags, second_flags


def _mega_push(ex, srcs, units, num_units, bank_off, epoch, grid, addends=None, add_delta=0):
    if not num_units:
        return
    ex.comm_stream.wait_stream(torch.cuda.current_stream())
    _comm.push(
        srcs,
        ex.heaps,
        ex.flag_ptrs,
        units,
        num_units,
        ex.counters,
        bank_off,
        epoch,
        grid,
        ex.comm_stream,
        addends,
        add_delta,
    )
    for t in tuple(srcs) + tuple(addends or ()):
        t.record_stream(ex.comm_stream)


def _side_copy(ex, dst, src):
    """``dst.copy_(src)`` on the side stream, under the work queued after it; ``_join_side`` ends it."""
    ex.save_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(ex.save_stream):
        dst.copy_(src)
    src.record_stream(ex.save_stream)


def _join_side(ex):
    torch.cuda.current_stream().wait_stream(ex.save_stream)


def _mega_forward(q, k, v, lay, group, softmax_scale, save):
    """Returns ``out``, the lse of q0 / q1 and, for the backward, K / V laid out

        [ remote slots (``_remote_layout``) | own first half | own second half ]

    where q1's keys are the whole buffer and q0's the first halves of the lower ranks plus its own: both
    end with their own chunk, so the backward runs two bottom-right causal calls as ``serial`` does."""
    n, r, c = lay.n, lay.r, lay.c
    tail_shape, dtype = k.shape[1:], k.dtype
    tok = k[0].numel() * k.element_size()
    fwd_bytes, bwd_bytes = _exchange_sizes(lay, tok)
    ex = _Exchange.get(group, 2 * (fwd_bytes + bwd_bytes))
    epoch = ex.begin("fwd")
    bank_off = (epoch & 1) * fwd_bytes
    units, num_units, q0_flags, q1_flags = _mega_fwd_plan(n, r, c, tok, fwd_bytes, k.device)
    cur = torch.cuda.current_stream()
    _mega_push(ex, (k, v), units, num_units, bank_off, epoch, _push_grid(n, _FWD_PUSH_BLOCKS))

    # local keys first, under the exchange: q0 sees its own chunk, q1 both own chunks; every remote key
    # is visible to every query, so the remote calls are non-causal and merge in by lse
    o0, l0 = _fwd(q[:c], k[:c], v[:c], softmax_scale)
    o1, l1 = _fwd(q[c:], k, v, softmax_scale)
    rem, low = (2 * n - 2 - r) * c, r * c
    kr, vr = (ex.view(r, bank_off + i * fwd_bytes // 2, (rem,) + tail_shape, dtype) for i in range(2))
    out = torch.empty_like(q)
    if r > 0:
        _comm.wait(ex.flag_ptrs, r, q0_flags, epoch, cur)
        ro, rl = _fwd_full(q[:c], kr[rem - low :], vr[rem - low :])
        lse0 = torch.empty_like(l0)
        _comm.merge(out[:c], lse0, o0, l0, ro, rl, cur)
    else:
        _side_copy(ex, out[:c], o0)
        lse0 = l0
    _comm.wait(ex.flag_ptrs, r, q1_flags, epoch, cur)
    saved = None
    if save:
        # a copy kernel beside q1's remote call: the SDMA engines copy within a GPU at ~60 GB/s, which
        # at 16K costs more than the kernel's share of the compute units
        ex.save_stream.wait_stream(cur)
        with torch.cuda.stream(ex.save_stream):
            saved = [torch.cat((buf, t)) for buf, t in ((kr, k), (vr, v))]
        for t in saved:
            t.record_stream(cur)
    ro, rl = _fwd_full(q[c:], kr, vr)
    lse1 = torch.empty_like(l1)
    _comm.merge(out[c:], lse1, o1, l1, ro, rl, cur)
    _join_side(ex)
    return out, lse0, lse1, saved


def _mega_backward(dout, q, kb, vb, out, lse0, lse1, lay, group, scale, det):
    n, r, c = lay.n, lay.r, lay.c
    tail_shape, dtype = kb.shape[1:], kb.dtype
    tok = kb[0].numel() * kb.element_size()
    fwd_bytes, bwd_bytes = _exchange_sizes(lay, tok)
    ex = _Exchange.get(group, 2 * (fwd_bytes + bwd_bytes))
    epoch = ex.begin("bwd")
    bank_off = 2 * fwd_bytes + (epoch & 1) * bwd_bytes
    units1, num1, units0, num0, first_flags, second_flags = _mega_bwd_plan(n, r, c, tok, bwd_bytes, kb.device)
    rem, low = (2 * n - 2 - r) * c, r * c
    keys0 = slice(rem - low, rem + c)

    seg = c * tok // kb.element_size()
    own_first, own_tail = (2 * n - 2 - r) * seg, (2 * n - 1 - r) * seg
    # q0's keys start at row rem - low of q1's: the pushes after q0 add q0's gradient rows of the lower
    # ranks' slots on the fly, the reduce q0's own-chunk rows (from row low), so no add runs between calls
    add_delta = (rem - low) * tok
    grid, exposed = _push_grid(n, _BWD_PUSH_BLOCKS), _push_grid(n, _EXPOSED_PUSH_BLOCKS)
    dq = torch.empty_like(q)
    if 2 * c * kb.shape[1] * kb.shape[2] >= _BWD_SPLIT_MIN_KEY_HEADS:
        # q1 against the remote keys first: its gradient rows then hold the final slots of the higher ranks
        # and go out under q0 and q1's own causal part; the slots of the lower ranks go out after q0, under
        # q1's own part
        own = slice(rem, rem + 2 * c)
        dq1, dk1, dv1 = _bwd_full(dout[c:], q[c:], kb[:rem], vb[:rem], out[c:], lse1, scale, det)
        _mega_push(ex, (dk1, dv1), units1, num1, bank_off, epoch, grid)
        dq0, dk0, dv0 = _bwd(dout[:c], q[:c], kb[keys0], vb[keys0], out[:c], lse0, scale, det)
        dk0, dv0 = dk0.contiguous(), dv0.contiguous()
        _mega_push(ex, (dk1, dv1), units0, num0, bank_off, epoch, grid, (dk0, dv0), add_delta)
        _side_copy(ex, dq[:c], dq0)
        dqb, dko, dvo = _bwd(dout[c:], q[c:], kb[own], vb[own], out[c:], lse1, scale, det)
        torch.add(dq1, dqb, out=dq[c:])
        own_k, own_v, own_first, own_tail = dko, dvo, 0, seg
    else:
        dq1, dk1, dv1 = _bwd(dout[c:], q[c:], kb, vb, out[c:], lse1, scale, det)
        dk1, dv1 = dk1.contiguous(), dv1.contiguous()
        _mega_push(ex, (dk1, dv1), units1, num1, bank_off, epoch, grid)
        _side_copy(ex, dq[c:], dq1)
        dq0, dk0, dv0 = _bwd(dout[:c], q[:c], kb[keys0], vb[keys0], out[:c], lse0, scale, det)
        dk0, dv0 = dk0.contiguous(), dv0.contiguous()
        # no attention is left to run beside this push
        _mega_push(ex, (dk1, dv1), units0, num0, bank_off, epoch, exposed, (dk0, dv0), add_delta)
        dq[:c].copy_(dq0)
        own_k, own_v = dk1, dv1

    dkv = torch.empty((2, 2 * c) + tail_shape, device=kb.device, dtype=dtype)
    _comm.reduce(
        ex.heaps,
        ex.flag_ptrs,
        own_k,
        own_v,
        dkv[0],
        dkv[1],
        r,
        n,
        seg,
        own_first,
        own_tail,
        bank_off,
        bwd_bytes // 4,
        first_flags,
        second_flags,
        epoch,
        torch.cuda.current_stream(),
        extra=(dk0, dv0, r * seg),
    )
    _join_side(ex)
    return dq, dkv[0], dkv[1]


def _sdma_forward(q, k, v, lay, group, softmax_scale, save):
    n, r, c = lay.n, lay.r, lay.c
    tail_shape, dtype = k.shape[1:], k.dtype
    tok = k[0].numel() * k.element_size()
    fwd_bytes, bwd_bytes = _exchange_sizes(lay, tok)
    ex = _Exchange.get(group, 2 * (fwd_bytes + bwd_bytes))
    epoch = ex.begin("fwd")
    base = [(epoch & 1) * fwd_bytes + i * fwd_bytes // 2 for i in range(2)]
    bufs = [ex.view(r, base[i], ((2 * n + 1) * c,) + tail_shape, dtype) for i in range(2)]

    ex.fork()
    first_off = lay.first_slot(r).start * tok
    second_off = lay.second_slot(r).start * tok
    for p in range(n):
        if p == r:
            continue
        for i, t in enumerate((k, v)):
            ex.push(p, base[i] + first_off, t[:c])
        ex.signal(p, _FWD_FIRST, "fwd", epoch)
        # q1 of rank p reads second-half slots p+1..n-1 only
        if p < r:
            for i, t in enumerate((k, v)):
                ex.push(p, base[i] + second_off, t[c:])
            ex.signal(p, _FWD_SECOND, "fwd", epoch)
    for buf, t in zip(bufs, (k, v)):
        buf[lay.first_slot(r)].copy_(t[:c])
        buf[lay.tail].copy_(t[c:])

    kb, vb = bufs
    ex.wait(_FWD_FIRST, range(r), epoch)
    out0, lse0 = _fwd(q[:c], kb[lay.keys0], vb[lay.keys0], softmax_scale)
    ex.wait(_FWD_FIRST, range(r + 1, n), epoch)
    ex.wait(_FWD_SECOND, range(r + 1, n), epoch)
    saved = None
    if save:
        # what both calls read, kept out of the symmetric buffer the next call overwrites
        cur = torch.cuda.current_stream()
        ex.save_stream.wait_stream(cur)
        with torch.cuda.stream(ex.save_stream):
            saved = [buf[lay.keys1].clone() for buf in bufs]
    out1, lse1 = _fwd(q[c:], kb[lay.keys1], vb[lay.keys1], softmax_scale)
    if save:
        cur.wait_stream(ex.save_stream)
        for t in saved:
            t.record_stream(cur)
    return out0, out1, lse0, lse1, saved


def _sdma_backward(dout, q, kb, vb, out0, out1, lse0, lse1, lay, group, scale, det, keys0, keys1):
    n, r, c = lay.n, lay.r, lay.c
    tail_shape, dtype = kb.shape[1:], kb.dtype
    tok = kb[0].numel() * kb.element_size()
    fwd_bytes, bwd_bytes = _exchange_sizes(lay, tok)
    ex = _Exchange.get(group, 2 * (fwd_bytes + bwd_bytes))
    epoch = ex.begin("bwd")
    base = [2 * fwd_bytes + (epoch & 1) * bwd_bytes + i * bwd_bytes // 2 for i in range(2)]
    # [K / V, first / second half, sender, c tokens]
    kv_slots = ex.view(r, base[0], (2, 2, n, c) + tail_shape, dtype)
    slots = kv_slots.unbind(0)
    slot_off = lambda half, sender: ((half * n + sender) * c) * tok

    # q1's keys: second-half slots r+1..n-1, first-half slots 0..n-1, own second half
    dq1, dk1, dv1 = _bwd(dout[c:], q[c:], kb[keys1], vb[keys1], out1, lse1, scale, det)
    in_first = lambda s: slice((n - r - 1 + s) * c, (n - r + s) * c)
    in_second = lambda s: slice((s - r - 1) * c, (s - r) * c)
    ex.fork()
    for s in range(r + 1, n):
        for i, g in enumerate((dk1, dv1)):
            ex.push(s, base[i] + slot_off(1, r), g[in_second(s)])
        ex.signal(s, _BWD_SECOND, "bwd", epoch)
        for i, g in enumerate((dk1, dv1)):
            ex.push(s, base[i] + slot_off(0, r), g[in_first(s)])
        ex.signal(s, _BWD_FIRST, "bwd", epoch)
    for sl, g in zip(slots, (dk1, dv1)):
        sl[1, r].copy_(g[-c:])

    dq0, dk0, dv0 = _bwd(dout[:c], q[:c], kb[keys0], vb[keys0], out0, lse0, scale, det)
    # disjoint from the slots still being pushed above
    lower = slice((n - r - 1) * c, n * c)
    dk1[lower].add_(dk0)
    dv1[lower].add_(dv0)
    ex.fork()
    for s in range(r):
        for i, g in enumerate((dk1, dv1)):
            ex.push(s, base[i] + slot_off(0, r), g[in_first(s)])
        ex.signal(s, _BWD_FIRST, "bwd", epoch)
    for sl, g in zip(slots, (dk1, dv1)):
        sl[0, r].copy_(g[in_first(r)])

    ex.wait(_BWD_FIRST, range(n), epoch)
    ex.wait(_BWD_SECOND, range(r), epoch)
    dkv = torch.empty((2, 2 * c) + tail_shape, device=kb.device, dtype=dtype)
    torch.sum(kv_slots[:, 0], dim=1, out=dkv[:, :c])
    torch.sum(kv_slots[:, 1, : r + 1], dim=1, out=dkv[:, c:])
    return torch.cat((dq0, dq1)), dkv[0], dkv[1]


# --------------------------------------------------------------------------------------------------------
# autograd
# --------------------------------------------------------------------------------------------------------


class _AGKVAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, group, mode, softmax_scale, deterministic):
        lay = _Layout(q.shape[0], group)
        n, c = lay.n, lay.c
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        tail = k.shape[1:]
        keys0, keys1 = lay.keys0, lay.keys1
        ctx.group, ctx.mode, ctx.lay = group, mode, lay
        ctx.softmax_scale, ctx.deterministic = softmax_scale, deterministic
        if mode == "mega":
            out, lse0, lse1, saved = _mega_forward(
                q, k, v, lay, group, softmax_scale, any(ctx.needs_input_grad[:3])
            )
            kb, vb = saved if saved is not None else (None, None)
            ctx.save_for_backward(q, kb, vb, out, lse0, lse1)
            return out
        if mode == "naive":
            kg = torch.empty((2 * n * c,) + tail, device=k.device, dtype=k.dtype)
            vg = torch.empty_like(kg)
            dist.all_gather_into_tensor(kg, k, group=group)
            dist.all_gather_into_tensor(vg, v, group=group)
            idx = torch.tensor(lay.seq_order, device=k.device)
            kb = kg.view((2 * n, c) + tail)[idx].view((2 * n * c,) + tail)
            vb = vg.view((2 * n, c) + tail)[idx].view((2 * n * c,) + tail)
            del kg, vg
            keys0, keys1 = slice(0, lay.len0), slice(0, lay.len1)
            out0, lse0 = _fwd(q[:c], kb[keys0], vb[keys0], softmax_scale)
            out1, lse1 = _fwd(q[c:], kb[keys1], vb[keys1], softmax_scale)
        elif mode == "serial":
            kb = torch.empty(((2 * n + 1) * c,) + tail, device=k.device, dtype=k.dtype)
            vb = torch.empty_like(kb)
            kb[lay.tail].copy_(k[c:])
            vb[lay.tail].copy_(v[c:])
            for buf, t in ((kb, k), (vb, v)):
                dist.all_gather_into_tensor(buf[lay.first], t[:c], group=group)
                dist.all_gather_into_tensor(buf[lay.second], t[c:], group=group)
            out0, lse0 = _fwd(q[:c], kb[keys0], vb[keys0], softmax_scale)
            out1, lse1 = _fwd(q[c:], kb[keys1], vb[keys1], softmax_scale)
        else:
            save = any(ctx.needs_input_grad[:3])
            out0, out1, lse0, lse1, saved = _sdma_forward(q, k, v, lay, group, softmax_scale, save)
            kb, vb = saved if saved is not None else (None, None)
            # saved holds keys1 = [(r+1)c, (2n+1)c) of the buffer
            off = keys1.start
            keys0, keys1 = slice(keys0.start - off, keys0.stop - off), slice(0, lay.len1)
        out = torch.cat((out0, out1))
        ctx.save_for_backward(q, kb, vb, out0, out1, lse0, lse1)
        ctx.keys = (keys0, keys1)
        return out

    @staticmethod
    def backward(ctx, dout):
        group, mode, lay = ctx.group, ctx.mode, ctx.lay
        scale, det = ctx.softmax_scale, ctx.deterministic
        dout = dout.contiguous()
        if mode == "mega":
            q, kb, vb, out, lse0, lse1 = ctx.saved_tensors
            dq, dk, dv = _mega_backward(dout, q, kb, vb, out, lse0, lse1, lay, group, scale, det)
            return dq, dk, dv, None, None, None, None

        q, kb, vb, out0, out1, lse0, lse1 = ctx.saved_tensors
        keys0, keys1 = ctx.keys
        n, r, c = lay.n, lay.r, lay.c
        tail = kb.shape[1:]

        if mode == "overlap":
            dq, dk, dv = _sdma_backward(
                dout, q, kb, vb, out0, out1, lse0, lse1, lay, group, scale, det, keys0, keys1
            )
            return dq, dk, dv, None, None, None, None

        dq1, dk1, dv1 = _bwd(dout[c:], q[c:], kb[keys1], vb[keys1], out1, lse1, scale, det)
        dq0, dk0, dv0 = _bwd(dout[:c], q[:c], kb[keys0], vb[keys0], out0, lse0, scale, det)
        dk = torch.empty((2 * c,) + tail, device=kb.device, dtype=kb.dtype)
        dv = torch.empty_like(dk)
        if mode == "naive":
            idx = torch.tensor(lay.seq_order, device=kb.device)
            for g0, g1, out in ((dk0, dk1, dk), (dv0, dv1, dv)):
                full = torch.zeros((2 * n * c,) + tail, device=kb.device, dtype=kb.dtype)
                full[: lay.len1].copy_(g1)
                full[: lay.len0].add_(g0)
                rank_major = torch.empty_like(full)
                rank_major.view((2 * n, c) + tail)[idx] = full.view((2 * n, c) + tail)
                dist.reduce_scatter_tensor(out, rank_major, group=group)
        else:
            # gathered-region gradient [second halves | first halves]; second-half slots 0..r-1 are chunks
            # after q1 and get nothing, slot r (own chunk 2n-1-r) gets q1's gradient of its own tail
            for g0, g1, out in ((dk0, dk1, dk), (dv0, dv1, dv)):
                g = torch.empty((2 * n * c,) + tail, device=kb.device, dtype=kb.dtype)
                g[: r * c].zero_()
                g[r * c : (r + 1) * c].copy_(g1[-c:])
                g[(r + 1) * c :].copy_(g1[:-c])
                g[keys0].add_(g0)
                dist.reduce_scatter_tensor(out[:c], g[lay.first], group=group)
                dist.reduce_scatter_tensor(out[c:], g[lay.second], group=group)
        return torch.cat((dq0, dq1)), dk, dv, None, None, None, None


def attention_agkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    mode: str = "overlap",
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """Causal attention of this rank's zigzag tokens against the whole sequence.

    q [s_local, b, hq, d], k / v [s_local, b, hkv, d], sbhd, bf16, d in (64, 128); s_local holds this
    rank's two chunks, chunk r then chunk 2n-1-r. Returns [s_local, b, hq, d] in the same token order.
    """
    assert mode in AGKV_MODES, f"mode must be one of {AGKV_MODES}, got {mode!r}"
    assert q.dim() == 4 and k.shape == v.shape and q.shape[:2] == k.shape[:2], "q / k / v must be sbhd"
    if mode == "mega":
        assert k.dtype == torch.bfloat16, "mode 'mega' reduces gradients in bf16 only"
        # only the flux build of the FlyDSL forward runs the non-causal (bidirectional) mask correctly
        assert flux_enabled("FLUX_ATTN_FLYDSL"), (
            "mode 'mega' needs the flux FlyDSL attention (FLUX_ATTN_FLYDSL=1)"
        )
    return _AGKVAttention.apply(q, k, v, group, mode, softmax_scale, deterministic)
