###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""q1's backward for the fused all-gather-KV attention, its dK / dV sent by the kernel itself.

The flux dK / dV kernel runs q1 (the rank's second zigzag chunk, ``c`` rows) against all its keys
``[remote | own first | own second]`` one q split per launch. The launch of the last split stores
through ``epilogue``: per work-group (BLOCK_KV key rows of one kv head and batch) it sums its
fp32 split with the earlier splits' bf16 partials and q0's gradient (``dk0`` / ``dv0``, key rows
``[add_lo, add_hi)``), and stores the result

    remote key chunk k  ->  slot (half, rank) of the owner's backward bank (system-scope stores)
    own chunk           ->  the local own-gradient buffer [own first | own second]

A work-group does not wait for its stores: waiting would hold its CU for the whole remote write and
serialize the traffic behind the compute instead of under it. The launch's completion is what
publishes them, so a one-block kernel queued behind it raises the owners' backward flags to the
epoch (``signal``); the owner's reduce kernel waits for them unchanged.

Per-call addresses (banks, flags, addends, buffers) ride a small int64 table in the dense path's
unused cu_seqlens argument:

    [epoch, dk0, dv0, (dK destination, dV destination, flag address) per key chunk]

where a chunk's destinations are the addresses of ITS first row (batch 0).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw

from primus_turbo.flydsl.attention import flash_attn_bwd_flux as _flux
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.fused_collective_gemm.common import SYS_NT, i64_base
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from primus_turbo.flydsl.utils.prims import cast, ld, st

P_EPOCH, P_ADD_K, P_ADD_V = range(3)
P_CHUNKS = 3
CHUNK_FIELDS = 3
SIGNAL_THREADS = 64

# host table copies in flight per (device, length) before the oldest is reused
_RING = 8


@ASTRewriter.transform
def _raise_flags(tid, params, num_chunks):
    if tid < fx.Int32(num_chunks):
        flag = ld(params, fx.Int32(P_CHUNKS + 2) + tid * fx.Int32(CHUNK_FIELDS), dtype=fx.T.i64)
        st(flag, fx.Int32(0), ld(params, fx.Int32(P_EPOCH), dtype=fx.T.i64), scope="sys", order="relaxed")


@functools.lru_cache(maxsize=64)
def _make_signal_kernel(num_chunks):
    @flyc.kernel(name="agkv_bwd_signal_kernel", known_block_size=[SIGNAL_THREADS, 1, 1])
    def kernel(PARAMS: fx.Tensor):
        _raise_flags(cast(fx.thread_idx.x, fx.T.i32), i64_base(PARAMS), num_chunks)

    return kernel


@flyc.jit
def _launch_signal(PARAMS, num_chunks: fx.Constexpr[int], stream: fx.Stream):
    _make_signal_kernel(num_chunks)(PARAMS).launch(grid=(1, 1, 1), block=(SIGNAL_THREADS, 1, 1), stream=stream)


def signal(params, num_chunks, stream):
    """Raise the flags of the table's first ``num_chunks`` key chunks to its epoch, once ``stream``'s work
    so far is done."""
    assert num_chunks <= SIGNAL_THREADS
    if num_chunks:
        run_launcher(_launch_signal, params, num_chunks, stream)


@functools.lru_cache(maxsize=64)
def epilogue(c, rem, add_lo, add_hi, tok, batch_bytes):
    """The flux dK / dV epilogue of one rank's q1 (lengths in key rows, ``tok`` bytes per key row of all
    batches, ``batch_bytes`` per batch)."""
    skv = rem + 2 * c

    def emit(dv_values, dk_values, info):
        assert info["token_stride"] * 2 == tok
        f32x4, bf16x4 = fx.T.vec(4, fx.T.f32), fx.T.vec(4, fx.T.bf16)
        params = i64_base(info["params"])
        kv_start = cast(info["kv_start"], fx.T.i32)
        b_off = cast(cast(info["batch_idx"], fx.T.i32) * fx.Int32(batch_bytes), fx.T.i64)
        chunk = kv_start // fx.Int32(c)
        entry = fx.Int32(P_CHUNKS) + chunk * fx.Int32(CHUNK_FIELDS)
        has_add = (kv_start >= fx.Int32(add_lo)) & (kv_start < fx.Int32(add_hi))
        add_bytes = ArithValue(has_add).select(fx.Int32(add_hi * tok), fx.Int32(0))
        for t, values, ws in ((1, dv_values, info["dv_ws"]), (0, dk_values, info["dk_ws"])):
            ws_base = i64_base(ws) + b_off
            terms = [
                create_buffer_resource_from_addr(ws_base + fx.Int64(s * skv * tok), num_records_bytes=skv * tok)
                for s in range(info["q_split"] - 1)
            ]
            add_base = ld(params, fx.Int32(P_ADD_K + t), dtype=fx.T.i64) + b_off - fx.Int64(add_lo * tok)
            terms.append(create_buffer_resource_from_addr(add_base, num_records_bytes=add_bytes))
            dst = (
                ld(params, entry + fx.Int32(t), dtype=fx.T.i64)
                + b_off
                - cast(chunk, fx.T.i64) * fx.Int64(c * tok)
            )
            dst_res = create_buffer_resource_from_addr(dst, num_records_bytes=(chunk + fx.Int32(1)) * fx.Int32(c * tok))
            values = [(_raw(v), cast(idx, fx.T.i32)) for v, idx in values]
            loads = [[buffer_load(res, idx, vec_width=4, dtype=fx.T.bf16) for res in terms] for _, idx in values]
            for (acc, idx), parts in zip(values, loads):
                for part in parts:
                    acc = fx.arith.addf(acc, fx.arith.extf(f32x4, part))
                buffer_store(fx.arith.trunc_f(bf16x4, acc), dst_res, idx, cache_modifier=SYS_NT)

    return emit


def block_kv(Skv, D):
    return _flux._fuse_blockkv_for(Skv, D, -1)


def _plan(B, Sq, Skv, Hq, Hkv, D, scale, deterministic, epi):
    """The dense plan of q1's backward with ``epi``, or None when it cannot run one launch per split."""
    plan = _flux._dense_plan(
        B, Sq, Skv, Hq, Hkv, D, scale, -1, True, deterministic, causal=True, dkdv_override={"epilogue": epi}
    )
    _, q_split, bkv, _, a16, _, _, _, _, band_span, _, _, _, _ = plan
    split_ok = q_split == 1 or (
        (bkv % (q_split * _flux._BWD_BLOCK_Q) == 0 or _flux._qsp_absolute(D, bkv, q_split))
        and _flux._qsp_cuttable(Sq, q_split)
    )
    if a16 or band_span or bkv != block_kv(Skv, D) or not split_ok:
        return None
    return plan


def supported(B, Sq, Skv, Hq, Hkv, D, scale, deterministic, c, tok):
    bkv = block_kv(Skv, D)
    if c % bkv or Skv * tok >= 1 << 31:
        return False
    epi = epilogue(c, Skv - 2 * c, 0, c, tok, Hkv * D * 2)
    return _plan(B, Sq, Skv, Hq, Hkv, D, scale, deterministic, epi) is not None


_RINGS = {}


def params_table(values, device):
    """``values`` as a device int64 tensor, copied from a pinned ring without a host synchronize."""
    key = (device, len(values))
    ring = _RINGS.get(key)
    if ring is None:
        ring = _RINGS[key] = [
            [
                torch.empty(len(values), dtype=torch.int64, pin_memory=True),
                torch.empty(len(values), dtype=torch.int64, device=device),
                torch.cuda.Event(),
            ]
            for _ in range(_RING)
        ] + [0]
    i = ring[-1]
    ring[-1] = (i + 1) % _RING
    host, dev, done = ring[i]
    done.synchronize()
    host.copy_(torch.tensor(values, dtype=torch.int64))
    dev.copy_(host, non_blocking=True)
    done.record()
    return dev


def q1_backward(dout, q, k, v, out, lse, scale, deterministic, epi, params, num_remote, dq):
    """dQ of ``q`` (the last ``Sq`` rows of a bottom-right causal call against ``k`` / ``v``) into ``dq``;
    dK / dV go out through ``epi``, whose runtime table is ``params``, then the flags of its first
    ``num_remote`` key chunks go up."""
    Sq, B, Hq, D = q.shape
    Skv, _, Hkv, _ = k.shape
    plan = _plan(B, Sq, Skv, Hq, Hkv, D, scale, deterministic, epi)
    assert plan is not None
    _, q_split, bkv, n_bands, _, _, wsq_ilv, wsq_ring, wsq_pair, _, pipe, dkdv_l, odo_l, _ = plan
    st = torch.cuda.current_stream()
    qf = _flux._prescale_q(q, scale)
    kf, vf, dof = k.reshape(-1), v.reshape(-1), dout.reshape(-1)
    o16 = out.to(q.dtype).reshape(-1)
    delta = torch.empty(B, Hq, Sq, device=q.device, dtype=torch.float32)
    df = delta.reshape(-1)
    odo_l(o16, dof, df, B, Sq, st)
    ws_dk, ws_dv = (
        torch.empty(max(q_split - 1, 1), Skv, B, Hkv, D, device=q.device, dtype=k.dtype).reshape(-1)
        for _ in range(2)
    )
    lsef = _flux._prescale_lse(lse.view(B, Sq, Hq).permute(0, 2, 1), st).reshape(-1)
    ws_dq, _ = _flux._dq_partial_ws(
        wsq_ring * wsq_ilv or n_bands,
        B,
        Sq,
        Hq * D,
        q.device,
        q.dtype,
        _flux._WSQ_BAND_PAD if D == 128 else 0,
        wsq_ilv,
        carry=False,
        pair=bool(wsq_pair),
    )
    side = _flux._SIDE_STREAM.get(q.device)
    if side is None:
        side = _flux._SIDE_STREAM[q.device] = torch.cuda.Stream(device=q.device)
    ph = _flux._cu_placeholder(q.device)

    def reduce(stream, qsp=(1, 0, None)):
        _flux._reduce_dq_partials(
            ws_dq,
            dq,
            bkv,
            Hq,
            D,
            dkdv_l.dq_scale,
            stream,
            qsp=qsp,
            causal_offset=Skv - Sq,
            window_left=-1,
            causal=True,
            n_bands=n_bands,
            sbhd=True,
            band_ring=wsq_ring,
            band_pair=wsq_pair,
            ph=ph,
        )

    for s in range(q_split):
        body = dkdv_l if q_split == 1 else dkdv_l.chunk(s, 1)
        body(qf, kf, vf, dof, lsef, df, ws_dk, ws_dv, params, params, ws_dq[0, 0].reshape(-1), B, Sq, Skv, 0, st)
        if pipe and s < q_split - 1:
            side.wait_stream(st)
            reduce(side, (q_split, s, 1))
    signal(params, num_remote, st)
    if pipe:
        reduce(st, (q_split, q_split - 1, 1))
        st.wait_stream(side)
    else:
        reduce(st)
    return dq
