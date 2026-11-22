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

Direct (``aux_blocks`` = 0): a work-group does not wait for its stores; the launch's completion
publishes them, and a one-block kernel queued behind it raises the owners' backward flags to the
epoch (``signal``). Each work-group's remote write still holds its CU after its compute (one
work-group per CU), so the traffic adds to the launch instead of hiding under it.

Pushed (``aux_blocks`` > 0): remote rows go to a local staging buffer and each work-group counts
itself into its key chunk; ``aux_blocks`` extra work-groups of the same launch, dispatched first,
copy each chunk to its owner once it is complete (whole-row 16 B copies, chunk order) while the
other CUs keep computing, and raise the owner's flag as the push kernel does. The owner's reduce
kernel waits for the flags unchanged either way.

Per-call addresses ride a small int64 table in the dense path's unused cu_seqlens argument:

    [epoch, dk0, dv0, ready counters, done counters, units, heaps, flag arrays, bank offset,
     staging K, staging V, (dK destination, dV destination, flag address) per key chunk]

where a chunk's destinations are the addresses of ITS first row (batch 0). Push units are the comm
kernel's (``UNIT_FIELDS`` int64 each), their group the key chunk.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw

from primus_turbo.flydsl.attention import flash_attn_bwd_flux as _flux
from primus_turbo.flydsl.attention.agkv_comm_kernel import (
    PUSH_BATCH,
    U_BYTES,
    U_DST,
    U_DST_OFF,
    U_FLAG,
    U_SRC,
    U_SRC_OFF,
    UNIT_BYTES,
    UNIT_FIELDS,
    WAIT_TIMEOUT,
    _unit_field,
)
from primus_turbo.flydsl.deep_ep.common import _asm
from primus_turbo.flydsl.deep_ep.launch_cache import run_launcher
from primus_turbo.flydsl.fused_collective_gemm.common import (
    BLOCK_THREADS,
    SYS_NT,
    VEC_I32,
    copy_rows,
    i64_base,
)
from primus_turbo.flydsl.utils.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    create_llvm_ptr,
)
from primus_turbo.flydsl.utils.prims import atomic_add, cast, ld, read_clock, spin_timed_out, st

(
    P_EPOCH,
    P_ADD_K,
    P_ADD_V,
    P_READY,
    P_DONE,
    P_UNITS,
    P_HEAPS,
    P_FLAGS,
    P_BANK,
    P_STAGE_K,
    P_STAGE_V,
    P_CHUNKS,
) = range(12)
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
    _make_signal_kernel(num_chunks)(PARAMS).launch(
        grid=(1, 1, 1), block=(SIGNAL_THREADS, 1, 1), stream=stream
    )


def signal(params, num_chunks, stream):
    """Raise the flags of the table's first ``num_chunks`` key chunks to its epoch, once ``stream``'s work
    so far is done."""
    assert num_chunks <= SIGNAL_THREADS
    if num_chunks:
        run_launcher(_launch_signal, params, num_chunks, stream)


@ASTRewriter.transform
def _when(cond, body):
    if cond:
        body()


@ASTRewriter.transform
def _count_ready(tid, remote, params, chunk):
    """One more work-group of key chunk ``chunk`` has its rows in the staging buffer."""
    if remote:
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if tid == fx.Int32(0):
            atomic_add(ld(params, fx.Int32(P_READY), dtype=fx.T.i64), chunk, fx.Int64(1), scope="agent")


def _copy_staged(tid, unit, srcs, heaps, units, bank_off, block_threads):
    """Copy unit ``unit`` out of the staging buffer (read past the caches: other XCDs wrote it)."""
    sel = _unit_field(units, unit, U_SRC)
    src_base = ArithValue(sel == fx.Int32(1)).select(srcs[1], srcs[0])
    nbytes = _unit_field(units, unit, U_BYTES)
    dst_heap = ld(heaps, _unit_field(units, unit, U_DST), dtype=fx.T.i64)
    src_res = create_buffer_resource_from_addr(
        src_base + _unit_field(units, unit, U_SRC_OFF, wide=True), num_records_bytes=nbytes
    )
    dst_res = create_buffer_resource_from_addr(
        dst_heap + bank_off + _unit_field(units, unit, U_DST_OFF, wide=True), num_records_bytes=nbytes
    )
    nvec = nbytes // fx.Int32(16)

    def coords(v):
        word = v * fx.Int32(VEC_I32)
        return word, word, v < nvec

    for k in range(BLOCK_THREADS // block_threads):
        copy_rows(
            tid + fx.Int32(k * block_threads),
            UNIT_BYTES // 16,
            coords,
            src_res,
            dst_res,
            SYS_NT,
            SYS_NT,
            PUSH_BATCH,
        )


@ASTRewriter.transform
def _wait_staged(ready, chunk, expected, rank):
    spin_start = read_clock()
    staged = ld(ready, chunk, scope="sys", dtype=fx.T.i64) >= fx.Int64(expected)
    while not staged:
        fx.rocdl.s_sleep(fx.Int32(1))
        if spin_timed_out(spin_start, WAIT_TIMEOUT):
            fx.printf(f"[agkv rank={rank} bwd push] chunk wait timeout: chunk={{}}\n", chunk)
            _asm("s_trap 2")
        staged = ld(ready, chunk, scope="sys", dtype=fx.T.i64) >= fx.Int64(expected)


@ASTRewriter.transform
def _count_pushed(done, ready, flags, units, chunk, first_unit, mine, chunk_units, epoch):
    """``mine`` more units of ``chunk`` retired; the chunk's last resets its counters and raises its flag."""
    if mine > fx.Int64(0):
        count = atomic_add(done, chunk, mine, scope="agent")
        if count + mine == fx.Int64(chunk_units):
            st(done, chunk, fx.Int64(0), scope="agent")
            st(ready, chunk, fx.Int64(0), scope="agent")
            dst_flags = ld(flags, _unit_field(units, first_unit, U_DST), dtype=fx.T.i64)
            st(dst_flags, _unit_field(units, first_unit, U_FLAG), epoch, scope="sys", order="relaxed")


@ASTRewriter.transform
def _push_staged(
    is_aux, tid, aux_id, params, num_chunks, chunk_units, aux_blocks, expected, block_threads, rank
):
    """The push blocks, key chunk by key chunk: wait until the chunk has all ``expected`` work-groups
    staged, issue this block's share of its units (every ``aux_blocks``-th) back to back, drain them
    once, and count them; the chunk's last count resets its counters and raises the owner's flag."""
    if is_aux:
        units = ld(params, fx.Int32(P_UNITS), dtype=fx.T.i64)
        ready = ld(params, fx.Int32(P_READY), dtype=fx.T.i64)
        done = ld(params, fx.Int32(P_DONE), dtype=fx.T.i64)
        heaps = ld(params, fx.Int32(P_HEAPS), dtype=fx.T.i64)
        flags = ld(params, fx.Int32(P_FLAGS), dtype=fx.T.i64)
        bank_off = ld(params, fx.Int32(P_BANK), dtype=fx.T.i64)
        epoch = ld(params, fx.Int32(P_EPOCH), dtype=fx.T.i64)
        srcs = [
            ld(params, fx.Int32(P_STAGE_K), dtype=fx.T.i64),
            ld(params, fx.Int32(P_STAGE_V), dtype=fx.T.i64),
        ]
        mine = ArithValue(aux_id < fx.Int32(chunk_units)).select(
            (fx.Int32(chunk_units - 1) - aux_id) // fx.Int32(aux_blocks) + fx.Int32(1), fx.Int32(0)
        )
        chunk = fx.Int32(0)
        while chunk < fx.Int32(num_chunks):
            first_unit = chunk * fx.Int32(chunk_units)
            if tid == fx.Int32(0):
                _wait_staged(ready, chunk, expected, rank)
            fx.gpu.barrier()
            j = aux_id
            while j < fx.Int32(chunk_units):
                _copy_staged(tid, first_unit + j, srcs, heaps, units, bank_off, block_threads)
                j = j + fx.Int32(aux_blocks)
            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            if tid == fx.Int32(0):
                _count_pushed(
                    done, ready, flags, units, chunk, first_unit, cast(mine, fx.T.i64), chunk_units, epoch
                )
            chunk = chunk + fx.Int32(1)


@functools.lru_cache(maxsize=64)
def epilogue(c, rem, add_lo, add_hi, tok, batch_bytes, aux_blocks=0, num_units=0, expected=0, rank=0):
    """The flux dK / dV epilogue of one rank's q1 (lengths in key rows, ``tok`` bytes per key row of all
    batches, ``batch_bytes`` per batch). ``aux_blocks`` > 0: remote rows go to the staging buffer and
    that many push blocks of the same launch send its ``num_units`` copy units, a chunk once its
    ``expected`` work-groups are in."""
    skv = rem + 2 * c

    def store(dv_values, dk_values, info):
        bf16x4 = fx.T.vec(4, fx.T.bf16)
        params = i64_base(info["params"])
        tid, head = cast(info["tid"], fx.T.i32), cast(info["kv_head_idx"], fx.T.i32)
        kv_start = cast(info["kv_start"], fx.T.i32)
        b_off = cast(cast(info["batch_idx"], fx.T.i32) * fx.Int32(batch_bytes), fx.T.i64)
        chunk = kv_start // fx.Int32(c)
        entry = fx.Int32(P_CHUNKS) + chunk * fx.Int32(CHUNK_FIELDS)
        has_add = (kv_start >= fx.Int32(add_lo)) & (kv_start < fx.Int32(add_hi))
        add_bytes = ArithValue(has_add).select(fx.Int32(add_hi * tok), fx.Int32(0))
        for t, values, ws in ((1, dv_values, info["dv_ws"]), (0, dk_values, info["dk_ws"])):
            ws_base = i64_base(ws) + b_off
            terms = [
                create_buffer_resource_from_addr(
                    ws_base + fx.Int64(s * skv * tok), num_records_bytes=skv * tok
                )
                for s in range(info["q_split"] - 1)
            ]
            add_base = ld(params, fx.Int32(P_ADD_K + t), dtype=fx.T.i64) + b_off - fx.Int64(add_lo * tok)
            terms.append(create_buffer_resource_from_addr(add_base, num_records_bytes=add_bytes))
            dst = (
                ld(params, entry + fx.Int32(t), dtype=fx.T.i64)
                + b_off
                - cast(chunk, fx.T.i64) * fx.Int64(c * tok)
            )
            dst_res = create_buffer_resource_from_addr(
                dst, num_records_bytes=(chunk + fx.Int32(1)) * fx.Int32(c * tok)
            )
            fragments = [
                (fx.arith.trunc_f(bf16x4, _raw(v)), cast(row, fx.T.i32) - kv_start, cast(col, fx.T.i32))
                for v, _, row, col in values
            ]
            _stage_and_store(fragments, terms, dst_res, kv_start, head, tid, info)
        if aux_blocks:
            _count_ready(tid, kv_start < fx.Int32(rem), params, chunk)

    def emit(dv_values, dk_values, info):
        assert info["token_stride"] * 2 == tok
        if aux_blocks:
            # the push blocks walk the empty tile past the keys
            _when(cast(info["kv_start"], fx.T.i32) < fx.Int32(skv), lambda: store(dv_values, dk_values, info))
        else:
            store(dv_values, dk_values, info)

    def aux(is_aux, info):
        _push_staged(
            is_aux,
            cast(info["tid"], fx.T.i32),
            cast(info["aux_id"], fx.T.i32),
            i64_base(info["params"]),
            rem // c,
            num_units // (rem // c),
            aux_blocks,
            expected,
            info["block_size"],
            rank,
        )

    if aux_blocks:
        emit.aux_blocks = aux_blocks
        emit.aux = aux
    return emit


# s_waitcnt lgkmcnt(0), vmcnt and expcnt left alone
_WAIT_LGKM = 0xC07F


def _lds_ptr(elem, lds_off):
    return create_llvm_ptr(elem * fx.Int32(2) + fx.Int32(lds_off), address_space=3)


def _stage_and_store(fragments, terms, dst_res, kv_start, head, tid, info):
    """Store one work-group's [BLOCK_KV, D] tile, held as bf16 4-element fragments ``(value, tile row,
    column)``, plus the same rows of ``terms`` (bf16 buffers laid out as the destination), summed in fp32,
    as whole rows: through the LDS the main loop is done with, then 16 B per lane, a wave's access four
    contiguous 256 B rows. Fragment accesses would move 32 B pieces, which the remote fabric stores at a
    fraction of the rate and which hold the attention's own loads behind them; the term rows are loaded
    before the LDS round trip and arrive under it."""
    bkv, hd, threads, lds_off = info["block_kv"], info["head_dim"], info["block_size"], info["lds_off"]
    row_elems = hd + 8  # padded: the fragment writes of 16 rows at one column fall in distinct banks
    vecs_per_row = hd // 8
    assert bkv * row_elems * 2 <= info["lds_bytes"] and bkv * vecs_per_row % threads == 0
    f32x8, bf16x8 = fx.T.vec(8, fx.T.f32), fx.T.vec(8, fx.T.bf16)
    coords = []
    for i in range(bkv * vecs_per_row // threads):
        vec = tid + fx.Int32(i * threads)
        row, part = vec // fx.Int32(vecs_per_row), vec % fx.Int32(vecs_per_row)
        idx = (kv_start + row) * fx.Int32(info["token_stride"]) + head * fx.Int32(hd) + part * fx.Int32(8)
        coords.append((row * fx.Int32(row_elems) + part * fx.Int32(8), idx))
    # the main loop's LDS reads and DMA writes retire before the LDS is reused
    fx.rocdl.s_waitcnt(0)
    loads = [[buffer_load(res, idx, vec_width=8, dtype=fx.T.bf16) for res in terms] for _, idx in coords]
    fx.gpu.barrier()
    for value, row, col in fragments:
        llvm.StoreOp(_raw(value), _lds_ptr(row * fx.Int32(row_elems) + col, lds_off))
    fx.rocdl.s_waitcnt(_WAIT_LGKM)
    fx.gpu.barrier()
    for (elem, idx), parts in zip(coords, loads):
        acc = fx.arith.extf(f32x8, llvm.LoadOp(bf16x8, _lds_ptr(elem, lds_off)).result)
        for part in parts:
            acc = fx.arith.addf(acc, fx.arith.extf(f32x8, part))
        buffer_store(fx.arith.trunc_f(bf16x8, acc), dst_res, idx, cache_modifier=SYS_NT)


@functools.lru_cache(maxsize=64)
def push_units(owners, rank, num_ranks, seg_bytes, bwd_bytes, device):
    """The push blocks' copy units, key chunk by key chunk: ``owners[k]`` = (owner rank, half, flag array)
    of remote key chunk k, whose rows sit at ``k * seg_bytes`` of the staging buffers and go to slot
    (half, rank) of the owner's backward bank (V region at ``bwd_bytes // 2``)."""
    pieces = -(-seg_bytes // UNIT_BYTES)
    rows = []
    for k, (owner, half, array) in enumerate(owners):
        slot = (half * num_ranks + rank) * seg_bytes
        for p in range(pieces):
            off = p * UNIT_BYTES
            for t in range(2):
                rows.append(
                    [
                        t,
                        k * seg_bytes + off,
                        owner,
                        t * (bwd_bytes // 2) + slot + off,
                        min(UNIT_BYTES, seg_bytes - off),
                        k,
                        2 * pieces,
                        array * num_ranks + rank,
                    ]
                )
    assert all(len(row) == UNIT_FIELDS for row in rows)
    return torch.tensor(rows, dtype=torch.int64, device=device), len(rows)


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
        body(
            qf,
            kf,
            vf,
            dof,
            lsef,
            df,
            ws_dk,
            ws_dv,
            params,
            params,
            ws_dq[0, 0].reshape(-1),
            B,
            Sq,
            Skv,
            0,
            st,
        )
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
